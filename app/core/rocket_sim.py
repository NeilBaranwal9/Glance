"""Parametric rocket geometry and a 3-DOF flight simulation.

Two jobs, one set of parameters:

1. :meth:`RocketParams.build_mesh` turns the dimensions into a shaded 3-D mesh
   for the viewer, so the drawing on screen is the vehicle being simulated
   rather than a decorative picture.
2. :func:`simulate` flies that vehicle and emits a multi-channel telemetry
   stream.

This is what makes geometry actually matter to the ML tab. Diameter sets frontal
area and therefore drag; fin count and span add drag and mass; propellant and
dry mass set the mass history; the motor sets thrust. Those propagate into
altitude, velocity, acceleration, dynamic pressure and Mach, and from there into
every model's output. Nothing is faked -- change the fin span and the flight
really is different, so the health score and anomaly timeline really do change.

A physical degradation process runs alongside: the nozzle erodes over the burn,
which bleeds chamber pressure, raises nozzle temperature and increases
vibration. That is the signal the anomaly detector is meant to find, and its
severity is a parameter.

No Qt import belongs here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

# Atmosphere (ISA-ish, adequate for sub-orbital hobby trajectories)
SEA_LEVEL_DENSITY = 1.225      # kg/m^3
SCALE_HEIGHT = 8500.0          # m
GRAVITY = 9.80665              # m/s^2
SEA_LEVEL_SOUND_SPEED = 340.29  # m/s

NOSE_SHAPES = ("ogive", "conical", "parabolic", "elliptical")


def air_density(altitude: np.ndarray | float) -> np.ndarray | float:
    """Exponential atmosphere. Clamped so very high altitudes stay finite."""
    return SEA_LEVEL_DENSITY * np.exp(-np.maximum(altitude, 0.0) / SCALE_HEIGHT)


def speed_of_sound(altitude: np.ndarray | float) -> np.ndarray | float:
    """Drops roughly linearly through the troposphere, then holds."""
    return SEA_LEVEL_SOUND_SPEED - 0.0038 * np.clip(altitude, 0.0, 11000.0)


@dataclass
class RocketParams:
    """Everything the viewer and the simulation need. Lengths in metres, mass in kg."""

    # --- airframe
    nose_length: float = 0.30
    nose_shape: str = "ogive"
    body_diameter: float = 0.10
    body_length: float = 1.00
    boattail_length: float = 0.08
    boattail_diameter: float = 0.075

    # --- fins
    fin_count: int = 4
    fin_root_chord: float = 0.20
    fin_tip_chord: float = 0.10
    fin_span: float = 0.12
    fin_sweep: float = 0.06
    fin_thickness: float = 0.004

    # --- mass and motor
    dry_mass: float = 3.0
    propellant_mass: float = 1.2
    motor_thrust: float = 350.0     # average thrust, N
    burn_time: float = 3.2          # s

    # --- environment / mission
    launch_angle: float = 3.0       # degrees from vertical
    drag_coefficient: float = 0.55  # base Cd at low Mach, before fin/Mach terms

    # --- degradation injected into the engine-health channels
    degradation: float = 0.35       # 0 = a perfect motor, 1 = severe erosion
    noise: float = 0.02             # relative sensor noise

    # ------------------------------------------------------------------ derived

    @property
    def total_length(self) -> float:
        return self.nose_length + self.body_length + self.boattail_length

    @property
    def frontal_area(self) -> float:
        return math.pi * (self.body_diameter / 2.0) ** 2

    @property
    def fin_area(self) -> float:
        """Planform area of one fin (trapezoid)."""
        return 0.5 * (self.fin_root_chord + self.fin_tip_chord) * self.fin_span

    @property
    def wet_mass(self) -> float:
        return self.dry_mass + self.propellant_mass

    @property
    def total_impulse(self) -> float:
        return self.motor_thrust * self.burn_time

    @property
    def effective_cd(self) -> float:
        """Body Cd plus a fin contribution referenced to frontal area."""
        if self.frontal_area <= 0:
            return self.drag_coefficient
        fin_drag = 0.012 * self.fin_count * self.fin_area / self.frontal_area
        return self.drag_coefficient + min(fin_drag, 1.5)

    @property
    def fineness_ratio(self) -> float:
        return self.total_length / max(self.body_diameter, 1e-6)

    def stability_margin(self) -> float:
        """Approximate static margin in calibres (Barrowman-style, simplified).

        Positive and roughly 1-2 calibres is the stable design range; the viewer
        surfaces this so parameter changes have an immediate engineering meaning.
        """
        diameter = max(self.body_diameter, 1e-6)
        # Centre of pressure: nose contribution plus fins.
        nose_cp = 0.466 * self.nose_length if self.nose_shape == "ogive" else 0.666 * self.nose_length
        fin_position = self.nose_length + self.body_length - self.fin_root_chord
        mid_chord = self.fin_sweep + 0.5 * (self.fin_root_chord - self.fin_tip_chord)
        fin_cn = (
            (4.0 * self.fin_count * (self.fin_span / diameter) ** 2)
            / (1.0 + math.sqrt(1.0 + (2.0 * math.hypot(mid_chord, self.fin_span)
                                      / (self.fin_root_chord + self.fin_tip_chord + 1e-9)) ** 2))
        )
        fin_cp = fin_position + self.fin_sweep / 3.0 * (
            (self.fin_root_chord + 2 * self.fin_tip_chord)
            / max(self.fin_root_chord + self.fin_tip_chord, 1e-9)
        ) + 0.25 * self.fin_root_chord
        nose_cn = 2.0
        total_cn = nose_cn + fin_cn
        cp = (nose_cn * nose_cp + fin_cn * fin_cp) / max(total_cn, 1e-9)

        # Centre of gravity: crude mass split between airframe and motor.
        airframe_cg = 0.55 * self.total_length
        motor_cg = self.nose_length + self.body_length - 0.15
        cg = (self.dry_mass * airframe_cg + self.propellant_mass * motor_cg) / max(self.wet_mass, 1e-9)
        return (cp - cg) / diameter

    # ------------------------------------------------------------------ geometry

    def _nose_radius(self, x: np.ndarray) -> np.ndarray:
        """Radius profile along the nose, x in [0, nose_length]."""
        radius = self.body_diameter / 2.0
        length = max(self.nose_length, 1e-6)
        fraction = np.clip(x / length, 0.0, 1.0)
        shape = self.nose_shape
        if shape == "conical":
            return radius * fraction
        if shape == "parabolic":
            return radius * (2 * fraction - fraction**2)
        if shape == "elliptical":
            return radius * np.sqrt(np.clip(1.0 - (1.0 - fraction) ** 2, 0.0, 1.0))
        # Tangent ogive
        rho = (radius**2 + length**2) / (2 * radius) if radius > 0 else length
        inner = np.clip(rho**2 - (length - fraction * length) ** 2, 0.0, None)
        return np.sqrt(inner) - (rho - radius)

    def build_mesh(self, radial_segments: int = 28, axial_segments: int = 26) -> "RocketMesh":
        """Tessellate the airframe into triangles, plus one quad pair per fin.

        The body is a surface of revolution; fins are flat plates swept out at
        equal angles. Coordinates are centred on the vehicle so the viewer can
        rotate about the middle.
        """
        radial_segments = max(6, int(radial_segments))
        axial_segments = max(6, int(axial_segments))

        radius = self.body_diameter / 2.0
        boat_radius = max(self.boattail_diameter / 2.0, 1e-4)

        # Axial stations and the radius at each.
        nose_x = np.linspace(0.0, self.nose_length, axial_segments)
        stations = [(x, r) for x, r in zip(nose_x, self._nose_radius(nose_x))]
        stations.append((self.nose_length, radius))
        stations.append((self.nose_length + self.body_length, radius))
        if self.boattail_length > 1e-6:
            stations.append((self.total_length, boat_radius))

        axis = np.array([s[0] for s in stations], dtype=np.float64)
        radii = np.array([max(s[1], 1e-5) for s in stations], dtype=np.float64)
        centre = self.total_length / 2.0
        axis = axis - centre  # centre the model on the origin

        angles = np.linspace(0.0, 2 * np.pi, radial_segments, endpoint=False)
        cos_a, sin_a = np.cos(angles), np.sin(angles)

        vertices: list[list[float]] = []
        for x, r in zip(axis, radii):
            for c, s in zip(cos_a, sin_a):
                vertices.append([float(x), float(r * c), float(r * s)])

        faces: list[tuple[int, int, int]] = []
        groups: list[str] = []
        rings = len(axis)
        for ring in range(rings - 1):
            base_a = ring * radial_segments
            base_b = (ring + 1) * radial_segments
            for i in range(radial_segments):
                j = (i + 1) % radial_segments
                faces.append((base_a + i, base_a + j, base_b + j))
                faces.append((base_a + i, base_b + j, base_b + i))
                group = "nose" if ring < axial_segments - 1 else "body"
                groups.extend([group, group])

        # Fins: flat trapezoids, two triangles each, both faces kept.
        fin_x0 = (self.nose_length + self.body_length - self.fin_root_chord) - centre
        for k in range(max(0, int(self.fin_count))):
            angle = 2 * np.pi * k / max(1, int(self.fin_count))
            ux, uy = math.cos(angle), math.sin(angle)
            root_in = radius * 0.98
            tip = root_in + self.fin_span
            quad = [
                [fin_x0, root_in * ux, root_in * uy],
                [fin_x0 + self.fin_root_chord, root_in * ux, root_in * uy],
                [fin_x0 + self.fin_sweep + self.fin_tip_chord, tip * ux, tip * uy],
                [fin_x0 + self.fin_sweep, tip * ux, tip * uy],
            ]
            start = len(vertices)
            vertices.extend(quad)
            faces.append((start, start + 1, start + 2))
            faces.append((start, start + 2, start + 3))
            groups.extend(["fin", "fin"])

        return RocketMesh(
            vertices=np.asarray(vertices, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int32),
            groups=groups,
            length=self.total_length,
            diameter=self.body_diameter,
        )

    # ------------------------------------------------------------------ io

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RocketParams":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (payload or {}).items() if k in fields})


@dataclass
class RocketMesh:
    """Triangle soup with per-face group tags, ready for a painter's-algorithm view."""

    vertices: np.ndarray           # (V, 3)
    faces: np.ndarray              # (F, 3) indices
    groups: list[str] = field(default_factory=list)
    length: float = 1.0
    diameter: float = 0.1

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def face_centroids(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.where(lengths < 1e-12, 1.0, lengths)


# ------------------------------------------------------------------ flight


TELEMETRY_CHANNELS = (
    "altitude_m",
    "velocity_ms",
    "accel_ms2",
    "mach",
    "dynamic_pressure_pa",
    "mass_kg",
    "thrust_n",
    "drag_n",
    "chamber_pressure_bar",
    "nozzle_temp_c",
    "vibration_g",
    "airframe_temp_c",
    "battery_v",
)


@dataclass
class FlightResult:
    """A simulated flight as a channel table."""

    time: np.ndarray
    channels: dict[str, np.ndarray]
    apogee: float
    max_velocity: float
    max_mach: float
    burnout_time: float
    flight_time: float
    stability_margin: float
    # Ground-truth engine condition, 100 = as-new. Derived from accumulated nozzle
    # erosion. This is a *label* for supervised training -- it is deliberately not
    # one of `channels`, because feeding it as an input would leak the answer.
    health_truth: np.ndarray = field(default_factory=lambda: np.empty(0))

    @property
    def names(self) -> list[str]:
        return list(self.channels)

    def matrix(self) -> np.ndarray:
        """``(n_samples, n_channels)`` in :attr:`names` order."""
        return np.column_stack([self.channels[n] for n in self.names])

    def summary(self) -> str:
        return (
            f"apogee {self.apogee:.0f} m, max speed {self.max_velocity:.0f} m/s "
            f"(Mach {self.max_mach:.2f}), burnout {self.burnout_time:.1f} s, "
            f"flight {self.flight_time:.1f} s, stability {self.stability_margin:.2f} cal"
        )


def simulate(
    params: RocketParams,
    dt: float = 0.05,
    max_time: float = 400.0,
    seed: int = 0,
) -> FlightResult:
    """Fly the vehicle and return its telemetry.

    Vertical 3-DOF: thrust along the flight path, quadratic drag, mass depletion
    over the burn, exponential atmosphere, and a simple transonic drag rise. The
    launch angle tilts the trajectory so range is non-zero but the model stays
    one-dimensional along the flight path.
    """
    rng = np.random.default_rng(seed)
    angle = math.radians(max(0.0, min(params.launch_angle, 45.0)))
    vertical = math.cos(angle)

    burn = max(params.burn_time, 1e-3)
    propellant = max(params.propellant_mass, 0.0)
    burn_rate = propellant / burn
    area = params.frontal_area
    base_cd = params.effective_cd

    time_s = 0.0
    altitude = 0.0
    velocity = 0.0
    mass = params.wet_mass

    # Nozzle erosion accumulates over the burn and drives the health channels.
    erosion = 0.0
    rows: list[list[float]] = []
    times: list[float] = []
    erosion_track: list[float] = []
    burnout_time = burn
    max_steps = int(max_time / dt)

    for _ in range(max_steps):
        burning = time_s < burn and propellant > 0.0

        if burning:
            erosion += (params.degradation * dt / burn)
            # Erosion costs thrust; a worn nozzle delivers less.
            thrust = params.motor_thrust * (1.0 - 0.22 * erosion)
            # Slight taper toward burnout, as in a real regressive grain.
            thrust *= 1.0 - 0.12 * (time_s / burn)
            mass = max(params.dry_mass, mass - burn_rate * dt)
        else:
            thrust = 0.0
            if burnout_time == burn and time_s >= burn:
                burnout_time = time_s

        rho = float(air_density(altitude))
        sound = float(speed_of_sound(altitude))
        mach = abs(velocity) / max(sound, 1e-6)
        # Transonic drag rise, smoothed so the derivative stays sane.
        cd = base_cd * (1.0 + 1.15 * math.exp(-((mach - 1.05) ** 2) / 0.045))
        drag = 0.5 * rho * velocity * abs(velocity) * cd * area

        accel = (thrust - drag) / max(mass, 1e-6) - GRAVITY * vertical
        velocity += accel * dt
        altitude += velocity * vertical * dt
        time_s += dt

        if altitude < 0.0 and time_s > burn:
            break

        q = 0.5 * rho * velocity**2
        noise = params.noise

        # Engine-health channels: meaningful only while burning, then decay.
        if burning:
            chamber = 55.0 * (1.0 - 0.30 * erosion) * (1.0 - 0.10 * time_s / burn)
            nozzle = 820.0 + 420.0 * (time_s / burn) + 260.0 * erosion
            vibration = 3.0 + 9.0 * erosion + 2.5 * (q / 50000.0)
        else:
            settle = math.exp(-(time_s - burn) / 4.0)
            chamber = 1.0 + 3.0 * settle
            nozzle = 300.0 + 700.0 * settle
            vibration = 0.4 + 1.2 * settle + 0.8 * (q / 50000.0)

        airframe = 18.0 + 0.045 * q / 100.0 + 22.0 * erosion * (1.0 if burning else 0.4)
        battery = 12.4 - 0.02 * time_s - 0.35 * (1.0 if burning else 0.0)

        def jitter(value: float, scale: float = 1.0) -> float:
            return float(value * (1.0 + rng.normal(0.0, noise * scale)))

        erosion_track.append(erosion)
        rows.append([
            max(altitude, 0.0),
            velocity,
            accel,
            mach,
            q,
            mass,
            thrust,
            drag,
            jitter(max(chamber, 0.0)),
            jitter(nozzle, 0.6),
            jitter(max(vibration, 0.0), 1.8),
            jitter(airframe, 0.5),
            jitter(battery, 0.15),
        ])
        times.append(time_s)

        if altitude <= 0.0 and velocity < 0.0 and time_s > burn:
            break

    if not rows:
        rows = [[0.0] * len(TELEMETRY_CHANNELS)]
        times = [0.0]
        erosion_track = [0.0]

    table = np.asarray(rows, dtype=np.float64)
    channels = {name: table[:, i] for i, name in enumerate(TELEMETRY_CHANNELS)}
    altitudes = channels["altitude_m"]
    velocities = channels["velocity_ms"]

    # Erosion is unrecoverable damage, so health is monotone non-increasing.
    erosion_arr = np.maximum.accumulate(np.asarray(erosion_track, dtype=np.float64))
    health_truth = 100.0 * np.clip(1.0 - erosion_arr, 0.0, 1.0)

    return FlightResult(
        health_truth=health_truth,
        time=np.asarray(times, dtype=np.float64),
        channels=channels,
        apogee=float(altitudes.max()),
        max_velocity=float(velocities.max()),
        max_mach=float(channels["mach"].max()),
        burnout_time=float(burnout_time),
        flight_time=float(times[-1]),
        stability_margin=float(params.stability_margin()),
    )
