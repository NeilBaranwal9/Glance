"""Build the pretrained model bundle that ships with the app.

    python ml/build_pretrained.py

Trains the four algorithms on the full NASA C-MAPSS FD001 training split using
the *same* generic, N-channel trainer the app uses for user-supplied data
(``app.ml.trainers.train_all``), then writes the result to
``app/models/pretrained_cmapss/``.

Using one trainer for both paths is the point: the bundled benchmark model and a
model the user trains on their own CSV are the same kind of object, load through
the same code, and run through the same online engine.

A second bundle is written for the rocket flight simulator
(``app/models/pretrained_rocket/``) so the Rocket Designer source has a model
matching its channel set out of the box.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.rocket_sim import RocketParams, simulate  # noqa: E402
from app.ml.dataset import load_table  # noqa: E402
from app.ml.persist import save_bundle  # noqa: E402
from app.ml.trainers import train_all  # noqa: E402
from ml.common import ARTIFACT_DIR, rel  # noqa: E402

CMAPSS_BUNDLE = ARTIFACT_DIR / "pretrained_cmapss"
ROCKET_BUNDLE = ARTIFACT_DIR / "pretrained_rocket"


def _progress(message: str, fraction: float) -> None:
    print(f"  [{fraction * 100:5.1f}%] {message}")


def build_cmapss(engines: int | None = None) -> None:
    print("=== pretrained bundle: NASA C-MAPSS FD001 ===")
    dataset = load_table("data/cmapss/train_FD001.txt")
    units = dataset.units()
    values = dataset.sensor_values()
    cycles = dataset.cycles()
    rul = dataset.rul(ceiling=125.0)

    if engines:
        keep = np.isin(units, np.unique(units)[:engines])
        values, units, cycles, rul = values[keep], units[keep], cycles[keep], rul[keep]

    print(f"  {len(values)} rows, {values.shape[1]} raw channels, "
          f"{len(np.unique(units))} engines, supervised RUL labels")
    models = train_all(
        values, dataset.roles.sensors, units=units, cycles=cycles, rul=rul,
        source="NASA C-MAPSS FD001 (run-to-failure, supervised)",
        progress=_progress,
    )
    path = save_bundle(models, CMAPSS_BUNDLE)
    print(f"  -> {rel(path)}")
    print("  " + models.summary().replace("\n", "\n  "))
    print("  top drivers: " + ", ".join(
        f"{d['feature']} ({d['importance']:.0f})" for d in models.top_drivers(5)))


def build_rocket(flights: int = 14) -> None:
    """Train on a spread of simulated flights so the model covers the design space."""
    print("\n=== pretrained bundle: rocket flight simulation ===")
    rng = np.random.default_rng(7)
    blocks, units_list, cycles_list, truth_list, nominal_list = [], [], [], [], []
    names = None

    for flight in range(flights):
        # Half the fleet is near-healthy so the detector has a real reference.
        degradation = 0.0 + rng.uniform(0.0, 0.10) if flight % 2 == 0 else rng.uniform(0.25, 0.95)
        params = RocketParams(
            body_diameter=float(rng.uniform(0.07, 0.16)),
            body_length=float(rng.uniform(0.7, 1.8)),
            nose_length=float(rng.uniform(0.2, 0.5)),
            fin_count=int(rng.integers(3, 7)),
            fin_span=float(rng.uniform(0.08, 0.22)),
            dry_mass=float(rng.uniform(2.0, 6.0)),
            propellant_mass=float(rng.uniform(0.6, 2.4)),
            motor_thrust=float(rng.uniform(200.0, 700.0)),
            burn_time=float(rng.uniform(2.0, 6.0)),
            # Spread of engine wear so the detector sees healthy and worn motors.
            degradation=float(degradation),
        )
        result = simulate(params, seed=flight)
        matrix = result.matrix()
        if names is None:
            names = result.names
        blocks.append(matrix)
        units_list.append(np.full(len(matrix), flight, dtype=np.int64))
        cycles_list.append(np.arange(len(matrix), dtype=np.float64))
        truth_list.append(result.health_truth)
        # The autoencoder's "nominal" reference is every sample of a healthy-motor
        # flight -- NOT the first 30% of every flight. A rocket's opening phase is
        # the boost, not a nominal baseline, so the usual leading-window rule would
        # calibrate against the wrong thing and flag ordinary coast as anomalous.
        nominal_list.append(np.full(len(matrix), degradation < 0.12, dtype=bool))

    values = np.vstack(blocks)
    units = np.concatenate(units_list)
    cycles = np.concatenate(cycles_list)
    truth = np.concatenate(truth_list)
    nominal = np.concatenate(nominal_list)
    print(f"  {len(values)} samples from {flights} flights, {values.shape[1]} channels")
    print(f"  supervised on true nozzle condition; {int(nominal.sum())} samples from "
          f"healthy-motor flights form the autoencoder baseline")

    models = train_all(
        values, names, units=units, cycles=cycles,
        health_truth=truth, nominal_mask=nominal,
        source="rocket flight simulation (supervised on nozzle condition)",
        progress=_progress,
    )
    path = save_bundle(models, ROCKET_BUNDLE)
    print(f"  -> {rel(path)}")
    print("  " + models.summary().replace("\n", "\n  "))
    print("  top drivers: " + ", ".join(
        f"{d['feature']} ({d['importance']:.0f})" for d in models.top_drivers(5)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engines", type=int, default=None,
                    help="limit C-MAPSS to the first N engines (faster, for testing)")
    ap.add_argument("--flights", type=int, default=14,
                    help="number of simulated flights for the rocket bundle")
    ap.add_argument("--only", choices=("cmapss", "rocket"), default=None)
    args = ap.parse_args(argv)

    if args.only != "rocket":
        build_cmapss(args.engines)
    if args.only != "cmapss":
        build_rocket(args.flights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
