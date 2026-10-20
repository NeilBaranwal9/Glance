####################################################################################################
"""
Interactive 3-D rocket viewer.

Renders the parametric airframe from app.core.rocket_sim with QPainter rather
than OpenGL: a painter's-algorithm pass over depth-sorted triangles with
back-face culling and Lambertian shading. That choice is deliberate -- it needs
no GPU, no driver, and no PyOpenGL, so it renders identically on any machine the
app starts on, and it can be verified in a headless test.

Mouse: drag to orbit, wheel to zoom, right-drag to pan, double-click to reset.
"""
####################################################################################################
# Imports

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.core.rocket_sim import RocketMesh, RocketParams

####################################################################################################

_BG_TOP = QColor("#22242a")
_BG_BOTTOM = QColor("#15161a")
_TEXT_DIM = QColor("#8e8e93")
_TEXT = QColor("#ffffff")

# Base colour per mesh group; shading multiplies these.
_GROUP_COLORS = {
    "nose": QColor("#d8dde6"),
    "body": QColor("#aeb6c2"),
    "fin": QColor("#0a84ff"),
}

_LIGHT = np.array([0.42, 0.72, 0.55])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)

_AMBIENT = 0.34
_MIN_ZOOM = 0.25
_MAX_ZOOM = 8.0


class Rocket3DView(QWidget):
    """Orbit-camera view of a :class:`RocketMesh`."""

    parameters_changed = Signal()

    def __init__(self, params: Optional[RocketParams] = None, parent=None):
        super().__init__(parent)
        self.setMinimumSize(260, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        self._params = params or RocketParams()
        self._mesh: Optional[RocketMesh] = None

        # Camera: yaw/pitch in degrees, zoom is a scale multiplier.
        self.yaw = -34.0
        self.pitch = 16.0
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self._drag_from: Optional[QPointF] = None
        self._drag_button: Optional[Qt.MouseButton] = None

        self._show_caption = True
        self.rebuild()

    # ------------------------------------------------------------------ model

    @property
    def params(self) -> RocketParams:
        return self._params

    def set_params(self, params: RocketParams) -> None:
        """Swap in new dimensions and re-tessellate."""
        self._params = params
        self.rebuild()
        self.parameters_changed.emit()

    def rebuild(self) -> None:
        try:
            self._mesh = self._params.build_mesh(radial_segments=40, axial_segments=30)
        except Exception:  # noqa: BLE001 - a bad parameter set must not kill the view
            self._mesh = None
        self.update()

    def reset_view(self) -> None:
        self.yaw, self.pitch, self.zoom = -34.0, 16.0, 1.0
        self.pan = QPointF(0.0, 0.0)
        self.update()

    def set_caption_visible(self, visible: bool) -> None:
        self._show_caption = visible
        self.update()

    # ------------------------------------------------------------------ camera

    def _rotation(self) -> np.ndarray:
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        # Rocket's long axis is +X; show it nose-up by rotating into screen space.
        about_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        about_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        # Mesh runs nose (-X) to tail (+X); map -X to screen-up so it stands nose-up.
        upright = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return about_y @ about_z @ upright

    def _project(self, points: np.ndarray, scale: float) -> np.ndarray:
        """World -> screen. Weak perspective keeps it readable without distortion."""
        centre_x, centre_y = self.width() / 2.0, self.height() / 2.0
        depth = points[:, 2]
        factor = 1.0 / (1.0 + 0.16 * depth / max(self._extent(), 1e-6))
        xs = centre_x + self.pan.x() + points[:, 0] * scale * factor
        ys = centre_y + self.pan.y() - points[:, 1] * scale * factor
        return np.column_stack([xs, ys])

    def _extent(self) -> float:
        if self._mesh is None or len(self._mesh.vertices) == 0:
            return 1.0
        return float(np.abs(self._mesh.vertices).max()) or 1.0

    # ------------------------------------------------------------------ painting

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_background(painter)

        mesh = self._mesh
        if mesh is None or mesh.n_faces == 0:
            self._paint_message(painter, "Invalid rocket geometry")
            painter.end()
            return

        rotation = self._rotation()
        rotated = mesh.vertices @ rotation.T

        span = max(self._extent() * 2.2, 1e-6)
        scale = (min(self.width(), self.height()) / span) * self.zoom
        screen = self._project(rotated, scale)

        tri_idx = mesh.faces
        tri_world = rotated[tri_idx]
        normals = np.cross(
            tri_world[:, 1] - tri_world[:, 0], tri_world[:, 2] - tri_world[:, 0]
        )
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms < 1e-12, 1.0, norms)

        depth = tri_world[:, :, 2].mean(axis=1)
        facing = normals[:, 2]

        # Fins are thin plates, so they must render from both sides; the hull is
        # closed, so its back faces can be culled.
        is_fin = np.array([g == "fin" for g in mesh.groups], dtype=bool)
        visible = (facing > 0.0) | is_fin
        order = np.argsort(depth)
        order = order[visible[order]]

        shade = np.abs(normals @ _LIGHT)
        intensity = np.clip(_AMBIENT + (1.0 - _AMBIENT) * shade, 0.0, 1.0)

        painter.setPen(Qt.PenStyle.NoPen)
        for face in order:
            group = mesh.groups[face] if face < len(mesh.groups) else "body"
            base = _GROUP_COLORS.get(group, _GROUP_COLORS["body"])
            k = float(intensity[face])
            colour = QColor(
                int(base.red() * k), int(base.green() * k), int(base.blue() * k)
            )
            polygon = QPolygonF([QPointF(*screen[v]) for v in tri_idx[face]])
            painter.setBrush(QBrush(colour))
            painter.drawPolygon(polygon)

        if self._show_caption:
            self._paint_caption(painter)
        painter.end()

    def _paint_background(self, painter: QPainter) -> None:
        from PySide6.QtGui import QLinearGradient

        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, _BG_TOP)
        gradient.setColorAt(1.0, _BG_BOTTOM)
        painter.fillRect(self.rect(), QBrush(gradient))

    def _paint_message(self, painter: QPainter, text: str) -> None:
        painter.setPen(QPen(_TEXT_DIM))
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)

    def _paint_caption(self, painter: QPainter) -> None:
        p = self._params
        stability = p.stability_margin()
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(_TEXT_DIM))
        left = 10
        y = self.height() - 40
        painter.drawText(
            left, y,
            f"L {p.total_length * 100:.0f} cm   Ø {p.body_diameter * 100:.1f} cm   "
            f"{p.fin_count} fins   {p.nose_shape}",
        )
        # Stability is the one number that can be wrong in a dangerous way.
        if stability < 1.0:
            painter.setPen(QPen(QColor("#ff453a") if stability < 0.5 else QColor("#ff9f0a")))
        else:
            painter.setPen(QPen(QColor("#30d158")))
        painter.drawText(left, y + 14, f"stability {stability:.2f} cal"
                         + ("  UNSTABLE" if stability < 0.5 else ""))
        painter.setPen(QPen(_TEXT_DIM))
        painter.drawText(left, y + 28, "drag to orbit · wheel to zoom · double-click to reset")

    # ------------------------------------------------------------------ input

    def mousePressEvent(self, event):
        self._drag_from = QPointF(event.position())
        self._drag_button = event.button()
        self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._drag_from is None:
            return
        current = QPointF(event.position())
        delta = current - self._drag_from
        self._drag_from = current
        if self._drag_button == Qt.MouseButton.RightButton:
            self.pan += delta
        else:
            self.yaw += delta.x() * 0.45
            self.pitch = max(-89.0, min(89.0, self.pitch + delta.y() * 0.45))
        self.update()

    def mouseReleaseEvent(self, event):
        self._drag_from = None
        self._drag_button = None
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        self.zoom = float(np.clip(self.zoom * (1.14**steps), _MIN_ZOOM, _MAX_ZOOM))
        self.update()

    def sizeHint(self):
        from PySide6.QtCore import QSize

        return QSize(320, 360)
