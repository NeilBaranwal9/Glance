####################################################################################################
"""
                                        ::
                                        ::
                                        ::
                                        ::
                                        ::
    ..    ..........    :.      ::      ::     .........  ..    ..........    ...      .
    ::    ::            : .:.   ::     .::.       ::      ::    ::       :    :: :.    :
    ::    ::   ..:::    :   .:. ::    ::::::      ::      ::    ::       :    ::   ::  :
    ::    ::......::    :      :::    ::::::      ::      ::    ::.......:    ::     :::
                                      ::::::
                                      :.::.:
                         .::::          ::          ::::.
                       .::::::::.       ::       .:::::::::
                       ::::::::::::....::::.....:::::::::::
                        .:::::::::::::::::::::::::::::::::.

                    Copyright (c) 2025 Ignition Software Department

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3, with the additional restriction
that this software may not be used for commercial purposes without
explicit written permission from the authors.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
####################################################################################################
# File:        ml_analysis_page.py
#
# @brief       Live predictive-maintenance analysis over an arbitrary sensor set.
# @details     Runs four algorithms - autoencoder, Gaussian Process, LSTM and
#              CatBoost - on a live stream, one sample at a time. The channel set
#              is whatever the attached source provides, so nothing is hardcoded
#              to a fixed sensor count.
#
#              Model: a pretrained bundle that ships with the app, or one trained
#              here on a dataset the user supplies.
#              Source: the app's live telemetry, the built-in rocket flight
#              simulation, or a dataset replay.
#
#              The Rocket Designer drives the simulation, so changing the airframe
#              changes the flight, which changes every algorithm's output.
####################################################################################################

####################################################################################################
# Imports

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from PySide6.QtCore import QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QConicalGradient, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg

from app.core.rocket_sim import NOSE_SHAPES, RocketParams, simulate
from app.widgets.rocket3d import Rocket3DView

####################################################################################################

# ---------------------------------------------------------------------------
# Colour constants -- consistent with the Glance dark theme
# ---------------------------------------------------------------------------
_BG_PRIMARY = "#1c1c1e"
_BG_CARD = "#2c2c2e"
_ACCENT = "#0a84ff"
_ACCENT_HOVER = "#0077ed"
_TEXT = "#ffffff"
_TEXT_DIM = "#8e8e93"
_BORDER = "rgba(255, 255, 255, 0.1)"
_SUCCESS = "#30d158"
_WARNING = "#ff9f0a"
_DANGER = "#ff453a"

_ACTUAL_COLOR = "#00e5ff"
_FORECAST_COLOR = "#ff9f0a"
_BAND_COLOR = (255, 159, 10, 55)
_HEALTH_COLOR = "#30d158"
_ANOMALY_COLOR = "#ff453a"

_STAGE_COLORS = [_SUCCESS, _WARNING, _DANGER]

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
# Order matters: the first available entry is loaded on open, and it must match
# the default source ("Rocket flight simulation") so the page is usable at once.
PRETRAINED = {
    "Pretrained · Rocket flight": MODELS_DIR / "pretrained_rocket",
    "Pretrained · NASA C-MAPSS turbofan": MODELS_DIR / "pretrained_cmapss",
}
TRAIN_ENTRY = "Train on my dataset…"

_MAX_PLOT_POINTS = 1500


####################################################################################################
# Training worker -- keeps model fitting off the UI thread
####################################################################################################

class TrainingWorker(QThread):
    """Runs ``train_all`` in the background, reporting progress."""

    progressed = Signal(str, float)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self._payload = payload
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from app.ml.trainers import train_all

            models = train_all(
                progress=lambda m, f: self.progressed.emit(m, f),
                is_cancelled=lambda: self._cancel,
                **self._payload,
            )
            if self._cancel:
                self.failed.emit("Training cancelled.")
            else:
                self.finished_ok.emit(models)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc(limit=3)}")


####################################################################################################
# Small presentational widgets
####################################################################################################

class MetricCard(QFrame):
    """A headline readout: caption, big value, sub-caption."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("MetricCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(3)

        self._label = QLabel(label)
        self._label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px; font-weight: 600;")
        layout.addWidget(self._label)

        self._value = QLabel("—")
        self._value.setStyleSheet(f"color: {_TEXT}; font-size: 21px; font-weight: 700;")
        layout.addWidget(self._value)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        layout.addWidget(self._detail)

        self.setStyleSheet(f"""
            #MetricCard {{
                background-color: {_BG_CARD};
                border: 1px solid {_BORDER};
                border-radius: 10px;
            }}
        """)

    def set_value(self, text: str, color: str = _TEXT):
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color}; font-size: 21px; font-weight: 700;")

    def set_detail(self, text: str):
        self._detail.setText(text)


class HealthGauge(QWidget):
    """Circular 0-100 health gauge; high is good, so the bands are inverted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 100.0
        self.setMinimumSize(180, 158)

    def set_value(self, value: float):
        self.value = float(max(0.0, min(100.0, value)))
        self.update()

    def band_color(self) -> QColor:
        if self.value < 24.0:
            return QColor(_DANGER)
        if self.value <= 60.0:
            return QColor(_WARNING)
        return QColor(_SUCCESS)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        side = min(width, height)
        rect = QRectF((width - side) / 2 + 12, (height - side) / 2 + 12, side - 24, side - 24)

        painter.setPen(QPen(QColor(255, 255, 255, 20), 9,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 135 * 16, 270 * 16)

        colour = self.band_color()
        gradient = QConicalGradient(rect.center(), 225)
        gradient.setColorAt(0.0, colour.darker(150))
        gradient.setColorAt(0.5, colour)
        gradient.setColorAt(1.0, colour.lighter(130))
        painter.setPen(QPen(QBrush(gradient), 9,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 135 * 16, int(-(self.value / 100.0) * 270 * 16))

        painter.setPen(QColor(_TEXT))
        painter.setFont(QFont("Segoe UI", 26, QFont.Weight.Bold))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.value:.0f}")
        painter.setPen(QColor(_TEXT_DIM))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Medium))
        painter.drawText(QRectF(rect.left(), rect.center().y() + 24, rect.width(), 18),
                         Qt.AlignmentFlag.AlignCenter, "HEALTH SCORE")
        painter.end()


class StageIndicator(QFrame):
    """Three-colour LED for the predicted degradation stage."""

    def __init__(self, labels: List[str], parent=None):
        super().__init__(parent)
        self._labels = list(labels)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("DEGRADATION STAGE")
        title.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self._led = QLabel("—")
        self._led.setFixedSize(150, 42)
        self._led.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._led.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        layout.addWidget(self._led, alignment=Qt.AlignmentFlag.AlignCenter)

        self._sub = QLabel("")
        self._sub.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._sub)

        self.setStyleSheet("background: transparent;")
        self.set_stage(None)

    def set_labels(self, labels: List[str]):
        self._labels = list(labels)

    def set_stage(self, stage: Optional[int], confidence: Optional[float] = None):
        if stage is None:
            self._led.setText("—")
            self._led.setStyleSheet("border-radius: 21px; background: #2a2a2a; "
                                    "border: 2px solid #444; color: #666;")
            self._led.setGraphicsEffect(None)
            self._sub.setText("")
            return
        colour = _STAGE_COLORS[stage % len(_STAGE_COLORS)]
        name = self._labels[stage] if stage < len(self._labels) else str(stage)
        swatch = QColor(colour)
        brightness = (swatch.red() * 299 + swatch.green() * 587 + swatch.blue() * 114) / 1000
        text_colour = "#000000" if brightness > 128 else "#ffffff"
        self._led.setText(name.upper())
        self._led.setStyleSheet(
            f"border-radius: 21px; background-color: {colour}; "
            f"border: 2px solid {colour}; color: {text_colour};")
        effect = self._led.graphicsEffect()
        if effect is None:
            glow = QGraphicsDropShadowEffect()
            glow.setBlurRadius(20)
            glow.setColor(QColor(colour))
            glow.setOffset(0, 0)
            self._led.setGraphicsEffect(glow)
        else:
            effect.setColor(QColor(colour))
        if confidence is not None:
            self._sub.setText(f"{confidence * 100:.0f}% confidence")


class Readout(QFrame):
    """Compact label/value row."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)
        name = QLabel(label)
        name.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        layout.addWidget(name)
        layout.addStretch()
        self._value = QLabel("—")
        self._value.setStyleSheet(f"color: {_TEXT}; font-size: 12px; font-weight: 600;")
        layout.addWidget(self._value)
        self.setStyleSheet("background: transparent;")

    def set_value(self, text: str, color: str = _TEXT):
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: 600;")


####################################################################################################
# MLAnalysisPage
####################################################################################################

class MLAnalysisPage(QWidget):
    """
    @brief  Live ML analysis over an arbitrary number of sensor channels.
    @details Self-contained page holding an OnlineMLEngine, a source that feeds it
             samples, and the widgets that render each algorithm's output.
    """

    close_requested = Signal()
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("MLAnalysisPage")

        self._engine = None            # app.ml.online.OnlineMLEngine (lazy)
        self._models = None            # app.ml.trainers.TrainedModels
        self._worker: Optional[TrainingWorker] = None
        self._current_label = ""

        self._source_kind = "rocket"   # 'live' | 'rocket' | 'replay'
        self._running = False
        self._stream_rows: Optional[np.ndarray] = None
        self._stream_position = 0
        self._live_channels: List[str] = []
        self._last_flight = None
        self._rocket_params = RocketParams()
        self._dropped_notice = ""

        self._stream_timer = QTimer(self)
        self._stream_timer.setInterval(40)          # simulated/replay feed
        self._stream_timer.timeout.connect(self._tick_stream)

        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(100)         # decouple painting from data rate
        self._redraw_timer.timeout.connect(self._refresh_views)

        self._build_ui()
        self._on_rocket_param_changed()   # populate the flight plot straight away
        QTimer.singleShot(0, self._load_default_model)

    # ==================================================================
    # Lazy engine -- torch/catboost stay unimported until the tab is used
    # ==================================================================

    def _ensure_engine(self):
        if self._engine is None:
            from app.ml.online import OnlineMLEngine

            self._engine = OnlineMLEngine(gp_every=5)
        return self._engine

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self):
        self.setStyleSheet(f"""
            #MLAnalysisPage {{ background-color: {_BG_PRIMARY}; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {_BG_CARD}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 7px;
                padding: 5px 9px; font-size: 12px; min-height: 18px;
            }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{
                background-color: {_BG_CARD}; color: {_TEXT};
                selection-background-color: {_ACCENT}; border: 1px solid {_BORDER};
            }}
            QPushButton#PrimaryButton {{
                background-color: {_ACCENT}; color: white; border: none;
                border-radius: 7px; padding: 7px 18px; font-size: 12px; font-weight: 600;
            }}
            QPushButton#PrimaryButton:hover {{ background-color: {_ACCENT_HOVER}; }}
            QPushButton#PrimaryButton:disabled {{ background-color: #3a3a3c; color: #7a7a7e; }}
            QPushButton#GhostButton {{
                background-color: transparent; color: {_TEXT_DIM};
                border: 1px solid {_BORDER}; border-radius: 7px;
                padding: 7px 14px; font-size: 12px;
            }}
            QPushButton#GhostButton:hover {{ color: {_TEXT}; }}
            QTabWidget::pane {{ border: 1px solid {_BORDER}; border-radius: 10px; }}
            QTabBar::tab {{
                background: transparent; color: {_TEXT_DIM};
                padding: 7px 16px; border: none; font-size: 12px;
            }}
            QTabBar::tab:selected {{ color: {_TEXT}; border-bottom: 2px solid {_ACCENT}; }}
            QTextEdit {{
                background-color: {_BG_CARD}; color: {_TEXT};
                border: 1px solid {_BORDER}; border-radius: 8px; font-size: 12px;
            }}
            QProgressBar {{
                background-color: {_BG_CARD}; border: 1px solid {_BORDER};
                border-radius: 6px; height: 6px; text-align: center; color: transparent;
            }}
            QProgressBar::chunk {{ background-color: {_ACCENT}; border-radius: 6px; }}
            QScrollArea {{ background: transparent; border: none; }}
            QLabel {{ color: {_TEXT}; }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        root.addWidget(self._build_header())
        root.addWidget(self._build_controls())
        root.addWidget(self._build_kpi_row())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(f"QSplitter::handle {{ background-color: {_BORDER}; }}")
        splitter.addWidget(self._build_tabs())
        splitter.addWidget(self._build_status_column())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([940, 300])
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------ header

    def _build_header(self) -> QWidget:
        bar = QWidget()
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)

        text = QWidget()
        text_layout = QVBoxLayout(text)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)
        title = QLabel("ML Analysis — Live Predictive Maintenance")
        title.setStyleSheet(f"color: {_TEXT}; font-size: 18px; font-weight: 700;")
        text_layout.addWidget(title)
        self._subtitle = QLabel("Loading model…")
        self._subtitle.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        text_layout.addWidget(self._subtitle)
        layout.addWidget(text)
        layout.addStretch()

        close_button = QPushButton("Back to Dashboard")
        close_button.setObjectName("GhostButton")
        close_button.clicked.connect(self._on_close)
        layout.addWidget(close_button)
        return bar

    # ------------------------------------------------------------------ controls

    def _build_controls(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("ControlPanel")
        panel.setStyleSheet(f"""
            #ControlPanel {{
                background-color: {_BG_CARD}; border: 1px solid {_BORDER};
                border-radius: 10px;
            }}
        """)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(12, 9, 12, 9)
        outer.setSpacing(7)

        row = QHBoxLayout()
        row.setSpacing(8)

        row.addWidget(self._caption("Model"))
        self._model_combo = QComboBox()
        for name in PRETRAINED:
            self._model_combo.addItem(name)
        self._model_combo.addItem(TRAIN_ENTRY)
        self._model_combo.setMinimumWidth(230)
        self._model_combo.activated.connect(self._on_model_selected)
        row.addWidget(self._model_combo)

        row.addWidget(self._caption("Source"))
        self._source_combo = QComboBox()
        self._source_combo.addItem("Rocket flight simulation", "rocket")
        self._source_combo.addItem("Live telemetry", "live")
        self._source_combo.addItem("Dataset replay", "replay")
        self._source_combo.setMinimumWidth(165)
        self._source_combo.activated.connect(self._on_source_selected)
        row.addWidget(self._source_combo)

        row.addWidget(self._caption("Forecast channel"))
        self._channel_combo = QComboBox()
        self._channel_combo.setMinimumWidth(175)
        self._channel_combo.activated.connect(self._on_channel_selected)
        row.addWidget(self._channel_combo)

        row.addStretch()

        self._start_button = QPushButton("▶  Start")
        self._start_button.setObjectName("PrimaryButton")
        self._start_button.clicked.connect(self.toggle_running)
        row.addWidget(self._start_button)

        self._reset_button = QPushButton("↺  Reset")
        self._reset_button.setObjectName("GhostButton")
        self._reset_button.clicked.connect(self.reset_run)
        row.addWidget(self._reset_button)
        outer.addLayout(row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        self._progress_label.setVisible(False)
        outer.addWidget(self._progress_label)
        return panel

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px; font-weight: 600;")
        return label

    # ------------------------------------------------------------------ KPI row

    def _build_kpi_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        self._kpi = {
            "autoencoder": MetricCard("Autoencoder · Anomaly Detection"),
            "gpr": MetricCard("Gaussian Process · Forecast"),
            "lstm": MetricCard("LSTM · Stage Classification"),
            "catboost": MetricCard("CatBoost · Health Score"),
        }
        for card in self._kpi.values():
            layout.addWidget(card)
        return row

    # ------------------------------------------------------------------ tabs

    def _build_tabs(self) -> QWidget:
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_live_tab(), "Live Analysis")
        self._tabs.addTab(self._build_rocket_tab(), "Rocket Designer")
        self._tabs.addTab(self._build_insights_tab(), "Model & Insights")
        return self._tabs

    def _style_plot(self, widget: pg.PlotWidget, title: str, y_label: str, x_label: str):
        widget.setBackground(_BG_CARD)
        widget.showGrid(x=True, y=True, alpha=0.08)
        widget.setAntialiasing(True)
        widget.setMenuEnabled(False)
        axis_pen = pg.mkPen(color=QColor(255, 255, 255, 80), width=1)
        for name in ("bottom", "left"):
            axis = widget.getAxis(name)
            axis.setPen(axis_pen)
            axis.setTextPen(QColor(255, 255, 255, 150))
        widget.setTitle(title, color="#ffffff", size="10pt")
        widget.setLabel("left", y_label, color="#aaaaaa")
        widget.setLabel("bottom", x_label, color="#aaaaaa")

    def _build_live_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        self._main_plot = pg.PlotWidget()
        self._style_plot(self._main_plot, "Measured vs Predicted", "value", "sample")
        self._main_plot.addLegend(offset=(10, 8), labelTextColor="#cccccc")

        self._band_upper = self._main_plot.plot(pen=pg.mkPen(None))
        self._band_lower = self._main_plot.plot(pen=pg.mkPen(None))
        self._band_fill = pg.FillBetweenItem(self._band_upper, self._band_lower,
                                             brush=pg.mkBrush(*_BAND_COLOR))
        self._main_plot.addItem(self._band_fill)
        self._forecast_curve = self._main_plot.plot(
            pen=pg.mkPen(_FORECAST_COLOR, width=2, style=Qt.PenStyle.DashLine),
            name="GP forecast + 95% band")
        self._actual_curve = self._main_plot.plot(
            pen=pg.mkPen(_ACTUAL_COLOR, width=2), name="Measured")
        self._anomaly_scatter = pg.ScatterPlotItem(
            size=9, symbol="x", pen=pg.mkPen("#ffffff", width=1),
            brush=pg.mkBrush(_ANOMALY_COLOR))
        self._main_plot.addItem(self._anomaly_scatter)
        layout.addWidget(self._main_plot, stretch=3)

        self._health_plot = pg.PlotWidget()
        self._style_plot(self._health_plot, "Predicted Health", "health (0-100)", "sample")
        self._health_plot.setYRange(0, 105, padding=0)
        self._health_plot.setXLink(self._main_plot)
        for value, colour in ((60.0, _WARNING), (24.0, _DANGER)):
            self._health_plot.addItem(pg.InfiniteLine(
                pos=value, angle=0,
                pen=pg.mkPen(colour, width=1, style=Qt.PenStyle.DotLine)))
        self._health_curve = self._health_plot.plot(pen=pg.mkPen(_HEALTH_COLOR, width=2))
        self._health_anomaly = pg.ScatterPlotItem(
            size=7, pen=pg.mkPen(None), brush=pg.mkBrush(_ANOMALY_COLOR))
        self._health_plot.addItem(self._health_anomaly)
        layout.addWidget(self._health_plot, stretch=2)
        return page

    def _build_rocket_tab(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        form_host = QScrollArea()
        form_host.setWidgetResizable(True)
        form_host.setFrameShape(QFrame.Shape.NoFrame)
        form_host.setMaximumWidth(290)
        # The viewport paints its own background and ignores the parent stylesheet,
        # so it renders white unless told otherwise explicitly.
        form_host.viewport().setAutoFillBackground(False)
        form_host.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget "
                                "{ background: transparent; border: none; }")
        inner = QWidget()
        inner.setAutoFillBackground(False)
        inner.setStyleSheet("background: transparent;")
        form = QFormLayout(inner)
        form.setContentsMargins(2, 2, 8, 2)
        form.setSpacing(5)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self._spins: Dict[str, QDoubleSpinBox] = {}

        def spin(key, lo, hi, value, step, suffix="", decimals=3):
            box = QDoubleSpinBox()
            box.setRange(lo, hi)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setValue(value)
            box.setSuffix(suffix)
            box.valueChanged.connect(self._on_rocket_param_changed)
            self._spins[key] = box
            return box

        params = self._rocket_params
        form.addRow(self._caption("AIRFRAME"), QLabel(""))
        self._nose_combo = QComboBox()
        self._nose_combo.addItems(list(NOSE_SHAPES))
        self._nose_combo.setCurrentText(params.nose_shape)
        self._nose_combo.activated.connect(self._on_rocket_param_changed)
        form.addRow("Nose shape", self._nose_combo)
        form.addRow("Nose length", spin("nose_length", 0.05, 1.5, params.nose_length, 0.02, " m"))
        form.addRow("Body ø", spin("body_diameter", 0.03, 0.40,
                                        params.body_diameter, 0.005, " m"))
        form.addRow("Body length", spin("body_length", 0.2, 4.0, params.body_length, 0.05, " m"))

        form.addRow(self._caption("FINS"), QLabel(""))
        self._fin_count = QSpinBox()
        self._fin_count.setRange(0, 8)
        self._fin_count.setValue(params.fin_count)
        self._fin_count.valueChanged.connect(self._on_rocket_param_changed)
        form.addRow("Fin count", self._fin_count)
        form.addRow("Fin span", spin("fin_span", 0.01, 0.6, params.fin_span, 0.01, " m"))
        form.addRow("Root chord", spin("fin_root_chord", 0.02, 0.8,
                                       params.fin_root_chord, 0.01, " m"))
        form.addRow("Tip chord", spin("fin_tip_chord", 0.0, 0.8,
                                      params.fin_tip_chord, 0.01, " m"))
        form.addRow("Sweep", spin("fin_sweep", 0.0, 0.5, params.fin_sweep, 0.01, " m"))

        form.addRow(self._caption("MASS & MOTOR"), QLabel(""))
        form.addRow("Dry mass", spin("dry_mass", 0.2, 50.0, params.dry_mass, 0.25, " kg", 2))
        form.addRow("Propellant", spin("propellant_mass", 0.05, 30.0,
                                       params.propellant_mass, 0.1, " kg", 2))
        form.addRow("Thrust", spin("motor_thrust", 20.0, 5000.0,
                                   params.motor_thrust, 25.0, " N", 1))
        form.addRow("Burn time", spin("burn_time", 0.3, 30.0, params.burn_time, 0.2, " s", 2))
        form.addRow("Launch angle", spin("launch_angle", 0.0, 45.0,
                                         params.launch_angle, 1.0, "°", 1))

        form.addRow(self._caption("ENGINE CONDITION"), QLabel(""))
        form.addRow("Nozzle wear", spin("degradation", 0.0, 1.0, params.degradation, 0.05, "", 2))
        form.addRow("Sensor noise", spin("noise", 0.0, 0.2, params.noise, 0.005, "", 3))

        self._flight_summary = QLabel("—")
        self._flight_summary.setWordWrap(True)
        self._flight_summary.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 11px;")
        form.addRow(self._flight_summary)

        form_host.setWidget(inner)
        layout.addWidget(form_host)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(7)

        self._rocket_view = Rocket3DView(self._rocket_params)
        right_layout.addWidget(self._rocket_view, stretch=3)

        self._flight_plot = pg.PlotWidget()
        self._style_plot(self._flight_plot, "Simulated Flight Profile",
                         "altitude (m)", "time (s)")
        self._altitude_curve = self._flight_plot.plot(pen=pg.mkPen(_ACTUAL_COLOR, width=2))
        self._flight_marker = pg.InfiniteLine(
            pos=0, angle=90, pen=pg.mkPen("#ffffff", width=1, style=Qt.PenStyle.DotLine))
        self._flight_plot.addItem(self._flight_marker)
        right_layout.addWidget(self._flight_plot, stretch=2)

        layout.addWidget(right, stretch=1)
        return page

    def _build_insights_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(7)

        self._insights = QTextEdit()
        self._insights.setReadOnly(True)
        layout.addWidget(self._insights, stretch=3)

        self._importance_plot = pg.PlotWidget()
        self._style_plot(self._importance_plot,
                         "Which channels drive the health score", "importance (%)", "")
        self._importance_bars = pg.BarGraphItem(x=[], height=[], width=0.6,
                                                brush=pg.mkBrush(_ACCENT))
        self._importance_plot.addItem(self._importance_bars)
        layout.addWidget(self._importance_plot, stretch=2)
        return page

    # ------------------------------------------------------------------ status column

    def _build_status_column(self) -> QWidget:
        column = QFrame()
        column.setObjectName("StatusColumn")
        column.setMinimumWidth(268)
        column.setMaximumWidth(360)
        column.setStyleSheet(f"""
            #StatusColumn {{
                background-color: {_BG_CARD}; border: 1px solid {_BORDER};
                border-radius: 10px;
            }}
        """)
        layout = QVBoxLayout(column)
        layout.setContentsMargins(13, 13, 13, 13)
        layout.setSpacing(9)

        self._gauge = HealthGauge()
        layout.addWidget(self._gauge, alignment=Qt.AlignmentFlag.AlignCenter)

        self._stage_indicator = StageIndicator(["healthy", "degrading", "critical"])
        layout.addWidget(self._stage_indicator)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {_BORDER};")
        layout.addWidget(divider)

        self._readouts = {
            "samples": Readout("Samples processed"),
            "rate": Readout("Throughput"),
            "score": Readout("Anomaly score"),
            "status": Readout("Status"),
            "anomalies": Readout("Anomalies"),
            "forecast": Readout("Forecast RMSE (live)"),
            "coverage": Readout("95% coverage (live)"),
        }
        for widget in self._readouts.values():
            layout.addWidget(widget)

        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setStyleSheet(f"color: {_BORDER};")
        layout.addWidget(divider2)

        caption = QLabel("TOP ANOMALY DRIVERS")
        caption.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px; font-weight: 600;")
        layout.addWidget(caption)
        self._drivers = QLabel("—")
        self._drivers.setWordWrap(True)
        self._drivers.setStyleSheet(f"color: {_TEXT}; font-size: 11px;")
        layout.addWidget(self._drivers)

        layout.addStretch()
        self._footnote = QLabel("")
        self._footnote.setWordWrap(True)
        self._footnote.setStyleSheet(f"color: {_TEXT_DIM}; font-size: 10px;")
        layout.addWidget(self._footnote)
        return column

    # ==================================================================
    # Model loading / training
    # ==================================================================

    def _load_default_model(self):
        for name, path in PRETRAINED.items():
            if (path / "manifest.json").exists():
                index = self._model_combo.findText(name)
                if index >= 0:
                    self._model_combo.setCurrentIndex(index)
                self._load_bundle(path, name)
                return
        self._subtitle.setText(
            "No pretrained model found — run:  python ml/build_pretrained.py")
        self._start_button.setEnabled(False)

    def _on_model_selected(self, index: int):
        text = self._model_combo.itemText(index)
        if text == TRAIN_ENTRY:
            self._choose_dataset()
            return
        path = PRETRAINED.get(text)
        if path is None:
            return
        if not (path / "manifest.json").exists():
            QMessageBox.warning(self, "Model not built",
                                f"{text} has not been built yet.\n\n"
                                "Run:  python ml/build_pretrained.py")
            self._restore_model_combo()
            return
        self._load_bundle(path, text)

    def _load_bundle(self, path: Path, label: str):
        self.stop()
        try:
            from app.ml.persist import load_bundle

            self._apply_models(load_bundle(path), label)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not load model", f"{type(exc).__name__}: {exc}")
            self._restore_model_combo()

    def _choose_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a dataset to train on", "",
            "Tabular data (*.csv *.txt *.tsv *.dat);;All files (*)")
        if not path:
            self._restore_model_combo()
            return
        try:
            from app.ml.dataset import load_table

            dataset = load_table(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not read file", f"{type(exc).__name__}: {exc}")
            self._restore_model_combo()
            return

        roles = dataset.roles
        labels = ("run-to-failure labels found — health will be supervised"
                  if roles.has_run_to_failure
                  else "no labels found — health will be baseline-calibrated")
        detail = (
            f"<b>{Path(path).name}</b><br>"
            f"{dataset.n_rows} rows &middot; <b>{dataset.n_sensors}</b> sensor channels<br><br>"
            f"Detected: {roles.describe()}<br><br>"
            f"{labels}<br><br>Train all four models on this data?"
        )
        answer = QMessageBox.question(
            self, "Train on this dataset", detail,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            self._restore_model_combo()
            return
        self._start_training(dataset, Path(path).name)

    def _start_training(self, dataset, label: str):
        self.stop()
        payload = {
            "values": dataset.sensor_values(),
            "names": list(dataset.roles.sensors),
            "units": dataset.units(),
            "cycles": dataset.cycles(),
            "rul": dataset.rul(ceiling=125.0),
            "source": label,
        }
        self._set_busy(True, f"Training on {label}…")
        self._worker = TrainingWorker(payload, self)
        self._worker.progressed.connect(self._on_training_progress)
        self._worker.finished_ok.connect(lambda m: self._on_training_done(m, label))
        self._worker.failed.connect(self._on_training_failed)
        self._worker.start()

    def _on_training_progress(self, message: str, fraction: float):
        self._progress.setValue(int(max(0.0, min(1.0, fraction)) * 100))
        self._progress_label.setText(message)

    def _on_training_done(self, models, label: str):
        self._set_busy(False)
        entry = f"Custom · {label}"
        if self._model_combo.findText(entry) < 0:
            self._model_combo.insertItem(self._model_combo.count() - 1, entry)
        self._model_combo.setCurrentIndex(self._model_combo.findText(entry))
        self._apply_models(models, entry)
        self.status_message.emit(f"ML model trained on {label}")

    def _on_training_failed(self, message: str):
        self._set_busy(False)
        self._restore_model_combo()
        QMessageBox.critical(self, "Training failed", message)

    def _set_busy(self, busy: bool, message: str = ""):
        self._progress.setVisible(busy)
        self._progress_label.setVisible(busy)
        self._progress_label.setText(message)
        self._progress.setValue(0)
        for widget in (self._model_combo, self._source_combo, self._start_button):
            widget.setEnabled(not busy)

    def _restore_model_combo(self):
        if self._current_label:
            index = self._model_combo.findText(self._current_label)
            if index >= 0:
                self._model_combo.setCurrentIndex(index)

    def _apply_models(self, models, label: str):
        """Install a bundle and rebuild everything that depends on its schema."""
        self._models = models
        self._current_label = label
        engine = self._ensure_engine()
        engine.attach(models)

        self._stage_indicator.set_labels(models.class_labels)
        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        for name in models.schema.names:
            self._channel_combo.addItem(str(name))
        index = self._channel_combo.findText(str(engine.forecast_channel))
        if index >= 0:
            self._channel_combo.setCurrentIndex(index)
        self._channel_combo.blockSignals(False)

        kind = "supervised" if models.labelled else "baseline-calibrated"
        self._subtitle.setText(
            f"{label}  ·  {models.n_features} channels  ·  {kind}  "
            f"·  window {models.window}, horizon {models.horizon}")
        self._dropped_notice = (
            f"Dropped {len(models.schema.dropped)} constant channel(s): "
            f"{', '.join(str(d) for d in models.schema.dropped[:6])}"
            if models.schema.dropped else "")
        self._footnote.setText(
            "Every value here is computed live, one sample at a time.\n"
            + self._dropped_notice)

        self._update_importance_plot()
        self._update_insights()
        self.reset_run()
        self._check_source_compatibility()

    # ==================================================================
    # Source selection
    # ==================================================================

    def _on_source_selected(self, index: int):
        self._source_kind = self._source_combo.itemData(index) or "rocket"
        self._stream_rows = None
        self.reset_run()
        self._check_source_compatibility()

    def _on_channel_selected(self, index: int):
        if self._engine is not None:
            self._engine.set_forecast_channel(self._channel_combo.itemText(index))
            self._refresh_views()

    def _check_source_compatibility(self) -> bool:
        """Warn when the model's channels cannot be supplied by the chosen source."""
        if self._models is None:
            return False
        needed = set(str(n) for n in self._models.schema.names)
        available: Optional[set] = None
        if self._source_kind == "rocket":
            from app.core.rocket_sim import TELEMETRY_CHANNELS

            available = set(TELEMETRY_CHANNELS)
        elif self._source_kind == "live":
            available = set(self._live_channels) if self._live_channels else None

        if available is not None and not needed.issubset(available):
            missing = sorted(needed - available)[:4]
            self._subtitle.setText(
                f"{self._current_label} — this source does not provide: "
                f"{', '.join(missing)}…   Choose a matching model, or train on this source.")
            self._start_button.setEnabled(False)
            return False
        self._start_button.setEnabled(True)
        return True

    # ==================================================================
    # Run control
    # ==================================================================

    def toggle_running(self):
        self.stop() if self._running else self.start()

    def start(self):
        if self._models is None or not self._check_source_compatibility():
            return
        if self._source_kind == "rocket":
            if not self._prepare_flight():
                return
        elif self._source_kind == "replay":
            if self._stream_rows is None and not self._prepare_replay():
                return
        self._running = True
        self._start_button.setText("▌▌  Pause")
        if self._source_kind != "live":
            self._stream_timer.start()
        self._redraw_timer.start()

    def stop(self):
        self._running = False
        self._stream_timer.stop()
        self._redraw_timer.stop()
        if hasattr(self, "_start_button"):
            self._start_button.setText("▶  Start")
        self._refresh_views()

    def reset_run(self):
        self.stop()
        if self._engine is not None:
            self._engine.reset()
        self._stream_position = 0
        self._refresh_views()

    # ------------------------------------------------------------------ sources

    def _prepare_flight(self) -> bool:
        """Fly the current airframe and queue the telemetry for streaming."""
        params = self._collect_params()
        self._rocket_params = params
        try:
            flight = simulate(params)
            rows = self._models.schema.select(flight.matrix(), flight.names)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Simulation failed", f"{type(exc).__name__}: {exc}")
            return False
        self._last_flight = flight
        self._stream_rows = rows
        self._stream_position = 0
        if self._engine is not None:
            self._engine.reset()
        self._altitude_curve.setData(flight.time, flight.channels["altitude_m"])
        self._flight_summary.setText(flight.summary())
        return True

    def _prepare_replay(self) -> bool:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose data to replay", "",
            "Tabular data (*.csv *.txt *.tsv *.dat);;All files (*)")
        if not path:
            return False
        try:
            from app.ml.dataset import load_table

            dataset = load_table(path)
            self._stream_rows = self._models.schema.select(
                dataset.sensor_values(), dataset.roles.sensors)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Cannot replay this file",
                f"{type(exc).__name__}: {exc}\n\n"
                "The file must contain the channels this model expects.")
            return False
        self._stream_position = 0
        return True

    def _tick_stream(self):
        """Feed the next few simulated/replayed samples into the engine."""
        if not self._running or self._stream_rows is None or self._engine is None:
            return
        for _ in range(3):
            if self._stream_position >= len(self._stream_rows):
                self.stop()
                self.status_message.emit("ML Analysis: run complete")
                return
            stamp = float(self._stream_position)
            if (self._source_kind == "rocket" and self._last_flight is not None
                    and self._stream_position < len(self._last_flight.time)):
                stamp = float(self._last_flight.time[self._stream_position])
            self._engine.push(self._stream_rows[self._stream_position], timestamp=stamp)
            self._stream_position += 1

    # ------------------------------------------------------------------ live feed

    def feed_live_sample(self, values: Dict[str, float], timestamp: float = 0.0):
        """Hook for the main window's telemetry path. Always safe to call."""
        if values:
            merged = set(self._live_channels) | set(values)
            if len(merged) != len(self._live_channels):
                self._live_channels = sorted(merged)
        if (not self._running or self._source_kind != "live"
                or self._engine is None or self._models is None):
            return
        try:
            self._engine.push_mapping(values, timestamp=timestamp)
        except Exception:  # noqa: BLE001 - inference must never break the data path
            pass

    def live_channels_available(self) -> List[str]:
        return list(self._live_channels)

    # ==================================================================
    # Rocket parameters
    # ==================================================================

    def _collect_params(self) -> RocketParams:
        params = RocketParams()
        for key, box in self._spins.items():
            if hasattr(params, key):
                setattr(params, key, float(box.value()))
        params.fin_count = int(self._fin_count.value())
        params.nose_shape = self._nose_combo.currentText()
        return params

    def _on_rocket_param_changed(self, *_):
        params = self._collect_params()
        self._rocket_params = params
        self._rocket_view.set_params(params)
        try:
            flight = simulate(params)
        except Exception:  # noqa: BLE001
            self._flight_summary.setText("Simulation failed for these parameters")
            return
        self._last_flight = flight
        self._altitude_curve.setData(flight.time, flight.channels["altitude_m"])
        self._flight_summary.setText(flight.summary())

    # ==================================================================
    # Rendering
    # ==================================================================

    def _refresh_views(self):
        engine = self._engine
        if engine is None or not engine.is_ready:
            return
        summary = engine.live_summary()
        result = engine.last_result

        self._update_kpis(summary, result)
        self._update_status(summary, result)
        self._update_plots(engine, result)
        if result is not None:
            self._gauge.set_value(result.health)
            self._stage_indicator.set_stage(result.stage, result.stage_confidence)
        if self._tabs.currentIndex() == 2:
            self._update_insights()

    def _update_kpis(self, summary: dict, result):
        if result is None:
            for card in self._kpi.values():
                card.set_value("—")
                card.set_detail("waiting for data")
            return

        score, threshold = summary["anomaly_score"], summary["anomaly_threshold"]
        ratio = score / threshold if threshold else float("nan")
        flagged = bool(result.anomaly_flag)
        self._kpi["autoencoder"].set_value(
            "ANOMALY" if flagged else "Nominal", _DANGER if flagged else _SUCCESS)
        self._kpi["autoencoder"].set_detail(
            f"score {score:.4f} · {ratio:.2f}× threshold · "
            f"{summary['anomalies']} flagged ({summary['anomaly_rate'] * 100:.1f}%)")

        if summary["forecast_scored"] > 0:
            self._kpi["gpr"].set_value(f"{summary['forecast_coverage'] * 100:.0f}% covered")
            self._kpi["gpr"].set_detail(
                f"{summary['forecast_channel']} · RMSE {summary['forecast_rmse']:.3g} "
                f"· {summary['forecast_scored']} predictions scored")
        else:
            self._kpi["gpr"].set_value("warming up")
            self._kpi["gpr"].set_detail(
                f"{summary['forecast_channel']} · needs {self._models.window} samples")

        self._kpi["lstm"].set_value(result.stage_label.capitalize(),
                                    _STAGE_COLORS[result.stage % 3])
        self._kpi["lstm"].set_detail(
            f"{result.stage_confidence * 100:.0f}% confidence · "
            f"over a {self._models.window}-sample window")

        health = result.health
        colour = _SUCCESS if health > 60 else (_WARNING if health >= 24 else _DANGER)
        self._kpi["catboost"].set_value(f"{health:.0f}/100", colour)
        drivers = self._models.top_drivers(2)
        self._kpi["catboost"].set_detail(
            ("driven by " + ", ".join(str(d["feature"]) for d in drivers)) if drivers else "")

    def _update_status(self, summary: dict, result):
        self._readouts["samples"].set_value(f"{summary['samples']}")
        self._readouts["anomalies"].set_value(
            f"{summary['anomalies']} ({summary['anomaly_rate'] * 100:.1f}%)")
        if result is not None:
            self._readouts["score"].set_value(f"{summary['anomaly_score']:.5f}")
            flagged = bool(result.anomaly_flag)
            self._readouts["status"].set_value(
                "ANOMALY" if flagged else "nominal", _DANGER if flagged else _SUCCESS)
            drivers = result.top_channels([str(n) for n in self._models.schema.names], 3)
            self._drivers.setText(
                "\n".join(f"{name}   {value:.4f}" for name, value in drivers) or "—")
        if summary["forecast_scored"] > 0:
            self._readouts["forecast"].set_value(f"{summary['forecast_rmse']:.4g}")
            coverage = summary["forecast_coverage"]
            colour = _SUCCESS if 0.90 <= coverage <= 0.99 else _WARNING
            self._readouts["coverage"].set_value(f"{coverage * 100:.1f}%", colour)
        rate = self._stream_rate()
        self._readouts["rate"].set_value(f"{rate:.0f} samples/s" if rate else "—")

    def _stream_rate(self) -> float:
        engine = self._engine
        if engine is None or len(engine.time_history) < 2:
            return 0.0
        span = engine.time_history[-1] - engine.time_history[0]
        return (len(engine.time_history) - 1) / span if span > 1e-9 else 0.0

    def _update_plots(self, engine, result):
        channel = engine.forecast_channel
        if not channel:
            return
        x, y = engine.channel_series(channel)
        if len(x) == 0:
            for curve in (self._actual_curve, self._forecast_curve,
                          self._band_upper, self._band_lower, self._health_curve):
                curve.setData([], [])
            self._anomaly_scatter.setData([], [])
            self._health_anomaly.setData([], [])
            return

        if len(x) > _MAX_PLOT_POINTS:
            step = len(x) // _MAX_PLOT_POINTS + 1
            x_plot, y_plot = x[::step], y[::step]
        else:
            x_plot, y_plot = x, y

        self._main_plot.setTitle(f"{channel}: measured vs predicted",
                                 color="#ffffff", size="10pt")
        self._actual_curve.setData(x_plot, y_plot)

        if result is not None and result.forecast_ready:
            joined_x = np.concatenate([[x[-1]], result.forecast_x])
            self._forecast_curve.setData(
                joined_x, np.concatenate([[y[-1]], result.forecast_mean]))
            self._band_upper.setData(
                joined_x, np.concatenate([[y[-1]], result.forecast_upper]))
            self._band_lower.setData(
                joined_x, np.concatenate([[y[-1]], result.forecast_lower]))
        else:
            for curve in (self._forecast_curve, self._band_upper, self._band_lower):
                curve.setData([], [])

        ax, ay = engine.anomaly_points(channel)
        self._anomaly_scatter.setData(ax, ay)

        health_x = np.asarray(engine.index_history, dtype=np.float64)
        health_y = np.asarray(engine.health_history, dtype=np.float64)
        if len(health_x) > _MAX_PLOT_POINTS:
            step = len(health_x) // _MAX_PLOT_POINTS + 1
            health_x, health_y = health_x[::step], health_y[::step]
        self._health_curve.setData(health_x, health_y)

        base = engine.index_history[0] if engine.index_history else 0
        marks = [(i, engine.health_history[i - base]) for i in engine.anomaly_indices
                 if 0 <= i - base < len(engine.health_history)]
        if marks:
            self._health_anomaly.setData([m[0] for m in marks], [m[1] for m in marks])
        else:
            self._health_anomaly.setData([], [])

        if self._last_flight is not None and self._source_kind == "rocket":
            position = min(self._stream_position, len(self._last_flight.time) - 1)
            self._flight_marker.setPos(float(self._last_flight.time[position]))

    def _update_importance_plot(self):
        if self._models is None:
            return
        top = self._models.top_drivers(12)
        if not top:
            return
        positions = list(range(len(top)))
        self._importance_bars.setOpts(
            x=positions, height=[d["importance"] for d in top], width=0.6)
        self._importance_plot.getAxis("bottom").setTicks(
            [[(i, str(d["feature"])) for i, d in enumerate(top)]])
        self._importance_plot.setXRange(-0.6, len(top) - 0.4, padding=0)

    def _update_insights(self):
        """Plain-language read of what each algorithm currently concludes."""
        if self._models is None or self._engine is None:
            return
        models, engine = self._models, self._engine
        summary = engine.live_summary()
        result = engine.last_result

        def block(title: str, body: str) -> str:
            return (f"<p style='margin:0 0 11px 0'><b style='color:#0a84ff'>{title}</b><br>"
                    f"<span style='color:#dddddd'>{body}</span></p>")

        parts = [
            f"<p style='color:#8e8e93;margin:0 0 12px 0'>Model <b style='color:#fff'>"
            f"{self._current_label}</b> &middot; {models.n_features} channels &middot; "
            f"{'supervised on labels' if models.labelled else 'baseline-calibrated'} &middot; "
            f"trained in {models.elapsed_s:.0f}s from {models.source}</p>"
        ]

        if result is None:
            parts.append("<p style='color:#8e8e93'>Press <b>Start</b> to begin streaming. "
                         "Every number below is computed live from incoming samples.</p>")
        else:
            ratio = (summary["anomaly_score"] / summary["anomaly_threshold"]
                     if summary["anomaly_threshold"] else float("nan"))
            names = [str(n) for n in models.schema.names]
            drivers = result.top_channels(names, 3)
            parts.append(block(
                "Autoencoder — what looks abnormal right now",
                f"Reconstruction error is <b>{summary['anomaly_score']:.5f}</b>, "
                f"{ratio:.2f}× the nominal threshold "
                f"({summary['anomaly_threshold']:.5f}). "
                f"{'This sample is flagged.' if result.anomaly_flag else 'This sample is nominal.'} "
                f"Across the run, {summary['anomalies']} of {summary['samples']} samples "
                f"({summary['anomaly_rate'] * 100:.1f}%) were flagged. The channels deviating "
                f"most from the learned healthy pattern are "
                f"<b>{', '.join(n for n, _ in drivers)}</b> — look there first."))

            if summary["forecast_scored"] > 0:
                calibrated = 0.90 <= summary["forecast_coverage"] <= 0.99
                parts.append(block(
                    "Gaussian Process — where it is heading, and can it be trusted",
                    f"Forecasting <b>{summary['forecast_channel']}</b> {models.horizon} samples "
                    f"ahead from the last {models.window}. Scored against what actually arrived: "
                    f"<b>RMSE {summary['forecast_rmse']:.4g}</b> over "
                    f"{summary['forecast_scored']} matured predictions, with "
                    f"<b>{summary['forecast_coverage'] * 100:.1f}%</b> of real values falling "
                    f"inside the 95% band. "
                    + ("The interval is well calibrated." if calibrated else
                       "The interval is miscalibrated on this run — treat the band with "
                       "caution.")))
            else:
                parts.append(block("Gaussian Process",
                                   "Collecting the first window of samples."))

            counts: Dict[int, int] = {}
            for stage in engine.stage_history:
                counts[stage] = counts.get(stage, 0) + 1
            spread = ", ".join(f"{models.class_labels[k]} {v}"
                               for k, v in sorted(counts.items()))
            parts.append(block(
                "LSTM — which degradation stage the trajectory implies",
                f"Current stage <b>{result.stage_label}</b> at "
                f"{result.stage_confidence * 100:.0f}% confidence, judged from the shape of the "
                f"last {models.window} samples rather than the newest reading alone. "
                f"Samples per stage this run: {spread}."))

            first = engine.health_history[0] if engine.health_history else result.health
            delta = result.health - first
            top = models.top_drivers(3)
            parts.append(block(
                "CatBoost — one number for condition, and what drives it",
                f"Health is <b>{result.health:.1f}/100</b> "
                f"({'down' if delta < 0 else 'up'} {abs(delta):.1f} since the run began). "
                f"Bands: healthy &gt;60, degrading 24–60, critical &lt;24. The channels "
                f"this model leans on most are "
                + ", ".join(f"<b>{d['feature']}</b> ({d['importance']:.0f}%)" for d in top)
                + ". Unlike the autoencoder's instantaneous residual, this attribution is "
                  "learned across the whole training set, so it says what generally predicts "
                  "condition."))

        if self._dropped_notice:
            parts.append(f"<p style='color:#8e8e93;font-size:11px'>{self._dropped_notice}</p>")
        self._insights.setHtml("".join(parts))

    # ==================================================================
    # Persistence -- selections and the rocket design
    # ==================================================================

    def get_save_state(self) -> dict:
        """
        @brief  Persist the selected model, source, channel and rocket design.
        """
        return {
            "version": 2,
            "model": self._current_label,
            "source": self._source_kind,
            "channel": self._channel_combo.currentText(),
            "rocket": self._collect_params().to_dict(),
        }

    def load_save_state(self, state: dict):
        """
        @brief  Restore selector positions and the rocket design.
        """
        if not isinstance(state, dict):
            return
        rocket = state.get("rocket") or {}
        if rocket:
            params = RocketParams.from_dict(rocket)
            for key, box in self._spins.items():
                if hasattr(params, key):
                    box.blockSignals(True)
                    box.setValue(float(getattr(params, key)))
                    box.blockSignals(False)
            self._fin_count.blockSignals(True)
            self._fin_count.setValue(int(params.fin_count))
            self._fin_count.blockSignals(False)
            self._nose_combo.setCurrentText(str(params.nose_shape))
            self._on_rocket_param_changed()

        source = state.get("source")
        if source:
            index = self._source_combo.findData(source)
            if index >= 0:
                self._source_combo.setCurrentIndex(index)
                self._source_kind = source
                self._check_source_compatibility()
        channel = state.get("channel")
        if channel:
            index = self._channel_combo.findText(channel)
            if index >= 0:
                self._channel_combo.setCurrentIndex(index)
                if self._engine is not None:
                    self._engine.set_forecast_channel(channel)

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def cleanup(self):
        """Stop timers and any running trainer. Called on shutdown."""
        self.stop()
        if self._worker is not None and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)

    def hideEvent(self, event):
        self.stop()
        super().hideEvent(event)

    def _on_close(self):
        self.stop()
        self.close_requested.emit()
