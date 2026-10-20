"""Live inference: push one sample in, get every algorithm's verdict out.

The engine keeps rolling buffers, so a caller feeds it samples as they arrive
from whatever source is attached (serial link, simulator, rocket flight model)
and receives an :class:`InferenceResult` per sample. Nothing is precomputed and
nothing is replayed.

Cost control matters because this runs on the UI's data path:

* autoencoder, health regressor and stage LSTM are tiny forward passes and run
  on every sample;
* the Gaussian Process needs a Cholesky of the window Gram matrix, so it is
  throttled to every ``gp_every`` samples and reuses the last band in between.

The engine also scores *itself*. Every forecast is parked until the cycles it
predicted actually arrive, then compared against reality to produce a running
forecast RMSE and empirical 95% interval coverage -- the "actual vs predicted"
readout, measured on this run rather than quoted from training.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.ml.labels import STAGE_LABELS, stage_from_health
from app.ml.trainers import (
    TrainedModels,
    gp_forecast,
    per_channel_residual,
    predict_stage,
    reconstruction_error,
)

DEFAULT_HISTORY = 4000


@dataclass
class InferenceResult:
    """One sample's verdict from all four algorithms."""

    index: int
    timestamp: float
    raw: np.ndarray
    scaled: np.ndarray

    # autoencoder
    anomaly_score: float = 0.0
    anomaly_flag: bool = False
    anomaly_threshold: float = 0.0
    channel_residuals: Optional[np.ndarray] = None

    # health regressor
    health: float = 100.0
    health_band: str = "healthy"

    # stage LSTM
    stage: int = 0
    stage_label: str = "healthy"
    stage_confidence: float = 0.0
    stage_probabilities: Optional[np.ndarray] = None

    # gaussian process
    forecast_ready: bool = False
    forecast_channel: str = ""
    forecast_x: Optional[np.ndarray] = None
    forecast_mean: Optional[np.ndarray] = None
    forecast_lower: Optional[np.ndarray] = None
    forecast_upper: Optional[np.ndarray] = None

    def top_channels(self, names: list[str], k: int = 3) -> list[tuple[str, float]]:
        """Channels contributing most to this sample's anomaly score."""
        if self.channel_residuals is None:
            return []
        order = np.argsort(self.channel_residuals)[::-1][:k]
        return [(names[i], float(self.channel_residuals[i])) for i in order]


@dataclass
class ForecastScore:
    """Running accuracy of the GP's own predictions on this run."""

    n: int = 0
    sum_sq_error: float = 0.0
    n_inside: int = 0

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.sum_sq_error / self.n)) if self.n else float("nan")

    @property
    def coverage(self) -> float:
        return float(self.n_inside / self.n) if self.n else float("nan")

    def update(self, actual: float, mean: float, lower: float, upper: float) -> None:
        self.n += 1
        self.sum_sq_error += float((mean - actual) ** 2)
        if lower <= actual <= upper:
            self.n_inside += 1


class OnlineMLEngine:
    """Stateful, per-sample inference over an attached :class:`TrainedModels`."""

    def __init__(self, history: int = DEFAULT_HISTORY, gp_every: int = 5):
        self.models: Optional[TrainedModels] = None
        self.history_limit = history
        self.gp_every = max(1, gp_every)

        self._window: deque[np.ndarray] = deque()
        self._count = 0
        self._forecast_channel_index = 0

        # Rolling series for plotting.
        self.index_history: list[int] = []
        self.time_history: list[float] = []
        self.raw_history: list[np.ndarray] = []
        self.scaled_history: list[np.ndarray] = []
        self.score_history: list[float] = []
        self.health_history: list[float] = []
        self.stage_history: list[int] = []
        self.anomaly_indices: list[int] = []

        # Pending forecasts awaiting the samples they predicted.
        self._pending: deque[tuple[int, float, float, float]] = deque()
        self.forecast_score = ForecastScore()
        self._last_forecast: Optional[tuple] = None
        self.last_result: Optional[InferenceResult] = None

    # ------------------------------------------------------------------ setup

    @property
    def is_ready(self) -> bool:
        return self.models is not None

    @property
    def channel_names(self) -> list[str]:
        return list(self.models.schema.names) if self.models else []

    def attach(self, models: TrainedModels, forecast_channel: Optional[str] = None) -> None:
        """Install a trained bundle and clear all run state."""
        self.models = models
        self.reset()
        self.set_forecast_channel(forecast_channel or models.metrics.get("lead_channel")
                                  or models.schema.names[0])

    def set_forecast_channel(self, channel: str) -> None:
        if not self.models:
            return
        try:
            self._forecast_channel_index = self.models.schema.index_of(channel)
        except KeyError:
            self._forecast_channel_index = 0
        # Previous predictions referred to a different channel.
        self._pending.clear()
        self.forecast_score = ForecastScore()
        self._last_forecast = None

    @property
    def forecast_channel(self) -> str:
        if not self.models:
            return ""
        return self.models.schema.names[self._forecast_channel_index]

    def reset(self) -> None:
        """Clear buffers and history, keeping the attached models."""
        self._window.clear()
        self._count = 0
        self.index_history.clear()
        self.time_history.clear()
        self.raw_history.clear()
        self.scaled_history.clear()
        self.score_history.clear()
        self.health_history.clear()
        self.stage_history.clear()
        self.anomaly_indices.clear()
        self._pending.clear()
        self.forecast_score = ForecastScore()
        self._last_forecast = None
        self.last_result = None

    # ------------------------------------------------------------------ inference

    def push(self, raw_row: np.ndarray, timestamp: float = 0.0) -> Optional[InferenceResult]:
        """Feed one sample (already in schema channel order). Returns its verdict."""
        if self.models is None:
            return None
        models = self.models
        raw = np.asarray(raw_row, dtype=np.float64).ravel()
        if raw.size != models.schema.n_features:
            raise ValueError(
                f"expected {models.schema.n_features} channels, got {raw.size}"
            )

        scaled = models.schema.transform(raw)
        self._window.append(scaled)
        while len(self._window) > models.window:
            self._window.popleft()

        index = self._count
        self._count += 1

        # --- 1. autoencoder -------------------------------------------------
        score = float(reconstruction_error(models.autoencoder, scaled.reshape(1, -1))[0])
        residuals = per_channel_residual(models.autoencoder, scaled)
        threshold = models.calibration.threshold if models.calibration else 0.0
        flagged = bool(score > threshold)

        # --- 2. health ------------------------------------------------------
        health = float(np.clip(models.health_model.predict(scaled.reshape(1, -1))[0], 0.0, 100.0))
        band = STAGE_LABELS[int(stage_from_health(np.array([health]))[0])]

        # --- 3. stage -------------------------------------------------------
        window = self._padded_window(models.window)
        stage, probabilities = predict_stage(models.stage_model, window)

        result = InferenceResult(
            index=index,
            timestamp=timestamp,
            raw=raw,
            scaled=scaled,
            anomaly_score=score,
            anomaly_flag=flagged,
            anomaly_threshold=threshold,
            channel_residuals=residuals,
            health=health,
            health_band=band,
            stage=int(stage),
            stage_label=models.class_labels[int(stage)],
            stage_confidence=float(probabilities[int(stage)]),
            stage_probabilities=probabilities,
        )

        # --- 4. gaussian process (throttled) --------------------------------
        self._mature_forecasts(index, scaled)
        if len(self._window) >= models.window and index % self.gp_every == 0:
            self._refresh_forecast(index)
        if self._last_forecast is not None:
            x, mean, lower, upper = self._last_forecast
            result.forecast_ready = True
            result.forecast_channel = self.forecast_channel
            result.forecast_x = x
            result.forecast_mean = mean
            result.forecast_lower = lower
            result.forecast_upper = upper

        self._record(result)
        self.last_result = result
        return result

    def push_mapping(self, sample: dict[str, float], timestamp: float = 0.0):
        """Feed a ``{channel: value}`` mapping; missing channels hold their last value."""
        if self.models is None:
            return None
        names = self.models.schema.names
        previous = self.raw_history[-1] if self.raw_history else np.zeros(len(names))
        row = np.array(
            [float(sample.get(n, previous[i])) for i, n in enumerate(names)],
            dtype=np.float64,
        )
        return self.push(row, timestamp)

    # ------------------------------------------------------------------ internals

    def _padded_window(self, size: int) -> np.ndarray:
        """Current window, left-padded with the oldest sample when short."""
        rows = list(self._window)
        if not rows:
            return np.zeros((size, self.models.schema.n_features), dtype=np.float32)
        if len(rows) < size:
            rows = [rows[0]] * (size - len(rows)) + rows
        return np.stack(rows).astype(np.float32)

    def _refresh_forecast(self, index: int) -> None:
        models = self.models
        column = self._forecast_channel_index
        buffer = np.array([row[column] for row in self._window], dtype=np.float64)
        try:
            mean, lower, upper = gp_forecast(
                models.gp_prototype, buffer, models.window, models.horizon
            )
        except Exception:  # noqa: BLE001 - a GP hiccup must not stop the stream
            return
        # Report in engineering units.
        channel = self.forecast_channel
        mean_raw = models.schema.inverse_transform_channel(mean, channel)
        lower_raw = models.schema.inverse_transform_channel(lower, channel)
        upper_raw = models.schema.inverse_transform_channel(upper, channel)
        x = np.arange(index + 1, index + 1 + models.horizon, dtype=np.float64)
        self._last_forecast = (x, mean_raw, lower_raw, upper_raw)

        for step in range(models.horizon):
            self._pending.append(
                (index + 1 + step, float(mean_raw[step]),
                 float(lower_raw[step]), float(upper_raw[step]))
            )
        while len(self._pending) > models.horizon * 40:
            self._pending.popleft()

    def _mature_forecasts(self, index: int, scaled: np.ndarray) -> None:
        """Score any parked prediction whose target sample has now arrived."""
        if not self._pending:
            return
        channel = self.forecast_channel
        actual = float(
            self.models.schema.inverse_transform_channel(
                np.array([scaled[self._forecast_channel_index]]), channel
            )[0]
        )
        remaining = deque()
        for target, mean, lower, upper in self._pending:
            if target == index:
                self.forecast_score.update(actual, mean, lower, upper)
            elif target > index:
                remaining.append((target, mean, lower, upper))
            # targets older than `index` are dropped as stale
        self._pending = remaining

    def _record(self, result: InferenceResult) -> None:
        self.index_history.append(result.index)
        self.time_history.append(result.timestamp)
        self.raw_history.append(result.raw)
        self.scaled_history.append(result.scaled)
        self.score_history.append(result.anomaly_score)
        self.health_history.append(result.health)
        self.stage_history.append(result.stage)
        if result.anomaly_flag:
            self.anomaly_indices.append(result.index)

        if len(self.index_history) > self.history_limit:
            trim = len(self.index_history) - self.history_limit
            del self.index_history[:trim]
            del self.time_history[:trim]
            del self.raw_history[:trim]
            del self.scaled_history[:trim]
            del self.score_history[:trim]
            del self.health_history[:trim]
            del self.stage_history[:trim]
            cutoff = self.index_history[0]
            self.anomaly_indices = [i for i in self.anomaly_indices if i >= cutoff]

    # ------------------------------------------------------------------ views

    def channel_series(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        """``(index, raw value)`` history for one channel."""
        if not self.models or not self.raw_history:
            return np.empty(0), np.empty(0)
        column = self.models.schema.index_of(channel)
        return (
            np.asarray(self.index_history, dtype=np.float64),
            np.array([row[column] for row in self.raw_history], dtype=np.float64),
        )

    def anomaly_points(self, channel: str) -> tuple[np.ndarray, np.ndarray]:
        """Positions of flagged samples, for scatter markers on a channel plot."""
        if not self.models or not self.anomaly_indices:
            return np.empty(0), np.empty(0)
        column = self.models.schema.index_of(channel)
        base = self.index_history[0] if self.index_history else 0
        xs, ys = [], []
        for i in self.anomaly_indices:
            offset = i - base
            if 0 <= offset < len(self.raw_history):
                xs.append(float(i))
                ys.append(float(self.raw_history[offset][column]))
        return np.asarray(xs), np.asarray(ys)

    def live_summary(self) -> dict[str, Any]:
        """Headline numbers measured on the current run, not quoted from training."""
        result = self.last_result
        total = len(self.index_history)
        return {
            "samples": total,
            "anomalies": len(self.anomaly_indices),
            "anomaly_rate": (len(self.anomaly_indices) / total) if total else 0.0,
            "health": result.health if result else float("nan"),
            "stage": result.stage_label if result else "-",
            "stage_confidence": result.stage_confidence if result else 0.0,
            "anomaly_score": result.anomaly_score if result else float("nan"),
            "anomaly_threshold": result.anomaly_threshold if result else float("nan"),
            "forecast_rmse": self.forecast_score.rmse,
            "forecast_coverage": self.forecast_score.coverage,
            "forecast_scored": self.forecast_score.n,
            "forecast_channel": self.forecast_channel,
        }
