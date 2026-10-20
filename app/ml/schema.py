"""Sensor schema and normalization for an arbitrary number of channels.

The offline C-MAPSS pipeline pinned 17 named features. At runtime the channel
set is whatever the attached source provides -- a rocket flight simulation, a
serial link with 40 parameters, or a CSV someone just loaded -- so the schema is
discovered from the data and carried alongside every model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

CLIP_LOW = -0.5
CLIP_HIGH = 1.5


@dataclass
class SensorSchema:
    """Channel names plus the min/max statistics used to normalize them.

    ``x_scaled = clip((x - min) / (max - min), -0.5, 1.5)``

    The clip range is deliberately wider than [0, 1]: readings beyond anything
    seen during fitting are exactly the interesting ones, so they are allowed to
    stay visibly out of range rather than being flattened onto the boundary.
    """

    names: list[str]
    minimum: np.ndarray
    maximum: np.ndarray
    clip_low: float = CLIP_LOW
    clip_high: float = CLIP_HIGH
    dropped: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ build

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        names: list[str],
        drop_constant: bool = True,
        tolerance: float = 1e-9,
    ) -> "SensorSchema":
        """Fit min/max over ``(n_samples, n_channels)``, optionally dropping flat ones."""
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {values.shape}")
        if values.shape[1] != len(names):
            raise ValueError(
                f"{values.shape[1]} columns but {len(names)} names were supplied"
            )
        if len(values) == 0:
            raise ValueError("cannot fit a schema on zero rows")

        low = np.nanmin(values, axis=0)
        high = np.nanmax(values, axis=0)
        spread = high - low

        keep = np.ones(len(names), dtype=bool)
        dropped: list[str] = []
        if drop_constant:
            keep = spread > tolerance
            if not keep.any():
                # Everything is flat; keep all channels rather than produce an
                # empty schema the models cannot be built from.
                keep = np.ones(len(names), dtype=bool)
            else:
                dropped = [n for n, k in zip(names, keep) if not k]

        return cls(
            names=[n for n, k in zip(names, keep) if k],
            minimum=low[keep],
            maximum=high[keep],
            dropped=dropped,
        )

    # ------------------------------------------------------------------ use

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def span(self) -> np.ndarray:
        spread = self.maximum - self.minimum
        return np.where(np.abs(spread) < 1e-12, 1.0, spread)

    def select(self, values: np.ndarray, source_names: list[str]) -> np.ndarray:
        """Reorder/subset raw columns from ``source_names`` into schema order."""
        index = {name: i for i, name in enumerate(source_names)}
        missing = [n for n in self.names if n not in index]
        if missing:
            raise KeyError(f"source is missing channels: {missing}")
        columns = [index[n] for n in self.names]
        return np.asarray(values, dtype=np.float64)[..., columns]

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Raw -> normalized float32. Accepts a single row or a batch."""
        raw = np.asarray(raw, dtype=np.float64)
        scaled = (raw - self.minimum) / self.span
        return np.clip(scaled, self.clip_low, self.clip_high).astype(np.float32)

    def inverse_transform_channel(self, scaled: np.ndarray, channel: str) -> np.ndarray:
        """Undo scaling for one channel, to report values in engineering units."""
        i = self.index_of(channel)
        return np.asarray(scaled, dtype=np.float64) * self.span[i] + self.minimum[i]

    def index_of(self, channel: str) -> int:
        try:
            return self.names.index(channel)
        except ValueError as exc:
            raise KeyError(f"unknown channel {channel!r}") from exc

    # ------------------------------------------------------------------ io

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "n_features": self.n_features,
            "min": [float(v) for v in self.minimum],
            "max": [float(v) for v in self.maximum],
            "clip_range": [self.clip_low, self.clip_high],
            "dropped_constant": list(self.dropped),
            "formula": "x_scaled = clip((x - min) / (max - min), -0.5, 1.5)",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SensorSchema":
        low, high = payload.get("clip_range", [CLIP_LOW, CLIP_HIGH])
        return cls(
            names=list(payload["names"]),
            minimum=np.asarray(payload["min"], dtype=np.float64),
            maximum=np.asarray(payload["max"], dtype=np.float64),
            clip_low=float(low),
            clip_high=float(high),
            dropped=list(payload.get("dropped_constant", [])),
        )

    def __len__(self) -> int:
        return self.n_features

    def __repr__(self) -> str:
        preview = ", ".join(self.names[:6])
        more = f", +{self.n_features - 6} more" if self.n_features > 6 else ""
        return f"SensorSchema({self.n_features} channels: {preview}{more})"
