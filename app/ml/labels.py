"""Deriving health and degradation stage, with or without failure labels.

Two regimes:

**Labelled (run-to-failure).** The dataset carries a unit/cycle structure or an
explicit RUL column, so health is the standard C-MAPSS style mapping of capped
RUL onto 0-100 and stages are RUL thresholds. This is what the offline pipeline
in ``ml/`` uses.

**Unlabelled (a live stream, or a plain log).** There is no ground-truth failure
time, so health is *calibrated against a healthy baseline*: an autoencoder is
fitted on an early window assumed nominal, and health is a bounded, monotone
function of how far the current reading deviates from that baseline in units of
the baseline's own spread. Deviation is expressed in "baseline sigmas" so the
scale is meaningful regardless of what the channels measure:

    d = (error - baseline_median) / (baseline_p99 - baseline_median)
    health = 100 * 2 ** (-d / HALF_LIFE_D)

``d = 0`` (a typical nominal reading) gives 100, ``d = 1`` (as unusual as the
99th percentile of nominal) still gives ~79, and health halves every
``HALF_LIFE_D`` further units. The map is smooth, bounded to (0, 100], and never
saturates, so a worsening trend keeps producing a falling number.

Stage bands are shared across both regimes so the two never disagree about what
"critical" means.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STAGE_LABELS = ["healthy", "degrading", "critical"]

# Stage bands expressed on the 0-100 health scale.
HEALTH_DEGRADING_MAX = 60.0   # health > 60           -> healthy
HEALTH_CRITICAL_MAX = 24.0    # 24 < health <= 60     -> degrading
#                               health <= 24          -> critical

# Equivalent RUL thresholds, for the labelled regime (ceiling 125 -> 60/24 health).
RUL_CEILING = 125.0
RUL_DEGRADING_MAX = 75.0
RUL_CRITICAL_MAX = 30.0

HALF_LIFE_D = 3.0  # deviation units over which health halves
ANOMALY_DEVIATIONS = 3.0  # flag fires here; equals health < 50 by construction


def health_from_rul(rul: np.ndarray, ceiling: float = RUL_CEILING) -> np.ndarray:
    """Labelled regime: capped RUL -> 0-100 health."""
    return np.clip(np.asarray(rul, dtype=np.float64) / ceiling, 0.0, 1.0) * 100.0


def stage_from_health(health: np.ndarray) -> np.ndarray:
    """Health -> stage index (0 healthy, 1 degrading, 2 critical)."""
    health = np.asarray(health, dtype=np.float64)
    stage = np.zeros(health.shape, dtype=np.int64)
    stage[health <= HEALTH_DEGRADING_MAX] = 1
    # Inclusive at the edge so this agrees with stage_from_rul, where RUL <= 30
    # is critical and maps to exactly health 24.
    stage[health <= HEALTH_CRITICAL_MAX] = 2
    return stage


def stage_from_rul(rul: np.ndarray) -> np.ndarray:
    """Labelled regime: RUL -> stage index, matching the health bands."""
    rul = np.asarray(rul, dtype=np.float64)
    stage = np.zeros(rul.shape, dtype=np.int64)
    stage[rul <= RUL_DEGRADING_MAX] = 1
    stage[rul <= RUL_CRITICAL_MAX] = 2
    return stage


@dataclass
class BaselineCalibration:
    """Maps an anomaly score onto 0-100 health, using a nominal reference window.

    ``reference`` is the median reconstruction error over the baseline window and
    ``spread`` the distance from that median to its 99th percentile -- a robust
    scale that does not assume the errors are normally distributed.
    """

    reference: float
    spread: float
    threshold: float          # anomaly flag: score above this is "not nominal"
    n_baseline: int
    percentile: float = 99.0
    half_life: float = HALF_LIFE_D
    threshold_deviations: float = ANOMALY_DEVIATIONS

    @classmethod
    def fit(
        cls,
        baseline_scores: np.ndarray,
        percentile: float = 99.0,
        half_life: float = HALF_LIFE_D,
        deviations: float = ANOMALY_DEVIATIONS,
    ) -> "BaselineCalibration":
        scores = np.asarray(baseline_scores, dtype=np.float64).ravel()
        if len(scores) == 0:
            raise ValueError("cannot calibrate on an empty baseline window")
        reference = float(np.median(scores))
        high = float(np.percentile(scores, percentile))
        spread = high - reference
        if not np.isfinite(spread) or spread <= 1e-12:
            # Degenerate baseline (constant error): fall back to a relative scale
            # so deviation stays meaningful instead of dividing by ~0.
            spread = max(abs(reference) * 0.1, 1e-9)
        # The flag fires at a fixed number of baseline deviations rather than at
        # the raw p99. A well-fitted autoencoder reconstructs its nominal data so
        # closely that p99 sits a hair above the median, and then anything merely
        # unfamiliar -- a different airframe, a new operating point -- trips it.
        # Tying the flag to the same deviation scale the health score uses keeps
        # the two consistent: the flag raises exactly when health falls below 50.
        threshold = reference + deviations * spread
        return cls(
            reference=reference,
            spread=float(spread),
            threshold=float(threshold),
            n_baseline=int(len(scores)),
            percentile=percentile,
            half_life=half_life,
            threshold_deviations=float(deviations),
        )

    def deviation(self, score: np.ndarray) -> np.ndarray:
        """Anomaly score -> deviation in baseline units (0 = typical nominal)."""
        return (np.asarray(score, dtype=np.float64) - self.reference) / self.spread

    def health(self, score: np.ndarray) -> np.ndarray:
        """Anomaly score -> 0-100 health. Monotone decreasing, never negative."""
        d = np.maximum(self.deviation(score), 0.0)
        return 100.0 * np.power(2.0, -d / self.half_life)

    def is_anomalous(self, score: np.ndarray) -> np.ndarray:
        return np.asarray(score, dtype=np.float64) > self.threshold

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "spread": self.spread,
            "threshold": self.threshold,
            "n_baseline": self.n_baseline,
            "percentile": self.percentile,
            "half_life": self.half_life,
            "threshold_deviations": self.threshold_deviations,
            "health_formula": "100 * 2 ** (-max(d, 0) / half_life), "
                              "d = (score - reference) / spread",
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BaselineCalibration":
        return cls(
            reference=float(payload["reference"]),
            spread=float(payload["spread"]),
            threshold=float(payload["threshold"]),
            n_baseline=int(payload.get("n_baseline", 0)),
            percentile=float(payload.get("percentile", 99.0)),
            half_life=float(payload.get("half_life", HALF_LIFE_D)),
            threshold_deviations=float(
                payload.get("threshold_deviations", ANOMALY_DEVIATIONS)),
        )


def stage_band_text() -> dict[str, str]:
    """Human-readable band edges, for the UI legend."""
    return {
        "healthy": f"health > {HEALTH_DEGRADING_MAX:.0f}",
        "degrading": f"{HEALTH_CRITICAL_MAX:.0f} - {HEALTH_DEGRADING_MAX:.0f}",
        "critical": f"health ≤ {HEALTH_CRITICAL_MAX:.0f}",
    }
