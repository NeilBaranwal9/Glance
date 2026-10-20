"""Shared paths, constants and helpers for the offline ML pipeline.

Nothing in ``ml/`` may import PySide6/Qt - training must run headless.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[1]
CMAPSS_DIR = REPO_ROOT / "data" / "cmapss"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACT_DIR = REPO_ROOT / "app" / "models"

PROCESSED_NPZ = PROCESSED_DIR / "fd001.npz"
PROCESSED_MANIFEST = PROCESSED_DIR / "fd001_manifest.json"

# ---------------------------------------------------------------- dataset schema
RAW_COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
ALL_FEATURES = ["op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# Constant across all of FD001 (verified: std < 1e-8 on the training split), so they
# carry no signal and would make min-max scaling divide by zero.
CONSTANT_FEATURES = ["op3", "s1", "s5", "s10", "s16", "s18", "s19"]

# ---------------------------------------------------------------- modelling constants
RUL_CEILING = 125          # standard C-MAPSS piecewise-linear RUL cap
WINDOW_LENGTH = 30         # see preprocess.py docstring for justification
GPR_TARGET_SENSOR = "s11"  # strongest monotonic degradation trend in FD001

# Degradation stages, derived from capped RUL (NOT flight phases).
STAGE_LABELS = ["healthy", "degrading", "critical"]
STAGE_CRITICAL_MAX_RUL = 30   # RUL <= 30            -> critical (class 2)
STAGE_DEGRADING_MAX_RUL = 75  # 30 < RUL <= 75       -> degrading (class 1)
#                               RUL > 75             -> healthy (class 0)

FAULT_RUL_THRESHOLD = 30      # anomaly-detection ground truth for the autoencoder
HEALTHY_LIFE_FRACTION = 0.30  # first 30% of each engine's life = "healthy baseline"

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)  # train / val / test, split by engine id
SPLIT_SEED = 1337

# ---------------------------------------------------------------- helpers


def kept_features() -> list[str]:
    """Model input features, in the exact canonical order used everywhere."""
    return [c for c in ALL_FEATURES if c not in CONSTANT_FEATURES]


def stage_from_rul(rul: np.ndarray) -> np.ndarray:
    """Map capped RUL to a degradation-stage class index (0 healthy .. 2 critical)."""
    stage = np.zeros(np.shape(rul), dtype=np.int64)
    stage[np.asarray(rul) <= STAGE_DEGRADING_MAX_RUL] = 1
    stage[np.asarray(rul) <= STAGE_CRITICAL_MAX_RUL] = 2
    return stage


def health_from_rul(rul: np.ndarray) -> np.ndarray:
    """Map capped RUL to a 0-100 health score (100 = as-new, 0 = at failure)."""
    return np.clip(np.asarray(rul, dtype=np.float64) / RUL_CEILING, 0.0, 1.0) * 100.0


def seed_everything(seed: int = SPLIT_SEED) -> None:
    """Make a training run reproducible across python/numpy/torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:
        pass


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write pretty, deterministic JSON (used for every *_meta.json)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def rel(path: Path) -> str:
    """Repo-relative POSIX path, for readable logs and manifests."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
    }
