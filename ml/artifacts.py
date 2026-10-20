"""The serialization contract between ``ml/`` (training) and ``app/`` (inference).

Every model writes its weights plus a sidecar ``<name>_meta.json`` into
``app/models/``. The sidecar is the *only* thing the UI layer needs to read in
order to drive a model correctly: it pins the exact input feature order, the
normalization statistics to apply at inference time, the output shape and what
that output means, and the decision thresholds.

All four models share one input feature contract -- the same 17 scaled columns
in the same order (``common.kept_features()``) -- so the UI can normalize a
telemetry row once and fan it out to every model. The models differ only in the
shape they consume:

    autoencoder      (N, 17)      one cycle per row
    catboost_health  (N, 17)      one cycle per row
    lstm             (N, 30, 17)  a 30-cycle window, oldest row first
    gpr              (N, 1)       the s11 column of a 30-cycle window

``META_SCHEMA_VERSION`` is bumped if the sidecar layout ever changes
incompatibly, so ``app/`` can fail loudly instead of mis-reading a stale file.
"""
from __future__ import annotations

import datetime as _dt
import platform
import sys
from pathlib import Path
from typing import Any

from ml.common import ARTIFACT_DIR, RUL_CEILING, STAGE_LABELS, rel, write_json

META_SCHEMA_VERSION = "1.0"


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _versions() -> dict[str, str]:
    """Record the libraries that produced the artifact, for reproducibility."""
    out: dict[str, str] = {"python": platform.python_version()}
    for mod in ("numpy", "torch", "sklearn", "catboost", "lightgbm", "onnx"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - absent or broken import is just "not used"
            continue
    return out


def dataset_block(dataset: Any) -> dict[str, Any]:
    """Provenance of the data a model was trained on."""
    return {
        "name": "NASA C-MAPSS FD001",
        "files": ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"],
        "split_by": "engine unit id",
        "split_seed": 1337,
        "engines": {
            "train": int(len(dataset.units("train"))),
            "val": int(len(dataset.units("val"))),
            "test": int(len(dataset.units("test"))),
        },
        "rul_ceiling": RUL_CEILING,
    }


def build_meta(
    *,
    name: str,
    task: str,
    dataset: Any,
    input_shape: list[int | None],
    output: dict[str, Any],
    artifacts: dict[str, str],
    inference: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
    window_length: int | None = None,
    input_feature_names: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one model's sidecar. Key order is fixed so diffs stay readable."""
    features = input_feature_names if input_feature_names is not None else list(dataset.features)
    meta: dict[str, Any] = {
        "name": name,
        "task": task,
        "schema_version": META_SCHEMA_VERSION,
        "created_utc": _now_utc(),
        "library_versions": _versions(),
        "dataset": dataset_block(dataset),
        "input": {
            "feature_names": features,
            "n_features": len(features),
            "shape": input_shape,
            "dtype": "float32",
            "layout": (
                f"(batch, {window_length}, {len(features)}) -- axis 1 is time, oldest first"
                if window_length
                else f"(batch, {len(features)}) -- one cycle per row"
            ),
            "window_length": window_length,
            "requires_normalization": True,
            "feature_order_is_binding": True,
        },
        "normalization": dataset.scaler_stats(),
        "output": output,
        "thresholds": thresholds or {},
        "artifacts": artifacts,
        "inference": inference,
        "metrics": metrics,
    }
    if extra:
        meta.update(extra)
    return meta


def save_meta(name: str, meta: dict[str, Any]) -> Path:
    """Write ``app/models/<name>_meta.json``."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return write_json(ARTIFACT_DIR / f"{name}_meta.json", meta)


def artifact_path(filename: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / filename


def report(name: str, metrics: dict[str, Any], written: list[Path]) -> None:
    """Uniform end-of-training summary."""
    print(f"\n--- {name}: metrics ---")
    _print_metrics(metrics)
    print(f"--- {name}: files ---")
    for path in written:
        size = path.stat().st_size if path.exists() else 0
        print(f"  {rel(path):44s} {size / 1024:9.1f} KiB")


def _print_metrics(metrics: dict[str, Any], indent: int = 2) -> None:
    pad = " " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_metrics(value, indent + 2)
        elif isinstance(value, float):
            print(f"{pad}{key:34s} {value:.4f}")
        elif isinstance(value, list) and value and isinstance(value[0], float):
            print(f"{pad}{key:34s} [{', '.join(f'{v:.4f}' for v in value)}]")
        else:
            print(f"{pad}{key:34s} {value}")


def stage_output_block() -> dict[str, Any]:
    """Shared description of the 3-class degradation-stage output."""
    return {
        "shape": [-1, 3],
        "dtype": "float32",
        "meaning": "unnormalized logits over degradation stage",
        "class_labels": list(STAGE_LABELS),
        "class_index": {label: i for i, label in enumerate(STAGE_LABELS)},
        "postprocess": "softmax(logits, axis=-1) for probabilities; argmax for the label",
        "note": "stages are derived from RUL thresholds, not from flight phases",
    }


def sanity_check_stdout() -> None:
    """Fail fast if Qt somehow leaked into the training process."""
    leaked = sorted(m for m in sys.modules if m.startswith(("PySide6", "PyQt5", "PyQt6")))
    if leaked:
        raise RuntimeError(f"Qt must not be imported during training, found: {leaked}")
