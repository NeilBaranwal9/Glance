"""Save and load a :class:`TrainedModels` bundle.

One format serves both halves of the tab: the pretrained benchmark bundle that
ships with the app and any model the user trains on their own data. That means
there is a single code path through the online engine, rather than a "real"
mode and a "demo" mode that drift apart.

A bundle is a directory:

    <name>/
        manifest.json          schema, window/horizon, metrics, importances
        autoencoder.pt         torch state dict
        stage_lstm.pt          torch state dict
        health.cbm             CatBoost model
        gp.pkl                 sklearn GaussianProcessRegressor (optimizer=None)

Torch modules are stored as state dicts and rebuilt from the schema's channel
count, so a bundle is only ever loaded into an architecture of the right size.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.ml.labels import STAGE_LABELS, BaselineCalibration
from app.ml.schema import SensorSchema
from app.ml.trainers import TrainedModels, _make_autoencoder, _make_stage_model

MANIFEST = "manifest.json"
BUNDLE_VERSION = "1.0"


def save_bundle(models: TrainedModels, directory: str | Path, overwrite: bool = True) -> Path:
    """Write a trained bundle to ``directory``."""
    import joblib
    import torch

    directory = Path(directory)
    if directory.exists() and overwrite:
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

    torch.save(models.autoencoder.state_dict(), directory / "autoencoder.pt")
    torch.save(models.stage_model.state_dict(), directory / "stage_lstm.pt")
    models.health_model.save_model(str(directory / "health.cbm"), format="cbm")
    joblib.dump(models.gp_prototype, directory / "gp.pkl")

    manifest: dict[str, Any] = {
        "bundle_version": BUNDLE_VERSION,
        "schema": models.schema.to_dict(),
        "window": int(models.window),
        "horizon": int(models.horizon),
        "labelled": bool(models.labelled),
        "class_labels": list(models.class_labels),
        "calibration": models.calibration.to_dict() if models.calibration else None,
        "feature_importances": models.feature_importances,
        "metrics": _jsonable(models.metrics),
        "elapsed_s": float(models.elapsed_s),
        "source": models.source,
        "files": {
            "autoencoder": "autoencoder.pt",
            "stage_lstm": "stage_lstm.pt",
            "health": "health.cbm",
            "gp": "gp.pkl",
        },
    }
    (directory / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def load_bundle(directory: str | Path) -> TrainedModels:
    """Read a bundle back into a ready-to-use :class:`TrainedModels`."""
    import joblib
    import torch
    from catboost import CatBoostRegressor

    directory = Path(directory)
    manifest_path = directory / MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"no model bundle at {directory}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    schema = SensorSchema.from_dict(manifest["schema"])
    n_features = schema.n_features

    autoencoder, _ = _make_autoencoder(n_features)
    autoencoder.load_state_dict(torch.load(directory / "autoencoder.pt", map_location="cpu"))
    autoencoder.eval()

    stage_model = _make_stage_model(n_features, len(manifest.get("class_labels", STAGE_LABELS)))
    stage_model.load_state_dict(torch.load(directory / "stage_lstm.pt", map_location="cpu"))
    stage_model.eval()

    health_model = CatBoostRegressor()
    health_model.load_model(str(directory / "health.cbm"))

    gp_prototype = joblib.load(directory / "gp.pkl")

    calibration = None
    if manifest.get("calibration"):
        calibration = BaselineCalibration.from_dict(manifest["calibration"])

    return TrainedModels(
        schema=schema,
        window=int(manifest.get("window", 30)),
        horizon=int(manifest.get("horizon", 15)),
        autoencoder=autoencoder,
        calibration=calibration,
        health_model=health_model,
        stage_model=stage_model,
        gp_prototype=gp_prototype,
        labelled=bool(manifest.get("labelled", False)),
        class_labels=list(manifest.get("class_labels", STAGE_LABELS)),
        feature_importances=list(manifest.get("feature_importances", [])),
        metrics=manifest.get("metrics", {}),
        elapsed_s=float(manifest.get("elapsed_s", 0.0)),
        source=manifest.get("source", str(directory.name)),
    )


def bundle_exists(directory: str | Path) -> bool:
    return (Path(directory) / MANIFEST).exists()


def describe_bundle(directory: str | Path) -> Optional[dict]:
    """Read just the manifest, for listing without loading the models."""
    path = Path(directory) / MANIFEST
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays so json.dump accepts the metrics blob."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
