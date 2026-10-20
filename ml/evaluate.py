"""Load all four serialized artifacts from app/models/ and report held-out metrics.

    python ml/evaluate.py                  # held-out test engines (default)
    python ml/evaluate.py --split official # the untouched official FD001 test set
    python ml/evaluate.py --split all      # val, test and official
    python ml/evaluate.py --json out.json  # also dump the numbers

This is the acceptance check for the Phase B contract. Every model is driven
**only** through its ``*_meta.json`` -- feature order, normalization statistics,
window length, thresholds and class labels are all read from the sidecar, never
hardcoded here. The models are loaded from disk in their serialized form, exactly
as ``app/`` will load them:

    autoencoder      onnxruntime session over autoencoder.onnx (TorchScript fallback)
    gpr              joblib.load -> sklearn GaussianProcessRegressor
    lstm             onnxruntime session over lstm.onnx (TorchScript fallback)
    catboost_health  CatBoostRegressor.load_model

So if this script prints sensible numbers, the UI layer can reproduce them with
the same five lines of code. No PySide6/Qt import appears anywhere in ml/, and
that is asserted at the end of the run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # allow `python ml/evaluate.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import ARTIFACT_DIR, PROCESSED_NPZ, rel  # noqa: E402

SPLIT_CHOICES = ("val", "test", "official", "all")


# ------------------------------------------------------------------ loading helpers


def load_meta(name: str) -> dict:
    path = ARTIFACT_DIR / f"{name}_meta.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{rel(path)} not found. Train it first: python ml/train_models.py --model {name}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class OnnxOrTorch:
    """Run a graph through onnxruntime, falling back to the TorchScript twin."""

    def __init__(self, meta: dict):
        self.backend = "none"
        self._session = None
        self._module = None
        onnx_name = meta.get("artifacts", {}).get("onnx")
        if onnx_name and (ARTIFACT_DIR / onnx_name).exists():
            try:
                import onnxruntime as ort

                self._session = ort.InferenceSession(
                    str(ARTIFACT_DIR / onnx_name), providers=["CPUExecutionProvider"]
                )
                self._input = self._session.get_inputs()[0].name
                self.backend = f"onnxruntime:{onnx_name}"
                return
            except Exception as exc:  # noqa: BLE001 - fall back below
                print(f"  (onnxruntime unavailable: {type(exc).__name__}; using TorchScript)")
        pt_name = meta.get("artifacts", {}).get("torchscript")
        if pt_name and (ARTIFACT_DIR / pt_name).exists():
            import torch

            self._module = torch.jit.load(str(ARTIFACT_DIR / pt_name))
            self._module.eval()
            self.backend = f"torchscript:{pt_name}"
            return
        raise FileNotFoundError(f"no runnable artifact for {meta.get('name')}")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self._session is not None:
            return np.asarray(self._session.run(None, {self._input: x})[0])
        import torch

        with torch.no_grad():
            return self._module(torch.from_numpy(x)).numpy()


def normalize_from_meta(raw: np.ndarray, meta: dict) -> np.ndarray:
    """Apply a sidecar's normalization block to raw feature values.

    Present to prove the meta.json alone is sufficient; the cached arrays are
    already normalized, so evaluation reads those and this is exercised by
    ``--check-normalization``.
    """
    block = meta["normalization"]
    low = np.asarray(block["min"], dtype=np.float64)
    high = np.asarray(block["max"], dtype=np.float64)
    span = np.where((high - low) < 1e-12, 1.0, high - low)
    lo_clip, hi_clip = block.get("clip_range", [-0.5, 1.5])
    return np.clip((raw - low) / span, lo_clip, hi_clip).astype(np.float32)


def _assert_feature_order(meta: dict, dataset_features: list[str], name: str) -> None:
    declared = meta["input"]["feature_names"]
    if name == "gpr":  # single-sensor model
        if declared[0] not in dataset_features:
            raise AssertionError(f"{name}: sensor {declared[0]} not in the dataset contract")
        return
    if declared != dataset_features:
        raise AssertionError(
            f"{name}: meta.json feature order does not match the dataset\n"
            f"  meta:    {declared}\n  dataset: {dataset_features}"
        )


# ------------------------------------------------------------------ per-model evaluation


def evaluate_autoencoder(ds, split: str) -> dict:
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    meta = load_meta("autoencoder")
    _assert_feature_order(meta, ds.features, "autoencoder")
    runner = OnnxOrTorch(meta)
    threshold = float(meta["thresholds"]["reconstruction_error"])

    view = ds.tabular(split)
    x = view["X"].astype(np.float32)
    recon = runner(x)
    scores = ((recon - x) ** 2).mean(axis=1)
    fault = view["fault"].astype(np.int64)
    predicted = (scores > threshold).astype(np.int64)

    out = {
        "backend": runner.backend,
        "threshold": threshold,
        "n_rows": int(len(x)),
        "fault_prevalence": float(fault.mean()),
        "flagged_fraction": float(predicted.mean()),
        "mean_reconstruction_error": float(scores.mean()),
    }
    if len(np.unique(fault)) > 1:
        precision, recall, f1, _ = precision_recall_fscore_support(
            fault, predicted, average="binary", zero_division=0
        )
        out.update(
            auroc=float(roc_auc_score(fault, scores)),
            average_precision=float(average_precision_score(fault, scores)),
            precision_at_threshold=float(precision),
            recall_at_threshold=float(recall),
            f1_at_threshold=float(f1),
        )
    return out


def evaluate_gpr(ds, split: str) -> dict:
    import joblib

    from ml.models.gpr import CONFIDENCE_Z, extract_windows, series_by_unit, time_grid

    meta = load_meta("gpr")
    sensor = meta["input"]["feature_names"][0]
    window = int(meta["input"]["window_length"])
    horizon = int(meta["output"]["shape"][0])
    gp = joblib.load(ARTIFACT_DIR / meta["artifacts"]["sklearn_pickle"])

    past_t, future_t = time_grid(window, horizon)
    y_past, y_future = extract_windows(
        series_by_unit(ds, split, sensor), window, horizon, stride=10
    )
    if len(y_past) == 0:
        return {"n_windows": 0}

    # Exactly the app's call sequence: condition on the window, then predict.
    means, stds = [], []
    for row in y_past:
        gp.fit(past_t, row)
        mean, std = gp.predict(future_t, return_std=True)
        means.append(mean)
        stds.append(std)
    mean = np.stack(means)
    std = np.stack(stds)

    error = mean - y_future
    inside = (y_future >= mean - CONFIDENCE_Z * std) & (y_future <= mean + CONFIDENCE_Z * std)
    col = ds.feature_index(sensor)
    raw_span = float(ds.scaler_max[col] - ds.scaler_min[col])
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "backend": f"joblib:{meta['artifacts']['sklearn_pickle']}",
        "target_sensor": sensor,
        "window": window,
        "horizon": horizon,
        "n_windows": int(len(y_past)),
        "rmse_normalized": rmse,
        "rmse_raw_sensor_units": rmse * raw_span,
        "mae_normalized": float(np.mean(np.abs(error))),
        "coverage_95": float(inside.mean()),
        "nominal_coverage": 0.95,
        "mean_interval_width": float(np.mean(2 * CONFIDENCE_Z * std)),
        "baseline_persistence_rmse": float(np.sqrt(np.mean((y_past[:, -1:] - y_future) ** 2))),
    }


def evaluate_lstm(ds, split: str) -> dict:
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    meta = load_meta("lstm")
    _assert_feature_order(meta, ds.features, "lstm")
    labels = meta["output"]["class_labels"]
    window = int(meta["input"]["window_length"])
    if window != ds.window:
        raise AssertionError(f"lstm meta window {window} != dataset window {ds.window}")
    runner = OnnxOrTorch(meta)

    view = ds.sequences(split)
    y_true = view["stage"].astype(np.int64)
    y_pred = runner(view["X"].astype(np.float32)).argmax(axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(labels)), zero_division=0
    )
    return {
        "backend": runner.backend,
        "window": window,
        "class_labels": labels,
        "n_sequences": int(len(y_true)),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(labels))
        ).tolist(),
    }


def evaluate_catboost(ds, split: str) -> dict:
    from catboost import CatBoostRegressor

    from ml.common import RUL_CEILING, regression_metrics

    meta = load_meta("catboost_health")
    _assert_feature_order(meta, ds.features, "catboost_health")
    model = CatBoostRegressor()
    model.load_model(str(ARTIFACT_DIR / meta["artifacts"]["catboost"]))

    view = ds.tabular(split)
    y_true = view["health"].astype(np.float64)
    y_pred = np.clip(model.predict(view["X"].astype(np.float64)), 0.0, 100.0)
    stats = regression_metrics(y_true, y_pred)
    top = meta.get("feature_importances", [])[:5]
    return {
        "backend": f"catboost:{meta['artifacts']['catboost']}",
        "n_rows": int(len(y_true)),
        "rmse_health_points": stats["rmse"],
        "mae_health_points": stats["mae"],
        "r2": stats["r2"],
        "rmse_rul_cycles": stats["rmse"] * RUL_CEILING / 100.0,
        "top_features": [f"{d['feature']} ({d['importance']:.1f})" for d in top],
    }


EVALUATORS = {
    "autoencoder": evaluate_autoencoder,
    "gpr": evaluate_gpr,
    "lstm": evaluate_lstm,
    "catboost_health": evaluate_catboost,
}


# ------------------------------------------------------------------ reporting


def _print_block(title: str, payload: dict, indent: int = 2) -> None:
    print(f"\n{title}")
    _print_values(payload, indent)


def _print_values(payload: dict, indent: int) -> None:
    pad = " " * indent
    for key, value in payload.items():
        if isinstance(value, dict):
            print(f"{pad}{key}:")
            _print_values(value, indent + 2)
        elif isinstance(value, float):
            print(f"{pad}{key:32s} {value:.4f}")
        elif isinstance(value, list):
            print(f"{pad}{key:32s} {value}")
        else:
            print(f"{pad}{key:32s} {value}")


def check_normalization(ds) -> None:
    """Verify a sidecar's normalization block reproduces the cached arrays.

    This is the load-bearing claim for Phase B: the UI will start from raw
    telemetry, so the min/max in meta.json must be all it needs.
    """
    import pandas as pd

    from ml.common import CMAPSS_DIR, RAW_COLUMNS
    from ml.preprocess import add_run_to_failure_rul, split_units

    meta = load_meta("catboost_health")
    raw = pd.read_csv(
        CMAPSS_DIR / "train_FD001.txt", sep=r"\s+", header=None, names=RAW_COLUMNS
    )
    labelled = add_run_to_failure_rul(raw)
    units = split_units(labelled["unit"].unique())["test"]
    frame = labelled[labelled["unit"].isin(units)].reset_index(drop=True)

    features = meta["input"]["feature_names"]
    from_meta = normalize_from_meta(frame[features].to_numpy(dtype=np.float64), meta)
    cached = ds.tabular("test")["X"]
    worst = float(np.abs(from_meta - cached).max())
    status = "ok" if worst < 1e-6 else "MISMATCH"
    print(
        f"\nnormalization round-trip from meta.json alone: "
        f"max abs diff vs cached arrays = {worst:.2e}  {status}"
    )
    if worst >= 1e-6:
        raise AssertionError("meta.json normalization does not reproduce the training arrays")


def assert_qt_free() -> None:
    leaked = sorted(m for m in sys.modules if m.startswith(("PySide6", "PyQt5", "PyQt6")))
    if leaked:
        raise AssertionError(f"Qt modules imported during evaluation: {leaked}")
    print("Qt-free check: no PySide6/PyQt module imported by ml/  ok")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", choices=SPLIT_CHOICES, default="test")
    ap.add_argument(
        "--model", choices=(*EVALUATORS, "all"), default="all", help="evaluate one model only"
    )
    ap.add_argument("--json", type=Path, default=None, help="also write the metrics as JSON")
    ap.add_argument(
        "--check-normalization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-derive normalized features from meta.json and compare to the cache",
    )
    args = ap.parse_args(argv)

    if not PROCESSED_NPZ.exists():
        raise SystemExit(
            f"{rel(PROCESSED_NPZ)} not found. Run: python ml/preprocess.py"
        )
    from ml.preprocess import Dataset

    ds = Dataset()

    splits = ("val", "test", "official") if args.split == "all" else (args.split,)
    names = tuple(EVALUATORS) if args.model == "all" else (args.model,)

    print("=" * 74)
    print(f"evaluating artifacts in {rel(ARTIFACT_DIR)}")
    print(f"features ({len(ds.features)}): {', '.join(ds.features)}")
    print(f"window: {ds.window}   scaler: {ds.scaler_mode}")
    print("=" * 74)

    results: dict[str, dict] = {}
    for split in splits:
        engines = len(np.unique(ds.tabular(split)["unit"]))
        header = f"### split '{split}'  ({engines} engines, {len(ds.tabular(split)['X'])} rows)"
        print(f"\n{header}\n{'-' * len(header)}")
        results[split] = {}
        for name in names:
            metrics = EVALUATORS[name](ds, split)
            results[split][name] = metrics
            _print_block(f"{name}:", metrics)

    if args.check_normalization:
        check_normalization(ds)

    print()
    assert_qt_free()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {rel(args.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
