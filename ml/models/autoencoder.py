"""Model 1/4 -- dense autoencoder for unsupervised fault detection.

Idea
----
Train a narrow autoencoder on *healthy* cycles only (the first 30% of each
training engine's life). It learns to reconstruct the as-new sensor signature
accurately. As an engine degrades, its readings drift away from that signature
and reconstruction error climbs -- no fault labels are used in training, which
is the point: the same detector works on a novel failure mode.

Architecture: 17 -> 32 -> 16 -> 4 (bottleneck) -> 16 -> 32 -> 17, tanh
activations, trained with MSE. The 4-unit bottleneck is deliberately tight: FD001
degradation is driven by roughly two latent factors (HPC efficiency and flow
degradation), so a wide bottleneck would let the net learn an identity map and
reconstruct faulty rows just as well as healthy ones.

Reported metric: AUROC of the per-row reconstruction error against RUL-based
fault labels (``RUL < 30`` = fault) on held-out engines. Also reports average
precision, and precision/recall/F1 at the operating threshold.

Output contract
---------------
The exported graph maps a batch of normalized cycles to their reconstruction,
``(N, 17) -> (N, 17)``. The *app* computes the anomaly score itself as the mean
squared error per row and compares it to ``thresholds.reconstruction_error``.
That keeps the graph a plain tensor op (portable to ONNX) and leaves the
threshold tunable without retraining.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn

from ml.artifacts import artifact_path, build_meta, report, save_meta
from ml.common import FAULT_RUL_THRESHOLD, HEALTHY_LIFE_FRACTION, seed_everything
from ml.models._torch_utils import (
    EarlyStopping,
    batches,
    count_params,
    export_onnx,
    export_torchscript,
    verify_onnx,
)
from ml.preprocess import Dataset, ensure_prepared

NAME = "autoencoder"

DEFAULTS = {
    "epochs": 300,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "bottleneck": 4,
    "patience": 25,
    "threshold_percentile": 99.0,
}


class DenseAutoencoder(nn.Module):
    """Symmetric dense autoencoder. Forward returns the reconstruction."""

    def __init__(self, n_features: int, bottleneck: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def reconstruction_error(model: nn.Module, x: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Mean squared error per row -- the anomaly score."""
    model.eval()
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.from_numpy(x[start : start + batch_size].astype(np.float32))
            recon = model(chunk)
            scores.append(((recon - chunk) ** 2).mean(dim=1).numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def _fault_metrics(scores: np.ndarray, fault: np.ndarray, threshold: float) -> dict:
    """AUROC/AP plus point metrics at the operating threshold."""
    if len(np.unique(fault)) < 2:  # single-class split -> ranking metrics undefined
        return {"auroc": float("nan"), "average_precision": float("nan")}
    predicted = (scores > threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        fault, predicted, average="binary", zero_division=0
    )
    return {
        "auroc": float(roc_auc_score(fault, scores)),
        "average_precision": float(average_precision_score(fault, scores)),
        "precision_at_threshold": float(precision),
        "recall_at_threshold": float(recall),
        "f1_at_threshold": float(f1),
        "flagged_fraction": float(predicted.mean()),
        "fault_prevalence": float(fault.mean()),
    }


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    ds: Dataset = ensure_prepared()
    n_features = len(ds.features)

    # Train on healthy rows only -- no fault labels enter the objective.
    x_train = ds.healthy("train")["X"].astype(np.float32)
    x_val = ds.healthy("val")["X"].astype(np.float32)
    print(
        f"training on {len(x_train)} healthy rows "
        f"(first {HEALTHY_LIFE_FRACTION:.0%} of each training engine's life), "
        f"validating on {len(x_val)}"
    )

    model = DenseAutoencoder(n_features, bottleneck=args.bottleneck)
    print(f"model: 17 -> 32 -> 16 -> {args.bottleneck} -> 16 -> 32 -> 17  "
          f"({count_params(model)} params)")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopping(patience=args.patience)

    val_tensor = torch.from_numpy(x_val)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for idx in batches(len(x_train), args.batch_size, seed=args.seed + epoch):
            chunk = torch.from_numpy(x_train[idx])
            optimizer.zero_grad()
            loss = loss_fn(model(chunk), chunk)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(idx)
        train_loss = running / len(x_train)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_tensor), val_tensor).item()
        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {train_loss:.6f}  val {val_loss:.6f}")
        if stopper.step(val_loss, model, epoch):
            print(f"  early stop at epoch {epoch} (best epoch {stopper.best_epoch})")
            break

    model = stopper.restore(model)
    print(f"  best val reconstruction MSE: {stopper.best:.6f} (epoch {stopper.best_epoch})")

    # Threshold from the healthy training distribution only -- never from labels.
    healthy_scores = reconstruction_error(model, x_train)
    threshold = float(np.percentile(healthy_scores, args.threshold_percentile))
    print(
        f"  anomaly threshold = p{args.threshold_percentile:g} of healthy train error "
        f"= {threshold:.6f}"
    )

    metrics: dict = {
        "best_val_reconstruction_mse": float(stopper.best),
        "best_epoch": int(stopper.best_epoch),
        "healthy_train_error_mean": float(healthy_scores.mean()),
        "healthy_train_error_p99": threshold,
    }
    for split in ("val", "test", "official"):
        tab = ds.tabular(split)
        scores = reconstruction_error(model, tab["X"].astype(np.float32))
        metrics[split] = _fault_metrics(scores, tab["fault"], threshold)

    # ------------------------------------------------------------ serialize
    example = torch.from_numpy(ds.tabular("test")["X"][:4].astype(np.float32))
    pt_path = export_torchscript(model, example, artifact_path(f"{NAME}.pt"))
    onnx_path = export_onnx(model, example, artifact_path(f"{NAME}.onnx"))
    written = [pt_path]
    if onnx_path is not None:
        with torch.no_grad():
            expected = model(example).numpy()
        verify_onnx(onnx_path, example, expected)
        written.append(onnx_path)

    artifacts = {"torchscript": pt_path.name}
    if onnx_path is not None:
        artifacts["onnx"] = onnx_path.name

    meta = build_meta(
        name=NAME,
        task="anomaly_detection_via_reconstruction_error",
        dataset=ds,
        input_shape=[-1, n_features],
        output={
            "shape": [-1, n_features],
            "dtype": "float32",
            "meaning": "reconstruction of the normalized input row",
            "anomaly_score": {
                "formula": "mean((output - input) ** 2, axis=-1)",
                "computed_by": "app (not baked into the graph, so the threshold stays tunable)",
                "direction": "higher = more anomalous",
            },
        },
        artifacts=artifacts,
        thresholds={
            "reconstruction_error": threshold,
            "percentile_of_healthy_train": args.threshold_percentile,
            "decision_rule": "fault if anomaly_score > reconstruction_error",
            "fault_label_definition": f"RUL < {FAULT_RUL_THRESHOLD} cycles",
        },
        inference={
            "protocol": [
                "1. Collect one telemetry cycle as the 17 raw features, in input.feature_names order.",
                "2. Normalize with normalization.min/max per the stated formula.",
                "3. Run the graph: (1, 17) float32 -> (1, 17) float32 reconstruction.",
                "4. score = mean((recon - x) ** 2).",
                "5. Flag a fault when score > thresholds.reconstruction_error.",
            ],
            "onnx_input_name": "input",
            "onnx_output_name": "output",
            "stateless": True,
            "healthy_baseline": (
                f"trained only on the first {HEALTHY_LIFE_FRACTION:.0%} of each engine's life"
            ),
        },
        metrics=metrics,
        extra={
            "architecture": {
                "type": "dense_autoencoder",
                "layers": [n_features, 32, 16, args.bottleneck, 16, 32, n_features],
                "activation": "tanh",
                "bottleneck": args.bottleneck,
                "parameters": count_params(model),
            }
        },
    )
    written.append(save_meta(NAME, meta))
    report(NAME, metrics, written)
    return metrics


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--bottleneck", type=int, default=DEFAULTS["bottleneck"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument(
        "--threshold-percentile", type=float, default=DEFAULTS["threshold_percentile"]
    )
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = add_arguments(argparse.ArgumentParser(description=__doc__))
    train(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
