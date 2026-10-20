"""Model 3/4 -- LSTM sequence classifier over degradation stage.

Task
----
Given a window of 30 consecutive cycles (17 normalized features each), classify
the engine's degradation stage *at the last cycle of the window*:

    0  healthy    RUL > 75
    1  degrading  30 < RUL <= 75
    2  critical   RUL <= 30

These stages are derived purely from RUL thresholds -- they are a health
classification, **not** flight phases. RUL is capped at 125 before thresholding,
consistent with the rest of the pipeline.

Architecture
------------
2-layer LSTM, hidden size 64, dropout 0.3 between layers, then a linear head on
the final timestep's hidden state. The final hidden state is the right readout
here because the label is defined at the window's last cycle.

Class imbalance: the stages are roughly 74/19/7 in the training split, so the
loss is weighted by inverse class frequency. Without that the model can score
~74% accuracy by always predicting "healthy" while never flagging a critical
engine -- exactly the failure mode that matters. Model selection uses macro-F1
on the validation split rather than accuracy, for the same reason.

Output contract
---------------
``(N, 30, 17) -> (N, 3)`` raw logits. The app applies softmax for probabilities
and argmax for the label. Logits (not probabilities) are exported so the graph
stays a plain tensor op and the app can apply its own temperature or per-class
threshold if it wants to bias toward recall on "critical".
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn

from ml.artifacts import artifact_path, build_meta, report, save_meta, stage_output_block
from ml.common import (
    STAGE_CRITICAL_MAX_RUL,
    STAGE_DEGRADING_MAX_RUL,
    STAGE_LABELS,
    seed_everything,
)
from ml.models._torch_utils import (
    EarlyStopping,
    batches,
    count_params,
    export_onnx,
    export_torchscript,
    verify_onnx,
)
from ml.preprocess import Dataset, ensure_prepared

NAME = "lstm"

DEFAULTS = {
    "epochs": 120,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "hidden": 64,
    "layers": 2,
    "dropout": 0.3,
    "patience": 15,
}


class StageClassifier(nn.Module):
    """LSTM over a cycle window -> logits for the stage at the final cycle."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        layers: int = 2,
        n_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])  # label is defined at the last cycle


def predict_logits(model: nn.Module, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            chunk = torch.from_numpy(x[start : start + batch_size].astype(np.float32))
            chunks.append(model(chunk).numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, len(STAGE_LABELS)), dtype=np.float32)


def _classification_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    y_pred = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(STAGE_LABELS)), zero_division=0
    )
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(STAGE_LABELS)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=range(len(STAGE_LABELS))
        ).tolist(),
        "confusion_matrix_axes": "rows = true stage, columns = predicted stage",
    }


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    ds: Dataset = ensure_prepared()
    n_features = len(ds.features)

    train_view, val_view = ds.sequences("train"), ds.sequences("val")
    x_train = train_view["X"].astype(np.float32)
    y_train = train_view["stage"].astype(np.int64)
    x_val = val_view["X"].astype(np.float32)
    y_val = val_view["stage"].astype(np.int64)

    counts = np.bincount(y_train, minlength=len(STAGE_LABELS))
    distribution = ", ".join(
        f"{label} {c} ({100 * c / len(y_train):.1f}%)" for label, c in zip(STAGE_LABELS, counts)
    )
    print(f"train sequences {len(x_train)} of shape {x_train.shape[1:]}  ->  {distribution}")
    print(f"val sequences   {len(x_val)}")

    # Inverse-frequency class weights, mean-normalized to keep the loss scale sane.
    weights = len(y_train) / (len(STAGE_LABELS) * np.maximum(counts, 1))
    weights = weights / weights.mean()
    print(f"class weights: {dict(zip(STAGE_LABELS, weights.round(3)))}")

    model = StageClassifier(
        n_features, hidden=args.hidden, layers=args.layers, dropout=args.dropout
    )
    print(
        f"model: LSTM(in={n_features}, hidden={args.hidden}, layers={args.layers}) "
        f"-> Linear({args.hidden}, {len(STAGE_LABELS)})  ({count_params(model)} params)"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights.astype(np.float32)))
    # Select on macro-F1, not accuracy: accuracy is dominated by the majority class.
    stopper = EarlyStopping(patience=args.patience, min_delta=1e-4)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for idx in batches(len(x_train), args.batch_size, seed=args.seed + epoch):
            xb = torch.from_numpy(x_train[idx])
            yb = torch.from_numpy(y_train[idx])
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(idx)
        train_loss = running / len(x_train)

        val_logits = predict_logits(model, x_val)
        val_macro_f1 = f1_score(y_val, val_logits.argmax(axis=1), average="macro", zero_division=0)
        val_accuracy = float((val_logits.argmax(axis=1) == y_val).mean())
        scheduler.step(1.0 - val_macro_f1)

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch:3d}  train loss {train_loss:.4f}  "
                f"val acc {val_accuracy:.4f}  val macroF1 {val_macro_f1:.4f}"
            )
        if stopper.step(1.0 - val_macro_f1, model, epoch):
            print(f"  early stop at epoch {epoch} (best epoch {stopper.best_epoch})")
            break

    model = stopper.restore(model)
    print(f"  best val macro-F1: {1.0 - stopper.best:.4f} (epoch {stopper.best_epoch})")

    metrics: dict = {
        "best_val_macro_f1": float(1.0 - stopper.best),
        "best_epoch": int(stopper.best_epoch),
        "class_weights": {label: float(w) for label, w in zip(STAGE_LABELS, weights)},
        "train_class_counts": {label: int(c) for label, c in zip(STAGE_LABELS, counts)},
    }
    for split in ("val", "test", "official"):
        view = ds.sequences(split)
        logits = predict_logits(model, view["X"].astype(np.float32))
        metrics[split] = _classification_metrics(view["stage"].astype(np.int64), logits)

    print("\nheld-out test split, per-class report:")
    test_view = ds.sequences("test")
    print(
        classification_report(
            test_view["stage"].astype(np.int64),
            predict_logits(model, test_view["X"].astype(np.float32)).argmax(axis=1),
            labels=range(len(STAGE_LABELS)),
            target_names=STAGE_LABELS,
            zero_division=0,
            digits=4,
        )
    )

    # ------------------------------------------------------------ serialize
    example = torch.from_numpy(ds.sequences("test")["X"][:4].astype(np.float32))
    pt_path = export_torchscript(model, example, artifact_path(f"{NAME}.pt"))
    onnx_path = export_onnx(model, example, artifact_path(f"{NAME}.onnx"))
    written = [pt_path]
    artifacts = {"torchscript": pt_path.name}
    if onnx_path is not None:
        model.eval()
        with torch.no_grad():
            expected = model(example).numpy()
        verify_onnx(onnx_path, example, expected)
        written.append(onnx_path)
        artifacts["onnx"] = onnx_path.name

    meta = build_meta(
        name=NAME,
        task="degradation_stage_classification",
        dataset=ds,
        input_shape=[-1, ds.window, n_features],
        window_length=ds.window,
        output=stage_output_block(),
        artifacts=artifacts,
        thresholds={
            "stage_definition_rul": {
                "healthy": f"RUL > {STAGE_DEGRADING_MAX_RUL}",
                "degrading": f"{STAGE_CRITICAL_MAX_RUL} < RUL <= {STAGE_DEGRADING_MAX_RUL}",
                "critical": f"RUL <= {STAGE_CRITICAL_MAX_RUL}",
            },
            "decision_rule": "argmax over the 3 logits",
            "note": "no probability cutoff is baked in; the app may apply a lower "
                    "threshold on 'critical' to trade precision for recall",
        },
        inference={
            "protocol": [
                f"1. Keep a rolling buffer of the last {ds.window} cycles, each holding the "
                f"{n_features} features in input.feature_names order.",
                "2. Normalize every value with normalization.min/max per the stated formula.",
                f"3. Shape the buffer to (1, {ds.window}, {n_features}) float32, "
                "axis 1 ordered oldest -> newest.",
                "4. Run the graph -> (1, 3) logits.",
                "5. probabilities = softmax(logits); stage = output.class_labels[argmax(logits)].",
            ],
            "onnx_input_name": "input",
            "onnx_output_name": "output",
            "stateless": True,
            "padding_rule": (
                f"fewer than {ds.window} cycles available: left-pad by repeating the "
                "oldest row, which is how training sequences were built"
            ),
            "label_position": "the stage refers to the LAST cycle in the window",
        },
        metrics=metrics,
        extra={
            "architecture": {
                "type": "lstm_classifier",
                "input_size": n_features,
                "hidden_size": args.hidden,
                "num_layers": args.layers,
                "dropout": args.dropout,
                "readout": "final timestep hidden state -> linear",
                "n_classes": len(STAGE_LABELS),
                "parameters": count_params(model),
            },
            "training": {
                "loss": "cross entropy with inverse-frequency class weights",
                "model_selection": "best validation macro-F1",
            },
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
    ap.add_argument("--hidden", type=int, default=DEFAULTS["hidden"])
    ap.add_argument("--layers", type=int, default=DEFAULTS["layers"])
    ap.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    ap.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = add_arguments(argparse.ArgumentParser(description=__doc__))
    train(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
