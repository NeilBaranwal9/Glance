"""Small torch helpers shared by the autoencoder and the LSTM.

Kept deliberately thin -- just the bits that would otherwise be copy-pasted:
an early-stopping tracker and the dual TorchScript/ONNX export.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ml.common import rel


class EarlyStopping:
    """Track the best validation loss and keep a copy of the winning weights."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = -1
        self.best_state: dict | None = None
        self._since_improved = 0

    def step(self, loss: float, model: nn.Module, epoch: int) -> bool:
        """Record an epoch. Returns True when training should stop."""
        if loss < self.best - self.min_delta:
            self.best = loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self._since_improved = 0
        else:
            self._since_improved += 1
        return self._since_improved >= self.patience

    def restore(self, model: nn.Module) -> nn.Module:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def batches(n: int, batch_size: int, shuffle: bool = True, seed: int | None = None):
    """Yield index slices over ``n`` rows."""
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def export_torchscript(model: nn.Module, example: torch.Tensor, path: Path) -> Path:
    """Trace to TorchScript so ``app/`` can load it with torch alone."""
    model.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(model, example)
    scripted = torch.jit.freeze(scripted)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(path))
    return path


def export_onnx(
    model: nn.Module,
    example: torch.Tensor,
    path: Path,
    input_name: str = "input",
    output_name: str = "output",
) -> Path | None:
    """Export to ONNX so ``app/`` can run on onnxruntime without torch.

    Returns None (with a warning) rather than failing the whole training run if
    the exporter is unavailable -- the TorchScript artifact is always written.
    """
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            example,
            str(path),
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    except Exception as exc:  # noqa: BLE001 - optional artifact
        print(f"  WARNING: ONNX export failed ({type(exc).__name__}: {exc})")
        print("           TorchScript artifact is still valid; app/ can use the .pt file.")
        return None
    return path


def verify_onnx(path: Path, example: torch.Tensor, expected: np.ndarray, tol: float = 1e-4) -> bool:
    """Run the exported graph through onnxruntime and compare to the torch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed; skipping ONNX numerical check)")
        return True
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: example.numpy()})[0]
    max_diff = float(np.abs(np.asarray(got) - expected).max())
    ok = max_diff < tol
    status = "ok" if ok else "MISMATCH"
    print(f"  onnx check: max|torch - onnx| = {max_diff:.2e}  {status}  ({rel(path)})")
    return ok


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
