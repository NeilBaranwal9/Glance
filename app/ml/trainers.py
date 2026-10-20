"""Train all four algorithms on an arbitrary N-channel dataset, in-process.

This is the "bring your own data" half of the ML Analysis tab. Unlike the
offline pipeline in ``ml/``, model shapes are derived from the data: the
autoencoder and LSTM are built with ``input_size = schema.n_features``, CatBoost
gets N columns, and the Gaussian Process is fitted per channel on demand.

Everything is sized to finish in seconds, because a person is watching a
progress bar. Heavy imports (torch, catboost) are deferred to call time so the
app starts fast when the tab is never opened.

All entry points take a ``TrainingProgress`` callback and an ``is_cancelled``
predicate so a Qt worker thread can report and abort without this module
importing Qt.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from app.ml.labels import (
    RUL_CEILING,
    STAGE_LABELS,
    BaselineCalibration,
    health_from_rul,
    stage_from_health,
    stage_from_rul,
)
from app.ml.schema import SensorSchema

ProgressFn = Callable[[str, float], None]
CancelFn = Callable[[], bool]

DEFAULT_WINDOW = 30
DEFAULT_HORIZON = 15
BASELINE_FRACTION = 0.30   # leading share of a run treated as nominal
CONFIDENCE_Z = 1.959963984540054


def _noop_progress(message: str, fraction: float) -> None:
    pass


def _never_cancelled() -> bool:
    return False


class TrainingCancelled(RuntimeError):
    """Raised when the caller's cancel predicate goes true mid-training."""


# ------------------------------------------------------------------ result


@dataclass
class TrainedModels:
    """Everything the online engine needs, for one N-channel schema."""

    schema: SensorSchema
    window: int = DEFAULT_WINDOW
    horizon: int = DEFAULT_HORIZON

    autoencoder: Any = None          # torch module (eval mode)
    calibration: Optional[BaselineCalibration] = None
    health_model: Any = None         # CatBoostRegressor
    stage_model: Any = None          # torch module (eval mode)
    gp_prototype: Any = None         # sklearn GaussianProcessRegressor, optimizer=None

    labelled: bool = False           # trained against true RUL rather than baseline
    class_labels: list[str] = field(default_factory=lambda: list(STAGE_LABELS))
    feature_importances: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    source: str = ""

    @property
    def n_features(self) -> int:
        return self.schema.n_features

    def summary(self) -> str:
        lines = [
            f"{self.n_features} channels, window {self.window}, horizon {self.horizon}",
            f"labels: {'run-to-failure RUL' if self.labelled else 'baseline-calibrated'}",
            f"trained in {self.elapsed_s:.1f}s from {self.source}",
        ]
        for name, block in self.metrics.items():
            if isinstance(block, dict):
                bits = ", ".join(
                    f"{k}={v:.3f}" for k, v in block.items() if isinstance(v, float)
                )
                if bits:
                    lines.append(f"  {name}: {bits}")
        return "\n".join(lines)

    def top_drivers(self, k: int = 5) -> list[dict]:
        return self.feature_importances[:k]


# ------------------------------------------------------------------ helpers


def build_windows(values: np.ndarray, units: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """One window per row (left-padded at the start of each unit).

    Returns ``(windows, row_index)`` so labels can be taken from the original rows.
    """
    values = np.asarray(values, dtype=np.float32)
    n, n_features = values.shape
    out = np.empty((n, window, n_features), dtype=np.float32)
    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        block = values[idx]
        for position, row in enumerate(idx):
            start = position - window + 1
            if start >= 0:
                out[row] = block[start : position + 1]
            else:
                pad = np.repeat(block[:1], -start, axis=0)
                out[row] = np.vstack([pad, block[: position + 1]])
    return out, np.arange(n)


def baseline_mask(units: np.ndarray, cycles: np.ndarray, fraction: float = BASELINE_FRACTION) -> np.ndarray:
    """Leading ``fraction`` of each unit's run -- the assumed-nominal window."""
    mask = np.zeros(len(units), dtype=bool)
    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        order = idx[np.argsort(cycles[idx])]
        keep = max(1, int(round(len(order) * fraction)))
        mask[order[:keep]] = True
    return mask


# ------------------------------------------------------------------ autoencoder


def _make_autoencoder(n_features: int):
    import torch
    from torch import nn

    hidden1 = int(max(16, min(64, 2 * n_features)))
    hidden2 = int(max(8, hidden1 // 2))
    bottleneck = int(max(2, min(8, max(2, n_features // 3))))

    class DenseAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden1), nn.Tanh(),
                nn.Linear(hidden1, hidden2), nn.Tanh(),
                nn.Linear(hidden2, bottleneck),
            )
            self.decoder = nn.Sequential(
                nn.Linear(bottleneck, hidden2), nn.Tanh(),
                nn.Linear(hidden2, hidden1), nn.Tanh(),
                nn.Linear(hidden1, n_features),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return DenseAutoencoder(), (n_features, hidden1, hidden2, bottleneck)


def train_autoencoder(
    healthy: np.ndarray,
    n_features: int,
    epochs: int = 120,
    batch_size: int = 256,
    lr: float = 1e-3,
    progress: ProgressFn = _noop_progress,
    is_cancelled: CancelFn = _never_cancelled,
    progress_span: tuple[float, float] = (0.0, 1.0),
):
    """Fit the reconstruction model on nominal rows only."""
    import torch
    from torch import nn

    model, shape = _make_autoencoder(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data = torch.from_numpy(np.asarray(healthy, dtype=np.float32))
    n = len(data)
    lo, hi = progress_span

    best = float("inf")
    best_state = None
    stale = 0
    rng = np.random.default_rng(0)
    for epoch in range(epochs):
        if is_cancelled():
            raise TrainingCancelled()
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            chunk = data[order[start : start + batch_size]]
            optimizer.zero_grad()
            loss = loss_fn(model(chunk), chunk)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(chunk)
        epoch_loss = total / max(n, 1)
        if epoch_loss < best - 1e-6:
            best, stale = epoch_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 20:
                break
        if epoch % 5 == 0:
            progress(f"Autoencoder epoch {epoch}  loss {epoch_loss:.5f}",
                     lo + (hi - lo) * epoch / epochs)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"reconstruction_mse": float(best), "layers": list(shape)}


def reconstruction_error(model, values: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    import torch

    values = np.asarray(values, dtype=np.float32)
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch):
            chunk = torch.from_numpy(values[start : start + batch])
            recon = model(chunk)
            out.append(((recon - chunk) ** 2).mean(dim=1).numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float64)


def per_channel_residual(model, row: np.ndarray) -> np.ndarray:
    """Squared error per channel for one sample -- the anomaly attribution."""
    import torch

    x = torch.from_numpy(np.asarray(row, dtype=np.float32).reshape(1, -1))
    model.eval()
    with torch.no_grad():
        recon = model(x)
        return ((recon - x) ** 2).numpy().ravel()


# ------------------------------------------------------------------ stage LSTM


def _make_stage_model(n_features: int, n_classes: int, hidden: int = 48, layers: int = 1):
    import torch
    from torch import nn

    class StageClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.head(output[:, -1, :])

    return StageClassifier()


def train_stage_model(
    windows: np.ndarray,
    stages: np.ndarray,
    n_features: int,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 2e-3,
    progress: ProgressFn = _noop_progress,
    is_cancelled: CancelFn = _never_cancelled,
    progress_span: tuple[float, float] = (0.0, 1.0),
):
    """Sequence classifier over degradation stage, class-weighted."""
    import torch
    from torch import nn

    n_classes = len(STAGE_LABELS)
    model = _make_stage_model(n_features, n_classes)
    counts = np.bincount(stages, minlength=n_classes).astype(np.float64)
    weights = len(stages) / (n_classes * np.maximum(counts, 1.0))
    weights = weights / weights.mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    y = torch.from_numpy(np.asarray(stages, dtype=np.int64))
    n = len(x)
    lo, hi = progress_span
    rng = np.random.default_rng(0)

    best, best_state, stale = float("inf"), None, 0
    for epoch in range(epochs):
        if is_cancelled():
            raise TrainingCancelled()
        model.train()
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            sel = order[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x[sel]), y[sel])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(sel)
        epoch_loss = total / max(n, 1)
        if epoch_loss < best - 1e-5:
            best, stale = epoch_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
        progress(f"Stage LSTM epoch {epoch}  loss {epoch_loss:.4f}",
                 lo + (hi - lo) * epoch / epochs)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predicted = model(x).argmax(dim=1).numpy()
    accuracy = float((predicted == stages).mean())
    per_class = {}
    for i, label in enumerate(STAGE_LABELS):
        mask = stages == i
        per_class[label] = float((predicted[mask] == i).mean()) if mask.any() else float("nan")
    return model, {"train_accuracy": accuracy, "recall_per_class": per_class,
                   "class_counts": {l: int(c) for l, c in zip(STAGE_LABELS, counts)}}


def predict_stage(model, window: np.ndarray) -> tuple[int, np.ndarray]:
    """One window ``(W, N)`` -> ``(stage index, probabilities)``."""
    import torch

    x = torch.from_numpy(np.asarray(window, dtype=np.float32)[None, ...])
    model.eval()
    with torch.no_grad():
        logits = model(x).numpy().ravel()
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    probabilities = exp / exp.sum()
    return int(logits.argmax()), probabilities


# ------------------------------------------------------------------ health regressor


def train_health_model(
    values: np.ndarray,
    health: np.ndarray,
    feature_names: list[str],
    iterations: int = 400,
    progress: ProgressFn = _noop_progress,
    is_cancelled: CancelFn = _never_cancelled,
):
    """Gradient-boosted regressor mapping channels -> 0-100 health.

    Its real contribution over the autoencoder is attribution: the feature
    importances say *which channels* are driving the health estimate, which is
    what an operator needs in order to act.
    """
    from catboost import CatBoostRegressor, Pool

    if is_cancelled():
        raise TrainingCancelled()
    progress("Fitting health regressor...", 0.0)

    model = CatBoostRegressor(
        iterations=iterations,
        depth=6,
        learning_rate=0.06,
        loss_function="RMSE",
        random_seed=0,
        verbose=False,
        allow_writing_files=False,
    )
    pool = Pool(np.asarray(values, dtype=np.float64),
                np.asarray(health, dtype=np.float64),
                feature_names=list(feature_names))
    model.fit(pool)

    predicted = np.clip(model.predict(np.asarray(values, dtype=np.float64)), 0.0, 100.0)
    residual = predicted - np.asarray(health, dtype=np.float64)
    raw = model.get_feature_importance(pool, type="PredictionValuesChange")
    importances = sorted(
        ({"feature": f, "importance": float(v)} for f, v in zip(feature_names, raw)),
        key=lambda d: -d["importance"],
    )
    return model, importances, {
        "rmse_health_points": float(np.sqrt(np.mean(residual**2))),
        "mae_health_points": float(np.mean(np.abs(residual))),
    }


# ------------------------------------------------------------------ GP forecaster


def make_gp_prototype(alpha: float = 1e-8):
    """A GP with fixed, sensible hyperparameters and ``optimizer=None``.

    Hyperparameters are fitted once per channel at attach time (see
    ``fit_gp_for_channel``); the prototype exists so the online engine can clone
    and condition cheaply on every tick without a hyperparameter search.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
        + ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )
    return GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, alpha=alpha)


def fit_gp_for_channel(
    series: np.ndarray,
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
    max_windows: int = 150,
    alpha: float = 1e-8,
):
    """Fit GP hyperparameters for one channel by marginal likelihood.

    Returns a conditioned-ready prototype with ``optimizer=None``.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from scipy.optimize import minimize

    series = np.asarray(series, dtype=np.float64).ravel()
    past_t = (np.arange(window) / window).reshape(-1, 1)

    starts = range(0, max(len(series) - window - horizon, 0) + 1)
    chosen = list(starts)[:: max(1, len(list(starts)) // max_windows or 1)][:max_windows]
    if not chosen:
        chosen = [0] if len(series) >= window else []
    if not chosen:
        return make_gp_prototype(alpha)

    template = make_gp_prototype(alpha).kernel
    conditioned = []
    for start in chosen:
        chunk = series[start : start + window]
        if len(chunk) < window:
            continue
        gp = GaussianProcessRegressor(kernel=template, optimizer=None,
                                      normalize_y=True, alpha=alpha)
        gp.fit(past_t, chunk)
        conditioned.append(gp)
    if not conditioned:
        return make_gp_prototype(alpha)

    def objective(theta):
        total_lml, total_grad = 0.0, np.zeros_like(theta)
        for gp in conditioned:
            lml, grad = gp.log_marginal_likelihood(theta, eval_gradient=True)
            total_lml += lml
            total_grad += grad
        n = len(conditioned)
        return -total_lml / n, -total_grad / n

    try:
        result = minimize(objective, template.theta, jac=True, method="L-BFGS-B",
                          bounds=template.bounds, options={"maxiter": 150})
        theta = result.x
    except Exception:  # noqa: BLE001 - a bad fit must not break training
        theta = template.theta

    prototype = make_gp_prototype(alpha)
    prototype.kernel = template.clone_with_theta(theta)
    prototype.fit(past_t, series[:window] if len(series) >= window else series)
    return prototype


def gp_forecast(prototype, buffer: np.ndarray, window: int, horizon: int):
    """Condition on a buffer and forecast ``horizon`` steps with a 95% band."""
    from sklearn.gaussian_process import GaussianProcessRegressor

    buffer = np.asarray(buffer, dtype=np.float64).ravel()
    if len(buffer) != window:
        raise ValueError(f"expected {window} samples, got {len(buffer)}")
    past_t = (np.arange(window) / window).reshape(-1, 1)
    future_t = (np.arange(window, window + horizon) / window).reshape(-1, 1)

    gp = GaussianProcessRegressor(
        kernel=prototype.kernel, optimizer=None,
        normalize_y=prototype.normalize_y, alpha=prototype.alpha,
    )
    gp.fit(past_t, buffer)
    mean, std = gp.predict(future_t, return_std=True)
    return mean, mean - CONFIDENCE_Z * std, mean + CONFIDENCE_Z * std


# ------------------------------------------------------------------ orchestration


def train_all(
    values: np.ndarray,
    names: list[str],
    units: Optional[np.ndarray] = None,
    cycles: Optional[np.ndarray] = None,
    rul: Optional[np.ndarray] = None,
    health_truth: Optional[np.ndarray] = None,
    nominal_mask: Optional[np.ndarray] = None,
    window: int = DEFAULT_WINDOW,
    horizon: int = DEFAULT_HORIZON,
    baseline_fraction: float = BASELINE_FRACTION,
    source: str = "dataset",
    progress: ProgressFn = _noop_progress,
    is_cancelled: CancelFn = _never_cancelled,
) -> TrainedModels:
    """Fit all four algorithms for an arbitrary N-channel table.

    Health target, in order of preference:

    ``health_truth`` -> supervised on a known 0-100 condition (the rocket
                        simulator knows its own nozzle erosion, so it can supply
                        this directly).
    ``rul``          -> supervised on run-to-failure RUL, the C-MAPSS case.
    neither          -> baseline-calibrated: an assumed-nominal window is taken
                        as the reference and health is deviation from it.

    ``nominal_mask`` overrides which rows the autoencoder treats as healthy.
    That matters for data with strong regimes rather than steady degradation --
    a rocket's first 30% is the boost phase, not a nominal baseline, so passing
    a mask of genuinely healthy runs avoids calibrating against the wrong thing.
    """
    started = time.perf_counter()
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("training needs a non-empty 2-D array")
    if units is None:
        units = np.zeros(len(values), dtype=np.int64)
    if cycles is None:
        cycles = np.arange(len(values), dtype=np.float64)

    progress("Fitting sensor schema...", 0.02)
    schema = SensorSchema.fit(values, names)
    scaled = schema.transform(schema.select(values, names))
    n_features = schema.n_features
    if n_features == 0:
        raise ValueError("no usable (non-constant) channels in this data")

    # -------------------------------------------------- 1. autoencoder
    if nominal_mask is not None:
        nominal = np.asarray(nominal_mask, dtype=bool)
    else:
        nominal = baseline_mask(units, cycles, baseline_fraction)
    if nominal.sum() < 8:
        nominal = np.ones(len(scaled), dtype=bool)
    progress(f"Training autoencoder on {int(nominal.sum())} nominal rows "
             f"({n_features} channels)...", 0.05)
    autoencoder, ae_metrics = train_autoencoder(
        scaled[nominal], n_features, progress=progress, is_cancelled=is_cancelled,
        progress_span=(0.05, 0.40),
    )

    # -------------------------------------------------- 2. health target
    scores = reconstruction_error(autoencoder, scaled)
    calibration = BaselineCalibration.fit(scores[nominal])
    if health_truth is not None:
        health = np.clip(np.asarray(health_truth, dtype=np.float64), 0.0, 100.0)
        stages = stage_from_health(health)
        labelled = True
    elif rul is not None:
        health = health_from_rul(rul)
        stages = stage_from_rul(np.asarray(rul, dtype=np.float64))
        labelled = True
    else:
        health = calibration.health(scores)
        stages = stage_from_health(health)
        labelled = False
    progress(f"Health target: {'true RUL' if labelled else 'baseline-calibrated'}", 0.45)

    # -------------------------------------------------- 3. health regressor
    health_model, importances, health_metrics = train_health_model(
        scaled, health, schema.names, progress=progress, is_cancelled=is_cancelled,
    )
    progress("Health regressor done.", 0.60)

    # -------------------------------------------------- 4. stage LSTM
    effective_window = int(max(2, min(window, max(2, len(scaled) // 4))))
    windows, _ = build_windows(scaled, units, effective_window)
    stage_model, stage_metrics = train_stage_model(
        windows, stages, n_features, progress=progress, is_cancelled=is_cancelled,
        progress_span=(0.60, 0.90),
    )

    # -------------------------------------------------- 5. GP on the lead channel
    progress("Fitting Gaussian Process hyperparameters...", 0.92)
    lead = int(np.argmax(np.std(scaled, axis=0)))
    gp_prototype = fit_gp_for_channel(
        scaled[:, lead].astype(np.float64), effective_window, horizon,
    )
    progress("Training complete.", 1.0)

    return TrainedModels(
        schema=schema,
        window=effective_window,
        horizon=horizon,
        autoencoder=autoencoder,
        calibration=calibration,
        health_model=health_model,
        stage_model=stage_model,
        gp_prototype=gp_prototype,
        labelled=labelled,
        feature_importances=importances,
        metrics={
            "autoencoder": ae_metrics,
            "health": health_metrics,
            "stage": stage_metrics,
            "baseline": calibration.to_dict(),
            "lead_channel": schema.names[lead],
            "n_rows": int(len(scaled)),
            "n_channels": n_features,
        },
        elapsed_s=time.perf_counter() - started,
        source=source,
    )
