"""Model 2/4 -- Gaussian Process Regression: near-future sensor forecast with bounds.

What it forecasts
-----------------
Sensor **s11** (HPC outlet static pressure), which has the strongest monotonic
degradation trend in FD001 (mean per-engine Spearman correlation with cycle
= +0.81, the highest of all 21 sensors). Given the last 30 cycles of s11, it
forecasts the next 15 cycles with a 95% credible interval.

Kernel
------
``C * DotProduct + C * RBF + WhiteKernel``

The linear (``DotProduct``) term is what makes the extrapolation useful: an
RBF-only GP reverts to the prior mean within a length-scale of the last
observation, which is useless 15 cycles out. The RBF term absorbs smooth local
curvature and the WhiteKernel absorbs sensor noise, which is what keeps the
interval honest.

How "training" works here, and why it is a two-stage fit
--------------------------------------------------------
A GP has no weights to learn -- it conditions on whatever data it is given. What
*is* learned offline is the kernel hyperparameter vector.

Stage 1 maximizes the mean **log marginal likelihood** across sampled training
windows (analytic gradient, L-BFGS-B). This is the textbook objective, and it is
used here only as an initializer, because on its own it gives a bad forecaster:
LML scores how well the kernel explains the 30 points *inside* the window, and
s11 has a low within-window signal-to-noise ratio, so LML drives the WhiteKernel
up to ~0.96 and the linear term to ~0. The GP then reverts to the window mean and
is beaten by naive linear extrapolation (measured: RMSE 0.088 vs 0.067, with
95% interval coverage of only 0.84).

Stage 2 therefore optimizes the objective we actually care about: the mean
**predictive negative log likelihood of the held-out future 15 cycles given the
past 30**, over every training window. This is leave-future-out, matches the
deployed task exactly, and calibrates the interval as a side effect (coverage is
part of the likelihood). Because the time grid is identical for every window,
only ``y`` differs, so one Cholesky factorization of the 30x30 Gram matrix serves
all windows at once -- the whole training set evaluates in ~6 ms, which is what
makes direct optimization of this objective practical. ``_BatchedGP`` implements
that and is asserted equal to scikit-learn's own ``predict`` to ~1e-14 before it
is trusted (see ``_assert_matches_sklearn``).

Inference contract
------------------
``gpr.pkl`` is a plain ``sklearn.gaussian_process.GaussianProcessRegressor`` with
the learned hyperparameters baked in and ``optimizer=None``. The app calls
``.fit()`` on the live 30-cycle window -- with ``optimizer=None`` that is pure
conditioning (one 30x30 Cholesky, sub-millisecond), *not* a hyperparameter
search -- then ``.predict(..., return_std=True)``. No custom class is unpickled,
so ``app/`` needs only scikit-learn, never an import from ``ml/``.

Reported metrics: horizon RMSE (normalized and raw sensor units) and empirical
coverage of the nominal 95% interval, against a persistence baseline and a
linear-extrapolation baseline for context.
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np
from scipy.linalg import cho_solve, cholesky
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel

from ml.artifacts import artifact_path, build_meta, report, save_meta
from ml.common import GPR_TARGET_SENSOR, WINDOW_LENGTH, rel, seed_everything
from ml.preprocess import Dataset, ensure_prepared

NAME = "gpr"

DEFAULTS = {
    "window": WINDOW_LENGTH,
    "horizon": 15,
    "lml_windows": 200,   # sampled windows for the stage-1 LML initializer
    "eval_stride": 10,    # stride between windows, for both fitting and evaluation
    "alpha": 1e-8,        # jitter; real observation noise is the WhiteKernel's job
    "maxiter": 300,
}

CONFIDENCE_Z = 1.959963984540054  # two-sided 95%


def make_kernel():
    """Linear trend + smooth local structure + observation noise."""
    return (
        ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
        + ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=0.5, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e1))
    )


def time_grid(window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Within-window time index, scaled by the window length to keep theta O(1)."""
    past = (np.arange(window, dtype=np.float64) / window).reshape(-1, 1)
    future = (np.arange(window, window + horizon, dtype=np.float64) / window).reshape(-1, 1)
    return past, future


def series_by_unit(ds: Dataset, split: str, sensor: str) -> dict[int, np.ndarray]:
    """Per-engine trajectory of one scaled sensor, ordered by cycle."""
    tab = ds.tabular(split)
    col = ds.feature_index(sensor)
    units, cycles, values = tab["unit"], tab["cycle"], tab["X"][:, col]
    out: dict[int, np.ndarray] = {}
    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        out[int(unit)] = values[idx[np.argsort(cycles[idx])]].astype(np.float64)
    return out


def extract_windows(
    series: dict[int, np.ndarray], window: int, horizon: int, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Stack every (past, future) pair of the right shape across engines.

    Returns ``(Y_past, Y_future)`` with shapes ``(n, window)`` and ``(n, horizon)``.
    """
    past_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    for values in series.values():
        last_start = len(values) - window - horizon
        for start in range(0, max(last_start + 1, 0), stride):
            past_rows.append(values[start : start + window])
            future_rows.append(values[start + window : start + window + horizon])
    if not past_rows:
        return np.empty((0, window)), np.empty((0, horizon))
    return np.stack(past_rows), np.stack(future_rows)


# ------------------------------------------------------------------ batched GP


class _BatchedGP:
    """Condition one GP on many windows at once.

    Every window shares the same input locations, so the Gram matrix -- and hence
    its Cholesky factor and the predictive variance -- is computed once and reused
    for all windows; only the right-hand side changes. Replicates scikit-learn's
    ``normalize_y=True`` behaviour (centre and scale each target to unit variance)
    so the numbers are interchangeable with the serialized artifact's.
    """

    def __init__(self, kernel, past_t: np.ndarray, future_t: np.ndarray, alpha: float):
        gram = kernel(past_t).copy()
        gram[np.diag_indices_from(gram)] += alpha
        self._chol = cholesky(gram, lower=True)
        self._cross = kernel(future_t, past_t)
        diag = kernel.diag(future_t)
        solved = cho_solve((self._chol, True), self._cross.T)
        variance = diag - np.einsum("ij,ji->i", self._cross, solved)
        self._std_unit = np.sqrt(np.maximum(variance, 1e-12))  # (horizon,)

    def predict(self, y_past: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(n, window)`` targets -> per-window ``(n, horizon)`` mean and std."""
        centre = y_past.mean(axis=1, keepdims=True)
        scale = y_past.std(axis=1, keepdims=True)
        scale = np.where(scale < 1e-12, 1.0, scale)
        weights = cho_solve((self._chol, True), ((y_past - centre) / scale).T)
        mean = (self._cross @ weights).T * scale + centre
        std = self._std_unit[None, :] * scale
        return mean, std


def _predictive_nll(mean: np.ndarray, std: np.ndarray, truth: np.ndarray) -> np.ndarray:
    std = np.maximum(std, 1e-9)
    return 0.5 * np.log(2 * np.pi * std**2) + (truth - mean) ** 2 / (2 * std**2)


def _assert_matches_sklearn(
    kernel, past_t: np.ndarray, future_t: np.ndarray, alpha: float, y_past: np.ndarray
) -> float:
    """Guard the fast path: batched math must equal sklearn's own predict."""
    batched = _BatchedGP(kernel, past_t, future_t, alpha)
    mean, std = batched.predict(y_past)
    worst = 0.0
    for row in range(min(len(y_past), 5)):
        gp = GaussianProcessRegressor(
            kernel=kernel, optimizer=None, normalize_y=True, alpha=alpha
        )
        gp.fit(past_t, y_past[row])
        ref_mean, ref_std = gp.predict(future_t, return_std=True)
        worst = max(worst, float(np.abs(ref_mean - mean[row]).max()))
        worst = max(worst, float(np.abs(ref_std - std[row]).max()))
    if worst > 1e-8:
        raise AssertionError(f"batched GP disagrees with sklearn by {worst:.2e}")
    return worst


# ------------------------------------------------------------------ fitting


def fit_lml(y_past: np.ndarray, past_t: np.ndarray, alpha: float) -> np.ndarray:
    """Stage 1: maximize the mean log marginal likelihood. Used as an initializer."""
    kernel = make_kernel()
    conditioned = []
    for values in y_past:
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=None, normalize_y=True, alpha=alpha)
        gp.fit(past_t, values)
        conditioned.append(gp)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        total_lml = 0.0
        total_grad = np.zeros_like(theta)
        for gp in conditioned:
            lml, grad = gp.log_marginal_likelihood(theta, eval_gradient=True)
            total_lml += lml
            total_grad += grad
        n = len(conditioned)
        return -total_lml / n, -total_grad / n

    start = -objective(kernel.theta)[0]
    result = minimize(
        objective, kernel.theta, jac=True, method="L-BFGS-B",
        bounds=kernel.bounds, options={"maxiter": 400},
    )
    print(
        f"  stage 1 (marginal likelihood, {len(conditioned)} windows): "
        f"mean LML {start:.3f} -> {-result.fun:.3f} ({result.nit} iters)"
    )
    return result.x


def fit_predictive(
    theta_init: np.ndarray,
    y_past: np.ndarray,
    y_future: np.ndarray,
    past_t: np.ndarray,
    future_t: np.ndarray,
    alpha: float,
    maxiter: int,
) -> np.ndarray:
    """Stage 2: minimize mean predictive NLL of the future window (leave-future-out).

    This is the objective that matches deployment, and it calibrates the
    predictive interval as a by-product.
    """
    template = make_kernel()

    def objective(theta: np.ndarray) -> float:
        try:
            batched = _BatchedGP(template.clone_with_theta(theta), past_t, future_t, alpha)
        except np.linalg.LinAlgError:
            return 1e6  # non-PSD corner of the hyperparameter space
        mean, std = batched.predict(y_past)
        return float(_predictive_nll(mean, std, y_future).mean())

    start = objective(theta_init)
    result = minimize(
        objective, theta_init, method="L-BFGS-B",
        bounds=template.bounds, options={"maxiter": maxiter, "eps": 1e-5},
    )
    # Keep whichever point is actually better -- L-BFGS-B on a finite-difference
    # gradient can return a worse iterate than it started from.
    theta = result.x if result.fun <= start else np.asarray(theta_init)
    print(
        f"  stage 2 (predictive NLL, {len(y_past)} windows): "
        f"{start:.4f} -> {min(result.fun, start):.4f} ({result.nit} iters)"
    )
    return theta


# ------------------------------------------------------------------ evaluation


def evaluate_forecast(
    kernel,
    y_past: np.ndarray,
    y_future: np.ndarray,
    past_t: np.ndarray,
    future_t: np.ndarray,
    alpha: float,
    raw_span: float,
) -> dict:
    """Horizon RMSE + empirical 95% coverage, with two baselines for context."""
    if len(y_past) == 0:
        return {"n_windows": 0}

    batched = _BatchedGP(kernel, past_t, future_t, alpha)
    mean, std = batched.predict(y_past)
    lower, upper = mean - CONFIDENCE_Z * std, mean + CONFIDENCE_Z * std
    error = mean - y_future

    # Baselines: hold the last value, and least-squares extrapolate the window.
    persistence = y_past[:, -1:] - y_future
    flat_t = past_t.ravel()
    design = np.vstack([flat_t, np.ones_like(flat_t)]).T
    coeffs, *_ = np.linalg.lstsq(design, y_past.T, rcond=None)   # (2, n)
    linear = (future_t.ravel()[:, None] * coeffs[0] + coeffs[1]).T - y_future

    def rmse(values: np.ndarray) -> float:
        return float(np.sqrt(np.mean(values**2)))

    inside = (y_future >= lower) & (y_future <= upper)
    return {
        "n_windows": int(len(y_past)),
        "n_points": int(error.size),
        "rmse_normalized": rmse(error),
        "rmse_raw_sensor_units": rmse(error) * raw_span,
        "mae_normalized": float(np.mean(np.abs(error))),
        "coverage_95": float(inside.mean()),
        "nominal_coverage": 0.95,
        "coverage_error": float(inside.mean() - 0.95),
        "mean_interval_width_normalized": float(np.mean(upper - lower)),
        "mean_negative_log_likelihood": float(_predictive_nll(mean, std, y_future).mean()),
        "rmse_by_horizon_step": [rmse(error[:, h]) for h in range(error.shape[1])],
        "coverage_by_horizon_step": [float(inside[:, h].mean()) for h in range(inside.shape[1])],
        "baseline_persistence_rmse": rmse(persistence),
        "baseline_linear_extrapolation_rmse": rmse(linear),
    }


# ------------------------------------------------------------------ training entry


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    ds: Dataset = ensure_prepared()
    sensor = args.sensor
    col = ds.feature_index(sensor)
    raw_span = float(ds.scaler_max[col] - ds.scaler_min[col])
    past_t, future_t = time_grid(args.window, args.horizon)

    print(
        f"forecasting {sensor} (feature index {col}) "
        f"{args.horizon} cycles ahead from a {args.window}-cycle window"
    )
    print(
        f"  {sensor} raw range on train: "
        f"[{ds.scaler_min[col]:.4f}, {ds.scaler_max[col]:.4f}] (span {raw_span:.4f})"
    )

    train_past, train_future = extract_windows(
        series_by_unit(ds, "train", sensor), args.window, args.horizon, args.eval_stride
    )
    print(f"  {len(train_past)} training windows")

    guard = _assert_matches_sklearn(make_kernel(), past_t, future_t, args.alpha, train_past)
    print(f"  batched-GP guard vs sklearn.predict: max abs diff {guard:.2e}")

    rng = np.random.default_rng(args.seed)
    sample = rng.choice(
        len(train_past), size=min(args.lml_windows, len(train_past)), replace=False
    )
    theta = fit_lml(train_past[sample], past_t, args.alpha)
    theta = fit_predictive(
        theta, train_past, train_future, past_t, future_t, args.alpha, args.maxiter
    )

    kernel = make_kernel().clone_with_theta(theta)
    print(f"  learned kernel: {kernel}")

    prototype = GaussianProcessRegressor(
        kernel=kernel, optimizer=None, normalize_y=True, alpha=args.alpha
    )
    # Pre-fit on one representative window so the artifact is usable as loaded,
    # before the app supplies live data.
    prototype.fit(past_t, train_past[0])

    metrics: dict = {
        "target_sensor": sensor,
        "window": args.window,
        "horizon": args.horizon,
        "kernel": str(kernel),
    }
    for split in ("train", "val", "test", "official"):
        past, future = extract_windows(
            series_by_unit(ds, split, sensor), args.window, args.horizon, args.eval_stride
        )
        metrics[split] = evaluate_forecast(
            kernel, past, future, past_t, future_t, args.alpha, raw_span
        )

    # ------------------------------------------------------------ serialize
    pkl_path = artifact_path(f"{NAME}.pkl")
    joblib.dump(prototype, pkl_path)

    # Round-trip check through the exact call sequence the app will use.
    reloaded = joblib.load(pkl_path)
    reloaded.fit(past_t, train_past[0])
    live_mean, live_std = reloaded.predict(future_t, return_std=True)
    batched_mean, batched_std = _BatchedGP(kernel, past_t, future_t, args.alpha).predict(
        train_past[:1]
    )
    diff = max(
        float(np.abs(live_mean - batched_mean[0]).max()),
        float(np.abs(live_std - batched_std[0]).max()),
    )
    print(
        f"  pickle round-trip via app call sequence: max abs diff {diff:.2e}  "
        f"{'ok' if diff < 1e-8 else 'MISMATCH'}  ({rel(pkl_path)})"
    )

    meta = build_meta(
        name=NAME,
        task="sensor_trajectory_forecast_with_uncertainty",
        dataset=ds,
        input_shape=[args.window, 1],
        input_feature_names=[sensor],
        window_length=args.window,
        output={
            "shape": [args.horizon],
            "dtype": "float64",
            "meaning": f"posterior mean of normalized {sensor} for the next {args.horizon} cycles",
            "returns": {
                "mean": f"predict(future_t) -> ({args.horizon},)",
                "std": f"predict(future_t, return_std=True) -> (mean, std), each ({args.horizon},)",
            },
            "confidence_interval": {
                "level": 0.95,
                "z": CONFIDENCE_Z,
                "lower": "mean - 1.959963984540054 * std",
                "upper": "mean + 1.959963984540054 * std",
                "includes_observation_noise": True,
            },
            "units": (
                f"normalized; to recover raw {sensor} units multiply by "
                "(normalization.max - normalization.min) at this sensor's index "
                "and add normalization.min"
            ),
        },
        artifacts={"sklearn_pickle": pkl_path.name},
        thresholds={},
        inference={
            "protocol": [
                f"1. Keep a rolling buffer of the last {args.window} cycles of {sensor}.",
                "2. Normalize each value using normalization.min/max at this sensor's index.",
                "3. gp = joblib.load('gpr.pkl')",
                f"4. past_t = (numpy.arange({args.window}) / {args.window}).reshape(-1, 1)",
                "5. gp.fit(past_t, buffer)  -- optimizer is None, so this only conditions "
                "(one 30x30 Cholesky, sub-millisecond); it does NOT refit hyperparameters",
                f"6. future_t = (numpy.arange({args.window}, {args.window + args.horizon}) "
                f"/ {args.window}).reshape(-1, 1)",
                "7. mean, std = gp.predict(future_t, return_std=True)",
                "8. band = mean +/- 1.959963984540054 * std",
            ],
            "time_grid": {
                "past_t": f"arange({args.window}) / {args.window}",
                "future_t": f"arange({args.window}, {args.window + args.horizon}) / {args.window}",
                "note": "time is divided by the window length; a different grid invalidates "
                        "the learned length scale",
            },
            "loader": "joblib.load -- a plain sklearn GaussianProcessRegressor; "
                      "no import from ml/ is required",
            "requires": ["scikit-learn", "joblib", "numpy", "scipy"],
            "refit_per_call": True,
            "stateless": False,
            "normalize_y": True,
            "cost": "one Cholesky of the 30x30 window Gram matrix per call",
        },
        metrics=metrics,
        extra={
            "kernel_detail": {
                "expression": "C * DotProduct + C * RBF + WhiteKernel",
                "fitted": str(kernel),
                "theta_log": [float(v) for v in theta],
                "hyperparameters_fit_by": (
                    "two stages: (1) mean log marginal likelihood as initializer, "
                    "(2) mean predictive NLL of the held-out future window, which is the "
                    "deployed task and also calibrates the interval"
                ),
                "normalize_y": True,
                "alpha": args.alpha,
            },
            "target_sensor": {
                "name": sensor,
                "feature_index_in_shared_contract": col,
                "chosen_because": "strongest monotonic degradation trend in FD001 "
                                  "(mean per-engine Spearman vs cycle = +0.81)",
                "raw_min": float(ds.scaler_min[col]),
                "raw_max": float(ds.scaler_max[col]),
            },
        },
    )
    written = [pkl_path, save_meta(NAME, meta)]
    report(NAME, metrics, written)
    return metrics


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--sensor", default=GPR_TARGET_SENSOR, help="sensor column to forecast")
    ap.add_argument("--window", type=int, default=DEFAULTS["window"])
    ap.add_argument("--horizon", type=int, default=DEFAULTS["horizon"])
    ap.add_argument("--lml-windows", type=int, default=DEFAULTS["lml_windows"])
    ap.add_argument("--eval-stride", type=int, default=DEFAULTS["eval_stride"])
    ap.add_argument("--alpha", type=float, default=DEFAULTS["alpha"])
    ap.add_argument("--maxiter", type=int, default=DEFAULTS["maxiter"])
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = add_arguments(argparse.ArgumentParser(description=__doc__))
    train(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
