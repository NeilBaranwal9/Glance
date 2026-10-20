"""Model 4/4 -- CatBoost regressor predicting a 0-100 health score.

Target
------
``health = 100 * clip(RUL / 125, 0, 1)``

100 means as-new, 0 means at failure. Because the target is an affine function of
the capped RUL, an RMSE of *h* health points is exactly *h * 1.25* cycles of RUL
error, and both are reported so the number can be read either way.

Why gradient boosting here
--------------------------
This is the per-cycle view: no window, no sequence, one row in and one score out.
Boosted trees are the right tool -- they handle the raw scale and mild
non-monotonicity of the sensor/health relationship without feature engineering,
train in seconds, and give usable feature importances, which is what makes the
score explainable in a UI ("health is down because s11 and s4 moved").

A LightGBM model is trained alongside on the identical split purely as a
reference point, so the CatBoost number can be judged as well-tuned or not.
Only CatBoost is serialized.

Output contract
---------------
``(N, 17) -> (N,)`` float, a health score that the app should clip to [0, 100]
before display -- a regressor can overshoot slightly past either end.
"""
from __future__ import annotations

import argparse

import numpy as np
from catboost import CatBoostRegressor, Pool

from ml.artifacts import artifact_path, build_meta, report, save_meta
from ml.common import RUL_CEILING, regression_metrics, seed_everything
from ml.preprocess import Dataset, ensure_prepared

NAME = "catboost_health"

DEFAULTS = {
    "iterations": 2000,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "early_stopping_rounds": 100,
}


def _health_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression metrics in health points, plus the RUL-cycle equivalent."""
    clipped = np.clip(y_pred, 0.0, 100.0)
    stats = regression_metrics(y_true, clipped)
    scale = RUL_CEILING / 100.0
    return {
        "rmse_health_points": stats["rmse"],
        "mae_health_points": stats["mae"],
        "r2": stats["r2"],
        "rmse_rul_cycles": stats["rmse"] * scale,
        "mae_rul_cycles": stats["mae"] * scale,
        "rmse_before_clipping": regression_metrics(y_true, y_pred)["rmse"],
        "n_rows": int(len(y_true)),
    }


def _lightgbm_reference(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
    seed: int,
) -> dict | None:
    """Train a LightGBM model on the same split, for context only."""
    try:
        import lightgbm as lgb
    except ImportError:
        return None
    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        feature_name=features,
    )
    return {
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "val": _health_metrics(y_val, model.predict(x_val)),
        "test": _health_metrics(y_test, model.predict(x_test)),
    }


def train(args: argparse.Namespace) -> dict:
    seed_everything(args.seed)
    ds: Dataset = ensure_prepared()
    features = ds.features

    views = {split: ds.tabular(split) for split in ("train", "val", "test", "official")}
    x_train = views["train"]["X"].astype(np.float64)
    y_train = views["train"]["health"].astype(np.float64)
    x_val = views["val"]["X"].astype(np.float64)
    y_val = views["val"]["health"].astype(np.float64)

    print(
        f"training on {len(x_train)} rows x {len(features)} features "
        f"({len(ds.units('train'))} engines), validating on {len(x_val)} rows"
    )
    print(f"target: health = 100 * clip(RUL / {RUL_CEILING}, 0, 1)   "
          f"range [{y_train.min():.1f}, {y_train.max():.1f}]")

    model = CatBoostRegressor(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=args.seed,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose=200,
        allow_writing_files=False,
    )
    train_pool = Pool(x_train, y_train, feature_names=list(features))
    val_pool = Pool(x_val, y_val, feature_names=list(features))
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    print(f"  best iteration: {model.get_best_iteration()} of {args.iterations}")

    metrics: dict = {
        "best_iteration": int(model.get_best_iteration()),
        "n_trees": int(model.tree_count_),
    }
    for split, view in views.items():
        metrics[split] = _health_metrics(
            view["health"].astype(np.float64), model.predict(view["X"].astype(np.float64))
        )

    # ------------------------------------------------------------ importances
    raw = model.get_feature_importance(train_pool, type="PredictionValuesChange")
    importances = sorted(
        ({"feature": f, "importance": float(v)} for f, v in zip(features, raw)),
        key=lambda d: -d["importance"],
    )
    print("\nfeature importances (PredictionValuesChange, % of total):")
    widest = max(len(d["feature"]) for d in importances)
    for rank, item in enumerate(importances, 1):
        bar = "#" * int(round(item["importance"] / 2))
        print(f"  {rank:2d}. {item['feature']:<{widest}}  {item['importance']:6.2f}  {bar}")
    metrics["feature_importances"] = importances

    if args.compare_lightgbm:
        reference = _lightgbm_reference(
            x_train, y_train, x_val, y_val,
            views["test"]["X"].astype(np.float64),
            views["test"]["health"].astype(np.float64),
            list(features), args.seed,
        )
        if reference is not None:
            metrics["lightgbm_reference"] = reference
            print(
                f"\nLightGBM reference on the same split: "
                f"test RMSE {reference['test']['rmse_health_points']:.3f} health points "
                f"vs CatBoost {metrics['test']['rmse_health_points']:.3f}"
            )

    # ------------------------------------------------------------ serialize
    cbm_path = artifact_path(f"{NAME}.cbm")
    model.save_model(str(cbm_path), format="cbm")

    reloaded = CatBoostRegressor()
    reloaded.load_model(str(cbm_path))
    probe = views["test"]["X"][:256].astype(np.float64)
    diff = float(np.abs(reloaded.predict(probe) - model.predict(probe)).max())
    print(f"\n  cbm round-trip: max|before - after| = {diff:.2e}  "
          f"{'ok' if diff < 1e-9 else 'MISMATCH'}")

    meta = build_meta(
        name=NAME,
        task="health_score_regression",
        dataset=ds,
        input_shape=[-1, len(features)],
        output={
            "shape": [-1],
            "dtype": "float64",
            "meaning": "health score, 100 = as-new, 0 = at failure",
            "range": [0, 100],
            "postprocess": "clip to [0, 100] -- the regressor can overshoot slightly",
            "target_definition": f"100 * clip(RUL / {RUL_CEILING}, 0, 1)",
            "rul_equivalent": (
                f"rul_cycles = health / 100 * {RUL_CEILING}; an RMSE of h health points "
                f"equals h * {RUL_CEILING / 100:.2f} cycles of RUL error"
            ),
        },
        artifacts={"catboost": cbm_path.name},
        thresholds={
            "suggested_display_bands": {
                "healthy": "health > 60",
                "degrading": "24 <= health <= 60",
                "critical": "health < 24",
            },
            "note": (
                "these bands are the LSTM's RUL stage thresholds (75 and 30 cycles) "
                "expressed as health points, so the two models agree by construction"
            ),
        },
        inference={
            "protocol": [
                "1. Collect one telemetry cycle as the 17 raw features, in "
                "input.feature_names order.",
                "2. Normalize with normalization.min/max per the stated formula.",
                "3. model = catboost.CatBoostRegressor(); model.load_model('catboost_health.cbm')",
                "4. health = float(model.predict(x.reshape(1, -1))[0])",
                "5. health = min(100.0, max(0.0, health))",
            ],
            "loader": "catboost.CatBoostRegressor().load_model",
            "requires": ["catboost", "numpy"],
            "stateless": True,
            "note": "expects the SAME normalized features as the autoencoder, so one "
                    "normalized row can be fed to both",
        },
        metrics=metrics,
        extra={
            "hyperparameters": {
                "iterations": args.iterations,
                "depth": args.depth,
                "learning_rate": args.learning_rate,
                "l2_leaf_reg": args.l2_leaf_reg,
                "loss_function": "RMSE",
                "early_stopping_rounds": args.early_stopping_rounds,
                "best_iteration": int(model.get_best_iteration()),
            },
            "feature_importances": importances,
            "feature_importance_type": "PredictionValuesChange (CatBoost default, "
                                       "normalized to sum to 100)",
        },
    )
    written = [cbm_path, save_meta(NAME, meta)]
    report(NAME, {k: v for k, v in metrics.items() if k != "feature_importances"}, written)
    return metrics


def add_arguments(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--iterations", type=int, default=DEFAULTS["iterations"])
    ap.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    ap.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    ap.add_argument("--l2-leaf-reg", type=float, default=DEFAULTS["l2_leaf_reg"])
    ap.add_argument(
        "--early-stopping-rounds", type=int, default=DEFAULTS["early_stopping_rounds"]
    )
    ap.add_argument(
        "--compare-lightgbm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also train LightGBM on the same split as a reference (not serialized)",
    )
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = add_arguments(argparse.ArgumentParser(description=__doc__))
    train(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
