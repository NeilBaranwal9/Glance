"""Single entry point for training the four FD001 models.

    python ml/train_models.py --model autoencoder
    python ml/train_models.py --model gpr
    python ml/train_models.py --model lstm
    python ml/train_models.py --model catboost_health
    python ml/train_models.py --model all

Any flag this script does not recognise is forwarded to the selected model's own
parser, so per-model options work unchanged:

    python ml/train_models.py --model lstm --hidden 128 --epochs 200
    python ml/train_models.py --model autoencoder --bottleneck 3
    python ml/train_models.py --model help --list      # show each model's flags

The dataset is downloaded and preprocessed on demand, so a clean checkout only
needs this one command.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # allow `python ml/train_models.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.artifacts import sanity_check_stdout  # noqa: E402
from ml.common import ARTIFACT_DIR, PROCESSED_NPZ, rel  # noqa: E402

MODELS = ("autoencoder", "gpr", "lstm", "catboost_health")


def _load(name: str):
    """Import a model module lazily, so training one model needs only its deps."""
    import importlib

    return importlib.import_module(f"ml.models.{name}")


def _ensure_data(force_preprocess: bool) -> None:
    """Download the dataset and build the processed cache if either is missing."""
    from ml.download_data import REQUIRED_FILES, CMAPSS_DIR, ensure_dataset

    if any(not (CMAPSS_DIR / n).exists() for n in REQUIRED_FILES):
        print("=== dataset ===")
        if not ensure_dataset():
            raise SystemExit("dataset unavailable; see the instructions above")
        print()

    from ml.preprocess import ensure_prepared

    if force_preprocess or not PROCESSED_NPZ.exists():
        print("=== preprocessing ===")
        ensure_prepared(force=True)
        print(f"wrote {rel(PROCESSED_NPZ)}\n")


def _summarize(name: str, metrics: dict) -> str:
    """One-line headline metric per model, for the closing table."""
    test = metrics.get("test", {})
    if name == "autoencoder":
        return f"AUROC {test.get('auroc', float('nan')):.4f} (fault = RUL < 30)"
    if name == "gpr":
        return (
            f"RMSE {test.get('rmse_normalized', float('nan')):.4f} normalized, "
            f"95% coverage {test.get('coverage_95', float('nan')):.4f}"
        )
    if name == "lstm":
        return (
            f"accuracy {test.get('accuracy', float('nan')):.4f}, "
            f"macro-F1 {test.get('macro_f1', float('nan')):.4f}"
        )
    if name == "catboost_health":
        return (
            f"RMSE {test.get('rmse_health_points', float('nan')):.4f} health points "
            f"({test.get('rmse_rul_cycles', float('nan')):.2f} RUL cycles)"
        )
    return ""


def _print_flags() -> None:
    print(__doc__)
    for name in MODELS:
        module = _load(name)
        parser = module.add_arguments(argparse.ArgumentParser(prog=name, add_help=False))
        flags = [
            "/".join(action.option_strings)
            for action in parser._actions
            if action.option_strings
        ]
        print(f"{name}:\n  {'  '.join(flags)}\n")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    top = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    top.add_argument(
        "--model",
        required=True,
        choices=(*MODELS, "all", "help"),
        help="which model to train ('all' trains every model in order)",
    )
    top.add_argument(
        "--force-preprocess", action="store_true", help="rebuild the processed cache first"
    )
    top.add_argument("--list", action="store_true", help="list each model's own flags and exit")
    known, rest = top.parse_known_args(argv)

    if known.model == "help" or known.list:
        _print_flags()
        return 0

    sanity_check_stdout()  # fail loudly if Qt leaked into the training process
    _ensure_data(known.force_preprocess)

    selected = MODELS if known.model == "all" else (known.model,)
    if known.model == "all" and rest:
        top.error(
            f"per-model flags cannot be used with --model all: {' '.join(rest)}\n"
            "train that model on its own to pass its flags"
        )

    results: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for name in selected:
        module = _load(name)
        parser = module.add_arguments(
            argparse.ArgumentParser(prog=f"train_models.py --model {name}")
        )
        args = parser.parse_args(rest)
        print(f"\n{'=' * 72}\n=== {name}\n{'=' * 72}")
        started = time.perf_counter()
        results[name] = module.train(args)
        timings[name] = time.perf_counter() - started
        sanity_check_stdout()

    print(f"\n{'=' * 72}\n=== summary (held-out test split of {rel(ARTIFACT_DIR)})\n{'=' * 72}")
    for name in selected:
        print(f"{name:18s} {timings[name]:7.1f}s  {_summarize(name, results[name])}")

    written = sorted(ARTIFACT_DIR.glob("*"))
    print(f"\nartifacts in {rel(ARTIFACT_DIR)}:")
    for path in written:
        print(f"  {path.name:34s} {path.stat().st_size / 1024:9.1f} KiB")
    if known.model == "all":
        print("\nnext: python ml/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
