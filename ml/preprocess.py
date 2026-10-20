"""C-MAPSS FD001 preprocessing: load, label, split, scale, and cache three views.

Run once; every training script then reads the cached arrays instead of
re-deriving them:

    python ml/preprocess.py            # writes data/processed/fd001.npz
    python ml/preprocess.py --force    # rebuild from scratch

Design decisions and their justification
----------------------------------------
**Dropped columns.** ``op3, s1, s5, s10, s16, s18, s19`` are exactly constant
across FD001 (std < 1e-8 on the training split). They carry zero signal and
would make min-max scaling divide by zero, so they are dropped, leaving 17
features in a fixed canonical order (see ``common.kept_features``).

**RUL labelling.** Standard C-MAPSS convention for run-to-failure data:
``RUL = max_cycle_for_engine - current_cycle``, then capped at
``RUL_CEILING = 125``. The cap matters because sensor readings are flat and
indistinguishable early in life, so asking a model to tell cycle 5 from cycle
100 apart is asking it to fit noise. The official ``test_FD001.txt`` engines are
truncated before failure, so their per-row RUL is reconstructed from
``RUL_FD001.txt`` as ``RUL(cycle) = final_rul + (max_cycle - cycle)``.

**Splitting by engine, not by row.** Consecutive cycles of one engine are almost
identical, so a row-wise split would put near-duplicates on both sides and
inflate every metric. The 100 run-to-failure engines in ``train_FD001.txt`` are
partitioned 70/15/15 by *unit id* under a fixed seed. The official
``test_FD001.txt`` is kept as a fourth, wholly untouched view.

**Scaling.** Min-max to [0, 1], fit on the training-split rows **only**, then
applied unchanged to val/test/official. Two modes:

  - ``global`` (default): one min/max vector for the whole fleet. This is the
    only mode whose statistics can be serialized and replayed at inference time,
    which is what the UI in Phase B needs -- a live engine has no known future
    min/max. The saved stats go into every ``*_meta.json``.
  - ``per_engine``: each engine scaled by its own observed min/max. Leakage-free
    with respect to the train/test boundary, but it needs an engine's whole run
    up front, so it cannot be reproduced on a streaming feed. Provided for
    offline comparison via ``--scaler per_engine``; not the default.

**Window length = 30 cycles** for the LSTM view. The binding constraint is the
official test set: its shortest engine is 31 cycles long, so 30 is the largest
window that still produces at least one sequence for all 100 test engines.
It is also ~15% of median engine life (199 cycles) -- long enough to carry a
visible degradation trend, short enough to keep ~14.6k training sequences.

**Healthy baseline** for the autoencoder: the first 30% of each engine's cycles.
The shortest engine in FD001 runs 128 cycles, so this window always sits far
from failure; the realised minimum RUL of the subset is reported in the manifest
so the "as-new" assumption stays auditable.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python ml/preprocess.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.common import (  # noqa: E402
    CMAPSS_DIR,
    CONSTANT_FEATURES,
    FAULT_RUL_THRESHOLD,
    HEALTHY_LIFE_FRACTION,
    PROCESSED_MANIFEST,
    PROCESSED_NPZ,
    RAW_COLUMNS,
    RUL_CEILING,
    SPLIT_FRACTIONS,
    SPLIT_SEED,
    STAGE_CRITICAL_MAX_RUL,
    STAGE_DEGRADING_MAX_RUL,
    STAGE_LABELS,
    WINDOW_LENGTH,
    health_from_rul,
    kept_features,
    rel,
    stage_from_rul,
    write_json,
)

SPLITS = ("train", "val", "test")
ALL_VIEWS = SPLITS + ("official",)


# ------------------------------------------------------------------ loading


def load_raw(data_dir: Path = CMAPSS_DIR) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Read the three FD001 text files. Whitespace-delimited, no header."""
    names = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
    paths = {n: data_dir / n for n in names}
    missing = [rel(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"C-MAPSS FD001 files not found: {', '.join(missing)}\n"
            "Run: python ml/download_data.py"
        )
    train = pd.read_csv(paths["train_FD001.txt"], sep=r"\s+", header=None, names=RAW_COLUMNS)
    test = pd.read_csv(paths["test_FD001.txt"], sep=r"\s+", header=None, names=RAW_COLUMNS)
    final_rul = pd.read_csv(paths["RUL_FD001.txt"], sep=r"\s+", header=None).iloc[:, 0].to_numpy()
    return train, test, final_rul


def add_run_to_failure_rul(df: pd.DataFrame, ceiling: int = RUL_CEILING) -> pd.DataFrame:
    """RUL = max_cycle_for_engine - current_cycle, capped. For run-to-failure data."""
    out = df.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul_raw"] = (last - out["cycle"]).astype(np.int64)
    out["rul"] = out["rul_raw"].clip(upper=ceiling).astype(np.float32)
    return out


def add_truncated_rul(
    df: pd.DataFrame, final_rul: np.ndarray, ceiling: int = RUL_CEILING
) -> pd.DataFrame:
    """RUL for the official test set, whose engines stop *before* failure."""
    out = df.copy()
    units_sorted = sorted(out["unit"].unique())
    if len(units_sorted) != len(final_rul):
        raise ValueError(
            f"RUL_FD001.txt has {len(final_rul)} rows but the test set has "
            f"{len(units_sorted)} engines"
        )
    final_by_unit = dict(zip(units_sorted, final_rul))
    last = out.groupby("unit")["cycle"].transform("max")
    tail = out["unit"].map(final_by_unit).astype(np.int64)
    out["rul_raw"] = (tail + (last - out["cycle"])).astype(np.int64)
    out["rul"] = out["rul_raw"].clip(upper=ceiling).astype(np.float32)
    return out


def split_units(
    units: np.ndarray,
    fractions: tuple[float, float, float] = SPLIT_FRACTIONS,
    seed: int = SPLIT_SEED,
) -> dict[str, np.ndarray]:
    """Partition engine ids (not rows) into train/val/test under a fixed seed."""
    shuffled = np.sort(np.asarray(units)).copy()
    np.random.default_rng(seed).shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return {
        "train": np.sort(shuffled[:n_train]),
        "val": np.sort(shuffled[n_train : n_train + n_val]),
        "test": np.sort(shuffled[n_train + n_val :]),
    }


# ------------------------------------------------------------------ scaling


@dataclass
class MinMaxStats:
    """Min-max scaling statistics, fit on the training split only."""

    features: list[str]
    minimum: np.ndarray
    maximum: np.ndarray
    mode: str = "global"

    @property
    def span(self) -> np.ndarray:
        span = self.maximum - self.minimum
        return np.where(span < 1e-12, 1.0, span)  # constant column -> maps to 0.0

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = frame[self.features].to_numpy(dtype=np.float64)
        # Unseen engines can sit slightly outside the train range; clamp loosely
        # rather than exactly to [0, 1] so genuine out-of-range drift stays visible.
        return np.clip((raw - self.minimum) / self.span, -0.5, 1.5).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "feature_names": list(self.features),
            "min": [float(v) for v in self.minimum],
            "max": [float(v) for v in self.maximum],
            "clip_range": [-0.5, 1.5],
        }


def fit_global_minmax(train_rows: pd.DataFrame, features: list[str]) -> MinMaxStats:
    raw = train_rows[features].to_numpy(dtype=np.float64)
    return MinMaxStats(features, raw.min(axis=0), raw.max(axis=0), mode="global")


def scale_per_engine(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Each engine scaled by its own min/max. Offline comparison only."""
    out = np.empty((len(frame), len(features)), dtype=np.float32)
    unit_col = frame["unit"].to_numpy()
    block_all = frame[features].to_numpy(dtype=np.float64)
    for unit in np.unique(unit_col):
        idx = np.flatnonzero(unit_col == unit)
        block = block_all[idx]
        lo, hi = block.min(axis=0), block.max(axis=0)
        span = np.where((hi - lo) < 1e-12, 1.0, hi - lo)
        out[idx] = ((block - lo) / span).astype(np.float32)
    return out


# ------------------------------------------------------------------ view builders


def build_sequences(
    scaled: np.ndarray,
    units: np.ndarray,
    cycles: np.ndarray,
    rul: np.ndarray,
    window: int = WINDOW_LENGTH,
) -> dict[str, np.ndarray]:
    """Sliding windows of ``window`` consecutive cycles, one engine at a time.

    Labels are taken at the *last* cycle of each window (predict-now semantics,
    matching a live feed where the newest sample is the one being judged).
    Engines shorter than ``window`` are left-padded by repeating their first row,
    so every engine contributes at least one sequence.
    """
    seq_x: list[np.ndarray] = []
    seq_rul: list[float] = []
    seq_unit: list[int] = []
    seq_end: list[int] = []

    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        idx = idx[np.argsort(cycles[idx])]
        block, block_rul, block_cycles = scaled[idx], rul[idx], cycles[idx]
        if len(idx) < window:  # pad short engines by repeating the first cycle
            pad = window - len(idx)
            block = np.vstack([np.repeat(block[:1], pad, axis=0), block])
            block_rul = np.concatenate([np.repeat(block_rul[:1], pad), block_rul])
            block_cycles = np.concatenate([np.repeat(block_cycles[:1], pad), block_cycles])
        for end in range(window - 1, len(block)):
            seq_x.append(block[end - window + 1 : end + 1])
            seq_rul.append(float(block_rul[end]))
            seq_unit.append(int(unit))
            seq_end.append(int(block_cycles[end]))

    if not seq_x:
        return {
            "X": np.empty((0, window, scaled.shape[1]), dtype=np.float32),
            "rul": np.empty(0, dtype=np.float32),
            "stage": np.empty(0, dtype=np.int64),
            "unit": np.empty(0, dtype=np.int64),
            "cycle_end": np.empty(0, dtype=np.int64),
        }

    rul_arr = np.asarray(seq_rul, dtype=np.float32)
    return {
        "X": np.stack(seq_x).astype(np.float32),
        "rul": rul_arr,
        "stage": stage_from_rul(rul_arr),
        "unit": np.asarray(seq_unit, dtype=np.int64),
        "cycle_end": np.asarray(seq_end, dtype=np.int64),
    }


def healthy_mask(
    units: np.ndarray, cycles: np.ndarray, fraction: float = HEALTHY_LIFE_FRACTION
) -> np.ndarray:
    """True for rows in the first ``fraction`` of their engine's observed cycles."""
    mask = np.zeros(len(units), dtype=bool)
    for unit in np.unique(units):
        idx = np.flatnonzero(units == unit)
        mask[idx] = cycles[idx] <= cycles[idx].max() * fraction
    return mask


# ------------------------------------------------------------------ orchestration


@dataclass
class Prepared:
    """All three views, for all four splits, plus the scaler and metadata."""

    features: list[str]
    scaler: MinMaxStats
    window: int
    unit_splits: dict[str, np.ndarray]
    scaler_mode: str = "global"
    tabular: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    sequences: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    healthy: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def to_npz_payload(self) -> dict[str, np.ndarray]:
        payload: dict[str, np.ndarray] = {
            "features": np.asarray(self.features, dtype=object),
            "window": np.asarray([self.window]),
            "scaler_min": self.scaler.minimum,
            "scaler_max": self.scaler.maximum,
            "scaler_mode": np.asarray([self.scaler_mode], dtype=object),
            "stage_labels": np.asarray(STAGE_LABELS, dtype=object),
        }
        for split, units in self.unit_splits.items():
            payload[f"units_{split}"] = units
        for view, store in (("tab", self.tabular), ("seq", self.sequences), ("hlt", self.healthy)):
            for split, arrays in store.items():
                for key, arr in arrays.items():
                    payload[f"{view}_{split}_{key}"] = arr
        return payload


def prepare(scaler_mode: str = "global", window: int = WINDOW_LENGTH) -> Prepared:
    """Build every view for every split. Pure numpy/pandas, no model imports."""
    features = kept_features()
    train_raw, test_raw, final_rul = load_raw()

    labelled_train = add_run_to_failure_rul(train_raw)
    labelled_official = add_truncated_rul(test_raw, final_rul)

    unit_splits = split_units(labelled_train["unit"].unique())
    train_rows = labelled_train[labelled_train["unit"].isin(unit_splits["train"])]

    # Fit on the training split only -- never on val/test/official.
    scaler = fit_global_minmax(train_rows, features)

    frames: dict[str, pd.DataFrame] = {
        split: labelled_train[labelled_train["unit"].isin(units)].reset_index(drop=True)
        for split, units in unit_splits.items()
    }
    frames["official"] = labelled_official.reset_index(drop=True)

    prepared = Prepared(features, scaler, window, unit_splits, scaler_mode=scaler_mode)
    for split, frame in frames.items():
        scaled = (
            scale_per_engine(frame, features)
            if scaler_mode == "per_engine"
            else scaler.transform(frame)
        )
        units = frame["unit"].to_numpy(dtype=np.int64)
        cycles = frame["cycle"].to_numpy(dtype=np.int64)
        rul = frame["rul"].to_numpy(dtype=np.float32)

        prepared.tabular[split] = {
            "X": scaled,
            "rul": rul,
            "rul_raw": frame["rul_raw"].to_numpy(dtype=np.int64),
            "health": health_from_rul(rul).astype(np.float32),
            "stage": stage_from_rul(rul),
            "fault": (rul < FAULT_RUL_THRESHOLD).astype(np.int64),
            "unit": units,
            "cycle": cycles,
        }
        prepared.sequences[split] = build_sequences(scaled, units, cycles, rul, window)
        mask = healthy_mask(units, cycles)
        prepared.healthy[split] = {
            "X": scaled[mask],
            "rul": rul[mask],
            "unit": units[mask],
            "cycle": cycles[mask],
        }
    return prepared


def build_manifest(prepared: Prepared) -> dict:
    counts = {}
    for split in ALL_VIEWS:
        tab = prepared.tabular[split]
        seq = prepared.sequences[split]
        hlt = prepared.healthy[split]
        counts[split] = {
            "engines": int(len(np.unique(tab["unit"]))),
            "tabular_rows": int(len(tab["X"])),
            "sequences": int(len(seq["X"])),
            "healthy_rows": int(len(hlt["X"])),
            "healthy_min_rul": float(hlt["rul"].min()) if len(hlt["rul"]) else None,
            "fault_rate": float(tab["fault"].mean()),
            "stage_counts": {
                label: int((tab["stage"] == i).sum()) for i, label in enumerate(STAGE_LABELS)
            },
        }
    scaler_dict = prepared.scaler.to_dict()
    scaler_dict["mode"] = prepared.scaler_mode
    return {
        "dataset": "NASA C-MAPSS FD001",
        "source_files": sorted(p.name for p in CMAPSS_DIR.glob("*FD001.txt")),
        "feature_names": prepared.features,
        "n_features": len(prepared.features),
        "dropped_constant_features": list(CONSTANT_FEATURES),
        "rul_ceiling": RUL_CEILING,
        "window_length": prepared.window,
        "stage_labels": list(STAGE_LABELS),
        "stage_thresholds_rul": {
            "critical_max": STAGE_CRITICAL_MAX_RUL,
            "degrading_max": STAGE_DEGRADING_MAX_RUL,
        },
        "fault_rul_threshold": FAULT_RUL_THRESHOLD,
        "healthy_life_fraction": HEALTHY_LIFE_FRACTION,
        "split_by": "engine unit id",
        "split_fractions": list(SPLIT_FRACTIONS),
        "split_seed": SPLIT_SEED,
        "unit_splits": {k: [int(u) for u in v] for k, v in prepared.unit_splits.items()},
        "scaler": scaler_dict,
        "counts": counts,
    }


def save(prepared: Prepared) -> tuple[Path, Path]:
    PROCESSED_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED_NPZ, **prepared.to_npz_payload())
    write_json(PROCESSED_MANIFEST, build_manifest(prepared))
    return PROCESSED_NPZ, PROCESSED_MANIFEST


# ------------------------------------------------------------------ cached access


class Dataset:
    """Read-only accessor over the cached ``fd001.npz``."""

    def __init__(self, path: Path = PROCESSED_NPZ):
        if not path.exists():
            raise FileNotFoundError(f"{rel(path)} not found. Run: python ml/preprocess.py")
        self._z = np.load(path, allow_pickle=True)
        self.features: list[str] = [str(f) for f in self._z["features"]]
        self.window = int(self._z["window"][0])
        self.scaler_min = self._z["scaler_min"]
        self.scaler_max = self._z["scaler_max"]
        self.scaler_mode = str(self._z["scaler_mode"][0])
        self.stage_labels = [str(s) for s in self._z["stage_labels"]]

    def tabular(self, split: str) -> dict[str, np.ndarray]:
        return self._view("tab", split)

    def sequences(self, split: str) -> dict[str, np.ndarray]:
        return self._view("seq", split)

    def healthy(self, split: str) -> dict[str, np.ndarray]:
        return self._view("hlt", split)

    def units(self, split: str) -> np.ndarray:
        return self._z[f"units_{split}"]

    def feature_index(self, name: str) -> int:
        return self.features.index(name)

    def scaler_stats(self) -> dict:
        """Exactly the ``normalization`` block that goes into each meta.json."""
        return {
            "mode": self.scaler_mode,
            "method": "min_max",
            "feature_names": list(self.features),
            "min": [float(v) for v in self.scaler_min],
            "max": [float(v) for v in self.scaler_max],
            "clip_range": [-0.5, 1.5],
            "formula": "x_scaled = clip((x - min) / (max - min), -0.5, 1.5)",
        }

    def _view(self, prefix: str, split: str) -> dict[str, np.ndarray]:
        tag = f"{prefix}_{split}_"
        found = {k[len(tag) :]: self._z[k] for k in self._z.files if k.startswith(tag)}
        if not found:
            raise KeyError(f"no '{prefix}' view for split '{split}'")
        return found


def ensure_prepared(force: bool = False, scaler_mode: str = "global") -> Dataset:
    """Build the cache if absent, then return an accessor."""
    if force or not PROCESSED_NPZ.exists():
        save(prepare(scaler_mode=scaler_mode))
    return Dataset()


# ------------------------------------------------------------------ cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Preprocess C-MAPSS FD001 into cached views.")
    ap.add_argument("--force", action="store_true", help="rebuild even if the cache exists")
    ap.add_argument(
        "--scaler",
        choices=("global", "per_engine"),
        default="global",
        help="min-max mode; 'global' (default) is the only one replayable at inference",
    )
    ap.add_argument("--window", type=int, default=WINDOW_LENGTH, help="LSTM window length")
    args = ap.parse_args(argv)

    if PROCESSED_NPZ.exists() and not args.force:
        print(f"{rel(PROCESSED_NPZ)} already exists (use --force to rebuild)")
    else:
        npz, manifest = save(prepare(scaler_mode=args.scaler, window=args.window))
        print(f"wrote {rel(npz)} ({npz.stat().st_size / 1e6:.1f} MB)")
        print(f"wrote {rel(manifest)}")

    ds = Dataset()
    print(f"\nfeatures ({len(ds.features)}): {', '.join(ds.features)}")
    print(f"window: {ds.window}   scaler: {ds.scaler_mode}")
    header = (
        f"\n{'split':10s} {'engines':>8s} {'rows':>7s} {'seqs':>7s} "
        f"{'healthy':>8s} {'fault%':>7s}  stages(h/d/c)"
    )
    print(header)
    for split in ALL_VIEWS:
        tab, seq, hlt = ds.tabular(split), ds.sequences(split), ds.healthy(split)
        stages = [int((tab["stage"] == i).sum()) for i in range(3)]
        print(
            f"{split:10s} {len(np.unique(tab['unit'])):8d} {len(tab['X']):7d} "
            f"{len(seq['X']):7d} {len(hlt['X']):8d} {100 * tab['fault'].mean():6.1f}%  "
            f"{stages[0]}/{stages[1]}/{stages[2]}"
        )
    tail = ", ".join(f"{s}={ds.healthy(s)['rul'].min():.0f}" for s in ALL_VIEWS)
    print(f"\nhealthy-subset min RUL per split: {tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
