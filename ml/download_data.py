"""Fetch the NASA C-MAPSS FD001 turbofan degradation files into data/cmapss/.

The original NASA PCoE download page (ti.arc.nasa.gov) is long dead. Two live
mirrors are tried in order:

  1. The NASA PHM datasets S3 bucket, which still serves the original
     "6. Turbofan Engine Degradation Simulation Data Set.zip" (all of FD001-FD004).
  2. Raw files from a public GitHub mirror, as a per-file fallback.

Usage:
    python ml/download_data.py            # download if missing
    python ml/download_data.py --force    # re-download even if present
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CMAPSS_DIR = REPO_ROOT / "data" / "cmapss"

REQUIRED_FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")

ZIP_URL = (
    "https://phm-datasets.s3.amazonaws.com/NASA/"
    "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
)
RAW_BASE = (
    "https://raw.githubusercontent.com/hankroark/"
    "Turbofan-Engine-Degradation/master/CMAPSSData"
)

# Expected row counts, used as a cheap sanity check on whatever we end up with.
EXPECTED_ROWS = {"train_FD001.txt": 20631, "test_FD001.txt": 13096, "RUL_FD001.txt": 100}

USER_AGENT = "glance-ml-dataset-fetch/1.0"
TIMEOUT = 120


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def _write(name: str, payload: bytes) -> Path:
    dest = CMAPSS_DIR / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest


def _row_count(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def _from_zip() -> bool:
    """Pull the three FD001 files out of the NASA S3 zip (possibly a nested zip)."""
    print(f"[1/2] trying NASA S3 archive: {ZIP_URL}")
    try:
        blob = _get(ZIP_URL)
    except Exception as exc:  # noqa: BLE001 - any network/HTTP error means "try next"
        print(f"      failed: {type(exc).__name__}: {exc}")
        return False

    print(f"      got {len(blob) / 1e6:.1f} MB, scanning archive...")
    try:
        outer = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        print("      not a zip file")
        return False

    # The NASA archive wraps CMAPSSData.zip inside the outer zip.
    candidates: list[zipfile.ZipFile] = [outer]
    for info in outer.infolist():
        if info.filename.lower().endswith(".zip"):
            try:
                candidates.append(zipfile.ZipFile(io.BytesIO(outer.read(info))))
            except zipfile.BadZipFile:
                continue

    found = 0
    for zf in candidates:
        by_base = {Path(n).name: n for n in zf.namelist()}
        for name in REQUIRED_FILES:
            if name in by_base and not (CMAPSS_DIR / name).exists():
                dest = _write(name, zf.read(by_base[name]))
                print(f"      wrote {dest.relative_to(REPO_ROOT)}")
                found += 1
    return found == len(REQUIRED_FILES)


def _from_raw() -> bool:
    """Per-file fallback against a public raw-file mirror."""
    print(f"[2/2] trying raw mirror: {RAW_BASE}")
    ok = True
    for name in REQUIRED_FILES:
        if (CMAPSS_DIR / name).exists():
            continue
        url = f"{RAW_BASE}/{name}"
        try:
            dest = _write(name, _get(url))
            print(f"      wrote {dest.relative_to(REPO_ROOT)}")
        except Exception as exc:  # noqa: BLE001
            print(f"      failed {name}: {type(exc).__name__}: {exc}")
            ok = False
    return ok


MANUAL_HELP = f"""
Automatic download failed. Fetch the files by hand:

  1. Download the C-MAPSS "Turbofan Engine Degradation Simulation Data Set":
       {ZIP_URL}
     (mirrors: Kaggle "NASA CMAPSS Jet Engine Simulated Data", or search the
      NASA Prognostics Center of Excellence data repository)
  2. Unzip it, then copy these three files into:
       {CMAPSS_DIR}
       - train_FD001.txt   ({EXPECTED_ROWS['train_FD001.txt']} rows, 100 engines)
       - test_FD001.txt    ({EXPECTED_ROWS['test_FD001.txt']} rows, 100 engines)
       - RUL_FD001.txt     ({EXPECTED_ROWS['RUL_FD001.txt']} rows)
  3. Re-run: python ml/download_data.py
""".strip()


def ensure_dataset(force: bool = False) -> bool:
    """Make sure the three FD001 files exist locally. Returns True on success."""
    CMAPSS_DIR.mkdir(parents=True, exist_ok=True)
    if force:
        for name in REQUIRED_FILES:
            (CMAPSS_DIR / name).unlink(missing_ok=True)

    missing = [n for n in REQUIRED_FILES if not (CMAPSS_DIR / n).exists()]
    if not missing:
        print(f"dataset already present in {CMAPSS_DIR.relative_to(REPO_ROOT)}")
    else:
        print(f"missing: {', '.join(missing)}")
        if not _from_zip():
            _from_raw()

    missing = [n for n in REQUIRED_FILES if not (CMAPSS_DIR / n).exists()]
    if missing:
        print(f"\nstill missing: {', '.join(missing)}\n")
        print(MANUAL_HELP)
        return False

    print("\nverifying:")
    ok = True
    for name in REQUIRED_FILES:
        path = CMAPSS_DIR / name
        rows = _row_count(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        expected = EXPECTED_ROWS[name]
        flag = "ok" if rows == expected else f"WARNING expected {expected}"
        ok = ok and rows == expected
        print(f"  {name:18s} {rows:6d} rows  sha256:{digest}  {flag}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)
    return 0 if ensure_dataset(force=args.force) else 1


if __name__ == "__main__":
    sys.exit(main())
