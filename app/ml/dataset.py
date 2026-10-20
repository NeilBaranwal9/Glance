"""Load an arbitrary CSV/TXT of sensor data and work out what the columns mean.

The user can point the ML Analysis tab at any tabular log. This module does the
detective work:

  * sniff the delimiter (comma, tab, semicolon, or runs of whitespace)
  * decide whether there is a header row
  * classify every column as a sensor channel, a unit/engine id, a cycle/time
    index, or an explicit RUL/label column
  * report what it found so the UI can show a mapping dialog for confirmation

Nothing here assumes the C-MAPSS layout; that dataset is simply one file this
loader handles correctly (whitespace-delimited, no header, unit + cycle + 24
channels).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Column-name heuristics. Matched case-insensitively against the whole name.
_UNIT_PATTERNS = (r"^unit.*", r"^engine.*", r"^machine.*", r"^asset.*", r"^id$", r".*_id$")
_CYCLE_PATTERNS = (r"^cycle.*", r"^time.*", r"^t$", r"^timestamp.*", r"^step.*", r"^index$")
_RUL_PATTERNS = (r"^rul.*", r".*remaining.*life.*", r"^ttf$", r".*time_to_fail.*")
_LABEL_PATTERNS = (r"^label.*", r"^fault.*", r"^failure.*", r"^stage.*", r"^health.*", r"^target$")

MAX_PREVIEW_ROWS = 5


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    lowered = name.strip().lower().replace(" ", "_")
    return any(re.fullmatch(p, lowered) for p in patterns)


@dataclass
class ColumnRoles:
    """Which column plays which part. All fields are column *names*."""

    sensors: list[str] = field(default_factory=list)
    unit: Optional[str] = None
    cycle: Optional[str] = None
    rul: Optional[str] = None
    label: Optional[str] = None
    ignored: list[str] = field(default_factory=list)

    @property
    def has_run_to_failure(self) -> bool:
        """True when RUL can be derived per row (unit + cycle) or is given outright."""
        return self.rul is not None or (self.unit is not None and self.cycle is not None)

    def describe(self) -> str:
        bits = [f"{len(self.sensors)} sensor channels"]
        for role in ("unit", "cycle", "rul", "label"):
            value = getattr(self, role)
            if value:
                bits.append(f"{role}={value}")
        if self.ignored:
            bits.append(f"{len(self.ignored)} ignored")
        return ", ".join(bits)


@dataclass
class LoadedDataset:
    """A parsed table plus the roles assigned to its columns."""

    frame: pd.DataFrame
    roles: ColumnRoles
    path: Optional[Path] = None
    delimiter: str = ","
    had_header: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_sensors(self) -> int:
        return len(self.roles.sensors)

    def sensor_values(self) -> np.ndarray:
        return self.frame[self.roles.sensors].to_numpy(dtype=np.float64)

    def units(self) -> np.ndarray:
        """Per-row unit id. Synthesises a single unit when the file has none."""
        if self.roles.unit is None:
            return np.zeros(len(self.frame), dtype=np.int64)
        codes = self.frame[self.roles.unit]
        if not np.issubdtype(codes.dtype, np.number):
            codes = codes.astype("category").cat.codes
        return codes.to_numpy(dtype=np.int64)

    def cycles(self) -> np.ndarray:
        """Per-row time index. Falls back to position within each unit."""
        if self.roles.cycle is not None:
            values = self.frame[self.roles.cycle].to_numpy(dtype=np.float64)
            if np.all(np.isfinite(values)):
                return values
        units = self.units()
        out = np.zeros(len(self.frame), dtype=np.float64)
        for unit in np.unique(units):
            mask = units == unit
            out[mask] = np.arange(int(mask.sum()), dtype=np.float64) + 1.0
        return out

    def rul(self, ceiling: Optional[float] = None) -> Optional[np.ndarray]:
        """Per-row remaining useful life, or None when it cannot be derived."""
        if self.roles.rul is not None:
            values = self.frame[self.roles.rul].to_numpy(dtype=np.float64)
        elif self.roles.unit is not None and self.roles.cycle is not None:
            units, cycles = self.units(), self.cycles()
            values = np.zeros(len(cycles), dtype=np.float64)
            for unit in np.unique(units):
                mask = units == unit
                values[mask] = cycles[mask].max() - cycles[mask]
        else:
            return None
        if ceiling is not None:
            values = np.clip(values, 0.0, ceiling)
        return values

    def summary(self) -> str:
        source = self.path.name if self.path else "in-memory table"
        lines = [
            f"{source}: {self.n_rows} rows x {len(self.frame.columns)} columns",
            f"  roles: {self.roles.describe()}",
            f"  run-to-failure labels: {'yes' if self.roles.has_run_to_failure else 'no'}",
        ]
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


# ------------------------------------------------------------------ sniffing


def _sniff_delimiter(sample: str) -> tuple[str, bool]:
    """Return ``(delimiter, is_whitespace_run)`` for a text sample."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter, False
    except csv.Error:
        pass
    lines = [ln for ln in sample.splitlines() if ln.strip()][:10]
    if lines and all(len(ln.split()) > 1 for ln in lines):
        counts = {len(ln.split()) for ln in lines}
        if len(counts) == 1:
            return r"\s+", True
    return ",", False


def _looks_like_header(row: list[str]) -> bool:
    """A header row is one where the cells are not all parseable as numbers."""
    if not row:
        return False
    numeric = 0
    for cell in row:
        try:
            float(cell)
            numeric += 1
        except (TypeError, ValueError):
            pass
    return numeric < max(1, len(row) // 2)


# ------------------------------------------------------------------ loading


def load_table(path: str | Path, max_rows: Optional[int] = None) -> LoadedDataset:
    """Read a delimited text file and assign a role to every column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")

    sample = path.read_text(encoding="utf-8", errors="replace")[:64_000]
    delimiter, is_whitespace = _sniff_delimiter(sample)

    first_line = next((ln for ln in sample.splitlines() if ln.strip()), "")
    cells = first_line.split() if is_whitespace else first_line.split(delimiter)
    has_header = _looks_like_header([c.strip() for c in cells])

    frame = pd.read_csv(
        path,
        sep=delimiter,
        engine="python" if is_whitespace else "c",
        header=0 if has_header else None,
        nrows=max_rows,
        skipinitialspace=True,
    )
    if not has_header:
        frame.columns = [f"col_{i}" for i in range(len(frame.columns))]

    notes: list[str] = []
    # Files exported with a trailing delimiter leave an all-NaN final column.
    empty = [c for c in frame.columns if frame[c].isna().all()]
    if empty:
        frame = frame.drop(columns=empty)
        notes.append(f"dropped {len(empty)} all-empty column(s)")

    dataset = assign_roles(frame, had_header=has_header)
    dataset.path = path
    dataset.delimiter = "whitespace" if is_whitespace else delimiter
    dataset.had_header = has_header
    dataset.notes = notes + dataset.notes
    return dataset


def assign_roles(frame: pd.DataFrame, had_header: bool = True) -> LoadedDataset:
    """Classify columns into sensors / unit / cycle / rul / label."""
    roles = ColumnRoles()
    notes: list[str] = []

    numeric_cols: list[str] = []
    for column in frame.columns:
        series = pd.to_numeric(frame[column], errors="coerce")
        # A column is usable if most of it parses as a number.
        if series.notna().mean() > 0.9:
            frame[column] = series
            numeric_cols.append(column)
        else:
            roles.ignored.append(str(column))

    if not numeric_cols:
        raise ValueError("no numeric columns found -- nothing to analyse")

    named = [c for c in numeric_cols if isinstance(c, str)]
    for column in named:
        if roles.unit is None and _matches(column, _UNIT_PATTERNS):
            roles.unit = column
        elif roles.cycle is None and _matches(column, _CYCLE_PATTERNS):
            roles.cycle = column
        elif roles.rul is None and _matches(column, _RUL_PATTERNS):
            roles.rul = column
        elif roles.label is None and _matches(column, _LABEL_PATTERNS):
            roles.label = column

    # Headerless files: fall back to the positional C-MAPSS-style convention,
    # but only when the first two columns actually behave like ids and counters.
    if not had_header and roles.unit is None and roles.cycle is None and len(numeric_cols) >= 3:
        first, second = numeric_cols[0], numeric_cols[1]
        first_vals = frame[first].to_numpy(dtype=np.float64)
        second_vals = frame[second].to_numpy(dtype=np.float64)
        looks_like_ids = (
            np.allclose(first_vals, np.round(first_vals))
            and len(np.unique(first_vals)) < max(2, len(frame) // 2)
        )
        looks_like_counter = (
            np.allclose(second_vals, np.round(second_vals)) and second_vals.min() >= 0
        )
        if looks_like_ids and looks_like_counter:
            roles.unit, roles.cycle = first, second
            notes.append(
                "headerless file: treated column 0 as unit id and column 1 as cycle "
                "(C-MAPSS convention)"
            )

    assigned = {roles.unit, roles.cycle, roles.rul, roles.label} - {None}
    roles.sensors = [c for c in numeric_cols if c not in assigned]

    if not roles.sensors:
        raise ValueError("every numeric column was consumed by metadata roles")
    return LoadedDataset(frame=frame, roles=roles, notes=notes)


def preview(dataset: LoadedDataset, rows: int = MAX_PREVIEW_ROWS) -> list[dict[str, Any]]:
    """First few rows as plain dicts, for a confirmation dialog."""
    return dataset.frame.head(rows).to_dict(orient="records")


def from_history(history: dict[str, list[dict]], names: Optional[list[str]] = None) -> LoadedDataset:
    """Build a dataset from the app's own ``data_history`` (a live capture).

    ``history`` maps parameter id -> list of ``{'timestamp', 'value',
    'filtered_value'}``. Channels are aligned on their common length, newest
    samples kept.
    """
    keys = [k for k in (names or list(history)) if history.get(k)]
    if not keys:
        raise ValueError("no captured channels to train on")
    length = min(len(history[k]) for k in keys)
    if length < 2:
        raise ValueError("not enough captured samples to train on")

    columns = {k: [p["filtered_value"] for p in history[k][-length:]] for k in keys}
    frame = pd.DataFrame(columns)
    frame.insert(0, "cycle", np.arange(1, length + 1, dtype=np.float64))

    roles = ColumnRoles(sensors=list(keys), cycle="cycle")
    return LoadedDataset(
        frame=frame,
        roles=roles,
        notes=["captured from the live telemetry buffer"],
    )
