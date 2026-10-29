"""Load NFHS-5 district factsheet indicators from a user-supplied CSV.

NFHS conventions handled: '*' = suppressed (fewer than 25 cases) -> NaN; '(x)' = based on
25-49 unweighted cases -> kept but flagged low-n; 'na', '-' -> NaN; thousands separators.
Factsheet column names vary by export, so map them explicitly via column_map.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from cfa.data.schemas import validate  # noqa: F401  (re-exported for convenience)

_MISSING = {"*", "na", "n/a", "-", "--", "", "nan"}
_PAREN = re.compile(r"^\(\s*([-+]?\d*\.?\d+)\s*\)$")


def parse_nfhs_value(raw: object) -> tuple[float, bool]:
    """Return (value, low_n_flag). Suppressed/missing -> (nan, False)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.nan, False
    if isinstance(raw, int | float | np.number):
        return float(raw), False
    text = str(raw).strip().replace(",", "")
    if text.lower() in _MISSING:
        return np.nan, False
    m = _PAREN.match(text)
    if m:
        return float(m.group(1)), True
    try:
        return float(text), False
    except ValueError:
        return np.nan, False


def load_nfhs_csv(
    path: str | Path,
    column_map: Mapping[str, str],
    percent_cols: tuple[str, ...] = ("hh_electricity", "hh_clean_fuel"),
) -> pd.DataFrame:
    """column_map: {csv_column: canonical}, canonical in district, state, households_surveyed,
    and any indicator names (e.g. hh_electricity, hh_clean_fuel). Adds <col>_low_n flags."""
    return standardise_nfhs(pd.read_csv(path, dtype=str), column_map, percent_cols)


def standardise_nfhs(
    df: pd.DataFrame, column_map: Mapping[str, str], percent_cols: tuple[str, ...] = ()
) -> pd.DataFrame:
    out = df.rename(columns=dict(column_map))
    missing = {"district", "state"} - set(out.columns)
    if missing:
        raise ValueError(f"column_map must produce columns {sorted(missing)}.")
    unknown = [c for c in percent_cols if c not in out.columns]
    if unknown:
        raise ValueError(f"Indicator columns not found after mapping: {unknown}")
    for col in percent_cols:
        parsed = out[col].map(parse_nfhs_value)
        out[col] = parsed.map(lambda t: t[0]).astype(float)
        out[f"{col}_low_n"] = parsed.map(lambda t: t[1]).astype(bool)
        bad = out[col].dropna()
        if ((bad < 0) | (bad > 100)).any():
            raise ValueError(f"{col}: percentages outside 0-100; check the column mapping.")
    if "households_surveyed" in out.columns:
        out["households_surveyed"] = out["households_surveyed"].map(
            lambda v: parse_nfhs_value(v)[0]
        )
    return out.reset_index(drop=True)
