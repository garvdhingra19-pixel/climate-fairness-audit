"""Load monitoring-station tables (e.g. a CPCB/CAAQMS export or any CSV) into a validated frame."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from cfa.data.schemas import StationSchema, validate


def load_stations_csv(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    """Read a CSV, rename columns via column_map {csv_name: station_id|lat|lon|completeness},
    drop duplicate station ids (with a warning) and validate against StationSchema."""
    df = pd.read_csv(path)
    return standardise_stations(df, column_map, source)


def standardise_stations(
    df: pd.DataFrame, column_map: Mapping[str, str] | None = None, source: str | None = None
) -> pd.DataFrame:
    out = df.rename(columns=dict(column_map or {}))
    missing = {"station_id", "lat", "lon"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)}; pass column_map to rename them.")
    for col in ("lat", "lon", "completeness"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    n_dupes = int(out["station_id"].duplicated().sum())
    if n_dupes:
        warnings.warn(f"Dropped {n_dupes} duplicate station_id rows (kept first).", stacklevel=2)
        out = out.drop_duplicates("station_id", keep="first")
    if source:
        out["source"] = source
    return validate(out.reset_index(drop=True), StationSchema, "stations")
