"""Pandera schemas: fail loudly (with ALL problems listed) before bad data reaches the DVI."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# Generous bounding box for India incl. islands. Catches swapped lat/lon and stray zeros.
INDIA_LAT = (6.0, 37.5)
INDIA_LON = (68.0, 98.0)

_lat = pa.Column(float, pa.Check.in_range(*INDIA_LAT), nullable=False, coerce=True)
_lon = pa.Column(float, pa.Check.in_range(*INDIA_LON), nullable=False, coerce=True)

StationSchema = pa.DataFrameSchema(
    {
        "station_id": pa.Column(str, unique=True, nullable=False, coerce=True),
        "lat": _lat,
        "lon": _lon,
        "completeness": pa.Column(
            float, pa.Check.in_range(0, 1), nullable=True, coerce=True, required=False
        ),
        "district_id": pa.Column(str, nullable=True, coerce=True, required=False),
    },
    strict=False,
)

ReferenceDistrictSchema = pa.DataFrameSchema(
    {
        "district_id": pa.Column(str, unique=True, nullable=False, coerce=True),
        "district": pa.Column(str, nullable=False, coerce=True),
        "state": pa.Column(str, nullable=False, coerce=True),
        "lat": _lat,
        "lon": _lon,
        "area_km2": pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True),
        "population": pa.Column(float, pa.Check.gt(0), nullable=False, coerce=True),
    },
    strict=False,
)


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema, name: str = "table") -> pd.DataFrame:
    """Validate lazily so every violation is reported at once, not just the first."""
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        bad = exc.failure_cases[["column", "check", "failure_case"]].head(15).to_string(index=False)
        raise ValueError(
            f"{name} failed validation ({len(exc.failure_cases)} problems):\n{bad}"
        ) from exc
