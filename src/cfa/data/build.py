"""Assemble the district table that feeds the DVI from reference, survey and station data."""

from __future__ import annotations

import pandas as pd

from cfa.data.schemas import ReferenceDistrictSchema, validate
from cfa.geo.crosswalk import crosswalk_report, match_districts


def build_district_table(
    reference: pd.DataFrame,
    stations: pd.DataFrame,
    nfhs: pd.DataFrame | None = None,
    cloud_free: pd.Series | None = None,
    min_match_rate: float = 0.9,
) -> tuple[pd.DataFrame, dict]:
    """Return (districts, report).

    reference: district_id, district, state, lat, lon, area_km2, population (validated).
    stations: needs district_id (assign first) and optional completeness.
    nfhs: standardised NFHS frame; matched to reference via the crosswalk.
    cloud_free: optional Series indexed by district_id; NaN is fine (DVI skips it).
    Raises if the NFHS match rate is below min_match_rate: a poor crosswalk must not pass silently.
    """
    ref = validate(reference, ReferenceDistrictSchema, "reference districts")
    out = ref[["district_id", "district", "state", "lat", "lon", "area_km2", "population"]].copy()
    report: dict = {}

    out["survey_n"] = float("nan")
    if nfhs is not None:
        cw = match_districts(nfhs, ref)
        report["nfhs_crosswalk"] = crosswalk_report(cw)
        if report["nfhs_crosswalk"]["match_rate"] < min_match_rate:
            raise ValueError(
                f"NFHS match rate {report['nfhs_crosswalk']['match_rate']:.1%} is below "
                f"{min_match_rate:.0%}; review unmatched names or add overrides."
            )
        matched = nfhs.loc[cw["source_index"]].assign(district_id=cw["district_id"].to_numpy())
        matched = matched.dropna(subset=["district_id"]).drop_duplicates("district_id", keep=False)
        for col in matched.columns.difference(["district", "state", "district_id"]):
            out[col] = out["district_id"].map(matched.set_index("district_id")[col])
        if "households_surveyed" in out.columns:
            out["survey_n"] = out["households_surveyed"]

    st = stations.dropna(subset=["district_id"])
    comp = (
        st.groupby("district_id")["completeness"].mean() if "completeness" in st.columns else None
    )
    out["completeness"] = out["district_id"].map(comp) if comp is not None else float("nan")
    out["cloud_free_frac"] = (
        out["district_id"].map(cloud_free) if cloud_free is not None else float("nan")
    )
    report["stations_unassigned"] = int(stations["district_id"].isna().sum())
    report["n_districts"] = int(len(out))
    return out, report
