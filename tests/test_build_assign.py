import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from cfa.data.build import build_district_table
from cfa.geo.assign import assign_by_nearest_centroid, assign_by_polygon
from cfa.visibility.dvi import compute_dvi, compute_visibility_inputs

REF = pd.DataFrame(
    {
        "district_id": ["D1", "D2", "D3", "D4"],
        "district": ["Gurugram", "Prayagraj", "Ahmedabad", "Pune"],
        "state": ["Haryana", "Uttar Pradesh", "Gujarat", "Maharashtra"],
        "lat": [28.46, 25.43, 23.02, 18.52],
        "lon": [77.03, 81.85, 72.57, 73.86],
        "area_km2": [1253.0, 5482.0, 8087.0, 15643.0],
        "population": [1.5e6, 6.0e6, 7.2e6, 9.4e6],
    }
)
NFHS = pd.DataFrame(
    {
        "district": ["Gurgaon", "Allahabad", "Ahmadabad", "Pune"],
        "state": ["Haryana", "Uttar Pradesh", "Gujarat", "Maharashtra"],
        "hh_electricity": [99.0, 95.0, 98.0, 99.5],
        "households_surveyed": [900.0, 700.0, 1000.0, 850.0],
    }
)


def _stations():
    return pd.DataFrame(
        {
            "station_id": ["s1", "s2", "s3"],
            "lat": [28.45, 23.03, 40.0],
            "lon": [77.02, 72.58, 77.0],
            "completeness": [0.9, 0.8, 0.5],
        }
    )


def test_nearest_centroid_assignment_flags_method_and_far_stations():
    out = assign_by_nearest_centroid(_stations(), REF, max_km=50)
    assert out["district_id"].tolist()[:2] == ["D1", "D3"]
    assert out["district_id"].isna().iloc[2]
    assert (out["assignment_method"] == "nearest_centroid_approx").all()


def test_polygon_assignment():
    polys = gpd.GeoDataFrame(
        {"district_id": ["P1", "P2"]},
        geometry=[box(77, 28, 78, 29), box(72, 23, 73, 24)],
        crs="EPSG:4326",
    )
    out = assign_by_polygon(_stations(), polys)
    assert out["district_id"].tolist()[:2] == ["P1", "P2"]
    assert pd.isna(out["district_id"].iloc[2]) and len(out) == 3
    with pytest.raises(ValueError):
        assign_by_polygon(_stations(), polys.set_crs(None, allow_override=True))


def test_build_table_feeds_dvi_end_to_end():
    st = assign_by_nearest_centroid(_stations(), REF, max_km=50)
    districts, report = build_district_table(REF, st, NFHS)
    assert report["nfhs_crosswalk"]["match_rate"] == 1.0 and report["stations_unassigned"] == 1
    assert districts.set_index("district_id").loc["D1", "survey_n"] == 900.0
    assert districts["cloud_free_frac"].isna().all()  # optional input absent -> skipped by DVI
    stations_ok = st.dropna(subset=["district_id"])
    scores = compute_dvi(compute_visibility_inputs(districts, stations_ok))
    assert scores["dvi"].between(0, 100).all()
    by_id = scores.set_index("district_id")["dvi"]
    assert by_id["D1"] > by_id["D4"]  # has stations vs none


def test_low_match_rate_raises():
    bad = NFHS.assign(district=["Qqq", "Www", "Eee", "Rrr"])
    st = assign_by_nearest_centroid(_stations(), REF, max_km=50)
    with pytest.raises(ValueError, match="match rate"):
        build_district_table(REF, st, bad)


def test_reference_validation_blocks_bad_rows():
    bad = REF.assign(lat=[28.46, 25.43, 23.02, 5.0])
    with pytest.raises(ValueError, match="failed validation"):
        build_district_table(bad, assign_by_nearest_centroid(_stations(), REF))
    assert np.isfinite(REF["area_km2"]).all()
