import numpy as np
import pandas as pd
import pytest

from cfa.data.ingest.nfhs import parse_nfhs_value, standardise_nfhs
from cfa.data.ingest.openaq import OpenAQClient, parse_locations
from cfa.data.ingest.stations import load_stations_csv, standardise_stations
from cfa.data.schemas import ReferenceDistrictSchema, validate

PAYLOAD = {
    "results": [
        {
            "id": 11,
            "name": "Anand Vihar",
            "coordinates": {"latitude": 28.65, "longitude": 77.31},
            "sensors": [{"parameter": {"name": "pm25"}}, {"parameter": {"name": "no2"}}],
            "datetimeFirst": {"utc": "2020-01-01T00:00:00Z"},
            "datetimeLast": {"utc": "2026-01-01T00:00:00Z"},
        },
        {"id": 12, "name": "No coords", "coordinates": None, "sensors": []},
    ]
}


def test_parse_nfhs_value_conventions():
    assert parse_nfhs_value("98.5") == (98.5, False)
    assert parse_nfhs_value("(75.2)") == (75.2, True)
    assert np.isnan(parse_nfhs_value("*")[0])
    assert np.isnan(parse_nfhs_value("na")[0])
    assert parse_nfhs_value("1,234") == (1234.0, False)
    assert np.isnan(parse_nfhs_value(None)[0])


def test_standardise_nfhs_flags_and_validates():
    raw = pd.DataFrame(
        {
            "Dist": ["A", "B", "C"],
            "St": ["X"] * 3,
            "Elec": ["99", "(80.5)", "*"],
            "HH": ["500", "40", "x"],
        }
    )
    out = standardise_nfhs(
        raw,
        {"Dist": "district", "St": "state", "Elec": "hh_electricity", "HH": "households_surveyed"},
        ("hh_electricity",),
    )
    assert out["hh_electricity_low_n"].tolist() == [False, True, False]
    assert np.isnan(out.loc[2, "hh_electricity"]) and out.loc[1, "households_surveyed"] == 40
    with pytest.raises(ValueError):
        standardise_nfhs(raw, {"Dist": "district", "St": "state"}, ("hh_electricity",))
    bad = raw.assign(Elec=["150", "1", "2"])
    with pytest.raises(ValueError):
        standardise_nfhs(
            bad, {"Dist": "district", "St": "state", "Elec": "hh_electricity"}, ("hh_electricity",)
        )


def test_parse_locations_and_missing_coords():
    df = parse_locations(PAYLOAD)
    assert df.loc[0, "station_id"] == "openaq_11" and df.loc[0, "has_pm25"]
    assert df.loc[0, "parameters"] == "no2,pm25"
    assert pd.isna(df.loc[1, "lat"])


def test_client_paginates_retries_and_hides_key(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _s: None)
    calls = []

    def fake_fetch(url, headers, timeout):
        calls.append((url, headers))
        if len(calls) == 1:
            raise OSError("transient")
        return {"results": PAYLOAD["results"][:1]} if "page=1" in url else {"results": []}

    client = OpenAQClient(api_key="secret-key", fetch=fake_fetch)
    df = client.get_locations(countries_id=9, limit=1)
    assert len(df) == 1 and len(calls) == 3
    assert all("secret-key" not in url for url, _ in calls)


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAQ_API_KEY", raising=False)
    with pytest.raises(ValueError):
        OpenAQClient()


def test_stations_validation_catches_swapped_latlon_and_reports_all():
    df = pd.DataFrame({"station_id": ["a", "b"], "lat": [77.2, 28.6], "lon": [28.6, 77.2]})
    with pytest.raises(ValueError, match="failed validation"):
        standardise_stations(df)


def test_stations_dedupe_rename_and_csv(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame(
        {"id": ["a", "a", "b"], "latitude": [28.6, 28.6, 19.0], "longitude": [77.2, 77.2, 72.8]}
    ).to_csv(p, index=False)
    with pytest.warns(UserWarning, match="duplicate"):
        out = load_stations_csv(
            p, {"id": "station_id", "latitude": "lat", "longitude": "lon"}, source="cpcb"
        )
    assert len(out) == 2 and (out["source"] == "cpcb").all()
    with pytest.raises(ValueError, match="Missing columns"):
        standardise_stations(pd.DataFrame({"x": [1]}))


def test_reference_schema_rejects_nonpositive_area():
    ref = pd.DataFrame(
        {
            "district_id": ["D1"],
            "district": ["A"],
            "state": ["S"],
            "lat": [20.0],
            "lon": [75.0],
            "area_km2": [0.0],
            "population": [1000.0],
        }
    )
    with pytest.raises(ValueError, match="reference"):
        validate(ref, ReferenceDistrictSchema, "reference districts")
