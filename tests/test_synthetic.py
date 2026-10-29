import pytest

from cfa.data.synthetic import add_synthetic_predictions, generate_synthetic_data


def test_shapes_and_unique_ids():
    districts, stations = generate_synthetic_data(120, seed=1)
    assert len(districts) == 120
    assert districts["district_id"].is_unique
    assert stations["station_id"].is_unique
    assert set(stations["district_id"]).issubset(set(districts["district_id"]))


def test_reproducible_with_same_seed():
    a, sa = generate_synthetic_data(80, seed=5)
    b, sb = generate_synthetic_data(80, seed=5)
    assert a.equals(b)
    assert sa.equals(sb)


def test_urban_districts_have_more_stations_than_rural():
    districts, stations = generate_synthetic_data(400, seed=3)
    counts = stations.groupby("district_id").size()
    districts["n_st"] = districts["district_id"].map(counts).fillna(0)
    means = districts.groupby("settlement")["n_st"].mean()
    assert means["urban"] > means["rural"]


def test_too_few_districts_rejected():
    with pytest.raises(ValueError):
        generate_synthetic_data(5)


def test_predictions_require_dvi():
    districts, _ = generate_synthetic_data(50)
    with pytest.raises(ValueError):
        add_synthetic_predictions(districts)
