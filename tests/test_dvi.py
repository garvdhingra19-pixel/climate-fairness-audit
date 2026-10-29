import numpy as np
import pandas as pd
import pytest

from cfa.data.synthetic import generate_synthetic_data
from cfa.visibility.dvi import (
    DEFAULT_COMPONENTS,
    assign_quintiles,
    compute_dvi,
    compute_visibility_inputs,
    normalise_component,
    sensitivity_analysis,
)


@pytest.fixture(scope="module")
def synthetic():
    districts, stations = generate_synthetic_data(200, seed=11)
    inputs = compute_visibility_inputs(districts, stations)
    return districts, stations, inputs, compute_dvi(inputs)


def test_normalise_bounds_and_direction():
    s = pd.Series(np.arange(100, dtype=float))
    up = normalise_component(s, higher_is_better=True)
    down = normalise_component(s, higher_is_better=False)
    assert up.min() == 0.0 and up.max() == 1.0
    np.testing.assert_allclose(up + down, 1.0)


def test_normalise_constant_column_is_uninformative():
    out = normalise_component(pd.Series([3.0, 3.0, 3.0]))
    assert (out == 0.5).all()


def test_normalise_handles_nan_and_inf():
    out = normalise_component(pd.Series([1.0, np.nan, np.inf, 5.0, 9.0]))
    assert np.isnan(out.iloc[1]) and np.isnan(out.iloc[2])


def test_log_rejects_negative():
    with pytest.raises(ValueError):
        normalise_component(pd.Series([-1.0, 2.0]), log=True)


def test_dvi_in_range(synthetic):
    dvi = synthetic[3]["dvi"].dropna()
    assert dvi.between(0, 100).all()


def test_more_stations_means_higher_dvi(synthetic):
    districts, stations, inputs, _ = synthetic
    target = inputs.loc[inputs["stations_per_1000km2"] == 0, "district_id"].iloc[0]
    row = districts[districts["district_id"] == target].iloc[0]
    extra = pd.DataFrame(
        {
            "station_id": [f"X{i}" for i in range(5)],
            "district_id": target,
            "lat": row["lat"],
            "lon": row["lon"],
            "completeness": 0.95,
        }
    )
    boosted = compute_visibility_inputs(districts, pd.concat([stations, extra], ignore_index=True))
    before = compute_dvi(inputs).set_index("district_id").loc[target, "dvi"]
    after = compute_dvi(boosted).set_index("district_id").loc[target, "dvi"]
    assert after > before


def test_missing_component_is_skipped_not_penalised(synthetic):
    _, _, inputs, _ = synthetic
    before = compute_dvi(inputs).loc[0, "n_components_used"]
    altered = inputs.copy()
    altered.loc[0, "survey_n"] = np.nan
    res = compute_dvi(altered)
    assert res.loc[0, "n_components_used"] == before - 1
    assert 0 <= res.loc[0, "dvi"] <= 100


def test_invalid_weights_rejected(synthetic):
    inputs = synthetic[2]
    names = list(DEFAULT_COMPONENTS)
    with pytest.raises(ValueError):
        compute_dvi(inputs, weights=dict.fromkeys(names, -1.0))
    with pytest.raises(ValueError):
        compute_dvi(inputs, weights={"only_one": 1.0})
    with pytest.raises(ValueError):
        compute_dvi(inputs, weights=dict.fromkeys(names, 0.0))


def test_weights_are_normalised(synthetic):
    inputs = synthetic[2]
    names = list(DEFAULT_COMPONENTS)
    a = compute_dvi(inputs, weights=dict.fromkeys(names, 1.0))["dvi"]
    b = compute_dvi(inputs, weights=dict.fromkeys(names, 7.0))["dvi"]
    np.testing.assert_allclose(a, b)


def test_quintiles_are_balanced(synthetic):
    q = assign_quintiles(synthetic[3]["dvi"])
    counts = q.value_counts()
    assert len(counts) == 5
    assert counts.max() - counts.min() <= 1


def test_quintile_order_follows_dvi(synthetic):
    scores = synthetic[3].copy()
    scores["q"] = assign_quintiles(scores["dvi"]).to_numpy()
    means = scores.groupby("q")["dvi"].mean()
    assert means["Q1_least_visible"] < means["Q3"] < means["Q5_most_visible"]


def test_missing_input_columns_raise(synthetic):
    districts, stations, _, _ = synthetic
    with pytest.raises(ValueError):
        compute_visibility_inputs(districts.drop(columns=["area_km2"]), stations)
    with pytest.raises(ValueError):
        compute_visibility_inputs(districts, stations.drop(columns=["lat"]))


def test_duplicate_district_ids_raise(synthetic):
    districts, stations, _, _ = synthetic
    with pytest.raises(ValueError):
        compute_visibility_inputs(pd.concat([districts, districts.iloc[:1]]), stations)


def test_sensitivity_is_stable_and_reproducible(synthetic):
    scores = synthetic[3]
    a = sensitivity_analysis(scores, n_draws=60, seed=2)
    b = sensitivity_analysis(scores, n_draws=60, seed=2)
    assert a.summary == b.summary
    assert a.summary["spearman_mean"] > 0.9
    assert a.district_stability.between(0, 1).all()


def test_sensitivity_rejects_bad_params(synthetic):
    with pytest.raises(ValueError):
        sensitivity_analysis(synthetic[3], n_draws=0)
