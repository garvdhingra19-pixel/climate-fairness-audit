import numpy as np
import pandas as pd
import pytest

from cfa.fairness.audit import error_by_bin, gap_with_ci, group_metrics, regression_metrics


def _frame(seed=0, n=400):
    rng = np.random.default_rng(seed)
    truth = rng.normal(50, 10, 2 * n)
    group = np.array(["low"] * n + ["high"] * n)
    noise = np.where(group == "low", 15.0, 4.0)
    return pd.DataFrame({"g": group, "pm25_true": truth, "pm25_pred": truth + rng.normal(0, noise)})


def test_regression_metrics_exact():
    m = regression_metrics([0, 0, 0, 0], [1, -1, 1, -1])
    assert m["rmse"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(1.0)
    assert m["bias"] == pytest.approx(0.0)
    assert m["n"] == 4


def test_regression_metrics_sign_convention_under_prediction_is_negative():
    assert regression_metrics([10, 10], [8, 8])["bias"] == pytest.approx(-2.0)


def test_regression_metrics_all_nan():
    assert regression_metrics([np.nan], [1.0])["n"] == 0


def test_group_metrics_ci_contains_estimate_and_orders_groups():
    res = group_metrics(_frame(), "g", n_boot=200, order=["low", "high"])
    assert list(res.index) == ["low", "high"]
    assert (res["rmse_lo"] <= res["rmse"]).all() and (res["rmse"] <= res["rmse_hi"]).all()
    assert res.loc["low", "rmse"] > res.loc["high", "rmse"]


def test_group_metrics_flags_small_groups():
    df = _frame()
    small = pd.concat(
        [df, pd.DataFrame({"g": ["tiny"] * 3, "pm25_true": [1.0] * 3, "pm25_pred": [2.0] * 3})]
    )
    res = group_metrics(small, "g", n_boot=100, min_n=10)
    assert not res.loc["tiny", "reliable"]
    assert res.loc["low", "reliable"]


def test_group_metrics_validates_inputs():
    with pytest.raises(ValueError):
        group_metrics(_frame(), "missing_col")
    with pytest.raises(ValueError):
        group_metrics(_frame(), "g", n_boot=5)


def test_gap_detects_real_difference():
    gap = gap_with_ci(_frame(), "g", "low", "high", n_boot=400)
    assert gap["gap"] > 0 and gap["significant"]
    assert gap["ratio"] > 1.5
    assert gap["gap_lo"] <= gap["gap"] <= gap["gap_hi"]


def test_gap_not_significant_when_groups_identical():
    rng = np.random.default_rng(1)
    n = 300
    truth = rng.normal(50, 10, 2 * n)
    df = pd.DataFrame(
        {
            "g": ["a"] * n + ["b"] * n,
            "pm25_true": truth,
            "pm25_pred": truth + rng.normal(0, 5, 2 * n),
        }
    )
    assert not gap_with_ci(df, "g", "a", "b", n_boot=400)["significant"]


def test_gap_rejects_bad_metric_and_tiny_group():
    with pytest.raises(ValueError):
        gap_with_ci(_frame(), "g", "low", "high", metric="nope")
    with pytest.raises(ValueError):
        gap_with_ci(_frame(), "g", "low", "absent")


def test_error_by_bin_shape():
    df = _frame().assign(x=np.arange(800, dtype=float))
    out = error_by_bin(df, "x", n_bins=8)
    assert len(out) == 8
    assert out["n"].sum() == 800
