import numpy as np
import pytest

pytest.importorskip("fairlearn")

from cfa.fairness.audit import group_metrics  # noqa: E402
from cfa.fairness.fairlearn_adapter import fairlearn_metric_frame, fairlearn_summary  # noqa: E402
from tests.test_audit import _frame  # noqa: E402


def test_metric_frame_matches_native_audit():
    df = _frame()
    frame = fairlearn_metric_frame(df, "g")
    native = group_metrics(df, "g", n_boot=50)
    for grp in ("low", "high"):
        assert frame.by_group.loc[grp, "rmse"] == pytest.approx(native.loc[grp, "rmse"])
        assert frame.by_group.loc[grp, "bias"] == pytest.approx(native.loc[grp, "bias"])


def test_summary_has_difference_and_ratio():
    summary = fairlearn_summary(fairlearn_metric_frame(_frame(), "g"))
    assert summary["max_difference"]["rmse"] > 0
    assert 0 < summary["ratio"]["rmse"] < 1
    assert np.isfinite(summary["max_difference"]["mae"])
