"""Optional Fairlearn integration (install with: pip install 'climate-fairness-audit[fairness]').

Fairlearn's MetricFrame accepts any metric f(y_true, y_pred), so it works for regression too
(group-wise RMSE/MAE/bias). Note that many Fairlearn/AIF360 *fairness definitions* (demographic
parity, equalized odds) are classification-only; for regression we report group-wise error gaps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def _bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def fairlearn_metric_frame(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "pm25_true",
    y_pred_col: str = "pm25_pred",
):
    """Build a fairlearn MetricFrame of rmse / mae / bias by group."""
    try:
        from fairlearn.metrics import MetricFrame
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Install Fairlearn: pip install 'climate-fairness-audit[fairness]'"
        ) from exc
    work = df.dropna(subset=[group_col, y_true_col, y_pred_col])
    return MetricFrame(
        metrics={"rmse": _rmse, "mae": _mae, "bias": _bias},
        y_true=work[y_true_col],
        y_pred=work[y_pred_col],
        sensitive_features=work[group_col],
    )


def fairlearn_summary(frame) -> dict[str, object]:
    """By-group table plus the largest between-group difference and ratio for each metric."""
    return {
        "by_group": frame.by_group,
        "max_difference": frame.difference(method="between_groups"),
        "ratio": frame.ratio(method="between_groups"),
    }
