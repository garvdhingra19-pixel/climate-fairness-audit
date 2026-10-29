"""Group-wise error audit with bootstrap confidence intervals (numpy only).

Sign convention: error = prediction - truth, so a negative bias means under-prediction.
Every headline gap should be reported with its confidence interval, never as a bare number.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

METRICS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "rmse": lambda e: np.sqrt(np.mean(e**2, axis=-1)),
    "mae": lambda e: np.mean(np.abs(e), axis=-1),
    "bias": lambda e: np.mean(e, axis=-1),
}


def _errors(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> np.ndarray:
    for col in (y_true_col, y_pred_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found.")
    err = (df[y_pred_col] - df[y_true_col]).to_numpy(dtype=float)
    return err[np.isfinite(err)]


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """n, rmse, mae and bias for one set of predictions."""
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan}
    return {"n": int(err.size), **{k: float(f(err)) for k, f in METRICS.items()}}


def _bootstrap(err: np.ndarray, metric: str, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap replicates of one metric, resampled in batches to bound memory."""
    n = err.size
    fn = METRICS[metric]
    batch = max(1, 2_000_000 // n)
    out = np.empty(n_boot)
    for start in range(0, n_boot, batch):
        size = min(batch, n_boot - start)
        out[start : start + size] = fn(err[rng.integers(0, n, size=(size, n))])
    return out


def group_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "pm25_true",
    y_pred_col: str = "pm25_pred",
    n_boot: int = 500,
    alpha: float = 0.05,
    min_n: int = 10,
    seed: int = 0,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Per-group n, rmse, mae, bias, each with a percentile bootstrap CI.

    Groups with fewer than min_n rows are kept but flagged reliable=False.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found.")
    if not 0 < alpha < 1 or n_boot < 50:
        raise ValueError("alpha must be in (0, 1) and n_boot must be >= 50.")
    rng = np.random.default_rng(seed)
    rows = []
    for name, sub in df.dropna(subset=[group_col]).groupby(group_col, observed=True):
        err = _errors(sub, y_true_col, y_pred_col)
        if err.size == 0:
            continue
        row: dict[str, object] = {
            group_col: name,
            "n": int(err.size),
            "reliable": err.size >= min_n,
        }
        for metric, fn in METRICS.items():
            reps = _bootstrap(err, metric, n_boot, rng)
            row[metric] = float(fn(err))
            row[f"{metric}_lo"] = float(np.quantile(reps, alpha / 2))
            row[f"{metric}_hi"] = float(np.quantile(reps, 1 - alpha / 2))
        rows.append(row)
    result = pd.DataFrame(rows).set_index(group_col)
    if order is not None:
        result = result.reindex([g for g in order if g in result.index])
    return result


def gap_with_ci(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    y_true_col: str = "pm25_true",
    y_pred_col: str = "pm25_pred",
    metric: str = "rmse",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, object]:
    """Gap = metric(group_a) - metric(group_b) with a bootstrap CI.

    Typical use: group_a = least visible quintile, group_b = most visible quintile.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    rng = np.random.default_rng(seed)
    errs = {}
    for g in (group_a, group_b):
        errs[g] = _errors(df[df[group_col] == g], y_true_col, y_pred_col)
        if errs[g].size < 2:
            raise ValueError(f"Group '{g}' has fewer than 2 usable rows.")
    fn = METRICS[metric]
    a_obs, b_obs = float(fn(errs[group_a])), float(fn(errs[group_b]))
    a_rep = _bootstrap(errs[group_a], metric, n_boot, rng)
    b_rep = _bootstrap(errs[group_b], metric, n_boot, rng)
    gap_rep = a_rep - b_rep
    result: dict[str, object] = {
        "metric": metric,
        "group_a": group_a,
        "group_b": group_b,
        "value_a": a_obs,
        "value_b": b_obs,
        "gap": a_obs - b_obs,
        "gap_lo": float(np.quantile(gap_rep, alpha / 2)),
        "gap_hi": float(np.quantile(gap_rep, 1 - alpha / 2)),
    }
    result["significant"] = bool(result["gap_lo"] > 0 or result["gap_hi"] < 0)
    if metric in ("rmse", "mae") and b_obs > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_rep = a_rep / b_rep
        result["ratio"] = a_obs / b_obs
        result["ratio_lo"] = float(np.nanquantile(ratio_rep, alpha / 2))
        result["ratio_hi"] = float(np.nanquantile(ratio_rep, 1 - alpha / 2))
    return result


def error_by_bin(
    df: pd.DataFrame,
    x_col: str = "dvi",
    y_true_col: str = "pm25_true",
    y_pred_col: str = "pm25_pred",
    n_bins: int = 10,
) -> pd.DataFrame:
    """Dose-response table: error metrics in equal-count bins of a continuous variable."""
    if x_col not in df.columns:
        raise ValueError(f"Column '{x_col}' not found.")
    work = df[[x_col, y_true_col, y_pred_col]].dropna().copy()
    work["bin"] = pd.qcut(work[x_col].rank(method="first"), n_bins, labels=False)
    rows = []
    for b, sub in work.groupby("bin"):
        m = regression_metrics(sub[y_true_col], sub[y_pred_col])
        rows.append({"bin": int(b), "x_mean": float(sub[x_col].mean()), **m})
    return pd.DataFrame(rows)
