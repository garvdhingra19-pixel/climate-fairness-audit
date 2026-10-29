"""Data Visibility Index (DVI): how well is each district 'seen' by climate data?

Each component is normalised to [0, 1] (1 = most visible) and combined as a weighted mean,
then scaled to 0-100. Missing components are skipped and the remaining weights renormalised,
so a district is never silently rewarded or punished for a gap in one source.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from cfa.config import QUINTILE_LABELS
from cfa.geo.distance import nearest_distance_km


@dataclass(frozen=True)
class ComponentSpec:
    """How one visibility component is built from a raw column."""

    column: str
    weight: float
    higher_is_better: bool = True
    log: bool = False


DEFAULT_COMPONENTS: dict[str, ComponentSpec] = {
    "station_density_area": ComponentSpec("stations_per_1000km2", 0.20, True, True),
    "station_density_pop": ComponentSpec("stations_per_100k", 0.20, True, True),
    "distance_to_station": ComponentSpec("dist_nearest_km", 0.20, False, True),
    "survey_sample": ComponentSpec("survey_n", 0.15, True, True),
    "completeness": ComponentSpec("completeness", 0.15, True, False),
    "satellite_quality": ComponentSpec("cloud_free_frac", 0.10, True, False),
}


def compute_visibility_inputs(districts: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Turn raw district + station tables into the raw DVI input columns."""
    need_d = {
        "district_id",
        "lat",
        "lon",
        "area_km2",
        "population",
        "survey_n",
        "completeness",
        "cloud_free_frac",
    }
    need_s = {"district_id", "lat", "lon"}
    if missing := need_d - set(districts.columns):
        raise ValueError(f"districts is missing columns: {sorted(missing)}")
    if missing := need_s - set(stations.columns):
        raise ValueError(f"stations is missing columns: {sorted(missing)}")
    if districts["district_id"].duplicated().any():
        raise ValueError("districts has duplicate district_id values.")
    if (districts["area_km2"] <= 0).any() or (districts["population"] <= 0).any():
        raise ValueError("area_km2 and population must be strictly positive.")

    counts = stations.groupby("district_id").size()
    n_st = districts["district_id"].map(counts).fillna(0).astype(float)
    return pd.DataFrame(
        {
            "district_id": districts["district_id"].to_numpy(),
            "stations_per_1000km2": (n_st / districts["area_km2"] * 1000).to_numpy(),
            "stations_per_100k": (n_st / districts["population"] * 1e5).to_numpy(),
            "dist_nearest_km": nearest_distance_km(
                districts["lat"], districts["lon"], stations["lat"], stations["lon"]
            ),
            "survey_n": districts["survey_n"].to_numpy(dtype=float),
            "completeness": districts["completeness"].to_numpy(dtype=float),
            "cloud_free_frac": districts["cloud_free_frac"].to_numpy(dtype=float),
        }
    )


def normalise_component(
    values: pd.Series | np.ndarray,
    higher_is_better: bool = True,
    log: bool = False,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> pd.Series:
    """Winsorise at percentiles, min-max scale to [0, 1], and flip if lower is better."""
    s = pd.Series(np.asarray(values, dtype=float)).replace([np.inf, -np.inf], np.nan)
    if log:
        if (s.dropna() < 0).any():
            raise ValueError("Cannot log-transform negative values.")
        s = np.log1p(s)
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    lo, hi = np.percentile(valid, [lo_pct, hi_pct])
    if hi <= lo:  # constant column: carries no information
        out = pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    else:
        out = ((s.clip(lo, hi) - lo) / (hi - lo)).astype(float)
    return out if higher_is_better else 1.0 - out


def _resolve_weights(
    weights: Mapping[str, float] | None, specs: Mapping[str, ComponentSpec]
) -> np.ndarray:
    if weights is None:
        w = np.array([specs[k].weight for k in specs], dtype=float)
    else:
        if set(weights) != set(specs):
            raise ValueError(f"weights must have exactly these keys: {sorted(specs)}")
        w = np.array([weights[k] for k in specs], dtype=float)
    if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
        raise ValueError("weights must be non-negative, finite, and not all zero.")
    return w / w.sum()


def _weighted_mean_skipna(scores: np.ndarray, w: np.ndarray) -> np.ndarray:
    mask = ~np.isnan(scores)
    num = np.where(mask, scores, 0.0) @ w
    den = mask @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def compute_dvi(
    inputs: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
    specs: Mapping[str, ComponentSpec] = DEFAULT_COMPONENTS,
) -> pd.DataFrame:
    """Return per-district component scores (score_*), the composite 'dvi' (0-100), and coverage."""
    scores = pd.DataFrame({"district_id": inputs["district_id"].to_numpy()})
    for name, spec in specs.items():
        if spec.column not in inputs.columns:
            raise ValueError(f"inputs is missing column '{spec.column}' for component '{name}'.")
        scores[f"score_{name}"] = normalise_component(
            inputs[spec.column], spec.higher_is_better, spec.log
        ).to_numpy()
    cols = [f"score_{k}" for k in specs]
    w = _resolve_weights(weights, specs)
    mat = scores[cols].to_numpy(dtype=float)
    scores["dvi"] = 100.0 * _weighted_mean_skipna(mat, w)
    scores["n_components_used"] = (~np.isnan(mat)).sum(axis=1)
    return scores


def _quintile_codes(values: np.ndarray) -> np.ndarray:
    """0..4 by rank (ties broken by order); -1 for NaN. Gives near-equal group sizes."""
    s = pd.Series(values)
    valid = s.notna()
    codes = np.full(len(s), -1, dtype=int)
    if valid.any():
        ranks = s[valid].rank(method="first").to_numpy()
        codes[valid.to_numpy()] = ((ranks - 1) * 5 // valid.sum()).astype(int)
    return codes


def assign_quintiles(
    dvi: pd.Series | np.ndarray, labels: tuple[str, ...] = QUINTILE_LABELS
) -> pd.Series:
    """Label each district Q1 (least visible) .. Q5 (most visible). NaN stays missing."""
    codes = _quintile_codes(np.asarray(dvi, dtype=float))
    lab = np.array(labels + (None,), dtype=object)
    return pd.Series(lab[np.where(codes >= 0, codes, 5)], dtype="object")


@dataclass
class SensitivityResult:
    summary: dict[str, float]
    district_stability: (
        pd.Series
    )  # share of weight draws keeping each district in its base quintile


def sensitivity_analysis(
    scores: pd.DataFrame,
    base_weights: Mapping[str, float] | None = None,
    specs: Mapping[str, ComponentSpec] = DEFAULT_COMPONENTS,
    n_draws: int = 500,
    concentration: float = 50.0,
    seed: int = 0,
) -> SensitivityResult:
    """How much do rankings change if the weights are perturbed?

    Weights are drawn from a Dirichlet centred on the base weights; a higher concentration
    means smaller perturbations. Reports rank correlation and quintile stability.
    """
    if n_draws < 1 or concentration <= 0:
        raise ValueError("n_draws must be >= 1 and concentration must be > 0.")
    cols = [f"score_{k}" for k in specs]
    mat = scores[cols].to_numpy(dtype=float)
    base_w = _resolve_weights(base_weights, specs)
    base = _weighted_mean_skipna(mat, base_w)
    base_codes = _quintile_codes(base)
    base_rank = pd.Series(base).rank()

    rng = np.random.default_rng(seed)
    alpha = np.maximum(base_w, 1e-3) * concentration
    draws = rng.dirichlet(alpha, size=n_draws)

    spearman, same, within_one = [], [], []
    stable = np.zeros(len(base))
    for w in draws:
        alt = _weighted_mean_skipna(mat, w)
        alt_codes = _quintile_codes(alt)
        ok = (base_codes >= 0) & (alt_codes >= 0)
        spearman.append(float(base_rank.corr(pd.Series(alt).rank())))
        same.append(float((base_codes[ok] == alt_codes[ok]).mean()))
        within_one.append(float((np.abs(base_codes[ok] - alt_codes[ok]) <= 1).mean()))
        stable += (base_codes == alt_codes) & ok

    summary = {
        "n_draws": float(n_draws),
        "concentration": float(concentration),
        "spearman_min": float(np.min(spearman)),
        "spearman_mean": float(np.mean(spearman)),
        "same_quintile_share_mean": float(np.mean(same)),
        "within_one_quintile_share_mean": float(np.mean(within_one)),
    }
    return SensitivityResult(summary, pd.Series(stable / n_draws, name="quintile_stability"))
