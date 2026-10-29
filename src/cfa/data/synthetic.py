"""Synthetic district + station data so the whole pipeline runs without any downloads.

The generator deliberately mimics the real-world pattern this project studies: monitoring
stations, survey samples and data completeness are concentrated in urban, populous districts.

EVERYTHING HERE IS SYNTHETIC. Never report numbers produced from it as real findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_data(
    n_districts: int = 300, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (districts, stations) DataFrames.

    districts columns: district_id, region, lat, lon, area_km2, population, urban_share,
        wealth_index, settlement, wealth_quintile, survey_n, completeness, cloud_free_frac
    stations columns: station_id, district_id, lat, lon, completeness
    """
    if n_districts < 20:
        raise ValueError("n_districts must be at least 20.")
    rng = np.random.default_rng(seed)
    n = n_districts

    lat = rng.uniform(8.5, 33.0, n)
    lon = rng.uniform(69.0, 96.0, n)
    urban_share = rng.beta(1.4, 3.2, n)
    population = rng.lognormal(mean=14.3, sigma=0.6, size=n) * (0.7 + 0.9 * urban_share)
    area_km2 = rng.lognormal(mean=8.3, sigma=0.7, size=n)
    wealth_index = np.clip(0.15 + 0.55 * urban_share + rng.normal(0, 0.15, n), 0, 1)

    # Stations concentrate where people and urban share are high.
    expected_stations = (population / 1e6) * (3.5 * urban_share) ** 2 * 0.45
    n_stations = rng.poisson(expected_stations)

    district_ids = np.array([f"D{i:04d}" for i in range(n)])
    rows = []
    for i in range(n):
        for _ in range(int(n_stations[i])):
            rows.append(
                {
                    "district_id": district_ids[i],
                    "lat": float(np.clip(lat[i] + rng.normal(0, 0.15), -90, 90)),
                    "lon": float(np.clip(lon[i] + rng.normal(0, 0.15), -180, 180)),
                    "completeness": float(rng.beta(4 + 6 * urban_share[i], 3)),
                }
            )
    stations = pd.DataFrame(rows, columns=["district_id", "lat", "lon", "completeness"])
    stations.insert(0, "station_id", [f"S{i:05d}" for i in range(len(stations))])

    mean_completeness = stations.groupby("district_id")["completeness"].mean()
    survey_n = rng.lognormal(mean=6.6 + 0.6 * urban_share, sigma=0.4, size=n).round()
    cloud_free = np.clip(rng.normal(0.7 - 0.2 * (lon - 69.0) / 27.0, 0.08), 0.2, 0.95)

    districts = pd.DataFrame(
        {
            "district_id": district_ids,
            "region": [
                f"R{int((la - 8.5) // 5):d}{int((lo - 69.0) // 9):d}"
                for la, lo in zip(lat, lon, strict=True)
            ],
            "lat": lat,
            "lon": lon,
            "area_km2": area_km2,
            "population": population,
            "urban_share": urban_share,
            "wealth_index": wealth_index,
            "survey_n": survey_n,
            "completeness": pd.Series(district_ids).map(mean_completeness).to_numpy(),
            "cloud_free_frac": cloud_free,
        }
    )
    districts["settlement"] = pd.cut(
        districts["urban_share"],
        bins=[0, 0.2, 0.45, 1.0],
        labels=["rural", "mixed", "urban"],
        include_lowest=True,
    ).astype(str)
    districts["wealth_quintile"] = pd.qcut(
        districts["wealth_index"].rank(method="first"),
        5,
        labels=["W1_poorest", "W2", "W3", "W4", "W5_richest"],
    ).astype(str)
    return districts, stations


def add_synthetic_predictions(districts: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Add pm25_true and pm25_pred columns. Needs a 'dvi' column (0-100) and 'urban_share'.

    Prediction error is built to grow (and bias downward) as visibility falls, so the audit
    code has a known effect to detect. This is a test fixture, not a scientific result.
    """
    if not {"dvi", "urban_share", "lat"}.issubset(districts.columns):
        raise ValueError("districts must contain 'dvi', 'urban_share' and 'lat'.")
    rng = np.random.default_rng(seed)
    invisibility = 1.0 - districts["dvi"].to_numpy() / 100.0
    truth = np.clip(
        30
        + 40 * districts["urban_share"].to_numpy()
        + 20 * (districts["lat"].to_numpy() > 24)
        + rng.normal(0, 10, len(districts)),
        5,
        300,
    )
    noise_sd = 4 + 14 * invisibility
    bias = -8 * invisibility
    out = districts.copy()
    out["pm25_true"] = truth
    out["pm25_pred"] = truth + bias + rng.normal(0, noise_sd)
    return out
