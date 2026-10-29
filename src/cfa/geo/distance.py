"""Great-circle distance helpers (pure numpy, no geopandas needed)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0088
_MAX_MATRIX_CELLS = 5_000_000  # caps memory of the pairwise distance block


def _validate_coords(lat: np.ndarray, lon: np.ndarray, name: str) -> None:
    finite = np.isfinite(lat) & np.isfinite(lon)
    if np.any(np.abs(lat[finite]) > 90) or np.any(np.abs(lon[finite]) > 180):
        raise ValueError(f"{name}: latitude must be in [-90, 90] and longitude in [-180, 180].")


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Inputs are degrees and broadcast like numpy arrays."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def nearest_distance_km(point_lat, point_lon, target_lat, target_lon) -> np.ndarray:
    """Distance from each point to its nearest target (e.g. district centroid -> nearest station).

    Returns NaN for every point when there are no targets. Memory use is bounded by chunking.
    """
    p_lat = np.asarray(point_lat, dtype=float)
    p_lon = np.asarray(point_lon, dtype=float)
    t_lat = np.asarray(target_lat, dtype=float)
    t_lon = np.asarray(target_lon, dtype=float)
    if p_lat.shape != p_lon.shape or t_lat.shape != t_lon.shape:
        raise ValueError("Latitude and longitude arrays must have the same shape.")
    _validate_coords(p_lat, p_lon, "points")
    _validate_coords(t_lat, t_lon, "targets")

    # Drop targets with missing coordinates.
    keep = np.isfinite(t_lat) & np.isfinite(t_lon)
    t_lat, t_lon = t_lat[keep], t_lon[keep]
    if t_lat.size == 0:
        return np.full(p_lat.shape, np.nan)

    out = np.empty(p_lat.shape, dtype=float)
    chunk = max(1, _MAX_MATRIX_CELLS // t_lat.size)
    for start in range(0, p_lat.size, chunk):
        sl = slice(start, start + chunk)
        dist = haversine_km(p_lat[sl, None], p_lon[sl, None], t_lat[None, :], t_lon[None, :])
        out[sl] = dist.min(axis=1)
    return out
