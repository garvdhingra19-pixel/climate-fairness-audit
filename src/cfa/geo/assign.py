"""Assign stations to districts: exact point-in-polygon (preferred) or nearest-centroid fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cfa.geo.distance import haversine_km


def assign_by_polygon(
    stations: pd.DataFrame, districts_gdf, id_col: str = "district_id"
) -> pd.DataFrame:
    """Point-in-polygon join (needs geopandas). Stations outside every polygon get NaN."""
    import geopandas as gpd

    if districts_gdf.crs is None:
        raise ValueError("districts_gdf needs a CRS (use EPSG:4326 for lat/lon).")
    pts = gpd.GeoDataFrame(
        stations.copy(),
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    ).to_crs(districts_gdf.crs)
    joined = gpd.sjoin(pts, districts_gdf[[id_col, "geometry"]], how="left", predicate="within")
    # Overlapping polygons can duplicate a point; keep the first so rows stay one-per-station.
    joined = joined[~joined.index.duplicated(keep="first")]
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["assignment_method"] = "polygon"
    return out


def assign_by_nearest_centroid(
    stations: pd.DataFrame, districts: pd.DataFrame, max_km: float = 75.0
) -> pd.DataFrame:
    """APPROXIMATE fallback: nearest centroid within max_km, else NaN (flagged in output)."""
    out = stations.copy()
    ids = np.full(len(out), None, dtype=object)
    for i, (la, lo) in enumerate(zip(out["lat"], out["lon"], strict=True)):
        d = haversine_km(la, lo, districts["lat"].to_numpy(), districts["lon"].to_numpy())
        j = int(np.argmin(d))
        if d[j] <= max_km:
            ids[i] = districts["district_id"].iloc[j]
    out["district_id"] = ids
    out["assignment_method"] = "nearest_centroid_approx"
    return out
