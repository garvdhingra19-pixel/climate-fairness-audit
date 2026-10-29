import numpy as np
import pytest

from cfa.geo.distance import haversine_km, nearest_distance_km


def test_haversine_known_distance_delhi_mumbai():
    d = haversine_km(28.6139, 77.2090, 19.0760, 72.8777)
    assert 1140 < float(d) < 1160


def test_haversine_zero_for_same_point():
    assert float(haversine_km(10, 20, 10, 20)) == pytest.approx(0.0, abs=1e-9)


def test_nearest_distance_picks_closest_target():
    d = nearest_distance_km([0.0], [0.0], [0.0, 10.0], [1.0, 10.0])
    assert d[0] == pytest.approx(float(haversine_km(0, 0, 0, 1)))


def test_nearest_distance_no_targets_gives_nan():
    d = nearest_distance_km([1.0, 2.0], [1.0, 2.0], [], [])
    assert np.isnan(d).all()


def test_nearest_distance_ignores_nan_targets():
    d = nearest_distance_km([0.0], [0.0], [np.nan, 0.0], [np.nan, 2.0])
    assert np.isfinite(d[0])


def test_invalid_coordinates_raise():
    with pytest.raises(ValueError):
        nearest_distance_km([95.0], [0.0], [0.0], [0.0])


def test_chunking_matches_unchunked(monkeypatch):
    rng = np.random.default_rng(0)
    p_lat, p_lon = rng.uniform(8, 33, 50), rng.uniform(69, 96, 50)
    t_lat, t_lon = rng.uniform(8, 33, 30), rng.uniform(69, 96, 30)
    full = nearest_distance_km(p_lat, p_lon, t_lat, t_lon)
    monkeypatch.setattr("cfa.geo.distance._MAX_MATRIX_CELLS", 60)  # force tiny chunks
    chunked = nearest_distance_km(p_lat, p_lon, t_lat, t_lon)
    np.testing.assert_allclose(full, chunked)
