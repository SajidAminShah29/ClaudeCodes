import numpy as np
import pytest

from resim.grid import build_cartesian
from resim.props import build_properties
from resim.props.geostat import covariance, gaussian_random_field, transform, truncate


def _correlation(field, lag, axis):
    a = np.moveaxis(field, axis, 0)
    return np.corrcoef(a[:-lag].ravel(), a[lag:].ravel())[0, 1]


def test_field_statistics_and_correlation_match_variogram():
    shape, h, r = (200, 200, 1), (1.0, 1.0, 1.0), 20.0
    fields = [gaussian_random_field(shape, h, "exponential", (r, r, 1.0), seed=s) for s in range(4)]
    y = np.stack(fields)
    assert abs(y.mean()) < 0.1
    assert abs(y.var() - 1.0) < 0.15
    for lag in (2, 5, 10):
        expected = covariance(np.array(lag / r), "exponential")
        measured = np.mean([_correlation(f, lag, 0) for f in fields])
        assert abs(measured - expected) < 0.08


def test_anisotropy_and_azimuth():
    shape, h = (150, 150, 1), (1.0, 1.0, 1.0)
    # azimuth 0 -> major range along +y
    y = gaussian_random_field(shape, h, "spherical", (60.0, 6.0, 1.0), azimuth=0.0, seed=1)
    assert _correlation(y, 5, 1) > _correlation(y, 5, 0) + 0.3
    y90 = gaussian_random_field(shape, h, "spherical", (60.0, 6.0, 1.0), azimuth=90.0, seed=1)
    assert _correlation(y90, 5, 0) > _correlation(y90, 5, 1) + 0.3


def test_reproducible_with_seed():
    a = gaussian_random_field((10, 10, 3), (1, 1, 1), ranges=5.0, seed=7)
    b = gaussian_random_field((10, 10, 3), (1, 1, 1), ranges=5.0, seed=7)
    assert np.array_equal(a, b)


def test_lognormal_transform_moments():
    y = np.random.default_rng(0).standard_normal(400_000)
    k = transform(y, 100.0, 50.0, "lognormal")
    assert abs(k.mean() - 100) < 1 and abs(k.std() - 50) < 1.5


def test_truncation_proportions():
    y = np.random.default_rng(0).standard_normal(200_000)
    codes = truncate(y, [0.6, 0.3, 0.1])
    counts = np.bincount(codes)[1:] / y.size
    assert np.allclose(counts, [0.6, 0.3, 0.1], atol=0.01)
    with pytest.raises(ValueError):
        truncate(y, [0.5, 0.2])


def test_property_sources_end_to_end():
    g = build_cartesian((30, 30, 5), 10.0, 10.0, 2.0)
    spec = {
        "SATNUM": {"source": "truncated_gaussian", "proportions": [0.7, 0.3], "variogram": {"range": [100, 50, 4]}, "seed": 3},
        "PORO": {"source": "gaussian_field", "mean": 0.2, "std": 0.05, "clip": [0.01, 0.4], "variogram": {"type": "exponential", "range": [120, 60, 6], "azimuth": 45}, "seed": 4},
        "PERMX": {"source": "poro_perm", "from": "PORO", "by": "SATNUM", "params": {1: {"a": 12, "b": -0.5}, 2: {"a": 8, "b": -1}}},
    }
    p = build_properties(g, spec)
    assert set(np.unique(p["SATNUM"])) == {1, 2}
    assert p["PORO"].min() >= 0.01 and p["PORO"].max() <= 0.4
    assert np.all(p["PERMX"] > 0)
