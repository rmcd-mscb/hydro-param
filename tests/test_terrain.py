"""Tests for terrain derivation (slope, aspect) with synthetic data.

Uses known-answer test cases: flat surfaces, uniformly tilted planes,
and cardinal-direction slopes.
"""

import numpy as np
import xarray as xr

from hydro_param.data_access import derive_aspect, derive_slope


def _make_elevation(
    data: np.ndarray,
    dx: float = 100.0,
    dy: float = 100.0,
) -> xr.DataArray:
    """Create an elevation DataArray with projected CRS coordinates.

    Parameters
    ----------
    data : np.ndarray
        2-D elevation values.
    dx, dy : float
        Cell size in meters.
    """
    ny, nx = data.shape
    x = np.arange(nx) * dx + dx / 2
    y = np.arange(ny) * dy + dy / 2
    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    return da


def test_flat_surface_slope_is_zero():
    """A flat surface (constant elevation) should have slope = 0."""
    data = np.full((10, 10), 500.0)
    elev = _make_elevation(data)
    slope = derive_slope(elev)
    np.testing.assert_allclose(slope.values, 0.0, atol=1e-10)


def test_flat_surface_aspect():
    """A flat surface has undefined aspect; we accept any value."""
    data = np.full((10, 10), 500.0)
    elev = _make_elevation(data)
    aspect = derive_aspect(elev)
    # Just check it's in valid range [0, 360)
    assert (aspect.values >= 0).all()
    assert (aspect.values < 360).all()


def test_uniform_slope_east():
    """Surface tilting east: elevation increases with x.

    If dz/dx = 1.0 and dz/dy = 0, slope = arctan(1) = 45°.
    """
    # 100m cells, elevation increases by 100m per cell eastward
    data = np.zeros((10, 10))
    for c in range(10):
        data[:, c] = c * 100.0  # 100m rise per 100m cell

    elev = _make_elevation(data, dx=100.0, dy=100.0)
    slope = derive_slope(elev)

    # Interior cells should have slope ≈ 45°
    interior = slope.values[2:-2, 2:-2]
    np.testing.assert_allclose(interior, 45.0, atol=1.0)


def test_uniform_slope_north():
    """Surface tilting north: elevation increases with y.

    dz/dy = 1.0, dz/dx = 0 → slope = 45°.
    """
    data = np.zeros((10, 10))
    for r in range(10):
        data[r, :] = r * 100.0  # 100m rise per 100m cell northward

    elev = _make_elevation(data, dx=100.0, dy=100.0)
    slope = derive_slope(elev)

    interior = slope.values[2:-2, 2:-2]
    np.testing.assert_allclose(interior, 45.0, atol=1.0)


def test_east_facing_aspect():
    """Surface tilting east → steepest descent is east → aspect = 90°.

    Aspect is measured clockwise from north. A surface that slopes
    downward to the east (elevation decreasing with increasing x)
    has aspect = 90°.
    """
    data = np.zeros((10, 10))
    for c in range(10):
        data[:, c] = (9 - c) * 100.0  # decreases eastward

    elev = _make_elevation(data, dx=100.0, dy=100.0)
    aspect = derive_aspect(elev)

    interior = aspect.values[2:-2, 2:-2]
    np.testing.assert_allclose(interior, 90.0, atol=1.0)


def test_north_facing_aspect():
    """Surface sloping downward to the north → aspect ≈ 0° (or 360°).

    Elevation decreases with increasing y (northward).
    """
    data = np.zeros((10, 10))
    for r in range(10):
        data[r, :] = (9 - r) * 100.0  # decreases northward

    elev = _make_elevation(data, dx=100.0, dy=100.0)
    aspect = derive_aspect(elev)

    interior = aspect.values[2:-2, 2:-2]
    # North-facing: aspect should be near 0 or 360
    # Normalize: values near 360 are equivalent to 0
    normalized = interior % 360
    close_to_north = (normalized < 1.0) | (normalized > 359.0)
    assert close_to_north.all()


def test_slope_output_shape_matches_input():
    data = np.random.default_rng(42).uniform(0, 1000, (15, 20))
    elev = _make_elevation(data)
    slope = derive_slope(elev)
    assert slope.shape == elev.shape
    assert list(slope.dims) == ["y", "x"]


def test_aspect_output_range():
    """Aspect values should be in [0, 360)."""
    data = np.random.default_rng(42).uniform(0, 1000, (15, 20))
    elev = _make_elevation(data)
    aspect = derive_aspect(elev)
    assert (aspect.values >= 0).all()
    assert (aspect.values < 360).all()


def test_slope_nonnegative():
    """Slope should always be non-negative."""
    data = np.random.default_rng(42).uniform(0, 1000, (15, 20))
    elev = _make_elevation(data)
    slope = derive_slope(elev)
    assert (slope.values >= 0).all()


def test_slope_has_attrs():
    data = np.full((5, 5), 100.0)
    elev = _make_elevation(data)
    slope = derive_slope(elev)
    assert slope.attrs["units"] == "degrees"
    assert "slope" in slope.attrs["long_name"].lower()


def test_aspect_has_attrs():
    data = np.full((5, 5), 100.0)
    elev = _make_elevation(data)
    aspect = derive_aspect(elev)
    assert aspect.attrs["units"] == "degrees"
    assert "aspect" in aspect.attrs["long_name"].lower()


def test_unsupported_slope_method():
    import pytest

    data = np.full((5, 5), 100.0)
    elev = _make_elevation(data)
    with pytest.raises(ValueError, match="Unsupported slope method"):
        derive_slope(elev, method="zevenbergen")


def test_unsupported_aspect_method():
    import pytest

    data = np.full((5, 5), 100.0)
    elev = _make_elevation(data)
    with pytest.raises(ValueError, match="Unsupported aspect method"):
        derive_aspect(elev, method="zevenbergen")
