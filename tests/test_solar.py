"""Tests for Swift (1976) solar radiation table computation."""

from __future__ import annotations

import numpy as np

from hydro_param.solar import NDOY, compute_soltab, r1, solar_declination


class TestSolarConstants:
    """Tests for module-level solar constants."""

    def test_ndoy_is_366(self) -> None:
        assert NDOY == 366

    def test_solar_declination_shape(self) -> None:
        assert solar_declination.shape == (366,)

    def test_solar_declination_range(self) -> None:
        assert np.all(solar_declination >= -0.45)
        assert np.all(solar_declination <= 0.45)

    def test_r1_shape(self) -> None:
        assert r1.shape == (366,)

    def test_r1_positive(self) -> None:
        assert np.all(r1 > 0)


class TestComputeSoltab:
    """Tests for the compute_soltab function."""

    def test_output_shapes(self) -> None:
        slopes = np.array([0.1, 0.3, 0.5])
        aspects = np.array([180.0, 90.0, 270.0])
        lats = np.array([42.0, 35.0, 48.0])
        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)
        assert potsw.shape == (366, 3)
        assert horad.shape == (366, 3)
        assert sunhrs.shape == (366, 3)

    def test_all_non_negative(self) -> None:
        slopes = np.array([0.1, 0.5])
        aspects = np.array([180.0, 0.0])
        lats = np.array([42.0, 42.0])
        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)
        assert np.all(potsw >= 0)
        assert np.all(horad >= 0)
        assert np.all(sunhrs >= 0)

    def test_flat_surface_equals_horizontal(self) -> None:
        slopes = np.array([0.0, 0.0])
        aspects = np.array([0.0, 180.0])
        lats = np.array([42.0, 35.0])
        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)
        np.testing.assert_allclose(potsw, horad, rtol=1e-10)

    def test_south_facing_more_than_north_mid_latitude(self) -> None:
        slopes = np.array([0.3, 0.3])
        aspects = np.array([180.0, 0.0])
        lats = np.array([42.0, 42.0])
        potsw, _horad, _sunhrs = compute_soltab(slopes, aspects, lats)
        annual_south = potsw[:, 0].sum()
        annual_north = potsw[:, 1].sum()
        assert annual_south > annual_north

    def test_single_hru(self) -> None:
        slopes = np.array([0.2])
        aspects = np.array([180.0])
        lats = np.array([40.0])
        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)
        assert potsw.shape == (366, 1)
        assert np.all(potsw >= 0)

    def test_sunhrs_reasonable_range(self) -> None:
        slopes = np.array([0.1, 0.5, 0.0])
        aspects = np.array([180.0, 270.0, 0.0])
        lats = np.array([42.0, 60.0, 0.0])
        _potsw, _horad, sunhrs = compute_soltab(slopes, aspects, lats)
        assert np.all(sunhrs >= 0)
        assert np.all(sunhrs <= 24)

    def test_equator_equinox_symmetry(self) -> None:
        slopes = np.array([0.0])
        aspects = np.array([0.0])
        lats = np.array([0.0])
        _potsw, _horad, sunhrs = compute_soltab(slopes, aspects, lats)
        np.testing.assert_allclose(sunhrs, 12.0, atol=0.5)
