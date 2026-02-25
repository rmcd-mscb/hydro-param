"""Tests for Swift (1976) solar radiation table computation."""

from __future__ import annotations

import logging

import numpy as np
import pytest

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

    def test_high_latitude_polar_day(self) -> None:
        """At 70N, summer should have ~24h sun, winter ~0h."""
        slopes = np.array([0.0])
        aspects = np.array([0.0])
        lats = np.array([70.0])
        _potsw, _horad, sunhrs = compute_soltab(slopes, aspects, lats)
        assert sunhrs[171, 0] > 20.0  # summer solstice
        assert sunhrs[354, 0] < 4.0  # winter solstice

    def test_high_latitude_southern_hemisphere(self) -> None:
        """Southern hemisphere high latitude produces valid output."""
        slopes = np.array([0.0])
        aspects = np.array([0.0])
        lats = np.array([-65.0])
        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)
        assert np.all(potsw >= 0)
        assert np.all(sunhrs >= 0)
        assert np.all(sunhrs <= 24)

    def test_steep_slope_wrap_around(self) -> None:
        """Steep slopes exercise wrap-around hour angle branches."""
        slopes = np.array([1.5, 2.0])
        aspects = np.array([180.0, 90.0])
        lats = np.array([45.0, 45.0])
        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)
        assert np.all(potsw >= 0)
        assert np.all(sunhrs >= 0)
        assert np.all(sunhrs <= 24)
        assert not np.any(np.isnan(potsw))

    def test_reference_values(self) -> None:
        """Validate specific values for a known configuration.

        Reference HRU: lat=40.5, slope=0.15 (rise/run), aspect=180 (south).
        Values computed from this implementation to guard against regressions
        in the trigonometric formulas.
        """
        slopes = np.array([0.15])
        aspects = np.array([180.0])
        lats = np.array([40.5])
        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)

        # Summer solstice (day 172)
        np.testing.assert_allclose(potsw[171, 0], 1010.088790, rtol=1e-6)
        np.testing.assert_allclose(horad[171, 0], 1022.596186, rtol=1e-6)
        np.testing.assert_allclose(sunhrs[171, 0], 14.094478, rtol=1e-5)

        # Winter solstice (day 355) — south-facing slope gets more than horad
        np.testing.assert_allclose(potsw[354, 0], 447.556212, rtol=1e-6)
        np.testing.assert_allclose(horad[354, 0], 322.702031, rtol=1e-6)

        # Equinox (day 80) — ~12h sunlight
        np.testing.assert_allclose(sunhrs[79, 0], 11.990138, rtol=1e-5)

    def test_negative_radiation_clamped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Steep north-facing slope at high latitude: values clamped, no NaN."""
        slopes = np.array([2.0])
        aspects = np.array([0.0])  # north-facing
        lats = np.array([60.0])
        with caplog.at_level(logging.DEBUG, logger="hydro_param.solar"):
            potsw, _horad, _sunhrs = compute_soltab(slopes, aspects, lats)
        assert np.all(potsw >= 0)
        assert not np.any(np.isnan(potsw))


class TestInputValidation:
    """Tests for compute_soltab input validation."""

    def test_nan_in_slopes_raises(self) -> None:
        slopes = np.array([0.1, np.nan])
        aspects = np.array([180.0, 90.0])
        lats = np.array([42.0, 42.0])
        with pytest.raises(ValueError, match="NaN"):
            compute_soltab(slopes, aspects, lats)

    def test_nan_in_lats_raises(self) -> None:
        slopes = np.array([0.1])
        aspects = np.array([180.0])
        lats = np.array([np.nan])
        with pytest.raises(ValueError, match="NaN"):
            compute_soltab(slopes, aspects, lats)

    def test_mismatched_lengths_raises(self) -> None:
        slopes = np.array([0.1, 0.2])
        aspects = np.array([180.0])
        lats = np.array([42.0])
        with pytest.raises(ValueError, match="equal length"):
            compute_soltab(slopes, aspects, lats)

    def test_empty_arrays_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            compute_soltab(np.array([]), np.array([]), np.array([]))

    def test_latitude_out_of_range_raises(self) -> None:
        slopes = np.array([0.1])
        aspects = np.array([180.0])
        lats = np.array([91.0])
        with pytest.raises(ValueError, match="Latitude"):
            compute_soltab(slopes, aspects, lats)

    def test_negative_slope_raises(self) -> None:
        slopes = np.array([-0.1])
        aspects = np.array([180.0])
        lats = np.array([42.0])
        with pytest.raises(ValueError, match="non-negative"):
            compute_soltab(slopes, aspects, lats)
