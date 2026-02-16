"""Tests for unit conversion registry."""

from __future__ import annotations

import numpy as np
import pytest

from hydro_param.units import convert, list_conversions


class TestConvert:
    """Tests for the convert() function."""

    def test_identity(self) -> None:
        """Same unit returns input unchanged."""
        values = np.array([1.0, 2.0, 3.0])
        result = convert(values, "m", "m")
        np.testing.assert_array_equal(result, values)

    def test_meters_to_feet(self) -> None:
        values = np.array([1.0, 100.0, 1000.0])
        result = convert(values, "m", "ft")
        np.testing.assert_allclose(result, [3.28084, 328.084, 3280.84])

    def test_feet_to_meters(self) -> None:
        values = np.array([3.28084])
        result = convert(values, "ft", "m")
        np.testing.assert_allclose(result, [1.0], atol=1e-4)

    def test_mm_to_inches(self) -> None:
        values = np.array([25.4, 50.8])
        result = convert(values, "mm", "in")
        np.testing.assert_allclose(result, [1.0, 2.0])

    def test_inches_to_mm(self) -> None:
        values = np.array([1.0, 2.0])
        result = convert(values, "in", "mm")
        np.testing.assert_allclose(result, [25.4, 50.8])

    def test_celsius_to_fahrenheit(self) -> None:
        values = np.array([0.0, 100.0, -40.0])
        result = convert(values, "C", "F")
        np.testing.assert_allclose(result, [32.0, 212.0, -40.0])

    def test_fahrenheit_to_celsius(self) -> None:
        values = np.array([32.0, 212.0, -40.0])
        result = convert(values, "F", "C")
        np.testing.assert_allclose(result, [0.0, 100.0, -40.0])

    def test_kelvin_to_fahrenheit(self) -> None:
        values = np.array([273.15, 373.15])
        result = convert(values, "K", "F")
        np.testing.assert_allclose(result, [32.0, 212.0])

    def test_kelvin_to_celsius(self) -> None:
        values = np.array([273.15, 373.15])
        result = convert(values, "K", "C")
        np.testing.assert_allclose(result, [0.0, 100.0])

    def test_sq_meters_to_acres(self) -> None:
        values = np.array([4046.8564224])  # 1 acre in m²
        result = convert(values, "m2", "acres")
        np.testing.assert_allclose(result, [1.0], atol=1e-3)

    def test_degrees_to_radians(self) -> None:
        values = np.array([0.0, 90.0, 180.0, 360.0])
        result = convert(values, "deg", "rad")
        np.testing.assert_allclose(result, [0.0, np.pi / 2, np.pi, 2 * np.pi])

    def test_round_trip_m_ft(self) -> None:
        """Round-trip conversion preserves values."""
        original = np.array([42.0, 1500.0, 0.0])
        result = convert(convert(original, "m", "ft"), "ft", "m")
        np.testing.assert_allclose(result, original, atol=1e-10)

    def test_round_trip_c_f(self) -> None:
        original = np.array([-10.0, 0.0, 25.0, 37.0])
        result = convert(convert(original, "C", "F"), "F", "C")
        np.testing.assert_allclose(result, original, atol=1e-10)

    def test_round_trip_mm_in(self) -> None:
        original = np.array([0.0, 10.0, 254.0])
        result = convert(convert(original, "mm", "in"), "in", "mm")
        np.testing.assert_allclose(result, original, atol=1e-10)

    def test_unknown_conversion_raises(self) -> None:
        with pytest.raises(KeyError, match="No conversion registered"):
            convert(np.array([1.0]), "parsecs", "furlongs")


class TestListConversions:
    """Tests for list_conversions()."""

    def test_returns_list(self) -> None:
        result = list_conversions()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_includes_standard_conversions(self) -> None:
        pairs = {(f, t) for f, t, _ in list_conversions()}
        assert ("m", "ft") in pairs
        assert ("C", "F") in pairs
        assert ("mm", "in") in pairs
        assert ("m2", "acres") in pairs
