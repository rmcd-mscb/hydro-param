"""Tests for USDA soil texture triangle classification."""

import numpy as np
import pytest
from hydro_param.classification import USDA_TEXTURE_CLASSES, classify_usda_texture


class TestClassifyUsdaTexture:
    """Tests for the vectorized USDA texture triangle classifier."""

    def test_pure_sand(self) -> None:
        result = classify_usda_texture(np.array([90.0]), np.array([5.0]), np.array([5.0]))
        assert result[0] == 1  # sand

    def test_pure_clay(self) -> None:
        result = classify_usda_texture(np.array([20.0]), np.array([20.0]), np.array([60.0]))
        assert result[0] == 12  # clay

    def test_loam_center(self) -> None:
        result = classify_usda_texture(np.array([40.0]), np.array([40.0]), np.array([20.0]))
        assert result[0] == 5  # loam

    def test_silt(self) -> None:
        result = classify_usda_texture(np.array([5.0]), np.array([90.0]), np.array([5.0]))
        assert result[0] == 8  # silt

    def test_silt_loam(self) -> None:
        result = classify_usda_texture(np.array([20.0]), np.array([60.0]), np.array([20.0]))
        assert result[0] == 6  # silt_loam

    def test_sandy_loam(self) -> None:
        result = classify_usda_texture(np.array([65.0]), np.array([25.0]), np.array([10.0]))
        assert result[0] == 3  # sandy_loam

    def test_loamy_sand(self) -> None:
        result = classify_usda_texture(np.array([82.0]), np.array([10.0]), np.array([8.0]))
        assert result[0] == 2  # loamy_sand

    def test_clay_loam(self) -> None:
        result = classify_usda_texture(np.array([30.0]), np.array([35.0]), np.array([35.0]))
        assert result[0] == 9  # clay_loam

    def test_silty_clay_loam(self) -> None:
        result = classify_usda_texture(np.array([10.0]), np.array([55.0]), np.array([35.0]))
        assert result[0] == 10  # silty_clay_loam

    def test_sandy_clay_loam(self) -> None:
        result = classify_usda_texture(np.array([60.0]), np.array([15.0]), np.array([25.0]))
        assert result[0] == 4  # sandy_clay_loam

    def test_sandy_clay(self) -> None:
        result = classify_usda_texture(np.array([50.0]), np.array([10.0]), np.array([40.0]))
        assert result[0] == 7  # sandy_clay

    def test_silty_clay(self) -> None:
        result = classify_usda_texture(np.array([5.0]), np.array([50.0]), np.array([45.0]))
        assert result[0] == 11  # silty_clay

    def test_vectorized_multiple_elements(self) -> None:
        sand = np.array([90.0, 20.0, 40.0])
        silt = np.array([5.0, 20.0, 40.0])
        clay = np.array([5.0, 60.0, 20.0])
        result = classify_usda_texture(sand, silt, clay)
        assert list(result) == [1, 12, 5]  # sand, clay, loam

    def test_nan_produces_nan(self) -> None:
        result = classify_usda_texture(np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
        assert np.isnan(result[0])

    def test_partial_nan(self) -> None:
        result = classify_usda_texture(
            np.array([90.0, np.nan]),
            np.array([5.0, 40.0]),
            np.array([5.0, np.nan]),
        )
        assert result[0] == 1  # sand
        assert np.isnan(result[1])

    def test_exhaustive_no_unclassified(self) -> None:
        """Every integer (sand, silt, clay) triple that sums to 100
        must classify into a valid USDA region (no NaN, no fallthrough).

        Sweeps all 5151 valid triples at 1% resolution.
        """
        valid_codes = set(USDA_TEXTURE_CLASSES.keys())
        for sand in range(0, 101):
            for clay in range(0, 101 - sand):
                silt = 100 - sand - clay
                result = classify_usda_texture(
                    np.array([float(sand)]),
                    np.array([float(silt)]),
                    np.array([float(clay)]),
                )
                code = result[0]
                assert not np.isnan(code), f"({sand}, {silt}, {clay}) classified as NaN"
                assert int(code) in valid_codes, f"({sand}, {silt}, {clay}) got invalid code {code}"

    def test_class_codes_dict_complete(self) -> None:
        assert len(USDA_TEXTURE_CLASSES) == 12
        assert set(USDA_TEXTURE_CLASSES.keys()) == set(range(1, 13))

    def test_fraction_scale_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Values in 0-1 range trigger a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            classify_usda_texture(np.array([0.4]), np.array([0.4]), np.array([0.2]))
        assert "fractions" in caplog.text.lower()
