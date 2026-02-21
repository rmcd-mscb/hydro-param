"""Tests for data_access module helpers."""

import pandas as pd
import pytest

from hydro_param.data_access import build_climr_cat_dict


@pytest.fixture()
def mock_catalog() -> pd.DataFrame:
    """Create a mock ClimateR catalog DataFrame."""
    return pd.DataFrame(
        {
            "id": ["gridmet", "gridmet", "gridmet", "daymet"],
            "variable": ["pr", "tmmx", "tmmn", "prcp"],
            "URL": [
                "http://example.com/gridmet/pr",
                "http://example.com/gridmet/tmmx",
                "http://example.com/gridmet/tmmn",
                "http://example.com/daymet/prcp",
            ],
            "type": ["opendap", "opendap", "opendap", "opendap"],
        }
    )


def test_build_climr_cat_dict_valid(mock_catalog: pd.DataFrame):
    result = build_climr_cat_dict(mock_catalog, "gridmet", ["pr", "tmmx"])
    assert "pr" in result
    assert "tmmx" in result
    assert result["pr"]["URL"] == "http://example.com/gridmet/pr"
    assert result["tmmx"]["variable"] == "tmmx"


def test_build_climr_cat_dict_single_var(mock_catalog: pd.DataFrame):
    result = build_climr_cat_dict(mock_catalog, "gridmet", ["tmmn"])
    assert len(result) == 1
    assert "tmmn" in result


def test_build_climr_cat_dict_missing_var(mock_catalog: pd.DataFrame):
    with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
        build_climr_cat_dict(mock_catalog, "gridmet", ["nonexistent"])


def test_build_climr_cat_dict_missing_catalog_id(mock_catalog: pd.DataFrame):
    with pytest.raises(ValueError, match="not found in ClimateR catalog"):
        build_climr_cat_dict(mock_catalog, "unknown_dataset", ["pr"])
