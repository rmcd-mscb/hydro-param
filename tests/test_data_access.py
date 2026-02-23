"""Tests for data_access module helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydro_param.data_access import _is_remote_url, build_climr_cat_dict, fetch_local_tiff
from hydro_param.dataset_registry import DatasetEntry


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


# ---------------------------------------------------------------------------
# _is_remote_url tests
# ---------------------------------------------------------------------------


def test_is_remote_url_http():
    assert _is_remote_url("http://example.com/data.vrt") is True


def test_is_remote_url_https():
    assert _is_remote_url("https://example.com/data.tif") is True


def test_is_remote_url_local_path():
    assert _is_remote_url("/data/local/file.tif") is False


def test_is_remote_url_relative_path():
    assert _is_remote_url("data/file.tif") is False


# ---------------------------------------------------------------------------
# fetch_local_tiff remote URL + variable_source tests
# ---------------------------------------------------------------------------


def _make_mock_da() -> MagicMock:
    """Create a mock object that behaves like rioxarray.open_rasterio output."""
    squeezed = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
    )
    # Build mock chain: open_rasterio() -> .squeeze() -> .rio.clip_box()
    mock_da = MagicMock()
    mock_squeezed = MagicMock()
    mock_da.squeeze.return_value = mock_squeezed
    mock_squeezed.rio.crs = "EPSG:4326"
    mock_squeezed.rio.clip_box.return_value = squeezed
    mock_squeezed.size = 16
    return mock_da


def test_fetch_local_tiff_remote_url():
    """HTTP URLs skip Path.exists() check and open correctly."""
    rioxarray = pytest.importorskip("rioxarray")

    mock_da = _make_mock_da()
    vrt_url = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/sand_mean_0_5.vrt"

    entry = DatasetEntry(strategy="local_tiff", source=vrt_url)
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with patch.object(rioxarray, "open_rasterio", return_value=mock_da) as mock_open:
        result = fetch_local_tiff(entry, bbox, dataset_name="polaris_30m")

    mock_open.assert_called_once_with(vrt_url, masked=True)
    assert result is not None


def test_fetch_local_tiff_variable_source_override():
    """variable_source takes precedence over entry.source."""
    rioxarray = pytest.importorskip("rioxarray")

    mock_da = _make_mock_da()
    var_url = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/clay_mean_0_5.vrt"

    entry = DatasetEntry(strategy="local_tiff", source="/some/default/path.tif")
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with patch.object(rioxarray, "open_rasterio", return_value=mock_da) as mock_open:
        result = fetch_local_tiff(entry, bbox, dataset_name="polaris_30m", variable_source=var_url)

    # Should use the variable_source URL, not the entry.source path
    mock_open.assert_called_once_with(var_url, masked=True)
    assert result is not None


def test_fetch_local_tiff_no_source_raises():
    """Raises ValueError when neither variable_source nor entry.source is set."""
    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841

    entry = DatasetEntry(strategy="local_tiff")
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with pytest.raises(ValueError, match="no 'source' path set"):
        fetch_local_tiff(entry, bbox, dataset_name="test_dataset")
