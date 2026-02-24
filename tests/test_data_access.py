"""Tests for data_access module helpers."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydro_param.data_access import (
    _is_remote_url,
    build_climr_cat_dict,
    fetch_local_tiff,
    save_to_geotiff,
)
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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("http://example.com/data.vrt", True),
        ("https://example.com/data.tif", True),
        ("/data/local/file.tif", False),
        ("data/file.tif", False),
        ("s3://bucket/key.tif", False),
    ],
)
def test_is_remote_url(source: str, expected: bool):
    assert _is_remote_url(source) is expected


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
    pytest.importorskip("rioxarray")

    entry = DatasetEntry(strategy="local_tiff")
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with pytest.raises(ValueError, match="requires a source path or URL"):
        fetch_local_tiff(entry, bbox, dataset_name="test_dataset")


def test_fetch_local_tiff_variable_source_local_path_not_found():
    """variable_source with a non-existent local path raises FileNotFoundError."""
    pytest.importorskip("rioxarray")

    entry = DatasetEntry(strategy="local_tiff", source="/valid/default.tif")
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with pytest.raises(FileNotFoundError, match="/nonexistent/override.tif"):
        fetch_local_tiff(entry, bbox, variable_source="/nonexistent/override.tif")


def test_fetch_local_tiff_remote_open_failure():
    """Remote open failure wraps error with dataset context."""
    rioxarray = pytest.importorskip("rioxarray")

    vrt_url = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/bad.vrt"
    entry = DatasetEntry(strategy="local_tiff", source=vrt_url)
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with patch.object(rioxarray, "open_rasterio", side_effect=Exception("GDAL error")):
        with pytest.raises(RuntimeError, match="Failed to open remote raster.*polaris"):
            fetch_local_tiff(entry, bbox, dataset_name="polaris_30m")


# ---------------------------------------------------------------------------
# fetch_stac_cog: per-variable asset_key
# ---------------------------------------------------------------------------


def test_fetch_stac_cog_uses_per_variable_asset_key():
    """fetch_stac_cog uses asset_key parameter instead of entry.asset_key when provided."""
    from hydro_param.data_access import fetch_stac_cog

    rioxarray = pytest.importorskip("rioxarray")

    squeezed = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
    )
    mock_da = MagicMock()
    mock_squeezed = MagicMock()
    mock_da.squeeze.return_value = mock_squeezed
    mock_squeezed.rio.crs = "EPSG:5070"
    mock_squeezed.rio.clip_box.return_value = squeezed
    mock_squeezed.size = 16

    # Build a mock STAC item with per-variable assets (no 'data' key)
    mock_asset = MagicMock()
    mock_asset.href = "https://soils.blob.core.windows.net/gnatsgo/aws0_100.tif"
    mock_item = MagicMock()
    mock_item.id = "test_tile"
    mock_item.properties = {}
    mock_item.assets = {"aws0_100": mock_asset, "rootznemc": MagicMock()}

    mock_search = MagicMock()
    mock_search.item_collection.return_value = [mock_item]

    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="gnatsgo-rasters",
        crs="EPSG:5070",
        asset_key="data",  # dataset-level default that would fail
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with (
        patch("pystac_client.Client.open", return_value=mock_client),
        patch.object(rioxarray, "open_rasterio", return_value=mock_da),
    ):
        # With per-variable asset_key, should succeed even though 'data' doesn't exist
        result = fetch_stac_cog(entry, bbox, asset_key="aws0_100")

    assert result is not None


def test_fetch_stac_cog_falls_back_to_entry_asset_key():
    """fetch_stac_cog uses entry.asset_key when asset_key parameter is None."""
    from hydro_param.data_access import fetch_stac_cog

    rioxarray = pytest.importorskip("rioxarray")

    squeezed = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
    )
    mock_da = MagicMock()
    mock_squeezed = MagicMock()
    mock_da.squeeze.return_value = mock_squeezed
    mock_squeezed.rio.crs = "EPSG:4269"
    mock_squeezed.rio.clip_box.return_value = squeezed
    mock_squeezed.size = 16

    mock_asset = MagicMock()
    mock_asset.href = "https://example.com/dem.tif"
    mock_item = MagicMock()
    mock_item.id = "test_tile"
    mock_item.properties = {}
    mock_item.assets = {"data": mock_asset}

    mock_search = MagicMock()
    mock_search.item_collection.return_value = [mock_item]

    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="3dep-seamless",
        crs="EPSG:4269",
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with (
        patch("pystac_client.Client.open", return_value=mock_client),
        patch.object(rioxarray, "open_rasterio", return_value=mock_da),
    ):
        # No asset_key param, should use entry.asset_key ("data")
        result = fetch_stac_cog(entry, bbox)

    assert result is not None


def test_fetch_stac_cog_missing_asset_key_raises_informative_error():
    """fetch_stac_cog raises KeyError with available assets when key is missing."""
    from hydro_param.data_access import fetch_stac_cog

    pytest.importorskip("rioxarray")

    # Build a mock STAC item with only specific assets
    mock_data_asset = MagicMock()
    mock_data_asset.roles = ["data"]
    mock_preview_asset = MagicMock()
    mock_preview_asset.roles = ["overview"]
    mock_item = MagicMock()
    mock_item.id = "test_tile_123"
    mock_item.properties = {}
    mock_item.assets = {
        "aws0_100": mock_data_asset,
        "rootznemc": mock_data_asset,
        "rendered_preview": mock_preview_asset,
    }

    mock_search = MagicMock()
    mock_search.item_collection.return_value = [mock_item]

    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="gnatsgo-rasters",
        crs="EPSG:5070",
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with (
        patch("pystac_client.Client.open", return_value=mock_client),
        pytest.raises(KeyError, match="'data' not found in STAC item 'test_tile_123'"),
    ):
        # Default asset_key is "data" which doesn't exist — should get helpful error
        fetch_stac_cog(entry, bbox)


def test_fetch_stac_cog_missing_asset_key_lists_available():
    """KeyError message includes available data asset keys."""
    from hydro_param.data_access import fetch_stac_cog

    pytest.importorskip("rioxarray")

    mock_data_asset = MagicMock()
    mock_data_asset.roles = ["data"]
    mock_item = MagicMock()
    mock_item.id = "tile_1"
    mock_item.properties = {}
    mock_item.assets = {"aws0_100": mock_data_asset, "rootznemc": mock_data_asset}

    mock_search = MagicMock()
    mock_search.item_collection.return_value = [mock_item]

    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="gnatsgo-rasters",
        crs="EPSG:5070",
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with (
        patch("pystac_client.Client.open", return_value=mock_client),
        pytest.raises(KeyError, match="aws0_100.*rootznemc"),
    ):
        fetch_stac_cog(entry, bbox, asset_key="nonexistent")


# ---------------------------------------------------------------------------
# query_stac_items
# ---------------------------------------------------------------------------


def test_query_stac_items_returns_items():
    """query_stac_items returns a list of STAC items."""
    from hydro_param.data_access import query_stac_items

    mock_item = MagicMock()
    mock_item.properties = {"gsd": 10}

    mock_search = MagicMock()
    mock_search.item_collection.return_value = [mock_item]

    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="3dep-seamless",
        crs="EPSG:4269",
        gsd=10,
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with patch("pystac_client.Client.open", return_value=mock_client):
        items = query_stac_items(entry, bbox)

    assert len(items) == 1
    assert items[0] is mock_item


def test_fetch_stac_cog_with_prequeried_items():
    """fetch_stac_cog skips STAC query when items are provided."""
    from hydro_param.data_access import fetch_stac_cog

    rioxarray = pytest.importorskip("rioxarray")

    squeezed = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
    )
    mock_da = MagicMock()
    mock_squeezed = MagicMock()
    mock_da.squeeze.return_value = mock_squeezed
    mock_squeezed.rio.crs = "EPSG:4269"
    mock_squeezed.rio.clip_box.return_value = squeezed
    mock_squeezed.size = 16

    mock_asset = MagicMock()
    mock_asset.href = "https://example.com/dem.tif"
    mock_item = MagicMock()
    mock_item.id = "test_tile"
    mock_item.properties = {}
    mock_item.assets = {"data": mock_asset}

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="3dep-seamless",
        crs="EPSG:4269",
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with (
        patch("pystac_client.Client.open") as mock_open,
        patch.object(rioxarray, "open_rasterio", return_value=mock_da),
    ):
        result = fetch_stac_cog(entry, bbox, items=[mock_item])

    # STAC client should NOT have been called
    mock_open.assert_not_called()
    assert result is not None


# ---------------------------------------------------------------------------
# save_to_geotiff: copy-free save
# ---------------------------------------------------------------------------


def test_save_to_geotiff_does_not_copy_array(tmp_path: Path):
    """save_to_geotiff should not create a full copy of the DataArray."""
    pytest.importorskip("rioxarray")

    da = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
        attrs={"_FillValue": -9999.0, "units": "meters"},
    )
    da = da.rio.set_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    out_path = tmp_path / "test.tif"
    save_to_geotiff(da, out_path)

    # Original attrs must be preserved (not mutated)
    assert "_FillValue" in da.attrs
    assert da.attrs["_FillValue"] == -9999.0
    assert da.attrs["units"] == "meters"
    assert out_path.exists()


def test_fetch_nhgf_stac_cog_basic():
    """fetch_nhgf_stac_cog fetches a COG and returns a DataArray."""
    from unittest.mock import MagicMock, patch

    import numpy as np
    import xarray as xr

    from hydro_param.data_access import fetch_nhgf_stac_cog

    mock_collection = MagicMock(name="pystac.Collection")
    mock_da = xr.DataArray(
        np.ones((3, 3)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]},
    )
    mock_nhgf = MagicMock(name="NHGFStacTiffData")
    mock_nhgf.ds = mock_da

    with (
        patch("gdptools.helpers.get_stac_collection", return_value=mock_collection),
        patch("gdptools.NHGFStacTiffData", return_value=mock_nhgf) as p_nhgf,
    ):
        result = fetch_nhgf_stac_cog(
            collection_id="nlcd-LndCov",
            variable_name="LndCov",
            year=2021,
        )

    assert isinstance(result, xr.DataArray)
    assert result.shape == (3, 3)
    # Verify year was passed as time_period
    nhgf_kwargs = p_nhgf.call_args.kwargs
    assert nhgf_kwargs["source_time_period"] == ["2021-01-01", "2021-12-31"]


def test_fetch_nhgf_stac_cog_no_year():
    """fetch_nhgf_stac_cog works without a year."""
    from unittest.mock import MagicMock, patch

    import numpy as np
    import xarray as xr

    from hydro_param.data_access import fetch_nhgf_stac_cog

    mock_collection = MagicMock()
    mock_da = xr.DataArray(np.ones((2, 2)), dims=["y", "x"])
    mock_nhgf = MagicMock()
    mock_nhgf.ds = mock_da

    with (
        patch("gdptools.helpers.get_stac_collection", return_value=mock_collection),
        patch("gdptools.NHGFStacTiffData", return_value=mock_nhgf) as p_nhgf,
    ):
        result = fetch_nhgf_stac_cog(
            collection_id="nlcd-LndCov",
            variable_name="LndCov",
            year=None,
        )

    assert isinstance(result, xr.DataArray)
    # source_time_period should NOT be in kwargs when year is None
    nhgf_kwargs = p_nhgf.call_args.kwargs
    assert "source_time_period" not in nhgf_kwargs
