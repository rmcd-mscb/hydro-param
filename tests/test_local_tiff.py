"""Tests for local GeoTIFF data access.

Tests fetch_local_tiff() with synthetic rasters: continuous and
categorical data, bbox clipping, missing file handling, and pipeline
dispatch via the _fetch helper.

Requires rioxarray (available in the 'full' pixi environment).
Tests are skipped when rioxarray is not installed.
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydro_param.dataset_registry import DatasetEntry

pytest.importorskip("rioxarray")

from hydro_param.data_access import fetch_local_tiff  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiff(path: Path, *, categorical: bool = False, crs: str = "EPSG:5070") -> Path:
    """Create a small synthetic GeoTIFF for testing.

    Parameters
    ----------
    path : Path
        Output file path.
    categorical : bool
        If True, write integer land cover classes; otherwise float elevation.
    crs : str
        CRS to assign to the raster.

    Returns
    -------
    Path
        The written file path.
    """
    ny, nx = 100, 100
    x = np.arange(nx) * 30.0 + 1_000_000.0  # 30m cells in projected CRS
    y = np.arange(ny) * 30.0 + 2_000_000.0

    if categorical:
        # NLCD-style classes: 11 (water), 21 (developed), 41 (forest)
        data = np.full((ny, nx), 41, dtype=np.uint8)
        data[:30, :] = 11
        data[30:60, :] = 21
    else:
        data = np.random.default_rng(42).uniform(100.0, 500.0, (ny, nx)).astype(np.float32)

    da = xr.DataArray(
        data,
        dims=["y", "x"],
        coords={"y": y, "x": x},
    )
    da = da.rio.write_crs(crs)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da.rio.to_raster(path)
    return path


def _make_entry(source: str, crs: str = "EPSG:5070", categorical: bool = False) -> DatasetEntry:
    """Create a minimal DatasetEntry for local_tiff strategy."""
    variables = [
        {
            "name": "land_cover" if categorical else "elevation",
            "band": 1,
            "categorical": categorical,
        }
    ]
    return DatasetEntry(
        strategy="local_tiff",
        source=source,
        crs=crs,
        x_coord="x",
        y_coord="y",
        variables=variables,
    )


# ---------------------------------------------------------------------------
# Tests: fetch_local_tiff
# ---------------------------------------------------------------------------


def test_fetch_local_tiff_loads_continuous(tmp_path: Path):
    """Load a continuous raster and clip to bbox."""
    tiff = _make_tiff(tmp_path / "elev.tif")
    entry = _make_entry(str(tiff))

    # Bbox covering roughly the middle of the raster
    bbox = [1_000_500.0, 2_000_500.0, 1_001_500.0, 2_001_500.0]
    da = fetch_local_tiff(entry, bbox)

    assert isinstance(da, xr.DataArray)
    assert da.size > 0
    # Clipped shape should be smaller than full 100x100
    assert da.shape[0] < 100 or da.shape[1] < 100


def test_fetch_local_tiff_loads_categorical(tmp_path: Path):
    """Load a categorical raster (NLCD-style)."""
    tiff = _make_tiff(tmp_path / "nlcd.tif", categorical=True)
    entry = _make_entry(str(tiff), categorical=True)

    # Full extent bbox
    bbox = [1_000_000.0, 2_000_000.0, 1_003_000.0, 2_003_000.0]
    da = fetch_local_tiff(entry, bbox)

    assert isinstance(da, xr.DataArray)
    # Should contain our NLCD classes
    values = da.values
    if np.issubdtype(values.dtype, np.floating):
        mask = np.isnan(values)
    else:
        mask = np.zeros_like(values, dtype=bool)
    unique = np.unique(values[~mask])
    assert 11 in unique or 21 in unique or 41 in unique


def test_fetch_local_tiff_clips_to_bbox(tmp_path: Path):
    """Bbox clipping produces a smaller raster than full extent."""
    tiff = _make_tiff(tmp_path / "elev.tif")
    entry = _make_entry(str(tiff))

    # Small bbox: ~10 cells x 10 cells
    bbox = [1_000_000.0, 2_000_000.0, 1_000_300.0, 2_000_300.0]
    da = fetch_local_tiff(entry, bbox)

    assert da.shape[0] <= 12  # ~10 cells + boundary tolerance
    assert da.shape[1] <= 12


def test_fetch_local_tiff_preserves_crs(tmp_path: Path):
    """Loaded raster retains the source CRS."""
    tiff = _make_tiff(tmp_path / "elev.tif", crs="EPSG:5070")
    entry = _make_entry(str(tiff), crs="EPSG:5070")

    bbox = [1_000_000.0, 2_000_000.0, 1_003_000.0, 2_003_000.0]
    da = fetch_local_tiff(entry, bbox)

    assert da.rio.crs is not None
    assert da.rio.crs.to_epsg() == 5070


def test_fetch_local_tiff_missing_file_raises(tmp_path: Path):
    """FileNotFoundError when source path does not exist."""
    entry = _make_entry(str(tmp_path / "nonexistent.tif"))

    with pytest.raises(FileNotFoundError, match="not found"):
        fetch_local_tiff(entry, [0.0, 0.0, 1.0, 1.0])


def test_fetch_local_tiff_no_data_in_bbox_raises(tmp_path: Path):
    """RuntimeError when bbox does not overlap raster extent."""
    tiff = _make_tiff(tmp_path / "elev.tif")
    entry = _make_entry(str(tiff))

    # Bbox far outside raster extent
    bbox = [0.0, 0.0, 1.0, 1.0]
    with pytest.raises(RuntimeError, match="No data in bbox"):
        fetch_local_tiff(entry, bbox)


# ---------------------------------------------------------------------------
# Tests: missing source error messages
# ---------------------------------------------------------------------------


def test_fetch_local_tiff_no_source_raises():
    """ValueError when source is None and no download info."""
    entry = DatasetEntry(strategy="local_tiff")
    with pytest.raises(ValueError, match="no 'source' path set"):
        fetch_local_tiff(entry, [0.0, 0.0, 1.0, 1.0])


def test_fetch_local_tiff_no_source_includes_download_info():
    """Error message includes download URL, size, and notes when available."""
    entry = DatasetEntry(
        strategy="local_tiff",
        download={
            "url": "s3://bucket/nlcd.tif",
            "size_gb": 1.5,
            "notes": "Use aws s3 cp",
        },
    )
    with pytest.raises(ValueError, match="s3://bucket/nlcd.tif") as exc_info:
        fetch_local_tiff(entry, [0.0, 0.0, 1.0, 1.0])
    msg = str(exc_info.value)
    assert "~1.5 GB" in msg
    assert "aws s3 cp" in msg


def test_fetch_local_tiff_no_source_suggests_config():
    """Error message suggests setting source in pipeline config."""
    entry = DatasetEntry(
        strategy="local_tiff",
        download={"url": "s3://bucket/data.tif"},
    )
    with pytest.raises(ValueError, match="source") as exc_info:
        fetch_local_tiff(entry, [0.0, 0.0, 1.0, 1.0])
    msg = str(exc_info.value)
    assert "pipeline config" in msg
    assert "your_dataset_name" in msg
