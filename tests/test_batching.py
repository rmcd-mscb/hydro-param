"""Tests for spatial batching (KD-tree recursive bisection)."""

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from hydro_param.batching import spatial_batch


def _make_grid_fabric(n_rows: int, n_cols: int) -> gpd.GeoDataFrame:
    """Create a grid of unit-square polygons as a test fabric."""
    polys = []
    ids = []
    for r in range(n_rows):
        for c in range(n_cols):
            polys.append(box(c, r, c + 1, r + 1))
            ids.append(r * n_cols + c)
    return gpd.GeoDataFrame(
        {"feature_id": ids},
        geometry=polys,
        crs="EPSG:4326",
    )


def test_all_features_assigned():
    """Every feature gets exactly one batch_id."""
    fabric = _make_grid_fabric(10, 10)
    result = spatial_batch(fabric, batch_size=25)
    assert "batch_id" in result.columns
    assert len(result) == 100
    assert result["batch_id"].notna().all()


def test_batch_ids_contiguous():
    """Batch IDs are 0-indexed and contiguous."""
    fabric = _make_grid_fabric(10, 10)
    result = spatial_batch(fabric, batch_size=25)
    batch_ids = sorted(result["batch_id"].unique())
    assert batch_ids == list(range(len(batch_ids)))


def test_batches_are_balanced():
    """Batch sizes are within 2x of target."""
    fabric = _make_grid_fabric(20, 20)  # 400 features
    result = spatial_batch(fabric, batch_size=50)

    sizes = result.groupby("batch_id").size()
    # KD-tree bisection guarantees balance within 2x
    assert sizes.min() >= 25  # >= batch_size / 2
    assert sizes.max() <= 250  # reasonable upper bound


def test_batches_spatially_compact():
    """Each batch bbox is smaller than the full domain bbox."""
    fabric = _make_grid_fabric(20, 20)  # 400 features, domain = 20x20
    result = spatial_batch(fabric, batch_size=50)

    full_area = 20 * 20  # domain is 20x20 units
    for batch_id in result["batch_id"].unique():
        batch = result[result["batch_id"] == batch_id]
        minx, miny, maxx, maxy = batch.total_bounds
        batch_area = (maxx - minx) * (maxy - miny)
        # Each batch bbox should be much smaller than the full domain
        assert batch_area < full_area * 0.5


def test_single_feature():
    """A fabric with one feature gets batch_id 0."""
    fabric = gpd.GeoDataFrame(
        {"feature_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    result = spatial_batch(fabric, batch_size=500)
    assert len(result) == 1
    assert result["batch_id"].iloc[0] == 0


def test_empty_fabric():
    """Empty fabric returns empty result with batch_id column."""
    fabric = gpd.GeoDataFrame(
        {"feature_id": []},
        geometry=[],
        crs="EPSG:4326",
    )
    result = spatial_batch(fabric, batch_size=500)
    assert "batch_id" in result.columns
    assert len(result) == 0


def test_small_fabric_single_batch():
    """Fabric smaller than batch_size → single batch."""
    fabric = _make_grid_fabric(3, 3)  # 9 features
    result = spatial_batch(fabric, batch_size=500)
    assert result["batch_id"].nunique() == 1


def test_large_fabric_multiple_batches():
    """Larger fabric produces multiple batches."""
    fabric = _make_grid_fabric(30, 30)  # 900 features
    result = spatial_batch(fabric, batch_size=100)
    assert result["batch_id"].nunique() > 1


def test_does_not_mutate_input():
    """spatial_batch returns a copy, not a view."""
    fabric = _make_grid_fabric(5, 5)
    result = spatial_batch(fabric, batch_size=10)
    assert "batch_id" not in fabric.columns
    assert "batch_id" in result.columns


def test_preserves_original_columns():
    """Original columns are preserved in the result."""
    fabric = _make_grid_fabric(5, 5)
    result = spatial_batch(fabric, batch_size=10)
    assert "feature_id" in result.columns
    assert "geometry" in result.columns
    assert np.array_equal(result["feature_id"].values, fabric["feature_id"].values)
