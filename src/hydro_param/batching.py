"""Spatial batching: assign features to spatially contiguous groups.

Uses KD-tree recursive bisection (design.md section 5.5.1, Approach 5)
to produce balanced batches with tight bounding boxes. This is the
foundation for memory-efficient processing of high-resolution rasters.
"""

from __future__ import annotations

import logging
import warnings

import geopandas as gpd
import numpy as np

logger = logging.getLogger(__name__)


def _recursive_bisect(
    centroids: np.ndarray,
    indices: np.ndarray,
    depth: int = 0,
    max_depth: int = 7,
    min_batch_size: int = 50,
) -> list[np.ndarray]:
    """Recursively bisect features along alternating axes.

    Parameters
    ----------
    centroids : np.ndarray
        (N, 2) array of centroid coordinates.
    indices : np.ndarray
        Indices into the original GeoDataFrame for this partition.
    depth : int
        Current recursion depth.
    max_depth : int
        Maximum recursion depth (produces up to 2^max_depth batches).
    min_batch_size : int
        Stop splitting if a partition has fewer features than this.

    Returns
    -------
    list[np.ndarray]
        List of index arrays, one per batch.
    """
    if depth >= max_depth or len(indices) <= min_batch_size:
        return [indices]

    axis = depth % 2  # alternate x/y
    coords = centroids[indices, axis]
    median = np.median(coords)
    left_mask = coords <= median
    right_mask = ~left_mask

    # Avoid empty partitions when all values equal the median
    if not left_mask.any() or not right_mask.any():
        return [indices]

    left = _recursive_bisect(centroids, indices[left_mask], depth + 1, max_depth, min_batch_size)
    right = _recursive_bisect(centroids, indices[right_mask], depth + 1, max_depth, min_batch_size)
    return left + right


def spatial_batch(
    gdf: gpd.GeoDataFrame,
    batch_size: int = 500,
) -> gpd.GeoDataFrame:
    """Assign spatially contiguous batch IDs via KD-tree recursive bisection.

    Groups features so that each batch's bounding box is compact,
    enabling efficient spatial subsetting of source rasters.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Target fabric with polygon geometries.
    batch_size : int
        Target number of features per batch.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of input with ``batch_id`` column added.
    """
    if gdf.empty:
        result = gdf.copy()
        result["batch_id"] = np.array([], dtype=int)
        return result

    # Short-circuit: single batch when all features fit
    if len(gdf) <= batch_size:
        result = gdf.copy()
        result["batch_id"] = 0
        logger.info(
            "Spatial batching: %d features → 1 batch (all fit in batch_size=%d)",
            len(gdf),
            batch_size,
        )
        return result

    # Geographic CRS centroid warning is expected — we only need
    # approximate centroids for spatial grouping, not precision.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*geographic CRS.*centroid.*")
        centroids = np.column_stack(
            [gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values]
        )

    n_batches = max(1, len(gdf) // batch_size)
    max_depth = max(1, int(np.ceil(np.log2(n_batches))))

    batches = _recursive_bisect(
        centroids,
        np.arange(len(gdf)),
        max_depth=max_depth,
        min_batch_size=max(1, batch_size // 2),
    )

    batch_ids = np.empty(len(gdf), dtype=int)
    for batch_id, indices in enumerate(batches):
        batch_ids[indices] = batch_id

    result = gdf.copy()
    result["batch_id"] = batch_ids

    logger.info(
        "Spatial batching: %d features → %d batches (target size=%d, actual range=%d–%d)",
        len(gdf),
        len(batches),
        batch_size,
        min(len(b) for b in batches),
        max(len(b) for b in batches),
    )

    return result
