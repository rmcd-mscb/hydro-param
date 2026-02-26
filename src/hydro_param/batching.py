"""Spatial batching: assign features to spatially contiguous groups.

Group polygon features into spatially contiguous batches using KD-tree
recursive bisection. This ensures that each batch's bounding box is
compact, which is critical for efficient spatial subsetting of
high-resolution source rasters (e.g., 3DEP at 10 m, gNATSGO at 30 m).

Without spatial batching, arbitrarily ordered fabric features would
produce large bounding boxes that fetch far more raster data than needed,
leading to excessive memory use and slow processing. The KD-tree approach
(design.md section 5.5.1, Approach 5) provides O(n log n) partitioning
with guaranteed spatial locality.

See Also
--------
design.md : Section 5.5.1 (spatial batching approaches and trade-offs).
hydro_param.pipeline : Orchestrator that processes batches sequentially.

References
----------
.. [1] design.md section 5.5.1 -- KD-tree recursive bisection (Approach 5).
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
    """Recursively bisect features along alternating axes (KD-tree style).

    Partition a set of point features by splitting at the median along
    alternating x/y axes, producing spatially compact groups. Recursion
    stops when the maximum depth is reached or a partition falls below
    the minimum batch size.

    This implements a simplified KD-tree construction where each internal
    node splits at the median rather than an arbitrary pivot. The
    alternating axis selection (x at even depths, y at odd depths) ensures
    balanced spatial coverage in both dimensions.

    Parameters
    ----------
    centroids : np.ndarray
        Shape ``(N, 2)`` array of centroid coordinates (x, y) for all
        features. Coordinates may be in any CRS -- only relative
        ordering matters for median splitting.
    indices : np.ndarray
        1-D integer array of indices into the original GeoDataFrame for
        this partition. Allows tracking which features belong to each
        resulting batch.
    depth : int
        Current recursion depth. Used to select the split axis
        (``depth % 2``: 0 = x-axis, 1 = y-axis).
    max_depth : int
        Maximum recursion depth. Produces up to ``2^max_depth`` batches.
        Computed by :func:`spatial_batch` from the target batch count.
    min_batch_size : int
        Stop splitting if a partition has fewer features than this
        threshold. Prevents creation of very small batches that would
        add per-batch overhead without meaningful memory savings.

    Returns
    -------
    list[np.ndarray]
        List of 1-D integer arrays, one per batch. Each array contains
        indices into the original GeoDataFrame. The union of all arrays
        covers all input indices exactly once.

    Notes
    -----
    When all centroid values along the split axis are equal (e.g.,
    features aligned along a meridian), the median split produces an
    empty partition. In this case, recursion stops and the current
    partition is returned as a single batch.
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

    Group polygon features into spatially compact batches so that each
    batch's bounding box covers a small geographic area. This is the
    primary entry point for spatial batching in the pipeline, called
    during stage 1 (resolve fabric) before any data access occurs.

    Compact bounding boxes are critical for memory efficiency: when the
    pipeline fetches source rasters clipped to a batch's bbox, a tight
    bbox means less data loaded into memory. For high-resolution datasets
    like gNATSGO (30 m, ~1.25 GB per variable), this prevents OOM errors.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Target fabric with polygon geometries (Polygon or MultiPolygon).
        May be in any CRS -- centroids are computed for spatial grouping
        only (approximate centroids are sufficient).
    batch_size : int
        Target number of features per batch. The actual batch sizes will
        vary due to the recursive bisection algorithm (typically within
        ``[batch_size/2, batch_size*2]``). Default is 500.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of input with a ``batch_id`` column (int) added. Batch IDs
        are sequential integers starting from 0. Features within the
        same batch are spatially contiguous.

    Notes
    -----
    For fabrics with geographic CRS (e.g., EPSG:4326), the centroid
    computation emits a ``UserWarning`` about geographic CRS accuracy.
    This warning is suppressed because only approximate centroids are
    needed for spatial grouping -- the batch boundaries do not need to be
    geometrically precise.

    The recursion depth is computed as ``ceil(log2(n_features / batch_size))``
    to produce approximately the right number of batches. The
    ``min_batch_size`` is set to ``batch_size / 2`` to prevent excessive
    fragmentation.

    Examples
    --------
    >>> import geopandas as gpd
    >>> fabric = gpd.read_file("nhru.gpkg")
    >>> batched = spatial_batch(fabric, batch_size=200)
    >>> batched["batch_id"].nunique()
    4  # for ~765 features with batch_size=200

    See Also
    --------
    _recursive_bisect : The recursive partitioning algorithm.
    hydro_param.pipeline : Uses batch IDs to iterate over spatial groups.
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
