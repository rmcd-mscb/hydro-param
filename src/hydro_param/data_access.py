"""Data access: STAC COG loading, local GeoTIFF loading, terrain derivation.

Provide unified functions for fetching raster source data from multiple
backends (Planetary Computer STAC, local GeoTIFFs, remote VRTs via GDAL
vsicurl) and for deriving terrain variables (slope, aspect) from elevation
grids using pure numpy.

The module supports three of the five processing strategies defined in the
pipeline architecture:

- ``stac_cog`` -- :func:`fetch_stac_cog` via Planetary Computer or USGS GDP STAC
- ``local_tiff`` -- :func:`fetch_local_tiff` for local files or HTTP(S) COGs
- ``climr_cat`` -- :func:`build_climr_cat_dict` for ClimateR-Catalog OPeNDAP

The remaining two strategies (``nhgf_stac`` static and temporal) are handled
directly in :mod:`hydro_param.processing` via gdptools ``NHGFStacTiffData``
and ``NHGFStacData`` classes.

See Also
--------
design.md : Sections 6.12 (local data preference) and 11.4 (access strategies).
hydro_param.processing : Zonal statistics and temporal aggregation.

References
----------
.. [1] design.md section 6.12 -- Local data over remote services.
.. [2] design.md section 11.4 -- Data access strategy details.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from hydro_param.classification import classify_usda_texture

if TYPE_CHECKING:
    from hydro_param.dataset_registry import DatasetEntry

logger = logging.getLogger(__name__)

_SENTINEL = object()

CLIMR_CATALOG_URL = (
    "https://github.com/mikejohnson51/climateR-catalogs/releases/download/June-2024/catalog.parquet"
)


# ---------------------------------------------------------------------------
# Terrain derivation (pure numpy, no external dependencies)
# ---------------------------------------------------------------------------


def _cell_sizes_meters(
    y_coords: np.ndarray, x_coords: np.ndarray, is_geographic: bool
) -> tuple[float, float]:
    """Compute cell sizes in meters from coordinate arrays.

    Convert raster cell spacing to meters so that gradient calculations
    produce physically correct slope and aspect values regardless of the
    source CRS. For geographic CRS (degrees), apply a latitude-dependent
    scale factor using a WGS-84 approximation (111,320 m/deg).

    Parameters
    ----------
    y_coords : np.ndarray
        1-D array of y (latitude or northing) coordinates.
    x_coords : np.ndarray
        1-D array of x (longitude or easting) coordinates.
    is_geographic : bool
        True if coordinates are in degrees (geographic CRS such as
        EPSG:4269 or EPSG:4326). False for projected CRS where
        native units are already meters.

    Returns
    -------
    dy_m : float
        Cell size in the y direction, in meters.
    dx_m : float
        Cell size in the x direction, in meters.

    Notes
    -----
    The latitude correction uses ``cos(center_lat)`` to approximate the
    east-west distance at the centroid latitude. This is adequate for the
    small spatial extents of individual processing batches but would
    introduce error for continent-scale grids.
    """
    dy = float(np.abs(np.median(np.diff(y_coords))))
    dx = float(np.abs(np.median(np.diff(x_coords))))

    if is_geographic:
        center_lat = np.radians(np.mean(y_coords))
        dy_m = dy * 111_320.0
        dx_m = dx * 111_320.0 * float(np.cos(center_lat))
    else:
        dy_m = dy
        dx_m = dx

    return dy_m, dx_m


def derive_slope(
    elevation: xr.DataArray,
    method: str = "horn",
    *,
    x_coord: str = "x",
    y_coord: str = "y",
) -> xr.DataArray:
    """Compute terrain slope in degrees from an elevation DataArray.

    Derive slope from a DEM raster using the Horn (1981) finite-difference
    method implemented via ``numpy.gradient``. The result is used by
    pywatershed derivation step 3 (topography) for parameters such as
    ``hru_slope``.

    Elevation units do not matter for slope computation as long as they
    are consistent with the cell size (both converted to meters internally
    via :func:`_cell_sizes_meters`).

    Parameters
    ----------
    elevation : xr.DataArray
        2-D elevation raster with spatial coordinates. May be in any CRS;
        geographic CRS coordinates (degrees) are automatically converted
        to meters for gradient computation.
    method : str
        Derivation method. Only ``"horn"`` is currently supported.
    x_coord : str
        Name of the x coordinate dimension in the DataArray.
    y_coord : str
        Name of the y coordinate dimension in the DataArray.

    Returns
    -------
    xr.DataArray
        Slope in degrees [0, 90], with the same shape, coordinates, and
        CRS as the input elevation. Output attributes include
        ``units="degrees"`` and ``long_name="Terrain slope"``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"horn"``.

    Notes
    -----
    The Horn method computes slope as ``arctan(sqrt(dz/dx² + dz/dy²))``,
    where the partial derivatives are estimated by ``numpy.gradient``
    using second-order central differences.

    References
    ----------
    .. [1] Horn, B.K.P. (1981). "Hill shading and the reflectance map."
       Proceedings of the IEEE, 69(1), 14-47.

    See Also
    --------
    derive_aspect : Compute terrain aspect from the same elevation input.
    _cell_sizes_meters : Convert coordinate spacing to meters.
    """
    if method != "horn":
        raise ValueError(f"Unsupported slope method: {method}")

    elev = elevation.values.astype(np.float64)
    y_coords = elevation.coords[y_coord].values
    x_coords = elevation.coords[x_coord].values

    has_crs = hasattr(elevation, "rio") and elevation.rio.crs
    is_geographic = has_crs and elevation.rio.crs.is_geographic
    dy_m, dx_m = _cell_sizes_meters(y_coords, x_coords, is_geographic)

    dz_dy, dz_dx = np.gradient(elev, dy_m, dx_m)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    result = elevation.copy(data=slope_deg)
    result.name = "slope"
    result.attrs = {"units": "degrees", "long_name": "Terrain slope"}
    return result


def derive_aspect(
    elevation: xr.DataArray,
    method: str = "horn",
    *,
    x_coord: str = "x",
    y_coord: str = "y",
) -> xr.DataArray:
    """Compute terrain aspect in degrees clockwise from north.

    Derive aspect (downslope direction) from a DEM raster using the Horn
    (1981) finite-difference method. The result is used by pywatershed
    derivation step 3 (topography) for parameters such as ``hru_aspect``.

    Aspect is measured as a compass bearing: 0 degrees = north, 90 = east,
    180 = south, 270 = west. Flat areas where both partial derivatives are
    zero will have an aspect of 0 degrees (north by convention).

    Parameters
    ----------
    elevation : xr.DataArray
        2-D elevation raster with spatial coordinates. May be in any CRS;
        geographic CRS coordinates (degrees) are automatically converted
        to meters for gradient computation.
    method : str
        Derivation method. Only ``"horn"`` is currently supported.
    x_coord : str
        Name of the x coordinate dimension in the DataArray.
    y_coord : str
        Name of the y coordinate dimension in the DataArray.

    Returns
    -------
    xr.DataArray
        Aspect in degrees [0, 360), with the same shape, coordinates, and
        CRS as the input elevation. Output attributes include
        ``units="degrees"`` and ``long_name="Terrain aspect (clockwise
        from north)"``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"horn"``.

    Notes
    -----
    Aspect is computed as ``atan2(-dz/dx, -dz/dy)`` (negated gradients
    give the descent direction), then converted from mathematical angle
    to compass bearing by adding 360 and taking modulo 360.

    References
    ----------
    .. [1] Horn, B.K.P. (1981). "Hill shading and the reflectance map."
       Proceedings of the IEEE, 69(1), 14-47.

    See Also
    --------
    derive_slope : Compute terrain slope from the same elevation input.
    _cell_sizes_meters : Convert coordinate spacing to meters.
    """
    if method != "horn":
        raise ValueError(f"Unsupported aspect method: {method}")

    elev = elevation.values.astype(np.float64)
    y_coords = elevation.coords[y_coord].values
    x_coords = elevation.coords[x_coord].values

    has_crs = hasattr(elevation, "rio") and elevation.rio.crs
    is_geographic = has_crs and elevation.rio.crs.is_geographic
    dy_m, dx_m = _cell_sizes_meters(y_coords, x_coords, is_geographic)

    dz_dy, dz_dx = np.gradient(elev, dy_m, dx_m)
    # Compass bearing: atan2(east_descent, north_descent)
    # Descent direction is opposite to gradient: (-dz_dx, -dz_dy)
    aspect_rad = np.arctan2(-dz_dx, -dz_dy)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = (aspect_deg + 360) % 360

    result = elevation.copy(data=aspect_deg)
    result.name = "aspect"
    result.attrs = {"units": "degrees", "long_name": "Terrain aspect (clockwise from north)"}
    return result


DERIVATION_FUNCTIONS = {
    "slope": derive_slope,
    "aspect": derive_aspect,
}


# ---------------------------------------------------------------------------
# Categorical derivation functions
# ---------------------------------------------------------------------------


def classify_usda_texture_raster(
    sand: xr.DataArray,
    silt: xr.DataArray,
    clay: xr.DataArray,
) -> xr.DataArray:
    """Classify sand/silt/clay percentage rasters into USDA texture classes.

    Thin wrapper around ``classify_usda_texture()`` that handles
    xarray DataArray I/O.  Returns a float64 raster with class codes
    (1--12) suitable for categorical zonal statistics.

    Parameters
    ----------
    sand : xr.DataArray
        Sand content as percentage (0--100), 2-D raster.
    silt : xr.DataArray
        Silt content as percentage (0--100), 2-D raster.
    clay : xr.DataArray
        Clay content as percentage (0--100), 2-D raster.

    Returns
    -------
    xr.DataArray
        Float64 raster with USDA texture class codes (1--12).
        Elements where any input is NaN remain NaN.

    See Also
    --------
    hydro_param.classification.classify_usda_texture : Core classifier.
    hydro_param.classification.USDA_TEXTURE_CLASSES : Code-to-name mapping.
    """
    codes = classify_usda_texture(
        sand.values.astype(np.float64).ravel(),
        silt.values.astype(np.float64).ravel(),
        clay.values.astype(np.float64).ravel(),
    )
    out = sand.copy(data=codes.reshape(sand.shape))
    out.name = "soil_texture"
    out.attrs = {"units": "class", "long_name": "USDA soil texture classification"}
    return out


CATEGORICAL_DERIVATION_FUNCTIONS: dict[str, Callable[..., xr.DataArray]] = {
    "usda_texture_triangle": classify_usda_texture_raster,
}


# ---------------------------------------------------------------------------
# STAC COG data access
# ---------------------------------------------------------------------------


def query_stac_items(
    entry: DatasetEntry,
    bbox: list[float],
) -> list[Any]:
    """Query a STAC catalog and return matching items for a bounding box.

    Handle client creation, optional Planetary Computer signing, spatial
    search, and GSD (ground sample distance) filtering. The returned items
    can be passed to ``fetch_stac_cog(..., items=...)`` to avoid repeated
    queries when multiple variables share the same STAC collection within
    a single processing batch.

    Enables STAC query reuse across variables, reducing redundant network
    calls during batch processing.

    Parameters
    ----------
    entry : DatasetEntry
        Registry entry with ``strategy="stac_cog"``. Must have non-None
        ``catalog_url`` and ``collection`` fields.
    bbox : list[float]
        Bounding box as ``[west, south, east, north]`` in EPSG:4326
        decimal degrees.

    Returns
    -------
    list[Any]
        Matching STAC items (signed if ``entry.sign`` is set). Items are
        ``pystac.Item`` objects suitable for asset access.

    Raises
    ------
    ValueError
        If ``entry.catalog_url`` or ``entry.collection`` is None.
    RuntimeError
        If no STAC items match the given collection and bounding box.

    Notes
    -----
    When ``entry.gsd`` is set, items are filtered to those matching the
    target ground sample distance (e.g., 10 m for 3DEP 1/3 arc-second).
    If no items match the GSD filter, all unfiltered items are returned
    with a warning.

    See Also
    --------
    fetch_stac_cog : Load and mosaic COG data from the returned items.
    """
    import pystac_client

    if entry.catalog_url is None:
        raise ValueError("stac_cog strategy requires 'catalog_url' on the dataset entry")
    if entry.collection is None:
        raise ValueError("stac_cog strategy requires 'collection' on the dataset entry")

    logger.info(
        "Querying STAC: catalog=%s collection=%s bbox=%s",
        entry.catalog_url,
        entry.collection,
        bbox,
    )

    modifier = None
    if entry.sign == "planetary_computer":
        import planetary_computer

        modifier = planetary_computer.sign_inplace

    client = pystac_client.Client.open(entry.catalog_url, modifier=modifier)

    search = client.search(
        collections=[entry.collection],
        bbox=bbox,
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError(f"No STAC items found for collection='{entry.collection}' bbox={bbox}")

    # Filter by GSD if specified
    if entry.gsd is not None:
        filtered = [i for i in items if i.properties.get("gsd") == entry.gsd]
        if filtered:
            items = filtered
        else:
            logger.warning(
                "No items with gsd=%d; using %d unfiltered items",
                entry.gsd,
                len(items),
            )

    logger.info("Found %d STAC items for bbox", len(items))
    return items


def fetch_stac_cog(
    entry: DatasetEntry,
    bbox: list[float],
    *,
    asset_key: str | None = None,
    items: list[Any] | None = None,
) -> xr.DataArray:
    """Load Cloud Optimized GeoTIFF(s) from a STAC catalog, clipped to bbox.

    Fetch one or more COG tiles from a STAC collection (e.g., 3DEP on
    Planetary Computer, gNATSGO rasters), clip each to the bounding box,
    and mosaic them into a single DataArray. This is the primary data
    loader for the ``stac_cog`` processing strategy.

    Multi-tile mosaicing is handled automatically when a bounding box
    spans multiple STAC items (common at DEM tile boundaries).

    Parameters
    ----------
    entry : DatasetEntry
        Registry entry with ``strategy="stac_cog"``. Must have non-None
        ``catalog_url`` and ``collection`` fields.
    bbox : list[float]
        Bounding box as ``[west, south, east, north]`` in EPSG:4326
        decimal degrees. Typically comes from
        :func:`hydro_param.pipeline._buffered_bbox`.
    asset_key : str or None
        Per-variable STAC asset key override. When not ``None``, this is
        used instead of ``entry.asset_key``. Necessary for collections
        like ``gnatsgo-rasters`` where each variable is a separate named
        asset (e.g., ``"sandtotal_r"``) rather than a single ``"data"``
        asset.
    items : list[Any] or None
        Pre-fetched STAC items from :func:`query_stac_items`. When
        provided, the STAC query is skipped entirely, saving redundant
        network calls when processing multiple variables from the same
        collection within a batch.

    Returns
    -------
    xr.DataArray
        2-D raster data clipped to the bounding box, in the source CRS
        (typically EPSG:4269 for 3DEP, EPSG:5070 for gNATSGO). The band
        dimension is squeezed out.

    Raises
    ------
    KeyError
        If ``asset_key`` (or ``entry.asset_key``) is not found in any
        STAC item. The error message lists available data assets.
    RuntimeError
        If no data remains after clipping all tiles to the bounding box.

    Notes
    -----
    Tiles that do not overlap the bounding box after precise clipping are
    silently skipped (logged at DEBUG level). This is expected when STAC
    search returns items whose coarse footprints overlap but whose actual
    data does not.

    See Also
    --------
    query_stac_items : Pre-query STAC items for reuse across variables.
    fetch_local_tiff : Load data from local GeoTIFF files.
    """
    import rioxarray  # noqa: F401

    if items is None:
        items = query_stac_items(entry, bbox)

    # Load and mosaic tiles
    resolved_key = asset_key if asset_key is not None else entry.asset_key
    logger.debug(
        "Using asset key '%s' (override=%s, default=%s)",
        resolved_key,
        asset_key,
        entry.asset_key,
    )
    arrays = []
    for item in items:
        try:
            asset = item.assets[resolved_key]
        except KeyError:
            available = sorted(k for k, a in item.assets.items() if a.roles and "data" in a.roles)
            raise KeyError(
                f"Asset key '{resolved_key}' not found in STAC item '{item.id}' "
                f"(collection='{entry.collection}'). "
                f"Available data assets: {available}. "
                f"Check the 'asset_key' field in your dataset registry."
            ) from None
        da = cast(xr.DataArray, rioxarray.open_rasterio(asset.href, masked=True))
        da = da.squeeze("band", drop=True)
        try:
            da = da.rio.clip_box(
                minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3], crs="EPSG:4326"
            )
        except (rioxarray.exceptions.NoDataInBounds, ValueError):
            # Item may not overlap after precise clipping
            logger.debug("Tile %s has no data in bbox, skipping", item.id)
            continue
        if da.size > 0:
            arrays.append(da)

    if not arrays:
        raise RuntimeError(f"No data after clipping to bbox={bbox}")

    if len(arrays) == 1:
        result = arrays[0]
    else:
        from rioxarray.merge import merge_arrays

        result = merge_arrays(arrays)
        logger.info("Mosaiced %d tiles", len(arrays))

    logger.info("Loaded raster: shape=%s, crs=%s", result.shape, result.rio.crs)
    return result


# ---------------------------------------------------------------------------
# Local GeoTIFF data access
# ---------------------------------------------------------------------------


def _is_remote_url(source: str) -> bool:
    """Return True if the source string is an HTTP or HTTPS URL.

    Only HTTP/HTTPS URLs are supported for direct remote access via GDAL
    vsicurl. Other remote schemes (``s3://``, ``gs://``) require different
    GDAL virtual filesystem handlers and are not handled here.
    """
    return source.startswith(("http://", "https://"))


def fetch_local_tiff(
    entry: DatasetEntry,
    bbox: list[float],
    *,
    dataset_name: str = "unknown",
    variable_source: str | None = None,
) -> xr.DataArray:
    """Load a local GeoTIFF or remote COG/VRT clipped to a bounding box.

    Read a raster file referenced by ``variable_source`` (per-variable
    override) or ``entry.source`` (dataset-level default), clip it to the
    bounding box, and return a 2-D DataArray. This is the primary data
    loader for the ``local_tiff`` processing strategy.

    Despite the name, this function supports both local file paths and
    remote HTTP(S) URLs (opened via GDAL vsicurl). The function name
    matches the ``strategy="local_tiff"`` enum value in the dataset
    registry schema.

    Source resolution order:

    1. ``variable_source`` (per-variable override, e.g., for POLARIS where
       each soil property is a separate file)
    2. ``entry.source`` (dataset-level path from pipeline config)
    3. Raise ``ValueError`` with download instructions if available

    Parameters
    ----------
    entry : DatasetEntry
        Registry entry with ``strategy="local_tiff"``.
    bbox : list[float]
        Bounding box as ``[west, south, east, north]`` in EPSG:4326
        decimal degrees.
    dataset_name : str
        Dataset name for use in error messages and download instructions.
    variable_source : str or None
        Per-variable source path or URL. Takes precedence over
        ``entry.source`` when set. Common for datasets where each
        variable is stored in a separate file (e.g., POLARIS soil
        properties).

    Returns
    -------
    xr.DataArray
        2-D raster data clipped to the bounding box, in the source CRS.
        The band dimension is squeezed out.

    Raises
    ------
    ValueError
        If no source is available (neither ``variable_source`` nor
        ``entry.source``). The error message includes download
        instructions when ``entry.download`` metadata is available.
    FileNotFoundError
        If the source is a local path that does not exist on disk.
    RuntimeError
        If no data remains after clipping to the bounding box, or if
        a remote URL fails to open.

    See Also
    --------
    fetch_stac_cog : Load data from STAC-cataloged COGs.
    save_to_geotiff : Write a DataArray back to GeoTIFF format.
    """
    import rioxarray  # noqa: F401
    from rioxarray.exceptions import NoDataInBounds

    # Resolve source: per-variable override > dataset-level
    source = variable_source if variable_source is not None else entry.source

    if source is None:
        msg = (
            f"Dataset '{dataset_name}' requires a source path or URL "
            f"(strategy: local_tiff) but neither variable_source nor entry.source is set."
        )
        if entry.download:
            if entry.download.files:
                msg += (
                    f"\n\nThis dataset has {len(entry.download.files)} "
                    f"downloadable files. Run:\n"
                    f"  hydro-param datasets info {dataset_name}"
                )
            elif entry.download.url_template:
                start, end = entry.download.year_range
                n_vars = len(entry.download.variables_available)
                msg += (
                    f"\n\nThis dataset has templated downloads "
                    f"({end - start + 1} years x {n_vars} variables). Run:\n"
                    f"  hydro-param datasets info {dataset_name}"
                )
            elif entry.download.url:
                msg += f"\n\nDownload from: {entry.download.url}"
                if entry.download.size_gb:
                    msg += f"\nExpected size: ~{entry.download.size_gb} GB"
                if entry.download.notes:
                    msg += f"\n{entry.download.notes.strip()}"
            msg += (
                f"\n\nSet 'source' in your pipeline config:\n"
                f"  datasets:\n"
                f"    - name: {dataset_name}\n"
                f"      source: /path/to/downloaded/file.tif"
            )
        raise ValueError(msg)

    if _is_remote_url(source):
        logger.info("Loading remote raster: %s bbox=%s", source, bbox)
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {source_path}")
        logger.info("Loading local GeoTIFF: %s bbox=%s", source_path, bbox)

    try:
        da = cast(xr.DataArray, rioxarray.open_rasterio(source, masked=True))
    except Exception as exc:
        if _is_remote_url(source):
            raise RuntimeError(
                f"Failed to open remote raster for dataset '{dataset_name}': {source}\n"
                f"Original error: {exc}"
            ) from exc
        raise
    da = da.squeeze("band", drop=True)

    try:
        da = da.rio.clip_box(
            minx=bbox[0],
            miny=bbox[1],
            maxx=bbox[2],
            maxy=bbox[3],
            crs="EPSG:4326",
        )
    except (NoDataInBounds, ValueError) as exc:
        raise RuntimeError(f"No data in bbox={bbox} for {source}") from exc

    if da.size == 0:
        raise RuntimeError(f"Empty raster after clipping to bbox={bbox}")

    logger.info("Loaded raster: shape=%s, crs=%s", da.shape, da.rio.crs)
    return da


def save_to_geotiff(da: xr.DataArray, path: Path) -> Path:
    """Write an xarray DataArray to a GeoTIFF file.

    Save raster data as a single-band GeoTIFF via rioxarray. This is used
    by the pipeline to cache intermediate raster results (e.g., clipped
    source data for a batch) before passing them to gdptools ZonalGen.

    The function temporarily removes ``_FillValue`` from attributes and
    encoding to avoid conflicts with rioxarray's internal encoding, then
    restores them afterward. This avoids a full ``.copy()`` of the
    DataArray, which was identified as a memory bottleneck in PR #72.

    Parameters
    ----------
    da : xr.DataArray
        Raster data with CRS information (via rioxarray accessor).
    path : Path
        Output file path. Parent directory must exist.

    Returns
    -------
    Path
        The output path (same as input), for convenience in chaining.

    Notes
    -----
    Uses a sentinel-based pop/restore pattern instead of ``da.copy()``
    to avoid doubling memory usage for large rasters (e.g., gNATSGO
    tiles at ~1.25 GB each). See PR #72 for the memory optimization
    rationale.

    See Also
    --------
    fetch_local_tiff : Load a GeoTIFF back into a DataArray.
    fetch_stac_cog : Load COG data from STAC.
    """
    import rioxarray  # noqa: F401

    # Temporarily remove _FillValue to avoid conflict with encoding,
    # then restore — avoids a full .copy() of the DataArray.
    fill_in_attrs = da.attrs.pop("_FillValue", _SENTINEL)
    fill_in_encoding = da.encoding.pop("_FillValue", _SENTINEL)
    try:
        da.rio.to_raster(path)
    finally:
        if fill_in_attrs is not _SENTINEL:
            da.attrs["_FillValue"] = fill_in_attrs
        if fill_in_encoding is not _SENTINEL:
            da.encoding["_FillValue"] = fill_in_encoding
    logger.debug("Saved GeoTIFF: %s (%s)", path, da.shape)
    return path


# ---------------------------------------------------------------------------
# ClimateR-Catalog helpers
# ---------------------------------------------------------------------------


def load_climr_catalog(
    catalog_url: str = CLIMR_CATALOG_URL,
) -> pd.DataFrame:
    """Load the ClimateR-Catalog as a pandas DataFrame.

    Fetch and parse the ClimateR-Catalog parquet file, which indexes
    1,700+ climate and environmental datasets with OPeNDAP endpoints.
    The catalog is used by the ``climr_cat`` processing strategy to
    resolve variable-level metadata for gdptools ``ClimRCatData``.

    Parameters
    ----------
    catalog_url : str
        URL to the catalog parquet file. Defaults to the June 2024
        release from the mikejohnson51/climateR-catalogs repository.

    Returns
    -------
    pd.DataFrame
        The full ClimateR catalog with columns including ``id``,
        ``variable``, ``URL``, ``units``, and spatial metadata.

    References
    ----------
    .. [1] Johnson, J.M. climateR-catalogs.
       https://github.com/mikejohnson51/climateR-catalogs

    See Also
    --------
    build_climr_cat_dict : Build per-variable dicts from the loaded catalog.
    """
    logger.info("Loading ClimateR catalog from %s", catalog_url)
    return pd.read_parquet(catalog_url)


def build_climr_cat_dict(
    catalog: pd.DataFrame,
    catalog_id: str,
    variable_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Build a ``source_cat_dict`` for gdptools ``ClimRCatData``.

    Extract per-variable metadata rows from the ClimateR-Catalog and
    format them as the dictionary structure expected by gdptools
    ``ClimRCatData`` constructor. Each entry contains the OPeNDAP URL,
    variable name, CRS, and spatial extent needed for data access.

    Parameters
    ----------
    catalog : pd.DataFrame
        Full ClimateR catalog loaded by :func:`load_climr_catalog`.
    catalog_id : str
        Dataset identifier in the catalog (e.g., ``"gridmet"`` for
        gridMET via OPeNDAP, ``"daymet"`` for Daymet).
    variable_names : list[str]
        Variables to extract (e.g., ``["pr", "tmmx"]`` for gridMET
        precipitation and max temperature).

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of variable name to catalog row as a dictionary. Each
        value contains all columns from the catalog for that variable,
        ready for ``ClimRCatData(source_cat_dict=...)``.

    Raises
    ------
    ValueError
        If ``catalog_id`` is not found in the catalog, or if any
        variable in ``variable_names`` is not available for the given
        catalog ID. The error message lists available options.

    See Also
    --------
    load_climr_catalog : Load the catalog DataFrame.
    TemporalProcessor.process_climr_cat : Use the dict for temporal processing.
    """
    if catalog_id not in catalog["id"].values:
        available_ids = sorted(catalog["id"].unique())
        raise ValueError(
            f"Catalog ID '{catalog_id}' not found in ClimateR catalog. Available: {available_ids}"
        )

    source_cat_dict: dict[str, dict[str, Any]] = {}
    for var_name in variable_names:
        matches = catalog[(catalog["id"] == catalog_id) & (catalog["variable"] == var_name)]
        if matches.empty:
            available = sorted(catalog.loc[catalog["id"] == catalog_id, "variable"].unique())
            raise ValueError(
                f"Variable '{var_name}' not found in ClimateR catalog for '{catalog_id}'. "
                f"Available: {available}"
            )
        source_cat_dict[var_name] = matches.iloc[0].to_dict()
    return source_cat_dict
