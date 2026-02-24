"""Data access: STAC COG loading, local GeoTIFF loading, terrain derivation.

Handles fetching source data from various backends (STAC, local files)
and deriving variables (slope, aspect from elevation). See design.md
sections 6.12 and 11.4 for access strategy details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from hydro_param.dataset_registry import DatasetEntry

logger = logging.getLogger(__name__)

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

    Parameters
    ----------
    y_coords, x_coords : np.ndarray
        1-D coordinate arrays.
    is_geographic : bool
        True if coordinates are in degrees (geographic CRS).

    Returns
    -------
    dy_m, dx_m : float
        Cell sizes in meters.
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
    """Compute slope in degrees from an elevation DataArray.

    Uses Horn (1981) method via numpy.gradient.

    Parameters
    ----------
    elevation : xr.DataArray
        2-D elevation raster with spatial coordinates.
    method : str
        Derivation method. Only ``"horn"`` is supported.
    x_coord : str
        Name of the x coordinate dimension in the DataArray.
    y_coord : str
        Name of the y coordinate dimension in the DataArray.

    Returns
    -------
    xr.DataArray
        Slope in degrees, same shape and coordinates as input.
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
    """Compute aspect in degrees (clockwise from north) from elevation.

    Parameters
    ----------
    elevation : xr.DataArray
        2-D elevation raster with spatial coordinates.
    method : str
        Derivation method. Only ``"horn"`` is supported.
    x_coord : str
        Name of the x coordinate dimension in the DataArray.
    y_coord : str
        Name of the y coordinate dimension in the DataArray.

    Returns
    -------
    xr.DataArray
        Aspect in degrees [0, 360), same shape and coordinates.
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
# STAC COG data access
# ---------------------------------------------------------------------------


def fetch_stac_cog(
    entry: DatasetEntry,
    bbox: list[float],
    *,
    asset_key: str | None = None,
) -> xr.DataArray:
    """Query a STAC catalog and load COG(s) clipped to the bounding box.

    Handles multi-tile mosaicing when a bounding box spans multiple
    STAC items.

    Parameters
    ----------
    entry : DatasetEntry
        Registry entry with ``strategy="stac_cog"``.
    bbox : list[float]
        ``[west, south, east, north]`` in the dataset's CRS.
    asset_key : str or None
        Per-variable STAC asset key override. When provided, this is used
        instead of ``entry.asset_key``. Required for collections like
        ``gnatsgo-rasters`` where each variable is a separate asset.

    Returns
    -------
    xr.DataArray
        Raster data clipped to the bounding box.
    """
    import pystac_client
    import rioxarray  # noqa: F401

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
            logger.warning("No items with gsd=%d; using %d unfiltered items", entry.gsd, len(items))

    logger.info("Found %d STAC items for bbox", len(items))

    # Load and mosaic tiles
    resolved_key = asset_key or entry.asset_key
    arrays = []
    for item in items:
        asset = item.assets[resolved_key]
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
    """Check if a source string is a remote HTTP(S) URL.

    Only HTTP/HTTPS URLs are supported for direct remote access via GDAL
    vsicurl. Other remote schemes (s3://, gs://) require different GDAL
    virtual filesystem handlers and are not handled here.
    """
    return source.startswith(("http://", "https://"))


def fetch_local_tiff(
    entry: DatasetEntry,
    bbox: list[float],
    *,
    dataset_name: str = "unknown",
    variable_source: str | None = None,
) -> xr.DataArray:
    """Load a local GeoTIFF (or remote VRT/COG) clipped to the bounding box.

    Reads the file referenced by ``variable_source`` (per-variable override),
    ``entry.source`` (dataset-level), or raises an error if neither is set.
    Remote HTTP(S) URLs are opened directly via GDAL vsicurl — no local
    file existence check is performed.

    Note: The function name ``fetch_local_tiff`` corresponds to the
    ``strategy="local_tiff"`` enum value in the registry schema. The strategy
    supports both local paths and remote HTTP(S) URLs via GDAL vsicurl.

    Parameters
    ----------
    entry : DatasetEntry
        Registry entry with ``strategy="local_tiff"``.
    bbox : list[float]
        ``[west, south, east, north]`` in EPSG:4326.
    dataset_name : str
        Dataset name for use in error messages.
    variable_source : str or None
        Per-variable source path or URL. Takes precedence over
        ``entry.source`` when set.

    Returns
    -------
    xr.DataArray
        Raster data clipped to the bounding box.

    Raises
    ------
    ValueError
        If no source is available (neither variable_source nor entry.source).
    FileNotFoundError
        If the source is a local path that does not exist.
    RuntimeError
        If no data remains after clipping to the bounding box.
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
    """Save an xarray DataArray as a GeoTIFF.

    Parameters
    ----------
    da : xr.DataArray
        Raster data with CRS information.
    path : Path
        Output file path.

    Returns
    -------
    Path
        The output path (same as input).
    """
    import rioxarray  # noqa: F401

    # Remove _FillValue from attrs to avoid conflict with encoding
    clean = da.copy()
    clean.attrs = {k: v for k, v in da.attrs.items() if k != "_FillValue"}
    clean.encoding = {k: v for k, v in da.encoding.items() if k != "_FillValue"}
    clean.rio.to_raster(path)
    logger.debug("Saved GeoTIFF: %s (%s)", path, da.shape)
    return path


# ---------------------------------------------------------------------------
# ClimateR-Catalog helpers
# ---------------------------------------------------------------------------


def load_climr_catalog(
    catalog_url: str = CLIMR_CATALOG_URL,
) -> pd.DataFrame:
    """Load the ClimateR-Catalog parquet file.

    Parameters
    ----------
    catalog_url : str
        URL to the catalog parquet file.

    Returns
    -------
    pd.DataFrame
        The full ClimateR catalog.
    """
    logger.info("Loading ClimateR catalog from %s", catalog_url)
    return pd.read_parquet(catalog_url)


def build_climr_cat_dict(
    catalog: pd.DataFrame,
    catalog_id: str,
    variable_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Build ``source_cat_dict`` for ``ClimRCatData`` from the catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Full ClimateR catalog (from :func:`load_climr_catalog`).
    catalog_id : str
        Dataset identifier in the catalog (e.g. ``"gridmet"``).
    variable_names : list[str]
        Variables to extract (e.g. ``["pr", "tmmx"]``).

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of variable name → catalog row dict.

    Raises
    ------
    ValueError
        If a variable is not found in the catalog for the given id.
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
