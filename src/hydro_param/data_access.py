"""Data access: STAC COG loading, terrain derivation, raster I/O.

Handles fetching source data from various backends (STAC, local files)
and deriving variables (slope, aspect from elevation). See design.md
section 11.4 for STAC integration details.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from hydro_param.dataset_registry import DatasetEntry

logger = logging.getLogger(__name__)


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


def derive_slope(elevation: xr.DataArray, method: str = "horn") -> xr.DataArray:
    """Compute slope in degrees from an elevation DataArray.

    Uses Horn (1981) method via numpy.gradient.

    Parameters
    ----------
    elevation : xr.DataArray
        2-D elevation raster with y/x coordinates.
    method : str
        Derivation method. Only ``"horn"`` is supported.

    Returns
    -------
    xr.DataArray
        Slope in degrees, same shape and coordinates as input.
    """
    if method != "horn":
        raise ValueError(f"Unsupported slope method: {method}")

    elev = elevation.values.astype(np.float64)
    y_coords = elevation.coords["y"].values
    x_coords = elevation.coords["x"].values

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


def derive_aspect(elevation: xr.DataArray, method: str = "horn") -> xr.DataArray:
    """Compute aspect in degrees (clockwise from north) from elevation.

    Parameters
    ----------
    elevation : xr.DataArray
        2-D elevation raster with y/x coordinates.
    method : str
        Derivation method. Only ``"horn"`` is supported.

    Returns
    -------
    xr.DataArray
        Aspect in degrees [0, 360), same shape and coordinates.
    """
    if method != "horn":
        raise ValueError(f"Unsupported aspect method: {method}")

    elev = elevation.values.astype(np.float64)
    y_coords = elevation.coords["y"].values
    x_coords = elevation.coords["x"].values

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

    Returns
    -------
    xr.DataArray
        Raster data clipped to the bounding box.
    """
    import pystac_client
    import rioxarray  # noqa: F401

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
    arrays = []
    for item in items:
        asset = item.assets[entry.asset_key]
        da = rioxarray.open_rasterio(asset.href, masked=True)
        da = da.squeeze("band", drop=True)
        try:
            da = da.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        except Exception:
            # Item may not overlap after precise clipping
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
    logger.info("Saved GeoTIFF: %s (%s)", path, da.shape)
    return path
