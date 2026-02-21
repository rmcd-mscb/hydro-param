"""Core processing: zonal statistics via gdptools ZonalGen.

Wraps gdptools UserTiffData + ZonalGen with support for both
continuous and categorical variables. See design.md section 11.5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import geopandas as gpd
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

ZonalEngine = Literal["serial", "parallel", "dask", "exactextract"]


class Processor(Protocol):
    """Protocol for spatial processing strategies."""

    def process(
        self,
        fabric: gpd.GeoDataFrame,
        tiff_path: Path,
        variable_name: str,
        id_field: str,
        *,
        engine: ZonalEngine = "exactextract",
        statistics: list[str] | None = None,
        categorical: bool = False,
        source_crs: str | None = None,
        x_coord: str = "x",
        y_coord: str = "y",
    ) -> pd.DataFrame: ...


class ZonalProcessor:
    """Zonal statistics using gdptools ZonalGen.

    Computes area-weighted statistics of a raster variable over
    polygon features using the specified engine (default: exactextract).
    """

    def process(
        self,
        fabric: gpd.GeoDataFrame,
        tiff_path: Path,
        variable_name: str,
        id_field: str,
        *,
        engine: ZonalEngine = "exactextract",
        statistics: list[str] | None = None,
        categorical: bool = False,
        source_crs: str | None = None,
        x_coord: str = "x",
        y_coord: str = "y",
    ) -> pd.DataFrame:
        """Compute zonal statistics for a raster variable.

        Parameters
        ----------
        fabric : gpd.GeoDataFrame
            Target polygons.
        tiff_path : Path
            Path to the GeoTIFF for this variable.
        variable_name : str
            Name of the variable being processed.
        id_field : str
            Column name for feature IDs in the fabric.
        engine : ZonalEngine
            gdptools zonal engine (``"exactextract"``, ``"serial"``,
            ``"parallel"``, ``"dask"``).
        statistics : list[str] or None
            Which statistics to return. Default is ``["mean"]``.
        categorical : bool
            If True, compute class fractions instead of continuous stats.
        source_crs : str or None
            Source dataset CRS from the dataset registry. If None, reads
            CRS from the GeoTIFF metadata.
        x_coord : str
            Source x coordinate name (maps to gdptools ``source_x_coord``).
        y_coord : str
            Source y coordinate name (maps to gdptools ``source_y_coord``).

        Returns
        -------
        pd.DataFrame
            DataFrame with statistics columns, indexed by feature ID.
        """
        import rioxarray  # noqa: F401
        from gdptools import UserTiffData, ZonalGen

        if statistics is None:
            statistics = ["mean"]

        logger.info(
            "Computing zonal %s for '%s' (engine=%s, categorical=%s)",
            statistics,
            variable_name,
            engine,
            categorical,
        )

        # Use registry CRS if provided, otherwise read from GeoTIFF
        if source_crs is None:
            ds = cast(xr.DataArray, rioxarray.open_rasterio(tiff_path))
            source_crs = str(ds.rio.crs)
            ds.close()

        user_data = UserTiffData(
            source_ds=str(tiff_path),
            source_crs=source_crs,
            source_x_coord=x_coord,
            source_y_coord=y_coord,
            source_var=variable_name,
            target_gdf=fabric[[id_field, "geometry"]].copy(),
            target_id=id_field,
        )

        zonal = ZonalGen(
            user_data=user_data,
            zonal_engine=engine,
            zonal_writer="csv",
            out_path=str(tiff_path.parent),
        )
        result_df = zonal.calculate_zonal(categorical=categorical)

        logger.info(
            "  %s: %d features, columns=%s",
            variable_name,
            len(result_df),
            list(result_df.columns),
        )

        # For continuous variables, validate and select requested statistics
        if not categorical:
            available = set(result_df.columns)
            missing = [s for s in statistics if s not in available]
            if missing:
                logger.warning(
                    "Requested statistics %s not available for '%s'. Available: %s",
                    missing,
                    variable_name,
                    sorted(available),
                )
            selected = [s for s in statistics if s in available]
            if selected:
                result_df = result_df[selected]

        return result_df

    def process_nhgf_stac(
        self,
        fabric: gpd.GeoDataFrame,
        collection_id: str,
        variable_name: str,
        id_field: str,
        *,
        year: int | None = None,
        engine: ZonalEngine = "exactextract",
        statistics: list[str] | None = None,
        categorical: bool = False,
        band: int = 1,
    ) -> pd.DataFrame:
        """Compute zonal statistics from an NHGF STAC collection.

        Uses gdptools ``NHGFStacTiffData`` to read COGs directly from
        the NHGF STAC catalog (e.g. NLCD on OSN), bypassing any
        intermediate GeoTIFF download.

        Parameters
        ----------
        fabric : gpd.GeoDataFrame
            Target polygons.
        collection_id : str
            NHGF STAC collection identifier (e.g. ``"nlcd-LndCov"``).
        variable_name : str
            Variable / layer name within the collection.
        id_field : str
            Column name for feature IDs in the fabric.
        year : int or None
            Select a specific STAC item by year. If None, uses the
            first available item.
        engine : ZonalEngine
            gdptools zonal engine.
        statistics : list[str] or None
            Which statistics to return. Default is ``["mean"]``.
        categorical : bool
            If True, compute class fractions instead of continuous stats.
        band : int
            Raster band to read (default 1).

        Returns
        -------
        pd.DataFrame
            DataFrame of zonal statistics indexed by feature ID.
        """
        from gdptools import NHGFStacTiffData, ZonalGen
        from gdptools.helpers import get_stac_collection

        if statistics is None:
            statistics = ["mean"]

        # gdptools expects list[str | Timestamp | datetime | None] | None;
        # list is invariant so list[str] doesn't satisfy — cast to Any.
        source_time_period = cast(
            "list[Any] | None",
            [f"{year}-01-01", f"{year}-12-31"] if year is not None else None,
        )

        logger.info(
            "NHGF STAC zonal: '%s' collection='%s' year=%s engine=%s categorical=%s",
            variable_name,
            collection_id,
            year,
            engine,
            categorical,
        )

        collection = get_stac_collection(collection_id)

        nhgf_data = NHGFStacTiffData(
            source_collection=collection,
            source_var=variable_name,
            target_gdf=fabric[[id_field, "geometry"]].copy(),
            target_id=id_field,
            source_time_period=source_time_period,
            band=band,
        )

        zonal = ZonalGen(
            user_data=nhgf_data,
            zonal_engine=engine,
            zonal_writer="csv",
            out_path=".",
        )
        result_df = zonal.calculate_zonal(categorical=categorical)

        logger.info(
            "  %s: %d features, columns=%s",
            variable_name,
            len(result_df),
            list(result_df.columns),
        )

        # For continuous variables, validate and select requested statistics
        if not categorical:
            available = set(result_df.columns)
            missing = [s for s in statistics if s not in available]
            if missing:
                logger.warning(
                    "Requested statistics %s not available for '%s'. Available: %s",
                    missing,
                    variable_name,
                    sorted(available),
                )
            selected = [s for s in statistics if s in available]
            if selected:
                result_df = result_df[selected]

        return result_df


def get_processor(fabric: gpd.GeoDataFrame) -> Processor:
    """Select the appropriate processor for a fabric geometry type.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Target fabric GeoDataFrame.

    Returns
    -------
    Processor
        A processor compatible with the fabric geometry type.

    Raises
    ------
    ValueError
        If the fabric is empty or has unsupported geometry types.
    """
    if fabric.empty:
        raise ValueError("Fabric GeoDataFrame is empty; cannot select a processor.")

    geom_types = set(fabric.geometry.geom_type.unique())
    polygon_types = {"Polygon", "MultiPolygon"}
    if geom_types <= polygon_types:
        return ZonalProcessor()

    unsupported = geom_types - polygon_types
    raise ValueError(f"Unsupported geometry types: {', '.join(sorted(unsupported))}")
