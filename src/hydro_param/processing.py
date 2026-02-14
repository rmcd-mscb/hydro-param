"""Core processing: zonal statistics via gdptools ZonalGen.

Wraps gdptools UserTiffData + ZonalGen with support for both
continuous and categorical variables. See design.md section 11.5.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)


class Processor(Protocol):
    """Protocol for spatial processing strategies."""

    def process(
        self,
        fabric: gpd.GeoDataFrame,
        tiff_path: Path,
        variable_name: str,
        id_field: str,
        *,
        engine: str = "exactextract",
        statistics: list[str] | None = None,
        categorical: bool = False,
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
        engine: str = "exactextract",
        statistics: list[str] | None = None,
        categorical: bool = False,
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
        engine : str
            gdptools zonal engine (``"exactextract"``, ``"serial"``).
        statistics : list[str] or None
            Which statistics to return. Default is ``["mean"]``.
        categorical : bool
            If True, compute class fractions instead of continuous stats.

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

        # Read CRS from GeoTIFF for UserTiffData
        ds = rioxarray.open_rasterio(tiff_path)
        source_crs = str(ds.rio.crs)
        ds.close()

        user_data = UserTiffData(
            source_ds=str(tiff_path),
            source_crs=source_crs,
            source_x_coord="x",
            source_y_coord="y",
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
