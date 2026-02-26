"""Core processing: zonal statistics and temporal aggregation via gdptools.

Wrap gdptools processing classes (``UserTiffData``, ``NHGFStacTiffData``,
``NHGFStacData``, ``ClimRCatData``) with ``ZonalGen``, ``WeightGen``, and
``AggGen`` to compute area-weighted statistics of raster data over polygon
features. Support both continuous variables (mean, median, etc.) and
categorical variables (class fraction extraction).

This module provides two processor classes:

- :class:`ZonalProcessor` -- static raster-on-polygon zonal statistics
  using ``ZonalGen`` with the exactextract engine. Supports local GeoTIFF,
  STAC COG, and NHGF STAC static strategies.
- :class:`TemporalProcessor` -- time-varying aggregation using the
  ``WeightGen`` + ``AggGen`` pipeline. Supports NHGF STAC temporal and
  ClimateR-Catalog (OPeNDAP) strategies.

The factory function :func:`get_processor` selects the appropriate processor
based on fabric geometry type.

Notes
-----
gdptools ``ZonalGen`` internally reprojects target polygons into the source
CRS, so no CRS alignment is needed in hydro-param. The exactextract engine
produces fractional coverage weights for partial cell overlap.

See Also
--------
design.md : Section 11.5 (processing architecture).
hydro_param.data_access : Data loading functions that produce the GeoTIFFs
    consumed by :class:`ZonalProcessor`.
hydro_param.pipeline : Orchestrator that calls these processors per batch.
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
    """Protocol defining the interface for spatial processing strategies.

    All processor implementations must provide a ``process`` method that
    computes zonal statistics for a raster variable over polygon features.
    This protocol enables the pipeline to work with different processing
    backends (currently only :class:`ZonalProcessor` for polygon targets;
    a future grid processor for raster-on-raster via xesmf is planned).

    See Also
    --------
    ZonalProcessor : Concrete implementation using gdptools ZonalGen.
    get_processor : Factory function that selects the appropriate processor.
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
    ) -> pd.DataFrame: ...


class ZonalProcessor:
    """Compute area-weighted zonal statistics via gdptools ZonalGen.

    Wrap the gdptools ``UserTiffData`` + ``ZonalGen`` pipeline to compute
    statistics (mean, median, majority, class fractions) of raster data
    over polygon features. This is the core processing class for the
    ``stac_cog``, ``local_tiff``, and ``nhgf_stac`` (static) strategies.

    The class is stateless -- each call to :meth:`process` or
    :meth:`process_nhgf_stac` creates fresh gdptools objects. This design
    allows safe reuse across batches and variables without accumulated state.

    Notes
    -----
    The default ``exactextract`` engine computes fractional coverage weights
    for partial cell overlap, producing accurate area-weighted statistics
    even when polygon boundaries do not align with raster cell edges.

    For categorical variables (e.g., NLCD land cover classes), set
    ``categorical=True`` to get per-class area fractions instead of
    continuous statistics.

    See Also
    --------
    TemporalProcessor : For time-varying datasets (SNODAS, gridMET).
    hydro_param.data_access.fetch_stac_cog : Produces the GeoTIFFs consumed here.
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
        """Compute zonal statistics for a raster variable over polygon features.

        Read a GeoTIFF, construct a gdptools ``UserTiffData`` + ``ZonalGen``
        pipeline, and compute area-weighted statistics for each polygon in
        the target fabric. This is the workhorse method called by the
        pipeline's ``_process_batch`` for ``stac_cog`` and ``local_tiff``
        strategies.

        Parameters
        ----------
        fabric : gpd.GeoDataFrame
            Target polygon features (a batch subset of the full fabric).
            Must contain at least the ``id_field`` column and ``geometry``.
        tiff_path : Path
            Path to the GeoTIFF for this variable. Typically a
            batch-clipped raster saved by :func:`data_access.save_to_geotiff`.
        variable_name : str
            Name of the variable being processed (e.g., ``"elevation"``,
            ``"sandtotal_r"``). Passed to gdptools as ``source_var``.
        id_field : str
            Column name for feature IDs in the fabric (e.g., ``"nhm_id"``).
            Used as the index in the output DataFrame.
        engine : ZonalEngine
            gdptools zonal engine. One of ``"exactextract"`` (default,
            recommended), ``"serial"``, ``"parallel"``, or ``"dask"``.
        statistics : list[str] or None
            Which statistics to compute and return (e.g., ``["mean"]``,
            ``["mean", "median"]``). Defaults to ``["mean"]``. Ignored
            when ``categorical=True``.
        categorical : bool
            If True, compute per-class area fractions instead of continuous
            statistics. Used for land cover variables (e.g., NLCD classes).
        source_crs : str or None
            Source dataset CRS as a string (e.g., ``"EPSG:5070"``). If
            None, the CRS is read from the GeoTIFF metadata via rioxarray.
        x_coord : str
            Name of the x coordinate in the source raster. Maps to
            gdptools ``source_x_coord``.
        y_coord : str
            Name of the y coordinate in the source raster. Maps to
            gdptools ``source_y_coord``.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per polygon feature. For continuous
            variables, columns are the requested statistics (e.g.,
            ``"mean"``). For categorical variables, columns are class
            fraction values as returned by gdptools.

        Notes
        -----
        gdptools ``ZonalGen`` handles CRS alignment internally by
        reprojecting the target polygons into the source raster CRS.
        No manual CRS transformation is needed in hydro-param.
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

        Use gdptools ``NHGFStacTiffData`` to read COGs directly from the
        NHGF STAC catalog hosted on OSN (Open Storage Network), bypassing
        any intermediate GeoTIFF download. This is the processing path for
        the ``nhgf_stac`` (static) strategy, used primarily for NLCD
        Annual land cover data (6 collections: ``nlcd-LndCov``,
        ``nlcd-FctImp``, etc.).

        Unlike :meth:`process`, this method does not require a local
        GeoTIFF -- gdptools handles the remote COG access, subsetting,
        and zonal computation in a single pipeline.

        Parameters
        ----------
        fabric : gpd.GeoDataFrame
            Target polygon features. Must contain at least the
            ``id_field`` column and ``geometry``.
        collection_id : str
            NHGF STAC collection identifier (e.g., ``"nlcd-LndCov"``
            for NLCD land cover, ``"nlcd-FctImp"`` for fractional
            impervious surface).
        variable_name : str
            Variable / layer name within the collection (passed to
            gdptools as ``source_var``).
        id_field : str
            Column name for feature IDs in the fabric (e.g., ``"nhm_id"``).
        year : int or None
            Select a specific STAC item by year (e.g., ``2019`` for
            NLCD 2019). Converted to a time period filter
            ``["YYYY-01-01", "YYYY-12-31"]``. If None, uses the first
            available item in the collection.
        engine : ZonalEngine
            gdptools zonal engine. Default ``"exactextract"``.
        statistics : list[str] or None
            Which statistics to compute. Defaults to ``["mean"]``.
            Ignored when ``categorical=True``.
        categorical : bool
            If True, compute per-class area fractions instead of
            continuous statistics. Required for NLCD land cover classes.
        band : int
            Raster band to read from COG files. Default is 1 (single-band).

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per polygon feature, indexed by
            ``id_field``. Columns depend on ``categorical`` flag.

        See Also
        --------
        process : Zonal stats from local GeoTIFF files.
        TemporalProcessor.process_nhgf_stac : Temporal aggregation from
            NHGF STAC Zarr collections.
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


class TemporalProcessor:
    """Compute temporal aggregation via gdptools WeightGen + AggGen.

    Handle time-varying datasets (e.g., SNODAS, CONUS404-BA, gridMET) by
    computing spatial intersection weights and then applying those weights
    to aggregate gridded time series onto polygon features. Output is an
    ``xr.Dataset`` with ``(time, features)`` dimensions.

    This class supports two data access backends:

    - :meth:`process_nhgf_stac` -- NHGF STAC Zarr collections via
      ``NHGFStacData`` (for SNODAS, CONUS404-BA)
    - :meth:`process_climr_cat` -- ClimateR-Catalog OPeNDAP datasets via
      ``ClimRCatData`` (for gridMET, Daymet)

    The class is stateless -- each method call creates fresh gdptools objects.

    Notes
    -----
    The processing pipeline is: data source -> ``WeightGen`` (compute
    area-weighted intersection matrix) -> ``AggGen`` (apply weights to
    aggregate time series). Weight generation uses EPSG:6931 (LAEA) by
    default for accurate area computation.

    The pipeline supports calendar-year splitting for multi-year datasets
    at the orchestrator level (see :mod:`hydro_param.pipeline`).

    See Also
    --------
    ZonalProcessor : For static raster-on-polygon zonal statistics.
    hydro_param.data_access.load_climr_catalog : Load the ClimateR catalog.
    """

    def process_nhgf_stac(
        self,
        fabric: gpd.GeoDataFrame,
        collection_id: str,
        variable_names: list[str],
        id_field: str,
        time_period: list[str],
        *,
        stat_method: str = "mean",
        weight_gen_crs: int = 6931,
    ) -> xr.Dataset:
        """Compute temporal aggregation from an NHGF STAC Zarr collection.

        Fetch gridded time series from the NHGF STAC catalog (Zarr format),
        compute polygon intersection weights, and aggregate to polygon
        features. Used for the ``nhgf_stac`` (temporal) strategy with
        datasets like SNODAS (snow water equivalent) and CONUS404-BA
        (basin-averaged atmospheric forcing).

        Parameters
        ----------
        fabric : gpd.GeoDataFrame
            Target polygon features. Must contain at least the
            ``id_field`` column and ``geometry``.
        collection_id : str
            NHGF STAC collection identifier (e.g., ``"snodas-swe"``).
        variable_names : list[str]
            Variables to process from the collection (e.g.,
            ``["swe_mm"]``). All variables share the same intersection
            weights.
        id_field : str
            Column name for feature IDs in the fabric (e.g., ``"nhm_id"``).
        time_period : list[str]
            Two-element list ``[start, end]`` of ISO date strings
            (e.g., ``["2020-01-01", "2020-12-31"]``).
        stat_method : str
            Aggregation statistic applied per time step. One of
            ``"mean"``, ``"median"``, ``"min"``, ``"max"``, etc.
        weight_gen_crs : int
            EPSG code for the CRS used during weight generation.
            Default is 6931 (LAEA -- Lambert Azimuthal Equal Area),
            which provides accurate area-weighted intersections.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions ``(time, <id_field>)`` and one
            data variable per requested variable name. Values are
            area-weighted aggregates in source units.

        See Also
        --------
        process_climr_cat : Temporal aggregation from ClimateR-Catalog.
        ZonalProcessor.process_nhgf_stac : Static zonal stats from NHGF STAC.
        """
        from gdptools import AggGen, NHGFStacData, WeightGen
        from gdptools.helpers import get_stac_collection

        logger.info(
            "NHGF STAC temporal: collection='%s' vars=%s period=%s stat=%s",
            collection_id,
            variable_names,
            time_period,
            stat_method,
        )

        collection = get_stac_collection(collection_id)

        nhgf_data = NHGFStacData(
            source_collection=collection,
            source_var=variable_names,
            target_gdf=fabric[[id_field, "geometry"]].copy(),
            target_id=id_field,
            source_time_period=cast("list[Any]", time_period),
        )
        wg = WeightGen(
            user_data=cast("Any", nhgf_data),
            method="serial",
            weight_gen_crs=weight_gen_crs,
        )
        weights = wg.calculate_weights()
        ag = AggGen(
            user_data=cast("Any", nhgf_data),
            stat_method=cast("Any", stat_method),
            agg_engine="serial",
            agg_writer="none",
            weights=weights,
        )
        _gdf, ds = ag.calculate_agg()

        logger.info(
            "  Temporal result: %d vars, %d time steps, %d features",
            len(ds.data_vars),
            ds.sizes.get("time", 0),
            ds.sizes.get(id_field, len(fabric)),
        )
        return ds

    def process_climr_cat(
        self,
        fabric: gpd.GeoDataFrame,
        catalog_id: str,
        variable_names: list[str],
        id_field: str,
        time_period: list[str],
        *,
        stat_method: str = "mean",
        weight_gen_crs: int = 6931,
    ) -> xr.Dataset:
        """Compute temporal aggregation from a ClimateR-Catalog OPeNDAP dataset.

        Fetch gridded time series via OPeNDAP from datasets indexed in the
        ClimateR-Catalog (e.g., gridMET, Daymet), compute polygon
        intersection weights, and aggregate to polygon features. Used for
        the ``climr_cat`` processing strategy.

        This is the preferred method for gridMET access because the gridMET
        copy on the USGS GDP STAC is not kept up to date, whereas OPeNDAP
        serves the canonical source.

        Parameters
        ----------
        fabric : gpd.GeoDataFrame
            Target polygon features. Must contain at least the
            ``id_field`` column and ``geometry``.
        catalog_id : str
            ClimateR catalog identifier (e.g., ``"gridmet"`` for
            gridMET, ``"daymet"`` for Daymet).
        variable_names : list[str]
            Variables to process (e.g., ``["pr", "tmmx"]`` for gridMET
            precipitation and max temperature). Must match variable
            names in the ClimateR catalog.
        id_field : str
            Column name for feature IDs in the fabric (e.g., ``"nhm_id"``).
        time_period : list[str]
            Two-element list ``[start, end]`` of ISO date strings
            (e.g., ``["2020-01-01", "2020-12-31"]``).
        stat_method : str
            Aggregation statistic applied per time step (e.g.,
            ``"mean"``, ``"median"``).
        weight_gen_crs : int
            EPSG code for weight generation CRS. Default is 6931
            (LAEA) for accurate area-weighted intersections.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions ``(time, <id_field>)`` and one
            data variable per requested variable name. Values are
            area-weighted aggregates in source units.

        See Also
        --------
        process_nhgf_stac : Temporal aggregation from NHGF STAC Zarr.
        hydro_param.data_access.build_climr_cat_dict : Build catalog dicts.
        hydro_param.data_access.load_climr_catalog : Load the catalog.
        """
        from gdptools import AggGen, ClimRCatData, WeightGen

        from hydro_param.data_access import build_climr_cat_dict, load_climr_catalog

        logger.info(
            "ClimR-Cat temporal: catalog_id='%s' vars=%s period=%s stat=%s",
            catalog_id,
            variable_names,
            time_period,
            stat_method,
        )

        catalog = load_climr_catalog()
        source_cat_dict = build_climr_cat_dict(catalog, catalog_id, variable_names)

        climr_data = ClimRCatData(
            source_cat_dict=source_cat_dict,
            target_gdf=fabric[[id_field, "geometry"]].copy(),
            target_id=id_field,
            source_time_period=cast("list[Any]", time_period),
        )
        wg = WeightGen(
            user_data=cast("Any", climr_data),
            method="serial",
            weight_gen_crs=weight_gen_crs,
        )
        weights = wg.calculate_weights()
        ag = AggGen(
            user_data=cast("Any", climr_data),
            stat_method=cast("Any", stat_method),
            agg_engine="serial",
            agg_writer="none",
            weights=weights,
        )
        _gdf, ds = ag.calculate_agg()

        logger.info(
            "  Temporal result: %d vars, %d time steps, %d features",
            len(ds.data_vars),
            ds.sizes.get("time", 0),
            ds.sizes.get(id_field, len(fabric)),
        )
        return ds


def get_processor(fabric: gpd.GeoDataFrame) -> Processor:
    """Select the appropriate processor for a fabric geometry type.

    Route polygon fabrics to :class:`ZonalProcessor`. This is the factory
    function for the processing pathway bifurcation described in design.md
    section 5.3. Currently only polygon targets (Polygon, MultiPolygon) are
    supported; grid target support via xesmf/rioxarray is planned.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Target fabric GeoDataFrame. Must be non-empty and contain only
        polygon geometry types.

    Returns
    -------
    Processor
        A :class:`ZonalProcessor` instance for polygon fabrics.

    Raises
    ------
    ValueError
        If the fabric is empty or contains unsupported geometry types
        (e.g., Point, LineString). The error message lists the
        unsupported types found.

    Notes
    -----
    Grid target support (raster-on-raster operations via xesmf) is a
    planned feature. When implemented, this function will also return
    a ``GridProcessor`` for grid-type fabrics.

    See Also
    --------
    ZonalProcessor : The processor returned for polygon fabrics.
    TemporalProcessor : Temporal aggregation (not selected by this factory;
        used directly by the pipeline for temporal strategies).
    """
    if fabric.empty:
        raise ValueError("Fabric GeoDataFrame is empty; cannot select a processor.")

    geom_types = set(fabric.geometry.geom_type.unique())
    polygon_types = {"Polygon", "MultiPolygon"}
    if geom_types <= polygon_types:
        return ZonalProcessor()

    unsupported = geom_types - polygon_types
    raise ValueError(f"Unsupported geometry types: {', '.join(sorted(unsupported))}")
