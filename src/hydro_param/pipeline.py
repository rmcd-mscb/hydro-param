"""Pipeline orchestrator: config-driven parameterization stages 1-5.

Implements the 5-stage pipeline from design.md section 4:
  1. Resolve Target Fabric
  2. Resolve Source Datasets
  3. Compute/Load Weights (handled internally by gdptools)
  4. Process Datasets (batch loop)
  5. Format Output (SIR → NetCDF/Parquet)

See design.md section 11 for MVP implementation details.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import xarray as xr

from hydro_param.batching import spatial_batch
from hydro_param.config import DatasetRequest, PipelineConfig, load_config
from hydro_param.data_access import (
    DERIVATION_FUNCTIONS,
    fetch_local_tiff,
    fetch_stac_cog,
    save_to_geotiff,
)
from hydro_param.dataset_registry import (
    DatasetEntry,
    DatasetRegistry,
    DerivedVariableSpec,
    VariableSpec,
    load_registry,
)
from hydro_param.processing import TemporalProcessor, ZonalProcessor, get_processor

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = Path("configs/datasets")


@dataclass
class Stage4Results:
    """Results from stage 4 processing, separating static and temporal outputs."""

    static: dict[str, pd.DataFrame] = field(default_factory=dict)
    temporal: dict[str, xr.Dataset] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Full pipeline result including static SIR, temporal data, and fabric."""

    sir: xr.Dataset
    temporal: dict[str, xr.Dataset] = field(default_factory=dict)
    fabric: gpd.GeoDataFrame | None = None


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def resolve_bbox(config: PipelineConfig) -> list[float]:
    """Extract the domain bounding box from config.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    list[float]
        ``[west, south, east, north]``.
    """
    if config.domain.type == "bbox":
        return config.domain.bbox  # type: ignore[return-value]
    raise NotImplementedError(
        f"Domain type '{config.domain.type}' is not yet supported. Use type='bbox'."
    )


def stage1_resolve_fabric(config: PipelineConfig) -> gpd.GeoDataFrame:
    """Stage 1: Load target fabric as GeoDataFrame."""
    logger.info("Stage 1: Loading target fabric from %s", config.target_fabric.path)
    fabric = gpd.read_file(config.target_fabric.path)

    if config.target_fabric.id_field not in fabric.columns:
        raise ValueError(
            f"ID field '{config.target_fabric.id_field}' not found in fabric. "
            f"Available columns: {list(fabric.columns)}"
        )

    logger.info(
        "Loaded %d features, id_field='%s', CRS=%s",
        len(fabric),
        config.target_fabric.id_field,
        fabric.crs,
    )

    # Apply domain filter to spatially subset fabric
    if config.domain.type == "bbox" and config.domain.bbox is not None:
        from shapely.geometry import box

        # Config bbox is assumed EPSG:4326; reproject if fabric CRS differs.
        bbox_geom = box(*config.domain.bbox)
        bbox_gdf = gpd.GeoDataFrame(index=[0], geometry=[bbox_geom], crs="EPSG:4326")
        if fabric.crs is not None and fabric.crs != bbox_gdf.crs:
            bbox_gdf = bbox_gdf.to_crs(fabric.crs)
        fabric = gpd.clip(fabric, bbox_gdf)
        logger.info("Domain bbox filter: %d features within bbox", len(fabric))
        if fabric.empty:
            raise ValueError("No features found within the specified domain bbox.")

    return fabric


def stage2_resolve_datasets(
    config: PipelineConfig,
    registry: DatasetRegistry,
) -> list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]]:
    """Stage 2: Resolve dataset names to registry entries + variable specs.

    Returns
    -------
    list of (DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec])
    """
    logger.info("Stage 2: Resolving %d datasets from registry", len(config.datasets))
    resolved = []
    for ds_req in config.datasets:
        entry = registry.get(ds_req.name)

        # Apply pipeline config source override
        if ds_req.source is not None:
            entry = entry.model_copy(update={"source": str(ds_req.source)})

        # Validate: local_tiff datasets must have a source
        if entry.strategy == "local_tiff" and entry.source is None:
            msg = (
                f"Dataset '{ds_req.name}' requires a local file "
                f"(strategy: local_tiff) but no 'source' path is set."
            )
            if entry.download:
                if entry.download.files:
                    msg += (
                        f"\n\nThis dataset has {len(entry.download.files)} "
                        f"downloadable files. Run:\n"
                        f"  hydro-param datasets info {ds_req.name}"
                    )
                elif entry.download.url_template:
                    start, end = entry.download.year_range
                    n_vars = len(entry.download.variables_available)
                    msg += (
                        f"\n\nThis dataset has templated downloads "
                        f"({end - start + 1} years x {n_vars} variables). Run:\n"
                        f"  hydro-param datasets info {ds_req.name}"
                    )
                elif entry.download.url:
                    msg += f"\n\nDownload from: {entry.download.url}"
                    if entry.download.size_gb:
                        msg += f"\nExpected size: ~{entry.download.size_gb} GB"
                    if entry.download.format:
                        msg += f"\nFormat: {entry.download.format}"
                    if entry.download.notes:
                        msg += f"\n{entry.download.notes.strip()}"
            msg += (
                f"\n\nThen set 'source' in your pipeline config:\n"
                f"  datasets:\n"
                f"    - name: {ds_req.name}\n"
                f"      source: /path/to/downloaded/file.tif"
            )
            raise ValueError(msg)

        # Validate: temporal datasets require time_period
        if entry.temporal and ds_req.time_period is None:
            raise ValueError(
                f"Dataset '{ds_req.name}' is temporal but no 'time_period' specified. "
                f"Add time_period: ['YYYY-MM-DD', 'YYYY-MM-DD'] to your pipeline config."
            )

        var_specs = [registry.resolve_variable(ds_req.name, v) for v in ds_req.variables]
        resolved.append((entry, ds_req, var_specs))
        logger.info(
            "  %s (%s): %d variables — %s",
            ds_req.name,
            entry.strategy,
            len(var_specs),
            [v.name for v in var_specs],
        )
    return resolved


def _buffered_bbox(fabric: gpd.GeoDataFrame, buffer_frac: float = 0.02) -> list[float]:
    """Compute WGS84 bbox from fabric with fractional buffer.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Spatial features (any CRS).
    buffer_frac : float
        Fractional buffer to add around bounds (default 2%).

    Returns
    -------
    list[float]
        ``[west, south, east, north]`` in EPSG:4326.
    """
    if fabric.crs and not fabric.crs.is_geographic:
        bounds = fabric.to_crs("EPSG:4326").total_bounds
    else:
        bounds = fabric.total_bounds
    dx = (bounds[2] - bounds[0]) * buffer_frac
    dy = (bounds[3] - bounds[1]) * buffer_frac
    return [bounds[0] - dx, bounds[1] - dy, bounds[2] + dx, bounds[3] + dy]


def _process_batch(
    batch_fabric: gpd.GeoDataFrame,
    entry: DatasetEntry,
    ds_req: DatasetRequest,
    var_specs: list[VariableSpec | DerivedVariableSpec],
    config: PipelineConfig,
    work_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Process all variables for a single spatial batch.

    Fetches source data clipped to the batch bounding box, derives
    any requested variables, and runs zonal statistics.

    Returns
    -------
    dict[str, pd.DataFrame]
        Variable name → DataFrame of zonal statistics.
    """
    processor = get_processor(batch_fabric)
    results: dict[str, pd.DataFrame] = {}

    # --- NHGF STAC direct pathway (no intermediate GeoTIFF) ---
    if entry.strategy == "nhgf_stac" and not entry.temporal:
        zonal_proc = cast(ZonalProcessor, processor)
        for var_spec in var_specs:
            if isinstance(var_spec, DerivedVariableSpec):
                raise NotImplementedError("Derived variables not supported for nhgf_stac strategy")
            df = zonal_proc.process_nhgf_stac(
                fabric=batch_fabric,
                collection_id=cast(str, entry.collection),
                variable_name=var_spec.name,
                id_field=config.target_fabric.id_field,
                year=cast(int | None, ds_req.year),
                engine=config.processing.engine,
                statistics=ds_req.statistics,
                categorical=var_spec.categorical,
                band=var_spec.band,
            )
            results[var_spec.name] = df
        return results

    # Compute WGS84 bbox with buffer for STAC/remote queries
    bbox = _buffered_bbox(batch_fabric)

    # Cache source data to avoid redundant fetches for derived variables
    source_cache: dict[str, xr.DataArray] = {}

    def _fetch(dataset_entry: DatasetEntry, fetch_bbox: list[float]) -> xr.DataArray:
        """Dispatch to the correct fetch function based on strategy."""
        if dataset_entry.strategy == "stac_cog":
            return fetch_stac_cog(dataset_entry, fetch_bbox)
        if dataset_entry.strategy == "local_tiff":
            return fetch_local_tiff(dataset_entry, fetch_bbox, dataset_name=ds_req.name)
        if dataset_entry.strategy == "nhgf_stac":
            raise NotImplementedError(
                "Temporal nhgf_stac datasets are not yet supported in the pipeline. "
                "Only static nhgf_stac (temporal: false) is implemented."
            )
        raise NotImplementedError(f"Strategy '{dataset_entry.strategy}' not yet supported")

    for i, var_spec in enumerate(var_specs):
        if isinstance(var_spec, DerivedVariableSpec):
            # Load source once, then derive
            if var_spec.source not in source_cache:
                source_cache[var_spec.source] = _fetch(entry, bbox)

            source_da = source_cache[var_spec.source]
            derive_fn = DERIVATION_FUNCTIONS.get(var_spec.name)
            if derive_fn is None:
                raise ValueError(f"No derivation function for '{var_spec.name}'")
            da = derive_fn(
                source_da,
                method=var_spec.method,
                x_coord=entry.x_coord,
                y_coord=entry.y_coord,
            )
        else:
            # Raw variable: fetch directly
            # TODO: Pass var_spec.band to fetch routine for multi-band datasets
            da = _fetch(entry, bbox)
            # Cache for potential derived variable reuse
            source_cache[var_spec.name] = da

        # Save as GeoTIFF for gdptools
        tiff_path = work_dir / f"{var_spec.name}.tif"
        save_to_geotiff(da, tiff_path)

        # Free derived/fetched raster before zonal stats to reduce peak memory
        if isinstance(var_spec, DerivedVariableSpec):
            del da

        # Determine if variable is categorical
        categorical = isinstance(var_spec, VariableSpec) and var_spec.categorical

        # Run zonal statistics
        df = processor.process(
            fabric=batch_fabric,
            tiff_path=tiff_path,
            variable_name=var_spec.name,
            id_field=config.target_fabric.id_field,
            engine=config.processing.engine,
            statistics=ds_req.statistics,
            categorical=categorical,
            source_crs=entry.crs,
            x_coord=entry.x_coord,
            y_coord=entry.y_coord,
        )
        results[var_spec.name] = df

        # Clean up GeoTIFF after zonal stats
        tiff_path.unlink(missing_ok=True)

        # Release source from cache when no remaining derived vars need it
        if isinstance(var_spec, DerivedVariableSpec):
            remaining = var_specs[i + 1 :]
            source_still_needed = any(
                isinstance(v, DerivedVariableSpec) and v.source == var_spec.source
                for v in remaining
            )
            if not source_still_needed and var_spec.source in source_cache:
                del source_cache[var_spec.source]

    return results


def _process_temporal(
    fabric: gpd.GeoDataFrame,
    entry: DatasetEntry,
    ds_req: DatasetRequest,
    var_specs: list[VariableSpec | DerivedVariableSpec],
    config: PipelineConfig,
) -> xr.Dataset:
    """Process a temporal dataset using WeightGen + AggGen.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Target fabric (full, not batched).
    entry : DatasetEntry
        Registry entry for the dataset.
    ds_req : DatasetRequest
        Pipeline config request.
    var_specs : list
        Resolved variable specifications.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    xr.Dataset
        Temporal dataset with ``(time, features)`` dimensions.
    """
    processor = TemporalProcessor()
    var_names = [v.name for v in var_specs]

    if any(isinstance(v, DerivedVariableSpec) for v in var_specs):
        raise NotImplementedError("Derived variables not supported for temporal datasets")

    time_period = cast(list[str], ds_req.time_period)

    if not ds_req.statistics:
        raise ValueError(
            f"Dataset '{ds_req.name}' is temporal but has no statistics specified. "
            "Temporal datasets require at least one statistic (e.g., 'mean')."
        )

    if len(ds_req.statistics) > 1:
        logger.warning(
            "Temporal processing for '%s' supports a single statistic. "
            "Multiple statistics provided (%s); only '%s' will be used.",
            ds_req.name,
            ds_req.statistics,
            ds_req.statistics[0],
        )

    stat_method = ds_req.statistics[0]

    if entry.strategy == "nhgf_stac":
        return processor.process_nhgf_stac(
            fabric=fabric,
            collection_id=cast(str, entry.collection),
            variable_names=var_names,
            id_field=config.target_fabric.id_field,
            time_period=time_period,
            stat_method=stat_method,
        )
    elif entry.strategy == "climr_cat":
        return processor.process_climr_cat(
            fabric=fabric,
            catalog_id=cast(str, entry.catalog_id),
            variable_names=var_names,
            id_field=config.target_fabric.id_field,
            time_period=time_period,
            stat_method=stat_method,
        )
    else:
        raise NotImplementedError(
            f"Temporal processing not supported for strategy '{entry.strategy}'"
        )


def _split_time_period_by_year(time_period: list[str]) -> list[list[str]]:
    """Split a time period into per-year chunks at calendar year boundaries.

    Parameters
    ----------
    time_period : list[str]
        ``[start, end]`` ISO date strings.

    Returns
    -------
    list[list[str]]
        List of ``[start, end]`` pairs, one per calendar year.

    Examples
    --------
    >>> _split_time_period_by_year(["2020-03-15", "2022-06-30"])
    [['2020-03-15', '2020-12-31'], ['2021-01-01', '2021-12-31'], ['2022-01-01', '2022-06-30']]
    """
    start = date.fromisoformat(time_period[0])
    end = date.fromisoformat(time_period[1])
    chunks: list[list[str]] = []
    while start <= end:
        year_end = date(start.year, 12, 31)
        chunk_end = min(year_end, end)
        chunks.append([start.isoformat(), chunk_end.isoformat()])
        start = date(start.year + 1, 1, 1)
    return chunks


def stage4_process(
    fabric: gpd.GeoDataFrame,
    resolved: list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
    config: PipelineConfig,
) -> Stage4Results:
    """Stage 4: Process all datasets with spatial batching.

    Temporal datasets skip batching and are processed full-fabric via
    ``WeightGen`` + ``AggGen``. Static datasets use the existing batch
    loop with ``ZonalGen``.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Target fabric with ``batch_id`` column from spatial batching.
    resolved : list
        Resolved dataset entries from stage 2.
    config : PipelineConfig
        Pipeline configuration.

    Returns
    -------
    Stage4Results
        Static and temporal results.
    """
    batch_ids = sorted(fabric["batch_id"].unique())
    logger.info("Stage 4: Processing %d datasets across %d batches", len(resolved), len(batch_ids))

    all_results: dict[str, list[pd.DataFrame]] = {}
    temporal_results: dict[str, xr.Dataset] = {}
    categories: dict[str, str] = {}

    for ds_idx, (entry, ds_req, var_specs) in enumerate(resolved, 1):
        category = entry.category or "uncategorized"
        var_names = [v.name for v in var_specs]

        if entry.temporal:
            # Split temporal processing by year to keep files manageable
            year_chunks = _split_time_period_by_year(cast(list[str], ds_req.time_period))
            logger.info(
                "Dataset %d/%d: %s [%s, temporal] vars=%s period=%s (%d year chunks)",
                ds_idx,
                len(resolved),
                ds_req.name,
                entry.strategy,
                var_names,
                ds_req.time_period,
                len(year_chunks),
            )
            t_ds = time.perf_counter()

            for chunk_period in year_chunks:
                chunk_year = chunk_period[0][:4]
                t_chunk = time.perf_counter()
                chunk_req = ds_req.model_copy(update={"time_period": chunk_period})
                ds = _process_temporal(fabric, entry, chunk_req, var_specs, config)
                result_key = f"{ds_req.name}_{chunk_year}"
                temporal_results[result_key] = ds
                categories[result_key] = category
                logger.info(
                    "  %s year %s: %d vars, %d time steps (%.1fs)",
                    ds_req.name,
                    chunk_year,
                    len(ds.data_vars),
                    ds.sizes.get("time", 0),
                    time.perf_counter() - t_chunk,
                )

            logger.info("  %s complete (%.1fs)", ds_req.name, time.perf_counter() - t_ds)
            continue

        # Expand years: list → iterate, bare int → [int], None → [None]
        if isinstance(ds_req.year, list):
            years: list[int | None] = list(ds_req.year)
        elif ds_req.year is not None:
            years = [ds_req.year]
        else:
            years = [None]

        year_label = years if len(years) > 1 else (years[0] if years[0] is not None else "none")
        logger.info(
            "Dataset %d/%d: %s [%s, static] vars=%s year=%s",
            ds_idx,
            len(resolved),
            ds_req.name,
            entry.strategy,
            var_names,
            year_label,
        )
        t_ds = time.perf_counter()

        for year in years:
            # Create single-year request for _process_batch
            year_req = ds_req.model_copy(update={"year": year})

            for batch_id in batch_ids:
                batch = fabric[fabric["batch_id"] == batch_id]
                t_batch = time.perf_counter()

                with tempfile.TemporaryDirectory(prefix="hydro_param_") as tmp:
                    work_dir = Path(tmp)
                    batch_results = _process_batch(
                        batch, entry, year_req, var_specs, config, work_dir
                    )

                logger.info(
                    "  Batch %d/%d: %d features, year=%s (%.1fs)",
                    batch_id + 1,
                    len(batch_ids),
                    len(batch),
                    year,
                    time.perf_counter() - t_batch,
                )

                for var_name, df in batch_results.items():
                    # Year-suffix result keys when multiple years are specified
                    result_key = f"{var_name}_{year}" if year is not None else var_name
                    all_results.setdefault(result_key, []).append(df)

            # Track categories with year-suffixed keys
            for var_spec in var_specs:
                result_key = f"{var_spec.name}_{year}" if year is not None else var_spec.name
                categories[result_key] = category

        logger.info("  %s complete (%.1fs)", ds_req.name, time.perf_counter() - t_ds)

    # Merge batch results per variable
    merged: dict[str, pd.DataFrame] = {}
    for var_name, dfs in all_results.items():
        merged[var_name] = pd.concat(dfs)
        logger.info("  Merged %s: %d total features", var_name, len(merged[var_name]))

    return Stage4Results(static=merged, temporal=temporal_results, categories=categories)


def stage5_format_output(
    results: dict[str, pd.DataFrame] | Stage4Results,
    config: PipelineConfig,
    fabric: gpd.GeoDataFrame,
) -> xr.Dataset:
    """Stage 5: Assemble results into SIR and write output.

    Parameters
    ----------
    results : dict[str, pd.DataFrame] or Stage4Results
        Static zonal statistics (legacy dict) or full Stage4Results
        containing both static and temporal outputs.
    config : PipelineConfig
        Pipeline configuration.
    fabric : gpd.GeoDataFrame
        Target fabric.

    Returns
    -------
    xr.Dataset
        The Standardized Internal Representation (CF-1.8 compliant).
        Temporal results are written as separate files.
    """
    logger.info("Stage 5: Assembling SIR output")
    id_field = config.target_fabric.id_field
    feature_ids = fabric[id_field].values

    # Support both legacy dict and Stage4Results
    if isinstance(results, Stage4Results):
        static_results = results.static
        temporal_results = results.temporal
        categories = results.categories
    else:
        static_results = results
        temporal_results = {}
        categories = {}

    sir_attrs = {
        "title": f"Hydrologic parameters: {config.output.sir_name}",
        "institution": "hydro-param",
        "source": "hydro-param pipeline",
        "history": (f"Created {datetime.now(timezone.utc).isoformat()} by hydro-param"),
        "Conventions": "CF-1.8",
        "target_fabric": str(config.target_fabric.path),
        "target_fabric_id_field": id_field,
        "n_features": len(feature_ids),
        "processing_engine": config.processing.engine,
    }

    data_vars: dict[str, tuple] = {}
    for var_name, df in static_results.items():
        for col in df.columns:
            sir_name = f"{var_name}_{col}" if col != "mean" else var_name
            # Reindex to ensure alignment with full fabric
            if hasattr(df.index, "name") and df.index.name == id_field:
                values = df[col].reindex(feature_ids).values
            else:
                values = df[col].values
            data_vars[sir_name] = (id_field, values)

    sir = xr.Dataset(
        data_vars,
        coords={id_field: feature_ids},
        attrs=sir_attrs,
    )

    # Write output
    output_dir = config.output.path
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write combined SIR (used by pywatershed plugin and PipelineResult)
    if not data_vars:
        logger.warning("No static results to write; skipping static SIR file.")
    if data_vars:
        if config.output.format == "netcdf":
            sir_path = output_dir / f"{config.output.sir_name}.nc"
            sir.to_netcdf(sir_path)
            logger.info("Wrote SIR → %s", sir_path)
        elif config.output.format == "parquet":
            sir_path = output_dir / f"{config.output.sir_name}.parquet"
            sir.to_dataframe().to_parquet(sir_path)
            logger.info("Wrote SIR → %s", sir_path)

    # Write per-variable files organized by category
    for var_name, df in static_results.items():
        category = categories.get(var_name, "uncategorized")
        var_dir = output_dir / category
        var_dir.mkdir(parents=True, exist_ok=True)

        # Build single-variable xr.Dataset
        var_data_vars: dict[str, tuple] = {}
        for col in df.columns:
            col_name = f"{var_name}_{col}" if col != "mean" else var_name
            if hasattr(df.index, "name") and df.index.name == id_field:
                values = df[col].reindex(feature_ids).values
            else:
                values = df[col].values
            var_data_vars[col_name] = (id_field, values)

        var_ds = xr.Dataset(
            var_data_vars,
            coords={id_field: feature_ids},
            attrs=sir_attrs,
        )

        if config.output.format == "netcdf":
            out_path = var_dir / f"{var_name}.nc"
            var_ds.to_netcdf(out_path)
        elif config.output.format == "parquet":
            out_path = var_dir / f"{var_name}.parquet"
            var_ds.to_dataframe().to_parquet(out_path)
        logger.info("Wrote %s → %s", var_name, out_path)

    # Write temporal results as per-dataset files in category directories
    for ds_name, ds in temporal_results.items():
        category = categories.get(ds_name, "climate")
        var_dir = output_dir / category
        var_dir.mkdir(parents=True, exist_ok=True)

        if config.output.format == "netcdf":
            temporal_path = var_dir / f"{ds_name}_temporal.nc"
            ds.to_netcdf(temporal_path)
        elif config.output.format == "parquet":
            temporal_path = var_dir / f"{ds_name}_temporal.parquet"
            ds.to_dataframe().to_parquet(temporal_path)
        logger.info("Wrote temporal %s → %s", ds_name, temporal_path)

    return sir


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline_from_config(
    config: PipelineConfig,
    registry: DatasetRegistry,
) -> PipelineResult:
    """Execute the full pipeline from pre-loaded config and registry.

    Parameters
    ----------
    config
        Pipeline configuration.
    registry
        Dataset registry.

    Returns
    -------
    PipelineResult
        Full results including static SIR, temporal data, and fabric.
    """
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("hydro-param pipeline: %s", config.output.sir_name)
    logger.info(
        "  Fabric: %s (id_field=%s)", config.target_fabric.path, config.target_fabric.id_field
    )
    logger.info(
        "  Datasets: %d, Engine: %s, Batch size: %d",
        len(config.datasets),
        config.processing.engine,
        config.processing.batch_size,
    )
    logger.info("  Output: %s (%s)", config.output.path, config.output.format)
    logger.info("=" * 60)

    # Stage 1: Resolve target fabric (applies domain filter if configured)
    t1 = time.perf_counter()
    fabric = stage1_resolve_fabric(config)

    # Spatial batching
    fabric = spatial_batch(fabric, batch_size=config.processing.batch_size)
    batch_ids = sorted(fabric["batch_id"].unique())
    batch_sizes = fabric.groupby("batch_id").size()
    logger.info(
        "Spatial batching: %d features → %d batches (min=%d, max=%d, mean=%d) (%.1fs)",
        len(fabric),
        len(batch_ids),
        batch_sizes.min(),
        batch_sizes.max(),
        batch_sizes.mean(),
        time.perf_counter() - t1,
    )

    # Stage 2: Resolve source datasets
    t2 = time.perf_counter()
    resolved = stage2_resolve_datasets(config, registry)
    logger.info("Stage 2 complete (%.1fs)", time.perf_counter() - t2)

    # Stage 3: Weights (handled internally by gdptools ZonalGen)
    logger.info("Stage 3: Weights computed internally by gdptools")

    # Stage 4: Process datasets
    t4 = time.perf_counter()
    results = stage4_process(fabric, resolved, config)
    logger.info(
        "Stage 4 complete: %d static vars, %d temporal datasets (%.1fs)",
        len(results.static),
        len(results.temporal),
        time.perf_counter() - t4,
    )

    # Stage 5: Format output
    t5 = time.perf_counter()
    sir = stage5_format_output(results, config, fabric)
    logger.info(
        "Stage 5 complete: SIR has %d variables, %d features (%.1fs)",
        len(sir.data_vars),
        sir.sizes.get(config.target_fabric.id_field, 0),
        time.perf_counter() - t5,
    )

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(
        "Pipeline complete: %d static + %d temporal datasets, %d features, %.1f seconds",
        len(results.static),
        len(results.temporal),
        len(fabric),
        elapsed,
    )
    logger.info("=" * 60)

    return PipelineResult(sir=sir, temporal=results.temporal, fabric=fabric)


def run_pipeline(
    config_path: str | Path,
    registry_path: str | Path | None = None,
) -> xr.Dataset:
    """Execute the full parameterization pipeline.

    Parameters
    ----------
    config_path : str or Path
        Path to the pipeline YAML config.
    registry_path : str or Path or None
        Path to a dataset registry YAML file or directory. Defaults to
        ``configs/datasets/``.

    Returns
    -------
    xr.Dataset
        The Standardized Internal Representation of computed parameters.
    """
    config = load_config(config_path)
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY
    registry = load_registry(registry_path)

    result = run_pipeline_from_config(config, registry)
    return result.sir


def main() -> int:
    """CLI entry point for ``python -m hydro_param.pipeline``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        logger.error("Usage: python -m hydro_param.pipeline <config.yml> [registry.yml]")
        return 1

    config_path = sys.argv[1]
    registry_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        run_pipeline(config_path, registry_path)
    except Exception:
        logger.exception("Pipeline failed.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
