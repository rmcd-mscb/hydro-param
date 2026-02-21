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
from datetime import datetime, timezone
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
                year=ds_req.year,
                engine=config.processing.engine,
                statistics=ds_req.statistics,
                categorical=var_spec.categorical,
                band=var_spec.band,
            )
            results[var_spec.name] = df
        return results

    # TODO: Reproject batch bounds into entry.crs when fabric CRS != dataset CRS
    bbox = list(batch_fabric.total_bounds)

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

    for var_spec in var_specs:
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

    for entry, ds_req, var_specs in resolved:
        logger.info("Processing dataset: %s", ds_req.name)

        if entry.temporal:
            ds = _process_temporal(fabric, entry, ds_req, var_specs, config)
            temporal_results[ds_req.name] = ds
            continue

        for batch_id in batch_ids:
            batch = fabric[fabric["batch_id"] == batch_id]
            logger.info("  Batch %d/%d: %d features", batch_id + 1, len(batch_ids), len(batch))

            with tempfile.TemporaryDirectory(prefix="hydro_param_") as tmp:
                work_dir = Path(tmp)
                batch_results = _process_batch(batch, entry, ds_req, var_specs, config, work_dir)

            for var_name, df in batch_results.items():
                all_results.setdefault(var_name, []).append(df)

    # Merge batch results per variable
    merged: dict[str, pd.DataFrame] = {}
    for var_name, dfs in all_results.items():
        merged[var_name] = pd.concat(dfs)
        logger.info("  %s: %d total features", var_name, len(merged[var_name]))

    return Stage4Results(static=merged, temporal=temporal_results)


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
    else:
        static_results = results
        temporal_results = {}

    data_vars: dict[str, tuple] = {}
    for var_name, df in static_results.items():
        for col in df.columns:
            sir_name = f"{var_name}_{col}" if col != "mean" else var_name
            # Reindex to ensure alignment with full fabric
            if hasattr(df.index, "name") and df.index.name == id_field:
                values = df[col].reindex(feature_ids).values
            else:
                values = df[col].values
            data_vars[sir_name] = ("hru_id", values)

    sir = xr.Dataset(
        data_vars,
        coords={"hru_id": feature_ids},
        attrs={
            "title": f"Hydrologic parameters: {config.output.sir_name}",
            "institution": "hydro-param",
            "source": "hydro-param pipeline",
            "history": (f"Created {datetime.now(timezone.utc).isoformat()} by hydro-param"),
            "Conventions": "CF-1.8",
            "target_fabric": str(config.target_fabric.path),
            "target_fabric_id_field": id_field,
            "n_features": len(feature_ids),
            "processing_engine": config.processing.engine,
        },
    )

    # Write output
    output_dir = config.output.path
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.output.format == "netcdf":
        out_path = output_dir / f"{config.output.sir_name}.nc"
        sir.to_netcdf(out_path)
        logger.info("Wrote SIR → %s", out_path)
    elif config.output.format == "parquet":
        out_path = output_dir / f"{config.output.sir_name}.parquet"
        sir.to_dataframe().to_parquet(out_path)
        logger.info("Wrote SIR → %s", out_path)

    # Write temporal results as separate files
    for ds_name, ds in temporal_results.items():
        if config.output.format == "netcdf":
            temporal_path = output_dir / f"{config.output.sir_name}_{ds_name}_temporal.nc"
            ds.to_netcdf(temporal_path)
            logger.info("Wrote temporal → %s", temporal_path)
        elif config.output.format == "parquet":
            temporal_path = output_dir / f"{config.output.sir_name}_{ds_name}_temporal.parquet"
            ds.to_dataframe().to_parquet(temporal_path)
            logger.info("Wrote temporal → %s", temporal_path)

    return sir


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


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
    t0 = time.perf_counter()

    config = load_config(config_path)
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY
    registry = load_registry(registry_path)

    logger.info("=" * 60)
    logger.info("hydro-param pipeline: %s", config.output.sir_name)
    logger.info("=" * 60)

    # Stage 1: Resolve target fabric (applies domain filter if configured)
    fabric = stage1_resolve_fabric(config)

    # Spatial batching
    fabric = spatial_batch(fabric, batch_size=config.processing.batch_size)

    # Stage 2: Resolve source datasets
    resolved = stage2_resolve_datasets(config, registry)

    # Stage 3: Weights (handled internally by gdptools ZonalGen)
    logger.info("Stage 3: Weights computed internally by gdptools ZonalGen")

    # Stage 4: Process datasets
    results = stage4_process(fabric, resolved, config)

    # Stage 5: Format output
    sir = stage5_format_output(results, config, fabric)

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

    return sir


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
