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
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

from hydro_param.batching import spatial_batch
from hydro_param.config import DatasetRequest, PipelineConfig, load_config
from hydro_param.data_access import (
    DERIVATION_FUNCTIONS,
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
from hydro_param.processing import get_processor

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = Path("configs/datasets.yml")


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
    # TODO: Reproject batch bounds into entry.crs when fabric CRS != dataset CRS
    bbox = list(batch_fabric.total_bounds)
    results: dict[str, pd.DataFrame] = {}

    # Cache source data to avoid redundant fetches for derived variables
    source_cache: dict[str, xr.DataArray] = {}

    for var_spec in var_specs:
        if isinstance(var_spec, DerivedVariableSpec):
            # Load source once, then derive
            if var_spec.source not in source_cache:
                if entry.strategy == "stac_cog":
                    source_cache[var_spec.source] = fetch_stac_cog(entry, bbox)
                else:
                    raise NotImplementedError(f"Strategy '{entry.strategy}' not yet supported")

            source_da = source_cache[var_spec.source]
            derive_fn = DERIVATION_FUNCTIONS.get(var_spec.name)
            if derive_fn is None:
                raise ValueError(f"No derivation function for '{var_spec.name}'")
            da = derive_fn(source_da, method=var_spec.method)
        else:
            # Raw variable: fetch directly
            # TODO: Pass var_spec.band to fetch routine for multi-band datasets
            if entry.strategy == "stac_cog":
                da = fetch_stac_cog(entry, bbox)
            else:
                raise NotImplementedError(f"Strategy '{entry.strategy}' not yet supported")
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
        )
        results[var_spec.name] = df

    return results


def stage4_process(
    fabric: gpd.GeoDataFrame,
    resolved: list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
    config: PipelineConfig,
) -> dict[str, pd.DataFrame]:
    """Stage 4: Process all datasets with spatial batching.

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
    dict[str, pd.DataFrame]
        Variable name → merged DataFrame of zonal statistics.
    """
    batch_ids = sorted(fabric["batch_id"].unique())
    logger.info("Stage 4: Processing %d datasets across %d batches", len(resolved), len(batch_ids))

    all_results: dict[str, list[pd.DataFrame]] = {}

    for entry, ds_req, var_specs in resolved:
        logger.info("Processing dataset: %s", ds_req.name)

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

    return merged


def stage5_format_output(
    results: dict[str, pd.DataFrame],
    config: PipelineConfig,
    fabric: gpd.GeoDataFrame,
) -> xr.Dataset:
    """Stage 5: Assemble results into SIR and write output.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        Variable name → zonal statistics DataFrame.
    config : PipelineConfig
        Pipeline configuration.
    fabric : gpd.GeoDataFrame
        Target fabric.

    Returns
    -------
    xr.Dataset
        The Standardized Internal Representation (CF-1.8 compliant).
    """
    logger.info("Stage 5: Assembling SIR output")
    id_field = config.target_fabric.id_field
    feature_ids = fabric[id_field].values

    data_vars: dict[str, tuple] = {}
    for var_name, df in results.items():
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
        Path to the dataset registry YAML. Defaults to ``configs/datasets.yml``.

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

    # Stage 1: Resolve target fabric
    # TODO: Apply config.domain to spatially subset fabric (bbox clip, HUC filter, etc.)
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
        "Pipeline complete: %d variables, %d features, %.1f seconds",
        len(results),
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
