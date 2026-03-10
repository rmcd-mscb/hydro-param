"""Pipeline orchestrator: execute the config-driven parameterization stages 1--5.

Implement the 5-stage pipeline from design.md section 4:

1. **Resolve Target Fabric** -- load the polygon mesh and apply optional
   domain filtering (bbox clip).
2. **Resolve Source Datasets** -- match dataset names in the pipeline config
   to registry entries and resolve variable specifications.
3. **Compute/Load Weights** -- handled internally by gdptools (no explicit
   stage function; gdptools ``ZonalGen`` computes coverage weights on the fly).
4. **Process Datasets** -- spatial batching loop for static datasets
   (``ZonalGen``) and full-fabric temporal processing (``WeightGen`` +
   ``AggGen``).  Per-variable and temporal output files are written
   incrementally as each dataset completes.
5. **Normalize SIR** -- canonical variable naming, unit conversion, and
   validation.  Produces the Standardized Internal Representation consumed
   by model plugins.

A lazy :meth:`PipelineResult.load_sir` method assembles a combined
``xr.Dataset`` on demand for downstream consumers.  Phase 2 model plugins
(e.g., pywatershed) consume SIR files from disk via ``SIRAccessor`` rather
than ``PipelineResult.load_sir``.

This module is **model-agnostic** by design -- all model-specific logic
(unit conversions, variable renaming, derived math, output formatting)
lives in model plugins under ``derivations/`` and ``formatters/``.

See Also
--------
design.md : Full architecture document (section 4: pipeline stages,
    section 11: MVP implementation).
hydro_param.config : Pydantic config schema consumed by every stage.
hydro_param.sir : SIR normalization and validation utilities.
hydro_param.sir_accessor : Lazy SIR loader used by Phase 2 plugins.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import xarray as xr

from hydro_param import manifest as _manifest_mod
from hydro_param.batching import spatial_batch
from hydro_param.config import DatasetRequest, PipelineConfig, load_config
from hydro_param.data_access import (
    DERIVATION_FUNCTIONS,
    fetch_local_tiff,
    fetch_stac_cog,
    query_stac_items,
    save_to_geotiff,
)
from hydro_param.dataset_registry import (
    AnyVariableSpec,
    DatasetEntry,
    DatasetRegistry,
    DerivedCategoricalSpec,
    DerivedVariableSpec,
    VariableSpec,
    load_registry,
)
from hydro_param.processing import TemporalProcessor, ZonalProcessor, get_processor
from hydro_param.sir import (
    SIRValidationWarning,
    SIRVariableSchema,
    build_sir_schema,
    normalize_sir,
    normalize_sir_temporal,
    validate_sir,
)

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY = Path(str(_pkg_files("hydro_param").joinpath("data/datasets")))

USER_REGISTRY_DIR = Path.home() / ".hydro-param" / "datasets"
"""User-local registry overlay directory (~/.hydro-param/datasets/).

YAML files in this directory extend the bundled dataset registry.
Overlay entries replace bundled entries on name collision.
"""


@dataclass
class Stage4Results:
    """Collect file paths and category metadata from stage 4 processing.

    Stage 4 writes per-variable CSV files (static datasets) and per-year
    NetCDF/Parquet files (temporal datasets) incrementally.  This dataclass
    aggregates the paths so that stage 5 can locate and normalize them.

    Attributes
    ----------
    static_files : dict[str, Path]
        Mapping of result key (e.g., ``"elevation"`` or ``"land_cover_2021"``)
        to the CSV file path written during stage 4.
    temporal_files : dict[str, Path]
        Mapping of result key (e.g., ``"gridmet_2020"``) to the NetCDF or
        Parquet file path written during stage 4.
    categories : dict[str, str]
        Mapping of result key to its dataset category (e.g., ``"topography"``),
        used for organizing output into subdirectories.
    """

    static_files: dict[str, Path] = field(default_factory=dict)
    temporal_files: dict[str, Path] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Encapsulate all pipeline outputs with lazy SIR loading.

    Per-variable and temporal files are written incrementally during stage 4.
    Stage 5 normalizes them into SIR files.  Use :meth:`load_sir` to
    assemble a combined ``xr.Dataset`` on demand rather than holding all
    data in memory.

    Attributes
    ----------
    output_dir : Path
        Root output directory (same as ``config.output.path``).
    static_files : dict[str, Path]
        Raw (pre-normalization) per-variable CSV paths from stage 4.
    temporal_files : dict[str, Path]
        Raw temporal NetCDF/Parquet paths from stage 4.
    categories : dict[str, str]
        Result key to dataset category mapping.
    fabric : gpd.GeoDataFrame or None
        Target fabric with ``batch_id`` column, retained for downstream
        consumers (e.g., model plugins that need geometry or topology).
    sir_files : dict[str, Path]
        Normalized SIR file paths from stage 5.
    sir_schema : list[SIRVariableSchema]
        Schema entries describing each SIR variable (canonical name, units,
        source dataset, statistic).
    sir_warnings : list[SIRValidationWarning]
        Validation warnings from stage 5 SIR validation.
    """

    output_dir: Path
    static_files: dict[str, Path] = field(default_factory=dict)
    temporal_files: dict[str, Path] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)
    fabric: gpd.GeoDataFrame | None = None
    sir_files: dict[str, Path] = field(default_factory=dict)
    sir_schema: list[SIRVariableSchema] = field(default_factory=list)
    sir_warnings: list[SIRValidationWarning] = field(default_factory=list)

    def load_sir(self) -> xr.Dataset:
        """Load normalized SIR files into a combined xr.Dataset.

        Assemble all per-variable SIR CSV files into a single
        ``xr.Dataset`` with the fabric ``id_field`` as the dimension.

        Returns
        -------
        xr.Dataset
            Combined dataset with one data variable per SIR variable.
            Returns an empty dataset if no SIR files are available.
        """
        if not self.sir_files:
            logger.warning("No SIR files available — returning empty dataset")
            return xr.Dataset()
        dfs = [pd.read_csv(p, index_col=0) for p in self.sir_files.values()]
        combined = pd.concat(dfs, axis=1)
        return xr.Dataset.from_dataframe(combined)

    def load_raw_sir(self) -> xr.Dataset:
        """Load raw (pre-normalization) static files into a combined xr.Dataset.

        Unlike :meth:`load_sir`, this always uses the stage 4 raw CSV files,
        bypassing SIR normalization.  Useful for debugging or inspecting
        source-native variable names and units.

        Returns
        -------
        xr.Dataset
            Combined dataset from raw static files, or an empty dataset if
            no static files exist.
        """
        if not self.static_files:
            return xr.Dataset()
        dfs = [pd.read_csv(p, index_col=0) for p in self.static_files.values()]
        combined = pd.concat(dfs, axis=1)
        return xr.Dataset.from_dataframe(combined)


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def resolve_bbox(config: PipelineConfig) -> list[float]:
    """Extract the domain bounding box from the pipeline config.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration containing an optional ``domain`` section.

    Returns
    -------
    list[float]
        Bounding box as ``[west, south, east, north]`` in EPSG:4326
        (decimal degrees).

    Raises
    ------
    ValueError
        If no domain is configured (the caller should use the fabric
        bounding box directly instead).
    NotImplementedError
        If the domain type is not ``"bbox"`` (HUC/gage extraction is
        planned but not yet implemented).
    """
    if config.domain is None:
        raise ValueError(
            "No domain configured. When domain is omitted, the pipeline uses "
            "the fabric bounding box automatically."
        )
    if config.domain.type == "bbox":
        return config.domain.bbox  # type: ignore[return-value]
    raise NotImplementedError(
        f"Domain type '{config.domain.type}' is not yet supported. Use type='bbox'."
    )


def stage1_resolve_fabric(config: PipelineConfig) -> gpd.GeoDataFrame:
    """Stage 1: Load the target fabric and apply optional domain filtering.

    Read the geospatial file specified by ``config.target_fabric.path``,
    validate that the ``id_field`` column exists, and optionally clip the
    fabric to a bounding box domain.  The domain bbox is assumed to be in
    EPSG:4326 and is reprojected to the fabric CRS if they differ.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with ``target_fabric`` and optional
        ``domain`` sections.

    Returns
    -------
    gpd.GeoDataFrame
        Target fabric, potentially spatially subsetted by the domain bbox.

    Raises
    ------
    FileNotFoundError
        If the fabric file does not exist on disk.
    ValueError
        If the ``id_field`` is not found in the fabric columns, or if
        domain filtering produces an empty result.
    """
    logger.info("Stage 1: Loading target fabric from %s", config.target_fabric.path)
    fabric_path = config.target_fabric.path
    if not fabric_path.exists():
        raise FileNotFoundError(
            f"Target fabric not found: {fabric_path}\n"
            f"Download or copy the fabric file before running the pipeline. "
            f"See 'hydro-param init --help' for project setup."
        )
    fabric = gpd.read_file(fabric_path)

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

    # Apply domain filter to spatially subset fabric (optional)
    if (
        config.domain is not None
        and config.domain.type == "bbox"
        and config.domain.bbox is not None
    ):
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


def _validate_time_range(
    ds_req: DatasetRequest,
    entry: DatasetEntry,
) -> None:
    """Validate that requested time period or year falls within the dataset's available range.

    Check the ``time_period`` start/end years and ``year`` values against
    the dataset's ``year_range`` metadata.  Raise ``ValueError`` with an
    actionable message if the requested range exceeds the available data.

    If the dataset entry has no ``year_range`` metadata, validation is
    skipped and the function returns immediately.

    Parameters
    ----------
    ds_req : DatasetRequest
        Pipeline request containing ``time_period`` and/or ``year``.
    entry : DatasetEntry
        Registry entry containing ``year_range`` metadata.

    Raises
    ------
    ValueError
        If the requested time range falls outside the dataset's
        ``year_range``.

    Notes
    -----
    Years are extracted from the first four characters of each ISO date
    string in ``time_period`` (e.g., ``"2020-01-01"`` -> ``2020``).
    This relies on ``DatasetRequest._validate_time_period`` having
    already confirmed that dates are well-formed ISO strings.
    """
    if entry.year_range is None:
        logger.debug(
            "Dataset '%s' has no year_range metadata; skipping time range validation",
            ds_req.name,
        )
        return

    avail_start, avail_end = entry.year_range

    # Validate time_period (temporal datasets)
    if ds_req.time_period is not None:
        req_start_year = int(ds_req.time_period[0][:4])
        req_end_year = int(ds_req.time_period[1][:4])
        if req_start_year < avail_start:
            raise ValueError(
                f"Dataset '{ds_req.name}' time_period {ds_req.time_period} "
                f"starts in {req_start_year}, but data is only "
                f"available from {avail_start}-{avail_end}. "
                f"Adjust time_period in your pipeline config."
            )
        if req_end_year > avail_end:
            raise ValueError(
                f"Dataset '{ds_req.name}' time_period {ds_req.time_period} "
                f"ends in {req_end_year}, but data is only "
                f"available from {avail_start}-{avail_end}. "
                f"Adjust time_period in your pipeline config."
            )

    # Validate year (e.g., multi-year static datasets like NLCD)
    if ds_req.year is not None:
        years = [ds_req.year] if isinstance(ds_req.year, int) else ds_req.year
        for y in years:
            if y < avail_start or y > avail_end:
                raise ValueError(
                    f"Dataset '{ds_req.name}' year {y} is outside the "
                    f"available range {avail_start}-{avail_end}. "
                    f"Adjust year in your pipeline config."
                )


def stage2_resolve_datasets(
    config: PipelineConfig,
    registry: DatasetRegistry,
) -> list[
    tuple[
        DatasetEntry,
        DatasetRequest,
        list[AnyVariableSpec],
    ]
]:
    """Stage 2: Resolve dataset names to registry entries and variable specs.

    For each :class:`~hydro_param.config.DatasetRequest` in the pipeline
    config, look up the corresponding :class:`~hydro_param.dataset_registry.DatasetEntry`
    in the registry.  Apply source overrides from the pipeline config,
    validate strategy-specific requirements (e.g., ``local_tiff`` needs a
    source path, temporal datasets need a ``time_period``), and resolve
    each requested variable name to its full specification.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration containing the ``datasets`` dict (keyed by category).
    registry : DatasetRegistry
        Dataset registry mapping names to entries and variable specs.

    Returns
    -------
    list[tuple[DatasetEntry, DatasetRequest, list[...]]]
        One tuple per dataset: the registry entry, the pipeline request,
        and the resolved variable specifications (VariableSpec,
        DerivedVariableSpec, or DerivedCategoricalSpec).

    Raises
    ------
    ValueError
        If a ``local_tiff`` dataset has no source path (dataset-level or
        per-variable), if a temporal dataset has no ``time_period``, or if
        the requested ``time_period`` or ``year`` falls outside the
        dataset's ``year_range``.
    KeyError
        If a dataset name is not found in the registry (raised by
        ``registry.get()``).
    """
    flat_datasets = config.flatten_datasets()
    logger.info("Stage 2: Resolving %d datasets from registry", len(flat_datasets))

    # Build dataset-name → config-category lookup for cross-validation
    ds_category_map: dict[str, str] = {}
    for category_key, ds_list in config.datasets.items():
        for _ds_req in ds_list:
            if _ds_req.name in ds_category_map:
                raise ValueError(
                    f"Dataset '{_ds_req.name}' appears in multiple categories: "
                    f"'{ds_category_map[_ds_req.name]}' and '{category_key}'. "
                    f"Each dataset should appear in exactly one category."
                )
            ds_category_map[_ds_req.name] = category_key

    resolved = []
    for ds_req in flat_datasets:
        entry = registry.get(ds_req.name)

        # Cross-validate config category vs registry category
        config_cat = ds_category_map[ds_req.name]
        if not entry.category:
            logger.debug(
                "Skipping category cross-validation for '%s': no category set in registry",
                ds_req.name,
            )
        elif entry.category != config_cat:
            logger.warning(
                "Category mismatch for dataset '%s': config key is '%s' "
                "but registry category is '%s'",
                ds_req.name,
                config_cat,
                entry.category,
            )

        # Apply pipeline config source override
        if ds_req.source is not None:
            entry = entry.model_copy(update={"source": str(ds_req.source)})

        # Validate: local_tiff datasets must have a source (dataset-level or per-variable)
        if entry.strategy == "local_tiff" and entry.source is None:
            # Check if all requested variables have per-variable source overrides
            requested_var_specs = [
                registry.resolve_variable(ds_req.name, v) for v in ds_req.variables
            ]
            all_vars_have_source = all(
                isinstance(vs, DerivedCategoricalSpec)
                or (isinstance(vs, VariableSpec) and vs.source_override is not None)
                for vs in requested_var_specs
            )
            if not all_vars_have_source:
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
                    f"    {config_cat}:\n"
                    f"      - name: {ds_req.name}\n"
                    f"        source: /path/to/downloaded/file.tif"
                )
                raise ValueError(msg)

        # Validate: temporal datasets require time_period
        if entry.temporal and ds_req.time_period is None:
            raise ValueError(
                f"Dataset '{ds_req.name}' is temporal but no 'time_period' specified. "
                f"Add time_period: ['YYYY-MM-DD', 'YYYY-MM-DD'] to your pipeline config."
            )

        # Validate time range against dataset availability
        _validate_time_range(ds_req, entry)

        # Auto-include source variables needed by derived categorical specs
        requested = set(ds_req.variables)
        extra_sources: list[str] = []
        for vname in ds_req.variables:
            spec = registry.resolve_variable(ds_req.name, vname)
            if isinstance(spec, DerivedCategoricalSpec):
                for src in spec.sources:
                    if src not in requested and src not in extra_sources:
                        extra_sources.append(src)
        if extra_sources:
            logger.info(
                "  Auto-including source variables for derived categorical: %s",
                extra_sources,
            )

        all_var_names = extra_sources + list(ds_req.variables)
        var_specs = [registry.resolve_variable(ds_req.name, v) for v in all_var_names]
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
    """Compute a WGS 84 bounding box from the fabric with a fractional buffer.

    Add a small buffer around the fabric total bounds to ensure edge
    features are fully covered by STAC/remote data queries.  Without
    this buffer, features at the boundary may receive partial or missing
    raster coverage.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Spatial features in any CRS.  If the CRS is projected, the fabric
        is temporarily reprojected to EPSG:4326 for bounds computation.
    buffer_frac : float
        Fractional buffer to add on each side, as a proportion of the
        extent in that dimension.  Default is 0.02 (2%).

    Returns
    -------
    list[float]
        Bounding box as ``[west, south, east, north]`` in EPSG:4326
        (decimal degrees).
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
    var_specs: list[AnyVariableSpec],
    config: PipelineConfig,
    work_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Process all variables for a single spatial batch.

    Fetch source raster data clipped to the batch bounding box, derive
    any terrain variables (slope, aspect) from their source, classify
    derived categorical variables (e.g., USDA texture from sand/silt/clay),
    save intermediate GeoTIFFs, and run zonal statistics via gdptools.

    Memory management is critical here because gNATSGO variables can
    each consume ~1.25 GB.  The ``source_cache`` is eagerly pruned:
    raw variables are released as soon as no downstream derived variable
    needs them, and ``gc.collect()`` runs after each variable.

    Parameters
    ----------
    batch_fabric : gpd.GeoDataFrame
        Subset of the target fabric for this spatial batch.
    entry : DatasetEntry
        Registry entry describing the source dataset (strategy, CRS, etc.).
    ds_req : DatasetRequest
        Pipeline config request (variables, statistics, year).
    var_specs : list[AnyVariableSpec]
        Resolved variable specifications from the registry.
        ``DerivedCategoricalSpec`` entries are processed in a second
        pass after all source variables.
    config : PipelineConfig
        Pipeline configuration (engine, id_field, etc.).
    work_dir : Path
        Temporary directory for intermediate GeoTIFF files.  Cleaned up
        by the caller (``tempfile.TemporaryDirectory``).

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of variable name to a DataFrame of zonal statistics,
        indexed by the fabric ``id_field``.

    Raises
    ------
    NotImplementedError
        If derived variables are requested for ``nhgf_stac`` strategy, or
        if temporal ``nhgf_stac`` is used (not yet supported in batch loop).
    ValueError
        If a derived variable has no registered derivation function,
        or if a derived categorical variable has no registered
        classification function in ``CATEGORICAL_DERIVATION_FUNCTIONS``.
    FileNotFoundError
        If source GeoTIFFs required by a derived categorical variable
        are missing from the work directory.

    Notes
    -----
    The NHGF STAC static pathway bypasses intermediate GeoTIFF files
    entirely -- gdptools reads directly from the STAC COGs.  All other
    strategies write a temporary GeoTIFF per variable, which gdptools
    reads for zonal statistics.

    See Also
    --------
    hydro_param.data_access.fetch_stac_cog : STAC COG fetch with bbox clip.
    hydro_param.data_access.fetch_local_tiff : Local GeoTIFF fetch.
    hydro_param.processing.ZonalProcessor : gdptools zonal statistics wrapper.
    """
    processor = get_processor(batch_fabric)
    results: dict[str, pd.DataFrame] = {}

    # --- NHGF STAC direct pathway (no intermediate GeoTIFF) ---
    if entry.strategy == "nhgf_stac" and not entry.temporal:
        zonal_proc = cast(ZonalProcessor, processor)
        for var_spec in var_specs:
            if isinstance(var_spec, DerivedVariableSpec | DerivedCategoricalSpec):
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

    # Pre-query STAC items once for all variables in this batch
    stac_items: list | None = None
    if entry.strategy == "stac_cog":
        stac_items = query_stac_items(entry, bbox)

    def _fetch(
        dataset_entry: DatasetEntry,
        fetch_bbox: list[float],
        *,
        variable_source: str | None = None,
        asset_key: str | None = None,
        items: list | None = None,
    ) -> xr.DataArray:
        """Dispatch to the correct fetch function based on dataset strategy.

        Routes to ``fetch_stac_cog`` or ``fetch_local_tiff`` depending on
        ``dataset_entry.strategy``.  Pre-queried STAC items can be passed
        via *items* to avoid redundant STAC API calls within a batch.
        """
        if dataset_entry.strategy == "stac_cog":
            return fetch_stac_cog(
                dataset_entry,
                fetch_bbox,
                asset_key=asset_key,
                items=items,
            )
        if dataset_entry.strategy == "local_tiff":
            return fetch_local_tiff(
                dataset_entry,
                fetch_bbox,
                dataset_name=ds_req.name,
                variable_source=variable_source,
            )
        if dataset_entry.strategy == "nhgf_stac":
            raise NotImplementedError(
                "Temporal nhgf_stac datasets are not yet supported in the pipeline. "
                "Only static nhgf_stac (temporal: false) is implemented."
            )
        raise NotImplementedError(f"Strategy '{dataset_entry.strategy}' not yet supported")

    for i, var_spec in enumerate(var_specs):
        if isinstance(var_spec, DerivedCategoricalSpec):
            continue  # Processed after all source variables

        if isinstance(var_spec, DerivedVariableSpec):
            # Load source once, then derive.
            # Note: asset_key is not passed here — derived variables currently
            # only exist on single-asset datasets (e.g. 3DEP slope/aspect from
            # elevation). If derived vars are added to per-asset datasets like
            # gNATSGO, the source VarSpec's asset_key will need to be resolved.
            if var_spec.source not in source_cache:
                source_cache[var_spec.source] = _fetch(entry, bbox, items=stac_items)

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
            # Per-variable source (e.g. POLARIS VRTs) overrides dataset-level source
            # Per-variable asset_key (e.g. gNATSGO) overrides dataset-level asset_key
            da = _fetch(
                entry,
                bbox,
                variable_source=var_spec.source_override,
                asset_key=var_spec.asset_key,
                items=stac_items,
            )
            # Cache for potential derived variable reuse
            source_cache[var_spec.name] = da

        # Save as GeoTIFF for gdptools
        tiff_path = work_dir / f"{var_spec.name}.tif"
        save_to_geotiff(da, tiff_path)

        # Free raster before zonal stats — gdptools reads from the file
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

        # Clean up GeoTIFF after zonal stats — keep if needed by derived categorical
        needed_by_dc = any(
            isinstance(dc, DerivedCategoricalSpec) and var_spec.name in dc.sources
            for dc in var_specs
        )
        if not needed_by_dc:
            tiff_path.unlink(missing_ok=True)

        # Release source_cache entries no longer needed
        if isinstance(var_spec, DerivedVariableSpec):
            remaining = var_specs[i + 1 :]
            source_still_needed = any(
                isinstance(v, DerivedVariableSpec) and v.source == var_spec.source
                for v in remaining
            )
            if not source_still_needed and var_spec.source in source_cache:
                del source_cache[var_spec.source]
        else:
            # Raw var: check if any later derived var uses this as source
            remaining = var_specs[i + 1 :]
            needed_by_derived = any(
                isinstance(v, DerivedVariableSpec) and v.source == var_spec.name for v in remaining
            )
            if not needed_by_derived and var_spec.name in source_cache:
                del source_cache[var_spec.name]

        gc.collect()

    # Process derived categorical specs last — re-read source GeoTIFFs
    # from disk rather than holding all sources in memory.
    from hydro_param.data_access import CATEGORICAL_DERIVATION_FUNCTIONS

    dc_specs = [v for v in var_specs if isinstance(v, DerivedCategoricalSpec)]
    if dc_specs:
        import rioxarray  # noqa: F401

    for dc_spec in dc_specs:
        derive_fn = CATEGORICAL_DERIVATION_FUNCTIONS.get(dc_spec.method)
        if derive_fn is None:
            raise ValueError(f"No categorical derivation function for method '{dc_spec.method}'")

        # Re-read source GeoTIFFs from disk

        source_das = []
        missing = []
        for src_name in dc_spec.sources:
            src_tiff = work_dir / f"{src_name}.tif"
            if not src_tiff.exists():
                missing.append(src_name)
                continue
            da = cast("xr.DataArray", rioxarray.open_rasterio(src_tiff))
            source_das.append(da.squeeze("band", drop=True))

        if missing:
            msg = (
                f"Cannot derive categorical variable '{dc_spec.name}': "
                f"missing source GeoTIFFs {missing} in {work_dir}. "
                f"This usually means the source variables failed to process "
                f"or were cleaned up prematurely."
            )
            raise FileNotFoundError(msg)

        # Classify pixels
        try:
            classified_da = derive_fn(*source_das)
        except TypeError as exc:
            raise ValueError(
                f"Derivation function '{dc_spec.method}' for '{dc_spec.name}' "
                f"received {len(source_das)} source arrays (from {dc_spec.sources}), "
                f"but the function signature does not match: {exc}"
            ) from exc

        # Free source arrays immediately
        del source_das
        gc.collect()

        # Save classified raster and run categorical zonal stats
        classified_tiff = work_dir / f"{dc_spec.name}.tif"
        save_to_geotiff(classified_da, classified_tiff)
        del classified_da

        df = processor.process(
            fabric=batch_fabric,
            tiff_path=classified_tiff,
            variable_name=dc_spec.name,
            id_field=config.target_fabric.id_field,
            engine=config.processing.engine,
            statistics=ds_req.statistics,
            categorical=True,
            source_crs=entry.crs,
            x_coord=entry.x_coord,
            y_coord=entry.y_coord,
        )
        results[dc_spec.name] = df

        # Clean up classified GeoTIFF; only delete source GeoTIFFs if no
        # remaining dc_specs still need them.
        classified_tiff.unlink(missing_ok=True)
        remaining_dc = dc_specs[dc_specs.index(dc_spec) + 1 :]
        for src_name in dc_spec.sources:
            still_needed = any(src_name in other.sources for other in remaining_dc)
            if not still_needed:
                (work_dir / f"{src_name}.tif").unlink(missing_ok=True)
        gc.collect()

    return results


def _process_temporal(
    fabric: gpd.GeoDataFrame,
    entry: DatasetEntry,
    ds_req: DatasetRequest,
    var_specs: list[AnyVariableSpec],
    config: PipelineConfig,
) -> xr.Dataset:
    """Process a temporal dataset using gdptools WeightGen + AggGen.

    Temporal datasets (e.g., gridMET, SNODAS, CONUS404-BA) are processed
    over the full fabric without spatial batching, because the gdptools
    ``WeightGen`` + ``AggGen`` pathway computes area-weighted time series
    in a single pass.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Target fabric (full, not spatially batched).
    entry : DatasetEntry
        Registry entry for the dataset.  Must have ``temporal=True`` and
        a strategy of ``"nhgf_stac"`` or ``"climr_cat"``.
    ds_req : DatasetRequest
        Pipeline config request with ``time_period`` and ``statistics``.
    var_specs : list[VariableSpec | DerivedVariableSpec]
        Resolved variable specifications.  Derived variables are not
        supported for temporal datasets.
    config : PipelineConfig
        Pipeline configuration (id_field, etc.).

    Returns
    -------
    xr.Dataset
        Temporal dataset with ``(time, features)`` dimensions containing
        area-weighted aggregated values.

    Raises
    ------
    NotImplementedError
        If derived variables are requested, or if the dataset strategy
        does not support temporal processing.
    ValueError
        If no statistics are specified in the dataset request.

    Notes
    -----
    Only the first statistic in ``ds_req.statistics`` is used.  If
    multiple statistics are provided, a warning is logged and subsequent
    entries are ignored.
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

    Temporal datasets that span multiple years are processed one year at a
    time to keep individual output files manageable and to enable resume
    support at year granularity.

    Parameters
    ----------
    time_period : list[str]
        ``[start, end]`` ISO date strings (``"YYYY-MM-DD"``).

    Returns
    -------
    list[list[str]]
        List of ``[start, end]`` pairs, one per calendar year.  The first
        chunk starts on the original start date; the last chunk ends on
        the original end date; intermediate chunks span full calendar years.

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


def _write_variable_file(
    var_name: str,
    df: pd.DataFrame,
    category: str,
    config: PipelineConfig,
    feature_ids: object,
) -> Path:
    """Write a single per-variable CSV file with standardized column naming.

    Rename statistic columns using ``var_name_{stat}`` (or just ``var_name``
    when the only statistic is ``"mean"``), reindex to the full set of
    feature IDs (inserting NaN for features not covered by any batch), sort
    by ``id_field``, and write a CSV with ``id_field`` as the first column.

    This function is called once per variable after all batches have been
    merged, producing the incremental output that stage 5 later normalizes.

    Parameters
    ----------
    var_name : str
        Variable name used as the filename stem and column prefix.
    df : pd.DataFrame
        Merged zonal statistics for this variable, concatenated from all
        spatial batches.  Index is the fabric ``id_field``.
    category : str
        Dataset category (e.g., ``"topography"``), used as the output
        subdirectory name.
    config : PipelineConfig
        Pipeline configuration (provides ``output.path`` and
        ``target_fabric.id_field``).
    feature_ids : array-like
        Complete set of feature IDs from the target fabric, used to
        reindex the output so every feature has a row.

    Returns
    -------
    Path
        Absolute path to the written CSV file (e.g.,
        ``output/topography/elevation.csv``).

    Warnings
    --------
    If the DataFrame index name does not match ``id_field``, a warning is
    logged and the index is renamed.  This may indicate a bug in the
    upstream zonal statistics output.
    """
    id_field = config.target_fabric.id_field
    output_dir = config.output.path
    var_dir = output_dir / category
    var_dir.mkdir(parents=True, exist_ok=True)

    # Rename columns: "mean" → var_name, others → var_name_{col}
    rename_map = {col: (var_name if col == "mean" else f"{var_name}_{col}") for col in df.columns}
    out_df = df.rename(columns=rename_map)

    # Warn if index name doesn't match id_field (may indicate upstream bug)
    if not (hasattr(out_df.index, "name") and out_df.index.name == id_field):
        logger.warning(
            "Index name mismatch for variable '%s': expected '%s', got '%s'. "
            "Renaming index — this may indicate a bug in zonal statistics output.",
            var_name,
            id_field,
            getattr(out_df.index, "name", None),
        )
        out_df.index.name = id_field

    # Reindex to full feature set (NaN for missing) and sort
    out_df = out_df.reindex(feature_ids).sort_index()

    out_path = var_dir / f"{var_name}.csv"
    out_df.to_csv(out_path, index=True)
    logger.info("Wrote %s → %s", var_name, out_path)
    return out_path


def _write_temporal_file(
    ds_name: str,
    ds: xr.Dataset,
    category: str,
    config: PipelineConfig,
) -> Path:
    """Write a single temporal output file in NetCDF or Parquet format.

    Temporal files are written immediately after each year-chunk is
    processed, keeping memory usage low for multi-year datasets.

    Parameters
    ----------
    ds_name : str
        Dataset name with year suffix (e.g., ``"gridmet_2020"``), used
        as the filename stem.
    ds : xr.Dataset
        Temporal dataset with ``(time, features)`` dimensions.
    category : str
        Dataset category (e.g., ``"climate"``), used as the output
        subdirectory name.
    config : PipelineConfig
        Pipeline configuration (provides ``output.path`` and
        ``output.format``).

    Returns
    -------
    Path
        Path to the written output file (e.g.,
        ``output/climate/gridmet_2020_temporal.nc``).

    Raises
    ------
    ValueError
        If ``config.output.format`` is not ``"netcdf"`` or ``"parquet"``.
    """
    output_dir = config.output.path
    var_dir = output_dir / category
    var_dir.mkdir(parents=True, exist_ok=True)

    if config.output.format == "netcdf":
        temporal_path = var_dir / f"{ds_name}_temporal.nc"
        ds.to_netcdf(temporal_path)
    elif config.output.format == "parquet":
        temporal_path = var_dir / f"{ds_name}_temporal.parquet"
        ds.to_dataframe().to_parquet(temporal_path)
    else:
        raise ValueError(
            f"Unsupported output format '{config.output.format}' for temporal "
            f"dataset '{ds_name}'. Supported formats: 'netcdf', 'parquet'."
        )
    logger.info("Wrote temporal %s → %s", ds_name, temporal_path)
    return temporal_path


def _save_manifest_to_disk(
    manifest: _manifest_mod.PipelineManifest,
    output_dir: Path,
) -> None:
    """Save the pipeline manifest to disk, logging errors non-fatally.

    The manifest enables resume support by recording which datasets have
    been processed and their fingerprints.  A write failure is logged but
    does not crash the pipeline -- outputs are still intact, only resume
    support may be degraded on the next run.

    Parameters
    ----------
    manifest : PipelineManifest
        Current manifest state to persist.
    output_dir : Path
        Output directory where the manifest JSON file is written.
    """
    try:
        manifest.save(output_dir)
    except OSError as exc:
        logger.error(
            "Failed to save manifest to %s: %s. "
            "Resume support may not work on next run, but outputs are intact.",
            output_dir,
            exc,
        )


def _save_manifest(
    manifest: _manifest_mod.PipelineManifest,
    ds_name: str,
    ds_fp: str,
    static_files: dict[str, Path],
    temporal_files: dict[str, Path],
    output_dir: Path,
) -> None:
    """Update a single dataset's manifest entry and persist to disk.

    Called after each dataset completes processing to record its
    fingerprint and output file paths for resume support.

    Parameters
    ----------
    manifest : PipelineManifest
        Manifest object to update in place.
    ds_name : str
        Dataset name (key in ``manifest.entries``).
    ds_fp : str
        Dataset fingerprint (hash of request + entry + processing config).
    static_files : dict[str, Path]
        Per-variable CSV paths for this dataset.
    temporal_files : dict[str, Path]
        Per-year temporal output paths for this dataset.
    output_dir : Path
        Output directory for relativizing file paths in the manifest.
    """
    manifest.entries[ds_name] = _manifest_mod.make_manifest_entry(
        ds_fp, static_files, temporal_files, output_dir
    )
    _save_manifest_to_disk(manifest, output_dir)


def stage4_process(
    fabric: gpd.GeoDataFrame,
    resolved: list[
        tuple[
            DatasetEntry,
            DatasetRequest,
            list[AnyVariableSpec],
        ]
    ],
    config: PipelineConfig,
) -> Stage4Results:
    """Stage 4: Process all datasets with spatial batching and incremental writes.

    Iterate over each resolved dataset.  Static datasets are processed
    through the spatial batch loop (KD-tree batches, per-batch GeoTIFF
    fetch, gdptools ``ZonalGen``).  Temporal datasets skip batching and
    are processed full-fabric via ``WeightGen`` + ``AggGen``.

    Per-variable CSV files (static) and per-year NetCDF/Parquet files
    (temporal) are written incrementally as each dataset completes,
    reducing peak memory usage compared to accumulating all results.

    Resume support is provided via a manifest that records dataset
    fingerprints and output paths.  When ``config.processing.resume`` is
    ``True``, datasets whose outputs are already current are skipped.

    Parameters
    ----------
    fabric : gpd.GeoDataFrame
        Target fabric with a ``batch_id`` column added by
        :func:`~hydro_param.batching.spatial_batch`.
    resolved : list[tuple[DatasetEntry, DatasetRequest, list[...]]]
        Resolved dataset entries from :func:`stage2_resolve_datasets`.
    config : PipelineConfig
        Pipeline configuration (output path, engine, batch size, resume
        flag, etc.).

    Returns
    -------
    Stage4Results
        Aggregated file paths and category metadata for all processed
        datasets.

    Raises
    ------
    ValueError
        If duplicate result keys are detected across datasets or years
        (indicates a config collision).

    Notes
    -----
    Multi-year static datasets (``year: [2020, 2021]``) produce
    year-suffixed result keys (e.g., ``"land_cover_2020"``).  Temporal
    datasets are split into per-calendar-year chunks via
    :func:`_split_time_period_by_year`.
    """
    batch_ids = sorted(fabric["batch_id"].unique())
    logger.info("Stage 4: Processing %d datasets across %d batches", len(resolved), len(batch_ids))

    id_field = config.target_fabric.id_field
    feature_ids = fabric[id_field].values

    # Ensure output directory exists for incremental writes
    config.output.path.mkdir(parents=True, exist_ok=True)

    # Always create manifest for resume support.
    # The resume flag controls whether completed datasets are *skipped*,
    # not whether the manifest is *written*.
    fab_fp = _manifest_mod.fabric_fingerprint(config)
    manifest = _manifest_mod.PipelineManifest(fabric_fingerprint=fab_fp)

    if config.processing.resume:
        existing = _manifest_mod.load_manifest(config.output.path)
        if existing is not None and existing.is_fabric_current(fab_fp):
            manifest = existing  # Preserve entries for skip checks
        elif existing is not None:
            logger.warning(
                "Fabric fingerprint changed — reprocessing all datasets (old=%s, new=%s)",
                existing.fabric_fingerprint,
                fab_fp,
            )

    static_files: dict[str, Path] = {}
    temporal_files: dict[str, Path] = {}
    categories: dict[str, str] = {}

    for ds_idx, (entry, ds_req, var_specs) in enumerate(resolved, 1):
        category = entry.category or "uncategorized"
        var_names = [v.name for v in var_specs]

        ds_fp = _manifest_mod.dataset_fingerprint(ds_req, entry, var_specs, config.processing)

        # Resume: check if this dataset can be skipped
        if config.processing.resume:
            if manifest.is_dataset_current(ds_req.name, ds_fp, config.output.path):
                cached = manifest.entries[ds_req.name]
                for k, rel in cached.static_files.items():
                    static_files[k] = config.output.path / rel
                    categories[k] = category
                for k, rel in cached.temporal_files.items():
                    temporal_files[k] = config.output.path / rel
                    categories[k] = category
                logger.info(
                    "Dataset %d/%d: %s — skipped (outputs current)",
                    ds_idx,
                    len(resolved),
                    ds_req.name,
                )
                continue

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

            # Track this dataset's temporal files explicitly
            ds_temporal_files: dict[str, Path] = {}

            for chunk_period in year_chunks:
                chunk_year = chunk_period[0][:4]
                t_chunk = time.perf_counter()
                chunk_req = ds_req.model_copy(update={"time_period": chunk_period})
                ds = _process_temporal(fabric, entry, chunk_req, var_specs, config)
                result_key = f"{ds_req.name}_{chunk_year}"
                categories[result_key] = category
                logger.info(
                    "  %s year %s: %d vars, %d time steps (%.1fs)",
                    ds_req.name,
                    chunk_year,
                    len(ds.data_vars),
                    ds.sizes.get("time", 0),
                    time.perf_counter() - t_chunk,
                )
                # Write temporal file immediately after each year chunk
                temporal_files[result_key] = _write_temporal_file(result_key, ds, category, config)
                ds_temporal_files[result_key] = temporal_files[result_key]

            logger.info("  %s complete (%.1fs)", ds_req.name, time.perf_counter() - t_ds)

            # Update manifest after temporal dataset completes
            _save_manifest(manifest, ds_req.name, ds_fp, {}, ds_temporal_files, config.output.path)

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

        # Collect batch results for this dataset only
        ds_batch_results: dict[str, list[pd.DataFrame]] = {}

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
                    ds_batch_results.setdefault(result_key, []).append(df)

            # Track categories with year-suffixed keys
            for var_spec in var_specs:
                result_key = f"{var_spec.name}_{year}" if year is not None else var_spec.name
                categories[result_key] = category

        # Merge and write per-variable files immediately after this dataset completes
        for var_key, dfs in ds_batch_results.items():
            if var_key in static_files:
                raise ValueError(
                    f"Duplicate static result key '{var_key}' from dataset "
                    f"'{ds_req.name}'. Overlapping variable names across "
                    f"datasets or years; adjust your configuration to avoid collisions."
                )
            merged_df = pd.concat(dfs)
            category = categories.get(var_key, "uncategorized")
            static_files[var_key] = _write_variable_file(
                var_key, merged_df, category, config, feature_ids
            )
            logger.info("  Merged %s: %d total features", var_key, len(merged_df))

        logger.info("  %s complete (%.1fs)", ds_req.name, time.perf_counter() - t_ds)

        # Update manifest after static dataset completes
        ds_static: dict[str, Path] = {k: static_files[k] for k in ds_batch_results}
        _save_manifest(manifest, ds_req.name, ds_fp, ds_static, {}, config.output.path)

    # Safety save: ensures manifest is on disk even if a per-dataset save was interrupted
    _save_manifest_to_disk(manifest, config.output.path)

    return Stage4Results(
        static_files=static_files, temporal_files=temporal_files, categories=categories
    )


def stage5_normalize_sir(
    stage4: Stage4Results,
    resolved: list[
        tuple[
            DatasetEntry,
            DatasetRequest,
            list[AnyVariableSpec],
        ]
    ],
    config: PipelineConfig,
) -> tuple[
    dict[str, Path],
    list[SIRVariableSchema],
    list[SIRValidationWarning],
    _manifest_mod.SIRManifestEntry,
]:
    """Stage 5: Normalize raw stage 4 output to canonical SIR format.

    Build a SIR schema from the resolved datasets, then normalize each
    raw per-variable CSV into a canonical SIR file with standardized
    variable names and units.  Temporal files are also normalized.
    Finally, validate the SIR files against the schema.

    The SIR (Standardized Internal Representation) is the contract
    between the generic pipeline and model plugins -- plugins consume
    SIR files with predictable names and units, never raw source output.

    Parameters
    ----------
    stage4 : Stage4Results
        Stage 4 results containing raw per-variable file paths.
    resolved : list[tuple[DatasetEntry, DatasetRequest, list[...]]]
        Resolved dataset entries from :func:`stage2_resolve_datasets`,
        used to build the SIR schema.
    config : PipelineConfig
        Pipeline configuration (output path, id_field, validation mode).

    Returns
    -------
    tuple[dict[str, Path], list[SIRVariableSchema], list[SIRValidationWarning], SIRManifestEntry]
        - Normalized SIR file paths (``sir/`` subdirectory)
        - Schema entries describing each SIR variable
        - Validation warnings (empty if all checks pass)
        - SIR manifest entry for persisting to the pipeline manifest

    Raises
    ------
    SIRValidationError
        If ``config.processing.sir_validation == "strict"`` and any
        validation warnings are found.

    See Also
    --------
    hydro_param.sir.normalize_sir : Per-file normalization logic.
    hydro_param.sir.validate_sir : SIR validation checks.
    """
    logger.info("Stage 5: SIR normalization")
    schema = build_sir_schema(resolved)
    logger.info("  SIR schema: %d variables", len(schema))

    sir_dir = config.output.path / "sir"
    sir_files = normalize_sir(
        raw_files=stage4.static_files,
        schema=schema,
        output_dir=sir_dir,
        id_field=config.target_fabric.id_field,
    )
    logger.info("  Normalized %d SIR files → %s", len(sir_files), sir_dir)

    # Normalize temporal files
    if stage4.temporal_files:
        temporal_sir = normalize_sir_temporal(
            temporal_files=stage4.temporal_files,
            schema=schema,
            resolved=resolved,
            output_dir=sir_dir,
        )
        sir_files.update(temporal_sir)
        logger.info("  Normalized %d temporal SIR files", len(temporal_sir))

    strict = config.processing.sir_validation == "strict"
    warnings = validate_sir(sir_files, schema, strict=strict)
    if warnings:
        logger.warning("  SIR validation: %d warnings", len(warnings))
        for w in warnings:
            logger.warning("    [%s] %s: %s", w.check_type, w.variable, w.message)
    else:
        logger.info("  SIR validation: passed")

    # Build SIR manifest entry for Phase 2 discovery
    output_path = config.output.path
    sir_manifest = _manifest_mod.SIRManifestEntry(
        static_files={
            k: str(v.relative_to(output_path))
            for k, v in sir_files.items()
            if str(v).endswith(".csv")
        },
        temporal_files={
            k: str(v.relative_to(output_path))
            for k, v in sir_files.items()
            if str(v).endswith(".nc")
        },
        sir_schema=[
            _manifest_mod.SIRSchemaEntry(
                name=s.canonical_name,
                units=s.canonical_units,
                statistic="categorical" if s.categorical else "continuous",
                source_dataset=s.dataset_name,
            )
            for s in schema
        ],
        completed_at=datetime.now(timezone.utc),
    )

    return sir_files, schema, warnings, sir_manifest


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pipeline_from_config(
    config: PipelineConfig,
    registry: DatasetRegistry,
) -> PipelineResult:
    """Execute the full 5-stage pipeline from pre-loaded config and registry.

    Orchestrate all pipeline stages in sequence: resolve fabric (stage 1),
    resolve datasets (stage 2), process datasets with spatial batching
    (stage 4), and normalize to SIR (stage 5).  Stage 3 (weights) is
    handled internally by gdptools during stage 4.

    GDAL HTTP timeout environment variables are set from
    ``config.processing.network_timeout`` for the duration of the pipeline
    and restored afterward.

    Parameters
    ----------
    config : PipelineConfig
        Validated pipeline configuration.
    registry : DatasetRegistry
        Dataset registry for resolving dataset names to entries.

    Returns
    -------
    PipelineResult
        Complete pipeline output including file paths for static and
        temporal outputs, the target fabric, normalized SIR files,
        schema, and validation warnings.  Use :meth:`PipelineResult.load_sir`
        to assemble a combined ``xr.Dataset`` on demand.

    See Also
    --------
    run_pipeline : Convenience wrapper that loads config and registry from paths.
    """
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("hydro-param pipeline: %s", config.output.sir_name)
    logger.info(
        "  Fabric: %s (id_field=%s)", config.target_fabric.path, config.target_fabric.id_field
    )
    logger.info(
        "  Datasets: %d, Engine: %s, Batch size: %d",
        len(config.flatten_datasets()),
        config.processing.engine,
        config.processing.batch_size,
    )
    logger.info("  Output: %s (%s)", config.output.path, config.output.format)
    logger.info("=" * 60)

    # Apply network timeout to GDAL HTTP operations (COG/vsicurl access)
    _timeout_s = str(config.processing.network_timeout)
    _prev_timeout = os.environ.get("GDAL_HTTP_TIMEOUT")
    _prev_connect = os.environ.get("GDAL_HTTP_CONNECTTIMEOUT")
    os.environ["GDAL_HTTP_TIMEOUT"] = _timeout_s
    os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = _timeout_s
    logger.info("  Network timeout: %ss", _timeout_s)

    try:
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

        # Stage 4: Process datasets + incremental writes
        t4 = time.perf_counter()
        results = stage4_process(fabric, resolved, config)
        logger.info(
            "Stage 4 complete: %d static vars, %d temporal datasets (%.1fs)",
            len(results.static_files),
            len(results.temporal_files),
            time.perf_counter() - t4,
        )

        # Stage 5: Normalize SIR
        t5 = time.perf_counter()
        sir_files, sir_schema, sir_warnings, sir_manifest_entry = stage5_normalize_sir(
            results, resolved, config
        )
        # Load manifest written by stage4 and append SIR section
        manifest = _manifest_mod.load_manifest(config.output.path)
        if manifest is None:
            manifest = _manifest_mod.PipelineManifest()
        manifest.sir = sir_manifest_entry
        _save_manifest_to_disk(manifest, config.output.path)
        logger.info("Stage 5 complete (%.1fs)", time.perf_counter() - t5)

        elapsed = time.perf_counter() - t0
        logger.info("=" * 60)
        logger.info(
            "Pipeline complete: %d static + %d temporal datasets, %d features, %.1f seconds",
            len(results.static_files),
            len(results.temporal_files),
            len(fabric),
            elapsed,
        )
        logger.info("=" * 60)

        return PipelineResult(
            output_dir=config.output.path,
            static_files=results.static_files,
            temporal_files=results.temporal_files,
            categories=results.categories,
            fabric=fabric,
            sir_files=sir_files,
            sir_schema=sir_schema,
            sir_warnings=sir_warnings,
        )
    finally:
        # Restore previous GDAL timeout settings
        if _prev_timeout is None:
            os.environ.pop("GDAL_HTTP_TIMEOUT", None)
        else:
            os.environ["GDAL_HTTP_TIMEOUT"] = _prev_timeout
        if _prev_connect is None:
            os.environ.pop("GDAL_HTTP_CONNECTTIMEOUT", None)
        else:
            os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = _prev_connect


def run_pipeline(
    config_path: str | Path,
    registry_path: str | Path | None = None,
) -> PipelineResult:
    """Execute the full parameterization pipeline from file paths.

    Convenience wrapper that loads the pipeline config and dataset registry
    from disk, then delegates to :func:`run_pipeline_from_config`.

    Parameters
    ----------
    config_path : str or Path
        Path to the pipeline YAML config file.
    registry_path : str or Path or None
        Path to a dataset registry YAML file or directory of YAML files.
        Defaults to the built-in registry bundled with the package.

    Returns
    -------
    PipelineResult
        Complete pipeline output.  Use :meth:`PipelineResult.load_sir`
        to assemble a combined ``xr.Dataset`` on demand.

    See Also
    --------
    run_pipeline_from_config : Core pipeline execution with pre-loaded objects.
    hydro_param.config.load_config : YAML config loader.
    hydro_param.dataset_registry.load_registry : Registry loader.
    """
    config = load_config(config_path)
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY
    registry = load_registry(registry_path, overlay_dirs=[USER_REGISTRY_DIR])

    return run_pipeline_from_config(config, registry)


def main() -> int:
    """Run the pipeline from the command line via ``python -m hydro_param.pipeline``.

    Parse ``sys.argv`` for a config path and optional registry path, configure
    logging, and execute the pipeline.  This is a minimal entry point for
    debugging; the primary CLI is :mod:`hydro_param.cli`.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
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
