"""Pipeline configuration: Pydantic models and YAML loader.

Define the declarative configuration schema for the hydro-param pipeline,
matching design.md section 11.6.  Configs express *what* to compute (target
fabric, datasets, statistics, output format) but never *how* -- all processing
logic lives in Python code, not in YAML.

The schema is validated at load time by Pydantic v2 so that invalid configs
fail fast with clear error messages before any data is fetched.

See Also
--------
hydro_param.pipeline : Orchestrator that consumes these config objects.
hydro_param.dataset_registry : Registry that resolves dataset names referenced
    in :class:`DatasetRequest`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from hydro_param.dataset_registry import VALID_CATEGORIES


class TargetFabricConfig(BaseModel):
    """Specify the target polygon fabric to parameterize.

    The target fabric is the spatial mesh (catchments, HRUs, grid cells) whose
    features receive zonal statistics from source datasets.  The fabric must be
    a pre-existing geospatial file -- hydro-param does not fetch or subset
    fabrics (use pynhd/pygeohydro upstream).

    Attributes
    ----------
    path : Path
        Path to the fabric file (GeoPackage, GeoParquet, or Shapefile).
    id_field : str
        Column name containing unique feature identifiers.  This becomes the
        index/dimension name in all output files and the SIR xarray Dataset.
    crs : str
        Coordinate reference system of the fabric file as an EPSG string.
        Defaults to ``"EPSG:4326"`` (WGS 84).

    Notes
    -----
    The ``id_field`` propagates through the entire pipeline: it controls the
    xarray dimension name in the SIR, the CSV index column, and the feature
    matching in the pywatershed derivation plugin.  Typical values are
    ``"nhm_id"`` (pywatershed/NHM), ``"featureid"`` (NHDPlus), or
    ``"hru_id"`` (custom fabrics).
    """

    path: Path
    id_field: str
    crs: str = "EPSG:4326"


class DomainConfig(BaseModel):
    """Define the spatial domain that restricts which fabric features are processed.

    When a domain is configured, stage 1 clips the target fabric to the
    specified extent before any data fetching or zonal statistics.  When
    omitted, the full fabric extent is used.

    Only ``type="bbox"`` is currently implemented; HUC and gage-based
    subsetting are planned.

    Attributes
    ----------
    type : {"bbox", "huc2", "huc4", "gage"}
        Domain specification method.
    bbox : list[float] or None
        Bounding box as ``[west, south, east, north]`` in EPSG:4326 (degrees).
        Required when ``type="bbox"``.
    id : str or None
        Identifier for HUC or gage-based domains (e.g., HUC-2 code or
        USGS gage ID).  Required when ``type`` is ``"huc2"``, ``"huc4"``,
        or ``"gage"``.

    Raises
    ------
    ValueError
        If the required field for the chosen ``type`` is missing.
    """

    type: Literal["bbox", "huc2", "huc4", "gage"]
    bbox: list[float] | None = None
    id: str | None = None

    @model_validator(mode="after")
    def _validate_domain(self) -> DomainConfig:
        """Ensure the correct field is provided for the chosen domain type."""
        if self.type == "bbox" and self.bbox is None:
            raise ValueError("bbox domain requires 'bbox' field")
        if self.type in ("huc2", "huc4", "gage") and self.id is None:
            raise ValueError(f"{self.type} domain requires 'id' field")
        return self


class DatasetRequest(BaseModel):
    """Request a dataset and its variables for pipeline processing.

    Each entry within a category list in the ``datasets:`` dict of a pipeline
    YAML config becomes one ``DatasetRequest``.  The ``name`` is resolved
    against the dataset registry to obtain fetch strategy, STAC collection,
    CRS, and variable metadata.

    Attributes
    ----------
    name : str
        Dataset name as it appears in the registry (e.g., ``"dem_3dep_10m"``).
    source : Path or None
        Local file path override for ``local_tiff`` datasets.  When set, this
        takes precedence over the registry-level ``source`` field.
    variables : list[str]
        Variable names to extract (e.g., ``["elevation", "slope"]``).
        Empty list means no variables requested (unusual but valid).
    statistics : list[str]
        Zonal statistics to compute for each variable.  Defaults to
        ``["mean"]``.  Common values: ``"mean"``, ``"majority"``,
        ``"minority"``, ``"sum"``, ``"min"``, ``"max"``, ``"median"``.
    year : int or list[int] or None
        Year(s) for multi-year static datasets (e.g., NLCD annual).  When a
        list is provided, the pipeline iterates over each year and produces
        year-suffixed output keys (e.g., ``"land_cover_2019"``).  Valid range:
        1900--2100.
    time_period : list[str] or None
        ``[start, end]`` ISO date strings (``"YYYY-MM-DD"``) for temporal
        datasets (e.g., gridMET, SNODAS).  Required when the registry marks
        the dataset as ``temporal: true``.

    Raises
    ------
    ValueError
        If ``year`` list is empty, a year is outside 1900--2100, or
        ``time_period`` dates are invalid or out of order.

    See Also
    --------
    hydro_param.dataset_registry.DatasetEntry : Registry metadata resolved
        from ``name``.
    """

    name: str
    source: Path | None = None
    variables: list[str] = []
    statistics: list[str] = Field(default_factory=lambda: ["mean"])
    year: int | list[int] | None = Field(default=None)
    time_period: list[str] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="[start, end] ISO date strings for temporal datasets",
    )

    @model_validator(mode="after")
    def _validate_year(self) -> DatasetRequest:
        """Validate year values are within 1900--2100."""
        if self.year is None:
            return self
        if isinstance(self.year, list) and len(self.year) == 0:
            raise ValueError("year list cannot be empty")
        years = [self.year] if isinstance(self.year, int) else self.year
        for y in years:
            if not (1900 <= y <= 2100):
                raise ValueError(f"Year {y} outside valid range 1900-2100")
        return self

    @model_validator(mode="after")
    def _validate_time_period(self) -> DatasetRequest:
        """Validate time_period dates are valid ISO format and in order."""
        if self.time_period is not None:
            start_str, end_str = self.time_period
            try:
                start = date.fromisoformat(start_str)
                end = date.fromisoformat(end_str)
            except ValueError as exc:
                raise ValueError(
                    f"time_period dates must be valid ISO format (YYYY-MM-DD): {exc}"
                ) from exc
            if start > end:
                raise ValueError(f"time_period start ({start_str}) must be <= end ({end_str})")
        return self


class OutputConfig(BaseModel):
    """Configure pipeline output location and format.

    Attributes
    ----------
    path : Path
        Directory for output files.  Created automatically if it does not
        exist.  Subdirectories are created per dataset category (e.g.,
        ``topography/``, ``soils/``).  Defaults to ``"./output"``.
    format : {"netcdf", "parquet"}
        File format for temporal output.  Static per-variable files are
        always written as CSV.  Defaults to ``"netcdf"``.
    sir_name : str
        Human-readable name for the output, used in CF-1.8 metadata
        attributes and log messages.  Defaults to ``"result"``.
    """

    path: Path = Path("./output")
    format: Literal["netcdf", "parquet"] = "netcdf"
    sir_name: str = "result"


class ProcessingConfig(BaseModel):
    """Control processing engine, batching, fault tolerance, and networking.

    Attributes
    ----------
    engine : {"exactextract", "serial"}
        Zonal statistics engine.  ``"exactextract"`` uses the
        exactextract C++ library via gdptools for fast, coverage-weighted
        statistics.  ``"serial"`` is a pure-Python fallback.  Defaults to
        ``"exactextract"``.
    batch_size : int
        Maximum number of features per spatial batch.  KD-tree recursive
        bisection groups nearby features to minimize data fetch extent.
        Must be > 0.  Defaults to 500.
    resume : bool
        When ``True``, skip datasets whose outputs are already current
        (checked via the pipeline manifest fingerprint).  Defaults to
        ``False``.
    sir_validation : {"tolerant", "strict"}
        SIR validation mode for stage 5.  ``"strict"`` raises on any
        validation warning; ``"tolerant"`` logs warnings and continues.
        Defaults to ``"tolerant"``.
    network_timeout : int
        Timeout in seconds for GDAL HTTP operations (COG/vsicurl access).
        Applied to both ``GDAL_HTTP_TIMEOUT`` and
        ``GDAL_HTTP_CONNECTTIMEOUT`` environment variables.  Must be > 0.
        Defaults to 120.

    """

    engine: Literal["exactextract", "serial"] = "exactextract"
    batch_size: int = Field(default=500, gt=0)
    resume: bool = False
    sir_validation: Literal["tolerant", "strict"] = "tolerant"
    network_timeout: int = Field(default=120, gt=0, description="Network timeout in seconds")


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration loaded from a YAML file.

    This is the root model that :func:`load_config` deserializes.  It
    composes all sub-configs and is consumed by every pipeline stage.

    Attributes
    ----------
    target_fabric : TargetFabricConfig
        Polygon mesh to parameterize.
    domain : DomainConfig or None
        Optional spatial subsetting.  When ``None``, the full fabric
        extent is used.
    datasets : dict[str, list[DatasetRequest]]
        Datasets organized by category (e.g., ``"topography"``, ``"soils"``).
        Category keys must be members of
        :data:`~hydro_param.dataset_registry.VALID_CATEGORIES`.
    output : OutputConfig
        Output location and format.
    processing : ProcessingConfig
        Engine, batching, and fault-tolerance settings.

    See Also
    --------
    load_config : Load and validate a YAML file into this model.
    hydro_param.pipeline.run_pipeline : Execute the pipeline from a config path.
    """

    target_fabric: TargetFabricConfig
    domain: DomainConfig | None = None
    datasets: dict[str, list[DatasetRequest]]
    output: OutputConfig = OutputConfig()
    processing: ProcessingConfig = ProcessingConfig()

    @model_validator(mode="after")
    def _validate_dataset_categories(self) -> PipelineConfig:
        """Validate that all dataset category keys are known registry categories."""
        unknown = set(self.datasets.keys()) - VALID_CATEGORIES
        if unknown:
            raise ValueError(
                f"Unknown dataset categories: {sorted(unknown)}. "
                f"Valid categories: {sorted(VALID_CATEGORIES)}"
            )
        return self

    def flatten_datasets(self) -> list[DatasetRequest]:
        """Flatten themed dataset dict into a single list for pipeline stages.

        Bridge the category-keyed config format to pipeline stages that expect
        a flat iterable of dataset requests.  This allows pipeline internals to
        remain agnostic to the themed grouping while the config YAML stays
        organized by domain category.

        Returns
        -------
        list[DatasetRequest]
            All dataset requests from all categories, preserving order
            within each category.

        Notes
        -----
        Dict insertion order (guaranteed since Python 3.7) preserves
        intra-category order.  Cross-category order follows YAML key order
        but is not semantically meaningful -- pipeline stages process each
        dataset independently.
        """
        return [ds for ds_list in self.datasets.values() for ds in ds_list]


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline YAML config file.

    Parse the YAML file at *path* and return a fully validated
    :class:`PipelineConfig`.  Pydantic model validators run during
    construction, so any schema violations raise immediately with
    descriptive error messages.

    After validation, all relative paths (``target_fabric.path``,
    ``output.path``, per-dataset ``source``) are resolved to absolute
    paths using the current working directory.  This ensures that
    downstream operations (manifest save/load, file existence checks)
    work consistently regardless of internal path manipulation.

    Parameters
    ----------
    path : str or Path
        Path to a YAML pipeline configuration file.

    Returns
    -------
    PipelineConfig
        Validated pipeline configuration with all paths resolved to
        absolute paths, ready for
        :func:`~hydro_param.pipeline.run_pipeline_from_config`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file is not valid YAML.
    pydantic.ValidationError
        If the YAML content does not match the config schema.

    Notes
    -----
    Relative paths in the YAML are interpreted relative to the current
    working directory (the standard convention when running
    ``hydro-param run configs/pipeline.yml`` from the project root).
    Absolute paths are left unchanged.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    config = PipelineConfig(**raw)
    return _resolve_paths(config)


def _resolve_paths(config: PipelineConfig) -> PipelineConfig:
    """Resolve all relative paths in a PipelineConfig to absolute.

    Convert ``target_fabric.path``, ``output.path``, and per-dataset
    ``source`` fields from relative to absolute using
    ``Path.resolve()`` (which anchors to the current working directory).
    Absolute paths remain absolute.

    Parameters
    ----------
    config : PipelineConfig
        Configuration with potentially relative paths.

    Returns
    -------
    PipelineConfig
        The same *config* object with path fields resolved in place.

    Notes
    -----
    This function mutates *config* in place and returns the same object
    for call-chaining convenience.  No copy is made.
    """
    config.target_fabric.path = config.target_fabric.path.resolve()
    config.output.path = config.output.path.resolve()
    for ds_list in config.datasets.values():
        for ds in ds_list:
            if ds.source is not None:
                ds.source = ds.source.resolve()
    return config
