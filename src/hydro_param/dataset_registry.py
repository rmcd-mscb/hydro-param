"""Dataset registry: load and resolve dataset definitions from YAML.

Map human-readable dataset names to access strategies, variable
specifications, and derivation rules.  The registry is the single source
of truth for "what datasets exist and how to access them."  Pipeline
stage 2 (``stage2_resolve_datasets``) consults the registry to resolve
user-requested datasets and variables into concrete access instructions.

The registry supports five access strategies (``stac_cog``, ``local_tiff``,
``nhgf_stac``, ``climr_cat``, ``native_zarr``/``converted_zarr``) and three
variable types (direct ``VariableSpec``, terrain-derived
``DerivedVariableSpec``, and multi-source categorical
``DerivedCategoricalSpec``).

References
----------
.. [1] docs/design.md, section 6.6 -- Dataset registry schema design.
.. [2] docs/design.md, section 11.3 -- Registry YAML conventions.

See Also
--------
hydro_param.config : Pipeline configuration schema (``DatasetRequest``).
hydro_param.data_access : Functions that use registry entries to fetch data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "climate",
        "geology",
        "hydrography",
        "land_cover",
        "snow",
        "soils",
        "topography",
        "water_bodies",
    }
)
"""Valid dataset registry categories.

These correspond to the per-category YAML files bundled in
``hydro_param/data/datasets/``.  Used by :class:`~hydro_param.config.PipelineConfig`
to validate category keys in the ``datasets:`` config section.
"""


class VariableSpec(BaseModel):
    """Describe a variable available directly in a source dataset.

    Each ``VariableSpec`` maps a logical variable name to its location
    within a source dataset (band number, STAC asset key, or file path
    override) and carries metadata for SIR normalization (units, long
    name, categorical flag).

    Attributes
    ----------
    name : str
        Logical variable name used throughout the pipeline (e.g.,
        ``"elevation"``, ``"land_cover"``, ``"ksat"``).
    band : int
        Raster band number for multi-band GeoTIFFs.  Default ``1``.
    units : str
        Source data units (e.g., ``"m"``, ``"log10(cm/hr)"``, ``"%"``).
        Empty string for dimensionless quantities.
    long_name : str
        Human-readable description for NetCDF attributes and documentation.
    native_name : str
        Variable name in the source data (e.g., OPeNDAP/CF name like
        ``"daily_mean_temperature_2m"``).  Required for temporal datasets
        to map gdptools output back to logical names.
    categorical : bool
        ``True`` for land-cover or other classification variables.
        Categorical variables produce per-class fraction columns in zonal
        statistics rather than continuous summary statistics.
    asset_key : str or None
        Per-variable STAC asset key override (e.g., ``"mukey"`` for
        gNATSGO).  When ``None``, uses the dataset-level ``asset_key``.
    source_override : str or None
        Per-variable source path or URL override (e.g., individual POLARIS
        VRT files).  When ``None``, uses the dataset-level ``source``.
    scale_factor : float or None
        Multiplicative scale factor for integer-encoded rasters (e.g.,
        ``0.01`` for values stored as ``value × 100``).  Follows
        CF-conventions ``scale_factor`` semantics.  When ``None``, no
        scaling is applied.  The pipeline applies this factor after
        zonal statistics so the SIR contains physically meaningful values.
    """

    name: str
    band: int = 1
    units: str = ""
    long_name: str = ""
    native_name: str = ""
    categorical: bool = False
    asset_key: str | None = None
    source_override: str | None = None
    scale_factor: float | None = None


class DerivedVariableSpec(BaseModel):
    """Describe a variable derived from another variable in the same dataset.

    Derived variables are computed from a source variable using a named
    method (e.g., slope and aspect from elevation via terrain analysis).
    They are resolved alongside direct variables in stage 2 and processed
    in stage 4.

    Attributes
    ----------
    name : str
        Logical name for the derived variable (e.g., ``"slope"``,
        ``"aspect"``).
    source : str
        Name of the source ``VariableSpec`` this is derived from (e.g.,
        ``"elevation"``).
    method : str
        Derivation method passed to the derivation function (e.g.,
        ``"horn"`` for Horn 1981 finite-difference terrain analysis).
        The derivation function is selected by variable ``name`` via
        ``hydro_param.data_access.DERIVATION_FUNCTIONS``.
    units : str
        Units of the derived variable (e.g., ``"degrees"``).
    long_name : str
        Human-readable description for metadata.
    """

    name: str
    source: str
    method: str
    units: str = ""
    long_name: str = ""


class DerivedCategoricalSpec(BaseModel):
    """Describe a categorical variable derived from multiple source variables.

    Multi-source categorical derivations classify pixels by combining
    two or more source bands (e.g., USDA texture triangle from
    sand/silt/clay percentages).  The result is a single-band
    categorical raster processed with categorical zonal statistics to
    produce per-class fraction columns.

    Unlike ``DerivedVariableSpec`` (single source, continuous output),
    this always produces categorical output with per-class fractions.

    Attributes
    ----------
    name : str
        Logical name for the derived variable (e.g.,
        ``"soil_texture"``).
    sources : list[str]
        Names of the source ``VariableSpec`` entries this is derived
        from (e.g., ``["sand", "silt", "clay"]``).  Must contain at
        least 2 entries.
    method : str
        Classification method key used to look up the derivation
        function via
        ``hydro_param.data_access.CATEGORICAL_DERIVATION_FUNCTIONS``.
    units : str
        Units of the derived variable (typically ``"class"``).
    long_name : str
        Human-readable description for metadata.
    """

    name: str
    sources: list[str]
    method: str
    units: str = ""
    long_name: str = ""

    @field_validator("sources")
    @classmethod
    def _check_min_sources(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            msg = "DerivedCategoricalSpec requires at least 2 sources"
            raise ValueError(msg)
        return v


#: Union of all variable specification types used throughout the pipeline.
AnyVariableSpec = VariableSpec | DerivedVariableSpec | DerivedCategoricalSpec


class DownloadFile(BaseModel):
    """Describe a single downloadable file in a multi-file dataset.

    Used by the ``hydro-param datasets download`` CLI command to stage
    local data files required by the ``local_tiff`` access strategy.

    Attributes
    ----------
    year : int
        Calendar year this file covers.
    variable : str
        Variable name this file provides (e.g., ``"ksat"``, ``"clay"``).
    url : str
        Direct download URL for the file.
    size_gb : float or None
        Approximate file size in gigabytes for progress reporting.
        ``None`` if unknown.
    """

    year: int
    variable: str
    url: str
    size_gb: float | None = None


class DownloadInfo(BaseModel):
    """Describe download provenance for datasets requiring local staging.

    Some datasets (e.g., POLARIS soil data, GFv1.1 rasters) cannot be
    accessed through STAC or OPeNDAP and must be downloaded to local disk
    before processing.  ``DownloadInfo`` records where to get the data,
    how large it is, and whether requester-pays access is needed.

    Supports two modes: **explicit files** (a fixed list of
    ``DownloadFile`` entries) and **template mode** (a URL template
    expanded over ``year_range x variables_available``).

    Attributes
    ----------
    url : str
        Single-file download URL (mutually exclusive with ``files``
        and ``url_template``).
    size_gb : float or None
        Approximate total download size in gigabytes.
    format : str
        File format description (e.g., ``"GeoTIFF"``, ``"VRT"``).
    notes : str
        Human-readable notes about access requirements.
    files : list[DownloadFile]
        Explicit list of downloadable files (multi-file datasets).
    url_template : str
        Python format string with ``{variable}`` and ``{year}``
        placeholders (e.g.,
        ``"https://example.com/{variable}_{year}.tif"``).
    year_range : list[int]
        Two-element ``[start, end]`` list for template expansion.
        Required when ``url_template`` is set.
    variables_available : list[str]
        Variable names available for template expansion.  Required when
        ``url_template`` is set.
    requester_pays : bool
        ``True`` if the data source requires requester-pays access
        (e.g., ``s3://usgs-landcover``).

    Raises
    ------
    ValueError
        If none of ``url``, ``files``, or ``url_template`` is provided,
        or if ``url_template`` is set without valid ``year_range`` and
        ``variables_available``.
    """

    url: str = ""
    size_gb: float | None = None
    format: str = ""
    notes: str = ""
    files: list[DownloadFile] = []
    url_template: str = ""
    year_range: list[int] = []
    variables_available: list[str] = []
    requester_pays: bool = False

    @model_validator(mode="after")
    def _require_download_source(self) -> DownloadInfo:
        """Validate that at least one download source is specified."""
        if not self.url and not self.files and not self.url_template:
            raise ValueError("DownloadInfo requires at least 'url', 'files', or 'url_template'")
        if self.url_template:
            if len(self.year_range) != 2 or self.year_range[0] > self.year_range[1]:
                raise ValueError(
                    "url_template requires 'year_range' as [start, end] with start <= end"
                )
            if not self.variables_available:
                raise ValueError("url_template requires a non-empty 'variables_available' list")
        return self

    def expand_files(
        self,
        *,
        years: set[int] | None = None,
        variables: set[str] | None = None,
    ) -> list[DownloadFile]:
        """Expand download sources into a concrete list of files.

        For **template mode**, iterate ``year_range x variables_available``
        and format the ``url_template`` with ``{variable}`` and ``{year}``
        placeholders.  For **explicit files mode**, return the ``files``
        list.  In both modes, optional ``years`` and ``variables`` filters
        restrict the output.

        Parameters
        ----------
        years : set[int] or None
            If given, only include files matching these calendar years.
        variables : set[str] or None
            If given, only include files matching these variable names.

        Returns
        -------
        list[DownloadFile]
            Expanded and filtered list of downloadable files.
        """
        if self.url_template:
            start, end = self.year_range
            result = []
            for yr in range(start, end + 1):
                if years is not None and yr not in years:
                    continue
                for var in self.variables_available:
                    if variables is not None and var not in variables:
                        continue
                    url = self.url_template.format(variable=var, year=yr)
                    result.append(DownloadFile(year=yr, variable=var, url=url))
            return result

        result = list(self.files)
        if years is not None:
            result = [f for f in result if f.year in years]
        if variables is not None:
            result = [f for f in result if f.variable in variables]
        return result


class DatasetEntry(BaseModel):
    """Describe a single dataset in the registry.

    Each entry captures everything needed to access, process, and normalize
    a source dataset: the access strategy, connection parameters (STAC
    catalog URL, collection, asset key, etc.), coordinate system, and the
    list of available variables.

    The ``strategy`` field determines which data access pathway is used:

    - ``"stac_cog"`` -- STAC COG via Planetary Computer (3DEP, gNATSGO).
    - ``"local_tiff"`` -- local GeoTIFF files (POLARIS, GFv1.1).
    - ``"nhgf_stac"`` -- NHGF STAC catalog (NLCD Annual on OSN).
    - ``"climr_cat"`` -- ClimateR-Catalog via OPeNDAP (gridMET).
    - ``"native_zarr"`` / ``"converted_zarr"`` -- Zarr stores (planned).

    Attributes
    ----------
    description : str
        Human-readable dataset description.
    strategy : str
        Data access strategy identifier.
    catalog_url : str or None
        STAC catalog URL (required for ``stac_cog``).
    collection : str or None
        STAC collection name (required for ``stac_cog`` and ``nhgf_stac``).
    asset_key : str
        Default STAC asset key.  Default ``"data"``.
    gsd : int or None
        Ground sample distance in metres (STAC COG spatial resolution).
    sign : str or None
        STAC signing method (e.g., ``"planetary-computer"``).
    source : str or None
        Local file path or remote URL for Zarr/local_tiff datasets.
    download : DownloadInfo or None
        Download provenance for datasets requiring local staging.
    catalog_id : str or None
        ClimateR-Catalog identifier (required for ``climr_cat``).
    crs : str
        Coordinate reference system as an EPSG string.  Default
        ``"EPSG:4326"``.
    x_coord : str
        Name of the x/longitude coordinate.  Default ``"x"``.
    y_coord : str
        Name of the y/latitude coordinate.  Default ``"y"``.
    t_coord : str or None
        Name of the time coordinate (required for temporal datasets).
    variables : list[VariableSpec]
        Variables directly available in this dataset.
    derived_variables : list[DerivedVariableSpec]
        Variables computed from other variables in this dataset.
    category : str
        Dataset category for grouping (e.g., ``"topography"``,
        ``"soils"``, ``"land_cover"``).
    temporal : bool
        ``True`` for time-indexed datasets (e.g., gridMET, SNODAS).
    time_step : {"daily", "monthly"} or None
        Temporal resolution of the dataset.  Required when
        ``temporal`` is ``True``.  ``None`` for static datasets.
    year_range : list[int] or None
        Two-element ``[start, end]`` list of available calendar years.
        Must satisfy ``start <= end``.

    Raises
    ------
    ValueError
        If required strategy-specific fields are missing, or if
        constraints are violated (e.g., temporal without ``t_coord``).
    """

    description: str = ""
    strategy: Literal[
        "stac_cog", "native_zarr", "converted_zarr", "local_tiff", "climr_cat", "nhgf_stac"
    ]
    catalog_url: str | None = None
    collection: str | None = None
    asset_key: str = "data"
    gsd: int | None = None
    sign: str | None = None
    source: str | None = None
    download: DownloadInfo | None = None
    catalog_id: str | None = None
    crs: str = "EPSG:4326"
    x_coord: str = "x"
    y_coord: str = "y"
    t_coord: str | None = None
    variables: list[VariableSpec] = []
    derived_variables: list[DerivedVariableSpec] = []
    derived_categorical_variables: list[DerivedCategoricalSpec] = []
    category: str = ""
    temporal: bool = False
    time_step: Literal["daily", "monthly"] | None = None
    year_range: list[int] | None = None

    @model_validator(mode="after")
    def _validate_strategy_fields(self) -> DatasetEntry:
        """Validate that strategy-specific required fields are present."""
        if self.strategy == "stac_cog":
            if not self.catalog_url or not self.collection:
                raise ValueError("stac_cog strategy requires 'catalog_url' and 'collection'")
        if self.strategy == "climr_cat":
            if not self.catalog_id:
                raise ValueError("climr_cat strategy requires 'catalog_id'")
            if not self.temporal:
                raise ValueError("climr_cat strategy requires 'temporal: true'")
        if self.strategy == "nhgf_stac":
            if not self.collection:
                raise ValueError("nhgf_stac strategy requires 'collection'")
        if self.temporal and not self.t_coord:
            raise ValueError("Temporal datasets require 't_coord'")
        if self.temporal and self.time_step is None:
            raise ValueError("Temporal datasets require 'time_step' (e.g., 'daily', 'monthly')")
        if self.temporal:
            missing = [v.name for v in self.variables if not v.native_name]
            if missing:
                raise ValueError(
                    f"Temporal datasets require 'native_name' on all variables. "
                    f"Missing native_name for: {', '.join(missing)}"
                )
        if self.year_range is not None:
            if len(self.year_range) != 2:
                raise ValueError("year_range must be a 2-element list [start, end]")
            if self.year_range[0] > self.year_range[1]:
                raise ValueError(
                    "year_range start must be <= end: "
                    f"got [{self.year_range[0]}, {self.year_range[1]}]"
                )
        return self


class DatasetRegistry(BaseModel):
    """Contain and query all registered datasets.

    Provides lookup by name and variable resolution across the full set
    of loaded datasets.  Typically created by ``load_registry()`` from
    one or more YAML files.

    Attributes
    ----------
    datasets : dict[str, DatasetEntry]
        Mapping of dataset name to entry.  Names must be unique across
        all registry files.

    See Also
    --------
    load_registry : Load a registry from YAML file(s).
    """

    datasets: dict[str, DatasetEntry]

    def get(self, name: str) -> DatasetEntry:
        """Look up a dataset by name.

        Parameters
        ----------
        name : str
            Dataset name as it appears in the registry YAML (e.g.,
            ``"3dep"``, ``"gnatsgo"``, ``"gridmet"``).

        Returns
        -------
        DatasetEntry
            The matching dataset entry.

        Raises
        ------
        KeyError
            If ``name`` is not found.  The error message lists all
            available dataset names for debugging.
        """
        if name not in self.datasets:
            available = ", ".join(sorted(self.datasets.keys()))
            raise KeyError(f"Dataset '{name}' not found in registry. Available: {available}")
        return self.datasets[name]

    def resolve_variable(
        self, dataset_name: str, variable_name: str
    ) -> VariableSpec | DerivedVariableSpec | DerivedCategoricalSpec:
        """Resolve a variable name to its specification within a dataset.

        Search direct variables, derived variables, and derived categorical
        variables in the named dataset.  Direct variables are checked first.

        Parameters
        ----------
        dataset_name : str
            Dataset name in the registry (e.g., ``"3dep"``).
        variable_name : str
            Variable name to look up (e.g., ``"elevation"``, ``"slope"``).

        Returns
        -------
        VariableSpec or DerivedVariableSpec or DerivedCategoricalSpec
            The matching variable specification.  Direct variables are
            checked first, then derived, then derived categorical.

        Raises
        ------
        KeyError
            If the dataset is not found in the registry, or the variable
            is not found in the dataset.  The error message lists all
            available variable names for debugging.
        """
        entry = self.get(dataset_name)
        for v in entry.variables:
            if v.name == variable_name:
                return v
        for dv in entry.derived_variables:
            if dv.name == variable_name:
                return dv
        for dcv in entry.derived_categorical_variables:
            if dcv.name == variable_name:
                return dcv
        available = (
            [v.name for v in entry.variables]
            + [dv.name for dv in entry.derived_variables]
            + [dcv.name for dcv in entry.derived_categorical_variables]
        )
        raise KeyError(
            f"Variable '{variable_name}' not found in dataset '{dataset_name}'. "
            f"Available: {', '.join(available)}"
        )


def get_all_dataset_names(registry: DatasetRegistry) -> set[str]:
    """Return the set of all dataset names in the registry.

    Parameters
    ----------
    registry : DatasetRegistry
        A loaded dataset registry.

    Returns
    -------
    set[str]
        All dataset names (e.g., ``{"dem_3dep_10m", "gridmet", ...}``).
    """
    return set(registry.datasets.keys())


def load_registry(
    path: str | Path,
    *,
    overlay_dirs: list[Path] | None = None,
) -> DatasetRegistry:
    """Load a dataset registry from YAML file(s), with optional overlays.

    When ``path`` is a directory, all ``*.yml`` and ``*.yaml`` files are
    loaded and merged into a single registry.  Dataset names must be
    unique across all files -- duplicates raise ``ValueError``.

    Overlay directories (e.g., ``~/.hydro-param/datasets/``) are scanned
    after the primary registry.  Overlay entries are merged into the
    result; on name collision, the overlay entry replaces the primary
    entry (no partial merge).  Non-existent or empty overlay directories
    are silently skipped.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a single registry YAML file, or a directory containing
        per-category YAML files (e.g., the bundled
        ``hydro_param.data.datasets``).  Each file must have a top-level
        ``datasets:`` key mapping dataset names to entries.
    overlay_dirs : list[Path] or None
        Optional list of directories containing user-local registry
        overlays.  Each directory is scanned for ``*.yml``/``*.yaml``
        files.  Later directories take precedence over earlier ones.

    Returns
    -------
    DatasetRegistry
        Merged registry containing all datasets found.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist, is neither file nor directory, or
        the directory contains no YAML files with datasets.
    ValueError
        If a dataset name appears in more than one YAML file within
        the *primary* registry directory.  Overlay collisions with the
        primary registry are resolved silently (overlay wins).

    Examples
    --------
    >>> from hydro_param.pipeline import DEFAULT_REGISTRY
    >>> registry = load_registry(DEFAULT_REGISTRY)
    >>> entry = registry.get("dem_3dep_10m")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry path does not exist: {path}")
    if path.is_file():
        registry = _load_registry_file(path)
    elif path.is_dir():
        registry = _load_registry_dir(path)
    else:
        raise FileNotFoundError(f"Registry path is neither a file nor directory: {path}")

    if overlay_dirs:
        registry = _merge_overlays(registry, overlay_dirs)

    return registry


def _load_registry_file(path: Path) -> DatasetRegistry:
    """Load a single registry YAML file and return a validated registry."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DatasetRegistry(**raw)


def _load_registry_dir(directory: Path) -> DatasetRegistry:
    """Load and merge all YAML files in a registry directory.

    Files are processed in sorted order for deterministic results.  YAML
    files that parse as empty or lack a ``datasets:`` key are silently
    skipped.

    Parameters
    ----------
    directory : pathlib.Path
        Directory containing ``*.yml`` and/or ``*.yaml`` files, each
        expected to have a ``datasets:`` root key.

    Returns
    -------
    DatasetRegistry
        Merged registry containing datasets from all files.

    Raises
    ------
    FileNotFoundError
        If the directory contains no YAML files or no files with
        datasets.
    ValueError
        If a dataset name appears in more than one file.  The error
        message identifies both conflicting files.
    """
    yaml_files = sorted(list(directory.glob("*.yml")) + list(directory.glob("*.yaml")))
    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML files (*.yml, *.yaml) found in registry directory: {directory}"
        )

    merged: dict[str, DatasetEntry] = {}
    source_files: dict[str, str] = {}
    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
        if raw is None or "datasets" not in raw:
            continue
        partial = DatasetRegistry(**raw)
        for name, entry in partial.datasets.items():
            if name in merged:
                raise ValueError(
                    f"Duplicate dataset name '{name}': found in "
                    f"'{yaml_file.name}' and '{source_files[name]}'. "
                    f"Dataset names must be unique across all registry files."
                )
            merged[name] = entry
            source_files[name] = yaml_file.name

    if not merged:
        raise FileNotFoundError(f"No datasets found in any YAML file in: {directory}")

    return DatasetRegistry(datasets=merged)


def _merge_overlays(base: DatasetRegistry, overlay_dirs: list[Path]) -> DatasetRegistry:
    """Merge user-local overlay datasets into a base registry.

    Scan each overlay directory for YAML files and merge their datasets
    into *base*.  Overlay entries replace base entries on name collision
    (no partial merge).  Non-existent or empty directories are silently
    skipped.

    Parameters
    ----------
    base : DatasetRegistry
        The primary (bundled) registry.
    overlay_dirs : list[Path]
        Directories to scan for overlay YAML files.  Later directories
        take precedence over earlier ones.

    Returns
    -------
    DatasetRegistry
        A new registry with overlay entries merged in.
    """
    merged = dict(base.datasets)
    for overlay_dir in overlay_dirs:
        if not overlay_dir.is_dir():
            logger.debug("Overlay directory does not exist, skipping: %s", overlay_dir)
            continue
        yaml_files = sorted(list(overlay_dir.glob("*.yml")) + list(overlay_dir.glob("*.yaml")))
        if not yaml_files:
            logger.debug("No YAML files in overlay directory: %s", overlay_dir)
            continue
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    raw = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                logger.warning("Could not parse overlay YAML, skipping: %s\n%s", yaml_file, exc)
                continue
            if raw is None:
                logger.warning("Overlay file is empty, skipping: %s", yaml_file)
                continue
            if "datasets" not in raw:
                logger.warning(
                    "Overlay file has no 'datasets' key (found keys: %s), skipping: %s",
                    list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__,
                    yaml_file,
                )
                continue
            try:
                partial = DatasetRegistry(**raw)
            except ValidationError as exc:
                logger.warning(
                    "Overlay file has invalid entries, skipping: %s\n%s",
                    yaml_file,
                    exc,
                )
                continue
            for name, entry in partial.datasets.items():
                if name in merged:
                    logger.info(
                        "Overlay dataset '%s' (from %s) replaces existing entry",
                        name,
                        yaml_file.name,
                    )
                merged[name] = entry
    return DatasetRegistry(datasets=merged)
