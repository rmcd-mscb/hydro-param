"""Define the pywatershed run configuration schema and YAML loader.

Provide Pydantic models that validate the YAML configuration for the
``hydro-param pywatershed run`` command.  This is a Phase 2 (model-specific)
config that consumes pre-existing SIR output from the generic Phase 1
pipeline.  It does NOT configure the Phase 1 pipeline itself.

The configuration covers nine sections: domain file paths, simulation
time period, SIR output location, static dataset declarations, forcing
time series, climate normals, manual parameter overrides, calibration
seed generation, and output file layout.

Notes
-----
Version 4.0 adds three data sections (``static_datasets``, ``forcing``,
``climate_normals``) that declare which pipeline datasets provide each
pywatershed parameter.  This creates a consumer-oriented, self-documenting
contract between the Phase 1 pipeline and the Phase 2 derivation plugin.

See Also
--------
hydro_param.sir_accessor.SIRAccessor : Lazy SIR variable loader.
hydro_param.plugins.DerivationContext : Derivation step context.
hydro_param.cli.pws_run_cmd : Two-phase workflow consumer.
"""

from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ParameterEntry(BaseModel):
    """Declare the SIR data source for a single pywatershed parameter.

    Each entry maps a pywatershed parameter to the pipeline dataset, source
    variable(s), and zonal statistic that produced the SIR data.

    Exactly one of ``variable`` (single) or ``variables`` (list) must be
    provided for entries backed by SIR data.  Both may be ``None`` only
    for entries whose ``source`` is not a pipeline dataset (e.g.,
    waterbody parameters derived from fabric overlay).

    Parameters
    ----------
    source : str
        Pipeline dataset registry name (e.g., ``"dem_3dep_10m"``),
        or a reference like ``"domain.waterbody_path"`` for non-SIR
        entries.
    variable : str or None
        Source variable name when a single variable is used.
    variables : list[str] or None
        Source variable names when multiple variables contribute
        (e.g., ``["sand", "silt", "clay"]`` for soil_type).
    statistic : str or None
        Zonal statistic applied (``"mean"``, ``"categorical"``).
    year : int or list[int] or None
        NLCD year(s) for multi-epoch land cover.
    time_period : list[str] or None
        Temporal range ``[start, end]`` in ISO format for temporal datasets.
    description : str
        Human-readable description of what this parameter represents.

    Raises
    ------
    ValueError
        If both ``variable`` and ``variables`` are set simultaneously.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    variable: str | None = None
    variables: list[str] | None = None
    statistic: str | None = None
    year: int | list[int] | None = None
    time_period: list[str] | None = None
    description: str

    @model_validator(mode="after")
    def _check_variable_fields(self) -> Self:
        """Validate that variable and variables are not both set."""
        if self.variable is not None and self.variables is not None:
            raise ValueError("Provide 'variable' (single) or 'variables' (list), not both.")
        return self


class TopographyDatasets(BaseModel):
    """Topography parameters derived from DEM zonal statistics.

    Parameters
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    hru_elev : ParameterEntry or None
        Mean HRU elevation.
    hru_slope : ParameterEntry or None
        Mean HRU land surface slope.
    hru_aspect : ParameterEntry or None
        Mean HRU aspect.
    """

    model_config = ConfigDict(extra="forbid")

    available: list[str] = Field(default_factory=list)
    hru_elev: ParameterEntry | None = None
    hru_slope: ParameterEntry | None = None
    hru_aspect: ParameterEntry | None = None


class SoilsDatasets(BaseModel):
    """Soil parameters derived from soil property datasets.

    Parameters
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    soil_type : ParameterEntry or None
        Soil type classification (1=sand, 2=loam, 3=clay).
    sat_threshold : ParameterEntry or None
        Gravity reservoir storage capacity (from porosity).
    soil_moist_max : ParameterEntry or None
        Maximum available water-holding capacity.
    soil_rechr_max_frac : ParameterEntry or None
        Recharge zone storage as fraction of soil_moist_max.
    """

    model_config = ConfigDict(extra="forbid")

    available: list[str] = Field(default_factory=list)
    soil_type: ParameterEntry | None = None
    sat_threshold: ParameterEntry | None = None
    soil_moist_max: ParameterEntry | None = None
    soil_rechr_max_frac: ParameterEntry | None = None


class LandcoverDatasets(BaseModel):
    """Land cover parameters for vegetation type, density, and interception.

    Parameters
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    cov_type : ParameterEntry or None
        Vegetation cover type.
    hru_percent_imperv : ParameterEntry or None
        Impervious surface fraction.
    covden_sum : ParameterEntry or None
        Summer vegetation cover density (0--1 fraction).
    covden_win : ParameterEntry or None
        Winter vegetation cover density (0--1 fraction).
    srain_intcp : ParameterEntry or None
        Summer rain interception storage capacity (inches).
    wrain_intcp : ParameterEntry or None
        Winter rain interception storage capacity (inches).
    snow_intcp : ParameterEntry or None
        Snow interception storage capacity (inches).
    """

    model_config = ConfigDict(extra="forbid")

    available: list[str] = Field(default_factory=list)
    cov_type: ParameterEntry | None = None
    hru_percent_imperv: ParameterEntry | None = None
    covden_sum: ParameterEntry | None = None
    covden_win: ParameterEntry | None = None
    srain_intcp: ParameterEntry | None = None
    wrain_intcp: ParameterEntry | None = None
    snow_intcp: ParameterEntry | None = None


class SnowDatasets(BaseModel):
    """Snow parameters from depletion curve classification and historical SWE data.

    Parameters
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    hru_deplcrv : ParameterEntry or None
        Snow depletion curve class per HRU.  Source is typically the GFv1.1
        CV_INT raster (categorical majority).  Indexes into the SDC table
        to populate ``snarea_curve``.
    snarea_thresh : ParameterEntry or None
        Snow depletion threshold (calibration seed from historical max SWE).
    """

    model_config = ConfigDict(extra="forbid")

    available: list[str] = Field(default_factory=list)
    hru_deplcrv: ParameterEntry | None = None
    snarea_thresh: ParameterEntry | None = None


class WaterbodyDatasets(BaseModel):
    """Depression storage and HRU type from waterbody overlay.

    Parameters
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    hru_type : ParameterEntry or None
        HRU type (0=inactive, 1=land, 2=lake, 3=swale).
    dprst_frac : ParameterEntry or None
        Fraction of HRU with surface depressions.
    """

    model_config = ConfigDict(extra="forbid")

    available: list[str] = Field(default_factory=list)
    hru_type: ParameterEntry | None = None
    dprst_frac: ParameterEntry | None = None


class StaticDatasetsConfig(BaseModel):
    """Static dataset declarations grouped by domain category.

    Each category contains explicit parameter fields that map to SIR data
    produced by the Phase 1 pipeline.

    Parameters
    ----------
    topography : TopographyDatasets
        DEM-derived parameters (elevation, slope, aspect).
    soils : SoilsDatasets
        Soil property parameters.
    landcover : LandcoverDatasets
        Land cover and impervious surface parameters.
    snow : SnowDatasets
        Historical snow parameters.
    waterbodies : WaterbodyDatasets
        Depression storage and HRU type.
    """

    model_config = ConfigDict(extra="forbid")

    topography: TopographyDatasets = Field(default_factory=TopographyDatasets)
    soils: SoilsDatasets = Field(default_factory=SoilsDatasets)
    landcover: LandcoverDatasets = Field(default_factory=LandcoverDatasets)
    snow: SnowDatasets = Field(default_factory=SnowDatasets)
    waterbodies: WaterbodyDatasets = Field(default_factory=WaterbodyDatasets)


class ForcingConfig(BaseModel):
    """Temporal forcing time series declarations.

    The Phase 2 derivation plugin converts forcing data from SIR units
    (metric: mm, degC) to PRMS units (inches, degF) during output
    formatting.  pywatershed expects one-variable-per-NetCDF.

    Parameters
    ----------
    available : list[str]
        Temporal-capable datasets available in the registry.
    prcp : ParameterEntry or None
        Daily precipitation.
    tmax : ParameterEntry or None
        Daily maximum temperature.
    tmin : ParameterEntry or None
        Daily minimum temperature.
    """

    model_config = ConfigDict(extra="forbid")

    available: list[str] = Field(default_factory=list)
    prcp: ParameterEntry | None = None
    tmax: ParameterEntry | None = None
    tmin: ParameterEntry | None = None


class ClimateNormalsConfig(BaseModel):
    """Long-term climate statistics for derived parameters.

    Can use the same source as forcing, or a different one (e.g.,
    forcing from CONUS404-BA but normals from gridMET).

    Parameters
    ----------
    available : list[str]
        Temporal-capable datasets available in the registry.
    jh_coef : ParameterEntry or None
        Jensen-Haise PET coefficient (monthly, from tmax/tmin normals).
    transp_beg : ParameterEntry or None
        Month transpiration begins (from monthly mean tmin threshold).
    transp_end : ParameterEntry or None
        Month transpiration ends (from monthly mean tmin threshold).
    """

    model_config = ConfigDict(extra="forbid")

    available: list[str] = Field(default_factory=list)
    jh_coef: ParameterEntry | None = None
    transp_beg: ParameterEntry | None = None
    transp_end: ParameterEntry | None = None


class PwsDomainConfig(BaseModel):
    """Define the spatial domain for pywatershed model setup.

    Point to pre-existing fabric and segment files on disk.  hydro-param
    does NOT fetch or subset fabrics — use pynhd or pygeohydro upstream.

    Attributes
    ----------
    fabric_path : Path
        Path to the HRU fabric file (GeoPackage or GeoParquet).
    segment_path : Path or None
        Path to the segment/flowline file for routing topology.
    waterbody_path : Path or None
        Path to NHDPlus waterbody polygon file (GeoPackage or GeoParquet)
        for depression storage overlay (step 6).  Must contain an ``ftype``
        column with values like ``"LakePond"`` and ``"Reservoir"``.
        When ``None``, step 6 uses zero defaults.
    id_field : str
        Feature ID column name in the fabric (default ``"nhm_id"``).
    segment_id_field : str
        Segment ID column name in the segment fabric (default
        ``"nhm_seg"``).

    Notes
    -----
    The ``fabric_path`` must point to a pre-existing file produced by
    pynhd, pygeohydro, or similar upstream tools.
    """

    fabric_path: Path
    segment_path: Path | None = None
    waterbody_path: Path | None = None
    id_field: str = "nhm_id"
    segment_id_field: str = "nhm_seg"


class PwsTimeConfig(BaseModel):
    """Define the simulation time period for pywatershed.

    Attributes
    ----------
    start : str
        Simulation start date in ISO format (e.g., ``"1980-10-01"``).
        Typically a water-year boundary for PRMS.
    end : str
        Simulation end date in ISO format (e.g., ``"1982-09-30"``).
    timestep : {"daily"}
        Temporal resolution.  Only daily is currently supported, which
        matches PRMS's native timestep.
    """

    start: str  # ISO date, e.g. "1980-10-01"
    end: str
    timestep: Literal["daily"] = "daily"

    @field_validator("start", "end")
    @classmethod
    def _validate_iso_date(cls, v: str) -> str:
        """Validate that start/end are valid ISO date strings."""
        try:
            _dt.date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date '{v}'. Expected ISO format (YYYY-MM-DD).") from None
        return v

    @model_validator(mode="after")
    def _check_date_order(self) -> Self:
        """Validate that start date is on or before end date."""
        start = _dt.date.fromisoformat(self.start)
        end = _dt.date.fromisoformat(self.end)
        if start > end:
            raise ValueError(f"start ({self.start}) must be on or before end ({self.end}).")
        return self


class PwsParameterOverrides(BaseModel):
    """Specify manual overrides for derived parameter values.

    Allow users to inject known-good values (e.g., from calibration)
    that bypass the standard derivation pipeline.  Overrides are
    applied after all other derivation steps complete.

    Attributes
    ----------
    values : dict[str, float | list[float]]
        Parameter name to scalar or per-HRU value mapping.  Scalars
        are broadcast to all HRUs.  List values must match the number
        of HRUs in the fabric.
    from_file : Path or None
        Path to a NetCDF or CSV file containing override values.
        Not yet implemented.
    """

    values: dict[str, float | list[float]] = Field(default_factory=dict)
    from_file: Path | None = None


class PwsCalibrationConfig(BaseModel):
    """Configure calibration seed generation for PRMS parameters.

    PRMS calibration parameters (e.g., ``carea_max``, ``soil_moist_max``,
    ``K_coef``) need physically plausible initial values.  This config
    controls whether and how those seeds are generated.

    Attributes
    ----------
    generate_seeds : bool
        Whether to generate calibration seed values.  Default ``True``.
    seed_method : {"physically_based", "all_defaults"}
        ``"physically_based"`` derives seeds from GIS data (e.g.,
        ``carea_max`` from impervious fraction).  ``"all_defaults"``
        uses PRMS default values for all calibration parameters.
    preserve_from_existing : list[str]
        Parameter names to preserve from an existing parameter file
        rather than re-deriving.  Useful for retaining calibrated
        values during fabric updates.
    """

    generate_seeds: bool = True
    seed_method: Literal["physically_based", "all_defaults"] = "physically_based"
    preserve_from_existing: list[str] = Field(default_factory=list)


class PwsOutputConfig(BaseModel):
    """Specify output file layout for pywatershed model setup.

    Control the directory structure and filenames for the four output
    components: static parameters, climate forcing, solar tables, and
    simulation control.

    Attributes
    ----------
    path : Path
        Root output directory.  Created if it does not exist.
        Default ``"./output"``.
    format : {"netcdf", "prms_text"}
        Output format.  ``"netcdf"`` produces CF-1.8 compliant files
        loadable by pywatershed.  ``"prms_text"`` is not yet
        implemented.
    parameter_file : str
        Filename for static parameters (default ``"parameters.nc"``).
    forcing_dir : str
        Subdirectory for climate forcing files (default ``"forcing"``).
    control_file : str
        Filename for simulation control (default ``"control.yml"``).
    soltab_file : str
        Filename for solar radiation tables (default ``"soltab.nc"``).
    """

    model_config = ConfigDict(extra="forbid")

    path: Path = Path("./output")
    format: Literal["netcdf", "prms_text"] = "netcdf"
    parameter_file: str = "parameters.nc"
    forcing_dir: str = "forcing"
    control_file: str = "control.yml"
    soltab_file: str = "soltab.nc"


class PywatershedRunConfig(BaseModel):
    """Define the top-level configuration for pywatershed model setup.

    A consumer-oriented, self-documenting contract between the Phase 1
    pipeline and the Phase 2 pywatershed derivation plugin.  Three data
    sections (``static_datasets``, ``forcing``, ``climate_normals``)
    declare which pipeline datasets provide each pywatershed parameter.

    Parameters
    ----------
    target_model : {"pywatershed"}
        Target model identifier (fixed to ``"pywatershed"``).
    version : str
        Config schema version (``"4.0"``).
    domain : PwsDomainConfig
        Domain fabric file paths and ID field names.
    time : PwsTimeConfig
        Simulation time period.
    sir_path : Path
        Path to the Phase 1 pipeline output directory containing
        ``.manifest.yml`` and ``sir/`` subdirectory.  Relative paths
        are resolved against the config file's parent directory.
    static_datasets : StaticDatasetsConfig
        Static dataset declarations grouped by domain category.
    forcing : ForcingConfig
        Temporal forcing time series declarations.
    climate_normals : ClimateNormalsConfig
        Long-term climate statistics for derived parameters.
    parameter_overrides : PwsParameterOverrides
        Manual parameter value overrides.
    calibration : PwsCalibrationConfig
        Calibration seed generation options.
    output : PwsOutputConfig
        Output directory structure and filenames.

    See Also
    --------
    load_pywatershed_config : YAML loader for this schema.
    hydro_param.cli.pws_run_cmd : Two-phase workflow consumer.
    """

    model_config = ConfigDict(extra="forbid")

    target_model: Literal["pywatershed"] = "pywatershed"
    version: Literal["4.0"] = "4.0"
    domain: PwsDomainConfig
    time: PwsTimeConfig
    sir_path: Path = Path("output")
    static_datasets: StaticDatasetsConfig = Field(default_factory=StaticDatasetsConfig)
    forcing: ForcingConfig = Field(default_factory=ForcingConfig)
    climate_normals: ClimateNormalsConfig = Field(default_factory=ClimateNormalsConfig)
    parameter_overrides: PwsParameterOverrides = PwsParameterOverrides()
    calibration: PwsCalibrationConfig = PwsCalibrationConfig()
    output: PwsOutputConfig = PwsOutputConfig()

    def declared_entries(self) -> dict[str, ParameterEntry]:
        """Collect all declared ParameterEntry objects from the config.

        Walk ``static_datasets``, ``forcing``, and ``climate_normals``
        sections and return a flat dictionary keyed by parameter name.

        Returns
        -------
        dict[str, ParameterEntry]
            Parameter name to entry mapping for all non-None entries.
        """
        entries: dict[str, ParameterEntry] = {}

        # Static datasets: walk each category
        for category in (
            self.static_datasets.topography,
            self.static_datasets.soils,
            self.static_datasets.landcover,
            self.static_datasets.snow,
            self.static_datasets.waterbodies,
        ):
            for field_name in type(category).model_fields:
                if field_name == "available":
                    continue
                value = getattr(category, field_name)
                if value is not None:
                    entries[field_name] = value

        # Forcing
        for field_name in ("prcp", "tmax", "tmin"):
            value = getattr(self.forcing, field_name)
            if value is not None:
                entries[field_name] = value

        # Climate normals
        for field_name in ("jh_coef", "transp_beg", "transp_end"):
            value = getattr(self.climate_normals, field_name)
            if value is not None:
                entries[field_name] = value

        return entries

    def validate_available_fields(self) -> None:
        """Check that ``available`` dataset names exist in the registry.

        Load the bundled dataset registry and verify that every name in
        each category's ``available`` list is a known dataset.  Unknown
        entries emit a ``UserWarning`` rather than raising, because they
        may refer to user-provided local datasets not in the bundled
        registry.

        Warnings
        --------
        UserWarning
            For each dataset name in an ``available`` list that is not
            found in the current registry.
        """
        from hydro_param.dataset_registry import get_all_dataset_names, load_registry
        from hydro_param.pipeline import DEFAULT_REGISTRY

        registry = load_registry(DEFAULT_REGISTRY)
        known = get_all_dataset_names(registry)

        categories: list[tuple[str, BaseModel]] = [
            ("topography", self.static_datasets.topography),
            ("soils", self.static_datasets.soils),
            ("landcover", self.static_datasets.landcover),
            ("snow", self.static_datasets.snow),
            ("waterbodies", self.static_datasets.waterbodies),
            ("forcing", self.forcing),
            ("climate_normals", self.climate_normals),
        ]
        for cat_name, category in categories:
            available: list[str] = getattr(category, "available", [])
            for ds_name in available:
                if ds_name not in known:
                    warnings.warn(
                        f"Dataset '{ds_name}' in {cat_name}.available "
                        f"is not in the registry. Known: {sorted(known)}",
                        UserWarning,
                        stacklevel=2,
                    )


def load_pywatershed_config(path: str | Path) -> PywatershedRunConfig:
    """Load and validate a pywatershed run configuration from YAML.

    Parse the YAML file and construct a fully validated
    ``PywatershedRunConfig`` with Pydantic's strict type coercion.

    Parameters
    ----------
    path
        Path to the YAML config file.

    Returns
    -------
    PywatershedRunConfig
        Validated configuration ready for ``pws_run_cmd()``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    yaml.YAMLError
        If the file contains invalid YAML.
    pydantic.ValidationError
        If the config fails schema validation (missing required
        fields, type mismatches, extra fields).

    Notes
    -----
    Path fields (``sir_path``, ``domain.fabric_path``, etc.) are
    returned as-is from the YAML.  Relative paths are resolved against
    the config file's parent directory by the CLI consumer
    (``pws_run_cmd``), not by this loader.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected YAML mapping in {path}, got {type(raw).__name__}. "
            f"Check that the file is non-empty and contains valid config."
        )
    return PywatershedRunConfig(**raw)
