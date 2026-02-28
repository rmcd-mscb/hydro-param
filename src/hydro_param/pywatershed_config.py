"""Define the pywatershed run configuration schema and YAML loader.

Provide Pydantic models that validate the YAML configuration for the
``hydro-param pywatershed run`` command.  This is a Phase 2 (model-specific)
config that consumes pre-existing SIR output from the generic Phase 1
pipeline.  It does NOT configure the Phase 1 pipeline itself.

The configuration covers six sections: domain file paths, simulation
time period, SIR output location, manual parameter overrides, calibration
seed generation, and output file layout.

Notes
-----
Version 3.0 of this schema eliminates Phase 1 fields (datasets, climate,
processing) that were previously translated into a ``PipelineConfig``.
Phase 2 now reads SIR output directly via ``SIRAccessor``.

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

    Notes
    -----
    The legacy config key ``cbh_dir`` is accepted with a deprecation
    warning and mapped to ``forcing_dir``.
    """

    path: Path = Path("./output")
    format: Literal["netcdf", "prms_text"] = "netcdf"
    parameter_file: str = "parameters.nc"
    forcing_dir: str = "forcing"
    control_file: str = "control.yml"
    soltab_file: str = "soltab.nc"

    @model_validator(mode="before")
    @classmethod
    def _migrate_cbh_dir(cls, values: dict) -> dict:
        """Map legacy ``cbh_dir`` to ``forcing_dir`` with a deprecation warning."""
        if isinstance(values, dict) and "cbh_dir" in values:
            if "forcing_dir" not in values:
                warnings.warn(
                    "Config key 'cbh_dir' is deprecated, use 'forcing_dir' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                values["forcing_dir"] = values.pop("cbh_dir")
            else:
                warnings.warn(
                    f"Both 'cbh_dir' and 'forcing_dir' specified; "
                    f"using 'forcing_dir: {values['forcing_dir']}' and "
                    f"ignoring 'cbh_dir: {values['cbh_dir']}'.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                values.pop("cbh_dir")
        return values


class PywatershedRunConfig(BaseModel):
    """Define the top-level configuration for pywatershed model setup.

    Phase 2 config that consumes pre-existing SIR output from the
    generic Phase 1 pipeline.  Specify the domain fabric files,
    simulation time period, SIR output location, manual parameter
    overrides, calibration seed generation, and output file layout.

    Attributes
    ----------
    target_model : {"pywatershed"}
        Target model identifier (fixed to ``"pywatershed"``).
    version : str
        Config schema version (default ``"3.0"``).
    domain : PwsDomainConfig
        Domain fabric file paths and ID field names.
    time : PwsTimeConfig
        Simulation time period.
    sir_path : Path
        Path to the Phase 1 pipeline output directory containing
        ``.manifest.yml`` and ``sir/`` subdirectory.  Relative paths
        are resolved against the config file's parent directory.
        Default ``"output"``.
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
    version: Literal["3.0"] = "3.0"
    domain: PwsDomainConfig
    time: PwsTimeConfig
    sir_path: Path = Path("output")
    parameter_overrides: PwsParameterOverrides = PwsParameterOverrides()
    calibration: PwsCalibrationConfig = PwsCalibrationConfig()
    output: PwsOutputConfig = PwsOutputConfig()


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
    return PywatershedRunConfig(**raw)
