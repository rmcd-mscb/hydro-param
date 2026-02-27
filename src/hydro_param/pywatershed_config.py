"""Define the pywatershed run configuration schema and YAML loader.

Provide Pydantic models that validate the YAML configuration for the
``hydro-param pywatershed run`` command.  This schema is separate from
``PipelineConfig`` because it represents a model-specific workflow
(complete pywatershed model setup) rather than generic zonal statistics.

The configuration covers seven sections: domain geometry, simulation
time period, climate forcing source, source dataset selections,
processing options, manual parameter overrides, calibration seed
generation, and output file layout.

Notes
-----
This config is translated into a generic ``PipelineConfig`` by
``cli._translate_pws_to_pipeline()`` for pipeline phases 1--5.
Model-specific post-processing (derivation, unit conversion, output
formatting) is driven directly from this config in phase 2.

See Also
--------
docs/reference/pywatershed_parameterization_guide.md : Section 2B describes
    the proposed config structure.
hydro_param.config.PipelineConfig : Generic pipeline configuration.
hydro_param.cli._translate_pws_to_pipeline : Config translation logic.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PwsDomainConfig(BaseModel):
    """Define the spatial domain for pywatershed model setup.

    Specify how the model domain is extracted (bounding box, HUC,
    pour point, or BANDIT) and where the pre-existing fabric and
    segment GeoPackage/GeoParquet files are located.

    Attributes
    ----------
    source : {"geospatial_fabric", "custom"}
        Whether to use the USGS Geospatial Fabric or a custom fabric.
    gf_version : str
        Geospatial Fabric version (default ``"1.1"``).
    extraction_method : {"bbox", "huc", "pour_point", "bandit"}
        Method for subsetting the domain.
    bbox : list[float] or None
        Bounding box as ``[minx, miny, maxx, maxy]`` in EPSG:4326.
    huc_id : str or None
        HUC identifier for HUC-based extraction.
    pour_point : list[float] or None
        Pour point as ``[x, y]`` in EPSG:4326.
    fabric_path : Path or None
        Path to the pre-existing HRU fabric file (GeoPackage or
        GeoParquet).  Required when ``source="custom"``.
    segment_path : Path or None
        Path to the segment/flowline fabric file for routing topology.
    id_field : str
        Feature ID column name in the fabric (default ``"nhm_id"``).
    segment_id_field : str
        Segment ID column name in the segment fabric (default
        ``"nhm_seg"``).

    Notes
    -----
    hydro-param does NOT fetch or subset fabrics.  The ``fabric_path``
    must point to a pre-existing file produced by pynhd, pygeohydro,
    or similar upstream tools.
    """

    source: Literal["geospatial_fabric", "custom"] = "geospatial_fabric"
    gf_version: str = "1.1"
    extraction_method: Literal["bbox", "huc", "pour_point", "bandit"] = "bbox"
    bbox: list[float] | None = None
    huc_id: str | None = None
    pour_point: list[float] | None = None
    fabric_path: Path | None = None
    segment_path: Path | None = None
    id_field: str = "nhm_id"
    segment_id_field: str = "nhm_seg"

    @model_validator(mode="after")
    def _validate_extraction(self) -> PwsDomainConfig:
        """Validate that required fields are present for the chosen extraction method."""
        if self.extraction_method == "bbox":
            if self.bbox is None:
                raise ValueError("bbox extraction requires 'bbox' field")
            if len(self.bbox) != 4:
                raise ValueError("bbox must contain 4 coordinates [minx, miny, maxx, maxy]")
        if self.extraction_method == "huc" and self.huc_id is None:
            raise ValueError("huc extraction requires 'huc_id' field")
        if self.extraction_method == "pour_point":
            if self.pour_point is None:
                raise ValueError("pour_point extraction requires 'pour_point' field")
            if len(self.pour_point) != 2:
                raise ValueError("pour_point must contain 2 coordinates [x, y]")
        if self.source == "custom" and self.fabric_path is None:
            raise ValueError("custom domain source requires 'fabric_path'")
        return self


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


class PwsClimateConfig(BaseModel):
    """Specify the climate forcing source and variables for PRMS.

    PRMS requires three Climate-by-HRU (CBH) time series: precipitation,
    maximum temperature, and minimum temperature.  These are area-weighted
    means of gridded climate data over each HRU polygon.

    Attributes
    ----------
    source : str
        Climate dataset name.  Accepts any string but only ``"gridmet"``
        is currently supported (accessed via OPeNDAP / ClimRCatData).
        Unsupported sources raise ``ValueError`` during pipeline
        translation.
    method : {"area_weighted_mean"}
        Aggregation method for zonal statistics.
    variables : list[str]
        PRMS-facing variable names: ``["prcp", "tmax", "tmin"]``.
        These are mapped to source-specific names (e.g., ``pr``,
        ``tmmx``, ``tmmn`` for gridMET) during pipeline translation.

    Notes
    -----
    The gridMET copy on the USGS GDP STAC is not kept up to date.
    Use the ``climr_cat`` strategy (OPeNDAP) for gridMET access.
    """

    source: str = "gridmet"
    method: Literal["area_weighted_mean"] = "area_weighted_mean"
    variables: list[str] = Field(default_factory=lambda: ["prcp", "tmax", "tmin"])


class PwsDatasetSources(BaseModel):
    """Select source datasets by category for pywatershed parameterization.

    Each attribute names a dataset in the hydro-param registry.  The
    defaults match the standard NHM-PRMS configuration.

    Attributes
    ----------
    topography : str
        DEM dataset name for elevation, slope, and aspect zonal stats.
        Default ``"dem_3dep_10m"`` (USGS 3DEP 1/3 arc-second).
    landcover : str
        Land cover dataset name for categorical fractions.  Default
        ``"nlcd_legacy"`` (local GeoTIFF).  Use ``"nlcd_osn_lndcov"``
        for NHGF STAC access.
    soils : str
        Soils dataset name.  Default ``"polaris_30m"`` (POLARIS 30m).
    hydrography : str or None
        Hydrography dataset name (e.g., NHDPlus).  Not yet used in
        the automated pipeline.
    """

    topography: str = "dem_3dep_10m"
    landcover: str = "nlcd_legacy"
    soils: str = "polaris_30m"
    hydrography: str | None = None


class PwsProcessingConfig(BaseModel):
    """Configure processing options for the zonal statistics engine.

    Attributes
    ----------
    zonal_method : {"exactextract", "serial"}
        Zonal statistics engine.  ``"exactextract"`` uses the
        exactextract C++ library via gdptools for fast, exact
        coverage-weighted statistics.  ``"serial"`` is a slower
        pure-Python fallback.
    batch_size : int
        Number of HRU features per spatial batch.  Must be > 0.
        Larger batches reduce STAC query overhead but increase
        per-batch memory usage.
    n_workers : int
        Number of parallel workers.  Must be >= 1.  Currently only
        ``1`` is supported (serial processing).
    """

    zonal_method: Literal["exactextract", "serial"] = "exactextract"
    batch_size: int = Field(default=500, gt=0)
    n_workers: int = Field(default=1, ge=1)


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
                # Both specified — drop legacy key silently
                values.pop("cbh_dir")
        return values


class PywatershedRunConfig(BaseModel):
    """Define the top-level configuration for pywatershed model setup.

    Specify everything needed for hydro-param to generate a complete
    pywatershed (NHM-PRMS) model: spatial domain, simulation time
    period, climate forcing source, source dataset selections,
    processing engine options, manual parameter overrides, calibration
    seed generation, and output file layout.

    This config is loaded from a YAML file by ``load_pywatershed_config()``
    and consumed by ``cli.pws_run_cmd()`` to drive the two-phase workflow.

    Attributes
    ----------
    target_model : {"pywatershed"}
        Target model identifier (fixed to ``"pywatershed"``).
    version : str
        Config schema version (default ``"2.0"``).
    domain : PwsDomainConfig
        Spatial domain specification.
    time : PwsTimeConfig
        Simulation time period.
    climate : PwsClimateConfig
        Climate forcing configuration.
    datasets : PwsDatasetSources
        Source dataset selections by category.
    processing : PwsProcessingConfig
        Zonal statistics engine and batching options.
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

    target_model: Literal["pywatershed"] = "pywatershed"
    version: str = "2.0"
    domain: PwsDomainConfig
    time: PwsTimeConfig
    climate: PwsClimateConfig = PwsClimateConfig()
    datasets: PwsDatasetSources = PwsDatasetSources()
    processing: PwsProcessingConfig = PwsProcessingConfig()
    parameter_overrides: PwsParameterOverrides = PwsParameterOverrides()
    calibration: PwsCalibrationConfig = PwsCalibrationConfig()
    output: PwsOutputConfig = PwsOutputConfig()


def load_pywatershed_config(path: str | Path) -> PywatershedRunConfig:
    """Load and validate a pywatershed run configuration from YAML.

    Parse the YAML file and construct a fully validated
    ``PywatershedRunConfig`` with Pydantic's strict type coercion
    and cross-field validation (e.g., bbox requires 4 coordinates,
    custom domain requires fabric_path).

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
        fields, type mismatches, cross-field constraint violations).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PywatershedRunConfig(**raw)
