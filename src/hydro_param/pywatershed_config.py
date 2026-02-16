"""pywatershed run configuration: Pydantic models + YAML loader.

Dedicated config schema for generating a complete pywatershed model
setup. Separate from ``PipelineConfig`` because it represents a
different workflow (model setup vs. generic zonal statistics).

See docs/reference/pywatershed_parameterization_guide.md §2B for the
proposed structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class PwsDomainConfig(BaseModel):
    """Domain specification for pywatershed setup."""

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
        if self.extraction_method == "bbox" and self.bbox is None:
            raise ValueError("bbox extraction requires 'bbox' field")
        if self.extraction_method == "huc" and self.huc_id is None:
            raise ValueError("huc extraction requires 'huc_id' field")
        if self.extraction_method == "pour_point" and self.pour_point is None:
            raise ValueError("pour_point extraction requires 'pour_point' field")
        if self.source == "custom" and self.fabric_path is None:
            raise ValueError("custom domain source requires 'fabric_path'")
        return self


class PwsTimeConfig(BaseModel):
    """Simulation time period."""

    start: str  # ISO date, e.g. "1980-10-01"
    end: str
    timestep: Literal["daily"] = "daily"


class PwsClimateConfig(BaseModel):
    """Climate forcing specification."""

    source: Literal["daymet_v4", "gridmet", "conus404_ba"] = "daymet_v4"
    method: Literal["area_weighted_mean"] = "area_weighted_mean"
    variables: list[str] = Field(default_factory=lambda: ["prcp", "tmax", "tmin"])


class PwsDatasetSources(BaseModel):
    """Dataset source selections by category."""

    topography: str = "dem_3dep_10m"
    landcover: str = "nlcd_legacy"
    soils: str = "polaris_30m"
    hydrography: str | None = None


class PwsProcessingConfig(BaseModel):
    """Processing options."""

    zonal_method: Literal["exactextract", "serial"] = "exactextract"
    batch_size: int = Field(default=500, gt=0)
    n_workers: int = Field(default=1, ge=1)


class PwsParameterOverrides(BaseModel):
    """Manual parameter value overrides."""

    values: dict[str, float | list[float]] = Field(default_factory=dict)
    from_file: Path | None = None


class PwsCalibrationConfig(BaseModel):
    """Calibration seed generation options."""

    generate_seeds: bool = True
    seed_method: Literal["physically_based", "all_defaults"] = "physically_based"
    preserve_from_existing: list[str] = Field(default_factory=list)


class PwsOutputConfig(BaseModel):
    """Output specification for pywatershed files."""

    path: Path = Path("./output")
    format: Literal["netcdf", "prms_text"] = "netcdf"
    parameter_file: str = "parameters.nc"
    cbh_dir: str = "cbh"
    control_file: str = "control.yml"
    soltab_file: str = "soltab.nc"


class PywatershedRunConfig(BaseModel):
    """Top-level configuration for pywatershed model setup.

    Specifies everything needed for hydro-param to generate a complete
    pywatershed model: domain, time period, climate forcing, datasets,
    processing, overrides, calibration, and output.
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
    """Load and validate a pywatershed run config from YAML.

    Parameters
    ----------
    path
        Path to the YAML config file.

    Returns
    -------
    PywatershedRunConfig
        Validated configuration.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PywatershedRunConfig(**raw)
