"""Pipeline configuration: Pydantic models + YAML loader.

Declarative config schema matching design.md section 11.6.
Configs say *what* to compute, not *how*.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class TargetFabricConfig(BaseModel):
    """Target fabric (polygon mesh) specification."""

    path: Path
    id_field: str
    crs: str = "EPSG:4326"


class DomainConfig(BaseModel):
    """Spatial domain specification."""

    type: Literal["bbox", "huc2", "huc4", "gage"]
    bbox: list[float] | None = None
    id: str | None = None

    @model_validator(mode="after")
    def _validate_domain(self) -> DomainConfig:
        if self.type == "bbox" and self.bbox is None:
            raise ValueError("bbox domain requires 'bbox' field")
        if self.type in ("huc2", "huc4", "gage") and self.id is None:
            raise ValueError(f"{self.type} domain requires 'id' field")
        return self


class DatasetRequest(BaseModel):
    """A dataset + variable selection from the pipeline config."""

    name: str
    variables: list[str] = []
    statistics: list[str] = Field(default_factory=lambda: ["mean"])


class OutputConfig(BaseModel):
    """Output specification."""

    path: Path = Path("./output")
    format: Literal["netcdf", "parquet"] = "netcdf"
    sir_name: str = "result"


class ProcessingConfig(BaseModel):
    """Processing options."""

    engine: Literal["exactextract", "serial"] = "exactextract"
    # TODO: Wire failure_mode into stage4 error handling (continue-on-failure with logging)
    failure_mode: Literal["strict", "tolerant"] = "strict"
    batch_size: int = Field(default=500, gt=0)


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration."""

    target_fabric: TargetFabricConfig
    domain: DomainConfig
    datasets: list[DatasetRequest]
    output: OutputConfig = OutputConfig()
    processing: ProcessingConfig = ProcessingConfig()


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline YAML config file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
