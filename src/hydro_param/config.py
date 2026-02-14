"""Pipeline configuration: Pydantic models + YAML loader."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    variables: list[str] = []


class PipelineConfig(BaseModel):
    target_fabric: str
    domain: dict[str, Any]
    datasets: list[DatasetConfig]
    output_path: str = "./output"


def load_config(path: str | Path) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw)
