"""Tests for config loading and validation."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hydro_param.config import (
    DatasetRequest,
    DomainConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
    TargetFabricConfig,
    load_config,
)


def test_load_config_from_yaml(tmp_path: Path):
    raw = {
        "target_fabric": {
            "path": "data/catchments.gpkg",
            "id_field": "featureid",
        },
        "domain": {"type": "bbox", "bbox": [0, 0, 2, 2]},
        "datasets": [
            {"name": "dem_3dep_10m", "variables": ["elevation", "slope"]},
        ],
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert isinstance(config, PipelineConfig)
    assert config.target_fabric.id_field == "featureid"
    assert len(config.datasets) == 1
    assert config.datasets[0].name == "dem_3dep_10m"
    assert config.datasets[0].variables == ["elevation", "slope"]


def test_config_full_yaml(tmp_path: Path):
    raw = {
        "target_fabric": {
            "path": "data/catchments.gpkg",
            "id_field": "hru_id",
            "crs": "EPSG:5070",
        },
        "domain": {"type": "bbox", "bbox": [-76.5, 38.5, -74.0, 42.6]},
        "datasets": [
            {"name": "dem", "variables": ["elevation"], "statistics": ["mean", "std"]},
        ],
        "output": {
            "path": "./results",
            "format": "parquet",
            "sir_name": "delaware_test",
        },
        "processing": {
            "engine": "serial",
            "failure_mode": "tolerant",
            "batch_size": 200,
        },
    }
    path = tmp_path / "full.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert config.target_fabric.crs == "EPSG:5070"
    assert config.output.format == "parquet"
    assert config.output.sir_name == "delaware_test"
    assert config.processing.engine == "serial"
    assert config.processing.failure_mode == "tolerant"
    assert config.processing.batch_size == 200


def test_config_defaults():
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )
    assert config.output.path == Path("./output")
    assert config.output.format == "netcdf"
    assert config.output.sir_name == "result"
    assert config.processing.engine == "exactextract"
    assert config.processing.failure_mode == "strict"
    assert config.processing.batch_size == 500


def test_dataset_request_defaults():
    ds = DatasetRequest(name="dem")
    assert ds.variables == []
    assert ds.statistics == ["mean"]


def test_target_fabric_requires_path_and_id():
    with pytest.raises(ValidationError):
        TargetFabricConfig(path="test.gpkg")  # missing id_field


def test_domain_bbox_requires_bbox_field():
    with pytest.raises(ValueError, match="bbox domain requires"):
        DomainConfig(type="bbox")


def test_domain_huc_requires_id_field():
    with pytest.raises(ValueError, match="huc2 domain requires"):
        DomainConfig(type="huc2")


def test_domain_gage_requires_id_field():
    with pytest.raises(ValueError, match="gage domain requires"):
        DomainConfig(type="gage")


def test_domain_huc_with_id():
    d = DomainConfig(type="huc2", id="02")
    assert d.id == "02"


def test_output_rejects_unknown_format():
    with pytest.raises(ValidationError):
        OutputConfig(format="csv")


def test_processing_rejects_unknown_engine():
    with pytest.raises(ValidationError):
        ProcessingConfig(engine="dask")


def test_processing_rejects_zero_batch_size():
    with pytest.raises(ValidationError):
        ProcessingConfig(batch_size=0)


def test_processing_rejects_negative_batch_size():
    with pytest.raises(ValidationError):
        ProcessingConfig(batch_size=-1)
