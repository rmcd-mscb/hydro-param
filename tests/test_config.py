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
        "datasets": {
            "topography": [
                {"name": "dem_3dep_10m", "variables": ["elevation", "slope"]},
            ],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert isinstance(config, PipelineConfig)
    assert config.target_fabric.id_field == "featureid"
    assert len(config.flatten_datasets()) == 1
    assert config.datasets["topography"][0].name == "dem_3dep_10m"
    assert config.datasets["topography"][0].variables == ["elevation", "slope"]


def test_config_full_yaml(tmp_path: Path):
    raw = {
        "target_fabric": {
            "path": "data/catchments.gpkg",
            "id_field": "hru_id",
            "crs": "EPSG:5070",
        },
        "domain": {"type": "bbox", "bbox": [-76.5, 38.5, -74.0, 42.6]},
        "datasets": {
            "topography": [
                {"name": "dem", "variables": ["elevation"], "statistics": ["mean", "std"]},
            ],
        },
        "output": {
            "path": "./results",
            "format": "parquet",
            "sir_name": "delaware_test",
        },
        "processing": {
            "batch_size": 200,
        },
    }
    path = tmp_path / "full.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert config.target_fabric.crs == "EPSG:5070"
    assert config.output.format == "parquet"
    assert config.output.sir_name == "delaware_test"
    assert config.processing.batch_size == 200


def test_config_defaults():
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets={},
    )
    assert config.output.path == Path("./output")
    assert config.output.format == "netcdf"
    assert config.output.sir_name == "result"
    assert config.processing.batch_size == 500
    assert config.processing.batch_size == 500


def test_dataset_request_defaults():
    ds = DatasetRequest(name="dem")
    assert ds.variables == []
    assert ds.statistics == ["mean"]
    assert ds.source is None


def test_dataset_request_accepts_source():
    ds = DatasetRequest(name="nlcd", source=Path("/data/nlcd.tif"))
    assert ds.source == Path("/data/nlcd.tif")


def test_dataset_request_source_from_yaml(tmp_path: Path):
    raw = {
        "target_fabric": {"path": "data/fabric.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": {
            "land_cover": [
                {
                    "name": "nlcd_2021",
                    "source": "data/nlcd.tif",
                    "variables": ["land_cover"],
                    "statistics": ["majority"],
                },
            ],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert config.datasets["land_cover"][0].source is not None
    assert config.datasets["land_cover"][0].source.is_absolute()
    assert config.datasets["land_cover"][0].source.name == "nlcd.tif"


def test_target_fabric_requires_path_and_id():
    with pytest.raises(ValidationError):
        TargetFabricConfig(path="test.gpkg")  # missing id_field


def test_domain_optional(tmp_path: Path):
    """Pipeline config without domain uses full fabric extent."""
    raw = {
        "target_fabric": {"path": "data/catchments.gpkg", "id_field": "featureid"},
        "datasets": {
            "topography": [{"name": "dem_3dep_10m", "variables": ["elevation"]}],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))
    config = load_config(str(path))
    assert config.domain is None


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


def test_processing_rejects_zero_batch_size():
    with pytest.raises(ValidationError):
        ProcessingConfig(batch_size=0)


def test_processing_rejects_negative_batch_size():
    with pytest.raises(ValidationError):
        ProcessingConfig(batch_size=-1)


def test_dataset_request_year_field():
    ds = DatasetRequest(name="nlcd_osn_lndcov", variables=["LndCov"], year=2021)
    assert ds.year == 2021


def test_dataset_request_year_default_none():
    ds = DatasetRequest(name="nlcd_osn_lndcov")
    assert ds.year is None


def test_dataset_request_year_rejects_invalid():
    with pytest.raises(ValidationError, match="outside valid range"):
        DatasetRequest(name="test", year=1800)
    with pytest.raises(ValidationError, match="outside valid range"):
        DatasetRequest(name="test", year=2200)


def test_dataset_request_year_accepts_list():
    ds = DatasetRequest(name="nlcd", variables=["LndCov"], year=[2019, 2020, 2021])
    assert ds.year == [2019, 2020, 2021]


def test_dataset_request_year_list_rejects_out_of_range():
    with pytest.raises(ValidationError, match="outside valid range"):
        DatasetRequest(name="test", year=[2020, 2200])


def test_dataset_request_year_list_rejects_empty():
    with pytest.raises(ValidationError, match="year list cannot be empty"):
        DatasetRequest(name="test", year=[])


def test_dataset_request_year_list_from_yaml(tmp_path: Path):
    raw = {
        "target_fabric": {"path": "data/fabric.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": {
            "land_cover": [
                {
                    "name": "nlcd",
                    "variables": ["LndCov"],
                    "statistics": ["categorical"],
                    "year": [2020, 2021],
                },
            ],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert config.datasets["land_cover"][0].year == [2020, 2021]


# ---------------------------------------------------------------------------
# time_period field
# ---------------------------------------------------------------------------


def test_dataset_request_time_period():
    ds = DatasetRequest(
        name="snodas",
        variables=["SWE"],
        time_period=["2020-01-01", "2020-12-31"],
    )
    assert ds.time_period == ["2020-01-01", "2020-12-31"]


def test_dataset_request_time_period_default_none():
    ds = DatasetRequest(name="dem")
    assert ds.time_period is None


def test_dataset_request_time_period_requires_two_elements():
    with pytest.raises(ValidationError):
        DatasetRequest(name="test", time_period=["2020-01-01"])
    with pytest.raises(ValidationError):
        DatasetRequest(name="test", time_period=["2020-01-01", "2020-06-01", "2020-12-31"])


def test_dataset_request_time_period_validates_date_format():
    with pytest.raises(ValidationError, match="valid ISO format"):
        DatasetRequest(name="test", time_period=["not-a-date", "2020-12-31"])


def test_dataset_request_time_period_validates_date_order():
    with pytest.raises(ValidationError, match="start.*must be <= end"):
        DatasetRequest(name="test", time_period=["2021-01-01", "2020-12-31"])


def test_dataset_request_time_period_from_yaml(tmp_path: Path):
    raw = {
        "target_fabric": {"path": "data/fabric.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": {
            "snow": [
                {
                    "name": "snodas",
                    "variables": ["SWE"],
                    "statistics": ["mean"],
                    "time_period": ["2020-01-01", "2020-12-31"],
                },
            ],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert config.datasets["snow"][0].time_period == ["2020-01-01", "2020-12-31"]


# ---------------------------------------------------------------------------
# sir_validation field
# ---------------------------------------------------------------------------


def test_sir_validation_default() -> None:
    """ProcessingConfig defaults sir_validation to 'tolerant'."""
    config = ProcessingConfig()
    assert config.sir_validation == "tolerant"


def test_sir_validation_strict() -> None:
    config = ProcessingConfig(sir_validation="strict")
    assert config.sir_validation == "strict"


# ---------------------------------------------------------------------------
# network_timeout field
# ---------------------------------------------------------------------------


def test_processing_config_network_timeout_default() -> None:
    """network_timeout defaults to 120."""
    pc = ProcessingConfig()
    assert pc.network_timeout == 120


def test_processing_config_network_timeout_custom() -> None:
    """network_timeout accepts positive int."""
    pc = ProcessingConfig(network_timeout=300)
    assert pc.network_timeout == 300


def test_processing_config_network_timeout_rejects_zero() -> None:
    """network_timeout rejects 0."""
    with pytest.raises(ValueError):
        ProcessingConfig(network_timeout=0)


def test_processing_config_network_timeout_rejects_negative() -> None:
    """network_timeout rejects negative values."""
    with pytest.raises(ValueError):
        ProcessingConfig(network_timeout=-10)


# ---------------------------------------------------------------------------
# Path resolution in load_config
# ---------------------------------------------------------------------------


def test_load_config_resolves_relative_paths(tmp_path: Path):
    """load_config resolves relative paths to absolute against CWD."""
    raw = {
        "target_fabric": {"path": "data/catchments.gpkg", "id_field": "id"},
        "datasets": {
            "topography": [
                {"name": "dem", "variables": ["elevation"]},
            ],
        },
        "output": {"path": "output"},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert config.target_fabric.path.is_absolute()
    assert config.target_fabric.path == Path.cwd() / "data" / "catchments.gpkg"
    assert config.output.path.is_absolute()
    assert config.output.path == Path.cwd() / "output"


def test_load_config_resolves_dotdot_paths(tmp_path: Path):
    """Paths with '..' components are fully resolved."""
    raw = {
        "target_fabric": {"path": "../sibling/catchments.gpkg", "id_field": "id"},
        "datasets": {},
        "output": {"path": "sub/../output"},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert ".." not in str(config.target_fabric.path)
    assert ".." not in str(config.output.path)
    assert config.target_fabric.path.is_absolute()
    assert config.output.path.is_absolute()


def test_load_config_preserves_absolute_paths(tmp_path: Path):
    """load_config does not modify already-absolute paths."""
    raw = {
        "target_fabric": {"path": "/abs/data/catchments.gpkg", "id_field": "id"},
        "datasets": {},
        "output": {"path": "/abs/output"},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert str(config.target_fabric.path) == "/abs/data/catchments.gpkg"
    assert str(config.output.path) == "/abs/output"


def test_load_config_resolves_dataset_source(tmp_path: Path):
    """load_config resolves relative dataset source paths to absolute."""
    raw = {
        "target_fabric": {"path": "fabric.gpkg", "id_field": "id"},
        "datasets": {
            "land_cover": [
                {"name": "nlcd", "source": "data/nlcd.tif", "variables": ["lc"]},
            ],
            "topography": [
                {"name": "dem", "variables": ["elevation"]},
            ],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert config.datasets["land_cover"][0].source is not None
    assert config.datasets["land_cover"][0].source.is_absolute()
    assert config.datasets["topography"][0].source is None


# ---------------------------------------------------------------------------
# Themed datasets (dict keyed by category)
# ---------------------------------------------------------------------------


def test_themed_datasets_from_yaml(tmp_path: Path):
    """Pipeline config accepts datasets organized by category."""
    raw = {
        "target_fabric": {"path": "data/catchments.gpkg", "id_field": "featureid"},
        "datasets": {
            "topography": [
                {"name": "dem_3dep_10m", "variables": ["elevation"]},
            ],
            "soils": [
                {"name": "gnatsgo_rasters", "variables": ["aws0_100"]},
            ],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert "topography" in config.datasets
    assert "soils" in config.datasets
    assert config.datasets["topography"][0].name == "dem_3dep_10m"


def test_themed_datasets_rejects_unknown_category():
    """Unknown category key raises ValidationError."""
    with pytest.raises(ValidationError, match="not_a_category"):
        PipelineConfig(
            target_fabric={"path": "test.gpkg", "id_field": "id"},
            datasets={"not_a_category": [{"name": "dem", "variables": ["elevation"]}]},
        )


def test_flatten_datasets():
    """flatten_datasets() merges all categories into a flat list."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        datasets={
            "topography": [
                DatasetRequest(name="dem", variables=["elevation"]),
            ],
            "soils": [
                DatasetRequest(name="gnatsgo", variables=["aws0_100"]),
                DatasetRequest(name="polaris", variables=["sand"]),
            ],
        },
    )
    flat = config.flatten_datasets()
    assert len(flat) == 3
    assert [ds.name for ds in flat] == ["dem", "gnatsgo", "polaris"]


def test_themed_datasets_empty_dict():
    """Empty datasets dict is valid."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        datasets={},
    )
    assert config.flatten_datasets() == []


def test_themed_datasets_empty_category_list():
    """Category with empty list is valid."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        datasets={"topography": []},
    )
    assert config.flatten_datasets() == []
