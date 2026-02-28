"""Tests for pywatershed run configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hydro_param.pywatershed_config import (
    PwsDomainConfig,
    PwsOutputConfig,
    PwsTimeConfig,
    PywatershedRunConfig,
    load_pywatershed_config,
)


@pytest.fixture()
def minimal_config_dict() -> dict:
    """Minimal valid config dictionary."""
    return {
        "target_model": "pywatershed",
        "domain": {
            "extraction_method": "bbox",
            "bbox": [-76.5, 38.5, -74.0, 42.6],
        },
        "time": {
            "start": "1980-10-01",
            "end": "2020-09-30",
        },
    }


@pytest.fixture()
def minimal_config_yaml(tmp_path: Path, minimal_config_dict: dict) -> Path:
    """Write minimal config to a YAML file."""
    path = tmp_path / "pws_config.yml"
    path.write_text(yaml.dump(minimal_config_dict))
    return path


class TestPwsDomainConfig:
    """Tests for domain configuration validation."""

    def test_bbox_valid(self) -> None:
        cfg = PwsDomainConfig(extraction_method="bbox", bbox=[-76.5, 38.5, -74.0, 42.6])
        assert cfg.bbox == [-76.5, 38.5, -74.0, 42.6]

    def test_bbox_requires_bbox_field(self) -> None:
        with pytest.raises(ValidationError, match="bbox extraction requires"):
            PwsDomainConfig(extraction_method="bbox")

    def test_huc_requires_huc_id(self) -> None:
        with pytest.raises(ValidationError, match="huc extraction requires"):
            PwsDomainConfig(extraction_method="huc")

    def test_huc_valid(self) -> None:
        cfg = PwsDomainConfig(extraction_method="huc", huc_id="01013500")
        assert cfg.huc_id == "01013500"

    def test_pour_point_requires_coords(self) -> None:
        with pytest.raises(ValidationError, match="pour_point extraction requires"):
            PwsDomainConfig(extraction_method="pour_point")

    def test_pour_point_valid(self) -> None:
        cfg = PwsDomainConfig(extraction_method="pour_point", pour_point=[-73.95, 42.45])
        assert cfg.pour_point == [-73.95, 42.45]

    def test_bbox_wrong_length(self) -> None:
        with pytest.raises(ValidationError, match="4 coordinates"):
            PwsDomainConfig(extraction_method="bbox", bbox=[-76.5, 38.5])

    def test_pour_point_wrong_length(self) -> None:
        with pytest.raises(ValidationError, match="2 coordinates"):
            PwsDomainConfig(extraction_method="pour_point", pour_point=[-73.95, 42.45, 100.0])

    def test_custom_requires_fabric_path(self) -> None:
        with pytest.raises(ValidationError, match="custom domain source requires"):
            PwsDomainConfig(source="custom", extraction_method="bbox", bbox=[0, 0, 1, 1])

    def test_custom_valid(self) -> None:
        cfg = PwsDomainConfig(
            source="custom",
            extraction_method="bbox",
            bbox=[0, 0, 1, 1],
            fabric_path=Path("/some/fabric.gpkg"),
        )
        assert cfg.fabric_path == Path("/some/fabric.gpkg")

    def test_waterbody_path_default_none(self) -> None:
        cfg = PwsDomainConfig(extraction_method="bbox", bbox=[-76.5, 38.5, -74.0, 42.6])
        assert cfg.waterbody_path is None

    def test_waterbody_path_accepted(self) -> None:
        cfg = PwsDomainConfig(
            extraction_method="bbox",
            bbox=[-76.5, 38.5, -74.0, 42.6],
            waterbody_path=Path("/some/waterbodies.gpkg"),
        )
        assert cfg.waterbody_path == Path("/some/waterbodies.gpkg")


class TestPwsTimeConfig:
    """Tests for time configuration."""

    def test_required_fields(self) -> None:
        cfg = PwsTimeConfig(start="1980-10-01", end="2020-09-30")
        assert cfg.start == "1980-10-01"
        assert cfg.end == "2020-09-30"
        assert cfg.timestep == "daily"

    def test_missing_start(self) -> None:
        with pytest.raises(ValidationError):
            PwsTimeConfig(end="2020-09-30")  # type: ignore[call-arg]


class TestPwsOutputConfig:
    """Tests for output configuration, including cbh_dir migration."""

    def test_defaults(self) -> None:
        cfg = PwsOutputConfig()
        assert cfg.forcing_dir == "forcing"

    def test_forcing_dir(self) -> None:
        cfg = PwsOutputConfig(forcing_dir="my_forcing")
        assert cfg.forcing_dir == "my_forcing"

    def test_legacy_cbh_dir_migrated(self) -> None:
        with pytest.warns(DeprecationWarning, match="cbh_dir"):
            cfg = PwsOutputConfig(**{"cbh_dir": "cbh"})
        assert cfg.forcing_dir == "cbh"

    def test_forcing_dir_takes_precedence_over_cbh_dir(self) -> None:
        cfg = PwsOutputConfig(**{"forcing_dir": "forcing", "cbh_dir": "cbh"})
        assert cfg.forcing_dir == "forcing"


class TestPywatershedRunConfig:
    """Tests for the top-level config."""

    def test_minimal(self, minimal_config_dict: dict) -> None:
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.target_model == "pywatershed"
        assert cfg.version == "2.0"
        assert cfg.domain.bbox == [-76.5, 38.5, -74.0, 42.6]
        assert cfg.time.start == "1980-10-01"

    def test_defaults(self, minimal_config_dict: dict) -> None:
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.climate.source == "gridmet"
        assert cfg.datasets.topography == "dem_3dep_10m"
        assert cfg.processing.zonal_method == "exactextract"
        assert cfg.calibration.generate_seeds is True
        assert cfg.output.format == "netcdf"

    def test_full_config(self, minimal_config_dict: dict) -> None:
        minimal_config_dict.update(
            {
                "climate": {
                    "source": "gridmet",
                    "variables": ["prcp", "tmax", "tmin", "srad"],
                },
                "datasets": {
                    "topography": "dem_3dep_10m",
                    "landcover": "nlcd_annual",
                    "soils": "polaris_30m",
                },
                "processing": {
                    "zonal_method": "exactextract",
                    "batch_size": 1000,
                    "n_workers": 4,
                },
                "parameter_overrides": {
                    "values": {"tmax_allsnow": 32.0, "den_max": 0.55},
                },
                "calibration": {
                    "generate_seeds": True,
                    "seed_method": "physically_based",
                },
                "output": {
                    "path": "./models/pws",
                    "parameter_file": "params.nc",
                },
            }
        )
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.climate.source == "gridmet"
        assert cfg.processing.n_workers == 4
        assert cfg.parameter_overrides.values["tmax_allsnow"] == 32.0

    def test_invalid_target_model(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["target_model"] = "swat"
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)


class TestLoadPywatershedConfig:
    """Tests for YAML loading."""

    def test_load_valid(self, minimal_config_yaml: Path) -> None:
        cfg = load_pywatershed_config(minimal_config_yaml)
        assert cfg.target_model == "pywatershed"
        assert cfg.domain.bbox is not None

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pywatershed_config("/nonexistent/config.yml")
