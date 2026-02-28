"""Tests for pywatershed run configuration (v3.0)."""

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
def minimal_config_dict(tmp_path: Path) -> dict:
    """Minimal valid v3.0 config dictionary."""
    fabric = tmp_path / "nhru.gpkg"
    fabric.touch()
    return {
        "target_model": "pywatershed",
        "version": "3.0",
        "domain": {
            "fabric_path": str(fabric),
            "id_field": "nhm_id",
        },
        "time": {
            "start": "2020-01-01",
            "end": "2020-12-31",
        },
    }


@pytest.fixture()
def minimal_config_yaml(tmp_path: Path, minimal_config_dict: dict) -> Path:
    """Write minimal config to a YAML file."""
    path = tmp_path / "pws_config.yml"
    path.write_text(yaml.dump(minimal_config_dict))
    return path


class TestPwsDomainConfig:
    """Tests for simplified domain configuration."""

    def test_fabric_path_required(self) -> None:
        with pytest.raises(ValidationError, match="fabric_path"):
            PwsDomainConfig()  # type: ignore[call-arg]

    def test_fabric_path_valid(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PwsDomainConfig(fabric_path=fabric)
        assert cfg.fabric_path == fabric
        assert cfg.id_field == "nhm_id"
        assert cfg.segment_id_field == "nhm_seg"

    def test_optional_paths(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        segments = tmp_path / "nseg.gpkg"
        waterbodies = tmp_path / "wb.gpkg"
        cfg = PwsDomainConfig(
            fabric_path=fabric,
            segment_path=segments,
            waterbody_path=waterbodies,
            id_field="hru_id",
            segment_id_field="seg_id",
        )
        assert cfg.segment_path == segments
        assert cfg.waterbody_path == waterbodies
        assert cfg.id_field == "hru_id"
        assert cfg.segment_id_field == "seg_id"

    def test_waterbody_path_default_none(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PwsDomainConfig(fabric_path=fabric)
        assert cfg.waterbody_path is None

    def test_no_extraction_method(self, tmp_path: Path) -> None:
        """v3.0 domain has no extraction_method field."""
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PwsDomainConfig(fabric_path=fabric)
        assert not hasattr(cfg, "extraction_method") or "extraction_method" not in cfg.model_fields
        assert not hasattr(cfg, "bbox") or "bbox" not in cfg.model_fields


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
    """Tests for the top-level v3.0 config."""

    def test_minimal(self, minimal_config_dict: dict) -> None:
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.target_model == "pywatershed"
        assert cfg.version == "3.0"
        assert cfg.sir_path == Path("output")

    def test_sir_path_override(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["sir_path"] = "/custom/sir/output"
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.sir_path == Path("/custom/sir/output")

    def test_rejects_old_datasets_field(self, minimal_config_dict: dict) -> None:
        """v3.0 should not accept datasets field (extra=forbid)."""
        minimal_config_dict["datasets"] = {"topography": "dem_3dep_10m"}
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_rejects_old_climate_field(self, minimal_config_dict: dict) -> None:
        """v3.0 should not accept climate field (extra=forbid)."""
        minimal_config_dict["climate"] = {"source": "gridmet"}
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_rejects_old_processing_field(self, minimal_config_dict: dict) -> None:
        """v3.0 should not accept processing field (extra=forbid)."""
        minimal_config_dict["processing"] = {"batch_size": 500}
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_invalid_target_model(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["target_model"] = "swat"
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_defaults(self, minimal_config_dict: dict) -> None:
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.calibration.generate_seeds is True
        assert cfg.output.format == "netcdf"
        assert cfg.parameter_overrides.values == {}

    def test_full_config(self, minimal_config_dict: dict) -> None:
        minimal_config_dict.update(
            {
                "sir_path": "/pipeline/output",
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
        assert cfg.sir_path == Path("/pipeline/output")
        assert cfg.parameter_overrides.values["tmax_allsnow"] == 32.0


class TestLoadPywatershedConfig:
    """Tests for YAML loading."""

    def test_load_valid(self, minimal_config_yaml: Path) -> None:
        cfg = load_pywatershed_config(minimal_config_yaml)
        assert cfg.target_model == "pywatershed"

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pywatershed_config("/nonexistent/config.yml")
