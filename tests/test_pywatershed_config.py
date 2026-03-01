"""Tests for pywatershed run configuration."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hydro_param.pywatershed_config import (
    ClimateNormalsConfig,
    ForcingConfig,
    LandcoverDatasets,
    ParameterEntry,
    PwsDomainConfig,
    PwsOutputConfig,
    PwsTimeConfig,
    PywatershedRunConfig,
    SnowDatasets,
    SoilsDatasets,
    StaticDatasetsConfig,
    TopographyDatasets,
    WaterbodyDatasets,
    load_pywatershed_config,
)


@pytest.fixture()
def minimal_config_dict(tmp_path: Path) -> dict:
    """Minimal valid v4.0 config dictionary."""
    fabric = tmp_path / "nhru.gpkg"
    fabric.touch()
    return {
        "target_model": "pywatershed",
        "version": "4.0",
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

    def test_invalid_start_date(self) -> None:
        with pytest.raises(ValidationError, match="Invalid date"):
            PwsTimeConfig(start="not-a-date", end="2020-09-30")

    def test_invalid_end_date(self) -> None:
        with pytest.raises(ValidationError, match="Invalid date"):
            PwsTimeConfig(start="1980-10-01", end="2020-13-45")

    def test_start_after_end_raises(self) -> None:
        """Reversed date range is rejected."""
        with pytest.raises(ValidationError, match="must be on or before"):
            PwsTimeConfig(start="2021-01-01", end="2020-01-01")

    def test_same_start_end_allowed(self) -> None:
        """Same start and end date (single day) is valid."""
        cfg = PwsTimeConfig(start="2020-06-15", end="2020-06-15")
        assert cfg.start == "2020-06-15"

    def test_empty_date_raises(self) -> None:
        """Empty string date is rejected."""
        with pytest.raises(ValidationError, match="Invalid date"):
            PwsTimeConfig(start="", end="2020-12-31")


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
        with pytest.warns(DeprecationWarning, match="cbh_dir"):
            cfg = PwsOutputConfig(**{"forcing_dir": "forcing", "cbh_dir": "cbh"})
        assert cfg.forcing_dir == "forcing"


class TestPywatershedRunConfig:
    """Tests for the top-level v4.0 config."""

    def test_minimal(self, minimal_config_dict: dict) -> None:
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.target_model == "pywatershed"
        assert cfg.version == "4.0"
        assert cfg.sir_path == Path("output")

    def test_sir_path_override(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["sir_path"] = "/custom/sir/output"
        cfg = PywatershedRunConfig(**minimal_config_dict)
        assert cfg.sir_path == Path("/custom/sir/output")

    def test_rejects_old_datasets_field(self, minimal_config_dict: dict) -> None:
        """v4.0 should not accept datasets field (extra=forbid)."""
        minimal_config_dict["datasets"] = {"topography": "dem_3dep_10m"}
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_rejects_old_climate_field(self, minimal_config_dict: dict) -> None:
        """v4.0 should not accept climate field (extra=forbid)."""
        minimal_config_dict["climate"] = {"source": "gridmet"}
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_rejects_old_processing_field(self, minimal_config_dict: dict) -> None:
        """v4.0 should not accept processing field (extra=forbid)."""
        minimal_config_dict["processing"] = {"batch_size": 500}
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_invalid_target_model(self, minimal_config_dict: dict) -> None:
        minimal_config_dict["target_model"] = "swat"
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**minimal_config_dict)

    def test_invalid_version(self, minimal_config_dict: dict) -> None:
        """v4.0 config rejects unknown version strings."""
        minimal_config_dict["version"] = "1.0"
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


class TestParameterEntry:
    """Tests for ParameterEntry schema."""

    def test_minimal_entry(self) -> None:
        entry = ParameterEntry(
            source="dem_3dep_10m",
            variable="elevation",
            statistic="mean",
            description="Mean HRU elevation",
        )
        assert entry.source == "dem_3dep_10m"
        assert entry.variable == "elevation"
        assert entry.statistic == "mean"

    def test_multi_variable_entry(self) -> None:
        entry = ParameterEntry(
            source="polaris_30m",
            variables=["sand", "silt", "clay"],
            statistic="mean",
            description="Soil type classification",
        )
        assert entry.variables == ["sand", "silt", "clay"]

    def test_temporal_entry_with_time_period(self) -> None:
        entry = ParameterEntry(
            source="snodas",
            variable="SWE",
            statistic="mean",
            time_period=["2020-01-01", "2021-12-31"],
            description="Snow depletion threshold",
        )
        assert entry.time_period == ["2020-01-01", "2021-12-31"]

    def test_entry_with_year(self) -> None:
        entry = ParameterEntry(
            source="nlcd_osn_lndcov",
            variable="LndCov",
            statistic="categorical",
            year=[2021],
            description="Vegetation cover type",
        )
        assert entry.year == [2021]

    def test_description_required(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            ParameterEntry(source="dem_3dep_10m", variable="elevation", statistic="mean")  # type: ignore[call-arg]

    def test_source_required(self) -> None:
        with pytest.raises(ValidationError, match="source"):
            ParameterEntry(variable="elevation", statistic="mean", description="test")  # type: ignore[call-arg]


class TestCategoryModels:
    """Tests for domain category Pydantic models."""

    def test_topography_defaults_empty(self) -> None:
        topo = TopographyDatasets()
        assert topo.available == []
        assert topo.hru_elev is None
        assert topo.hru_slope is None
        assert topo.hru_aspect is None

    def test_topography_with_entries(self) -> None:
        topo = TopographyDatasets(
            available=["dem_3dep_10m"],
            hru_elev=ParameterEntry(
                source="dem_3dep_10m",
                variable="elevation",
                statistic="mean",
                description="Mean HRU elevation",
            ),
        )
        assert topo.hru_elev is not None
        assert topo.hru_elev.source == "dem_3dep_10m"
        assert topo.hru_slope is None

    def test_soils_with_mixed_sources(self) -> None:
        soils = SoilsDatasets(
            available=["polaris_30m", "gnatsgo_rasters"],
            soil_type=ParameterEntry(
                source="polaris_30m",
                variables=["sand", "silt", "clay"],
                statistic="mean",
                description="Soil type classification",
            ),
            soil_moist_max=ParameterEntry(
                source="gnatsgo_rasters",
                variable="aws0_100",
                statistic="mean",
                description="Max available water-holding capacity",
            ),
        )
        assert soils.soil_type.source == "polaris_30m"
        assert soils.soil_moist_max.source == "gnatsgo_rasters"

    def test_forcing_defaults_empty(self) -> None:
        forcing = ForcingConfig()
        assert forcing.prcp is None
        assert forcing.tmax is None
        assert forcing.tmin is None

    def test_climate_normals_defaults_empty(self) -> None:
        cn = ClimateNormalsConfig()
        assert cn.jh_coef is None
        assert cn.transp_beg is None
        assert cn.transp_end is None

    def test_static_datasets_nests_all_categories(self) -> None:
        sd = StaticDatasetsConfig()
        assert isinstance(sd.topography, TopographyDatasets)
        assert isinstance(sd.soils, SoilsDatasets)
        assert isinstance(sd.landcover, LandcoverDatasets)
        assert isinstance(sd.snow, SnowDatasets)
        assert isinstance(sd.waterbodies, WaterbodyDatasets)


class TestPywatershedRunConfigV4:
    """Tests for v4.0 config with static_datasets, forcing, climate_normals."""

    def test_v4_accepts_new_sections(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(
                    available=["dem_3dep_10m"],
                    hru_elev=ParameterEntry(
                        source="dem_3dep_10m",
                        variable="elevation",
                        statistic="mean",
                        description="Mean HRU elevation",
                    ),
                ),
            ),
        )
        assert cfg.version == "4.0"
        assert cfg.static_datasets.topography.hru_elev is not None

    def test_v4_defaults_all_sections_empty(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
        )
        assert cfg.static_datasets.topography.hru_elev is None
        assert cfg.forcing.prcp is None
        assert cfg.climate_normals.jh_coef is None

    def test_v4_full_config_from_yaml(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        config_dict = {
            "target_model": "pywatershed",
            "version": "4.0",
            "sir_path": "output",
            "domain": {"fabric_path": str(fabric), "id_field": "nhm_id"},
            "time": {"start": "2020-01-01", "end": "2020-12-31"},
            "static_datasets": {
                "topography": {
                    "available": ["dem_3dep_10m"],
                    "hru_elev": {
                        "source": "dem_3dep_10m",
                        "variable": "elevation",
                        "statistic": "mean",
                        "description": "Mean HRU elevation",
                    },
                },
                "soils": {
                    "available": ["polaris_30m", "gnatsgo_rasters"],
                    "soil_type": {
                        "source": "polaris_30m",
                        "variables": ["sand", "silt", "clay"],
                        "statistic": "mean",
                        "description": "Soil type classification",
                    },
                },
            },
            "forcing": {
                "available": ["gridmet"],
                "prcp": {
                    "source": "gridmet",
                    "variable": "pr",
                    "statistic": "mean",
                    "description": "Daily precipitation",
                },
                "tmax": {
                    "source": "gridmet",
                    "variable": "tmmx",
                    "statistic": "mean",
                    "description": "Daily maximum temperature",
                },
                "tmin": {
                    "source": "gridmet",
                    "variable": "tmmn",
                    "statistic": "mean",
                    "description": "Daily minimum temperature",
                },
            },
            "climate_normals": {
                "available": ["gridmet"],
                "jh_coef": {
                    "source": "gridmet",
                    "variables": ["tmmx", "tmmn"],
                    "description": "Jensen-Haise PET coefficient",
                },
                "transp_beg": {
                    "source": "gridmet",
                    "variable": "tmmn",
                    "description": "Month transpiration begins",
                },
                "transp_end": {
                    "source": "gridmet",
                    "variable": "tmmn",
                    "description": "Month transpiration ends",
                },
            },
        }
        config_path = tmp_path / "pws.yml"
        config_path.write_text(yaml.dump(config_dict))
        cfg = load_pywatershed_config(config_path)
        assert cfg.static_datasets.topography.hru_elev.source == "dem_3dep_10m"
        assert cfg.forcing.prcp.source == "gridmet"
        assert cfg.climate_normals.transp_beg.source == "gridmet"

    def test_v4_rejects_unknown_top_level_field(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        with pytest.raises(ValidationError, match="extra"):
            PywatershedRunConfig(
                version="4.0",
                domain=PwsDomainConfig(fabric_path=fabric),
                time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
                bogus_field="oops",
            )


class TestConfigDeclaredEntries:
    """Tests for collecting declared parameter entries from config."""

    def test_collect_static_entries(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(
                    hru_elev=ParameterEntry(
                        source="dem_3dep_10m",
                        variable="elevation",
                        statistic="mean",
                        description="Mean HRU elevation",
                    ),
                ),
            ),
        )
        entries = cfg.declared_entries()
        assert "hru_elev" in entries
        assert entries["hru_elev"].source == "dem_3dep_10m"

    def test_collect_forcing_entries(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            forcing=ForcingConfig(
                prcp=ParameterEntry(
                    source="gridmet",
                    variable="pr",
                    statistic="mean",
                    description="Daily precipitation",
                ),
            ),
        )
        entries = cfg.declared_entries()
        assert "prcp" in entries

    def test_collect_climate_normals_entries(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            climate_normals=ClimateNormalsConfig(
                jh_coef=ParameterEntry(
                    source="gridmet",
                    variables=["tmmx", "tmmn"],
                    description="Jensen-Haise PET coefficient",
                ),
            ),
        )
        entries = cfg.declared_entries()
        assert "jh_coef" in entries

    def test_empty_config_returns_no_entries(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
        )
        entries = cfg.declared_entries()
        assert len(entries) == 0


class TestAvailableFieldValidation:
    """Tests for validating 'available' fields against the dataset registry."""

    def test_valid_available_passes(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(available=["dem_3dep_10m"]),
            ),
        )
        # Should not raise or warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg.validate_available_fields()

    def test_unknown_available_warns(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(available=["nonexistent_dataset"]),
            ),
        )
        with pytest.warns(UserWarning, match="nonexistent_dataset"):
            cfg.validate_available_fields()

    def test_empty_available_passes(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
        )
        # All available lists are empty by default — should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg.validate_available_fields()


class TestExampleConfig:
    """Tests for example config files."""

    def test_example_drb_2yr_pywatershed_loads(self) -> None:
        """Example config should parse without validation errors."""
        config_path = Path("configs/examples/drb_2yr_pywatershed.yml")
        if not config_path.exists():
            pytest.skip("Example config not found")
        cfg = load_pywatershed_config(config_path)
        assert cfg.version == "4.0"
        assert cfg.static_datasets.topography.hru_elev is not None
        assert cfg.static_datasets.topography.hru_elev.source == "dem_3dep_10m"
        assert cfg.forcing.prcp is not None
        assert cfg.climate_normals.jh_coef is not None
