"""Tests for pipeline derivation dispatch and config translation."""

from __future__ import annotations

import pytest
import xarray as xr

from hydro_param.config import OutputConfig, PipelineConfig
from hydro_param.pipeline import PipelineResult


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_default_values(self) -> None:
        sir = xr.Dataset({"elevation": ("hru_id", [100.0])}, coords={"hru_id": [1]})
        result = PipelineResult(sir=sir)
        assert result.sir is sir
        assert result.temporal == {}
        assert result.fabric is None

    def test_with_temporal(self) -> None:
        sir = xr.Dataset(coords={"hru_id": [1]})
        temporal_ds = xr.Dataset({"temp": ("time", [1.0, 2.0])})
        result = PipelineResult(sir=sir, temporal={"gridmet": temporal_ds})
        assert "gridmet" in result.temporal


class TestOutputConfigDerivation:
    """Tests for derivation fields in OutputConfig."""

    def test_default_no_derivation(self) -> None:
        config = OutputConfig()
        assert config.derivation is None
        assert config.derivation_options == {}

    def test_with_derivation(self) -> None:
        config = OutputConfig(
            derivation="pywatershed",
            derivation_options={"id_field": "nhm_id", "start": "2020-01-01", "end": "2021-12-31"},
        )
        assert config.derivation == "pywatershed"
        assert config.derivation_options["id_field"] == "nhm_id"

    def test_pipeline_config_with_derivation(self) -> None:
        config = PipelineConfig(
            target_fabric={"path": "test.gpkg", "id_field": "id"},
            domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
            datasets=[],
            output={
                "derivation": "pywatershed",
                "derivation_options": {"start": "2020-01-01", "end": "2021-12-31"},
            },
        )
        assert config.output.derivation == "pywatershed"


class TestPwsConfigTranslation:
    """Tests for PywatershedRunConfig → PipelineConfig translation."""

    def test_translate_basic(self) -> None:
        from hydro_param.cli import _translate_pws_to_pipeline
        from hydro_param.pywatershed_config import PywatershedRunConfig

        pws_config = PywatershedRunConfig(
            domain={
                "source": "custom",
                "extraction_method": "bbox",
                "bbox": [-75.8, 39.6, -74.4, 42.5],
                "fabric_path": "data/nhru.gpkg",
                "segment_path": "data/nsegment.gpkg",
            },
            time={"start": "2020-01-01", "end": "2021-12-31"},
            climate={"source": "gridmet", "variables": ["prcp", "tmax", "tmin"]},
            datasets={"topography": "dem_3dep_10m", "landcover": "nlcd_osn_lndcov"},
        )

        pipeline_config = _translate_pws_to_pipeline(pws_config)

        assert pipeline_config.output.derivation == "pywatershed"
        assert pipeline_config.target_fabric.id_field == "nhm_id"
        assert pipeline_config.domain.type == "bbox"
        assert len(pipeline_config.datasets) == 3  # topo + landcover + climate

        # Check datasets
        ds_names = [d.name for d in pipeline_config.datasets]
        assert "dem_3dep_10m" in ds_names
        assert "nlcd_osn_lndcov" in ds_names
        assert "gridmet" in ds_names

        # NLCD should use categorical statistics
        nlcd_ds = next(d for d in pipeline_config.datasets if d.name == "nlcd_osn_lndcov")
        assert nlcd_ds.statistics == ["categorical"]
        assert nlcd_ds.year == 2021

        # gridMET should have time_period and mapped variable names
        gridmet_ds = next(d for d in pipeline_config.datasets if d.name == "gridmet")
        assert gridmet_ds.time_period == ["2020-01-01", "2021-12-31"]
        assert gridmet_ds.variables == ["pr", "tmmx", "tmmn"]

    def test_translate_derivation_options(self) -> None:
        from hydro_param.cli import _translate_pws_to_pipeline
        from hydro_param.pywatershed_config import PywatershedRunConfig

        pws_config = PywatershedRunConfig(
            domain={
                "source": "custom",
                "extraction_method": "bbox",
                "bbox": [-75.8, 39.6, -74.4, 42.5],
                "fabric_path": "data/nhru.gpkg",
                "segment_path": "data/nsegment.gpkg",
            },
            time={"start": "2020-01-01", "end": "2021-12-31"},
        )

        pipeline_config = _translate_pws_to_pipeline(pws_config)
        opts = pipeline_config.output.derivation_options

        assert opts["start"] == "2020-01-01"
        assert opts["end"] == "2021-12-31"
        assert opts["segment_path"] == "data/nsegment.gpkg"
        assert "temporal_renames" in opts
        assert "temporal_conversions" in opts

    def test_translate_missing_fabric_raises(self) -> None:
        from hydro_param.cli import _translate_pws_to_pipeline
        from hydro_param.pywatershed_config import PywatershedRunConfig

        pws_config = PywatershedRunConfig(
            domain={
                "source": "geospatial_fabric",
                "extraction_method": "bbox",
                "bbox": [-75.8, 39.6, -74.4, 42.5],
            },
            time={"start": "2020-01-01", "end": "2021-12-31"},
        )

        with pytest.raises(ValueError, match="fabric_path"):
            _translate_pws_to_pipeline(pws_config)
