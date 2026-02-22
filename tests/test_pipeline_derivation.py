"""Tests for pipeline result dataclass and config translation."""

from __future__ import annotations

import pytest
import xarray as xr

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

        # No derivation on the generic pipeline config
        assert not hasattr(pipeline_config.output, "derivation")
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

    def test_translate_no_derivation_options(self) -> None:
        """OutputConfig has no derivation or derivation_options fields."""
        from hydro_param.cli import _translate_pws_to_pipeline
        from hydro_param.pywatershed_config import PywatershedRunConfig

        pws_config = PywatershedRunConfig(
            domain={
                "source": "custom",
                "extraction_method": "bbox",
                "bbox": [-75.8, 39.6, -74.4, 42.5],
                "fabric_path": "data/nhru.gpkg",
            },
            time={"start": "2020-01-01", "end": "2021-12-31"},
        )

        pipeline_config = _translate_pws_to_pipeline(pws_config)
        # OutputConfig should only have path, format, sir_name
        assert pipeline_config.output.sir_name == "pywatershed_sir"
        assert pipeline_config.output.format == "netcdf"
