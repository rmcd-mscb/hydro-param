"""Tests for pipeline result dataclass and config translation."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from hydro_param.pipeline import PipelineResult


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_default_values(self) -> None:
        result = PipelineResult(output_dir=Path("/tmp"))
        assert result.static_files == {}
        assert result.temporal_files == {}
        assert result.categories == {}
        assert result.fabric is None

    def test_with_files(self, tmp_path: Path) -> None:
        elev_path = tmp_path / "elevation.nc"
        temporal_path = tmp_path / "gridmet_temporal.nc"
        result = PipelineResult(
            output_dir=tmp_path,
            static_files={"elevation": elev_path},
            temporal_files={"gridmet": temporal_path},
        )
        assert "elevation" in result.static_files
        assert "gridmet" in result.temporal_files

    def test_load_sir_from_files(self, tmp_path: Path) -> None:
        """load_sir() assembles a combined Dataset from per-variable CSV files."""
        import pandas as pd

        df = pd.DataFrame({"elevation": [100.0]}, index=pd.Index([1], name="nhm_id"))
        path = tmp_path / "elevation.csv"
        df.to_csv(path, index=True)

        result = PipelineResult(
            output_dir=tmp_path,
            static_files={"elevation": path},
        )
        sir = result.load_sir()
        assert "elevation" in sir.data_vars

    def test_load_sir_empty(self) -> None:
        """load_sir() returns empty Dataset when no files."""
        result = PipelineResult(output_dir=Path("/tmp"))
        sir = result.load_sir()
        assert isinstance(sir, xr.Dataset)
        assert len(sir.data_vars) == 0


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
        assert len(pipeline_config.datasets) == 4  # topo + landcover + soils + climate

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

    def test_translate_unsupported_climate_raises(self) -> None:
        """Unsupported climate sources raise ValueError with helpful message."""
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
            climate={"source": "daymet_v4"},
        )

        with pytest.raises(ValueError, match="not yet supported"):
            _translate_pws_to_pipeline(pws_config)

    def test_translate_includes_soils(self) -> None:
        """Soils dataset is included in translated pipeline config."""
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
            datasets={"topography": "dem_3dep_10m", "landcover": "nlcd_osn_lndcov", "soils": "polaris_30m"},
        )

        pipeline_config = _translate_pws_to_pipeline(pws_config)
        ds_names = [d.name for d in pipeline_config.datasets]
        assert "polaris_30m" in ds_names
        assert len(pipeline_config.datasets) == 4  # topo + landcover + soils + climate

        soils_ds = next(d for d in pipeline_config.datasets if d.name == "polaris_30m")
        assert "sand" in soils_ds.variables
        assert "clay" in soils_ds.variables
        assert "silt" in soils_ds.variables
        assert soils_ds.statistics == ["mean"]
