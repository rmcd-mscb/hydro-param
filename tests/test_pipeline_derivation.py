"""Tests for pipeline derivation dispatch and config translation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import Polygon

from hydro_param.config import OutputConfig, PipelineConfig
from hydro_param.pipeline import PipelineResult, _run_derivation


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

    def test_temporal_renames_use_short_names(self) -> None:
        """Verify temporal_renames maps short gridMET names, not CF long names."""
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
            climate={"source": "gridmet", "variables": ["prcp", "tmax", "tmin"]},
        )

        pipeline_config = _translate_pws_to_pipeline(pws_config)
        renames = pipeline_config.output.derivation_options["temporal_renames"]
        assert renames == {"pr": "prcp", "tmmx": "tmax", "tmmn": "tmin"}


class TestRunDerivation:
    """Tests for _run_derivation temporal processing."""

    @pytest.fixture()
    def sir_with_topo(self) -> xr.Dataset:
        return xr.Dataset(
            {
                "elevation": ("hru_id", np.array([100.0, 500.0])),
                "slope": ("hru_id", np.array([5.0, 15.0])),
                "aspect": ("hru_id", np.array([0.0, 90.0])),
            },
            coords={"hru_id": [1, 2]},
        )

    @pytest.fixture()
    def fabric(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {"nhm_id": [1, 2]},
            geometry=[
                Polygon([(0, 40), (1, 40), (1, 41), (0, 41)]),
                Polygon([(1, 40), (2, 40), (2, 41), (1, 41)]),
            ],
            crs="EPSG:4326",
        )

    @pytest.fixture()
    def temporal_data(self) -> dict[str, xr.Dataset]:
        """Synthetic temporal dataset mimicking gridMET output."""
        times = ["2020-01-01", "2020-01-02"]
        return {
            "gridmet": xr.Dataset(
                {
                    "pr": (("nhm_id", "time"), np.array([[1.0, 2.0], [3.0, 4.0]])),
                    "tmmx": (("nhm_id", "time"), np.array([[300.0, 301.0], [302.0, 303.0]])),
                    "tmmn": (("nhm_id", "time"), np.array([[280.0, 281.0], [282.0, 283.0]])),
                },
                coords={"nhm_id": [1, 2], "time": times},
            )
        }

    def test_temporal_rename_and_conversion(
        self,
        sir_with_topo: xr.Dataset,
        fabric: gpd.GeoDataFrame,
        temporal_data: dict,
        tmp_path: Path,
    ) -> None:
        """Temporal vars renamed pr→prcp, tmmx→tmax; K→C conversion applied."""
        config = PipelineConfig(
            target_fabric={"path": "test.gpkg", "id_field": "nhm_id"},
            domain={"type": "bbox", "bbox": [0, 0, 2, 41]},
            datasets=[],
            output={
                "path": str(tmp_path),
                "derivation": "pywatershed",
                "derivation_options": {
                    "start": "2020-01-01",
                    "end": "2020-01-02",
                    "id_field": "nhm_id",
                    "temporal_renames": {"pr": "prcp", "tmmx": "tmax", "tmmn": "tmin"},
                    "temporal_conversions": {"tmax": ("K", "C"), "tmin": ("K", "C")},
                },
            },
        )

        # Mock the formatter to capture what it receives
        mock_formatter = MagicMock()
        mock_formatter.write.return_value = []
        mock_formatter.validate.return_value = []

        with patch("hydro_param.output.get_formatter", return_value=mock_formatter):
            _run_derivation(sir_with_topo, temporal_data, fabric, config)

        # Verify formatter was called
        assert mock_formatter.write.called
        derived = mock_formatter.write.call_args[0][0]

        # Temporal vars should be renamed
        assert "prcp" in derived
        assert "tmax" in derived
        assert "tmin" in derived
        assert "pr" not in derived
        assert "tmmx" not in derived

        # K→C conversion: 300K = 26.85°C
        np.testing.assert_allclose(derived["tmax"].values[0, 0], 26.85, atol=0.01)

        # Feature dimension should be nhru (aligned with derived params)
        assert "nhru" in derived["prcp"].dims

    def test_missing_start_end_raises(
        self, sir_with_topo: xr.Dataset, fabric: gpd.GeoDataFrame
    ) -> None:
        """Validate that missing start/end raises early."""
        config = PipelineConfig(
            target_fabric={"path": "test.gpkg", "id_field": "nhm_id"},
            domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
            datasets=[],
            output={
                "derivation": "pywatershed",
                "derivation_options": {"id_field": "nhm_id"},
            },
        )

        with pytest.raises(ValueError, match="derivation_options.*'start'"):
            _run_derivation(sir_with_topo, {}, fabric, config)
