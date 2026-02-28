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
