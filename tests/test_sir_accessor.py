"""Tests for SIRAccessor lazy variable loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture()
def sir_dir(tmp_path: Path) -> Path:
    """Create a minimal SIR directory with CSV and NC files."""
    sir = tmp_path / "sir"
    sir.mkdir()
    # Static CSV
    df = pd.DataFrame(
        {"elevation_m_mean": [100.0, 200.0, 300.0]},
        index=pd.Index([1, 2, 3], name="nhm_id"),
    )
    df.to_csv(sir / "elevation_m_mean.csv")
    # Temporal NC
    ds = xr.Dataset({"pr": xr.DataArray(np.ones((3, 365)), dims=["nhm_id", "time"])})
    ds.to_netcdf(sir / "gridmet_2020.nc")
    return tmp_path


@pytest.fixture()
def sir_dir_with_manifest(sir_dir: Path) -> Path:
    """SIR directory with a valid manifest."""
    from hydro_param.manifest import PipelineManifest, SIRManifestEntry

    sir_entry = SIRManifestEntry(
        static_files={"elevation_m_mean": "sir/elevation_m_mean.csv"},
        temporal_files={"gridmet_2020": "sir/gridmet_2020.nc"},
        sir_schema=[{"name": "elevation_m_mean", "units": "m", "statistic": "mean"}],
    )
    manifest = PipelineManifest(sir=sir_entry)
    manifest.save(sir_dir)
    return sir_dir


class TestSIRAccessor:
    def test_from_manifest(self, sir_dir_with_manifest: Path) -> None:
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "elevation_m_mean" in acc.available_variables()
        assert "gridmet_2020" in acc.available_temporal()

    def test_load_variable(self, sir_dir_with_manifest: Path) -> None:
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        da = acc.load_variable("elevation_m_mean")
        assert isinstance(da, xr.DataArray)
        assert len(da) == 3
        assert float(da.values[0]) == 100.0

    def test_load_temporal(self, sir_dir_with_manifest: Path) -> None:
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        ds = acc.load_temporal("gridmet_2020")
        assert isinstance(ds, xr.Dataset)
        assert "pr" in ds

    def test_missing_variable_raises(self, sir_dir_with_manifest: Path) -> None:
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        with pytest.raises(KeyError, match="no_such_var"):
            acc.load_variable("no_such_var")

    def test_glob_fallback_no_manifest(self, sir_dir: Path) -> None:
        """Without manifest, SIRAccessor falls back to globbing sir/."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir)
        assert "elevation_m_mean" in acc.available_variables()
        assert "gridmet_2020" in acc.available_temporal()

    def test_contains_check(self, sir_dir_with_manifest: Path) -> None:
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "elevation_m_mean" in acc
        assert "no_such_var" not in acc

    def test_getitem(self, sir_dir_with_manifest: Path) -> None:
        """SIRAccessor[name] loads a variable (Dataset-compatible API)."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        da = acc["elevation_m_mean"]
        assert isinstance(da, xr.DataArray)

    def test_data_vars_property(self, sir_dir_with_manifest: Path) -> None:
        """data_vars returns available static variable names."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "elevation_m_mean" in acc.data_vars

    def test_missing_file_raises_at_init(self, tmp_path: Path) -> None:
        """If manifest references missing files, fail at init."""
        from hydro_param.manifest import PipelineManifest, SIRManifestEntry

        sir_entry = SIRManifestEntry(
            static_files={"ghost": "sir/ghost.csv"},
        )
        manifest = PipelineManifest(sir=sir_entry)
        manifest.save(tmp_path)
        from hydro_param.sir_accessor import SIRAccessor

        with pytest.raises(FileNotFoundError, match="ghost"):
            SIRAccessor(tmp_path)
