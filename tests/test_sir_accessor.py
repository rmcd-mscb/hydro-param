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

    def test_contains_check_static(self, sir_dir_with_manifest: Path) -> None:
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "elevation_m_mean" in acc
        assert "no_such_var" not in acc

    def test_contains_check_temporal(self, sir_dir_with_manifest: Path) -> None:
        """__contains__ includes temporal datasets."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "gridmet_2020" in acc

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

    def test_empty_sir_dir_raises(self, tmp_path: Path) -> None:
        """Glob fallback with no sir/ dir raises FileNotFoundError."""
        from hydro_param.sir_accessor import SIRAccessor

        with pytest.raises(FileNotFoundError, match="No SIR output files found"):
            SIRAccessor(tmp_path)

    def test_empty_sir_subdir_raises(self, tmp_path: Path) -> None:
        """Glob fallback with empty sir/ subdir raises FileNotFoundError."""
        (tmp_path / "sir").mkdir()
        from hydro_param.sir_accessor import SIRAccessor

        with pytest.raises(FileNotFoundError, match="No SIR output files found"):
            SIRAccessor(tmp_path)

    def test_corrupt_csv_raises(self, sir_dir_with_manifest: Path) -> None:
        """Corrupt CSV produces an actionable error (OSError or ValueError)."""
        from hydro_param.sir_accessor import SIRAccessor

        # Write binary garbage that pandas cannot parse
        csv_path = sir_dir_with_manifest / "sir" / "elevation_m_mean.csv"
        csv_path.write_bytes(b"\x00\x01\x02\x03\x04\x05")

        acc = SIRAccessor(sir_dir_with_manifest)
        with pytest.raises((OSError, ValueError), match="elevation_m_mean"):
            acc.load_variable("elevation_m_mean")

    def test_corrupt_netcdf_raises_oserror(self, sir_dir_with_manifest: Path) -> None:
        """Corrupt NetCDF produces an actionable OSError."""
        from hydro_param.sir_accessor import SIRAccessor

        # Corrupt the NC file
        nc_path = sir_dir_with_manifest / "sir" / "gridmet_2020.nc"
        nc_path.write_bytes(b"not a netcdf file")

        acc = SIRAccessor(sir_dir_with_manifest)
        with pytest.raises(OSError, match="gridmet_2020"):
            acc.load_temporal("gridmet_2020")

    def test_missing_temporal_key_raises(self, sir_dir_with_manifest: Path) -> None:
        """load_temporal raises KeyError for missing key."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        with pytest.raises(KeyError, match="nonexistent"):
            acc.load_temporal("nonexistent")

    def test_sir_schema_returns_copy(self, sir_dir_with_manifest: Path) -> None:
        """sir_schema returns a copy; mutating it does not affect internal state."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        schema = acc.sir_schema
        original_len = len(schema)
        schema.append({"name": "injected"})
        assert len(acc.sir_schema) == original_len


def test_load_dataset_returns_all_columns(tmp_path: Path) -> None:
    """load_dataset returns full xr.Dataset with all CSV columns."""
    import textwrap

    from hydro_param.sir_accessor import SIRAccessor

    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame(
        {"lndcov_frac_2021_11": [0.8, 0.1], "lndcov_frac_2021_41": [0.2, 0.9]},
        index=pd.Index([1, 2], name="nhm_id"),
    )
    df.to_csv(sir_dir / "lndcov_frac_2021.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            lndcov_frac_2021: sir/lndcov_frac_2021.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)
    sir = SIRAccessor(tmp_path)
    ds = sir.load_dataset("lndcov_frac_2021")
    assert isinstance(ds, xr.Dataset)
    assert "lndcov_frac_2021_11" in ds.data_vars
    assert "lndcov_frac_2021_41" in ds.data_vars
    assert len(ds.data_vars) == 2


def test_find_variable_exact_match(tmp_path: Path) -> None:
    """find_variable returns exact match when available."""
    import textwrap

    from hydro_param.sir_accessor import SIRAccessor

    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame({"val": [1.0]}, index=pd.Index([1], name="nhm_id"))
    df.to_csv(sir_dir / "elevation_m_mean.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            elevation_m_mean: sir/elevation_m_mean.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)
    sir = SIRAccessor(tmp_path)
    assert sir.find_variable("elevation_m_mean") == "elevation_m_mean"


def test_find_variable_year_suffix(tmp_path: Path) -> None:
    """find_variable matches year-suffixed variant."""
    import textwrap

    from hydro_param.sir_accessor import SIRAccessor

    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame({"val": [5.0]}, index=pd.Index([1], name="nhm_id"))
    df.to_csv(sir_dir / "fctimp_pct_mean_2021.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            fctimp_pct_mean_2021: sir/fctimp_pct_mean_2021.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)
    sir = SIRAccessor(tmp_path)
    assert sir.find_variable("fctimp_pct_mean") == "fctimp_pct_mean_2021"


def test_find_variable_picks_most_recent_year(tmp_path: Path) -> None:
    """find_variable returns the most recent year when multiple matches exist."""
    import textwrap

    from hydro_param.sir_accessor import SIRAccessor

    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame({"val": [5.0]}, index=pd.Index([1], name="nhm_id"))
    for year in [2019, 2021, 2020]:
        df.to_csv(sir_dir / f"fctimp_pct_mean_{year}.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            fctimp_pct_mean_2019: sir/fctimp_pct_mean_2019.csv
            fctimp_pct_mean_2020: sir/fctimp_pct_mean_2020.csv
            fctimp_pct_mean_2021: sir/fctimp_pct_mean_2021.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)
    sir = SIRAccessor(tmp_path)
    assert sir.find_variable("fctimp_pct_mean") == "fctimp_pct_mean_2021"


def test_find_variable_not_found(tmp_path: Path) -> None:
    """find_variable returns None when no match exists."""
    import textwrap

    from hydro_param.sir_accessor import SIRAccessor

    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame({"val": [1.0]}, index=pd.Index([1], name="nhm_id"))
    df.to_csv(sir_dir / "elevation_m_mean.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            elevation_m_mean: sir/elevation_m_mean.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)
    sir = SIRAccessor(tmp_path)
    assert sir.find_variable("bogus_var") is None
