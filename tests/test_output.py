"""Tests for output formatter protocol and factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydro_param.output import NetCDFFormatter, ParquetFormatter, get_formatter


def _has_parquet_engine() -> bool:
    """Check if a parquet engine (pyarrow or fastparquet) is available."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        pass
    try:
        import fastparquet  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture()
def sample_dataset() -> xr.Dataset:
    """Minimal xr.Dataset for testing output formatters."""
    return xr.Dataset(
        {"elevation": ("hru_id", np.array([100.0, 200.0, 300.0]))},
        coords={"hru_id": [1, 2, 3]},
        attrs={"title": "test"},
    )


class TestGetFormatter:
    """Tests for the get_formatter() factory function."""

    def test_netcdf(self) -> None:
        fmt = get_formatter("netcdf")
        assert fmt.name == "netcdf"
        assert isinstance(fmt, NetCDFFormatter)

    def test_parquet(self) -> None:
        fmt = get_formatter("parquet")
        assert fmt.name == "parquet"
        assert isinstance(fmt, ParquetFormatter)

    def test_pywatershed(self) -> None:
        from hydro_param.formatters.pywatershed import PywatershedFormatter

        fmt = get_formatter("pywatershed")
        assert fmt.name == "pywatershed"
        assert isinstance(fmt, PywatershedFormatter)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown output formatter"):
            get_formatter("prms_v3_text")


class TestNetCDFFormatter:
    """Tests for NetCDFFormatter."""

    def test_write(self, tmp_path: Path, sample_dataset: xr.Dataset) -> None:
        fmt = NetCDFFormatter()
        paths = fmt.write(sample_dataset, tmp_path, {"sir_name": "test_output"})
        assert len(paths) == 1
        assert paths[0].name == "test_output.nc"
        assert paths[0].exists()

        # Verify content
        ds = xr.open_dataset(paths[0])
        assert "elevation" in ds.data_vars
        ds.close()

    def test_validate_empty(self, sample_dataset: xr.Dataset) -> None:
        fmt = NetCDFFormatter()
        assert fmt.validate(sample_dataset) == []


class TestParquetFormatter:
    """Tests for ParquetFormatter."""

    @pytest.mark.skipif(not _has_parquet_engine(), reason="pyarrow or fastparquet not installed")
    def test_write(self, tmp_path: Path, sample_dataset: xr.Dataset) -> None:
        fmt = ParquetFormatter()
        paths = fmt.write(sample_dataset, tmp_path, {"sir_name": "test_output"})
        assert len(paths) == 1
        assert paths[0].name == "test_output.parquet"
        assert paths[0].exists()

    def test_validate_empty(self, sample_dataset: xr.Dataset) -> None:
        fmt = ParquetFormatter()
        assert fmt.validate(sample_dataset) == []
