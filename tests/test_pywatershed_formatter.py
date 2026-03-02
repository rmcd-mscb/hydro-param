"""Tests for pywatershed output formatter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from hydro_param.formatters.pywatershed import PywatershedFormatter


@pytest.fixture()
def formatter() -> PywatershedFormatter:
    """Formatter with bundled metadata."""
    return PywatershedFormatter()


@pytest.fixture()
def param_dataset() -> xr.Dataset:
    """Synthetic pywatershed parameter dataset (already in PRMS units)."""
    return xr.Dataset(
        {
            "hru_elev": ("nhru", np.array([500.0, 1000.0, 2000.0])),
            "hru_slope": ("nhru", np.array([0.05, 0.15, 0.50])),
            "hru_aspect": ("nhru", np.array([0.0, 90.0, 270.0])),
            "hru_lat": ("nhru", np.array([42.0, 41.5, 43.0])),
            "hru_area": ("nhru", np.array([1000.0, 2000.0, 500.0])),
            "cov_type": ("nhru", np.array([4, 1, 2])),
            "covden_sum": ("nhru", np.array([0.8, 0.3, 0.4])),
            "covden_win": ("nhru", np.array([0.8, 0.15, 0.2])),
            "hru_percent_imperv": ("nhru", np.array([0.05, 0.20, 0.0])),
            "srain_intcp": ("nhru", np.array([0.08, 0.05, 0.05])),
            "wrain_intcp": ("nhru", np.array([0.08, 0.05, 0.05])),
            "snow_intcp": ("nhru", np.array([0.06, 0.01, 0.02])),
            "imperv_stor_max": ("nhru", np.array([0.03, 0.03, 0.03])),
            "tmax_allsnow": np.float64(32.0),
            "den_init": np.float64(0.10),
            "gwstor_init": np.float64(2.0),
        },
        coords={"nhru": [1, 2, 3]},
    )


@pytest.fixture()
def forcing_dataset() -> xr.Dataset:
    """Synthetic forcing time series (metric source units)."""
    nhru = 3
    ntime = 5
    return xr.Dataset(
        {
            "prcp": (["time", "nhru"], np.full((ntime, nhru), 25.4)),  # 25.4 mm = 1 inch
            "tmax": (["time", "nhru"], np.full((ntime, nhru), 0.0)),  # 0°C = 32°F
            "tmin": (["time", "nhru"], np.full((ntime, nhru), -10.0)),  # -10°C = 14°F
        },
        coords={
            "time": np.arange(ntime),
            "nhru": [1, 2, 3],
        },
    )


class TestWriteParameters:
    """Tests for write_parameters()."""

    def test_creates_file(
        self, formatter: PywatershedFormatter, param_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        out = tmp_path / "parameters.nc"
        formatter.write_parameters(param_dataset, out)
        assert out.exists()

    def test_contains_static_params(
        self, formatter: PywatershedFormatter, param_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        out = tmp_path / "parameters.nc"
        formatter.write_parameters(param_dataset, out)
        ds = xr.open_dataset(out)
        assert "hru_elev" in ds
        assert "cov_type" in ds
        assert "tmax_allsnow" in ds
        ds.close()

    def test_excludes_forcing_vars(self, formatter: PywatershedFormatter, tmp_path: Path) -> None:
        """Forcing variables (prcp, tmax, tmin) are excluded from params file."""
        ds = xr.Dataset(
            {
                "hru_elev": ("nhru", np.array([100.0])),
                "prcp": (["time", "nhru"], np.array([[1.0]])),
                "tmax": (["time", "nhru"], np.array([[20.0]])),
            },
        )
        out = tmp_path / "parameters.nc"
        formatter.write_parameters(ds, out)
        result = xr.open_dataset(out)
        assert "hru_elev" in result
        assert "prcp" not in result
        assert "tmax" not in result
        result.close()

    def test_cf_attributes(
        self, formatter: PywatershedFormatter, param_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        out = tmp_path / "parameters.nc"
        formatter.write_parameters(param_dataset, out)
        ds = xr.open_dataset(out)
        assert ds.attrs["Conventions"] == "CF-1.8"
        assert "hydro-param" in ds.attrs["source"]
        ds.close()


class TestWriteForcingNetcdf:
    """Tests for write_forcing_netcdf()."""

    def test_creates_files(
        self, formatter: PywatershedFormatter, forcing_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        paths = formatter.write_forcing_netcdf(forcing_dataset, tmp_path / "forcing")
        assert len(paths) == 3
        assert all(p.exists() for p in paths)

    def test_prcp_mm_to_inches(
        self, formatter: PywatershedFormatter, forcing_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        paths = formatter.write_forcing_netcdf(forcing_dataset, tmp_path / "forcing")
        prcp_path = [p for p in paths if p.name == "prcp.nc"][0]
        ds = xr.open_dataset(prcp_path)
        # 25.4 mm → 1.0 inch
        np.testing.assert_allclose(ds["prcp"].values, 1.0)
        assert ds["prcp"].attrs["units"] == "in"
        ds.close()

    def test_tmax_c_to_f(
        self, formatter: PywatershedFormatter, forcing_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        paths = formatter.write_forcing_netcdf(forcing_dataset, tmp_path / "forcing")
        tmax_path = [p for p in paths if p.name == "tmax.nc"][0]
        ds = xr.open_dataset(tmax_path)
        # 0°C → 32°F
        np.testing.assert_allclose(ds["tmax"].values, 32.0)
        assert ds["tmax"].attrs["units"] == "F"
        ds.close()

    def test_tmin_c_to_f(
        self, formatter: PywatershedFormatter, forcing_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        paths = formatter.write_forcing_netcdf(forcing_dataset, tmp_path / "forcing")
        tmin_path = [p for p in paths if p.name == "tmin.nc"][0]
        ds = xr.open_dataset(tmin_path)
        # -10°C → 14°F
        np.testing.assert_allclose(ds["tmin"].values, 14.0)
        ds.close()

    def test_forcing_contains_nhm_id(
        self, formatter: PywatershedFormatter, forcing_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        """Forcing NetCDF must include nhm_id for pywatershed NetCdfRead."""
        paths = formatter.write_forcing_netcdf(forcing_dataset, tmp_path / "forcing")
        for path in paths:
            ds = xr.open_dataset(path)
            assert "nhm_id" in ds, f"nhm_id missing from {path.name}"
            np.testing.assert_array_equal(ds["nhm_id"].values, [1, 2, 3])
            ds.close()

    def test_skips_when_no_forcing(
        self, formatter: PywatershedFormatter, param_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        """No forcing vars → no forcing files, no error."""
        paths = formatter.write_forcing_netcdf(param_dataset, tmp_path / "forcing")
        assert paths == []


class TestWriteSoltab:
    """Tests for write_soltab()."""

    def test_creates_file(self, formatter: PywatershedFormatter, tmp_path: Path) -> None:
        ds = xr.Dataset(
            {
                "soltab_potsw": (["nhru", "doy"], np.ones((3, 366))),
                "soltab_horad_potsw": (["nhru", "doy"], np.ones((3, 366)) * 0.5),
            },
        )
        out = tmp_path / "soltab.nc"
        formatter.write_soltab(ds, out)
        assert out.exists()
        result = xr.open_dataset(out)
        assert "soltab_potsw" in result
        assert "soltab_horad_potsw" in result
        result.close()

    def test_skips_when_no_soltab(
        self, formatter: PywatershedFormatter, param_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        out = tmp_path / "soltab.nc"
        formatter.write_soltab(param_dataset, out)
        assert not out.exists()


class TestWriteControl:
    """Tests for write_control()."""

    def test_creates_file(self, formatter: PywatershedFormatter, tmp_path: Path) -> None:
        config = {"start": "1980-10-01", "end": "2020-09-30"}
        out = tmp_path / "control.yml"
        formatter.write_control(config, out)
        assert out.exists()

    def test_content(self, formatter: PywatershedFormatter, tmp_path: Path) -> None:
        config = {"start": "1980-10-01", "end": "2020-09-30"}
        out = tmp_path / "control.yml"
        formatter.write_control(config, out)
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["start_time"] == "1980-10-01"
        assert data["end_time"] == "2020-09-30"
        assert data["time_step"] == "24:00:00"

    def test_raises_on_missing_start_end(
        self, formatter: PywatershedFormatter, tmp_path: Path
    ) -> None:
        out = tmp_path / "control.yml"
        with pytest.raises(ValueError, match="'start' and 'end'"):
            formatter.write_control({}, out)


class TestWrite:
    """Tests for the top-level write() method."""

    def test_creates_all_files(
        self, formatter: PywatershedFormatter, param_dataset: xr.Dataset, tmp_path: Path
    ) -> None:
        config = {
            "parameter_file": "parameters.nc",
            "forcing_dir": "forcing",
            "control_file": "control.yml",
            "start": "1980-10-01",
            "end": "2020-09-30",
        }
        paths = formatter.write(param_dataset, tmp_path, config)
        # Should have at least params + control (no forcing/soltab in this dataset)
        assert len(paths) >= 2
        names = [p.name for p in paths]
        assert "parameters.nc" in names
        assert "control.yml" in names


class TestValidate:
    """Tests for validate()."""

    def test_complete_dataset_no_warnings(
        self, formatter: PywatershedFormatter, param_dataset: xr.Dataset
    ) -> None:
        warnings = formatter.validate(param_dataset)
        # No range violations should exist for our valid fixture values
        range_warnings = [w for w in warnings if "below" in w or "above" in w]
        assert range_warnings == []

    def test_detects_missing_required(self, formatter: PywatershedFormatter) -> None:
        empty_ds = xr.Dataset()
        warnings = formatter.validate(empty_ds)
        # Should flag missing required parameters
        assert any("missing" in w.lower() for w in warnings)

    def test_detects_out_of_range(self, formatter: PywatershedFormatter) -> None:
        ds = xr.Dataset(
            {
                "hru_slope": ("nhru", np.array([-1.0, 0.5])),  # negative slope invalid
            },
        )
        warnings = formatter.validate(ds)
        assert any("hru_slope" in w and "below" in w for w in warnings)

    def test_no_metadata_returns_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Formatter returns warning when metadata YAML is missing."""
        monkeypatch.setattr(
            PywatershedFormatter,
            "_default_metadata_path",
            staticmethod(lambda: Path("/nonexistent/metadata.yml")),
        )
        fmt = PywatershedFormatter()
        ds = xr.Dataset({"hru_elev": ("nhru", np.array([100.0]))})
        warnings = fmt.validate(ds)
        assert len(warnings) == 1
        assert "unavailable" in warnings[0].lower()

    def test_formatter_no_init_args(self) -> None:
        """Formatter can be constructed with no arguments."""
        fmt = PywatershedFormatter()
        assert fmt.name == "pywatershed"
