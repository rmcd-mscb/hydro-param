"""Tests for plugin protocols, DerivationContext, and registries."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from hydro_param.plugins import (
    DerivationContext,
    DerivationPlugin,
    FormatterPlugin,
    NetCDFFormatter,
    ParquetFormatter,
    get_derivation,
    get_formatter,
)


class TestDerivationContext:
    """Tests for DerivationContext construction and validation."""

    def test_valid_context(self) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1, 2, 3]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        assert ctx.fabric_id_field == "nhm_id"
        assert ctx.fabric is None
        assert ctx.segments is None

    def test_missing_dim_raises(self) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1, 2, 3]})
        with pytest.raises(KeyError, match="wrong_field"):
            DerivationContext(sir=sir, fabric_id_field="wrong_field")

    def test_resolved_lookup_tables_dir_override(self, tmp_path: Path) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", lookup_tables_dir=tmp_path)
        assert ctx.resolved_lookup_tables_dir == tmp_path

    def test_resolved_lookup_tables_dir_default(self) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        result = ctx.resolved_lookup_tables_dir
        assert result.exists()
        assert (result / "nlcd_to_prms_cov_type.yml").exists()

    def test_frozen_immutable(self) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        with pytest.raises(AttributeError):
            ctx.fabric_id_field = "other"  # type: ignore[misc]


class TestProtocolSatisfaction:
    """Verify plugin implementations satisfy their protocols."""

    def test_pywatershed_derivation_satisfies_protocol(self) -> None:
        from hydro_param.derivations.pywatershed import PywatershedDerivation

        assert isinstance(PywatershedDerivation(), DerivationPlugin)

    def test_pywatershed_formatter_satisfies_protocol(self) -> None:
        from hydro_param.formatters.pywatershed import PywatershedFormatter

        assert isinstance(PywatershedFormatter(), FormatterPlugin)


class TestGetDerivation:
    """Tests for the get_derivation() factory."""

    def test_pywatershed(self) -> None:
        from hydro_param.derivations.pywatershed import PywatershedDerivation

        plugin = get_derivation("pywatershed")
        assert isinstance(plugin, PywatershedDerivation)

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown derivation plugin"):
            get_derivation("nextgen")


class TestGetFormatter:
    """Tests for the get_formatter() factory."""

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


class TestFabricIdFieldValidation:
    """Tests for fabric_id_field validation against fabric columns."""

    def test_fabric_id_field_not_in_fabric_raises(self) -> None:
        import geopandas as gpd
        from shapely.geometry import Polygon

        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"wrong_col": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        with pytest.raises(KeyError, match="not found in fabric columns"):
            DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")


class TestLookupTablesDirValidation:
    """Tests for lookup_tables_dir existence validation."""

    def test_lookup_tables_dir_nonexistent_raises(self, tmp_path: Path) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        bad_dir = tmp_path / "nonexistent"
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", lookup_tables_dir=bad_dir)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _ = ctx.resolved_lookup_tables_dir
