"""Tests for plugin protocols, DerivationContext, and registries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hydro_param.plugins import (
    DerivationContext,
    DerivationPlugin,
    FormatterPlugin,
    NetCDFFormatter,
    ParquetFormatter,
    get_derivation,
    get_formatter,
)
from hydro_param.sir_accessor import SIRAccessor


def _make_sir_accessor(
    tmp_path: Path,
    variables: dict[str, list[float]] | None = None,
    index_name: str = "nhm_id",
    index_values: list[int] | None = None,
) -> SIRAccessor:
    """Create a SIRAccessor from simple variable data for testing."""
    from hydro_param.manifest import PipelineManifest, SIRManifestEntry
    from hydro_param.sir_accessor import SIRAccessor

    sir_dir = tmp_path / "sir"
    sir_dir.mkdir(exist_ok=True)

    if variables is None:
        variables = {"val": [1.0, 2.0, 3.0]}
    if index_values is None:
        first = next(iter(variables.values()))
        index_values = list(range(1, len(first) + 1))

    idx = pd.Index(index_values, name=index_name)
    static_files = {}
    for name, values in variables.items():
        df = pd.DataFrame({name: values}, index=idx)
        df.to_csv(sir_dir / f"{name}.csv")
        static_files[name] = f"sir/{name}.csv"

    manifest = PipelineManifest(sir=SIRManifestEntry(static_files=static_files))
    manifest.save(tmp_path)
    return SIRAccessor(tmp_path)


class TestDerivationContext:
    """Tests for DerivationContext construction and validation."""

    def test_valid_context(self, tmp_path: Path) -> None:
        sir = _make_sir_accessor(tmp_path)
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        assert ctx.fabric_id_field == "nhm_id"
        assert ctx.fabric is None
        assert ctx.segments is None

    def test_sir_accessor_contains(self, tmp_path: Path) -> None:
        sir = _make_sir_accessor(tmp_path, variables={"elevation_m_mean": [100.0, 200.0]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        assert "elevation_m_mean" in ctx.sir
        assert "nonexistent" not in ctx.sir

    def test_resolved_lookup_tables_dir_override(self, tmp_path: Path) -> None:
        sir = _make_sir_accessor(tmp_path)
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", lookup_tables_dir=tmp_path)
        assert ctx.resolved_lookup_tables_dir == tmp_path

    def test_resolved_lookup_tables_dir_default(self, tmp_path: Path) -> None:
        sir = _make_sir_accessor(tmp_path)
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        result = ctx.resolved_lookup_tables_dir
        assert result.exists()
        assert (result / "nlcd_to_prms_cov_type.yml").exists()

    def test_frozen_immutable(self, tmp_path: Path) -> None:
        sir = _make_sir_accessor(tmp_path)
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

    def test_fabric_id_field_not_in_fabric_raises(self, tmp_path: Path) -> None:
        import geopandas as gpd
        from shapely.geometry import Polygon

        sir = _make_sir_accessor(tmp_path)
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
        sir = _make_sir_accessor(tmp_path)
        bad_dir = tmp_path / "nonexistent"
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", lookup_tables_dir=bad_dir)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _ = ctx.resolved_lookup_tables_dir
