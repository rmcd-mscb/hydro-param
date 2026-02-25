"""Tests for plugin protocols, DerivationContext, and registries."""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr
from hydro_param.plugins import (
    DerivationContext,
    DerivationPlugin,
    FormatterPlugin,
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
