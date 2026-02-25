# pywatershed Plugin Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the pywatershed plugin system to use formal Protocol contracts, a typed DerivationContext, and a unified plugin registry in `plugins.py`.

**Architecture:** New `plugins.py` module defines `DerivationPlugin` and `FormatterPlugin` protocols plus `DerivationContext` dataclass. Existing derivation/formatter implementations are refactored to implement these protocols. `output.py` is retired — its contents migrate to `plugins.py`. Lookup tables and parameter metadata move into the Python package under `src/hydro_param/data/` for `importlib.resources` access.

**Tech Stack:** Python typing.Protocol, dataclasses, importlib.resources, xarray, geopandas, hatchling (package data)

**Design doc:** `docs/plans/2026-02-25-pywatershed-plugin-architecture-design.md`

---

### Task 1: Create `plugins.py` with protocols and DerivationContext

**Files:**
- Create: `src/hydro_param/plugins.py`
- Create: `tests/test_plugins.py`

**Step 1: Write the failing tests**

```python
# tests/test_plugins.py
"""Tests for plugin protocols, DerivationContext, and registries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
        ctx = DerivationContext(
            sir=sir, fabric_id_field="nhm_id", lookup_tables_dir=tmp_path
        )
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
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev python -m pytest tests/test_plugins.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydro_param.plugins'`

**Step 3: Write `plugins.py`**

```python
# src/hydro_param/plugins.py
"""Plugin protocols, context, and registries.

Defines the contracts that all model plugins (derivation + formatter) must
satisfy, the typed ``DerivationContext`` input bundle, and factory functions
for plugin discovery.

This module is the single source of truth for "what is a plugin?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Protocol, runtime_checkable

import geopandas as gpd
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DerivationContext — typed input bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DerivationContext:
    """Everything a derivation plugin needs.

    Bundles SIR output, target fabric geometry, and configuration into
    a single immutable object.  Validates that ``fabric_id_field`` exists
    as a dimension in the SIR on construction.

    Parameters
    ----------
    sir
        Normalized Standardized Internal Representation (SIR) dataset.
    fabric
        Target HRU polygon GeoDataFrame.
    segments
        Stream segment line GeoDataFrame.
    fabric_id_field
        Column name for HRU identifiers in the fabric.  Must exist as a
        dimension in ``sir``.
    segment_id_field
        Column name for segment identifiers in the segments GeoDataFrame.
    config
        Plugin-specific configuration dict.
    lookup_tables_dir
        Override path to lookup table YAML files.  When ``None``, defaults
        to the package-bundled tables via ``importlib.resources``.
    """

    sir: xr.Dataset
    fabric: gpd.GeoDataFrame | None = None
    segments: gpd.GeoDataFrame | None = None
    fabric_id_field: str = "nhm_id"
    segment_id_field: str | None = None
    config: dict = field(default_factory=dict)
    lookup_tables_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.fabric_id_field not in self.sir.dims:
            raise KeyError(
                f"Expected dimension '{self.fabric_id_field}' not found in SIR. "
                f"Available dims: {list(self.sir.dims)}"
            )

    @property
    def resolved_lookup_tables_dir(self) -> Path:
        """Resolve the lookup tables directory.

        Returns the explicit override if set, otherwise the
        package-bundled default via ``importlib.resources``.
        """
        if self.lookup_tables_dir is not None:
            return self.lookup_tables_dir
        return Path(str(files("hydro_param").joinpath("data/lookup_tables")))


# ---------------------------------------------------------------------------
# Plugin protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class DerivationPlugin(Protocol):
    """Contract for model-specific parameter derivation.

    Implementations transform a normalized SIR dataset into
    model-specific parameters (e.g., PRMS units, variable names,
    lookup-table reclassification).
    """

    name: str

    def derive(self, context: DerivationContext) -> xr.Dataset:
        """Derive model parameters from the SIR.

        Parameters
        ----------
        context
            Typed input bundle containing SIR, fabric, config, etc.

        Returns
        -------
        xr.Dataset
            Model-specific parameter dataset.
        """
        ...


@runtime_checkable
class FormatterPlugin(Protocol):
    """Contract for model-specific output formatting.

    Implementations write derived parameters to the file format(s)
    expected by the target model.
    """

    name: str

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """Validate parameters before writing.

        Returns
        -------
        list[str]
            Validation warnings.  Empty if all checks pass.
        """
        ...

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write model-specific output files.

        Parameters
        ----------
        parameters
            Derived model parameters.
        output_path
            Output directory.
        config
            Formatter-specific configuration options.

        Returns
        -------
        list[Path]
            Paths to all files written.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in formatters (migrated from output.py)
# ---------------------------------------------------------------------------


class NetCDFFormatter:
    """Simple NetCDF output (wraps SIR → NetCDF)."""

    name: str = "netcdf"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.nc"
        parameters.to_netcdf(out_file)
        logger.info("Wrote NetCDF: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        return []


class ParquetFormatter:
    """Simple Parquet output (wraps SIR → Parquet)."""

    name: str = "parquet"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.parquet"
        parameters.to_dataframe().to_parquet(out_file)
        logger.info("Wrote Parquet: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        return []


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------


def get_derivation(name: str) -> DerivationPlugin:
    """Select a derivation plugin by name.

    Parameters
    ----------
    name
        Plugin name: ``"pywatershed"``.

    Raises
    ------
    ValueError
        If the name is not recognized.
    """
    if name == "pywatershed":
        from hydro_param.derivations.pywatershed import PywatershedDerivation

        return PywatershedDerivation()

    available = "pywatershed"
    raise ValueError(f"Unknown derivation plugin '{name}'. Available: {available}")


def get_formatter(name: str) -> FormatterPlugin:
    """Select an output formatter by name.

    Parameters
    ----------
    name
        Formatter name: ``"netcdf"``, ``"parquet"``, or ``"pywatershed"``.

    Raises
    ------
    ValueError
        If the name is not recognized.
    """
    if name == "netcdf":
        return NetCDFFormatter()
    if name == "parquet":
        return ParquetFormatter()
    if name == "pywatershed":
        from hydro_param.formatters.pywatershed import PywatershedFormatter

        return PywatershedFormatter()

    available = "netcdf, parquet, pywatershed"
    raise ValueError(f"Unknown output formatter '{name}'. Available: {available}")
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_plugins.py -v`
Expected: 5 pass (context tests), 2 FAIL (protocol satisfaction — derivation/formatter not refactored yet)

Note: The protocol satisfaction tests will pass only after Tasks 2 and 4 are complete. For now, confirm the 5 context tests pass.

**Step 5: Commit**

```bash
git add src/hydro_param/plugins.py tests/test_plugins.py
git commit -m "feat: add plugins.py with DerivationContext, protocols, and registries"
```

---

### Task 2: Move lookup tables and metadata into package

**Files:**
- Create: `src/hydro_param/data/lookup_tables/` (copy 4 YAML files)
- Create: `src/hydro_param/data/pywatershed/` (copy 1 YAML file)
- Create: `src/hydro_param/data/__init__.py` (empty)
- Modify: `pyproject.toml` (add package data inclusion)

**Step 1: Copy data files into the package**

```bash
mkdir -p src/hydro_param/data/lookup_tables
mkdir -p src/hydro_param/data/pywatershed
cp configs/lookup_tables/*.yml src/hydro_param/data/lookup_tables/
cp configs/pywatershed/parameter_metadata.yml src/hydro_param/data/pywatershed/
touch src/hydro_param/data/__init__.py
```

**Step 2: Add package data to `pyproject.toml`**

Add under `[build-system]` section or as a hatchling-specific config. Since the project uses hatchling, add:

```toml
[tool.hatch.build.targets.wheel]
packages = ["src/hydro_param"]

[tool.hatch.build.targets.wheel.force-include]
"src/hydro_param/data" = "hydro_param/data"
```

If a `[tool.hatch.build]` section already exists, just add the force-include. The key point: hatchling must include `src/hydro_param/data/**/*.yml` in the wheel.

**Step 3: Verify package data is accessible**

Run: `pixi run -e dev python -c "from importlib.resources import files; p = files('hydro_param').joinpath('data/lookup_tables'); print(p); assert (p / 'nlcd_to_prms_cov_type.yml').is_file()"`
Expected: Prints path, no assertion error.

**Step 4: Run DerivationContext default path test**

Run: `pixi run -e dev python -m pytest tests/test_plugins.py::TestDerivationContext::test_resolved_lookup_tables_dir_default -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/data/ pyproject.toml
git commit -m "feat: move lookup tables and metadata into package for importlib.resources"
```

---

### Task 3: Add registry tests to `test_plugins.py`

**Files:**
- Modify: `tests/test_plugins.py`

**Step 1: Add registry tests**

Append to `tests/test_plugins.py`:

```python
from hydro_param.plugins import get_derivation, get_formatter, NetCDFFormatter, ParquetFormatter


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
```

**Step 2: Run tests**

Run: `pixi run -e dev python -m pytest tests/test_plugins.py -v`
Expected: All registry tests PASS, protocol satisfaction tests may still fail.

**Step 3: Commit**

```bash
git add tests/test_plugins.py
git commit -m "test: add registry factory tests for get_derivation and get_formatter"
```

---

### Task 4: Refactor `PywatershedDerivation` to accept `DerivationContext`

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Modify: `tests/test_pywatershed_derivation.py`

This is the largest task. The derivation math is **unchanged** — only the interface is refactored.

**Step 1: Refactor `derivations/pywatershed.py`**

Key changes to `src/hydro_param/derivations/pywatershed.py`:

1. Delete module-level `_sir_id_dim()` function (lines 30-42). Its logic is now in `DerivationContext.__post_init__`.

2. Delete `rename_sir_variables()` method (lines 166-196). Deprecated, SIR normalization handles this.

3. Refactor `derive()` method signature — replace 6 args with `DerivationContext`:

```python
from hydro_param.plugins import DerivationContext

class PywatershedDerivation:
    name: str = "pywatershed"

    def __init__(self) -> None:
        self._lookup_cache: dict[str, dict] = {}

    def derive(self, context: DerivationContext) -> xr.Dataset:
        sir = context.sir
        id_field = context.fabric_id_field
        nhru = sir.sizes[id_field]
        ds = xr.Dataset()

        if id_field in sir.coords:
            ds = ds.assign_coords(nhru=sir[id_field].values)

        ds = self._derive_geometry(context, ds)
        ds = self._derive_topology(context, ds)
        ds = self._derive_topography(context, ds)
        ds = self._derive_landcover(context, ds)
        ds = self._apply_lookup_tables(context, ds)
        ds = self._apply_defaults(ds, nhru)
        ds = self._apply_overrides(ds, context.config)
        return ds
```

4. Refactor each step method to take `(self, ctx: DerivationContext, ds: xr.Dataset)`:

   - `_derive_geometry(self, ctx, ds)` — use `ctx.sir`, `ctx.fabric`, `ctx.fabric_id_field`
   - `_derive_topology(self, ctx, ds)` — guard clause if `ctx.fabric is None or ctx.segments is None`, use `ctx.fabric_id_field`, `ctx.segment_id_field`
   - `_derive_topography(self, ctx, ds)` — use `ctx.sir`
   - `_derive_landcover(self, ctx, ds)` — use `ctx.sir`
   - `_apply_lookup_tables(self, ctx, ds)` — use `ctx.resolved_lookup_tables_dir`
   - `_apply_defaults(self, ds, nhru)` — unchanged (no context needed)
   - `_apply_overrides(self, ds, config)` — unchanged (takes config dict)

5. Refactor `_load_lookup_table()` to accept path:

```python
def _load_lookup_table(self, name: str, tables_dir: Path) -> dict:
    if name not in self._lookup_cache:
        path = tables_dir / f"{name}.yml"
        with open(path) as f:
            self._lookup_cache[name] = yaml.safe_load(f)
    return self._lookup_cache[name]
```

6. Refactor `merge_temporal_into_derived()` — replace hardcoded `"nhru"` with `id_field` parameter:

```python
def merge_temporal_into_derived(
    derived: xr.Dataset,
    temporal: dict[str, xr.Dataset],
    renames: dict[str, str] | None = None,
    conversions: dict[str, tuple[str, str]] | None = None,
    id_field: str = "nhru",
) -> xr.Dataset:
```

Update the hardcoded `"nhru"` check on line 140 to use the `id_field` parameter:
```python
if feat_dims and id_field in derived.dims and feat_dims[0] != id_field:
    da = da.rename({feat_dims[0]: id_field})
```

7. Add `name` class attribute: `name: str = "pywatershed"`

**Step 2: Refactor `tests/test_pywatershed_derivation.py`**

All test calls to `derivation.derive(sir, ...)` become `derivation.derive(ctx)` where `ctx = DerivationContext(...)`.

Key pattern — update the `derivation` fixture:

```python
from hydro_param.plugins import DerivationContext

# Remove old LOOKUP_TABLES_DIR constant since DerivationContext handles it

@pytest.fixture()
def derivation() -> PywatershedDerivation:
    """Derivation plugin instance."""
    return PywatershedDerivation()
```

Update each test class. Example pattern for `TestDeriveTopography`:

```python
# Before:
ds = derivation.derive(sir_topography)

# After:
ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
ds = derivation.derive(ctx)
```

For topology tests that pass fabric/segments:

```python
# Before:
ds = derivation.derive(sir_minimal, fabric=synthetic_fabric, segments=synthetic_segments)

# After:
ctx = DerivationContext(
    sir=sir_minimal,
    fabric=synthetic_fabric,
    segments=synthetic_segments,
    fabric_id_field="nhm_id",
    segment_id_field="nhm_seg",
)
ds = derivation.derive(ctx)
```

For tests that pass `id_field="nhm_id"` explicitly:

```python
# Before:
ds = derivation.derive(sir, fabric=fabric, id_field="nhm_id")

# After:
ctx = DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")
ds = derivation.derive(ctx)
```

For tests that pass `config=`:

```python
# Before:
config = {"parameter_overrides": {"values": {"tmax_allsnow": 30.0}}}
ds = derivation.derive(sir_topography, config=config)

# After:
ctx = DerivationContext(
    sir=sir_topography, fabric_id_field="nhm_id", config=config
)
ds = derivation.derive(ctx)
```

Delete `TestSirVariableRenaming` class entirely (lines 758-784). The method is removed.

Update `TestMergeTemporalIntoDerived` — no changes needed unless the function signature changes to require `id_field`. Since we're adding `id_field` as a keyword argument with default `"nhru"`, existing tests continue to work.

**Step 3: Run tests**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py tests/test_plugins.py -v`
Expected: ALL PASS (including protocol satisfaction for derivation)

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "refactor: PywatershedDerivation accepts DerivationContext, remove deprecated methods"
```

---

### Task 5: Refactor `PywatershedFormatter` for protocol compliance

**Files:**
- Modify: `src/hydro_param/formatters/pywatershed.py`
- Modify: `tests/test_pywatershed_formatter.py`

**Step 1: Refactor `formatters/pywatershed.py`**

Changes are minimal:

1. Remove `__init__` parameter `metadata_path` — resolve via `importlib.resources` or `config["metadata_path"]` in `write()`:

```python
from importlib.resources import files

class PywatershedFormatter:
    name: str = "pywatershed"

    def __init__(self) -> None:
        self._metadata_cache: dict | None = None

    @staticmethod
    def _default_metadata_path() -> Path:
        return Path(str(files("hydro_param").joinpath("data/pywatershed/parameter_metadata.yml")))
```

2. Update `_load_metadata()` to accept optional path override:

```python
def _load_metadata(self, metadata_path: Path | None = None) -> dict | None:
    if self._metadata_cache is not None:
        return self._metadata_cache
    path = metadata_path or self._default_metadata_path()
    try:
        with open(path) as f:
            self._metadata_cache = yaml.safe_load(f)
        return self._metadata_cache
    except FileNotFoundError:
        logger.warning("Parameter metadata not found: %s", path)
        return None
```

3. Update `validate()` to return a warning when metadata is unavailable:

```python
def validate(self, parameters: xr.Dataset) -> list[str]:
    warnings: list[str] = []
    metadata = self._load_metadata()
    if metadata is None:
        return ["Parameter metadata unavailable — validation skipped"]
    # ... rest unchanged
```

4. Update `write()` to pass `config.get("metadata_path")` through to validate/load:

```python
def write(self, parameters, output_path, config):
    # ...
    metadata_path = config.get("metadata_path")
    warnings = self.validate(parameters, metadata_path=metadata_path)
    # ...
```

Actually — this changes the `FormatterPlugin.validate()` protocol signature. Keep `validate(self, parameters)` clean and let `_load_metadata` use the default path. If users need a custom metadata path, they pass it via `config` to `write()`, and `write()` handles it internally. `validate()` stays protocol-compliant.

Simpler approach: `validate()` signature unchanged. Metadata path resolution is internal to the class.

**Step 2: Update formatter tests**

In `tests/test_pywatershed_formatter.py`, find the test that checks validate with missing metadata and update the assertion:

```python
# Before (if exists):
assert formatter.validate(params) == []

# After:
warnings = formatter.validate(params)
assert any("unavailable" in w for w in warnings)
```

Also verify that the formatter no longer requires `metadata_path` in `__init__`:

```python
def test_formatter_no_init_args(self) -> None:
    fmt = PywatershedFormatter()
    assert fmt.name == "pywatershed"
```

**Step 3: Run tests**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_formatter.py tests/test_plugins.py -v`
Expected: ALL PASS (including protocol satisfaction for formatter)

**Step 4: Commit**

```bash
git add src/hydro_param/formatters/pywatershed.py tests/test_pywatershed_formatter.py
git commit -m "refactor: PywatershedFormatter uses importlib.resources, validate warns on missing metadata"
```

---

### Task 6: Retire `output.py` and update imports

**Files:**
- Delete: `src/hydro_param/output.py`
- Modify: `src/hydro_param/__init__.py`
- Modify: `src/hydro_param/cli.py` (line 552)
- Modify: `tests/test_output.py` → update imports to `plugins`

**Step 1: Update `src/hydro_param/__init__.py`**

Replace:
```python
from hydro_param.output import OutputFormatter, get_formatter
```

With:
```python
from hydro_param.plugins import DerivationContext, DerivationPlugin, FormatterPlugin, get_derivation, get_formatter
```

Update `__all__`:
```python
__all__ = [
    "DatasetRegistry",
    "DerivationContext",
    "DerivationPlugin",
    "FormatterPlugin",
    "PipelineConfig",
    "PywatershedRunConfig",
    "get_derivation",
    "get_formatter",
    "load_config",
    "load_pywatershed_config",
    "load_registry",
    "run_pipeline",
]
```

**Step 2: Update `src/hydro_param/cli.py`**

At line 552, replace:
```python
from hydro_param.output import get_formatter
```

With:
```python
from hydro_param.plugins import DerivationContext, get_formatter
```

At lines 580-599, replace the derivation call:
```python
# Before:
plugin = PywatershedDerivation()
# ...
derived = plugin.derive(
    sir,
    config=derivation_config,
    fabric=result.fabric,
    segments=segments,
    id_field=pws_config.domain.id_field,
    segment_id_field=pws_config.domain.segment_id_field,
)

# After:
from hydro_param.derivations.pywatershed import PywatershedDerivation

plugin = PywatershedDerivation()
ctx = DerivationContext(
    sir=sir,
    fabric=result.fabric,
    segments=segments,
    fabric_id_field=pws_config.domain.id_field,
    segment_id_field=pws_config.domain.segment_id_field,
    config=derivation_config,
)
derived = plugin.derive(ctx)
```

**Step 3: Update `tests/test_output.py` imports**

Replace:
```python
from hydro_param.output import NetCDFFormatter, ParquetFormatter, get_formatter
```

With:
```python
from hydro_param.plugins import NetCDFFormatter, ParquetFormatter, get_formatter
```

All test logic stays identical — only the import path changes.

**Step 4: Delete `src/hydro_param/output.py`**

```bash
git rm src/hydro_param/output.py
```

**Step 5: Run all tests**

Run: `pixi run -e dev python -m pytest tests/test_output.py tests/test_plugins.py tests/test_pywatershed_derivation.py tests/test_pywatershed_formatter.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/hydro_param/__init__.py src/hydro_param/cli.py tests/test_output.py
git rm src/hydro_param/output.py
git commit -m "refactor: retire output.py, migrate to plugins.py"
```

---

### Task 7: Run full test suite and pre-push checks

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `pixi run -e dev test`
Expected: ALL tests pass (460+ tests)

**Step 2: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass (ruff, mypy, detect-secrets)

**Step 3: Run full check suite**

Run: `pixi run -e dev check`
Expected: lint, format-check, typecheck, and tests all pass

**Step 4: Fix any issues found**

If mypy complains about Protocol structural typing, add explicit type annotations. If ruff finds formatting issues, they should auto-fix. If tests fail, trace back to the specific refactoring step.

**Step 5: Final commit if fixes were needed**

```bash
git add -u
git commit -m "fix: address linting/typing issues from plugin refactoring"
```

---

### Task 8: Update `__init__.py` re-exports in derivations and formatters

**Files:**
- Modify: `src/hydro_param/derivations/__init__.py`
- Modify: `src/hydro_param/formatters/__init__.py`

**Step 1: Update derivations `__init__.py`**

Add `DerivationContext` re-export for convenience:

```python
"""Model-specific parameter derivation plugins."""

from hydro_param.derivations.pywatershed import PywatershedDerivation
from hydro_param.plugins import DerivationContext

__all__ = ["DerivationContext", "PywatershedDerivation"]
```

**Step 2: Update formatters `__init__.py`**

No changes needed — `PywatershedFormatter` re-export is sufficient.

**Step 3: Run tests**

Run: `pixi run -e dev python -m pytest tests/test_plugins.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/__init__.py
git commit -m "refactor: re-export DerivationContext from derivations package"
```

---

## Task Dependency Graph

```
Task 1 (plugins.py) ──┬──→ Task 3 (registry tests)
                       │
Task 2 (package data) ─┤
                       │
                       ├──→ Task 4 (refactor derivation) ──→ Task 6 (retire output.py)
                       │                                            │
                       └──→ Task 5 (refactor formatter) ───────────→├──→ Task 7 (full verification)
                                                                    │
                                                                    └──→ Task 8 (update __init__)
```

Tasks 1 and 2 can run in parallel. Task 3 depends on Task 1. Tasks 4 and 5 depend on Tasks 1 and 2. Task 6 depends on Tasks 4 and 5. Task 7 depends on Task 6. Task 8 depends on Task 7.
