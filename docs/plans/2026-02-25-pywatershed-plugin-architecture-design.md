# pywatershed Plugin Architecture Redesign

**Date:** 2026-02-25
**Status:** Approved
**Scope:** Redesign the plugin contract, interfaces, and module layout so the pywatershed plugin serves as an exemplary template for future model plugins.

## Motivation

The pywatershed plugin (derivation + formatter) works but has accumulated several architectural issues:

1. **No formal plugin contract.** A NextGen developer has no single place to learn what a plugin must implement.
2. **Monolithic derive() signature.** Six loose arguments (`sir`, `config`, `fabric`, `segments`, `id_field`, `segment_id_field`) make the interface hard to extend and hard to read.
3. **Tight coupling to internals.** Plugins reach into SIR dimensions, config dicts, and GeoDataFrames through ad-hoc patterns. The boundary between pipeline output and plugin input isn't typed.
4. **Scattered infrastructure.** `output.py` has an informal `OutputFormatter` protocol; derivation has no protocol at all; registries are split across files.
5. **Deprecated cruft.** `rename_sir_variables()` is deprecated but still present. `_sir_id_dim()` validates context state but lives as a module-level helper.
6. **Packaged-install breakage.** Lookup tables and metadata default to `Path("configs/...")` relative to project root, which doesn't exist in `pip install` environments.

## Approach

**Protocol + Context + Unified Registry.** Define the plugin contract formally with `typing.Protocol`, bundle all plugin inputs into a typed `DerivationContext` dataclass, and consolidate plugin discovery in a single `plugins.py` module.

Alternatives considered:
- **Minimal cleanup** (add protocols, keep 6-arg signature) — rejected: doesn't fix the interface or boundary problems.
- **Full plugin framework with base classes** (ABC + shared infrastructure) — rejected: over-engineering for 2-3 plugins, violates YAGNI.

## Design

### 1. Plugin Contract (`plugins.py`)

A single new module that answers "what must a plugin implement?"

```python
# src/hydro_param/plugins.py

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Protocol, runtime_checkable

import geopandas as gpd
import xarray as xr


@dataclass(frozen=True)
class DerivationContext:
    """Typed input bundle for derivation plugins.

    Bundles everything a derivation plugin needs into a single
    immutable object. Validates inputs on construction.
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
        if self.lookup_tables_dir is not None:
            return self.lookup_tables_dir
        return Path(str(files("hydro_param").joinpath("data/lookup_tables")))


@runtime_checkable
class DerivationPlugin(Protocol):
    """Contract for model-specific parameter derivation."""

    name: str

    def derive(self, context: DerivationContext) -> xr.Dataset: ...


@runtime_checkable
class FormatterPlugin(Protocol):
    """Contract for model-specific output formatting."""

    name: str

    def validate(self, parameters: xr.Dataset) -> list[str]: ...

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]: ...
```

The module also contains:
- `NetCDFFormatter` and `ParquetFormatter` (migrated from `output.py`, unchanged)
- `get_derivation(name: str) -> DerivationPlugin` — config-driven factory with lazy imports
- `get_formatter(name: str) -> FormatterPlugin` — config-driven factory with lazy imports

### 2. Derivation Plugin Refactoring

`PywatershedDerivation` keeps its single-class-with-private-methods structure. Changes:

**Interface:**
- `derive()` accepts `DerivationContext` instead of 6 separate arguments.
- Every step method takes `(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset`.
- Lookup tables loaded from `ctx.resolved_lookup_tables_dir`.

**Cleanup:**
- `rename_sir_variables()` deleted entirely (SIR normalization handles this in Stage 5).
- Module-level `_sir_id_dim()` deleted (validation moved to `DerivationContext.__post_init__`).
- `merge_temporal_into_derived()` becomes a method, uses `ctx.fabric_id_field` instead of hardcoded `"nhru"`.

**Orchestrator:**
```python
def derive(self, context: DerivationContext) -> xr.Dataset:
    nhru = context.sir.sizes[context.fabric_id_field]
    ds = xr.Dataset()
    ds = ds.assign_coords(nhru=context.sir[context.fabric_id_field].values)

    ds = self._derive_geometry(context, ds)
    ds = self._derive_topology(context, ds)
    ds = self._derive_topography(context, ds)
    ds = self._derive_landcover(context, ds)
    ds = self._apply_lookup_tables(context, ds)
    ds = self._apply_defaults(ds, nhru)
    ds = self._apply_overrides(ds, context.config)
    return ds
```

**Unchanged:** All derivation math (unit conversions, NLCD reclassification, majority extraction, lookup table application, default values, topology validation). The `_DEFAULTS` dict, `_IMPERV_STOR_MAX_DEFAULT`, and all static/validation helper methods stay as-is.

### 3. Formatter Plugin Refactoring

`PywatershedFormatter` changes are minimal:

- `__init__` drops `metadata_path` parameter; metadata path resolved via `importlib.resources` default or `config["metadata_path"]` override in `write()`.
- `validate()` returns `["Parameter metadata unavailable — validation skipped"]` when metadata file is missing (instead of silent empty list).
- Satisfies `FormatterPlugin` protocol (already structurally compatible).

**Unchanged:** All four `write_*` methods, `_FORCING_VARS`, `_SOLTAB_VARS`, CF-1.8 attributes, one-variable-per-file forcing pattern.

### 4. Package Data Migration

Lookup tables and parameter metadata move into the Python package for `importlib.resources` access:

```
src/hydro_param/data/
  lookup_tables/
    nlcd_to_prms_cov_type.yml
    cov_type_to_interception.yml
    cov_type_winter_reduction.yml
    soil_texture_to_prms_type.yml
  pywatershed/
    parameter_metadata.yml
```

The original `configs/` directory at project root stays for human reference and example pipeline configs. `pyproject.toml` gets a `[tool.setuptools.package-data]` entry to include `data/**/*.yml`.

`DerivationContext.lookup_tables_dir` defaults to `None` (uses `importlib.resources`), but can be overridden for development or custom table sets.

### 5. Error Handling

Consistent pattern across the plugin boundary:

| Location | Behavior |
|----------|----------|
| `DerivationContext.__post_init__` | Fail hard (`KeyError`) if `fabric_id_field` not in SIR dims |
| `derive()` entry | No additional validation needed (context is pre-validated) |
| Individual step methods | Guard clause + `logger.info` skip when optional data missing |
| Topology validators | Fail hard (`ValueError`) for corrupted data (self-loops, range violations) |
| `FormatterPlugin.validate()` | Return warning strings (caller decides severity) |
| `FormatterPlugin.write()` | Call `validate()` first, log warnings, write anyway |

### 6. Module Retired

`src/hydro_param/output.py` is deleted. Its contents:
- `OutputFormatter` protocol -> replaced by `FormatterPlugin` in `plugins.py`
- `NetCDFFormatter` -> moved to `plugins.py`
- `ParquetFormatter` -> moved to `plugins.py`
- `get_formatter()` -> moved to `plugins.py`

### 7. Testing Strategy

**New:** `tests/test_plugins.py` (~10 tests)
- `DerivationContext` validation (missing dim raises `KeyError`, resolved paths)
- Registry functions (`get_derivation`, `get_formatter` for valid and invalid names)
- Protocol satisfaction checks (`isinstance` with `runtime_checkable`)

**Modified:** `tests/test_pywatershed_derivation.py`
- Migrate all `derive()` calls to use `DerivationContext`
- All assertion logic unchanged (same math, same expected values)
- Confirm `rename_sir_variables` is removed

**Modified:** `tests/test_pywatershed_formatter.py`
- Assert `validate()` returns warning when metadata missing (not empty list)

**Modified:** Any tests importing from `output.py` — update imports to `plugins.py`

**Estimated:** ~10 new tests, 0 net change in existing test count.

## File Change Summary

| File | Change |
|------|--------|
| `src/hydro_param/plugins.py` | **New** — protocols, context, registries, built-in formatters |
| `src/hydro_param/data/lookup_tables/*.yml` | **Moved** from `configs/lookup_tables/` |
| `src/hydro_param/data/pywatershed/*.yml` | **Moved** from `configs/pywatershed/` |
| `src/hydro_param/derivations/pywatershed.py` | Refactored — `DerivationContext`, step signatures, remove deprecated |
| `src/hydro_param/formatters/pywatershed.py` | Minor — metadata via `importlib.resources`, validate warning |
| `src/hydro_param/output.py` | **Deleted** — migrated to `plugins.py` |
| `src/hydro_param/derivations/__init__.py` | Unchanged |
| `src/hydro_param/formatters/__init__.py` | Unchanged |
| `src/hydro_param/cli.py` | Construct `DerivationContext`, import from `plugins` |
| `src/hydro_param/pipeline.py` | Import from `plugins` instead of `output` |
| `tests/test_plugins.py` | **New** — plugin infrastructure tests |
| `tests/test_pywatershed_derivation.py` | Migrate to `DerivationContext` |
| `tests/test_pywatershed_formatter.py` | Minor — validate warning assertion |
| `pyproject.toml` | Add `package-data` for `data/**/*.yml` |

## Scope Boundaries

**This redesign does NOT:**
- Implement remaining derivation steps (5-7, 9-12, 14)
- Change the SIR format or pipeline stages 1-5
- Change lookup table YAML schema
- Change formatter output file formats
- Add entry-point-based plugin discovery
- Introduce base classes or inheritance
- Change `pywatershed_config.py` (Pydantic config schema unchanged; CLI constructs `DerivationContext` from it)
