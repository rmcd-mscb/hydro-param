# Design: Dataset-Prefixed SIR Filenames for Source Provenance

**Date:** 2026-02-28

## Problem

SIR files use canonical names (`sand_pct_mean.csv`) that strip the source
dataset identity. If a user switches from POLARIS to SSURGO for soils, or
uses a different landcover source, the derivation plugin cannot determine
which dataset produced the file. This prevents routing to dataset-specific
processing logic.

## Solution

Prefix SIR filenames with the dataset registry key using a double-underscore
separator: `{dataset}__{canonical_name}.{ext}`

**Examples:**
```
sir/polaris_30m__sand_pct_mean.csv
sir/gnatsgo__aws0_100_cm_mean.csv
sir/dem_3dep_10m__elevation_m_mean.csv
sir/nlcd_osn_lndcov__lndcov_frac_2021.csv
sir/gridmet__tmmx_C_mean_2021.nc
sir/snodas__swe_m_mean_2021.nc
```

## Scope

**This PR:** Provenance in filenames, manifest, and SIRAccessor API.
**Future PR:** Derivation routing based on dataset prefix.

## Design Details

### 1. Separator Convention

Double-underscore `__` separates dataset prefix from canonical name:
- Dataset keys use single `_` (e.g., `polaris_30m`, `dem_3dep_10m`)
- Canonical names use single `_` (e.g., `sand_pct_mean`)
- `__` is unambiguous — no existing name contains it

### 2. `SIRVariableSchema` Changes (`sir.py`)

Add `dataset_name: str` field to `SIRVariableSchema`. Populated from
`DatasetRequest.name` during `build_sir_schema()`.

Add `prefixed_name()` method or update `canonical_name()` to accept an
optional `dataset_name` parameter that prepends `{dataset}__`.

### 3. `build_sir_schema()` Changes (`sir.py`)

Pass `ds_req.name` into each `SIRVariableSchema` entry as `dataset_name`.

### 4. `normalize_sir()` and `normalize_sir_temporal()` Changes (`sir.py`)

Use the schema's `dataset_name` to generate prefixed filenames when writing
normalized SIR files. The mapping keys in the returned dict also use the
prefixed name.

### 5. Stage 5 Manifest Changes (`pipeline.py`)

The SIR manifest `static_files` and `temporal_files` keys become prefixed
names (e.g., `polaris_30m__sand_pct_mean`). The `sir_schema` entries gain
a `source_dataset` field.

### 6. `SIRSchemaEntry` Changes (`manifest.py`)

Add `source_dataset: str` field to `SIRSchemaEntry` TypedDict.

### 7. `SIRAccessor` Changes (`sir_accessor.py`)

- `__contains__` and `__getitem__` accept both prefixed
  (`polaris_30m__sand_pct_mean`) and unprefixed (`sand_pct_mean`) keys.
  Unprefixed lookup searches for any variable whose canonical portion
  matches — backward compatible with existing derivation code.
- New `source_for(variable_name) -> str | None` method returns the dataset
  prefix for a given variable.
- `_glob_sir_static()` and `_glob_sir_temporal()` parse the `__` separator
  from filenames during glob fallback discovery.
- `find_variable()` works with unprefixed base names as before.

### 8. Backward Compatibility

- Existing derivation code (`"elevation_m_mean" in sir`, `sir["sand_pct_mean"]`)
  continues to work — SIRAccessor matches on the canonical portion after `__`.
- Glob fallback handles both old (unprefixed) and new (prefixed) filenames.
- Manifest loading handles both old and new formats gracefully.

### 9. Test Impact

- All tests that reference SIR filenames need updating to include prefixes.
- New tests for `source_for()` method and unprefixed/prefixed lookup.
- New tests for `_glob_sir_static` parsing of `__` separator.
