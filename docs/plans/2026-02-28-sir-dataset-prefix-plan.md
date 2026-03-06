# Dataset-Prefixed SIR Filenames — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prefix SIR filenames with the dataset registry key (`{dataset}__{canonical}.csv`) so source provenance is preserved on disk and accessible to derivation plugins.

**Architecture:** Thread `dataset_name` from `DatasetRequest.name` through `SIRVariableSchema` → `normalize_sir()` → SIR filenames → manifest → `SIRAccessor`. Backward-compatible: unprefixed lookups still work in derivation code.

**Tech Stack:** Python, xarray, pandas, pytest

---

### Task 1: Create issue and feature branch

**Step 1: Create GitHub issue**

```bash
gh issue create \
  --title "feat: prefix SIR filenames with dataset name for source provenance" \
  --body "$(cat <<'EOF'
## Problem

SIR files use canonical names (`sand_pct_mean.csv`) that strip source dataset identity. If a user switches from POLARIS to SSURGO for soils, or uses a different landcover source, the derivation plugin has no way to know which dataset produced a file. This prevents routing to dataset-specific processing logic.

## Solution

Prefix SIR filenames with the dataset registry key using a double-underscore separator:

```
sir/polaris_30m__sand_pct_mean.csv
sir/dem_3dep_10m__elevation_m_mean.csv
sir/nlcd_osn_lndcov__lndcov_frac_2021.csv
sir/gridmet__tmmx_C_mean_2021.nc
```

## Scope

This issue covers provenance in filenames, manifest, and SIRAccessor API. Derivation routing based on dataset prefix is a separate future issue.

## Changes

1. `SIRVariableSchema` gains `dataset_name` field
2. `build_sir_schema()` populates it from `DatasetRequest.name`
3. `normalize_sir()` and `normalize_sir_temporal()` use prefixed filenames
4. `SIRSchemaEntry` (manifest) gains `source_dataset` field
5. `SIRAccessor` supports both prefixed and unprefixed lookups, adds `source_for()` method
6. Glob fallback parses `__` separator from filenames

## Design

See `docs/plans/2026-02-28-sir-dataset-prefix-design.md`
EOF
)"
```

**Step 2: Create feature branch**

```bash
# Use the issue number from step 1
git checkout -b feat/<ISSUE_NUM>-sir-dataset-prefix main
```

---

### Task 2: Add `dataset_name` to `SIRVariableSchema` and `build_sir_schema()`

**Files:**
- Modify: `src/hydro_param/sir.py:170-307`
- Test: `tests/test_sir.py`

**Step 1: Write failing test**

Add a test that `build_sir_schema()` populates `dataset_name` on each schema entry:

```python
def test_build_sir_schema_includes_dataset_name():
    """build_sir_schema populates dataset_name from DatasetRequest.name."""
    from hydro_param.sir import build_sir_schema
    # Use existing test fixtures for resolved datasets
    # Assert schema[i].dataset_name == ds_req.name for each entry
```

**Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_sir.py -k "dataset_name" -v -x
```

**Step 3: Add `dataset_name` field to `SIRVariableSchema`**

In `src/hydro_param/sir.py`, add field to the dataclass at line ~212:

```python
dataset_name: str = ""
```

**Step 4: Populate `dataset_name` in `build_sir_schema()`**

In the loop at line ~250, pass `ds_req.name` to each `SIRVariableSchema`:

```python
for entry, ds_req, var_specs in resolved:
    ...
    # In both categorical and continuous branches:
    SIRVariableSchema(
        ...
        dataset_name=ds_req.name,
    )
```

**Step 5: Run test to verify it passes**

```bash
pixi run -e dev pytest tests/test_sir.py -k "dataset_name" -v -x
```

**Step 6: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add dataset_name to SIRVariableSchema (#<ISSUE>)

build_sir_schema() now populates dataset_name from DatasetRequest.name
on every schema entry, threading source provenance through the SIR
normalization pipeline."
```

---

### Task 3: Generate prefixed SIR filenames in `normalize_sir()`

**Files:**
- Modify: `src/hydro_param/sir.py:358-532` (`normalize_sir`)
- Modify: `src/hydro_param/sir.py:535-630` (`normalize_sir_temporal`)
- Test: `tests/test_sir.py`

**Step 1: Write failing test**

Test that `normalize_sir()` produces files with dataset prefix:

```python
def test_normalize_sir_uses_dataset_prefix(tmp_path):
    """normalize_sir() writes files with {dataset}__{canonical}.csv pattern."""
    # Create a raw CSV, build schema with dataset_name="dem_3dep_10m"
    # Call normalize_sir()
    # Assert output file is "dem_3dep_10m__elevation_m_mean.csv"
    # Assert returned dict key is "dem_3dep_10m__elevation_m_mean"
```

**Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_sir.py -k "dataset_prefix" -v -x
```

**Step 3: Update `normalize_sir()` to use prefixed filenames**

The schema entries now have `dataset_name`. When writing the output file, use `{dataset_name}__{canonical_name}` as both the filename stem and the dict key.

Key change in the normalization loop: where it currently builds the output filename from `entry.canonical_name`, prefix it with `entry.dataset_name + "__"`.

**Step 4: Update `normalize_sir_temporal()` similarly**

The temporal normalizer needs the dataset name from the schema entry. The `file_key` (e.g., `gridmet_2020`) already contains the dataset name as a prefix, but use the schema's `dataset_name` for consistency.

Output key becomes: `{dataset_name}__{canonical_name_with_year_suffix}`
Output file becomes: `{dataset_name}__{canonical_name_with_year_suffix}.nc`

**Step 5: Run tests**

```bash
pixi run -e dev pytest tests/test_sir.py -v -x
```

Expect existing tests to fail because filenames changed. Update them in the next step.

**Step 6: Update existing `normalize_sir` tests for new filename pattern**

All assertions checking SIR filenames need to expect the `{dataset}__` prefix.

**Step 7: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: prefix SIR filenames with dataset name (#<ISSUE>)

normalize_sir() and normalize_sir_temporal() now write files as
{dataset}__{canonical}.csv/.nc. The double-underscore separator is
unambiguous since neither dataset keys nor canonical names contain __."
```

---

### Task 4: Update `SIRSchemaEntry` manifest and stage 5

**Files:**
- Modify: `src/hydro_param/manifest.py:96-112`
- Modify: `src/hydro_param/pipeline.py:1416-1438`
- Test: `tests/test_manifest.py` (if exists), `tests/test_pipeline.py`

**Step 1: Add `source_dataset` to `SIRSchemaEntry`**

In `src/hydro_param/manifest.py`:

```python
class SIRSchemaEntry(TypedDict):
    name: str
    units: str
    statistic: str
    source_dataset: str  # NEW
```

**Step 2: Update stage 5 manifest construction**

In `src/hydro_param/pipeline.py` at line ~1430, add `source_dataset`:

```python
sir_schema=[
    _manifest_mod.SIRSchemaEntry(
        name=s.canonical_name,
        units=s.canonical_units,
        statistic="categorical" if s.categorical else "continuous",
        source_dataset=s.dataset_name,
    )
    for s in schema
],
```

Note: The manifest `static_files` and `temporal_files` keys will automatically
use prefixed names because `normalize_sir()` now returns prefixed keys.

**Step 3: Run tests**

```bash
pixi run -e dev pytest tests/test_pipeline.py tests/test_sir.py -v -x 2>&1 | tail -30
```

Fix any failures from the manifest schema change.

**Step 4: Commit**

```bash
git add src/hydro_param/manifest.py src/hydro_param/pipeline.py tests/
git commit -m "feat: add source_dataset to SIR manifest schema (#<ISSUE>)

SIRSchemaEntry now includes source_dataset, persisted in .manifest.yml.
Stage 5 populates it from SIRVariableSchema.dataset_name."
```

---

### Task 5: Update `SIRAccessor` for prefixed lookups

**Files:**
- Modify: `src/hydro_param/sir_accessor.py`
- Test: `tests/test_sir_accessor.py`

**Step 1: Write failing tests**

```python
def test_sir_accessor_unprefixed_lookup():
    """SIRAccessor finds prefixed files via unprefixed canonical name."""
    # Manifest has key "polaris_30m__sand_pct_mean" -> "sir/polaris_30m__sand_pct_mean.csv"
    # sir["sand_pct_mean"] should resolve to it
    # "sand_pct_mean" in sir should be True

def test_sir_accessor_prefixed_lookup():
    """SIRAccessor finds files by full prefixed name."""
    # sir["polaris_30m__sand_pct_mean"] should work directly

def test_sir_accessor_source_for():
    """SIRAccessor.source_for() returns dataset name."""
    # sir.source_for("sand_pct_mean") == "polaris_30m"
    # sir.source_for("polaris_30m__sand_pct_mean") == "polaris_30m"

def test_sir_accessor_find_variable_unprefixed():
    """find_variable() works with unprefixed base names on prefixed files."""
    # sir.find_variable("fctimp_pct_mean") resolves to
    # "nlcd_osn_fctimp__fctimp_pct_mean_2021" (most recent year)
```

**Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_sir_accessor.py -k "prefixed or source_for" -v -x
```

**Step 3: Implement SIRAccessor changes**

Key changes to `SIRAccessor`:

1. **Internal index:** Build a secondary `_canonical_to_prefixed` dict mapping
   canonical names (the part after `__`) to their full prefixed keys. This enables
   unprefixed lookups.

2. **`__contains__`:** Check prefixed key first, then canonical-to-prefixed fallback.

3. **`__getitem__`:** Same resolution order.

4. **`source_for(name: str) -> str | None`:** Parse `__` from the resolved key,
   return the dataset prefix.

5. **`find_variable(base_name)`:** Search both prefixed keys and canonical portions
   for year-suffix matching.

6. **`_glob_sir_static()` / `_glob_sir_temporal()`:** When globbing, use the full
   stem (including prefix) as the dict key. The canonical-to-prefixed index handles
   backward-compatible lookup.

**Step 4: Run full test suite**

```bash
pixi run -e dev pytest tests/test_sir_accessor.py -v -x
```

**Step 5: Update existing SIRAccessor tests**

Tests that create manifest entries or glob fixtures need updated filenames.

**Step 6: Commit**

```bash
git add src/hydro_param/sir_accessor.py tests/test_sir_accessor.py
git commit -m "feat: SIRAccessor supports prefixed and unprefixed lookups (#<ISSUE>)

SIRAccessor now handles dataset-prefixed SIR keys (polaris_30m__sand_pct_mean)
while maintaining backward compatibility with unprefixed lookups
(sand_pct_mean). New source_for() method returns the dataset name for a
given variable."
```

---

### Task 6: Update derivation and CLI tests for prefixed SIR

**Files:**
- Modify: `tests/test_pywatershed_derivation.py` (test fixtures)
- Modify: `tests/test_cli.py` (test fixtures)
- Possibly: other test files that create SIR fixtures

**Step 1: Find all test files referencing SIR variable names**

```bash
pixi run -e dev grep -rn "elevation_m_mean\|sand_pct_mean\|lndcov_frac\|sir/" tests/ | head -40
```

**Step 2: Update test fixtures**

Tests using `_MockSIRAccessor` or creating manifest fixtures with unprefixed
SIR names should continue to work if the SIRAccessor backward compatibility
is correct. Verify this by running:

```bash
pixi run -e dev pytest tests/ -v -x 2>&1 | tail -40
```

Only fix tests that actually fail — the backward-compatible lookup should
handle most cases.

**Step 3: Commit any test fixes**

```bash
git add tests/
git commit -m "test: update test fixtures for prefixed SIR filenames (#<ISSUE>)"
```

---

### Task 7: Run full check suite

**Step 1: Run full checks**

```bash
pixi run -e dev check
```

**Step 2: Run pre-commit**

```bash
pixi run -e dev pre-commit
```

**Step 3: Fix any issues and commit**

```bash
git add -u
git commit -m "chore: fix lint/type issues from SIR prefix changes (#<ISSUE>)"
```
