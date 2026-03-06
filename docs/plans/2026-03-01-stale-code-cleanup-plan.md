# Stale Code & Backward Compatibility Cleanup Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove all backward compatibility code, dead code, and stale test/doc references from the codebase.

**Architecture:** Pure deletion/simplification pass. No new features, no schema changes. v4.0 config stays v4.0. Manifest drops v1, keeps v2 only.

**Tech Stack:** Python, Pydantic, pytest

---

### Task 1: Delete dead `_build_sir_attrs()` from pipeline.py

**Files:**
- Modify: `src/hydro_param/pipeline.py:1008-1034`
- Test: `tests/test_pipeline.py` (no changes needed — function was never tested)

**Step 1: Delete `_build_sir_attrs` function**

Remove lines 1008–1034 from `pipeline.py`. This function is never called anywhere.

**Step 2: Run tests to verify nothing breaks**

Run: `pixi run -e dev pytest tests/test_pipeline.py -v --tb=short -q`
Expected: All pass

**Step 3: Commit**

```bash
git add src/hydro_param/pipeline.py
git commit -m "refactor: remove unused _build_sir_attrs() from pipeline"
```

---

### Task 2: Remove `load_sir()` static_files fallback from PipelineResult

**Files:**
- Modify: `src/hydro_param/pipeline.py:153-186`

**Step 1: Simplify `load_sir()` to use only `sir_files`**

Replace the `load_sir()` method body. Remove the `elif self.static_files` fallback branch. The method should use `sir_files` directly — if empty, return empty dataset.

Updated `load_sir()`:
```python
def load_sir(self) -> xr.Dataset:
    """Load normalized SIR files into a combined xr.Dataset.

    Assemble all per-variable SIR CSV files into a single
    ``xr.Dataset`` with the fabric ``id_field`` as the dimension.

    Returns
    -------
    xr.Dataset
        Combined dataset with one data variable per SIR variable.
        Returns an empty dataset if no files are available.
    """
    if not self.sir_files:
        logger.warning("No SIR files available — returning empty dataset")
        return xr.Dataset()
    dfs = [pd.read_csv(p, index_col=0) for p in self.sir_files.values()]
    combined = pd.concat(dfs, axis=1)
    return xr.Dataset.from_dataframe(combined)
```

Also update the class docstring (around line 118–120) to remove the fallback mention.

**Step 2: Run tests**

Run: `pixi run -e dev pytest tests/test_pipeline.py -v --tb=short -q`
Expected: All pass (the test `test_load_sir_fallback_to_static` may fail — delete it if so)

**Step 3: Commit**

```bash
git add src/hydro_param/pipeline.py
git commit -m "refactor: remove static_files fallback from PipelineResult.load_sir()"
```

---

### Task 3: Remove `cbh_dir` migration from pywatershed_config.py

**Files:**
- Modify: `src/hydro_param/pywatershed_config.py:432-466`
- Modify: `tests/test_pywatershed_config.py:141-160`

**Step 1: Delete the `_migrate_cbh_dir` validator**

Remove lines 445–466 (the `@model_validator(mode="before")` and its body). Also remove the `cbh_dir` mention from the `PwsOutputConfig` docstring (lines 432–435: "The legacy config key ``cbh_dir`` is accepted with a deprecation warning and mapped to ``forcing_dir``.").

**Step 2: Check if `import warnings` is still needed**

Search for other `warnings.warn` calls in the file. Line 600 uses `warnings.warn` in `validate_available_fields()`, so keep `import warnings`.

**Step 3: Delete legacy tests**

Delete `test_legacy_cbh_dir_migrated` (lines 152-155) and `test_forcing_dir_takes_precedence_over_cbh_dir` (lines 157-160). Update `TestPwsOutputConfig` class docstring from "Tests for output configuration, including cbh_dir migration." to "Tests for output configuration."

**Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py -v --tb=short -q`
Expected: All pass

**Step 5: Commit**

```bash
git add src/hydro_param/pywatershed_config.py tests/test_pywatershed_config.py
git commit -m "refactor: remove cbh_dir backward compatibility migration"
```

---

### Task 4: Remove stale v3.0 field rejection tests and anti-patterns

**Files:**
- Modify: `tests/test_pywatershed_config.py`

**Step 1: Delete old-field rejection tests**

Delete these tests from `TestPywatershedRunConfig`:
- `test_rejects_old_datasets_field` (lines 177-181)
- `test_rejects_old_climate_field` (lines 183-187)
- `test_rejects_old_processing_field` (lines 189-193)

These are redundant — `extra="forbid"` on the model already ensures any unknown field is rejected, and `test_v4_rejects_unknown_top_level_field` covers this generically.

**Step 2: Delete anti-pattern test**

Delete `test_no_extraction_method` (lines 95-101) from `TestPwsDomainConfig`. The `v3.0` comment and negative `hasattr` checks are anti-patterns. `extra="forbid"` already prevents unknown fields.

**Step 3: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py -v --tb=short -q`
Expected: All pass

**Step 4: Commit**

```bash
git add tests/test_pywatershed_config.py
git commit -m "refactor: remove stale v3.0 field rejection tests"
```

---

### Task 5: Rename v4-prefixed tests to generic names

**Files:**
- Modify: `tests/test_pywatershed_config.py`

**Step 1: Rename the test class and methods**

Rename `TestPywatershedRunConfigV4` → `TestPywatershedRunConfigSections` (lines 362-484).
Rename methods:
- `test_v4_accepts_new_sections` → `test_accepts_dataset_sections`
- `test_v4_defaults_all_sections_empty` → `test_defaults_all_sections_empty`
- `test_v4_full_config_from_yaml` → `test_full_config_from_yaml`
- `test_v4_rejects_unknown_top_level_field` → `test_rejects_unknown_top_level_field`

Also update the class docstring from "Tests for v4.0 config with static_datasets, forcing, climate_normals." to "Tests for config sections: static_datasets, forcing, climate_normals."

**Step 2: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py -v --tb=short -q`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_pywatershed_config.py
git commit -m "refactor: rename v4-prefixed config tests to generic names"
```

---

### Task 6: Drop manifest v1 support

**Files:**
- Modify: `src/hydro_param/manifest.py:54,87-93,156-162`
- Modify: `tests/test_manifest.py:386-393`

**Step 1: Change supported versions**

In `manifest.py` line 54, change:
```python
_SUPPORTED_VERSIONS = {1, 2}
```
to:
```python
_SUPPORTED_VERSIONS = {2}
```

**Step 2: Simplify `_parse_completed_at` validators**

In `ManifestEntry` (line 87-93), simplify to:
```python
@field_validator("completed_at", mode="before")
@classmethod
def _parse_completed_at(cls, v: object) -> object:
    """Parse ISO date strings."""
    if isinstance(v, str):
        return datetime.fromisoformat(v)
    return v
```

Same change in `SIRManifestEntry` (lines 156-162).

**Step 3: Remove "legacy" from docstrings**

- Line 79: Change "``datetime.min`` for incomplete or legacy entries." → "``datetime.min`` for incomplete entries."
- Line 109: Change "Empty string for legacy entries." → remove this line.

**Step 4: Update test**

Change `test_manifest_version_1_has_no_sir` (lines 386-393) to verify v1 is **rejected**:
```python
def test_manifest_version_1_rejected(self, tmp_path):
    """Version 1 manifests are no longer supported."""
    manifest_path = tmp_path / ".manifest.yml"
    manifest_path.write_text("version: 1\nfabric_fingerprint: abc\nentries: {}\n")
    loaded = load_manifest(tmp_path)
    assert loaded is None  # load_manifest returns None on validation error
```

**Step 5: Run tests**

Run: `pixi run -e dev pytest tests/test_manifest.py -v --tb=short -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/hydro_param/manifest.py tests/test_manifest.py
git commit -m "refactor: drop manifest v1 support, require v2 only"
```

---

### Task 7: Remove SIR accessor glob fallback

**Files:**
- Modify: `src/hydro_param/sir_accessor.py:71-100,488-559`
- Modify: `tests/test_sir_accessor.py`

**Step 1: Replace glob fallback in `__init__` with error**

Replace lines 75-96 with:
```python
manifest = load_manifest(self._output_dir)
if manifest is None or manifest.sir is None:
    raise FileNotFoundError(
        f"No valid manifest with SIR section at {self._output_dir}. "
        f"Run 'hydro-param run pipeline.yml' to produce SIR output "
        f"before running Phase 2."
    )
self._static = dict(manifest.sir.static_files)
self._temporal = dict(manifest.sir.temporal_files)
self._sir_schema = list(manifest.sir.sir_schema)
```

**Step 2: Delete glob helper functions**

Delete `_glob_sir_static()` (lines 524-540) and `_glob_sir_temporal()` (lines 543-559).

**Step 3: Remove backward-compat comment**

Delete the comment at line 492: "backward-compatible lookups. If two prefixed keys share the same canonical name, the last one wins (alphabetically)". Replace with just: "If two prefixed keys share the same canonical name, the last one wins (alphabetically)".

**Step 4: Update tests**

- Delete `test_glob_fallback_no_manifest` (line 84-90) — it uses the `sir_dir` fixture without manifest.
- Delete `test_glob_fallback_prefixed` (line 587-599) — same glob path.
- Add a new test verifying `FileNotFoundError` when no manifest:

```python
def test_no_manifest_raises(self, tmp_path: Path) -> None:
    """SIRAccessor requires a manifest — no glob fallback."""
    from hydro_param.sir_accessor import SIRAccessor

    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    pd.DataFrame({"val": [1.0]}).to_csv(sir_dir / "foo.csv")
    with pytest.raises(FileNotFoundError, match="No valid manifest"):
        SIRAccessor(tmp_path)
```

**Step 5: Run tests**

Run: `pixi run -e dev pytest tests/test_sir_accessor.py -v --tb=short -q`
Expected: All pass

**Step 6: Commit**

```bash
git add src/hydro_param/sir_accessor.py tests/test_sir_accessor.py
git commit -m "refactor: remove SIR accessor glob fallback, require manifest"
```

---

### Task 8: Fix stale docstring in test_cli.py

**Files:**
- Modify: `tests/test_cli.py:789`

**Step 1: Fix version reference**

Change line 789 from:
```python
"""Write a v3.0 pywatershed run config YAML for testing."""
```
to:
```python
"""Write a v4.0 pywatershed run config YAML for testing."""
```

**Step 2: Run tests**

Run: `pixi run -e dev pytest tests/test_cli.py -v --tb=short -q`
Expected: All pass

**Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "docs: fix stale v3.0 version reference in test_cli.py"
```

---

### Task 9: Final verification

**Step 1: Run full test suite**

Run: `pixi run -e dev check`
Expected: All lint, typecheck, and tests pass

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Verify no remaining backward-compat references**

Search for remaining legacy mentions:
```bash
grep -rn "backward.compat\|legacy\|cbh_dir\|v3\.0\|_glob_sir" src/hydro_param/ tests/
```

Expected: Only legitimate mentions (e.g., `nlcd_legacy` dataset name, which is a real dataset not compat code).
