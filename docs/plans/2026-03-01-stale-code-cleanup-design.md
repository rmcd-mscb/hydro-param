# Stale Code & Backward Compatibility Cleanup

**Date:** 2026-03-01
**Status:** Design

## Problem

After rapid iteration (PRs #116–#130), the codebase contains:

1. **Dead code** — `_build_sir_attrs()` in pipeline.py, never called
2. **Config backward compat** — `cbh_dir` → `forcing_dir` migration validator (pre-v4.0 artifact)
3. **Manifest v1 support** — `_SUPPORTED_VERSIONS = {1, 2}` with empty-string datetime parsing for legacy entries
4. **SIR glob fallback** — `_glob_sir_static()` / `_glob_sir_temporal()` when manifest is missing
5. **Pipeline stage 4 fallback** — raw stage 4 file fallback when SIR unavailable
6. **Stale tests** — tests for removed fields, old version references, anti-pattern assertions
7. **Stale docstrings** — version references ("v3.0"), outdated comments

Since this is a pre-alpha project with no external users depending on backward compatibility, all compat code should be removed.

## Changes

### pipeline.py
- Delete `_build_sir_attrs()` (~26 lines)
- Remove raw stage 4 fallback path (if present)

### pywatershed_config.py
- Delete `_migrate_cbh_dir` model validator (~20 lines)
- Remove `import warnings` if no longer needed

### manifest.py
- Change `_SUPPORTED_VERSIONS = {1, 2}` → `{2}`
- Remove empty-string datetime parsing in `_parse_completed_at` validators (2 occurrences)

### sir_accessor.py
- Remove glob fallback in `__init__`
- Delete `_glob_sir_static()` and `_glob_sir_temporal()` helper functions
- Remove backward-compat lookup comments
- Raise an error (or log error + empty state) when manifest is missing

### test_pywatershed_config.py
- Delete `test_legacy_cbh_dir_migrated`
- Delete `test_forcing_dir_takes_precedence_over_cbh_dir`
- Delete `test_rejects_old_datasets_field`, `test_rejects_old_climate_field`, `test_rejects_old_processing_field`
- Delete `test_no_extraction_method` anti-pattern checks
- Rename `test_v4_*` tests to generic names

### test_cli.py
- Fix `_write_pws_config()` docstring "v3.0" → "v4.0"

### test_manifest.py
- Update `test_manifest_version_1_has_no_sir` → should now expect rejection of v1
- Keep `test_manifest_version_2_with_sir` (current version)

### test_sir_accessor.py (if exists)
- Remove tests that rely on glob fallback behavior

## Non-goals

- No new features
- No config schema changes (v4.0 stays v4.0)
- No changes to dataset registry (`nlcd_legacy` stays — it's a valid dataset, not compat code)
