# Pre-release UX Audit Design

**Date:** 2026-02-27
**Scope:** Fix all 15 gaps identified in the end-to-end user workflow review.
**Branch:** Single issue + branch.

## Context

Before running a complete end-to-end pywatershed test (init → pipeline → pywatershed derivation), a review of the CLI workflow from a first-time user's perspective revealed 15 gaps ranging from silent data omissions to stale documentation. This design covers all fixes.

The end-to-end test target: DRB fabric, gridMET climate from 1980-01-01 to 2025-12-31.

## Code Fixes

### Gap 3 — Soils dataset not fetched in `pws_run_cmd` (HIGH)

**Problem:** `_translate_pws_to_pipeline()` in `cli.py` builds a `PipelineConfig` from the pywatershed config but omits the soils dataset (`cfg.datasets.soils`). Step 5 (`_derive_soils`) silently produces defaults instead of derived values.

**Fix:** Add the soils dataset to the translated pipeline config. The soils dataset uses `local_tiff` strategy (POLARIS default) with variables `sand`, `silt`, `clay`, `awc`. Mirror the pattern used for topography and landcover datasets already in the translator.

**Location:** `src/hydro_param/cli.py`, `_translate_pws_to_pipeline()` (lines ~644–716).

### Gap 4 — `daymet_v4`/`conus404_ba` not in registry (HIGH)

**Problem:** `_CLIMATE_SOURCE_MAP` passes `"daymet_v4"` and `"conus404_ba"` through as dataset names, but these don't exist in the dataset registry. Users get an opaque `KeyError`.

**Fix:** Validate the climate source early in `_translate_pws_to_pipeline`. If the source is not in a supported set (currently just `gridmet`), raise a clear `ValueError` with the message: `"Climate source '{source}' is not yet supported. Available sources: gridmet"`. For the end-to-end test, gridMET via `climr_cat` (OPeNDAP) is the target.

**Location:** `src/hydro_param/cli.py`, `_translate_pws_to_pipeline()`.

### Gap 5 — Land cover year hardcoded to 2021 (MEDIUM)

**Problem:** The NLCD dataset request in the translator hardcodes `year=2021` regardless of the user's time period.

**Fix:** Use the end year of `cfg.time_period` (or the most recent available NLCD year if the end year exceeds coverage). The NHGF STAC NLCD collection has annual data 1985–2024.

**Location:** `src/hydro_param/cli.py`, `_translate_pws_to_pipeline()`, line ~664.

### Gap 6 — `failure_mode: tolerant` is a no-op (MEDIUM)

**Problem:** `PipelineConfig.failure_mode` exists but `stage4_process` never checks it. Setting `tolerant` has no effect.

**Fix:** Remove the config option and the TODO comment. Add it back when actually implemented, so users aren't misled by a non-functional setting. Remove from the generated template in `project.py` as well.

**Location:** `src/hydro_param/config.py` line ~249, `src/hydro_param/project.py` template.

### Gap 11 — Unclosed xarray datasets (LOW)

**Problem:** `pws_run_cmd` opens temporal NetCDF files with `xr.open_dataset` but never closes them.

**Fix:** Use a try/finally pattern or explicitly close datasets after `merge_temporal_into_derived` completes.

**Location:** `src/hydro_param/cli.py`, `pws_run_cmd()`, line ~834.

### Gap 14 — `validate` missing logging setup (LOW)

**Problem:** `pws_validate_cmd` never calls `logging.basicConfig()`, so any warnings from the validator are silently discarded.

**Fix:** Add `logging.basicConfig(level=logging.INFO)` at the start of `pws_validate_cmd`.

**Location:** `src/hydro_param/cli.py`, `pws_validate_cmd()`, line ~857.

## Documentation Fixes

### Gap 1 — Stale README (HIGH)

**Problem:** README.md says "Design & planning phase" and "source coming soon."

**Fix:** Update status to "Pre-alpha MVP", add current project structure, quick usage example, and link to full docs.

### Gap 2 — CLI docs missing pywatershed commands (HIGH)

**Problem:** `docs/user-guide/cli.md` ends after `hydro-param run`. No docs for `pywatershed run` or `pywatershed validate`.

**Fix:** Add sections for both commands with usage, arguments, and example output.

### Gap 7 — Output structure undocumented (MEDIUM)

**Problem:** No docs explain what files the pipeline produces or where they go.

**Fix:** Add an "Output files" section to the quickstart or CLI docs explaining the output directory structure for both `run` and `pywatershed run`.

### Gap 8 — Fabric prerequisite missing from quickstart (MEDIUM)

**Problem:** Quickstart assumes user has a fabric file but doesn't explain how to get one.

**Fix:** Add a prerequisite note explaining that fabrics must be obtained via pynhd/pygeohydro or other means before running hydro-param.

### Gap 9 — Nonexistent pip extras in install docs (MEDIUM)

**Problem:** Docs reference `pip install hydro-param[gdp]` etc. but these extras aren't defined in `pyproject.toml`.

**Fix:** Remove pip extras references. Document pixi as the primary install method. Keep a simple `pip install hydro-param` for users who don't use pixi.

## Lower Priority Fixes

### Gap 10 — Translated config invisible for debugging (LOW)

**Fix:** Log the translated `PipelineConfig` at DEBUG level in `pws_run_cmd` so users can see what was generated with `--verbose` or `HYDRO_PARAM_LOG_LEVEL=DEBUG`.

### Gap 12 — Silent soltab skip (LOW)

**Fix:** Add an INFO-level message in `pws_run_cmd` when soltab arrays are not present in the derived dataset (i.e., `soltab.nc` will not be written).

### Gap 13 — Confusing download status for remote datasets (LOW)

**Fix:** Change `datasets list` status labels: remote-access datasets show "remote (no download needed)" instead of ambiguous labels.

### Gap 15 — Example configs reference non-shipped fabric (LOW)

**Fix:** Add a comment at the top of example configs noting the fabric must be obtained separately, with a pointer to the DRB test data or pynhd.

## Test Configuration

End-to-end test target:
- **Fabric:** DRB nhru.gpkg (765 HRUs)
- **Climate:** gridMET via `climr_cat` (OPeNDAP), 1980-01-01 to 2025-12-31
- **Land cover:** NLCD OSN via NHGF STAC
- **Topography:** 3DEP 10m via Planetary Computer STAC
- **Soils:** POLARIS 30m via local_tiff
- **All 14 derivation steps** exercised
