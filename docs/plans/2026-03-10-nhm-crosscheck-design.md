# NHM Reference Cross-Check Improvements — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align hydro-param's processing with the NHM reference workflow where
scientifically justified, and fix the GFv1.1 CV_INT misregistration discovered
during cross-check analysis.

**Issue:** #185

**Architecture:** All changes respect the two-phase separation. Items 1–2 are
derivation plugin changes (Phase 2). Item 3 is a registry fix + new derivation
step. Item 4 updates configs. The pipeline (Phase 1) is unchanged except for
config-level statistic selection.

**Tech Stack:** Python, xarray, numpy, geopandas, shapely, rasterio, gdptools

---

## Context

A cross-check of hydro-param against the NHM GFv1.1 reference parameterization
workflow (Jupyter notebooks from the NHM GIS processing repository) identified
several processing differences and one data registration error. This design
addresses the items that should be fixed now; three larger items are tracked as
separate issues (#200, #154, #73).

### Raster Inspection Results

| File | dtype | Sample values | Shape | What it is |
|------|-------|--------------|-------|------------|
| CV_INT.tif | uint8 | 1, 2, 6 | 3301×4663 (~16 MB) | Snow CV integer class for `hru_deplcrv` |
| Snow.tif | uint8 | 0, 2, 3, 5, 7, 10 | 124131×166734 (30m CONUS) | Snow interception (NALCMS crosswalk lookup, hundredths-inches) |
| CNPY.tif | int16 | 0–100 | 110020×155443 (30m CONUS) | Tree canopy cover percent (MOD44B) → `covden_sum` source |

### SDC_table.csv

9 curves × 11 values (matching PRMS `ndeplval=11`). Columns Val1–Val9 are indexed
by CV_INT class number. Each column defines the fractional snow-covered area at
11 evenly-spaced SWE fractions (0.0 to 1.0).

```
Val1,Val2,Val3,Val4,Val5,Val6,Val7,Val8,Val9
0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00
0.96,0.85,0.75,0.61,0.29,0.22,0.17,0.14,0.10
...
1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00
```

---

## Change 1: Elevation statistic — mean → median

**Rationale:** Median is more robust to outlier cells (cliff pixels, water surface
pixels) that can skew the HRU mean elevation. The NHM reference uses median.
gdptools supports `median` as a zonal statistic.

**Scope:**
- `src/hydro_param/derivations/pywatershed.py` — `_derive_topography()`:
  prefer `elevation_m_median`, fall back to `elevation_m_mean` with warning
- Pipeline configs: change statistic from `mean` to `median` for elevation
  datasets (example configs + pw-check configs)
- Dataset registry: no change needed (gdptools handles the statistic)

**SIR variable:** `elevation_m_median` (new canonical name alongside existing
`elevation_m_mean`)

**Backward compat:** The derivation checks for `elevation_m_median` first, then
falls back to `elevation_m_mean` with a deprecation-style warning. Existing SIR
output continues to work.

---

## Change 2: Latitude — centroid → representative_point()

**Rationale:** `representative_point()` guarantees the point falls inside the
polygon, which matters for concave or horseshoe-shaped HRUs. The NHM reference
uses this approach. Shapely's `representative_point()` is a drop-in replacement
with negligible performance difference.

**Scope:**
- `src/hydro_param/derivations/pywatershed.py` — `_derive_geometry()`:
  change `.centroid` to `.representative_point()` for latitude/longitude
  computation
- Update docstring and `long_name` attribute from "centroid" to
  "representative point"

**Note:** HRU area computation continues to use `.geometry.area` (polygon area,
not point-based). Only the lat/lon extraction changes.

---

## Change 3: Fix CV_INT misregistration + wire snow depletion curves

This is the largest change. It has three sub-parts:

### 3a. Remove gfv11_covden_sum (misregistered)

The `gfv11_covden_sum` entry in `gfv11.py` maps to `CV_INT.tif`, which is NOT
summer cover density — it's the snow CV integer classification raster. The actual
covden_sum source is `CNPY.tif` (registered as `gfv11_cnpy`).

**Scope:**
- `src/hydro_param/gfv11.py` — remove the `gfv11_covden_sum` entry from
  `GFV11_DATASETS`
- `src/hydro_param/gfv11.py` — update `FILE_DIRECTORY_MAP` comment for
  CV_INT.zip if needed

### 3b. Add gfv11_cv_int dataset

Register CV_INT.tif correctly as a categorical snow CV raster.

**Scope:**
- `src/hydro_param/gfv11.py` — add `gfv11_cv_int` entry to `GFV11_DATASETS`:
  - filename: `CV_INT.tif`
  - subdir: `land_cover`
  - variable name: `cv_int` (snow CV integer class)
  - categorical: `True`
  - units: `class_index`

### 3c. Wire snow depletion curves into derivation

Add SDC_table.csv consumption to populate `hru_deplcrv` and `snarea_curve`.

**Data flow:**
1. Pipeline Phase 1: CV_INT.tif → zonal stats (majority) → SIR variable
   `cv_int_frac` (categorical fractions per HRU)
2. Derivation Phase 2:
   - Extract majority CV class per HRU from `cv_int_frac` (same pattern as
     `cov_type` categorical extraction)
   - Load SDC_table.csv (bundled in package data)
   - Index into SDC table by CV class → 11 values per curve
   - Populate `hru_deplcrv` (integer index, 1-based) and `snarea_curve`
     (ndeplval × ncurves array)

**Scope:**
- `src/hydro_param/data/pywatershed/lookup_tables/sdc_table.yml` — convert
  SDC_table.csv to YAML format (consistent with other lookup tables) or keep
  as CSV and add a loader
- `src/hydro_param/derivations/pywatershed.py` — new helper or extend step 13
  (defaults) to derive `hru_deplcrv` + `snarea_curve` from CV_INT + SDC table
- Pre-computed pass-through: support `_try_precomputed(ctx, "hru_deplcrv",
  categorical=True)` for configs that declare CV_INT as a precomputed source

**SDC table mapping:** CV_INT raster values (1–9) map to SDC_table columns
(Val1–Val9). Each column has 11 rows = `ndeplval` values for `snarea_curve`.
The derivation assigns each HRU's `hru_deplcrv` to the 1-based curve index
matching its majority CV_INT class, then collects the unique curves into
`snarea_curve`.

**PRMS dimension model:** `snarea_curve` has dimension `ndeplval` (11 values
per curve) × `ndepl` (number of unique curves). `hru_deplcrv` is dimensioned
`nhru` and indexes into the curve array (1-based).

**Fallback:** When CV_INT is unavailable (no `gfv11_cv_int` in SIR), keep
current behavior: `hru_deplcrv = 1` (all HRUs use curve 1) and
`snarea_curve = [0.0, ..., 1.0]` (linear depletion, 11 values).

---

## Change 4: Update configs

**Scope:**
- `pw-check/configs/gfv11_static_pipeline.yml` — remove `gfv11_covden_sum`,
  add `gfv11_cv_int`
- `pw-check/configs/gfv11_static_pywatershed.yml` — remove `covden_sum`
  reference to `gfv11_covden_sum`, ensure `covden_sum` derives from
  `gfv11_cnpy` (canopy% / 100); add `hru_deplcrv` entry pointing to
  `gfv11_cv_int`
- `configs/examples/gfv11_static_pipeline.yml` — same updates

---

## What we're NOT changing

| Item | Reason |
|------|--------|
| Segment slope | Keep NHDPlus VAA lookup — endpoint sampling is unreliable |
| soil_moist_max source | Keep gNATSGO aws0_100 (already integrated product) |
| Snow.tif mapping | Confirmed correct — NALCMS crosswalk snow interception |
| Subsurface flux rescaling | Future issue #154 (needs GLHYMPS data source) |
| Derived-raster pathway | Future issue #200 (new stage 4 strategy) |
| Spatial gap-fill | Future issue #73 (scope expanded to static + temporal) |

---

## Testing Strategy

- Unit tests for elevation median fallback (median preferred, mean accepted)
- Unit test for representative_point() lat/lon (verify inside polygon)
- Unit test for `gfv11_cv_int` registration (categorical, correct filename)
- Unit test for SDC table loading + `hru_deplcrv` / `snarea_curve` derivation
- Integration: re-run pw-check Phase 1 + Phase 2 with corrected configs

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Existing SIR output has `elevation_m_mean` | Fallback to mean with warning |
| CV_INT values don't map 1:1 to SDC columns | Verify mapping with raster VAT + table dimensions |
| SDC_table has 9 columns but CV_INT has values 1–6 | Only 6 of 9 curves used; unused curves are still loaded |
| `gfv11_covden_sum` removal breaks existing configs | Update all configs in same PR; overlay regeneration needed |
