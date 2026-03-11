# GFv1.1 Static Parameter Validation — Design

**Issue:** #171
**Goal:** Validate GFv1.1 ScienceBase rasters against the DRB test fabric and
document differences versus the current 3DEP/gNATSGO/NLCD-based pipeline.

## Scope

Phase A (this design): Static parameters only — all 21 GFv1.1 rasters through
ZonalGen, no climate/forcing data.

Phase B (future): Add gridMET + SNODAS for forcing and climate normals, producing
a complete pywatershed parameterization from GFv1.1 sources.

## Target Fabric

- Path: `data/pywatershed_gis/drb_2yr/nhru.gpkg`
- 765 HRUs, `id_field: nhm_id`
- Segments: `data/pywatershed_gis/drb_2yr/nsegment.gpkg`, `segment_id_field: nhm_seg`

## Deliverables

1. **`configs/examples/gfv11_static_pipeline.yml`** — Pipeline config (Phase 1)
   for all 21 GFv1.1 datasets.
2. **`configs/examples/gfv11_static_pywatershed.yml`** — Pywatershed run config
   (Phase 2) mapping GFv1.1 SIR to PRMS parameters.
3. **Scale factor support in pipeline** — Apply `scale_factor` from registry
   metadata to zonal statistics output.

## Architecture

### Two-Phase Flow

```
Phase 1: hydro-param run configs/examples/gfv11_static_pipeline.yml
  → SIR with 21+ variables (physically meaningful values)

Phase 2: hydro-param pywatershed run configs/examples/gfv11_static_pywatershed.yml
  → parameters.nc with PRMS parameters derived from GFv1.1 sources
```

### Scale Factor Implementation

**Problem:** Three GFv1.1 rasters (`slope100X.tif`, `asp100X.tif`, `twi100X.tif`)
store values as `real_value × 100` in uint32.  The GeoTIFF files have no
`scale_factor` in their GDAL metadata — the factor is only recorded in
hydro-param's registry (`VariableSpec.scale_factor = 0.01`).

The pipeline currently ignores `scale_factor`; the docstring says "consumers
apply it."  This is wrong — scale factor is a data encoding concern, not a
model-specific transform.  The SIR should contain physically meaningful values.

**Fix:** After `processor.process()` returns the zonal statistics DataFrame,
multiply numeric columns by `var_spec.scale_factor` when it is not `None`.

**Location:** `pipeline.py`, `_process_batch()`, after line 783 (`results[var_spec.name] = df`).

**Implementation:**

```python
# Apply scale factor for integer-encoded rasters (e.g., slope × 100)
if isinstance(var_spec, VariableSpec) and var_spec.scale_factor is not None:
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols] * var_spec.scale_factor
    logger.debug(
        "Applied scale_factor %.4f to %s", var_spec.scale_factor, var_spec.name
    )

results[var_spec.name] = df
```

**Why not apply earlier (before ZonalGen)?**  ZonalGen operates on the raw
GeoTIFF.  Converting uint32 to float64 before writing the intermediate TIFF
would double memory usage and change the exactextract statistics for integer
rasters.  Applying after zonal stats is cheaper and equally correct for `mean`
and other linear statistics.

**Categorical variables are unaffected** — `scale_factor` is only set on
continuous variables.  The `if isinstance(var_spec, VariableSpec)` guard
excludes `DerivedVariableSpec` and `DerivedCategoricalSpec`.

### GFv1.1 Dataset → Variable Mapping

All variable names below are as registered in `gfv11.py:GFV11_DATASETS`.

**Topography (5 datasets):**

| Dataset | Variable | Statistic | Scale Factor | Notes |
|---------|----------|-----------|--------------|-------|
| `gfv11_dem` | `elevation` | mean | — | SRTM 30m, meters |
| `gfv11_slope` | `slope` | mean | 0.01 | Integer ×100, degrees |
| `gfv11_aspect` | `aspect` | mean | 0.01 | Integer ×100, degrees |
| `gfv11_twi` | `twi` | mean | 0.01 | Integer ×100, unitless |
| `gfv11_fdr` | `flow_dir` | majority | — | D8 categorical |

**Soils (5 datasets):**

| Dataset | Variable | Statistic | Notes |
|---------|----------|-----------|-------|
| `gfv11_sand` | `sand_pct` | mean | SoilGrids250m, % |
| `gfv11_silt` | `silt_pct` | mean | SoilGrids250m, % |
| `gfv11_clay` | `clay_pct` | mean | SoilGrids250m, % |
| `gfv11_awc` | `awc` | mean | SoilGrids250m, mm |
| `gfv11_text_prms` | `soil_type` | majority | PRMS class (1/2/3) |

**Land Cover (10 datasets):**

| Dataset | Variable | Statistic | Notes |
|---------|----------|-----------|-------|
| `gfv11_lulc` | `cov_type` | majority | NALCMS → PRMS class |
| `gfv11_imperv` | `imperv_pct` | mean | GMIS, % |
| `gfv11_cnpy` | `canopy_pct` | mean | MODIS, % |
| `gfv11_covden_sum` | `covden_sum` | mean | Pre-computed, fraction |
| `gfv11_covden_win` | `covden_win` | mean | Pre-computed, fraction |
| `gfv11_covden_loss` | `covden_loss` | mean | Pre-computed, fraction |
| `gfv11_srain` | `srain_intcp` | mean | Pre-computed, inches |
| `gfv11_wrain` | `wrain_intcp` | mean | Pre-computed, inches |
| `gfv11_snow_intcp` | `snow_intcp` | mean | Pre-computed, inches |
| `gfv11_root_depth` | `root_depth` | mean | Pre-computed, inches |

**Water Bodies (1 dataset):**

| Dataset | Variable | Statistic | Notes |
|---------|----------|-----------|-------|
| `gfv11_wbg` | `waterbody` | majority | NHD HR mask, categorical |

### Pipeline Config: `gfv11_static_pipeline.yml`

Uses the themed `datasets` dict format (PR #187).  All 21 datasets organized
by category.  No `domain` section needed — the fabric bbox is used automatically.

### Pywatershed Run Config: `gfv11_static_pywatershed.yml`

Maps GFv1.1 SIR variables to PRMS parameters.  Key differences from the
standard `drb_2yr_pywatershed.yml`:

- **Topography:** `gfv11_dem` instead of `dem_3dep_10m`
- **Soils:** `gfv11_sand/silt/clay/awc` instead of `polaris_30m/gnatsgo_rasters`
- **Land cover:** `gfv11_lulc/imperv/cnpy` instead of `nlcd_osn_*`
- **Pre-computed params:** `gfv11_covden_sum`, `gfv11_srain`, etc. bypass
  derivation — the pipeline provides them directly as zonal means
- **No forcing/climate:** Omitted for Phase A
- **No snow:** No SNODAS for snarea_thresh calibration seed

### Expected Data Source Differences

When comparing GFv1.1 output against the current pipeline:

| Parameter | GFv1.1 Source | Current Source | Expected Difference |
|-----------|--------------|----------------|---------------------|
| Elevation | SRTM 30m | 3DEP 10m | Small — both DEMs, different resolution |
| Slope/Aspect | TGF 30m | Horn from 3DEP | Moderate — integer encoding vs continuous |
| Soil texture | SoilGrids250m | POLARIS 30m | Significant — different products entirely |
| AWC | SoilGrids250m | gNATSGO rootznaws | Significant — different soil databases |
| Cov type | NALCMS 2015 | NLCD 2021 | Significant — different classification, vintage |
| Imperviousness | GMIS | NLCD FctImp | Moderate — different methodology/vintage |
| Canopy | MODIS MOD44B | Not currently used | N/A — new parameter |
| Cover density | Pre-computed | Derived from NLCD | Interesting — validates derivation logic |
| Interception | Pre-computed | Derived from NLCD | Interesting — validates derivation logic |

## Out of Scope

- Grid processing pathway (polygon only)
- Climate/forcing data (Phase B)
- Comparison notebook/script (manual for now)
- Scale factor for derived variables (only raw `VariableSpec` supported)

## Risks

- **CRS mismatch:** GFv1.1 rasters are EPSG:5070, DRB fabric is EPSG:5070.
  gdptools handles reprojection automatically, but worth verifying.
- **NoData handling:** GFv1.1 slope/aspect use `_FillValue = 2147483647` (uint32
  max).  exactextract should exclude these, but verify means aren't inflated.
- **Pre-computed parameter units:** GFv1.1 interception/root_depth are in inches
  (PRMS native units).  The SIR should preserve these; no conversion needed
  since the pywatershed plugin expects inches for these parameters.
