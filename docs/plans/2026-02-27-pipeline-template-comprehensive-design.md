# Pipeline Template: Comprehensive Dataset Coverage

**Date:** 2026-02-27
**Status:** Design
**Scope:** `src/hydro_param/project.py` — `generate_pipeline_template()`

## Problem

The `hydro-param init` command generates a `pipeline.yml` template with only a
single DEM dataset example and a commented-out NLCD entry. This does not reflect
the 5 data access strategies and 7 dataset configurations that have been tested
end-to-end during pywatershed development. Users who run `init` get a config
that is too minimal to be useful as a starting point.

## Solution

Rewrite `generate_pipeline_template()` to produce a comprehensive template
matching the tested `configs/examples/drb_2yr_pipeline.yml`, but with generic
placeholders instead of DRB-specific values.

## Template Structure

```yaml
target_fabric:
  path: "data/fabrics/catchments.gpkg"
  id_field: "featureid"
  crs: "EPSG:4326"

datasets:
  # --- Topography ---
  - name: dem_3dep_10m           # stac_cog strategy
    variables: [elevation, slope, aspect]
    statistics: [mean]

  # --- Soils ---
  - name: gnatsgo_rasters        # stac_cog strategy
    variables: [aws0_100, rootznemc, rootznaws]
    statistics: [mean]

  - name: polaris_30m            # local_tiff strategy
    variables: [sand, silt, clay, theta_s, ksat]
    statistics: [mean]

  # --- Land Cover ---
  - name: nlcd_osn_lndcov        # nhgf_stac static strategy
    variables: [LndCov]
    statistics: [categorical]
    year: [2021]

  - name: nlcd_osn_fctimp        # nhgf_stac static strategy
    variables: [FctImp]
    statistics: [mean]
    year: [2021]

  # --- Snow ---
  - name: snodas                  # nhgf_stac temporal strategy
    variables: [SWE]
    statistics: [mean]
    time_period: ["2020-01-01", "2021-12-31"]

  # --- Climate ---
  - name: gridmet                 # climr_cat strategy
    variables: [pr, tmmx, tmmn, srad, pet, vs]
    statistics: [mean]
    time_period: ["2020-01-01", "2021-12-31"]

output:
  path: "output"
  format: netcdf
  sir_name: "{project_name}"

processing:
  engine: exactextract
  batch_size: 500
```

### Design Decisions

1. **All 7 datasets active (uncommented).** Users remove what they don't need.
   This matches the tested drb_2yr config and demonstrates all 5 access strategies.

2. **Generic placeholders for target_fabric.** `catchments.gpkg` / `featureid` /
   `EPSG:4326` — consistent with init scaffolding. Not DRB-specific.

3. **Single year for NLCD** (2021 instead of [2020, 2021]). Simpler starting
   point; multi-year is documented in the drb_2yr example.

4. **Inline comments per dataset.** Each entry gets a comment noting the access
   strategy (stac_cog, local_tiff, nhgf_stac, climr_cat) so users understand
   the data access pattern.

5. **Section headers.** Group datasets by domain (Topography, Soils, Land Cover,
   Snow, Climate) for readability.

## Files Changed

| File | Change |
|------|--------|
| `src/hydro_param/project.py` | Rewrite `generate_pipeline_template()` body (lines 145–209) |
| `tests/test_project.py` | Update `TestGeneratePipelineTemplate` assertions |

### Test Updates

- `test_template_is_valid_yaml` — no change (still checks for same top-level keys)
- `test_template_references_data_dirs` — remove `data/land_cover/` assertion
  (NLCD OSN doesn't use local paths); keep `data/fabrics/`
- `test_template_uses_project_name` — no change
- **New test:** `test_template_includes_all_dataset_strategies` — verify all 7
  dataset names are present in the generated YAML

## What Does NOT Change

- `generate_pywatershed_template()` — confirmed correct
- `generate_gitignore()` — unrelated
- `init_project()` — no logic changes needed
- `configs/examples/drb_2yr_pipeline.yml` — reference example stays as-is
