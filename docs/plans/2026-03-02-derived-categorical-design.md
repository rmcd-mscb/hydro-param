# Design: Derived Categorical Pipeline Strategy

**Issue:** #135
**Date:** 2026-03-02

## Problem

The pipeline currently processes each variable independently through
continuous zonal statistics (area-weighted mean).  For soil texture
classification, pixels must be classified into USDA texture classes
*before* zonal aggregation (classify-then-aggregate) to produce
accurate per-HRU texture class fractions.  This requires combining
multiple source bands (sand%, silt%, clay%) into a single classified
raster, then running categorical zonal stats.

## Design

### User-Facing Config

Users request derived categorical variables the same way they request
derived continuous variables (slope, aspect) — by name in the variables
list.  The registry defines the derivation recipe:

```yaml
# pipeline.yml
- name: polaris_30m
  variables: [sand, silt, clay, theta_s, ksat, soil_texture]
  statistics: [mean]
```

The user adds `soil_texture` to the variables list.  The registry knows
it derives from `[sand, silt, clay]` using the `usda_texture_triangle`
method and produces categorical output.

### DerivedCategoricalSpec

New Pydantic model in `dataset_registry.py`:

```python
class DerivedCategoricalSpec(BaseModel):
    name: str                # "soil_texture"
    sources: list[str]       # ["sand", "silt", "clay"]
    method: str              # "usda_texture_triangle"
    units: str               # "class"
    long_name: str           # "USDA soil texture classification"
```

Registry entry in `soils.yml`:

```yaml
polaris_30m:
  variables: [...]
  derived_categorical_variables:
    - name: soil_texture
      sources: [sand, silt, clay]
      method: usda_texture_triangle
      units: "class"
      long_name: "USDA soil texture classification"
```

### Stage 2 Resolution

When the user requests `soil_texture`:

1. Registry looks up the name in `derived_categorical_variables`.
2. Returns a `DerivedCategoricalSpec` alongside regular `VariableSpec`
   and `DerivedVariableSpec` entries.
3. Auto-includes source variables (`sand`, `silt`, `clay`) if the user
   didn't explicitly list them — same pattern as slope auto-including
   elevation.

### Classification Function

`usda_texture_triangle()` lives in `data_access.py` alongside
`derive_slope` / `derive_aspect`.  Registered in a new
`CATEGORICAL_DERIVATION_FUNCTIONS` dict:

```python
CATEGORICAL_DERIVATION_FUNCTIONS = {
    "soil_texture": classify_usda_texture_raster,
}
```

The function takes 3 aligned `xr.DataArray` inputs (sand, silt, clay),
applies the USDA texture triangle decision tree per-pixel, and returns
a single-band integer raster with class codes (1-12).  NaN pixels in
any source band classify as NaN (nodata).

### Pipeline Processing (`_process_batch()`)

Processing order within a dataset's variable loop:

1. **`VariableSpec` entries first** — fetch, save GeoTIFF, continuous
   zonal stats (one at a time, normal flow).
2. **`DerivedVariableSpec` entries next** — existing pattern
   (slope/aspect from elevation).
3. **`DerivedCategoricalSpec` entries last** — re-read source GeoTIFFs
   from disk (already saved in step 1), classify per-pixel, save
   classified GeoTIFF, run categorical zonal stats (`categorical=True`).

This ordering is the memory management strategy: source rasters are
processed and released one at a time during step 1, then re-read from
local disk for classification.  Peak memory during classification is
3 float32 arrays + 1 int8 array, briefly.

### SIR Output

Categorical zonal stats produce per-class fraction columns.  These land
in the SIR as `soil_texture_frac_sand`, `soil_texture_frac_loamy_sand`,
etc. — exactly the format `_compute_soil_type()` already expects as its
preferred input path (fraction columns → argmax → PRMS soil_type).

The #121 fallback (aggregate-then-classify from mean percentages)
becomes a true fallback — only used when the user didn't request
`soil_texture` in their pipeline config.

### Memory Management

- Process derived categorical variables **last** within each batch.
- Re-read source GeoTIFFs from disk rather than holding all sources in
  the source cache simultaneously.
- Release all source arrays immediately after classification.
- Peak memory: 3 float32 + 1 int8, briefly during classification.

### Error Handling

- **Missing sources in user config:** Stage 2 auto-includes source
  variables needed by derived categorical entries.
- **Failed source fetch:** If a source GeoTIFF is missing at
  classification time (fetch failed for that batch), skip classification
  for that batch and log a warning (fault-tolerance).
- **NaN pixels:** Any source band NaN → classified pixel is NaN
  (nodata), excluded from zonal fractions by gdptools.

## Affected Components

| Component | File | Change |
|-----------|------|--------|
| `DerivedCategoricalSpec` | `dataset_registry.py` | New Pydantic model |
| Registry parsing | `dataset_registry.py` | Parse `derived_categorical_variables` |
| Stage 2 resolution | `dataset_registry.py` / `pipeline.py` | Resolve names, auto-include sources |
| `classify_usda_texture_raster()` | `data_access.py` | New raster classification function |
| `CATEGORICAL_DERIVATION_FUNCTIONS` | `data_access.py` | New function registry |
| `_process_batch()` ordering | `pipeline.py` | Process derived categorical last |
| Registry entry | `soils.yml` | Add `derived_categorical_variables` |
| SIR schema | `sir.py` | Handle categorical derived variables |
| Tests | Multiple test files | Classifier, integration, config |

## Scope Boundaries

- MVP: USDA texture triangle for POLARIS only.
- The `CATEGORICAL_DERIVATION_FUNCTIONS` registry is extensible for
  future multi-band classifications (e.g., gNATSGO texture).
- No changes to temporal processing pathway.
- No changes to existing continuous or single-source derived processing.
