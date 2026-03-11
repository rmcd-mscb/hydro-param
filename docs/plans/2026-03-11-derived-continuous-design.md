# DerivedContinuousSpec Design — Pixel-Level Raster Math Before Zonal Stats

## Problem

The current pipeline computes zonal statistics on each raster independently,
then model plugins combine the results. For parameters like
`soil_moist_max = RootDepth × AWC`, the NHM reference multiplies rasters
pixel-by-pixel *before* zonal stats to preserve within-HRU spatial correlation.
Post-multiplying the zonal means loses this information.

## Decision

Add a new `DerivedContinuousSpec` Pydantic model that follows the same
second-pass pattern as `DerivedCategoricalSpec`: source variables are processed
in the first pass (saved as GeoTIFFs), then `DerivedContinuousSpec` variables
are processed in a second pass that reads those GeoTIFFs, aligns them, applies
an arithmetic operation, and runs continuous zonal stats on the product.

## DerivedContinuousSpec Model

```python
class DerivedContinuousSpec(BaseModel):
    name: str                          # output variable name
    sources: list[str]                 # >= 2 sibling VariableSpec names
    operation: Literal["multiply", "divide", "add", "subtract"]
    align_to: str                      # name of template source for resampling
    units: str | None = None
    scale_factor: float | None = None  # applied after zonal stats
    resampling_method: str = "nearest" # rasterio.enums.Resampling name
```

Validators:
- `len(sources) >= 2`
- `align_to in sources`

Added to `AnyVariableSpec = VariableSpec | DerivedVariableSpec | DerivedCategoricalSpec | DerivedContinuousSpec`.

## Raster Alignment

New function in `data_access.py`:

```python
def align_rasters(
    sources: list[xr.DataArray],
    template: xr.DataArray,
    method: str = "nearest",
) -> list[xr.DataArray]:
```

Uses `rioxarray .rio.reproject_match(template, resampling=Resampling[method])`.
Template is passed through unchanged; only non-template sources are reprojected.

## Arithmetic Operations

New function in `data_access.py`:

```python
def apply_raster_operation(
    sources: list[xr.DataArray],
    operation: str,
) -> xr.DataArray:
```

`functools.reduce` with the operation over the source list (left-to-right fold).
Four operations: `multiply → *`, `divide → /`, `add → +`, `subtract → -`.

## Pipeline Integration

In `_process_batch()`, after the existing `DerivedCategoricalSpec` second pass:

1. Collect `DerivedContinuousSpec` entries from `var_specs`
2. For each spec:
   - Re-read source GeoTIFFs from disk
   - Identify template (the `align_to` source)
   - Call `align_rasters()` to match all sources to template grid
   - Call `apply_raster_operation()` to produce product raster
   - Save product GeoTIFF
   - Run continuous zonal stats via `processor.process()`
   - Apply `scale_factor` if present
3. Source GeoTIFF retention logic extended to keep files needed by
   `DerivedContinuousSpec` (same pattern as `DerivedCategoricalSpec`)

## Config Example

```yaml
variables:
  - name: root_depth
    asset_key: muaggatt_rootznemc
    units: cm
  - name: awc
    asset_key: muaggatt_aws0100
    units: mm
  - name: soil_moist_product
    sources: [root_depth, awc]
    operation: multiply
    align_to: awc
    units: "cm*mm"
    scale_factor: 0.01
```

## Scope Boundaries

- No changes to the pywatershed plugin (consuming from SIR is separate work)
- No new dataset registry entries (configured per use case)
- Sources must be siblings in the same dataset (no cross-dataset references)
- Same-dataset constraint matches `DerivedCategoricalSpec` architecture

## Files Modified

| File | Change |
|------|--------|
| `src/hydro_param/dataset_registry.py` | Add `DerivedContinuousSpec`, update `AnyVariableSpec` |
| `src/hydro_param/data_access.py` | Add `align_rasters()`, `apply_raster_operation()` |
| `src/hydro_param/pipeline.py` | Second-pass block + source retention for `DerivedContinuousSpec` |
| `tests/test_data_access.py` | Unit tests for alignment and operations |
| `tests/test_pipeline.py` | Integration test for derived continuous processing |

## Related Issues

- #137 — DerivedCategoricalSpec (established the pattern)
- #154 — GLHYMPS permeability (potential consumer)
- #185 — NHM cross-check (identified the need)
