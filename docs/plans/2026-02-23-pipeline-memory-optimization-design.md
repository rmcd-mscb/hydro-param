# Pipeline Memory Optimization + STAC Query Reuse

**Date:** 2026-02-23
**Status:** Approved
**Scope:** Fix OOM in stage 4 batch processing; reduce redundant STAC queries

## Problem

The pipeline OOM-kills during gNATSGO processing (dataset 2/5, batch 8) on a
32 GB machine. The log cuts off mid-raster-load with no Python traceback — the
hallmark of a Linux OOM kill.

Root causes traced to `_process_batch` in `pipeline.py` and `save_to_geotiff`
in `data_access.py`:

1. **`source_cache` never releases raw VariableSpec entries.** For gNATSGO,
   all 3 variables (~1.1–1.25 GB each) accumulate. Cleanup logic (lines
   422–430) only runs for `DerivedVariableSpec`.
2. **Raw variable DataArrays survive past save_to_geotiff.** The `del da`
   (line 399) only fires for derived vars. For raw vars, the array stays in
   scope through zonal stats (which reads from the GeoTIFF file).
3. **`save_to_geotiff` does `da.copy()`** just to strip `_FillValue` from
   attrs, momentarily doubling memory per raster.
4. **No GC between variables** — circular refs from xarray/numpy accumulate.

Secondary issue: gNATSGO queries the same STAC catalog 3× per batch (once per
variable). The item list is identical; only the asset_key differs.

## Design

### Section 1: Memory leak fixes

**1a. Release source_cache for raw vars after save_to_geotiff.**
After caching the raw variable and saving to GeoTIFF, check if any later
`DerivedVariableSpec` needs it as a source. If not, delete from `source_cache`
immediately.

**1b. `del da` for raw vars after save.**
Extend the existing `del da` (currently derived-only) to also cover raw vars
once the GeoTIFF is written.

**1c. Fix `save_to_geotiff` copy.**
Replace `clean = da.copy()` with in-place attr/encoding pop of `_FillValue`,
call `rio.to_raster`, then restore the original values. No full-array copy.

**1d. `gc.collect()` after each variable's zonal stats.**
Belt-and-suspenders for circular refs from xarray/numpy internals.

### Section 2: STAC query reuse

**2a. Extract `query_stac_items()` from `fetch_stac_cog`.**
New public function in `data_access.py`:
`query_stac_items(entry, bbox) -> list[pystac.Item]`
Handles client creation, signing, search, GSD filtering. Returns signed items.

**2b. Add optional `items` parameter to `fetch_stac_cog`.**
`fetch_stac_cog(entry, bbox, *, asset_key=None, items=None)`
If `items` is provided, skip the STAC query. Fully backward-compatible.

**2c. Cache STAC items per-batch in `_process_batch`.**
For `stac_cog` strategy, call `query_stac_items()` once before the variable
loop, pass items into each `_fetch()` call. Cache lives only for one batch
iteration — no long-lived state.

## Scope boundaries

**In scope:** 4 memory fixes + STAC query reuse (2 files).

**Not in scope:** batch size tuning, chunked raster reading, temporal pathway
changes, fetch-per-variable pattern refactoring.

## Files modified

- `src/hydro_param/pipeline.py` — `_process_batch` memory management + items caching
- `src/hydro_param/data_access.py` — `query_stac_items`, `fetch_stac_cog` items param, `save_to_geotiff` fix

## Testing

- Existing tests pass unchanged (backward-compatible)
- Unit test for `query_stac_items` → `fetch_stac_cog` round-trip
- Unit test verifying `save_to_geotiff` doesn't modify input DataArray attrs

## Risk

- Section 1: Very low — earlier cleanup of objects already cleaned up later
- Section 2: Low — pure extraction + additive parameter; existing callers unchanged
