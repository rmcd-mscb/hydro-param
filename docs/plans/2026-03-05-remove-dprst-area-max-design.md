# Design: Remove dprst_area_max from Output (#162)

## Problem

The parameter audit found that `dprst_area_max` is produced by step 6
(`_derive_waterbody()`) but is **not** an input parameter to pywatershed.
pywatershed computes it internally as `dprst_frac × hru_area` in both
`PRMSRunoff` and `PRMSSoilzone`. Outputting it is redundant and confusing.

## Investigation

- **pywatershed `get_parameters()`**: Does NOT list `dprst_area_max`
- **pywatershed `variables.yaml`**: Lists it as a runtime *variable*, not a parameter
- **PRMSRunoff/PRMSSoilzone**: Both compute `dprst_area_max = dprst_frac * hru_area`
- **hydro-param**: Computes it in step 6 from waterbody overlay (m² → acres)

## Decision

**Remove `dprst_area_max` entirely.** The fraction (`dprst_frac`) and area
(`hru_area`) are both already output — anyone needing absolute waterbody area
can compute `dprst_frac × hru_area`.

## Changes

| File | Change |
|------|--------|
| `src/hydro_param/derivations/pywatershed.py` | Remove from `_waterbody_defaults()` and `_derive_waterbody()` |
| `src/hydro_param/data/pywatershed/parameter_metadata.yml` | Remove `dprst_area_max` entry |
| `src/hydro_param/pywatershed_config.py` | Remove `dprst_area_max` field from `PwsWaterbodyConfig` |
| `tests/test_pywatershed_derivation.py` | Remove `dprst_area_max` assertions |
