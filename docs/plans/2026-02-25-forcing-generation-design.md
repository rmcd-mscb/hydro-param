# Design: Step 7 — Forcing Generation (`_derive_forcing`)

**Date:** 2026-02-25
**Issue:** #83 (partial — forcing generation)
**Status:** Approved

## Scope

Formalize step 7 (forcing generation) as a `_derive_forcing()` method inside
`PywatershedDerivation`. Takes SIR-normalized temporal NetCDFs, renames
variables to PRMS names, converts units, and merges into the derived dataset.
YAML config defines per-source-dataset variable mappings.

**Variables in scope:** prcp, tmax, tmin (CBH required) + swrad, potet.

## Architecture

### Data flow

```
SIR temporal files (pr_mm_mean, tmmx_C_mean, tmmn_C_mean, srad_W_m2_mean, pet_mm_mean)
  → _derive_forcing() reads YAML mappings for the source dataset
  → renames: pr_mm_mean → prcp, tmmx_C_mean → tmax, etc.
  → unit converts to intermediate units (mm stays mm, C stays C, W/m2 → Langleys/day)
  → merges into derived xr.Dataset
  → formatter write_forcing_netcdf() does final PRMS conversion (mm→in, C→F)
```

### Why SIR-normalized (not raw) temporal files

Stage 5 SIR normalization already converts temperatures from K to C and
applies canonical naming. Using SIR-normalized files means the plugin only
maps `{SIR canonical name → PRMS name}` — a clean two-phase boundary. The
plugin never sees source-native names like `daily_maximum_temperature`.

## Changes to `DerivationContext`

Add optional `temporal` field to `plugins.py`:

```python
@dataclass(frozen=True)
class DerivationContext:
    sir: xr.Dataset
    temporal: dict[str, xr.Dataset] | None = None  # NEW
    fabric: gpd.GeoDataFrame | None = None
    segments: gpd.GeoDataFrame | None = None
    fabric_id_field: str = "nhm_id"
    segment_id_field: str | None = None
    config: dict = field(default_factory=dict)
    lookup_tables_dir: Path | None = None
```

The pipeline CLI populates `temporal` from SIR-normalized temporal files. The
field defaults to `None` for backward compatibility — existing callers that
don't pass temporal data continue to work unchanged.

## New YAML: `configs/lookup_tables/forcing_variables.yml`

```yaml
name: forcing_variables
description: "SIR temporal variable to PRMS forcing variable mappings"
source: "gridMET via climateR-catalogs"

datasets:
  gridmet:
    variables:
      prcp:
        sir_name: pr_mm_mean
        sir_unit: mm
        intermediate_unit: mm
      tmax:
        sir_name: tmmx_C_mean
        sir_unit: C
        intermediate_unit: C
      tmin:
        sir_name: tmmn_C_mean
        sir_unit: C
        intermediate_unit: C
      swrad:
        sir_name: srad_W_m2_mean
        sir_unit: W/m2
        intermediate_unit: Langleys/day
      potet:
        sir_name: pet_mm_mean
        sir_unit: mm
        intermediate_unit: in
```

Additional dataset sections (daymet, conus404_ba) can be added later using
the same structure.

## `_derive_forcing()` method

1. If `context.temporal` is None or empty → log info, return ds unchanged.
2. Load `forcing_variables.yml` via `_load_lookup_table()`.
3. Detect source dataset by matching SIR variable names against each
   `datasets.*.variables.*.sir_name` entry. Pick the dataset with the most
   matches.
4. Concat multi-year chunks along `time` dimension (same grouping logic as
   current `merge_temporal_into_derived`: strip `_YYYY` suffix, sort by
   first time value).
5. For each mapped variable:
   a. Find the SIR variable by `sir_name`.
   b. Rename to the PRMS name (dict key).
   c. Convert `sir_unit` → `intermediate_unit` if they differ.
   d. Align feature dimension to match derived dataset's id_field.
6. Add temporal DataArrays to derived dataset.
7. Log summary of variables merged (count, time range).

### Missing variable handling

- Missing SIR variable that is mapped → log warning, skip that variable.
- No temporal data at all → log info, return ds unchanged.
- Unknown SIR variables (not in any mapping) → ignore (they stay in temporal
  but don't get merged).

## Formatter extension

Extend `_FORCING_VARS` in `formatters/pywatershed.py`:

```python
_FORCING_VARS: dict[str, tuple[str, str]] = {
    "prcp": ("mm", "in"),
    "tmax": ("C", "F"),
    "tmin": ("C", "F"),
    "swrad": ("Langleys/day", "Langleys/day"),  # identity — already converted
    "potet": ("in", "in"),                       # identity — already converted
}
```

The formatter writes one file per variable into the forcing directory.

## Unit conversions

| Variable | SIR unit | Intermediate | Formatter final | New? |
|----------|----------|-------------|-----------------|------|
| prcp | mm | mm | in | No |
| tmax | C | C | F | No |
| tmin | C | C | F | No |
| swrad | W/m2 | Langleys/day | Langleys/day | Yes |
| potet | mm | in | in | No |

New `units.py` entry: `("W/m2", "Langleys/day")` = `× 2.065`
(1 W/m2 ≈ 2.065 cal/cm2/day = 2.065 Langleys/day).

## Deprecation of `merge_temporal_into_derived()`

The existing module-level function is replaced by `_derive_forcing()`.
Mark it deprecated with a log warning on call. Remove in a future release.

## `derive()` integration

```python
def derive(self, context: DerivationContext) -> xr.Dataset:
    ...
    # Step 9: Solar radiation tables
    ds = self._derive_soltab(context, ds)
    # Step 13: Defaults
    ds = self._apply_defaults(ds, nhru)
    # Step 14: Calibration seeds
    ds = self._derive_calibration_seeds(context, ds)
    # Step 7: Forcing (temporal merge) — runs late because it only merges,
    # no downstream steps depend on forcing variables
    ds = self._derive_forcing(context, ds)
    # Parameter overrides
    ...
    return ds
```

Step 7 runs after step 14 because no other derivation step depends on
forcing variables. This avoids requiring temporal data for static parameter
derivation.

## File changes summary

| Change | File | Notes |
|--------|------|-------|
| Modified | `src/hydro_param/plugins.py` | Add `temporal` field to `DerivationContext` |
| New | `src/hydro_param/data/lookup_tables/forcing_variables.yml` | Variable mappings |
| Modified | `src/hydro_param/derivations/pywatershed.py` | `_derive_forcing()`, deprecate `merge_temporal_into_derived()` |
| Modified | `src/hydro_param/formatters/pywatershed.py` | Extend `_FORCING_VARS` |
| Modified | `src/hydro_param/units.py` | Add W/m2 → Langleys/day |
| Modified | `tests/test_pywatershed_derivation.py` | Tests for `_derive_forcing()` |

## Testing strategy

- **Unit tests for `_derive_forcing()`:**
  - Synthetic temporal datasets with known values → verify rename + conversion
  - Multi-year concat → verify time dimension ordering
  - Feature dim alignment → verify nhm_id → nhru rename
  - Missing temporal data → returns ds unchanged
  - Missing SIR variable → logs warning, skips variable
  - swrad W/m2 → Langleys/day conversion accuracy
  - potet mm → in conversion accuracy
- **YAML validation test:** Load forcing_variables.yml, verify structure
- **Formatter test:** Verify extended `_FORCING_VARS` writes swrad/potet files
- **Integration test:** Full `derive()` with temporal context → formatter output

## Out of scope

- Daymet / CONUS404-BA dataset mappings (add later to YAML)
- Steps 6 (waterbody), 10 (PET derivation), 11 (transp), 12 (routing)
- Changes to pipeline temporal processing
- CLI wiring updates (separate task — update `cli.py` to populate
  `DerivationContext.temporal` from SIR files)
