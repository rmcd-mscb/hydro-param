# Fix Temporal Forcing Detection for Per-Variable SIR Files

**Issue:** #126
**Date:** 2026-02-28

## Problem

SIR normalizes temporal data into per-variable-per-year NetCDF files
(`pr_mm_mean_2020.nc`, `tmmx_C_mean_2020.nc`).  Both `_derive_forcing()`
and `_compute_monthly_normals()` group temporal keys by base name after
stripping `_YYYY` suffixes.  This produces variable-keyed groups like
`{"pr_mm_mean": [...], "tmmx_C_mean": [...]}` instead of source-keyed
groups like `{"gridmet": [...]}`.

**Forcing (step 7):** Each variable is treated as a separate "source".
The fuzzy matcher in `_detect_forcing_dataset()` finds a 1/5 match to
gridmet config, then warns that the other 4 variables are missing.
All 5 forcing files are eventually produced (5 separate passes) but
with ~25 spurious warnings.

**Climate normals (steps 10/11):** `_compute_monthly_normals()` needs
both `tmmx_C_mean` and `tmmn_C_mean` in the same dataset to compute
monthly means.  Since each dataset contains only one variable, it never
finds both.  PET coefficients and transpiration timing fall back to
scalar defaults.

## Design Constraint: Memory

At CONUS or Alaska scale, merging 6 variables × 40+ years into a single
xr.Dataset would consume excessive memory.  The fix must keep temporal
data as per-variable datasets and look up configuration per-variable.

## Solution

### 1. Build a reverse lookup from forcing_variables.yml

Add a private helper `_build_sir_to_forcing_lookup()` that inverts the
`forcing_variables.yml` mapping:

```
Input (forcing_variables.yml):
  gridmet:
    prcp:  {sir_name: pr_mm_mean, sir_unit: mm, intermediate_unit: mm}
    tmax:  {sir_name: tmmx_C_mean, sir_unit: C, intermediate_unit: C}
    ...

Output (reverse lookup):
  {
    "pr_mm_mean":      {"prms_name": "prcp", "sir_unit": "mm", "intermediate_unit": "mm", "source": "gridmet"},
    "tmmx_C_mean":     {"prms_name": "tmax", "sir_unit": "C", "intermediate_unit": "C", "source": "gridmet"},
    "tmmn_C_mean":     {"prms_name": "tmin", "sir_unit": "C", "intermediate_unit": "C", "source": "gridmet"},
    "srad_W_m2_mean":  {"prms_name": "swrad", ...},
    "pet_mm_mean":     {"prms_name": "potet", ...},
  }
```

### 2. Refactor _derive_forcing()

Replace the current source-grouped iteration with per-variable iteration:

1. Build the reverse lookup.
2. Iterate over `ctx.temporal.items()`.
3. Strip `_YYYY` suffix to get the variable base name.
4. Look up the base name in the reverse lookup.  If not found, log at
   DEBUG (not WARNING — variables like `swe_m_mean` and `vs_m_s_mean`
   are legitimately not forcing variables) and skip.
5. Concat multi-year chunks of the same variable (same logic as now,
   just scoped to one variable at a time).
6. Apply unit conversion using the looked-up config.
7. Align the feature dimension and assign to the output dataset.

This eliminates the per-source grouping, `_detect_forcing_dataset()`
calls, and the "missing variable" warnings entirely.

### 3. Refactor _compute_monthly_normals()

Replace the source-grouped search with direct variable lookup:

1. Build the reverse lookup (or accept it as a parameter from the
   caller, since `_derive_forcing` already builds one).
2. Find the temporal keys whose base names match the `tmax` and `tmin`
   SIR names from the lookup (`tmmx_C_mean` and `tmmn_C_mean`).
3. Concat multi-year chunks of each independently.
4. Compute monthly means and convert C → F.

No need for both variables in one dataset — they're processed into
separate numpy arrays immediately.

### 4. Remove _detect_forcing_dataset()

This method exists solely to match source-keyed temporal data to
forcing config.  With per-variable lookup, it's no longer needed.

### 5. Reduce log noise

| Old behavior | New behavior |
|---|---|
| WARNING for each unmatched temporal source (swe, vs) | DEBUG: not a forcing variable |
| WARNING for each missing variable within a source (4 per pass × 5 passes = 20) | Eliminated entirely |
| 5 INFO "Step 7: merged N forcing variables" messages | 1 INFO summary with total count |

## Files Changed

| File | Change |
|------|--------|
| `derivations/pywatershed.py` | Add `_build_sir_to_forcing_lookup()`, refactor `_derive_forcing()` and `_compute_monthly_normals()`, delete `_detect_forcing_dataset()` |
| `tests/test_pywatershed_derivation.py` | Update forcing/normals tests for per-variable temporal input, add test for reverse lookup, remove `_detect_forcing_dataset` tests |

## What Does Not Change

- SIR file format (per-variable-per-year NetCDFs)
- `forcing_variables.yml` config format
- CLI temporal loading (PR #125)
- `DerivationContext.temporal` type signature
- Formatter behavior
