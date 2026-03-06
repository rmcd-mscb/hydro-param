# Design: Derive soil_rechr_max_frac from gNATSGO AWC ratio

**Date:** 2026-03-04
**Issue:** #151
**Status:** Approved

## Problem

`soil_rechr_max_frac` (fraction of soil moisture in the recharge zone) is set to a
uniform default of 0.4 for all HRUs. The authoritative derivation spec defines it as:

```
soil_rechr_max_frac = AWC_upper_18inches / AWC_total_rootzone
```

gNATSGO already provides both variables:
- `aws0_50` — available water storage, 0–50 cm (upper ~18 inches = recharge zone)
- `aws0_100` — available water storage, 0–100 cm (full root zone)

The ratio `aws0_50 / aws0_100` directly approximates the recharge-to-total fraction.

## Approach

### 1. Dataset registry (`soils.yml`)

Add `aws0_50` as a variable in the `gnatsgo_rasters` dataset entry, mirroring the
existing `aws0_100` entry:

```yaml
- name: aws0_50
  band: 1
  units: "mm"
  long_name: "Available water storage 0-50cm"
  native_name: "aws0_50"
  categorical: false
  asset_key: "aws0_50"
```

### 2. Derivation code (`_derive_soils()`)

Replace the constant-default block for `soil_rechr_max_frac` with:

1. Look up `aws0_50_mm_mean` and `aws0_100_mm_mean` in the SIR via `find_variable()`
2. If both present: compute `ratio = aws0_50 / np.where(aws0_100 > 0, aws0_100, 1.0)`
3. Clip to [0.1, 0.9] (physical bounds — can't be 0% or 100% of root zone)
4. If either is missing: fall back to 0.4 default (existing behavior)

Guard against division by zero with `np.where(aws0_100 > 0, aws0_100, 1.0)` and
set those HRUs to the default 0.4 (zero total AWC means no data).

### 3. Reference doc (`pywatershed_dataset_param_map.yml`)

Update `soil_rechr_max_frac` entry:
- `derivation_type`: `default` → `derived_formula`
- `method`: document the formula and fallback
- `source_dataset`: add gNATSGO reference

### 4. Tests

- **Derived path:** Provide both `aws0_50_mm_mean` and `aws0_100_mm_mean` in mock SIR,
  verify ratio computation and clipping
- **Division by zero:** HRU with aws0_100 = 0 gets default 0.4
- **Fallback path:** Only aws0_100 present (no aws0_50) → default 0.4
- **No data path:** Neither variable present → default 0.4

## Files changed

| File | Change |
|------|--------|
| `src/hydro_param/data/datasets/soils.yml` | Add `aws0_50` variable |
| `src/hydro_param/derivations/pywatershed.py` | AWC ratio in `_derive_soils()` |
| `docs/reference/pywatershed_dataset_param_map.yml` | Update derivation spec |
| `tests/test_pywatershed_derivation.py` | New test cases |

## Not in scope

- Issue #152 (TWI-based carea_max/smidx_coef) — deferred, requires DEM flow
  accumulation infrastructure
- No new dependencies required
