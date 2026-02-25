# Design: Derivation Steps 5, 9, 14 — Soils, Soltab, Calibration Seeds

**Date:** 2026-02-25
**Issue:** #83 (partial — pure computation + soils subset)
**Status:** Approved

## Scope

Implement three of the eight remaining pywatershed derivation steps from
issue #83:

| Step | Name | Type | Dependencies |
|------|------|------|--------------|
| 9 | `solar_radiation_tables` | Pure computation | hru_lat, hru_slope, hru_aspect (step 3) |
| 5 | `soils_zonal_stats` | SIR data + lookup | gNATSGO zonal stats in SIR |
| 14 | `calibration_seeds` | YAML-driven formulas | Steps 3, 4, 5 outputs |

Three separate PRs, each following TDD. Implementation order: 9 → 5 → 14.

## Step 9: Solar Radiation Tables (`_derive_soltab`)

### Approach

Port pywatershed's `PRMSSolarGeometry.compute_soltab()` (Swift 1976 algorithm)
into hydro-param as a standalone pure-function module. The pywatershed class
requires `Control`, `Parameters`, and `Process` infrastructure that hydro-param
does not have. The algorithm itself is a ~100-line static method with clear
inputs and outputs — clean to extract.

### New module: `src/hydro_param/solar.py`

Module-level constants ported from pywatershed's `solar_constants.py`:

- `ndoy = 366`
- `solar_declination` — array of length 366, Fourier approximation
- `r1` — solar constant adjusted for orbital eccentricity, length 366
- `pi_12 = 12 / pi`

Public function:

```python
def compute_soltab(
    slopes: np.ndarray,   # decimal fraction (rise/run), shape (nhru,)
    aspects: np.ndarray,   # degrees, 0=north, shape (nhru,)
    lats: np.ndarray,      # decimal degrees, shape (nhru,)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute potential solar radiation tables (Swift 1976).

    Returns:
        soltab_potsw: potential SW radiation on sloped surface,
            cal/cm2/day (Langleys), shape (ndoy, nhru)
        soltab_horad_potsw: potential SW radiation on horizontal surface,
            cal/cm2/day (Langleys), shape (ndoy, nhru)
        soltab_sunhrs: hours of direct sunlight,
            shape (ndoy, nhru)
    """
```

Internal helpers (module-level, private):

- `_compute_t(lats, solar_declination)` — sunrise equation, returns hour
  angle array (ndoy, nhru)
- `_func3(v, w, x, y)` — Swift 1976 equation 6, potential solar radiation
  on a surface (cal/cm2/day)

### Derivation step: `_derive_soltab(ctx, ds)`

1. Read `hru_lat`, `hru_slope`, `hru_aspect` from the dataset (derived in
   step 3).
2. `hru_slope` is stored as decimal fraction (tan of slope angle). This is
   what `compute_soltab` expects — pywatershed's `hru_slope` parameter is
   also rise/run (decimal fraction), and the algorithm applies `arctan`
   internally.
3. Call `compute_soltab(slopes, aspects, lats)`.
4. Add `soltab_potsw` and `soltab_horad_potsw` as 2D `xr.DataArray` with
   dims `(ndoy, nhru)`. Also add `soltab_sunhrs`.
5. Units: cal/cm2/day (Langleys) — matches PRMS convention.

### Algorithm summary (Swift 1976)

For each HRU and each day of year (1–366):

1. Compute the latitude of an "equivalent slope" — a horizontal surface at a
   different latitude that receives the same radiation as the actual sloped
   surface (Lee 1963, equation 13).
2. Compute the longitude offset between the actual location and the
   equivalent slope (Lee 1963, equation 12).
3. Compute sunrise/sunset hour angles on both the equivalent slope and the
   horizontal surface at the HRU latitude.
4. Clip slope sunrise/sunset to not exceed horizontal sunrise/sunset (the
   slope cannot see the sun before/after the horizontal surface does).
5. Integrate potential radiation using `func3` (Swift 1976, equation 6).
6. Handle wrap-around cases where the slope's effective sunrise/sunset
   crosses the noon meridian.

All operations are vectorized over (ndoy, nhru) using numpy broadcasting.

### References

- Swift, L.W. Jr. (1976). Algorithm for solar radiation on mountain slopes.
  Water Resources Research, 12(1):108.
- Lee, R. (1963). Evaluation of solar beam irradiation as a climatic parameter
  of mountain watersheds. Colorado State University Hydrology Paper No. 2.
- Markstrom et al. (2015). PRMS-IV, the precipitation-runoff modeling system,
  version 4. USGS TM 6-B7.
- pywatershed source: `EC-USGS/pywatershed`, `pywatershed/atmosphere/prms_solar_geometry.py`

## Step 5: Soils Zonal Stats (`_derive_soils`)

### What the pipeline provides

The gNATSGO dataset is already processed via the `stac_cog` strategy. The SIR
contains zonal stats variables from gNATSGO following the canonical naming
convention (e.g., `soil_texture_frac_*`, `awc_mm_mean`, `soil_depth_cm_mean`).

### Derivation step: `_derive_soils(ctx, ds)`

Three output parameters:

**`soil_type`** (integer, 1=sand, 2=loam, 3=clay):
1. Scan SIR for soil texture fraction variables matching `soil_texture_frac_*`
   pattern (same approach as step 4's `lndcov_frac_*` scan).
2. Compute majority class via `argmax` across fraction variables.
3. Reclassify USDA texture class → PRMS soil_type using the existing
   `soil_texture_to_prms_type.yml` lookup table.
4. Fallback: if no fraction variables found, look for a single
   `soil_texture` or `soil_texture_majority` variable (direct dominant class).

**`soil_moist_max`** (inches):
1. Read AWC (available water capacity) and soil depth from SIR.
2. Compute `soil_moist_max = awc * soil_depth`.
3. Convert from metric (mm or cm) to inches.
4. Clip to physically reasonable range (e.g., 0.5–20.0 inches).

**`soil_rechr_max_frac`** (dimensionless, 0–1):
1. If layer-resolved AWC is available in SIR: top-layer AWC / total AWC.
2. Otherwise: literature default of 0.4 (Regan et al. 2018).
3. Clip to [0.01, 1.0].

### Fallback behavior

Matches step 4 pattern:
- Categorical fractions preferred → single dominant value fallback
- Missing variables → `KeyError` with clear message (consistent with PR #84
  "explicit field missing → raise" pattern)
- `soil_rechr_max_frac` is the exception: falls back to 0.4 with warning
  since layer data is often unavailable

### Lookup tables

No new tables needed. Uses existing `configs/lookup_tables/soil_texture_to_prms_type.yml`
via the `_load_lookup_table()` cache mechanism.

### SIR variable naming

The exact SIR variable names for gNATSGO outputs depend on what the pipeline's
SIR normalization (PR #70) produces. The derivation step will document expected
variable name patterns and raise `KeyError` with a descriptive message if
they're not found. Expected patterns:

- `soil_texture_frac_{class}` — fraction of each USDA texture class per HRU
- `awc_mm_mean` or `awc_cm_mean` — available water capacity (zonal mean)
- `soil_depth_cm_mean` or `soil_depth_mm_mean` — total soil depth (zonal mean)

## Step 14: Calibration Seeds (`_derive_calibration_seeds`)

### Approach

YAML-driven seed definitions. A new config file defines each calibration
parameter's computation method, required inputs, default value, and valid
range. The derivation step reads this file and evaluates each seed using a
safe dispatch mechanism.

### New file: `configs/lookup_tables/calibration_seeds.yml`

```yaml
name: calibration_seeds
description: "Physically-based initial values for calibration parameters"
source: "Regan et al. 2018 (TM6-B9), Hay et al. 2023"

seeds:
  carea_max:
    method: linear
    params: {input: hru_percent_imperv, scale: 0.6, offset: 0.2}
    range: [0.0, 1.0]
    default: 0.4

  smidx_coef:
    method: exponential_scale
    params: {input: hru_slope, scale: 0.005, exponent: 3.0}
    range: [0.001, 0.06]
    default: 0.01

  smidx_exp:
    method: constant
    params: {value: 0.3}
    range: [0.1, 0.8]
    default: 0.3

  soil2gw_max:
    method: fraction_of
    params: {input: soil_moist_max, fraction: 0.1}
    range: [0.0, 5.0]
    default: 0.1

  ssr2gw_rate:
    method: constant
    params: {value: 0.1}
    range: [0.0, 1.0]
    default: 0.1

  ssr2gw_exp:
    method: constant
    params: {value: 1.0}
    range: [0.0, 3.0]
    default: 1.0

  slowcoef_lin:
    method: constant
    params: {value: 0.015}
    range: [0.001, 0.5]
    default: 0.015

  slowcoef_sq:
    method: constant
    params: {value: 0.1}
    range: [0.0, 1.0]
    default: 0.1

  fastcoef_lin:
    method: constant
    params: {value: 0.09}
    range: [0.001, 0.8]
    default: 0.09

  fastcoef_sq:
    method: constant
    params: {value: 0.8}
    range: [0.0, 1.0]
    default: 0.8

  pref_flow_den:
    method: constant
    params: {value: 0.0}
    range: [0.0, 0.1]
    default: 0.0

  gwflow_coef:
    method: constant
    params: {value: 0.015}
    range: [0.001, 0.5]
    default: 0.015

  gwsink_coef:
    method: constant
    params: {value: 0.0}
    range: [0.0, 1.0]
    default: 0.0

  snarea_thresh:
    method: fraction_of
    params: {input: soil_moist_max, fraction: 0.8}
    range: [0.0, 200.0]
    default: 50.0

  rain_cbh_adj:
    method: constant
    params: {value: 1.0}
    range: [0.5, 2.0]
    default: 1.0

  snow_cbh_adj:
    method: constant
    params: {value: 1.0}
    range: [0.5, 2.0]
    default: 1.0

  tmax_cbh_adj:
    method: constant
    params: {value: 0.0}
    range: [-10.0, 10.0]
    default: 0.0

  tmin_cbh_adj:
    method: constant
    params: {value: 0.0}
    range: [-10.0, 10.0]
    default: 0.0

  tmax_allrain_offset:
    method: constant
    params: {value: 1.0}
    range: [0.0, 10.0]
    default: 1.0

  adjmix_rain:
    method: constant
    params: {value: 1.0}
    range: [0.6, 1.4]
    default: 1.0

  dday_slope:
    method: constant
    params: {value: 0.4}
    range: [0.2, 0.9]
    default: 0.4

  dday_intcp:
    method: constant
    params: {value: -40.0}
    range: [-60.0, 10.0]
    default: -40.0
```

### Formula dispatch

A small dict of named operations — no `eval()`, no arbitrary code execution:

```python
_SEED_METHODS: dict[str, Callable] = {
    "linear": lambda ds, p: p["scale"] * ds[p["input"]] + p["offset"],
    "exponential_scale": lambda ds, p: p["scale"] * np.exp(p["exponent"] * ds[p["input"]]),
    "fraction_of": lambda ds, p: p["fraction"] * ds[p["input"]],
    "constant": lambda ds, p: p["value"],
}
```

This gives users the ability to tune coefficients in YAML without touching
Python, while keeping execution safe and predictable.

### Derivation step: `_derive_calibration_seeds(ctx, ds)`

1. Load `calibration_seeds.yml` via `_load_lookup_table()`.
2. Iterate over `seeds` entries.
3. For each seed:
   a. Check if all `params.input` fields exist in the dataset.
   b. If yes: dispatch to the named method, compute the value.
   c. If no: use `default` value and log a warning.
   d. Clip result to `range`.
   e. Add as `xr.DataArray` to dataset.
4. Skip seeds whose parameter name already exists in the dataset (allows
   prior steps or user overrides to take precedence).

### Graceful degradation

If a required input is missing (e.g., `soil_moist_max` not yet computed
because step 5 was skipped), the seed falls back to its `default` value with
a warning. This means step 14 produces useful output even when upstream steps
are incomplete — though with less physically-based values.

## Integration into `PywatershedDerivation.derive()`

The `derive()` method currently runs steps 1 → 2 → 3 → 4 → 8 → 13. With
these additions the sequence becomes:

```
1 (geometry) → 2 (topology) → 3 (topo) → 4 (landcover) →
5 (soils) → 8 (lookups) → 9 (soltab) → 13 (defaults) → 14 (calibration)
```

Step 9 (soltab) runs after step 8 because it depends only on step 3 outputs
and has no interaction with lookups. Step 14 (calibration) runs last because
it consumes outputs from all prior steps.

## File changes summary

| Change | File | Notes |
|--------|------|-------|
| New module | `src/hydro_param/solar.py` | Soltab algorithm (Swift 1976) |
| New lookup | `configs/lookup_tables/calibration_seeds.yml` | Seed definitions |
| Modified | `src/hydro_param/derivations/pywatershed.py` | 3 new step methods + updated `derive()` |
| New tests | `tests/test_solar.py` | Unit tests for soltab algorithm |
| Modified | `tests/test_pywatershed_derivation.py` | Tests for steps 5, 9, 14 |

## Testing strategy

### Step 9 (soltab)

- **Unit tests** (`test_solar.py`):
  - Flat surface (slope=0): `soltab_potsw == soltab_horad_potsw`
  - Equator at equinox: symmetric sunrise/sunset
  - South-facing slope at mid-latitude: more radiation than horizontal
  - North-facing steep slope: reduced radiation
  - Single-HRU edge case
  - All outputs non-negative
- **Reference validation**: Download `parameters_PRMSSolarGeometry.nc` from
  pywatershed's DRB 2-year test data and compare hydro-param's output for
  matching inputs (hru_lat, hru_slope, hru_aspect). Tolerance: < 0.1%
  relative difference.
- **Integration test**: Run through full `derive()` pipeline, verify
  `soltab_potsw` has shape (366, nhru) and correct dims.

### Step 5 (soils)

- **Unit tests**:
  - Majority class extraction from fraction variables (argmax)
  - USDA texture → PRMS soil_type reclassification (all 12 classes)
  - `soil_moist_max` unit conversion (mm → inches)
  - `soil_rechr_max_frac` with and without layer data (default fallback)
  - Missing SIR variables → `KeyError` with descriptive message
- **Integration test**: Synthetic SIR with gNATSGO-like variables through
  full pipeline.

### Step 14 (calibration seeds)

- **Unit tests**:
  - Each seed method type: linear, exponential_scale, fraction_of, constant
  - Range clipping (values outside range get clipped)
  - Missing input fallback to default (with warning)
  - Pre-existing parameter in dataset → not overwritten
  - Unknown method name in YAML → clear error
- **YAML validation test**: Load `calibration_seeds.yml`, verify all entries
  have required fields (method, params, range, default).
- **Integration test**: Full pipeline produces all ~22 calibration parameters.

## Out of scope

- Steps 6 (waterbody), 7 (forcing), 10 (PET), 11 (transp), 12 (routing) —
  tracked in issue #83, separate design needed
- Changes to SIR normalization or pipeline stages
- New dataset registry entries
