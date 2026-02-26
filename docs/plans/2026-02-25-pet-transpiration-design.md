# Design: Derivation Steps 10 & 11 — PET Coefficients & Transpiration Timing

**Date:** 2026-02-25
**Issue:** #83 (partial — climate-derived parameters)
**Status:** Approved

## Scope

Implement two climate-derived pywatershed derivation steps:

| Step | Name | Type | Dependencies |
|------|------|------|--------------|
| 10 | `pet_coefficients` | Climate-derived formula | hru_elev (step 3), temporal tmax/tmin |
| 11 | `climate_derived_params` | Climate-derived analysis | temporal tmin |

Both steps compute from monthly climate normals aggregated from the temporal
forcing data already available in `DerivationContext.temporal`.

## Climate Normals Source

Monthly normals are computed on-the-fly from `ctx.temporal` (gridMET or
similar multi-year data). The temporal pipeline already provides SIR-normalized
tmax/tmin in °C. Steps 10 and 11 aggregate to monthly means and convert to °F
for PRMS formulas.

If no temporal data is available, both steps fall back to published defaults
with a warning.

## Step 10: PET Coefficients (`_derive_pet_coefficients`)

### Parameters produced

- **`jh_coef`** — Jensen-Haise monthly PET coefficient, shape `(nhru, 12)`
- **`jh_coef_hru`** — Per-HRU elevation-adjusted coefficient, shape `(nhru,)`

### Algorithm

#### 1. Compute monthly climate normals

Reuse temporal data from `ctx.temporal`:
1. Concat multi-year chunks (same pattern as `_derive_forcing`).
2. Detect tmax/tmin variables by scanning `forcing_variables.yml` config
   or known SIR canonical names.
3. Group by month (`da.groupby("time.month").mean()`).
4. Convert °C → °F: `T_F = T_C * 9/5 + 32`.
5. Result: `monthly_tmax`, `monthly_tmin` each shape `(12, nhru)` in °F.

#### 2. Saturation vapor pressure (Magnus formula)

```python
def _sat_vp(temp_f: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure (hPa) from temperature in °F."""
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    return 6.1078 * np.exp(17.269 * temp_c / (temp_c + 237.3))
```

#### 3. jh_coef (per-HRU, monthly)

PRMS-IV equation 1-26:

```
jh_coef[hru, month] = 27.5 - 0.25 * (sat_vp(tmax) - sat_vp(tmin)) / sat_vp(tmax)
```

- Input: monthly_tmax, monthly_tmin in °F
- Clip to `[0.005, 0.06]` per degree-F per day (physical bounds)

#### 4. jh_coef_hru (per-HRU)

Elevation-based adjustment using July (warmest-month) normals:

```
jh_coef_hru = mean(jh_coef_july) + elevation_lapse_adjustment
```

Where the elevation adjustment accounts for reduced atmospheric pressure at
altitude increasing saturation vapor pressure deficit. Uses `hru_elev` from
step 3 (in feet).

Clip to `[0.005, 0.06]`.

### Fallback

If temporal data is unavailable:
- `jh_coef`: constant 0.014 per °F per day (NHM default)
- `jh_coef_hru`: constant 0.014 per °F per day

## Step 11: Transpiration Timing (`_derive_transp_timing`)

### Parameters produced

- **`transp_beg`** — Month transpiration begins, integer 1–12, shape `(nhru,)`
- **`transp_end`** — Month transpiration ends, integer 1–12, shape `(nhru,)`

### Algorithm

#### 1. Reuse monthly tmin normals

Same monthly_tmin array computed for step 10 (or recomputed — cheap).

#### 2. transp_beg (spring frost detection)

For each HRU:
1. Scan months January (1) through December (12).
2. Find first month where monthly mean tmin > 32°F.
3. If no month found (permanent frost), default to month 4 (April).

#### 3. transp_end (fall frost detection)

For each HRU:
1. Scan months July (7) through December (12).
2. Find first month where monthly mean tmin < 32°F.
3. If no month found (tropical, no frost), default to month 10 (October).

### Fallback

If temporal data is unavailable:
- `transp_beg`: constant 4 (April)
- `transp_end`: constant 10 (October)

## Shared Helper: `_compute_monthly_normals`

Private method shared by steps 10 and 11:

```python
def _compute_monthly_normals(
    self, ctx: DerivationContext,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute monthly mean tmax and tmin from temporal forcing.

    Returns
    -------
    tuple of (monthly_tmax, monthly_tmin) each shape (12, nhru) in °F,
    or None if temporal data is unavailable.
    """
```

This method:
1. Gets temporal data from `ctx.temporal`.
2. Detects tmax/tmin variables using `forcing_variables.yml` config.
3. Concats multi-year chunks by base name.
4. Groups by month, computes mean.
5. Converts °C → °F.
6. Returns as numpy arrays.

## Pipeline Ordering

```
... → Step 9 (soltab) → Step 10 (PET) → Step 11 (transp) →
Step 13 (defaults) → Step 14 (calibration) → Step 7 (forcing)
```

Steps 10 and 11 run before step 13 (defaults) so that defaults only fill in
where climate-derived computation failed or was skipped.

## File Changes

| Change | File | Notes |
|--------|------|-------|
| Modified | `src/hydro_param/derivations/pywatershed.py` | 4 new methods + updated `derive()` |
| Modified | `tests/test_pywatershed_derivation.py` | Tests for steps 10, 11 |

No new files, no new YAML configs, no new dependencies.

## Testing Strategy

### Step 10 (PET coefficients)

- **Unit tests:**
  - `_sat_vp` correctness: known values (32°F → 6.11 hPa, 212°F → 1013 hPa)
  - `jh_coef` formula: hand-computed values for known tmax/tmin
  - Range clipping: values outside [0.005, 0.06] are clipped
  - Single-HRU edge case
- **Fallback test:** no temporal data → defaults (0.014)
- **Integration test:** full `derive()` with synthetic temporal data produces
  `jh_coef` shape (nhru, 12) and `jh_coef_hru` shape (nhru,)

### Step 11 (transpiration timing)

- **Unit tests:**
  - Warm climate (no frost): transp_beg=1, transp_end=default
  - Cold climate (short season): transp_beg=5 or 6, transp_end=9
  - Temperate climate: transp_beg=3–4, transp_end=10–11
  - All values in [1, 12] range
- **Fallback test:** no temporal data → defaults (beg=4, end=10)
- **Edge case:** permanent frost (all tmin < 32°F) → sensible defaults
- **Integration test:** full pipeline produces integer month values

## Out of Scope

- Steps 6 (waterbody), 12 (routing) — separate design needed
- Separate normals dataset support (can be added later)
- YAML-configurable algorithm constants
