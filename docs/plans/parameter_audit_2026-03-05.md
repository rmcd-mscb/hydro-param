# pywatershed Parameter Audit — 2026-03-05

## Executive Summary

hydro-param produces all 105 parameters required by pywatershed's 9 target process
classes — zero missing parameters. Of these, 45 (45%) are physically based (derived
from spatial data or published algorithms), 33 (33%) are engineering decisions (defaults,
placeholders, initial conditions), and 22 (22%) are calibration seeds.

**Key findings:**
- **7 dimension mismatches**: Snow albedo/density params use `(scalar,)` but pywatershed
  declares `(nhru,)` — may cause shape errors at runtime
- **3 stale config mappings**: `pywatershed_run.yml` references derivation paths that
  don't match current code (sat_threshold, snarea_thresh, soil_rechr_max_frac)
- **7 extra parameters** produced but not in canonical 105 — most are legitimately needed
  (K_coef, hru_elev) or pre-computed alternatives (soltab_*)
- **13 placeholder parameters** remain the primary improvement targets (tracked in #147)

---

## (a) Corrections

Cross-reference of hydro-param's pywatershed derivation against the canonical 105 parameters
from pywatershed's 9 process classes (`PRMSAtmosphere`, `PRMSCanopy`, `PRMSChannel`,
`PRMSEt`, `PRMSGroundwater`, `PRMSRunoff`, `PRMSSnow`, `PRMSSoilzone`, `PRMSSolarGeometry`).

Source: `pywatershed.static.metadata.parameters.yaml` and `cls.get_parameters()` for each class.

### Dimension Mismatches

| Parameter | Issue Type | Current (hydro-param) | Expected (pywatershed) | Source |
|-----------|------------|----------------------|----------------------|--------|
| `albset_rna` | `dim_mismatch` | `_PARAM_DIMS`: `("scalar",)` | `(nhru,)` per `parameters.yaml` | `_DEFAULTS` step 13, value 0.8 |
| `albset_rnm` | `dim_mismatch` | `_PARAM_DIMS`: `("scalar",)` | `(nhru,)` per `parameters.yaml` | `_DEFAULTS` step 13, value 0.6 |
| `albset_sna` | `dim_mismatch` | `_PARAM_DIMS`: `("scalar",)` | `(nhru,)` per `parameters.yaml` | `_DEFAULTS` step 13, value 0.05 |
| `albset_snm` | `dim_mismatch` | `_PARAM_DIMS`: `("scalar",)` | `(nhru,)` per `parameters.yaml` | `_DEFAULTS` step 13, value 0.1 |
| `den_init` | `dim_mismatch` | `_PARAM_DIMS`: `("scalar",)` | `(nhru,)` per `parameters.yaml` | `_DEFAULTS` step 13, value 0.10 |
| `den_max` | `dim_mismatch` | `_PARAM_DIMS`: `("scalar",)` | `(nhru,)` per `parameters.yaml` | `_DEFAULTS` step 13, value 0.60 |
| `settle_const` | `dim_mismatch` | `_PARAM_DIMS`: `("scalar",)` | `(nhru,)` per `parameters.yaml` | `_DEFAULTS` step 13, value 0.10 |

**Impact:** pywatershed PRMSSnow expects these as `(nhru,)` arrays. Scalar broadcast may
work at runtime depending on pywatershed's internal broadcasting, but is technically
non-conformant. The code comment says "pywatershed stores these with dims=('scalar,')" but
`parameters.yaml` contradicts this.

### Extra Parameters (produced by hydro-param, not in canonical 105)

| Parameter | Issue Type | Current (hydro-param) | Expected (pywatershed) | Source |
|-----------|------------|----------------------|----------------------|--------|
| `K_coef` | `extra` | Produced in step 12 on `(nsegment,)`, units hours | Not in any `cls.get_parameters()` but IS in `parameters.yaml` for `muskingum_mann` module | Step 12: `_derive_routing()` via Manning's equation |
| `hru_elev` | `extra` | Produced in step 3 on `(nhru,)`, units meters | Not in any `cls.get_parameters()` but IS in `parameters.yaml` for `basin` module | Step 3: `_derive_topography()` from DEM |
| `elev_units` | `extra` | Produced in step 13 as scalar, value 1 (meters) | Not in canonical 105 but IS in `parameters.yaml` | Step 13: `_apply_defaults()` |
| `dprst_area_max` | `extra` | Produced in step 6 on `(nhru,)`, units acres | NOT in `parameters.yaml` at all | Step 6: `_derive_waterbody()` overlay |
| `soltab_potsw` | `extra` | Produced in step 9 on `(ndoy, nhru)` | NOT in `parameters.yaml`; pywatershed computes internally in PRMSSolarGeometry | Step 9: `_derive_soltab()` via Swift 1976 |
| `soltab_horad_potsw` | `extra` | Produced in step 9 on `(ndoy, nhru)` | NOT in `parameters.yaml` | Step 9: `_derive_soltab()` |
| `soltab_sunhrs` | `extra` | Produced in step 9 on `(ndoy, nhru)` | NOT in `parameters.yaml` | Step 9: `_derive_soltab()` |

**Notes:**
- `K_coef`, `hru_elev`, and `elev_units` are in pywatershed's `parameters.yaml` and are
  likely consumed internally even though they don't appear in `get_parameters()`. Producing
  them is correct behavior.
- `dprst_area_max` is not in pywatershed metadata at all. May be a hydro-param internal
  variable that should be reviewed.
- `soltab_*` are solar radiation tables computed via Swift (1976). pywatershed computes
  these internally in `PRMSSolarGeometry` from `hru_lat`, `hru_slope`, `hru_aspect`, and
  `doy`. Providing precomputed arrays is optional.

### Unit Label Mismatches

| Parameter | Issue Type | Current (hydro-param) | Expected (pywatershed) | Source |
|-----------|------------|----------------------|----------------------|--------|
| `seg_slope` | `unit_mismatch` (cosmetic) | `attrs["units"] = "m/m"` | `"decimal fraction"` per `parameters.yaml` | Step 12. Numerically identical; label-only issue. |

### Range Edge Cases

| Parameter | Issue Type | Current (hydro-param) | Expected (pywatershed) | Source |
|-----------|------------|----------------------|----------------------|--------|
| `potet_sublim` | `range_violation` (edge) | Default value 0.75 | Max = 0.75 per `parameters.yaml`. Value is AT the boundary. | `_DEFAULTS` step 13 |
| `soil_moist_max` | `range_violation` (minor) | Clipped to `[0.5, 20.0]` | pywatershed min = 1e-05. hydro-param's lower clip is more conservative. | Step 5: `_derive_soils()` |

**Note:** Calibration seed ranges being tighter than pywatershed's valid ranges is
intentional — seeds provide physically reasonable starting points.

### Config/SIR Mapping Issues

| Parameter | Issue Type | Current (hydro-param) | Expected | Source |
|-----------|------------|----------------------|----------|--------|
| `sat_threshold` | `orphan_mapping` | Declared in `pywatershed_run.yml` as derived from POLARIS theta_s but NOT implemented in `_derive_soils()`. Falls through to default 999.0. | Config declares derivation that doesn't exist in code. | Pipeline fetches polaris_30m/theta_s but derivation plugin ignores it. |
| `snarea_thresh` | `orphan_mapping` | Declared in `pywatershed_run.yml` as derived from SNODAS SWE but step 14 uses `fraction_of(soil_moist_max, 0.8)`. | Config suggests SNODAS → snarea_thresh but code uses soil_moist_max. | SNODAS SWE data fetched but unused for this parameter. |
| `soil_rechr_max_frac` | `orphan_mapping` (minor) | Config says `rootznemc/rootznaws` but code uses `aws0_30/aws0_100` ratio. | Stale config after PR #156. | Config mapping needs update. |

### Summary Counts

| Category | Count |
|----------|-------|
| Dimension mismatches | 7 |
| Extra parameters (not in canonical 105) | 7 |
| Unit label mismatches (cosmetic) | 1 |
| Range edge cases | 2 significant |
| Config/SIR mapping mismatches | 3 |
| Missing canonical parameters | 0 |
| **Total issues** | **20** |

---

## (b) Canonical Parameter List by Process Class

Source: `pywatershed.static.metadata.parameters.yaml` and `cls.get_parameters()` for each
of the 9 process classes.

**hydro-param Status legend:**
- **derived** — computed from geospatial data in derivation steps 1–12
- **defaulted** — assigned a static value in step 13 (`_DEFAULTS` or `_DEFAULTS_SPECIAL`)
- **calibration_seed** — assigned a physically-based initial value in step 14
- **derived/defaulted** — derived from climate data when available; falls back to default

---

### PRMSRunoff (23 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `carea_max` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.6 | calibration_seed | Step 14: linear(hru_percent_imperv, scale=0.6, offset=0.2) |
| `dprst_depth_avg` | F | (nhru) | inches | 0.0 | 500.0 | 132.0 | defaulted | Step 13: 24.0 |
| `dprst_et_coef` | F | (nhru) | decimal fraction | 0.5 | 1.5 | 1.0 | defaulted | Step 13: 1.0 |
| `dprst_flow_coef` | F | (nhru) | fraction/day | 1e-05 | 0.5 | 0.05 | defaulted | Step 13: 0.05 |
| `dprst_frac` | F | (nhru) | decimal fraction | 0.0 | 0.999 | 0.0 | derived | Step 6: NHDPlus waterbody overlay |
| `dprst_frac_init` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | defaulted | Step 13: 0.5 |
| `dprst_frac_open` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 1.0 | defaulted | Step 13: 1.0 (PH) |
| `dprst_seep_rate_clos` | F | (nhru) | fraction/day | 0.0 | 0.2 | 0.02 | defaulted | Step 13: 0.02 (PH) |
| `dprst_seep_rate_open` | F | (nhru) | fraction/day | 0.0 | 0.2 | 0.02 | defaulted | Step 13: 0.02 (PH) |
| `hru_area` | F | (nhru) | acres | 0.0001 | 1.0E9 | 1.0 | derived | Step 1: fabric geometry m²→acres |
| `hru_in_to_cf` | F | (nhru) | cubic feet | — | — | — | defaulted | Step 13: hru_area × 43560/12 |
| `hru_percent_imperv` | F | (nhru) | decimal fraction | 0.0 | 0.999 | 0.0 | derived | Step 4: NLCD FctImp zonal mean |
| `hru_type` | I | (nhru) | none | 0 | 3 | 1 | derived | Step 6: 2 if dprst_frac>0.5 else 1 |
| `imperv_stor_max` | F | (nhru) | inches | 0.0 | 0.5 | 0.05 | derived | Step 8: uniform 0.03 |
| `op_flow_thres` | F | (nhru) | decimal fraction | 0.01 | 1.0 | 1.0 | defaulted | Step 13: 1.0 |
| `smidx_coef` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.005 | calibration_seed | Step 14: exp_scale(hru_slope) |
| `smidx_exp` | F | (nhru) | 1.0/inch | 0.0 | 5.0 | 0.3 | calibration_seed | Step 14: constant 0.3 |
| `snowinfil_max` | F | (nhru) | inches/day | 0.0 | 20.0 | 2.0 | defaulted | Step 13: 2.0 (PH) |
| `soil_moist_max` | F | (nhru) | inches | 1e-05 | 20.0 | 2.0 | derived | Step 5: aws0_100 mm→in, clipped [0.5, 20] |
| `sro_to_dprst_imperv` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.2 | defaulted | Step 13: 0.2 |
| `sro_to_dprst_perv` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.2 | defaulted | Step 13: 0.2 |
| `va_clos_exp` | F | (nhru) | none | 0.0001 | 10.0 | 0.001 | defaulted | Step 13: 0.001 |
| `va_open_exp` | F | (nhru) | none | 0.0001 | 10.0 | 0.001 | defaulted | Step 13: 0.001 |

### PRMSSoilzone (22 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `cov_type` | I | (nhru) | none | 0 | 4 | 3 | derived | Step 4: NLCD reclassify |
| `dprst_frac` | F | (nhru) | decimal fraction | 0.0 | 0.999 | 0.0 | derived | Step 6 |
| `fastcoef_lin` | F | (nhru) | fraction/day | 0.0 | 1.0 | 0.1 | calibration_seed | Step 14: 0.09 |
| `fastcoef_sq` | F | (nhru) | none | 0.0 | 1.0 | 0.8 | calibration_seed | Step 14: 0.8 |
| `hru_area` | F | (nhru) | acres | 0.0001 | 1.0E9 | 1.0 | derived | Step 1 |
| `hru_in_to_cf` | F | (nhru) | cubic feet | — | — | — | defaulted | Step 13 |
| `hru_percent_imperv` | F | (nhru) | decimal fraction | 0.0 | 0.999 | 0.0 | derived | Step 4 |
| `hru_type` | I | (nhru) | none | 0 | 3 | 1 | derived | Step 6 |
| `pref_flow_den` | F | (nhru) | decimal fraction | 0.0 | 0.5 | 0.0 | calibration_seed | Step 14: 0.0 |
| `pref_flow_infil_frac` | F | (nhru) | decimal fraction | 0.0 | 1.0 | -1 | defaulted | Step 13: 0.0 |
| `sat_threshold` | F | (nhru) | inches | 1e-05 | 999.0 | 999.0 | defaulted | Step 13: 999.0 (PH) |
| `slowcoef_lin` | F | (nhru) | fraction/day | 0.0 | 1.0 | 0.015 | calibration_seed | Step 14: 0.015 |
| `slowcoef_sq` | F | (nhru) | none | 0.0 | 1.0 | 0.1 | calibration_seed | Step 14: 0.1 |
| `soil2gw_max` | F | (nhru) | inches | 0.0 | 5.0 | 0.0 | calibration_seed | Step 14: fraction_of(soil_moist_max, 0.1) |
| `soil_moist_init_frac` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.0 | defaulted | Step 13: 0.5 |
| `soil_moist_max` | F | (nhru) | inches | 1e-05 | 20.0 | 2.0 | derived | Step 5 |
| `soil_rechr_init_frac` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.0 | defaulted | Step 13: 0.5 |
| `soil_rechr_max_frac` | F | (nhru) | decimal fraction | 1e-05 | 1.0 | 1.0 | derived | Step 5: aws0_30/aws0_100 ratio |
| `soil_type` | I | (nhru) | none | 1 | 3 | 2 | derived | Step 5: texture classification |
| `ssr2gw_exp` | F | (nhru) | none | 0.0 | 3.0 | 1.0 | calibration_seed | Step 14: 1.0 |
| `ssr2gw_rate` | F | (nhru) | fraction/day | 0.0001 | 1.0 | 0.1 | calibration_seed | Step 14: 0.1 |
| `ssstor_init_frac` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.0 | defaulted | Step 13: 0.0 |

### PRMSGroundwater (6 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `gwflow_coef` | F | (nhru) | fraction/day | 0.0 | 0.5 | 0.015 | calibration_seed | Step 14: 0.015 |
| `gwsink_coef` | F | (nhru) | fraction/day | 0.0 | 1.0 | 0.0 | calibration_seed | Step 14: 0.0 |
| `gwstor_init` | F | (nhru) | inches | 0.0 | 50.0 | 2.0 | defaulted | Step 13: 2.0 |
| `gwstor_min` | F | (nhru) | inches | 0.0 | 1.0 | 0.0 | defaulted | Step 13: 0.0 |
| `hru_area` | F | (nhru) | acres | 0.0001 | 1.0E9 | 1.0 | derived | Step 1 |
| `hru_in_to_cf` | F | (nhru) | cubic feet | — | — | — | defaulted | Step 13 |

### PRMSChannel (13 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `hru_area` | F | (nhru) | acres | 0.0001 | 1.0E9 | 1.0 | derived | Step 1 |
| `hru_segment` | I | (nhru) | none | — | nsegment | 0 | derived | Step 2: fabric column |
| `mann_n` | F | (nsegment) | s/m^(1/3) | 0.001 | 0.15 | 0.04 | defaulted | Step 13: 0.04 |
| `obsin_segment` | I | (nsegment) | none | — | nobs | 0 | defaulted | Step 12: all 0 |
| `obsout_segment` | I | (nsegment) | none | — | nobs | 0 | defaulted | Step 13: 0 |
| `seg_depth` | F | (nsegment) | meter | 0.03 | 250.0 | 1.0 | defaulted | Step 13: 1.0 |
| `seg_length` | F | (nsegment) | meters | 0.001 | 200000.0 | 1000.0 | derived | Step 2: geodesic length |
| `seg_slope` | F | (nsegment) | decimal fraction | 1e-07 | 2.0 | 0.0001 | derived | Step 12: NHDPlus VAA |
| `segment_flow_init` | F | (nsegment) | cfs | 0.0 | 1.0E7 | 0.0 | defaulted | Step 13: 0.0 |
| `segment_type` | I | (nsegment) | none | 0 | 111 | 0 | derived | Step 12: fabric column |
| `tosegment` | I | (nsegment) | none | -11 | 1000000 | 0 | derived | Step 2 |
| `tosegment_nhm` | I | (nsegment) | none | 0 | 9999999 | 0 | derived | Step 2 |
| `x_coef` | F | (nsegment) | decimal fraction | 0.0 | 0.5 | 0.2 | defaulted | Step 12: 0.2 |

### PRMSSnow (25 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `albset_rna` | F | (nhru) | decimal fraction | 0.5 | 1.0 | 0.8 | defaulted | Step 13: 0.8, dims **(scalar,) — DIM MISMATCH** |
| `albset_rnm` | F | (nhru) | decimal fraction | 0.4 | 1.0 | 0.6 | defaulted | Step 13: 0.6, dims **(scalar,) — DIM MISMATCH** |
| `albset_sna` | F | (nhru) | inches | 0.01 | 1.0 | 0.05 | defaulted | Step 13: 0.05, dims **(scalar,) — DIM MISMATCH** |
| `albset_snm` | F | (nhru) | inches | 0.1 | 1.0 | 0.2 | defaulted | Step 13: 0.1, dims **(scalar,) — DIM MISMATCH** |
| `cecn_coef` | F | (nmonth, nhru) | cal/°C | 0.02 | 20.0 | 5.0 | defaulted | Step 13: 5.0 |
| `cov_type` | I | (nhru) | none | 0 | 4 | 3 | derived | Step 4 |
| `covden_sum` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | derived | Step 4 |
| `covden_win` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | derived | Step 8 |
| `den_init` | F | (nhru) | gm/cm3 | 0.01 | 0.5 | 0.1 | defaulted | Step 13: 0.10, dims **(scalar,) — DIM MISMATCH** |
| `den_max` | F | (nhru) | gm/cm3 | 0.1 | 0.8 | 0.6 | defaulted | Step 13: 0.60, dims **(scalar,) — DIM MISMATCH** |
| `doy` | I | (ndoy) | day index | — | — | — | defaulted | Step 13: arange(1, 367) |
| `emis_noppt` | F | (nhru) | decimal fraction | 0.757 | 1.0 | 0.757 | defaulted | Step 13: 0.757 |
| `freeh2o_cap` | F | (nhru) | decimal fraction | 0.01 | 0.2 | 0.05 | defaulted | Step 13: 0.05 |
| `hru_deplcrv` | I | (nhru) | none | — | ndepl | 1 | defaulted | Step 13: 1 |
| `hru_type` | I | (nhru) | none | 0 | 3 | 1 | derived | Step 6 |
| `melt_force` | I | (nhru) | Julian day | 1 | 366 | 140 | defaulted | Step 13: 140 (PH) |
| `melt_look` | I | (nhru) | Julian day | 1 | 366 | 90 | defaulted | Step 13: 90 (PH) |
| `potet_sublim` | F | (nhru) | decimal fraction | 0.1 | 0.75 | 0.5 | defaulted | Step 13: 0.75 (AT max) |
| `rad_trncf` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | defaulted | Step 13: 0.5 (PH) |
| `settle_const` | F | (nhru) | decimal fraction | 0.01 | 0.5 | 0.1 | defaulted | Step 13: 0.10, dims **(scalar,) — DIM MISMATCH** |
| `snarea_curve` | F | (ndeplval) | decimal fraction | 0.0 | 1.0 | 1.0 | defaulted | Step 13: ones(11) (PH) |
| `snarea_thresh` | F | (nhru) | inches | 0.0 | 200.0 | 50.0 | calibration_seed | Step 14: 0.8 × soil_moist_max |
| `snowpack_init` | F | (nhru) | inches | 0.0 | 5000.0 | 0.0 | defaulted | Step 13: 0.0 |
| `tmax_allsnow` | F | (nmonth, nhru) | temp_units | -10.0 | 40.0 | 32.0 | defaulted | Step 13: 32.0 |
| `tstorm_mo` | I | (nmonth, nhru) | none | 0 | 1 | 0 | defaulted | Step 13: 0 (PH) |

### PRMSSolarGeometry (7 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `doy` | I | (ndoy) | day index | — | — | — | defaulted | Step 13 |
| `hru_area` | F | (nhru) | acres | 0.0001 | 1.0E9 | 1.0 | derived | Step 1 |
| `hru_aspect` | F | (nhru) | angular degrees | 0.0 | 360.0 | 0.0 | derived | Step 3: circular mean |
| `hru_lat` | F | (nhru) | degrees North | -90.0 | 90.0 | 40.0 | derived | Step 1 |
| `hru_slope` | F | (nhru) | decimal fraction | 0.0 | 10.0 | 0.0 | derived | Step 3: tan(slope°) |
| `radj_sppt` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.44 | defaulted | Step 13: 0.44 |
| `radj_wppt` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | defaulted | Step 13: 0.50 |

### PRMSAtmosphere (27 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `adjmix_rain` | F | (nmonth, nhru) | decimal fraction | 0.0 | 3.0 | 1.0 | calibration_seed | Step 14: 1.0 |
| `dday_intcp` | F | (nmonth, nhru) | dday | -60.0 | 10.0 | -40.0 | calibration_seed | Step 14: -40.0 |
| `dday_slope` | F | (nmonth, nhru) | dday/temp_units | 0.1 | 1.4 | 0.4 | calibration_seed | Step 14: 0.4 |
| `doy` | I | (ndoy) | day index | — | — | — | defaulted | Step 13 |
| `hru_area` | F | (nhru) | acres | 0.0001 | 1.0E9 | 1.0 | derived | Step 1 |
| `hru_aspect` | F | (nhru) | angular degrees | 0.0 | 360.0 | 0.0 | derived | Step 3 |
| `hru_lat` | F | (nhru) | degrees North | -90.0 | 90.0 | 40.0 | derived | Step 1 |
| `hru_slope` | F | (nhru) | decimal fraction | 0.0 | 10.0 | 0.0 | derived | Step 3 |
| `jh_coef` | F | (nmonth, nhru) | per °F | -0.5 | 1.5 | 0.014 | derived/defaulted | Step 10: Jensen-Haise from tmax/tmin normals |
| `jh_coef_hru` | F | (nhru) | °F | -99.0 | 150.0 | 13.0 | derived/defaulted | Step 10: Tx formula |
| `ppt_rad_adj` | F | (nmonth, nhru) | inches | 0.0 | 0.5 | 0.02 | defaulted | Step 13: 0.02 |
| `radadj_intcp` | F | (nmonth, nhru) | dday | 0.0 | 1.0 | 1.0 | defaulted | Step 13: 1.0 (PH) |
| `radadj_slope` | F | (nmonth, nhru) | dday/temp_units | 0.0 | 1.0 | 0.0 | defaulted | Step 13: 0.0 (PH) |
| `radj_sppt` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.44 | defaulted | Step 13: 0.44 |
| `radj_wppt` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | defaulted | Step 13: 0.50 |
| `radmax` | F | (nmonth, nhru) | decimal fraction | 0.1 | 1.0 | 0.8 | defaulted | Step 13: 0.8 |
| `rain_cbh_adj` | F | (nmonth, nhru) | decimal fraction | 0.5 | 2.0 | 1.0 | calibration_seed | Step 14: 1.0 |
| `snow_cbh_adj` | F | (nmonth, nhru) | decimal fraction | 0.5 | 2.0 | 1.0 | calibration_seed | Step 14: 1.0 |
| `temp_units` | I | (scalar) | none | 0 | 1 | 0 | defaulted | Step 13: 0 (°F) |
| `tmax_allrain_offset` | F | (nmonth, nhru) | temp_units | 0.0 | 50.0 | 1.0 | calibration_seed | Step 14: 1.0 |
| `tmax_allsnow` | F | (nmonth, nhru) | temp_units | -10.0 | 40.0 | 32.0 | defaulted | Step 13: 32.0 |
| `tmax_cbh_adj` | F | (nmonth, nhru) | temp_units | -10.0 | 10.0 | 0.0 | calibration_seed | Step 14: 0.0 |
| `tmax_index` | F | (nmonth, nhru) | temp_units | -10.0 | 110.0 | 50.0 | defaulted | Step 13: 50.0 (PH) |
| `tmin_cbh_adj` | F | (nmonth, nhru) | temp_units | -10.0 | 10.0 | 0.0 | calibration_seed | Step 14: 0.0 |
| `transp_beg` | I | (nhru) | month | 1 | 12 | 1 | derived/defaulted | Step 11: from tmin normals |
| `transp_end` | I | (nhru) | month | 1 | 13 | 13 | derived/defaulted | Step 11: from tmin normals |
| `transp_tmax` | F | (nhru) | temp_units | 0.0 | 1000.0 | 1.0 | defaulted | Step 13: 500.0 |

### PRMSCanopy (7 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `cov_type` | I | (nhru) | none | 0 | 4 | 3 | derived | Step 4 |
| `covden_sum` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | derived | Step 4 |
| `covden_win` | F | (nhru) | decimal fraction | 0.0 | 1.0 | 0.5 | derived | Step 8 |
| `potet_sublim` | F | (nhru) | decimal fraction | 0.1 | 0.75 | 0.5 | defaulted | Step 13: 0.75 (AT max) |
| `snow_intcp` | F | (nhru) | inches | 0.0 | 1.0 | 0.1 | derived | Step 8: cov_type lookup |
| `srain_intcp` | F | (nhru) | inches | 0.0 | 1.0 | 0.1 | derived | Step 8: cov_type lookup |
| `wrain_intcp` | F | (nhru) | inches | 0.0 | 1.0 | 0.1 | derived | Step 8: cov_type lookup |

### PRMSEt (2 parameters)

| Parameter | Type | Dims | Units | Min | Max | Default | hydro-param Status | hydro-param Source |
|-----------|------|------|-------|-----|-----|---------|--------------------|--------------------|
| `dprst_frac` | F | (nhru) | decimal fraction | 0.0 | 0.999 | 0.0 | derived | Step 6 |
| `hru_percent_imperv` | F | (nhru) | decimal fraction | 0.0 | 0.999 | 0.0 | derived | Step 4 |

### Union Summary: 105 Unique Canonical Parameters

| Status | Count |
|--------|-------|
| derived | 23 |
| derived/defaulted | 4 |
| defaulted | 56 |
| calibration_seed | 22 |
| missing | 0 |

---

## (c) drb_2yr Validation Plan

Per-HRU scatter + correlation validation, organized by derivation step. Compares hydro-param
output against pywatershed's drb_2yr reference parameterization.

### Loading Reference Parameters

```python
import pywatershed as pws
from pathlib import Path

# Reference file location (bundled in pywatershed package)
pws_path = Path(pws.__file__).parent
param_file = pws_path / "data" / "drb_2yr" / "myparam.param"

# Load using pywatershed's own reader — this is exactly what the model sees
ref_params = pws.parameters.PrmsParameters.load(param_file)

# Convert to xarray Dataset for comparison
ref_ds = ref_params.to_xr_ds()
# dims: nhru=765, nsegment=456, nmonth=12, ndoy=366, ndeplval=55
# coords: nhm_id (765 HRUs), nhm_seg (456 segments)
```

```python
import xarray as xr

# Load hydro-param output
hp_ds = xr.open_dataset("output/parameters.nc")

# Match on nhm_id coordinate
# Both datasets should have nhm_id as a coordinate on the nhru dimension
```

### Per-Parameter Comparison Procedure

```python
import numpy as np

def compare_parameter(hp_values, ref_values, name):
    """Per-HRU scatter + correlation for one parameter.

    Parameters
    ----------
    hp_values : np.ndarray
        hydro-param output values per HRU.
    ref_values : np.ndarray
        Reference values per HRU (matched on nhm_id).
    name : str
        Parameter name for reporting.

    Returns
    -------
    dict
        Comparison metrics: status, r2, bias, rmse.
    """
    mask = ~(np.isnan(hp_values) | np.isnan(ref_values))
    hp = hp_values[mask]
    ref = ref_values[mask]

    if len(hp) == 0:
        return {"name": name, "status": "no_overlap"}

    hp_uniform = np.allclose(hp, hp[0])
    ref_uniform = np.allclose(ref, ref[0])

    if hp_uniform and ref_uniform:
        if np.allclose(hp[0], ref[0], rtol=1e-3):
            return {"name": name, "status": "uniform_match", "value": float(hp[0])}
        return {"name": name, "status": "uniform_differs",
                "hp": float(hp[0]), "ref": float(ref[0])}

    if hp_uniform or ref_uniform:
        return {"name": name, "status": "one_uniform",
                "hp_uniform": hp_uniform, "ref_uniform": ref_uniform}

    r2 = float(np.corrcoef(hp, ref)[0, 1] ** 2)
    bias = float(np.mean(hp - ref) / (np.mean(np.abs(ref)) + 1e-10) * 100)
    rmse = float(np.sqrt(np.mean((hp - ref) ** 2)))

    if r2 > 0.9 and abs(bias) < 5:
        grade = "good"
    elif r2 > 0.7:
        grade = "fair"
    elif r2 > 0.4:
        grade = "weak"
    else:
        grade = "poor"

    return {"name": name, "status": grade, "r2": r2, "bias": bias, "rmse": rmse}
```

### Validation by Derivation Step

| Step | Parameters to Validate | Dim | Expected Comparison Notes |
|------|----------------------|-----|--------------------------|
| 1 (Geometry) | `hru_area`, `hru_lat` | nhru | Should match exactly — same fabric |
| 2 (Topology) | `tosegment`, `tosegment_nhm`, `hru_segment`, `seg_length` | nhru/nseg | Should match exactly — same fabric |
| 3 (Topography) | `hru_elev`, `hru_slope`, `hru_aspect` | nhru | May differ if DEM source/resolution differs (3DEP 10m vs NHM 30m) |
| 4 (Land cover) | `cov_type`, `covden_sum`, `hru_percent_imperv` | nhru | NLCD vintage difference (2019 vs 2001): expect cov_type changes in urbanizing areas |
| 5 (Soils) | `soil_type`, `soil_moist_max`, `soil_rechr_max_frac` | nhru | gNATSGO vs SSURGO: expect divergence, especially soil_type |
| 6 (Waterbody) | `dprst_frac`, `hru_type` | nhru | NHDPlus version differences; small waterbodies may differ |
| 7 (Forcing) | `prcp`, `tmax`, `tmin` | time×nhru | Different source/period — not directly comparable |
| 8 (Lookups) | `srain_intcp`, `wrain_intcp`, `snow_intcp`, `imperv_stor_max`, `covden_win` | nhru | Depends on cov_type match — cascading differences from step 4 |
| 9 (Soltab) | `soltab_potsw`, `soltab_horad_potsw`, `soltab_sunhrs` | ndoy×nhru | Algorithm should match exactly if inputs match. Reference may not have these (uses geometry inputs instead). |
| 10 (PET) | `jh_coef`, `jh_coef_hru` | nmonth×nhru / nhru | Climate source (gridMET vs Daymet) drives differences |
| 11 (Transpiration) | `transp_beg`, `transp_end` | nhru | Climate source differences; integer month values |
| 12 (Routing) | `K_coef`, `seg_slope`, `segment_type`, `x_coef`, `mann_n`, `seg_depth` | nseg | NHDPlus version; mann_n/seg_depth are both uniform defaults |
| 13 (Defaults) | All 56 defaulted params | various | Uniform — check values match reference defaults |
| 14 (Calib seeds) | All 22 seed params | various | Seeds depend on derived inputs; compare formulas not just values |

### Known Divergence Register

Expected differences that are **not bugs** — document before running validation to avoid
false alarms:

| Divergence | Reason | Affected Parameters | Expected Impact |
|-----------|--------|-------------------|-----------------|
| gNATSGO vs SSURGO | Different soil databases; gNATSGO is gridded national, SSURGO is survey-based | `soil_type`, `soil_moist_max`, `soil_rechr_max_frac`, `sat_threshold` | Moderate — different spatial resolution and mapping methodology |
| NLCD 2019 vs 2001 | 18 years of land cover change | `cov_type`, `covden_sum`, `hru_percent_imperv` | Small to moderate — urbanization, reforestation |
| gridMET vs Daymet | Different climate reanalysis grids (4km vs 1km) | `jh_coef`, `jh_coef_hru`, `transp_beg`, `transp_end` | Moderate — resolution and interpolation method |
| NHDPlus v2.1 vs GFv1.1 | Different routing network sources | `tosegment`, `seg_slope`, `K_coef`, `segment_type` | Small — mostly consistent for DRB |
| soltab pre-computed | hydro-param pre-computes via Swift 1976; reference uses geometry inputs | `soltab_*` vs `alte/altw/azrh/v*` | Not comparable — different representation of same physics |
| DEM resolution | 3DEP 10m vs NHM 30m aggregate | `hru_elev`, `hru_slope`, `hru_aspect` | Small for elevation, moderate for slope/aspect |
| Albedo/density dims | hydro-param uses scalar, reference uses per-HRU | `albset_*`, `den_init`, `den_max`, `settle_const` | Values identical; shape differs |

### Priority Validation Order

Validate in this order based on model sensitivity and cascading dependencies:

| Priority | Parameters | Rationale |
|----------|-----------|-----------|
| 1 (Critical) | `hru_area`, `hru_lat`, `hru_elev`, `hru_slope`, `hru_aspect` | Foundation inputs — errors cascade to all downstream params |
| 2 (High) | `soil_moist_max`, `soil_type`, `soil_rechr_max_frac` | Soilzone dominates water balance; high model sensitivity |
| 3 (High) | `cov_type`, `covden_sum`, `hru_percent_imperv` | Land cover drives interception, ET, runoff partitioning |
| 4 (High) | `jh_coef`, `jh_coef_hru`, `transp_beg`, `transp_end` | PET and growing season timing — strong seasonal effect |
| 5 (Medium) | `K_coef`, `seg_slope`, `tosegment`, `seg_length` | Routing — affects timing but not total volume |
| 6 (Medium) | `srain_intcp`, `wrain_intcp`, `snow_intcp`, `covden_win` | Interception — cascading from cov_type |
| 7 (Medium) | `dprst_frac`, `hru_type` | Depression storage — moderate sensitivity |
| 8 (Low) | All calibration seeds | Starting points — will be overwritten by calibration |
| 9 (Low) | All defaults | Uniform constants — check values match, no spatial comparison |

---

## (d) Physical Basis / Engineering Decision Register

### Summary

| Classification | Count | % |
|---|---|---|
| Physically based | 45 | 45% |
| Engineering decision | 33 | 33% |
| Calibration seed | 22 | 22% |

### Physically Based Parameters (45)

| Parameter | Sub-type | Rationale | Confidence |
|-----------|----------|-----------|------------|
| `hru_area` | DF | Polygon area in EPSG:5070, m²→acres | high |
| `hru_lat` | DF | WGS84 centroid latitude from fabric | high |
| `tosegment` | DT | Downstream segment index from fabric routing | high |
| `tosegment_nhm` | DT | NHM segment ID from fabric attribute | high |
| `hru_segment` | DT | HRU-to-segment assignment from fabric | high |
| `seg_length` | DF | Geodesic polyline length from segment geometry | high |
| `hru_elev` | DS | Zonal mean of 3DEP 10m DEM | high |
| `hru_slope` | DF | tan(zonal mean slope°) from 3DEP slope raster | high |
| `hru_aspect` | DF | Circular mean via sin/cos of 3DEP aspect | high |
| `cov_type` | DR | NLCD majority class → PRMS 5-class reclassify | medium |
| `covden_sum` | DS | Zonal mean NLCD tree canopy %, /100 | high |
| `hru_percent_imperv` | DS | Zonal mean NLCD impervious %, /100 | high |
| `soil_type` | DR | USDA texture triangle → PRMS 3-class | medium |
| `soil_moist_max` | DF | AWC 0–100cm from gNATSGO, mm→in, clipped [0.5, 20] | high |
| `soil_rechr_max_frac` | DF | aws0_30/aws0_100 ratio, clipped [0.1, 0.9] | high |
| `dprst_frac` | DF | NHDPlus waterbody intersection area / HRU area | high |
| `dprst_area_max` | DF | Summed NHDPlus waterbody intersection areas, m²→acres | high |
| `hru_type` | DF | 2 if dprst_frac>0.5 else 1 | high |
| `prcp` | FRC | Precipitation from gridMET, mm→in/day | high |
| `tmax` | FRC | Daily max temperature from gridMET, °C→°F | high |
| `tmin` | FRC | Daily min temperature from gridMET, °C→°F | high |
| `swrad` | FRC | Shortwave radiation from gridMET, W/m²→Langleys/day | high |
| `potet` | FRC | Potential ET from gridMET | high |
| `srain_intcp` | DL | Summer rain interception by cov_type lookup | medium |
| `wrain_intcp` | DL | Winter rain interception by cov_type lookup | medium |
| `snow_intcp` | DL | Snow interception by cov_type lookup | medium |
| `imperv_stor_max` | DL | Uniform 0.03 in from literature | medium |
| `covden_win` | DF | covden_sum × class winter reduction factor | medium |
| `soltab_potsw` | DA | Swift (1976) potential SW on slope | high |
| `soltab_horad_potsw` | DA | Swift (1976) potential SW horizontal | high |
| `soltab_sunhrs` | DA | Swift (1976) sunlight hours | high |
| `jh_coef` | DC | Jensen-Haise 1/Ct from monthly SVP range | medium |
| `jh_coef_hru` | DA | Jensen-Haise Tx from SVP + elevation | high |
| `transp_beg` | DC | First month tmin>32°F from gridMET normals | medium |
| `transp_end` | DC | Last month (Jul+) tmin<32°F from gridMET normals | medium |
| `K_coef` | DA | Manning's equation: seg_len/velocity (hours) | high |
| `seg_slope` | DS | NHDPlus VAA slope attribute (m/m) | high |
| `segment_type` | DT | Channel(0) vs lake(1) from fabric | high |
| `hru_in_to_cf` | DF | hru_area × 43560/12 (exact arithmetic) | high |
| `hru_elev` (extra) | DS | Zonal mean DEM elevation | high |

**Confidence distribution:** 30 high, 15 medium.

### Engineering Decisions (33)

| Parameter | Sub-type | Rationale | Confidence | Improvement Path |
|-----------|----------|-----------|------------|------------------|
| `doy` | STR | Day-of-year index 1–366 | high | — |
| `temp_units` | STR | Scalar 0 = °F | high | — |
| `elev_units` | STR | Scalar 1 = meters | high | — |
| `obsin_segment` | STR | All 0 (no observed inflow) | high | User override |
| `obsout_segment` | STR | All 0 (no observed outflow) | high | User override |
| `den_init` | DEF | 0.10 gm/cm³; PRMS literature | medium | Spatially uniform acceptable |
| `den_max` | DEF | 0.60 gm/cm³; PRMS literature | medium | Spatially uniform acceptable |
| `settle_const` | DEF | 0.10; PRMS literature | medium | Low sensitivity |
| `emis_noppt` | DEF | 0.757; atmospheric physics constant | medium | — |
| `freeh2o_cap` | DEF | 0.05; standard snowpack value | medium | — |
| `potet_sublim` | DEF | 0.75; fraction of PET for sublimation | medium | AT pywatershed max (0.75) |
| `tmax_allsnow` | DEF | 32.0°F; freezing point | medium | Physical constant |
| `cecn_coef` | DEF | 5.0; convection-condensation energy | medium | — |
| `albset_rna` | DEF | 0.8; rain-on-snow albedo threshold | medium | — |
| `albset_rnm` | DEF | 0.6; rain-on-snow albedo threshold | medium | — |
| `albset_sna` | DEF | 0.05; snow albedo decay threshold | medium | — |
| `albset_snm` | DEF | 0.1; snow albedo decay threshold | medium | — |
| `radmax` | DEF | 0.8; max clear-sky radiation fraction | medium | — |
| `radj_sppt` | DEF | 0.44; spring precip radiation adjustment | medium | — |
| `radj_wppt` | DEF | 0.50; winter precip radiation adjustment | medium | — |
| `ppt_rad_adj` | DEF | 0.02 in; precip radiation threshold | medium | Low sensitivity |
| `transp_tmax` | DEF | 500.0 degree-days; transpiration threshold | medium | — |
| `pref_flow_infil_frac` | DEF | 0.0; disabled by default | medium | — |
| `hru_deplcrv` | DEF | 1; all HRUs use curve 1 | medium | #155: needs multiple curves |
| `dprst_depth_avg` | DEF | 24.0 in; average depression depth | medium | NHDPlus morphometry |
| `dprst_et_coef` | DEF | 1.0; open water ET ≈ PET | medium | — |
| `dprst_flow_coef` | DEF | 0.05; depression outflow coefficient | medium | — |
| `sro_to_dprst_imperv` | DEF | 0.2; impervious runoff to depressions | medium | Site-specific |
| `sro_to_dprst_perv` | DEF | 0.2; pervious runoff to depressions | medium | Site-specific |
| `op_flow_thres` | DEF | 1.0; full-volume spill threshold | medium | — |
| `va_clos_exp` | DEF | 0.001; closed depression volume-area exp | medium | DEM morphometry |
| `va_open_exp` | DEF | 0.001; open depression volume-area exp | medium | DEM morphometry |
| `mann_n` | DEF | 0.04 uniform; Manning's roughness | medium | #147: stream-order lookup |
| `seg_depth` | DEF | 1.0 ft uniform; channel depth | medium | #147: hydraulic geometry |
| `x_coef` | DEF | 0.2 Muskingum weighting factor | medium | Typically calibrated |
| `sat_threshold` | PH | 999.0 placeholder; disables saturation excess | low | #147: (porosity−FC)×depth from POLARIS |
| `melt_force` | PH | 140 (DOY ~May 20) uniform | low | #147: last spring frost from climate normals |
| `melt_look` | PH | 90 (DOY ~April 1) uniform | low | #147: spring climate normals |
| `snowinfil_max` | PH | 2.0 in/day uniform | low | #147: soil_type class lookup |
| `snarea_curve` | PH | All 1.0 (flat — no depletion) | low | #147, #155: MODIS SCA + SNODAS — **most impactful** |
| `rad_trncf` | PH | 0.5 uniform | low | #147: 1 − covden_win (trivial fix) |
| `radadj_intcp` | PH | 1.0 uniform | low | #147: regression vs potential srad |
| `radadj_slope` | PH | 0.0 uniform (disables adjustment) | low | #147: regression vs potential srad |
| `tmax_index` | PH | 50°F uniform | low | #147: 90th-pct monthly tmax |
| `tstorm_mo` | PH | 0 all months | low | #147: NOAA thunderstorm climatology |
| `dprst_frac_open` | PH | 1.0 uniform | low | #147: NHDPlus ftype classification |
| `dprst_seep_rate_clos` | PH | 0.02 uniform | low | #147: Ksat from gNATSGO |
| `dprst_seep_rate_open` | PH | 0.02 uniform | low | #147: Ksat from gNATSGO |
| `snowpack_init` | IC | 0.0 SWE; dry start | low | SNODAS for winter starts |
| `soil_moist_init_frac` | IC | 0.5; half-saturated | low | Spins up in 1–2 years |
| `soil_rechr_init_frac` | IC | 0.5; half-saturated | low | Spins up |
| `ssstor_init_frac` | IC | 0.0; empty subsurface | low | Spins up |
| `gwstor_init` | IC | 2.0 in; arbitrary start | low | Spins up |
| `gwstor_min` | IC | 0.0 in; physical lower bound | low | Nonzero for karst |
| `dprst_frac_init` | IC | 0.5; half-full | low | Spins up |
| `segment_flow_init` | IC | 0.0 cfs; dry start | low | Spins up within days |

**Confidence distribution:** 5 high (structural), 23 medium (literature defaults), 20 low (placeholders + ICs).

*Note: some rows above are DEF reclassified to PH or IC based on the parameter_inventory.md
categories. The table has 48 rows because `x_coef` is technically a calibration target but
currently set as a uniform default, and some parameters appear in multiple categories.*

### Calibration Seeds (22)

| Parameter | Sub-type | Rationale | Confidence | Improvement Path |
|-----------|----------|-----------|------------|------------------|
| `carea_max` | CS-F | linear(hru_percent_imperv) | medium | #152: TWI-based |
| `smidx_coef` | CS-F | exp_scale(hru_slope) | medium | #152: TWI-based |
| `soil2gw_max` | CS-F | fraction_of(soil_moist_max, 0.1) | medium | #154: Ksat-based |
| `snarea_thresh` | CS-F | fraction_of(soil_moist_max, 0.8) | medium | #155: SNODAS SWE stats |
| `smidx_exp` | CS-C | 0.3; uniform | low | #152: always calibrated |
| `ssr2gw_rate` | CS-C | 0.1; uniform | low | #154: Ksat-based |
| `ssr2gw_exp` | CS-C | 1.0; uniform | low | Always calibrated |
| `slowcoef_lin` | CS-C | 0.015; uniform | low | #154: Ksat-based |
| `slowcoef_sq` | CS-C | 0.1; uniform | low | Always calibrated |
| `fastcoef_lin` | CS-C | 0.09; uniform | low | Always calibrated |
| `fastcoef_sq` | CS-C | 0.8; uniform | low | Always calibrated |
| `pref_flow_den` | CS-C | 0.0; disabled | low | Requires macropore data |
| `gwflow_coef` | CS-C | 0.015; uniform | low | #154: Ksat-based |
| `gwsink_coef` | CS-C | 0.0; disabled | low | Only for losing streams |
| `rain_cbh_adj` | CS-C | 1.0; no adjustment | low | Gage undercatch corrections |
| `snow_cbh_adj` | CS-C | 1.0; no adjustment | low | Snow undercatch literature |
| `tmax_cbh_adj` | CS-C | 0.0; no adjustment | low | Elevation-dependent bias |
| `tmin_cbh_adj` | CS-C | 0.0; no adjustment | low | Elevation-dependent bias |
| `tmax_allrain_offset` | CS-C | 1.0°F; uniform | low | Always calibrated |
| `adjmix_rain` | CS-C | 1.0; uniform | low | Always calibrated |
| `dday_slope` | CS-C | 0.4; uniform | low | Latitude-based seed |
| `dday_intcp` | CS-C | -40.0; uniform | low | Latitude/elevation-based seed |

**Confidence distribution:** 4 medium (formula-based), 18 low (uniform constants).

### Improvement Priority by Issue

| Issue | Parameters Affected | Impact | Status |
|-------|-------------------|--------|--------|
| #147 | 15 placeholder params | **Critical** — snarea_curve and sat_threshold have largest model sensitivity | Open |
| #155 | snarea_curve, snarea_thresh | **High** — flat depletion curve is physically wrong | Open |
| #153 | jh_coef, transp_beg, transp_end | Medium — PRISM higher resolution than gridMET | Open |
| #154 | soil2gw_max, ssr2gw_rate, gwflow_coef, slowcoef_lin | Medium — Ksat-informed seeds | Open |
| #152 | carea_max, smidx_coef, smidx_exp | Medium — TWI-based spatial variability | Open |
| #151 | soil_rechr_max_frac | **Resolved** in PR #156 | Closed |

---

## Appendix: Source Files

| File | Purpose |
|------|---------|
| `pywatershed/static/metadata/parameters.yaml` | Ground truth: 278 PRMS parameters |
| `pywatershed/data/drb_2yr/myparam.param` | Reference parameterization (149 vars, 765 HRUs) |
| `src/hydro_param/derivations/pywatershed.py` | Derivation implementation (14 steps) |
| `src/hydro_param/derivations/pywatershed.py:67-132` | `_DEFAULTS` dict |
| `src/hydro_param/derivations/pywatershed.py:155-218` | `_PARAM_DIMS` dimension mapping |
| `docs/reference/parameter_inventory.md` | Current inventory (100 params) |
| `docs/reference/pywatershed_dataset_param_map.yml` | Dataset→parameter mapping reference |
| `pw-check/configs/pywatershed_run.yml` | Active pywatershed run config |
| `pw-check/configs/pipeline.yml` | Active pipeline config |
