# Design: pywatershed v2.0 runtime compatibility

**Date:** 2026-03-02

## Problem

hydro-param produces a `parameters.nc` file that pywatershed v2.0's
`Model` class cannot consume.  Two categories of issues prevent
`pws.Model(nhm_processes, control=control, parameters=params)` from
running:

### Issue 1: Missing parameters (34 total)

pywatershed's 8 NHM process classes collectively require ~105 parameters.
Our `parameters.nc` contains 75 data vars.  The 34 missing parameters
fall into five groups:

| Group | Count | Parameters | Source |
|-------|-------|-----------|--------|
| Structural / coordinate | 3 | `doy`, `hru_in_to_cf`, `temp_units` | Derivable from existing data |
| Depression storage (operational) | 10 | `dprst_et_coef`, `dprst_flow_coef`, `dprst_frac_init`, `dprst_frac_open`, `dprst_seep_rate_clos/open`, `sro_to_dprst_imperv/perv`, `op_flow_thres`, `va_clos/open_exp` | pywatershed defaults |
| Snow process | 6 | `snarea_curve`, `hru_deplcrv`, `cecn_coef`, `rad_trncf`, `melt_force/look`, `tstorm_mo`, `snowpack_init`, `snowinfil_max` | pywatershed defaults |
| Atmosphere | 5 | `ppt_rad_adj`, `radadj_intcp/slope`, `tmax_index` | pywatershed defaults |
| Soilzone / Channel | 7 | `sat_threshold`, `pref_flow_infil_frac`, `mann_n`, `seg_depth`, `segment_flow_init`, `obsout_segment`, `tosegment_nhm` | Derivable or defaults |

Every missing parameter has a well-defined default in pywatershed's own
`meta.parameters` dictionary.  The original PRMS Fortran also uses these
same defaults.

### Issue 2: Scalar shape mismatch (21 parameters)

Our step 13 (`_apply_defaults`) writes default parameters as 0-dimensional
scalars via `np.float64(default_val)`.  pywatershed expects all parameters
to be correctly dimensioned arrays:

- Per-HRU parameters: shape `(nhru,)` — e.g., `den_max`, `emis_noppt`
- Per-month-per-HRU: shape `(nmonths, nhru)` — e.g., `radmax`, `tmax_allsnow`

The 21 affected parameters are all from our `_DEFAULTS` dict: `albset_rna`,
`albset_rnm`, `albset_sna`, `albset_snm`, `den_init`, `den_max`,
`dprst_depth_avg`, `emis_noppt`, `freeh2o_cap`, `gwstor_init`,
`gwstor_min`, `jh_coef_hru`, `potet_sublim`, `radj_sppt`, `radj_wppt`,
`radmax`, `settle_const`, `soil_moist_init_frac`, `soil_rechr_init_frac`,
`ssstor_init_frac`, `tmax_allsnow`, `transp_tmax`.

## Root cause

Both issues stem from the **derivation plugin** (`_apply_defaults` in step
13), not the formatter.  The formatter faithfully writes whatever the
derivation produces — it doesn't add or reshape parameters.

1. **Missing params**: `_DEFAULTS` dict only includes ~20 params.  The
   other ~34 params that pywatershed needs were never added.

2. **Scalar shapes**: The non-special defaults loop writes 0-d DataArrays
   (`np.float64(val)`) instead of broadcasting to `(nhru,)` or
   `(nmonths, nhru)`.

## Fix

All changes are in the derivation plugin (`derivations/pywatershed.py`),
specifically step 13 (`_apply_defaults`).  The formatter stays unchanged.

### 1. Add missing parameters to `_DEFAULTS`

Add all 34 missing parameters with their pywatershed-standard defaults:

```python
# Structural
"doy": special  # np.arange(1, 367), dim=(ndoy,)
"hru_in_to_cf": special  # derived from hru_area
"temp_units": 0  # scalar: 0=Fahrenheit

# Depression storage (operational defaults)
"dprst_et_coef": 1.0
"dprst_flow_coef": 0.05
"dprst_frac_init": 0.5
"dprst_frac_open": 1.0
"dprst_seep_rate_clos": 0.02
"dprst_seep_rate_open": 0.02
"sro_to_dprst_imperv": 0.2
"sro_to_dprst_perv": 0.2
"op_flow_thres": 1.0
"va_clos_exp": 0.001
"va_open_exp": 0.001

# Snow
"cecn_coef": 5.0  # (nmonths, nhru)
"rad_trncf": 0.5
"melt_force": 140  # Julian day
"melt_look": 90   # Julian day
"snowinfil_max": 2.0
"snowpack_init": 0.0
"hru_deplcrv": 1  # index into snarea_curve
"tstorm_mo": 0    # (nmonths, nhru)
"snarea_curve": special  # (ndeplval=11,), all 1.0

# Atmosphere
"ppt_rad_adj": 0.02  # (nmonths, nhru)
"radadj_intcp": 1.0  # (nmonths, nhru)
"radadj_slope": 0.0  # (nmonths, nhru)
"tmax_index": 50.0   # (nmonths, nhru)

# Soilzone
"sat_threshold": 999.0
"pref_flow_infil_frac": -1.0

# Channel
"mann_n": 0.04         # (nsegment,)
"seg_depth": 1.0       # (nsegment,)
"segment_flow_init": 0.0  # (nsegment,)
"obsout_segment": 0    # (nsegment,)
"tosegment_nhm": special  # copy from tosegment
```

### 2. Broadcast all defaults to correct dimensions

Replace the scalar default loop with dimension-aware broadcasting.
Add a `_PARAM_DIMS` mapping that associates each default parameter
with its expected pywatershed dimension:

```python
_PARAM_DIMS: dict[str, tuple[str, ...]] = {
    "den_init": ("nhru",),
    "den_max": ("nhru",),
    "radmax": ("nmonths", "nhru"),
    "tmax_allsnow": ("nmonths", "nhru"),
    "cecn_coef": ("nmonths", "nhru"),
    # ... etc for all defaults
}
```

The default loop then creates correctly-shaped arrays:

```python
for param_name, default_val in _DEFAULTS.items():
    if param_name in ds:
        continue
    dims = _PARAM_DIMS.get(param_name, ("nhru",))
    shape = tuple(dim_sizes[d] for d in dims)
    ds[param_name] = xr.DataArray(
        np.full(shape, default_val, dtype=...),
        dims=dims,
    )
```

### 3. Special-case parameters

Expand `_DEFAULTS_SPECIAL` to include:

- `doy`: `np.arange(1, 367)` with dim `("ndoy",)`
- `hru_in_to_cf`: `hru_area * (43560.0 / 12.0)` — unit conversion
  factor (1 inch over 1 acre = 3630 ft³)
- `temp_units`: scalar `0` (Fahrenheit)
- `snarea_curve`: `np.ones(11)` with dim `("ndeplval",)`
- `tosegment_nhm`: copy from `tosegment` if present

### 4. Update `parameter_metadata.yml`

Add metadata entries (dimension, units, valid_range) for all newly
added parameters so the formatter's `validate()` can check them.

## Scope

- **Changed**: `derivations/pywatershed.py` (step 13 `_apply_defaults`)
- **Changed**: `data/pywatershed/parameter_metadata.yml` (new entries)
- **Unchanged**: `formatters/pywatershed.py` (no formatter changes needed)
- **New tests**: verify all defaults are correctly shaped, verify
  pywatershed `PrmsParameters.from_netcdf()` can load the output

## Verification

After the fix, this should work:

```python
params = pws.parameters.PrmsParameters.from_netcdf("parameters.nc", use_xr=True)
control = pws.Control(start_time=..., end_time=..., time_step=...)
model = pws.Model(nhm_processes, control=control, parameters=params)
model.advance()
model.calculate()
```
