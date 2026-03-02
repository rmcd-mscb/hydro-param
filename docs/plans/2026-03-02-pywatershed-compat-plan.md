# pywatershed v2.0 Runtime Compatibility — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make hydro-param's `parameters.nc` output directly consumable by pywatershed v2.0's `Model` class.

**Architecture:** Fix the derivation plugin's step 13 (`_apply_defaults`) to (a) add all 34 missing parameters with pywatershed-standard defaults, and (b) broadcast all defaults to correct dimensions (`(nhru,)`, `(nmonths, nhru)`, `(nsegment,)`, etc.) instead of writing scalars. Also update `parameter_metadata.yml` with entries for the new parameters.

**Tech Stack:** Python, xarray, numpy, pywatershed v2.0 (pws-test pixi env for verification)

**Design doc:** `docs/plans/2026-03-02-pywatershed-compat-design.md`

---

### Task 1: Add `_PARAM_DIMS` mapping and refactor default loop to broadcast correctly

This is the core fix. Replace the scalar default loop with dimension-aware broadcasting.

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:67-110` (constants) and `:2231-2305` (step 13)
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing test — defaults have correct shapes**

Add a new test to `TestApplyDefaults` that checks all non-special defaults are `(nhru,)` or `(nmonths, nhru)` arrays, not scalars.

```python
def test_defaults_have_correct_shapes(
    self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
) -> None:
    """All defaults must be correctly-dimensioned arrays, not 0-d scalars."""
    ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    nhru = 3  # sir_topography fixture has 3 HRUs

    # Per-HRU defaults must be 1-D arrays of length nhru
    per_hru = [
        "den_init", "den_max", "settle_const", "emis_noppt",
        "freeh2o_cap", "potet_sublim", "albset_rna", "albset_snm",
        "albset_rnm", "albset_sna", "radj_sppt", "radj_wppt",
        "soil_moist_init_frac", "soil_rechr_init_frac",
        "ssstor_init_frac", "gwstor_init", "gwstor_min",
        "dprst_depth_avg", "transp_tmax", "jh_coef_hru",
    ]
    for name in per_hru:
        assert name in ds, f"Missing default: {name}"
        assert ds[name].ndim == 1, (
            f"{name}: expected 1-D (nhru,), got ndim={ds[name].ndim}"
        )
        assert ds[name].shape == (nhru,), (
            f"{name}: expected shape ({nhru},), got {ds[name].shape}"
        )

    # Per-month-per-HRU defaults must be 2-D (nmonths, nhru)
    per_month_hru = ["tmax_allsnow", "radmax"]
    for name in per_month_hru:
        if name in ds:
            assert ds[name].ndim == 2, (
                f"{name}: expected 2-D (nmonths, nhru), got ndim={ds[name].ndim}"
            )
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestApplyDefaults::test_defaults_have_correct_shapes -v`
Expected: FAIL — current defaults are 0-d scalars.

**Step 3: Add `_PARAM_DIMS` mapping**

After the existing `_DEFAULTS_SPECIAL` line (~107), add a dimension mapping:

```python
# Dimension mapping for default parameters.  Every entry in _DEFAULTS
# that is NOT in _DEFAULTS_SPECIAL must appear here.  pywatershed v2.0
# requires all parameters as correctly-dimensioned arrays.
_PARAM_DIMS: dict[str, tuple[str, ...]] = {
    # Per-HRU (nhru,)
    "den_init": ("nhru",),
    "den_max": ("nhru",),
    "settle_const": ("nhru",),
    "emis_noppt": ("nhru",),
    "freeh2o_cap": ("nhru",),
    "potet_sublim": ("nhru",),
    "albset_rna": ("nhru",),
    "albset_snm": ("nhru",),
    "albset_rnm": ("nhru",),
    "albset_sna": ("nhru",),
    "radj_sppt": ("nhru",),
    "radj_wppt": ("nhru",),
    "soil_moist_init_frac": ("nhru",),
    "soil_rechr_init_frac": ("nhru",),
    "ssstor_init_frac": ("nhru",),
    "gwstor_init": ("nhru",),
    "gwstor_min": ("nhru",),
    "dprst_depth_avg": ("nhru",),
    "transp_tmax": ("nhru",),
    "jh_coef_hru": ("nhru",),
    # Per-month-per-HRU (nmonths, nhru)
    "tmax_allsnow": ("nmonths", "nhru"),
    "radmax": ("nmonths", "nhru"),
}
```

**Step 4: Refactor `_apply_defaults` to use `_PARAM_DIMS`**

Replace the non-special default loop (lines ~2297-2304) with:

```python
        # Dimension sizes for broadcasting
        dim_sizes: dict[str, int] = {
            "nhru": nhru,
            "nmonths": 12,
        }

        for param_name, default_val in _DEFAULTS.items():
            if param_name in _DEFAULTS_SPECIAL:
                continue  # handled above
            if param_name in ds:
                continue  # data-derived value takes precedence
            dims = _PARAM_DIMS.get(param_name, ("nhru",))
            shape = tuple(dim_sizes[d] for d in dims)
            dtype = np.int32 if isinstance(default_val, int) else np.float64
            ds[param_name] = xr.DataArray(
                np.full(shape, default_val, dtype=dtype),
                dims=dims,
                attrs={"long_name": param_name.replace("_", " ").title()},
            )
        return ds
```

**Step 5: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestApplyDefaults -v`
Expected: All tests pass. Note: `test_defaults_present` uses `.item()` which works for both 0-d and 1-d single-element arrays, but the fixture has 3 HRUs so we need to update it.

**Step 6: Update `test_defaults_present` for array values**

The existing test checks `ds["tmax_allsnow"].item() == 32.0` which will fail for a `(12, 3)` array. Update:

```python
def test_defaults_present(
    self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
) -> None:
    ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    # Check values (all elements should be the default)
    np.testing.assert_allclose(ds["tmax_allsnow"].values, 32.0)
    np.testing.assert_allclose(ds["den_init"].values, 0.10)
    np.testing.assert_allclose(ds["gwstor_init"].values, 2.0)
    np.testing.assert_allclose(ds["radmax"].values, 0.8)
```

**Step 7: Run all tests and commit**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: All pass.

Commit: `feat: broadcast default parameters to correct dimensions for pywatershed v2.0`

---

### Task 2: Add 34 missing parameters to step 13

Add all parameters that pywatershed requires but we don't currently produce.

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:67-110` (_DEFAULTS, _PARAM_DIMS, _DEFAULTS_SPECIAL)
- Modify: `src/hydro_param/derivations/pywatershed.py:2231-2305` (_apply_defaults)
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing test — all pywatershed-required defaults present**

```python
def test_all_pywatershed_required_defaults(
    self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
) -> None:
    """All parameters required by pywatershed NHM processes must be present."""
    ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    required = {
        # Structural
        "doy", "hru_in_to_cf", "temp_units",
        # Depression storage operational
        "dprst_et_coef", "dprst_flow_coef", "dprst_frac_init",
        "dprst_frac_open", "dprst_seep_rate_clos", "dprst_seep_rate_open",
        "sro_to_dprst_imperv", "sro_to_dprst_perv", "op_flow_thres",
        "va_clos_exp", "va_open_exp",
        # Snow
        "cecn_coef", "rad_trncf", "melt_force", "melt_look",
        "snowinfil_max", "snowpack_init", "hru_deplcrv", "tstorm_mo",
        "snarea_curve",
        # Atmosphere
        "ppt_rad_adj", "radadj_intcp", "radadj_slope", "tmax_index",
        # Soilzone
        "sat_threshold", "pref_flow_infil_frac",
        # Channel (only if segments present)
        # "mann_n", "seg_depth", "segment_flow_init", "obsout_segment",
        # "tosegment_nhm",  — these need nsegment dim, skip for HRU-only test
    }
    missing = required - set(ds.data_vars)
    assert not missing, f"Missing pywatershed defaults: {sorted(missing)}"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestApplyDefaults::test_all_pywatershed_required_defaults -v`
Expected: FAIL — none of these new params exist yet.

**Step 3: Add new parameters to `_DEFAULTS` and `_PARAM_DIMS`**

Add to `_DEFAULTS`:

```python
    # --- Depression storage (operational) ---
    "dprst_et_coef": 1.0,
    "dprst_flow_coef": 0.05,
    "dprst_frac_init": 0.5,
    "dprst_frac_open": 1.0,
    "dprst_seep_rate_clos": 0.02,
    "dprst_seep_rate_open": 0.02,
    "sro_to_dprst_imperv": 0.2,
    "sro_to_dprst_perv": 0.2,
    "op_flow_thres": 1.0,
    "va_clos_exp": 0.001,
    "va_open_exp": 0.001,
    # --- Snow (additional) ---
    "rad_trncf": 0.5,
    "melt_force": 140,    # Julian day
    "melt_look": 90,      # Julian day
    "snowinfil_max": 2.0,
    "snowpack_init": 0.0,
    "hru_deplcrv": 1,     # index into snarea_curve
    # --- Atmosphere ---
    "ppt_rad_adj": 0.02,
    "radadj_intcp": 1.0,
    "radadj_slope": 0.0,
    "tmax_index": 50.0,   # degF
    # --- Soilzone ---
    "sat_threshold": 999.0,
    "pref_flow_infil_frac": -1.0,
```

Add to `_PARAM_DIMS`:

```python
    # Depression storage
    "dprst_et_coef": ("nhru",),
    "dprst_flow_coef": ("nhru",),
    "dprst_frac_init": ("nhru",),
    "dprst_frac_open": ("nhru",),
    "dprst_seep_rate_clos": ("nhru",),
    "dprst_seep_rate_open": ("nhru",),
    "sro_to_dprst_imperv": ("nhru",),
    "sro_to_dprst_perv": ("nhru",),
    "op_flow_thres": ("nhru",),
    "va_clos_exp": ("nhru",),
    "va_open_exp": ("nhru",),
    # Snow (additional)
    "rad_trncf": ("nhru",),
    "melt_force": ("nhru",),
    "melt_look": ("nhru",),
    "snowinfil_max": ("nhru",),
    "snowpack_init": ("nhru",),
    "hru_deplcrv": ("nhru",),
    "cecn_coef": ("nmonths", "nhru"),
    "tstorm_mo": ("nmonths", "nhru"),
    # Atmosphere
    "ppt_rad_adj": ("nmonths", "nhru"),
    "radadj_intcp": ("nmonths", "nhru"),
    "radadj_slope": ("nmonths", "nhru"),
    "tmax_index": ("nmonths", "nhru"),
    # Soilzone
    "sat_threshold": ("nhru",),
    "pref_flow_infil_frac": ("nhru",),
```

Add `cecn_coef` and `tstorm_mo` to `_DEFAULTS`:

```python
    "cecn_coef": 5.0,
    "tstorm_mo": 0,
```

**Step 4: Add special-case parameters in `_apply_defaults`**

Add these before the main default loop:

```python
        # doy: coordinate array 1-366
        if "doy" not in ds:
            ds["doy"] = xr.DataArray(
                np.arange(1, 367, dtype=np.int32),
                dims=("ndoy",),
                attrs={"long_name": "Day of year"},
            )

        # hru_in_to_cf: unit conversion factor (inches*acres → cubic feet)
        # 1 inch over 1 acre = 43560/12 = 3630 ft³
        if "hru_in_to_cf" not in ds and "hru_area" in ds:
            ds["hru_in_to_cf"] = xr.DataArray(
                ds["hru_area"].values * (43560.0 / 12.0),
                dims=("nhru",),
                attrs={
                    "units": "cubic_feet_per_inch_acre",
                    "long_name": "Inches to cubic feet conversion factor",
                },
            )

        # temp_units: 0 = Fahrenheit (PRMS convention)
        if "temp_units" not in ds:
            ds["temp_units"] = xr.DataArray(
                np.int32(0),
                attrs={"long_name": "Temperature units (0=F, 1=C)"},
            )

        # snarea_curve: snow depletion curve (11 values, default all 1.0)
        if "snarea_curve" not in ds:
            ds["snarea_curve"] = xr.DataArray(
                np.ones(11, dtype=np.float64),
                dims=("ndeplval",),
                attrs={"long_name": "Snow area depletion curve"},
            )
```

Add `"doy"`, `"hru_in_to_cf"`, `"temp_units"`, `"snarea_curve"` to `_DEFAULTS_SPECIAL`.

**Step 5: Add segment-level defaults in `_apply_defaults`**

After the HRU defaults, add segment-level defaults (only when nsegment dim exists):

```python
        # Segment-level defaults (only when routing topology is present)
        nseg_vars = [v for v in ds.data_vars if "nsegment" in (ds[v].dims or ())]
        if nseg_vars:
            nsegment = ds[nseg_vars[0]].sizes.get("nsegment", 0)
            if nsegment > 0:
                seg_defaults: dict[str, tuple[float | int, str]] = {
                    "mann_n": (0.04, "Manning's roughness coefficient"),
                    "seg_depth": (1.0, "Bankfull water depth"),
                    "segment_flow_init": (0.0, "Initial flow in segment"),
                    "obsout_segment": (0, "Observed streamflow segment index"),
                }
                for name, (val, desc) in seg_defaults.items():
                    if name not in ds:
                        dtype = np.int32 if isinstance(val, int) else np.float64
                        ds[name] = xr.DataArray(
                            np.full(nsegment, val, dtype=dtype),
                            dims=("nsegment",),
                            attrs={"long_name": desc},
                        )

                # tosegment_nhm: copy from tosegment if available
                if "tosegment_nhm" not in ds and "tosegment" in ds:
                    ds["tosegment_nhm"] = xr.DataArray(
                        ds["tosegment"].values.copy(),
                        dims=("nsegment",),
                        attrs={"long_name": "NHM downstream segment ID"},
                    )
```

**Step 6: Run tests to verify**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestApplyDefaults -v`
Expected: All pass.

**Step 7: Commit**

Commit: `feat: add 34 missing pywatershed parameters to step 13 defaults`

---

### Task 3: Update `parameter_metadata.yml` with new entries

Add metadata entries for all newly added parameters so validation works.

**Files:**
- Modify: `src/hydro_param/data/pywatershed/parameter_metadata.yml`

**Step 1: Add metadata entries**

Add after the existing depression storage section (~line 183):

```yaml
  # --- Depression storage (operational defaults) ---
  dprst_et_coef:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Fraction of unsatisfied PET applied to depression storage"

  dprst_flow_coef:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Coefficient for open depression linear flow routing"

  dprst_frac_init:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Initial depression storage as fraction of maximum"

  dprst_frac_open:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Fraction of open depression area generating runoff"

  dprst_seep_rate_clos:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Seepage rate coefficient for closed depressions"

  dprst_seep_rate_open:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Seepage rate coefficient for open depressions"

  sro_to_dprst_imperv:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Fraction of impervious runoff to depression storage"

  sro_to_dprst_perv:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Fraction of pervious runoff to depression storage"

  op_flow_thres:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Open depression outflow threshold fraction"

  va_clos_exp:
    dimension: nhru
    units: dimensionless
    valid_range: [0.0001, 10.0]
    required: false
    description: "Closed depression storage-area exponent"

  va_open_exp:
    dimension: nhru
    units: dimensionless
    valid_range: [0.0001, 10.0]
    required: false
    description: "Open depression storage-area exponent"
```

Add snow/atmosphere/soilzone/channel entries:

```yaml
  # --- Snow (additional defaults) ---
  cecn_coef:
    dimension: [nhru, 12]
    units: calories_per_degC
    valid_range: [0.0, 20.0]
    required: false
    description: "Monthly convection condensation energy coefficient"

  rad_trncf:
    dimension: nhru
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Solar radiation transmission coefficient through canopy"

  melt_force:
    dimension: nhru
    units: julian_day
    valid_range: [1, 366]
    required: false
    description: "Julian date to force spring snowmelt stage"

  melt_look:
    dimension: nhru
    units: julian_day
    valid_range: [1, 366]
    required: false
    description: "Julian date to start looking for spring snowmelt"

  snowinfil_max:
    dimension: nhru
    units: inches
    valid_range: [0.0, 20.0]
    required: false
    description: "Maximum snow infiltration per day"

  snowpack_init:
    dimension: nhru
    units: inches
    valid_range: [0.0, 5000.0]
    required: false
    description: "Initial snowpack water equivalent"

  hru_deplcrv:
    dimension: nhru
    units: integer
    valid_range: [1, 10]
    required: false
    description: "Index of snow depletion curve"

  tstorm_mo:
    dimension: [nhru, 12]
    units: integer
    valid_range: [0, 1]
    required: false
    description: "Monthly thunderstorm prevalence flag"

  snarea_curve:
    dimension: ndeplval
    units: decimal_fraction
    valid_range: [0.0, 1.0]
    required: false
    description: "Snow area depletion curve values"

  # --- Atmosphere (additional defaults) ---
  ppt_rad_adj:
    dimension: [nhru, 12]
    units: inches
    valid_range: [0.0, 0.5]
    required: false
    description: "Precipitation threshold for radiation adjustment"

  radadj_intcp:
    dimension: [nhru, 12]
    units: dimensionless
    valid_range: [0.0, 1.0]
    required: false
    description: "Temperature range adjustment intercept for radiation"

  radadj_slope:
    dimension: [nhru, 12]
    units: dimensionless
    valid_range: [0.0, 1.0]
    required: false
    description: "Temperature range adjustment slope for radiation"

  tmax_index:
    dimension: [nhru, 12]
    units: degF
    valid_range: [0.0, 120.0]
    required: false
    description: "Monthly maximum temperature index"

  temp_units:
    dimension: one
    units: integer
    valid_range: [0, 1]
    required: false
    description: "Temperature units (0=Fahrenheit, 1=Celsius)"

  # --- Soilzone (additional defaults) ---
  pref_flow_infil_frac:
    dimension: nhru
    units: decimal_fraction
    valid_range: [-1.0, 1.0]
    required: false
    description: "Fraction of infiltration to preferential flow reservoir"

  # --- Channel (additional defaults) ---
  mann_n:
    dimension: nsegment
    units: dimensionless
    valid_range: [0.001, 1.0]
    required: false
    description: "Manning's roughness coefficient"

  seg_depth:
    dimension: nsegment
    units: feet
    valid_range: [0.0, 1000.0]
    required: false
    description: "Bankfull water depth in segment"

  segment_flow_init:
    dimension: nsegment
    units: cfs
    valid_range: [0.0, 100000.0]
    required: false
    description: "Initial flow in stream segment"

  obsout_segment:
    dimension: nsegment
    units: integer
    valid_range: [0, 100000]
    required: false
    description: "Index of measured streamflow station replacing segment outflow"

  tosegment_nhm:
    dimension: nsegment
    units: integer
    valid_range: [0, 1000000]
    required: false
    description: "NHM downstream segment ID"

  # --- Structural ---
  hru_in_to_cf:
    dimension: nhru
    units: cubic_feet_per_inch_acre
    valid_range: [0.0, 1000000000.0]
    required: false
    description: "Inches-to-cubic-feet conversion factor per HRU"

  seg_slope:
    dimension: nsegment
    units: decimal_fraction
    valid_range: [0.0, 10.0]
    required: false
    description: "Stream segment slope"

  segment_type:
    dimension: nsegment
    units: integer
    valid_range: [0, 3]
    required: false
    description: "Segment type (0=segment, 1=headwater, 2=lake, 3=replacement)"

  obsin_segment:
    dimension: nsegment
    units: integer
    valid_range: [0, 100000]
    required: false
    description: "Index of observed inflow station for segment"
```

Also update existing entries that say `dimension: one` to say `dimension: nhru` (the 21 scalars that should be per-HRU):

- `den_init`, `den_max`, `settle_const`, `emis_noppt`, `freeh2o_cap`, `potet_sublim`, `albset_rna`, `albset_snm`, `albset_rnm`, `albset_sna`, `radj_sppt`, `radj_wppt`, `soil_moist_init_frac`, `soil_rechr_init_frac`, `ssstor_init_frac`, `gwstor_init`, `gwstor_min`, `dprst_depth_avg`, `transp_tmax` — all change `dimension: one` → `dimension: nhru`
- `radmax` — change `dimension: one` → `dimension: [nhru, 12]`
- `tmax_allsnow` — change `dimension: one` → `dimension: [nhru, 12]`

**Step 2: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py tests/test_pywatershed_formatter.py -v`
Expected: All pass.

**Step 3: Commit**

Commit: `feat: add parameter metadata for 34 new pywatershed defaults`

---

### Task 4: Verify against pywatershed v2.0

End-to-end verification: re-run derivation on DRB data and attempt to load in pywatershed.

**Step 1: Re-run pywatershed derivation on DRB data**

```bash
pixi run -e dev hydro-param pywatershed run \
    --config /tmp/drb-e2e-3/configs/pywatershed_run.yml
```

**Step 2: Verify parameters.nc loads in pywatershed**

```bash
pixi run -e pws-test python -c "
import pywatershed as pws
import numpy as np
from pathlib import Path

model_dir = Path('/tmp/drb-e2e-3/models/pywatershed')
params = pws.parameters.PrmsParameters.from_netcdf(
    model_dir / 'parameters.nc', use_xr=True
)
print(f'Loaded {len(params.data_vars)} params')

control = pws.Control(
    start_time=np.datetime64('1980-10-01T00:00:00'),
    end_time=np.datetime64('1980-10-05T00:00:00'),
    time_step=np.timedelta64(24, 'h'),
    options={
        'input_dir': str(model_dir / 'forcing'),
        'budget_type': 'warn',
        'calc_method': 'numpy',
    },
)

nhm_processes = [
    pws.PRMSSolarGeometry, pws.PRMSAtmosphere,
    pws.PRMSCanopy, pws.PRMSSnow,
    pws.PRMSRunoff, pws.PRMSSoilzone,
    pws.PRMSGroundwater, pws.PRMSChannel,
]

model = pws.Model(nhm_processes, control=control, parameters=params)
print('Model created!')
model.advance()
model.calculate()
print('First timestep OK!')
"
```

**Step 3: Fix any remaining issues discovered during verification**

This step is intentionally open-ended.  pywatershed may surface additional
issues (dtype mismatches, missing coords, soltab loading).  Document and
fix each as discovered.

**Step 4: Run full test suite**

Run: `pixi run -e dev check`
Expected: All pass (lint, format, typecheck, tests).

**Step 5: Commit any verification fixes**

Commit: `fix: address pywatershed v2.0 runtime issues from verification`

---

### Task 5: Final commit and PR

**Step 1: Verify all checks pass**

Run: `pixi run -e dev check` and `pixi run -e dev pre-commit`

**Step 2: Create PR**

Branch: `fix/<issue-number>-pywatershed-compat`
PR title: `feat: pywatershed v2.0 runtime compatibility — add missing params and fix shapes`
