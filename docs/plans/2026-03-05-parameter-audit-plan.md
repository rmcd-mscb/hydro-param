# Parameter Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce a single audit findings document (`docs/plans/parameter_audit_2026-03-05.md`) that cross-references hydro-param's pywatershed parameterization against pywatershed's actual source-code interface.

**Architecture:** Source-code-first audit. Extract canonical parameter list from pywatershed's `parameters.yaml`, cross-reference against hydro-param's derivation code, flag all mismatches, build validation plan. Research/documentation only — no code changes.

**Tech Stack:** pywatershed (installed in `pws-test` pixi environment), Python for metadata extraction, Markdown for output.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `.pixi/envs/pws-test/lib/python3.11/site-packages/pywatershed/static/metadata/parameters.yaml` | **Ground truth** — 278 PRMS parameters with dims, units, ranges, module assignments |
| `src/hydro_param/derivations/pywatershed.py` | hydro-param's derivation implementation (lines 67–226: `_DEFAULTS`, `_PARAM_DIMS`) |
| `docs/reference/parameter_inventory.md` | Current inventory (100 params: 95 static + 5 forcing) |
| `docs/reference/pywatershed_dataset_param_map.yml` | Authoritative dataset→parameter mapping reference |
| `pw-check/configs/pywatershed_run.yml` | Active pywatershed run config (SIR mappings) |
| `pw-check/configs/pipeline.yml` | Active pipeline config (dataset definitions) |

## Target Process Classes (9)

The canonical parameter list is the **union** of parameters consumed by these 9 classes. Extract via `cls.get_parameters()`:

| Class | Param Count | Key Parameters |
|-------|-------------|----------------|
| PRMSRunoff | 23 | carea_max, dprst_*, smidx_*, soil_moist_max, snowinfil_max |
| PRMSSoilzone | 22 | cov_type, soil_type, soil_moist_max, sat_threshold, pref_flow_* |
| PRMSSnow | 25 | albset_*, cecn_coef, covden_*, den_*, melt_*, snarea_*, rad_trncf |
| PRMSAtmosphere | 27 | adjmix_rain, dday_*, jh_coef*, radmax, rain/snow_cbh_adj, tmax_* |
| PRMSChannel | 13 | K_coef, mann_n, seg_*, tosegment*, x_coef |
| PRMSSolarGeometry | 7 | doy, hru_area, hru_aspect, hru_lat, hru_slope, radj_sppt, radj_wppt |
| PRMSCanopy | 7 | cov_type, covden_*, potet_sublim, *_intcp |
| PRMSGroundwater | 6 | gwflow_coef, gwsink_coef, gwstor_*, hru_area, hru_in_to_cf |
| PRMSEt | 2 | dprst_frac, hru_percent_imperv |

**Union: 105 unique parameters** (vs hydro-param's current 100).

---

## Task 1: Extract Canonical Parameter Metadata

**Files:**
- Read: `.pixi/envs/pws-test/lib/python3.11/site-packages/pywatershed/static/metadata/parameters.yaml`
- Output: Start building `docs/plans/parameter_audit_2026-03-05.md` Section (b)

**Step 1: Run extraction script**

Run this in the `pws-test` environment to produce the canonical list:

```bash
pixi run -e pws-test python -c "
import yaml
from pathlib import Path
import pywatershed as pws

# Load metadata
meta_path = Path('.pixi/envs/pws-test/lib/python3.11/site-packages/pywatershed/static/metadata/parameters.yaml')
with open(meta_path) as f:
    all_meta = yaml.safe_load(f)

# Get params per class
classes = {
    'PRMSRunoff': pws.PRMSRunoff,
    'PRMSSoilzone': pws.PRMSSoilzone,
    'PRMSGroundwater': pws.PRMSGroundwater,
    'PRMSChannel': pws.PRMSChannel,
    'PRMSSnow': pws.PRMSSnow,
    'PRMSSolarGeometry': pws.PRMSSolarGeometry,
    'PRMSAtmosphere': pws.PRMSAtmosphere,
    'PRMSCanopy': pws.PRMSCanopy,
    'PRMSEt': pws.PRMSEt,
}

for cls_name, cls in classes.items():
    params = sorted(set(cls.get_parameters()))
    print(f'\n### {cls_name} ({len(params)} parameters)\n')
    print('| Parameter | Type | Dims | Units | Min | Max | Default |')
    print('|-----------|------|------|-------|-----|-----|---------|')
    for p in params:
        m = all_meta.get(p, {})
        dims = ', '.join(m.get('dims', {}).values()) if m.get('dims') else 'scalar'
        print(f'| \`{p}\` | {m.get(\"type\",\"?\")} | ({dims}) | {m.get(\"units\",\"?\")} | {m.get(\"minimum\",\"?\")} | {m.get(\"maximum\",\"?\")} | {m.get(\"default\",\"?\")} |')
"
```

**Step 2: Record the output as Section (b) of the audit document**

Copy the table output into the audit document. For each parameter, add two columns manually in the next task: `hydro-param status` and `hydro-param source`.

---

## Task 2: Cross-Reference Against hydro-param Inventory

**Files:**
- Read: `docs/reference/parameter_inventory.md` (lines 48–278 — the full parameter table)
- Read: `src/hydro_param/derivations/pywatershed.py` (lines 67–218 — `_DEFAULTS` and `_PARAM_DIMS`)
- Output: Complete Section (b) status columns + start Section (a)

**Step 1: For each of the 105 canonical parameters, classify hydro-param status**

Walk the canonical list from Task 1. For each parameter, determine status:

- **`derived`** — hydro-param computes it in a `_derive_*()` function (steps 1–12)
- **`defaulted`** — hydro-param assigns a constant in `_DEFAULTS` dict or `_apply_defaults()` (step 13)
- **`calibration_seed`** — hydro-param assigns in `_derive_calibration_seeds()` (step 14)
- **`missing`** — pywatershed needs it but hydro-param doesn't produce it
- **`extra`** — hydro-param produces it but pywatershed's 9 target classes don't need it

Cross-reference sources:
- `_DEFAULTS` dict (lines 67–132): 66 defaulted parameters
- `_DEFAULTS_SPECIAL` (lines 135–148): 10 special-case defaults
- `_PARAM_DIMS` (lines 155–218): dimension assignments
- Derivation steps 1–12: `_derive_geometry`, `_derive_topology`, `_derive_topography`, `_derive_landcover`, `_derive_soils`, `_derive_waterbodies`, `_derive_forcing`, `_derive_lookup_tables`, `_derive_soltab`, `_compute_pet_coefficients`, `_derive_climate_params`, `_derive_routing`
- Calibration seeds step 14: `_derive_calibration_seeds`

**Step 2: Check dimension mismatches**

Compare `_PARAM_DIMS` entries against `parameters.yaml` dims. Flag any where hydro-param has a different dimension than pywatershed expects.

Known dimension issues to check:
- `albset_rna/rnm/sna/snm`: hydro-param uses `(scalar,)`, pywatershed says `(nhru,)` — **likely mismatch**
- `den_init`, `den_max`, `settle_const`: hydro-param uses `(scalar,)`, pywatershed says `(nhru,)` — **likely mismatch**
- All `(nmonth, nhru)` params: verify hydro-param matches

**Step 3: Check unit mismatches**

For each derived parameter, verify the unit conversion in hydro-param matches pywatershed's expected units:
- `hru_elev`: pywatershed expects `feet` — hydro-param converts m→ft ✓
- `hru_area`: pywatershed expects `acres` — hydro-param converts m²→acres ✓
- `seg_length`: pywatershed expects `meters` — check hydro-param
- `K_coef`: pywatershed expects `hours` — hydro-param computes seg_len/velocity in hours
- `soil_moist_max`: pywatershed expects `inches` — hydro-param converts mm→inches
- `jh_coef`: pywatershed expects `per degrees Fahrenheit` — check
- `mann_n`: pywatershed expects `seconds / meter ** (1/3)` — check

**Step 4: Check valid range violations**

For derived parameters, check if the code clips/bounds to pywatershed's valid range:
- `soil_moist_max`: pywatershed range [1e-05, 20.0] — hydro-param clips [0.5, 20] ✓
- `hru_slope`: pywatershed range [0.0, 10.0] — check hydro-param
- `cov_type`: pywatershed range [0, 4] — check lookup table
- `soil_type`: pywatershed range [1, 3] — check lookup table
- `K_coef`: pywatershed range [0.01, 24.0] — check clipping

**Step 5: Record all findings as Section (a) corrections table**

For each mismatch found, add a row to the corrections table:

```markdown
| Parameter | Issue Type | Current | Expected | Source |
|-----------|------------|---------|----------|--------|
| `albset_rna` | dim_mismatch | (scalar,) | (nhru,) | parameters.yaml |
```

---

## Task 3: Audit Datasets and SIR Mappings

**Files:**
- Read: `pw-check/configs/pipeline.yml` — active pipeline config
- Read: `pw-check/configs/pywatershed_run.yml` — active pywatershed run config
- Read: `src/hydro_param/data/datasets/*.yml` — all 8 dataset registry files
- Output: Add orphan_dataset and orphan_mapping entries to Section (a)

**Step 1: List all datasets in pipeline.yml**

For each dataset, trace forward to the canonical parameter it ultimately feeds:
- `dem_3dep_10m` → `hru_elev`, `hru_slope`, `hru_aspect` ✓
- `gnatsgo_rasters` → `soil_type`, `soil_moist_max`, `soil_rechr_max_frac` ✓
- `nlcd_osn_lndcov` → `cov_type` ✓
- `nlcd_osn_fctimp` → `hru_percent_imperv` ✓
- `polaris_30m` → `soil_type`, `sat_threshold` ✓
- `gridmet` → `jh_coef`, `transp_beg`, `transp_end`, forcing ✓
- `snodas` → `snarea_thresh` ✓
- Any other datasets → flag if no canonical consumer

**Step 2: List all SIR variable mappings in pywatershed_run.yml**

For each mapping (`sir_variable → parameter`), verify the target parameter name exists in the 105-parameter canonical list.

**Step 3: Flag orphans**

- `orphan_dataset`: dataset in pipeline.yml → SIR variable → no canonical param
- `orphan_mapping`: SIR mapping → parameter name not in pywatershed's metadata

---

## Task 4: Build drb_2yr Validation Plan (Section c)

**Files:**
- Read: pywatershed source for how it reads `myparam.param`
- Read: `docs/reference/parameter_inventory.md` (DRB comparison section, lines 338–439)
- Output: Section (c) of audit document

**Step 1: Document how to load reference parameters**

pywatershed reads `myparam.param` via its `Parameters` class. Document the exact loading code:

```python
import pywatershed as pws
from pathlib import Path

# Load reference parameters the way pywatershed does
params = pws.Parameters.from_netcdf(
    Path("test_data/drb_2yr/myparam.param"),
    use_xr=True
)
# OR if it's a PRMS-format file:
params = pws.parameters.PrmsParameters.load(
    Path("test_data/drb_2yr/myparam.param")
)
```

Verify which loading method pywatershed actually uses and document it.

**Step 2: Design per-parameter comparison procedure**

For each of the 105 canonical parameters (where both hydro-param and reference have values):

```python
import numpy as np
from scipy import stats

def compare_parameter(hp_values, ref_values, name):
    """Per-HRU scatter + correlation for one parameter."""
    # Filter NaN
    mask = ~(np.isnan(hp_values) | np.isnan(ref_values))
    hp = hp_values[mask]
    ref = ref_values[mask]

    if len(hp) == 0:
        return {"name": name, "status": "no_overlap"}

    # Check if both uniform
    hp_uniform = np.allclose(hp, hp[0])
    ref_uniform = np.allclose(ref, ref[0])

    if hp_uniform and ref_uniform:
        if np.allclose(hp[0], ref[0]):
            return {"name": name, "status": "uniform_match", "value": hp[0]}
        else:
            return {"name": name, "status": "uniform_differs",
                    "hp": hp[0], "ref": ref[0]}

    # Compute stats
    r2 = np.corrcoef(hp, ref)[0, 1] ** 2
    bias = np.mean(hp - ref) / (np.mean(ref) + 1e-10) * 100  # % bias
    rmse = np.sqrt(np.mean((hp - ref) ** 2))

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

**Step 3: Organize by derivation step**

Group the 105 parameters by derivation step (1–14) so validation can proceed incrementally:

| Step | Parameters to Validate | Expected Comparison Notes |
|------|----------------------|--------------------------|
| 1 | hru_area, hru_lat | Should match exactly (same fabric) |
| 2 | tosegment, hru_segment, seg_length | Should match exactly (same fabric) |
| 3 | hru_elev, hru_slope, hru_aspect | May differ if DEM source/resolution differs |
| 4 | cov_type, covden_sum, hru_percent_imperv | May differ by NLCD vintage (2019 vs 2001) |
| 5 | soil_type, soil_moist_max, soil_rechr_max_frac | gNATSGO vs SSURGO — expect divergence |
| 6 | dprst_frac, dprst_area_max, hru_type | NHDPlus version differences |
| 7 | prcp, tmax, tmin | Different forcing source/period |
| 8 | srain_intcp, wrain_intcp, snow_intcp, covden_win | Depends on cov_type match |
| 9 | soltab_potsw, soltab_horad_potsw, soltab_sunhrs | Algorithm should match exactly |
| 10 | jh_coef, jh_coef_hru | Climate source differences |
| 11 | transp_beg, transp_end | Climate source differences |
| 12 | K_coef, seg_slope, segment_type | Fabric/NHDPlus version |
| 13 | All defaults | Uniform — check values match |
| 14 | All calibration seeds | Seeds — compare formulas |

**Step 4: Document known divergence register**

List expected differences that are **not bugs**:

| Divergence | Reason | Affected Parameters |
|-----------|--------|-------------------|
| gNATSGO vs SSURGO | Different soil data sources | soil_type, soil_moist_max, soil_rechr_max_frac |
| NLCD 2019 vs 2001 | Different land cover vintage | cov_type, covden_sum, hru_percent_imperv |
| gridMET vs Daymet | Different climate source | jh_coef, transp_beg, transp_end |
| NHDPlus v2.1 vs GFv1.1 | Different routing network | tosegment, seg_slope, K_coef |
| soltab pre-computed vs geometry | Equivalent but different approach | soltab_*, solar geometry params |

**Step 5: Write Section (c) as a validation checklist**

Structure as a step-by-step validation procedure with specific commands to run and expected outcomes.

---

## Task 5: Build Physical Basis Register (Section d)

**Files:**
- Read: `docs/reference/parameter_inventory.md` (category codes and full table)
- Read: `src/hydro_param/derivations/pywatershed.py` (derivation logic)
- Output: Section (d) of audit document

**Step 1: Map inventory categories to audit classifications**

| Inventory Category | Audit Classification |
|-------------------|---------------------|
| DS (Derived Spatial) | Physically based |
| DF (Derived Formula) | Physically based |
| DR (Derived Reclassify) | Physically based |
| DL (Derived Lookup Table) | Physically based (if table is empirically grounded) |
| DA (Derived Algorithm) | Physically based |
| DT (Derived Topology) | Physically based |
| DC (Derived Climate) | Physically based |
| CS-F (Calib Seed Formula) | Calibration seed |
| CS-C (Calib Seed Constant) | Calibration seed |
| DEF (Default Physical) | Engineering decision (if uniform; physically based if spatially variable) |
| PH (Placeholder) | Engineering decision |
| IC (Initial Condition) | Engineering decision |
| FRC (Forcing) | Physically based |
| STR (Structural) | Engineering decision |

**Step 2: For each parameter, assign confidence level**

- **high** — spatial data + validated algorithm, matches reference within expected tolerance
- **medium** — data-derived but from a different source than reference, or uses simplifying assumptions
- **low** — placeholder, uniform default, or known to need improvement

**Step 3: For each engineering decision / low-confidence entry, document improvement path**

Cross-reference with open issues:
- #147 (master placeholder issue)
- #151 (soil_rechr_max_frac — now FIXED)
- #152 (TWI-based seeds)
- #153 (PRISM upgrade)
- #154 (Ksat-based seeds)
- #155 (snow depletion curves)

**Step 4: Write Section (d) as the register table**

```markdown
| Parameter | Classification | Rationale | Confidence | Improvement Path |
|-----------|---------------|-----------|------------|------------------|
| `hru_elev` | Physically based | 3DEP 10m DEM zonal mean | high | — |
| `sat_threshold` | Engineering decision | 999 placeholder, no physics | low | #147: (porosity-FC)×depth from POLARIS |
```

---

## Task 6: Assemble Final Audit Document

**Files:**
- Create: `docs/plans/parameter_audit_2026-03-05.md`

**Step 1: Combine all sections**

Assemble the audit document with this structure:

```markdown
# pywatershed Parameter Audit — 2026-03-05

## Executive Summary
[2-3 sentences: total params, coverage %, key findings]

## (a) Corrections
[Table from Task 2 Step 5 + Task 3 Step 3]

## (b) Canonical Parameter List
[Tables from Task 1 + Task 2 status columns, organized by process class]

## (c) drb_2yr Validation Plan
[Procedure from Task 4, organized by derivation step]

## (d) Physical Basis / Engineering Decision Register
[Register from Task 5]

## Appendix: Data Sources
[Quick reference to pywatershed metadata location and hydro-param source files]
```

**Step 2: Verify completeness**

Check that:
- All 105 canonical parameters appear in Section (b)
- Every issue found in Tasks 2–3 appears in Section (a)
- Section (c) covers all derivation steps 1–14
- Section (d) covers all 100 hydro-param parameters

**Step 3: Commit**

```bash
git add docs/plans/parameter_audit_2026-03-05.md
git commit -m "docs: add pywatershed parameter audit findings

Cross-references hydro-param's 100-parameter inventory against
pywatershed's 105-parameter canonical interface. Documents
corrections, validation plan, and physical basis register."
```

---

## Execution Notes

- **No code changes** — this is purely a research/documentation task
- **Environment**: Use `pws-test` pixi environment for all pywatershed introspection
- **Ground truth**: Always defer to `parameters.yaml` over PRMS docs or NHM tables
- **Known gap**: hydro-param produces 100 params, pywatershed needs 105 — the 5 missing are already documented in `parameter_inventory.md` lines 359–400 (solar geometry, topology IDs, segment spatial params)
- The `soltab_*` and `doy` parameters that hydro-param produces but aren't in the canonical 105 are consumed by pywatershed through different mechanisms — document how
