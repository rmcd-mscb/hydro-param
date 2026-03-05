# Parameter Audit Design

**Date:** 2026-03-05
**Status:** Approved
**Output:** `docs/plans/parameter_audit_2026-03-05.md` (audit findings document)

## Purpose

Audit hydro-param's pywatershed parameterization against pywatershed's actual
source-code interface. Identify drift toward PRMS/NHM conventions, missing
parameters, incorrect mappings, and engineering decisions masquerading as
physical derivations. Planning only — no code changes.

## Ground Truth

- **Primary source:** `pywatershed/static/metadata/parameters.yaml` (278 PRMS
  parameters with dims, units, valid ranges, module assignments)
- **Install path:** `.pixi/envs/pws-test/lib/python3.11/site-packages/pywatershed/`
- **Reference parameterization:** `drb_2yr/myparam.param` (read by pywatershed
  at runtime — this is the working ground truth for the test domain)

## Scope

**In scope:**
- Parameters consumed by the 9 target process classes: PRMSRunoff, PRMSSoilzone,
  PRMSGroundwater, PRMSChannel, PRMSSnow, PRMSSolarGeometry, PRMSAtmosphere,
  PRMSCanopy, PRMSEt
- Datasets in `pipeline.yml` that feed these parameters
- SIR field-to-parameter mappings in `pywatershed_run.yml`

**Out of scope:**
- Control variables (138)
- Lake, cascade, and dynamic parameter sets
- Process classes hydro-param doesn't target (starfit, MODFLOW)

## Approach: Source-Code-First (Approach A)

Start from pywatershed source → work backward to hydro-param.

1. Extract canonical parameter list from pywatershed process class metadata
2. Cross-reference against hydro-param's derivation code and config
3. Flag mismatches in names, units, dimensions, valid ranges
4. Audit datasets and SIR mappings for orphans or misalignments
5. Classify each parameter as physically based vs. engineering decision
6. Build per-HRU validation plan keyed to canonical list

## Output Document Structure

### Section (a): Corrections (Tasks 1–3)

Corrections table with columns:

| Parameter | Issue Type | Current | Expected | Source |

Issue types:
- `missing` — pywatershed needs it, hydro-param doesn't produce it
- `name_mismatch` — hydro-param uses a different name than pywatershed expects
- `unit_mismatch` — output units don't match `parameters.yaml` units
- `dim_mismatch` — output dimensions don't match (e.g., nhru vs nmonth×nhru)
- `range_violation` — derived values outside `parameters.yaml` valid range
- `orphan_dataset` — dataset in pipeline.yml with no canonical param consumer
- `orphan_mapping` — SIR mapping targeting a name not in pywatershed metadata

### Section (b): Canonical Parameter List (Task 4)

Organized by process class. For each of the 9 target classes, extract every
parameter from the `modules` field in `parameters.yaml`.

Per-process-class table:

| Parameter | pywatershed dims | pywatershed units | hydro-param status | hydro-param source | Notes |

Status values: `derived`, `defaulted`, `missing`, `extra`.

### Section (c): drb_2yr Validation Plan (Task 5)

Per-HRU scatter + correlation, aligned to the 15-step derivation DAG.

**Method:**
1. Load hydro-param `parameters.nc` and drb_2yr `myparam.param` (as read by
   pywatershed at runtime), matched on `nhm_id`
2. Per parameter: R², bias (mean difference), RMSE, scatter plot
3. Classification thresholds:
   - **Good:** R² > 0.9, bias < 5%
   - **Fair:** R² > 0.7
   - **Weak:** R² > 0.4
   - **Poor:** R² ≤ 0.4
   - **Uniform-match:** both constant, same value
   - **Uniform-differs:** both constant, different values
4. Group results by derivation step for incremental validation
5. Known divergence register — document expected differences upfront
   (gNATSGO vs SSURGO, NLCD vintage, etc.)

**Key requirement:** Load `myparam.param` using pywatershed's own reader so the
reference values are exactly what the model sees at runtime.

### Section (d): Physical Basis / Engineering Decision Register (Task 6)

Three classifications:

| Classification | Definition |
|---|---|
| **Physically based** | Derived from spatial data, empirical relationships, or process equations |
| **Engineering decision** | Value driven by architectural constraints, pragmatic defaults, or code design |
| **Calibration seed** | Initial guess intended for calibration — reasonable range matters, not exact value |

Per-parameter table:

| Parameter | Classification | Rationale | Confidence | Improvement Path |

- **Confidence:** high / medium / low
- **Improvement path:** for engineering decisions and low-confidence entries,
  what data source or method would upgrade it (with links to open issues)

## Non-Goals

- No code changes in this audit
- No new tests or scripts (validation plan is a design, not implementation)
- No changes to pywatershed config or derivation code
- No performance analysis
