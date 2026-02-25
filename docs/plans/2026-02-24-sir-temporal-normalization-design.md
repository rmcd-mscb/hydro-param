# Temporal SIR Normalization + Categorical Validation Fix

**Date:** 2026-02-24
**Status:** Approved
**Scope:** Normalize temporal NetCDF output through the SIR layer; fix categorical range validation

## Problem

Stage 5 SIR validation produces 5 spurious warnings on a successful DRB pipeline run:

3 `[missing]` warnings:
- `pr_mm_mean`, `tmmx_K_mean`, `tmmn_K_mean` not found in SIR output

2 `[range]` warnings:
- `lndcov_frac_2020_count` and `lndcov_frac_2021_count` values [33, 435660]
  outside expected range [0.0, 1.0]

### Root causes

**Temporal variables bypass SIR normalization.** `build_sir_schema()` generates
schema entries for all resolved datasets including temporal ones (gridMET).
But `normalize_sir()` only receives `stage4.static_files` — temporal NetCDFs
in `stage4.temporal_files` are never normalized. `validate_sir()` then flags
the temporal schema entries as missing.

**Categorical validation checks all columns.** NLCD categorical CSV files contain
fraction columns (`lndcov_frac_11`, `lndcov_frac_21`, ...) plus a `count` column
(pixel counts per feature). The validation loop iterates ALL columns and checks
them against `valid_range=(0.0, 1.0)`. Pixel counts (33–435,660) fail this check.

### Non-issue: "missing" features

Investigation (see `notebooks/temporal_missing_features.ipynb`) confirmed that
the 134 features "missing" from output are HRUs outside the domain bbox — not a
processing bug. The pipeline correctly processes all 631 features within the
configured bbox `[-75.8, 39.6, -74.4, 42.5]`. Nearest-neighbor gap-filling for
genuine grid coverage gaps is deferred to issue #73.

## Design

### Section 1: Temporal SIR normalization

**1a. New unit table entries and conversion.**

Add to `_UNIT_TABLE` in `sir.py`:

| Source unit | Abbreviation | Canonical unit | Conversion |
|-------------|-------------|----------------|------------|
| `K`         | `C`         | `°C`           | `K_to_C`   |
| `mm`        | `mm`        | `mm`           | None       |
| `W/m2`      | `W_m2`      | `W/m2`         | None       |
| `kg/kg`     | `kg_kg`     | `kg/kg`        | None       |
| `m/s`       | `m_s`       | `m/s`          | None       |

Add `"K_to_C"` case to `apply_conversion()`: `values - 273.15`.

**1b. Schema changes.**

Add `temporal: bool = False` field to `SIRVariableSchema`.

`build_sir_schema()` sets `temporal=True` for entries where
`DatasetEntry.temporal` is true. This requires passing the `DatasetEntry`
information through — the existing signature already receives
`(DatasetEntry, DatasetRequest, var_specs)` tuples.

**1c. New `normalize_sir_temporal()` function.**

```
normalize_sir_temporal(
    temporal_files: dict[str, Path],
    schema: list[SIRVariableSchema],
    resolved: list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec]]],
    output_dir: Path,
) -> dict[str, Path]
```

For each temporal NetCDF:
1. Open the dataset.
2. Build a reverse lookup from gdptools long names to registry variable specs
   (using `VariableSpec.long_name`).
3. For each data variable, map to the canonical SIR name and apply unit
   conversion.
4. Rename variables in the dataset to canonical names.
5. Write normalized NetCDF to `output_dir/`.

Returns mapping of canonical name → normalized file path.

**1d. Variable name mapping.**

Temporal NetCDFs use gdptools long names (`precipitation_amount`,
`daily_maximum_temperature`, `daily_minimum_temperature`). The registry
`VariableSpec` has a `long_name` field. `normalize_sir_temporal()` receives
the resolved var specs and builds: `{long_name: (var_spec, schema_entry)}`.

### Section 2: Categorical validation fix

**2a. Filter columns in `validate_sir()`.**

For categorical schema entries (where `entry.categorical` is true), only
validate columns whose name contains `_frac_`. Skip `count` and any other
non-fraction columns.

### Section 3: Pipeline integration

**3a. Update `stage5_normalize_sir()` in `pipeline.py`.**

After calling `normalize_sir()` for static files:
1. Call `normalize_sir_temporal()` with `stage4.temporal_files`, schema,
   resolved specs, and the SIR output directory.
2. Merge returned temporal SIR files into the `sir_files` dict.
3. Pass the combined `sir_files` to `validate_sir()`.

## Scope boundaries

**In scope:** Temporal normalization (rename + unit convert), categorical
validation fix, unit table expansion.

**Not in scope:** Nearest-neighbor gap-filling (#73), temporal chunking
changes, new temporal processing strategies.

## Files modified

- `src/hydro_param/sir.py` — unit table, `SIRVariableSchema.temporal`,
  `build_sir_schema()`, `normalize_sir_temporal()`, `apply_conversion()`,
  `validate_sir()`
- `src/hydro_param/pipeline.py` — `stage5_normalize_sir()` calls temporal
  normalization
- `tests/test_sir.py` — temporal normalization, K→°C, categorical validation

## Testing

- Existing tests pass unchanged (backward-compatible)
- Unit test: `normalize_sir_temporal()` renames variables and converts K → °C
- Unit test: `apply_conversion()` handles `"K_to_C"`
- Unit test: `validate_sir()` skips count columns for categorical variables
- Unit test: `build_sir_schema()` marks temporal entries correctly

## Risk

- Low — additive changes; existing static normalization path unchanged
- `normalize_sir_temporal()` is new code but follows the same patterns as
  `normalize_sir()`
