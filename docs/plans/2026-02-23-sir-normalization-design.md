# SIR Normalization Layer Design

**Date:** 2026-02-23
**Status:** Approved
**Implements:** design.md Appendix A.3 (v5.3), A.8 (A')

## Problem

The pipeline currently produces raw gdptools output with source-native variable
names (`elevation`, `slope`, `LndCov`, `FctImp`, `ksat`) and source-native units
(`log10(cm/hr)`, `%`, `m3/m3`). Derivation plugins must guess names and handle
unit conversions themselves. The design doc requires a Standardized Internal
Representation (SIR) with self-documenting names, guaranteed canonical units, and
schema validation.

## Design Decisions

1. **Full design doc naming convention:** `<base>_<unit>_<aggregation>` (e.g.,
   `elevation_m_mean`, `slope_deg_mean`, `clay_pct_mean`).
2. **Schema auto-generated from registry metadata:** No new config files. The
   canonical name is derived from `VariableSpec.name` + `.units` + requested
   statistic.
3. **Unit normalization at the SIR boundary:** All log-transformed, unusual, or
   non-SI units are converted to canonical SI at the SIR boundary. Derivation
   plugins trust SIR units blindly.
4. **New Stage 5 in the pipeline:** Reads raw stage 4 per-variable files,
   normalizes (rename + convert + attrs), writes normalized per-variable files.
5. **Per-variable files with canonical names:** Same incremental pattern as
   stage 4. No monolithic NetCDF/Zarr. Files stored in `output_dir/sir/`.

## Canonical Naming Convention

### Function: `canonical_name(name, units, stat) -> str`

Pattern: `<base>_<unit_abbrev>[_<stat>]`

- `<base>` is the physical quantity in lowercase (e.g., `elevation`, `slope`,
  `clay`, `ksat`).
- `<unit_abbrev>` is a short canonical abbreviation (see table below).
  Dimensionless quantities omit `<unit_abbrev>` (e.g., `lambda_mean`).
- `<stat>` is the aggregation method (`mean`, `min`, `max`, `std`, `sum`,
  `majority`, etc.). Always present for zonal statistics.

### Unit Abbreviation Table

| Source Units | Canonical Units | Abbreviation | Conversion |
|-------------|----------------|--------------|------------|
| `m` | `m` | `m` | none |
| `degrees` | `degrees` | `deg` | none |
| `%` | `%` | `pct` | none |
| `m3/m3` | `m3/m3` | `m3_m3` | none |
| `g/cm3` | `g/cm3` | `g_cm3` | none |
| `cm` | `cm` | `cm` | none |
| `cm/hr` | `cm/hr` | `cm_hr` | none |
| `log10(cm/hr)` | `cm/hr` | `cm_hr` | `10^x` |
| `log10(kPa)` | `kPa` | `kPa` | `10^x` |
| `log10(%)` | `%` | `pct` | `10^x` |
| `log10(kPa^-1)` | `kPa^-1` | `kPa_inv` | `10^x` |
| `day_of_year` | `day_of_year` | `doy` | none |
| (empty) | (dimensionless) | (omit) | none |

### Categorical Variables

Categorical fraction columns from `ZonalGen(categorical=True)`:
- Source: `LndCov_11`, `LndCov_21`, `LndCov_41`, ...
- Canonical: `lndcov_frac_11`, `lndcov_frac_21`, `lndcov_frac_41`, ...

The prefix is lowercased and `_frac_` is inserted before the class code.

## Module Structure

### New module: `src/hydro_param/sir.py`

```python
@dataclass
class SIRVariableSchema:
    """Schema entry for a single SIR variable."""
    canonical_name: str         # e.g., "elevation_m_mean"
    source_name: str            # e.g., "elevation" (raw gdptools column)
    source_units: str           # e.g., "m"
    canonical_units: str        # e.g., "m"
    long_name: str              # CF-1.8 long_name
    categorical: bool           # True for NLCD class fractions
    valid_range: tuple | None   # (min, max) or None
    conversion: str | None      # e.g., "log10_to_linear" or None


def canonical_name(name: str, units: str, stat: str) -> str:
    """Generate canonical SIR variable name from components."""
    ...


def build_sir_schema(
    resolved: list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec]]],
    config: PipelineConfig,
) -> list[SIRVariableSchema]:
    """Auto-generate SIR schema from stage 2 resolved datasets."""
    ...


def normalize_sir(
    raw_files: dict[str, Path],
    schema: list[SIRVariableSchema],
    output_dir: Path,
    feature_ids: object,
    id_field: str,
) -> dict[str, Path]:
    """Normalize raw per-variable files to canonical SIR format.

    Reads raw CSVs, renames columns, converts units, writes
    normalized per-variable CSVs to output_dir/sir/.

    Returns {canonical_name: Path} for normalized files.
    """
    ...


@dataclass
class SIRValidationWarning:
    variable: str
    check_type: str   # "missing", "units", "range", "nan_coverage"
    message: str


def validate_sir(
    sir_files: dict[str, Path],
    schema: list[SIRVariableSchema],
    *,
    strict: bool = False,
) -> list[SIRValidationWarning]:
    """Validate normalized SIR files against schema.

    Returns list of warnings. Raises SIRValidationError if
    strict=True and any warnings are produced.
    """
    ...
```

## Pipeline Integration

### Updated pipeline flow

```
Stage 1: Resolve fabric
Stage 2: Resolve datasets + build SIR schema
Stage 3: Weights (gdptools internal)
Stage 4: Process datasets -> raw per-variable CSVs
          (source names, source units, output_dir/<category>/<var>.csv)
Stage 5: Normalize SIR -> normalized per-variable CSVs
          (canonical names, canonical units, output_dir/sir/<var>.csv)
          + validate_sir() logs warnings or raises in strict mode
```

### `PipelineResult` changes

```python
@dataclass
class PipelineResult:
    output_dir: Path
    static_files: dict[str, Path]      # raw stage 4 output (unchanged)
    temporal_files: dict[str, Path]     # raw temporal output (unchanged)
    categories: dict[str, str]
    fabric: gpd.GeoDataFrame | None
    sir_files: dict[str, Path]         # NEW: normalized SIR files
    sir_schema: list[SIRVariableSchema] # NEW: schema used for normalization
```

### `load_sir()` update

`load_sir()` now loads from `sir_files` (normalized) instead of `static_files`
(raw). A `load_raw_sir()` method is available for debugging/provenance.

### Config addition

```yaml
processing:
  sir_validation: "tolerant"  # or "strict" for dev/debugging
```

## Validation Rules

### Checks performed by `validate_sir()`:

1. **Completeness**: All schema variables present in output. Missing = warning
   (pipeline configs may request subsets).

2. **Attrs**: Every variable has `units` and `long_name` xarray attributes.

3. **Value ranges**: If `valid_range` declared in schema, warn if values fall
   outside. Not an error -- real data can have edge cases.

4. **NaN coverage**:
   - **Informational**: Individual NaN values are expected (partial spatial
     coverage, e.g., gridMET not fully covering CONUS, coastal HRUs). gdptools
     returns NaN for features without source data coverage. This is a feature,
     not a bug.
   - **Warning**: Entire variable is 100% NaN (likely processing failure).
   - **No gap-filling**: SIR reports what it has. Gap-filling (nearest-neighbor)
     is a downstream concern for derivation plugins.

5. **Dimension consistency**: All variables share the same feature dimension
   with identical coordinates.

### Error handling

- Default: warnings logged at WARNING level, pipeline continues (fault tolerant).
- `strict=True`: any warning raises `SIRValidationError` (dev/debugging mode).
- Controlled by `processing.sir_validation` config field.

## Impact on Existing Code

### pywatershed derivation plugin

- `rename_sir_variables()` becomes a no-op or is removed -- SIR already uses
  canonical names.
- Plugin needs updating to expect canonical SIR names (e.g.,
  `elevation_m_mean` instead of `elevation`).
- Unit conversions FROM canonical SI TO PRMS units (m->ft, deg->rad) stay in
  the plugin. These are model-specific, not SIR concerns.

### Existing tests

- No breakage -- Stage 5 is additive.
- New tests needed for `sir.py` functions.
- Derivation tests need updating to use canonical SIR names in fixtures.

## Future Considerations

- **Temporal SIR normalization**: Currently scoped to static variables only.
  Temporal datasets (gridMET, SNODAS) pass through to derivation plugins
  unchanged. Temporal SIR normalization is a future extension.
- **Gap-filling utilities**: A future `sir_gap_fill()` function could apply
  nearest-neighbor or spatial interpolation to fill NaN values. This is NOT
  part of SIR normalization -- it would be a separate optional step.
- **SIR schema export**: The schema could be written to a metadata file
  alongside the SIR for provenance and reproducibility.
