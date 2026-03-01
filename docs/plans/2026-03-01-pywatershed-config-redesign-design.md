# Design: pywatershed_run.yml Config Redesign

**Date:** 2026-03-01
**Status:** Draft
**Related issues:** #124 (SIR dataset prefix), #120 (temporal DerivationContext)

## Problem

The current `pywatershed_run.yml` is a black box. It declares `sir_path` and
trusts the derivation plugin to figure out which datasets produced which SIR
variables. The user has no visibility into the data contract, the plugin has no
way to validate expectations against reality, and there is no mechanism for
dataset-specific processing logic (e.g., polaris vs gnatsgo soil derivation).

## Design Principle

The `pywatershed_run.yml` becomes a **consumer-oriented, self-documenting
contract** between the pipeline (Phase 1) and the pywatershed derivation
plugin (Phase 2).

- **Consumer-oriented:** Every section is keyed by what pywatershed needs
  (parameter names), not by what the source dataset provides (variable names).
- **Self-documenting:** Each entry includes a human-readable description so a
  domain scientist can understand the data flow without reading Python code.
- **Contract:** The derivation plugin validates at startup that the SIR
  contains the declared data. Mismatches produce early errors, not silent
  failures deep in derivation logic.

## Config Structure

Three data sections organized by processing mode:

| Section | Purpose | Keys |
|---------|---------|------|
| `static_datasets` | Zonal stats over HRUs | Grouped by domain category (topography, soils, landcover, snow, waterbodies) |
| `forcing` | Temporal time series for model input | pywatershed forcing variable names (prcp, tmax, tmin) |
| `climate_normals` | Long-term statistics for derived params | Derived parameter names (jh_coef, transp_beg, transp_end) |

### Entry Rule

**Only parameters that require SIR data from the pipeline get a config entry.**
Internally derived parameters — lookup table outputs (covden_sum, covden_win,
interception capacities, imperv_stor_max), formula-derived values (jh_coef_hru
from hru_elev), calibration seeds — are the derivation plugin's internal
business and do not appear in the config.

### Parameter Entry Schema

Each entry declares:

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | Pipeline dataset registry name (e.g., `dem_3dep_10m`) |
| `variable` / `variables` | `str \| list[str] \| None` | Source variable(s) used |
| `statistic` | `str \| None` | Zonal statistic applied (mean, categorical) |
| `year` | `int \| list[int] \| None` | NLCD year(s) |
| `time_period` | `list[str] \| None` | Temporal range [start, end] |
| `description` | `str` | Human-readable purpose |

### Category Discovery

Each category has an `available: list[str]` field:
- Populated by `hydro-param init` from the dataset registry
- Validated at runtime against the current registry
- Surfaces available dataset choices directly in the YAML

### Runtime Behavior: Validate and Route

At startup, the derivation plugin:

1. Reads each parameter entry from the config
2. Queries the SIR (via `SIRAccessor`) for matching data using the `source`
   field and issue #124's dataset-prefixed filenames
3. Validates that declared data exists — raises early errors for mismatches
4. Uses `source` to route dataset-specific processing logic (e.g., polaris
   soil texture derivation differs from gnatsgo)

## Issue #124 Integration

This config redesign and issue #124 (SIR dataset prefix) are symbiotic.
Neither is fully useful alone:

- **#124** provides SIR-side provenance: prefixed filenames
  (`polaris_30m__sand_pct_mean.csv`) and manifest `source_dataset` field
- **This redesign** provides the consumer-side contract: the config declares
  `source: polaris_30m` for `soil_type`
- **Together** they create validated, end-to-end data lineage:
  pipeline dataset → SIR file → derived parameter

**#124 is a prerequisite for this redesign.** The prefixed filenames and
manifest `source_dataset` field are what make the `source:` contract
enforceable.

## Pydantic Schema

Explicit fields per category. Each pywatershed parameter is a named `Optional`
field on its category model. This makes the code self-documenting for domain
scientists — parameter names are visible as class attributes, not buried in
generic dicts.

```python
class ParameterEntry(BaseModel):
    source: str
    variable: str | list[str] | None = None
    statistic: str | None = None
    year: int | list[int] | None = None
    time_period: list[str] | None = None
    description: str

class TopographyDatasets(BaseModel):
    available: list[str] = []
    hru_elev: ParameterEntry | None = None
    hru_slope: ParameterEntry | None = None
    hru_aspect: ParameterEntry | None = None

class SoilsDatasets(BaseModel):
    available: list[str] = []
    soil_type: ParameterEntry | None = None
    sat_threshold: ParameterEntry | None = None
    soil_moist_max: ParameterEntry | None = None
    soil_rechr_max_frac: ParameterEntry | None = None

class LandcoverDatasets(BaseModel):
    available: list[str] = []
    cov_type: ParameterEntry | None = None
    hru_percent_imperv: ParameterEntry | None = None

class SnowDatasets(BaseModel):
    available: list[str] = []
    snarea_thresh: ParameterEntry | None = None

class WaterbodyDatasets(BaseModel):
    available: list[str] = []
    hru_type: ParameterEntry | None = None
    dprst_frac: ParameterEntry | None = None
    dprst_area_max: ParameterEntry | None = None

class ForcingConfig(BaseModel):
    available: list[str] = []
    prcp: ParameterEntry | None = None
    tmax: ParameterEntry | None = None
    tmin: ParameterEntry | None = None

class ClimateNormalsConfig(BaseModel):
    available: list[str] = []
    jh_coef: ParameterEntry | None = None
    transp_beg: ParameterEntry | None = None
    transp_end: ParameterEntry | None = None

class StaticDatasetsConfig(BaseModel):
    topography: TopographyDatasets = TopographyDatasets()
    soils: SoilsDatasets = SoilsDatasets()
    landcover: LandcoverDatasets = LandcoverDatasets()
    snow: SnowDatasets = SnowDatasets()
    waterbodies: WaterbodyDatasets = WaterbodyDatasets()
```

The top-level `PywatershedRunConfig` gains three new fields:
- `static_datasets: StaticDatasetsConfig`
- `forcing: ForcingConfig`
- `climate_normals: ClimateNormalsConfig`

`sir_path` is retained — it declares where to find the SIR files on disk.
The new sections declare what to expect in the SIR.

Version bumps to `"4.0"`.

## Example Config

```yaml
target_model: pywatershed
version: "4.0"

sir_path: "../output"

domain:
  fabric_path: "data/fabrics/nhru.gpkg"
  segment_path: "data/fabrics/nsegment.gpkg"
  waterbody_path: "data/fabrics/waterbodies.gpkg"
  id_field: "nhm_id"
  segment_id_field: "nhm_seg"

time:
  start: "1980-10-01"
  end: "2020-09-30"
  timestep: daily

static_datasets:

  topography:
    available: [dem_3dep_10m]
    hru_elev:
      source: dem_3dep_10m
      variable: elevation
      statistic: mean
      description: "Mean HRU elevation"
    hru_slope:
      source: dem_3dep_10m
      variable: slope
      statistic: mean
      description: "Mean land surface slope"
    hru_aspect:
      source: dem_3dep_10m
      variable: aspect
      statistic: mean
      description: "Mean HRU aspect"

  soils:
    available: [polaris_30m, gnatsgo_rasters]
    soil_type:
      source: polaris_30m
      variables: [sand, silt, clay]
      statistic: mean
      description: "Soil type classification (1=sand, 2=loam, 3=clay)"
    sat_threshold:
      source: polaris_30m
      variable: theta_s
      statistic: mean
      description: "Gravity reservoir storage capacity (from porosity)"
    soil_moist_max:
      source: gnatsgo_rasters
      variable: aws0_100
      statistic: mean
      description: "Maximum available water-holding capacity"
    soil_rechr_max_frac:
      source: gnatsgo_rasters
      variables: [rootznemc, rootznaws]
      statistic: mean
      description: "Recharge zone storage as fraction of soil_moist_max"

  landcover:
    available: [nlcd_osn_lndcov, nlcd_osn_fctimp, nlcd_legacy, nlcd_annual]
    cov_type:
      source: nlcd_osn_lndcov
      variable: LndCov
      statistic: categorical
      year: [2021]
      description: "Vegetation cover type (0=bare, 1=grasses, 2=shrubs, 3=trees, 4=coniferous)"
    hru_percent_imperv:
      source: nlcd_osn_fctimp
      variable: FctImp
      statistic: mean
      year: [2021]
      description: "Impervious surface fraction"

  snow:
    available: [snodas]
    snarea_thresh:
      source: snodas
      variable: SWE
      statistic: mean
      time_period: ["2020-01-01", "2021-12-31"]
      description: "Snow depletion threshold (calibration seed from historical max SWE)"

  waterbodies:
    available: []
    hru_type:
      source: domain.waterbody_path
      description: "HRU type (0=inactive, 1=land, 2=lake, 3=swale)"
    dprst_frac:
      source: domain.waterbody_path
      description: "Fraction of HRU with surface depressions"
    dprst_area_max:
      source: domain.waterbody_path
      description: "Maximum surface depression area"

forcing:
  available: [gridmet, snodas]
  prcp:
    source: gridmet
    variable: pr
    statistic: mean
    description: "Daily precipitation"
  tmax:
    source: gridmet
    variable: tmmx
    statistic: mean
    description: "Daily maximum temperature"
  tmin:
    source: gridmet
    variable: tmmn
    statistic: mean
    description: "Daily minimum temperature"

climate_normals:
  available: [gridmet, snodas]
  jh_coef:
    source: gridmet
    variables: [tmmx, tmmn]
    description: "Jensen-Haise PET coefficient (monthly, from tmax/tmin normals)"
  transp_beg:
    source: gridmet
    variable: tmmn
    description: "Month transpiration begins (from last spring frost)"
  transp_end:
    source: gridmet
    variable: tmmn
    description: "Month transpiration ends (from first fall killing frost)"

calibration:
  generate_seeds: true
  seed_method: physically_based
  preserve_from_existing: []

parameter_overrides:
  values: {}

output:
  path: "models/pywatershed"
  format: netcdf
  parameter_file: "parameters.nc"
  forcing_dir: "forcing"
  control_file: "control.yml"
  soltab_file: "soltab.nc"
```

## Scope and Dependencies

### Prerequisites
- Issue #124 (SIR dataset prefix) — provides the provenance mechanism that
  makes `source:` validation possible

### In scope
- New Pydantic schema (`pywatershed_config.py`)
- `hydro-param init` template generation with `available:` fields
- Derivation plugin startup validation (SIR ↔ config contract check)
- Dataset-specific routing in derivation plugin
- Example config update (`configs/examples/`)
- DRB e2e config migration

### Out of scope (future work)
- Adding new curated datasets to the registry
- New derivation logic for alternative datasets (e.g., gnatsgo-specific
  soil type derivation) — the routing infrastructure is built, but
  alternative code paths are added as datasets are onboarded
- Grid processing pathway

## Decisions Log

| Decision | Rationale |
|----------|-----------|
| Consumer-oriented keys (pywatershed param names) | Config reads as "what am I building?" not "what did I process?" |
| Three sections (static, forcing, normals) | Forcing and normals can come from different sources; maximum clarity |
| Only SIR-backed entries in config | Internal derivations (lookups, formulas) are plugin business, not user decisions |
| Explicit Pydantic fields per category | Self-documenting code for domain scientists; parameter names visible as class attributes |
| `available:` as active validated field | Discoverability in YAML + runtime safety net against registry drift |
| Validate and route at runtime | Early failure on missing data; `source` drives dataset-specific processing |
| #124 as prerequisite | Prefixed filenames are the mechanism that makes `source:` enforceable |
| Version 4.0 | Breaking change to schema; pre-alpha so no migration concern |
