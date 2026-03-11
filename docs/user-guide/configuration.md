# Configuration

hydro-param uses two YAML configuration files that correspond to its
two-phase architecture:

1. **Pipeline config (Phase 1)** --- Declares what datasets to fetch and what
   zonal statistics to compute. Produces a Standardized Internal
   Representation (SIR) on disk.
2. **pywatershed run config (Phase 2)** --- Maps SIR output to model-specific
   parameters, forcing files, and control settings for pywatershed.

Both configs are **declarative only**: they say *what* to compute, not *how*.
No variables, conditionals, or templating. All logic lives in Python.

!!! info "Config validation"
    Both configs are validated at load time by [Pydantic v2](https://docs.pydantic.dev/).
    Invalid configs fail fast with clear error messages before any data is
    fetched or parameters are derived.

---

## Pipeline config (Phase 1)

The pipeline config drives the `hydro-param run` command. It has five
top-level sections:

| Section | Required | Purpose |
|---------|----------|---------|
| `target_fabric` | Yes | Polygon mesh to parameterize |
| `datasets` | Yes | Datasets, variables, and statistics to compute |
| `output` | No | Output directory, format, and name |
| `processing` | No | Batch size, resume, timeout |
| `domain` | No | Spatial subsetting (bbox, HUC, gage) |

### `target_fabric`

The spatial mesh whose features receive zonal statistics. Must be a
pre-existing geospatial file --- hydro-param does not fetch or subset
fabrics (use [pynhd](https://github.com/hyriver/pynhd) upstream).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | *required* | Path to fabric file (GeoPackage or GeoParquet) |
| `id_field` | string | *required* | Column with unique feature IDs. Becomes the xarray dimension name in all output. |
| `crs` | string | `"EPSG:4326"` | Coordinate reference system as an EPSG string |

!!! tip "Choosing `id_field`"
    The `id_field` propagates through the entire pipeline: it controls the
    xarray dimension name in the SIR, the CSV index column, and feature
    matching in the pywatershed derivation plugin. Common values:

    - `nhm_id` --- pywatershed / National Hydrologic Model
    - `featureid` --- NHDPlus
    - `hru_id` --- custom fabrics

### `datasets`

Datasets are organized by category. Each category key must be one of the
valid registry categories:

`climate`, `geology`, `hydrography`, `land_cover`, `snow`, `soils`,
`topography`, `water_bodies`

Each category contains a list of dataset requests:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | *required* | Dataset name from the registry (e.g., `dem_3dep_10m`) |
| `source` | string | `null` | Local file path override for `local_tiff` datasets |
| `variables` | list | `[]` | Variable names to extract |
| `statistics` | list | `["mean"]` | Zonal statistics: `mean`, `categorical`, `majority`, `sum`, `min`, `max`, `median` |
| `year` | int or list | `null` | Year(s) for multi-year static datasets (e.g., NLCD). Range: 1900--2100. |
| `time_period` | list | `null` | `[start, end]` ISO dates for temporal datasets |

!!! note "Static vs. temporal datasets"
    - **Static datasets** (topography, soils, land cover) use `year` for
      multi-epoch data. The pipeline produces year-suffixed output keys
      (e.g., `lndcov_frac_2021`).
    - **Temporal datasets** (climate, snow) use `time_period` with ISO date
      strings. The pipeline splits processing by calendar year automatically.

### `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `"./output"` | Output directory (created automatically) |
| `format` | string | `"netcdf"` | Format for temporal output: `netcdf` or `parquet` |
| `sir_name` | string | `"result"` | Name used in CF-1.8 metadata and log messages |

### `processing`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | `500` | Max features per spatial batch. KD-tree bisection groups nearby features. |
| `resume` | bool | `false` | Skip datasets whose outputs already exist (checked via manifest) |
| `sir_validation` | string | `"tolerant"` | `"tolerant"` logs warnings; `"strict"` raises on any issue |
| `network_timeout` | int | `120` | GDAL HTTP timeout in seconds for COG/vsicurl access |

### `domain` (optional)

When present, clips the target fabric to the specified extent before
processing. When omitted, the full fabric extent is used.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | *required* | `bbox`, `huc2`, `huc4`, or `gage` |
| `bbox` | list | `null` | `[west, south, east, north]` in EPSG:4326. Required when `type: bbox`. |
| `id` | string | `null` | HUC code or gage ID. Required when type is `huc2`, `huc4`, or `gage`. |

### Complete example

This is a production pipeline config for the Delaware River Basin, fetching
topography, soils, land cover, snow, and climate data over a 2-year period:

```yaml title="configs/examples/drb_2yr_pipeline.yml"
# DRB 2-year pipeline config: pywatershed data requirements
# Produces a SIR with topographic, soils, land cover, snow, and climate data.
#
# Usage:
#   hydro-param run configs/examples/drb_2yr_pipeline.yml

target_fabric:
  path: data/pywatershed_gis/drb_2yr/nhru.gpkg  # (1)!
  id_field: nhm_id  # (2)!

datasets:
  topography:
    # 3DEP 10m DEM — elevation, slope, sin/cos aspect for circular mean
    - name: dem_3dep_10m
      variables: [elevation, slope, sin_aspect, cos_aspect]
      statistics: [mean]

  soils:
    # gNATSGO pre-summarized soil properties (Planetary Computer)
    - name: gnatsgo_rasters
      variables: [aws0_100, rootznemc, rootznaws]
      statistics: [mean]

    # POLARIS soil texture properties, 30m (remote VRT, no download needed)
    - name: polaris_30m
      variables: [sand, silt, clay, theta_s, ksat, soil_texture]
      statistics: [mean]

  land_cover:
    # NLCD Land Cover via NHGF STAC OSN — categorical fractions (multi-year)
    - name: nlcd_osn_lndcov
      variables: [LndCov]
      statistics: [categorical]  # (3)!
      year: [2020, 2021]  # (4)!

    # NLCD Fractional Impervious via NHGF STAC OSN
    - name: nlcd_osn_fctimp
      variables: [FctImp]
      statistics: [mean]
      year: [2020, 2021]

  snow:
    # SNODAS daily snow — historical SWE for snarea_thresh calibration seed
    - name: snodas
      variables: [SWE]
      statistics: [mean]
      time_period: ["2020-01-01", "2021-12-31"]  # (5)!

  climate:
    # gridMET daily climate (OPeNDAP via ClimateR catalog)
    - name: gridmet
      variables: [pr, tmmx, tmmn, srad, pet, vs]
      statistics: [mean]
      time_period: ["2020-01-01", "2021-12-31"]

output:
  path: output
  format: netcdf
  sir_name: drb_2yr_sir

processing:
  batch_size: 240  # (6)!
  resume: true  # (7)!
```

1. Path to the HRU fabric GeoPackage. Must exist before running.
2. The `nhm_id` column becomes the xarray dimension in all SIR output files.
3. `categorical` produces per-class fraction columns (e.g., `lndcov_frac_11`,
   `lndcov_frac_21`, ...) instead of a single summary statistic.
4. Multi-year requests produce year-suffixed output (e.g., `lndcov_frac_2021`).
   The most recent year is used during Phase 2 derivation.
5. Temporal datasets require `time_period` with ISO date strings. The pipeline
   splits processing by calendar year automatically.
6. Smaller batch sizes reduce peak memory. 240 works well for DRB's 765 HRUs.
7. `resume: true` skips datasets whose manifest fingerprints are current,
   allowing interrupted runs to continue where they left off.

---

## pywatershed run config (Phase 2)

The pywatershed run config drives the `hydro-param pywatershed run` command.
It consumes SIR output from Phase 1 and produces model-ready parameter files,
forcing time series, solar tables, and a control file.

This config creates a **consumer-oriented contract** between the generic
pipeline and the model-specific derivation plugin. Each parameter entry
declares exactly which SIR dataset and variable provides its source data.

| Section | Required | Purpose |
|---------|----------|---------|
| `target_model` | No | Fixed to `"pywatershed"` |
| `version` | No | Schema version (currently `"4.0"`) |
| `sir_path` | No | Path to Phase 1 output directory |
| `domain` | Yes | Fabric and segment file paths |
| `time` | Yes | Simulation time period |
| `static_datasets` | No | Maps SIR data to static parameters |
| `forcing` | No | Maps SIR data to forcing time series |
| `climate_normals` | No | Maps SIR data to climate-derived parameters |
| `parameter_overrides` | No | Manual parameter value overrides |
| `calibration` | No | Calibration seed generation options |
| `output` | No | Output directory and file layout |

!!! warning "Extra fields are rejected"
    All models in the pywatershed config use `extra="forbid"`. Typos or
    unknown fields produce immediate validation errors rather than being
    silently ignored.

### `sir_path`

Path to the Phase 1 pipeline output directory containing the `.manifest.yml`
and SIR data files. Relative paths are resolved against the config file's
parent directory.

### `domain`

Points to pre-existing fabric and segment files. hydro-param does not fetch
or subset fabrics.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fabric_path` | string | *required* | Path to HRU fabric (GeoPackage or GeoParquet) |
| `segment_path` | string | `null` | Path to segment/flowline file for routing topology |
| `waterbody_path` | string | `null` | Path to NHDPlus waterbody polygons for depression storage overlay |
| `id_field` | string | `"nhm_id"` | Feature ID column in the fabric |
| `segment_id_field` | string | `"nhm_seg"` | Segment ID column in the segment file |

### `time`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `start` | string | *required* | Start date in ISO format (e.g., `"2020-01-01"`) |
| `end` | string | *required* | End date in ISO format (e.g., `"2021-12-31"`) |
| `timestep` | string | `"daily"` | Temporal resolution. Only `daily` is supported. |

### `static_datasets`

Groups static parameter declarations by category. Each parameter entry maps
a pywatershed parameter name to the SIR dataset that provides its source data.

The five categories are: `topography`, `soils`, `landcover`, `snow`, and
`waterbodies`. Each category has an `available` list declaring which registry
datasets were used, plus named parameter fields.

**Parameter entry fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `source` | string | *required* | Registry dataset name (e.g., `dem_3dep_10m`) |
| `variable` | string | `null` | Single source variable name |
| `variables` | list | `null` | Multiple source variables (use one of `variable` or `variables`, not both) |
| `statistic` | string | `null` | Zonal statistic applied (`mean`, `categorical`) |
| `year` | int or list | `null` | NLCD year(s) for multi-epoch land cover |
| `time_period` | list | `null` | Temporal range for temporal datasets |
| `description` | string | *required* | Human-readable description |

??? info "Available parameters by category"

    **Topography:** `hru_elev`, `hru_slope`, `hru_aspect`

    **Soils:** `soil_type`, `sat_threshold`, `soil_moist_max`, `soil_rechr_max_frac`

    **Landcover:** `cov_type`, `hru_percent_imperv`, `covden_sum`, `covden_win`,
    `srain_intcp`, `wrain_intcp`, `snow_intcp`

    **Snow:** `hru_deplcrv`, `snarea_thresh`

    **Waterbodies:** `hru_type`, `dprst_frac`

### `forcing`

Temporal forcing time series. The derivation plugin converts forcing from SIR
units (metric: mm, degC) to PRMS units (inches, degF) during output formatting.

| Field | Type | Description |
|-------|------|-------------|
| `available` | list | Temporal-capable datasets in the registry |
| `prcp` | ParameterEntry | Daily precipitation |
| `tmax` | ParameterEntry | Daily maximum temperature |
| `tmin` | ParameterEntry | Daily minimum temperature |

### `climate_normals`

Long-term climate statistics for derived parameters. Can use the same source
as forcing or a different one (e.g., forcing from CONUS404-BA but normals
from gridMET).

| Field | Type | Description |
|-------|------|-------------|
| `available` | list | Temporal-capable datasets in the registry |
| `jh_coef` | ParameterEntry | Jensen-Haise PET coefficient (monthly, from tmax/tmin normals) |
| `transp_beg` | ParameterEntry | Month transpiration begins (from tmin threshold) |
| `transp_end` | ParameterEntry | Month transpiration ends (from tmin threshold) |

### `calibration`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `generate_seeds` | bool | `true` | Whether to generate calibration seed values |
| `seed_method` | string | `"physically_based"` | `"physically_based"` or `"all_defaults"` |
| `preserve_from_existing` | list | `[]` | Parameter names to keep from an existing file |

### `parameter_overrides`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `values` | dict | `{}` | Parameter name to scalar or per-HRU value. Scalars broadcast to all HRUs. |
| `from_file` | string | `null` | Path to NetCDF/CSV with override values (not yet implemented) |

### `output`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `path` | string | `"./output"` | Root output directory |
| `format` | string | `"netcdf"` | `"netcdf"` (CF-1.8) or `"prms_text"` (not yet implemented) |
| `parameter_file` | string | `"parameters.nc"` | Static parameter filename |
| `forcing_dir` | string | `"forcing"` | Subdirectory for forcing files |
| `control_file` | string | `"control.yml"` | Simulation control filename |
| `soltab_file` | string | `"soltab.nc"` | Solar radiation table filename |

### Complete example

```yaml title="configs/examples/drb_2yr_pywatershed.yml"
# DRB 2-year pywatershed model setup config (v4.0)
# Generates parameters.nc, forcing/, and control.yml for pywatershed.
#
# Usage:
#   hydro-param pywatershed run configs/examples/drb_2yr_pywatershed.yml

target_model: pywatershed
version: "4.0"

sir_path: "../../output"  # (1)!

domain:
  fabric_path: data/pywatershed_gis/drb_2yr/nhru.gpkg
  segment_path: data/pywatershed_gis/drb_2yr/nsegment.gpkg
  id_field: nhm_id
  segment_id_field: nhm_seg

time:
  start: "2020-01-01"
  end: "2021-12-31"
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

  soils:
    available: [polaris_30m, gnatsgo_rasters]
    soil_type:  # (2)!
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
    available: [nlcd_osn_lndcov, nlcd_osn_fctimp]
    cov_type:  # (3)!
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
    available: []  # (4)!

forcing:
  available: [gridmet]
  prcp:  # (5)!
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

climate_normals:  # (6)!
  available: [gridmet]
  jh_coef:
    source: gridmet
    variables: [tmmx, tmmn]
    description: "Jensen-Haise PET coefficient (monthly, from tmax/tmin normals)"
  transp_beg:
    source: gridmet
    variable: tmmn
    description: "Month transpiration begins (from monthly mean tmin threshold)"
  transp_end:
    source: gridmet
    variable: tmmn
    description: "Month transpiration ends (from monthly mean tmin threshold)"

calibration:
  generate_seeds: true
  seed_method: physically_based
  preserve_from_existing: []

parameter_overrides:
  values: {}

output:
  path: models/pywatershed
  format: netcdf
  parameter_file: parameters.nc
  forcing_dir: forcing
  control_file: control.yml
  soltab_file: soltab.nc
```

1. Relative to the config file's parent directory. Points to the Phase 1
   pipeline output containing `.manifest.yml` and SIR data files.
2. `soil_type` uses `variables` (plural) because classification requires
   sand, silt, and clay percentages together.
3. `cov_type` uses `statistic: categorical` because vegetation type is derived
   from NLCD class fractions via grouped majority, not a continuous mean.
4. No waterbody dataset is used here. Depression storage and HRU type default
   to zero when `waterbody_path` is not provided.
5. Forcing entries map SIR variable names (metric: `pr` in mm, `tmmx`/`tmmn`
   in degC) to PRMS parameters (imperial: inches, degF). Unit conversion
   happens automatically during derivation.
6. `climate_normals` can use a different source than `forcing`. For example,
   forcing from CONUS404-BA but normals from gridMET for better long-term
   temperature statistics.

---

## Dataset registry

hydro-param ships a bundled dataset registry containing metadata for all
supported datasets. The registry defines fetch strategies, STAC endpoints,
variable names, CRS, and resolution for each dataset.

### Bundled datasets

The bundled registry is organized by category in
`src/hydro_param/data/datasets/`:

```
datasets/
  climate.yml         # gridMET, Daymet, PRISM, CONUS404-BA
  geology.yml         # (reserved)
  hydrography.yml     # GFv1.1 rasters
  land_cover.yml      # NLCD (6 OSN collections)
  snow.yml            # SNODAS
  soils.yml           # gNATSGO, POLARIS
  topography.yml      # 3DEP DEM
  water_bodies.yml    # (reserved)
```

List all available datasets with:

```console
$ hydro-param datasets list
```

Get detailed info for a specific dataset:

```console
$ hydro-param datasets info dem_3dep_10m
```

### User-local overlay

You can extend the registry without modifying the package by placing YAML
files in `~/.hydro-param/datasets/`. Files in this directory are merged on
top of the bundled registry at load time, allowing you to:

- Add custom local datasets
- Override fetch strategies for existing datasets
- Register new STAC collections

The overlay follows the same schema as the bundled registry files. For
example, the `hydro-param datasets download gfv11` command automatically
registers GFv1.1 rasters via a user-local overlay file.

!!! tip "Overlay precedence"
    User-local entries override bundled entries with the same dataset name.
    This lets you point a dataset at a local file path instead of a remote
    service for faster processing or offline use.

---

## Config reference

For complete field-level documentation, see the API reference:

**Pipeline config (Phase 1):**

- [`PipelineConfig`](../api/config.md#hydro_param.config.PipelineConfig) --- Top-level pipeline configuration
- [`TargetFabricConfig`](../api/config.md#hydro_param.config.TargetFabricConfig) --- Target polygon mesh
- [`DomainConfig`](../api/config.md#hydro_param.config.DomainConfig) --- Spatial extent
- [`DatasetRequest`](../api/config.md#hydro_param.config.DatasetRequest) --- Dataset selection
- [`OutputConfig`](../api/config.md#hydro_param.config.OutputConfig) --- Output format and location
- [`ProcessingConfig`](../api/config.md#hydro_param.config.ProcessingConfig) --- Batching and fault tolerance

**pywatershed run config (Phase 2):**

- [`PywatershedRunConfig`](../api/pywatershed-config.md#hydro_param.pywatershed_config.PywatershedRunConfig) --- Top-level pywatershed configuration
- [`PwsDomainConfig`](../api/pywatershed-config.md#hydro_param.pywatershed_config.PwsDomainConfig) --- Domain fabric paths
- [`PwsTimeConfig`](../api/pywatershed-config.md#hydro_param.pywatershed_config.PwsTimeConfig) --- Simulation time period
- [`StaticDatasetsConfig`](../api/pywatershed-config.md#hydro_param.pywatershed_config.StaticDatasetsConfig) --- Static parameter declarations
- [`ForcingConfig`](../api/pywatershed-config.md#hydro_param.pywatershed_config.ForcingConfig) --- Forcing time series
- [`ClimateNormalsConfig`](../api/pywatershed-config.md#hydro_param.pywatershed_config.ClimateNormalsConfig) --- Climate-derived parameters
- [`ParameterEntry`](../api/pywatershed-config.md#hydro_param.pywatershed_config.ParameterEntry) --- Individual parameter declaration
- [`PwsOutputConfig`](../api/pywatershed-config.md#hydro_param.pywatershed_config.PwsOutputConfig) --- Output file layout
