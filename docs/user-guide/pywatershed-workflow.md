# pywatershed Workflow

hydro-param's primary target model is
[pywatershed](https://github.com/EC-USGS/pywatershed) --- the USGS
National Hydrologic Model (NHM-PRMS) implemented in Python. This page
walks through the end-to-end workflow for generating a complete set of
pywatershed model parameters from geospatial source data.

---

## Two-Phase Architecture

Generating pywatershed parameters is a two-phase process:

```
Phase 1 (Generic Pipeline)          Phase 2 (pywatershed Derivation)
┌───────────────────────────┐       ┌──────────────────────────────────┐
│ hydro-param run            │       │ hydro-param pywatershed run      │
│                           │       │                                  │
│ Fetch datasets            │       │ Read SIR from disk               │
│ Compute zonal statistics  │──SIR──│ Derive ~100 PRMS parameters      │
│ Write CSV/NetCDF per var  │       │ Write parameters.nc, forcing/,   │
│                           │       │       soltab.nc, control.yml     │
│ Model-agnostic            │       │ pywatershed-specific             │
└───────────────────────────┘       └──────────────────────────────────┘
```

**Phase 1** runs the generic five-stage pipeline. It fetches geospatial
datasets (3DEP, NLCD, POLARIS, gridMET, etc.), computes zonal statistics
against your target fabric, and writes the results as a **Standardized
Internal Representation (SIR)** --- one file per variable, organized by
dataset category. Phase 1 knows nothing about pywatershed or any other
model.

**Phase 2** reads the SIR output and derives all parameters that
pywatershed needs. It performs unit conversions (metric to PRMS internal
units: feet, inches, degrees F), variable reclassification (NLCD codes
to PRMS cover types), lookup-table joins, solar geometry computations,
climate normal derivations, and gap-filling. The result is a set of
model-ready files that pywatershed can load directly.

!!! info "Why two phases?"
    Separating data access from model derivation means you can re-run
    Phase 2 with different pywatershed settings without re-fetching
    datasets. It also means the same SIR output could drive other models
    in the future.

---

## Prerequisites

Before starting, you need:

**Required:**

- **HRU fabric** --- a polygon GeoPackage or GeoParquet file defining
  your Hydrologic Response Units (HRUs). Obtain this from the
  [USGS Geospatial Fabric](https://www.sciencebase.gov/catalog/item/5e29d1a0e4b0a79317cf7f63)
  or generate it with [pynhd](https://github.com/hyriver/pynhd).

- **Segment fabric** --- a line GeoPackage defining stream segments
  (needed for routing parameters in Phase 2). Typically the NHDPlus
  flowline geometry for your domain.

**Optional:**

- **Local GFv1.1 rasters** --- pre-downloaded Geospatial Fabric v1.1
  rasters for land cover parameters. Download with
  `hydro-param gfv11 download --output-dir /path/to/data/gfv11`.

- **Waterbody polygons** --- NHDPlus waterbody geometries for depression
  storage and HRU type classification.

!!! warning "hydro-param does not fetch fabrics"
    hydro-param expects pre-existing geospatial files as input. Use
    [pynhd](https://github.com/hyriver/pynhd) or
    [pygeohydro](https://github.com/hyriver/pygeohydro) to obtain
    and subset fabric files for your study area before running
    hydro-param.

---

## Step 1: Run the Generic Pipeline (Phase 1)

Phase 1 is driven by a **pipeline config** file that declares your
target fabric, datasets, variables, and output settings.

### Pipeline config overview

```yaml title="configs/examples/drb_2yr_pipeline.yml (abbreviated)"
target_fabric:
  path: data/pywatershed_gis/drb_2yr/nhru.gpkg
  id_field: nhm_id

datasets:
  topography:
    - name: dem_3dep_10m
      variables: [elevation, slope, sin_aspect, cos_aspect]
      statistics: [mean]

  soils:
    - name: gnatsgo_rasters
      variables: [aws0_100, rootznemc, rootznaws]
      statistics: [mean]
    - name: polaris_30m
      variables: [sand, silt, clay, theta_s, ksat, soil_texture]
      statistics: [mean]

  land_cover:
    - name: nlcd_osn_lndcov
      variables: [LndCov]
      statistics: [categorical]
      year: [2020, 2021]
    - name: nlcd_osn_fctimp
      variables: [FctImp]
      statistics: [mean]
      year: [2020, 2021]

  climate:
    - name: gridmet
      variables: [pr, tmmx, tmmn, srad, pet, vs]
      statistics: [mean]
      time_period: ["2020-01-01", "2021-12-31"]

output:
  path: output
  format: netcdf
  sir_name: drb_2yr_sir

processing:
  batch_size: 240
  resume: true
```

Key points:

- **`target_fabric`** points to your HRU polygon file and its unique
  identifier field.
- **`datasets`** lists what to fetch, organized by category. Each entry
  names a registered dataset, the variables to extract, and the
  statistics to compute.
- **`statistics: [categorical]`** computes class fractions (for NLCD
  land cover codes); `[mean]` computes continuous zonal means.
- **`time_period`** triggers temporal processing for climate datasets.
- **`batch_size`** controls how many HRUs are processed at once
  (spatial batching keeps memory bounded).
- **`resume: true`** allows restarting interrupted runs.

See [Configuration](configuration.md) for the full config reference.

### Run it

```console
$ hydro-param run configs/examples/drb_2yr_pipeline.yml
```

### What it produces

Phase 1 writes the SIR to the output directory:

```
output/
  .manifest.yml                                        # SIR manifest (resume + Phase 2 lookup)
  topography/
    dem_3dep_10m_elevation_mean.csv                    # Mean elevation per HRU
    dem_3dep_10m_slope_mean.csv                        # Mean slope per HRU
    dem_3dep_10m_sin_aspect_mean.csv                   # Sin of aspect (for circular mean)
    dem_3dep_10m_cos_aspect_mean.csv                   # Cos of aspect (for circular mean)
  soils/
    gnatsgo_rasters_aws0_100_mean.csv                  # Available water storage 0-100cm
    polaris_30m_sand_mean.csv                          # Sand fraction per HRU
    ...
  land_cover/
    nlcd_osn_lndcov_LndCov_categorical_2021.csv        # NLCD class fractions per HRU
    nlcd_osn_fctimp_FctImp_mean_2021.csv               # Fractional impervious per HRU
  climate/
    gridmet_pr_mm_mean_2020.nc                         # Daily precipitation (temporal)
    gridmet_tmmx_C_mean_2020.nc                        # Daily max temperature (temporal)
    ...
```

Each file contains one variable with the HRU identifier as the index
or coordinate dimension. Temporal datasets produce one NetCDF per
variable per year. Static datasets produce CSVs.

!!! tip "Check the manifest"
    The `.manifest.yml` file in the output directory lists every SIR
    file with its dataset, variable, statistic, and file path. Phase 2
    uses this manifest to locate SIR data on disk.

---

## Step 2: Derive pywatershed Parameters (Phase 2)

Phase 2 reads the SIR and applies pywatershed-specific derivations to
produce model-ready parameter files.

### pywatershed run config overview

```yaml title="configs/examples/drb_2yr_pywatershed.yml (abbreviated)"
target_model: pywatershed
version: "4.0"

sir_path: "../../output"

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
    hru_slope:
      source: dem_3dep_10m
      variable: slope
      statistic: mean
  soils:
    available: [polaris_30m, gnatsgo_rasters]
    soil_type:
      source: polaris_30m
      variables: [sand, silt, clay]
      statistic: mean
    soil_moist_max:
      source: gnatsgo_rasters
      variable: aws0_100
      statistic: mean
    # ... more soil parameters

  landcover:
    available: [nlcd_osn_lndcov, nlcd_osn_fctimp]
    cov_type:
      source: nlcd_osn_lndcov
      variable: LndCov
      statistic: categorical
      year: [2021]
    # ... more landcover parameters

forcing:
  available: [gridmet]
  prcp:
    source: gridmet
    variable: pr
    statistic: mean
  tmax:
    source: gridmet
    variable: tmmx
    statistic: mean
  tmin:
    source: gridmet
    variable: tmmn
    statistic: mean

climate_normals:
  available: [gridmet]
  jh_coef:
    source: gridmet
    variables: [tmmx, tmmn]
  transp_beg:
    source: gridmet
    variable: tmmn
  transp_end:
    source: gridmet
    variable: tmmn

calibration:
  generate_seeds: true
  seed_method: physically_based

output:
  path: models/pywatershed
  format: netcdf
  parameter_file: parameters.nc
  forcing_dir: forcing
  control_file: control.yml
  soltab_file: soltab.nc
```

Key points:

- **`sir_path`** points to the Phase 1 output directory (relative
  paths are resolved from the config file's location).
- **`domain`** provides both HRU and segment fabrics --- segments are
  needed for routing parameter derivation (Step 12).
- **`static_datasets`** maps PRMS parameter names to SIR variables.
  Each entry says which dataset, variable, and statistic to read from
  the SIR.
- **`forcing`** maps PRMS forcing variables (prcp, tmax, tmin) to
  their SIR temporal data sources.
- **`climate_normals`** identifies which climate variables to use for
  deriving PET coefficients and transpiration timing.
- **`calibration.generate_seeds: true`** computes physically-based
  initial values for calibration parameters.

See [Configuration](configuration.md#pywatershed-run-config-phase-2)
for the full config reference.

### Run it

```console
$ hydro-param pywatershed run configs/examples/drb_2yr_pywatershed.yml
```

### What it produces

```
models/pywatershed/
  parameters.nc      # Static PRMS parameters (CF-1.8 NetCDF)
  soltab.nc          # Potential solar radiation tables (nhru x 366)
  control.yml        # Simulation time period and file paths
  forcing/
    prcp.nc          # Daily precipitation (inches/day)
    tmax.nc          # Daily maximum temperature (degrees F)
    tmin.nc          # Daily minimum temperature (degrees F)
```

---

## What the Derivation Steps Produce

Phase 2 executes 14 ordered derivation steps. Each step may depend on
parameters computed in earlier steps (the steps form a directed acyclic
graph).

| Step | Category | Key Parameters | Source |
|------|----------|---------------|--------|
| 1 | Geometry | `hru_area`, `hru_lat`, `hru_lon` | Computed from fabric polygon geometry |
| 2 | Topology | `hru_segment`, `tosegment_nhm`, `hru_up_id` | Spatial join of fabric to NHDPlus segments |
| 3 | Topography | `hru_elev`, `hru_slope`, `hru_aspect` | 3DEP DEM zonal statistics |
| 4 | Land cover | `cov_type`, `covden_sum`, `covden_win`, `srain_intcp`, `wrain_intcp`, `snow_intcp` | NLCD reclassification + PRMS lookup tables |
| 5 | Soils | `soil_type`, `soil_moist_max`, `soil_rechr_max`, `soil_rechr_max_frac` | POLARIS/gNATSGO zonal stats + USDA texture triangle |
| 6 | Waterbodies | `hru_type`, `dprst_frac` | NHDPlus waterbody overlay |
| 7 | Forcing | `prcp`, `tmax`, `tmin` (daily) | gridMET temporal data, converted to PRMS units |
| 8 | Lookup tables | `hru_deplcrv`, `snarea_curve`, `rad_trncf` | PRMS tables keyed by `cov_type` / forest type |
| 9 | Solar tables | `soltab_potsw`, `soltab_horad` (nhru x 366) | Computed from latitude, slope, and aspect |
| 10 | PET coefficients | `jh_coef` (monthly) | Jensen-Haise formula from tmax/tmin normals |
| 11 | Transpiration | `transp_beg`, `transp_end` | Monthly mean tmin threshold analysis |
| 12 | Routing | `K_coef`, `x_coef`, `seg_cum_area`, `seg_slope` | Muskingum coefficients from segment geometry |
| 13 | Defaults | `tmax_allsnow`, `dday_slope`, `cecn_coef`, etc. | PRMS default values |
| 14 | Calibration seeds | `snarea_thresh`, `fastcoef_lin`, etc. | Physically-based initial values |

!!! note "Step ordering matters"
    Steps must run in order because later steps depend on earlier
    results. For example, Step 8 (lookup tables) needs `cov_type` from
    Step 4, and Step 9 (solar tables) needs `hru_lat`, `hru_slope`, and
    `hru_aspect` from Steps 1 and 3.

### Unit conversions

PRMS uses non-SI internal units. Phase 2 handles all conversions:

| Quantity | SIR Units | PRMS Units | Conversion |
|----------|-----------|------------|------------|
| Elevation | meters | feet | multiply by 3.28084 |
| Area | square meters | acres | divide by 4046.86 |
| Precipitation | mm/day | inches/day | divide by 25.4 |
| Temperature | degrees C | degrees F | T_F = T_C x 9/5 + 32 |
| Soil moisture | mm | inches | divide by 25.4 |

---

## Step 3: Validate Output

After Phase 2 completes, validate the parameter file to confirm all
required parameters are present and within acceptable ranges:

```console
$ hydro-param pywatershed validate models/pywatershed/parameters.nc
```

The validator checks:

- All required PRMS parameters are present in the file
- Values fall within the valid ranges defined in the bundled parameter
  metadata
- Dimension sizes are consistent (nhru, nsegment, nmonths, etc.)

A clean run prints a success message. Any issues are reported with the
parameter name, the expected range, and the actual min/max values found.

!!! tip "Run validation after every change"
    If you modify the pywatershed run config --- for example, changing
    `parameter_overrides` or switching data sources --- re-run validation
    to catch any out-of-range values early.

---

## Output File Reference

| File | Format | Dimensions | Description |
|------|--------|-----------|-------------|
| `parameters.nc` | CF-1.8 NetCDF | `nhru`, `nsegment`, `nmonths`, `ndeplval` | Static PRMS parameters. Load with `pws.Parameters.from_netcdf()`. |
| `forcing/prcp.nc` | NetCDF | `nhru` x `time` | Daily precipitation in inches/day. |
| `forcing/tmax.nc` | NetCDF | `nhru` x `time` | Daily maximum temperature in degrees F. |
| `forcing/tmin.nc` | NetCDF | `nhru` x `time` | Daily minimum temperature in degrees F. |
| `soltab.nc` | NetCDF | `nhru` x 366 | Potential shortwave radiation and horizontal radiation by Julian day. |
| `control.yml` | YAML | --- | Simulation start/end dates, timestep, and paths to parameter/forcing files. |

---

## Running pywatershed

Once you have the output files, you can run a pywatershed simulation:

```python
import pywatershed as pws

control = pws.Control.load("models/pywatershed/control.yml")
params = pws.Parameters.from_netcdf("models/pywatershed/parameters.nc")

model = pws.Model(control=control, parameters=params)
model.run()
```

Refer to the
[pywatershed documentation](https://ec-usgs.github.io/pywatershed/)
for details on model configuration, process selection, and output
analysis.

---

## Next Steps

- **[Configuration](configuration.md)** --- Full reference for both
  pipeline and pywatershed run configs, including all supported options.
- **[Datasets](datasets.md)** --- Browse available datasets, check
  registration details, and download local data.
- **[CLI Reference](cli.md)** --- Complete command reference for
  `hydro-param run`, `hydro-param pywatershed run`, and other commands.
