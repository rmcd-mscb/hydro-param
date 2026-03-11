# Quick Start

A minimal end-to-end example using the Delaware River Basin.

This guide walks through a complete pipeline run: from project setup to
inspecting output files. By the end, you will have zonal statistics for
elevation, slope, and land cover computed over a polygon fabric.

## Prerequisites

- **pixi** installed ([installation guide](https://pixi.sh))
- **hydro-param** cloned and environment synced:

```bash
git clone git@github.com:rmcd-mscb/hydro-param.git
cd hydro-param
pixi install
```

## Step 1: Initialize a project

```bash
hydro-param init delaware-demo
cd delaware-demo
```

This creates a standard project directory with template configs,
data directories organized by dataset category (e.g., `data/topography/`,
`data/soils/`), and a `.gitignore`.

## Step 2: Obtain a target fabric

hydro-param does not fetch or subset fabrics. You need a pre-existing
GeoPackage or GeoParquet file containing your HRU polygons. Options:

- **pynhd / pygeohydro** -- `pygeohydro.get_camels()` or NHDPlus catchments
- **USGS Geospatial Fabric** -- download from ScienceBase
- **Custom** -- any polygon mesh with a unique integer ID column

Copy your fabric into the project:

```bash
cp /path/to/catchments.gpkg data/fabrics/
```

!!! note "Fabric requirements"
    The fabric must be a GeoPackage (`.gpkg`) or GeoParquet (`.parquet`)
    file with polygon geometries and a unique integer ID column (e.g.,
    `featureid`, `nhm_id`). hydro-param reads this file as-is and does
    not modify it.

## Step 3: Explore available datasets

List all registered datasets grouped by category:

```bash
hydro-param datasets list
```

Get details on a specific dataset, including its variables, access
strategy, and spatial extent:

```bash
hydro-param datasets info dem_3dep_10m
```

## Step 4: Write a pipeline config

Create `configs/pipeline.yml` with the following content. This minimal
config requests two datasets: 3DEP elevation and slope, plus NLCD land
cover fractions.

```yaml title="configs/pipeline.yml"
target_fabric:
  path: "data/fabrics/catchments.gpkg"
  id_field: "featureid"
  crs: "EPSG:5070"

domain:
  type: bbox
  bbox: [-76.5, 38.5, -74.0, 42.6]  # (1)!

datasets:
  topography:  # (2)!
    - name: dem_3dep_10m
      variables: [elevation, slope]
      statistics: [mean]

  land_cover:
    - name: nlcd_osn_lndcov
      variables: [LndCov]
      statistics: [categorical]  # (3)!
      year: [2021]

output:
  path: "./output"
  format: netcdf

processing:
  batch_size: 500
```

1. Bounding box as `[west, south, east, north]` in longitude/latitude.
   This covers the Delaware River Basin.
2. Datasets are grouped by category. Valid categories include
   `topography`, `soils`, `land_cover`, `snow`, and `climate`.
3. Use `categorical` for land cover to get per-class area fractions
   rather than a single summary statistic.

!!! tip "Config validation"
    hydro-param validates the config on load using Pydantic. Misspelled
    field names, unknown categories, or missing required fields produce
    clear error messages before any processing begins.

## Step 5: Run the pipeline

```bash
hydro-param run configs/pipeline.yml
```

The pipeline executes five stages:

1. **Resolve fabric** -- loads and spatially subsets the target fabric
2. **Resolve datasets** -- validates dataset names against the registry
3. **Compute weights** -- gdptools builds intersection weights per batch
4. **Process datasets** -- computes zonal statistics for each variable
5. **Format output** -- writes results to the output directory

Progress is logged to the console. A typical run with 500-feature batches
takes a few minutes depending on network speed (remote datasets) or disk
I/O (local data).

## Step 6: Inspect results

After the pipeline completes, the output directory contains one CSV per
variable, organized by category:

```
output/
  topography/
    dem_3dep_10m_elevation_mean.csv       # (1)!
    dem_3dep_10m_slope_mean.csv
  land_cover/
    nlcd_osn_lndcov_LndCov_categorical_2021.csv  # (2)!
  .manifest.yml                           # (3)!
```

1. Each CSV is indexed by the fabric ID field (`featureid` in this
   example) with one row per feature.
2. Categorical output contains one column per land cover class, with
   values representing the fractional area of each class within each
   feature.
3. The manifest tracks dataset fingerprints for resume support. Re-run
   the same config with `resume: true` under `processing:` to skip
   datasets whose inputs have not changed.

Load a result in Python:

```python
import pandas as pd

elevation = pd.read_csv("output/topography/dem_3dep_10m_elevation_mean.csv", index_col=0)
print(elevation.head())
```

Or with xarray for downstream analysis:

```python
import xarray as xr

ds = xr.Dataset.from_dataframe(elevation)
```

## Next steps

- [Configuration](../user-guide/configuration.md) -- full reference for
  all config fields, dataset options, and processing settings
- [Datasets](../user-guide/datasets.md) -- browse the dataset registry
  and learn about access strategies
- [pywatershed Workflow](../user-guide/pywatershed-workflow.md) -- use
  pipeline output to derive pywatershed model parameters
