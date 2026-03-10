# Quick Start

A minimal end-to-end example using the Delaware River Basin.

## 1. Initialize a project

```bash
hydro-param init delaware-demo
cd delaware-demo
```

## 2. Obtain a target fabric

hydro-param does not fetch or subset fabrics. You need a pre-existing
GeoPackage or GeoParquet file containing your HRU polygons. Options:

- **pynhd/pygeohydro**: `pygeohydro.get_camels()` or NHDPlus catchments
- **USGS Geospatial Fabric**: Download from ScienceBase
- **Custom**: Any polygon mesh with a unique ID column

Copy your fabric to `data/fabrics/` inside the project.

## 3. Explore available datasets

```bash
hydro-param datasets list
```

## 4. Get details on a specific dataset

```bash
hydro-param datasets info dem_3dep_10m
```

## 5. Edit the pipeline config

Edit `configs/pipeline.yml` to specify your target fabric, domain, and
datasets. See [Configuration](../user-guide/configuration.md) for details.

```yaml
target_fabric:
  path: "data/fabrics/catchments.gpkg"
  id_field: "featureid"
  crs: "EPSG:4326"

domain:
  type: bbox
  bbox: [-76.5, 38.5, -74.0, 42.6]

datasets:
  - name: dem_3dep_10m
    variables: [elevation, slope, aspect]
    statistics: [mean]

output:
  path: "./output"
  format: netcdf
  sir_name: "delaware_terrain"

processing:
  batch_size: 500
```

## 6. Run the pipeline

```bash
hydro-param run configs/pipeline.yml
```

## 7. Inspect results

The pipeline writes files to `output/`:

- `output/<category>/<variable>.csv` — raw zonal statistics per variable
- `output/sir/<variable>.csv` — normalized SIR (standardized names/units)
- `output/.hydro_param_manifest.json` — resume manifest

For temporal datasets, output is `output/<category>/<dataset>_<year>_temporal.nc`.
