# Quick Start

A minimal end-to-end example using the Delaware River Basin.

## 1. Initialize a project

```bash
hydro-param init delaware-demo
cd delaware-demo
```

## 2. Explore available datasets

```bash
hydro-param datasets list
```

## 3. Get details on a specific dataset

```bash
hydro-param datasets info dem_3dep_10m
```

## 4. Edit the pipeline config

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
  engine: exactextract
  batch_size: 500
```

## 5. Run the pipeline

```bash
hydro-param run configs/pipeline.yml
```

## 6. Inspect results

Output is written to `output/` as NetCDF or Parquet, depending on your config.
