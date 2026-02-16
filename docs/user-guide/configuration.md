# Configuration

hydro-param uses YAML configuration files to define pipelines declaratively.

## Pipeline Config Structure

A pipeline config (`configs/pipeline.yml`) has five sections:

- **target_fabric** --- The polygon mesh to parameterize
- **domain** --- Spatial extent for the analysis
- **datasets** --- Which datasets and variables to process
- **output** --- Output format and location
- **processing** --- Engine and batching options

## Example

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
  - name: nlcd_annual
    source: "data/land_cover/nlcd_annual_2021_land_cover.tif"
    variables: [land_cover]
    statistics: [majority]

output:
  path: "./output"
  format: netcdf
  sir_name: "delaware_params"

processing:
  engine: exactextract
  failure_mode: strict
  batch_size: 500
```

## Config Reference

See the [API Reference](../api/config.md) for full details on each config model:

- [`PipelineConfig`](../api/config.md#hydro_param.config.PipelineConfig) --- Top-level pipeline configuration
- [`TargetFabricConfig`](../api/config.md#hydro_param.config.TargetFabricConfig) --- Target polygon mesh
- [`DomainConfig`](../api/config.md#hydro_param.config.DomainConfig) --- Spatial extent
- [`DatasetRequest`](../api/config.md#hydro_param.config.DatasetRequest) --- Dataset selection
- [`OutputConfig`](../api/config.md#hydro_param.config.OutputConfig) --- Output format and location
- [`ProcessingConfig`](../api/config.md#hydro_param.config.ProcessingConfig) --- Engine and batching
