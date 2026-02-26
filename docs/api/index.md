# API Reference

Auto-generated API documentation from source code docstrings.

## Public API

The following are exported from the top-level `hydro_param` package:

- [`PipelineConfig`](config.md) --- Pipeline configuration model
- [`DatasetRegistry`](registry.md) --- Dataset registry container
- [`load_config`](config.md) --- Load pipeline YAML
- [`load_registry`](registry.md) --- Load dataset registry
- [`run_pipeline`](pipeline.md) --- Execute the pipeline

## Module Index

### Core

| Module | Description |
|---|---|
| [`config`](config.md) | Pipeline configuration models and YAML loader |
| [`pipeline`](pipeline.md) | 5-stage pipeline orchestrator |
| [`sir`](sir.md) | Standardized Internal Representation normalization |
| [`dataset_registry`](registry.md) | Dataset registry schema and loader |
| [`data_access`](data-access.md) | STAC COG, local GeoTIFF, terrain derivation |
| [`processing`](processing.md) | Zonal statistics via gdptools |
| [`batching`](batching.md) | Spatial batching via KD-tree bisection |
| [`manifest`](manifest.md) | Pipeline manifest for incremental runs |
| [`units`](units.md) | Unit conversion utilities (SI to PRMS imperial) |
| [`project`](project.md) | Project scaffolding and root detection |

### Plugins

| Module | Description |
|---|---|
| [`plugins`](plugins.md) | Plugin protocols, DerivationContext, and factory functions |
| [`derivations.pywatershed`](derivations-pywatershed.md) | pywatershed parameter derivation (steps 1--14) |
| [`formatters.pywatershed`](formatters-pywatershed.md) | pywatershed output formatter (NetCDF, CSV, control) |
| [`pywatershed_config`](pywatershed-config.md) | pywatershed-specific configuration schema |

### Utilities

| Module | Description |
|---|---|
| [`cli`](cli.md) | Command-line interface (cyclopts) |
| [`solar`](solar.md) | Clear-sky solar radiation (Swift 1976 soltab) |
