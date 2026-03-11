# hydro-param

**Configuration-driven hydrologic parameterization for any model, any fabric.**

[![CI](https://github.com/rmcd-mscb/hydro-param/actions/workflows/ci.yml/badge.svg)](https://github.com/rmcd-mscb/hydro-param/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status:** Pre-alpha MVP. 1,015 tests passing. See the [documentation site](https://rmcd-mscb.github.io/hydro-param/) and [`docs/design.md`](docs/design.md) for the full architecture.

## The Problem

Every hydrologic modeling project requires the same tedious workflow: download large geospatial datasets (soils, land cover, elevation, climate), intersect them with a target mesh of watersheds or grid cells, compute area-weighted statistics, and format the results for a specific model. This process is typically done with ad-hoc scripts that are fragile, slow, and not reproducible.

## The Approach

- **Config-driven:** Declare what you want in YAML -- target fabric, datasets, parameters, output format -- and the engine handles the rest.
- **Fabric-agnostic:** Works with NHM GFv1.1, NOAA NextGen hydrofabric, HUC12 watersheds, regular grids, or any polygon/grid mesh.
- **Five data access strategies:** STAC COG, NHGF STAC (static and temporal), ClimR OPeNDAP, and local GeoTIFF -- covering 3DEP, gNATSGO, NLCD, POLARIS, GFv1.1, SNODAS, CONUS404-BA, and gridMET.
- **Two-phase architecture:** A generic pipeline produces a Standardized Internal Representation (SIR). All model-specific logic -- unit conversions, variable renaming, derived math, output formatting -- lives in plugins.
- **Cloud-native and local:** Transparently accesses data from Planetary Computer STAC, USGS NHGF STAC (OSN), OPeNDAP, or local GeoTIFF files.

## Architecture

hydro-param separates concerns into two phases:

```
Phase 1: Generic Pipeline (model-agnostic)
==========================================

YAML Config
    |
    v
 Stage 1: Resolve Fabric -----> GeoDataFrame (HRUs / grid cells)
 Stage 2: Resolve Datasets ----> Dataset registry + access strategy
 Stage 3: Compute Weights -----> gdptools spatial weights
 Stage 4: Process Datasets ----> Zonal statistics per feature
 Stage 5: Format Output -------> Standardized Internal Representation (SIR)
                                     |
                                     v
Phase 2: Model Plugin (e.g., pywatershed)
=========================================

 SIR on disk (CSV / NetCDF per dataset)
    |
    v
 Load SIR --> Derive parameters --> Unit conversions --> Format output
              (lookup tables,        (m -> ft,            (NetCDF,
               reclassify,            C -> F,              Parquet,
               formulas)              mm -> in)            PRMS)
```

### Data Access Strategies

| Strategy | gdptools Class | Datasets |
|----------|---------------|----------|
| `stac_cog` | UserTiffData | 3DEP 10m DEM, gNATSGO soils |
| `local_tiff` | UserTiffData | POLARIS 30m soils, GFv1.1 rasters |
| `nhgf_stac` (static) | NHGFStacTiffData | NLCD Annual (6 collections on OSN) |
| `nhgf_stac` (temporal) | NHGFStacData | SNODAS, CONUS404-BA |
| `climr_cat` | ClimRCatData | gridMET (OPeNDAP) |

## Quick Start

### Install

```bash
git clone https://github.com/rmcd-mscb/hydro-param.git
cd hydro-param
pixi install
```

### Path 1: Terrain-only pipeline

```bash
# Scaffold a project
hydro-param init my-project && cd my-project

# Explore available datasets
hydro-param datasets list
hydro-param datasets info dem_3dep_10m

# Run the pipeline with your config
hydro-param run configs/pipeline.yml
```

### Path 2: Full pywatershed parameterization

```bash
# Run the generic pipeline to produce SIR
hydro-param run configs/pipeline.yml

# Generate pywatershed parameters from SIR
hydro-param pywatershed run configs/pywatershed_run.yml

# Validate the output
hydro-param pywatershed validate output/parameters.nc
```

### Path 3: Download GFv1.1 national rasters

```bash
# Download GFv1.1 rasters from ScienceBase
hydro-param gfv11 download
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `hydro-param init [DIR]` | Scaffold a new project directory |
| `hydro-param datasets list` | Show available datasets grouped by category |
| `hydro-param datasets info NAME` | Show full details for a dataset |
| `hydro-param datasets download NAME` | Download dataset files via AWS CLI |
| `hydro-param run CONFIG` | Run the 5-stage generic pipeline |
| `hydro-param pywatershed run CONFIG` | Generate pywatershed model parameters from SIR |
| `hydro-param pywatershed validate FILE` | Validate parameter file against metadata |
| `hydro-param gfv11 download` | Download GFv1.1 national rasters from ScienceBase |

## Project Structure

```
hydro-param/
├── src/hydro_param/           # Package source
│   ├── cli.py                 # CLI entry point (cyclopts)
│   ├── config.py              # Pydantic config schema + YAML loader
│   ├── pipeline.py            # 5-stage generic orchestrator
│   ├── processing.py          # gdptools ZonalGen wrapper
│   ├── data_access.py         # STAC COG / local GeoTIFF fetch
│   ├── batching.py            # KD-tree spatial batching
│   ├── dataset_registry.py    # Registry schema + variable resolution
│   ├── classification.py      # USDA soil texture classifier
│   ├── sir_accessor.py        # Lazy per-variable SIR loading
│   ├── plugins.py             # Plugin protocols + factory functions
│   ├── gfv11.py               # GFv1.1 ScienceBase download
│   ├── solar.py               # Solar geometry (soltab)
│   ├── units.py               # Unit conversion utilities
│   ├── derivations/           # Model-specific derivation plugins
│   │   └── pywatershed.py     # pywatershed parameter derivation
│   ├── formatters/            # Output formatter plugins
│   │   └── pywatershed.py     # pywatershed output formatter
│   └── data/                  # Bundled data (importlib.resources)
│       ├── datasets/          # Dataset registry (8 YAML files)
│       └── pywatershed/       # Parameter metadata + lookup tables
├── tests/                     # 1,015 tests
├── configs/                   # Example pipeline configs
├── docs/                      # MkDocs documentation
└── pyproject.toml             # Package + pixi workspace config
```

## Documentation

Full documentation is available at **[rmcd-mscb.github.io/hydro-param](https://rmcd-mscb.github.io/hydro-param/)**.

Key references:
- [`docs/design.md`](docs/design.md) -- Architecture document with design decisions and trade-off analyses
- [`docs/reference/pywatershed_dataset_param_map.yml`](docs/reference/pywatershed_dataset_param_map.yml) -- Authoritative dataset-to-parameter mapping for pywatershed

## Development

hydro-param uses [pixi](https://pixi.sh) for environment management.

```bash
pixi install                     # Create/sync environments
pixi run -e dev test             # Run tests
pixi run -e dev check            # Run lint + format-check + typecheck + tests
pixi run -e dev pre-commit       # Run all pre-commit hooks
pixi run -e docs docs-build      # Build documentation
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

## Related Projects

| Project | Relationship |
|---------|-------------|
| [gdptools](https://gdptools.readthedocs.io/) | Core dependency -- polygon intersection engine |
| [pywatershed](https://github.com/EC-USGS/pywatershed) | Target model -- parameter consumer |
| [HyTEST](https://hytest-org.github.io/hytest/) | Data ecosystem -- Zarr stores on OSN/S3 |
| [climateR](https://github.com/mikejohnson51/climateR) | Complementary -- climate data catalog (R) |
| [NGIAB_data_preprocess](https://github.com/CIROH-UA/NGIAB_data_preprocess) | Adjacent -- NextGen-specific preprocessing |

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Rich McDonald -- [Connected Waters LLC](https://connectedwatersllc.com)
