# hydro-param

**Configuration-driven hydrologic parameterization for any model, any fabric.**

!!! warning "Pre-alpha"

    This project is in active development. The API may change without notice.
    Currently at 1,015 tests passing.

## What It Does

hydro-param generates static physical parameters (soils, vegetation, topography,
land cover, climate normals) for hydrologic models by intersecting geospatial
source datasets with target model fabrics. You declare what parameters you need
in a YAML configuration file, and the engine handles data access, spatial
intersection, zonal statistics, and model-specific derivations.

## Key Features

- **Config-driven YAML pipelines** --- Declare what to compute, not how.
  No variables, conditionals, or templating in config files.
- **Five data access strategies** --- STAC COG (3DEP, gNATSGO), NHGF STAC
  static (NLCD) and temporal (SNODAS, CONUS404-BA), ClimR OPeNDAP (gridMET),
  and local GeoTIFF (POLARIS, GFv1.1).
- **Two-phase architecture** --- A generic, model-agnostic pipeline produces
  standardized intermediate results. Model plugins handle all domain-specific
  logic independently.
- **Fabric-agnostic** --- Works with NHM GFv1.1, NOAA NextGen, HUC12, or any
  polygon mesh provided as a GeoPackage or GeoParquet file.
- **Incremental processing with manifest-based resume** --- Failed or
  interrupted runs pick up where they left off. Failed features are logged,
  not fatal.
- **pywatershed as primary target model** --- 14 derivation steps producing
  ~100 PRMS parameters, including lookup-table reclassification, unit
  conversions, climate normals, and solar geometry.

## Two-Phase Architecture

hydro-param enforces a strict separation between generic data processing and
model-specific logic. This makes the core engine reusable across any target
model.

!!! info "Phase 1: Generic Pipeline"

    The 5-stage pipeline is model-agnostic. It resolves the target fabric,
    resolves source datasets, computes intersection weights, processes zonal
    statistics, and writes results to the **Standardized Internal
    Representation (SIR)** --- a collection of per-variable CSV and NetCDF
    files in source units with source names.

    **Stages:** Resolve fabric | Resolve datasets | Compute weights | Process datasets | Format output

!!! info "Phase 2: Model Plugin"

    A model plugin (e.g., pywatershed) reads the SIR and performs all
    model-specific transforms: unit conversions (meters to feet, Celsius to
    Fahrenheit), variable renaming, majority extraction, gap-filling, derived
    math (soil moisture capacity, Jensen-Haise coefficients), lookup-table
    reclassification, and output formatting.

    **The pipeline knows nothing about any target model.** All model-specific
    logic lives in the plugin.

```
YAML Config
    |
    v
+-----------------------+       +---------------------------+
| Phase 1: Pipeline     |       | Phase 2: Model Plugin     |
|                       |       |                           |
| 1. Resolve fabric     |       | - Unit conversions        |
| 2. Resolve datasets   |  SIR  | - Variable renaming       |
| 3. Compute weights    | ----> | - Lookup-table reclassify |
| 4. Process datasets   |       | - Derived parameters      |
| 5. Format output      |       | - Output formatting       |
+-----------------------+       +---------------------------+
                                            |
                                            v
                                   Model Parameter File
```

## Quick Links

| | |
|---|---|
| [Installation](getting-started/installation.md) | Set up hydro-param with pixi or pip |
| [Quick Start](getting-started/quickstart.md) | Run your first parameterization in minutes |
| [pywatershed Workflow](user-guide/pywatershed-workflow.md) | End-to-end guide for pywatershed/PRMS users |
| [CLI Reference](user-guide/cli.md) | Command-line interface documentation |
| [API Reference](api/index.md) | Python library API |
| [Architecture](design.md) | Full design document with trade-off analyses |
| [Development Roadmap](plans/development-roadmap.md) | Current priorities and future plans |
