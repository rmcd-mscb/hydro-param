# hydro-param

**Configuration-driven hydrologic parameterization for any model, any fabric.**

`hydro-param` is a Python tool for generating spatially distributed parameters from gridded and polygon geospatial datasets, mapped onto arbitrary hydrologic response unit (HRU) fabrics. It fills the "missing middle" between raw data access libraries and hydrologic model execution.

> **Status:** Pre-alpha MVP. See [`docs/design.md`](docs/design.md) for the full architecture document.

## The Problem

Every hydrologic modeling project requires the same tedious workflow: download large geospatial datasets (soils, land cover, elevation, climate), intersect them with a target mesh of watersheds or grid cells, compute area-weighted statistics, and format the results for a specific model. This process is typically done with ad-hoc scripts that are fragile, slow, and not reproducible.

## The Approach

- **Config-driven:** Declare what you want in YAML — target fabric, datasets, parameters, output format — and the engine handles the rest.
- **Fabric-agnostic:** Works with NHM GFv1.1, NOAA NextGen hydrofabric, HUC12 watersheds, regular grids, or any polygon/grid mesh.
- **Multi-source data access:** Five strategies — STAC COG (3DEP, gNATSGO), NHGF STAC (NLCD on OSN), ClimR OPeNDAP (gridMET), local GeoTIFF (POLARIS, GFv1.1), and temporal NHGF STAC (SNODAS, CONUS404-BA).
- **Dask for lazy I/O only:** Uses Dask for efficient spatial subsetting of large stores. All computation uses numpy and gdptools.

## Key Design Decisions

- **Spatial batching** for I/O-efficient parallel processing (KD-tree recursive bisection)
- **Five data access strategies:** STAC COG, NHGF STAC (static and temporal), ClimR OPeNDAP, local GeoTIFF
- **POLARIS** (30m) over SSURGO/gNATSGO for soils
- **Plugin output formatters** for PRMS, NextGen, pywatershed, generic Parquet/NetCDF
- **Two-phase separation:** Generic pipeline produces a Standardized Internal Representation (SIR); all model-specific logic lives in plugins

## Architecture

```
YAML Config  →  Engine  →  Standardized Internal Representation  →  Formatter  →  Model Input
                  │
         ┌───────┼────────┐
         │       │        │
     gdptools  xesmf  rioxarray
    (polygon)  (grid)  (reproject)
```

## Project Structure

```
hydro-param/
├── src/hydro_param/       # Package source
│   ├── cli.py             # CLI (cyclopts)
│   ├── config.py          # Pydantic config
│   ├── pipeline.py        # 5-stage orchestrator
│   ├── processing.py      # gdptools wrapper
│   ├── derivations/       # Model-specific derivations
│   └── formatters/        # Output formatters
├── configs/               # Dataset registry + examples
├── tests/                 # 635 tests
├── docs/                  # MkDocs documentation
└── pyproject.toml         # Package + pixi config
```

## Quick Start

```bash
# Install
git clone https://github.com/rmcd-mscb/hydro-param.git
cd hydro-param && pixi install

# Initialize a project
hydro-param init my-project && cd my-project

# Explore datasets
hydro-param datasets list
hydro-param datasets info dem_3dep_10m

# Run the pipeline
hydro-param run configs/pipeline.yml
```

## Related Projects

| Project | Relationship |
|---------|-------------|
| [gdptools](https://gdptools.readthedocs.io/) | Core dependency — polygon intersection engine |
| [pywatershed](https://github.com/EC-USGS/pywatershed) | Target model — parameter consumer |
| [HyTEST](https://hytest-org.github.io/hytest/) | Data ecosystem — Zarr stores on OSN/S3 |
| [climateR](https://github.com/mikejohnson51/climateR) | Complementary — climate data catalog (R) |
| [NGIAB_data_preprocess](https://github.com/CIROH-UA/NGIAB_data_preprocess) | Adjacent — NextGen-specific preprocessing |

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

Rich McDonald — [Connected Waters LLC](https://connectedwatersllc.com)
