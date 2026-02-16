# hydro-param

**Configuration-driven hydrologic parameterization for any model, any fabric.**

hydro-param is a Python tool for generating spatially distributed parameters
from gridded and polygon geospatial datasets, mapped onto arbitrary hydrologic
response unit (HRU) fabrics.

!!! warning "Pre-alpha"

    This project is in active development. The API may change.

## The Problem

Every hydrologic modeling project requires the same tedious workflow: download
large geospatial datasets (soils, land cover, elevation, climate), intersect
them with a target mesh of watersheds or grid cells, compute area-weighted
statistics, and format the results for a specific model.

## The Approach

- **Config-driven** --- Declare what you want in YAML and the engine handles the rest.
- **Fabric-agnostic** --- Works with NHM GFv1.1, NOAA NextGen, HUC12, or any polygon mesh.
- **Cloud-native data** --- Reads directly from STAC catalogs. No bulk downloads required.
- **Scalable** --- Same config runs on a laptop (regional) or HPC (CONUS).

## Quick Links

- [Installation](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [CLI Reference](user-guide/cli.md)
- [API Reference](api/index.md)
- [Architecture](design.md)
