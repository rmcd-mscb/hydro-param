# hydro-param

**Configuration-driven hydrologic parameterization for any model, any fabric.**

`hydro-param` is a Python tool for generating spatially distributed parameters from gridded and polygon geospatial datasets, mapped onto arbitrary hydrologic response unit (HRU) fabrics. It fills the "missing middle" between raw data access libraries and hydrologic model execution.

> **Status:** Design & planning phase. See [`docs/design.md`](docs/design.md) for the full architecture document.

## The Problem

Every hydrologic modeling project requires the same tedious workflow: download large geospatial datasets (soils, land cover, elevation, climate), intersect them with a target mesh of watersheds or grid cells, compute area-weighted statistics, and format the results for a specific model. This process is typically done with ad-hoc scripts that are fragile, slow, and not reproducible.

## The Approach

- **Config-driven:** Declare what you want in YAML — target fabric, datasets, parameters, output format — and the engine handles the rest.
- **Fabric-agnostic:** Works with NHM GFv1.1, NOAA NextGen hydrofabric, HUC12 watersheds, regular grids, or any polygon/grid mesh.
- **Cloud-native data:** Reads directly from Zarr stores on S3/OSN. No bulk downloads required.
- **Scalable:** Runs on a laptop (regional), HPC via SLURM (CONUS), or cloud via Coiled/AWS (burst computing). Same config, any backend.
- **Honest about Dask:** Uses Dask for lazy I/O (what it's good at), not for distributed scheduling on HPC (where it's fragile). Parallelism via joblib and SLURM arrays.

## Key Design Decisions

- **Spatial batching** for I/O-efficient parallel processing (Hilbert curve sorting)
- **Three-tier data strategy:** native Zarr → virtual Zarr (Kerchunk) → converted Zarr
- **POLARIS** (100m probabilistic) over SSURGO/gNATSGO for soils
- **Plugin output formatters** for PRMS, NextGen, pywatershed, generic Parquet/NetCDF
- **Separate data package** (`hydro-param-data`) for standalone dataset access

## Planned Architecture

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
├── docs/
│   └── design.md          # Full architecture & design document
├── src/
│   └── hydro_param/       # Package source (coming soon)
├── tests/                  # Test suite (coming soon)
├── pyproject.toml          # Package metadata
├── LICENSE
└── README.md
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
