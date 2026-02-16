# Installation

## Prerequisites

- Python >= 3.10
- [pixi](https://pixi.sh) (recommended) or pip

## Install with pixi (recommended)

pixi handles Python, geospatial C libraries (GDAL, GEOS, PROJ), and all
dependencies from conda-forge:

```bash
git clone https://github.com/rmcd-mscb/hydro-param.git
cd hydro-param
pixi install
```

## Install with pip

```bash
pip install hydro-param
```

For full processing capabilities, install optional dependency groups:

```bash
pip install hydro-param[gdp]     # gdptools + exactextract
pip install hydro-param[stac]    # STAC catalog access
pip install hydro-param[regrid]  # xESMF grid regridding
```

## Verify installation

```bash
hydro-param --help
```
