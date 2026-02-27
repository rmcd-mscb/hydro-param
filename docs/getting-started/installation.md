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

!!! note
    pip installation requires that GDAL, GEOS, and PROJ system libraries
    are already installed. Using pixi (above) is recommended as it handles
    these automatically.

## Verify installation

```bash
hydro-param --help
```
