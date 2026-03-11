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

!!! info "Pixi environments"
    pixi creates three environments:

    - `default` --- core dependencies only
    - `dev` --- development (pytest, ruff, mypy, pre-commit)
    - `docs` --- documentation building (mkdocs-material)

    Most development commands use the `dev` environment:
    `pixi run -e dev test`, `pixi run -e dev check`, etc.

## Install with pip

!!! warning
    hydro-param is not yet published to PyPI. Install from source using
    pixi (above) or `pip install -e .` in a development environment with
    GDAL, GEOS, and PROJ system libraries pre-installed.

## Verify installation

```bash
pixi run -e dev -- hydro-param --help
```

Run the test suite to confirm everything works:

```bash
pixi run -e dev test
```
