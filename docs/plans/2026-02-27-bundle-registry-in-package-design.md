# Bundle Dataset Registry in Package

**Date:** 2026-02-27
**Status:** Approved

## Problem

`DEFAULT_REGISTRY = Path("configs/datasets")` in `pipeline.py` is a relative
path that resolves against CWD. When hydro-param is pip-installed and run from
a user's project directory, the dataset registry is not found:

```
FileNotFoundError: Registry path does not exist: configs/datasets
```

## Solution

Move the dataset registry into the package tree and load it via
`importlib.resources`, following the existing pattern used by lookup tables
(`src/hydro_param/data/lookup_tables/`) and pywatershed metadata
(`src/hydro_param/data/pywatershed/`).

## Scope

### Move into package

- `configs/datasets/*.yml` (8 files) → `src/hydro_param/data/datasets/`

### Delete from `configs/` (already have canonical copies in package)

- `configs/datasets/` — moving to package (this PR)
- `configs/lookup_tables/` — package copy is canonical and more complete
  (has 2 extra files: `calibration_seeds.yml`, `forcing_variables.yml`)
- `configs/pywatershed/` — package copy is identical

### Keep at `configs/` (user-facing, not runtime data)

- `configs/examples/` — reference configs for users/developers
- `configs/delaware_terrain*.yml` — dev/sandbox configs

### Code changes

**`pipeline.py`** — Change `DEFAULT_REGISTRY` from relative `Path` to
`importlib.resources` resolution:

```python
# Before
DEFAULT_REGISTRY = Path("configs/datasets")

# After
from importlib.resources import files
DEFAULT_REGISTRY = Path(str(files("hydro_param").joinpath("data/datasets")))
```

**Docstrings** — Update any references to `configs/datasets/` path.

**No `pyproject.toml` changes** — Hatchling auto-includes non-Python files
within `packages = ["src/hydro_param"]`.

## Testing

- Existing tests pass (they use `load_registry()` with explicit paths)
- Add a test that `DEFAULT_REGISTRY` resolves to an existing directory
  containing the expected 8 YAML files
