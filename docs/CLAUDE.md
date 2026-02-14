# CLAUDE.md — Project Context for AI Assistants

This file provides context for AI coding assistants (Claude Code, GitHub Copilot, etc.) working on the hydro-param project.

## Project Summary

hydro-param is a configuration-driven hydrologic parameterization system. It generates static physical parameters (soils, vegetation, topography, land cover) for hydrologic models by intersecting geospatial source datasets with target model fabrics.

**Read `docs/design.md` for the full architecture.** It contains detailed design decisions, trade-off analyses, and the rationale behind every major architectural choice.

## Key Architectural Decisions

1. **Config is declarative only.** YAML configs say *what* to compute, not *how*. No variables, conditionals, or templating in config files. Logic belongs in Python scripts using the library API.

2. **Two processing pathways.** Polygon targets (HRUs, catchments) use gdptools. Grid targets use xesmf/rioxarray. A factory function routes based on target fabric type.

3. **Dask is for I/O only.** Dask lazy loading for efficient spatial subsetting of Zarr stores. All computation uses numpy/gdptools via joblib or SLURM arrays. Do NOT introduce dask.distributed.

4. **Spatial batching is foundational.** Features must be grouped into spatially contiguous batches before processing. Use Hilbert curve sorting or KD-tree bisection. Never assume fabric row order is spatially coherent.

5. **Output formatters are plugins.** Core engine produces xarray Datasets (Standardized Internal Representation). Model-specific output formats (PRMS, NextGen, pywatershed) are separate adapter classes.

6. **Cache by stable IDs, not geometry hashes.** Weight and batch caches use composite keys from fabric version + dataset ID + CRS + method. Never hash floating-point geometry coordinates.

7. **Fault tolerance by default.** Production runs use tolerant mode — failed HRUs logged to CSV, processing continues. Strict mode for development/debugging.

## Dependencies & Environment

- Python ≥ 3.11
- Core: xarray, geopandas, numpy, pydantic (config validation)
- Spatial: gdptools, rioxarray, pyproj
- Compute: joblib (default), optional: coiled, boto3
- I/O: zarr, fsspec, s3fs
- Optional: xesmf (grid regridding), dask (lazy I/O only)

Package management: **pixi** (configured in `pyproject.toml` under `[tool.pixi.*]`). Use `pixi install` to create/sync environments. Run local development tasks (tests, linting, formatting) via `pixi run -e dev`. CI runs tests in dedicated `test-py311`/`test-py312` environments.

## Coding Conventions

- src layout (`src/hydro_param/`)
- Type hints on all public functions
- Pydantic models for config validation
- pytest for testing
- NumPy-style docstrings
- Logging via Python stdlib `logging`, not print statements

## Common Tasks

- **Run tests:** `pixi run -e dev test`
- **Type check:** `pixi run -e dev typecheck`
- **Format:** `pixi run -e dev format`
- **Lint:** `pixi run -e dev lint`
- **All checks:** `pixi run -e dev check`
- **Pre-commit:** `pixi run -e dev pre-commit`
- **Install/sync environment:** `pixi install`

## Development Workflow

See `CONTRIBUTING.md` for the full guide. Key rules for AI assistants:

- **GitHub Flow**: All changes go through feature branches and pull requests against `main`. Never commit directly to `main`.
- **Issue first**: Every code change starts with a GitHub issue. Reference the issue number in branch names and PR descriptions.
- **Branch naming**: `<type>/<issue-number>-<short-description>` (e.g., `feat/12-config-schema`).
- **Conventional commits**: Use `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:` prefixes.
- **Pre-commit hooks**: Run `pre-commit run --all-files` before suggesting a commit. Hooks enforce ruff, mypy, and secrets detection.
- **PR template**: Fill out the summary, related issue (`Closes #N`), test plan, and checklist.

## Before Every Push

Always run these locally before pushing to a branch:

1. `pixi run -e dev check` — runs lint, format-check, typecheck, and tests
2. `pixi run -e dev pre-commit` — runs all pre-commit hooks (ruff, mypy, detect-secrets)

If you modified `pyproject.toml`, also run `pixi install` to regenerate `pixi.lock` — CI uses `--locked` and will fail if the lock file is stale.

## What NOT to Do

- Don't add dask.distributed as a dependency
- Don't put conditional logic in YAML configs
- Don't hash geometry coordinates for cache keys
- Don't hardcode model-specific output logic in the core engine
- Don't treat grids as polygons for raster-on-raster operations
- Don't commit directly to `main` — always use a feature branch and PR
- Don't create commits without conventional commit prefixes
- Don't skip pre-commit hooks
