# CLAUDE.md — Project Context for AI Assistants

This file provides context for AI coding assistants (Claude Code, GitHub Copilot, etc.) working on the hydro-param project.

## Project Summary

hydro-param is a configuration-driven hydrologic parameterization system. It generates static physical parameters (soils, vegetation, topography, land cover) for hydrologic models by intersecting geospatial source datasets with target model fabrics.

**Read `docs/design.md` (v5.4) for the full architecture.** It contains detailed design decisions, trade-off analyses, two external review responses (Appendix A: Gemini 3 Pro, Appendix B: Gemini 2.5 Pro), CLI design, user workflow, processing pathway bifurcation, and the rationale behind every major architectural choice.

## Current Status

**Pre-alpha MVP.** The 5-stage config-driven pipeline works end-to-end for the Delaware River Basin test domain using 3DEP DEM (STAC COG) and NLCD (local GeoTIFF):

| Stage | Status | Module |
|-------|--------|--------|
| 1. Resolve fabric | Working | `pipeline.py` → `stage1_resolve_fabric()` |
| 2. Resolve datasets | Working | `pipeline.py` → `stage2_resolve_datasets()` |
| 3. Compute weights | Working (gdptools internal) | — |
| 4. Process datasets | Working (stac_cog + local_tiff*) | `pipeline.py` → `stage4_process()` |
| 5. Format output | Working (NetCDF/Parquet) | `pipeline.py` → `stage5_format_output()` |

\* `local_tiff` requires the `source` path to be manually specified in the dataset registry entry. CLI-based download and pipeline config `source` override are planned for Phase 1.

**Implemented data access strategies:** `stac_cog` (Planetary Computer), `local_tiff` (local GeoTIFF).
**Planned (Phase 1):** CLI with `datasets download` command, registry `download` blocks, pipeline config `source` override for user-supplied paths.
**Planned (Phase 2):** `climr_cat` (gridMET via ClimateR-Catalog + gdptools ClimRCatData/AggGen).
**Not yet implemented:** `native_zarr`, `converted_zarr`.

## Key Architectural Decisions

1. **Config is declarative only.** YAML configs say *what* to compute, not *how*. No variables, conditionals, or templating in config files. Logic belongs in Python scripts using the library API.

2. **Two processing pathways.** Polygon targets (HRUs, catchments) use gdptools. Grid targets use xesmf/rioxarray. A factory function routes based on target fabric type.

3. **Dask is for I/O only.** Dask lazy loading for efficient spatial subsetting of Zarr stores. All computation uses numpy/gdptools via joblib or SLURM arrays. Do NOT introduce dask.distributed.

4. **Spatial batching is foundational.** Features must be grouped into spatially contiguous batches before processing. KD-tree recursive bisection is the MVP method. Never assume fabric row order is spatially coherent.

5. **Output formatters are plugins.** Core engine produces xarray Datasets (Standardized Internal Representation with strict schema validation). Model-specific output formats (PRMS, NextGen, pywatershed) are separate adapter classes.

6. **Cache by stable IDs, not geometry hashes.** Weight and batch caches use composite keys from fabric version + dataset ID + CRS + method. Never hash floating-point geometry coordinates.

7. **Fault tolerance by default.** Production runs and MVP use tolerant mode — failed HRUs logged to CSV, processing continues. Strict mode for development/debugging.

8. **Local data over remote services.** Spatial batching generates many concurrent data requests per run. Remote services (WMS/WCS) throttle under this load. Prefer local COG/GeoTIFF via transparent cache over remote endpoints (see design.md §6.12).

9. **SIR schema is enforced.** The Standardized Internal Representation uses canonical variable names encoding units (e.g., `elevation_m`, `slope_deg`, `ksat_cm_hr`), guaranteed unit conversions, and a `validate_sir()` step (see design.md §A.8 A′).

10. **hydro-param does NOT fetch/subset fabrics.** Input must be a pre-existing geospatial file (GeoPackage/Parquet). Use pynhd/pygeohydro upstream.

11. **Library-managed transparent caching.** No project directory scaffolding. Data caching is pooch-style, managed by the library. The pipeline config is the single source of truth (see design.md §6.11).

12. **gdptools handles CRS alignment.** `ZonalGen` reprojects target geometry into source CRS by design. No additional CRS reprojection needed in hydro-param batch processing.

## Module Architecture

```
src/hydro_param/
  config.py            — Pydantic config schema + YAML loader
  dataset_registry.py  — Registry schema + YAML loader + variable resolution
  data_access.py       — STAC COG fetch, local GeoTIFF fetch, terrain derivation
  batching.py          — KD-tree spatial batching
  processing.py        — gdptools ZonalGen wrapper (continuous + categorical)
  pipeline.py          — 5-stage orchestrator with batch loop
```

**Planned (Phase 1):**
```
  cli.py               — CLI entry point: datasets list/info/download, run
```

**Config files:**
```
configs/
  datasets.yml         — Dataset registry (3DEP, POLARIS, NLCD entries)
  delaware_terrain.yml — Example pipeline config for Delaware River Basin
```

## Dependencies & Environment

- Python ≥ 3.11
- Core: xarray, geopandas, numpy, pydantic (config validation)
- Spatial: gdptools, rioxarray, pyproj
- Compute: joblib (default), optional: coiled, boto3
- I/O: zarr, fsspec, s3fs
- Optional: xesmf (grid regridding), dask (lazy I/O only)

Package management: **pixi** (configured in `pyproject.toml` under `[tool.pixi.*]`). Use `pixi install` to create/sync environments.

**Pixi environments:**
- `dev` — daily development (pytest, ruff, mypy, pre-commit). Does NOT include rioxarray/gdptools.
- `full` — all optional deps including gdptools, rioxarray, exactextract. Use for integration tests.
- `test-py311` / `test-py312` — CI matrix environments.
- `download` — includes pynhd for fabric download scripts.

## Coding Conventions

- src layout (`src/hydro_param/`)
- Type hints on all public functions
- Pydantic models for config validation
- pytest for testing; use `pytest.importorskip()` for tests requiring optional deps (rioxarray, gdptools)
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
- **Run integration tests (with gdptools):** `pixi run -e full pytest tests/`

## Development Workflow

See `CONTRIBUTING.md` for the full guide. Key rules for AI assistants:

- **GitHub Flow**: All changes go through feature branches and pull requests against `main`. Never commit directly to `main`.
- **Issue first**: Every code change starts with a GitHub issue. Reference the issue number in branch names and PR descriptions.
- **Branch naming**: `<type>/<issue-number>-<short-description>` (e.g., `feat/19-local-tiff-access`).
- **Conventional commits**: Use `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `ci:` prefixes.
- **Pre-commit hooks**: Run `pre-commit run --all-files` before suggesting a commit. Hooks enforce ruff, mypy, and secrets detection.
- **PR template**: Fill out the summary, related issue (`Closes #N`), test plan, and checklist.
- **Git remote**: SSH (`git@github.com:rmcd-mscb/hydro-param.git`).

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
- Don't wire remote data services (WMS/WCS) into the batch processing loop — use local files
- Don't build fabric subsetting/fetching into hydro-param — that's pynhd's job

## Open Work (design.md §10.4)

**Phase 1 priorities (current sprint):**

- CLI: `hydro-param datasets list/info/download` + `hydro-param run` (cyclopts) — §11.9
- Registry `download` block for NLCD with real download URLs — §6.6
- Pipeline config `source` override for `local_tiff` datasets — §6.6
- NLCD categorical processing end-to-end (Delaware domain test)

**Phase 1 priorities (remaining):**

- Processing pathway bifurcation: polygon vs grid targets (gdptools vs xesmf)
- Compute backend interface (serial + joblib)
- SIR schema validation with `validate_sir()`
- Library-managed transparent data caching
- Registry version pinning
- Spatial correctness validation (regression tests)
- Dockerfile + Apptainer docs

**Phase 2 (designed, deferred):**

- Temporal processing: gridMET via ClimRCatData → WeightGen → AggGen — §11.11
- Weight caching (critical for AggGen reuse across variables)
