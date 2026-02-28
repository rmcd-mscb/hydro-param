# CLAUDE.md — Project Context for AI Assistants

This file provides context for AI coding assistants (Claude Code, GitHub Copilot, etc.) working on the hydro-param project.

## Project Summary

hydro-param is a configuration-driven hydrologic parameterization system. It generates static physical parameters (soils, vegetation, topography, land cover) for hydrologic models by intersecting geospatial source datasets with target model fabrics.

**Read `docs/design.md` for the full architecture.** It contains detailed design decisions, trade-off analyses, two external review responses (Appendix A: Gemini 3 Pro, Appendix B: Gemini 2.5 Pro), CLI design, user workflow, processing pathway bifurcation, and the rationale behind every major architectural choice.

## Current Status

**Pre-alpha MVP.** The 5-stage config-driven pipeline works end-to-end with 635 tests passing. Five data access strategies are implemented:

| Stage | Status | Module |
|-------|--------|--------|
| 1. Resolve fabric | Working | `pipeline.py` → `stage1_resolve_fabric()` |
| 2. Resolve datasets | Working | `pipeline.py` → `stage2_resolve_datasets()` |
| 3. Compute weights | Working (gdptools internal) | — |
| 4. Process datasets | Working (all 5 strategies) | `pipeline.py` → `stage4_process()` |
| 5. Format output | Working (NetCDF/Parquet) | `pipeline.py` → `stage5_format_output()` |

**Processing pathways (gdptools):**

| Strategy | gdptools Class | Datasets |
|----------|---------------|----------|
| `stac_cog` | UserTiffData → ZonalGen | 3DEP, gNATSGO |
| `local_tiff` | UserTiffData → ZonalGen | NLCD legacy, POLARIS |
| `nhgf_stac` (static) | NHGFStacTiffData → ZonalGen | NLCD OSN (6 collections) |
| `nhgf_stac` (temporal) | NHGFStacData → WeightGen → AggGen | SNODAS, CONUS404-BA |
| `climr_cat` | ClimRCatData → WeightGen → AggGen | gridMET (OPeNDAP) |

Temporal processing supports calendar-year splitting for multi-year datasets.

## Key Architectural Decisions

1. **Two-phase separation (pipeline ↔ plugin).** The generic pipeline (stages 1–5) is model-agnostic — it knows nothing about any target model. All model-specific logic (unit conversions, variable renaming, majority extraction, gap-filling, derived math, output formatting) lives in model plugins (e.g., `derivations/pywatershed.py`, `formatters/pywatershed.py`). **Never put model-specific logic in the pipeline.**

2. **Config is declarative only.** YAML configs say *what* to compute, not *how*. No variables, conditionals, or templating in config files. Logic belongs in Python scripts using the library API.

3. **Two processing pathways.** Polygon targets (HRUs, catchments) use gdptools. Grid targets use xesmf/rioxarray. A factory function routes based on target fabric type.

4. **Dask is for I/O only.** Dask lazy loading for efficient spatial subsetting of Zarr stores. All computation uses numpy/gdptools. Do NOT introduce dask.distributed.

5. **Spatial batching is foundational.** Features must be grouped into spatially contiguous batches before processing. KD-tree recursive bisection is the MVP method. Never assume fabric row order is spatially coherent.

6. **Output formatters are plugins.** Core engine produces a Standardized Internal Representation (SIR). Model-specific output formats (PRMS, NextGen, pywatershed) are separate adapter classes.

7. **Cache by stable IDs, not geometry hashes.** Weight and batch caches use composite keys from fabric version + dataset ID + CRS + method. Never hash floating-point geometry coordinates.

8. **Fault tolerance by default.** Production runs and MVP use tolerant mode — failed HRUs logged to CSV, processing continues. Strict mode for development/debugging.

9. **Local data over remote services.** Spatial batching generates many concurrent data requests per run. Remote services (WMS/WCS) throttle under this load. Prefer local COG/GeoTIFF via transparent cache over remote endpoints (see design.md §6.12).

10. **SIR normalization is planned.** The Standardized Internal Representation currently returns raw gdptools output (source units, source names). A planned normalization step will standardize names/units at the SIR boundary with a `validate_sir()` function before returning to consumers. Not yet implemented.

11. **hydro-param does NOT fetch/subset fabrics.** Input must be a pre-existing geospatial file (GeoPackage/Parquet). Use pynhd/pygeohydro upstream.

12. **Library-managed transparent caching.** No project directory scaffolding. Data caching is pooch-style, managed by the library. The pipeline config is the single source of truth (see design.md §6.11).

13. **gdptools handles CRS alignment.** `ZonalGen` reprojects target geometry into source CRS by design. No additional CRS reprojection needed in hydro-param batch processing.

## Module Architecture

```
src/hydro_param/
  __init__.py
  __main__.py
  cli.py                — CLI entry point (cyclopts): init, datasets, run, pywatershed
  config.py             — Pydantic config schema + YAML loader
  dataset_registry.py   — Registry schema + YAML loader + variable resolution
  data_access.py        — STAC COG fetch, local GeoTIFF fetch, terrain derivation
  batching.py           — KD-tree spatial batching
  processing.py         — gdptools ZonalGen wrapper (continuous + categorical)
  pipeline.py           — 5-stage generic orchestrator (model-agnostic)
  project.py            — Project context management
  units.py              — Unit conversion utilities
  output.py             — Output writing (NetCDF/Parquet)
  pywatershed_config.py — pywatershed-specific configuration
  derivations/
    __init__.py
    pywatershed.py      — pywatershed parameter derivation plugin
  formatters/
    __init__.py
    pywatershed.py      — pywatershed output formatter plugin
```

**Bundled data** (loaded via `importlib.resources`):
```
src/hydro_param/data/
  datasets/              — Dataset registry (8 per-category YAML files)
  lookup_tables/         — PRMS parameter derivation tables (6 YAML files)
  pywatershed/           — pywatershed parameter metadata
```

**Config files** (user-facing reference, not loaded at runtime):
```
configs/
  examples/              — Example pipeline configs (DRB 2-year)
  delaware_terrain.yml   — Delaware River Basin pipeline config
```

## Dependencies & Environment

- Python ≥ 3.10
- Core: gdptools (≥0.3.10), xarray, geopandas, numpy, pandas, pydantic (≥2.0), cyclopts
- Spatial: rioxarray, pyproj, shapely
- STAC: pystac-client, planetary-computer
- I/O: pyyaml

Package management: **pixi** (configured in `pyproject.toml` under `[tool.pixi.*]`). Use `pixi install` to create/sync environments.

**Pixi environments:**
- `default` — core dependencies only (gdptools, geopandas, xarray, etc.)
- `dev` — daily development (default + pytest, ruff, mypy, pre-commit)
- `docs` — documentation building (mkdocs-material, mkdocstrings)

## Execution Style

When executing an approved plan (e.g., from `/superpowers:writing-plans` or `/superpowers:executing-plans`), **prioritize doing over re-planning**. Do not re-read design docs, re-analyze architecture, or ask redundant clarifying questions when the plan already covers the approach. Start executing tasks immediately. Only pause to investigate if a task fails or produces unexpected results.

## Coding Conventions

- src layout (`src/hydro_param/`)
- Type hints on all public functions
- Pydantic models for config validation
- pytest for testing; use `pytest.importorskip()` for tests requiring optional deps (rioxarray, gdptools)
- NumPy-style docstrings (see Docstring Requirements below)
- Logging via Python stdlib `logging`, not print statements

### Docstring Requirements

Write thorough NumPy-style docstrings. This code is maintained by domain scientists and physical reviewers who need to understand intent, not just interface.

**Scope:**
- **Public functions, classes, methods:** Full docstrings always.
- **Private (`_foo`) methods:** Full docstrings when logic is non-trivial (algorithmic steps, unit conversions, GIS operations). Simple helpers get a one-line summary.

**Required sections** (include all that apply):

```
Summary line (imperative mood, one line).

Extended description explaining *why* this exists, the approach taken,
and any domain context a maintainer would need.

Parameters
----------
param_name : type
    Description including units, valid ranges, and defaults.

Returns
-------
type
    Description including units and structure.

Raises
------
ExceptionType
    When and why this is raised.

Notes
-----
Implementation details, algorithm references, unit conversion
rationale, or caveats that aren't obvious from the code.

Examples
--------
>>> short_usage_example()
expected_output

References
----------
Citation or link to the source algorithm, paper, or PRMS documentation.

See Also
--------
related_function : Brief note on relationship.

Warnings
--------
Known limitations, precision issues, or conditions where results
may be unreliable.
```

**Guidelines:**
- Always document units (feet, acres, m², °F, etc.) in parameter and return descriptions.
- For derivation steps, reference the step number and source (e.g., "Step 6 per `pywatershed_dataset_param_map.yml`").
- For unit conversions, document both the source and target units and the conversion factor.
- Include Examples when they help a reader understand usage or expected behavior, not just when they're easy to write.
- Omit sections that genuinely don't apply (e.g., no Raises if nothing is raised), but err on the side of including Notes and References.

## Common Tasks

- **Run tests:** `pixi run -e dev test`
- **Type check:** `pixi run -e dev typecheck`
- **Format:** `pixi run -e dev format`
- **Lint:** `pixi run -e dev lint`
- **All checks:** `pixi run -e dev check`
- **Pre-commit:** `pixi run -e dev pre-commit`
- **Install/sync environment:** `pixi install`
- **Build docs:** `pixi run -e docs docs-build`

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

If you modified `pyproject.toml`, also run `pixi install` to regenerate `pixi.lock` — CI uses `locked: true` in setup-pixi and will fail if the lock file is stale.

## What NOT to Do

- Don't put model-specific logic in the pipeline — all model logic belongs in plugins (two-phase boundary)
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

## Open Work

**Current priorities:**
- Processing pathway bifurcation: polygon vs grid targets (gdptools vs xesmf)
- Library-managed transparent data caching
- Dockerfile + Apptainer docs

**Implemented (done):**
- All 14 pywatershed derivation steps: 1 (geometry), 2 (topology), 3 (topo), 4 (landcover), 5 (soils), 6 (waterbody), 8 (lookups), 9 (soltab), 10 (PET), 11 (transp), 12 (routing), 13 (defaults), 14 (calibration seeds)
- All 5 data access strategies (stac_cog, local_tiff, nhgf_stac static/temporal, climr_cat)
- Temporal processing with calendar-year splitting
- CLI: `hydro-param init`, `datasets list/info/download`, `run`, `pywatershed run/validate`

## Data Catalogs

- **USGS GDP STAC:** `https://api.water.usgs.gov/gdp/pygeoapi/stac` — USGS Water Mission Area
  STAC catalog. Hosts gridMET, SNODAS, CONUS404-BA, NLCD Annual, PRISM, and more.
  Accessed via gdptools `NHGFStacData` class.
- **NHGF STAC:** `https://code.usgs.gov/wma/nhgf/stac/-/raw/main/catalog/catalog.json`
- **Planetary Computer:** `https://planetarycomputer.microsoft.com/api/stac/v1` — 3DEP, gNATSGO
- **ClimateR-Catalog:** `https://github.com/mikejohnson51/climateR-catalogs` — gridMET, Daymet,
  1700+ datasets. Accessed via gdptools `ClimRCatData` class.

Note: The gridMET copy on the USGS GDP STAC is not kept up to date. Use `climr_cat`
strategy (ClimRCatData via OPeNDAP) for gridMET, not `nhgf_stac`.

## pywatershed Parameterization Reference

hydro-param's primary target model is pywatershed (USGS NHM-PRMS in Python).
Complete documentation of the dataset→parameter mapping is in:

- `docs/reference/pywatershed_dataset_param_map.yml` — Authoritative reference
  for all ~100+ PRMS parameters, their source datasets, derivation methods,
  lookup tables, default values, calibration seeds, and a 15-step derivation
  pipeline DAG.

- `docs/reference/pywatershed_parameterization_guide.md` — Implementation guide
  covering data.yml catalog design, pywatershed config file structure, output
  plugin API, unit conversions, and process class I/O signatures.

When working on dataset catalogs, parameter derivation, or the pywatershed
output plugin, read these files first.

Key facts:
- pywatershed needs 3 CBH time series (prcp, tmax, tmin) + ~100 static params
- 7 core source datasets: 3DEP DEM, NLCD, STATSGO2, Daymet, GF, NHDPlus, soltab
- Parameters fall into: GIS zonal stats, reclassify, lookup tables, formulas,
  climate-derived, defaults, and calibration seeds
- PRMS internal units: feet, inches, °F, acres
- The derivation pipeline has a clear DAG (15 ordered steps)
- HyRiver packages (pynhd, pygeohydro, pydaymet) are the primary data accessors
