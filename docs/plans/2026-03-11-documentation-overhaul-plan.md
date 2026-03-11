# Documentation Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Bring all user-facing documentation up to date with the current codebase and add two new pages (pywatershed workflow guide, development roadmap).

**Architecture:** Content-first rewrite of 11 files. No Python code changes. Each task rewrites or creates one documentation file, verifies the mkdocs build, and commits. The development roadmap task is the largest (synthesizes 38 plan docs into themed summaries).

**Tech Stack:** Markdown, MkDocs Material, pixi

---

### Task 1: Update mkdocs.yml nav and verify build

**Files:**
- Modify: `mkdocs.yml`
- Create: `docs/user-guide/pywatershed-workflow.md` (placeholder)
- Create: `docs/plans/development-roadmap.md` (placeholder)

Update the nav first so all subsequent tasks can verify their pages render.

**Step 1: Update mkdocs.yml nav**

Replace the `nav:` section in `mkdocs.yml` with:

```yaml
nav:
  - Home:
      - index.md
  - Getting Started:
      - getting-started/index.md
      - getting-started/installation.md
      - getting-started/project-init.md
      - getting-started/quickstart.md
  - User Guide:
      - user-guide/index.md
      - user-guide/cli.md
      - user-guide/configuration.md
      - user-guide/pywatershed-workflow.md
      - user-guide/datasets.md
  - API Reference:
      - api/index.md
      - Core:
          - api/config.md
          - api/pipeline.md
          - api/sir.md
          - api/registry.md
          - api/data-access.md
          - api/processing.md
          - api/batching.md
          - api/manifest.md
          - api/units.md
          - api/project.md
      - Plugins:
          - api/plugins.md
          - api/derivations-pywatershed.md
          - api/formatters-pywatershed.md
          - api/pywatershed-config.md
      - Utilities:
          - api/cli.md
          - api/solar.md
  - Architecture:
      - design.md
      - plans/development-roadmap.md
  - Contributing:
      - contributing.md
      - ai-workflow-guide.md
```

**Step 2: Create placeholder files**

Create `docs/user-guide/pywatershed-workflow.md`:
```markdown
# pywatershed Workflow

*This page is under construction.*
```

Create `docs/plans/development-roadmap.md`:
```markdown
# Development Roadmap

*This page is under construction.*
```

**Step 3: Verify mkdocs build**

Run: `pixi run -e docs docs-build`
Expected: Build succeeds with no errors. Warnings about missing cross-references are OK.

**Step 4: Commit**

```bash
git add mkdocs.yml docs/user-guide/pywatershed-workflow.md docs/plans/development-roadmap.md
git commit -m "docs: update mkdocs nav and add placeholder pages"
```

---

### Task 2: Rewrite README.md

**Files:**
- Modify: `README.md`

**Context needed:**
- Current test count: 1015
- CLI commands: `init`, `datasets list/info/download`, `run`, `pywatershed run/validate`, `gfv11 download`
- 5 data access strategies: stac_cog, local_tiff, nhgf_stac (static), nhgf_stac (temporal), climr_cat
- Project structure: `src/hydro_param/` with `derivations/`, `formatters/`, `data/`

**Step 1: Rewrite README.md**

Replace the entire file. The new README should have these sections in order:

1. **Title + one-liner** — `# hydro-param` + tagline
2. **Badges** — CI status, Python version, license (use shields.io badge URLs pointing to `github.com/rmcd-mscb/hydro-param`)
3. **The Problem** — Keep existing text (it's good as-is)
4. **The Approach** — Updated bullet points reflecting current capabilities:
   - Config-driven (YAML → engine → output)
   - Fabric-agnostic (NHM GFv1.1, NextGen, HUC12, any polygon mesh)
   - Five data strategies (STAC COG, NHGF STAC, ClimR OPeNDAP, local GeoTIFF, temporal NHGF STAC)
   - Two-phase architecture (generic pipeline → model plugin)
   - Cloud-native + local (reads from STAC catalogs or local files)
5. **Architecture** — Updated ASCII diagram showing the two-phase flow:
   ```
   YAML Config → Pipeline (5 stages) → SIR → Model Plugin → Model Input
                     │                         │
              ┌──────┼──────┐          ┌───────┼────────┐
              │      │      │          │       │        │
           gdptools xesmf rioxarray  pywatershed NextGen PRMS
          (polygon) (grid) (reproject)
   ```
6. **Quick Start** — Three paths:
   - **Terrain-only** (3 commands: clone, install, run)
   - **Full pywatershed setup** (pipeline → pywatershed run)
   - **GFv1.1 national data layers** (gfv11 download → pipeline)
7. **CLI Commands** — Summary table of all commands with one-line descriptions:
   | Command | Description |
   |---------|-------------|
   | `hydro-param init` | Scaffold a new project |
   | `hydro-param datasets list` | Show available datasets |
   | `hydro-param datasets info NAME` | Dataset details |
   | `hydro-param datasets download NAME` | Download dataset files |
   | `hydro-param run CONFIG` | Run the generic pipeline |
   | `hydro-param pywatershed run CONFIG` | Generate pywatershed model setup |
   | `hydro-param pywatershed validate FILE` | Validate parameter file |
   | `hydro-param gfv11 download` | Download GFv1.1 national rasters |
8. **Project Structure** — Updated tree showing current modules:
   ```
   hydro-param/
   ├── src/hydro_param/
   │   ├── cli.py                 # CLI (cyclopts)
   │   ├── config.py              # Pipeline config (Pydantic)
   │   ├── pywatershed_config.py  # pywatershed config (Pydantic)
   │   ├── pipeline.py            # 5-stage orchestrator
   │   ├── processing.py          # gdptools wrapper
   │   ├── data_access.py         # STAC/local data fetch
   │   ├── batching.py            # KD-tree spatial batching
   │   ├── sir_accessor.py        # SIR lazy loader
   │   ├── classification.py      # USDA texture triangle
   │   ├── derivations/           # Model-specific derivations
   │   │   └── pywatershed.py
   │   ├── formatters/            # Output formatters
   │   │   └── pywatershed.py
   │   └── data/                  # Bundled registry + lookup tables
   │       ├── datasets/          # 8 dataset registry YAMLs
   │       └── pywatershed/       # Parameter metadata + tables
   ├── configs/examples/          # Example pipeline configs
   ├── tests/                     # 1015 tests
   ├── docs/                      # MkDocs documentation
   └── pyproject.toml             # Package + pixi config
   ```
9. **Documentation** — Link to docs site: `https://rmcd-mscb.github.io/hydro-param/`
10. **Related Projects** — Keep existing table
11. **License + Contact** — Keep existing

**Step 2: Verify render**

Run: `pixi run -e docs docs-build`
Expected: Builds successfully.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with current project state"
```

---

### Task 3: Rewrite docs/index.md (MkDocs landing page)

**Files:**
- Modify: `docs/index.md`

**Step 1: Rewrite docs/index.md**

The landing page should be richer than the README but not duplicate it. Structure:

1. **Title + tagline** — Same as README
2. **Pre-alpha warning** — Keep the admonition
3. **What it does** — 2-3 sentences: config-driven parameterization, SIR output, model plugins
4. **Key features** — Bulleted list with brief explanations:
   - Config-driven YAML pipelines
   - Five data access strategies (with dataset examples)
   - Two-phase architecture (pipeline → plugin)
   - Fabric-agnostic (any polygon mesh)
   - Incremental processing with resume
   - 1015 tests
5. **Two-Phase Architecture** — Brief explanation with a simple diagram or callout box:
   - Phase 1: Generic pipeline produces Standardized Internal Representation (SIR)
   - Phase 2: Model plugin (e.g., pywatershed) derives model-specific parameters
6. **Quick Links** — Updated link list:
   - [Installation](getting-started/installation.md)
   - [Quick Start](getting-started/quickstart.md)
   - [pywatershed Workflow](user-guide/pywatershed-workflow.md)
   - [CLI Reference](user-guide/cli.md)
   - [API Reference](api/index.md)
   - [Architecture](design.md)
   - [Development Roadmap](plans/development-roadmap.md)

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/index.md
git commit -m "docs: rewrite landing page with feature highlights"
```

---

### Task 4: Update docs/getting-started/installation.md

**Files:**
- Modify: `docs/getting-started/installation.md`

**Step 1: Update installation.md**

Changes to make:
- After "pixi install", add a note about the three environments:
  ```
  pixi creates three environments:
  - `default` — core dependencies
  - `dev` — development (pytest, ruff, mypy, pre-commit)
  - `docs` — documentation (mkdocs-material)
  ```
- Update verify step to: `pixi run -e dev -- hydro-param --help`
- Add a note that `pip install hydro-param` is not yet published to PyPI (pre-alpha):
  ```
  !!! warning
      hydro-param is not yet published to PyPI. Install from source using pixi (above)
      or `pip install -e .` in a development environment with GDAL/GEOS/PROJ.
  ```
- Add "Run tests" verification: `pixi run -e dev test`

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/getting-started/installation.md
git commit -m "docs: update installation with pixi environments and verify steps"
```

---

### Task 5: Rewrite docs/getting-started/quickstart.md

**Files:**
- Modify: `docs/getting-started/quickstart.md`

**Context needed:**
- Read `configs/examples/drb_2yr_pipeline.yml` for a real working config example
- Current output structure: `output/<category>/<variable>.csv` + `.manifest.yml`
- No `sir/` subdirectory exists
- Config schema: `target_fabric`, `domain`, `datasets` (list), `output`, `processing`

**Step 1: Rewrite quickstart.md**

Structure:

1. **Title** — "Quick Start"
2. **Intro** — "A minimal end-to-end example using the Delaware River Basin."
3. **Prerequisites** — pixi installed, hydro-param cloned
4. **Step 1: Initialize** — `hydro-param init delaware-demo && cd delaware-demo`
5. **Step 2: Obtain a fabric** — Keep existing text about pynhd/pygeohydro/USGS GF. Copy to `data/fabrics/`.
6. **Step 3: Explore datasets** — `hydro-param datasets list` + `hydro-param datasets info dem_3dep_10m`
7. **Step 4: Configure the pipeline** — Show a minimal but accurate config based on `drb_2yr_pipeline.yml` schema. Use current field names (`target_fabric.path`, `target_fabric.id_field`, `target_fabric.crs`, `domain.type`, `domain.bbox`, `datasets` as a list with `name`, `variables`, `statistics`, `output.path`, `output.format`, `processing.batch_size`). Include only 1-2 datasets to keep it minimal (DEM + NLCD).
8. **Step 5: Run** — `hydro-param run configs/pipeline.yml`
9. **Step 6: Inspect results** — Describe actual output:
   - `output/topography/dem_3dep_10m_elevation_mean.csv`
   - `output/land_cover/nlcd_osn_lndcov_LndCov_categorical_2021.csv`
   - `output/.manifest.yml`
10. **Next steps** — Link to:
    - [Configuration](../user-guide/configuration.md) for full config reference
    - [pywatershed Workflow](../user-guide/pywatershed-workflow.md) for model-specific output
    - [Datasets](../user-guide/datasets.md) for the full dataset catalog

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/getting-started/quickstart.md
git commit -m "docs: rewrite quickstart with current config schema and output structure"
```

---

### Task 6: Update docs/user-guide/cli.md

**Files:**
- Modify: `docs/user-guide/cli.md`

**Context needed:**
- Full CLI help output from all 11 commands (gathered above)

**Step 1: Update cli.md**

Rewrite to match actual CLI output. Structure:

1. **Intro** — hydro-param CLI overview
2. **Top-level commands** — Table from `hydro-param --help`
3. **Project Management**
   - `hydro-param init` — copy from `--help` output, add parameter table
4. **Dataset Discovery**
   - `hydro-param datasets list` — parameters, example output
   - `hydro-param datasets info NAME` — parameters
   - `hydro-param datasets download NAME` — parameters, note about AWS CLI
5. **Pipeline Execution**
   - `hydro-param run CONFIG` — parameters including `--resume`, describe 5-stage pipeline briefly
6. **pywatershed Model Setup**
   - `hydro-param pywatershed run CONFIG` — parameters, output files table, two-phase explanation
   - `hydro-param pywatershed validate PARAM-FILE` — parameters
7. **GFv1.1 Data Layers** (NEW section)
   - `hydro-param gfv11 download` — parameters including `--output-dir` and `--items`

For each command, include:
- Usage line
- Description (from help text)
- Parameter table
- Example invocation where helpful

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/user-guide/cli.md
git commit -m "docs: update CLI reference with all commands including gfv11"
```

---

### Task 7: Rewrite docs/user-guide/configuration.md

**Files:**
- Modify: `docs/user-guide/configuration.md`

**Context needed:**
- `src/hydro_param/config.py` — PipelineConfig Pydantic models
- `src/hydro_param/pywatershed_config.py` — PywatershedRunConfig Pydantic models
- `configs/examples/drb_2yr_pipeline.yml` — real Phase 1 example
- `configs/examples/drb_2yr_pywatershed.yml` — real Phase 2 example

**Step 1: Rewrite configuration.md**

Structure:

1. **Intro** — Two config files: pipeline config (Phase 1) and pywatershed run config (Phase 2)
2. **Pipeline Config (Phase 1)** — `pipeline.yml`
   - `target_fabric` — path, id_field, crs
   - `domain` — type (bbox/huc2/huc4/gage), bbox or id
   - `datasets` — list of DatasetRequest (name, source, variables, statistics, year, time_period)
   - `output` — path, format (netcdf/parquet), sir_name
   - `processing` — batch_size, resume, sir_validation, network_timeout
   - Include a complete annotated example based on `drb_2yr_pipeline.yml`
3. **pywatershed Run Config (Phase 2)** — `pywatershed_run.yml`
   - `target_model`, `version`
   - `sir_path` — path to Phase 1 output
   - `domain` — fabric_path, segment_path, waterbody_path, id_field, segment_id_field
   - `time` — start, end, timestep
   - `static_datasets` — topography, soils, landcover, snow, waterbodies (each with ParameterEntry fields)
   - `forcing` — prcp, tmax, tmin
   - `climate_normals` — jh_coef, transp_beg, transp_end
   - `parameter_overrides`, `calibration`, `output`
   - Include a complete annotated example based on `drb_2yr_pywatershed.yml`
4. **Dataset Registry** — Brief explanation of the bundled registry + user-local overlay mechanism (`~/.hydro-param/datasets/`). Link to datasets page for details.
5. **Config Reference** — Links to API docs for PipelineConfig and PywatershedRunConfig

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/user-guide/configuration.md
git commit -m "docs: rewrite configuration guide with Phase 1 and Phase 2 configs"
```

---

### Task 8: Update docs/user-guide/datasets.md

**Files:**
- Modify: `docs/user-guide/datasets.md`

**Context needed:**
- Bundled registry categories and dataset names
- GFv1.1 overlay mechanism
- `hydro-param datasets list` actual output

**Step 1: Update datasets.md**

Changes:
- Fix category table: replace "gSSURGO" with "gNATSGO" and "POLARIS"
- Add GFv1.1 datasets section explaining:
  - `hydro-param gfv11 download` fetches national rasters
  - Download auto-registers a user-local overlay at `~/.hydro-param/datasets/gfv11.yml`
  - GFv1.1 datasets then appear in `datasets list` and can be referenced in pipeline configs
- Update the download example to show `gfv11 download` instead of generic `datasets download`
- Add a note about the five data access strategies and which datasets use which:
  | Strategy | Datasets |
  |----------|----------|
  | `stac_cog` | 3DEP 10m DEM, gNATSGO |
  | `local_tiff` | POLARIS 30m, GFv1.1 rasters |
  | `nhgf_stac` (static) | NLCD Annual (6 collections) |
  | `nhgf_stac` (temporal) | SNODAS, CONUS404-BA |
  | `climr_cat` | gridMET (OPeNDAP) |
- Link to API reference for DatasetEntry schema

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/user-guide/datasets.md
git commit -m "docs: update datasets guide with GFv1.1, access strategies, overlay mechanism"
```

---

### Task 9: Create docs/user-guide/pywatershed-workflow.md

**Files:**
- Modify: `docs/user-guide/pywatershed-workflow.md` (replace placeholder)

**Context needed:**
- Two-phase architecture: Phase 1 (pipeline) → SIR → Phase 2 (derivation)
- `configs/examples/drb_2yr_pipeline.yml` and `drb_2yr_pywatershed.yml`
- 14 derivation steps (from CLAUDE.md)
- Output files: parameters.nc, forcing/*.nc, soltab.nc, control.yml

**Step 1: Write pywatershed-workflow.md**

This is the primary new page. Structure:

1. **Intro** — hydro-param's primary target model is pywatershed (USGS NHM-PRMS). This guide walks through the end-to-end workflow.
2. **Overview** — Two-phase architecture explained in user terms:
   - Phase 1: Generic pipeline fetches and processes geospatial datasets → produces SIR
   - Phase 2: pywatershed plugin reads SIR + config → derives ~100 PRMS parameters
   - Diagram showing the flow
3. **Prerequisites**
   - Target fabric (HRU polygons + segment flowlines as GeoPackage)
   - Waterbody polygons (optional, for lake/depression parameters)
   - Local data (GFv1.1 rasters if using NHM data layers)
4. **Step 1: Run Phase 1 (Generic Pipeline)**
   - Show the pipeline config (annotated `drb_2yr_pipeline.yml`)
   - `hydro-param run configs/pipeline.yml`
   - What it produces: SIR files in `output/<category>/`
5. **Step 2: Run Phase 2 (pywatershed Derivation)**
   - Show the pywatershed run config (annotated `drb_2yr_pywatershed.yml`)
   - `hydro-param pywatershed run configs/pywatershed_run.yml`
   - What it produces: parameters.nc, forcing/, soltab.nc, control.yml
6. **What the Derivation Steps Produce** — High-level summary table:
   | Step | Category | Parameters | Source |
   |------|----------|-----------|--------|
   | 1 | Geometry | hru_area, hru_lat, hru_lon | Fabric geometry |
   | 2 | Topology | hru_segment, tosegment_nhm, etc. | Fabric + NHDPlus |
   | 3 | Topography | hru_elev, hru_slope | 3DEP DEM |
   | 4 | Land cover | cov_type, covden_sum/win, interception | NLCD / GFv1.1 |
   | 5 | Soils | soil_type, soil_moist_max, soil_rechr_max | POLARIS / gNATSGO |
   | 6 | Waterbodies | hru_type, dprst_frac | NHDPlus waterbodies |
   | 7 | Forcing | prcp, tmax, tmin (daily time series) | gridMET |
   | 8 | Lookup tables | hru_deplcrv, snarea_curve, etc. | PRMS lookup tables |
   | 9 | Solar tables | soltab_potsw, soltab_horad | Computed from lat/slope |
   | 10 | PET | jh_coef (monthly) | Climate normals |
   | 11 | Transpiration | transp_beg, transp_end | Climate normals |
   | 12 | Routing | K_coef, x_coef | Segment geometry |
   | 13 | Defaults | tmax_allsnow, dday_slope, etc. | PRMS defaults |
   | 14 | Calibration seeds | snarea_thresh, etc. | Derived from other params |
7. **Step 3: Validate Output**
   - `hydro-param pywatershed validate models/pywatershed/parameters.nc`
8. **Output File Reference** — Table of output files and their contents
9. **Next Steps** — Links to:
   - [Configuration](configuration.md) for config details
   - [Parameter Inventory](../reference/parameter_inventory.md) for the full parameter list
   - [Parameterization Guide](../reference/pywatershed_parameterization_guide.md) for derivation details

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/user-guide/pywatershed-workflow.md
git commit -m "docs: add pywatershed two-phase workflow guide"
```

---

### Task 10: Create docs/plans/development-roadmap.md

**Files:**
- Modify: `docs/plans/development-roadmap.md` (replace placeholder)

**Context needed:**
- The 38-entry plan catalog (gathered above)
- Open issues from CLAUDE.md: #73, #92, #100, #154, #200
- Merged PRs referenced in MEMORY.md

**Step 1: Write development-roadmap.md**

This is the largest task. Structure:

1. **Title + Intro** — "Development Roadmap" + brief explanation: this page summarizes the design decisions and implementation work that shaped hydro-param. Each theme groups related work with links to the full design documents in `docs/plans/`.

2. **Core Pipeline** — Narrative paragraph (3-5 sentences) about how the 5-stage pipeline, SIR normalization, spatial batching, and incremental processing came together. Then a table:
   | Date | Design | Summary |
   |------|--------|---------|
   | 2026-02-23 | [SIR Normalization](2026-02-23-sir-normalization-design.md) | Standardized variable names and units at pipeline output boundary |
   | 2026-02-23 | [Memory Optimization](2026-02-23-pipeline-memory-optimization-design.md) | STAC query reuse and memory-efficient batch processing |
   | 2026-02-24 | [Temporal SIR](2026-02-24-sir-temporal-normalization-design.md) | Extended SIR normalization to temporal (multi-year) datasets |
   | 2026-02-24 | [Pipeline Resilience](2026-02-24-pipeline-resilience-optimization-plan.md) | Manifest-based resume, pre-fetch, network timeout |
   | 2026-02-28 | [Variable Naming](2026-02-28-sir-variable-naming-fix.md) | Year-suffixed SIR variable name resolution |
   | 2026-02-28 | [Dataset Prefix](2026-02-28-sir-dataset-prefix-design.md) | Dataset name prefix in SIR filenames for disambiguation |
   | 2026-03-02 | [Shared Classification](2026-03-02-shared-classification-design.md) | USDA texture triangle as shared module |
   | 2026-03-02 | [Derived Categorical](2026-03-02-derived-categorical-design.md) | Pixel-level multi-source classification before zonal stats |

3. **pywatershed Plugin** — Narrative about the plugin architecture, two-phase separation, and the 14-step derivation pipeline. Table of 13 entries covering plugin architecture, derivation steps, config redesign, forcing, decouple, compat, soil texture, soil_rechr_max_frac.

4. **Data Access** — Narrative about the five strategies, GFv1.1 integration, and user-local registry overlays. Table of 3 entries.

5. **Validation & QA** — Narrative about parameter auditing and cross-checking against NHM reference parameters. Table of 5 entries.

6. **Infrastructure** — Narrative about UX audit, bundled registry, config schema, stale code cleanup, themed datasets. Table of 6 entries.

7. **Open / Planned** — Current priorities and open issues:
   - Grid processing pathway (polygon targets use gdptools; grid targets planned with xesmf)
   - Transparent data caching (pooch-style, library-managed)
   - Derived-raster pathway (#200 — pixel-level raster math before zonal stats)
   - PRMS legacy formatter (#92 — pyPRMS-based)
   - NextGen hydrofabric slopes (#100)
   - Subsurface flux rescaling (#154 — needs GLHYMPS data source)
   - Nearest-neighbor gap-fill (#73 — temporal SIR features missing grid coverage)
   - DerivedContinuousSpec (2026-03-11 — design complete, implementation pending)

Each table row links to the full plan doc. Keep summaries to one sentence each.

**Step 2: Verify build**

Run: `pixi run -e docs docs-build`

**Step 3: Commit**

```bash
git add docs/plans/development-roadmap.md
git commit -m "docs: add development roadmap with themed plan summaries"
```

---

### Task 11: Update docs/contributing.md and final verification

**Files:**
- Modify: `docs/contributing.md`

**Step 1: Update contributing.md**

Add a section after the existing content:

```markdown
## Design Documents

Design decisions and implementation plans are archived in `docs/plans/`,
organized by date. Each significant feature or change starts with a design
document that captures the problem, proposed approaches, and the chosen
solution.

For a themed summary of all design work to date, see the
[Development Roadmap](../plans/development-roadmap.md).
```

**Step 2: Full build verification**

Run: `pixi run -e docs docs-build`
Expected: Clean build, all pages render, no broken links within the nav.

**Step 3: Spot-check with docs-serve**

Run: `pixi run -e docs docs-serve` (manually browse key pages)
Check: Home, Quick Start, pywatershed Workflow, CLI, Configuration, Roadmap

**Step 4: Commit**

```bash
git add docs/contributing.md
git commit -m "docs: add design documents section to contributing guide"
```

---

## Task Dependency Graph

```
Task 1 (nav + placeholders)
  ├── Task 2 (README)
  ├── Task 3 (index.md)
  ├── Task 4 (installation)
  ├── Task 5 (quickstart)
  ├── Task 6 (CLI)
  ├── Task 7 (configuration)
  ├── Task 8 (datasets)
  ├── Task 9 (pywatershed workflow)
  ├── Task 10 (roadmap)
  └── Task 11 (contributing + final verify)
```

Task 1 must be done first (nav setup). Tasks 2–10 are independent and can be
done in any order. Task 11 should be last (final verification).

## Verification Checklist

After all tasks are complete:

- [ ] `pixi run -e docs docs-build` succeeds with no errors
- [ ] All 11 nav pages render correctly
- [ ] No broken internal links
- [ ] README reflects current state (1015 tests, all CLI commands, current project structure)
- [ ] Quick start uses current config schema and output structure
- [ ] pywatershed workflow page covers the full two-phase flow
- [ ] Development roadmap links to all 38 plan docs
- [ ] CLI reference includes `gfv11 download`
- [ ] Configuration guide covers both Phase 1 and Phase 2 configs
