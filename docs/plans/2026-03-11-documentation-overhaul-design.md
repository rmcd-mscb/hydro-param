# Documentation Overhaul Design

**Date:** 2026-03-11
**Issue:** TBD
**Status:** Approved

## Goal

Bring all user-facing documentation up to date with the current codebase (1015
tests, v4.0 pywatershed config, 5 data access strategies, GFv1.1 support, 14
derivation steps). Serve two audiences equally: USGS scientists who want to use
hydro-param and open-source developers who might contribute.

## Approach

Content-first rewrite of narrative documentation. API reference pages
(mkdocstrings auto-generated from docstrings) are not in scope — the gap is in
prose content: README, quickstart, user guides, and workflow documentation.

## Deliverables

| Page | Action | Description |
|------|--------|-------------|
| `README.md` | Rewrite | Fix test count, project structure, add missing CLI commands, update architecture diagram |
| `docs/index.md` | Rewrite | Richer landing page with feature highlights |
| `docs/getting-started/installation.md` | Update | Add pixi environments, fix verify step, note pip not published |
| `docs/getting-started/quickstart.md` | Rewrite | Current config schema, actual output structure, pywatershed cross-link |
| `docs/user-guide/cli.md` | Update | Add `gfv11 download`, update `pywatershed run` for v4.0 config |
| `docs/user-guide/configuration.md` | Rewrite | Current pipeline schema, pywatershed run config, registry overlays |
| `docs/user-guide/datasets.md` | Update | Fix dataset names, add GFv1.1, overlay mechanism |
| `docs/user-guide/pywatershed-workflow.md` | **New** | End-to-end two-phase workflow guide |
| `docs/plans/development-roadmap.md` | **New** | Themed summary of all design/plan docs |
| `docs/contributing.md` | Update | Add paragraph about `docs/plans/` as decision archive |
| `mkdocs.yml` | Update | Add new pages to nav |

## Design Decisions

### 1. Plan docs stay out of nav

The 80+ `docs/plans/` design and implementation documents are internal working
artifacts. They remain in the repo as a decision trail but are not linked in
the mkdocs navigation. A brief mention in `contributing.md` tells contributors
where to find them. The new Development Roadmap page provides a themed executive
summary with links to individual plans.

### 2. Development Roadmap structure (grouped by theme)

The roadmap groups completed and planned work into themes rather than a
chronological list:

1. **Core Pipeline** — 5-stage orchestrator, spatial batching, SIR, incremental writes
2. **Data Access** — 5 strategies, GFv1.1 download, registry overlays
3. **pywatershed Plugin** — plugin architecture, 14 derivation steps, config v4.0, forcing, soltab, routing
4. **Validation & QA** — NHM cross-check, parameter audit, DRB validation
5. **Infrastructure** — CI/CD, pre-commit, Claude Code automations, bundled registry
6. **Open / Planned** — grid pathway, transparent caching, derived-raster (#200), NextGen/PRMS formatters

Each theme gets a narrative paragraph plus a table of completed items (date,
one-liner, link to plan doc). Open items get descriptions with issue links.

### 3. README scope

The README serves as a concise entry point. Detailed "Key Design Decisions"
content moves to `docs/design.md` (where it already lives in full). The README
focuses on: what the tool does, how to install, how to run, project structure,
and related projects.

### 4. New pywatershed Workflow page

The two-phase workflow (generic pipeline → pywatershed derivation) is the
primary use case but has no dedicated user-facing guide. The new page explains
the workflow in user terms: fabric setup, Phase 1 config, Phase 2 config,
output files, and what the 14 derivation steps produce. Links to the parameter
inventory and parameterization guide for deep dives.

### 5. mkdocs nav structure

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

## Out of Scope

- `docs/design.md` rewrite (2300+ lines, separate effort)
- API reference pages (auto-generated from docstrings)
- `docs/reference/` files (pywatershed parameterization guide, parameter
  inventory, pipeline flow, GFv1.1 provenance — these are reference material,
  not user guides)
- Deleting or reorganizing `docs/plans/` files
