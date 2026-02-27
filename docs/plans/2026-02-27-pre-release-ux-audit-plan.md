# Pre-release UX Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 15 gaps from the UX audit so the full pywatershed workflow (init → pipeline → derivation) works end-to-end without silent failures or misleading docs.

**Architecture:** Code fixes in `cli.py`, `config.py`, `project.py`, `pywatershed_config.py`. Doc fixes in `README.md`, `docs/`. Single issue + branch. TDD for code changes.

**Tech Stack:** Python, Pydantic, cyclopts, pytest, MkDocs

---

### Task 1: Create GitHub issue and feature branch

**Step 1: Create the issue**

```bash
gh issue create \
  --title "fix: pre-release UX audit — 15 gaps in pywatershed workflow" \
  --body "Fixes 15 gaps identified in end-to-end user workflow review.

## Code fixes
- Gap 3: Soils dataset not fetched in _translate_pws_to_pipeline
- Gap 4: daymet_v4/conus404_ba not validated (not yet supported)
- Gap 5: Land cover year hardcoded to 2021
- Gap 6: failure_mode: tolerant is a no-op (remove until implemented)
- Gap 11: Unclosed xarray datasets in pws_run_cmd
- Gap 14: validate command missing logging setup

## Doc fixes
- Gap 1: README stale ('Design & planning phase')
- Gap 2: CLI docs missing pywatershed commands
- Gap 7: Output structure undocumented
- Gap 8: Quickstart missing fabric prerequisite
- Gap 9: Nonexistent pip extras in install docs
- Gap 10: Translated config not visible for debugging
- Gap 12: Silent soltab skip
- Gap 13: Confusing download status labels
- Gap 15: Example configs reference non-shipped fabric

Design: docs/plans/2026-02-27-pre-release-ux-audit-design.md"
```

**Step 2: Create the feature branch**

```bash
git checkout -b fix/<issue-number>-pre-release-ux-audit
```

---

### Task 2: Gap 3 — Add soils dataset to `_translate_pws_to_pipeline`

**Files:**
- Modify: `tests/test_pipeline_derivation.py` (TestPwsConfigTranslation)
- Modify: `src/hydro_param/cli.py:644-716` (_translate_pws_to_pipeline)

**Step 1: Write the failing test**

Add to `tests/test_pipeline_derivation.py` class `TestPwsConfigTranslation`:

```python
def test_translate_includes_soils(self) -> None:
    """Soils dataset is included in translated pipeline config."""
    from hydro_param.cli import _translate_pws_to_pipeline
    from hydro_param.pywatershed_config import PywatershedRunConfig

    pws_config = PywatershedRunConfig(
        domain={
            "source": "custom",
            "extraction_method": "bbox",
            "bbox": [-75.8, 39.6, -74.4, 42.5],
            "fabric_path": "data/nhru.gpkg",
        },
        time={"start": "2020-01-01", "end": "2021-12-31"},
        climate={"source": "gridmet", "variables": ["prcp", "tmax", "tmin"]},
        datasets={"topography": "dem_3dep_10m", "landcover": "nlcd_osn_lndcov", "soils": "polaris_30m"},
    )

    pipeline_config = _translate_pws_to_pipeline(pws_config)
    ds_names = [d.name for d in pipeline_config.datasets]
    assert "polaris_30m" in ds_names
    assert len(pipeline_config.datasets) == 4  # topo + landcover + soils + climate

    soils_ds = next(d for d in pipeline_config.datasets if d.name == "polaris_30m")
    assert "sand" in soils_ds.variables
    assert "clay" in soils_ds.variables
    assert "silt" in soils_ds.variables
    assert soils_ds.statistics == ["mean"]
```

**Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_pipeline_derivation.py::TestPwsConfigTranslation::test_translate_includes_soils -v
```

Expected: FAIL — soils not in datasets (len == 3, not 4).

**Step 3: Implement the fix**

In `src/hydro_param/cli.py`, after the land cover block (~line 674) and before the climate block (~line 676), add:

```python
# Soils
datasets.append(
    DatasetRequest(
        name=cfg.datasets.soils,
        variables=["sand", "silt", "clay", "ksat", "theta_s", "bd"],
        statistics=["mean"],
    )
)
```

**Step 4: Update the existing test assertion**

In `test_translate_basic` (line 83), update expected dataset count from 3 to 4:

```python
assert len(pipeline_config.datasets) == 4  # topo + landcover + soils + climate
```

**Step 5: Run tests to verify**

```bash
pixi run -e dev pytest tests/test_pipeline_derivation.py::TestPwsConfigTranslation -v
```

Expected: ALL PASS.

**Step 6: Commit**

```bash
git add tests/test_pipeline_derivation.py src/hydro_param/cli.py
git commit -m "fix: add soils dataset to pywatershed pipeline translation (gap 3)"
```

---

### Task 3: Gap 4 — Validate unsupported climate sources

**Files:**
- Modify: `tests/test_pipeline_derivation.py`
- Modify: `src/hydro_param/cli.py:676-697`
- Modify: `src/hydro_param/pywatershed_config.py:128-156`

**Step 1: Write the failing test**

Add to `TestPwsConfigTranslation`:

```python
def test_translate_unsupported_climate_raises(self) -> None:
    """Unsupported climate sources raise ValueError with helpful message."""
    from hydro_param.cli import _translate_pws_to_pipeline
    from hydro_param.pywatershed_config import PywatershedRunConfig

    pws_config = PywatershedRunConfig(
        domain={
            "source": "custom",
            "extraction_method": "bbox",
            "bbox": [-75.8, 39.6, -74.4, 42.5],
            "fabric_path": "data/nhru.gpkg",
        },
        time={"start": "2020-01-01", "end": "2021-12-31"},
        climate={"source": "daymet_v4"},
    )

    with pytest.raises(ValueError, match="not yet supported"):
        _translate_pws_to_pipeline(pws_config)
```

**Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_pipeline_derivation.py::TestPwsConfigTranslation::test_translate_unsupported_climate_raises -v
```

Expected: FAIL — no ValueError raised.

**Step 3: Implement the fix**

In `src/hydro_param/cli.py`, replace the `_CLIMATE_SOURCE_MAP` block (~lines 677–682) with:

```python
# Climate (temporal) — validate source and map variable names
_SUPPORTED_CLIMATE_SOURCES = {"gridmet"}
if cfg.climate.source not in _SUPPORTED_CLIMATE_SOURCES:
    raise ValueError(
        f"Climate source '{cfg.climate.source}' is not yet supported. "
        f"Available sources: {', '.join(sorted(_SUPPORTED_CLIMATE_SOURCES))}"
    )
climate_ds_name = cfg.climate.source
```

Also update `PwsClimateConfig` in `pywatershed_config.py` to accept any string instead of a strict Literal, so validation happens at translation time with a clear message rather than at config load time with a cryptic Pydantic error. Change line 154:

```python
source: str = "gridmet"
```

**Step 4: Run tests**

```bash
pixi run -e dev pytest tests/test_pipeline_derivation.py::TestPwsConfigTranslation -v
```

Expected: ALL PASS.

**Step 5: Commit**

```bash
git add tests/test_pipeline_derivation.py src/hydro_param/cli.py src/hydro_param/pywatershed_config.py
git commit -m "fix: validate unsupported climate sources with helpful error (gap 4)"
```

---

### Task 4: Gap 5 — Use time period end year for land cover

**Files:**
- Modify: `tests/test_pipeline_derivation.py`
- Modify: `src/hydro_param/cli.py:656-674`

**Step 1: Write the failing test**

Add to `TestPwsConfigTranslation`:

```python
def test_translate_landcover_year_from_time_period(self) -> None:
    """Land cover year uses end year of time period, not hardcoded 2021."""
    from hydro_param.cli import _translate_pws_to_pipeline
    from hydro_param.pywatershed_config import PywatershedRunConfig

    pws_config = PywatershedRunConfig(
        domain={
            "source": "custom",
            "extraction_method": "bbox",
            "bbox": [-75.8, 39.6, -74.4, 42.5],
            "fabric_path": "data/nhru.gpkg",
        },
        time={"start": "2018-01-01", "end": "2019-12-31"},
        climate={"source": "gridmet"},
        datasets={"landcover": "nlcd_osn_lndcov"},
    )

    pipeline_config = _translate_pws_to_pipeline(pws_config)
    nlcd_ds = next(d for d in pipeline_config.datasets if d.name == "nlcd_osn_lndcov")
    assert nlcd_ds.year == 2019  # end year, not 2021
```

**Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_pipeline_derivation.py::TestPwsConfigTranslation::test_translate_landcover_year_from_time_period -v
```

Expected: FAIL — year is 2021.

**Step 3: Implement the fix**

In `src/hydro_param/cli.py`, replace the hardcoded `year=2021` (~line 664) with:

```python
from datetime import date as _date

# Use end year of simulation period, clamped to NLCD availability (1985–2024)
_end_year = _date.fromisoformat(cfg.time.end).year
_nlcd_year = min(_end_year, 2024)
```

Then in the DatasetRequest: `year=_nlcd_year`.

Move the `from datetime import date as _date` to the top of the function or use the module-level import if one exists.

**Step 4: Update `test_translate_basic`**

The existing test uses time period 2020–2021, so `nlcd_ds.year` should still be 2021 (end year). The assertion `assert nlcd_ds.year == 2021` at line 94 should remain correct. Verify.

**Step 5: Run all translation tests**

```bash
pixi run -e dev pytest tests/test_pipeline_derivation.py::TestPwsConfigTranslation -v
```

Expected: ALL PASS.

**Step 6: Commit**

```bash
git add tests/test_pipeline_derivation.py src/hydro_param/cli.py
git commit -m "fix: derive land cover year from time period end year (gap 5)"
```

---

### Task 5: Gap 6 — Remove non-functional `failure_mode` config option

**Files:**
- Modify: `src/hydro_param/config.py:248-250`
- Modify: `src/hydro_param/project.py:207-208` (pipeline template)
- Modify: `tests/` — search for any tests referencing `failure_mode`

**Step 1: Check for existing tests**

```bash
pixi run -e dev pytest -k "failure_mode" --collect-only 2>/dev/null || echo "no tests"
```

Also grep:

```bash
grep -rn "failure_mode" tests/ src/
```

**Step 2: Remove from config.py**

In `src/hydro_param/config.py`, delete lines 249-250 (the TODO comment and the `failure_mode` field):

```python
# TODO: Wire failure_mode into stage4 error handling (continue-on-failure with logging)
failure_mode: Literal["strict", "tolerant"] = "strict"
```

Also remove the Notes section in the docstring (~lines 244-245) referencing it.

**Step 3: Remove from pipeline template**

In `src/hydro_param/project.py`, delete line 208 from the template string:

```
  failure_mode: strict                    # strict (fail fast) or tolerant (log and continue)
```

**Step 4: Fix any tests that reference `failure_mode`**

Update or remove any assertions about `failure_mode`.

**Step 5: Run full test suite**

```bash
pixi run -e dev pytest -x -q
```

Expected: ALL PASS.

**Step 6: Commit**

```bash
git add src/hydro_param/config.py src/hydro_param/project.py
git commit -m "fix: remove non-functional failure_mode config option (gap 6)"
```

---

### Task 6: Gap 11 — Close temporal xarray datasets + Gap 14 — Add logging to validate

**Files:**
- Modify: `src/hydro_param/cli.py:834` (temporal close)
- Modify: `src/hydro_param/cli.py:857-896` (validate logging)

**Step 1: Fix unclosed datasets (gap 11)**

In `src/hydro_param/cli.py` `pws_run_cmd()`, replace line 834:

```python
temporal = {name: xr.open_dataset(path) for name, path in result.temporal_files.items()}
```

With a try/finally pattern:

```python
temporal = {name: xr.open_dataset(path) for name, path in result.temporal_files.items()}
try:
    derived = merge_temporal_into_derived(
        derived,
        temporal,
        renames={"pr": "prcp", "tmmx": "tmax", "tmmn": "tmin"},
        conversions={"tmax": ("K", "C"), "tmin": ("K", "C")},
    )
finally:
    for ds in temporal.values():
        ds.close()
```

And remove the original `derived = merge_temporal_into_derived(...)` call (lines 835-840) since it's now inside the try block.

**Step 2: Fix missing logging in validate (gap 14)**

In `src/hydro_param/cli.py` `pws_validate_cmd()`, add at the start of the function body (after the docstring, before `import xarray`):

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
```

**Step 3: Run tests**

```bash
pixi run -e dev pytest tests/test_pipeline_derivation.py tests/test_cli.py -v -q
```

Expected: ALL PASS.

**Step 4: Commit**

```bash
git add src/hydro_param/cli.py
git commit -m "fix: close temporal datasets after merge, add logging to validate (gaps 11, 14)"
```

---

### Task 7: Gap 10 — Log translated config at DEBUG level

**Files:**
- Modify: `src/hydro_param/cli.py:798` (after `_translate_pws_to_pipeline`)

**Step 1: Add debug logging**

After line 798 (`pipeline_config = _translate_pws_to_pipeline(pws_config)`), add:

```python
logger.debug("Translated pipeline config: %s", pipeline_config.model_dump_json(indent=2))
```

**Step 2: Commit**

```bash
git add src/hydro_param/cli.py
git commit -m "fix: log translated pipeline config at DEBUG level (gap 10)"
```

---

### Task 8: Gap 12 — Log when soltab.nc is not produced

**Files:**
- Modify: `src/hydro_param/cli.py` (in `pws_run_cmd`, near the formatter.write call)

**Step 1: Add info log**

After `formatter.write(derived, ...)` (line 852), add:

```python
soltab_path = Path(pws_config.output.path) / pws_config.output.soltab_file
if not soltab_path.exists():
    logger.info(
        "soltab.nc was not produced (missing elevation/slope/aspect in SIR). "
        "Solar radiation tables will not be available."
    )
```

**Step 2: Commit**

```bash
git add src/hydro_param/cli.py
git commit -m "fix: log when soltab.nc is not produced (gap 12)"
```

---

### Task 9: Gap 13 — Improve dataset list status labels

**Files:**
- Modify: `src/hydro_param/cli.py:84-116` (_access_status)
- Modify: `tests/` — check for tests on `_access_status`

**Step 1: Check for existing tests**

```bash
grep -rn "_access_status" tests/
```

**Step 2: Update labels**

In `_access_status()`, change the remote return value at line 113:

```python
if entry.strategy in ("stac_cog", "native_zarr", "climr_cat", "nhgf_stac"):
    return "remote (no download needed)"
```

**Step 3: Fix any tests that assert on the old label**

**Step 4: Run tests**

```bash
pixi run -e dev pytest -x -q
```

**Step 5: Commit**

```bash
git add src/hydro_param/cli.py
git commit -m "fix: clarify remote dataset status labels (gap 13)"
```

---

### Task 10: Gap 15 — Add comment to example configs about fabric

**Files:**
- Modify: `configs/examples/drb_2yr_pywatershed.yml`
- Modify: `configs/examples/drb_2yr_pipeline.yml`

**Step 1: Add comment**

At the top of each example config, after the existing header comments, add:

```yaml
# NOTE: This config references DRB test fabric files that are not shipped
# with the repository. Obtain fabric files via pynhd/pygeohydro or from
# the USGS Geospatial Fabric before running.
```

**Step 2: Commit**

```bash
git add configs/examples/
git commit -m "docs: note fabric prerequisite in example configs (gap 15)"
```

---

### Task 11: Gap 1 — Update stale README

**Files:**
- Modify: `README.md`

**Step 1: Rewrite README.md**

Replace the status line and project structure with current state. Key changes:

- Status: "Pre-alpha MVP" with 635 tests passing
- Remove "coming soon" from project structure
- Update project structure to match actual `src/hydro_param/` layout
- Update "Planned Architecture" to "Architecture" (it works now)
- Remove references to Hilbert curve (we use KD-tree)
- Remove reference to `hydro-param-data` separate package
- Remove "Scalable" Dask/Coiled bullet (not implemented)
- Add Quick Start section with actual commands

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README to reflect pre-alpha MVP status (gap 1)"
```

---

### Task 12: Gap 2 — Add pywatershed commands to CLI docs

**Files:**
- Modify: `docs/user-guide/cli.md`

**Step 1: Add sections**

After the `hydro-param run` section, add:

```markdown
### `hydro-param pywatershed run`

Generate a complete pywatershed model setup (parameters, forcing, control).

\`\`\`bash
hydro-param pywatershed run CONFIG [--registry PATH]
\`\`\`

Runs a two-phase workflow:

1. **Phase 1 (generic pipeline):** Fetch and process source datasets (topography, land cover, soils, climate) via the 5-stage pipeline.
2. **Phase 2 (pywatershed derivation):** Derive PRMS parameters from the raw SIR, merge temporal climate data with unit conversions, and write output files.

**Output files:**

| File | Description |
|------|-------------|
| `parameters.nc` | Static PRMS parameters (CF-1.8 NetCDF) |
| `forcing/prcp.nc` | Precipitation forcing (inches/day) |
| `forcing/tmax.nc` | Maximum temperature forcing (°F) |
| `forcing/tmin.nc` | Minimum temperature forcing (°F) |
| `soltab.nc` | Potential solar radiation tables (nhru × 366) |
| `control.yml` | Simulation time period configuration |

### `hydro-param pywatershed validate`

Validate a pywatershed parameter file against metadata constraints.

\`\`\`bash
hydro-param pywatershed validate PARAM_FILE
\`\`\`

Checks that required PRMS parameters are present and values fall within valid ranges.
```

Also fix the duplicate description on line 38 (`datasets download` says "Display all datasets grouped by category" — should be "Download dataset files to local storage").

**Step 2: Commit**

```bash
git add docs/user-guide/cli.md
git commit -m "docs: add pywatershed run/validate to CLI reference (gap 2)"
```

---

### Task 13: Gaps 7, 8, 9 — Fix quickstart and installation docs

**Files:**
- Modify: `docs/getting-started/quickstart.md`
- Modify: `docs/getting-started/installation.md`

**Step 1: Fix quickstart (gaps 7 + 8)**

After step 1 ("Initialize a project"), add a new step 2:

```markdown
## 2. Obtain a target fabric

hydro-param does not fetch or subset fabrics. You need a pre-existing
GeoPackage or GeoParquet file containing your HRU polygons. Options:

- **pynhd/pygeohydro**: `pygeohydro.get_camels()` or NHDPlus catchments
- **USGS Geospatial Fabric**: Download from ScienceBase
- **Custom**: Any polygon mesh with a unique ID column

Copy your fabric to `data/fabrics/` inside the project.
```

Renumber subsequent steps (3–7 instead of 2–6).

Update step 7 ("Inspect results") to describe actual output structure:

```markdown
## 7. Inspect results

The pipeline writes files to `output/`:

- `output/<category>/<variable>.csv` — raw zonal statistics per variable
- `output/sir/<variable>.csv` — normalized SIR (standardized names/units)
- `output/.hydro_param_manifest.json` — resume manifest

For temporal datasets, output is `output/<category>/<dataset>_<year>_temporal.nc`.
```

**Step 2: Fix installation docs (gap 9)**

Replace the pip extras section with:

```markdown
## Install with pip

```bash
pip install hydro-param
```

!!! note
    pip installation requires that GDAL, GEOS, and PROJ system libraries
    are already installed. Using pixi (above) is recommended as it handles
    these automatically.
```

Remove the `[gdp]`, `[stac]`, `[regrid]` extras.

**Step 3: Commit**

```bash
git add docs/getting-started/quickstart.md docs/getting-started/installation.md
git commit -m "docs: add fabric prerequisite, output structure, fix pip extras (gaps 7, 8, 9)"
```

---

### Task 14: Run full check suite

**Step 1: Run all checks**

```bash
pixi run -e dev check
```

**Step 2: Run pre-commit**

```bash
pixi run -e dev pre-commit
```

**Step 3: Fix any issues that come up**

**Step 4: Commit fixes if needed**

```bash
git commit -m "chore: fix lint/type/format issues from UX audit"
```

---

### Task 15: Create PR

**Step 1: Push and create PR**

```bash
git push -u origin fix/<issue-number>-pre-release-ux-audit
```

```bash
gh pr create \
  --title "fix: pre-release UX audit — 15 workflow gaps" \
  --body "$(cat <<'EOF'
## Summary

Fixes 15 gaps identified in an end-to-end user workflow review of
`hydro-param pywatershed run`. Closes #<issue-number>.

### Code fixes
- **Gap 3:** Soils dataset now included in `_translate_pws_to_pipeline`
- **Gap 4:** Unsupported climate sources (`daymet_v4`, `conus404_ba`) raise clear `ValueError`
- **Gap 5:** Land cover year derived from time period end year (was hardcoded 2021)
- **Gap 6:** Removed non-functional `failure_mode` config option
- **Gap 10:** Translated pipeline config logged at DEBUG level
- **Gap 11:** Temporal xarray datasets properly closed after merge
- **Gap 12:** INFO log when soltab.nc not produced
- **Gap 13:** Remote dataset status label clarified
- **Gap 14:** `validate` command now sets up logging

### Doc fixes
- **Gap 1:** README updated from "Design phase" to pre-alpha MVP
- **Gap 2:** CLI docs now include `pywatershed run` and `pywatershed validate`
- **Gap 7:** Output file structure documented in quickstart
- **Gap 8:** Fabric prerequisite added to quickstart
- **Gap 9:** Nonexistent pip extras removed from install docs
- **Gap 15:** Example configs note fabric prerequisite

## Test plan
- [ ] New tests for gaps 3, 4, 5 (translation tests)
- [ ] Existing tests pass (`pixi run -e dev check`)
- [ ] Pre-commit hooks pass
- [ ] End-to-end `hydro-param pywatershed run` with DRB fabric + gridMET 1980–2025

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
