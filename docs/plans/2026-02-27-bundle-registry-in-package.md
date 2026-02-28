# Bundle Dataset Registry in Package — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the dataset registry into the Python package so hydro-param works when pip-installed and run from any directory.

**Architecture:** Copy `configs/datasets/` → `src/hydro_param/data/datasets/`, change `DEFAULT_REGISTRY` to use `importlib.resources` (matching existing lookup_tables pattern), delete stale `configs/` copies, update tests and docstrings.

**Tech Stack:** Python `importlib.resources`, hatchling packaging, pytest

**Design doc:** `docs/plans/2026-02-27-bundle-registry-in-package-design.md`

---

### Task 1: Move dataset registry into package

**Files:**
- Create: `src/hydro_param/data/datasets/` (8 YAML files)
- Delete: `configs/datasets/` (moved, not needed)
- Delete: `configs/lookup_tables/` (stale copy — package has canonical version)
- Delete: `configs/pywatershed/` (stale copy — package has canonical version)

**Step 1: Copy dataset registry files into package**

```bash
cp -r configs/datasets/ src/hydro_param/data/datasets/
```

**Step 2: Verify the copy**

```bash
ls src/hydro_param/data/datasets/
# Expected: climate.yml geology.yml hydrography.yml land_cover.yml
#           snow.yml soils.yml topography.yml water_bodies.yml
```

**Step 3: Delete stale configs/ directories**

```bash
rm -rf configs/datasets/ configs/lookup_tables/ configs/pywatershed/
```

After deletion, `configs/` should only contain:
```
configs/
  examples/
    drb_2yr_pipeline.yml
    drb_2yr_pywatershed.yml
  delaware_terrain.yml
  delaware_terrain_sandbox.yml
```

**Step 4: Commit**

```bash
git add src/hydro_param/data/datasets/ && git add -u
git commit -m "refactor: move dataset registry into package, remove stale configs"
```

---

### Task 2: Fix DEFAULT_REGISTRY to use importlib.resources

**Files:**
- Modify: `src/hydro_param/pipeline.py:81`

**Step 1: Write the failing test**

Add to `tests/test_pipeline.py` (or `tests/test_registry.py` — wherever DEFAULT_REGISTRY is tested):

```python
def test_default_registry_resolves_to_existing_directory():
    """DEFAULT_REGISTRY must resolve to the bundled registry directory."""
    from hydro_param.pipeline import DEFAULT_REGISTRY

    assert DEFAULT_REGISTRY.is_dir(), f"DEFAULT_REGISTRY does not exist: {DEFAULT_REGISTRY}"
    yamls = sorted(p.name for p in DEFAULT_REGISTRY.glob("*.yml"))
    expected = [
        "climate.yml",
        "geology.yml",
        "hydrography.yml",
        "land_cover.yml",
        "snow.yml",
        "soils.yml",
        "topography.yml",
        "water_bodies.yml",
    ]
    assert yamls == expected
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_registry.py::test_default_registry_resolves_to_existing_directory -v`

Expected: FAIL — `DEFAULT_REGISTRY` is still `Path("configs/datasets")` which no longer exists after Task 1.

**Step 3: Fix DEFAULT_REGISTRY**

In `src/hydro_param/pipeline.py`, change line 81 from:

```python
DEFAULT_REGISTRY = Path("configs/datasets")
```

to:

```python
from importlib.resources import files as _pkg_files

DEFAULT_REGISTRY = Path(str(_pkg_files("hydro_param").joinpath("data/datasets")))
```

Note: Import `files` with an alias `_pkg_files` to avoid shadowing — `pipeline.py` already imports many names. Place the import near the top of the file with the other imports, and the `DEFAULT_REGISTRY` assignment stays at line 81.

Actually, check if `importlib.resources.files` is already imported in pipeline.py. If not, add it to the imports section. The existing pattern in `plugins.py` line 7 uses:
```python
from importlib.resources import files
```

Follow the same pattern in pipeline.py.

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_registry.py::test_default_registry_resolves_to_existing_directory -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_registry.py
git commit -m "fix: resolve DEFAULT_REGISTRY via importlib.resources"
```

---

### Task 3: Update tests that reference configs/datasets/

**Files:**
- Modify: `tests/test_registry.py` (~9 test functions)

There are 9 tests in `tests/test_registry.py` that use `Path("configs/datasets")` with a skip guard:

```python
registry_path = Path("configs/datasets")
if not registry_path.exists():
    pytest.skip("configs/datasets/ not found")
```

These tests load the real registry to validate its structure. Now that the registry lives in the package, update them to use `DEFAULT_REGISTRY`:

```python
from hydro_param.pipeline import DEFAULT_REGISTRY

# In each test, replace:
#   registry_path = Path("configs/datasets")
#   if not registry_path.exists():
#       pytest.skip("configs/datasets/ not found")
# With:
#   registry_path = DEFAULT_REGISTRY
```

The skip guard is no longer needed — `DEFAULT_REGISTRY` always resolves via `importlib.resources` and will always exist when the package is installed.

Apply this replacement to all 9 test functions that use this pattern:
- `test_load_real_registry` (line 523)
- `test_real_registry_nlcd_legacy_has_download` (line 538)
- `test_real_registry_variables_have_required_fields` (line 559)
- `test_real_registry_temporal_datasets` (line 579)
- `test_real_registry_stac_cog_datasets` (line 596)
- `test_real_registry_nhgf_stac_datasets` (line 612)
- `test_real_registry_local_tiff_datasets` (line 627)
- `test_real_registry_climr_cat_datasets` (line 756)
- `test_real_registry_download_blocks` (line 835)

**Step 1: Update all 9 tests**

Replace `Path("configs/datasets")` with `DEFAULT_REGISTRY` and remove skip guards.

**Step 2: Add the import at the top of the test file**

Near the existing imports in `tests/test_registry.py`, add:

```python
from hydro_param.pipeline import DEFAULT_REGISTRY
```

**Step 3: Run tests**

Run: `pixi run -e dev pytest tests/test_registry.py -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add tests/test_registry.py
git commit -m "test: update registry tests to use DEFAULT_REGISTRY"
```

---

### Task 4: Update docstrings referencing configs/datasets/

**Files:**
- Modify: `src/hydro_param/pipeline.py:1485`
- Modify: `src/hydro_param/dataset_registry.py:476,496`
- Modify: `src/hydro_param/project.py:35,48`

**Step 1: Update docstrings**

In `pipeline.py` line 1485, change:
```
Defaults to ``configs/datasets/`` (the built-in registry).
```
to:
```
Defaults to the built-in registry bundled with the package.
```

In `dataset_registry.py` line 476, change:
```
per-category YAML files (e.g., ``configs/datasets/``).
```
to:
```
per-category YAML files (e.g., the bundled ``hydro_param.data.datasets``).
```

In `dataset_registry.py` line 496, change:
```python
>>> registry = load_registry("configs/datasets/")
```
to:
```python
>>> from hydro_param.pipeline import DEFAULT_REGISTRY
>>> registry = load_registry(DEFAULT_REGISTRY)
```

In `project.py` lines 35 and 48, change references to `configs/datasets/` to `hydro_param.data.datasets` (these are comments about category names matching YAML file names).

**Step 2: Run full test suite**

Run: `pixi run -e dev check`
Expected: All tests pass, lint/format/typecheck clean.

**Step 3: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass.

**Step 4: Commit**

```bash
git add src/hydro_param/pipeline.py src/hydro_param/dataset_registry.py src/hydro_param/project.py
git commit -m "docs: update docstrings for bundled registry location"
```

---

### Task 5: Verify end-to-end from user project directory

**Step 1: Reinstall in test venv**

```bash
cd /tmp/drb-e2e
source .venv/bin/activate
pip install -e /home/rmcd/projects/usgs/hydro-param
```

**Step 2: Run hydro-param datasets list**

```bash
hydro-param datasets list
```

Expected: Lists all datasets from the bundled registry (dem_3dep_10m, polaris_30m, etc.) — no FileNotFoundError.

**Step 3: Verify pipeline starts (will fail on missing fabric, that's OK)**

```bash
hydro-param run configs/pipeline.yml
```

Expected: Gets past registry loading, fails on missing fabric file (not on missing registry). The error should be about `data/fabrics/catchments.gpkg` not existing, NOT `configs/datasets`.
