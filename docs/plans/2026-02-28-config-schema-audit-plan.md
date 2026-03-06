# Config Schema Audit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `waterbody_path` to the pywatershed config schema and relocate PRMS lookup tables under `data/pywatershed/`.

**Architecture:** Two independent changes: (1) new optional `waterbody_path` field in `PwsDomainConfig` wired through CLI to `DerivationContext.waterbodies`, (2) move `data/lookup_tables/` → `data/pywatershed/lookup_tables/` with a one-line default path update.

**Tech Stack:** Pydantic config models, geopandas, importlib.resources, pytest

**Design doc:** `docs/plans/2026-02-28-config-schema-audit-design.md`

---

### Task 1: Relocate lookup tables (file move + path update)

**Files:**
- Move: `src/hydro_param/data/lookup_tables/*.yml` → `src/hydro_param/data/pywatershed/lookup_tables/`
- Delete: `src/hydro_param/data/lookup_tables/` (empty directory)
- Modify: `src/hydro_param/plugins.py:146`

**Step 1: Move the lookup table files**

```bash
mkdir -p src/hydro_param/data/pywatershed/lookup_tables
mv src/hydro_param/data/lookup_tables/*.yml src/hydro_param/data/pywatershed/lookup_tables/
rmdir src/hydro_param/data/lookup_tables
```

**Step 2: Update the default path in `plugins.py`**

In `src/hydro_param/plugins.py`, line 146, change:

```python
# Before:
return Path(str(files("hydro_param").joinpath("data/lookup_tables")))

# After:
return Path(str(files("hydro_param").joinpath("data/pywatershed/lookup_tables")))
```

**Step 3: Run the existing lookup table tests to verify**

Run: `pixi run -e dev pytest tests/test_plugins.py::TestDerivationContext::test_resolved_lookup_tables_dir_default -v`
Expected: PASS — the default path resolves to the new location and `nlcd_to_prms_cov_type.yml` exists there.

Run: `pixi run -e dev pytest tests/test_plugins.py -v`
Expected: All pass.

**Step 4: Commit**

```bash
git add -A src/hydro_param/data/pywatershed/lookup_tables/
git add src/hydro_param/plugins.py
git rm -r src/hydro_param/data/lookup_tables/
git commit -m "refactor: relocate PRMS lookup tables under data/pywatershed/

Move data/lookup_tables/ → data/pywatershed/lookup_tables/ to group all
pywatershed/PRMS-specific bundled data in one directory. Update the
default path in DerivationContext.resolved_lookup_tables_dir."
```

---

### Task 2: Add `waterbody_path` to `PwsDomainConfig`

**Files:**
- Modify: `src/hydro_param/pywatershed_config.py:62-64`
- Test: `tests/test_pywatershed_config.py`

**Step 1: Write the failing test**

Add to `tests/test_pywatershed_config.py` in `class TestPwsDomainConfig`:

```python
def test_waterbody_path_default_none(self) -> None:
    cfg = PwsDomainConfig(extraction_method="bbox", bbox=[-76.5, 38.5, -74.0, 42.6])
    assert cfg.waterbody_path is None

def test_waterbody_path_accepted(self) -> None:
    cfg = PwsDomainConfig(
        extraction_method="bbox",
        bbox=[-76.5, 38.5, -74.0, 42.6],
        waterbody_path=Path("/some/waterbodies.gpkg"),
    )
    assert cfg.waterbody_path == Path("/some/waterbodies.gpkg")
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestPwsDomainConfig::test_waterbody_path_accepted -v`
Expected: FAIL — `waterbody_path` is not a valid field.

**Step 3: Add the field to `PwsDomainConfig`**

In `src/hydro_param/pywatershed_config.py`, add after `segment_path` (line 84):

```python
waterbody_path: Path | None = None
```

Update the class docstring `Attributes` section to include:

```
waterbody_path : Path or None
    Path to NHDPlus waterbody polygon file (GeoPackage or GeoParquet)
    for depression storage overlay (step 6).  Must contain an ``ftype``
    column with values like ``"LakePond"`` and ``"Reservoir"``.
    When ``None``, step 6 uses zero defaults.
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestPwsDomainConfig -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/hydro_param/pywatershed_config.py tests/test_pywatershed_config.py
git commit -m "feat: add waterbody_path to PwsDomainConfig schema

Add optional waterbody_path field for NHDPlus waterbody polygon overlay.
When provided, step 6 (depression storage) will compute dprst_frac,
dprst_area_max, and hru_type from waterbody-HRU intersections."
```

---

### Task 3: Wire `waterbody_path` through CLI to `DerivationContext`

**Files:**
- Modify: `src/hydro_param/cli.py:841-858`

**Step 1: Add waterbody loading in `pws_run_cmd()`**

In `src/hydro_param/cli.py`, after the segments block (line 843), add:

```python
waterbodies = None
if pws_config.domain.waterbody_path is not None:
    waterbodies = gpd.read_file(pws_config.domain.waterbody_path)
```

Then update the `DerivationContext` construction (line 851) to pass it:

```python
ctx = DerivationContext(
    sir=sir,
    fabric=result.fabric,
    segments=segments,
    waterbodies=waterbodies,
    fabric_id_field=pws_config.domain.id_field,
    segment_id_field=pws_config.domain.segment_id_field,
    config=derivation_config,
)
```

**Step 2: Run existing tests**

Run: `pixi run -e dev pytest tests/test_cli.py -v`
Expected: All pass (no behavioral change when `waterbody_path` is `None`).

**Step 3: Commit**

```bash
git add src/hydro_param/cli.py
git commit -m "feat: wire waterbody_path through CLI to DerivationContext

Load waterbody GeoDataFrame from config path and pass to
DerivationContext so step 6 (depression storage) receives
actual waterbody data instead of always using defaults."
```

---

### Task 4: Update init template

**Files:**
- Modify: `src/hydro_param/project.py:288-295`
- Test: `tests/test_project.py`

**Step 1: Write the failing test**

Add to `tests/test_project.py`:

```python
from hydro_param.project import generate_pywatershed_template

def test_pywatershed_template_contains_waterbody_path() -> None:
    template = generate_pywatershed_template("test_project")
    assert "waterbody_path" in template
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_project.py::test_pywatershed_template_contains_waterbody_path -v`
Expected: FAIL — template doesn't contain `waterbody_path`.

**Step 3: Update the template**

In `src/hydro_param/project.py`, in `generate_pywatershed_template()`, after the `segment_path` line (line 293), add:

```yaml
  waterbody_path: "data/fabrics/waterbodies.gpkg"  # NHDPlus waterbody polygons (optional)
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_project.py::test_pywatershed_template_contains_waterbody_path -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/hydro_param/project.py tests/test_project.py
git commit -m "feat: add waterbody_path to pywatershed init template

Include waterbody_path in the generated pywatershed_run.yml so
users see the option alongside fabric_path and segment_path."
```

---

### Task 5: Run full test suite and pre-commit

**Step 1: Run all tests**

Run: `pixi run -e dev test`
Expected: All pass.

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All pass.

**Step 3: Review changes**

Run: `git log --oneline main..HEAD`
Expected: 4 commits (lookup relocation, schema field, CLI wiring, template update).
