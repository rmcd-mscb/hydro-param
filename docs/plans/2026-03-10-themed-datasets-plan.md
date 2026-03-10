# Themed Pipeline Config Datasets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `PipelineConfig.datasets: list[DatasetRequest]` with `dict[str, list[DatasetRequest]]` keyed by registry category, with validation and stage 2 cross-check.

**Architecture:** Add `VALID_CATEGORIES` frozenset to `dataset_registry.py`. Change `PipelineConfig.datasets` to `dict[str, list[DatasetRequest]]` with a Pydantic validator checking keys. Add `flatten_datasets()` method for downstream consumers. Stage 2 cross-validates dataset category against config key.

**Tech Stack:** Pydantic v2 model validators, Python logging, pytest

---

### Task 1: Add VALID_CATEGORIES to dataset_registry.py

**Files:**
- Modify: `src/hydro_param/dataset_registry.py` (top-level constant)
- Test: `tests/test_dataset_registry.py`

**Step 1: Write the failing test**

In `tests/test_dataset_registry.py`, add:

```python
from hydro_param.dataset_registry import VALID_CATEGORIES


def test_valid_categories_is_frozenset():
    """VALID_CATEGORIES contains the 8 registry category names."""
    assert isinstance(VALID_CATEGORIES, frozenset)
    assert VALID_CATEGORIES == frozenset({
        "climate", "geology", "hydrography", "land_cover",
        "snow", "soils", "topography", "water_bodies",
    })
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_dataset_registry.py::test_valid_categories_is_frozenset -v`
Expected: FAIL with ImportError (VALID_CATEGORIES not defined)

**Step 3: Write minimal implementation**

Add near the top of `src/hydro_param/dataset_registry.py` (after imports, before class definitions):

```python
VALID_CATEGORIES: frozenset[str] = frozenset({
    "climate",
    "geology",
    "hydrography",
    "land_cover",
    "snow",
    "soils",
    "topography",
    "water_bodies",
})
"""Valid dataset registry categories.

These correspond to the per-category YAML files bundled in
``hydro_param/data/datasets/``.  Used by :class:`~hydro_param.config.PipelineConfig`
to validate category keys in the ``datasets:`` config section.
"""
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_dataset_registry.py::test_valid_categories_is_frozenset -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/dataset_registry.py tests/test_dataset_registry.py
git commit -m "refactor: add VALID_CATEGORIES frozenset to dataset_registry (#182)"
```

---

### Task 2: Change PipelineConfig.datasets to dict and add flatten_datasets()

**Files:**
- Modify: `src/hydro_param/config.py` (PipelineConfig, _resolve_paths)
- Test: `tests/test_config.py`

**Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
from hydro_param.dataset_registry import VALID_CATEGORIES


def test_themed_datasets_from_yaml(tmp_path: Path):
    """Pipeline config accepts datasets organized by category."""
    raw = {
        "target_fabric": {"path": "data/catchments.gpkg", "id_field": "featureid"},
        "datasets": {
            "topography": [
                {"name": "dem_3dep_10m", "variables": ["elevation"]},
            ],
            "soils": [
                {"name": "gnatsgo_rasters", "variables": ["aws0_100"]},
            ],
        },
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))

    config = load_config(str(path))
    assert "topography" in config.datasets
    assert "soils" in config.datasets
    assert config.datasets["topography"][0].name == "dem_3dep_10m"


def test_themed_datasets_rejects_unknown_category():
    """Unknown category key raises ValidationError."""
    with pytest.raises(ValidationError, match="not_a_category"):
        PipelineConfig(
            target_fabric={"path": "test.gpkg", "id_field": "id"},
            datasets={"not_a_category": [{"name": "dem", "variables": ["elevation"]}]},
        )


def test_flatten_datasets():
    """flatten_datasets() merges all categories into a flat list."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        datasets={
            "topography": [
                DatasetRequest(name="dem", variables=["elevation"]),
            ],
            "soils": [
                DatasetRequest(name="gnatsgo", variables=["aws0_100"]),
                DatasetRequest(name="polaris", variables=["sand"]),
            ],
        },
    )
    flat = config.flatten_datasets()
    assert len(flat) == 3
    names = [ds.name for ds in flat]
    assert "dem" in names
    assert "gnatsgo" in names
    assert "polaris" in names


def test_themed_datasets_empty_dict():
    """Empty datasets dict is valid."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        datasets={},
    )
    assert config.flatten_datasets() == []


def test_themed_datasets_empty_category_list():
    """Category with empty list is valid."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        datasets={"topography": []},
    )
    assert config.flatten_datasets() == []
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_config.py::test_themed_datasets_from_yaml tests/test_config.py::test_themed_datasets_rejects_unknown_category tests/test_config.py::test_flatten_datasets tests/test_config.py::test_themed_datasets_empty_dict tests/test_config.py::test_themed_datasets_empty_category_list -v`
Expected: FAIL (datasets is still list type)

**Step 3: Write implementation**

In `src/hydro_param/config.py`:

1. Add import: `from hydro_param.dataset_registry import VALID_CATEGORIES`

2. Change `PipelineConfig.datasets` type and add validator + method:

```python
class PipelineConfig(BaseModel):
    """...(existing docstring)..."""

    target_fabric: TargetFabricConfig
    domain: DomainConfig | None = None
    datasets: dict[str, list[DatasetRequest]]
    output: OutputConfig = OutputConfig()
    processing: ProcessingConfig = ProcessingConfig()

    @model_validator(mode="after")
    def _validate_dataset_categories(self) -> PipelineConfig:
        """Validate that all dataset category keys are known registry categories."""
        unknown = set(self.datasets.keys()) - VALID_CATEGORIES
        if unknown:
            raise ValueError(
                f"Unknown dataset categories: {sorted(unknown)}. "
                f"Valid categories: {sorted(VALID_CATEGORIES)}"
            )
        return self

    def flatten_datasets(self) -> list[DatasetRequest]:
        """Flatten themed dataset dict into a single list for pipeline stages.

        Returns
        -------
        list[DatasetRequest]
            All dataset requests from all categories, preserving order
            within each category.
        """
        return [ds for ds_list in self.datasets.values() for ds in ds_list]
```

3. Update `_resolve_paths()` to iterate nested dict:

```python
def _resolve_paths(config: PipelineConfig) -> PipelineConfig:
    config.target_fabric.path = config.target_fabric.path.resolve()
    config.output.path = config.output.path.resolve()
    for ds_list in config.datasets.values():
        for ds in ds_list:
            if ds.source is not None:
                ds.source = ds.source.resolve()
    return config
```

**Step 4: Run new tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_config.py::test_themed_datasets_from_yaml tests/test_config.py::test_themed_datasets_rejects_unknown_category tests/test_config.py::test_flatten_datasets tests/test_config.py::test_themed_datasets_empty_dict tests/test_config.py::test_themed_datasets_empty_category_list -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/config.py tests/test_config.py
git commit -m "refactor: change PipelineConfig.datasets to dict keyed by category (#182)"
```

---

### Task 3: Update existing test_config.py tests

All existing tests that construct configs with `"datasets": [...]` (flat list) must be updated to `"datasets": {"<category>": [...]}` (themed dict). Tests that access `config.datasets[0]` must use `config.flatten_datasets()[0]` or `config.datasets["<category>"][0]`.

**Files:**
- Modify: `tests/test_config.py`

**Step 1: Update all existing test fixtures**

The following tests need their `"datasets"` value changed from a flat list to a category-keyed dict. Use `"topography"` as the default category for DEM-like test datasets:

- `test_load_config_from_yaml`: `"datasets": [{"name": "dem_3dep_10m", ...}]` → `"datasets": {"topography": [{"name": "dem_3dep_10m", ...}]}`; assertions change to `config.datasets["topography"][0]` or `config.flatten_datasets()[0]`
- `test_config_full_yaml`: same pattern
- `test_config_defaults`: `datasets=[]` → `datasets={}`
- `test_dataset_request_source_from_yaml`: wrap in `"land_cover"` category, update assertions to use `flatten_datasets()`
- `test_domain_optional`: wrap in `"topography"`
- `test_dataset_request_year_list_from_yaml`: wrap in `"land_cover"`; update assertion to `config.flatten_datasets()[0].year`
- `test_dataset_request_time_period_from_yaml`: wrap in `"snow"`; update assertion to `config.flatten_datasets()[0].time_period`
- `test_load_config_resolves_relative_paths`: wrap in `"topography"`
- `test_load_config_resolves_dotdot_paths`: `datasets=[]` → `datasets={}`
- `test_load_config_preserves_absolute_paths`: `datasets=[]` → `datasets={}`
- `test_load_config_resolves_dataset_source`: wrap in `"land_cover"` and `"topography"`; update assertions to use `flatten_datasets()`

**Step 2: Run all config tests**

Run: `pixi run -e dev pytest tests/test_config.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_config.py
git commit -m "test: update config tests for themed datasets dict (#182)"
```

---

### Task 4: Update pipeline.py consumers to use flatten_datasets()

**Files:**
- Modify: `src/hydro_param/pipeline.py` (3 locations)
- Test: `tests/test_pipeline.py`

**Step 1: Update pipeline.py**

Three locations in `pipeline.py` access `config.datasets`:

1. **Line 429** (stage 2 log message): `len(config.datasets)` → `len(config.flatten_datasets())`
2. **Line 431** (stage 2 loop): `for ds_req in config.datasets:` → `for ds_req in config.flatten_datasets():`
3. **Line 1594** (run_pipeline log): `len(config.datasets)` → `len(config.flatten_datasets())`

**Step 2: Update test_pipeline.py fixtures**

All test fixtures in `test_pipeline.py` that construct config dicts with `"datasets": [...]` need updating to `"datasets": {"<category>": [...]}`. There are ~18 locations (lines 68, 134, 160, 184, 207, 251, 283, 315, 964, 1003, 1041, 1079, 1117, 1156, 1668, 1707, 2434).

For each:
- Wrap the list in an appropriate category key (use `"topography"` for DEM tests, `"soils"` for soil tests, etc.; `"topography"` is a safe default for generic test datasets)
- Update any `config.datasets[0]` access to `config.flatten_datasets()[0]` (line 2442)

**Step 3: Run all pipeline tests**

Run: `pixi run -e dev pytest tests/test_pipeline.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "refactor: update pipeline to use flatten_datasets() (#182)"
```

---

### Task 5: Add stage 2 category cross-validation warning

**Files:**
- Modify: `src/hydro_param/pipeline.py` (stage2_resolve_datasets)
- Modify: `src/hydro_param/config.py` (add category_for_dataset helper or pass category through)
- Test: `tests/test_pipeline.py`

**Step 1: Design the cross-validation**

The challenge: `stage2_resolve_datasets` currently receives a flat list from `flatten_datasets()`. To cross-validate, it needs to know which config category key each dataset was listed under.

Approach: Change `stage2_resolve_datasets` to iterate the dict directly (not the flattened list). For each `(category_key, ds_req)` pair, after resolving the registry entry, compare `entry.category` with `category_key`.

**Step 2: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_stage2_warns_on_category_mismatch(tmp_path, caplog):
    """Stage 2 warns when a dataset's registry category doesn't match its config key."""
    import logging

    # Create a config where dem_3dep_10m (topography) is under "soils"
    raw = {
        "target_fabric": {"path": str(tmp_path / "fabric.gpkg"), "id_field": "nhm_id"},
        "datasets": {
            "soils": [
                {"name": "dem_3dep_10m", "variables": ["elevation"], "statistics": ["mean"]},
            ],
        },
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.dump(raw))
    config = load_config(cfg_path)

    registry = load_registry()

    with caplog.at_level(logging.WARNING, logger="hydro_param.pipeline"):
        resolved = stage2_resolve_datasets(config, registry)

    assert any("category mismatch" in msg.lower() for msg in caplog.messages)
    # Still resolves successfully despite mismatch
    assert len(resolved) == 1
```

**Step 3: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_stage2_warns_on_category_mismatch -v`
Expected: FAIL (no warning emitted)

**Step 4: Implement cross-validation**

Update `stage2_resolve_datasets` in `pipeline.py` to iterate the dict:

```python
def stage2_resolve_datasets(config, registry):
    all_datasets = config.flatten_datasets()
    logger.info("Stage 2: Resolving %d datasets from registry", len(all_datasets))

    # Build dataset-name → config-category lookup
    ds_category_map: dict[str, str] = {}
    for category_key, ds_list in config.datasets.items():
        for ds_req in ds_list:
            ds_category_map[ds_req.name] = category_key

    resolved = []
    for ds_req in all_datasets:
        entry = registry.get(ds_req.name)

        # Cross-validate config category vs registry category
        config_cat = ds_category_map.get(ds_req.name, "")
        if entry.category and config_cat and entry.category != config_cat:
            logger.warning(
                "Category mismatch for dataset '%s': config key is '%s' "
                "but registry category is '%s'",
                ds_req.name,
                config_cat,
                entry.category,
            )

        # ... rest of existing resolution logic unchanged ...
```

**Step 5: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_stage2_warns_on_category_mismatch -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "feat: warn on dataset category mismatch in stage 2 (#182)"
```

---

### Task 6: Update example config and run full checks

**Files:**
- Modify: `configs/examples/drb_2yr_pipeline.yml`

**Step 1: Update example config to themed format**

```yaml
datasets:
  topography:
    - name: dem_3dep_10m
      variables: [elevation, slope, aspect]
      statistics: [mean]

  soils:
    - name: gnatsgo_rasters
      variables: [aws0_100, rootznemc, rootznaws]
      statistics: [mean]
    - name: polaris_30m
      variables: [sand, silt, clay, theta_s, ksat, soil_texture]
      statistics: [mean]

  land_cover:
    - name: nlcd_osn_lndcov
      variables: [LndCov]
      statistics: [categorical]
      year: [2020, 2021]
    - name: nlcd_osn_fctimp
      variables: [FctImp]
      statistics: [mean]
      year: [2020, 2021]

  snow:
    - name: snodas
      variables: [SWE]
      statistics: [mean]
      time_period: ["2020-01-01", "2021-12-31"]

  climate:
    - name: gridmet
      variables: [pr, tmmx, tmmn, srad, pet, vs]
      statistics: [mean]
      time_period: ["2020-01-01", "2021-12-31"]
```

**Step 2: Run full check suite**

Run: `pixi run -e dev check`
Expected: ALL PASS (lint, format, typecheck, tests)

Run: `pixi run -e dev pre-commit`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add configs/examples/drb_2yr_pipeline.yml
git commit -m "docs: update example pipeline config to themed dataset format (#182)"
```

---

### Task 7: Check for any other consumers of config.datasets

**Files:** Various — search-and-fix pass

**Step 1: Search for remaining config.datasets references**

Search across the entire `src/` and `tests/` trees for any remaining references to `config.datasets` that aren't going through `flatten_datasets()`. Also search for `\.datasets\[` (direct indexing into what was a list).

```bash
rg "config\.datasets" src/ tests/ --glob '*.py'
rg "\.datasets\[" src/ tests/ --glob '*.py'
```

Fix any found references. Common patterns:
- `config.datasets[N]` → `config.flatten_datasets()[N]` or `config.datasets["category"][N]`
- `len(config.datasets)` → `len(config.flatten_datasets())`
- `for ds in config.datasets:` → `for ds in config.flatten_datasets():`

Also check `src/hydro_param/cli.py` for any direct access.

**Step 2: Run full test suite**

Run: `pixi run -e dev pytest -v`
Expected: ALL PASS

**Step 3: Commit if changes were needed**

```bash
git add -u
git commit -m "fix: update remaining config.datasets references to flatten_datasets() (#182)"
```
