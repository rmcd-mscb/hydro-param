# Design: Organize Pipeline Config Datasets by Registry Category

**Issue:** #182
**Date:** 2026-03-10

## Change Summary

Replace `PipelineConfig.datasets: list[DatasetRequest]` with `datasets: dict[str, list[DatasetRequest]]`, where keys are registry category names. The dict is validated at config load time against a canonical category set, then flattened to a list for downstream pipeline stages. At stage 2, a cross-validation warning fires if a dataset's registry category doesn't match its config key.

## Design Decisions

- **Clean break** — no backward compatibility with flat list format
- **Validated keys** — category keys must match a known registry category
- **Warn on mismatch** — cross-validate dataset category at stage 2 (warn, not error)
- **VALID_CATEGORIES frozenset** in `dataset_registry.py` — single source of truth
- **Cross-validation at stage 2** — when registry entries are loaded, not at config time

## Components

### 1. `dataset_registry.py` — `VALID_CATEGORIES` frozenset

Export a `VALID_CATEGORIES: frozenset[str]` constant with the 8 known categories: `climate`, `geology`, `hydrography`, `land_cover`, `snow`, `soils`, `topography`, `water_bodies`.

### 2. `config.py` — Schema change

- `PipelineConfig.datasets` type changes to `dict[str, list[DatasetRequest]]`
- Pydantic `model_validator` checks all keys against `VALID_CATEGORIES`, raising `ValidationError` for unknown keys
- `_resolve_paths()` updated to iterate the nested dict structure

### 3. `config.py` — `flatten_datasets()` helper

A method on `PipelineConfig` that flattens dict values into `list[DatasetRequest]`. All downstream consumers call this instead of accessing `config.datasets` directly.

### 4. `pipeline.py` — Stage 2 cross-validation

After resolving each dataset against the registry, compare the config key (category) with `DatasetEntry.category`. Log a warning on mismatch.

### 5. `pipeline.py` — Consumer updates

The 3 places that access `config.datasets` switch to `config.flatten_datasets()` or iterate the dict directly as appropriate.

### 6. Example config + tests

- `drb_2yr_pipeline.yml` updated to themed dict format
- `test_config.py`: valid themed config, unknown category key rejected, empty category list
- `test_pipeline.py`: cross-validation warning on category mismatch

## Data Flow

```
YAML dict[str, list[DatasetRequest]]
  → Pydantic validates keys against VALID_CATEGORIES
  → PipelineConfig.datasets stored as dict
  → flatten_datasets() produces list[DatasetRequest] for pipeline stages
  → Stage 2: resolve each dataset, warn if registry category ≠ config key
```

## Not Changing

- `DatasetRequest` model
- Pipeline stages 1, 3, 4, 5 (receive flattened list)
- Registry files
- `pywatershed_run.yml` (separate config)
