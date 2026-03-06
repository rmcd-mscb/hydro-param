# Config Schema Audit: waterbody_path + Lookup Tables Relocation

**Date:** 2026-02-28
**Status:** Proposed

## Problem

A comprehensive audit of `PywatershedRunConfig` against `DerivationContext` revealed two gaps:

1. **Missing `waterbody_path`:** `DerivationContext.waterbodies` is never populated because `PwsDomainConfig` has no `waterbody_path` field. Step 6 (depression storage: `dprst_frac`, `dprst_area_max`, `hru_type`) always falls back to zero defaults.

2. **Misplaced lookup tables:** The 6 PRMS-specific lookup table YAMLs live at `data/lookup_tables/` — a generic package-level path. They belong under `data/pywatershed/` alongside `parameter_metadata.yml`, since they are exclusively pywatershed/PRMS data.

## Design

### 1. Add `waterbody_path` to `PwsDomainConfig`

Add to `pywatershed_config.py`:

```python
waterbody_path: Path | None = None
```

Same pattern as `fabric_path` and `segment_path`. In `cli.py` `pws_run_cmd()`, read the file and pass to `DerivationContext`:

```python
waterbodies = None
if pws_config.domain.waterbody_path is not None:
    waterbodies = gpd.read_file(pws_config.domain.waterbody_path)

ctx = DerivationContext(
    ...
    waterbodies=waterbodies,
)
```

The init template (`project.py`) gets a commented line next to `segment_path`:

```yaml
waterbody_path: "data/fabrics/waterbodies.gpkg"  # NHDPlus waterbody polygons (optional)
```

### 2. Relocate lookup tables under `data/pywatershed/`

Move `src/hydro_param/data/lookup_tables/` → `src/hydro_param/data/pywatershed/lookup_tables/`.

Final structure:

```
src/hydro_param/data/
  datasets/              — Dataset registry (model-agnostic)
  pywatershed/           — All pywatershed-specific bundled data
    parameter_metadata.yml
    lookup_tables/       — 6 PRMS lookup table YAMLs
      calibration_seeds.yml
      cov_type_to_interception.yml
      cov_type_winter_reduction.yml
      forcing_variables.yml
      nlcd_to_prms_cov_type.yml
      soil_texture_to_prms_type.yml
```

One code change — update the default in `plugins.py` `DerivationContext.resolved_lookup_tables_dir`:

```python
# Before:
return Path(str(files("hydro_param").joinpath("data/lookup_tables")))

# After:
return Path(str(files("hydro_param").joinpath("data/pywatershed/lookup_tables")))
```

All consumers use `ctx.resolved_lookup_tables_dir`, so the relocation is transparent.

### 3. Tests

- Config round-trip test for `waterbody_path` in `PwsDomainConfig`
- CLI wiring test: `waterbodies` passed to `DerivationContext` when `waterbody_path` is set
- Update any tests that explicitly reference the old `data/lookup_tables` path

## Files Changed

| Change | File |
|---|---|
| Add `waterbody_path` field | `src/hydro_param/pywatershed_config.py` |
| Wire waterbodies in CLI | `src/hydro_param/cli.py` |
| Update init template | `src/hydro_param/project.py` |
| Move lookup tables | `src/hydro_param/data/lookup_tables/` → `src/hydro_param/data/pywatershed/lookup_tables/` |
| Update default path | `src/hydro_param/plugins.py` |
| Delete old directory | `src/hydro_param/data/lookup_tables/` |
| Tests | Existing test modules |

## Decision: `lookup_tables_dir` config override

Decided **not** to expose a `lookup_tables_dir` field in the config schema. The bundled PRMS tables are sufficient. The `DerivationContext.lookup_tables_dir` override remains available for programmatic use but is not surfaced in YAML.
