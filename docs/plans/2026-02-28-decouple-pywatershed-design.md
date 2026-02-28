# Design: Decouple pywatershed run from Phase 1 Pipeline

**Issue:** #117
**Date:** 2026-02-28
**Status:** Approved

## Problem

`hydro-param pywatershed run` executes both Phase 1 (generic pipeline) and
Phase 2 (pywatershed derivation + formatting) inline. This violates the
two-phase separation principle and has practical consequences:

- Re-running derivation after a config tweak re-runs the entire pipeline.
- Dataset mismatches between `pipeline.yml` and `pywatershed_run.yml` cause
  confusing errors.
- The pywatershed config contains fields (`datasets:`, `processing:`,
  `climate:`) that belong to Phase 1, not Phase 2.

## Desired Behavior

```bash
# Phase 1: generic pipeline produces SIR (unchanged)
hydro-param run configs/pipeline.yml

# Phase 2: pywatershed consumes existing SIR (decoupled)
hydro-param pywatershed run configs/pywatershed_run.yml
```

## Design Decisions

1. **Project structure is the coupling point.** Both configs share the same
   project layout via `hydro-param init`. Decoupling is at execution level,
   not project level.

2. **Manifest is the contract between Phase 1 and Phase 2.** The pipeline
   manifest (`.manifest.yml`) is extended with a `sir` section that tracks
   normalized SIR files, temporal files, and schema metadata.

3. **Explicit `sir_path` with project-convention default.** The pywatershed
   config gains a `sir_path` field (default: `"output"`) pointing to the
   pipeline output directory. Advanced users can override it.

4. **Clean break — remove Phase 1 fields.** `datasets:`, `processing:`, and
   `climate:` are removed from the pywatershed config. No convenience mode.

5. **Lazy per-variable loading via SIRAccessor.** Derivation steps load only
   the SIR variables they need on demand, then release. No upfront memory hit.

6. **Manifest preferred, glob fallback.** If the manifest is missing or
   corrupt, the SIRAccessor falls back to discovering SIR files by globbing
   `sir/` with a warning.

## 1. Manifest Extension

Extend `PipelineManifest` with a `sir` field and bump version 1 → 2.

```python
class SIRManifestEntry(BaseModel):
    static_files: dict[str, str] = {}      # {sir_var_name: relative_path}
    temporal_files: dict[str, str] = {}    # {dataset_year: relative_path}
    schema: list[dict] = []                 # SIRVariableSchema serialized
    completed_at: datetime = datetime.min

class PipelineManifest(BaseModel):
    version: int = 2                        # bumped from 1
    fabric_fingerprint: str = ""
    entries: dict[str, ManifestEntry] = {}  # existing per-dataset tracking
    sir: SIRManifestEntry | None = None     # NEW: stage 5 output
```

Stage 5 writes the `sir` section after normalization. Phase 2 reads it to
discover all SIR files.

**Migration:** Version 1 manifests missing `sir` are treated as "SIR not yet
produced." Phase 2 tells the user to run `hydro-param run pipeline.yml` first
(or falls back to globbing if files exist on disk).

**Hardening:**
- Atomic writes (write to `.manifest.yml.tmp`, rename).
- File existence validation for every manifest reference at SIRAccessor init.
- Schema version check with clear migration messages.

## 2. Pywatershed Config Simplification

Remove Phase 1 fields, add `sir_path`, bump version to `"3.0"`.

**Before (v2.0 — current):**

```yaml
target_model: pywatershed
version: "2.0"
domain:
  source: custom
  extraction_method: bbox
  bbox: [-76.5, 38.5, -74.0, 42.6]
  fabric_path: "data/fabrics/nhru.gpkg"
  segment_path: "data/fabrics/nsegment.gpkg"
  id_field: "nhm_id"
  segment_id_field: "nhm_seg"
time: { ... }
climate: { source: gridmet, ... }
datasets: { topography: dem_3dep_10m, ... }
processing: { zonal_method: exactextract, ... }
parameter_overrides: { ... }
calibration: { ... }
output: { ... }
```

**After (v3.0 — decoupled):**

```yaml
target_model: pywatershed
version: "3.0"

domain:
  fabric_path: "data/fabrics/nhru.gpkg"
  segment_path: "data/fabrics/nsegment.gpkg"
  # waterbody_path: "data/fabrics/waterbodies.gpkg"
  id_field: "nhm_id"
  segment_id_field: "nhm_seg"

time:
  start: "1980-10-01"
  end: "2020-09-30"

sir_path: "output"

parameter_overrides:
  values: {}

calibration:
  generate_seeds: true
  seed_method: physically_based
  preserve_from_existing: []

output:
  path: "models/pywatershed"
  format: netcdf
  parameter_file: "parameters.nc"
  forcing_dir: "forcing"
  control_file: "control.yml"
  soltab_file: "soltab.nc"
```

**Removed:**
- `climate:` — Phase 1 concern (which climate dataset to fetch).
- `datasets:` — Phase 1 concern (which datasets to process).
- `processing:` — Phase 1 concern (zonal engine, batch size).
- `domain.source`, `domain.extraction_method`, `domain.bbox`,
  `domain.gf_version`, `domain.huc_id`, `domain.pour_point` — Phase 1
  extraction concerns.

**Kept:**
- `domain:` — file paths and id fields (Phase 2 needs fabric/segments/waterbodies).
- `time:` — start/end for control file and temporal merge.
- `parameter_overrides:`, `calibration:`, `output:` — Phase 2 concerns.

**Added:**
- `sir_path: Path = Path("output")` — pipeline output directory.

**Simplified `PwsDomainConfig`:**

```python
class PwsDomainConfig(BaseModel):
    fabric_path: Path
    segment_path: Path | None = None
    waterbody_path: Path | None = None
    id_field: str = "nhm_id"
    segment_id_field: str = "nhm_seg"
```

## 3. Phase 2 Execution Flow — Lazy Loading

### SIRAccessor

A thin class wrapping the manifest for per-variable on-demand loading:

```python
class SIRAccessor:
    """Lazy accessor for SIR files backed by a pipeline manifest.

    Does not hold data in memory. Each call to load_variable()
    reads from disk; the caller releases when done.
    """

    def __init__(self, output_dir: Path):
        manifest = load_manifest(output_dir)
        if manifest and manifest.sir:
            self._static = manifest.sir.static_files
            self._temporal = manifest.sir.temporal_files
            self._schema = manifest.sir.schema
        else:
            logger.warning(
                "No valid manifest at %s — discovering SIR files by "
                "scanning sir/. Schema metadata will not be available. "
                "Consider re-running 'hydro-param run pipeline.yml' to "
                "regenerate the manifest.",
                output_dir,
            )
            self._static = _glob_sir_static(output_dir / "sir")
            self._temporal = _glob_sir_temporal(output_dir / "sir")
            self._schema = []
        # Validate all referenced files exist (fail fast)
        self._validate_files(output_dir)

    def available_variables(self) -> list[str]: ...
    def available_temporal(self) -> list[str]: ...
    def load_variable(self, name: str) -> xr.DataArray: ...
    def load_temporal(self, name: str) -> xr.Dataset: ...
```

### DerivationContext Change

```python
@dataclass(frozen=True)
class DerivationContext:
    sir: SIRAccessor                          # was: xr.Dataset
    fabric: gpd.GeoDataFrame | None = None
    segments: gpd.GeoDataFrame | None = None
    waterbodies: gpd.GeoDataFrame | None = None
    fabric_id_field: str = "nhm_id"
    segment_id_field: str | None = None
    config: dict = field(default_factory=dict)
    lookup_tables_dir: Path | None = None
```

### Derivation Step Pattern

Each step loads only what it needs:

```python
def _derive_topography(self, ctx, ds):
    elevation = ctx.sir.load_variable("elevation")
    slope = ctx.sir.load_variable("slope")
    aspect = ctx.sir.load_variable("aspect")
    # ... compute PRMS params ...
    # elevation, slope, aspect go out of scope → GC'd

def _compute_monthly_normals(self, ctx):
    for key in ctx.sir.available_temporal():
        chunk = ctx.sir.load_temporal(key)
        # accumulate monthly stats
        chunk.close()
```

### Revised pws_run_cmd Flow

```
pws_run_cmd(config)
  ├─ Load PywatershedRunConfig (v3.0)
  ├─ Resolve sir_path → create SIRAccessor(sir_path)  # no data loaded
  ├─ Load fabric GeoDataFrame from domain.fabric_path
  ├─ Load segments GeoDataFrame from domain.segment_path (optional)
  ├─ Load waterbodies GeoDataFrame from domain.waterbody_path (optional)
  ├─ Build DerivationContext(sir=accessor, fabric, segments, ...)
  ├─ PywatershedDerivation().derive(ctx)
  │   ├─ Step 1 (geometry): ctx.fabric only
  │   ├─ Step 3 (topo): load elevation, slope, aspect → release
  │   ├─ Step 4 (landcover): load LndCov fractions → release
  │   ├─ Step 5 (soils): load sand, silt, clay, ... → release
  │   ├─ Step 10 (PET): load temporal per-year → accumulate normals → release
  │   └─ ... each step loads and releases
  ├─ merge_temporal_into_derived()
  └─ PywatershedFormatter().write()
```

## 4. Init Template Updates

`generate_pywatershed_template()` produces the v3.0 config.

"Next steps" output from `hydro-param init`:

```
Next steps:
  1. Place your fabric files in data/fabrics/
  2. Edit configs/pipeline.yml (dataset sources, domain)
  3. Edit configs/pywatershed_run.yml (time period, output options)
  4. Run the pipeline:
       hydro-param run configs/pipeline.yml
  5. Run pywatershed parameterization:
       hydro-param pywatershed run configs/pywatershed_run.yml
```

## 5. Deletions

**Deleted:**
- `_translate_pws_to_pipeline()` in `cli.py` (~160 lines).
- `PwsDatasetSources` in `pywatershed_config.py`.
- `PwsProcessingConfig` in `pywatershed_config.py`.
- `PwsClimateConfig` in `pywatershed_config.py`.
- `PwsDomainConfig` extraction fields: `source`, `gf_version`,
  `extraction_method`, `bbox`, `huc_id`, `pour_point`.

**Added:**
- `sir_path: Path = Path("output")` on `PywatershedRunConfig`.
- `SIRAccessor` class (new module or in `sir.py`).
- `SIRManifestEntry` in `manifest.py`.
- Manifest version bump 1 → 2.

**Modified:**
- `DerivationContext.sir` type: `xr.Dataset` → `SIRAccessor`.
- `DerivationPlugin` protocol: `sir` field type changes.
- All 14 derivation steps: `ctx.sir["var"]` → `ctx.sir.load_variable("var")`.
- `pws_run_cmd()`: rewritten — no pipeline execution, reads manifest + SIRAccessor.
- `generate_pywatershed_template()`: simplified v3.0 template.
- Init "Next steps": two-phase workflow guidance.

## 6. Testing Strategy

**Unit tests:**
- `SIRAccessor` — load_variable correctness, missing variable KeyError, missing
  file FileNotFoundError, glob fallback when no manifest.
- `SIRManifestEntry` — serialization roundtrip, version 2 parsing, version 1
  treated as no-SIR.
- `PywatershedRunConfig` v3.0 — validates new schema, rejects old fields.
- `PwsDomainConfig` simplified — validates paths, rejects removed fields.

**Integration tests:**
- `pws_run_cmd` with fixture SIR on disk — write CSVs + temporal NCs + manifest
  v2, run Phase 2, verify output.
- Missing manifest → warning + glob fallback.
- Manifest with no `sir` section → warning + glob fallback.
- Missing SIR files → FileNotFoundError at init.

**Derivation step tests:**
- Fixtures change from pre-built `xr.Dataset` to mock/fixture `SIRAccessor`
  backed by temp CSV files.

**Deleted tests:**
- `_translate_pws_to_pipeline()` tests.
- Tests coupling pywatershed config to pipeline execution.
