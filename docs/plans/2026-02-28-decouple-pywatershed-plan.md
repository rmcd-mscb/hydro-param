# Decouple pywatershed run from Phase 1 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `hydro-param pywatershed run` consume existing SIR output from disk instead of re-running the Phase 1 pipeline inline.

**Architecture:** Extend the pipeline manifest with SIR tracking (version 2), introduce a lazy `SIRAccessor` that loads variables on demand, strip Phase 1 fields from the pywatershed config (v3.0), and rewrite `pws_run_cmd()` to read from manifest + SIRAccessor.

**Tech Stack:** Python, Pydantic, xarray, geopandas, pytest, YAML

**Design doc:** `docs/plans/2026-02-28-decouple-pywatershed-design.md`

---

### Task 1: Extend Manifest with SIR Tracking

**Files:**
- Modify: `src/hydro_param/manifest.py:48` (`_SUPPORTED_VERSION`), `:89-120` (`PipelineManifest`)
- Modify: `src/hydro_param/pipeline.py:1331-1408` (`stage5_normalize_sir`)
- Test: `tests/test_manifest.py`

**Step 1: Write failing tests for SIRManifestEntry and version 2 manifest**

Add to `tests/test_manifest.py`:

```python
class TestSIRManifestEntry:
    """Test the SIR section of the pipeline manifest."""

    def test_sir_manifest_entry_defaults(self):
        from hydro_param.manifest import SIRManifestEntry

        entry = SIRManifestEntry()
        assert entry.static_files == {}
        assert entry.temporal_files == {}
        assert entry.schema == []

    def test_sir_manifest_entry_roundtrip(self):
        from hydro_param.manifest import SIRManifestEntry

        entry = SIRManifestEntry(
            static_files={"elevation_m_mean": "sir/elevation_m_mean.csv"},
            temporal_files={"gridmet_2020": "sir/gridmet_2020.nc"},
            schema=[{"name": "elevation_m_mean", "units": "m", "statistic": "mean"}],
        )
        data = entry.model_dump(mode="json")
        restored = SIRManifestEntry(**data)
        assert restored.static_files == entry.static_files
        assert restored.temporal_files == entry.temporal_files
        assert restored.schema == entry.schema

    def test_manifest_version_2_with_sir(self, tmp_path):
        from hydro_param.manifest import PipelineManifest, SIRManifestEntry

        sir = SIRManifestEntry(
            static_files={"elevation_m_mean": "sir/elevation_m_mean.csv"},
        )
        manifest = PipelineManifest(version=2, sir=sir)
        manifest.save(tmp_path)
        from hydro_param.manifest import load_manifest

        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.sir is not None
        assert loaded.sir.static_files == {"elevation_m_mean": "sir/elevation_m_mean.csv"}

    def test_manifest_version_1_has_no_sir(self, tmp_path):
        """Version 1 manifests loaded as version 2 should have sir=None."""
        from hydro_param.manifest import PipelineManifest

        manifest = PipelineManifest(sir=None)
        manifest.save(tmp_path)
        from hydro_param.manifest import load_manifest

        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert loaded.sir is None

    def test_manifest_atomic_write(self, tmp_path):
        """Manifest save should be atomic (no partial writes)."""
        from hydro_param.manifest import PipelineManifest, SIRManifestEntry

        sir = SIRManifestEntry(static_files={"a": "sir/a.csv"})
        manifest = PipelineManifest(sir=sir)
        manifest.save(tmp_path)
        # File should exist and be valid
        from hydro_param.manifest import load_manifest

        assert load_manifest(tmp_path) is not None
        # No .tmp file should remain
        assert not (tmp_path / ".manifest.yml.tmp").exists()
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_manifest.py::TestSIRManifestEntry -v`
Expected: FAIL — `SIRManifestEntry` does not exist yet.

**Step 3: Implement SIRManifestEntry and update PipelineManifest**

In `src/hydro_param/manifest.py`:

- Add `SIRManifestEntry` Pydantic model after `ManifestEntry` (after line 87):

```python
class SIRManifestEntry(BaseModel):
    """Track normalized SIR output from stage 5.

    Records the file paths, schema metadata, and completion time for
    the SIR normalization step.  Used by Phase 2 (model plugins) to
    discover what the pipeline produced without re-running it.

    Attributes
    ----------
    static_files : dict[str, str]
        Mapping of SIR variable names to file paths relative to the
        output directory (e.g., ``{"elevation_m_mean": "sir/elevation_m_mean.csv"}``).
    temporal_files : dict[str, str]
        Mapping of temporal dataset keys to file paths relative to the
        output directory (e.g., ``{"gridmet_2020": "sir/gridmet_2020.nc"}``).
    schema : list[dict]
        Serialized SIR variable schema entries from ``build_sir_schema()``.
    completed_at : datetime
        UTC timestamp when SIR normalization completed.
    """

    static_files: dict[str, str] = {}
    temporal_files: dict[str, str] = {}
    schema: list[dict] = []
    completed_at: datetime = datetime.min

    @field_validator("completed_at", mode="before")
    @classmethod
    def _parse_completed_at(cls, v: object) -> object:
        if isinstance(v, str):
            return datetime.fromisoformat(v) if v else datetime.min
        return v
```

- Change `_SUPPORTED_VERSION` from 1 to 2 (line 48).

- Add `sir` field to `PipelineManifest`:

```python
class PipelineManifest(BaseModel):
    version: int = _SUPPORTED_VERSION
    fabric_fingerprint: str = ""
    entries: dict[str, ManifestEntry] = {}
    sir: SIRManifestEntry | None = None  # NEW
```

- Update `_check_version` to accept both 1 and 2:

```python
@field_validator("version")
@classmethod
def _check_version(cls, v: int) -> int:
    if v not in (1, 2):
        raise ValueError(f"Unsupported manifest version {v} (expected 1 or 2)")
    return v
```

- Make `save()` atomic (write to `.tmp`, rename):

```python
def save(self, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME
    tmp_path = output_dir / f"{MANIFEST_FILENAME}.tmp"
    data = self.model_dump(mode="json")
    tmp_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    tmp_path.replace(manifest_path)
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_manifest.py::TestSIRManifestEntry -v`
Expected: PASS

**Step 5: Wire stage 5 to write SIR manifest entry**

In `src/hydro_param/pipeline.py`, at the end of `stage5_normalize_sir()` (after line 1408), populate the manifest SIR section. This requires passing the manifest into stage 5 or returning the SIR entry for the caller to write.

Preferred approach: return the `SIRManifestEntry` from stage 5, and have `run_pipeline_from_config()` save it to the manifest before returning.

In `pipeline.py:stage5_normalize_sir` — change return type to include `SIRManifestEntry`:

```python
from hydro_param.manifest import SIRManifestEntry

def stage5_normalize_sir(...) -> tuple[dict[str, Path], list[SIRVariableSchema], list[SIRValidationWarning], SIRManifestEntry]:
    # ... existing code ...

    sir_manifest = SIRManifestEntry(
        static_files={k: str(v.relative_to(config.output.path)) for k, v in sir_files.items()
                      if str(v).endswith(".csv")},
        temporal_files={k: str(v.relative_to(config.output.path)) for k, v in sir_files.items()
                        if str(v).endswith(".nc")},
        schema=[s.__dict__ if hasattr(s, "__dict__") else s for s in schema],
        completed_at=datetime.now(timezone.utc),
    )
    return sir_files, schema, warnings, sir_manifest
```

In `pipeline.py:run_pipeline_from_config` — after stage 5, save SIR manifest:

```python
sir_files, sir_schema, sir_warnings, sir_manifest_entry = stage5_normalize_sir(...)
# Save SIR info to manifest
manifest.sir = sir_manifest_entry
_save_manifest_to_disk(manifest, config.output.path)
```

**Step 6: Run full manifest test suite**

Run: `pixi run -e dev pytest tests/test_manifest.py -v`
Expected: PASS (existing tests + new tests)

**Step 7: Commit**

```bash
git add src/hydro_param/manifest.py src/hydro_param/pipeline.py tests/test_manifest.py
git commit -m "feat: extend manifest with SIR tracking (version 2) (#117)"
```

---

### Task 2: Create SIRAccessor

**Files:**
- Create: `src/hydro_param/sir_accessor.py`
- Test: `tests/test_sir_accessor.py`

**Step 1: Write failing tests for SIRAccessor**

Create `tests/test_sir_accessor.py`:

```python
"""Tests for SIRAccessor lazy variable loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture()
def sir_dir(tmp_path: Path) -> Path:
    """Create a minimal SIR directory with CSV and NC files."""
    sir = tmp_path / "sir"
    sir.mkdir()
    # Static CSV
    df = pd.DataFrame({"elevation_m_mean": [100.0, 200.0, 300.0]}, index=pd.Index([1, 2, 3], name="nhm_id"))
    df.to_csv(sir / "elevation_m_mean.csv")
    # Temporal NC
    ds = xr.Dataset({"pr": xr.DataArray(np.ones((3, 365)), dims=["nhm_id", "time"])})
    ds.to_netcdf(sir / "gridmet_2020.nc")
    return tmp_path


@pytest.fixture()
def sir_dir_with_manifest(sir_dir: Path) -> Path:
    """SIR directory with a valid manifest."""
    from hydro_param.manifest import PipelineManifest, SIRManifestEntry

    sir_entry = SIRManifestEntry(
        static_files={"elevation_m_mean": "sir/elevation_m_mean.csv"},
        temporal_files={"gridmet_2020": "sir/gridmet_2020.nc"},
        schema=[{"name": "elevation_m_mean", "units": "m", "statistic": "mean"}],
    )
    manifest = PipelineManifest(sir=sir_entry)
    manifest.save(sir_dir)
    return sir_dir


class TestSIRAccessor:
    def test_from_manifest(self, sir_dir_with_manifest):
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "elevation_m_mean" in acc.available_variables()
        assert "gridmet_2020" in acc.available_temporal()

    def test_load_variable(self, sir_dir_with_manifest):
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        da = acc.load_variable("elevation_m_mean")
        assert isinstance(da, xr.DataArray)
        assert len(da) == 3
        assert float(da.values[0]) == 100.0

    def test_load_temporal(self, sir_dir_with_manifest):
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        ds = acc.load_temporal("gridmet_2020")
        assert isinstance(ds, xr.Dataset)
        assert "pr" in ds

    def test_missing_variable_raises(self, sir_dir_with_manifest):
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        with pytest.raises(KeyError, match="no_such_var"):
            acc.load_variable("no_such_var")

    def test_glob_fallback_no_manifest(self, sir_dir):
        """Without manifest, SIRAccessor falls back to globbing sir/."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir)
        assert "elevation_m_mean" in acc.available_variables()
        assert "gridmet_2020" in acc.available_temporal()

    def test_contains_check(self, sir_dir_with_manifest):
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "elevation_m_mean" in acc
        assert "no_such_var" not in acc

    def test_getitem(self, sir_dir_with_manifest):
        """SIRAccessor[name] loads a variable (Dataset-compatible API)."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        da = acc["elevation_m_mean"]
        assert isinstance(da, xr.DataArray)

    def test_data_vars_property(self, sir_dir_with_manifest):
        """data_vars returns available static variable names."""
        from hydro_param.sir_accessor import SIRAccessor

        acc = SIRAccessor(sir_dir_with_manifest)
        assert "elevation_m_mean" in acc.data_vars

    def test_missing_file_raises_at_init(self, tmp_path):
        """If manifest references missing files, fail at init."""
        from hydro_param.manifest import PipelineManifest, SIRManifestEntry

        sir_entry = SIRManifestEntry(
            static_files={"ghost": "sir/ghost.csv"},
        )
        manifest = PipelineManifest(sir=sir_entry)
        manifest.save(tmp_path)
        from hydro_param.sir_accessor import SIRAccessor

        with pytest.raises(FileNotFoundError, match="ghost"):
            SIRAccessor(tmp_path)
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir_accessor.py -v`
Expected: FAIL — `sir_accessor` module does not exist.

**Step 3: Implement SIRAccessor**

Create `src/hydro_param/sir_accessor.py`:

```python
"""Lazy accessor for SIR files backed by a pipeline manifest.

Provides per-variable on-demand loading from the SIR output directory.
No data is held in memory between calls — each ``load_variable()`` reads
from disk and the caller is responsible for releasing.

Supports a Dataset-compatible API (``__contains__``, ``__getitem__``,
``data_vars``) so derivation steps can check variable availability and
load data with minimal code changes.

When the manifest is missing or corrupt, falls back to discovering SIR
files by globbing the ``sir/`` subdirectory with a warning.

See Also
--------
hydro_param.manifest : Manifest schema that tracks SIR output.
hydro_param.plugins.DerivationContext : Consumer of SIRAccessor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from hydro_param.manifest import load_manifest

logger = logging.getLogger(__name__)


class SIRAccessor:
    """Lazy accessor for SIR files backed by a pipeline manifest.

    Does not hold data in memory.  Each call to ``load_variable()``
    reads from disk; the caller releases when done.

    Parameters
    ----------
    output_dir : Path
        Pipeline output directory containing ``.manifest.yml`` and
        ``sir/`` subdirectory.

    Raises
    ------
    FileNotFoundError
        If a file referenced in the manifest does not exist on disk.

    Notes
    -----
    Implements ``__contains__`` and ``__getitem__`` for compatibility
    with derivation steps that check ``"var_name" in sir`` and access
    ``sir["var_name"]``.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._schema: list[dict] = []

        manifest = load_manifest(self._output_dir)
        if manifest is not None and manifest.sir is not None:
            self._static = dict(manifest.sir.static_files)
            self._temporal = dict(manifest.sir.temporal_files)
            self._schema = list(manifest.sir.schema)
        else:
            logger.warning(
                "No valid manifest with SIR section at %s — discovering SIR "
                "files by scanning sir/. Schema metadata will not be "
                "available. Consider re-running 'hydro-param run pipeline.yml' "
                "to regenerate the manifest.",
                self._output_dir,
            )
            self._static = _glob_sir_static(self._output_dir / "sir")
            self._temporal = _glob_sir_temporal(self._output_dir / "sir")

        self._validate_files()

    def _validate_files(self) -> None:
        """Verify all referenced files exist on disk.

        Raises
        ------
        FileNotFoundError
            If any referenced file is missing.
        """
        for name, rel_path in self._static.items():
            full = self._output_dir / rel_path
            if not full.exists():
                raise FileNotFoundError(
                    f"SIR static file for '{name}' not found at {full}. "
                    f"Re-run 'hydro-param run pipeline.yml' to regenerate."
                )
        for name, rel_path in self._temporal.items():
            full = self._output_dir / rel_path
            if not full.exists():
                raise FileNotFoundError(
                    f"SIR temporal file for '{name}' not found at {full}. "
                    f"Re-run 'hydro-param run pipeline.yml' to regenerate."
                )

    def available_variables(self) -> list[str]:
        """List all static SIR variable names.

        Returns
        -------
        list[str]
            Variable names available for ``load_variable()``.
        """
        return list(self._static.keys())

    def available_temporal(self) -> list[str]:
        """List all temporal SIR dataset keys.

        Returns
        -------
        list[str]
            Dataset keys available for ``load_temporal()``.
        """
        return list(self._temporal.keys())

    @property
    def data_vars(self) -> list[str]:
        """Static variable names (Dataset-compatible API).

        Returns
        -------
        list[str]
            Same as ``available_variables()``.
        """
        return self.available_variables()

    @property
    def schema(self) -> list[dict]:
        """SIR variable schema metadata.

        Returns
        -------
        list[dict]
            Schema entries from the manifest, or empty list if
            discovered via glob fallback.
        """
        return self._schema

    def load_variable(self, name: str) -> xr.DataArray:
        """Load a single SIR variable from disk.

        Parameters
        ----------
        name : str
            SIR variable name (e.g., ``"elevation_m_mean"``).

        Returns
        -------
        xr.DataArray
            Variable data with the fabric id field as dimension.

        Raises
        ------
        KeyError
            If the variable name is not in the SIR.
        """
        if name not in self._static:
            raise KeyError(
                f"SIR variable '{name}' not found. "
                f"Available: {sorted(self._static.keys())}"
            )
        path = self._output_dir / self._static[name]
        df = pd.read_csv(path, index_col=0)
        ds = xr.Dataset.from_dataframe(df)
        if name in ds:
            return ds[name]
        # Single-column CSV: return the first (only) variable
        return next(iter(ds.data_vars.values()))

    def load_temporal(self, name: str) -> xr.Dataset:
        """Load a single temporal SIR file from disk.

        Parameters
        ----------
        name : str
            Temporal dataset key (e.g., ``"gridmet_2020"``).

        Returns
        -------
        xr.Dataset
            Temporal dataset.  Caller should close when done.

        Raises
        ------
        KeyError
            If the temporal key is not in the SIR.
        """
        if name not in self._temporal:
            raise KeyError(
                f"SIR temporal dataset '{name}' not found. "
                f"Available: {sorted(self._temporal.keys())}"
            )
        path = self._output_dir / self._temporal[name]
        return xr.open_dataset(path)

    def __contains__(self, name: object) -> bool:
        """Check if a variable name is available (Dataset-compatible API)."""
        return isinstance(name, str) and name in self._static

    def __getitem__(self, name: str) -> xr.DataArray:
        """Load a variable by name (Dataset-compatible API).

        Equivalent to ``load_variable(name)``.
        """
        return self.load_variable(name)


def _glob_sir_static(sir_dir: Path) -> dict[str, str]:
    """Discover static SIR files by globbing CSV files.

    Parameters
    ----------
    sir_dir : Path
        The ``sir/`` subdirectory to scan.

    Returns
    -------
    dict[str, str]
        Mapping of variable names (stem) to paths relative to the
        parent output directory.
    """
    if not sir_dir.is_dir():
        return {}
    return {
        p.stem: str(p.relative_to(sir_dir.parent))
        for p in sorted(sir_dir.glob("*.csv"))
    }


def _glob_sir_temporal(sir_dir: Path) -> dict[str, str]:
    """Discover temporal SIR files by globbing NetCDF files.

    Parameters
    ----------
    sir_dir : Path
        The ``sir/`` subdirectory to scan.

    Returns
    -------
    dict[str, str]
        Mapping of dataset keys (stem) to paths relative to the
        parent output directory.
    """
    if not sir_dir.is_dir():
        return {}
    return {
        p.stem: str(p.relative_to(sir_dir.parent))
        for p in sorted(sir_dir.glob("*.nc"))
    }
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir_accessor.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir_accessor.py tests/test_sir_accessor.py
git commit -m "feat: add SIRAccessor for lazy per-variable loading (#117)"
```

---

### Task 3: Update DerivationContext and Plugin Protocol

**Files:**
- Modify: `src/hydro_param/plugins.py:38-147` (`DerivationContext`), `:149-193` (`DerivationPlugin`)
- Test: `tests/test_plugins.py`

**Step 1: Write failing tests for SIRAccessor-based DerivationContext**

Add to `tests/test_plugins.py`:

```python
class TestDerivationContextWithAccessor:
    """Test DerivationContext with SIRAccessor instead of xr.Dataset."""

    def test_context_accepts_sir_accessor(self, tmp_path):
        from hydro_param.manifest import PipelineManifest, SIRManifestEntry
        from hydro_param.sir_accessor import SIRAccessor

        # Create minimal SIR
        sir_dir = tmp_path / "sir"
        sir_dir.mkdir()
        df = pd.DataFrame(
            {"val": [1.0, 2.0]},
            index=pd.Index([1, 2], name="nhm_id"),
        )
        df.to_csv(sir_dir / "val.csv")
        manifest = PipelineManifest(
            sir=SIRManifestEntry(static_files={"val": "sir/val.csv"})
        )
        manifest.save(tmp_path)

        acc = SIRAccessor(tmp_path)
        ctx = DerivationContext(sir=acc, fabric_id_field="nhm_id")
        assert "val" in ctx.sir

    def test_context_rejects_none_sir(self):
        with pytest.raises(TypeError):
            DerivationContext(sir=None)
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_plugins.py::TestDerivationContextWithAccessor -v`
Expected: FAIL — type annotation mismatch or validation error.

**Step 3: Update DerivationContext**

In `src/hydro_param/plugins.py`:

- Change `sir` type annotation from `xr.Dataset` to `SIRAccessor`:

```python
from hydro_param.sir_accessor import SIRAccessor

@dataclass(frozen=True)
class DerivationContext:
    sir: SIRAccessor
    temporal: dict[str, xr.Dataset] | None = None
    fabric: gpd.GeoDataFrame | None = None
    ...
```

- Update `__post_init__` validation: remove the check for `fabric_id_field` in `sir.dims` (SIRAccessor doesn't have dims — it loads per-variable).

- Update the `DerivationPlugin` protocol docstrings to reference `SIRAccessor`.

**Step 4: Fix existing test fixtures**

Update `tests/test_plugins.py` existing `TestDerivationContext` tests to create an `SIRAccessor` instead of `xr.Dataset`. Tests that created `xr.Dataset({"val": ...}, coords={"nhm_id": ...})` need to write CSV fixtures to tmp_path and create an SIRAccessor.

**Step 5: Run full plugin test suite**

Run: `pixi run -e dev pytest tests/test_plugins.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/hydro_param/plugins.py tests/test_plugins.py
git commit -m "refactor: change DerivationContext.sir to SIRAccessor (#117)"
```

---

### Task 4: Update Derivation Steps to Use SIRAccessor

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (all `_derive_*` methods)
- Test: `tests/test_pywatershed_derivation.py`

This is the largest task. Each derivation step currently does `sir = ctx.sir` and then accesses it like an `xr.Dataset`. The SIRAccessor's `__contains__` and `__getitem__` support means most access patterns work unchanged, but `sir.sizes`, `sir.coords`, and `sir.data_vars` iteration need updating.

**Step 1: Update `derive()` entry point (lines 347-354)**

Current:
```python
sir = context.sir
id_field = context.fabric_id_field
nhru = sir.sizes.get(id_field, 0)
ds = xr.Dataset()
if id_field in sir.coords:
    ds = ds.assign_coords(nhru=sir[id_field].values)
```

New — derive `nhru` from fabric (which is always loaded) instead of SIR:
```python
sir = context.sir
id_field = context.fabric_id_field
fabric = context.fabric
nhru = len(fabric) if fabric is not None else 0
ds = xr.Dataset()
if fabric is not None and id_field in fabric.columns:
    ds = ds.assign_coords(nhru=fabric[id_field].values)
```

This is cleaner — the HRU count and IDs come from the authoritative fabric, not the SIR.

**Step 2: Update `_derive_geometry()` (line 449-455)**

Current:
```python
sir = ctx.sir
...
hru_ids = sir[id_field].values if id_field in sir.coords else None
```

New — get HRU IDs from fabric:
```python
fabric = ctx.fabric
...
hru_ids = fabric[id_field].values if fabric is not None and id_field in fabric.columns else None
```

**Step 3: Update `_derive_topography()` (lines 1306-1330)**

Current pattern works with SIRAccessor — `"elevation_m_mean" in sir` and `sir["elevation_m_mean"].values` both work via `__contains__` and `__getitem__`. No changes needed.

**Step 4: Update `_derive_landcover()` (lines 1386-1484)**

Current pattern for fraction discovery:
```python
sir = ctx.sir
fraction_vars = sorted(str(v) for v in sir.data_vars if str(v).startswith(prefix))
```

New — use `SIRAccessor.data_vars` property:
```python
sir = ctx.sir
fraction_vars = sorted(v for v in sir.data_vars if v.startswith(prefix))
```

This works because `SIRAccessor.data_vars` returns `list[str]`. The only change is removing the `str(v)` cast since they're already strings.

**Step 5: Update `_derive_soils()` (lines 1574-1649)**

Same pattern as landcover — `sir.data_vars` iteration for fraction prefixes. Same fix.

**Step 6: Update `_derive_waterbody()` (line 1877)**

Current:
```python
hru_ids = ctx.sir[id_field].values
```

New — get from fabric:
```python
hru_ids = ctx.fabric[id_field].values
```

**Step 7: Update `_derive_forcing()` (lines 2396-2406)**

Current uses `ctx.temporal` (separate from `ctx.sir`). No changes needed — temporal is still a `dict[str, xr.Dataset]` on DerivationContext.

**Step 8: Update `_compute_monthly_normals()` (lines 2579-2588)**

Same — uses `ctx.temporal`, not `ctx.sir`. No changes needed.

**Step 9: Update test fixtures in `tests/test_pywatershed_derivation.py`**

Every test that creates an `xr.Dataset` as the `sir` argument to `DerivationContext` needs to be updated to write CSV files to `tmp_path` and create an `SIRAccessor`. Create a shared fixture:

```python
@pytest.fixture()
def make_sir_accessor(tmp_path):
    """Factory fixture to create SIRAccessor from a dict of DataArrays."""
    def _make(variables: dict[str, np.ndarray], index_name: str = "nhm_id",
              index_values: np.ndarray | None = None):
        sir_dir = tmp_path / "sir"
        sir_dir.mkdir(exist_ok=True)
        if index_values is None:
            first = next(iter(variables.values()))
            index_values = np.arange(1, len(first) + 1)
        idx = pd.Index(index_values, name=index_name)
        static_files = {}
        for name, values in variables.items():
            df = pd.DataFrame({name: values}, index=idx)
            path = sir_dir / f"{name}.csv"
            df.to_csv(path)
            static_files[name] = f"sir/{name}.csv"
        from hydro_param.manifest import PipelineManifest, SIRManifestEntry
        manifest = PipelineManifest(
            sir=SIRManifestEntry(static_files=static_files)
        )
        manifest.save(tmp_path)
        from hydro_param.sir_accessor import SIRAccessor
        return SIRAccessor(tmp_path)
    return _make
```

**Step 10: Run derivation tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: PASS

**Step 11: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "refactor: update derivation steps for SIRAccessor (#117)"
```

---

### Task 5: Simplify PywatershedRunConfig (v3.0)

**Files:**
- Modify: `src/hydro_param/pywatershed_config.py:38-420`
- Test: `tests/test_pywatershed_config.py`

**Step 1: Write failing tests for v3.0 config**

Add to `tests/test_pywatershed_config.py`:

```python
class TestPywatershedRunConfigV3:
    """Test the simplified v3.0 config schema."""

    def test_minimal_v3_config(self, tmp_path):
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        config_data = {
            "target_model": "pywatershed",
            "version": "3.0",
            "domain": {"fabric_path": str(fabric), "id_field": "nhm_id"},
            "time": {"start": "2020-01-01", "end": "2020-12-31"},
        }
        from hydro_param.pywatershed_config import PywatershedRunConfig
        cfg = PywatershedRunConfig(**config_data)
        assert cfg.sir_path == Path("output")
        assert cfg.version == "3.0"

    def test_sir_path_override(self, tmp_path):
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        config_data = {
            "target_model": "pywatershed",
            "version": "3.0",
            "domain": {"fabric_path": str(fabric)},
            "time": {"start": "2020-01-01", "end": "2020-12-31"},
            "sir_path": "/custom/sir/output",
        }
        from hydro_param.pywatershed_config import PywatershedRunConfig
        cfg = PywatershedRunConfig(**config_data)
        assert cfg.sir_path == Path("/custom/sir/output")

    def test_rejects_old_datasets_field(self, tmp_path):
        """v3.0 should not accept datasets field."""
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        config_data = {
            "target_model": "pywatershed",
            "version": "3.0",
            "domain": {"fabric_path": str(fabric)},
            "time": {"start": "2020-01-01", "end": "2020-12-31"},
            "datasets": {"topography": "dem_3dep_10m"},
        }
        from hydro_param.pywatershed_config import PywatershedRunConfig
        with pytest.raises(ValidationError):
            PywatershedRunConfig(**config_data)

    def test_simplified_domain(self, tmp_path):
        """Domain should only have file paths and id fields."""
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        segments = tmp_path / "nseg.gpkg"
        segments.touch()
        domain_data = {
            "fabric_path": str(fabric),
            "segment_path": str(segments),
            "id_field": "nhm_id",
            "segment_id_field": "nhm_seg",
        }
        from hydro_param.pywatershed_config import PwsDomainConfig
        domain = PwsDomainConfig(**domain_data)
        assert domain.fabric_path == fabric
        assert not hasattr(domain, "bbox")
        assert not hasattr(domain, "extraction_method")
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestPywatershedRunConfigV3 -v`
Expected: FAIL — old schema still in place.

**Step 3: Rewrite pywatershed_config.py**

Simplify `PwsDomainConfig` — keep only:
- `fabric_path: Path`
- `segment_path: Path | None = None`
- `waterbody_path: Path | None = None`
- `id_field: str = "nhm_id"`
- `segment_id_field: str = "nhm_seg"`

Delete:
- `source`, `gf_version`, `extraction_method`, `bbox`, `huc_id`, `pour_point`
- All validators related to extraction methods

Delete these classes entirely:
- `PwsClimateConfig`
- `PwsDatasetSources`
- `PwsProcessingConfig`

Simplify `PywatershedRunConfig`:
- Add `sir_path: Path = Path("output")`
- Remove `climate`, `datasets`, `processing` fields
- Use `model_config = ConfigDict(extra="forbid")` to reject unknown fields
- Bump default version to `"3.0"`

Keep unchanged:
- `PwsTimeConfig`
- `PwsParameterOverrides`
- `PwsCalibrationConfig`
- `PwsOutputConfig`
- `load_pywatershed_config()`

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py -v`
Expected: PASS (update or remove tests for deleted classes)

**Step 5: Commit**

```bash
git add src/hydro_param/pywatershed_config.py tests/test_pywatershed_config.py
git commit -m "refactor: simplify PywatershedRunConfig to v3.0 (#117)"
```

---

### Task 6: Rewrite pws_run_cmd()

**Files:**
- Modify: `src/hydro_param/cli.py:581-950` (delete `_translate_pws_to_pipeline`, rewrite `pws_run_cmd`)
- Test: `tests/test_cli.py`

**Step 1: Write integration test for decoupled pws_run_cmd**

Add to `tests/test_cli.py`:

```python
class TestPwsRunDecoupled:
    """Test Phase 2 pywatershed run consuming existing SIR."""

    @pytest.fixture()
    def project_with_sir(self, tmp_path):
        """Create a project with pre-built SIR output."""
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import box

        # Fabric
        fabrics_dir = tmp_path / "data" / "fabrics"
        fabrics_dir.mkdir(parents=True)
        gdf = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)]},
            crs="EPSG:4326",
        )
        gdf.to_file(fabrics_dir / "nhru.gpkg", driver="GPKG")

        # SIR output
        output_dir = tmp_path / "output"
        sir_dir = output_dir / "sir"
        sir_dir.mkdir(parents=True)

        idx = pd.Index([1, 2], name="nhm_id")
        for var, vals in [
            ("elevation_m_mean", [100.0, 200.0]),
            ("slope_deg_mean", [5.0, 10.0]),
            ("aspect_deg_mean", [180.0, 90.0]),
        ]:
            pd.DataFrame({var: vals}, index=idx).to_csv(sir_dir / f"{var}.csv")

        # Manifest
        from hydro_param.manifest import PipelineManifest, SIRManifestEntry
        sir_entry = SIRManifestEntry(
            static_files={
                "elevation_m_mean": "sir/elevation_m_mean.csv",
                "slope_deg_mean": "sir/slope_deg_mean.csv",
                "aspect_deg_mean": "sir/aspect_deg_mean.csv",
            },
        )
        PipelineManifest(sir=sir_entry).save(output_dir)

        # Config
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_text = f"""\
target_model: pywatershed
version: "3.0"
domain:
  fabric_path: "{fabrics_dir / 'nhru.gpkg'}"
  id_field: nhm_id
time:
  start: "2020-01-01"
  end: "2020-12-31"
sir_path: "{output_dir}"
output:
  path: "{tmp_path / 'models' / 'pywatershed'}"
"""
        (config_dir / "pywatershed_run.yml").write_text(config_text)
        return tmp_path

    def test_phase2_loads_sir_from_disk(self, project_with_sir):
        """Phase 2 should not run Phase 1 — just load SIR from manifest."""
        # This test verifies the decoupled flow works end-to-end.
        # Full integration test; mark as slow if needed.
        pass  # Placeholder — fill in during implementation
```

**Step 2: Delete `_translate_pws_to_pipeline()` (lines 581-738)**

Remove the entire function. Remove its imports.

**Step 3: Rewrite `pws_run_cmd()` (lines 746-950)**

New flow:
1. Load `PywatershedRunConfig` (v3.0)
2. Resolve `sir_path` to absolute
3. Create `SIRAccessor(sir_path)`
4. Load fabric, segments, waterbodies from `config.domain.*_path`
5. Build temporal dict from `SIRAccessor.available_temporal()`
6. Build `DerivationContext`
7. Run `PywatershedDerivation().derive(ctx)`
8. `merge_temporal_into_derived()`
9. `PywatershedFormatter().write()`

```python
def pws_run_cmd(config: Path, *, registry: Path | None = None) -> None:
    pws_config = load_pywatershed_config(config)

    # Resolve SIR path
    sir_path = pws_config.sir_path
    if not sir_path.is_absolute():
        sir_path = config.parent / sir_path
    sir_path = sir_path.resolve()

    # Load SIR accessor (lazy)
    from hydro_param.sir_accessor import SIRAccessor
    try:
        sir = SIRAccessor(sir_path)
    except FileNotFoundError as exc:
        logger.error("SIR output not found: %s", exc)
        logger.error("Run 'hydro-param run pipeline.yml' first to produce SIR output.")
        raise SystemExit(1) from exc

    # Load fabric
    fabric = gpd.read_file(pws_config.domain.fabric_path)

    # Load optional segments/waterbodies
    segments = None
    if pws_config.domain.segment_path:
        segments = gpd.read_file(pws_config.domain.segment_path)
    waterbodies = None
    if pws_config.domain.waterbody_path:
        waterbodies = gpd.read_file(pws_config.domain.waterbody_path)

    # Build derivation context
    derivation_config = {}
    if pws_config.parameter_overrides.values:
        derivation_config["parameter_overrides"] = {
            "values": pws_config.parameter_overrides.values,
        }

    ctx = DerivationContext(
        sir=sir,
        fabric=fabric,
        segments=segments,
        waterbodies=waterbodies,
        fabric_id_field=pws_config.domain.id_field,
        segment_id_field=pws_config.domain.segment_id_field,
        config=derivation_config,
    )

    # Derive
    plugin = PywatershedDerivation()
    derived = plugin.derive(ctx)

    # Load and merge temporal data
    temporal = {}
    for name in sir.available_temporal():
        temporal[name] = sir.load_temporal(name)
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

    # Format and write
    formatter = get_formatter("pywatershed")
    formatter_config = {
        "parameter_file": pws_config.output.parameter_file,
        "forcing_dir": pws_config.output.forcing_dir,
        "soltab_file": pws_config.output.soltab_file,
        "control_file": pws_config.output.control_file,
        "start": pws_config.time.start,
        "end": pws_config.time.end,
    }
    formatter.write(derived, pws_config.output.path, formatter_config)

    logger.info("pywatershed model setup complete: %s", pws_config.output.path)
```

**Step 4: Remove dead imports**

Remove imports for `PipelineConfig`, `run_pipeline_from_config`, and any Phase 1 types no longer needed by `pws_run_cmd`.

**Step 5: Run CLI tests**

Run: `pixi run -e dev pytest tests/test_cli.py -v`
Expected: PASS (update or remove tests for `_translate_pws_to_pipeline`)

**Step 6: Commit**

```bash
git add src/hydro_param/cli.py tests/test_cli.py
git commit -m "refactor: rewrite pws_run_cmd for decoupled Phase 2 (#117)"
```

---

### Task 7: Update Init Template

**Files:**
- Modify: `src/hydro_param/project.py:242-350` (`generate_pywatershed_template`), `:508-517` (Next steps)
- Test: `tests/test_project.py` (if exists)

**Step 1: Update `generate_pywatershed_template()` to v3.0**

Replace the template string with the simplified v3.0 config (see design doc Section 4).

**Step 2: Update "Next steps" output in `init_project()`**

Change lines 508-517 to show the two-phase workflow:

```python
print("\nNext steps:")
print("  1. Place your fabric files in data/fabrics/")
print("  2. Edit configs/pipeline.yml (dataset sources, domain)")
print("  3. Edit configs/pywatershed_run.yml (time period, output options)")
print("  4. Run the pipeline:")
print("       hydro-param run configs/pipeline.yml")
print("  5. Run pywatershed parameterization:")
print("       hydro-param pywatershed run configs/pywatershed_run.yml")
```

**Step 3: Run project tests**

Run: `pixi run -e dev pytest tests/test_project.py -v` (if exists, otherwise test manually with `hydro-param init /tmp/test-project`)

**Step 4: Commit**

```bash
git add src/hydro_param/project.py
git commit -m "refactor: update init template for two-phase workflow (#117)"
```

---

### Task 8: Run Full Test Suite and Fix Breakage

**Step 1: Run all tests**

Run: `pixi run -e dev test`

**Step 2: Fix any remaining test failures**

Common expected failures:
- Tests that import deleted config classes (`PwsDatasetSources`, etc.)
- Tests that create `DerivationContext(sir=xr.Dataset(...))` — need SIRAccessor
- Tests for `_translate_pws_to_pipeline()` — delete these

**Step 3: Run lint, typecheck, pre-commit**

Run: `pixi run -e dev check`
Run: `pixi run -e dev pre-commit`

**Step 4: Commit fixes**

```bash
git commit -m "test: fix remaining test breakage from decoupling (#117)"
```

---

### Task 9: Final Verification

**Step 1: Run full check suite**

Run: `pixi run -e dev check`
Expected: All green (lint, format, typecheck, tests)

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Verify test count**

Run: `pixi run -e dev pytest --co -q | tail -1`
Verify test count is reasonable (should be close to current ~635, minus deleted tests, plus new tests).

**Step 4: Final commit if needed, then prepare for PR**

---

## Task Dependency Graph

```
Task 1 (manifest) ─┐
                    ├─ Task 2 (SIRAccessor) ─┐
                    │                         ├─ Task 3 (DerivationContext) ─┐
                    │                         │                              ├─ Task 4 (derivation steps)
                    │                         │                              │
Task 5 (config) ────┤                         │                              │
                    │                         │                              │
                    └─────────────────────────┴──────────────────────────────┴─ Task 6 (CLI rewrite)
                                                                                │
Task 7 (init template) ────────────────────────────────────────────────────────┘
                                                                                │
Task 8 (fix breakage) ─────────────────────────────────────────────────────────┘
                                                                                │
Task 9 (final verification) ───────────────────────────────────────────────────┘
```

Tasks 1 and 5 can run in parallel.
Tasks 2 depends on 1.
Task 3 depends on 2.
Tasks 4 and 6 depend on 3 and 5.
Task 7 depends on 5.
Task 8 depends on all prior tasks.
Task 9 depends on 8.
