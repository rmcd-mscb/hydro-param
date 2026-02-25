# Temporal SIR Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Normalize temporal NetCDF output through the SIR layer and fix categorical validation false positives.

**Architecture:** Extend `sir.py` with temporal normalization (rename + unit convert), expand the unit table, and wire it into `stage5_normalize_sir()` in `pipeline.py`. Fix categorical validation to skip non-fraction columns.

**Tech Stack:** xarray, numpy, pandas, pytest

---

### Task 1: Unit table expansion + K_to_C conversion

**Files:**
- Modify: `src/hydro_param/sir.py:29-42` (unit table)
- Modify: `src/hydro_param/sir.py:195-218` (apply_conversion)
- Test: `tests/test_sir.py`

**Step 1: Write failing tests for new units and K_to_C conversion**

Add to `tests/test_sir.py`:

```python
# In TestUnitAbbreviation:
def test_kelvin(self) -> None:
    assert unit_abbreviation("K") == "C"

def test_millimeters(self) -> None:
    assert unit_abbreviation("mm") == "mm"

def test_watts_per_m2(self) -> None:
    assert unit_abbreviation("W/m2") == "W_m2"

def test_kg_per_kg(self) -> None:
    assert unit_abbreviation("kg/kg") == "kg_kg"

def test_meters_per_second(self) -> None:
    assert unit_abbreviation("m/s") == "m_s"

# In TestCanonicalName:
def test_temperature_kelvin_converts_to_celsius(self) -> None:
    assert canonical_name("tmmx", "K", "mean") == "tmmx_C_mean"

# In TestApplyConversion:
def test_k_to_c(self) -> None:
    values = np.array([273.15, 283.15, 293.15])
    result = apply_conversion(values, "K_to_C")
    np.testing.assert_allclose(result, [0.0, 10.0, 20.0])

def test_k_to_c_with_nan(self) -> None:
    values = np.array([273.15, np.nan, 293.15])
    result = apply_conversion(values, "K_to_C")
    np.testing.assert_allclose(result[0], 0.0)
    assert np.isnan(result[1])
    np.testing.assert_allclose(result[2], 20.0)
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir.py -k "kelvin or millimeters or watts or kg_per_kg or meters_per_second or k_to_c or temperature_kelvin" -v`
Expected: FAIL

**Step 3: Add unit table entries and K_to_C conversion**

In `sir.py` `_UNIT_TABLE`, add:
```python
"K": ("C", "°C", "K_to_C"),
"mm": ("mm", "mm", None),
"W/m2": ("W_m2", "W/m2", None),
"kg/kg": ("kg_kg", "kg/kg", None),
"m/s": ("m_s", "m/s", None),
```

In `apply_conversion()`, add before the `raise`:
```python
if conversion == "K_to_C":
    return values - 273.15
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py -k "kelvin or millimeters or watts or kg_per_kg or meters_per_second or k_to_c or temperature_kelvin" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add MKS unit table entries and K_to_C conversion"
```

---

### Task 2: Add `temporal` field to SIRVariableSchema + build_sir_schema

**Files:**
- Modify: `src/hydro_param/sir.py:107-118` (SIRVariableSchema)
- Modify: `src/hydro_param/sir.py:121-192` (build_sir_schema)
- Test: `tests/test_sir.py`

**Step 1: Write failing tests**

```python
# In TestSIRVariableSchema:
def test_temporal_field_default_false(self) -> None:
    s = SIRVariableSchema(
        canonical_name="test", source_name="test", source_units="m",
        canonical_units="m", long_name="Test", categorical=False,
        valid_range=None, conversion=None,
    )
    assert s.temporal is False

def test_temporal_field_explicit(self) -> None:
    s = SIRVariableSchema(
        canonical_name="test", source_name="test", source_units="m",
        canonical_units="m", long_name="Test", categorical=False,
        valid_range=None, conversion=None, temporal=True,
    )
    assert s.temporal is True

# In TestBuildSIRSchema:
def test_temporal_dataset_marked(self) -> None:
    """Schema entries from temporal datasets have temporal=True."""
    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.sir import build_sir_schema

    entry = DatasetEntry(
        strategy="climr_cat",
        catalog_id="gridmet",
        temporal=True,
        t_coord="day",
        variables=[VariableSpec(name="pr", units="mm", long_name="Precipitation")],
        category="climate",
    )
    ds_req = DatasetRequest(
        name="gridmet",
        variables=["pr"],
        statistics=["mean"],
        time_period=["2020-01-01", "2020-12-31"],
    )
    var_specs = [VariableSpec(name="pr", units="mm", long_name="Precipitation")]
    resolved = [(entry, ds_req, var_specs)]
    schema = build_sir_schema(resolved)
    assert len(schema) == 1
    assert schema[0].temporal is True

def test_static_dataset_not_temporal(self) -> None:
    """Schema entries from static datasets have temporal=False."""
    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import VariableSpec
    from hydro_param.sir import build_sir_schema

    entry = self._make_entry(
        variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
    )
    ds_req = DatasetRequest(name="test", variables=["elevation"], statistics=["mean"])
    var_specs = [VariableSpec(name="elevation", units="m", long_name="Elevation")]
    resolved = [(entry, ds_req, var_specs)]
    schema = build_sir_schema(resolved)
    assert len(schema) == 1
    assert schema[0].temporal is False
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir.py -k "temporal_field or temporal_dataset_marked or static_dataset_not_temporal" -v`
Expected: FAIL

**Step 3: Implement**

Add `temporal: bool = False` to `SIRVariableSchema` dataclass.

In `build_sir_schema()`:
- Change type annotation from `tuple[object, ...]` to use `DatasetEntry` (import it).
- Read `_entry` → `entry` (no longer ignored).
- Check `hasattr(entry, 'temporal') and entry.temporal` to set `temporal=True` on
  schema entries. Use `hasattr` for backward compatibility with tests passing
  plain objects.

Pass `temporal=is_temporal` to every `SIRVariableSchema(...)` constructor call.

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py -k "temporal_field or temporal_dataset_marked or static_dataset_not_temporal" -v`
Expected: PASS

**Step 5: Run all existing tests to confirm no regressions**

Run: `pixi run -e dev pytest tests/test_sir.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add temporal flag to SIRVariableSchema"
```

---

### Task 3: Categorical validation fix

**Files:**
- Modify: `src/hydro_param/sir.py:434-473` (validate_sir)
- Test: `tests/test_sir.py`

**Step 1: Write failing test**

```python
# In TestValidateSIR:
def test_categorical_count_column_not_range_checked(self, tmp_path: Path) -> None:
    """Count columns in categorical CSVs should not trigger range warnings."""
    import pandas as pd
    from hydro_param.sir import SIRVariableSchema, validate_sir

    # Simulate NLCD categorical output with fraction + count columns
    df = pd.DataFrame({
        "lndcov_frac_11": [0.3, 0.5],
        "lndcov_frac_21": [0.7, 0.5],
        "count": [1000, 2000],  # pixel counts — NOT fractions
    }, index=[1, 2])
    df.index.name = "nhm_id"
    path = tmp_path / "lndcov_frac.csv"
    df.to_csv(path)

    schema = [SIRVariableSchema(
        canonical_name="lndcov_frac",
        source_name="LndCov",
        source_units="",
        canonical_units="",
        long_name="Land Cover",
        categorical=True,
        valid_range=(0.0, 1.0),
        conversion=None,
    )]

    warnings = validate_sir({"lndcov_frac": path}, schema)
    # count column values [1000, 2000] should NOT produce range warnings
    range_warnings = [w for w in warnings if w.check_type == "range"]
    assert len(range_warnings) == 0
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_sir.py::TestValidateSIR::test_categorical_count_column_not_range_checked -v`
Expected: FAIL (currently reports range warning for count column)

**Step 3: Fix validate_sir**

In `validate_sir()`, inside the column loop at line 441, add a filter for
categorical entries: only validate columns containing `_frac_` when the schema
entry is categorical.

```python
for col in df.columns:
    # For categorical entries, only validate fraction columns
    if matching and matching[0].categorical and "_frac_" not in col:
        continue
    # ... existing validation code
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_sir.py::TestValidateSIR::test_categorical_count_column_not_range_checked -v`
Expected: PASS

**Step 5: Run all validation tests**

Run: `pixi run -e dev pytest tests/test_sir.py::TestValidateSIR -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "fix: skip non-fraction columns in categorical SIR validation"
```

---

### Task 4: `normalize_sir_temporal()` function

**Files:**
- Modify: `src/hydro_param/sir.py` (add new function after `normalize_sir`)
- Test: `tests/test_sir.py`

**Step 1: Write failing test**

```python
class TestNormalizeSIRTemporal:
    """Tests for normalize_sir_temporal()."""

    def test_renames_variables_and_converts_units(self, tmp_path: Path) -> None:
        """Temporal normalization renames long names to canonical and converts K to °C."""
        import xarray as xr
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import DatasetEntry, VariableSpec
        from hydro_param.sir import (
            SIRVariableSchema,
            build_sir_schema,
            normalize_sir_temporal,
        )

        # Create a synthetic temporal NetCDF with gdptools-style long names
        ds = xr.Dataset(
            {
                "daily_maximum_temperature": (["time", "nhm_id"],
                    np.array([[300.0, 310.0], [305.0, 315.0]])),
                "precipitation_amount": (["time", "nhm_id"],
                    np.array([[5.0, 10.0], [3.0, 7.0]])),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=2),
                "nhm_id": [1, 2],
            },
        )
        nc_path = tmp_path / "input" / "gridmet_2020_temporal.nc"
        nc_path.parent.mkdir()
        ds.to_netcdf(nc_path)

        temporal_files = {"gridmet_2020": nc_path}

        entry = DatasetEntry(
            strategy="climr_cat",
            catalog_id="gridmet",
            temporal=True,
            t_coord="day",
            variables=[
                VariableSpec(name="tmmx", units="K",
                             long_name="daily_maximum_temperature"),
                VariableSpec(name="pr", units="mm",
                             long_name="precipitation_amount"),
            ],
            category="climate",
        )
        ds_req = DatasetRequest(
            name="gridmet",
            variables=["tmmx", "pr"],
            statistics=["mean"],
            time_period=["2020-01-01", "2020-12-31"],
        )
        var_specs = entry.variables
        resolved = [(entry, ds_req, var_specs)]
        schema = build_sir_schema(resolved)

        out_dir = tmp_path / "sir"
        result = normalize_sir_temporal(
            temporal_files=temporal_files,
            schema=schema,
            resolved=resolved,
            output_dir=out_dir,
        )

        # Should have produced normalized files
        assert len(result) > 0

        # Check temperature was converted K -> °C
        tmmx_files = {k: v for k, v in result.items() if "tmmx" in k}
        assert len(tmmx_files) == 1
        tmmx_path = list(tmmx_files.values())[0]
        out_ds = xr.open_dataset(tmmx_path)
        tmmx_key = [k for k in out_ds.data_vars if "tmmx" in k][0]
        np.testing.assert_allclose(
            out_ds[tmmx_key].values[0],
            [300.0 - 273.15, 310.0 - 273.15],
        )

        # Check precipitation passthrough (no conversion)
        pr_files = {k: v for k, v in result.items() if "pr" in k}
        assert len(pr_files) == 1
        pr_path = list(pr_files.values())[0]
        out_ds_pr = xr.open_dataset(pr_path)
        pr_key = [k for k in out_ds_pr.data_vars if "pr" in k][0]
        np.testing.assert_allclose(
            out_ds_pr[pr_key].values[0], [5.0, 10.0]
        )
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_sir.py::TestNormalizeSIRTemporal -v`
Expected: FAIL (function doesn't exist)

**Step 3: Implement `normalize_sir_temporal()`**

Add to `sir.py` after `normalize_sir()`:

```python
def normalize_sir_temporal(
    temporal_files: dict[str, Path],
    schema: list[SIRVariableSchema],
    resolved: Sequence[tuple[object, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
    output_dir: Path,
) -> dict[str, Path]:
    """Normalize temporal NetCDF files to canonical SIR format.

    Reads raw temporal NetCDFs from stage 4, renames variables from gdptools
    long names to canonical SIR names, applies unit conversions, and writes
    normalized per-variable NetCDFs.

    Parameters
    ----------
    temporal_files
        Mapping of dataset key (e.g. ``"gridmet_2020"``) to raw NetCDF path.
    schema
        SIR variable schema entries (from ``build_sir_schema()``).
    resolved
        Resolved dataset entries from stage 2.
    output_dir
        Directory to write normalized NetCDF files.

    Returns
    -------
    dict[str, Path]
        Mapping of canonical name to normalized NetCDF file path.
    """
    import xarray as xr

    output_dir.mkdir(parents=True, exist_ok=True)
    sir_files: dict[str, Path] = {}

    # Build reverse lookup: gdptools long_name -> (var_spec, schema_entries)
    # Only for temporal datasets.
    long_name_lookup: dict[str, tuple[VariableSpec, list[SIRVariableSchema]]] = {}
    for entry_obj, _ds_req, var_specs in resolved:
        if not (hasattr(entry_obj, "temporal") and entry_obj.temporal):
            continue
        for vs in var_specs:
            if isinstance(vs, VariableSpec) and vs.long_name:
                matching = [s for s in schema if s.source_name == vs.name and s.temporal]
                if matching:
                    long_name_lookup[vs.long_name] = (vs, matching)

    for file_key, nc_path in temporal_files.items():
        ds = xr.open_dataset(nc_path)

        for data_var in list(ds.data_vars):
            lookup = long_name_lookup.get(str(data_var))
            if lookup is None:
                logger.warning(
                    "No SIR schema match for temporal variable '%s' in %s — skipping",
                    data_var, nc_path.name,
                )
                continue

            var_spec, schema_entries = lookup
            # Use the first matching schema entry for conversion info
            schema_entry = schema_entries[0]

            # Apply unit conversion
            values = ds[data_var].values.astype(np.float64)
            if schema_entry.conversion is not None:
                values = apply_conversion(values, schema_entry.conversion)

            # Build canonical variable name
            cname = schema_entry.canonical_name

            # Create single-variable dataset with canonical name
            out_ds = xr.Dataset(
                {cname: (ds[data_var].dims, values)},
                coords=ds.coords,
            )
            out_path = output_dir / f"{cname}.nc"
            out_ds.to_netcdf(out_path)
            sir_files[cname] = out_path
            logger.info("SIR temporal normalized: %s/%s → %s", file_key, data_var, out_path.name)

        ds.close()

    return sir_files
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py::TestNormalizeSIRTemporal -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add normalize_sir_temporal for temporal NetCDF SIR normalization"
```

---

### Task 5: Pipeline integration

**Files:**
- Modify: `src/hydro_param/pipeline.py:930-979` (stage5_normalize_sir)
- Test: `tests/test_pipeline.py`

**Step 1: Write failing test**

```python
def test_stage5_normalize_sir_includes_temporal(tmp_path: Path) -> None:
    """stage5_normalize_sir normalizes temporal files alongside static."""
    import xarray as xr
    from hydro_param.pipeline import Stage4Results, stage5_normalize_sir
    from hydro_param.config import DatasetRequest, PipelineConfig
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    # Create minimal static file
    static_df = pd.DataFrame({"elevation": [100.0]}, index=[1])
    static_df.index.name = "nhm_id"
    static_path = tmp_path / "raw" / "elevation.csv"
    static_path.parent.mkdir()
    static_df.to_csv(static_path)

    # Create minimal temporal file
    ds = xr.Dataset(
        {"daily_maximum_temperature": (["time", "nhm_id"], np.array([[300.0]]))},
        coords={"time": pd.date_range("2020-01-01", periods=1), "nhm_id": [1]},
    )
    temporal_path = tmp_path / "climate" / "gridmet_2020_temporal.nc"
    temporal_path.parent.mkdir()
    ds.to_netcdf(temporal_path)

    stage4 = Stage4Results(
        static_files={"elevation": static_path},
        temporal_files={"gridmet_2020": temporal_path},
    )

    # Build resolved with both static and temporal
    static_entry = DatasetEntry(
        strategy="stac_cog", catalog_url="https://example.com",
        collection="test", variables=[
            VariableSpec(name="elevation", units="m", long_name="Elevation")
        ], category="topography",
    )
    temporal_entry = DatasetEntry(
        strategy="climr_cat", catalog_id="gridmet", temporal=True,
        t_coord="day", variables=[
            VariableSpec(name="tmmx", units="K",
                         long_name="daily_maximum_temperature"),
        ], category="climate",
    )
    resolved = [
        (static_entry,
         DatasetRequest(name="dem", variables=["elevation"], statistics=["mean"]),
         [VariableSpec(name="elevation", units="m", long_name="Elevation")]),
        (temporal_entry,
         DatasetRequest(name="gridmet", variables=["tmmx"], statistics=["mean"],
                        time_period=["2020-01-01", "2020-12-31"]),
         [VariableSpec(name="tmmx", units="K",
                       long_name="daily_maximum_temperature")]),
    ]

    # Minimal config
    config = ...  # construct a PipelineConfig with output.path = tmp_path, etc.
    # (Use existing test helpers or mock as needed)

    sir_files, schema, warnings = stage5_normalize_sir(stage4, resolved, config)

    # Temporal variable should be in sir_files
    temporal_keys = [k for k in sir_files if "tmmx" in k]
    assert len(temporal_keys) == 1
    # No missing warnings for temporal vars
    missing = [w for w in warnings if w.check_type == "missing" and "tmmx" in w.variable]
    assert len(missing) == 0
```

Note: The test may need adaptation to construct a valid `PipelineConfig` — use
existing test fixtures/helpers from the test file as a reference.

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_stage5_normalize_sir_includes_temporal -v`
Expected: FAIL

**Step 3: Implement pipeline integration**

In `pipeline.py`, update `stage5_normalize_sir()`:

1. Add import: `from hydro_param.sir import normalize_sir_temporal`
2. After the existing `normalize_sir()` call and before `validate_sir()`:

```python
# Normalize temporal files
if stage4.temporal_files:
    temporal_sir = normalize_sir_temporal(
        temporal_files=stage4.temporal_files,
        schema=schema,
        resolved=resolved,
        output_dir=sir_dir,
    )
    sir_files.update(temporal_sir)
    logger.info("  Normalized %d temporal SIR files", len(temporal_sir))
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_stage5_normalize_sir_includes_temporal -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pixi run -e dev pytest -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "feat: integrate temporal SIR normalization into stage 5"
```

---

### Task 6: Final verification

**Step 1: Run full test suite**

Run: `pixi run -e dev check`
Expected: ALL PASS (lint, format, typecheck, tests)

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: ALL PASS

**Step 3: Commit any lint/type fixes**

If needed:
```bash
git add -u
git commit -m "chore: lint and type fixes for temporal SIR normalization"
```
