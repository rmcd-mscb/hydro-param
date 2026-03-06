# SIR Variable Naming Convention Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the naming convention mismatch between SIR output (year-suffixed variable names) and pywatershed derivation code (base names), unblocking derivation steps 4, 5, 8, and 14.

**Architecture:** The fix has three parts: (1) add `load_dataset()` to SIRAccessor so multi-column CSVs can be fully loaded, (2) add a `find_variable()` helper for year-suffix fuzzy matching, (3) update derivation steps 4 and 5 to use these new capabilities. Step 8 and 14 will cascade-fix automatically once step 4 produces `cov_type` and `hru_percent_imperv`.

**Tech Stack:** Python, xarray, numpy, pytest. No new dependencies.

**Issue:** #119

---

### Task 1: Add `load_dataset()` and `find_variable()` to SIRAccessor

**Files:**
- Modify: `src/hydro_param/sir_accessor.py`
- Test: `tests/test_sir_accessor.py`

**Context:** Currently `load_variable(name)` returns a single `xr.DataArray`, but multi-column CSVs (like `lndcov_frac_2021.csv` with 17 columns) lose all but the first column. We need `load_dataset()` to return the full `xr.Dataset`. We also need `find_variable(base_name)` that matches year-suffixed variants (e.g., `fctimp_pct_mean` → `fctimp_pct_mean_2021`).

**Step 1: Write failing tests**

In `tests/test_sir_accessor.py`, add tests for both new methods:

```python
def test_load_dataset_returns_all_columns(tmp_path):
    """load_dataset returns full xr.Dataset with all CSV columns."""
    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    # Multi-column categorical CSV
    df = pd.DataFrame(
        {"lndcov_frac_2021_11": [0.8, 0.1], "lndcov_frac_2021_41": [0.2, 0.9]},
        index=pd.Index([1, 2], name="nhm_id"),
    )
    df.to_csv(sir_dir / "lndcov_frac_2021.csv")
    # Write minimal manifest
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            lndcov_frac_2021: sir/lndcov_frac_2021.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)

    sir = SIRAccessor(tmp_path)
    ds = sir.load_dataset("lndcov_frac_2021")
    assert isinstance(ds, xr.Dataset)
    assert "lndcov_frac_2021_11" in ds.data_vars
    assert "lndcov_frac_2021_41" in ds.data_vars
    assert len(ds.data_vars) == 2


def test_find_variable_exact_match(tmp_path):
    """find_variable returns exact match when available."""
    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame({"val": [1.0]}, index=pd.Index([1], name="nhm_id"))
    df.to_csv(sir_dir / "elevation_m_mean.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            elevation_m_mean: sir/elevation_m_mean.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)

    sir = SIRAccessor(tmp_path)
    assert sir.find_variable("elevation_m_mean") == "elevation_m_mean"


def test_find_variable_year_suffix(tmp_path):
    """find_variable matches year-suffixed variant."""
    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame({"val": [5.0]}, index=pd.Index([1], name="nhm_id"))
    df.to_csv(sir_dir / "fctimp_pct_mean_2021.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            fctimp_pct_mean_2021: sir/fctimp_pct_mean_2021.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)

    sir = SIRAccessor(tmp_path)
    assert sir.find_variable("fctimp_pct_mean") == "fctimp_pct_mean_2021"


def test_find_variable_not_found(tmp_path):
    """find_variable returns None when no match exists."""
    sir_dir = tmp_path / "sir"
    sir_dir.mkdir()
    df = pd.DataFrame({"val": [1.0]}, index=pd.Index([1], name="nhm_id"))
    df.to_csv(sir_dir / "elevation_m_mean.csv")
    manifest_content = textwrap.dedent("""\
        version: 2
        fabric_fingerprint: test
        entries: {}
        sir:
          static_files:
            elevation_m_mean: sir/elevation_m_mean.csv
          temporal_files: {}
          sir_schema: []
    """)
    (tmp_path / ".manifest.yml").write_text(manifest_content)

    sir = SIRAccessor(tmp_path)
    assert sir.find_variable("bogus_var") is None
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir_accessor.py -x -v -k "load_dataset or find_variable"`
Expected: FAIL — `load_dataset` and `find_variable` don't exist.

**Step 3: Implement `load_dataset()` and `find_variable()`**

In `src/hydro_param/sir_accessor.py`, add two methods to `SIRAccessor`:

```python
def load_dataset(self, name: str) -> xr.Dataset:
    """Load a static SIR file as a full Dataset (all columns).

    Unlike ``load_variable()`` which returns a single DataArray,
    this method returns the complete ``xr.Dataset`` with all columns
    from a multi-column CSV.  Useful for categorical fraction files
    (e.g., ``lndcov_frac_2021``) where each column represents a
    class fraction.

    Parameters
    ----------
    name : str
        SIR variable name (must match a static file key).

    Returns
    -------
    xr.Dataset
        Full dataset with all columns from the CSV.

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
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        raise OSError(
            f"Failed to read SIR file for '{name}' at {path}: {exc}."
        ) from exc
    return xr.Dataset.from_dataframe(df)


def find_variable(self, base_name: str) -> str | None:
    """Find a static variable by base name, allowing year suffixes.

    Return ``base_name`` if it exists as-is.  Otherwise, search for
    variables matching ``{base_name}_{year}`` where year is a 4-digit
    number.  Returns the first match (most recent year if multiple).

    Parameters
    ----------
    base_name : str
        Variable base name (e.g., ``"fctimp_pct_mean"``).

    Returns
    -------
    str or None
        The actual SIR variable name, or ``None`` if not found.
    """
    if base_name in self._static:
        return base_name
    import re
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{4}})$")
    matches = [v for v in self._static if pattern.match(v)]
    if matches:
        # Return most recent year
        return sorted(matches)[-1]
    return None
```

Also add `find_variable` and `load_dataset` to `_MockSIRAccessor` in `tests/test_pywatershed_derivation.py`:

```python
def load_dataset(self, name: str) -> xr.Dataset:
    # For mock, find all vars starting with name and return as Dataset
    matching = {k: v for k, v in self._ds.data_vars.items()
                if str(k).startswith(name)}
    if not matching:
        raise KeyError(f"SIR variable '{name}' not found.")
    return xr.Dataset(matching)

def find_variable(self, base_name: str) -> str | None:
    if base_name in self._ds:
        return base_name
    import re
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{4}})$")
    matches = [str(v) for v in self._ds.data_vars if pattern.match(str(v))]
    if matches:
        return sorted(matches)[-1]
    return None
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir_accessor.py -x -v -k "load_dataset or find_variable"`
Expected: PASS

**Step 5: Commit**

```
git add src/hydro_param/sir_accessor.py tests/test_sir_accessor.py tests/test_pywatershed_derivation.py
git commit -m "feat: add load_dataset() and find_variable() to SIRAccessor (#119)"
```

---

### Task 2: Fix `_compute_majority_from_fractions` to handle year-suffixed columns

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (lines 1455-1528)
- Test: `tests/test_pywatershed_derivation.py`

**Context:** `_compute_majority_from_fractions` currently searches `sir.data_vars` for columns starting with `lndcov_frac_`. With the real SIR, `data_vars` only contains `lndcov_frac_2021` (a file-level key), not the individual columns like `lndcov_frac_2021_11`. The method needs to: (1) detect the file-level key, (2) load it as a full Dataset via `load_dataset()`, (3) extract class codes from the column names.

**Step 1: Write failing test**

Add a test that uses year-suffixed column names in the mock SIR:

```python
def test_majority_from_fractions_year_suffixed(self, derivation):
    """Majority class from year-suffixed fraction columns (lndcov_frac_2021_XX)."""
    sir = _MockSIRAccessor(
        xr.Dataset(
            {
                "lndcov_frac_2021_11": ("nhm_id", np.array([0.1, 0.0])),
                "lndcov_frac_2021_41": ("nhm_id", np.array([0.8, 0.1])),
                "lndcov_frac_2021_42": ("nhm_id", np.array([0.1, 0.9])),
            },
            coords={"nhm_id": [1, 2]},
        )
    )
    result = derivation._compute_majority_from_fractions(sir)
    assert result is not None
    np.testing.assert_array_equal(result, [41, 42])
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -x -v -k "test_majority_from_fractions_year_suffixed"`
Expected: FAIL — returns None because `data_vars` has `lndcov_frac_2021_11` etc., which start with `lndcov_frac_` but the suffix `2021_11` isn't parseable as a single int.

Wait — in the mock, `data_vars` DOES have `lndcov_frac_2021_11` as individual vars. The real SIR only has `lndcov_frac_2021` as a file key. Let me reconsider.

The mock test will actually pass because the mock exposes individual columns as data_vars. The real failure is in SIRAccessor where data_vars only returns file-level keys.

We need a test with a real SIRAccessor OR update the test to match the real structure. The correct approach: update `_compute_majority_from_fractions` to try file-level keys when column-level search fails.

**Step 1 (revised): Write failing test using file-level mock**

```python
def test_majority_from_fractions_file_level_key(self, derivation):
    """Majority class from a file-level SIR key containing fraction columns.

    Real SIR has data_vars=['lndcov_frac_2021'] with individual columns
    lndcov_frac_2021_11, lndcov_frac_2021_41 inside the CSV. The mock
    simulates this with a Dataset where lndcov_frac_2021 is the only
    data_var key, and load_dataset() returns the inner columns.
    """
    # Inner columns (what load_dataset returns)
    inner_ds = xr.Dataset(
        {
            "lndcov_frac_2021_11": ("nhm_id", np.array([0.1, 0.0])),
            "lndcov_frac_2021_41": ("nhm_id", np.array([0.8, 0.1])),
            "lndcov_frac_2021_42": ("nhm_id", np.array([0.1, 0.9])),
        },
        coords={"nhm_id": [1, 2]},
    )
    # Outer Dataset (what data_vars reports)
    outer_ds = xr.Dataset(
        {"lndcov_frac_2021": ("nhm_id", np.array([0.0, 0.0]))},
        coords={"nhm_id": [1, 2]},
    )

    class _FileKeyMock(_MockSIRAccessor):
        def __init__(self):
            super().__init__(outer_ds)
            self._inner = inner_ds

        def load_dataset(self, name: str) -> xr.Dataset:
            if name == "lndcov_frac_2021":
                return self._inner
            raise KeyError(name)

    sir = _FileKeyMock()
    result = derivation._compute_majority_from_fractions(sir)
    assert result is not None
    np.testing.assert_array_equal(result, [41, 42])
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -x -v -k "test_majority_from_fractions_file_level_key"`
Expected: FAIL — current code doesn't call `load_dataset()`.

**Step 3: Update `_compute_majority_from_fractions`**

Replace the method body (lines 1491-1526) to add a fallback that loads the full dataset when column-level search fails:

```python
@staticmethod
def _compute_majority_from_fractions(
    sir: SIRAccessor,
    prefixes: tuple[str, ...] = ("lndcov_frac_",),
) -> np.ndarray | None:
    # ... (keep existing docstring) ...

    for prefix in prefixes:
        # --- First, try column-level search in data_vars ---
        fraction_vars = sorted(v for v in sir.data_vars if v.startswith(prefix))
        class_codes: list[int] = []
        valid_vars: list[str] = []

        for v in fraction_vars:
            suffix = v[len(prefix):]
            try:
                class_codes.append(int(suffix))
                valid_vars.append(v)
            except ValueError:
                # Might be a file-level key like "lndcov_frac_2021" (year suffix)
                # Try loading it as a dataset and extracting columns
                if hasattr(sir, "load_dataset"):
                    try:
                        inner_ds = sir.load_dataset(v)
                    except (KeyError, OSError):
                        logger.debug(
                            "Skipping variable '%s': suffix '%s' not an integer "
                            "and load_dataset failed",
                            v, suffix,
                        )
                        continue

                    # Search inner columns: prefix is now the file key + "_"
                    inner_prefix = v + "_"
                    inner_codes: list[int] = []
                    inner_vars: list[str] = []
                    for col in sorted(inner_ds.data_vars):
                        col_str = str(col)
                        if not col_str.startswith(inner_prefix):
                            continue
                        col_suffix = col_str[len(inner_prefix):]
                        try:
                            inner_codes.append(int(col_suffix))
                            inner_vars.append(col_str)
                        except ValueError:
                            logger.debug(
                                "Skipping column '%s': suffix '%s' not an integer",
                                col_str, col_suffix,
                            )

                    if len(inner_codes) >= 2:
                        fractions = np.column_stack(
                            [inner_ds[v].values for v in inner_vars]
                        )
                        codes = np.array(inner_codes)
                        majority_idx = np.argmax(fractions, axis=1)
                        majority_class = codes[majority_idx]
                        logger.info(
                            "Computed majority class from %d categorical fraction "
                            "columns (file=%r)",
                            len(inner_vars), v,
                        )
                        return majority_class
                else:
                    logger.debug(
                        "Skipping variable '%s': suffix '%s' is not an "
                        "integer class code",
                        v, suffix,
                    )
                continue

        if len(class_codes) < 2:
            continue

        # Stack fractions into (nhru, n_classes) array
        fractions = np.column_stack([sir[v].values for v in valid_vars])
        codes = np.array(class_codes)
        majority_idx = np.argmax(fractions, axis=1)
        majority_class = codes[majority_idx]

        logger.info(
            "Computed majority class from %d categorical fraction columns "
            "(prefix=%r)",
            len(valid_vars), prefix,
        )
        return majority_class

    return None
```

**Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -x -v -k "majority_from_fractions"`
Expected: ALL majority tests PASS (both old column-level and new file-level).

**Step 5: Commit**

```
git commit -m "fix: handle year-suffixed landcover fraction columns in majority computation (#119)"
```

---

### Task 3: Fix `_derive_landcover` to find year-suffixed `fctimp_pct_mean`

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (line 1445)
- Test: `tests/test_pywatershed_derivation.py`

**Context:** Line 1445 does `if "fctimp_pct_mean" in sir:` but the SIR has `fctimp_pct_mean_2021`. Use the new `find_variable()` method.

**Step 1: Write failing test**

```python
def test_derive_landcover_year_suffixed_imperv(self, derivation):
    """hru_percent_imperv derived from year-suffixed fctimp_pct_mean_2021."""
    sir = _MockSIRAccessor(
        xr.Dataset(
            {
                "lndcov_frac_11": ("nhm_id", np.array([0.8, 0.2])),
                "lndcov_frac_42": ("nhm_id", np.array([0.2, 0.8])),
                "fctimp_pct_mean_2021": ("nhm_id", np.array([10.0, 50.0])),
            },
            coords={"nhm_id": [1, 2]},
        )
    )
    ctx = _make_context(sir)
    ds = derivation._derive_landcover(ctx, xr.Dataset())
    assert "hru_percent_imperv" in ds
    np.testing.assert_allclose(ds["hru_percent_imperv"].values, [0.1, 0.5])
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -x -v -k "test_derive_landcover_year_suffixed_imperv"`
Expected: FAIL — `hru_percent_imperv` not in ds.

**Step 3: Update `_derive_landcover`**

Replace lines 1445-1451:

```python
# Before:
if "fctimp_pct_mean" in sir:
    ...

# After:
fctimp_key = sir.find_variable("fctimp_pct_mean")
if fctimp_key is not None:
    ds["hru_percent_imperv"] = xr.DataArray(
        np.clip(sir[fctimp_key].values / 100.0, 0.0, 1.0),
        dims="nhru",
        attrs={"units": "decimal_fraction", "long_name": "HRU impervious fraction"},
    )
```

**Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -x -v -k "landcover"`
Expected: PASS for all landcover tests.

**Step 5: Commit**

```
git commit -m "fix: find year-suffixed fctimp_pct_mean in landcover derivation (#119)"
```

---

### Task 4: Fix `_derive_soils` to find `aws0_100_cm_mean` and convert units

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (lines 1599-1612)
- Test: `tests/test_pywatershed_derivation.py`

**Context:** Line 1600 does `if "awc_mm_mean" in sir:` but the SIR has `aws0_100_cm_mean` (available water storage, 0-100 cm depth, in centimeters). The derivation needs to: (1) search for both names, (2) convert cm → mm (multiply by 10) when using the cm variant.

**Step 1: Write failing test**

```python
def test_derive_soils_aws_cm_fallback(self, derivation):
    """soil_moist_max derived from aws0_100_cm_mean with cm->mm conversion."""
    sir = _MockSIRAccessor(
        xr.Dataset(
            {
                "aws0_100_cm_mean": ("nhm_id", np.array([5.0, 15.0, 8.0])),
                "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.1, 0.0])),
                "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.8, 0.1])),
                "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1, 0.9])),
            },
            coords={"nhm_id": [1, 2, 3]},
        )
    )
    ctx = _make_context(sir)
    ds = derivation._derive_soils(ctx, xr.Dataset())
    assert "soil_moist_max" in ds
    # 5 cm = 50 mm -> convert(50, mm, in) = 50/25.4 ≈ 1.969
    # 15 cm = 150 mm -> convert(150, mm, in) = 150/25.4 ≈ 5.906
    # 8 cm = 80 mm -> convert(80, mm, in) = 80/25.4 ≈ 3.150
    expected = np.clip(np.array([50.0, 150.0, 80.0]) / 25.4, 0.5, 20.0)
    np.testing.assert_allclose(ds["soil_moist_max"].values, expected, rtol=1e-3)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -x -v -k "test_derive_soils_aws_cm_fallback"`
Expected: FAIL — `soil_moist_max` not in ds.

**Step 3: Update `_derive_soils`**

Replace lines 1599-1612 with:

```python
# --- soil_moist_max ---
if "awc_mm_mean" in sir:
    awc_mm = sir["awc_mm_mean"].values.astype(np.float64)
    soil_moist_max = convert(awc_mm, "mm", "in")
    soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
    ds["soil_moist_max"] = xr.DataArray(
        soil_moist_max,
        dims="nhru",
        attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
    )
elif "aws0_100_cm_mean" in sir:
    # Available water storage in cm — convert to mm first
    aws_cm = sir["aws0_100_cm_mean"].values.astype(np.float64)
    awc_mm = aws_cm * 10.0  # cm -> mm
    soil_moist_max = convert(awc_mm, "mm", "in")
    soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
    ds["soil_moist_max"] = xr.DataArray(
        soil_moist_max,
        dims="nhru",
        attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
    )
    logger.info(
        "Used aws0_100_cm_mean (cm -> mm -> in) for soil_moist_max"
    )
else:
    logger.warning(
        "Skipping soil_moist_max derivation (step 5): neither 'awc_mm_mean' "
        "nor 'aws0_100_cm_mean' found in SIR."
    )
```

**Step 4: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -x -v -k "soil"`
Expected: PASS for all soil tests (existing tests use `awc_mm_mean`; new test uses `aws0_100_cm_mean`).

**Step 5: Commit**

```
git commit -m "fix: support aws0_100_cm_mean as fallback for soil_moist_max with cm→mm conversion (#119)"
```

---

### Task 5: Run full test suite and verify cascade fixes

**Files:** None (verification only)

**Step 1: Run full derivation tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: All existing tests still pass, new tests pass.

**Step 2: Run full test suite**

Run: `pixi run -e dev check`
Expected: All checks pass (lint, format, typecheck, tests).

**Step 3: Manual verification against DRB SIR**

Run: `cd /tmp/drb-e2e-2 && hydro-param pywatershed run configs/pywatershed_run.yml`
Expected:
- `cov_type`, `covden_sum`, `covden_win` now present
- `hru_percent_imperv` now present
- `srain_intcp`, `wrain_intcp`, `snow_intcp`, `imperv_stor_max` now present (cascade from step 8)
- `soil_moist_max` now present
- `carea_max` calibration seed now uses real data (not default)
- Remaining warnings (steps 10, 11, soil_type) are expected (separate issues #120, #121)

**Step 4: Commit any final adjustments**

If manual testing reveals additional issues, fix and commit.
