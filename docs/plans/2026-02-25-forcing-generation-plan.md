# Forcing Generation (Step 7) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Formalize step 7 (forcing generation) as `_derive_forcing()` inside `PywatershedDerivation`, consuming SIR-normalized temporal data via a YAML-configured variable mapping.

**Architecture:** Add `temporal` field to `DerivationContext`. A new `forcing_variables.yml` defines SIR→PRMS variable name/unit mappings per source dataset (gridMET initially). `_derive_forcing()` loads this YAML, renames variables, converts units, and merges temporal DataArrays into the derived dataset. The formatter then writes one NetCDF per forcing variable with final PRMS units.

**Tech Stack:** Python 3.10+, xarray, numpy, PyYAML, pytest

---

### Task 1: Add `temporal` field to `DerivationContext`

**Files:**
- Modify: `src/hydro_param/plugins.py:24-58`
- Test: `tests/test_pywatershed_derivation.py` (existing fixtures)

**Step 1: Write the failing test**

Add to `tests/test_pywatershed_derivation.py` near the top, after existing imports:

```python
class TestDerivationContextTemporal:
    """Tests for temporal field on DerivationContext."""

    def test_temporal_defaults_to_none(self, sir_topography: xr.Dataset) -> None:
        """DerivationContext.temporal is None when not provided."""
        ctx = DerivationContext(sir=sir_topography)
        assert ctx.temporal is None

    def test_temporal_accepts_dict(self, sir_topography: xr.Dataset) -> None:
        """DerivationContext.temporal accepts a dict of datasets."""
        temporal_ds = xr.Dataset(
            {"pr_mm_mean": (("time", "nhm_id"), np.ones((3, 3)))},
            coords={"time": [0, 1, 2], "nhm_id": [1, 2, 3]},
        )
        ctx = DerivationContext(
            sir=sir_topography,
            temporal={"gridmet_2020": temporal_ds},
        )
        assert ctx.temporal is not None
        assert "gridmet_2020" in ctx.temporal
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDerivationContextTemporal -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'temporal'`

**Step 3: Add `temporal` field to `DerivationContext`**

In `src/hydro_param/plugins.py`, add the field after `sir`:

```python
@dataclass(frozen=True)
class DerivationContext:
    # ... existing docstring — add temporal to the Parameters section ...

    sir: xr.Dataset
    temporal: dict[str, xr.Dataset] | None = None  # <-- NEW
    fabric: gpd.GeoDataFrame | None = None
    segments: gpd.GeoDataFrame | None = None
    fabric_id_field: str = "nhm_id"
    segment_id_field: str | None = None
    config: dict = field(default_factory=dict)
    lookup_tables_dir: Path | None = None
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDerivationContextTemporal -v`
Expected: PASS (2 tests)

**Step 5: Run full test suite to check nothing broke**

Run: `pixi run -e dev pytest --tb=short -q`
Expected: All tests pass (existing DerivationContext callers don't pass `temporal`, so the default `None` is backward-compatible)

**Step 6: Commit**

```bash
git add src/hydro_param/plugins.py tests/test_pywatershed_derivation.py
git commit -m "feat: add temporal field to DerivationContext"
```

---

### Task 2: Register W/m2 → Langleys/day unit conversion

**Files:**
- Modify: `src/hydro_param/units.py:100-131`
- Test: `tests/test_units.py` (existing test file)

**Step 1: Write the failing test**

Add to `tests/test_units.py`:

```python
def test_wm2_to_langleys_per_day():
    """W/m2 → Langleys/day conversion (1 W/m2 ≈ 2.065 Langleys/day)."""
    values = np.array([1.0, 100.0, 250.0])
    result = convert(values, "W/m2", "Langleys/day")
    expected = values * 2.065
    np.testing.assert_allclose(result, expected, rtol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_units.py::test_wm2_to_langleys_per_day -v`
Expected: FAIL — `KeyError: No conversion registered: W/m2 -> Langleys/day`

**Step 3: Register the conversion**

Add to `src/hydro_param/units.py` after the angular conversions block (~line 124):

```python
# Irradiance
register("W/m2", "Langleys/day", lambda v: v * 2.065, "watts per square meter to Langleys per day")
```

The factor 2.065 comes from: 1 W/m2 = 1 J/(s·m2) × 86400 s/day × 1 cal/4.184 J × 1/(10000 cm2/m2) = 2.065 cal/(cm2·day) = 2.065 Langleys/day.

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_units.py::test_wm2_to_langleys_per_day -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/units.py tests/test_units.py
git commit -m "feat: register W/m2 to Langleys/day unit conversion"
```

---

### Task 3: Create `forcing_variables.yml` lookup table

**Files:**
- Create: `src/hydro_param/data/lookup_tables/forcing_variables.yml`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing test**

Add a new test class to `tests/test_pywatershed_derivation.py`:

```python
class TestForcingVariablesYAML:
    """Validate the forcing_variables.yml lookup table."""

    def test_forcing_variables_yaml_loads(self, derivation: PywatershedDerivation) -> None:
        """forcing_variables.yml loads and has required structure."""
        from importlib.resources import files as pkg_files

        tables_dir = Path(str(pkg_files("hydro_param").joinpath("data/lookup_tables")))
        data = derivation._load_lookup_table("forcing_variables", tables_dir)
        assert "mapping" in data
        datasets = data["mapping"]
        assert "gridmet" in datasets

    def test_gridmet_variables_have_required_keys(self, derivation: PywatershedDerivation) -> None:
        """Each gridmet variable entry has sir_name, sir_unit, intermediate_unit."""
        from importlib.resources import files as pkg_files

        tables_dir = Path(str(pkg_files("hydro_param").joinpath("data/lookup_tables")))
        data = derivation._load_lookup_table("forcing_variables", tables_dir)
        gridmet = data["mapping"]["gridmet"]
        required_keys = {"sir_name", "sir_unit", "intermediate_unit"}
        for prms_name, entry in gridmet.items():
            missing = required_keys - set(entry.keys())
            assert not missing, f"{prms_name} missing keys: {missing}"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestForcingVariablesYAML -v`
Expected: FAIL — `FileNotFoundError: Lookup table 'forcing_variables.yml' not found`

**Step 3: Create the YAML file**

Create `src/hydro_param/data/lookup_tables/forcing_variables.yml`:

```yaml
# Forcing variable mappings: SIR canonical names → PRMS parameter names.
# One section per source dataset. Each variable maps an SIR temporal
# variable to its PRMS name and specifies the unit the derivation step
# should deliver (intermediate_unit). The formatter then does the final
# PRMS unit conversion (e.g., mm → in, C → F).
#
# sir_name: canonical SIR variable name (from stage 5 normalization)
# sir_unit: unit of the SIR variable
# intermediate_unit: unit to deliver to the formatter

name: forcing_variables
description: "SIR temporal variable to PRMS forcing variable mappings"
source: "gridMET via climateR-catalogs"

mapping:
  gridmet:
    prcp:
      sir_name: pr_mm_mean
      sir_unit: mm
      intermediate_unit: mm
    tmax:
      sir_name: tmmx_C_mean
      sir_unit: C
      intermediate_unit: C
    tmin:
      sir_name: tmmn_C_mean
      sir_unit: C
      intermediate_unit: C
    swrad:
      sir_name: srad_W_m2_mean
      sir_unit: W/m2
      intermediate_unit: Langleys/day
    potet:
      sir_name: pet_mm_mean
      sir_unit: mm
      intermediate_unit: in
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestForcingVariablesYAML -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/data/lookup_tables/forcing_variables.yml tests/test_pywatershed_derivation.py
git commit -m "feat: add forcing_variables.yml lookup table for gridMET"
```

---

### Task 4: Implement `_derive_forcing()` method

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing tests**

Add test fixture and test class to `tests/test_pywatershed_derivation.py`:

```python
@pytest.fixture()
def temporal_gridmet() -> dict[str, xr.Dataset]:
    """Synthetic SIR-normalized temporal data mimicking gridMET output.

    Two years of daily data for 3 HRUs. Variable names match SIR canonical
    names (post-stage-5 normalization).
    """
    nhru = 3
    ntime_2020 = 366  # leap year
    ntime_2021 = 365
    rng = np.random.default_rng(42)

    def _make_ds(ntime: int, year: int) -> xr.Dataset:
        times = np.arange(ntime)
        return xr.Dataset(
            {
                "pr_mm_mean": (("time", "nhm_id"), rng.uniform(0, 20, (ntime, nhru))),
                "tmmx_C_mean": (("time", "nhm_id"), rng.uniform(10, 35, (ntime, nhru))),
                "tmmn_C_mean": (("time", "nhm_id"), rng.uniform(-5, 15, (ntime, nhru))),
                "srad_W_m2_mean": (("time", "nhm_id"), rng.uniform(50, 300, (ntime, nhru))),
                "pet_mm_mean": (("time", "nhm_id"), rng.uniform(0, 8, (ntime, nhru))),
            },
            coords={"time": times, "nhm_id": [1, 2, 3]},
        )

    return {
        "gridmet_2020": _make_ds(ntime_2020, 2020),
        "gridmet_2021": _make_ds(ntime_2021, 2021),
    }


class TestDeriveForcing:
    """Tests for _derive_forcing (step 7)."""

    def test_no_temporal_returns_unchanged(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """When temporal is None, dataset is returned unchanged."""
        ctx = DerivationContext(sir=sir_topography, temporal=None)
        ds = xr.Dataset({"hru_elev": ("nhru", [100.0, 200.0, 300.0])})
        result = derivation._derive_forcing(ctx, ds)
        assert set(result.data_vars) == {"hru_elev"}

    def test_renames_sir_to_prms(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """SIR canonical names are renamed to PRMS names."""
        ctx = DerivationContext(sir=sir_topography, temporal=temporal_gridmet)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert "prcp" in result
        assert "tmax" in result
        assert "tmin" in result
        assert "swrad" in result
        assert "potet" in result
        # Original SIR names should NOT be present
        assert "pr_mm_mean" not in result
        assert "tmmx_C_mean" not in result

    def test_multiyear_concat(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """Multi-year temporal data is concatenated along time."""
        ctx = DerivationContext(sir=sir_topography, temporal=temporal_gridmet)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        # 366 (2020) + 365 (2021) = 731 timesteps
        assert result["prcp"].sizes["time"] == 731

    def test_swrad_converted_to_langleys(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """swrad is converted from W/m2 to Langleys/day."""
        temporal = {
            "gridmet_2020": xr.Dataset(
                {"srad_W_m2_mean": (("time", "nhm_id"), np.array([[100.0, 200.0, 300.0]]))},
                coords={"time": [0], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        expected = np.array([100.0, 200.0, 300.0]) * 2.065
        np.testing.assert_allclose(result["swrad"].values[0], expected, rtol=1e-6)

    def test_potet_converted_to_inches(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """potet is converted from mm to inches."""
        temporal = {
            "gridmet_2020": xr.Dataset(
                {"pet_mm_mean": (("time", "nhm_id"), np.array([[25.4, 50.8, 0.0]]))},
                coords={"time": [0], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        np.testing.assert_allclose(result["potet"].values[0], [1.0, 2.0, 0.0], rtol=1e-6)

    def test_feature_dim_aligned(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Temporal feature dim is renamed to match derived dataset nhru."""
        temporal = {
            "gridmet_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert "nhru" in result["prcp"].dims

    def test_missing_sir_variable_skipped(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Missing SIR variables are skipped with a warning."""
        temporal = {
            "gridmet_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        # Only prcp should be present (other SIR vars not in temporal)
        assert "prcp" in result
        assert "tmax" not in result

    def test_empty_temporal_returns_unchanged(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Empty temporal dict returns dataset unchanged."""
        ctx = DerivationContext(sir=sir_topography, temporal={})
        ds = xr.Dataset({"hru_elev": ("nhru", [100.0, 200.0, 300.0])})
        result = derivation._derive_forcing(ctx, ds)
        assert set(result.data_vars) == {"hru_elev"}
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveForcing -v`
Expected: FAIL — `AttributeError: 'PywatershedDerivation' object has no attribute '_derive_forcing'`

**Step 3: Implement `_derive_forcing()`**

Add to `src/hydro_param/derivations/pywatershed.py`, after the `_derive_calibration_seeds` method:

```python
    # ------------------------------------------------------------------
    # Step 7: Forcing generation (temporal merge)
    # ------------------------------------------------------------------

    def _derive_forcing(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Step 7: Merge SIR-normalized temporal forcing into derived dataset.

        Loads variable mappings from ``forcing_variables.yml``, detects the
        source dataset by matching SIR variable names, renames to PRMS names,
        applies unit conversions, and merges into *ds*.

        Skips gracefully when no temporal data is available.
        """
        if ctx.temporal is None or len(ctx.temporal) == 0:
            logger.info("No temporal data provided; skipping forcing generation.")
            return ds

        tables_dir = ctx.resolved_lookup_tables_dir
        config = self._load_lookup_table("forcing_variables", tables_dir)
        datasets_config = config["mapping"]

        # Concat multi-year chunks by base name (strip _YYYY suffix)
        chunks_by_source: dict[str, list[xr.Dataset]] = {}
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            chunks_by_source.setdefault(base_name, []).append(tds)

        for source_name, chunks in chunks_by_source.items():
            if len(chunks) > 1:
                chunks.sort(key=lambda c: c["time"].values[0])
                merged_temporal = xr.concat(chunks, dim="time")
            else:
                merged_temporal = chunks[0]

            # Detect dataset config by matching source name or SIR variable names
            dataset_cfg = self._detect_forcing_dataset(
                source_name, merged_temporal, datasets_config
            )
            if dataset_cfg is None:
                logger.warning(
                    "Could not match temporal source '%s' to any forcing dataset "
                    "config; skipping.",
                    source_name,
                )
                continue

            # Process each mapped variable
            for prms_name, var_cfg in dataset_cfg.items():
                sir_name = var_cfg["sir_name"]
                sir_unit = var_cfg["sir_unit"]
                intermediate_unit = var_cfg["intermediate_unit"]

                if sir_name not in merged_temporal:
                    logger.warning(
                        "Forcing variable '%s' (SIR name '%s') not found in "
                        "temporal data; skipping.",
                        prms_name,
                        sir_name,
                    )
                    continue

                da = merged_temporal[sir_name].copy(deep=True)

                # Unit conversion (SIR unit → intermediate unit)
                if sir_unit != intermediate_unit:
                    da.values = convert(
                        da.values.astype(np.float64), sir_unit, intermediate_unit
                    )

                # Align feature dimension to derived dataset
                feat_dims = [d for d in da.dims if d != "time"]
                if feat_dims and "nhru" in ds.dims and feat_dims[0] != "nhru":
                    da = da.rename({feat_dims[0]: "nhru"})

                ds[prms_name] = da

            n_vars = sum(1 for p in dataset_cfg if p in ds)
            logger.info(
                "Step 7: merged %d forcing variables from '%s' "
                "(%d timesteps).",
                n_vars,
                source_name,
                merged_temporal.sizes.get("time", 0),
            )

        return ds

    @staticmethod
    def _detect_forcing_dataset(
        source_name: str,
        temporal: xr.Dataset,
        datasets_config: dict,
    ) -> dict | None:
        """Match a temporal dataset to its forcing config section.

        Tries exact name match first, then falls back to counting SIR
        variable name matches.
        """
        # Exact match on source name
        if source_name in datasets_config:
            return datasets_config[source_name]

        # Fuzzy match: pick config with most SIR variable name hits
        best_match: str | None = None
        best_count = 0
        temporal_vars = set(temporal.data_vars)
        for cfg_name, cfg_vars in datasets_config.items():
            sir_names = {v["sir_name"] for v in cfg_vars.values()}
            count = len(sir_names & temporal_vars)
            if count > best_count:
                best_count = count
                best_match = cfg_name

        if best_match is not None and best_count > 0:
            logger.info(
                "Matched temporal source '%s' to forcing config '%s' "
                "(%d/%d variables matched).",
                source_name,
                best_match,
                best_count,
                len(datasets_config[best_match]),
            )
            return datasets_config[best_match]

        return None
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveForcing -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: implement _derive_forcing for step 7 forcing generation"
```

---

### Task 5: Wire `_derive_forcing()` into `derive()` and extend formatter

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:158-214` (derive method)
- Modify: `src/hydro_param/formatters/pywatershed.py:31-35` (_FORCING_VARS)
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing integration test**

Add to `tests/test_pywatershed_derivation.py`:

```python
class TestDeriveIntegrationForcing:
    """Integration test: full derive() with temporal data produces forcing."""

    def test_derive_with_temporal_produces_forcing(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """Full derive() call with temporal context includes forcing vars."""
        ctx = DerivationContext(
            sir=sir_topography,
            temporal=temporal_gridmet,
        )
        result = derivation.derive(ctx)
        # CBH variables
        assert "prcp" in result
        assert "tmax" in result
        assert "tmin" in result
        # Additional forcings
        assert "swrad" in result
        assert "potet" in result
        # Time dimension present
        assert "time" in result["prcp"].dims
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveIntegrationForcing -v`
Expected: FAIL — `derive()` doesn't call `_derive_forcing()` yet

**Step 3: Wire into `derive()` and extend formatter**

In `src/hydro_param/derivations/pywatershed.py`, update the `derive()` method. Add after the calibration seeds step (before parameter overrides):

```python
        # Step 7: Forcing (temporal merge — runs late, no downstream deps)
        ds = self._derive_forcing(context, ds)
```

Also update the class docstring (line 148-150) to include step 7.

In `src/hydro_param/formatters/pywatershed.py`, extend `_FORCING_VARS`:

```python
_FORCING_VARS: dict[str, tuple[str, str]] = {
    "prcp": ("mm", "in"),
    "tmax": ("C", "F"),
    "tmin": ("C", "F"),
    "swrad": ("Langleys/day", "Langleys/day"),
    "potet": ("in", "in"),
}
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveIntegrationForcing -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pixi run -e dev pytest --tb=short -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py src/hydro_param/formatters/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: wire _derive_forcing into derive() and extend formatter"
```

---

### Task 6: Deprecate `merge_temporal_into_derived()`

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:74-140`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the test**

```python
class TestMergeTemporalDeprecation:
    """Test that merge_temporal_into_derived logs a deprecation warning."""

    def test_deprecation_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Calling merge_temporal_into_derived logs a deprecation warning."""
        from hydro_param.derivations.pywatershed import merge_temporal_into_derived

        ds = xr.Dataset({"x": ("nhru", [1.0])})
        with caplog.at_level(logging.WARNING):
            merge_temporal_into_derived(ds, {})
        assert any("deprecated" in r.message.lower() for r in caplog.records)
```

Note: you'll need to add `import logging` to the test file imports if not already there.

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestMergeTemporalDeprecation -v`
Expected: FAIL — no deprecation warning logged yet

**Step 3: Add deprecation warning**

In `src/hydro_param/derivations/pywatershed.py`, add at the top of `merge_temporal_into_derived()` (after the docstring):

```python
    import warnings
    warnings.warn(
        "merge_temporal_into_derived() is deprecated. "
        "Use PywatershedDerivation._derive_forcing() via DerivationContext.temporal instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "merge_temporal_into_derived() is deprecated; "
        "use DerivationContext.temporal with _derive_forcing() instead."
    )
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestMergeTemporalDeprecation -v`
Expected: PASS

**Step 5: Run full test suite + checks**

Run: `pixi run -e dev check`
Expected: All pass

**Step 6: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "refactor: deprecate merge_temporal_into_derived in favor of _derive_forcing"
```

---

### Task 7: Update module docstring and run full checks

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:1-8`

**Step 1: Update docstring**

Change line 7 from:
```
Foundation implementation covers steps 1, 2, 3, 4, 5, 8, 9, 13, and 14.
```
to:
```
Foundation implementation covers steps 1, 2, 3, 4, 5, 7, 8, 9, 13, and 14.
```

**Step 2: Run full check suite**

Run: `pixi run -e dev check`
Expected: All pass (lint, format, typecheck, tests)

**Step 3: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All pass

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "docs: update module docstring to include step 7"
```

---

### Task 8: Create PR

**Step 1: Push branch**

```bash
git push -u origin feat/<issue-number>-forcing-generation
```

**Step 2: Create PR**

```bash
gh pr create --title "feat: step 7 forcing generation (_derive_forcing)" --body "$(cat <<'EOF'
## Summary
- Add `temporal` field to `DerivationContext` for passing SIR-normalized temporal data
- New `forcing_variables.yml` lookup table with gridMET SIR→PRMS variable mappings
- Implement `_derive_forcing()` method: renames, unit converts, multi-year concat, dim alignment
- Register W/m2 → Langleys/day unit conversion
- Extend formatter `_FORCING_VARS` for swrad and potet
- Deprecate `merge_temporal_into_derived()` (replaced by `_derive_forcing`)

Closes #<issue-number>

## Test plan
- [ ] `TestDerivationContextTemporal` — temporal field on DerivationContext
- [ ] `TestForcingVariablesYAML` — YAML structure validation
- [ ] `TestDeriveForcing` — 8 unit tests (rename, concat, conversion, dim alignment, missing vars)
- [ ] `TestDeriveIntegrationForcing` — full derive() with temporal context
- [ ] `TestMergeTemporalDeprecation` — deprecation warning
- [ ] `test_wm2_to_langleys_per_day` — unit conversion accuracy
- [ ] Full test suite passes
- [ ] `pixi run -e dev check` passes
- [ ] `pixi run -e dev pre-commit` passes

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
