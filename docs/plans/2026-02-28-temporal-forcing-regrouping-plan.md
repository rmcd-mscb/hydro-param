# Fix Temporal Forcing Detection for Per-Variable SIR Files — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix `_derive_forcing()` and `_compute_monthly_normals()` to work with per-variable temporal SIR files instead of requiring all variables grouped by source.

**Architecture:** Build a reverse lookup from `forcing_variables.yml` (`sir_name → config`), refactor both methods to iterate per-variable instead of per-source, and find tmax/tmin independently for climate normals. Delete `_detect_forcing_dataset()`.

**Tech Stack:** Python, xarray, pytest

---

### Task 1: Create issue and feature branch

**Step 1: Create feature branch**

```bash
git checkout -b fix/126-temporal-forcing-per-variable main
```

**Step 2: Verify branch**

```bash
git branch --show-current
```

Expected: `fix/126-temporal-forcing-per-variable`

---

### Task 2: Update temporal_gridmet fixture to match real SIR format

The existing `temporal_gridmet` fixture uses source-keyed datasets (`gridmet_2020` with all 5 variables). Real SIR produces variable-keyed datasets (`pr_mm_mean_2020` with 1 variable). All tests using this fixture need to work with the realistic format.

**Files:**
- Modify: `tests/test_pywatershed_derivation.py:174-198`

**Step 1: Replace the `temporal_gridmet` fixture**

Replace lines 174-198 with a fixture that produces per-variable-per-year datasets matching real SIR output:

```python
@pytest.fixture()
def temporal_gridmet() -> dict[str, xr.Dataset]:
    """Synthetic SIR-normalized temporal data mimicking per-variable SIR output.

    Real SIR normalizes temporal data into per-variable-per-year NetCDF files.
    Each dataset contains a single variable with its SIR canonical name.
    """
    import pandas as pd

    nhru = 3
    rng = np.random.default_rng(42)

    variables = {
        "pr_mm_mean": lambda n, h: rng.uniform(0, 20, (n, h)),
        "tmmx_C_mean": lambda n, h: rng.uniform(10, 35, (n, h)),
        "tmmn_C_mean": lambda n, h: rng.uniform(-5, 15, (n, h)),
        "srad_W_m2_mean": lambda n, h: rng.uniform(50, 300, (n, h)),
        "pet_mm_mean": lambda n, h: rng.uniform(0, 8, (n, h)),
    }

    result: dict[str, xr.Dataset] = {}
    for year in [2020, 2021]:
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        ntime = len(times)
        for var_name, gen_fn in variables.items():
            key = f"{var_name}_{year}"
            result[key] = xr.Dataset(
                {var_name: (("time", "nhm_id"), gen_fn(ntime, nhru))},
                coords={"time": times, "nhm_id": [1, 2, 3]},
            )

    return result
```

**Step 2: Run existing tests to see what breaks**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveForcing -v -x 2>&1 | tail -20
```

Expected: Tests that depend on source-keyed format will fail (e.g., `test_multiyear_concat` expects 731 timesteps from 2 gridmet datasets with all vars, now they're separate).

**Step 3: Commit**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: update temporal_gridmet fixture to match per-variable SIR format (#126)

Real SIR normalizes temporal data into per-variable-per-year files
(e.g., pr_mm_mean_2020.nc) not per-source files (gridmet_2020.nc).
Update the fixture to match reality. Tests that depend on source-keyed
format will be fixed in subsequent commits."
```

---

### Task 3: Add _build_sir_to_forcing_lookup() helper

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Add the reverse lookup method**

Add this as a new method on `PywatershedDerivation`, just before `_derive_forcing()` (around line 2370):

```python
    def _build_sir_to_forcing_lookup(
        self,
        tables_dir: Path,
    ) -> dict[str, dict[str, str]]:
        """Build reverse lookup from SIR variable names to forcing config.

        Invert the ``forcing_variables.yml`` mapping so that each SIR
        canonical name maps to its PRMS name, source, and unit config.
        This allows per-variable temporal data to be matched to forcing
        config without requiring all variables in a single dataset.

        Parameters
        ----------
        tables_dir : pathlib.Path
            Directory containing ``forcing_variables.yml``.

        Returns
        -------
        dict[str, dict[str, str]]
            Mapping from SIR name to config dict with keys:
            ``prms_name``, ``sir_unit``, ``intermediate_unit``, ``source``.

        Examples
        --------
        >>> lookup = deriv._build_sir_to_forcing_lookup(tables_dir)
        >>> lookup["pr_mm_mean"]
        {'prms_name': 'prcp', 'sir_unit': 'mm', 'intermediate_unit': 'mm', 'source': 'gridmet'}
        """
        config = self._load_lookup_table("forcing_variables", tables_dir)
        datasets_config = config["mapping"]

        lookup: dict[str, dict[str, str]] = {}
        for source_name, variables in datasets_config.items():
            for prms_name, var_cfg in variables.items():
                sir_name = var_cfg["sir_name"]
                lookup[sir_name] = {
                    "prms_name": prms_name,
                    "sir_unit": var_cfg["sir_unit"],
                    "intermediate_unit": var_cfg["intermediate_unit"],
                    "source": source_name,
                }
        return lookup
```

**Step 2: Run tests (should still pass — new code, no callers yet)**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py -x --co -q 2>&1 | tail -5
```

**Step 3: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "feat: add _build_sir_to_forcing_lookup reverse mapping (#126)

Invert forcing_variables.yml so SIR canonical names map to their
PRMS name, source, and unit config. Enables per-variable temporal
data lookup without requiring all variables in a single dataset."
```

---

### Task 4: Refactor _derive_forcing() for per-variable iteration

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:2370-2487` (the `_derive_forcing` method)

**Step 1: Replace _derive_forcing implementation**

Replace the entire method body (after the docstring) with per-variable iteration logic. Keep the method signature and docstring unchanged. The new logic:

1. Build the reverse lookup.
2. Group temporal keys by variable base name (strip `_YYYY`), concat multi-year chunks.
3. For each concatenated variable, look up its config in the reverse lookup.
4. If not found, log at DEBUG and skip.
5. Apply unit conversion and feature dimension alignment.
6. Assign to output dataset using the PRMS name.
7. Log a single summary at the end.

```python
        if ctx.temporal is None or len(ctx.temporal) == 0:
            logger.info("No temporal data provided; skipping forcing generation.")
            return ds

        tables_dir = ctx.resolved_lookup_tables_dir
        sir_lookup = self._build_sir_to_forcing_lookup(tables_dir)

        # Group per-variable multi-year chunks: strip _YYYY suffix
        chunks_by_var: dict[str, list[xr.Dataset]] = {}
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            chunks_by_var.setdefault(base_name, []).append(tds)

        forced_count = 0
        for var_base, chunks in chunks_by_var.items():
            # Look up config for this SIR variable
            var_cfg = sir_lookup.get(var_base)
            if var_cfg is None:
                logger.debug(
                    "Temporal variable '%s' is not a forcing variable; skipping.",
                    var_base,
                )
                continue

            prms_name = var_cfg["prms_name"]
            sir_unit = var_cfg["sir_unit"]
            intermediate_unit = var_cfg["intermediate_unit"]

            # Concat multi-year chunks
            if len(chunks) > 1:
                chunks.sort(key=lambda c: c["time"].values[0])
                merged = xr.concat(chunks, dim="time")
            else:
                merged = chunks[0]

            if var_base not in merged:
                logger.warning(
                    "Forcing variable '%s' (SIR name '%s') not found in "
                    "temporal data after concat; skipping.",
                    prms_name,
                    var_base,
                )
                continue

            da = merged[var_base]

            # Unit conversion (SIR unit → intermediate unit)
            if sir_unit != intermediate_unit:
                try:
                    converted = convert(
                        da.values.astype(np.float64), sir_unit, intermediate_unit
                    )
                except KeyError:
                    logger.error(
                        "No unit conversion registered for '%s' → '%s' "
                        "(forcing variable '%s'). Register the conversion "
                        "in units.py or fix forcing_variables.yml.",
                        sir_unit,
                        intermediate_unit,
                        prms_name,
                    )
                    continue
                da = da.copy(data=converted)
                da.attrs["units"] = intermediate_unit

            # Align feature dimension to derived dataset
            target_dim = "nhru"
            feat_dims = [d for d in da.dims if d != "time"]
            if feat_dims and target_dim in ds.dims and feat_dims[0] != target_dim:
                da = da.rename({feat_dims[0]: target_dim})

            ds[prms_name] = da
            forced_count += 1

        if forced_count > 0:
            logger.info("Step 7: merged %d forcing variables.", forced_count)

        return ds
```

**Step 2: Run forcing tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveForcing -v -x 2>&1 | tail -20
```

Expected: Most tests pass. Tests that use source-keyed inline temporal dicts (e.g., `test_swrad_converted_to_langleys` with `"gridmet_2020"` key containing `"srad_W_m2_mean"`) should still pass because the base name becomes `"gridmet"` which isn't in the reverse lookup — but the variable name `srad_W_m2_mean` IS the data var inside the dataset, and the key after stripping `_2020` becomes `"gridmet"` which won't match the reverse lookup.

**These inline temporal dicts need updating to use variable-keyed format.** Update each test that constructs inline temporal dicts with `"gridmet_2020"` as the key to use `"<var_name>_2020"` as the key instead. For example:

`test_swrad_converted_to_langleys`: Change `"gridmet_2020"` → `"srad_W_m2_mean_2020"`
`test_potet_converted_to_inches`: Change `"gridmet_2020"` → `"pet_mm_mean_2020"`
`test_feature_dim_aligned`: Change `"gridmet_2020"` → `"pr_mm_mean_2020"`
`test_missing_sir_variable_skipped`: Change `"gridmet_2020"` → `"pr_mm_mean_2020"`
`test_unknown_source_skipped`: No change needed (already uses `"unknown_source_2020"` with `"some_unknown_var"`)
`test_fuzzy_match_by_variable_names`: This test specifically tested fuzzy matching by source name — the concept no longer applies. Replace it with a test that verifies per-variable lookup works when temporal keys use variable names.
`test_unregistered_conversion_skipped`: Change `"gridmet_2020"` → use variable-keyed format and update custom YAML to match.
`TestDeriveIntegrationForcing::test_derive_with_temporal_produces_forcing`: Uses the `temporal_gridmet` fixture — already updated in Task 2.

**Step 3: Update inline temporal dicts in TestDeriveForcing**

For `test_swrad_converted_to_langleys` (line 1685-1690), change:
```python
        temporal = {
            "srad_W_m2_mean_2020": xr.Dataset(
                {"srad_W_m2_mean": (("time", "nhm_id"), np.array([[100.0, 200.0, 300.0]]))},
                coords={"time": [0], "nhm_id": [1, 2, 3]},
            ),
        }
```

For `test_potet_converted_to_inches` (line 1704-1709), change:
```python
        temporal = {
            "pet_mm_mean_2020": xr.Dataset(
                {"pet_mm_mean": (("time", "nhm_id"), np.array([[25.4, 50.8, 0.0]]))},
                coords={"time": [0], "nhm_id": [1, 2, 3]},
            ),
        }
```

For `test_feature_dim_aligned` (line 1722-1727), change:
```python
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
```

For `test_missing_sir_variable_skipped` (line 1740-1745), change:
```python
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
```

For `test_fuzzy_match_by_variable_names` (line 1770-1793): Replace with a test that verifies per-variable lookup works correctly. The concept of fuzzy matching is gone; replace with:
```python
    def test_per_variable_temporal_matched(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Per-variable temporal keys are matched via reverse lookup."""
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), np.ones((2, 3)) * 20.0)},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
            "tmmn_C_mean_2020": xr.Dataset(
                {"tmmn_C_mean": (("time", "nhm_id"), np.ones((2, 3)) * 5.0)},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert "prcp" in result
        assert "tmax" in result
        assert "tmin" in result
```

For `test_unregistered_conversion_skipped` (line 1827-1834): Change temporal to use variable-keyed format and update the custom YAML so the bogus entry has a unique sir_name:
```python
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), np.ones((2, 3)) * 20.0)},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
```

And update the custom YAML to use a valid sir_name for the bogus entry so it gets matched:
```python
        custom_yaml = {
            "name": "forcing_variables",
            "description": "test",
            "mapping": {
                "gridmet": {
                    "prcp": {
                        "sir_name": "pr_mm_mean",
                        "sir_unit": "mm",
                        "intermediate_unit": "mm",
                    },
                    "bogus": {
                        "sir_name": "tmmx_C_mean",
                        "sir_unit": "C",
                        "intermediate_unit": "furlongs",
                    },
                },
            },
        }
```

**Step 4: Run all forcing tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveForcing tests/test_pywatershed_derivation.py::TestDeriveIntegrationForcing -v -x 2>&1 | tail -20
```

Expected: All pass.

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: refactor _derive_forcing for per-variable temporal data (#126)

Replace source-grouped iteration with per-variable lookup via
_build_sir_to_forcing_lookup(). Each SIR variable is matched to its
forcing config independently. Eliminates ~25 spurious warnings per
run and removes the need for all variables in a single dataset."
```

---

### Task 5: Refactor _compute_monthly_normals() for per-variable lookup

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:2549-2668`

**Step 1: Replace _compute_monthly_normals implementation**

Replace the method body (after the docstring) with direct variable lookup. The new logic:

1. Build the reverse lookup to find tmax and tmin SIR names.
2. Search temporal keys for tmax and tmin base names independently.
3. Concat multi-year chunks of each.
4. Compute monthly means and convert C → F.

```python
        if ctx.temporal is None or len(ctx.temporal) == 0:
            return None

        tables_dir = ctx.resolved_lookup_tables_dir
        sir_lookup = self._build_sir_to_forcing_lookup(tables_dir)

        # Find tmax and tmin SIR names from config
        tmax_sir: str | None = None
        tmin_sir: str | None = None
        for sir_name, cfg in sir_lookup.items():
            if cfg["prms_name"] == "tmax":
                tmax_sir = sir_name
            elif cfg["prms_name"] == "tmin":
                tmin_sir = sir_name

        if tmax_sir is None or tmin_sir is None:
            logger.warning(
                "Forcing config is missing tmax and/or tmin entries; "
                "cannot compute climate normals."
            )
            return None

        # Collect multi-year chunks for tmax and tmin independently
        tmax_chunks: list[xr.Dataset] = []
        tmin_chunks: list[xr.Dataset] = []
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            if base_name == tmax_sir:
                tmax_chunks.append(tds)
            elif base_name == tmin_sir:
                tmin_chunks.append(tds)

        if not tmax_chunks or not tmin_chunks:
            logger.warning(
                "No tmax/tmin variables found in temporal data for climate normals."
            )
            return None

        # Concat multi-year chunks
        if len(tmax_chunks) > 1:
            tmax_chunks.sort(key=lambda c: c["time"].values[0])
            tmax_merged = xr.concat(tmax_chunks, dim="time")
        else:
            tmax_merged = tmax_chunks[0]

        if len(tmin_chunks) > 1:
            tmin_chunks.sort(key=lambda c: c["time"].values[0])
            tmin_merged = xr.concat(tmin_chunks, dim="time")
        else:
            tmin_merged = tmin_chunks[0]

        if tmax_sir not in tmax_merged or tmin_sir not in tmin_merged:
            logger.warning(
                "Temporal data missing tmax='%s' or tmin='%s' after concat.",
                tmax_sir,
                tmin_sir,
            )
            return None

        # Group by month, compute mean, convert C -> F
        tmax_monthly = tmax_merged[tmax_sir].groupby("time.month").mean(dim="time")
        tmin_monthly = tmin_merged[tmin_sir].groupby("time.month").mean(dim="time")

        # Validate full 12-month coverage
        n_months = tmax_monthly.sizes.get("month", 0)
        if n_months != 12:
            logger.warning(
                "Temporal tmax data covers only %d of 12 months; "
                "cannot compute reliable monthly normals. Skipping.",
                n_months,
            )
            return None

        tmax_f = tmax_monthly.values * 9.0 / 5.0 + 32.0
        tmin_f = tmin_monthly.values * 9.0 / 5.0 + 32.0

        # Ensure 2-D shape (12, nhru) for single-HRU case
        if tmax_f.ndim == 1:
            tmax_f = tmax_f[:, np.newaxis]
            tmin_f = tmin_f[:, np.newaxis]

        logger.info(
            "Computed monthly climate normals from tmax='%s', tmin='%s' "
            "(%d + %d timesteps, %d HRUs).",
            tmax_sir,
            tmin_sir,
            tmax_merged.sizes.get("time", 0),
            tmin_merged.sizes.get("time", 0),
            tmax_f.shape[1],
        )
        return tmax_f, tmin_f
```

**Step 2: Run PET and transpiration tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDerivePetCoefficients tests/test_pywatershed_derivation.py::TestDeriveTranspTiming -v -x 2>&1 | tail -20
```

Expected: All pass (these use the updated `temporal_gridmet` fixture from Task 2).

**Step 3: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "feat: refactor _compute_monthly_normals for per-variable temporal (#126)

Find tmax and tmin independently from temporal dict instead of
requiring both in the same dataset. Fixes climate normals computation
that was broken with per-variable SIR temporal files — PET and
transpiration now use climate-derived values instead of defaults."
```

---

### Task 6: Delete _detect_forcing_dataset()

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:2490-2543`

**Step 1: Delete the method**

Remove the entire `_detect_forcing_dataset()` static method (the one between `_derive_forcing` and `_compute_monthly_normals`). It is no longer called by anything.

**Step 2: Run all derivation tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py -v -x 2>&1 | tail -20
```

Expected: All pass.

**Step 3: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "refactor: delete _detect_forcing_dataset (#126)

No longer needed — per-variable reverse lookup replaces source-based
fuzzy matching."
```

---

### Task 7: Add test for _build_sir_to_forcing_lookup

**Files:**
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Add a test class for the reverse lookup helper**

Add this near `TestDeriveForcing` (around line 1620):

```python
class TestBuildSirToForcingLookup:
    """Tests for _build_sir_to_forcing_lookup reverse mapping."""

    def test_returns_all_five_gridmet_variables(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """Reverse lookup contains all 5 gridmet SIR variable names."""
        from importlib.resources import files

        tables_dir = Path(
            str(files("hydro_param").joinpath("data/pywatershed/lookup_tables"))
        )
        lookup = derivation._build_sir_to_forcing_lookup(tables_dir)
        expected_sir_names = {
            "pr_mm_mean", "tmmx_C_mean", "tmmn_C_mean",
            "srad_W_m2_mean", "pet_mm_mean",
        }
        assert set(lookup.keys()) == expected_sir_names

    def test_prms_names_correct(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """Each SIR name maps to the correct PRMS name."""
        from importlib.resources import files

        tables_dir = Path(
            str(files("hydro_param").joinpath("data/pywatershed/lookup_tables"))
        )
        lookup = derivation._build_sir_to_forcing_lookup(tables_dir)
        assert lookup["pr_mm_mean"]["prms_name"] == "prcp"
        assert lookup["tmmx_C_mean"]["prms_name"] == "tmax"
        assert lookup["tmmn_C_mean"]["prms_name"] == "tmin"
        assert lookup["srad_W_m2_mean"]["prms_name"] == "swrad"
        assert lookup["pet_mm_mean"]["prms_name"] == "potet"

    def test_source_field_present(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """Each entry includes the source dataset name."""
        from importlib.resources import files

        tables_dir = Path(
            str(files("hydro_param").joinpath("data/pywatershed/lookup_tables"))
        )
        lookup = derivation._build_sir_to_forcing_lookup(tables_dir)
        for entry in lookup.values():
            assert entry["source"] == "gridmet"
```

**Step 2: Run new tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestBuildSirToForcingLookup -v 2>&1 | tail -10
```

Expected: All 3 pass.

**Step 3: Commit**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: add tests for _build_sir_to_forcing_lookup (#126)"
```

---

### Task 8: Run full check suite and verify

**Step 1: Run full checks**

```bash
pixi run -e dev check
```

Expected: All pass (lint, format, typecheck, tests).

**Step 2: Run pre-commit**

```bash
pixi run -e dev pre-commit
```

Expected: All hooks pass.

**Step 3: Commit any fixups if needed**

If pre-commit hooks made formatting changes, commit them.
