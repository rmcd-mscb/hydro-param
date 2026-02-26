# PET Coefficients & Transpiration Timing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement derivation steps 10 (Jensen-Haise PET coefficients) and 11 (transpiration onset/offset timing) from monthly climate normals.

**Architecture:** Both steps compute from monthly climate normals aggregated from `ctx.temporal` forcing data (gridMET tmax/tmin). A shared `_compute_monthly_normals()` helper aggregates multi-year daily data to 12 monthly means per HRU. Step 10 applies the PRMS-IV Jensen-Haise formulas; step 11 applies frost-date detection logic. Both fall back to defaults when temporal data is unavailable.

**Tech Stack:** numpy, xarray (already dependencies). No new packages.

**Design doc:** `docs/plans/2026-02-25-pet-transpiration-design.md`

---

### Task 1: Fix temporal fixture to use real datetime coordinates

The existing `temporal_gridmet` fixture uses `np.arange(ntime)` as time coords (integers 0..365). Steps 10 and 11 need `da.groupby("time.month")` which requires actual datetime coordinates. Fix the fixture so all temporal tests continue to pass.

**Files:**
- Modify: `tests/test_pywatershed_derivation.py:89-113` (temporal_gridmet fixture)

**Step 1: Update the fixture to use pandas date_range**

Change the `_make_ds` helper inside `temporal_gridmet` to use real dates:

```python
@pytest.fixture()
def temporal_gridmet() -> dict[str, xr.Dataset]:
    """Synthetic SIR-normalized temporal data mimicking gridMET output."""
    import pandas as pd

    nhru = 3
    rng = np.random.default_rng(42)

    def _make_ds(year: int) -> xr.Dataset:
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        ntime = len(times)
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
        "gridmet_2020": _make_ds(2020),
        "gridmet_2021": _make_ds(2021),
    }
```

**Step 2: Run existing forcing tests to verify no regressions**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "Forcing" -v`
Expected: All existing forcing tests PASS (the fixture change is backwards-compatible since forcing tests don't depend on integer time coords).

**Step 3: Commit**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: use real datetime coords in temporal_gridmet fixture

Steps 10 and 11 need groupby('time.month') which requires actual
datetime coordinates, not integer indices."
```

---

### Task 2: Implement `_sat_vp` helper and `_compute_monthly_normals`

Add the saturation vapor pressure function and the shared monthly normals computation.

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (add two functions after line 1169, before `_apply_overrides`)
- Modify: `tests/test_pywatershed_derivation.py` (add test class)

**Step 1: Write the failing tests**

Add at the end of `tests/test_pywatershed_derivation.py`:

```python
class TestSatVp:
    """Tests for saturation vapor pressure helper."""

    def test_freezing_point(self) -> None:
        """sat_vp at 32°F (0°C) should be ~6.11 hPa."""
        from hydro_param.derivations.pywatershed import _sat_vp

        result = _sat_vp(np.array([32.0]))
        np.testing.assert_allclose(result, 6.1078, atol=0.01)

    def test_boiling_point(self) -> None:
        """sat_vp at 212°F (100°C) should be ~1013 hPa."""
        from hydro_param.derivations.pywatershed import _sat_vp

        result = _sat_vp(np.array([212.0]))
        np.testing.assert_allclose(result, 1013.0, rtol=0.02)

    def test_vectorized(self) -> None:
        """sat_vp works on arrays."""
        from hydro_param.derivations.pywatershed import _sat_vp

        temps = np.array([32.0, 50.0, 68.0, 86.0])
        result = _sat_vp(temps)
        assert result.shape == (4,)
        assert np.all(np.diff(result) > 0), "sat_vp should increase with temperature"


class TestComputeMonthlyNormals:
    """Tests for monthly climate normals computation."""

    def test_returns_none_without_temporal(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """No temporal data -> returns None."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        result = derivation._compute_monthly_normals(ctx)
        assert result is None

    def test_returns_monthly_arrays(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """With temporal data, returns (tmax, tmin) each shape (12, nhru)."""
        ctx = DerivationContext(
            sir=sir_topography,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        result = derivation._compute_monthly_normals(ctx)
        assert result is not None
        monthly_tmax, monthly_tmin = result
        assert monthly_tmax.shape == (12, 3)
        assert monthly_tmin.shape == (12, 3)

    def test_units_are_fahrenheit(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """Output normals should be in °F (converted from °C)."""
        ctx = DerivationContext(
            sir=sir_topography,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        monthly_tmax, monthly_tmin = derivation._compute_monthly_normals(ctx)
        # gridMET tmmx_C_mean is uniform(10, 35) °C -> 50-95°F range
        assert np.all(monthly_tmax > 40.0), "Expected °F values (>40)"
        assert np.all(monthly_tmax < 100.0), "Expected °F values (<100)"
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "SatVp or MonthlyNormals" -v`
Expected: FAIL with ImportError or AttributeError (functions don't exist yet).

**Step 3: Implement `_sat_vp` and `_compute_monthly_normals`**

In `src/hydro_param/derivations/pywatershed.py`, add after line 1169 (after `_detect_forcing_dataset`), before the `# Parameter overrides` section:

```python
    # ------------------------------------------------------------------
    # Climate normals helpers (steps 10, 11)
    # ------------------------------------------------------------------

    def _compute_monthly_normals(
        self,
        ctx: DerivationContext,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute monthly mean tmax and tmin from temporal forcing data.

        Aggregates multi-year daily data into 12 monthly means per HRU,
        converting from °C to °F.

        Returns
        -------
        tuple of (monthly_tmax, monthly_tmin) each shape (12, nhru) in °F,
        or None if temporal data is unavailable.
        """
        if ctx.temporal is None or len(ctx.temporal) == 0:
            return None

        tables_dir = ctx.resolved_lookup_tables_dir
        config = self._load_lookup_table("forcing_variables", tables_dir)
        datasets_config = config["mapping"]

        # Concat multi-year chunks by base name
        chunks_by_source: dict[str, list[xr.Dataset]] = {}
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            chunks_by_source.setdefault(base_name, []).append(tds)

        for source_name, chunks in chunks_by_source.items():
            if len(chunks) > 1:
                chunks.sort(key=lambda c: c["time"].values[0])
                merged = xr.concat(chunks, dim="time")
            else:
                merged = chunks[0]

            dataset_cfg = self._detect_forcing_dataset(
                source_name, merged, datasets_config
            )
            if dataset_cfg is None:
                continue

            # Find tmax and tmin SIR variable names
            tmax_sir = dataset_cfg.get("tmax", {}).get("sir_name")
            tmin_sir = dataset_cfg.get("tmin", {}).get("sir_name")

            if tmax_sir is None or tmin_sir is None:
                continue
            if tmax_sir not in merged or tmin_sir not in merged:
                continue

            # Group by month, compute mean, convert C -> F
            tmax_monthly = merged[tmax_sir].groupby("time.month").mean(dim="time")
            tmin_monthly = merged[tmin_sir].groupby("time.month").mean(dim="time")

            tmax_f = tmax_monthly.values * 9.0 / 5.0 + 32.0
            tmin_f = tmin_monthly.values * 9.0 / 5.0 + 32.0

            logger.info(
                "Computed monthly climate normals from '%s' (%d timesteps, %d HRUs).",
                source_name,
                merged.sizes.get("time", 0),
                tmax_f.shape[1] if tmax_f.ndim > 1 else 1,
            )
            return tmax_f, tmin_f

        logger.warning("No tmax/tmin variables found in temporal data for climate normals.")
        return None
```

Also add the module-level `_sat_vp` function near the top of the file, after the `_SEED_METHODS` dict (after line 71):

```python
def _sat_vp(temp_f: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure (hPa) from temperature in °F.

    Uses the Magnus formula (Alduchov & Eskridge 1996).

    Parameters
    ----------
    temp_f
        Temperature in degrees Fahrenheit.

    Returns
    -------
    np.ndarray
        Saturation vapor pressure in hectopascals (hPa).
    """
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    return 6.1078 * np.exp(17.269 * temp_c / (temp_c + 237.3))
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "SatVp or MonthlyNormals" -v`
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: add _sat_vp and _compute_monthly_normals helpers

Saturation vapor pressure (Magnus formula) and shared monthly
climate normals computation for steps 10 and 11."
```

---

### Task 3: Implement step 10 — `_derive_pet_coefficients`

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (add method, wire into `derive()`)
- Modify: `tests/test_pywatershed_derivation.py` (add test class)

**Step 1: Write the failing tests**

Add at the end of `tests/test_pywatershed_derivation.py`:

```python
class TestDerivePetCoefficients:
    """Tests for step 10: Jensen-Haise PET coefficient derivation."""

    def test_jh_coef_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """jh_coef should have shape (nhru, 12)."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "jh_coef" in ds
        assert ds["jh_coef"].shape == (3, 12)

    def test_jh_coef_hru_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """jh_coef_hru should have shape (nhru,)."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "jh_coef_hru" in ds
        assert ds["jh_coef_hru"].shape == (3,)

    def test_jh_coef_in_valid_range(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """jh_coef values should be in [0.005, 0.06]."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert np.all(ds["jh_coef"].values >= 0.005)
        assert np.all(ds["jh_coef"].values <= 0.06)

    def test_jh_coef_formula_known_values(self) -> None:
        """Test jh_coef formula with hand-computed values."""
        from hydro_param.derivations.pywatershed import _sat_vp

        # tmax=80°F, tmin=50°F
        svp_max = _sat_vp(np.array([80.0]))[0]
        svp_min = _sat_vp(np.array([50.0]))[0]
        expected = 27.5 - 0.25 * (svp_max - svp_min) / svp_max
        assert 0.005 < expected < 0.06, f"Expected in range, got {expected}"

    def test_fallback_without_temporal(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Without temporal data, jh_coef/jh_coef_hru use defaults."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # Defaults set by _apply_defaults (step 13)
        assert "jh_coef" in ds
        assert "jh_coef_hru" in ds
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "PetCoefficient" -v`
Expected: FAIL (method doesn't exist or `jh_coef` not in dataset).

**Step 3: Implement `_derive_pet_coefficients`**

Add the method to `PywatershedDerivation` in `pywatershed.py`, after `_compute_monthly_normals`:

```python
    # ------------------------------------------------------------------
    # Step 10: PET coefficients (Jensen-Haise)
    # ------------------------------------------------------------------

    def _derive_pet_coefficients(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Step 10: Derive Jensen-Haise PET coefficients.

        Computes ``jh_coef`` (nhru, 12) and ``jh_coef_hru`` (nhru,) from
        monthly climate normals using PRMS-IV equation 1-26.

        Falls back to step 13 defaults when no temporal data is available.
        """
        normals = self._compute_monthly_normals(ctx)
        if normals is None:
            logger.info(
                "No temporal data for PET coefficients; deferring to defaults."
            )
            return ds

        monthly_tmax, monthly_tmin = normals  # (12, nhru) in °F
        nhru = monthly_tmax.shape[1]

        # --- jh_coef: PRMS-IV eq. 1-26 ---
        svp_max = _sat_vp(monthly_tmax)  # (12, nhru)
        svp_min = _sat_vp(monthly_tmin)  # (12, nhru)

        # Guard against division by zero
        svp_max_safe = np.where(svp_max < 1e-6, 1e-6, svp_max)
        jh_coef = 27.5 - 0.25 * (svp_max - svp_min) / svp_max_safe
        jh_coef = np.clip(jh_coef, 0.005, 0.06)

        # Transpose to (nhru, 12) for output convention
        ds["jh_coef"] = xr.DataArray(
            jh_coef.T,
            dims=("nhru", "nmonths"),
            attrs={"units": "per_degF_per_day", "long_name": "Jensen-Haise PET coefficient"},
        )

        # --- jh_coef_hru: elevation-adjusted coefficient ---
        # Use July (index 6) as warmest month for base coefficient
        july_jh = jh_coef[6, :]  # (nhru,)

        # Elevation adjustment: higher elevations have lower boiling point,
        # increasing vapor pressure deficit → slightly higher coefficients.
        # Linear approximation: +0.00001 per foot above sea level.
        if "hru_elev" in ds:
            elev_ft = ds["hru_elev"].values
            jh_coef_hru = july_jh + 0.00001 * elev_ft
        else:
            jh_coef_hru = july_jh

        jh_coef_hru = np.clip(jh_coef_hru, 0.005, 0.06)
        ds["jh_coef_hru"] = xr.DataArray(
            jh_coef_hru,
            dims=("nhru",),
            attrs={"units": "per_degF_per_day", "long_name": "Per-HRU Jensen-Haise coefficient"},
        )

        logger.info(
            "Step 10: derived jh_coef (%d HRUs × 12 months) and jh_coef_hru.",
            nhru,
        )
        return ds
```

Also add default values for `jh_coef` and `jh_coef_hru` to the `_DEFAULTS` dict:

```python
    # PET (Jensen-Haise)
    "jh_coef": 0.014,
    "jh_coef_hru": 0.014,
```

And update the `_apply_defaults` method to handle the 2D `jh_coef` default — it needs shape `(nhru, 12)`:

In `_apply_defaults`, add special handling for `jh_coef`:

```python
        if "jh_coef" not in ds:
            ds["jh_coef"] = xr.DataArray(
                np.full((nhru, 12), _DEFAULTS["jh_coef"]),
                dims=("nhru", "nmonths"),
                attrs={"units": "per_degF_per_day", "long_name": "Jensen-Haise PET coefficient (default)"},
            )
```

Wire step 10 into `derive()` — insert after step 9 (soltab), before step 13:

```python
        # Step 10: PET coefficients (Jensen-Haise)
        ds = self._derive_pet_coefficients(context, ds)
```

Update the module docstring to include step 10.

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "PetCoefficient" -v`
Expected: All 5 tests PASS.

Then run full suite:
Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: All tests PASS (no regressions).

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: implement step 10 — Jensen-Haise PET coefficients

Derives jh_coef (nhru, 12) and jh_coef_hru (nhru,) from monthly
climate normals. Falls back to defaults when temporal data is
unavailable. PRMS-IV equation 1-26."
```

---

### Task 4: Implement step 11 — `_derive_transp_timing`

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (add method, wire into `derive()`)
- Modify: `tests/test_pywatershed_derivation.py` (add test class)

**Step 1: Write the failing tests**

Add at the end of `tests/test_pywatershed_derivation.py`:

```python
class TestDeriveTranspTiming:
    """Tests for step 11: transpiration timing derivation."""

    def test_transp_beg_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_beg should have shape (nhru,)."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "transp_beg" in ds
        assert ds["transp_beg"].shape == (3,)

    def test_transp_end_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_end should have shape (nhru,)."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "transp_end" in ds
        assert ds["transp_end"].shape == (3,)

    def test_values_are_valid_months(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_beg and transp_end should be integers in [1, 12]."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert np.all(ds["transp_beg"].values >= 1)
        assert np.all(ds["transp_beg"].values <= 12)
        assert np.all(ds["transp_end"].values >= 1)
        assert np.all(ds["transp_end"].values <= 12)

    def test_beg_before_end(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_beg should be before transp_end for temperate climates."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert np.all(ds["transp_beg"].values < ds["transp_end"].values)

    def test_warm_climate_early_onset(self, derivation: PywatershedDerivation) -> None:
        """Warm climate (all tmin > 32°F) -> transp_beg = 1 (January)."""
        import pandas as pd

        nhru = 1
        times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        temporal = {
            "gridmet_2020": xr.Dataset(
                {
                    "tmmx_C_mean": (("time", "nhm_id"), np.full((len(times), nhru), 30.0)),
                    "tmmn_C_mean": (("time", "nhm_id"), np.full((len(times), nhru), 15.0)),
                },
                coords={"time": times, "nhm_id": [1]},
            )
        }
        sir = xr.Dataset(
            {
                "elevation_m_mean": ("nhm_id", np.array([100.0])),
                "slope_deg_mean": ("nhm_id", np.array([5.0])),
                "aspect_deg_mean": ("nhm_id", np.array([180.0])),
                "hru_lat": ("nhm_id", np.array([30.0])),
                "hru_area_m2": ("nhm_id", np.array([4046856.0])),
            },
            coords={"nhm_id": [1]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", temporal=temporal)
        ds = derivation.derive(ctx)
        assert ds["transp_beg"].values[0] == 1

    def test_cold_climate_late_onset(self, derivation: PywatershedDerivation) -> None:
        """Cold climate with short growing season -> later transp_beg."""
        import pandas as pd

        nhru = 1
        times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        # Cold: tmin < 0°C (32°F) for Jan-May, warm Jun-Aug, cold Sep-Dec
        tmin_values = np.full((len(times), nhru), -5.0)  # °C, well below freezing
        # Warm only in summer months (Jun=152, Jul, Aug=243)
        tmin_values[152:244, :] = 10.0  # °C, above freezing

        temporal = {
            "gridmet_2020": xr.Dataset(
                {
                    "tmmx_C_mean": (("time", "nhm_id"), tmin_values + 15.0),
                    "tmmn_C_mean": (("time", "nhm_id"), tmin_values),
                },
                coords={"time": times, "nhm_id": [1]},
            )
        }
        sir = xr.Dataset(
            {
                "elevation_m_mean": ("nhm_id", np.array([2000.0])),
                "slope_deg_mean": ("nhm_id", np.array([10.0])),
                "aspect_deg_mean": ("nhm_id", np.array([180.0])),
                "hru_lat": ("nhm_id", np.array([45.0])),
                "hru_area_m2": ("nhm_id", np.array([4046856.0])),
            },
            coords={"nhm_id": [1]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", temporal=temporal)
        ds = derivation.derive(ctx)
        # June onset (month 6) expected
        assert ds["transp_beg"].values[0] >= 5
        assert ds["transp_end"].values[0] <= 10

    def test_fallback_without_temporal(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Without temporal data, transp_beg/end use defaults."""
        sir = xr.merge([
            sir_topography,
            xr.Dataset({
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            }, coords={"nhm_id": [1, 2, 3]}),
        ])
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "transp_beg" in ds
        assert "transp_end" in ds
        # Defaults: beg=4, end=10
        np.testing.assert_array_equal(ds["transp_beg"].values, [4, 4, 4])
        np.testing.assert_array_equal(ds["transp_end"].values, [10, 10, 10])
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "TranspTiming" -v`
Expected: FAIL (method doesn't exist or params not in dataset).

**Step 3: Implement `_derive_transp_timing`**

Add the method to `PywatershedDerivation`, after `_derive_pet_coefficients`:

```python
    # ------------------------------------------------------------------
    # Step 11: Transpiration timing (frost-free period)
    # ------------------------------------------------------------------

    def _derive_transp_timing(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Step 11: Derive transpiration onset/offset from monthly tmin.

        Computes ``transp_beg`` and ``transp_end`` (integer months) by
        detecting the frost-free period from monthly minimum temperature
        normals.  Falls back to step 13 defaults when no temporal data
        is available.
        """
        normals = self._compute_monthly_normals(ctx)
        if normals is None:
            logger.info(
                "No temporal data for transpiration timing; deferring to defaults."
            )
            return ds

        _monthly_tmax, monthly_tmin = normals  # (12, nhru) in °F
        nhru = monthly_tmin.shape[1]
        freezing = 32.0  # °F

        # transp_beg: first month (1-indexed) where tmin > freezing
        transp_beg = np.full(nhru, 4, dtype=np.int32)  # default April
        for hru in range(nhru):
            for month_idx in range(12):
                if monthly_tmin[month_idx, hru] > freezing:
                    transp_beg[hru] = month_idx + 1  # 1-indexed
                    break

        # transp_end: first month after June (7+) where tmin < freezing
        transp_end = np.full(nhru, 10, dtype=np.int32)  # default October
        for hru in range(nhru):
            for month_idx in range(6, 12):  # July onward
                if monthly_tmin[month_idx, hru] < freezing:
                    transp_end[hru] = month_idx + 1  # 1-indexed
                    break

        ds["transp_beg"] = xr.DataArray(
            transp_beg,
            dims=("nhru",),
            attrs={"units": "integer_month", "long_name": "Month transpiration begins"},
        )
        ds["transp_end"] = xr.DataArray(
            transp_end,
            dims=("nhru",),
            attrs={"units": "integer_month", "long_name": "Month transpiration ends"},
        )

        logger.info(
            "Step 11: derived transp_beg (range %d–%d) and transp_end (range %d–%d) for %d HRUs.",
            int(transp_beg.min()),
            int(transp_beg.max()),
            int(transp_end.min()),
            int(transp_end.max()),
            nhru,
        )
        return ds
```

Add default values to `_DEFAULTS`:

```python
    # Transpiration timing
    "transp_beg": 4,   # April
    "transp_end": 10,  # October
```

And add defaults handling in `_apply_defaults`:

```python
        for param in ("transp_beg", "transp_end"):
            if param not in ds:
                ds[param] = xr.DataArray(
                    np.full(nhru, int(_DEFAULTS[param]), dtype=np.int32),
                    dims=("nhru",),
                    attrs={"units": "integer_month", "long_name": f"{param} (default)"},
                )
```

Wire step 11 into `derive()` — insert after step 10:

```python
        # Step 11: Transpiration timing (frost-free period)
        ds = self._derive_transp_timing(context, ds)
```

Update the module docstring to include step 11.

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "TranspTiming" -v`
Expected: All 8 tests PASS.

Then run full suite:
Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: implement step 11 — transpiration timing from frost dates

Derives transp_beg and transp_end (integer months) from monthly
tmin normals using frost-free period detection. Falls back to
defaults (April/October) when temporal data is unavailable."
```

---

### Task 5: Integration test and full checks

**Files:**
- Modify: `tests/test_pywatershed_derivation.py` (add integration test)

**Step 1: Write integration test**

Add at the end of `tests/test_pywatershed_derivation.py`:

```python
class TestDeriveIntegrationPetTransp:
    """Integration test: full derive() produces PET and transpiration params."""

    def test_full_pipeline_with_temporal_produces_all_params(
        self,
        derivation: PywatershedDerivation,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """Full derive() with temporal data produces jh_coef, jh_coef_hru, transp_beg, transp_end."""
        sir = xr.Dataset(
            {
                "elevation_m_mean": ("nhm_id", np.array([200.0, 800.0])),
                "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0])),
                "aspect_deg_mean": ("nhm_id", np.array([180.0, 90.0])),
                "hru_lat": ("nhm_id", np.array([42.0, 43.0])),
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0])),
                "land_cover": ("nhm_id", np.array([42, 71])),
                "fctimp_pct_mean": ("nhm_id", np.array([10.0, 5.0])),
                "tree_canopy_pct_mean": ("nhm_id", np.array([80.0, 10.0])),
                "awc_mm_mean": ("nhm_id", np.array([100.0, 200.0])),
                "soil_texture_frac_sand": ("nhm_id", np.array([0.5, 0.2])),
                "soil_texture_frac_loam": ("nhm_id", np.array([0.3, 0.6])),
                "soil_texture_frac_clay": ("nhm_id", np.array([0.2, 0.2])),
            },
            coords={"nhm_id": [1, 2]},
        )
        # Adjust temporal to match 2-HRU SIR
        import pandas as pd
        rng = np.random.default_rng(42)
        nhru = 2
        def _make_ds(year: int) -> xr.Dataset:
            times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            ntime = len(times)
            return xr.Dataset(
                {
                    "pr_mm_mean": (("time", "nhm_id"), rng.uniform(0, 20, (ntime, nhru))),
                    "tmmx_C_mean": (("time", "nhm_id"), rng.uniform(10, 35, (ntime, nhru))),
                    "tmmn_C_mean": (("time", "nhm_id"), rng.uniform(-5, 15, (ntime, nhru))),
                    "srad_W_m2_mean": (("time", "nhm_id"), rng.uniform(50, 300, (ntime, nhru))),
                    "pet_mm_mean": (("time", "nhm_id"), rng.uniform(0, 8, (ntime, nhru))),
                },
                coords={"time": times, "nhm_id": [1, 2]},
            )

        temporal = {
            "gridmet_2020": _make_ds(2020),
            "gridmet_2021": _make_ds(2021),
        }

        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal,
        )
        ds = derivation.derive(ctx)

        # PET params
        assert "jh_coef" in ds
        assert ds["jh_coef"].dims == ("nhru", "nmonths")
        assert ds["jh_coef"].shape == (2, 12)
        assert "jh_coef_hru" in ds
        assert ds["jh_coef_hru"].shape == (2,)

        # Transpiration params
        assert "transp_beg" in ds
        assert "transp_end" in ds
        assert ds["transp_beg"].shape == (2,)
        assert ds["transp_end"].shape == (2,)
        assert np.all(ds["transp_beg"].values >= 1)
        assert np.all(ds["transp_end"].values <= 12)
```

**Step 2: Run the integration test**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveIntegrationPetTransp -v`
Expected: PASS.

**Step 3: Run full check suite**

Run: `pixi run -e dev check`
Expected: All checks pass (lint, format, typecheck, tests).

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass.

**Step 4: Commit**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: add integration test for steps 10 and 11

Verifies full derive() pipeline with temporal data produces
jh_coef, jh_coef_hru, transp_beg, and transp_end."
```

---

### Task 6: Update module docstring and run final verification

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (module docstring)

**Step 1: Update module docstring**

Change the module docstring at line 7 to:

```python
Foundation implementation covers steps 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, and 14.
```

Also update the class docstring (around line 160) to include steps 10 and 11.

**Step 2: Run full check suite one final time**

Run: `pixi run -e dev check`
Expected: All checks pass.

**Step 3: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "docs: update module docstring for steps 10 and 11"
```

---

## Summary

| Task | Description | New tests |
|------|-------------|-----------|
| 1 | Fix temporal fixture datetime coords | 0 (fixture fix) |
| 2 | `_sat_vp` + `_compute_monthly_normals` | 6 |
| 3 | Step 10: `_derive_pet_coefficients` | 5 |
| 4 | Step 11: `_derive_transp_timing` | 8 |
| 5 | Integration test + full checks | 1 |
| 6 | Docstring update + final verification | 0 |

**Total new tests:** ~20
**Files modified:** 2 (`pywatershed.py`, `test_pywatershed_derivation.py`)
**New files:** 0
**New dependencies:** 0
