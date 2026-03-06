# soil_rechr_max_frac from gNATSGO AWC Ratio — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Derive `soil_rechr_max_frac` from the ratio of gNATSGO `aws0_50` (0–50cm) to `aws0_100` (0–100cm) instead of using a uniform 0.4 default.

**Architecture:** Add `aws0_50` to the gNATSGO dataset registry, then replace the constant-default block in `_derive_soils()` with a ratio computation that falls back to 0.4 when data is missing. No new dependencies.

**Tech Stack:** numpy, xarray, pytest, YAML registry

---

### Task 1: Add `aws0_50` variable to soils.yml

**Files:**
- Modify: `src/hydro_param/data/datasets/soils.yml:197-225`

**Step 1: Add the variable entry**

Add `aws0_50` as the first variable in the `gnatsgo_rasters.variables` list (before `aws0_100`), following the same pattern:

```yaml
      - name: aws0_50
        band: 1
        units: "mm"
        long_name: "Available water storage 0-50cm"
        native_name: "aws0_50"
        categorical: false
        asset_key: "aws0_50"
```

Insert this block at line 198, before the existing `aws0_100` entry.

**Step 2: Verify YAML is valid**

Run: `python -c "import yaml; yaml.safe_load(open('src/hydro_param/data/datasets/soils.yml'))"`
Expected: No output (clean parse)

**Step 3: Commit**

```bash
git add src/hydro_param/data/datasets/soils.yml
git commit -m "feat: add aws0_50 variable to gNATSGO dataset registry (#151)"
```

---

### Task 2: Write failing tests for AWC-ratio soil_rechr_max_frac

**Files:**
- Modify: `tests/test_pywatershed_derivation.py` (in `class TestDeriveSoils`)

**Step 1: Write 3 new tests after `test_soil_rechr_max_frac_default`**

Add these tests inside `class TestDeriveSoils`, after the existing `test_soil_rechr_max_frac_default` method (around line 462):

```python
    def test_soil_rechr_max_frac_from_awc_ratio(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Computes aws0_50/aws0_100 ratio when both variables present."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.3])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.5])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.2])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([100.0, 200.0])),
                    "aws0_50_mm_mean": ("nhm_id", np.array([60.0, 80.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_rechr_max_frac" in ds
        # 60/100 = 0.6, 80/200 = 0.4
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, [0.6, 0.4], atol=1e-6)
        assert ds["soil_rechr_max_frac"].attrs["units"] == "decimal_fraction"

    def test_soil_rechr_max_frac_clipped(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Ratio clipped to [0.1, 0.9] physical bounds."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.7])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.2])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([100.0, 100.0])),
                    "aws0_50_mm_mean": ("nhm_id", np.array([5.0, 99.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # 5/100 = 0.05 -> clips to 0.1;  99/100 = 0.99 -> clips to 0.9
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, [0.1, 0.9])

    def test_soil_rechr_max_frac_zero_aws100_uses_default(
        self, derivation: PywatershedDerivation
    ) -> None:
        """HRUs with aws0_100 = 0 get the 0.4 default (no data)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.7])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.2])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([0.0, 100.0])),
                    "aws0_50_mm_mean": ("nhm_id", np.array([0.0, 40.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # HRU 1: aws0_100=0 -> default 0.4;  HRU 2: 40/100=0.4
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, [0.4, 0.4])
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSoils::test_soil_rechr_max_frac_from_awc_ratio tests/test_pywatershed_derivation.py::TestDeriveSoils::test_soil_rechr_max_frac_clipped tests/test_pywatershed_derivation.py::TestDeriveSoils::test_soil_rechr_max_frac_zero_aws100_uses_default -v`

Expected: FAIL — `test_soil_rechr_max_frac_from_awc_ratio` fails because current code always returns 0.4. The clipping and zero-aws100 tests may pass or fail depending on values.

**Step 3: Commit failing tests**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: add soil_rechr_max_frac AWC ratio test cases (#151)"
```

---

### Task 3: Implement AWC ratio derivation in `_derive_soils()`

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:1944-1961`

**Step 1: Replace the constant-default block**

Replace the `# --- soil_rechr_max_frac ---` block (lines 1944–1960) with:

```python
        # --- soil_rechr_max_frac ---
        # Prefer derived ratio: aws0_50 (0-50cm, ~upper 18 inches) / aws0_100
        # (0-100cm, full root zone).  Falls back to constant default when
        # either variable is missing from the SIR.
        aws50_key = sir.find_variable("aws0_50_mm_mean")
        if aws50_key is not None and aws_key is not None:
            aws50_mm = sir[aws50_key].values.astype(np.float64)
            aws100_mm = sir[aws_key].values.astype(np.float64)
            # Guard division by zero: HRUs with zero total AWC get default
            valid = aws100_mm > 0
            ratio = np.full_like(aws100_mm, self._SOIL_RECHR_MAX_FRAC_DEFAULT)
            ratio[valid] = aws50_mm[valid] / aws100_mm[valid]
            ratio = np.clip(ratio, 0.1, 0.9)
            ds["soil_rechr_max_frac"] = xr.DataArray(
                ratio,
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Fraction of soil moisture in recharge zone",
                },
            )
            logger.info(
                "soil_rechr_max_frac derived from aws0_50/aws0_100 ratio for %d HRUs "
                "(%.1f%% had zero aws0_100, set to default %.2f)",
                len(ratio),
                100.0 * np.sum(~valid) / len(ratio),
                self._SOIL_RECHR_MAX_FRAC_DEFAULT,
            )
        elif "soil_type" in ds:
            nhru = len(ds["soil_type"])
            ds["soil_rechr_max_frac"] = xr.DataArray(
                np.full(nhru, self._SOIL_RECHR_MAX_FRAC_DEFAULT),
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Fraction of soil moisture in recharge zone",
                },
            )
            logger.debug(
                "soil_rechr_max_frac set to default %.2f for %d HRUs "
                "(aws0_50_mm_mean not available in SIR)",
                self._SOIL_RECHR_MAX_FRAC_DEFAULT,
                nhru,
            )
```

Note: `aws_key` is already resolved earlier in the method (line 1899) as `sir.find_variable("aws0_100_mm_mean")`. We reuse it here.

**Step 2: Update the docstring**

In the `_derive_soils` docstring (around line 1861), replace:

```
``soil_rechr_max_frac`` is set to a constant default of 0.4
(no soil layer depth data is currently available from the SIR
to compute it from first principles).
```

with:

```
``soil_rechr_max_frac`` is derived from the ratio
``aws0_50_mm / aws0_100_mm`` when both variables are present
in the SIR.  ``aws0_50`` (0–50 cm ≈ upper 18 inches) approximates
the PRMS recharge zone; ``aws0_100`` (0–100 cm) approximates the
full root zone.  HRUs with zero ``aws0_100`` receive the default
0.4.  The ratio is clipped to ``[0.1, 0.9]``.  Falls back to a
uniform 0.4 when ``aws0_50_mm_mean`` is absent from the SIR.
```

**Step 3: Run the new tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSoils -v`
Expected: ALL PASS (including the 3 new tests and the existing `test_soil_rechr_max_frac_default`)

**Step 4: Run the full test suite**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "feat: derive soil_rechr_max_frac from aws0_50/aws0_100 ratio (#151)"
```

---

### Task 4: Update pywatershed_dataset_param_map.yml

**Files:**
- Modify: `docs/reference/pywatershed_dataset_param_map.yml:449-458`

**Step 1: Update the soil_rechr_max_frac entry**

Replace the existing entry:

```yaml
  soil_rechr_max_frac:
    description: "Max recharge zone storage as fraction of soil_moist_max"
    units: "decimal fraction (0–1)"
    dimension: nhru
    source_dataset: statsgo2
    derivation_type: derived_formula
    method: >
      soil_rechr_max_frac = AWC_upper_18inches / AWC_total_rootzone
      Uses depth-weighted AWC of upper soil layers vs total.
    dependencies: [awc, soil_depth]
```

with:

```yaml
  soil_rechr_max_frac:
    description: "Max recharge zone storage as fraction of soil_moist_max"
    units: "decimal fraction (0–1)"
    dimension: nhru
    source_dataset: gnatsgo
    derivation_type: derived_formula
    method: >
      soil_rechr_max_frac = aws0_50_mm / aws0_100_mm
      where aws0_50 (0-50cm) approximates the PRMS recharge zone (~upper 18 inches)
      and aws0_100 (0-100cm) approximates the full root zone.
      Clipped to [0.1, 0.9].  Falls back to constant 0.4 when aws0_50 is absent.
    dependencies: [aws0_50, aws0_100]
```

**Step 2: Commit**

```bash
git add docs/reference/pywatershed_dataset_param_map.yml
git commit -m "docs: update soil_rechr_max_frac derivation spec for AWC ratio (#151)"
```

---

### Task 5: Run full checks and verify

**Step 1: Run all checks**

Run: `pixi run -e dev check`
Expected: ALL PASS (lint, format, typecheck, tests)

**Step 2: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: ALL PASS

**Step 3: Final commit if any auto-formatting changes**

If ruff/formatting made changes:
```bash
git add -u
git commit -m "chore: apply formatting fixes"
```
