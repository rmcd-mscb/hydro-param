# Soil Texture Triangle Classification — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add USDA soil texture triangle classification so `_compute_soil_type()` can derive PRMS soil_type from continuous POLARIS sand/silt/clay percentages.

**Architecture:** Add a private `_classify_usda_texture()` helper that implements the standard USDA texture triangle conditional logic, then add a third fallback path in `_compute_soil_type()` that uses it when `sand_pct_mean`, `silt_pct_mean`, `clay_pct_mean` are available in SIR.

**Tech Stack:** numpy, xarray (existing deps only)

**Design doc:** `docs/plans/2026-03-02-soil-texture-triangle-design.md`

---

### Task 1: USDA texture triangle classifier — tests

**Files:**
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing tests for `_classify_usda_texture()`**

Add a new test class after `TestDeriveSoils` (around line 580). The function is a private helper on `PywatershedDerivation`, so access it via `derivation._classify_usda_texture()`.

```python
class TestClassifyUsdaTexture:
    """Tests for USDA soil texture triangle classification."""

    def test_pure_sand(self, derivation: PywatershedDerivation) -> None:
        """High sand, low clay -> sand."""
        result = derivation._classify_usda_texture(
            np.array([90.0]), np.array([5.0]), np.array([5.0])
        )
        assert result[0] == "sand"

    def test_pure_clay(self, derivation: PywatershedDerivation) -> None:
        """High clay -> clay."""
        result = derivation._classify_usda_texture(
            np.array([20.0]), np.array([20.0]), np.array([60.0])
        )
        assert result[0] == "clay"

    def test_loam_center(self, derivation: PywatershedDerivation) -> None:
        """Classic loam composition."""
        result = derivation._classify_usda_texture(
            np.array([40.0]), np.array([40.0]), np.array([20.0])
        )
        assert result[0] == "loam"

    def test_silt(self, derivation: PywatershedDerivation) -> None:
        """Very high silt, low clay -> silt."""
        result = derivation._classify_usda_texture(
            np.array([5.0]), np.array([90.0]), np.array([5.0])
        )
        assert result[0] == "silt"

    def test_silt_loam(self, derivation: PywatershedDerivation) -> None:
        """High silt, moderate clay -> silt_loam."""
        result = derivation._classify_usda_texture(
            np.array([20.0]), np.array([60.0]), np.array([20.0])
        )
        assert result[0] == "silt_loam"

    def test_sandy_loam(self, derivation: PywatershedDerivation) -> None:
        """Moderate sand, low clay -> sandy_loam."""
        result = derivation._classify_usda_texture(
            np.array([65.0]), np.array([25.0]), np.array([10.0])
        )
        assert result[0] == "sandy_loam"

    def test_loamy_sand(self, derivation: PywatershedDerivation) -> None:
        """High sand but not pure sand -> loamy_sand."""
        result = derivation._classify_usda_texture(
            np.array([80.0]), np.array([10.0]), np.array([10.0])
        )
        assert result[0] == "loamy_sand"

    def test_clay_loam(self, derivation: PywatershedDerivation) -> None:
        """Moderate clay, moderate sand -> clay_loam."""
        result = derivation._classify_usda_texture(
            np.array([30.0]), np.array([35.0]), np.array([35.0])
        )
        assert result[0] == "clay_loam"

    def test_silty_clay_loam(self, derivation: PywatershedDerivation) -> None:
        """Moderate clay, high silt, low sand -> silty_clay_loam."""
        result = derivation._classify_usda_texture(
            np.array([10.0]), np.array([55.0]), np.array([35.0])
        )
        assert result[0] == "silty_clay_loam"

    def test_sandy_clay_loam(self, derivation: PywatershedDerivation) -> None:
        """Moderate clay, high sand, low silt -> sandy_clay_loam."""
        result = derivation._classify_usda_texture(
            np.array([60.0]), np.array([15.0]), np.array([25.0])
        )
        assert result[0] == "sandy_clay_loam"

    def test_sandy_clay(self, derivation: PywatershedDerivation) -> None:
        """High clay + high sand -> sandy_clay."""
        result = derivation._classify_usda_texture(
            np.array([50.0]), np.array([10.0]), np.array([40.0])
        )
        assert result[0] == "sandy_clay"

    def test_silty_clay(self, derivation: PywatershedDerivation) -> None:
        """High clay + high silt -> silty_clay."""
        result = derivation._classify_usda_texture(
            np.array([5.0]), np.array([50.0]), np.array([45.0])
        )
        assert result[0] == "silty_clay"

    def test_vectorized_multiple_hrus(self, derivation: PywatershedDerivation) -> None:
        """Classifies multiple HRUs at once."""
        sand = np.array([90.0, 20.0, 40.0])
        silt = np.array([5.0, 20.0, 40.0])
        clay = np.array([5.0, 60.0, 20.0])
        result = derivation._classify_usda_texture(sand, silt, clay)
        assert list(result) == ["sand", "clay", "loam"]

    def test_nan_values_default_to_loam(self, derivation: PywatershedDerivation) -> None:
        """NaN inputs classify as loam (safe default)."""
        result = derivation._classify_usda_texture(
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        )
        assert result[0] == "loam"
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py -k "TestClassifyUsdaTexture" -v`
Expected: FAIL — `AttributeError: 'PywatershedDerivation' has no attribute '_classify_usda_texture'`

**Step 3: Commit failing tests**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: add USDA texture triangle classification tests (red)"
```

---

### Task 2: USDA texture triangle classifier — implementation

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Add `_classify_usda_texture()` method**

Add this method to the `PywatershedDerivation` class, before `_compute_soil_type()` (around line 1622, after `_derive_soils()` returns).

```python
    def _classify_usda_texture(
        self,
        sand: np.ndarray,
        silt: np.ndarray,
        clay: np.ndarray,
    ) -> np.ndarray:
        """Classify sand/silt/clay percentages into USDA texture class names.

        Apply the standard USDA soil texture triangle decision tree to
        assign one of 12 texture classes to each element.  Boundaries
        follow the USDA Natural Resources Conservation Service (NRCS)
        soil texture calculator definitions.

        Parameters
        ----------
        sand : np.ndarray
            Sand content as percentage (0–100), shape ``(n,)``.
        silt : np.ndarray
            Silt content as percentage (0–100), shape ``(n,)``.
        clay : np.ndarray
            Clay content as percentage (0–100), shape ``(n,)``.

        Returns
        -------
        np.ndarray
            Array of USDA texture class name strings, shape ``(n,)``.
            Valid classes: ``sand``, ``loamy_sand``, ``sandy_loam``,
            ``loam``, ``silt_loam``, ``silt``, ``sandy_clay_loam``,
            ``clay_loam``, ``silty_clay_loam``, ``sandy_clay``,
            ``silty_clay``, ``clay``.  NaN inputs default to ``loam``.

        Notes
        -----
        The decision tree evaluates conditions in a specific order to
        handle overlapping boundary regions correctly.  This follows
        the standard formulation used by the NRCS Soil Texture
        Calculator and Gerakis & Baer (1999).

        References
        ----------
        Gerakis, A. and B. Baer, 1999. A computer program for soil
        textural classification. Soil Science Society of America
        Journal, 63:807-808.

        USDA-NRCS Soil Texture Calculator:
        https://www.nrcs.usda.gov/resources/education-and-teaching-materials/soil-texture-calculator
        """
        n = len(sand)
        result = np.full(n, "loam", dtype=object)

        for i in range(n):
            s, si, c = float(sand[i]), float(silt[i]), float(clay[i])

            # NaN guard — default to loam
            if np.isnan(s) or np.isnan(si) or np.isnan(c):
                continue

            if si + 1.5 * c < 15:
                result[i] = "sand"
            elif si + 1.5 * c >= 15 and si + 2 * c < 30:
                result[i] = "loamy_sand"
            elif (c >= 7 and c < 20 and s > 52 and si + 2 * c >= 30) or (
                c < 7 and si < 50 and si + 2 * c >= 30
            ):
                result[i] = "sandy_loam"
            elif c >= 7 and c < 27 and si >= 28 and si < 50 and s <= 52:
                result[i] = "loam"
            elif (si >= 50 and c >= 12 and c < 27) or (
                si >= 50 and si < 80 and c < 12
            ):
                result[i] = "silt_loam"
            elif si >= 80 and c < 12:
                result[i] = "silt"
            elif c >= 20 and c < 35 and si < 28 and s > 45:
                result[i] = "sandy_clay_loam"
            elif c >= 27 and c < 40 and s > 20 and s <= 45:
                result[i] = "clay_loam"
            elif c >= 27 and c < 40 and s <= 20:
                result[i] = "silty_clay_loam"
            elif c >= 35 and s > 45:
                result[i] = "sandy_clay"
            elif c >= 40 and si >= 40:
                result[i] = "silty_clay"
            elif c >= 40:
                result[i] = "clay"
            # else: stays "loam" (default)

        return result
```

**Step 2: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py -k "TestClassifyUsdaTexture" -v`
Expected: all 14 tests PASS

**Step 3: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "feat: add USDA soil texture triangle classifier"
```

---

### Task 3: Continuous percentage fallback — tests

**Files:**
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing tests for the percentage fallback in `_compute_soil_type()`**

Add these to `TestDeriveSoils` (after the existing soil tests, around line 580):

```python
    def test_soil_type_from_continuous_percentages(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Falls back to USDA texture triangle when only continuous percentages available."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "sand_pct_mean": ("nhm_id", np.array([90.0, 40.0, 20.0])),
                    "silt_pct_mean": ("nhm_id", np.array([5.0, 40.0, 20.0])),
                    "clay_pct_mean": ("nhm_id", np.array([5.0, 20.0, 60.0])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0, 80.0, 100.0])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # sand(90/5/5) -> PRMS 1, loam(40/40/20) -> PRMS 2, clay(20/20/60) -> PRMS 3
        np.testing.assert_array_equal(ds["soil_type"].values, [1, 2, 3])

    def test_soil_type_percentages_preferred_over_skip(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Percentages path used when no fractions or single texture available."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "sand_pct_mean": ("nhm_id", np.array([80.0])),
                    "silt_pct_mean": ("nhm_id", np.array([10.0])),
                    "clay_pct_mean": ("nhm_id", np.array([10.0])),
                    "aws0_100_cm_mean": ("nhm_id", np.array([5.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # loamy_sand(80/10/10) -> PRMS 1 (coarse)
        assert ds["soil_type"].values[0] == 1

    def test_soil_type_fractions_preferred_over_percentages(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Fraction columns take priority over continuous percentages."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    # Fractions say sand dominant
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.8])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.2])),
                    # Percentages say clay dominant (should be ignored)
                    "sand_pct_mean": ("nhm_id", np.array([10.0])),
                    "silt_pct_mean": ("nhm_id", np.array([10.0])),
                    "clay_pct_mean": ("nhm_id", np.array([80.0])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # Fractions win: sand=0.8 > clay=0.2 -> PRMS 1
        assert ds["soil_type"].values[0] == 1
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py -k "test_soil_type_from_continuous_percentages or test_soil_type_percentages_preferred or test_soil_type_fractions_preferred" -v`
Expected: `test_soil_type_from_continuous_percentages` and `test_soil_type_percentages_preferred` FAIL (soil_type not in ds); `test_soil_type_fractions_preferred` should PASS (fractions path already works).

**Step 3: Commit failing tests**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: add continuous percentage fallback tests (red)"
```

---

### Task 4: Continuous percentage fallback — implementation

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Add percentage fallback to `_compute_soil_type()`**

In `_compute_soil_type()`, modify the early-return check (line 1650) to also check for continuous percentages, and add a third fallback block after the single-texture-class block (after line 1704).

First, update the availability check at line 1646-1651:

```python
        # Check data availability before loading lookup table
        prefix = "soil_texture_frac_"
        fraction_vars = sorted(v for v in sir.data_vars if v.startswith(prefix))
        has_single = any(c in sir for c in ("soil_texture", "soil_texture_majority"))
        has_continuous = all(
            v in sir for v in ("sand_pct_mean", "silt_pct_mean", "clay_pct_mean")
        )

        if len(fraction_vars) < 2 and not has_single and not has_continuous:
            return None
```

Then, after the single-texture-class fallback block (after line 1704, before `return None`), add:

```python
        # Fallback: classify continuous sand/silt/clay percentages via
        # the USDA texture triangle.  This is an aggregate-then-classify
        # approach — HRU-mean percentages are classified directly, which
        # may differ from pixel-level classification.
        if has_continuous:
            sand = sir["sand_pct_mean"].values.astype(np.float64)
            silt = sir["silt_pct_mean"].values.astype(np.float64)
            clay = sir["clay_pct_mean"].values.astype(np.float64)
            texture_names = self._classify_usda_texture(sand, silt, clay)
            logger.info(
                "soil_type: classified %d HRUs from continuous sand/silt/clay "
                "percentages via USDA texture triangle (aggregate-then-classify)",
                len(texture_names),
            )
            return np.array([mapping.get(name, 2) for name in texture_names])
```

**Step 2: Update the warning message in `_derive_soils()`**

At line 1570-1574, update the warning to mention the percentage path:

```python
            logger.warning(
                "Skipping soil_type derivation (step 5): no soil texture data "
                "found in SIR. Expected soil_texture_frac_* columns, "
                "soil_texture/soil_texture_majority variable, or continuous "
                "sand_pct_mean/silt_pct_mean/clay_pct_mean percentages."
            )
```

**Step 3: Update docstrings**

Update `_derive_soils()` docstring (line 1515-1523) to add mode 3:

```
        3. **Continuous percentages** (fallback): SIR contains
           ``sand_pct_mean``, ``silt_pct_mean``, ``clay_pct_mean``
           from POLARIS.  Each HRU's mean percentages are classified
           via the USDA soil texture triangle, then mapped to PRMS
           soil_type.  This is an aggregate-then-classify approach.
```

Update `_compute_soil_type()` docstring (line 1625-1629) to mention the third path:

```
        Try fraction columns first (argmax across texture classes), then
        fall back to a single texture class variable, then to continuous
        sand/silt/clay percentages via USDA texture triangle classification.
        Unrecognized texture names default to loam (soil_type=2).
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py -k "TestDeriveSoils" -v`
Expected: all soil tests PASS (existing + new)

**Step 5: Run full test suite**

Run: `pixi run -e dev check`
Expected: all checks pass

**Step 6: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "feat: add continuous percentage fallback to _compute_soil_type()"
```

---

### Task 5: Final verification and cleanup

**Step 1: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: all hooks pass

**Step 2: Commit any formatting fixes**

If ruff/mypy produced fixes, commit them.

**Step 3: Push and create PR**

```bash
git push -u origin feat/121-soil-texture-triangle
gh pr create --title "feat: derive PRMS soil_type from continuous sand/silt/clay percentages" \
  --body "Closes #121"
```
