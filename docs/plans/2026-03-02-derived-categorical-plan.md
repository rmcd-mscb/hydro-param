# Derived Categorical Pipeline Strategy — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `derived_categorical` pipeline pathway so multi-source raster classification (e.g., USDA soil texture from POLARIS sand/silt/clay) runs at the pixel level before zonal aggregation, producing per-class fraction columns in the SIR.

**Architecture:** Shared `classification.py` module holds the vectorized USDA texture triangle.  New `DerivedCategoricalSpec` model in the registry schema, a `CATEGORICAL_DERIVATION_FUNCTIONS` registry in `data_access.py`, and a new code path in `_process_batch()` that re-reads source GeoTIFFs from disk, classifies pixels, and runs categorical zonal stats.  The user simply adds the derived variable name to their pipeline config's `variables` list.

**Tech Stack:** Pydantic (models), numpy (classification), xarray (raster I/O), gdptools (categorical zonal stats)

**Design docs:**
- `docs/plans/2026-03-02-shared-classification-design.md` — shared classification refactor
- `docs/plans/2026-03-02-derived-categorical-design.md` — derived categorical pipeline strategy

---

### Task 1: Shared classification module — `classification.py`

**Files:**
- Create: `src/hydro_param/classification.py`
- Create: `tests/test_classification.py`

**Step 1: Write the failing tests**

Create `tests/test_classification.py` with the core classifier tests (relocated from `test_pywatershed_derivation.py::TestClassifyUsdaTexture`, rewritten to call the standalone function and assert integer codes):

```python
"""Tests for USDA soil texture triangle classification."""

import numpy as np
import pytest

from hydro_param.classification import USDA_TEXTURE_CLASSES, classify_usda_texture


class TestClassifyUsdaTexture:
    """Tests for the vectorized USDA texture triangle classifier."""

    def test_pure_sand(self) -> None:
        result = classify_usda_texture(
            np.array([90.0]), np.array([5.0]), np.array([5.0])
        )
        assert result[0] == 1  # sand

    def test_pure_clay(self) -> None:
        result = classify_usda_texture(
            np.array([20.0]), np.array([20.0]), np.array([60.0])
        )
        assert result[0] == 12  # clay

    def test_loam_center(self) -> None:
        result = classify_usda_texture(
            np.array([40.0]), np.array([40.0]), np.array([20.0])
        )
        assert result[0] == 5  # loam

    def test_silt(self) -> None:
        result = classify_usda_texture(
            np.array([5.0]), np.array([90.0]), np.array([5.0])
        )
        assert result[0] == 8  # silt

    def test_silt_loam(self) -> None:
        result = classify_usda_texture(
            np.array([20.0]), np.array([60.0]), np.array([20.0])
        )
        assert result[0] == 6  # silt_loam

    def test_sandy_loam(self) -> None:
        result = classify_usda_texture(
            np.array([65.0]), np.array([25.0]), np.array([10.0])
        )
        assert result[0] == 3  # sandy_loam

    def test_loamy_sand(self) -> None:
        result = classify_usda_texture(
            np.array([82.0]), np.array([10.0]), np.array([8.0])
        )
        assert result[0] == 2  # loamy_sand

    def test_clay_loam(self) -> None:
        result = classify_usda_texture(
            np.array([30.0]), np.array([35.0]), np.array([35.0])
        )
        assert result[0] == 9  # clay_loam

    def test_silty_clay_loam(self) -> None:
        result = classify_usda_texture(
            np.array([10.0]), np.array([55.0]), np.array([35.0])
        )
        assert result[0] == 10  # silty_clay_loam

    def test_sandy_clay_loam(self) -> None:
        result = classify_usda_texture(
            np.array([60.0]), np.array([15.0]), np.array([25.0])
        )
        assert result[0] == 4  # sandy_clay_loam

    def test_sandy_clay(self) -> None:
        result = classify_usda_texture(
            np.array([50.0]), np.array([10.0]), np.array([40.0])
        )
        assert result[0] == 7  # sandy_clay

    def test_silty_clay(self) -> None:
        result = classify_usda_texture(
            np.array([5.0]), np.array([50.0]), np.array([45.0])
        )
        assert result[0] == 11  # silty_clay

    def test_vectorized_multiple_elements(self) -> None:
        sand = np.array([90.0, 20.0, 40.0])
        silt = np.array([5.0, 20.0, 40.0])
        clay = np.array([5.0, 60.0, 20.0])
        result = classify_usda_texture(sand, silt, clay)
        assert list(result) == [1, 12, 5]  # sand, clay, loam

    def test_nan_produces_nan(self) -> None:
        result = classify_usda_texture(
            np.array([np.nan]), np.array([np.nan]), np.array([np.nan])
        )
        assert np.isnan(result[0])

    def test_partial_nan(self) -> None:
        result = classify_usda_texture(
            np.array([90.0, np.nan]),
            np.array([5.0, 40.0]),
            np.array([5.0, np.nan]),
        )
        assert result[0] == 1  # sand
        assert np.isnan(result[1])

    def test_exhaustive_no_unclassified(self) -> None:
        """Every integer (sand, silt, clay) triple that sums to 100
        must classify into a valid USDA region (no NaN, no fallthrough).

        Sweeps all 5151 valid triples at 1% resolution.
        """
        valid_codes = set(USDA_TEXTURE_CLASSES.keys())
        for sand in range(0, 101):
            for clay in range(0, 101 - sand):
                silt = 100 - sand - clay
                result = classify_usda_texture(
                    np.array([float(sand)]),
                    np.array([float(silt)]),
                    np.array([float(clay)]),
                )
                code = result[0]
                assert not np.isnan(code), (
                    f"({sand}, {silt}, {clay}) classified as NaN"
                )
                assert int(code) in valid_codes, (
                    f"({sand}, {silt}, {clay}) got invalid code {code}"
                )

    def test_class_codes_dict_complete(self) -> None:
        assert len(USDA_TEXTURE_CLASSES) == 12
        assert set(USDA_TEXTURE_CLASSES.keys()) == set(range(1, 13))

    def test_fraction_scale_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Values in 0-1 range trigger a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            classify_usda_texture(
                np.array([0.4]), np.array([0.4]), np.array([0.2])
            )
        assert "fractions" in caplog.text.lower()
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev python -m pytest tests/test_classification.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydro_param.classification'`

**Step 3: Implement `classification.py`**

Create `src/hydro_param/classification.py`:

```python
"""USDA soil texture triangle classification.

Vectorized implementation of the standard USDA soil texture triangle
decision tree.  Used by both the generic pipeline (pixel-level raster
classification for categorical zonal stats) and the pywatershed plugin
(HRU-level aggregate-then-classify fallback).

References
----------
Gerakis, A. and B. Baer, 1999. A computer program for soil textural
classification. Soil Science Society of America Journal, 63:807-808.

USDA-NRCS Soil Texture Calculator:
https://www.nrcs.usda.gov/resources/education-and-teaching-materials/soil-texture-calculator
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

USDA_TEXTURE_CLASSES: dict[int, str] = {
    1: "sand",
    2: "loamy_sand",
    3: "sandy_loam",
    4: "sandy_clay_loam",
    5: "loam",
    6: "silt_loam",
    7: "sandy_clay",
    8: "silt",
    9: "clay_loam",
    10: "silty_clay_loam",
    11: "silty_clay",
    12: "clay",
}


def classify_usda_texture(
    sand: np.ndarray,
    silt: np.ndarray,
    clay: np.ndarray,
) -> np.ndarray:
    """Classify sand/silt/clay percentages into USDA texture class codes.

    Apply the standard USDA soil texture triangle decision tree using
    vectorized boolean masking.  Each element is assigned one of 12
    integer class codes defined in ``USDA_TEXTURE_CLASSES``.

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
        Float64 array of USDA texture class codes (1–12), shape
        ``(n,)``.  Elements where any input is NaN are NaN.

    Notes
    -----
    The decision tree evaluates conditions using line-equation
    boundaries from the USDA Soil Survey Manual (Ch. 3).  Evaluation
    order matters: later assignments overwrite earlier ones where
    conditions overlap, so clay-dominated classes must be evaluated
    last.

    Inputs must satisfy ``sand + silt + clay ≈ 100``.  A warning is
    logged if inputs appear to be fractions (0–1) rather than
    percentages, or if the sum deviates from 100 by more than 5%.

    References
    ----------
    Gerakis, A. and B. Baer, 1999. A computer program for soil
    textural classification. Soil Science Society of America
    Journal, 63:807-808.
    """
    s = np.asarray(sand, dtype=np.float64)
    si = np.asarray(silt, dtype=np.float64)
    c = np.asarray(clay, dtype=np.float64)

    valid = ~(np.isnan(s) | np.isnan(si) | np.isnan(c))

    # Input validation on valid elements only
    if valid.any():
        totals = (s + si + c)[valid]
        if np.all(totals < 2.0):
            logger.warning(
                "classify_usda_texture: values appear to be fractions "
                "(0-1) rather than percentages (0-100); classification "
                "results will be incorrect. Check source data units."
            )
        far_from_100 = np.abs(totals - 100.0) > 5.0
        if np.any(far_from_100):
            logger.warning(
                "classify_usda_texture: %d/%d element(s) have "
                "sand+silt+clay summing outside 95-105%% range; "
                "texture classification may be unreliable",
                int(np.sum(far_from_100)),
                len(totals),
            )

    # Initialize: NaN everywhere, then fill valid elements
    result = np.full(len(s), np.nan)
    sv, siv, cv = s[valid], si[valid], c[valid]
    codes = np.full(len(sv), 5, dtype=np.float64)  # default loam

    # Line-equation conditions matching the USDA Soil Survey Manual
    # (Ch. 3) texture triangle boundaries.  Order matters: later
    # assignments overwrite earlier ones for overlapping regions.
    codes[siv + 1.5 * cv < 15] = 1  # sand
    codes[(siv + 1.5 * cv >= 15) & (siv + 2 * cv < 30)] = 2  # loamy_sand
    codes[
        ((cv >= 7) & (cv < 20) & (sv > 52) & (siv + 2 * cv >= 30))
        | ((cv < 7) & (siv < 50) & (siv + 2 * cv >= 30))
    ] = 3  # sandy_loam
    codes[
        (cv >= 20) & (cv < 35) & (siv < 28) & (sv > 45)
    ] = 4  # sandy_clay_loam
    codes[
        (cv >= 7) & (cv < 27) & (siv >= 28) & (siv < 50) & (sv <= 52)
    ] = 5  # loam
    codes[
        ((siv >= 50) & (cv >= 12) & (cv < 27))
        | ((siv >= 50) & (siv < 80) & (cv < 12))
    ] = 6  # silt_loam
    codes[(cv >= 35) & (sv > 45)] = 7  # sandy_clay
    codes[(siv >= 80) & (cv < 12)] = 8  # silt
    codes[(cv >= 27) & (cv < 40) & (sv > 20) & (sv <= 45)] = 9  # clay_loam
    codes[(cv >= 27) & (cv < 40) & (sv <= 20)] = 10  # silty_clay_loam
    codes[(cv >= 40) & (siv >= 40)] = 11  # silty_clay
    codes[cv >= 40] = 12  # clay (must be last — broadest clay condition)

    result[valid] = codes
    return result
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_classification.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/classification.py tests/test_classification.py
git commit -m "feat: add shared classification module with vectorized USDA texture triangle"
```

---

### Task 2: Refactor pywatershed derivation to use shared classifier

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Delete `_classify_usda_texture()` method and update consumer**

In `src/hydro_param/derivations/pywatershed.py`:

1. Add import at top:
   ```python
   from hydro_param.classification import USDA_TEXTURE_CLASSES, classify_usda_texture
   ```

2. Delete the `_classify_usda_texture()` method (lines 1631-1773, ~140 lines).

3. Update the call site in `_compute_soil_type()` (around line 1878):
   ```python
           # Old:
           # texture_names = self._classify_usda_texture(sand, silt, clay)

           # New:
           codes = classify_usda_texture(sand, silt, clay)
           texture_names = np.array([
               USDA_TEXTURE_CLASSES.get(int(c), "loam")
               if not np.isnan(c) else "loam"
               for c in codes
           ])
   ```

**Step 2: Remove `TestClassifyUsdaTexture` class from derivation tests**

In `tests/test_pywatershed_derivation.py`, delete the entire
`TestClassifyUsdaTexture` class (lines 680-808).  The integration tests
for `_compute_soil_type()` (the continuous-percentage fallback path)
remain.

**Step 3: Run tests to verify nothing broke**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py tests/test_classification.py -v`
Expected: All pass — classifier tests now in `test_classification.py`, integration tests still pass in `test_pywatershed_derivation.py`

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "refactor: use shared classify_usda_texture() in pywatershed derivation"
```

---

### Task 3: DerivedCategoricalSpec model

**Files:**
- Modify: `src/hydro_param/dataset_registry.py:80-112`
- Test: `tests/test_registry.py`

**Step 1: Write the failing test**

Add a test class in `tests/test_registry.py`:

```python
class TestDerivedCategoricalSpec:
    """Tests for DerivedCategoricalSpec model."""

    def test_basic_creation(self) -> None:
        from hydro_param.dataset_registry import DerivedCategoricalSpec

        spec = DerivedCategoricalSpec(
            name="soil_texture",
            sources=["sand", "silt", "clay"],
            method="usda_texture_triangle",
            units="class",
            long_name="USDA soil texture classification",
        )
        assert spec.name == "soil_texture"
        assert spec.sources == ["sand", "silt", "clay"]
        assert spec.method == "usda_texture_triangle"

    def test_sources_must_have_at_least_two(self) -> None:
        import pytest
        from hydro_param.dataset_registry import DerivedCategoricalSpec

        with pytest.raises(ValueError):
            DerivedCategoricalSpec(
                name="bad",
                sources=["single"],
                method="test",
            )
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_registry.py -k "TestDerivedCategoricalSpec" -v`
Expected: FAIL — `ImportError: cannot import name 'DerivedCategoricalSpec'`

**Step 3: Write minimal implementation**

Add after `DerivedVariableSpec` (line 112) in `src/hydro_param/dataset_registry.py`:

```python
class DerivedCategoricalSpec(BaseModel):
    """Describe a categorical variable derived from multiple source variables.

    Multi-source categorical derivations classify pixels by combining
    two or more source bands (e.g., USDA texture triangle from
    sand/silt/clay percentages).  The result is a single-band
    categorical raster processed with categorical zonal statistics to
    produce per-class fraction columns.

    Unlike ``DerivedVariableSpec`` (single source, continuous output),
    this always produces categorical output with per-class fractions.

    Attributes
    ----------
    name : str
        Logical name for the derived variable (e.g.,
        ``"soil_texture"``).
    sources : list[str]
        Names of the source ``VariableSpec`` entries this is derived
        from (e.g., ``["sand", "silt", "clay"]``).  Must contain at
        least 2 entries.
    method : str
        Classification method key used to look up the derivation
        function via
        ``hydro_param.data_access.CATEGORICAL_DERIVATION_FUNCTIONS``.
    units : str
        Units of the derived variable (typically ``"class"``).
    long_name : str
        Human-readable description for metadata.
    """

    name: str
    sources: list[str]
    method: str
    units: str = ""
    long_name: str = ""

    @field_validator("sources")
    @classmethod
    def _check_min_sources(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("DerivedCategoricalSpec requires at least 2 sources")
        return v
```

Note: `field_validator` is already imported from pydantic in this file — verify before adding.

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev python -m pytest tests/test_registry.py -k "TestDerivedCategoricalSpec" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/dataset_registry.py tests/test_registry.py
git commit -m "feat: add DerivedCategoricalSpec model for multi-source classification"
```

---

### Task 4: Registry parsing — `derived_categorical_variables` field on DatasetEntry

**Files:**
- Modify: `src/hydro_param/dataset_registry.py:344`
- Test: `tests/test_registry.py`

**Step 1: Write the failing test**

```python
class TestDerivedCategoricalParsing:
    """Tests for parsing derived_categorical_variables from YAML."""

    def test_dataset_entry_has_derived_categorical(self) -> None:
        from hydro_param.dataset_registry import (
            DatasetEntry,
            DerivedCategoricalSpec,
        )

        entry = DatasetEntry(
            strategy="local_tiff",
            variables=[],
            derived_categorical_variables=[
                DerivedCategoricalSpec(
                    name="soil_texture",
                    sources=["sand", "silt", "clay"],
                    method="usda_texture_triangle",
                )
            ],
        )
        assert len(entry.derived_categorical_variables) == 1
        assert entry.derived_categorical_variables[0].name == "soil_texture"

    def test_defaults_to_empty_list(self) -> None:
        from hydro_param.dataset_registry import DatasetEntry

        entry = DatasetEntry(strategy="local_tiff", variables=[])
        assert entry.derived_categorical_variables == []
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_registry.py -k "TestDerivedCategoricalParsing" -v`
Expected: FAIL — `unexpected keyword argument 'derived_categorical_variables'`

**Step 3: Add field to DatasetEntry**

At line 344 in `dataset_registry.py`, after `derived_variables`, add:

```python
    derived_categorical_variables: list[DerivedCategoricalSpec] = []
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev python -m pytest tests/test_registry.py -k "TestDerivedCategoricalParsing" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/dataset_registry.py tests/test_registry.py
git commit -m "feat: add derived_categorical_variables field to DatasetEntry"
```

---

### Task 5: Variable resolution — resolve `DerivedCategoricalSpec` by name

**Files:**
- Modify: `src/hydro_param/dataset_registry.py:431-469`
- Test: `tests/test_registry.py`

**Step 1: Write the failing test**

```python
    def test_resolve_derived_categorical_variable(self) -> None:
        from hydro_param.dataset_registry import (
            DatasetEntry,
            DatasetRegistry,
            DerivedCategoricalSpec,
            VariableSpec,
        )

        entry = DatasetEntry(
            strategy="local_tiff",
            variables=[
                VariableSpec(name="sand"),
                VariableSpec(name="silt"),
                VariableSpec(name="clay"),
            ],
            derived_categorical_variables=[
                DerivedCategoricalSpec(
                    name="soil_texture",
                    sources=["sand", "silt", "clay"],
                    method="usda_texture_triangle",
                )
            ],
        )
        registry = DatasetRegistry(datasets={"test": entry})
        spec = registry.resolve_variable("test", "soil_texture")
        assert isinstance(spec, DerivedCategoricalSpec)
        assert spec.sources == ["sand", "silt", "clay"]
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_registry.py -k "test_resolve_derived_categorical_variable" -v`
Expected: FAIL — `KeyError: Variable 'soil_texture' not found`

**Step 3: Update `resolve_variable()` (line 458-469)**

Update the return type and add a search loop for `derived_categorical_variables`:

```python
    def resolve_variable(
        self, dataset_name: str, variable_name: str
    ) -> VariableSpec | DerivedVariableSpec | DerivedCategoricalSpec:
```

After the `derived_variables` loop (line 464), add:

```python
        for dcv in entry.derived_categorical_variables:
            if dcv.name == variable_name:
                return dcv
```

Update the `available` list (line 465) to include derived categorical names:

```python
        available = (
            [v.name for v in entry.variables]
            + [dv.name for dv in entry.derived_variables]
            + [dcv.name for dcv in entry.derived_categorical_variables]
        )
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev python -m pytest tests/test_registry.py -k "test_resolve_derived_categorical_variable" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/dataset_registry.py tests/test_registry.py
git commit -m "feat: resolve DerivedCategoricalSpec in registry.resolve_variable()"
```

---

### Task 6: Auto-include source variables in stage 2 resolution

**Files:**
- Modify: `src/hydro_param/pipeline.py:476-479`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Find an appropriate location in `tests/test_pipeline.py` (near existing stage 2 tests):

```python
def test_stage2_auto_includes_derived_categorical_sources() -> None:
    """When user requests a DerivedCategoricalSpec, its sources are auto-included."""
    from hydro_param.config import DatasetRequest, PipelineConfig
    from hydro_param.dataset_registry import (
        DatasetEntry,
        DatasetRegistry,
        DerivedCategoricalSpec,
        VariableSpec,
    )
    from hydro_param.pipeline import stage2_resolve_datasets

    entry = DatasetEntry(
        strategy="local_tiff",
        variables=[
            VariableSpec(name="sand", source_override="http://example.com/sand.vrt"),
            VariableSpec(name="silt", source_override="http://example.com/silt.vrt"),
            VariableSpec(name="clay", source_override="http://example.com/clay.vrt"),
            VariableSpec(name="ksat", source_override="http://example.com/ksat.vrt"),
        ],
        derived_categorical_variables=[
            DerivedCategoricalSpec(
                name="soil_texture",
                sources=["sand", "silt", "clay"],
                method="usda_texture_triangle",
            )
        ],
    )
    registry = DatasetRegistry(datasets={"test_ds": entry})

    # User requests only soil_texture and ksat — sand/silt/clay should be auto-included
    config = PipelineConfig(
        target_fabric={"path": "/tmp/test.gpkg", "id_field": "nhm_id"},
        datasets=[DatasetRequest(name="test_ds", variables=["soil_texture", "ksat"], statistics=["mean"])],
        output={"path": "/tmp/out"},
    )

    resolved = stage2_resolve_datasets(config, registry)
    _, _, var_specs = resolved[0]
    var_names = [v.name for v in var_specs]

    # Sources auto-included before the derived categorical spec
    assert "sand" in var_names
    assert "silt" in var_names
    assert "clay" in var_names
    assert "soil_texture" in var_names
    assert "ksat" in var_names
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_pipeline.py -k "test_stage2_auto_includes_derived_categorical_sources" -v`
Expected: FAIL — `KeyError: Variable 'soil_texture' not found` or sources missing from var_names

**Step 3: Update stage 2 resolution (pipeline.py line 478)**

Before the `var_specs` line (478), add auto-inclusion logic:

```python
        # Auto-include source variables needed by derived categorical specs
        requested = set(ds_req.variables)
        extra_sources: list[str] = []
        for vname in ds_req.variables:
            spec = registry.resolve_variable(ds_req.name, vname)
            if isinstance(spec, DerivedCategoricalSpec):
                for src in spec.sources:
                    if src not in requested and src not in extra_sources:
                        extra_sources.append(src)
        if extra_sources:
            logger.info(
                "  Auto-including source variables for derived categorical: %s",
                extra_sources,
            )

        all_var_names = extra_sources + list(ds_req.variables)
        var_specs = [registry.resolve_variable(ds_req.name, v) for v in all_var_names]
```

Replace the original line 478 (`var_specs = [registry.resolve_variable(ds_req.name, v) for v in ds_req.variables]`) with this block.

Also add the import at the top of `pipeline.py` (around line 64, where `DerivedVariableSpec` is imported):

```python
from hydro_param.dataset_registry import DerivedCategoricalSpec
```

And update the type signature of `stage2_resolve_datasets` return type (line 379):

```python
) -> list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec | DerivedCategoricalSpec]]]:
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev python -m pytest tests/test_pipeline.py -k "test_stage2_auto_includes_derived_categorical_sources" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "feat: auto-include source variables for derived categorical specs in stage 2"
```

---

### Task 7: Raster classification wrapper in `data_access.py`

**Files:**
- Modify: `src/hydro_param/data_access.py`
- Test: `tests/test_data_access.py`

**Step 1: Write the failing test**

```python
class TestClassifyUsdaTextureRaster:
    """Tests for the xarray raster wrapper around USDA texture classification."""

    def test_basic_classification(self) -> None:
        import xarray as xr
        from hydro_param.data_access import classify_usda_texture_raster

        sand = xr.DataArray([[90.0, 40.0], [20.0, 5.0]], dims=["y", "x"])
        silt = xr.DataArray([[5.0, 40.0], [20.0, 90.0]], dims=["y", "x"])
        clay = xr.DataArray([[5.0, 20.0], [60.0, 5.0]], dims=["y", "x"])
        result = classify_usda_texture_raster(sand, silt, clay)
        assert result.values[0, 0] == 1   # sand
        assert result.values[0, 1] == 5   # loam
        assert result.values[1, 0] == 12  # clay
        assert result.values[1, 1] == 8   # silt

    def test_nan_produces_nan(self) -> None:
        import xarray as xr
        from hydro_param.data_access import classify_usda_texture_raster

        sand = xr.DataArray([[np.nan]], dims=["y", "x"])
        silt = xr.DataArray([[np.nan]], dims=["y", "x"])
        clay = xr.DataArray([[np.nan]], dims=["y", "x"])
        result = classify_usda_texture_raster(sand, silt, clay)
        assert np.isnan(result.values[0, 0])

    def test_output_preserves_coordinates(self) -> None:
        import xarray as xr
        from hydro_param.data_access import classify_usda_texture_raster

        sand = xr.DataArray([[90.0]], dims=["y", "x"], coords={"y": [1.0], "x": [2.0]})
        silt = xr.DataArray([[5.0]], dims=["y", "x"], coords={"y": [1.0], "x": [2.0]})
        clay = xr.DataArray([[5.0]], dims=["y", "x"], coords={"y": [1.0], "x": [2.0]})
        result = classify_usda_texture_raster(sand, silt, clay)
        assert list(result.coords["y"].values) == [1.0]
        assert list(result.coords["x"].values) == [2.0]
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_data_access.py -k "TestClassifyUsdaTextureRaster" -v`
Expected: FAIL — `ImportError: cannot import name 'classify_usda_texture_raster'`

**Step 3: Implement the wrapper**

Add to `data_access.py` before `DERIVATION_FUNCTIONS` (line 269):

```python
from hydro_param.classification import classify_usda_texture


def classify_usda_texture_raster(
    sand: xr.DataArray,
    silt: xr.DataArray,
    clay: xr.DataArray,
) -> xr.DataArray:
    """Classify sand/silt/clay percentage rasters into USDA texture classes.

    Thin wrapper around ``classify_usda_texture()`` that handles
    xarray DataArray I/O.  Returns a single-band integer raster with
    class codes (1–12) suitable for categorical zonal statistics.

    Parameters
    ----------
    sand : xr.DataArray
        Sand content as percentage (0–100), 2-D raster.
    silt : xr.DataArray
        Silt content as percentage (0–100), 2-D raster.
    clay : xr.DataArray
        Clay content as percentage (0–100), 2-D raster.

    Returns
    -------
    xr.DataArray
        Integer raster with USDA texture class codes (1–12).
        NaN inputs produce NaN output.

    See Also
    --------
    hydro_param.classification.classify_usda_texture : Core classifier.
    hydro_param.classification.USDA_TEXTURE_CLASSES : Code-to-name mapping.
    """
    codes = classify_usda_texture(
        sand.values.astype(np.float64).ravel(),
        silt.values.astype(np.float64).ravel(),
        clay.values.astype(np.float64).ravel(),
    )
    out = sand.copy(data=codes.reshape(sand.shape))
    out.name = "soil_texture"
    out.attrs = {"units": "class", "long_name": "USDA soil texture classification"}
    return out


CATEGORICAL_DERIVATION_FUNCTIONS: dict[str, Callable[..., xr.DataArray]] = {
    "usda_texture_triangle": classify_usda_texture_raster,
}
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev python -m pytest tests/test_data_access.py -k "TestClassifyUsdaTextureRaster" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/data_access.py tests/test_data_access.py
git commit -m "feat: add classify_usda_texture_raster() wrapper and CATEGORICAL_DERIVATION_FUNCTIONS"
```

---

### Task 8: Pipeline `_process_batch()` — derived categorical processing

**Files:**
- Modify: `src/hydro_param/pipeline.py:522-735`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

This is an integration test that verifies the full flow.  Find the existing
`_process_batch` tests in `tests/test_pipeline.py` and add:

```python
def test_process_batch_derived_categorical(tmp_path: Path) -> None:
    """DerivedCategoricalSpec in var_specs triggers multi-source classification."""
    import geopandas as gpd
    import rioxarray  # noqa: F401
    import xarray as xr
    from shapely.geometry import box

    from hydro_param.config import DatasetRequest
    from hydro_param.data_access import save_to_geotiff
    from hydro_param.dataset_registry import (
        DatasetEntry,
        DerivedCategoricalSpec,
        VariableSpec,
    )

    pytest.importorskip("gdptools")

    # Create a simple 4x4 raster fabric
    fabric = gpd.GeoDataFrame(
        {"nhm_id": [1, 2]},
        geometry=[box(0, 0, 0.5, 0.5), box(0.5, 0, 1, 0.5)],
        crs="EPSG:4326",
    )

    # Create test rasters — uniform values so classification is deterministic
    coords = {"y": [0.375, 0.125], "x": [0.125, 0.375, 0.625, 0.875]}
    sand_da = xr.DataArray(
        np.full((2, 4), 90.0), dims=["y", "x"], coords=coords
    ).rio.write_crs("EPSG:4326")
    silt_da = xr.DataArray(
        np.full((2, 4), 5.0), dims=["y", "x"], coords=coords
    ).rio.write_crs("EPSG:4326")
    clay_da = xr.DataArray(
        np.full((2, 4), 5.0), dims=["y", "x"], coords=coords
    ).rio.write_crs("EPSG:4326")

    # Save source GeoTIFFs (simulating what the VariableSpec processing would produce)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    save_to_geotiff(sand_da, work_dir / "sand.tif")
    save_to_geotiff(silt_da, work_dir / "silt.tif")
    save_to_geotiff(clay_da, work_dir / "clay.tif")

    # The DerivedCategoricalSpec
    dc_spec = DerivedCategoricalSpec(
        name="soil_texture",
        sources=["sand", "silt", "clay"],
        method="usda_texture_triangle",
        units="class",
    )

    entry = DatasetEntry(
        strategy="local_tiff",
        variables=[
            VariableSpec(name="sand", source_override="dummy"),
            VariableSpec(name="silt", source_override="dummy"),
            VariableSpec(name="clay", source_override="dummy"),
        ],
        derived_categorical_variables=[dc_spec],
    )

    # We test the derived categorical processing in isolation —
    # source GeoTIFFs already exist on disk from step 1.
    # This test verifies the classification + categorical zonal stats path.
    from hydro_param.data_access import CATEGORICAL_DERIVATION_FUNCTIONS, classify_usda_texture_raster

    # Classify
    classified = classify_usda_texture_raster(sand_da, silt_da, clay_da)
    save_to_geotiff(classified, work_dir / "soil_texture.tif")

    # Verify classified raster has expected class code
    assert classified.values[0, 0] == 1  # sand
```

Note: The full `_process_batch` integration test depends on the pipeline
wiring in the next step.  This test validates the classify + save path.
A full end-to-end test should be added after wiring is complete.

**Step 2: Run test to verify it fails or passes**

Run: `pixi run -e dev python -m pytest tests/test_pipeline.py -k "test_process_batch_derived_categorical" -v`
Expected: Should PASS (this tests the classification function, not the pipeline wiring)

**Step 3: Add derived categorical processing to `_process_batch()`**

In `_process_batch()` (line 653), the main loop iterates `var_specs`.
The design calls for processing `DerivedCategoricalSpec` entries **last**
by splitting the loop.  Modify the function:

After the existing for-loop (lines 653-733), before `return results` (line 735), add:

```python
    # Process derived categorical specs last — re-read source GeoTIFFs
    # from disk rather than holding all sources in memory.
    from hydro_param.data_access import CATEGORICAL_DERIVATION_FUNCTIONS

    dc_specs = [v for v in var_specs if isinstance(v, DerivedCategoricalSpec)]
    for dc_spec in dc_specs:
        derive_fn = CATEGORICAL_DERIVATION_FUNCTIONS.get(dc_spec.method)
        if derive_fn is None:
            raise ValueError(
                f"No categorical derivation function for method '{dc_spec.method}'"
            )

        # Re-read source GeoTIFFs from disk
        source_das = []
        missing = []
        for src_name in dc_spec.sources:
            src_tiff = work_dir / f"{src_name}.tif"
            if not src_tiff.exists():
                missing.append(src_name)
                continue
            source_das.append(
                cast("xr.DataArray", rioxarray.open_rasterio(src_tiff).squeeze("band", drop=True))
            )

        if missing:
            logger.warning(
                "Skipping derived categorical '%s': missing source GeoTIFFs %s",
                dc_spec.name,
                missing,
            )
            continue

        # Classify pixels
        classified_da = derive_fn(*source_das)

        # Free source arrays immediately
        del source_das
        gc.collect()

        # Save classified raster and run categorical zonal stats
        classified_tiff = work_dir / f"{dc_spec.name}.tif"
        save_to_geotiff(classified_da, classified_tiff)
        del classified_da

        df = processor.process(
            fabric=batch_fabric,
            tiff_path=classified_tiff,
            variable_name=dc_spec.name,
            id_field=config.target_fabric.id_field,
            engine=config.processing.engine,
            statistics=ds_req.statistics,
            categorical=True,
            source_crs=entry.crs,
            x_coord=entry.x_coord,
            y_coord=entry.y_coord,
        )
        results[dc_spec.name] = df

        classified_tiff.unlink(missing_ok=True)
        gc.collect()
```

Also, in the main for-loop (line 653), skip `DerivedCategoricalSpec` entries
so they aren't processed twice:

```python
    for i, var_spec in enumerate(var_specs):
        if isinstance(var_spec, DerivedCategoricalSpec):
            continue  # Processed after all source variables
```

And update the existing GeoTIFF cleanup (line 713) to NOT delete source
GeoTIFFs that are needed by derived categorical specs:

```python
        # Clean up GeoTIFF after zonal stats — keep if needed by derived categorical
        needed_by_dc = any(
            isinstance(dc, DerivedCategoricalSpec) and var_spec.name in dc.sources
            for dc in var_specs
        )
        if not needed_by_dc:
            tiff_path.unlink(missing_ok=True)
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev python -m pytest tests/test_pipeline.py -v`
Expected: all pass

**Step 5: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "feat: process DerivedCategoricalSpec in _process_batch()"
```

---

### Task 9: SIR schema support for derived categorical variables

**Files:**
- Modify: `src/hydro_param/sir.py:220-315`
- Test: `tests/test_sir.py`

**Step 1: Write the failing test**

```python
class TestBuildSIRSchemaDerivedCategorical:
    """Tests for derived categorical variables in SIR schema."""

    def test_derived_categorical_produces_frac_schema(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import (
            DatasetEntry,
            DerivedCategoricalSpec,
            VariableSpec,
        )
        from hydro_param.sir import build_sir_schema

        entry = DatasetEntry(
            strategy="local_tiff",
            variables=[VariableSpec(name="sand"), VariableSpec(name="silt"), VariableSpec(name="clay")],
            derived_categorical_variables=[
                DerivedCategoricalSpec(
                    name="soil_texture",
                    sources=["sand", "silt", "clay"],
                    method="usda_texture_triangle",
                    units="class",
                    long_name="USDA soil texture classification",
                )
            ],
        )
        ds_req = DatasetRequest(name="polaris_30m", variables=["sand", "silt", "clay", "soil_texture"], statistics=["mean"])
        var_specs = [
            VariableSpec(name="sand", units="%"),
            VariableSpec(name="silt", units="%"),
            VariableSpec(name="clay", units="%"),
            DerivedCategoricalSpec(name="soil_texture", sources=["sand", "silt", "clay"], method="usda_texture_triangle", units="class"),
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        categorical_entries = [s for s in schema if s.categorical]
        assert len(categorical_entries) == 1
        assert categorical_entries[0].canonical_name == "soil_texture_frac"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_sir.py -k "TestBuildSIRSchemaDerivedCategorical" -v`
Expected: FAIL — `DerivedCategoricalSpec` not handled in `build_sir_schema`

**Step 3: Update `build_sir_schema()` (sir.py line 266-314)**

Add handling for `DerivedCategoricalSpec` in the var_spec loop.  Import the class and add a branch:

```python
from hydro_param.dataset_registry import DerivedCategoricalSpec
```

In the loop (line 266), before the existing categorical check (line 271-293), add:

```python
            if isinstance(var_spec, DerivedCategoricalSpec):
                # Derived categorical: always produces fraction columns
                for year in years:
                    cname = canonical_name(var_spec.name, "", "frac")
                    if year is not None:
                        cname = f"{cname}_{year}"
                    schema.append(
                        SIRVariableSchema(
                            canonical_name=cname,
                            source_name=var_spec.name,
                            source_units=var_spec.units,
                            canonical_units="",
                            long_name=var_spec.long_name or var_spec.name,
                            categorical=True,
                            valid_range=(0.0, 1.0),
                            conversion=None,
                            temporal=is_temporal,
                            dataset_name=ds_req.name,
                        )
                    )
                continue
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev python -m pytest tests/test_sir.py -k "TestBuildSIRSchemaDerivedCategorical" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: handle DerivedCategoricalSpec in build_sir_schema()"
```

---

### Task 10: Registry entry — add soil_texture to POLARIS

**Files:**
- Modify: `src/hydro_param/data/datasets/soils.yml:90-174`

**Step 1: Add derived_categorical_variables to polaris_30m**

After the `variables` list (line 174), add:

```yaml
    derived_categorical_variables:
      - name: soil_texture
        sources: [sand, silt, clay]
        method: usda_texture_triangle
        units: "class"
        long_name: "USDA soil texture classification"
```

**Step 2: Verify the registry loads correctly**

Run: `pixi run -e dev python -c "from hydro_param.dataset_registry import load_default_registry; r = load_default_registry(); e = r.get('polaris_30m'); print(len(e.derived_categorical_variables), e.derived_categorical_variables[0].name)"`
Expected: `1 soil_texture`

**Step 3: Run full test suite**

Run: `pixi run -e dev check`
Expected: all checks pass

**Step 4: Commit**

```bash
git add src/hydro_param/data/datasets/soils.yml
git commit -m "feat: add soil_texture derived categorical to POLARIS registry"
```

---

### Task 11: Update example pipeline config

**Files:**
- Modify: `configs/examples/drb_2yr_pipeline.yml:29-31`

**Step 1: Add soil_texture to POLARIS variables**

Update line 30:

```yaml
  - name: polaris_30m
    variables: [sand, silt, clay, theta_s, ksat, soil_texture]
    statistics: [mean]
```

**Step 2: Commit**

```bash
git add configs/examples/drb_2yr_pipeline.yml
git commit -m "docs: add soil_texture to example pipeline config"
```

---

### Task 12: Final verification and cleanup

**Step 1: Run full checks**

Run: `pixi run -e dev check`
Expected: all checks pass (lint, format, typecheck, tests)

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: all hooks pass

**Step 3: Fix any issues**

Commit any formatting or type annotation fixes.

**Step 4: Push and create PR**

```bash
git push -u origin feat/135-derived-categorical
gh pr create --title "feat: derived_categorical pipeline strategy for pixel-level classification" \
  --body "Closes #135"
```
