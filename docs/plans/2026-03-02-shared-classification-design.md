# Design: Shared USDA Texture Triangle Classification Module

**Issue:** #135 (prerequisite refactor)
**Date:** 2026-03-02

## Problem

PR #136 (issue #121) added `_classify_usda_texture()` as a private method
on `PywatershedDerivation` in `derivations/pywatershed.py`.  Issue #135
(derived categorical pipeline strategy) needs the same decision tree in
`data_access.py` for pixel-level raster classification.  Duplicating the
~50-line decision tree violates DRY and risks drift.

## Design

### New module: `classification.py`

Create `src/hydro_param/classification.py` with two exports:

```python
USDA_TEXTURE_CLASSES: dict[int, str] = {
    1: "sand", 2: "loamy_sand", 3: "sandy_loam",
    4: "sandy_clay_loam", 5: "loam", 6: "silt_loam",
    7: "sandy_clay", 8: "silt", 9: "clay_loam",
    10: "silty_clay_loam", 11: "silty_clay", 12: "clay",
}

def classify_usda_texture(
    sand: np.ndarray, silt: np.ndarray, clay: np.ndarray,
) -> np.ndarray:
    """Vectorized USDA texture triangle classification.

    Returns float64 array of class codes (1-12).
    NaN where any input is NaN.
    """
```

Pure numpy — no xarray, no model knowledge.  Includes input validation
(fraction-scale warning, sum-to-100 check) from PR #136.  Uses vectorized
boolean masking instead of the per-element for-loop.

### Consumer 1: `data_access.py` (pipeline)

```python
from hydro_param.classification import classify_usda_texture

def classify_usda_texture_raster(
    sand: xr.DataArray, silt: xr.DataArray, clay: xr.DataArray,
) -> xr.DataArray:
    """Thin wrapper: unwrap DataArrays, call core, wrap result."""
```

Registered in `CATEGORICAL_DERIVATION_FUNCTIONS`.

### Consumer 2: `derivations/pywatershed.py` (plugin)

Delete `_classify_usda_texture()` method (~90 lines).  In
`_compute_soil_type()`, the continuous-percentage fallback calls:

```python
from hydro_param.classification import USDA_TEXTURE_CLASSES, classify_usda_texture

codes = classify_usda_texture(sand, silt, clay)
code_to_name = USDA_TEXTURE_CLASSES
texture_names = np.array([
    code_to_name.get(int(c), "loam") if not np.isnan(c) else "loam"
    for c in codes
])
```

NaN codes map to `"loam"` (preserving PRMS default soil_type=2).

### Test relocation

- **Move** `TestClassifyUsdaTexture` from `test_pywatershed_derivation.py`
  to new `tests/test_classification.py`.  Rewrite to call the standalone
  function (no `derivation` fixture needed).  Assert integer class codes
  instead of string names.
- **Keep** the integration tests in `test_pywatershed_derivation.py` that
  test the continuous-percentage fallback path through `_compute_soil_type()`.
- **Add** raster wrapper tests in `tests/test_data_access.py`.

## Affected Files

| File | Change |
|------|--------|
| `src/hydro_param/classification.py` | **New** — core function + class codes dict |
| `src/hydro_param/data_access.py` | Raster wrapper + `CATEGORICAL_DERIVATION_FUNCTIONS` |
| `src/hydro_param/derivations/pywatershed.py` | Delete `_classify_usda_texture()`, import from `classification` |
| `tests/test_classification.py` | **New** — relocated + rewritten classifier tests |
| `tests/test_data_access.py` | Raster wrapper tests |
| `tests/test_pywatershed_derivation.py` | Remove `TestClassifyUsdaTexture` class |

## Scope

- This is a prerequisite refactor before the main #135 work.
- No behavioral changes — same decision tree, same boundaries, same results.
- The vectorized implementation replaces the per-element for-loop.
