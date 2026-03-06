# Design: Derive PRMS soil_type from continuous sand/silt/clay percentages

**Issue:** #121
**Date:** 2026-03-02

## Problem

Step 5 (`_derive_soils`) expects categorical soil texture class fractions
(`soil_texture_frac_sand`, `soil_texture_frac_loam`, etc.) or a single
pre-classified texture variable. POLARIS provides continuous percentages
(`sand_pct_mean`, `silt_pct_mean`, `clay_pct_mean`) which don't match
either expected format, causing soil_type derivation to be skipped.

## Design

### USDA Texture Triangle Classifier

A pure function implementing the standard USDA soil texture triangle
boundaries. Takes sand%, silt%, clay% as numpy arrays, returns an array
of USDA texture class names (one of 12 classes: sand, loamy_sand,
sandy_loam, loam, silt_loam, silt, sandy_clay_loam, clay_loam,
silty_clay_loam, sandy_clay, silty_clay, clay).

Lives as a private helper in `derivations/pywatershed.py` since the only
consumer is `_compute_soil_type()`. If other consumers emerge later, it
can be promoted to a shared module.

Standard USDA boundaries (all percentages 0-100):

| Class | Conditions |
|-------|-----------|
| clay | clay >= 40 |
| silty_clay | clay 40-60 AND silt >= 40 |
| sandy_clay | clay 35-55 AND sand >= 45 |
| silty_clay_loam | clay 27-40 AND silt >= 40 AND sand < 20 |
| clay_loam | clay 27-40 AND sand 20-45 |
| sandy_clay_loam | clay 20-35 AND sand >= 45 AND silt < 28 |
| silt | silt >= 80 AND clay < 12 |
| silt_loam | (silt >= 50 AND clay 12-27) OR (silt 50-80 AND clay < 12) |
| loam | clay 7-27 AND silt 28-50 AND sand <= 52 |
| sandy_loam | (sand >= 43 AND clay < 7 AND silt < 50) OR (sand 43-52 AND clay 7-20) |
| loamy_sand | sand 70-90 AND clay < 15 |
| sand | sand >= 85 AND clay < 10 |

Note: Boundaries must be evaluated in the correct order to handle
overlapping regions. The canonical evaluation order starts from the
finest classes (clay) and works toward coarsest (sand).

### Fallback Path in `_compute_soil_type()`

Add a third input mode after the existing two:

1. **Soil texture fractions** (preferred): `soil_texture_frac_*` columns
   → argmax → lookup to PRMS type.
2. **Single texture class**: `soil_texture` / `soil_texture_majority`
   → direct lookup to PRMS type.
3. **Continuous percentages** (new fallback): `sand_pct_mean` +
   `silt_pct_mean` + `clay_pct_mean` → USDA texture triangle classifier
   → lookup to PRMS type. Logs at INFO level to distinguish from the
   fraction-based path.

### Scientific Note

Classifying HRU-mean percentages (aggregate-then-classify) can differ
from classifying pixels then aggregating (classify-then-aggregate)
because the texture triangle is nonlinear. A future enhancement could
add a `derived_categorical` pipeline strategy for pixel-level
classification before zonal aggregation. The DRB e2e test case can
quantify this difference.

## Scope

- Add `_classify_usda_texture()` helper function
- Add continuous percentage fallback in `_compute_soil_type()`
- Update `_derive_soils()` docstring to document the third input mode
- Update warning message to mention percentage fallback
- Tests for the texture triangle classifier (boundary cases)
- Tests for the fallback integration path
- No pipeline changes, no config changes, no new modules
