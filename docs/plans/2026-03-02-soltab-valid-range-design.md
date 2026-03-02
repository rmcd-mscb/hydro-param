# Design: Fix soltab valid_range upper bound

**Issue:** #122
**Date:** 2026-03-02

## Problem

The formatter validation reports ~32K values of `soltab_potsw` and ~35K of
`soltab_horad_potsw` exceeding `valid_range` maximum of 1000.0 Langleys/day
in the DRB end-to-end run (765 HRUs x 366 days).

## Root cause

The `valid_range: [0.0, 1000.0]` in `parameter_metadata.yml` is too
conservative.  The Swift (1976) algorithm computes potential clear-sky
direct-beam radiation, which can exceed 1000 Langleys/day on sloped and
horizontal surfaces at certain latitudes and times of year.  The test suite
already confirms this: lat=40.5N, slope=0.15, south-facing produces ~1010
Langleys at summer solstice.

## Algorithm verification

Line-by-line comparison of `solar.py` against pywatershed's
`PRMSSolarGeometry` confirms **zero algorithmic differences** in the core
computation (equivalent slope latitude, hour-angle clipping, radiation
integral, wrap-around corrections, flat-surface override, negative clamping).
The solar constant, declination formula, and all intermediate calculations are
identical.  The original PRMS Fortran does not define valid_range bounds for
these computed output variables.

The only non-functional difference is `DNEARZERO` (1e-12 vs 2.22e-16 machine
epsilon), which is irrelevant for all realistic inputs.

## Fix

1. Update `parameter_metadata.yml`: change `valid_range` from `[0.0, 1000.0]`
   to `[0.0, 2000.0]` for both `soltab_potsw` and `soltab_horad_potsw`.
2. Update any tests that assert on the old range threshold.
