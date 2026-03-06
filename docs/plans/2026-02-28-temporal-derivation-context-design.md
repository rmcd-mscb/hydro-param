# Design: Wire Temporal SIR Data into DerivationContext (Issue #120)

**Date:** 2026-02-28
**Issue:** #120 — CLI doesn't pass temporal data to DerivationContext

## Problem

`pws_run_cmd` constructs `DerivationContext(temporal=None)` even though
`SIRAccessor` has temporal data (gridMET, SNODAS). Steps 10 (PET coefficients)
and 11 (transpiration timing) fall back to scalar defaults. A post-hoc
`merge_temporal_into_derived()` workaround handles forcing (step 7) separately.

## Solution

Move temporal loading before `DerivationContext` construction, pass it through
the context, and remove the redundant workaround.

## Changes

### 1. `cli.py` — `pws_run_cmd` (primary fix)

- Load temporal data from SIR before constructing `DerivationContext`
- Pass `temporal={...}` to `DerivationContext`
- Remove the post-hoc `merge_temporal_into_derived()` block
- Keep error handling for temporal loading failures
- Ensure datasets are closed in a `finally` block after derivation

### 2. `derivations/pywatershed.py` (cleanup)

- Delete `merge_temporal_into_derived()` function
- Remove the docstring note about CLI not passing temporal

### 3. Tests (update)

- Remove/update tests for `merge_temporal_into_derived()`
- Add test verifying temporal is passed to `DerivationContext`
- Verify steps 10/11 work when temporal data is available

## What stays the same

- `DerivationContext.temporal` type signature
- `SIRAccessor.load_temporal()` and `available_temporal()`
- All derivation step implementations (`_derive_forcing`, `_compute_monthly_normals`,
  `_derive_pet_coefficients`, `_derive_transp_timing`)
- Dataset lifecycle management pattern
