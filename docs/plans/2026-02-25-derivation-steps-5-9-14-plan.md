# Derivation Steps 5, 9, 14 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three pywatershed derivation steps — soltab (9), soils (5), and calibration seeds (14) — as three separate PRs.

**Architecture:** Each step is a private method on `PywatershedDerivation` following the existing pattern: `_step_name(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset`. Step 9 also introduces a new `solar.py` module for the Swift (1976) algorithm. Step 14 introduces a YAML-driven seed config with a safe formula dispatch.

**Tech Stack:** numpy (vectorized computation), xarray (DataArrays), PyYAML (lookup tables), pytest (TDD)

**Design doc:** `docs/plans/2026-02-25-derivation-steps-5-9-14-design.md`

---

## PR A: Step 9 — Solar Radiation Tables (soltab)

### Task 1: Create GitHub issue and feature branch

**Step 1: Create issue and branch**

```bash
# Use the /issue-branch skill or manually:
gh issue create --title "feat: implement step 9 solar radiation tables (soltab)" \
  --body "Implement Swift (1976) solar radiation table computation for pywatershed derivation pipeline. Part of #83.

## What
- New \`solar.py\` module with \`compute_soltab()\` function
- New \`_derive_soltab()\` step in \`PywatershedDerivation\`
- Produces \`soltab_potsw\`, \`soltab_horad_potsw\`, \`soltab_sunhrs\` (nhru × 366)

## References
- Swift, L.W. Jr. (1976). Algorithm for solar radiation on mountain slopes. WRR 12(1):108.
- pywatershed source: \`EC-USGS/pywatershed\`, \`prms_solar_geometry.py\`
"
# Then create branch: feat/<issue-number>-soltab
```

### Task 2: Write solar constants module

**Files:**
- Create: `src/hydro_param/solar.py`

**Step 1: Write failing test for solar constants**

**Test file:** `tests/test_solar.py`

```python
"""Tests for Swift (1976) solar radiation table computation."""

from __future__ import annotations

import numpy as np
import pytest

from hydro_param.solar import NDOY, compute_soltab, solar_declination, r1


class TestSolarConstants:
    """Verify solar constants are correct shape and range."""

    def test_ndoy_is_366(self) -> None:
        assert NDOY == 366

    def test_solar_declination_shape(self) -> None:
        assert solar_declination.shape == (366,)

    def test_solar_declination_range(self) -> None:
        # Solar declination ranges roughly -0.41 to 0.41 radians
        assert np.all(solar_declination >= -0.45)
        assert np.all(solar_declination <= 0.45)

    def test_r1_shape(self) -> None:
        assert r1.shape == (366,)

    def test_r1_positive(self) -> None:
        assert np.all(r1 > 0)
```

**Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_solar.py::TestSolarConstants -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hydro_param.solar'`

**Step 3: Write solar constants in `src/hydro_param/solar.py`**

```python
"""Swift (1976) solar radiation table computation.

Computes potential clear-sky solar radiation on sloped and horizontal
surfaces for every day of the year.  Ported from pywatershed's
``PRMSSolarGeometry`` (EC-USGS/pywatershed), which implements PRMS 5.2.1.

References
----------
Swift, L.W. Jr. (1976). Algorithm for solar radiation on mountain slopes.
    Water Resources Research, 12(1):108.
Lee, R. (1963). Evaluation of solar beam irradiation as a climatic parameter
    of mountain watersheds. Colorado State University Hydrology Paper No. 2.
Markstrom et al. (2015). PRMS-IV. USGS TM 6-B7.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

# ---------------------------------------------------------------------------
# Constants (ported from pywatershed/atmosphere/solar_constants.py)
# ---------------------------------------------------------------------------

NDOY: int = 366
"""Number of days in the solar table (includes leap day)."""

_DNEARZERO: float = 1.0e-12
_N_DAYS_PER_YEAR_FLT: float = 365.242
_ECCENTRICITY: float = 0.01671

_PI: float = math.pi
_TWO_PI: float = 2.0 * _PI
_PI_12: float = 12.0 / _PI
_RAD_DAY: float = _TWO_PI / _N_DAYS_PER_YEAR_FLT

_julian_days = np.arange(NDOY) + 1
_obliquity = 1.0 - (_ECCENTRICITY * np.cos((_julian_days - 3) * _RAD_DAY))

_yy = (_julian_days - 1) * _RAD_DAY
_yy2 = _yy * 2
_yy3 = _yy * 3

solar_declination: np.ndarray = (
    0.006918
    - 0.399912 * np.cos(_yy)
    + 0.070257 * np.sin(_yy)
    - 0.006758 * np.cos(_yy2)
    + 0.000907 * np.sin(_yy2)
    - 0.002697 * np.cos(_yy3)
    + 0.00148 * np.sin(_yy3)
)
"""Solar declination for each day of year (radians), shape (366,)."""

# Solar constant: 2.0 cal/cm2/min (Drummond et al. 1968)
_R0: float = 2.0
r1: np.ndarray = (60.0 * _R0) / (_obliquity**2)
"""Solar constant adjusted for orbital eccentricity, shape (366,)."""
```

**Step 4: Run test to verify it passes**

```bash
pixi run -e dev pytest tests/test_solar.py::TestSolarConstants -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/solar.py tests/test_solar.py
git commit -m "feat: add solar constants for Swift (1976) soltab computation"
```

### Task 3: Write compute_soltab function

**Files:**
- Modify: `src/hydro_param/solar.py`
- Modify: `tests/test_solar.py`

**Step 1: Write failing tests for compute_soltab**

Add to `tests/test_solar.py`:

```python
class TestComputeSoltab:
    """Tests for compute_soltab() function."""

    def test_output_shapes(self) -> None:
        """Output arrays have shape (366, nhru)."""
        slopes = np.array([0.1, 0.3, 0.5])
        aspects = np.array([180.0, 90.0, 270.0])
        lats = np.array([42.0, 35.0, 48.0])

        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)

        assert potsw.shape == (366, 3)
        assert horad.shape == (366, 3)
        assert sunhrs.shape == (366, 3)

    def test_all_non_negative(self) -> None:
        """All output values are non-negative."""
        slopes = np.array([0.1, 0.5])
        aspects = np.array([180.0, 0.0])
        lats = np.array([42.0, 42.0])

        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)

        assert np.all(potsw >= 0)
        assert np.all(horad >= 0)
        assert np.all(sunhrs >= 0)

    def test_flat_surface_equals_horizontal(self) -> None:
        """For slope=0, soltab_potsw should equal soltab_horad_potsw."""
        slopes = np.array([0.0, 0.0])
        aspects = np.array([0.0, 180.0])  # aspect irrelevant for flat
        lats = np.array([42.0, 35.0])

        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)

        np.testing.assert_allclose(potsw, horad, rtol=1e-10)

    def test_south_facing_more_than_north_mid_latitude(self) -> None:
        """South-facing slope gets more annual radiation than north-facing at mid-latitude."""
        slopes = np.array([0.3, 0.3])
        aspects = np.array([180.0, 0.0])  # south, north
        lats = np.array([42.0, 42.0])

        potsw, _horad, _sunhrs = compute_soltab(slopes, aspects, lats)

        annual_south = potsw[:, 0].sum()
        annual_north = potsw[:, 1].sum()
        assert annual_south > annual_north

    def test_single_hru(self) -> None:
        """Works with a single HRU."""
        slopes = np.array([0.2])
        aspects = np.array([180.0])
        lats = np.array([40.0])

        potsw, horad, sunhrs = compute_soltab(slopes, aspects, lats)

        assert potsw.shape == (366, 1)
        assert np.all(potsw >= 0)

    def test_sunhrs_reasonable_range(self) -> None:
        """Sunlight hours per day should be 0-24."""
        slopes = np.array([0.1, 0.5, 0.0])
        aspects = np.array([180.0, 270.0, 0.0])
        lats = np.array([42.0, 60.0, 0.0])

        _potsw, _horad, sunhrs = compute_soltab(slopes, aspects, lats)

        assert np.all(sunhrs >= 0)
        assert np.all(sunhrs <= 24)

    def test_equator_equinox_symmetry(self) -> None:
        """At equator on flat surface, ~12 hours of sunlight year-round."""
        slopes = np.array([0.0])
        aspects = np.array([0.0])
        lats = np.array([0.0])

        _potsw, _horad, sunhrs = compute_soltab(slopes, aspects, lats)

        # Should be approximately 12 hours throughout the year
        np.testing.assert_allclose(sunhrs, 12.0, atol=0.5)
```

**Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_solar.py::TestComputeSoltab -v
```

Expected: FAIL with `ImportError` (compute_soltab not defined yet)

**Step 3: Implement compute_soltab in `src/hydro_param/solar.py`**

Append to the module after the constants:

```python
# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_t(lats: np.ndarray, decl: np.ndarray) -> np.ndarray:
    """Sunrise equation: hour angle from local noon to sunrise/sunset.

    Parameters
    ----------
    lats
        Latitudes in radians, shape (nhru,).
    decl
        Solar declination for each day, shape (ndoy,).

    Returns
    -------
    np.ndarray
        Hour angles, shape (ndoy, nhru).
    """
    nhru = len(lats)
    lats_mat = np.tile(-np.tan(lats), (NDOY, 1))
    sol_dec_mat = np.transpose(np.tile(np.tan(decl), (nhru, 1)))
    tx = lats_mat * sol_dec_mat

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in arccos")
        result = np.arccos(np.copy(tx))

    result[np.where(tx < -1.0)] = _PI
    result[np.where(tx > 1.0)] = 0.0
    return result


def _func3(
    v: np.ndarray,
    w: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Potential solar radiation on a surface (Swift 1976, equation 6).

    Parameters
    ----------
    v
        Longitude offset between actual and equivalent slope (nhru,).
    w
        Latitude of the equivalent slope (nhru,).
    x
        Hour angle of sunset on equivalent slope (ndoy, nhru).
    y
        Hour angle of sunrise on equivalent slope (ndoy, nhru).

    Returns
    -------
    np.ndarray
        Potential solar radiation in cal/cm2/day, shape (ndoy, nhru).
    """
    nhru = len(v)
    vv = np.tile(v, (NDOY, 1))
    ww = np.tile(w, (NDOY, 1))
    rr = np.transpose(np.tile(r1, (nhru, 1)))
    dd = np.transpose(np.tile(solar_declination, (nhru, 1)))

    return (
        rr
        * _PI_12
        * (
            np.sin(dd) * np.sin(ww) * (x - y)
            + np.cos(dd) * np.cos(ww) * (np.sin(x + vv) - np.sin(y + vv))
        )
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_soltab(
    slopes: np.ndarray,
    aspects: np.ndarray,
    lats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute potential solar radiation tables (Swift 1976).

    Pre-computes daily potential clear-sky solar radiation for every HRU
    and every day of the year (1–366), accounting for terrain slope and
    aspect.

    Parameters
    ----------
    slopes
        HRU slopes as decimal fraction (rise/run), shape (nhru,).
    aspects
        HRU aspects in degrees (0=north, 180=south), shape (nhru,).
    lats
        HRU latitudes in decimal degrees, shape (nhru,).

    Returns
    -------
    soltab_potsw
        Potential SW radiation on sloped surface, cal/cm2/day (Langleys),
        shape (ndoy, nhru).
    soltab_horad_potsw
        Potential SW radiation on horizontal surface, cal/cm2/day,
        shape (ndoy, nhru).
    soltab_sunhrs
        Hours of direct sunlight, shape (ndoy, nhru).
    """
    nhru = len(slopes)

    # --- Horizontal surface (slope=0, aspect=0) ---
    horad, _ = _compute_soltab_core(
        np.zeros(nhru), np.zeros(nhru), lats
    )

    # --- Sloped surface ---
    potsw, sunhrs = _compute_soltab_core(slopes, aspects, lats)

    return potsw, horad, sunhrs


def _compute_soltab_core(
    slopes: np.ndarray,
    aspects: np.ndarray,
    lats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Core soltab computation for one surface configuration.

    Returns (solt, sunh) arrays of shape (ndoy, nhru).
    """
    nhru = len(slopes)

    # Slope derived quantities
    sl = np.arctan(slopes)
    sl_sin = np.sin(sl)
    sl_cos = np.cos(sl)

    # Aspect derived quantities
    aspects_rad = np.radians(aspects)
    aspects_cos = np.cos(aspects_rad)

    # Latitude derived quantities
    x0 = np.radians(lats)
    x0_cos = np.cos(x0)

    # Latitude of equivalent slope (Lee 1963, equation 13)
    x1 = np.arcsin(sl_cos * np.sin(x0) + sl_sin * x0_cos * aspects_cos)

    # Denominator of Lee 1963 equation 12
    d1 = sl_cos * x0_cos - sl_sin * np.sin(x0) * aspects_cos
    d1 = np.where(np.abs(d1) < _DNEARZERO, _DNEARZERO, d1)

    # Longitude difference (Lee 1963, equation 12)
    x2 = np.arctan(sl_sin * np.sin(aspects_rad) / d1)
    wh_d1_lt_zero = np.where(d1 < 0.0)
    if len(wh_d1_lt_zero[0]) > 0:
        x2[wh_d1_lt_zero] = x2[wh_d1_lt_zero] + _PI

    # Hour angles for equivalent slope
    tt = _compute_t(x1, solar_declination)
    t6 = -tt - np.tile(x2, (NDOY, 1))
    t7 = tt - np.tile(x2, (NDOY, 1))

    # Hour angles for horizontal surface at HRU latitude
    tt = _compute_t(x0, solar_declination)
    t0 = -tt
    t1 = tt

    # Clip slope sunrise/sunset to horizontal bounds
    t3 = np.copy(t7)
    wh = np.where(t7 > t1)
    if len(wh[0]) > 0:
        t3[wh] = t1[wh]

    t2 = np.copy(t6)
    wh = np.where(t6 < t0)
    if len(wh[0]) > 0:
        t2[wh] = t0[wh]

    t6_shifted = t6 + _TWO_PI
    t7_shifted = t7 - _TWO_PI

    wh = np.where(t3 < t2)
    if len(wh[0]):
        t2[wh] = 0.0
        t3[wh] = 0.0

    # Base case
    solt = _func3(x2, x1, t3, t2)
    sunh = (t3 - t2) * _PI_12

    # Wrap-around: t7_shifted > t0
    wh = np.where(t7_shifted > t0)
    if len(wh[0]):
        solt[wh] = _func3(x2, x1, t3, t2)[wh] + _func3(x2, x1, t7_shifted, t0)[wh]
        sunh[wh] = (t3 - t2 + t7_shifted - t0)[wh] * _PI_12

    # Wrap-around: t6_shifted < t1
    wh = np.where(t6_shifted < t1)
    if len(wh[0]):
        solt[wh] = _func3(x2, x1, t3, t2)[wh] + _func3(x2, x1, t1, t6_shifted)[wh]
        sunh[wh] = (t3 - t2 + t1 - t6_shifted)[wh] * _PI_12

    # Flat surfaces: override with horizontal computation
    mask_flat = np.tile(np.abs(sl), (NDOY, 1)) < _DNEARZERO
    solt = np.where(mask_flat, _func3(np.zeros(nhru), x0, t1, t0), solt)
    sunh = np.where(mask_flat, (t1 - t0) * _PI_12, sunh)

    # Clamp negatives
    sunh = np.where(sunh < _DNEARZERO, 0.0, sunh)
    wh = np.where(solt < 0.0)
    if len(wh[0]):
        solt[wh] = 0.0
        warnings.warn(
            f"{len(wh[0])}/{np.prod(solt.shape)} locations-times with "
            f"negative potential solar radiation (clamped to zero)."
        )

    return solt, sunh
```

**Step 4: Run tests to verify they pass**

```bash
pixi run -e dev pytest tests/test_solar.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/solar.py tests/test_solar.py
git commit -m "feat: implement compute_soltab (Swift 1976 algorithm)"
```

### Task 4: Wire soltab into derivation pipeline

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing test for _derive_soltab**

Add to `tests/test_pywatershed_derivation.py`:

```python
from hydro_param.solar import NDOY


class TestDeriveSoltab:
    """Tests for step 9: solar radiation tables."""

    def test_soltab_output_shape(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """soltab arrays have shape (366, nhru)."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soltab_potsw" in ds
        assert "soltab_horad_potsw" in ds
        assert ds["soltab_potsw"].shape == (NDOY, 3)
        assert ds["soltab_horad_potsw"].shape == (NDOY, 3)

    def test_soltab_dims(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """soltab arrays have (ndoy, nhru) dimensions."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["soltab_potsw"].dims == ("ndoy", "nhru")
        assert ds["soltab_horad_potsw"].dims == ("ndoy", "nhru")

    def test_soltab_non_negative(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert np.all(ds["soltab_potsw"].values >= 0)
        assert np.all(ds["soltab_horad_potsw"].values >= 0)

    def test_soltab_units_langleys(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["soltab_potsw"].attrs["units"] == "cal/cm2/day"

    def test_soltab_requires_topo_params(self, derivation: PywatershedDerivation) -> None:
        """Soltab is skipped when topo params are missing."""
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soltab_potsw" not in ds

    def test_soltab_sunhrs_present(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soltab_sunhrs" in ds
        assert ds["soltab_sunhrs"].shape == (NDOY, 3)
```

**Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSoltab -v
```

Expected: FAIL (no `soltab_potsw` in output)

**Step 3: Implement _derive_soltab in derivation module**

Add to `src/hydro_param/derivations/pywatershed.py`:

1. Add import at top (after existing imports):
```python
from hydro_param.solar import NDOY, compute_soltab
```

2. Add step method after `_apply_lookup_tables` (before `_apply_defaults`):
```python
    # ------------------------------------------------------------------
    # Step 9: Solar radiation tables
    # ------------------------------------------------------------------

    def _derive_soltab(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Step 9: Compute potential solar radiation tables (Swift 1976).

        Requires ``hru_lat``, ``hru_slope``, and ``hru_aspect`` from step 3.
        Produces 2-D arrays of shape (ndoy=366, nhru) for potential solar
        radiation on sloped and horizontal surfaces.
        """
        required = ("hru_lat", "hru_slope", "hru_aspect")
        if not all(v in ds for v in required):
            missing = [v for v in required if v not in ds]
            logger.info("Skipping soltab: missing %s", missing)
            return ds

        potsw, horad, sunhrs = compute_soltab(
            slopes=ds["hru_slope"].values,
            aspects=ds["hru_aspect"].values,
            lats=ds["hru_lat"].values,
        )

        ds["soltab_potsw"] = xr.DataArray(
            potsw,
            dims=("ndoy", "nhru"),
            attrs={"units": "cal/cm2/day", "long_name": "Potential SW radiation on slope"},
        )
        ds["soltab_horad_potsw"] = xr.DataArray(
            horad,
            dims=("ndoy", "nhru"),
            attrs={"units": "cal/cm2/day", "long_name": "Potential SW radiation on horizontal"},
        )
        ds["soltab_sunhrs"] = xr.DataArray(
            sunhrs,
            dims=("ndoy", "nhru"),
            attrs={"units": "hours", "long_name": "Hours of direct sunlight"},
        )
        return ds
```

3. Wire into `derive()` method — add after step 8 and before step 13:
```python
        # Step 9: Solar radiation tables (soltab)
        ds = self._derive_soltab(context, ds)
```

4. Update module docstring (line 7) to include step 9:
```python
Foundation implementation covers steps 1, 2, 3, 4, 8, 9, and 13.
```

5. Update class docstring (line 138-139) similarly.

**Step 4: Run tests to verify they pass**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSoltab -v
pixi run -e dev pytest tests/test_pywatershed_derivation.py -v  # ensure no regressions
pixi run -e dev pytest tests/test_solar.py -v
```

Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: wire soltab (step 9) into derivation pipeline"
```

### Task 5: Run full checks and create PR

**Step 1: Run all checks**

```bash
pixi run -e dev check
pixi run -e dev pre-commit
```

Expected: All pass

**Step 2: Push and create PR**

```bash
git push -u origin feat/<issue-number>-soltab
gh pr create --title "feat: implement step 9 solar radiation tables (soltab)" \
  --body "$(cat <<'EOF'
## Summary
- New `solar.py` module implementing Swift (1976) clear-sky radiation algorithm
- New `_derive_soltab()` step in `PywatershedDerivation`
- Produces `soltab_potsw`, `soltab_horad_potsw`, `soltab_sunhrs` (366 × nhru)
- Algorithm ported from pywatershed's `PRMSSolarGeometry`

Closes #<issue-number>
Part of #83

## Test plan
- [ ] Solar constants shape and range validation
- [ ] Output shapes (366, nhru) for all three arrays
- [ ] Flat surface: potsw == horad
- [ ] South-facing > north-facing at mid-latitude
- [ ] Single HRU edge case
- [ ] Sunlight hours in [0, 24] range
- [ ] Missing topo params → graceful skip
- [ ] No regressions in existing derivation tests

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## PR B: Step 5 — Soils Zonal Stats

### Task 6: Create GitHub issue and feature branch

**Step 1: Create issue and branch**

```bash
gh issue create --title "feat: implement step 5 soils zonal stats" \
  --body "Implement soil parameter derivation from gNATSGO zonal statistics. Part of #83.

## What
- New \`_derive_soils()\` step in \`PywatershedDerivation\`
- Produces \`soil_type\`, \`soil_moist_max\`, \`soil_rechr_max_frac\`
- Uses existing \`soil_texture_to_prms_type.yml\` lookup table
"
# Then create branch: feat/<issue-number>-soils
```

### Task 7: Write failing tests for soil derivation

**Files:**
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Add SIR fixture and test class**

Add fixture after existing fixtures:

```python
@pytest.fixture()
def sir_soils() -> xr.Dataset:
    """Synthetic SIR with soil data (gNATSGO-like)."""
    return xr.Dataset(
        {
            # Soil texture fractions (USDA classes)
            "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.1, 0.0])),
            "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.8, 0.1])),
            "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1, 0.9])),
            # Available water capacity and depth
            "awc_mm_mean": ("nhm_id", np.array([50.0, 150.0, 80.0])),
            "soil_depth_cm_mean": ("nhm_id", np.array([100.0, 150.0, 80.0])),
        },
        coords={"nhm_id": [1, 2, 3]},
    )
```

Add test class:

```python
class TestDeriveSoils:
    """Tests for step 5: soils zonal stats."""

    def test_soil_type_from_fractions(
        self, derivation: PywatershedDerivation, sir_soils: xr.Dataset
    ) -> None:
        """Majority texture class reclassified to PRMS soil_type."""
        ctx = DerivationContext(sir=sir_soils, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # HRU 1: sand dominant -> soil_type 1
        assert ds["soil_type"].values[0] == 1
        # HRU 2: loam dominant -> soil_type 2
        assert ds["soil_type"].values[1] == 2
        # HRU 3: clay dominant -> soil_type 3
        assert ds["soil_type"].values[2] == 3

    def test_soil_moist_max(
        self, derivation: PywatershedDerivation, sir_soils: xr.Dataset
    ) -> None:
        """soil_moist_max = awc * depth, converted to inches."""
        ctx = DerivationContext(sir=sir_soils, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_moist_max" in ds
        # HRU 1: 50 mm AWC/m * 1.0m depth = 50 mm = 1.9685 inches
        # (awc_mm_mean is mm of water per unit soil column, soil_depth_cm_mean is depth)
        # soil_moist_max = awc_mm_mean * (soil_depth_cm_mean / 100) converted to inches
        # Actually: awc_mm_mean is already total AWC in mm for the column
        # 50 mm = 50 / 25.4 = 1.9685 inches
        np.testing.assert_allclose(ds["soil_moist_max"].values[0], 50.0 / 25.4, atol=0.01)
        assert ds["soil_moist_max"].attrs["units"] == "inches"

    def test_soil_rechr_max_frac_default(
        self, derivation: PywatershedDerivation, sir_soils: xr.Dataset
    ) -> None:
        """soil_rechr_max_frac defaults to 0.4 when no layer data available."""
        ctx = DerivationContext(sir=sir_soils, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_rechr_max_frac" in ds
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, 0.4)

    def test_soil_moist_max_clipped(self, derivation: PywatershedDerivation) -> None:
        """soil_moist_max is clipped to [0.5, 20.0] inches."""
        sir = xr.Dataset(
            {
                "soil_texture_frac_sand": ("nhm_id", np.array([1.0, 1.0])),
                "soil_texture_frac_loam": ("nhm_id", np.array([0.0, 0.0])),
                "awc_mm_mean": ("nhm_id", np.array([1.0, 1000.0])),  # very low, very high
            },
            coords={"nhm_id": [1, 2]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["soil_moist_max"].values[0] >= 0.5
        assert ds["soil_moist_max"].values[1] <= 20.0

    def test_soils_missing_sir_vars(self, derivation: PywatershedDerivation) -> None:
        """Soils step is skipped when no soil data in SIR."""
        sir = xr.Dataset(
            {"elevation_m_mean": ("nhm_id", np.array([100.0]))},
            coords={"nhm_id": [1]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" not in ds
        assert "soil_moist_max" not in ds

    def test_soil_type_single_value_fallback(self, derivation: PywatershedDerivation) -> None:
        """Falls back to single soil_texture variable if no fractions."""
        sir = xr.Dataset(
            {
                "soil_texture": ("nhm_id", np.array(["sand", "clay"])),
                "awc_mm_mean": ("nhm_id", np.array([50.0, 80.0])),
            },
            coords={"nhm_id": [1, 2]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        assert ds["soil_type"].values[0] == 1  # sand
        assert ds["soil_type"].values[1] == 3  # clay
```

**Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSoils -v
```

Expected: FAIL (`soil_type` not in output)

### Task 8: Implement _derive_soils

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Add the _derive_soils method**

Add after `_derive_landcover` (before `_apply_lookup_tables`):

```python
    # ------------------------------------------------------------------
    # Step 5: Soils zonal stats
    # ------------------------------------------------------------------

    def _derive_soils(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Step 5: Derive soil parameters from gNATSGO/STATSGO2 zonal stats.

        Supports two input modes:

        1. **Soil texture fractions** (preferred): SIR contains columns
           like ``soil_texture_frac_sand``, ``soil_texture_frac_loam``, etc.
           Majority class via argmax, then reclassify to PRMS soil_type.
        2. **Single texture class**: ``soil_texture`` or
           ``soil_texture_majority`` containing the dominant USDA class name.

        Also derives ``soil_moist_max`` from available water capacity (AWC).
        """
        sir = ctx.sir

        # --- soil_type ---
        soil_type = self._compute_soil_type(sir, ctx)
        if soil_type is not None:
            ds["soil_type"] = xr.DataArray(
                soil_type,
                dims="nhru",
                attrs={"units": "integer", "long_name": "PRMS soil type (1=sand, 2=loam, 3=clay)"},
            )

        # --- soil_moist_max ---
        if "awc_mm_mean" in sir:
            awc_mm = sir["awc_mm_mean"].values.astype(np.float64)
            soil_moist_max = convert(awc_mm, "mm", "in")
            soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
            ds["soil_moist_max"] = xr.DataArray(
                soil_moist_max,
                dims="nhru",
                attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
            )

        # --- soil_rechr_max_frac ---
        if "soil_type" in ds:
            # Default: 0.4 (Regan et al. 2018) when layer data unavailable
            nhru = len(ds["soil_type"])
            ds["soil_rechr_max_frac"] = xr.DataArray(
                np.full(nhru, 0.4),
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Fraction of soil moisture in recharge zone",
                },
            )

        return ds

    def _compute_soil_type(
        self, sir: xr.Dataset, ctx: DerivationContext
    ) -> np.ndarray | None:
        """Compute PRMS soil_type from SIR soil texture data.

        Returns array of soil_type values (1/2/3) or None if no soil data.
        """
        tables_dir = ctx.resolved_lookup_tables_dir
        soil_table = self._load_lookup_table("soil_texture_to_prms_type", tables_dir)
        mapping = soil_table["mapping"]

        # Try fraction columns first
        prefix = "soil_texture_frac_"
        fraction_vars = sorted(str(v) for v in sir.data_vars if str(v).startswith(prefix))
        if len(fraction_vars) >= 2:
            class_names: list[str] = []
            valid_vars: list[str] = []
            for v in fraction_vars:
                name = v[len(prefix):]
                if name in mapping:
                    class_names.append(name)
                    valid_vars.append(v)
                else:
                    logger.debug("Skipping soil fraction '%s': class '%s' not in lookup", v, name)

            if len(valid_vars) >= 2:
                fractions = np.column_stack([sir[v].values for v in valid_vars])
                majority_idx = np.argmax(fractions, axis=1)
                majority_names = [class_names[i] for i in majority_idx]
                return np.array([mapping[name] for name in majority_names])

        # Fallback: single texture class variable
        for candidate in ("soil_texture", "soil_texture_majority"):
            if candidate in sir:
                texture_values = sir[candidate].values
                return np.array([mapping.get(str(v), 2) for v in texture_values])

        return None
```

**Step 2: Wire into derive() method**

Add after step 4 and before step 8:

```python
        # Step 5: Soils parameters (soil_type, soil_moist_max, soil_rechr_max_frac)
        ds = self._derive_soils(context, ds)
```

Update module and class docstrings to include step 5.

**Step 3: Run tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSoils -v
pixi run -e dev pytest tests/test_pywatershed_derivation.py -v  # no regressions
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: implement step 5 soils zonal stats derivation"
```

### Task 9: Run full checks and create PR

**Step 1: Run all checks**

```bash
pixi run -e dev check
pixi run -e dev pre-commit
```

**Step 2: Push and create PR**

```bash
git push -u origin feat/<issue-number>-soils
gh pr create --title "feat: implement step 5 soils zonal stats" \
  --body "$(cat <<'EOF'
## Summary
- New `_derive_soils()` step in `PywatershedDerivation`
- Produces `soil_type` (1=sand, 2=loam, 3=clay), `soil_moist_max` (inches), `soil_rechr_max_frac`
- Supports texture fraction columns (majority via argmax) and single-value fallback
- Uses existing `soil_texture_to_prms_type.yml` lookup table

Closes #<issue-number>
Part of #83

## Test plan
- [ ] Majority from fraction columns (sand/loam/clay)
- [ ] Reclassification to PRMS soil_type (1/2/3)
- [ ] soil_moist_max from AWC (mm → inches conversion)
- [ ] soil_moist_max clipped to [0.5, 20.0] inches
- [ ] soil_rechr_max_frac defaults to 0.4
- [ ] Single texture fallback
- [ ] Missing SIR vars → graceful skip
- [ ] No regressions

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## PR C: Step 14 — Calibration Seeds

### Task 10: Create GitHub issue and feature branch

**Step 1: Create issue and branch**

```bash
gh issue create --title "feat: implement step 14 calibration seeds" \
  --body "Implement YAML-driven calibration parameter seed generation. Part of #83.

## What
- New \`calibration_seeds.yml\` lookup table with ~22 seed definitions
- New \`_derive_calibration_seeds()\` step in \`PywatershedDerivation\`
- Safe formula dispatch (no eval) with 4 method types
- Graceful degradation: missing inputs → default values with warnings
"
# Then create branch: feat/<issue-number>-calibration-seeds
```

### Task 11: Create calibration_seeds.yml

**Files:**
- Create: `configs/lookup_tables/calibration_seeds.yml`

**Step 1: Write failing test for YAML structure**

Add to `tests/test_pywatershed_derivation.py`:

```python
class TestCalibrationSeedsYAML:
    """Tests for calibration_seeds.yml structure."""

    def test_yaml_loads(self, derivation: PywatershedDerivation) -> None:
        """calibration_seeds.yml loads and has required structure."""
        from hydro_param.plugins import DerivationContext

        tables_dir = DerivationContext(
            sir=xr.Dataset(coords={"nhm_id": [1]}),
            fabric_id_field="nhm_id",
        ).resolved_lookup_tables_dir
        data = derivation._load_lookup_table("calibration_seeds", tables_dir)
        assert "mapping" in data  # uses 'mapping' key like other tables
        for name, seed in data["mapping"].items():
            assert "method" in seed, f"Seed '{name}' missing 'method'"
            assert "params" in seed, f"Seed '{name}' missing 'params'"
            assert "range" in seed, f"Seed '{name}' missing 'range'"
            assert "default" in seed, f"Seed '{name}' missing 'default'"
            assert len(seed["range"]) == 2, f"Seed '{name}' range must be [min, max]"
```

**Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestCalibrationSeedsYAML -v
```

Expected: FAIL (file not found)

**Step 3: Create `configs/lookup_tables/calibration_seeds.yml`**

Note: this file must also be accessible via `importlib.resources` at
`hydro_param/data/lookup_tables/calibration_seeds.yml`. Check whether
the project copies configs into the package data directory or serves them
from the repo root. The existing tables are bundled at
`src/hydro_param/data/lookup_tables/`. Create the file there:

**Create:** `src/hydro_param/data/lookup_tables/calibration_seeds.yml`

```yaml
# Calibration parameter seed values for pywatershed/PRMS.
# Source: Regan et al. 2018 (TM6-B9), Hay et al. 2023, NHM methodology.
#
# Each seed has:
#   method: one of linear, exponential_scale, fraction_of, constant
#   params: method-specific parameters (input, scale, offset, etc.)
#   range: [min, max] — output is clipped to this range
#   default: fallback value when required inputs are missing

name: calibration_seeds
description: "Physically-based initial values for calibration parameters"
source: "Regan et al. 2018 (TM6-B9), Hay et al. 2023"

mapping:
  carea_max:
    method: linear
    params: {input: hru_percent_imperv, scale: 0.6, offset: 0.2}
    range: [0.0, 1.0]
    default: 0.4

  smidx_coef:
    method: exponential_scale
    params: {input: hru_slope, scale: 0.005, exponent: 3.0}
    range: [0.001, 0.06]
    default: 0.01

  smidx_exp:
    method: constant
    params: {value: 0.3}
    range: [0.1, 0.8]
    default: 0.3

  soil2gw_max:
    method: fraction_of
    params: {input: soil_moist_max, fraction: 0.1}
    range: [0.0, 5.0]
    default: 0.1

  ssr2gw_rate:
    method: constant
    params: {value: 0.1}
    range: [0.0, 1.0]
    default: 0.1

  ssr2gw_exp:
    method: constant
    params: {value: 1.0}
    range: [0.0, 3.0]
    default: 1.0

  slowcoef_lin:
    method: constant
    params: {value: 0.015}
    range: [0.001, 0.5]
    default: 0.015

  slowcoef_sq:
    method: constant
    params: {value: 0.1}
    range: [0.0, 1.0]
    default: 0.1

  fastcoef_lin:
    method: constant
    params: {value: 0.09}
    range: [0.001, 0.8]
    default: 0.09

  fastcoef_sq:
    method: constant
    params: {value: 0.8}
    range: [0.0, 1.0]
    default: 0.8

  pref_flow_den:
    method: constant
    params: {value: 0.0}
    range: [0.0, 0.1]
    default: 0.0

  gwflow_coef:
    method: constant
    params: {value: 0.015}
    range: [0.001, 0.5]
    default: 0.015

  gwsink_coef:
    method: constant
    params: {value: 0.0}
    range: [0.0, 1.0]
    default: 0.0

  snarea_thresh:
    method: fraction_of
    params: {input: soil_moist_max, fraction: 0.8}
    range: [0.0, 200.0]
    default: 50.0

  rain_cbh_adj:
    method: constant
    params: {value: 1.0}
    range: [0.5, 2.0]
    default: 1.0

  snow_cbh_adj:
    method: constant
    params: {value: 1.0}
    range: [0.5, 2.0]
    default: 1.0

  tmax_cbh_adj:
    method: constant
    params: {value: 0.0}
    range: [-10.0, 10.0]
    default: 0.0

  tmin_cbh_adj:
    method: constant
    params: {value: 0.0}
    range: [-10.0, 10.0]
    default: 0.0

  tmax_allrain_offset:
    method: constant
    params: {value: 1.0}
    range: [0.0, 10.0]
    default: 1.0

  adjmix_rain:
    method: constant
    params: {value: 1.0}
    range: [0.6, 1.4]
    default: 1.0

  dday_slope:
    method: constant
    params: {value: 0.4}
    range: [0.2, 0.9]
    default: 0.4

  dday_intcp:
    method: constant
    params: {value: -40.0}
    range: [-60.0, 10.0]
    default: -40.0
```

**Step 4: Run test to verify it passes**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestCalibrationSeedsYAML -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/data/lookup_tables/calibration_seeds.yml tests/test_pywatershed_derivation.py
git commit -m "feat: add calibration_seeds.yml lookup table"
```

### Task 12: Write failing tests for calibration seed derivation

**Files:**
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Add test class**

```python
class TestDeriveCalibrationSeeds:
    """Tests for step 14: calibration seed generation."""

    def test_constant_seeds_present(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """Constant seeds are always produced (no input dependencies)."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        constant_params = [
            "smidx_exp", "ssr2gw_rate", "ssr2gw_exp", "slowcoef_lin",
            "slowcoef_sq", "fastcoef_lin", "fastcoef_sq", "pref_flow_den",
            "gwflow_coef", "gwsink_coef", "rain_cbh_adj", "snow_cbh_adj",
            "tmax_cbh_adj", "tmin_cbh_adj", "tmax_allrain_offset",
            "adjmix_rain", "dday_slope", "dday_intcp",
        ]
        for param in constant_params:
            assert param in ds, f"Missing calibration seed: {param}"

    def test_linear_seed_carea_max(self, derivation: PywatershedDerivation) -> None:
        """carea_max = 0.6 * hru_percent_imperv + 0.2, clipped to [0, 1]."""
        sir = xr.Dataset(
            {
                "land_cover": ("nhm_id", np.array([42, 71])),
                "fctimp_pct_mean": ("nhm_id", np.array([50.0, 10.0])),  # 50%, 10%
            },
            coords={"nhm_id": [1, 2]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "carea_max" in ds
        # 0.6 * 0.50 + 0.2 = 0.50
        np.testing.assert_allclose(ds["carea_max"].values[0], 0.50, atol=0.01)
        # 0.6 * 0.10 + 0.2 = 0.26
        np.testing.assert_allclose(ds["carea_max"].values[1], 0.26, atol=0.01)

    def test_exponential_seed_smidx_coef(self, derivation: PywatershedDerivation) -> None:
        """smidx_coef = 0.005 * exp(3.0 * hru_slope), clipped."""
        sir = xr.Dataset(
            {
                "elevation_m_mean": ("nhm_id", np.array([100.0])),
                "slope_deg_mean": ("nhm_id", np.array([5.0])),
                "aspect_deg_mean": ("nhm_id", np.array([180.0])),
            },
            coords={"nhm_id": [1]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "smidx_coef" in ds
        slope_frac = np.tan(np.radians(5.0))  # ~0.0875
        expected = 0.005 * np.exp(3.0 * slope_frac)
        expected = np.clip(expected, 0.001, 0.06)
        np.testing.assert_allclose(ds["smidx_coef"].values[0], expected, rtol=0.01)

    def test_fraction_of_seed(self, derivation: PywatershedDerivation) -> None:
        """soil2gw_max = 0.1 * soil_moist_max."""
        sir = xr.Dataset(
            {
                "soil_texture_frac_sand": ("nhm_id", np.array([1.0])),
                "soil_texture_frac_loam": ("nhm_id", np.array([0.0])),
                "awc_mm_mean": ("nhm_id", np.array([127.0])),  # 127mm = 5.0 inches
            },
            coords={"nhm_id": [1]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil2gw_max" in ds
        # soil_moist_max ~ 5.0 inches, soil2gw_max = 0.1 * 5.0 = 0.5
        np.testing.assert_allclose(ds["soil2gw_max"].values[0], 0.5, atol=0.05)

    def test_missing_input_uses_default(self, derivation: PywatershedDerivation) -> None:
        """When input variable missing, seed uses default value."""
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # carea_max depends on hru_percent_imperv which is missing -> default 0.4
        assert "carea_max" in ds
        np.testing.assert_allclose(ds["carea_max"].values, 0.4)

    def test_range_clipping(self, derivation: PywatershedDerivation) -> None:
        """Seeds are clipped to their defined range."""
        sir = xr.Dataset(
            {
                "elevation_m_mean": ("nhm_id", np.array([100.0])),
                "slope_deg_mean": ("nhm_id", np.array([85.0])),  # very steep
                "aspect_deg_mean": ("nhm_id", np.array([180.0])),
            },
            coords={"nhm_id": [1]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # Very steep slope -> smidx_coef would be huge, but clipped to 0.06
        assert ds["smidx_coef"].values[0] <= 0.06

    def test_existing_param_not_overwritten(self, derivation: PywatershedDerivation) -> None:
        """Calibration seeds don't overwrite existing parameters."""
        sir = xr.Dataset(coords={"nhm_id": [1]})
        config = {"parameter_overrides": {"values": {"gwflow_coef": 0.999}}}
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", config=config)
        ds = derivation.derive(ctx)
        # Override applied after calibration seeds, so override wins
        np.testing.assert_allclose(ds["gwflow_coef"].values, 0.999)

    def test_all_seeds_produced(self, derivation: PywatershedDerivation) -> None:
        """All 22 calibration seeds are produced."""
        sir = xr.Dataset(coords={"nhm_id": [1]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        expected_seeds = [
            "carea_max", "smidx_coef", "smidx_exp", "soil2gw_max",
            "ssr2gw_rate", "ssr2gw_exp", "slowcoef_lin", "slowcoef_sq",
            "fastcoef_lin", "fastcoef_sq", "pref_flow_den", "gwflow_coef",
            "gwsink_coef", "snarea_thresh", "rain_cbh_adj", "snow_cbh_adj",
            "tmax_cbh_adj", "tmin_cbh_adj", "tmax_allrain_offset",
            "adjmix_rain", "dday_slope", "dday_intcp",
        ]
        for param in expected_seeds:
            assert param in ds, f"Missing calibration seed: {param}"
```

**Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveCalibrationSeeds -v
```

Expected: FAIL

### Task 13: Implement _derive_calibration_seeds

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Add seed method dispatch and derivation step**

Add after the `_IMPERV_STOR_MAX_DEFAULT` constant near the top of the file:

```python
# Seed computation methods — safe dispatch (no eval)
_SEED_METHODS: dict[str, callable] = {
    "linear": lambda ds, p: p["scale"] * ds[p["input"]].values + p["offset"],
    "exponential_scale": lambda ds, p: p["scale"] * np.exp(p["exponent"] * ds[p["input"]].values),
    "fraction_of": lambda ds, p: p["fraction"] * ds[p["input"]].values,
    "constant": lambda ds, p: np.float64(p["value"]),
}
```

Add the step method on `PywatershedDerivation`:

```python
    # ------------------------------------------------------------------
    # Step 14: Calibration seeds
    # ------------------------------------------------------------------

    def _derive_calibration_seeds(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Step 14: Generate physically-based calibration parameter seeds.

        Reads seed definitions from ``calibration_seeds.yml`` and evaluates
        each using a safe formula dispatch.  Falls back to default values
        when required inputs are missing.
        """
        tables_dir = ctx.resolved_lookup_tables_dir
        seed_table = self._load_lookup_table("calibration_seeds", tables_dir)
        seeds = seed_table["mapping"]

        nhru = ds.sizes.get("nhru", 0)

        for param_name, seed_def in seeds.items():
            if param_name in ds:
                continue  # Don't overwrite existing params

            method_name = seed_def["method"]
            params = seed_def["params"]
            val_range = seed_def["range"]
            default = seed_def["default"]

            if method_name not in _SEED_METHODS:
                logger.warning("Unknown seed method '%s' for '%s', using default", method_name, param_name)
                value = default
            else:
                # Check if required input exists
                input_field = params.get("input")
                if input_field and input_field not in ds:
                    logger.info(
                        "Calibration seed '%s' input '%s' not available, using default %.4g",
                        param_name, input_field, default,
                    )
                    value = default
                else:
                    value = _SEED_METHODS[method_name](ds, params)

            # Convert scalar to array if we have spatial dimension
            if isinstance(value, (int, float, np.floating)):
                if nhru > 0:
                    value = np.full(nhru, value, dtype=np.float64)
                else:
                    value = np.float64(value)

            # Clip to valid range
            value = np.clip(value, val_range[0], val_range[1])

            dims = ("nhru",) if isinstance(value, np.ndarray) and value.ndim == 1 else ()
            ds[param_name] = xr.DataArray(
                value,
                dims=dims if dims else None,
                attrs={"long_name": f"{param_name} (calibration seed)"},
            )

        return ds
```

**Step 2: Wire into derive() method**

Add after step 13 (defaults) and before parameter overrides:

```python
        # Step 14: Calibration seeds
        ds = self._derive_calibration_seeds(context, ds)
```

Update module and class docstrings to include step 14.

**Step 3: Run tests**

```bash
pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveCalibrationSeeds -v
pixi run -e dev pytest tests/test_pywatershed_derivation.py -v  # no regressions
```

Expected: All PASS

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: implement step 14 YAML-driven calibration seeds"
```

### Task 14: Run full checks and create PR

**Step 1: Run all checks**

```bash
pixi run -e dev check
pixi run -e dev pre-commit
```

**Step 2: Push and create PR**

```bash
git push -u origin feat/<issue-number>-calibration-seeds
gh pr create --title "feat: implement step 14 calibration seeds" \
  --body "$(cat <<'EOF'
## Summary
- New `calibration_seeds.yml` with 22 seed definitions
- New `_derive_calibration_seeds()` step in `PywatershedDerivation`
- Safe formula dispatch: `linear`, `exponential_scale`, `fraction_of`, `constant`
- Graceful degradation: missing inputs → default values with info log
- Seeds don't overwrite existing params (overrides still win)

Closes #<issue-number>
Part of #83

## Test plan
- [ ] All 22 seeds produced
- [ ] Constant seeds always present
- [ ] Linear seed (carea_max) computed correctly
- [ ] Exponential seed (smidx_coef) computed correctly
- [ ] fraction_of seed (soil2gw_max) computed correctly
- [ ] Missing input → default with log message
- [ ] Range clipping works
- [ ] Existing params not overwritten
- [ ] YAML structure validation
- [ ] No regressions

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Verification Checklist

After all three PRs are merged, verify:

1. `pixi run -e dev check` passes on main
2. `derive()` pipeline order is: 1 → 2 → 3 → 4 → 5 → 8 → 9 → 13 → 14 → overrides
3. Module docstring lists all implemented steps
4. Class docstring lists all implemented steps
5. Full derivation integration test produces soltab + soil + calibration params
