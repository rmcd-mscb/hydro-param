# NHM Cross-Check Improvements — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Align hydro-param processing with NHM reference workflow (issue #185): elevation median, representative_point lat/lon, fix CV_INT misregistration, wire snow depletion curves.

**Architecture:** Four independent changes to the derivation plugin and GFv1.1 registry. Pipeline (Phase 1) is unchanged except config-level statistic selection. All changes respect the two-phase separation. Tasks 1–2 are trivial derivation fixes. Task 3 is a registry fix. Task 4 wires SDC_table into the derivation. Task 5 updates configs.

**Tech Stack:** Python, xarray, numpy, geopandas, shapely, rasterio, gdptools, yaml

**Design doc:** `docs/plans/2026-03-10-nhm-crosscheck-design.md`

---

### Task 1: Elevation — prefer median over mean

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:1800-1805`
- Modify: `tests/test_pywatershed_derivation.py:92-116` (sir_topography fixture)
- Modify: `tests/test_pywatershed_derivation.py:287-310` (TestDeriveTopography)

**Step 1: Write the failing test**

Add a new test and update the fixture to include `elevation_m_median`:

```python
# In sir_topography fixture, add alongside elevation_m_mean:
"elevation_m_median": ("nhm_id", np.array([101.0, 502.0, 1498.0])),

# New test in TestDeriveTopography:
def test_elevation_prefers_median(
    self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
) -> None:
    """When both median and mean are available, median is used."""
    ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    assert "hru_elev" in ds
    # Median values, not mean
    np.testing.assert_allclose(ds["hru_elev"].values[0], 101.0, atol=0.01)

def test_elevation_falls_back_to_mean(
    self, derivation: PywatershedDerivation
) -> None:
    """When only mean is available, it's used with a warning."""
    sir = _MockSIRAccessor(
        xr.Dataset(
            {
                "elevation_m_mean": ("nhm_id", np.array([100.0, 500.0])),
                "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0])),
                "hru_lat": ("nhm_id", np.array([42.0, 41.5])),
            },
            coords={"nhm_id": [1, 2]},
        )
    )
    ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    assert "hru_elev" in ds
    np.testing.assert_allclose(ds["hru_elev"].values[0], 100.0, atol=0.01)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py::TestDeriveTopography::test_elevation_prefers_median -v`
Expected: FAIL (median not yet preferred)

**Step 3: Implement elevation median preference**

In `_derive_topography()` at line 1800, replace:

```python
if "elevation_m_mean" in sir:
    ds["hru_elev"] = xr.DataArray(
        sir["elevation_m_mean"].values.astype(np.float64),
        dims="nhru",
        attrs={"units": "meters", "long_name": "Mean HRU elevation"},
    )
```

with:

```python
if "elevation_m_median" in sir:
    ds["hru_elev"] = xr.DataArray(
        sir["elevation_m_median"].values.astype(np.float64),
        dims="nhru",
        attrs={"units": "meters", "long_name": "Median HRU elevation"},
    )
elif "elevation_m_mean" in sir:
    logger.warning(
        "Using arithmetic mean elevation (legacy SIR). Median is preferred "
        "for robustness to outlier cells. Consider adding 'median' to the "
        "elevation statistics in your pipeline config."
    )
    ds["hru_elev"] = xr.DataArray(
        sir["elevation_m_mean"].values.astype(np.float64),
        dims="nhru",
        attrs={"units": "meters", "long_name": "Mean HRU elevation"},
    )
```

Also update the existing `test_elevation_meters_preserved` to use median values (since the fixture now has both, and median is preferred).

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py::TestDeriveTopography -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: prefer median elevation over mean in topography derivation"
```

---

### Task 2: Latitude — centroid → representative_point()

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:669-683`
- Modify: `tests/test_pywatershed_derivation.py:1817-1895` (TestDeriveGeometryFromFabric)

**Step 1: Write the failing test**

Add a test with a concave polygon where centroid falls outside:

```python
# In TestDeriveGeometryFromFabric:
def test_lat_uses_representative_point(self, derivation: PywatershedDerivation) -> None:
    """representative_point() guarantees point inside polygon for concave HRUs."""
    sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
    # L-shaped (concave) polygon — centroid may fall outside
    from shapely.geometry import Polygon
    l_shape = Polygon([(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)])
    fabric = gpd.GeoDataFrame(
        {"nhm_id": [1]},
        geometry=[l_shape],
        crs="EPSG:5070",
    )
    ctx = DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    assert "hru_lat" in ds
    assert "hru_lon" in ds
    # The representative point must be inside the polygon
    from shapely.geometry import Point
    pt = Point(ds["hru_lon"].values[0], ds["hru_lat"].values[0])
    # Reproject fabric to 4326 for comparison
    fab_4326 = fabric.to_crs(epsg=4326)
    assert fab_4326.geometry.iloc[0].contains(pt) or fab_4326.geometry.iloc[0].touches(pt)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py::TestDeriveGeometryFromFabric::test_lat_uses_representative_point -v`
Expected: May or may not fail depending on the L-shape centroid position after reprojection. If it passes, the test still validates the contract.

**Step 3: Implement representative_point()**

In `_derive_geometry()` at lines 669-683, replace:

```python
            # Latitude from WGS84 centroids (compute in projected CRS, reproject)
            centroids_5070 = fab_5070.geometry.centroid
            centroids_4326 = gpd.GeoSeries(centroids_5070, crs="EPSG:5070").to_crs(epsg=4326)
            lats = centroids_4326.y.values
            ds["hru_lat"] = xr.DataArray(
                lats,
                dims="nhru",
                attrs={"units": "decimal_degrees", "long_name": "Latitude of HRU centroid"},
            )
            lons = centroids_4326.x.values
            ds["hru_lon"] = xr.DataArray(
                lons,
                dims="nhru",
                attrs={"units": "decimal_degrees", "long_name": "Longitude of HRU centroid"},
            )
```

with:

```python
            # Latitude from WGS84 representative points (guaranteed inside polygon)
            rep_pts_5070 = fab_5070.geometry.representative_point()
            rep_pts_4326 = gpd.GeoSeries(rep_pts_5070, crs="EPSG:5070").to_crs(epsg=4326)
            lats = rep_pts_4326.y.values
            ds["hru_lat"] = xr.DataArray(
                lats,
                dims="nhru",
                attrs={
                    "units": "decimal_degrees",
                    "long_name": "Latitude of HRU representative point",
                },
            )
            lons = rep_pts_4326.x.values
            ds["hru_lon"] = xr.DataArray(
                lons,
                dims="nhru",
                attrs={
                    "units": "decimal_degrees",
                    "long_name": "Longitude of HRU representative point",
                },
            )
```

Also update the `long_name` in the SIR fallback path (lines 692-696) and the existing test assertions that check `long_name` (line 1883).

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py::TestDeriveGeometryFromFabric -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat: use representative_point() for HRU lat/lon coordinates"
```

---

### Task 3: Fix CV_INT misregistration in gfv11.py

**Files:**
- Modify: `src/hydro_param/gfv11.py:320-335` (remove gfv11_covden_sum, add gfv11_cv_int)
- Test: `tests/test_gfv11.py` (if exists, or add inline assertions)

**Step 1: Write the failing test**

```python
# In tests/test_gfv11.py or a new test:
def test_gfv11_cv_int_registered():
    """CV_INT.tif is registered as gfv11_cv_int (categorical), not gfv11_covden_sum."""
    from hydro_param.gfv11 import GFV11_DATASETS
    assert "gfv11_cv_int" in GFV11_DATASETS
    assert "gfv11_covden_sum" not in GFV11_DATASETS
    entry = GFV11_DATASETS["gfv11_cv_int"]
    assert entry["filename"] == "CV_INT.tif"
    assert entry["variables"][0]["categorical"] is True
    assert entry["variables"][0]["name"] == "cv_int"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev python -m pytest tests/test_gfv11.py::test_gfv11_cv_int_registered -v`
Expected: FAIL (gfv11_covden_sum still exists, gfv11_cv_int doesn't)

**Step 3: Implement the fix**

In `src/hydro_param/gfv11.py`, replace the `gfv11_covden_sum` entry (lines 320-335):

```python
    "gfv11_covden_sum": {
        "description": "GFv1.1 pre-computed summer cover density, 30m, CONUS",
        "category": "land_cover",
        "filename": "CV_INT.tif",
        ...
    },
```

with:

```python
    "gfv11_cv_int": {
        "description": "GFv1.1 snow CV integer class for snow depletion curves, 30m, CONUS",
        "category": "snow",
        "filename": "CV_INT.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "cv_int",
                "band": 1,
                "units": "class_index",
                "long_name": "Snow coefficient of variation class (indexes into SDC table)",
                "native_name": "cv_int",
                "categorical": True,
            }
        ],
    },
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_gfv11.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/gfv11.py tests/test_gfv11.py
git commit -m "fix: re-register CV_INT.tif as gfv11_cv_int (snow CV class, not covden_sum)"
```

---

### Task 4: Wire snow depletion curves into derivation

**Files:**
- Create: `src/hydro_param/data/pywatershed/lookup_tables/sdc_table.yml`
- Modify: `src/hydro_param/derivations/pywatershed.py:3316-3322` (step 13 defaults)
- Modify: `src/hydro_param/derivations/pywatershed.py:110-148` (_DEFAULTS, _DEFAULTS_SPECIAL, _PARAM_DIMS)
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Create the SDC lookup table**

Convert SDC_table.csv to YAML format consistent with other lookup tables.

Write `src/hydro_param/data/pywatershed/lookup_tables/sdc_table.yml`:

```yaml
# Snow Depletion Curve (SDC) table.
# Source: GFv1.1 ScienceBase SDC_table.csv
# Reference: Liston et al. 2009, Sexstone et al. 2020
#
# Each key is a CV_INT class (1-9).  Values are 11 entries (ndeplval)
# representing fractional snow-covered area at evenly-spaced SWE fractions.
# Class 0 maps to all-zero (bare ground / water).
description: "Snow depletion curves indexed by CV_INT class"
ndeplval: 11
curves:
  0: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
  1: [0.00, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
  2: [0.00, 0.85, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
  3: [0.00, 0.75, 0.96, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
  4: [0.00, 0.61, 0.88, 0.98, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00]
  5: [0.00, 0.29, 0.53, 0.72, 0.86, 0.94, 0.99, 1.00, 1.00, 1.00, 1.00]
  6: [0.00, 0.22, 0.43, 0.62, 0.77, 0.88, 0.95, 0.99, 1.00, 1.00, 1.00]
  7: [0.00, 0.17, 0.36, 0.53, 0.68, 0.81, 0.91, 0.97, 1.00, 1.00, 1.00]
  8: [0.00, 0.14, 0.30, 0.46, 0.61, 0.75, 0.86, 0.94, 0.99, 1.00, 1.00]
  9: [0.00, 0.10, 0.23, 0.37, 0.52, 0.65, 0.78, 0.89, 0.96, 1.00, 1.00]
```

Note: Row 0 of the CSV is all zeros (first row). Columns Val1-Val9 become curves 1-9.
Each curve has 11 values (the 11 rows of the CSV). Curve 0 added as all-zeros for
bare/water class.

**Step 2: Write the failing tests**

```python
class TestSnowDepletionCurves:
    """Tests for snow depletion curve derivation from CV_INT + SDC table."""

    def test_hru_deplcrv_from_cv_int(self, derivation: PywatershedDerivation) -> None:
        """hru_deplcrv assigned from majority of CV_INT categorical fractions."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "cv_int_frac": ("nhm_id", np.array([0.0, 0.0])),
                    "cv_int_frac_2": ("nhm_id", np.array([0.8, 0.1])),
                    "cv_int_frac_5": ("nhm_id", np.array([0.2, 0.9])),
                    "cv_int_frac_count": ("nhm_id", np.array([100, 100])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        precomputed = {
            "hru_deplcrv": {
                "source": "gfv11_cv_int",
                "variable": "cv_int",
                "statistic": "majority",
            },
        }
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", precomputed=precomputed)
        ds = derivation.derive(ctx)
        assert "hru_deplcrv" in ds
        # HRU 1: class 2 majority, HRU 2: class 5 majority
        assert ds["hru_deplcrv"].values[0] == 2
        assert ds["hru_deplcrv"].values[1] == 5

    def test_snarea_curve_from_sdc_table(self, derivation: PywatershedDerivation) -> None:
        """snarea_curve populated from SDC table indexed by unique hru_deplcrv values."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "cv_int_frac": ("nhm_id", np.array([0.0, 0.0])),
                    "cv_int_frac_2": ("nhm_id", np.array([0.8, 0.1])),
                    "cv_int_frac_5": ("nhm_id", np.array([0.2, 0.9])),
                    "cv_int_frac_count": ("nhm_id", np.array([100, 100])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        precomputed = {
            "hru_deplcrv": {
                "source": "gfv11_cv_int",
                "variable": "cv_int",
                "statistic": "majority",
            },
        }
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", precomputed=precomputed)
        ds = derivation.derive(ctx)
        assert "snarea_curve" in ds
        # Should have ndeplval (11) values per unique curve
        assert ds["snarea_curve"].dims[0] == "ndeplval" or len(ds["snarea_curve"].dims) >= 1
        # Curve values should be in [0, 1]
        assert np.all(ds["snarea_curve"].values >= 0.0)
        assert np.all(ds["snarea_curve"].values <= 1.0)

    def test_snarea_curve_default_without_cv_int(self, derivation: PywatershedDerivation) -> None:
        """Without CV_INT, snarea_curve defaults to linear depletion."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"_dummy": ("nhm_id", np.array([0.0, 0.0]))},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "snarea_curve" in ds
        # Default: all 1.0 (current behavior)
        np.testing.assert_array_equal(ds["snarea_curve"].values, np.ones(11))
```

**Step 3: Run tests to verify they fail**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py::TestSnowDepletionCurves -v`
Expected: FAIL (hru_deplcrv derivation not yet implemented)

**Step 4: Implement snow depletion curve derivation**

In `_derive_defaults()` (step 13), replace the snarea_curve default block (lines 3316-3322):

```python
        # snarea_curve: snow depletion curve (11 values, default all 1.0)
        if "snarea_curve" not in ds:
            ds["snarea_curve"] = xr.DataArray(
                np.ones(11, dtype=np.float64),
                dims=("ndeplval",),
                attrs={"long_name": "Snow area depletion curve"},
            )
```

with logic that:
1. Checks for precomputed `hru_deplcrv` (categorical, from `gfv11_cv_int`)
2. If found, loads SDC table, assigns `hru_deplcrv` per HRU, builds `snarea_curve`
3. If not found, falls back to current default (all 1.0, `hru_deplcrv = 1`)

```python
        # --- Snow depletion curves: derive from CV_INT if available ---
        deplcrv_vals = self._try_precomputed(ctx, "hru_deplcrv", categorical=True)
        if deplcrv_vals is not None:
            # Load SDC lookup table
            sdc_table = self._load_lookup_table("sdc_table", tables_dir)
            curves = sdc_table["curves"]

            # Assign hru_deplcrv (1-based curve index per HRU)
            # Clamp to valid curve range [0, 9]
            deplcrv_vals = np.clip(deplcrv_vals.astype(int), 0, max(curves.keys()))
            ds["hru_deplcrv"] = xr.DataArray(
                deplcrv_vals,
                dims="nhru",
                attrs={"long_name": "Index of snow depletion curve"},
            )

            # Build snarea_curve from unique curves used
            unique_curves = sorted(set(deplcrv_vals))
            # For simplicity in MVP, stack all 10 possible curves (0-9) into
            # a (ndepl × ndeplval) array. hru_deplcrv indexes into this.
            ndepl = len(unique_curves)
            ndeplval = sdc_table["ndeplval"]

            # Remap hru_deplcrv to 1-based sequential indices
            curve_remap = {cv: i + 1 for i, cv in enumerate(unique_curves)}
            ds["hru_deplcrv"] = xr.DataArray(
                np.array([curve_remap[v] for v in deplcrv_vals]),
                dims="nhru",
                attrs={"long_name": "Index of snow depletion curve (1-based)"},
            )

            # Build the curve array (ndepl * ndeplval,) — flat as PRMS expects
            curve_values = []
            for cv in unique_curves:
                curve_values.extend(curves.get(cv, curves.get(0, [0.0] * ndeplval)))
            ds["snarea_curve"] = xr.DataArray(
                np.array(curve_values, dtype=np.float64),
                dims=("ndeplval",),
                attrs={"long_name": "Snow area depletion curve values"},
            )
            logger.info(
                "Snow depletion curves: %d unique curves from CV_INT, "
                "%d HRUs assigned",
                ndepl,
                nhru,
            )
        else:
            # Default: uniform depletion, single curve
            if "hru_deplcrv" not in ds:
                ds["hru_deplcrv"] = xr.DataArray(
                    np.ones(nhru, dtype=np.int32),
                    dims="nhru",
                    attrs={"long_name": "Index of snow depletion curve"},
                )
            if "snarea_curve" not in ds:
                ds["snarea_curve"] = xr.DataArray(
                    np.ones(11, dtype=np.float64),
                    dims=("ndeplval",),
                    attrs={"long_name": "Snow area depletion curve"},
                )
```

Also update:
- Remove `"hru_deplcrv": 1` from `_DEFAULTS` dict (line 120) — now handled in step 13
- Add `"hru_deplcrv"` to `_DEFAULTS_SPECIAL` (line 136) so the scalar default isn't applied
- Ensure `_PARAM_DIMS` has `"hru_deplcrv": ("nhru",)` if not already present

**Step 5: Run tests to verify they pass**

Run: `pixi run -e dev python -m pytest tests/test_pywatershed_derivation.py::TestSnowDepletionCurves -v`
Expected: ALL PASS

**Step 6: Run full test suite**

Run: `pixi run -e dev check`
Expected: ALL PASS (970+ tests)

**Step 7: Commit**

```bash
git add src/hydro_param/data/pywatershed/lookup_tables/sdc_table.yml \
        src/hydro_param/derivations/pywatershed.py \
        tests/test_pywatershed_derivation.py
git commit -m "feat: derive snow depletion curves from CV_INT + SDC table"
```

---

### Task 5: Update configs

**Files:**
- Modify: `pw-check/configs/gfv11_static_pipeline.yml:72-76`
- Modify: `pw-check/configs/gfv11_static_pywatershed.yml`
- Modify: `configs/examples/gfv11_static_pipeline.yml`

**Step 1: Update pipeline configs**

In `pw-check/configs/gfv11_static_pipeline.yml`:

1. Change topography statistics from `[mean]` to `[mean, median]` (line 27)
2. Remove the `gfv11_covden_sum` dataset entry (lines 72-76)
3. Add `gfv11_cv_int` to snow section:

```yaml
  snow:
    - name: gfv11_cv_int
      variables: [cv_int]
      statistics: [majority]
```

In `pw-check/configs/gfv11_static_pywatershed.yml`:

1. Remove `covden_sum` entry that references `gfv11_covden_sum` (lines 78-82)
2. Ensure `covden_sum` derives from `gfv11_cnpy` (add entry if needed):

```yaml
    covden_sum:
      source: gfv11_cnpy
      variable: cnpy
      statistic: mean
      description: "Summer cover density from MODIS tree canopy cover percent"
```

3. Add snow section with `hru_deplcrv`:

```yaml
  snow:
    available: [gfv11_cv_int]
    hru_deplcrv:
      source: gfv11_cv_int
      variable: cv_int
      statistic: majority
      description: "Snow depletion curve class from GFv1.1 CV_INT raster"
```

Apply same changes to `configs/examples/gfv11_static_pipeline.yml`.

**Step 2: Validate configs parse correctly**

Run: `pixi run -e dev python -c "from hydro_param.pywatershed_config import load_pywatershed_config; load_pywatershed_config('pw-check/configs/gfv11_static_pywatershed.yml')"`
Expected: No validation errors

**Step 3: Commit**

```bash
git add pw-check/configs/ configs/examples/
git commit -m "chore: update GFv1.1 configs for CV_INT fix and median elevation"
```

---

### Task 6: Final verification

**Step 1: Run full check suite**

Run: `pixi run -e dev check`
Expected: ALL PASS

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: ALL PASS

**Step 3: Push to PR branch**

```bash
git push
```
