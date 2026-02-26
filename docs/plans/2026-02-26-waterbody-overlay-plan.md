# Waterbody Overlay (Step 6) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement derivation step 6 — polygon-on-polygon overlay of NHDPlus waterbodies against HRU polygons to derive `dprst_frac`, `dprst_area_max`, and `hru_type`.

**Architecture:** Add `waterbodies` field to `DerivationContext`, implement `_derive_waterbody()` in pywatershed.py using `gpd.overlay`, wire into `derive()` between steps 5 and 8.

**Tech Stack:** geopandas (overlay), numpy, xarray

**Design doc:** `docs/plans/2026-02-26-waterbody-overlay-design.md`

---

### Task 1: Add `waterbodies` field to DerivationContext

**Files:**
- Modify: `src/hydro_param/plugins.py:55-62`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Add the field**

In `src/hydro_param/plugins.py`, add `waterbodies` to `DerivationContext`:

```python
sir: xr.Dataset
temporal: dict[str, xr.Dataset] | None = None
fabric: gpd.GeoDataFrame | None = None
segments: gpd.GeoDataFrame | None = None
waterbodies: gpd.GeoDataFrame | None = None
fabric_id_field: str = "nhm_id"
segment_id_field: str | None = None
config: dict = field(default_factory=dict)
lookup_tables_dir: Path | None = None
```

Also update the docstring to document the new field:

```python
waterbodies
    NHDPlus waterbody polygon GeoDataFrame for depression storage
    parameters.  When ``None``, step 6 (waterbody overlay) is skipped.
```

**Step 2: Run tests to verify no regression**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v --tb=short 2>&1 | tail -20`
Expected: All existing tests pass (field defaults to None, no behavior change)

**Step 3: Commit**

```bash
git add src/hydro_param/plugins.py
git commit -m "feat: add waterbodies field to DerivationContext"
```

---

### Task 2: Write failing tests for step 6

**Files:**
- Create test fixtures and test class in: `tests/test_pywatershed_derivation.py`

**Step 1: Add waterbody test fixtures**

Add these fixtures after the existing `sir_topo_with_area` fixture:

```python
@pytest.fixture()
def waterbody_fabric() -> gpd.GeoDataFrame:
    """Synthetic HRU fabric with known areas for waterbody overlay tests.

    Two 100m x 100m square HRUs in EPSG:5070 (area = 10,000 m² each).
    """
    hru1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    hru2 = Polygon([(200, 0), (300, 0), (300, 100), (200, 100)])
    return gpd.GeoDataFrame(
        {"nhm_id": [1, 2], "geometry": [hru1, hru2]},
        crs="EPSG:5070",
    )


@pytest.fixture()
def waterbody_sir() -> xr.Dataset:
    """Synthetic SIR for waterbody overlay tests.

    Includes hru_area_m2 matching the waterbody_fabric (10,000 m² each).
    """
    return xr.Dataset(
        {"hru_area_m2": ("nhm_id", np.array([10000.0, 10000.0]))},
        coords={"nhm_id": [1, 2]},
    )


@pytest.fixture()
def sample_waterbodies() -> gpd.GeoDataFrame:
    """Synthetic waterbody polygons for overlay tests.

    - Waterbody A: 60m x 100m LakePond overlapping HRU 1 (60% coverage)
    - Waterbody B: 30m x 100m Reservoir overlapping HRU 2 (30% coverage)
    - Waterbody C: SwampMarsh overlapping HRU 1 (should be filtered out)
    """
    wb_a = Polygon([(0, 0), (60, 0), (60, 100), (0, 100)])
    wb_b = Polygon([(200, 0), (230, 0), (230, 100), (200, 100)])
    wb_c = Polygon([(70, 0), (90, 0), (90, 100), (70, 100)])
    return gpd.GeoDataFrame(
        {
            "comid": [101, 102, 103],
            "ftype": ["LakePond", "Reservoir", "SwampMarsh"],
            "geometry": [wb_a, wb_b, wb_c],
        },
        crs="EPSG:5070",
    )
```

**Step 2: Add TestDeriveWaterbody class with failing tests**

```python
class TestDeriveWaterbody:
    """Tests for step 6: waterbody overlay."""

    def test_overlay_fraction_and_area(
        self, derivation, waterbody_sir, waterbody_fabric, sample_waterbodies
    ):
        """Verify dprst_frac and dprst_area_max from known geometry."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=sample_waterbodies,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ds, ctx)

        # HRU 1: 60% LakePond coverage (SwampMarsh excluded)
        assert ds["dprst_frac"].values[0] == pytest.approx(0.6, abs=0.01)
        # HRU 2: 30% Reservoir coverage
        assert ds["dprst_frac"].values[1] == pytest.approx(0.3, abs=0.01)

        # Area in acres: 6000 m² and 3000 m²
        assert ds["dprst_area_max"].values[0] == pytest.approx(6000.0 / 4046.8564224, abs=0.01)
        assert ds["dprst_area_max"].values[1] == pytest.approx(3000.0 / 4046.8564224, abs=0.01)

    def test_hru_type_threshold(
        self, derivation, waterbody_sir, waterbody_fabric, sample_waterbodies
    ):
        """HRU with >50% coverage gets type=2 (lake), others type=1."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=sample_waterbodies,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ds, ctx)

        assert ds["hru_type"].values[0] == 2  # 60% > 50%
        assert ds["hru_type"].values[1] == 1  # 30% < 50%

    def test_no_waterbodies_fallback(self, derivation, waterbody_sir, waterbody_fabric):
        """When waterbodies=None, assign defaults."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=None,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ds, ctx)

        np.testing.assert_array_equal(ds["dprst_frac"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["dprst_area_max"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["hru_type"].values, [1, 1])

    def test_swamp_only_fallback(self, derivation, waterbody_sir, waterbody_fabric):
        """When only SwampMarsh waterbodies exist, assign defaults."""
        swamp = gpd.GeoDataFrame(
            {
                "comid": [201],
                "ftype": ["SwampMarsh"],
                "geometry": [Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=swamp,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ds, ctx)

        np.testing.assert_array_equal(ds["dprst_frac"].values, [0.0, 0.0])

    def test_partial_overlap(self, derivation, waterbody_sir, waterbody_fabric):
        """Waterbody extending beyond HRU — only clipped area counted."""
        # Waterbody extends 50m beyond HRU 1 boundary
        big_wb = gpd.GeoDataFrame(
            {
                "comid": [301],
                "ftype": ["LakePond"],
                "geometry": [Polygon([(-50, 0), (80, 0), (80, 100), (-50, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=big_wb,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ds, ctx)

        # Only 80m of 100m HRU covered (80%)
        assert ds["dprst_frac"].values[0] == pytest.approx(0.8, abs=0.01)
        # HRU 2 has no overlap
        assert ds["dprst_frac"].values[1] == pytest.approx(0.0, abs=0.01)

    def test_multiple_waterbodies_per_hru(self, derivation, waterbody_sir, waterbody_fabric):
        """Two waterbodies in one HRU — areas summed."""
        wb1 = Polygon([(0, 0), (20, 0), (20, 100), (0, 100)])
        wb2 = Polygon([(40, 0), (60, 0), (60, 100), (40, 100)])
        multi_wb = gpd.GeoDataFrame(
            {
                "comid": [401, 402],
                "ftype": ["LakePond", "LakePond"],
                "geometry": [wb1, wb2],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=multi_wb,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ds, ctx)

        # 20m + 20m = 40m of 100m → 40%
        assert ds["dprst_frac"].values[0] == pytest.approx(0.4, abs=0.01)

    def test_crs_mismatch_auto_reproject(self, derivation, waterbody_sir, waterbody_fabric):
        """Waterbodies in different CRS are reprojected to fabric CRS."""
        # Create waterbody in WGS84 that overlaps HRU 1 when reprojected
        # Use a small polygon near the origin in EPSG:5070, converted to 4326
        wb_5070 = gpd.GeoDataFrame(
            {
                "comid": [501],
                "ftype": ["LakePond"],
                "geometry": [Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        wb_4326 = wb_5070.to_crs("EPSG:4326")
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=wb_4326,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ds, ctx)

        # Should get ~50% coverage after reprojection
        assert ds["dprst_frac"].values[0] == pytest.approx(0.5, abs=0.05)
```

**Step 3: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveWaterbody -v --tb=short`
Expected: FAIL — `_derive_waterbody` does not exist yet

**Step 4: Commit failing tests**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: add failing tests for step 6 waterbody overlay"
```

---

### Task 3: Implement `_derive_waterbody`

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Add the M2_PER_ACRE constant near module top**

After the `_IMPERV_STOR_MAX_DEFAULT` line (~line 72):

```python
# Square meters per acre for area conversion
_M2_PER_ACRE = 4046.8564224
```

**Step 2: Implement `_derive_waterbody` method**

Add after the `_derive_soils` method (before `_apply_lookup_tables`):

```python
# ------------------------------------------------------------------
# Step 6: Waterbody overlay (depression storage)
# ------------------------------------------------------------------

def _derive_waterbody(
    self,
    ds: xr.Dataset,
    ctx: DerivationContext,
) -> xr.Dataset:
    """Step 6: Derive depression storage from waterbody overlay.

    Performs polygon-on-polygon overlay of NHDPlus waterbody polygons
    against HRU fabric to compute depression fraction, area, and HRU
    type classification.

    Parameters
    ----------
    ds
        In-progress parameter dataset (must contain ``hru_area``).
    ctx
        Derivation context (must contain ``fabric`` and ``waterbodies``).

    Returns
    -------
    xr.Dataset
        Dataset with ``dprst_frac``, ``dprst_area_max``, and ``hru_type``.
    """
    id_field = ctx.fabric_id_field
    nhru = ds.sizes.get("nhru", 0) or ds.sizes.get(id_field, 0)

    # Fallback: no waterbody data
    if ctx.waterbodies is None:
        logger.warning("No waterbody data provided; using defaults for step 6")
        ds["dprst_frac"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["dprst_area_max"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["hru_type"] = xr.DataArray(np.ones(nhru, dtype=np.int32), dims=id_field)
        return ds

    # Filter to LakePond and Reservoir only
    wb = ctx.waterbodies[ctx.waterbodies["ftype"].isin({"LakePond", "Reservoir"})].copy()
    if wb.empty:
        logger.info("No LakePond/Reservoir waterbodies found; using defaults for step 6")
        ds["dprst_frac"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["dprst_area_max"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["hru_type"] = xr.DataArray(np.ones(nhru, dtype=np.int32), dims=id_field)
        return ds

    fabric = ctx.fabric
    if fabric is None:
        logger.warning("No fabric provided; using defaults for step 6")
        ds["dprst_frac"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["dprst_area_max"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["hru_type"] = xr.DataArray(np.ones(nhru, dtype=np.int32), dims=id_field)
        return ds

    # Ensure matching CRS
    if wb.crs != fabric.crs:
        logger.info("Reprojecting waterbodies from %s to %s", wb.crs, fabric.crs)
        wb = wb.to_crs(fabric.crs)

    # Polygon overlay: intersection of fabric × waterbodies
    intersections = gpd.overlay(
        fabric[[id_field, "geometry"]],
        wb[["geometry"]],
        how="intersection",
    )

    if intersections.empty:
        logger.info("No waterbody-HRU intersections found; using defaults for step 6")
        ds["dprst_frac"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["dprst_area_max"] = xr.DataArray(np.zeros(nhru), dims=id_field)
        ds["hru_type"] = xr.DataArray(np.ones(nhru, dtype=np.int32), dims=id_field)
        return ds

    # Compute clipped areas and group by HRU
    intersections["_clip_area_m2"] = intersections.geometry.area
    area_by_hru = intersections.groupby(id_field)["_clip_area_m2"].sum()

    # Build arrays aligned to ds coordinate order
    hru_ids = ds[id_field].values if id_field in ds.coords else fabric[id_field].values
    clipped_acres = np.zeros(len(hru_ids))
    for i, hid in enumerate(hru_ids):
        if hid in area_by_hru.index:
            clipped_acres[i] = area_by_hru[hid] / _M2_PER_ACRE

    # Compute fraction from hru_area (already in acres from step 1)
    hru_area_acres = ds["hru_area"].values
    dprst_frac = np.where(hru_area_acres > 0, clipped_acres / hru_area_acres, 0.0)
    dprst_frac = np.clip(dprst_frac, 0.0, 1.0)

    # HRU type: 2 (lake) if >50% waterbody, else 1 (land)
    hru_type = np.where(dprst_frac > 0.5, 2, 1).astype(np.int32)

    ds["dprst_frac"] = xr.DataArray(dprst_frac, dims=id_field)
    ds["dprst_area_max"] = xr.DataArray(clipped_acres, dims=id_field)
    ds["hru_type"] = xr.DataArray(hru_type, dims=id_field)

    n_lake = int((hru_type == 2).sum())
    n_with_water = int((dprst_frac > 0).sum())
    logger.info(
        "Step 6 waterbody overlay: %d/%d HRUs with waterbodies, %d lake-type",
        n_with_water, nhru, n_lake,
    )

    return ds
```

**Step 3: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveWaterbody -v --tb=short`
Expected: All 7 tests PASS

**Step 4: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "feat: implement step 6 — waterbody overlay for depression storage"
```

---

### Task 4: Wire step 6 into `derive()` and add geopandas import

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:200-260` (derive method)

**Step 1: Ensure geopandas import at module top**

Check that `import geopandas as gpd` is present near the top of pywatershed.py. If not, add it.

**Step 2: Wire step 6 into derive()**

After the step 5 call (line ~234) and before step 8 (line ~237), add:

```python
        # Step 6: Waterbody overlay (dprst_frac, dprst_area_max, hru_type)
        ds = self._derive_waterbody(ds, context)
```

**Step 3: Update module docstring**

Update the module docstring to include step 6:

```
Foundation implementation covers steps 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, and 14.
```

Update the class docstring similarly.

**Step 4: Run full test suite**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v --tb=short 2>&1 | tail -20`
Expected: All tests pass (existing tests create DerivationContext without waterbodies, which defaults to None → step 6 produces defaults silently)

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "feat: wire step 6 waterbody overlay into derive() pipeline"
```

---

### Task 5: Add integration test and run full checks

**Files:**
- Modify: `tests/test_pywatershed_derivation.py`

**Step 1: Add integration test**

```python
class TestDeriveIntegrationWaterbody:
    """Integration test: full derive() with waterbody data."""

    def test_full_derive_with_waterbodies(self, derivation, waterbody_fabric):
        """Full pipeline produces waterbody params when waterbodies provided."""
        sir = xr.Dataset(
            {
                "hru_area_m2": ("nhm_id", np.array([10000.0, 10000.0])),
                "elevation_m_mean": ("nhm_id", np.array([100.0, 500.0])),
                "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0])),
                "aspect_deg_mean": ("nhm_id", np.array([0.0, 90.0])),
                "hru_lat": ("nhm_id", np.array([42.0, 41.5])),
                "land_cover": ("nhm_id", np.array([42, 71])),
            },
            coords={"nhm_id": [1, 2]},
        )
        wb = gpd.GeoDataFrame(
            {
                "comid": [101],
                "ftype": ["LakePond"],
                "geometry": [Polygon([(0, 0), (70, 0), (70, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=waterbody_fabric,
            waterbodies=wb,
        )
        ds = derivation.derive(ctx)

        assert "dprst_frac" in ds
        assert "dprst_area_max" in ds
        assert "hru_type" in ds
        assert ds["dprst_frac"].shape == (2,)
        assert ds["hru_type"].dtype == np.int32
        # HRU 1 should have 70% lake coverage → type 2
        assert ds["hru_type"].values[0] == 2
```

**Step 2: Run full checks**

Run: `pixi run -e dev check`
Expected: All lint, format, typecheck, and tests pass

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Commit**

```bash
git add tests/test_pywatershed_derivation.py
git commit -m "test: add integration test for step 6 waterbody overlay"
```

---

### Task 6: Update defaults and docstrings

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Add waterbody defaults to `_DEFAULTS` dict**

Add to the `_DEFAULTS` dict (if not already covered by step 13):

```python
# Depression storage
"dprst_frac": 0.0,
"dprst_area_max": 0.0,
"hru_type": 1,
```

Also add `"hru_type"` to `_DEFAULTS_SPECIAL` so `_apply_defaults` handles integer dtype:

```python
_DEFAULTS_SPECIAL: frozenset[str] = frozenset({"jh_coef", "transp_beg", "transp_end", "hru_type"})
```

**Step 2: Update `_apply_defaults` to handle hru_type as int32**

Ensure the special handling in `_apply_defaults` casts `hru_type` to int32, similar to `transp_beg`/`transp_end`.

**Step 3: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v --tb=short 2>&1 | tail -20`
Expected: All tests pass

**Step 4: Run full checks**

Run: `pixi run -e dev check && pixi run -e dev pre-commit`
Expected: All pass

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "chore: add waterbody defaults and update docstrings"
```
