# Add Missing Segment Spatial Parameters (#159) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `seg_lat`, `seg_cum_area`, and `seg_elev` parameters to the pywatershed derivation.

**Architecture:** `seg_lat` extends `_derive_topology()` (step 2) using WGS84 segment centroids. `seg_cum_area` extends `_derive_routing()` (step 12) by expanding the existing VAA spatial-join to include `totdasqkm`. `seg_elev` adds a new `_derive_segment_elevation()` method using gdptools `InterpGen` to sample a 3DEP DEM raster along segment polylines. All three get metadata entries.

**Tech Stack:** xarray, geopandas, numpy, gdptools (InterpGen, UserTiffData), pynhd, pytest

---

### Task 1: Add `seg_lat` to topology derivation

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (`_derive_topology`, ~line 733-778)
- Modify: `src/hydro_param/data/pywatershed/parameter_metadata.yml`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing test**

Add to `TestDeriveTopology` class (after the existing topology tests):

```python
def test_seg_lat_from_segments(
    self,
    derivation: PywatershedDerivation,
    sir_minimal: _MockSIRAccessor,
    synthetic_fabric: gpd.GeoDataFrame,
    synthetic_segments: gpd.GeoDataFrame,
) -> None:
    """seg_lat computed from segment centroid latitude."""
    ctx = DerivationContext(
        sir=sir_minimal,
        fabric=synthetic_fabric,
        segments=synthetic_segments,
        fabric_id_field="nhm_id",
        segment_id_field="nhm_seg",
    )
    ds = derivation.derive(ctx)
    assert "seg_lat" in ds
    assert ds["seg_lat"].dims == ("nsegment",)
    # synthetic_segments are at y=0.5 (EPSG:4326)
    np.testing.assert_allclose(ds["seg_lat"].values, [0.5, 0.5, 0.5], atol=0.01)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveTopology::test_seg_lat_from_segments -v`
Expected: FAIL — `seg_lat` not in ds

**Step 3: Implement `seg_lat` in `_derive_topology()`**

In `_derive_topology()`, after the `hru_segment_nhm` block (before `return ds`), add:

```python
        # --- seg_lat: segment centroid latitude (WGS84) ---
        if segments.crs is not None and not segments.crs.is_geographic:
            segs_4326 = segments.to_crs(epsg=4326)
        else:
            segs_4326 = segments
        seg_centroids = segs_4326.geometry.centroid
        ds["seg_lat"] = xr.DataArray(
            seg_centroids.y.values,
            dims="nsegment",
            attrs={
                "units": "decimal_degrees",
                "long_name": "Latitude of segment centroid",
            },
        )
```

Update the docstring to mention `seg_lat` in the Returns section:
- Add ``seg_lat`` : decimal degrees on ``nsegment``

**Step 4: Add metadata entry**

In `parameter_metadata.yml`, add after the topology section (near `seg_length` or routing params):

```yaml
  seg_lat:
    dimension: nsegment
    units: decimal_degrees
    valid_range: [-90.0, 90.0]
    required: false
    description: "Latitude of segment centroid"
```

**Step 5: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveTopology::test_seg_lat_from_segments -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py src/hydro_param/data/pywatershed/parameter_metadata.yml tests/test_pywatershed_derivation.py
git commit -m "feat: add seg_lat to topology derivation (#159)"
```

---

### Task 2: Add `seg_cum_area` to routing derivation

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (`_fetch_vaa`, `_get_slopes_from_comid`, `_get_slopes_spatial_join`, `_derive_routing`)
- Modify: `src/hydro_param/data/pywatershed/parameter_metadata.yml`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing test**

Add to `TestDeriveRouting` class (find it by searching for `class TestDeriveRouting` or the routing test section):

```python
def test_seg_cum_area_from_vaa(
    self,
    derivation: PywatershedDerivation,
) -> None:
    """seg_cum_area derived from VAA totdasqkm via spatial join."""
    sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
    fabric = gpd.GeoDataFrame(
        {"nhm_id": [1, 2], "hru_segment": [1, 2]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        {"nhm_seg": [101, 102], "tosegment": [2, 0]},
        geometry=[
            LineString([(0.5, 0.5), (1.0, 0.5)]),
            LineString([(1.0, 0.5), (2.0, 0.5)]),
        ],
        crs="EPSG:4326",
    )
    ctx = DerivationContext(
        sir=sir,
        fabric=fabric,
        segments=segments,
        fabric_id_field="nhm_id",
        segment_id_field="nhm_seg",
    )
    ds = derivation.derive(ctx)
    assert "seg_cum_area" in ds
    assert ds["seg_cum_area"].dims == ("nsegment",)
    assert ds["seg_cum_area"].attrs["units"] == "acres"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "test_seg_cum_area" -v`
Expected: FAIL — `seg_cum_area` not in ds

**Step 3: Expand `_fetch_vaa()` to include `totdasqkm`**

In `_fetch_vaa()` (~line 1219), change the column selection:

```python
# Old:
result = vaa[["comid", "slope"]].dropna(subset=["slope"])

# New:
cols = ["comid", "slope"]
if "totdasqkm" in vaa.columns:
    cols.append("totdasqkm")
result = vaa[cols].dropna(subset=["slope"])
```

**Step 4: Add cumulative area extraction to COMID path**

In `_get_slopes_from_comid()` (~line 925), expand the return signature to also return cumulative areas. However, to keep changes minimal, instead extract `seg_cum_area` directly in `_derive_routing()` after slopes are obtained.

In `_derive_routing()`, after the slopes block (~line 1376) and before `# --- seg_slope ---`, add:

```python
        # --- seg_cum_area: cumulative drainage area from VAA ---
        _KM2_TO_ACRES = 247.10538146717  # 1 km² = 247.105 acres
        if vaa is not None and "totdasqkm" in vaa.columns:
            if comid_col is not None:
                # COMID path: direct lookup
                cum_area = self._get_cum_area_from_comid(
                    segments, vaa, comid_col
                )
            else:
                # Spatial join path: length-weighted mean
                cum_area = self._get_cum_area_spatial_join(
                    segments, nhd_flowlines, vaa
                )
            ds["seg_cum_area"] = xr.DataArray(
                cum_area * _KM2_TO_ACRES,
                dims="nsegment",
                attrs={
                    "units": "acres",
                    "long_name": "Cumulative drainage area of segment",
                },
            )
        else:
            logger.warning(
                "VAA totdasqkm unavailable; skipping seg_cum_area"
            )
```

Note: The `nhd_flowlines` variable is only available in the spatial-join branch. Restructure slightly: capture `nhd_flowlines` as a local variable set to `None` before the slope if/elif blocks, and set it inside the GF path branch.

**Step 5: Implement `_get_cum_area_from_comid` helper**

Add as a new static method (near `_get_slopes_from_comid`):

```python
@staticmethod
def _get_cum_area_from_comid(
    segments: gpd.GeoDataFrame,
    vaa: pd.DataFrame,
    comid_col: str,
) -> np.ndarray:
    """Look up cumulative drainage area from VAA by COMID.

    Parameters
    ----------
    segments : gpd.GeoDataFrame
        Segment GeoDataFrame with a COMID column.
    vaa : pd.DataFrame
        VAA table with ``comid`` and ``totdasqkm`` columns.
    comid_col : str
        Name of the COMID column in *segments*.

    Returns
    -------
    np.ndarray
        Cumulative drainage area in km² per segment.  Unmatched
        segments get 0.0.
    """
    comids = segments[comid_col].values
    vaa_areas = dict(
        zip(vaa["comid"].values, vaa["totdasqkm"].values, strict=True)
    )
    return np.array(
        [vaa_areas.get(c, 0.0) for c in comids], dtype=np.float64
    )
```

**Step 6: Implement `_get_cum_area_spatial_join` helper**

Add as a new static method (near `_get_slopes_spatial_join`):

```python
@staticmethod
def _get_cum_area_spatial_join(
    segments: gpd.GeoDataFrame,
    nhd_flowlines: gpd.GeoDataFrame | None,
    vaa: pd.DataFrame,
) -> np.ndarray:
    """Get cumulative area via spatial join to NHDPlus flowlines.

    For each segment, find the nearest NHD flowline (by length-weighted
    match in the buffer corridor) and use its ``totdasqkm`` value.

    Parameters
    ----------
    segments : gpd.GeoDataFrame
        Segment GeoDataFrame (no COMID column).
    nhd_flowlines : gpd.GeoDataFrame or None
        NHDPlus flowlines with COMID.  If ``None``, returns zeros.
    vaa : pd.DataFrame
        VAA table with ``comid`` and ``totdasqkm`` columns.

    Returns
    -------
    np.ndarray
        Cumulative drainage area in km² per segment.  Unmatched
        segments get 0.0.
    """
    nseg = len(segments)
    if nhd_flowlines is None:
        return np.zeros(nseg, dtype=np.float64)

    # Ensure same CRS
    if segments.crs != nhd_flowlines.crs:
        nhd_flowlines = nhd_flowlines.to_crs(segments.crs)

    segs = segments.reset_index(drop=True)
    nhd = nhd_flowlines.reset_index(drop=True)

    # Buffer segments into corridors
    seg_buffers = segs.copy()
    seg_buffers["geometry"] = segs.geometry.buffer(_SPATIAL_JOIN_BUFFER_M)

    joined = gpd.sjoin(seg_buffers, nhd, how="left", predicate="intersects")

    # Find COMID column in flowlines
    fl_comid_col = next(
        (c for c in nhd.columns if c.lower() == "comid"), None
    )
    if fl_comid_col is None:
        return np.zeros(nseg, dtype=np.float64)

    # Build COMID -> totdasqkm lookup
    vaa_areas = dict(
        zip(vaa["comid"].values, vaa["totdasqkm"].values, strict=True)
    )

    # For each segment, take max totdasqkm among matched flowlines
    # (cumulative area of the largest matched flowline is the best proxy)
    cum_area = np.zeros(nseg, dtype=np.float64)
    for seg_idx in range(nseg):
        matches = joined[joined.index == seg_idx]
        if matches.empty or matches[fl_comid_col].isna().all():
            continue
        areas = [
            vaa_areas.get(int(c), 0.0)
            for c in matches[fl_comid_col].dropna()
        ]
        if areas:
            cum_area[seg_idx] = max(areas)

    return cum_area
```

**Step 7: Restructure `_derive_routing` to share `nhd_flowlines`**

The current code only fetches `nhd_flowlines` in the GF (no-COMID) branch. We need it accessible for `seg_cum_area` too. Refactor:

Before the slope if/elif block, add:
```python
        nhd_flowlines = None
```

In the GF branch, capture:
```python
        elif vaa is not None:
            nhd_flowlines = self._fetch_nhd_flowlines(segments, vaa)
            if nhd_flowlines is not None:
                slopes = self._get_slopes_spatial_join(segments, nhd_flowlines)
            ...
```

This is already the structure — just ensure `nhd_flowlines` is defined before use.

**Step 8: Add metadata entry**

In `parameter_metadata.yml`:

```yaml
  seg_cum_area:
    dimension: nsegment
    units: acres
    valid_range: [0.0, 1000000000.0]
    required: false
    description: "Cumulative drainage area of segment"
```

**Step 9: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "test_seg_cum_area or TestDeriveRouting" -v`
Expected: PASS (the test will likely see `seg_cum_area` because the mock test lacks VAA access — need to verify the test handles this gracefully, or mock the VAA fetch)

Note: Since `_fetch_vaa()` makes a network call (pynhd), the test above will try to download VAA. For unit testing, either:
- Accept that `seg_cum_area` is skipped (VAA unavailable warning) and test only that it doesn't error, OR
- Mock `_fetch_vaa()` to return a DataFrame with `totdasqkm`

If the existing routing tests already mock VAA, follow that pattern. If they make real network calls (slow/flaky), add a focused unit test that patches `_fetch_vaa`:

```python
def test_seg_cum_area_with_mocked_vaa(
    self,
    derivation: PywatershedDerivation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """seg_cum_area computed from mocked VAA totdasqkm."""
    vaa_df = pd.DataFrame({
        "comid": [1001, 1002],
        "slope": [0.01, 0.02],
        "totdasqkm": [100.0, 500.0],
    })
    monkeypatch.setattr(
        PywatershedDerivation, "_fetch_vaa",
        staticmethod(lambda: vaa_df),
    )
    # Create segments with COMID column for direct lookup
    sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
    fabric = gpd.GeoDataFrame(
        {"nhm_id": [1, 2], "hru_segment": [1, 2]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        {"nhm_seg": [101, 102], "comid": [1001, 1002], "tosegment": [2, 0]},
        geometry=[
            LineString([(0.5, 0.5), (1.0, 0.5)]),
            LineString([(1.0, 0.5), (2.0, 0.5)]),
        ],
        crs="EPSG:4326",
    )
    ctx = DerivationContext(
        sir=sir,
        fabric=fabric,
        segments=segments,
        fabric_id_field="nhm_id",
        segment_id_field="nhm_seg",
    )
    ds = derivation.derive(ctx)
    assert "seg_cum_area" in ds
    _KM2_TO_ACRES = 247.10538146717
    np.testing.assert_allclose(
        ds["seg_cum_area"].values,
        [100.0 * _KM2_TO_ACRES, 500.0 * _KM2_TO_ACRES],
        rtol=1e-5,
    )
```

**Step 10: Run tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -k "seg_cum_area" -v`
Expected: PASS

**Step 11: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py src/hydro_param/data/pywatershed/parameter_metadata.yml tests/test_pywatershed_derivation.py
git commit -m "feat: add seg_cum_area to routing derivation (#159)"
```

---

### Task 3: Add `seg_elev` via InterpGen

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (new `_derive_segment_elevation`)
- Modify: `src/hydro_param/data/pywatershed/parameter_metadata.yml`
- Modify: `pw-check/configs/pywatershed_run.yml` (add optional `dem_path` config)
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing test**

Add a new test class `TestDeriveSegmentElevation`:

```python
class TestDeriveSegmentElevation:
    """Tests for segment elevation via InterpGen."""

    @pytest.fixture()
    def tmp_dem(self, tmp_path: Path) -> Path:
        """Create a small synthetic DEM GeoTIFF for testing."""
        rioxarray = pytest.importorskip("rioxarray")
        dem_data = np.full((10, 10), 500.0, dtype=np.float32)
        # Gradient: row 0 = 500m, row 9 = 590m
        for i in range(10):
            dem_data[i, :] = 500.0 + i * 10.0
        da = xr.DataArray(
            dem_data,
            dims=["y", "x"],
            coords={
                "y": np.linspace(1.0, 0.0, 10),
                "x": np.linspace(0.0, 1.0, 10),
            },
        )
        da = da.rio.set_crs("EPSG:4326")
        da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
        dem_path = tmp_path / "test_dem.tif"
        da.rio.to_raster(str(dem_path))
        return dem_path

    def test_seg_elev_from_dem(
        self,
        derivation: PywatershedDerivation,
        tmp_dem: Path,
    ) -> None:
        """seg_elev computed from DEM via InterpGen."""
        pytest.importorskip("gdptools")
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [101, 102], "tosegment": [2, 0]},
            geometry=[
                LineString([(0.1, 0.5), (0.9, 0.5)]),
                LineString([(0.1, 0.2), (0.9, 0.2)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
            config={"dem_path": str(tmp_dem)},
        )
        ds = derivation.derive(ctx)
        assert "seg_elev" in ds
        assert ds["seg_elev"].dims == ("nsegment",)
        assert ds["seg_elev"].attrs["units"] == "feet"
        # Both segments at approximately mid-elevation of the DEM
        # Values should be positive and in feet (meters * 3.28084)
        assert np.all(ds["seg_elev"].values > 0)

    def test_seg_elev_skipped_without_dem(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """seg_elev skipped gracefully when no dem_path in config."""
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1], "hru_segment": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [101], "tosegment": [0]},
            geometry=[LineString([(0.1, 0.5), (0.9, 0.5)])],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "seg_elev" not in ds
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSegmentElevation -v`
Expected: FAIL — test class doesn't exist yet (after adding tests, `seg_elev` not in ds)

**Step 3: Implement `_derive_segment_elevation()`**

Add new method in `pywatershed.py` (after `_derive_topography`, before step 4):

```python
    def _derive_segment_elevation(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Derive mean segment elevation from a DEM raster (step 3b).

        Sample a 3DEP DEM along each segment polyline using gdptools
        ``InterpGen`` (grid-to-line interpolation) and compute the mean
        elevation.  Convert from meters to feet.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context.  Must have ``segments`` and a
            ``dem_path`` key in ``config`` pointing to a local
            GeoTIFF DEM.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with ``seg_elev`` (feet) on ``nsegment``, or
            ``ds`` unchanged if DEM path is unavailable or segments
            are ``None``.

        Notes
        -----
        Step 3b of the derivation DAG (runs after step 3, before step 4).
        The DEM path is provided via ``config["dem_path"]``.  When absent,
        this step is skipped with a debug log message.

        ``InterpGen`` samples the raster at 50 m intervals along each
        segment polyline and returns per-segment statistics.  The
        ``"mean"`` statistic gives the average elevation along the
        stream channel.

        References
        ----------
        gdptools InterpGen: grid-to-line interpolation for polyline
        geometries.
        """
        segments = ctx.segments
        dem_path = ctx.config.get("dem_path")

        if segments is None or dem_path is None:
            if dem_path is None:
                logger.debug(
                    "No dem_path in config; skipping seg_elev derivation"
                )
            return ds

        dem_path = Path(dem_path)
        if not dem_path.exists():
            logger.warning("DEM path %s does not exist; skipping seg_elev", dem_path)
            return ds

        try:
            from gdptools import InterpGen, UserTiffData
        except ImportError:
            logger.warning("gdptools not available; skipping seg_elev")
            return ds

        segment_id_field = ctx.segment_id_field or "nhm_seg"

        # Prepare segments in geographic CRS for InterpGen
        if segments.crs is not None and not segments.crs.is_geographic:
            segs_geo = segments.to_crs(epsg=4326)
        else:
            segs_geo = segments

        # Open DEM to get metadata
        import rioxarray  # noqa: F401

        dem_ds = xr.open_dataset(dem_path, engine="rasterio")
        # Identify the data variable (usually 'band_data' or first var)
        dem_var = list(dem_ds.data_vars)[0]
        dem_da = dem_ds[dem_var]

        # Determine coordinate names
        x_coord = "x" if "x" in dem_da.coords else "longitude"
        y_coord = "y" if "y" in dem_da.coords else "latitude"

        # Determine CRS from the DEM
        if hasattr(dem_da, "rio") and dem_da.rio.crs is not None:
            dem_crs = dem_da.rio.crs.to_epsg() or 4326
        else:
            dem_crs = 4326

        # Build segment ID column for InterpGen target_id
        if segment_id_field in segs_geo.columns:
            target_id = segment_id_field
        else:
            segs_geo = segs_geo.copy()
            segs_geo["_seg_idx"] = range(len(segs_geo))
            target_id = "_seg_idx"

        user_data = UserTiffData(
            f=str(dem_path),
            var=dem_var,
            x_coord=x_coord,
            y_coord=y_coord,
            crs=dem_crs,
            t_gdf=segs_geo,
            t_id=target_id,
        )

        interp = InterpGen(
            user_data,
            pt_spacing=50,
            stat="mean",
            interp_method="linear",
        )

        stats_df = interp.calc_interp()

        # Extract mean elevation per segment, maintaining order
        _M_TO_FT = 3.28084
        nseg = len(segments)
        seg_elev = np.full(nseg, np.nan, dtype=np.float64)

        if segment_id_field in segs_geo.columns:
            seg_id_order = segs_geo[segment_id_field].values
            for i, sid in enumerate(seg_id_order):
                row = stats_df[stats_df["line_id"] == sid]
                if not row.empty:
                    seg_elev[i] = row["mean"].iloc[0] * _M_TO_FT
        else:
            for i in range(nseg):
                row = stats_df[stats_df["line_id"] == i]
                if not row.empty:
                    seg_elev[i] = row["mean"].iloc[0] * _M_TO_FT

        # Fill NaN with 0 and warn
        nan_count = int(np.isnan(seg_elev).sum())
        if nan_count > 0:
            logger.warning(
                "%d of %d segments have no DEM elevation; using 0.0 feet",
                nan_count,
                nseg,
            )
            seg_elev = np.nan_to_num(seg_elev, nan=0.0)

        ds["seg_elev"] = xr.DataArray(
            seg_elev,
            dims="nsegment",
            attrs={"units": "feet", "long_name": "Mean segment channel elevation"},
        )

        return ds
```

**Step 4: Wire into `derive()` method**

In the `derive()` method, after Step 3 (topography) and before Step 4 (landcover):

```python
        # Step 3b: Segment elevation (InterpGen + 3DEP DEM)
        ds = self._derive_segment_elevation(context, ds)
```

Update the module docstring step list to include "3b. Segment elevation".

Update the `derive()` Notes docstring to include step 3b in the execution order.

**Step 5: Add metadata entry**

In `parameter_metadata.yml`:

```yaml
  seg_elev:
    dimension: nsegment
    units: feet
    valid_range: [-1000.0, 30000.0]
    required: false
    description: "Mean segment channel elevation"
```

**Step 6: Add `dem_path` to example config**

In `pw-check/configs/pywatershed_run.yml`, in the `static_datasets.topography` section, add a comment about `dem_path`:

```yaml
  topography:
    available: [dem_3dep_10m]
    # Optional: path to local DEM GeoTIFF for segment elevation derivation
    # via gdptools InterpGen. If omitted, seg_elev is skipped.
    # dem_path: "data/rasters/dem_3dep_10m.tif"
    hru_elev:
      ...
```

**Step 7: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveSegmentElevation -v`
Expected: PASS

**Step 8: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py src/hydro_param/data/pywatershed/parameter_metadata.yml tests/test_pywatershed_derivation.py pw-check/configs/pywatershed_run.yml
git commit -m "feat: add seg_elev via InterpGen DEM interpolation (#159)"
```

---

### Task 4: Run full test suite and verify

**Step 1: Run all checks**

Run: `pixi run -e dev check`
Expected: All tests pass, no lint/type errors

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Final commit if any formatting changes**

If ruff/mypy required fixes, commit them:
```bash
git commit -m "chore: fix lint/format from pre-commit (#159)"
```
