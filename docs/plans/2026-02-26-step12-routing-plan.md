# Step 12: Routing Coefficients — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Derive Muskingum routing parameters (K_coef, x_coef, seg_slope, segment_type, obsin_segment) for stream segments, handling both NHD flowlines (with COMIDs) and post-processed GF/PRMS segments (spatial join to NHDPlus).

**Architecture:** A new `_derive_routing()` method on `PywatershedDerivation` in `src/hydro_param/derivations/pywatershed.py`.  It detects segment type by column presence (COMID → direct VAA lookup; no COMID → spatial join to NHDPlus flowlines for slope), computes K_coef via Manning's equation, and assigns defaults for x_coef, segment_type, and obsin_segment.  Follows the existing step-method pattern (Steps 1–14).

**Tech Stack:** geopandas (spatial join), pynhd (NHDPlus flowlines + VAA), numpy, xarray, pyproj

**Design doc:** `docs/plans/2026-02-26-step12-routing-design.md`

---

### Task 1: Add routing constants and helper — slope fetching

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (add constants near line 108, after `_IMPERV_STOR_MAX_DEFAULT`)
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing tests for `_get_slopes_from_comid`**

Add a new test class `TestRoutingSlopes` with a test for the NHD (COMID) path.
This helper takes a segments GeoDataFrame with a `comid` column and a VAA
DataFrame with `comid` and `slope` columns, and returns a slope array aligned
to segments.

```python
class TestRoutingSlopes:
    """Tests for Step 12 slope fetching helpers."""

    def test_slopes_from_comid_direct(self, derivation):
        """NHD segments with COMID get slope directly from VAA."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 102, 103],
                "tosegment": [2, 3, 0],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
                LineString([(2, 0), (3, 0)]),
            ],
            crs="EPSG:4326",
        )
        vaa = pd.DataFrame({
            "comid": [101, 102, 103, 999],
            "slope": [0.01, 0.005, 0.001, 0.1],
        })
        slopes = derivation._get_slopes_from_comid(segments, vaa)
        np.testing.assert_array_almost_equal(slopes, [0.01, 0.005, 0.001])

    def test_slopes_from_comid_missing_uses_fallback(self, derivation):
        """COMIDs not in VAA get fallback slope."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 999],
                "tosegment": [2, 0],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        vaa = pd.DataFrame({"comid": [101], "slope": [0.01]})
        slopes = derivation._get_slopes_from_comid(segments, vaa)
        np.testing.assert_array_almost_equal(slopes, [0.01, 1e-4])

    def test_slopes_from_comid_case_insensitive(self, derivation):
        """Column name 'COMID' (upper case) also works."""
        segments = gpd.GeoDataFrame(
            {
                "COMID": [101],
                "tosegment": [0],
            },
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        vaa = pd.DataFrame({"comid": [101], "slope": [0.02]})
        slopes = derivation._get_slopes_from_comid(segments, vaa)
        np.testing.assert_array_almost_equal(slopes, [0.02])
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestRoutingSlopes -v`
Expected: FAIL — `_get_slopes_from_comid` does not exist

**Step 3: Add routing constants and implement `_get_slopes_from_comid`**

Add constants after `_IMPERV_STOR_MAX_DEFAULT` (around line 108):

```python
# Routing constants (Step 12)
_MANNING_N = 0.04           # natural channel roughness coefficient
_DEFAULT_DEPTH_FT = 1.0     # bankfull depth placeholder (feet)
_MIN_SLOPE = 1e-7           # pywatershed floor for seg_slope
_FALLBACK_SLOPE = 1e-4      # for segments with no NHDPlus match
_K_COEF_MIN = 0.01          # hours
_K_COEF_MAX = 24.0          # hours
_DEFAULT_K_COEF = 1.0       # hours, when computation not possible
_DEFAULT_X_COEF = 0.2       # standard Muskingum weighting
_LAKE_K_COEF = 24.0         # travel time for lake segments
_LAKE_SEGMENT_TYPE = 1      # segment_type value for lake
_CHANNEL_SEGMENT_TYPE = 0   # segment_type value for channel
```

Add `import pandas as pd` to the imports at the top of the file.

Add method to `PywatershedDerivation` class (after `_validate_hru_segment`,
before the Step 3 section):

```python
@staticmethod
def _get_slopes_from_comid(
    segments: gpd.GeoDataFrame,
    vaa: pd.DataFrame,
) -> np.ndarray:
    """Look up NHDPlus VAA slope by COMID (direct join).

    Parameters
    ----------
    segments : gpd.GeoDataFrame
        Segment GeoDataFrame with ``comid`` or ``COMID`` column.
    vaa : pd.DataFrame
        NHDPlus VAA table with ``comid`` and ``slope`` columns.

    Returns
    -------
    np.ndarray
        Slope values aligned to segment order.  Segments with no
        matching COMID in the VAA get ``_FALLBACK_SLOPE``.
    """
    # Normalize column name
    seg_comid_col = "comid" if "comid" in segments.columns else "COMID"
    comids = segments[seg_comid_col].values

    # Build lookup from VAA
    vaa_slopes = dict(zip(vaa["comid"].values, vaa["slope"].values))

    slopes = np.array([
        vaa_slopes.get(c, _FALLBACK_SLOPE) for c in comids
    ], dtype=np.float64)

    n_missing = np.sum(slopes == _FALLBACK_SLOPE)
    if n_missing > 0:
        logger.warning(
            "%d of %d segments have no matching COMID in VAA; "
            "using fallback slope %.1e",
            n_missing, len(comids), _FALLBACK_SLOPE,
        )
    return slopes
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestRoutingSlopes -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat(routing): add routing constants and COMID slope lookup helper"
```

---

### Task 2: Spatial join slope helper for GF segments

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing tests for `_get_slopes_spatial_join`**

This helper takes GF segments (no COMID) and NHDPlus flowlines (with slope
attribute), performs a spatial join, and returns length-weighted mean slope
per segment.

```python
def test_slopes_spatial_join_basic(self, derivation):
    """GF segments get length-weighted slope from intersecting NHD flowlines."""
    # Two GF segments, each overlapping one NHD flowline exactly
    gf_segments = gpd.GeoDataFrame(
        {
            "nhm_seg": [1, 2],
            "tosegment": [2, 0],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
        ],
        crs="EPSG:4326",
    )
    nhd_flowlines = gpd.GeoDataFrame(
        {
            "comid": [101, 102],
            "slope": [0.01, 0.005],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
        ],
        crs="EPSG:4326",
    )
    slopes = derivation._get_slopes_spatial_join(gf_segments, nhd_flowlines)
    np.testing.assert_array_almost_equal(slopes, [0.01, 0.005])

def test_slopes_spatial_join_no_match_uses_fallback(self, derivation):
    """GF segments with no NHD match get fallback slope."""
    gf_segments = gpd.GeoDataFrame(
        {
            "nhm_seg": [1],
            "tosegment": [0],
        },
        geometry=[LineString([(100, 100), (101, 100)])],
        crs="EPSG:4326",
    )
    nhd_flowlines = gpd.GeoDataFrame(
        {
            "comid": [101],
            "slope": [0.01],
        },
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:4326",
    )
    slopes = derivation._get_slopes_spatial_join(gf_segments, nhd_flowlines)
    np.testing.assert_array_almost_equal(slopes, [_FALLBACK_SLOPE])

def test_slopes_spatial_join_multiple_nhd_per_segment(self, derivation):
    """GF segment overlapping two NHD flowlines gets length-weighted slope."""
    # One GF segment spans two NHD flowlines with different slopes
    gf_segments = gpd.GeoDataFrame(
        {
            "nhm_seg": [1],
            "tosegment": [0],
        },
        geometry=[LineString([(0, 0), (2, 0)])],
        crs="EPSG:4326",
    )
    nhd_flowlines = gpd.GeoDataFrame(
        {
            "comid": [101, 102],
            "slope": [0.01, 0.03],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),   # equal length
            LineString([(1, 0), (2, 0)]),    # equal length
        ],
        crs="EPSG:4326",
    )
    slopes = derivation._get_slopes_spatial_join(gf_segments, nhd_flowlines)
    # Equal length segments → simple average: (0.01 + 0.03) / 2 = 0.02
    np.testing.assert_array_almost_equal(slopes, [0.02], decimal=3)
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestRoutingSlopes::test_slopes_spatial_join_basic -v`
Expected: FAIL — `_get_slopes_spatial_join` does not exist

**Step 3: Implement `_get_slopes_spatial_join`**

```python
@staticmethod
def _get_slopes_spatial_join(
    segments: gpd.GeoDataFrame,
    nhd_flowlines: gpd.GeoDataFrame,
) -> np.ndarray:
    """Get slopes via spatial join to NHDPlus flowlines (length-weighted).

    For each segment, find all NHDPlus flowlines that intersect it,
    compute the intersection length, and return a length-weighted
    mean slope.

    Parameters
    ----------
    segments : gpd.GeoDataFrame
        Segment GeoDataFrame (no COMID column).
    nhd_flowlines : gpd.GeoDataFrame
        NHDPlus flowlines with ``slope`` column and line geometries.

    Returns
    -------
    np.ndarray
        Length-weighted mean slope per segment.  Segments with no
        NHD match get ``_FALLBACK_SLOPE``.
    """
    # Ensure same CRS
    if segments.crs != nhd_flowlines.crs:
        nhd_flowlines = nhd_flowlines.to_crs(segments.crs)

    # Reset index to use positional alignment
    segs = segments.reset_index(drop=True)
    nhd = nhd_flowlines.reset_index(drop=True)

    # Spatial join — find all NHD flowlines intersecting each segment
    joined = gpd.sjoin(segs, nhd, how="left", predicate="intersects")

    slopes = np.full(len(segs), _FALLBACK_SLOPE, dtype=np.float64)

    for seg_idx in range(len(segs)):
        matches = joined[joined.index == seg_idx]
        matches = matches.dropna(subset=["slope"])
        if matches.empty:
            continue

        # Compute intersection lengths for weighting
        seg_geom = segs.geometry.iloc[seg_idx]
        weights = []
        match_slopes = []
        for _, row in matches.iterrows():
            nhd_idx = int(row["index_right"])
            nhd_geom = nhd.geometry.iloc[nhd_idx]
            intersection = seg_geom.intersection(nhd_geom)
            if intersection.is_empty:
                continue
            weights.append(intersection.length)
            match_slopes.append(row["slope"])

        if weights:
            weights = np.array(weights)
            match_slopes = np.array(match_slopes)
            total_weight = weights.sum()
            if total_weight > 0:
                slopes[seg_idx] = np.average(match_slopes, weights=weights)

    n_fallback = np.sum(slopes == _FALLBACK_SLOPE)
    if n_fallback > 0:
        logger.warning(
            "%d of %d segments have no intersecting NHDPlus flowlines; "
            "using fallback slope %.1e",
            n_fallback, len(segs), _FALLBACK_SLOPE,
        )
    return slopes
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestRoutingSlopes -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat(routing): add spatial join slope helper for GF segments"
```

---

### Task 3: Manning's equation K_coef computation

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing tests for `_compute_k_coef`**

This static method takes slope and seg_length arrays and returns K_coef
using Manning's equation.

```python
class TestManningKCoef:
    """Tests for K_coef computation via Manning's equation."""

    def test_basic_k_coef(self, derivation):
        """K_coef computed from Manning's equation."""
        slopes = np.array([0.01])
        lengths = np.array([10000.0])  # 10 km in meters
        # velocity = (1/0.04) * sqrt(0.01) * 1.0^(2/3) * 3600
        # velocity = 25 * 0.1 * 1.0 * 3600 = 9000 ft/hr
        # seg_length_ft = 10000 * 3.28084 = 32808.4 ft
        # K_coef = 32808.4 / 9000 = 3.645 hours
        k_coef = derivation._compute_k_coef(slopes, lengths)
        assert k_coef[0] == pytest.approx(3.645, abs=0.01)

    def test_k_coef_clamped_max(self, derivation):
        """K_coef > 24 is clamped to 24."""
        # Very low slope + long segment → large K
        slopes = np.array([1e-7])
        lengths = np.array([100000.0])
        k_coef = derivation._compute_k_coef(slopes, lengths)
        assert k_coef[0] == _K_COEF_MAX

    def test_k_coef_clamped_min(self, derivation):
        """K_coef < 0.01 is clamped to 0.01."""
        # Very steep + short segment → tiny K
        slopes = np.array([1.0])
        lengths = np.array([1.0])
        k_coef = derivation._compute_k_coef(slopes, lengths)
        assert k_coef[0] == _K_COEF_MIN

    def test_slope_floor_applied(self, derivation):
        """Slopes below 1e-7 are clamped to 1e-7."""
        slopes = np.array([0.0, -0.001])
        lengths = np.array([10000.0, 10000.0])
        k_coef = derivation._compute_k_coef(slopes, lengths)
        # Both should produce same result (clamped to 1e-7)
        assert k_coef[0] == k_coef[1]
        assert k_coef[0] <= _K_COEF_MAX

    def test_zero_length_gets_default(self, derivation):
        """Zero-length segments get default K_coef."""
        slopes = np.array([0.01])
        lengths = np.array([0.0])
        k_coef = derivation._compute_k_coef(slopes, lengths)
        assert k_coef[0] == _DEFAULT_K_COEF
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestManningKCoef -v`
Expected: FAIL — `_compute_k_coef` does not exist

**Step 3: Implement `_compute_k_coef`**

```python
@staticmethod
def _compute_k_coef(
    slopes: np.ndarray,
    seg_lengths_m: np.ndarray,
) -> np.ndarray:
    """Compute Muskingum K_coef via Manning's equation.

    Parameters
    ----------
    slopes : np.ndarray
        Channel slope (m/m) per segment.
    seg_lengths_m : np.ndarray
        Segment lengths in meters.

    Returns
    -------
    np.ndarray
        K_coef in hours, clamped to ``[_K_COEF_MIN, _K_COEF_MAX]``.
        Zero-length segments receive ``_DEFAULT_K_COEF``.
    """
    k_coef = np.full(len(slopes), _DEFAULT_K_COEF, dtype=np.float64)

    # Mask valid segments (nonzero length)
    valid = seg_lengths_m > 0
    if not np.any(valid):
        return k_coef

    # Clamp slopes
    s = np.clip(slopes[valid], _MIN_SLOPE, None)

    # Manning's equation: velocity in ft/hr
    # velocity = (1/n) * sqrt(slope) * depth^(2/3) * 3600
    velocity = (1.0 / _MANNING_N) * np.sqrt(s) * (_DEFAULT_DEPTH_FT ** (2.0 / 3.0)) * 3600.0

    # Convert segment length from meters to feet
    seg_length_ft = seg_lengths_m[valid] * 3.28084

    # K = length / velocity (hours)
    k_coef[valid] = seg_length_ft / velocity

    # Clamp to valid range
    k_coef = np.clip(k_coef, _K_COEF_MIN, _K_COEF_MAX)

    return k_coef
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestManningKCoef -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat(routing): add Manning's equation K_coef computation"
```

---

### Task 4: Detect segment type and fetch NHDPlus VAA

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing tests for `_has_comid` detection and `_fetch_nhd_slopes`**

```python
class TestSegmentTypeDetection:
    """Tests for NHD vs GF segment detection."""

    def test_has_comid_lowercase(self, derivation):
        """Segments with 'comid' column detected as NHD."""
        segments = gpd.GeoDataFrame(
            {"comid": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        assert derivation._has_comid(segments) is True

    def test_has_comid_uppercase(self, derivation):
        """Segments with 'COMID' column detected as NHD."""
        segments = gpd.GeoDataFrame(
            {"COMID": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        assert derivation._has_comid(segments) is True

    def test_no_comid(self, derivation):
        """Segments without COMID column detected as GF."""
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        assert derivation._has_comid(segments) is False
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestSegmentTypeDetection -v`
Expected: FAIL — `_has_comid` does not exist

**Step 3: Implement `_has_comid` and `_fetch_nhd_slopes`**

```python
@staticmethod
def _has_comid(segments: gpd.GeoDataFrame) -> bool:
    """Check whether segments carry a COMID column (NHD flowlines).

    Parameters
    ----------
    segments : gpd.GeoDataFrame
        Segment GeoDataFrame to inspect.

    Returns
    -------
    bool
        ``True`` if ``comid`` or ``COMID`` column exists.
    """
    cols_lower = {c.lower() for c in segments.columns}
    return "comid" in cols_lower

@staticmethod
def _fetch_nhd_slopes(
    segments: gpd.GeoDataFrame,
) -> tuple[pd.DataFrame | None, gpd.GeoDataFrame | None]:
    """Fetch NHDPlus VAA slopes and flowlines for the segment extent.

    Downloads NHDPlus VAA (cached 245 MB parquet) for slope values.
    For the spatial join path, also fetches NHDPlus flowline geometries
    within the segment bounding box via pynhd WaterData service.

    Parameters
    ----------
    segments : gpd.GeoDataFrame
        Segment GeoDataFrame used to determine bounding box.

    Returns
    -------
    tuple[pd.DataFrame | None, gpd.GeoDataFrame | None]
        (vaa_df, nhd_flowlines) — VAA table with ``comid`` and ``slope``
        columns, and NHDPlus flowline GeoDataFrame (None if COMID path
        is used and flowlines are not needed).  Returns ``(None, None)``
        if the fetch fails.
    """
    pynhd = pytest.importorskip("pynhd") if False else None  # noqa: never runs
    try:
        import pynhd as nhd
    except ImportError:
        logger.warning("pynhd not installed; cannot fetch NHDPlus slopes")
        return None, None

    try:
        # Fetch VAA (cached parquet, ~245 MB first download)
        vaa = nhd.nhdplus_vaa()
        vaa = vaa[["comid", "slope"]].dropna(subset=["slope"])

        # Fetch flowlines for spatial join (only needed for GF path)
        bbox = segments.to_crs("EPSG:4326").total_bounds
        wd = nhd.WaterData("nhdflowline_network")
        flowlines = wd.bybox(tuple(bbox))

        return vaa, flowlines

    except Exception:
        logger.exception("Failed to fetch NHDPlus data")
        return None, None
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestSegmentTypeDetection -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat(routing): add segment type detection and NHD fetch helper"
```

---

### Task 5: Main `_derive_routing` method

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py` (add method + wire into `derive()`)
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write failing tests for `_derive_routing`**

Test the full orchestration with synthetic data (no network calls).
We mock `_fetch_nhd_slopes` to avoid network dependencies in unit tests.

```python
class TestDeriveRouting:
    """Tests for Step 12: _derive_routing orchestration."""

    def test_routing_no_segments_returns_defaults(self, derivation):
        """No segments → warn and return defaults."""
        sir = xr.Dataset(coords={"nhm_id": [1, 2, 3]})
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir["nhm_id"].values)
        ds = derivation._derive_routing(ctx, ds)
        # Should not have routing params (no nsegment dimension)
        assert "K_coef" not in ds
        assert "x_coef" not in ds

    def test_routing_with_comid_segments(self, derivation, monkeypatch):
        """NHD segments with COMID produce K_coef from VAA slope."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 102],
                "tosegment": [2, 0],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )

        vaa = pd.DataFrame({"comid": [101, 102], "slope": [0.01, 0.005]})

        monkeypatch.setattr(
            derivation, "_fetch_nhd_slopes",
            staticmethod(lambda segs: (vaa, None)),
        )

        ctx = DerivationContext(
            sir=sir, fabric=fabric, segments=segments,
            fabric_id_field="nhm_id", segment_id_field="comid",
        )
        ds = derivation.derive(ctx)

        assert "K_coef" in ds
        assert "x_coef" in ds
        assert "seg_slope" in ds
        assert "segment_type" in ds
        assert "obsin_segment" in ds
        assert ds["K_coef"].dims == ("nsegment",)
        assert np.all(ds["K_coef"].values > 0)
        assert np.all(ds["K_coef"].values <= 24.0)
        np.testing.assert_array_equal(ds["x_coef"].values, [0.2, 0.2])
        np.testing.assert_array_equal(ds["segment_type"].values, [0, 0])
        np.testing.assert_array_equal(ds["obsin_segment"].values, [0, 0])

    def test_routing_gf_segments_spatial_join(self, derivation, monkeypatch):
        """GF segments without COMID use spatial join for slopes."""
        segments = gpd.GeoDataFrame(
            {
                "nhm_seg": [1, 2],
                "tosegment": [2, 0],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "comid": [101, 102],
                "slope": [0.01, 0.005],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )

        vaa = pd.DataFrame({"comid": [101, 102], "slope": [0.01, 0.005]})

        monkeypatch.setattr(
            derivation, "_fetch_nhd_slopes",
            staticmethod(lambda segs: (vaa, nhd_flowlines)),
        )

        ctx = DerivationContext(
            sir=sir, fabric=fabric, segments=segments,
            fabric_id_field="nhm_id", segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)

        assert "K_coef" in ds
        assert "seg_slope" in ds
        np.testing.assert_array_almost_equal(
            ds["seg_slope"].values, [0.01, 0.005]
        )

    def test_routing_segment_type_passthrough(self, derivation, monkeypatch):
        """segment_type column in segments GeoDataFrame is passed through."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 102],
                "tosegment": [2, 0],
                "segment_type": [0, 1],  # channel, lake
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )

        vaa = pd.DataFrame({"comid": [101, 102], "slope": [0.01, 0.005]})
        monkeypatch.setattr(
            derivation, "_fetch_nhd_slopes",
            staticmethod(lambda segs: (vaa, None)),
        )

        ctx = DerivationContext(
            sir=sir, fabric=fabric, segments=segments,
            fabric_id_field="nhm_id", segment_id_field="comid",
        )
        ds = derivation.derive(ctx)

        np.testing.assert_array_equal(ds["segment_type"].values, [0, 1])
        # Lake segment should have K_coef = 24.0
        assert ds["K_coef"].values[1] == 24.0

    def test_routing_fetch_failure_uses_fallbacks(self, derivation, monkeypatch):
        """Failed NHD fetch produces fallback slopes and default K_coef."""
        segments = gpd.GeoDataFrame(
            {
                "nhm_seg": [1, 2],
                "tosegment": [2, 0],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )

        monkeypatch.setattr(
            derivation, "_fetch_nhd_slopes",
            staticmethod(lambda segs: (None, None)),
        )

        ctx = DerivationContext(
            sir=sir, fabric=fabric, segments=segments,
            fabric_id_field="nhm_id", segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)

        assert "K_coef" in ds
        assert "seg_slope" in ds
        # All fallback slopes
        np.testing.assert_array_equal(
            ds["seg_slope"].values, [_FALLBACK_SLOPE, _FALLBACK_SLOPE]
        )
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveRouting -v`
Expected: FAIL — `_derive_routing` does not exist

**Step 3: Implement `_derive_routing` and wire into `derive()`**

Add the method after the Step 8 section (lookup tables) and before Step 9
(soltab).  The exact insertion point is after `_apply_lookup_tables` and
its helpers:

```python
# ------------------------------------------------------------------
# Step 12: Routing coefficients
# ------------------------------------------------------------------

def _derive_routing(
    self,
    ctx: DerivationContext,
    ds: xr.Dataset,
) -> xr.Dataset:
    """Derive Muskingum routing parameters from channel geometry (step 12).

    Compute ``K_coef`` (travel time) via Manning's equation using
    NHDPlus VAA slopes, and assign ``x_coef``, ``seg_slope``,
    ``segment_type``, and ``obsin_segment``.

    Supports two segment types:

    - **NHD segments** (have ``comid``/``COMID`` column): direct
      COMID-to-VAA slope lookup.
    - **GF/PRMS segments** (no COMID): spatial join to NHDPlus
      flowlines with length-weighted slope averaging.

    Parameters
    ----------
    ctx : DerivationContext
        Derivation context providing ``segments`` GeoDataFrame.
    ds : xr.Dataset
        In-progress parameter dataset (must already contain
        ``seg_length`` on ``nsegment`` from Step 2).

    Returns
    -------
    xr.Dataset
        Dataset with ``K_coef`` (hours), ``x_coef`` (dimensionless),
        ``seg_slope`` (m/m), ``segment_type`` (integer), and
        ``obsin_segment`` (integer) added on ``nsegment``.
        Returns ``ds`` unchanged if ``ctx.segments`` is ``None``.
    """
    segments = ctx.segments
    if segments is None:
        logger.warning("No segments provided; skipping routing derivation")
        return ds

    nseg = len(segments)
    if "nsegment" not in ds.dims:
        logger.warning("nsegment dimension not in dataset; skipping routing")
        return ds

    # --- Fetch NHDPlus slopes ---
    vaa, nhd_flowlines = self._fetch_nhd_slopes(segments)

    if vaa is not None and self._has_comid(segments):
        # NHD path: direct COMID lookup
        slopes = self._get_slopes_from_comid(segments, vaa)
    elif vaa is not None and nhd_flowlines is not None:
        # GF path: spatial join
        slopes = self._get_slopes_spatial_join(segments, nhd_flowlines)
    else:
        # Fetch failed — use fallback slopes everywhere
        logger.warning(
            "NHDPlus data unavailable; using fallback slope %.1e for all %d segments",
            _FALLBACK_SLOPE, nseg,
        )
        slopes = np.full(nseg, _FALLBACK_SLOPE, dtype=np.float64)

    # --- seg_slope ---
    ds["seg_slope"] = xr.DataArray(
        slopes,
        dims="nsegment",
        attrs={"units": "m/m", "long_name": "Channel slope"},
    )

    # --- K_coef ---
    seg_lengths = ds["seg_length"].values if "seg_length" in ds else np.zeros(nseg)
    k_coef = self._compute_k_coef(slopes, seg_lengths)

    # --- segment_type ---
    if "segment_type" in segments.columns:
        seg_type = segments["segment_type"].values.astype(np.int32)
    else:
        seg_type = np.full(nseg, _CHANNEL_SEGMENT_TYPE, dtype=np.int32)

    # Force lake segments to max K_coef
    lake_mask = seg_type == _LAKE_SEGMENT_TYPE
    k_coef[lake_mask] = _LAKE_K_COEF

    ds["K_coef"] = xr.DataArray(
        k_coef,
        dims="nsegment",
        attrs={"units": "hours", "long_name": "Muskingum storage time coefficient"},
    )

    # --- x_coef ---
    ds["x_coef"] = xr.DataArray(
        np.full(nseg, _DEFAULT_X_COEF, dtype=np.float64),
        dims="nsegment",
        attrs={"units": "none", "long_name": "Muskingum routing weighting factor"},
    )

    # --- segment_type ---
    ds["segment_type"] = xr.DataArray(
        seg_type,
        dims="nsegment",
        attrs={"units": "none", "long_name": "Segment type (0=channel, 1=lake)"},
    )

    # --- obsin_segment ---
    ds["obsin_segment"] = xr.DataArray(
        np.zeros(nseg, dtype=np.int32),
        dims="nsegment",
        attrs={"units": "none", "long_name": "Observed inflow segment (0=none)"},
    )

    return ds
```

Wire into `derive()` — add after Step 8 (lookup tables) and before Step 9
(soltab).  Find the line:

```python
        # Step 9: Solar radiation tables (soltab)
```

Insert before it:

```python
        # Step 12: Routing coefficients (K_coef, x_coef, seg_slope)
        ds = self._derive_routing(context, ds)
```

Also update the module docstring to list Step 12 as implemented, and update
the `derive()` docstring step order note.

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveRouting -v`
Expected: PASS (5 tests)

**Step 5: Run full test suite**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py tests/test_pywatershed_derivation.py
git commit -m "feat(routing): implement _derive_routing with NHD/GF fork and Manning K_coef"
```

---

### Task 6: Remove x_coef from _DEFAULTS and update docstrings

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`

**Step 1: Remove `"x_coef": 0.2` from `_DEFAULTS` dict**

Now that Step 12 sets x_coef explicitly as an nsegment-dimensioned array,
the scalar default in `_DEFAULTS` is no longer needed.  The `_apply_defaults`
loop would overwrite Step 12's array with a scalar if Step 12 didn't run
(no segments), but that case is handled by the guard in `_derive_routing`.

Find line 82:

```python
    "x_coef": 0.2,
```

Remove it.

**Step 2: Update module docstring**

Replace the line:

```
Step 12 (routing parameters) is not yet implemented.
```

with:

```
12. Routing --- Muskingum K_coef, x_coef, seg_slope from NHDPlus VAA
```

**Step 3: Update `derive()` step order note**

Find the line:

```python
        Step execution order: 1 (geometry) -> 2 (topology) -> 3 (topo) ->
        4 (landcover) -> 5 (soils) -> 6 (waterbody) -> 8 (lookups) ->
        9 (soltab) -> 10 (PET) -> 11 (transp) -> 13 (defaults) ->
        14 (calibration) -> 7 (forcing) -> overrides.
```

Replace with:

```python
        Step execution order: 1 (geometry) -> 2 (topology) -> 3 (topo) ->
        4 (landcover) -> 5 (soils) -> 6 (waterbody) -> 8 (lookups) ->
        12 (routing) -> 9 (soltab) -> 10 (PET) -> 11 (transp) ->
        13 (defaults) -> 14 (calibration) -> 7 (forcing) -> overrides.
```

**Step 4: Run full test suite**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py -v`
Expected: All tests pass (x_coef now set by Step 12 when segments present,
not needed in _DEFAULTS since tests without segments don't check for x_coef)

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py
git commit -m "refactor(routing): remove x_coef from _DEFAULTS, update docstrings for step 12"
```

---

### Task 7: Run full checks and verify

**Files:**
- No new code changes

**Step 1: Run lint and format**

Run: `pixi run -e dev format && pixi run -e dev lint`
Expected: Clean

**Step 2: Run type check**

Run: `pixi run -e dev typecheck`
Expected: Clean (may need `pd.DataFrame` type annotations adjusted)

**Step 3: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass

**Step 4: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 5: Run all checks together**

Run: `pixi run -e dev check`
Expected: All green

**Step 6: Final commit if any formatting/lint fixes needed**

```bash
git add -u
git commit -m "chore: lint and format fixes for step 12 routing"
```
