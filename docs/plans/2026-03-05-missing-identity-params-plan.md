# Add Missing Topology/Identity Parameters (#158) — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `hru_lon`, `nhm_id`, `nhm_seg`, and `hru_segment_nhm` parameters to pywatershed derivation.

**Architecture:** `hru_lon` extends `_derive_geometry()` using the WGS84 centroids already computed for `hru_lat`. The three identity/topology params extend `_derive_topology()` using config-driven `id_field` and `segment_id_field` (not hardcoded column names). All four get metadata entries.

**Tech Stack:** xarray, geopandas, numpy, pytest

---

### Task 1: Add `hru_lon` to geometry derivation

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:479-553` (`_derive_geometry`)
- Modify: `src/hydro_param/data/pywatershed/parameter_metadata.yml:39-52`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing tests**

Add to `TestDeriveGeometryFromFabric` class after `test_lat_from_fabric` (~line 1652):

```python
def test_lon_from_fabric(self, derivation: PywatershedDerivation) -> None:
    """hru_lon computed from fabric centroid longitude."""
    sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
    fabric = gpd.GeoDataFrame(
        {"nhm_id": [1, 2]},
        geometry=[
            Polygon([(0, 40), (1, 40), (1, 41), (0, 41)]),
            Polygon([(2, 42), (3, 42), (3, 43), (2, 43)]),
        ],
        crs="EPSG:4326",
    )
    ctx = DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    assert "hru_lon" in ds
    np.testing.assert_allclose(ds["hru_lon"].values, [0.5, 2.5], atol=0.01)
```

Add SIR fallback test to `TestDeriveGeometryFromFabric`:

```python
def test_lon_fallback_from_sir(self, derivation: PywatershedDerivation) -> None:
    """Without fabric, hru_lon falls back to SIR."""
    sir = _MockSIRAccessor(
        xr.Dataset(
            {"hru_lon": ("nhm_id", np.array([-75.0, -76.0]))},
            coords={"nhm_id": [1, 2]},
        )
    )
    ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
    ds = derivation.derive(ctx)
    assert "hru_lon" in ds
    np.testing.assert_allclose(ds["hru_lon"].values, [-75.0, -76.0])
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveGeometryFromFabric::test_lon_from_fabric tests/test_pywatershed_derivation.py::TestDeriveGeometryFromFabric::test_lon_fallback_from_sir -v`
Expected: FAIL — `hru_lon` not in ds

**Step 3: Implement `hru_lon` in `_derive_geometry()`**

In `_derive_geometry()`, after the `hru_lat` DataArray assignment (line 538), add:

```python
            lons = centroids_4326.x.values
            ds["hru_lon"] = xr.DataArray(
                lons,
                dims="nhru",
                attrs={"units": "decimal_degrees", "long_name": "Longitude of HRU centroid"},
            )
```

In the SIR fallback block (after line 552), add:

```python
            if "hru_lon" in sir:
                ds["hru_lon"] = xr.DataArray(
                    sir["hru_lon"].values,
                    dims="nhru",
                    attrs={"units": "decimal_degrees", "long_name": "Longitude of HRU centroid"},
                )
```

Update the docstring (line 484-506) to mention `hru_lon` alongside `hru_lat`:
- Summary: "Compute HRU area, centroid latitude, and centroid longitude..."
- "Derive ``hru_area`` (acres), ``hru_lat``, and ``hru_lon`` (decimal degrees)..."
- Returns: "Dataset with ``hru_area`` (acres), ``hru_lat``, and ``hru_lon`` (decimal degrees)..."
- Falls back: "Falls back to SIR variables ``hru_area_m2``, ``hru_lat``, and ``hru_lon``..."

**Step 4: Add metadata entry**

In `parameter_metadata.yml`, after `hru_lat` (line 45), add:

```yaml
  hru_lon:
    dimension: nhru
    units: decimal_degrees
    valid_range: [-180.0, 180.0]
    required: true
    description: "Longitude of HRU centroid"
```

**Step 5: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveGeometryFromFabric -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py src/hydro_param/data/pywatershed/parameter_metadata.yml tests/test_pywatershed_derivation.py
git commit -m "feat: add hru_lon to geometry derivation step 1"
```

---

### Task 2: Add `nhm_id`, `nhm_seg`, `hru_segment_nhm` to topology derivation

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py:559-706` (`_derive_topology`)
- Modify: `src/hydro_param/data/pywatershed/parameter_metadata.yml`
- Test: `tests/test_pywatershed_derivation.py`

**Step 1: Write the failing tests**

Add to `TestDeriveTopology` class (after `test_nsegment_coordinate`, ~line 1290):

```python
def test_nhm_id_from_fabric(
    self,
    derivation: PywatershedDerivation,
    sir_minimal: _MockSIRAccessor,
    synthetic_fabric: gpd.GeoDataFrame,
    synthetic_segments: gpd.GeoDataFrame,
) -> None:
    """nhm_id emitted from fabric id_field column."""
    ctx = DerivationContext(
        sir=sir_minimal,
        fabric=synthetic_fabric,
        segments=synthetic_segments,
        fabric_id_field="nhm_id",
        segment_id_field="nhm_seg",
    )
    ds = derivation.derive(ctx)
    assert "nhm_id" in ds
    np.testing.assert_array_equal(ds["nhm_id"].values, [101, 102, 103])
    assert ds["nhm_id"].dims == ("nhru",)

def test_nhm_seg_from_segments(
    self,
    derivation: PywatershedDerivation,
    sir_minimal: _MockSIRAccessor,
    synthetic_fabric: gpd.GeoDataFrame,
    synthetic_segments: gpd.GeoDataFrame,
) -> None:
    """nhm_seg emitted from segments segment_id_field column."""
    ctx = DerivationContext(
        sir=sir_minimal,
        fabric=synthetic_fabric,
        segments=synthetic_segments,
        fabric_id_field="nhm_id",
        segment_id_field="nhm_seg",
    )
    ds = derivation.derive(ctx)
    assert "nhm_seg" in ds
    np.testing.assert_array_equal(ds["nhm_seg"].values, [201, 202, 203])
    assert ds["nhm_seg"].dims == ("nsegment",)

def test_hru_segment_nhm_mapping(
    self,
    derivation: PywatershedDerivation,
    sir_minimal: _MockSIRAccessor,
    synthetic_fabric: gpd.GeoDataFrame,
    synthetic_segments: gpd.GeoDataFrame,
) -> None:
    """hru_segment_nhm maps each HRU to its NHM segment ID."""
    # synthetic_fabric has hru_segment=[1,2,2], seg_ids=[201,202,203]
    # HRU 101 -> segment 1 -> nhm_seg 201
    # HRU 102 -> segment 2 -> nhm_seg 202
    # HRU 103 -> segment 2 -> nhm_seg 202
    ctx = DerivationContext(
        sir=sir_minimal,
        fabric=synthetic_fabric,
        segments=synthetic_segments,
        fabric_id_field="nhm_id",
        segment_id_field="nhm_seg",
    )
    ds = derivation.derive(ctx)
    assert "hru_segment_nhm" in ds
    np.testing.assert_array_equal(ds["hru_segment_nhm"].values, [201, 202, 202])
    assert ds["hru_segment_nhm"].dims == ("nhru",)
```

Add test for custom column names (verifies config-driven, not hardcoded):

```python
def test_identity_params_with_custom_column_names(
    self,
    derivation: PywatershedDerivation,
) -> None:
    """Identity params work with non-NHM column names."""
    sir = _MockSIRAccessor(xr.Dataset(coords={"my_hru": [10, 20]}))
    fabric = gpd.GeoDataFrame(
        {"my_hru": [10, 20], "hru_segment": [1, 2]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        {"my_seg": [500, 600], "tosegment": [2, 0]},
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
        fabric_id_field="my_hru",
        segment_id_field="my_seg",
    )
    ds = derivation.derive(ctx)
    # Output params always named nhm_id/nhm_seg regardless of source column
    assert "nhm_id" in ds
    np.testing.assert_array_equal(ds["nhm_id"].values, [10, 20])
    assert "nhm_seg" in ds
    np.testing.assert_array_equal(ds["nhm_seg"].values, [500, 600])
    assert "hru_segment_nhm" in ds
    np.testing.assert_array_equal(ds["hru_segment_nhm"].values, [500, 600])
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveTopology::test_nhm_id_from_fabric tests/test_pywatershed_derivation.py::TestDeriveTopology::test_nhm_seg_from_segments tests/test_pywatershed_derivation.py::TestDeriveTopology::test_hru_segment_nhm_mapping tests/test_pywatershed_derivation.py::TestDeriveTopology::test_identity_params_with_custom_column_names -v`
Expected: FAIL — parameters not in ds

**Step 3: Implement identity params in `_derive_topology()`**

After the `seg_length` block (line 704), before `return ds` (line 706), add:

```python
        # --- nhm_id: HRU identifier from fabric id_field ---
        if id_field in fabric.columns:
            if "nhru" in ds.coords and id_field in fabric.columns:
                fabric_indexed = fabric.set_index(id_field)
                hru_ids = ds.coords["nhru"].values
                nhm_id_vals = hru_ids
            else:
                nhm_id_vals = fabric[id_field].values
            ds["nhm_id"] = xr.DataArray(
                nhm_id_vals,
                dims="nhru",
                attrs={"units": "none", "long_name": "HRU identifier"},
            )

        # --- nhm_seg: segment identifier from segment_id_field ---
        ds["nhm_seg"] = xr.DataArray(
            seg_ids,
            dims="nsegment",
            attrs={"units": "none", "long_name": "Segment identifier"},
        )

        # --- hru_segment_nhm: map HRU segment index to segment ID ---
        # hru_segment is 1-based index into segments; 0 means no segment.
        hru_seg_nhm = np.where(
            hru_segment > 0,
            seg_ids[hru_segment.astype(int) - 1],
            0,
        )
        ds["hru_segment_nhm"] = xr.DataArray(
            hru_seg_nhm,
            dims="nhru",
            attrs={
                "units": "none",
                "long_name": "Segment identifier for HRU contributing flow",
            },
        )
```

Update the docstring (lines 564-606) to mention the new parameters:
- "Read ``tosegment``, ``hru_segment``, ``seg_length``, ``nhm_id``, ``nhm_seg``, and ``hru_segment_nhm``..."
- Returns: add ``nhm_id`` (integer on nhru), ``nhm_seg`` (integer on nsegment), ``hru_segment_nhm`` (integer on nhru)
- Notes: "``nhm_id`` and ``nhm_seg`` are read from the config-declared ``id_field`` and ``segment_id_field`` columns. The output parameter names are always ``nhm_id`` and ``nhm_seg`` (pywatershed convention) regardless of the source column name."

**Step 4: Add metadata entries**

In `parameter_metadata.yml`, add in the topology section (after existing topology params, locate by searching for `tosegment`):

```yaml
  nhm_id:
    dimension: nhru
    units: none
    type: integer
    required: false
    description: "HRU identifier from fabric id_field"

  nhm_seg:
    dimension: nsegment
    units: none
    type: integer
    required: false
    description: "Segment identifier from fabric segment_id_field"

  hru_segment_nhm:
    dimension: nhru
    units: none
    type: integer
    required: false
    description: "Segment identifier for HRU contributing flow"
```

**Step 5: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py::TestDeriveTopology -v`
Expected: PASS

**Step 6: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py src/hydro_param/data/pywatershed/parameter_metadata.yml tests/test_pywatershed_derivation.py
git commit -m "feat: add nhm_id, nhm_seg, hru_segment_nhm to topology derivation step 2"
```

---

### Task 3: Run full test suite and verify

**Step 1: Run all checks**

Run: `pixi run -e dev check`
Expected: All tests pass, no lint/type errors

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Final commit if any formatting changes**

If ruff/mypy required fixes, commit them:
```bash
git commit -m "chore: fix lint/format from pre-commit"
```
