# Pipeline Memory Optimization + STAC Query Reuse — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix OOM kills during stage 4 batch processing by eliminating memory leaks in `_process_batch` and reducing redundant STAC queries.

**Architecture:** Four surgical memory fixes (source_cache cleanup, early `del`, copy-free GeoTIFF save, gc.collect) plus extracting STAC item querying into a reusable function that `_process_batch` calls once per batch instead of once per variable.

**Tech Stack:** Python, xarray, rioxarray, pystac-client, gc

---

### Task 1: Fix `save_to_geotiff` to avoid full array copy

**Files:**
- Modify: `src/hydro_param/data_access.py:425-448` (`save_to_geotiff`)
- Test: `tests/test_data_access.py`

**Step 1: Write the failing test**

Add to `tests/test_data_access.py`:

```python
def test_save_to_geotiff_does_not_copy_array(tmp_path: Path):
    """save_to_geotiff should not create a full copy of the DataArray."""
    rioxarray = pytest.importorskip("rioxarray")

    da = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
        attrs={"_FillValue": -9999.0, "units": "meters"},
    )
    da = da.rio.set_crs("EPSG:4326")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    out_path = tmp_path / "test.tif"
    save_to_geotiff(da, out_path)

    # Original attrs must be preserved (not mutated)
    assert "_FillValue" in da.attrs
    assert da.attrs["_FillValue"] == -9999.0
    assert da.attrs["units"] == "meters"
    assert out_path.exists()
```

Add necessary imports at top of file: `from pathlib import Path` (already present),
`from hydro_param.data_access import save_to_geotiff`.

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_data_access.py::test_save_to_geotiff_does_not_copy_array -v`
Expected: PASS (current code preserves attrs since it copies — but this establishes the contract)

**Step 3: Implement the copy-free save**

Replace `save_to_geotiff` in `src/hydro_param/data_access.py`:

```python
def save_to_geotiff(da: xr.DataArray, path: Path) -> Path:
    """Save an xarray DataArray as a GeoTIFF.

    Parameters
    ----------
    da : xr.DataArray
        Raster data with CRS information.
    path : Path
        Output file path.

    Returns
    -------
    Path
        The output path (same as input).
    """
    import rioxarray  # noqa: F401

    # Temporarily remove _FillValue to avoid conflict with encoding,
    # then restore — avoids a full .copy() of the DataArray.
    fill_in_attrs = da.attrs.pop("_FillValue", _SENTINEL)
    fill_in_encoding = da.encoding.pop("_FillValue", _SENTINEL)
    try:
        da.rio.to_raster(path)
    finally:
        if fill_in_attrs is not _SENTINEL:
            da.attrs["_FillValue"] = fill_in_attrs
        if fill_in_encoding is not _SENTINEL:
            da.encoding["_FillValue"] = fill_in_encoding
    logger.debug("Saved GeoTIFF: %s (%s)", path, da.shape)
    return path
```

Add the sentinel near the top of the module (after imports):

```python
_SENTINEL = object()
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_data_access.py::test_save_to_geotiff_does_not_copy_array -v`
Expected: PASS

**Step 5: Run all data_access tests**

Run: `pixi run -e dev pytest tests/test_data_access.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/hydro_param/data_access.py tests/test_data_access.py
git commit -m "fix: avoid full array copy in save_to_geotiff"
```

---

### Task 2: Extract `query_stac_items` from `fetch_stac_cog`

**Files:**
- Modify: `src/hydro_param/data_access.py:172-286` (`fetch_stac_cog`)
- Test: `tests/test_data_access.py`

**Step 1: Write the failing test**

Add to `tests/test_data_access.py`:

```python
def test_query_stac_items_returns_items():
    """query_stac_items returns a list of STAC items."""
    from hydro_param.data_access import query_stac_items

    mock_item = MagicMock()
    mock_item.properties = {"gsd": 10}

    mock_search = MagicMock()
    mock_search.item_collection.return_value = [mock_item]

    mock_client = MagicMock()
    mock_client.search.return_value = mock_search

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="3dep-seamless",
        crs="EPSG:4269",
        gsd=10,
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with patch("pystac_client.Client.open", return_value=mock_client):
        items = query_stac_items(entry, bbox)

    assert len(items) == 1
    assert items[0] is mock_item


def test_fetch_stac_cog_with_prequeried_items():
    """fetch_stac_cog skips STAC query when items are provided."""
    from hydro_param.data_access import fetch_stac_cog

    rioxarray = pytest.importorskip("rioxarray")

    squeezed = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
    )
    mock_da = MagicMock()
    mock_squeezed = MagicMock()
    mock_da.squeeze.return_value = mock_squeezed
    mock_squeezed.rio.crs = "EPSG:4269"
    mock_squeezed.rio.clip_box.return_value = squeezed
    mock_squeezed.size = 16

    mock_asset = MagicMock()
    mock_asset.href = "https://example.com/dem.tif"
    mock_item = MagicMock()
    mock_item.id = "test_tile"
    mock_item.properties = {}
    mock_item.assets = {"data": mock_asset}

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="3dep-seamless",
        crs="EPSG:4269",
    )
    bbox = [-75.8, 39.6, -74.4, 42.5]

    with (
        patch("pystac_client.Client.open") as mock_open,
        patch.object(rioxarray, "open_rasterio", return_value=mock_da),
    ):
        result = fetch_stac_cog(entry, bbox, items=[mock_item])

    # STAC client should NOT have been called
    mock_open.assert_not_called()
    assert result is not None
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_data_access.py::test_query_stac_items_returns_items tests/test_data_access.py::test_fetch_stac_cog_with_prequeried_items -v`
Expected: FAIL with ImportError (query_stac_items doesn't exist yet)

**Step 3: Extract `query_stac_items` and add `items` parameter**

In `src/hydro_param/data_access.py`, add new function before `fetch_stac_cog`:

```python
def query_stac_items(
    entry: DatasetEntry,
    bbox: list[float],
) -> list[Any]:
    """Query a STAC catalog and return matching items.

    Handles client creation, optional Planetary Computer signing,
    search, and GSD filtering. The returned items can be passed to
    ``fetch_stac_cog(..., items=...)`` to avoid repeated queries.

    Parameters
    ----------
    entry : DatasetEntry
        Registry entry with ``strategy="stac_cog"``.
    bbox : list[float]
        ``[west, south, east, north]`` in EPSG:4326.

    Returns
    -------
    list
        Matching STAC items (signed if required).
    """
    import pystac_client

    if entry.catalog_url is None:
        raise ValueError("stac_cog strategy requires 'catalog_url' on the dataset entry")
    if entry.collection is None:
        raise ValueError("stac_cog strategy requires 'collection' on the dataset entry")

    logger.info(
        "Querying STAC: catalog=%s collection=%s bbox=%s",
        entry.catalog_url,
        entry.collection,
        bbox,
    )

    modifier = None
    if entry.sign == "planetary_computer":
        import planetary_computer

        modifier = planetary_computer.sign_inplace

    client = pystac_client.Client.open(entry.catalog_url, modifier=modifier)

    search = client.search(
        collections=[entry.collection],
        bbox=bbox,
    )
    items = list(search.item_collection())
    if not items:
        raise RuntimeError(
            f"No STAC items found for collection='{entry.collection}' bbox={bbox}"
        )

    # Filter by GSD if specified
    if entry.gsd is not None:
        filtered = [i for i in items if i.properties.get("gsd") == entry.gsd]
        if filtered:
            items = filtered
        else:
            logger.warning(
                "No items with gsd=%d; using %d unfiltered items",
                entry.gsd,
                len(items),
            )

    logger.info("Found %d STAC items for bbox", len(items))
    return items
```

Then modify `fetch_stac_cog` to accept `items` parameter and delegate querying:

```python
def fetch_stac_cog(
    entry: DatasetEntry,
    bbox: list[float],
    *,
    asset_key: str | None = None,
    items: list[Any] | None = None,
) -> xr.DataArray:
    """Query a STAC catalog and load COG(s) clipped to the bounding box.

    ...existing docstring, plus:

    items : list or None
        Pre-queried STAC items from ``query_stac_items``. When provided,
        the STAC query is skipped entirely.
    """
    import rioxarray  # noqa: F401

    if items is None:
        items = query_stac_items(entry, bbox)

    # Load and mosaic tiles (rest of existing code, starting from resolved_key)
    ...
```

Remove the duplicated query logic from `fetch_stac_cog` (everything from
`import pystac_client` through the GSD filtering) — it now lives in
`query_stac_items`.

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_data_access.py -v`
Expected: All pass (new tests + existing tests)

**Step 5: Commit**

```bash
git add src/hydro_param/data_access.py tests/test_data_access.py
git commit -m "refactor: extract query_stac_items from fetch_stac_cog"
```

---

### Task 3: Memory cleanup in `_process_batch` + STAC items caching

**Files:**
- Modify: `src/hydro_param/pipeline.py:289-432` (`_process_batch`)
- Modify: `src/hydro_param/pipeline.py` (imports)
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_process_batch_releases_source_cache(tmp_path: Path):
    """_process_batch releases source_cache entries for raw vars after save."""
    import gc
    from unittest.mock import MagicMock, patch

    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import _process_batch

    # Create a mock raster that tracks deletion
    mock_da = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
        attrs={"units": "cm"},
    )
    mock_da = mock_da.rio.set_crs("EPSG:5070")
    mock_da = mock_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # Build minimal fixtures
    fabric = gpd.GeoDataFrame(
        {"nhm_id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com/stac",
        collection="test",
        crs="EPSG:5070",
    )

    config_raw = {
        "target_fabric": {"path": "dummy.gpkg", "id_field": "nhm_id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 2, 2]},
        "datasets": [{"name": "test_ds", "variables": ["var_a", "var_b"], "statistics": ["mean"]}],
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.dump(config_raw))
    config = load_config(cfg_path)

    ds_req = config.datasets[0]

    var_specs = [
        VariableSpec(name="var_a", band=1, units="cm", categorical=False, asset_key="var_a"),
        VariableSpec(name="var_b", band=1, units="cm", categorical=False, asset_key="var_b"),
    ]

    mock_zonal_df = pd.DataFrame({"mean": [1.0, 2.0]}, index=pd.Index([1, 2], name="nhm_id"))

    with (
        patch("hydro_param.pipeline.fetch_stac_cog", return_value=mock_da),
        patch("hydro_param.pipeline.query_stac_items", return_value=[MagicMock()]),
        patch("hydro_param.pipeline.save_to_geotiff", return_value=tmp_path / "mock.tif"),
        patch("hydro_param.processing.ZonalProcessor.process", return_value=mock_zonal_df),
    ):
        results = _process_batch(fabric, entry, ds_req, var_specs, config, tmp_path)

    assert "var_a" in results
    assert "var_b" in results
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_process_batch_releases_source_cache -v`
Expected: FAIL (ImportError for `query_stac_items` import in pipeline.py — not wired yet)

**Step 3: Implement memory cleanup + STAC caching in `_process_batch`**

In `src/hydro_param/pipeline.py`:

1. Add to imports:

```python
import gc

from hydro_param.data_access import (
    DERIVATION_FUNCTIONS,
    fetch_local_tiff,
    fetch_stac_cog,
    query_stac_items,
    save_to_geotiff,
)
```

2. In `_process_batch`, before the `for i, var_spec in enumerate(var_specs):` loop,
   add STAC items pre-query:

```python
    # Pre-query STAC items once for all variables in this batch
    stac_items: list | None = None
    if entry.strategy == "stac_cog":
        stac_items = query_stac_items(entry, bbox)
```

3. Update `_fetch` to accept and pass `items`:

```python
    def _fetch(
        dataset_entry: DatasetEntry,
        fetch_bbox: list[float],
        *,
        variable_source: str | None = None,
        asset_key: str | None = None,
        items: list | None = None,
    ) -> xr.DataArray:
        """Dispatch to the correct fetch function based on strategy."""
        if dataset_entry.strategy == "stac_cog":
            return fetch_stac_cog(
                dataset_entry, fetch_bbox, asset_key=asset_key, items=items,
            )
        ...  # rest unchanged
```

4. In the var loop, pass `stac_items` to `_fetch`:

```python
        # For DerivedVariableSpec:
        if var_spec.source not in source_cache:
            source_cache[var_spec.source] = _fetch(entry, bbox, items=stac_items)

        # For raw VariableSpec:
        da = _fetch(
            entry,
            bbox,
            variable_source=var_spec.source_override,
            asset_key=var_spec.asset_key,
            items=stac_items,
        )
```

5. After `save_to_geotiff(da, tiff_path)`, delete the DataArray for ALL vars
   (not just derived):

```python
        # Save as GeoTIFF for gdptools
        tiff_path = work_dir / f"{var_spec.name}.tif"
        save_to_geotiff(da, tiff_path)

        # Free raster before zonal stats — gdptools reads from the file
        del da
```

6. After zonal stats complete (after `results[var_spec.name] = df`), release
   raw var from source_cache if no derived var needs it, and run gc:

```python
        results[var_spec.name] = df

        # Clean up GeoTIFF after zonal stats
        tiff_path.unlink(missing_ok=True)

        # Release source_cache entries no longer needed
        if isinstance(var_spec, DerivedVariableSpec):
            remaining = var_specs[i + 1 :]
            source_still_needed = any(
                isinstance(v, DerivedVariableSpec) and v.source == var_spec.source
                for v in remaining
            )
            if not source_still_needed and var_spec.source in source_cache:
                del source_cache[var_spec.source]
        else:
            # Raw var: check if any later derived var uses this as source
            remaining = var_specs[i + 1 :]
            needed_by_derived = any(
                isinstance(v, DerivedVariableSpec) and v.source == var_spec.name
                for v in remaining
            )
            if not needed_by_derived and var_spec.name in source_cache:
                del source_cache[var_spec.name]

        gc.collect()
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_process_batch_releases_source_cache -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `pixi run -e dev pytest -v`
Expected: All existing + new tests pass

**Step 6: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "fix: release source_cache and add gc.collect in _process_batch"
```

---

### Task 4: Final verification

**Step 1: Run full check suite**

Run: `pixi run -e dev check`
Expected: lint, format, typecheck, and all tests pass

**Step 2: Run pre-commit**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Commit any formatting/lint fixes**

If pre-commit auto-fixes anything:

```bash
git add -u
git commit -m "chore: formatting fixes from pre-commit"
```
