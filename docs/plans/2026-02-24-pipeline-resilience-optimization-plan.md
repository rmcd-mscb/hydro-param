# Pipeline Resilience & Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three pipeline resilience issues: always-write manifest (resume works retroactively), NHGF STAC COG pre-fetch (eliminate redundant downloads), and network timeout (prevent silent hangs).

**Architecture:** The manifest is always written regardless of `resume` flag; `resume` only controls whether completed datasets are skipped. NHGF STAC COGs are fetched once per dataset+year and saved as local GeoTIFFs before the batch loop. A single `network_timeout` config value sets `GDAL_HTTP_TIMEOUT` at pipeline start.

**Tech Stack:** Python, Pydantic, xarray, rioxarray, pystac-client, gdptools, pytest

**Design doc:** `docs/plans/2026-02-24-pipeline-resilience-optimization-design.md`

---

### Task 1: Add `network_timeout` to ProcessingConfig

**Files:**
- Modify: `src/hydro_param/config.py:92-100`
- Test: `tests/test_config.py`

**Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_processing_config_network_timeout_default():
    """network_timeout defaults to 120."""
    from hydro_param.config import ProcessingConfig

    pc = ProcessingConfig()
    assert pc.network_timeout == 120


def test_processing_config_network_timeout_custom():
    """network_timeout accepts positive int."""
    from hydro_param.config import ProcessingConfig

    pc = ProcessingConfig(network_timeout=300)
    assert pc.network_timeout == 300


def test_processing_config_network_timeout_rejects_zero():
    """network_timeout rejects 0."""
    import pytest
    from hydro_param.config import ProcessingConfig

    with pytest.raises(Exception):
        ProcessingConfig(network_timeout=0)


def test_processing_config_network_timeout_rejects_negative():
    """network_timeout rejects negative values."""
    import pytest
    from hydro_param.config import ProcessingConfig

    with pytest.raises(Exception):
        ProcessingConfig(network_timeout=-10)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_config.py::test_processing_config_network_timeout_default -v`
Expected: FAIL — `ProcessingConfig` has no `network_timeout` field.

**Step 3: Write minimal implementation**

In `src/hydro_param/config.py`, add to `ProcessingConfig` (line 99, after `resume`):

```python
network_timeout: int = Field(default=120, gt=0, description="Network timeout in seconds")
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_config.py -k network_timeout -v`
Expected: All 4 PASS.

**Step 5: Commit**

```bash
git add src/hydro_param/config.py tests/test_config.py
git commit -m "feat: add network_timeout to ProcessingConfig (default 120s)"
```

---

### Task 2: Apply network timeout in `run_pipeline_from_config()`

**Files:**
- Modify: `src/hydro_param/pipeline.py:1008-1040`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_run_pipeline_sets_gdal_http_timeout(tmp_path: Path):
    """run_pipeline_from_config sets GDAL_HTTP_TIMEOUT from config."""
    import os
    from unittest.mock import patch

    from hydro_param.config import PipelineConfig

    gpkg_path = tmp_path / "test.gpkg"
    gpkg_path.write_text("fake")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg_path), "id_field": "hru_id"},
        datasets=[],
        output={"path": str(tmp_path / "output")},
        processing={"network_timeout": 300},
    )

    # Patch stage1 to short-circuit the pipeline (no real processing)
    with patch("hydro_param.pipeline.stage1_resolve_fabric") as mock_s1:
        mock_s1.side_effect = RuntimeError("stop early")
        try:
            from hydro_param.pipeline import run_pipeline_from_config
            from hydro_param.dataset_registry import DatasetRegistry

            registry = DatasetRegistry(datasets={})
            run_pipeline_from_config(config, registry)
        except RuntimeError:
            pass

    assert os.environ.get("GDAL_HTTP_TIMEOUT") == "300"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_run_pipeline_sets_gdal_http_timeout -v`
Expected: FAIL — `GDAL_HTTP_TIMEOUT` is not set.

**Step 3: Write minimal implementation**

In `src/hydro_param/pipeline.py`, add at the top of `run_pipeline_from_config()` (after the logging block, around line 1041):

```python
    # Apply network timeout to all GDAL/HTTP operations
    import os
    timeout_s = str(config.processing.network_timeout)
    os.environ["GDAL_HTTP_TIMEOUT"] = timeout_s
    os.environ["GDAL_HTTP_CONNECTTIMEOUT"] = timeout_s
    logger.info("  Network timeout: %ss", timeout_s)
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_run_pipeline_sets_gdal_http_timeout -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "feat: apply network_timeout as GDAL_HTTP_TIMEOUT at pipeline start"
```

---

### Task 3: Always-write manifest (decouple from `resume` flag)

**Files:**
- Modify: `src/hydro_param/pipeline.py:760-775` (manifest init), `:789` (skip guard), `:846` and `:927` (manifest save guards), `:932` (final save guard)
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_stage4_always_writes_manifest(tmp_path: Path):
    """Manifest is written even when resume=False."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest, PipelineConfig
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.manifest import MANIFEST_FILENAME, load_manifest
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    gpkg_path = tmp_path / "test.gpkg"
    gpkg_path.write_text("fake")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg_path), "id_field": "hru_id"},
        datasets=[],
        output={"path": str(tmp_path / "output")},
        processing={"resume": False},  # resume OFF
    )

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df):
        stage4_process(fabric, [(entry, ds_req, [var_spec])], config)

    # Manifest should exist even though resume=False
    manifest = load_manifest(config.output.path)
    assert manifest is not None
    assert "nlcd_osn_lndcov" in manifest.entries


def test_stage4_resume_false_does_not_skip(tmp_path: Path):
    """With resume=False, stage4 does NOT skip even if manifest is current."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest, PipelineConfig
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.manifest import (
        ManifestEntry,
        PipelineManifest,
        dataset_fingerprint,
        fabric_fingerprint,
    )
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    output_dir = tmp_path / "output"
    lc_dir = output_dir / "land_cover"
    lc_dir.mkdir(parents=True)
    (lc_dir / "LndCov.csv").write_text("hru_id,LndCov\na,11\n")

    gpkg_path = tmp_path / "test.gpkg"
    gpkg_path.write_text("fake")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg_path), "id_field": "hru_id"},
        datasets=[],
        output={"path": str(output_dir)},
        processing={"resume": False},  # resume OFF
    )

    # Write manifest that matches perfectly
    ds_fp = dataset_fingerprint(ds_req, entry, [var_spec], config.processing)
    fab_fp = fabric_fingerprint(config)
    manifest = PipelineManifest(
        fabric_fingerprint=fab_fp,
        entries={
            "nlcd_osn_lndcov": ManifestEntry(
                fingerprint=ds_fp,
                static_files={"LndCov": "land_cover/LndCov.csv"},
            ),
        },
    )
    manifest.save(output_dir)

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    # Even with valid manifest, process_nhgf_stac SHOULD be called (resume=False)
    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        stage4_process(fabric, [(entry, ds_req, [var_spec])], config)
        mock_method.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_stage4_always_writes_manifest -v`
Expected: FAIL — manifest is not written when `resume=False`.

**Step 3: Write minimal implementation**

Modify `stage4_process()` in `src/hydro_param/pipeline.py` (lines 760–775):

**Replace** the current manifest init block:

```python
    # Resume support: load manifest if resume is enabled
    manifest: _manifest_mod.PipelineManifest | None = None
    fab_fp: str = ""
    if config.processing.resume:
        manifest = _manifest_mod.load_manifest(config.output.path)
        fab_fp = _manifest_mod.fabric_fingerprint(config)
        if manifest is not None and not manifest.is_fabric_current(fab_fp):
            # Fabric changed: discard stale manifest and start fresh
            logger.warning(
                "Fabric fingerprint changed — reprocessing all datasets (old=%s, new=%s)",
                manifest.fabric_fingerprint,
                fab_fp,
            )
            manifest = None
        if manifest is None:
            manifest = _manifest_mod.PipelineManifest(fabric_fingerprint=fab_fp)
```

**With:**

```python
    # Always create manifest for resume support.
    # The resume flag controls whether completed datasets are *skipped*,
    # not whether the manifest is *written*.
    fab_fp = _manifest_mod.fabric_fingerprint(config)
    manifest = _manifest_mod.PipelineManifest(fabric_fingerprint=fab_fp)

    if config.processing.resume:
        existing = _manifest_mod.load_manifest(config.output.path)
        if existing is not None and existing.is_fabric_current(fab_fp):
            manifest = existing  # Preserve entries for skip checks
        elif existing is not None:
            logger.warning(
                "Fabric fingerprint changed — reprocessing all datasets (old=%s, new=%s)",
                existing.fabric_fingerprint,
                fab_fp,
            )
```

**Also update** the skip guard (line 789) — change:

```python
        if manifest is not None:
            if manifest.is_dataset_current(ds_req.name, ds_fp, config.output.path):
```

To:

```python
        if config.processing.resume:
            if manifest.is_dataset_current(ds_req.name, ds_fp, config.output.path):
```

**Also update** the manifest save guards — remove the `if manifest is not None:` guards at lines 846, 927, and 932. Since `manifest` is always created now, these guards are unnecessary. Change:

Line 846: `if manifest is not None:` → remove the `if`, dedent the body.
Line 927: `if manifest is not None:` → remove the `if`, dedent the body.
Line 932: `if manifest is not None:` → remove the `if`, dedent the body.

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pipeline.py -k "manifest or resume" -v`
Expected: All manifest/resume tests PASS, including both new tests and the 4 existing resume tests.

**Step 5: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "fix: always write manifest regardless of resume flag"
```

---

### Task 4: Add `fetch_nhgf_stac_cog()` to data_access.py

**Files:**
- Modify: `src/hydro_param/data_access.py` (add new function after `save_to_geotiff`)
- Test: `tests/test_data_access.py`

**Step 1: Write the failing test**

Add to `tests/test_data_access.py`:

```python
def test_fetch_nhgf_stac_cog_basic(tmp_path: Path):
    """fetch_nhgf_stac_cog fetches a COG and returns a DataArray."""
    from unittest.mock import MagicMock, patch

    import numpy as np
    import xarray as xr

    from hydro_param.data_access import fetch_nhgf_stac_cog

    # Mock the STAC collection
    mock_collection = MagicMock(name="pystac.Collection")

    # Mock NHGFStacTiffData — its .ds attribute is the DataArray
    mock_da = xr.DataArray(
        np.ones((3, 3)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0], "x": [1.0, 2.0, 3.0]},
    )
    mock_nhgf = MagicMock(name="NHGFStacTiffData")
    mock_nhgf.ds = mock_da

    with (
        patch("gdptools.helpers.get_stac_collection", return_value=mock_collection),
        patch("gdptools.NHGFStacTiffData", return_value=mock_nhgf),
    ):
        result = fetch_nhgf_stac_cog(
            collection_id="nlcd-LndCov",
            variable_name="LndCov",
            year=2021,
        )

    assert isinstance(result, xr.DataArray)
    assert result.shape == (3, 3)


def test_fetch_nhgf_stac_cog_no_year(tmp_path: Path):
    """fetch_nhgf_stac_cog works without a year (static dataset)."""
    from unittest.mock import MagicMock, patch

    import numpy as np
    import xarray as xr

    from hydro_param.data_access import fetch_nhgf_stac_cog

    mock_collection = MagicMock()
    mock_da = xr.DataArray(np.ones((2, 2)), dims=["y", "x"])
    mock_nhgf = MagicMock()
    mock_nhgf.ds = mock_da

    with (
        patch("gdptools.helpers.get_stac_collection", return_value=mock_collection),
        patch("gdptools.NHGFStacTiffData", return_value=mock_nhgf) as p_nhgf,
    ):
        result = fetch_nhgf_stac_cog(
            collection_id="nlcd-LndCov",
            variable_name="LndCov",
            year=None,
        )

    assert isinstance(result, xr.DataArray)
    # source_time_period should NOT be passed when year is None
    nhgf_kwargs = p_nhgf.call_args.kwargs
    assert "source_time_period" not in nhgf_kwargs or nhgf_kwargs["source_time_period"] is None
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_data_access.py::test_fetch_nhgf_stac_cog_basic -v`
Expected: FAIL — `fetch_nhgf_stac_cog` does not exist.

**Step 3: Write minimal implementation**

Add to `src/hydro_param/data_access.py` after the `save_to_geotiff()` function (after line 492):

```python
# ---------------------------------------------------------------------------
# NHGF STAC COG pre-fetch (one-shot download for batch loop)
# ---------------------------------------------------------------------------


def fetch_nhgf_stac_cog(
    collection_id: str,
    variable_name: str,
    *,
    year: int | None = None,
    band: int = 1,
) -> xr.DataArray:
    """Fetch a COG from the NHGF STAC catalog and return as a DataArray.

    Uses ``gdptools.NHGFStacTiffData`` to fetch the raster once. This is
    intended to be called once per dataset+year at the pipeline level, and
    the result saved as a local GeoTIFF for the batch loop.

    Parameters
    ----------
    collection_id : str
        NHGF STAC collection ID (e.g. ``"nlcd-LndCov"``).
    variable_name : str
        Variable name within the collection.
    year : int or None
        Year filter. When ``None``, no time period is applied.
    band : int
        Band number (default 1).

    Returns
    -------
    xr.DataArray
        The fetched raster.
    """
    import gdptools

    logger.info(
        "Fetching NHGF STAC COG: collection=%s var=%s year=%s",
        collection_id,
        variable_name,
        year,
    )

    collection = gdptools.helpers.get_stac_collection(collection_id)

    kwargs: dict[str, Any] = {
        "source_collection": collection,
        "source_var": variable_name,
        "band": band,
    }
    if year is not None:
        kwargs["source_time_period"] = [f"{year}-01-01", f"{year}-12-31"]

    nhgf_data = gdptools.NHGFStacTiffData(**kwargs)

    logger.info("NHGF STAC COG fetched: shape=%s", nhgf_data.ds.shape)
    return nhgf_data.ds
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_data_access.py -k fetch_nhgf_stac_cog -v`
Expected: Both tests PASS.

**Step 5: Commit**

```bash
git add src/hydro_param/data_access.py tests/test_data_access.py
git commit -m "feat: add fetch_nhgf_stac_cog() for one-shot NHGF STAC downloads"
```

---

### Task 5: Route NHGF STAC static datasets through pre-fetch in stage4

**Files:**
- Modify: `src/hydro_param/pipeline.py:853-929` (the static dataset branch of stage4_process)
- Test: `tests/test_pipeline.py`

This is the integration task. Before the batch loop, for `nhgf_stac` static datasets:
1. Fetch the COG once with `fetch_nhgf_stac_cog()`
2. Save to a temp GeoTIFF
3. Create a modified entry with `strategy="local_tiff"` and `source` pointing to the temp file
4. Convert var_specs to strip NHGF-specific fields, add `source_override`
5. Run the batch loop using the `local_tiff` path

**Step 1: Write the failing test**

Add to `tests/test_pipeline.py`:

```python
def test_stage4_nhgf_stac_prefetch_skips_per_batch_nhgf(tmp_path: Path):
    """NHGF STAC static datasets use pre-fetch, NOT per-batch process_nhgf_stac."""
    from unittest.mock import MagicMock, patch

    import numpy as np
    import xarray as xr

    from hydro_param.config import DatasetRequest, PipelineConfig
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"], "batch_id": [0, 1]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    gpkg_path = tmp_path / "test.gpkg"
    gpkg_path.write_text("fake")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg_path), "id_field": "hru_id"},
        datasets=[],
        output={"path": str(tmp_path / "output")},
    )

    # Mock fetch_nhgf_stac_cog to return a DataArray
    mock_da = xr.DataArray(
        np.ones((3, 3)),
        dims=["y", "x"],
        coords={"y": [0.5, 1.0, 1.5], "x": [0.5, 1.0, 1.5]},
    )
    mock_da.attrs["_FillValue"] = -9999

    # Mock processor.process to return a DataFrame
    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    with (
        patch("hydro_param.pipeline.fetch_nhgf_stac_cog", return_value=mock_da) as p_fetch,
        patch.object(ZonalProcessor, "process_nhgf_stac") as p_nhgf,
        patch.object(ZonalProcessor, "process", return_value=mock_df),
    ):
        stage4_process(fabric, [(entry, ds_req, [var_spec])], config)

        # fetch_nhgf_stac_cog should be called ONCE (not per batch)
        p_fetch.assert_called_once()

        # process_nhgf_stac should NOT be called (we use local_tiff path now)
        p_nhgf.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_stage4_nhgf_stac_prefetch_skips_per_batch_nhgf -v`
Expected: FAIL — `process_nhgf_stac` is still called per batch.

**Step 3: Write minimal implementation**

In `src/hydro_param/pipeline.py`:

1. Add import at the top of the file (with other data_access imports around line 50):
```python
from hydro_param.data_access import fetch_nhgf_stac_cog
```

2. In `stage4_process()`, modify the static dataset branch. Before the year/batch loop (around line 853, after the `years` expansion), add NHGF STAC pre-fetch logic:

Replace the section from the year expansion (line 853) through the end of the static dataset processing (line 929) with logic that:

- Detects `entry.strategy == "nhgf_stac" and not entry.temporal`
- Creates a temporary directory for the pre-fetched COGs
- For each year, calls `fetch_nhgf_stac_cog()` once per variable
- Saves each as a local GeoTIFF
- Creates a modified entry with `strategy="local_tiff"` and updates var_specs with `source_override` pointing to the temp file
- Runs the batch loop using the `local_tiff` path (existing `_process_batch` code)
- Cleans up temp files

The key code structure:

```python
        # NHGF STAC pre-fetch: download COG once, then batch via local_tiff
        nhgf_prefetch = entry.strategy == "nhgf_stac" and not entry.temporal
        prefetch_tmpdir = None

        if nhgf_prefetch:
            prefetch_tmpdir = tempfile.mkdtemp(prefix="hydro_param_nhgf_")
            local_entry = entry.model_copy(update={"strategy": "local_tiff"})

        for year in years:
            year_req = ds_req.model_copy(update={"year": year})

            # Pre-fetch NHGF STAC COGs for this year
            year_var_specs = var_specs
            if nhgf_prefetch:
                year_var_specs = []
                for vs in var_specs:
                    if isinstance(vs, DerivedVariableSpec):
                        raise NotImplementedError(
                            "Derived variables not supported for nhgf_stac strategy"
                        )
                    da = fetch_nhgf_stac_cog(
                        collection_id=cast(str, entry.collection),
                        variable_name=vs.name,
                        year=year,
                        band=vs.band,
                    )
                    tiff_path = Path(prefetch_tmpdir) / f"{vs.name}_{year}.tif"
                    save_to_geotiff(da, tiff_path)
                    del da
                    year_var_specs.append(
                        VariableSpec(
                            name=vs.name,
                            band=1,
                            categorical=vs.categorical,
                            source_override=str(tiff_path),
                        )
                    )
                    logger.info("  Pre-fetched %s year=%s → %s", vs.name, year, tiff_path)

            batch_entry = local_entry if nhgf_prefetch else entry

            for batch_id in batch_ids:
                batch = fabric[fabric["batch_id"] == batch_id]
                t_batch = time.perf_counter()

                with tempfile.TemporaryDirectory(prefix="hydro_param_") as tmp:
                    work_dir = Path(tmp)
                    batch_results = _process_batch(
                        batch, batch_entry, year_req, year_var_specs, config, work_dir
                    )

                # ... (rest of batch logging and result collection unchanged)

        # Clean up pre-fetch temp dir
        if prefetch_tmpdir is not None:
            import shutil
            shutil.rmtree(prefetch_tmpdir, ignore_errors=True)
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_stage4_nhgf_stac_prefetch_skips_per_batch_nhgf -v`
Expected: PASS.

**Step 5: Run existing NHGF STAC and resume tests**

Run: `pixi run -e dev pytest tests/test_pipeline.py -k "nhgf_stac or resume or manifest" -v`
Expected: All PASS.

**Step 6: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "feat: pre-fetch NHGF STAC COGs once per dataset+year, route batches through local_tiff"
```

---

### Task 6: Add timeout error message with guidance

**Files:**
- Modify: `src/hydro_param/data_access.py` (wrap `fetch_nhgf_stac_cog` and network calls)
- Test: `tests/test_data_access.py`

**Step 1: Write the failing test**

Add to `tests/test_data_access.py`:

```python
def test_fetch_nhgf_stac_cog_timeout_message():
    """Timeout errors include helpful guidance."""
    from unittest.mock import patch

    from hydro_param.data_access import fetch_nhgf_stac_cog

    with patch(
        "gdptools.helpers.get_stac_collection",
        side_effect=Exception("HTTP timeout"),
    ):
        with pytest.raises(RuntimeError, match="Network timeout.*increase the timeout"):
            fetch_nhgf_stac_cog(
                collection_id="nlcd-LndCov",
                variable_name="LndCov",
                year=2021,
            )
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_data_access.py::test_fetch_nhgf_stac_cog_timeout_message -v`
Expected: FAIL — raw exception propagates without helpful message.

**Step 3: Write minimal implementation**

Wrap the body of `fetch_nhgf_stac_cog()` in a try/except that catches network-related exceptions and re-raises with guidance:

```python
    try:
        collection = gdptools.helpers.get_stac_collection(collection_id)
        # ... (existing code)
    except Exception as exc:
        exc_str = str(exc).lower()
        if any(kw in exc_str for kw in ("timeout", "timed out", "connection")):
            raise RuntimeError(
                f"Network timeout fetching NHGF STAC COG "
                f"(collection='{collection_id}', var='{variable_name}', year={year}).\n\n"
                f"  If this endpoint is slow, increase the timeout:\n"
                f"    processing:\n"
                f"      network_timeout: 300\n\n"
                f"  To skip completed datasets on re-run:\n"
                f"    processing:\n"
                f"      resume: true\n\n"
                f"  Original error: {exc}"
            ) from exc
        raise
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_data_access.py::test_fetch_nhgf_stac_cog_timeout_message -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/hydro_param/data_access.py tests/test_data_access.py
git commit -m "feat: add helpful error message for network timeouts in NHGF STAC fetch"
```

---

### Task 7: Final verification and cleanup

**Files:**
- All modified files from Tasks 1–6

**Step 1: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass.

**Step 2: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass.

**Step 3: Run full checks**

Run: `pixi run -e dev check`
Expected: All checks pass (lint, format, typecheck, tests).

**Step 4: Commit any fixups from pre-commit**

If pre-commit made formatting changes:
```bash
git add -u
git commit -m "style: apply pre-commit formatting fixes"
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | `network_timeout` config field | `config.py`, `tests/test_config.py` |
| 2 | Apply timeout as GDAL env var | `pipeline.py`, `tests/test_pipeline.py` |
| 3 | Always-write manifest | `pipeline.py`, `tests/test_pipeline.py` |
| 4 | `fetch_nhgf_stac_cog()` function | `data_access.py`, `tests/test_data_access.py` |
| 5 | Pre-fetch routing in stage4 | `pipeline.py`, `tests/test_pipeline.py` |
| 6 | Timeout error guidance | `data_access.py`, `tests/test_data_access.py` |
| 7 | Final verification | All |

**Estimated commits:** 6–7 (one per task, plus possible style fixup)
