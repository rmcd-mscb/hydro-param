# GFv1.1 Static Parameter Validation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate GFv1.1 ScienceBase rasters against the DRB test fabric by adding scale factor support to the pipeline and providing two example config files.

**Architecture:** Three deliverables: (1) apply `scale_factor` from `VariableSpec` after zonal stats in `pipeline.py`, (2) `gfv11_static_pipeline.yml` config referencing all 21 GFv1.1 datasets, (3) `gfv11_static_pywatershed.yml` config mapping GFv1.1 SIR variables to PRMS parameters. Each deliverable is independently testable.

**Tech Stack:** Python, pytest, Pydantic, YAML configs, gdptools/exactextract

**Design doc:** `docs/plans/2026-03-10-gfv11-validation-design.md`

---

### Task 1: Add scale factor support to pipeline — failing test

**Files:**
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

Add a test that verifies `_process_batch()` applies `scale_factor` when present on a `VariableSpec`. The test should mock `ZonalProcessor.process()` to return known numeric values, use a `VariableSpec` with `scale_factor=0.01`, and assert the output DataFrame has values multiplied by 0.01.

```python
class TestScaleFactor:
    """Verify scale_factor application after zonal statistics."""

    def test_scale_factor_applied(self, tmp_path: Path) -> None:
        """Pipeline multiplies numeric columns by scale_factor when set."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        from hydro_param.dataset_registry import DatasetEntry, VariableSpec

        # Create a minimal fabric
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )

        # Mock zonal stats output: raw integer-encoded values (slope * 100)
        raw_df = pd.DataFrame(
            {"slope_mean": [4500.0, 3200.0]},
            index=pd.Index([1, 2], name="nhm_id"),
        )

        var_spec = VariableSpec(
            name="slope",
            band=1,
            units="degrees",
            long_name="Terrain slope",
            scale_factor=0.01,
        )

        entry = DatasetEntry(
            name="gfv11_slope",
            description="test",
            strategy="local_tiff",
            category="topography",
            source=str(tmp_path / "slope.tif"),
            variables=[var_spec],
        )

        # Create a dummy GeoTIFF
        da = xr.DataArray(
            np.array([[4500, 3200]], dtype=np.uint32),
            dims=["y", "x"],
            coords={"y": [0.5], "x": [0.5, 1.5]},
        )
        da.rio.set_crs("EPSG:4326")
        da.rio.to_raster(tmp_path / "slope.tif")

        # Mock the processor to return our known raw values
        mock_processor = MagicMock(spec=ZonalProcessor)
        mock_processor.process.return_value = raw_df

        config = MagicMock()
        config.target_fabric.id_field = "nhm_id"

        ds_req = MagicMock()
        ds_req.statistics = ["mean"]

        with patch("hydro_param.pipeline._fetch", return_value=da):
            results = _process_batch(
                batch_fabric=fabric,
                config=config,
                ds_req=ds_req,
                entry=entry,
                var_specs=[var_spec],
                bbox=(0, 0, 2, 1),
                work_dir=tmp_path,
                processor=mock_processor,
            )

        # Scale factor 0.01 should have been applied
        result_df = results["slope"]
        assert result_df["slope_mean"].iloc[0] == pytest.approx(45.0)
        assert result_df["slope_mean"].iloc[1] == pytest.approx(32.0)

    def test_no_scale_factor_unchanged(self, tmp_path: Path) -> None:
        """Pipeline does NOT modify values when scale_factor is None."""
        import numpy as np
        from unittest.mock import MagicMock, patch

        from hydro_param.dataset_registry import DatasetEntry, VariableSpec

        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2]},
            geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        )

        raw_df = pd.DataFrame(
            {"elevation_mean": [100.0, 200.0]},
            index=pd.Index([1, 2], name="nhm_id"),
        )

        var_spec = VariableSpec(
            name="elevation",
            band=1,
            units="m",
            long_name="Elevation",
            # scale_factor defaults to None
        )

        entry = DatasetEntry(
            name="gfv11_dem",
            description="test",
            strategy="local_tiff",
            category="topography",
            source=str(tmp_path / "dem.tif"),
            variables=[var_spec],
        )

        da = xr.DataArray(
            np.array([[100, 200]], dtype=np.int16),
            dims=["y", "x"],
            coords={"y": [0.5], "x": [0.5, 1.5]},
        )
        da.rio.set_crs("EPSG:4326")
        da.rio.to_raster(tmp_path / "dem.tif")

        mock_processor = MagicMock(spec=ZonalProcessor)
        mock_processor.process.return_value = raw_df

        config = MagicMock()
        config.target_fabric.id_field = "nhm_id"

        ds_req = MagicMock()
        ds_req.statistics = ["mean"]

        with patch("hydro_param.pipeline._fetch", return_value=da):
            results = _process_batch(
                batch_fabric=fabric,
                config=config,
                ds_req=ds_req,
                entry=entry,
                var_specs=[var_spec],
                bbox=(0, 0, 2, 1),
                work_dir=tmp_path,
                processor=mock_processor,
            )

        result_df = results["elevation"]
        assert result_df["elevation_mean"].iloc[0] == pytest.approx(100.0)
        assert result_df["elevation_mean"].iloc[1] == pytest.approx(200.0)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::TestScaleFactor -v`
Expected: PASS for `test_no_scale_factor_unchanged` (no-op case), FAIL for `test_scale_factor_applied` (scale not applied yet)

**Step 3: Commit test**

```bash
git add tests/test_pipeline.py
git commit -m "test: add scale_factor tests for _process_batch"
```

---

### Task 2: Implement scale factor in pipeline — make test pass

**Files:**
- Modify: `src/hydro_param/pipeline.py:783` (after `processor.process()`, before `results[var_spec.name] = df`)

**Step 1: Add scale factor application**

In `_process_batch()`, after line 782 (`processor.process()` call), before line 783 (`results[var_spec.name] = df`), insert:

```python
        # Apply scale factor for integer-encoded rasters (e.g., slope × 100)
        if isinstance(var_spec, VariableSpec) and var_spec.scale_factor is not None:
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols] * var_spec.scale_factor
            logger.debug(
                "Applied scale_factor %.4f to %s",
                var_spec.scale_factor,
                var_spec.name,
            )

        results[var_spec.name] = df
```

Ensure `VariableSpec` is imported from `hydro_param.dataset_registry` at the top of `pipeline.py`. Check existing imports first — it is likely already imported.

**Step 2: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pipeline.py::TestScaleFactor -v`
Expected: Both tests PASS

**Step 3: Run full test suite**

Run: `pixi run -e dev pytest tests/test_pipeline.py -v`
Expected: All existing pipeline tests still pass

**Step 4: Commit**

```bash
git add src/hydro_param/pipeline.py
git commit -m "feat: apply scale_factor to zonal stats output in pipeline"
```

---

### Task 3: Update scale_factor docstring in VariableSpec

**Files:**
- Modify: `src/hydro_param/dataset_registry.py:91-96`

**Step 1: Update the docstring**

The current docstring says "The pipeline passes this through as metadata; consumers apply it." This is now wrong — the pipeline applies scale_factor itself. Update:

```python
    scale_factor : float or None
        Multiplicative scale factor for integer-encoded rasters (e.g.,
        ``0.01`` for values stored as ``value × 100``).  Follows
        CF-conventions ``scale_factor`` semantics.  When ``None``, no
        scaling is applied.  The pipeline applies this factor after
        zonal statistics so the SIR contains physically meaningful values.
```

**Step 2: Commit**

```bash
git add src/hydro_param/dataset_registry.py
git commit -m "docs: update scale_factor docstring to match pipeline behavior"
```

---

### Task 4: Create `gfv11_static_pipeline.yml` config

**Files:**
- Create: `configs/examples/gfv11_static_pipeline.yml`
- Test: Validate with `load_config()` in a test

**Step 1: Write the config file**

Model after `configs/examples/drb_2yr_pipeline.yml`. All 21 GFv1.1 datasets organized by category. No `domain` section. No `time_period` or temporal datasets.

```yaml
# GFv1.1 static parameter pipeline config — all 21 ScienceBase rasters
# Produces a SIR with topographic, soils, land cover, and water body data
# from the NHM Geospatial Fabric v1.1 pre-processed rasters.
#
# Prerequisites:
#   hydro-param gfv11 download /path/to/gfv11/data
#
# Usage:
#   hydro-param run configs/examples/gfv11_static_pipeline.yml
#
# NOTE: This config references DRB test fabric files that are not shipped
# with the repository. Obtain fabric files via pynhd/pygeohydro or from
# the USGS Geospatial Fabric before running.

target_fabric:
  path: data/pywatershed_gis/drb_2yr/nhru.gpkg
  id_field: nhm_id

datasets:
  topography:
    # GFv1.1 SRTM 30m DEM — elevation
    - name: gfv11_dem
      variables: [elevation]
      statistics: [mean]

    # GFv1.1 terrain slope (integer-encoded × 100, scale_factor applied by pipeline)
    - name: gfv11_slope
      variables: [slope]
      statistics: [mean]

    # GFv1.1 terrain aspect (integer-encoded × 100, scale_factor applied by pipeline)
    - name: gfv11_aspect
      variables: [aspect]
      statistics: [mean]

    # GFv1.1 topographic wetness index (integer-encoded × 100, scale_factor applied)
    - name: gfv11_twi
      variables: [twi]
      statistics: [mean]

    # GFv1.1 D8 flow direction — categorical majority
    - name: gfv11_fdr
      variables: [flow_dir]
      statistics: [majority]

  soils:
    # GFv1.1 SoilGrids250m sand percentage
    - name: gfv11_sand
      variables: [sand_pct]
      statistics: [mean]

    # GFv1.1 SoilGrids250m silt percentage
    - name: gfv11_silt
      variables: [silt_pct]
      statistics: [mean]

    # GFv1.1 SoilGrids250m clay percentage
    - name: gfv11_clay
      variables: [clay_pct]
      statistics: [mean]

    # GFv1.1 SoilGrids250m available water capacity
    - name: gfv11_awc
      variables: [awc]
      statistics: [mean]

    # GFv1.1 PRMS soil texture class (categorical)
    - name: gfv11_text_prms
      variables: [soil_type]
      statistics: [majority]

  land_cover:
    # GFv1.1 NALCMS 2015 → PRMS cover type (categorical)
    - name: gfv11_lulc
      variables: [cov_type]
      statistics: [majority]

    # GFv1.1 GMIS impervious surface percentage
    - name: gfv11_imperv
      variables: [imperv_pct]
      statistics: [mean]

    # GFv1.1 MODIS tree canopy cover percentage
    - name: gfv11_cnpy
      variables: [canopy_pct]
      statistics: [mean]

    # GFv1.1 pre-computed summer cover density
    - name: gfv11_covden_sum
      variables: [covden_sum]
      statistics: [mean]

    # GFv1.1 pre-computed winter cover density
    - name: gfv11_covden_win
      variables: [covden_win]
      statistics: [mean]

    # GFv1.1 pre-computed seasonal cover density loss
    - name: gfv11_covden_loss
      variables: [covden_loss]
      statistics: [mean]

    # GFv1.1 pre-computed summer rain interception
    - name: gfv11_srain
      variables: [srain_intcp]
      statistics: [mean]

    # GFv1.1 pre-computed winter rain interception
    - name: gfv11_wrain
      variables: [wrain_intcp]
      statistics: [mean]

    # GFv1.1 pre-computed snow interception
    - name: gfv11_snow_intcp
      variables: [snow_intcp]
      statistics: [mean]

    # GFv1.1 pre-computed root depth
    - name: gfv11_root_depth
      variables: [root_depth]
      statistics: [mean]

  water_bodies:
    # GFv1.1 NHD HR waterbody mask (categorical)
    - name: gfv11_wbg
      variables: [waterbody]
      statistics: [majority]

output:
  path: output
  format: netcdf
  sir_name: gfv11_static_sir

processing:
  batch_size: 240
  resume: true
```

**Step 2: Write a config validation test**

Add to `tests/test_pipeline.py` (or a small dedicated test):

```python
class TestGfv11Config:
    """Validate GFv1.1 example config files load without errors."""

    def test_gfv11_static_pipeline_loads(self) -> None:
        """gfv11_static_pipeline.yml parses and validates."""
        config_path = Path("configs/examples/gfv11_static_pipeline.yml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        config = load_config(config_path)
        assert config.target_fabric.id_field == "nhm_id"
        # 21 datasets across 4 categories
        total = sum(len(ds) for ds in config.datasets.values())
        assert total == 21
```

**Step 3: Run test**

Run: `pixi run -e dev pytest tests/test_pipeline.py::TestGfv11Config -v`
Expected: PASS

**Step 4: Commit**

```bash
git add configs/examples/gfv11_static_pipeline.yml tests/test_pipeline.py
git commit -m "feat: add gfv11_static_pipeline.yml example config (21 datasets)"
```

---

### Task 5: Create `gfv11_static_pywatershed.yml` config

**Files:**
- Create: `configs/examples/gfv11_static_pywatershed.yml`
- Modify: `tests/test_pipeline.py` (add validation test)

**Step 1: Write the pywatershed run config**

Model after `configs/examples/drb_2yr_pywatershed.yml`. Key differences:
- GFv1.1 sources instead of 3DEP/POLARIS/NLCD/gNATSGO
- No `forcing` section (Phase A: static only)
- No `climate_normals` section
- No `snow` section (no SNODAS)
- Pre-computed parameters (covden, interception, root_depth) go directly to PRMS params

```yaml
# GFv1.1 static-only pywatershed config — validates GFv1.1 rasters against DRB
# Generates parameters.nc from GFv1.1 ScienceBase sources (Phase A: no forcing).
#
# Prerequisites:
#   hydro-param run configs/examples/gfv11_static_pipeline.yml
#
# Usage:
#   hydro-param pywatershed run configs/examples/gfv11_static_pywatershed.yml
#
# NOTE: This config references DRB test fabric files that are not shipped
# with the repository. Obtain fabric files via pynhd/pygeohydro or from
# the USGS Geospatial Fabric before running.

target_model: pywatershed
version: "4.0"

sir_path: "../../output"

domain:
  fabric_path: data/pywatershed_gis/drb_2yr/nhru.gpkg
  segment_path: data/pywatershed_gis/drb_2yr/nsegment.gpkg
  id_field: nhm_id
  segment_id_field: nhm_seg

static_datasets:

  topography:
    available: [gfv11_dem, gfv11_slope, gfv11_aspect]
    hru_elev:
      source: gfv11_dem
      variable: elevation
      statistic: mean
      description: "Mean HRU elevation (SRTM 30m)"
    hru_slope:
      source: gfv11_slope
      variable: slope
      statistic: mean
      description: "Mean land surface slope (TGF 30m, scale_factor applied)"
    hru_aspect:
      source: gfv11_aspect
      variable: aspect
      statistic: mean
      description: "Mean HRU aspect (TGF 30m, scale_factor applied)"

  soils:
    available: [gfv11_sand, gfv11_silt, gfv11_clay, gfv11_awc, gfv11_text_prms]
    soil_type:
      source: gfv11_text_prms
      variable: soil_type
      statistic: majority
      description: "PRMS soil type from GFv1.1 pre-classified raster"
    soil_moist_max:
      source: gfv11_awc
      variable: awc
      statistic: mean
      description: "Max available water-holding capacity (SoilGrids250m AWC)"

  landcover:
    available: [gfv11_lulc, gfv11_imperv, gfv11_cnpy, gfv11_covden_sum, gfv11_covden_win, gfv11_covden_loss, gfv11_srain, gfv11_wrain, gfv11_snow_intcp, gfv11_root_depth]
    cov_type:
      source: gfv11_lulc
      variable: cov_type
      statistic: majority
      description: "Vegetation cover type from NALCMS 2015 (GFv1.1 pre-classified)"
    hru_percent_imperv:
      source: gfv11_imperv
      variable: imperv_pct
      statistic: mean
      description: "Impervious surface fraction (GMIS)"
    covden_sum:
      source: gfv11_covden_sum
      variable: covden_sum
      statistic: mean
      description: "Summer cover density (GFv1.1 pre-computed)"
    covden_win:
      source: gfv11_covden_win
      variable: covden_win
      statistic: mean
      description: "Winter cover density (GFv1.1 pre-computed)"
    srain_intcp:
      source: gfv11_srain
      variable: srain_intcp
      statistic: mean
      description: "Summer rain interception (GFv1.1 pre-computed, inches)"
    wrain_intcp:
      source: gfv11_wrain
      variable: wrain_intcp
      statistic: mean
      description: "Winter rain interception (GFv1.1 pre-computed, inches)"
    snow_intcp:
      source: gfv11_snow_intcp
      variable: snow_intcp
      statistic: mean
      description: "Snow interception (GFv1.1 pre-computed, inches)"

  waterbodies:
    available: []

calibration:
  generate_seeds: false

parameter_overrides:
  values: {}

output:
  path: models/pywatershed
  format: netcdf
  parameter_file: parameters.nc
```

**Step 2: Write a config validation test**

Add to the `TestGfv11Config` class in `tests/test_pipeline.py`:

```python
    def test_gfv11_static_pywatershed_loads(self) -> None:
        """gfv11_static_pywatershed.yml parses and validates."""
        from hydro_param.pywatershed_config import load_pywatershed_config

        config_path = Path("configs/examples/gfv11_static_pywatershed.yml")
        if not config_path.exists():
            pytest.skip("Config file not found")
        config = load_pywatershed_config(config_path)
        assert config.domain.id_field == "nhm_id"
        assert "topography" in config.static_datasets
        assert "soils" in config.static_datasets
        assert "landcover" in config.static_datasets
```

**Step 3: Run test**

Run: `pixi run -e dev pytest tests/test_pipeline.py::TestGfv11Config -v`
Expected: PASS

**Step 4: Commit**

```bash
git add configs/examples/gfv11_static_pywatershed.yml tests/test_pipeline.py
git commit -m "feat: add gfv11_static_pywatershed.yml example config"
```

---

### Task 6: Run full checks and final commit

**Step 1: Run full test suite**

Run: `pixi run -e dev check`
Expected: All lint, format, typecheck, tests pass

**Step 2: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Fix any issues found**

If any checks fail, fix and re-run.

**Step 4: Final commit if needed**

If fixes were needed, commit them with appropriate messages.
