# DerivedContinuousSpec Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add pixel-level raster math before zonal stats via a new `DerivedContinuousSpec` variable spec type.

**Architecture:** New Pydantic model follows `DerivedCategoricalSpec`'s second-pass pattern. Source GeoTIFFs are aligned via `rioxarray.reproject_match()`, combined with a reduce-based arithmetic operation, then processed through continuous zonal stats.

**Tech Stack:** pydantic, rioxarray, numpy, xarray, functools.reduce

---

### Task 1: Add `DerivedContinuousSpec` model

**Files:**
- Modify: `src/hydro_param/dataset_registry.py:152-199`
- Test: `tests/test_dataset_registry.py` (or inline in existing tests)

**Step 1: Write the failing test**

In `tests/test_dataset_registry.py`, add tests for the new model:

```python
from hydro_param.dataset_registry import DerivedContinuousSpec


class TestDerivedContinuousSpec:
    def test_valid_spec(self):
        spec = DerivedContinuousSpec(
            name="product",
            sources=["a", "b"],
            operation="multiply",
            align_to="a",
        )
        assert spec.name == "product"
        assert spec.operation == "multiply"
        assert spec.resampling_method == "nearest"

    def test_scale_factor_optional(self):
        spec = DerivedContinuousSpec(
            name="product",
            sources=["a", "b"],
            operation="divide",
            align_to="b",
            scale_factor=0.01,
        )
        assert spec.scale_factor == 0.01

    def test_rejects_single_source(self):
        with pytest.raises(ValidationError):
            DerivedContinuousSpec(
                name="bad",
                sources=["a"],
                operation="multiply",
                align_to="a",
            )

    def test_rejects_invalid_operation(self):
        with pytest.raises(ValidationError):
            DerivedContinuousSpec(
                name="bad",
                sources=["a", "b"],
                operation="power",
                align_to="a",
            )

    def test_align_to_must_be_in_sources(self):
        with pytest.raises(ValidationError):
            DerivedContinuousSpec(
                name="bad",
                sources=["a", "b"],
                operation="multiply",
                align_to="c",
            )

    def test_three_sources(self):
        spec = DerivedContinuousSpec(
            name="triple",
            sources=["a", "b", "c"],
            operation="add",
            align_to="b",
        )
        assert len(spec.sources) == 3
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_dataset_registry.py::TestDerivedContinuousSpec -v`
Expected: FAIL — `ImportError` (class doesn't exist yet)

**Step 3: Write the implementation**

In `src/hydro_param/dataset_registry.py`, after `DerivedCategoricalSpec` (line ~196), add:

```python
class DerivedContinuousSpec(BaseModel):
    """Describe a continuous variable derived from pixel-level arithmetic on multiple sources.

    Multi-source continuous derivations apply an arithmetic operation
    (multiply, divide, add, subtract) to two or more aligned source
    rasters *before* zonal statistics.  This preserves within-HRU
    spatial correlation that would be lost by aggregating each raster
    independently and combining the results.

    Unlike ``DerivedCategoricalSpec`` (multi-source, categorical output),
    this always produces continuous output processed with standard
    zonal statistics (mean, median, etc.).

    Parameters
    ----------
    name : str
        Logical name for the derived variable (e.g.,
        ``"soil_moist_product"``).
    sources : list[str]
        Names of the source ``VariableSpec`` entries to combine.
        Must contain at least 2 entries.  All sources must belong
        to the same dataset.
    operation : {"multiply", "divide", "add", "subtract"}
        Arithmetic operation applied left-to-right across sources
        via ``functools.reduce``.
    align_to : str
        Name of the source whose grid (resolution, extent, CRS)
        is used as the resampling template.  Must be one of
        ``sources``.
    units : str
        Units of the derived variable after the operation.
    long_name : str
        Human-readable description for metadata.
    scale_factor : float or None
        Multiplicative factor applied to zonal statistics output
        (e.g., 0.01 to convert from percent to fraction).
    resampling_method : str
        Rasterio resampling method name for aligning non-template
        sources (default ``"nearest"``).
    """

    name: str
    sources: list[str]
    operation: Literal["multiply", "divide", "add", "subtract"]
    align_to: str
    units: str = ""
    long_name: str = ""
    scale_factor: float | None = None
    resampling_method: str = "nearest"

    @field_validator("sources")
    @classmethod
    def _check_min_sources(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            msg = "DerivedContinuousSpec requires at least 2 sources"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _check_align_to_in_sources(self) -> "DerivedContinuousSpec":
        if self.align_to not in self.sources:
            msg = (
                f"align_to '{self.align_to}' must be one of sources "
                f"{self.sources}"
            )
            raise ValueError(msg)
        return self
```

Update the `AnyVariableSpec` union (line ~199):

```python
AnyVariableSpec = VariableSpec | DerivedVariableSpec | DerivedCategoricalSpec | DerivedContinuousSpec
```

Add `Literal` to the imports from `typing` if not already present, and `model_validator` from pydantic if not imported.

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_dataset_registry.py::TestDerivedContinuousSpec -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/dataset_registry.py tests/test_dataset_registry.py
git commit -m "feat: add DerivedContinuousSpec model for pixel-level raster math"
```

---

### Task 2: Add `align_rasters()` and `apply_raster_operation()`

**Files:**
- Modify: `src/hydro_param/data_access.py`
- Test: `tests/test_data_access.py`

**Step 1: Write the failing tests**

In `tests/test_data_access.py`, add:

```python
class TestAlignRasters:
    """Tests for align_rasters() template-based raster alignment."""

    def test_matching_grids_unchanged(self):
        """Sources already on the same grid are returned as-is."""
        template = xr.DataArray(
            np.ones((4, 4)),
            dims=("y", "x"),
            coords={"y": [40.0, 39.0, 38.0, 37.0], "x": [-75.0, -74.0, -73.0, -72.0]},
        )
        template = template.rio.write_crs("EPSG:4326").rio.set_spatial_dims("x", "y")
        other = template * 2

        result = align_rasters([other, template], template)
        assert len(result) == 2
        assert result[0].shape == template.shape
        assert result[1].shape == template.shape

    def test_different_resolution_aligned(self):
        """Coarser source is resampled to match finer template."""
        fine = xr.DataArray(
            np.ones((4, 4)),
            dims=("y", "x"),
            coords={"y": [40.0, 39.0, 38.0, 37.0], "x": [-75.0, -74.0, -73.0, -72.0]},
        )
        fine = fine.rio.write_crs("EPSG:4326").rio.set_spatial_dims("x", "y")
        coarse = xr.DataArray(
            np.full((2, 2), 5.0),
            dims=("y", "x"),
            coords={"y": [40.0, 37.0], "x": [-75.0, -72.0]},
        )
        coarse = coarse.rio.write_crs("EPSG:4326").rio.set_spatial_dims("x", "y")

        result = align_rasters([coarse, fine], fine)
        assert result[0].shape == fine.shape  # coarse upsampled
        assert result[1].shape == fine.shape  # template unchanged

    def test_template_not_resampled(self):
        """The template source is passed through without resampling."""
        template = xr.DataArray(
            np.arange(16, dtype=float).reshape(4, 4),
            dims=("y", "x"),
            coords={"y": [40.0, 39.0, 38.0, 37.0], "x": [-75.0, -74.0, -73.0, -72.0]},
        )
        template = template.rio.write_crs("EPSG:4326").rio.set_spatial_dims("x", "y")

        result = align_rasters([template], template)
        xr.testing.assert_identical(result[0], template)


class TestApplyRasterOperation:
    """Tests for apply_raster_operation() arithmetic operations."""

    @pytest.fixture()
    def a(self):
        return xr.DataArray(np.array([[2.0, 3.0], [4.0, 5.0]]), dims=("y", "x"))

    @pytest.fixture()
    def b(self):
        return xr.DataArray(np.array([[10.0, 20.0], [30.0, 40.0]]), dims=("y", "x"))

    @pytest.fixture()
    def c(self):
        return xr.DataArray(np.array([[0.5, 0.5], [0.5, 0.5]]), dims=("y", "x"))

    def test_multiply(self, a, b):
        result = apply_raster_operation([a, b], "multiply")
        expected = a * b
        xr.testing.assert_equal(result, expected)

    def test_divide(self, a, b):
        result = apply_raster_operation([a, b], "divide")
        expected = a / b
        xr.testing.assert_equal(result, expected)

    def test_add(self, a, b):
        result = apply_raster_operation([a, b], "add")
        expected = a + b
        xr.testing.assert_equal(result, expected)

    def test_subtract(self, a, b):
        result = apply_raster_operation([a, b], "subtract")
        expected = a - b
        xr.testing.assert_equal(result, expected)

    def test_three_sources_multiply(self, a, b, c):
        """Left-to-right fold: (a * b) * c."""
        result = apply_raster_operation([a, b, c], "multiply")
        expected = (a * b) * c
        xr.testing.assert_equal(result, expected)

    def test_invalid_operation(self, a, b):
        with pytest.raises(ValueError, match="Unsupported"):
            apply_raster_operation([a, b], "power")
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_data_access.py::TestAlignRasters tests/test_data_access.py::TestApplyRasterOperation -v`
Expected: FAIL — `ImportError` (functions don't exist yet)

**Step 3: Write the implementation**

In `src/hydro_param/data_access.py`, add near the bottom (before or after the existing derivation function registries):

```python
import functools
import operator

def align_rasters(
    sources: list[xr.DataArray],
    template: xr.DataArray,
    method: str = "nearest",
) -> list[xr.DataArray]:
    """Align source rasters to a template grid via reprojection.

    Non-template sources are reprojected to match the template's CRS,
    resolution, and extent using ``rioxarray.reproject_match()``.  The
    template itself is passed through unchanged.

    Parameters
    ----------
    sources : list[xr.DataArray]
        Source rasters to align.
    template : xr.DataArray
        Reference raster whose grid defines the target CRS,
        resolution, and extent.
    method : str
        Rasterio resampling method name (default ``"nearest"``).

    Returns
    -------
    list[xr.DataArray]
        Aligned rasters in the same order as *sources*, all
        sharing the template's grid.
    """
    from rasterio.enums import Resampling

    resampling = Resampling[method]
    aligned = []
    for da in sources:
        if da is template:
            aligned.append(da)
        else:
            aligned.append(da.rio.reproject_match(template, resampling=resampling))
    return aligned


_RASTER_OPS: dict[str, Any] = {
    "multiply": operator.mul,
    "divide": operator.truediv,
    "add": operator.add,
    "subtract": operator.sub,
}


def apply_raster_operation(
    sources: list[xr.DataArray],
    operation: str,
) -> xr.DataArray:
    """Apply an arithmetic operation across source rasters via left-to-right fold.

    Parameters
    ----------
    sources : list[xr.DataArray]
        Two or more rasters to combine.  Must be on the same grid
        (use ``align_rasters`` first if resolutions differ).
    operation : {"multiply", "divide", "add", "subtract"}
        Arithmetic operation to apply.

    Returns
    -------
    xr.DataArray
        Result of ``reduce(op, sources)``.

    Raises
    ------
    ValueError
        If *operation* is not one of the supported operations.
    """
    op_fn = _RASTER_OPS.get(operation)
    if op_fn is None:
        msg = f"Unsupported raster operation '{operation}'. Choose from {list(_RASTER_OPS)}"
        raise ValueError(msg)
    return functools.reduce(op_fn, sources)
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_data_access.py::TestAlignRasters tests/test_data_access.py::TestApplyRasterOperation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/data_access.py tests/test_data_access.py
git commit -m "feat: add align_rasters() and apply_raster_operation() for pixel-level math"
```

---

### Task 3: Wire `DerivedContinuousSpec` into `_process_batch()`

**Files:**
- Modify: `src/hydro_param/pipeline.py:586-913`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing integration test**

In `tests/test_pipeline.py`, add a test that mirrors the existing `DerivedCategoricalSpec` integration test pattern but for continuous output:

```python
def test_process_batch_derived_continuous(tmp_path, sample_fabric):
    """DerivedContinuousSpec multiplies source rasters before zonal stats."""
    from hydro_param.dataset_registry import DerivedContinuousSpec, VariableSpec

    # Create two synthetic source rasters
    src_a = tmp_path / "a.tif"
    src_b = tmp_path / "b.tif"
    _write_synthetic_tiff(src_a, value=2.0, ...)
    _write_synthetic_tiff(src_b, value=3.0, ...)

    var_specs = [
        VariableSpec(name="a", source_override=str(src_a)),
        VariableSpec(name="b", source_override=str(src_b)),
        DerivedContinuousSpec(
            name="product",
            sources=["a", "b"],
            operation="multiply",
            align_to="a",
        ),
    ]

    results = _process_batch(...)
    assert "product" in results
    # Product of uniform 2.0 * 3.0 = 6.0
    assert results["product"]["product_mean"].iloc[0] == pytest.approx(6.0)
```

The exact test setup should follow the existing `_process_batch` test helpers already in `test_pipeline.py`. Look at how `test_process_batch_local_tiff_passes_variable_source` or the derived categorical tests are structured and mirror them.

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_process_batch_derived_continuous -v`
Expected: FAIL — `DerivedContinuousSpec` is skipped in first pass but no second-pass handler exists

**Step 3: Write the implementation**

Three changes in `src/hydro_param/pipeline.py`:

**3a. Update imports** (line ~66-69):

Add `DerivedContinuousSpec` to the import from `dataset_registry`.

**3b. Update source retention logic** (line ~809-815):

The `needed_by_dc` check must also consider `DerivedContinuousSpec`:

```python
needed_by_dc = any(
    isinstance(dc, DerivedCategoricalSpec | DerivedContinuousSpec)
    and var_spec.name in dc.sources
    for dc in var_specs
)
```

**3c. Skip `DerivedContinuousSpec` in first pass** (line ~734):

The existing `isinstance(var_spec, DerivedCategoricalSpec): continue` check
must also skip `DerivedContinuousSpec`:

```python
if isinstance(var_spec, DerivedCategoricalSpec | DerivedContinuousSpec):
    continue
```

**3d. Add second-pass block** (after the dc_specs block, before `return results` at line ~913):

```python
    # Process derived continuous specs — same second-pass pattern as
    # categorical, but runs continuous (not categorical) zonal stats.
    from hydro_param.data_access import align_rasters, apply_raster_operation

    dcont_specs = [v for v in var_specs if isinstance(v, DerivedContinuousSpec)]
    if dcont_specs:
        import rioxarray  # noqa: F401

    for dcont_spec in dcont_specs:
        # Re-read source GeoTIFFs from disk
        source_das: list[xr.DataArray] = []
        template_da: xr.DataArray | None = None
        missing: list[str] = []
        for src_name in dcont_spec.sources:
            src_tiff = work_dir / f"{src_name}.tif"
            if not src_tiff.exists():
                missing.append(src_name)
                continue
            da = cast("xr.DataArray", rioxarray.open_rasterio(src_tiff))
            da = da.squeeze("band", drop=True)
            source_das.append(da)
            if src_name == dcont_spec.align_to:
                template_da = da

        if missing:
            msg = (
                f"Cannot derive continuous variable '{dcont_spec.name}': "
                f"missing source GeoTIFFs {missing} in {work_dir}. "
                f"This usually means the source variables failed to process "
                f"or were cleaned up prematurely."
            )
            raise FileNotFoundError(msg)

        if template_da is None:
            msg = (
                f"align_to source '{dcont_spec.align_to}' not found "
                f"among loaded sources for '{dcont_spec.name}'"
            )
            raise ValueError(msg)

        # Align all sources to template grid
        aligned = align_rasters(
            source_das, template_da, method=dcont_spec.resampling_method
        )
        del source_das
        gc.collect()

        # Apply arithmetic operation
        product_da = apply_raster_operation(aligned, dcont_spec.operation)
        del aligned
        gc.collect()

        # Save product raster and run continuous zonal stats
        product_tiff = work_dir / f"{dcont_spec.name}.tif"
        save_to_geotiff(product_da, product_tiff)
        del product_da

        df = processor.process(
            fabric=batch_fabric,
            tiff_path=product_tiff,
            variable_name=dcont_spec.name,
            id_field=config.target_fabric.id_field,
            statistics=ds_req.statistics,
            categorical=False,
            source_crs=entry.crs,
            x_coord=entry.x_coord,
            y_coord=entry.y_coord,
        )

        # Apply scale_factor if present
        if dcont_spec.scale_factor is not None:
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = df[numeric_cols] * dcont_spec.scale_factor
            logger.info(
                "Applied scale_factor %.4f to derived continuous variable %s",
                dcont_spec.scale_factor,
                dcont_spec.name,
            )

        results[dcont_spec.name] = df

        # Clean up product GeoTIFF; only delete source GeoTIFFs if no
        # remaining dcont_specs or dc_specs still need them.
        product_tiff.unlink(missing_ok=True)
        remaining_dcont = dcont_specs[dcont_specs.index(dcont_spec) + 1 :]
        for src_name in dcont_spec.sources:
            still_needed = any(
                src_name in other.sources
                for other in remaining_dcont
            )
            # Also check categorical specs haven't been cleaned up yet
            still_needed = still_needed or any(
                isinstance(dc, DerivedCategoricalSpec) and src_name in dc.sources
                for dc in var_specs
                if dc not in dc_specs  # already processed
            )
            if not still_needed:
                (work_dir / f"{src_name}.tif").unlink(missing_ok=True)
        gc.collect()
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_pipeline.py::test_process_batch_derived_continuous -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/pipeline.py tests/test_pipeline.py
git commit -m "feat: wire DerivedContinuousSpec into _process_batch() second pass"
```

---

### Task 4: Update stage2 variable resolution and first-pass skip logic

**Files:**
- Modify: `src/hydro_param/pipeline.py` (stage2 area, lines ~392-535)

**Step 1: Verify stage2 handles new spec type**

Check `stage2_resolve_datasets()` — it currently handles `DerivedCategoricalSpec`
by resolving its sources from the dataset's variables list. The same logic must
apply to `DerivedContinuousSpec`. Read the function, identify where
`DerivedCategoricalSpec` is referenced, and add `DerivedContinuousSpec` to
those same checks.

Also check the `nhgf_stac` first-pass skip at line ~663:
```python
if isinstance(var_spec, DerivedVariableSpec | DerivedCategoricalSpec):
```
This must also include `DerivedContinuousSpec`.

**Step 2: Update**

Add `DerivedContinuousSpec` alongside `DerivedCategoricalSpec` in all isinstance
checks in stage2 and the nhgf_stac first-pass handler.

**Step 3: Run full test suite**

Run: `pixi run -e dev pytest -x -q`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/hydro_param/pipeline.py
git commit -m "feat: handle DerivedContinuousSpec in stage2 resolution and nhgf_stac skip"
```

---

### Task 5: Final verification

**Step 1: Run full check suite**

```bash
pixi run -e dev check
pixi run -e dev pre-commit
```

**Step 2: Verify no regressions**

All existing tests must pass. New tests:
- `TestDerivedContinuousSpec` (6 model validation tests)
- `TestAlignRasters` (3 alignment tests)
- `TestApplyRasterOperation` (6 arithmetic tests)
- `test_process_batch_derived_continuous` (1 integration test)

**Step 3: Commit any final fixes**

If pre-commit or type-checking reveals issues, fix and commit.
