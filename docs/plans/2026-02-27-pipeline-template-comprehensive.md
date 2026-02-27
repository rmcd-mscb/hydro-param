# Comprehensive Pipeline Template Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update `generate_pipeline_template()` to produce a comprehensive config with all 7 tested datasets instead of a minimal DEM-only example.

**Architecture:** Single function rewrite in `project.py` + test updates. The template matches the tested `drb_2yr_pipeline.yml` structure but with generic placeholders for target_fabric and output.

**Tech Stack:** Python, YAML string templates, pytest

**Design doc:** `docs/plans/2026-02-27-pipeline-template-comprehensive-design.md`

---

### Task 1: Update tests for comprehensive template

**Files:**
- Modify: `tests/test_project.py:138-157`

**Step 1: Update existing tests and add new coverage**

Replace the `TestGeneratePipelineTemplate` class with tests that expect the comprehensive template:

```python
class TestGeneratePipelineTemplate:
    def test_template_is_valid_yaml(self):
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        assert parsed is not None
        assert "target_fabric" in parsed
        assert "datasets" in parsed
        assert "output" in parsed
        assert "processing" in parsed

    def test_template_references_data_dirs(self):
        content = generate_pipeline_template("test_project")
        assert "data/fabrics/" in content

    def test_template_uses_project_name(self):
        content = generate_pipeline_template("my_watershed")
        parsed = yaml.safe_load(content)
        assert parsed["output"]["sir_name"] == "my_watershed"

    def test_template_includes_all_dataset_strategies(self):
        """Template covers all 5 access strategies via 7 dataset entries."""
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        dataset_names = [d["name"] for d in parsed["datasets"]]
        expected = [
            "dem_3dep_10m",       # stac_cog
            "gnatsgo_rasters",    # stac_cog
            "polaris_30m",        # local_tiff
            "nlcd_osn_lndcov",    # nhgf_stac static
            "nlcd_osn_fctimp",    # nhgf_stac static
            "snodas",             # nhgf_stac temporal
            "gridmet",            # climr_cat
        ]
        assert dataset_names == expected

    def test_template_temporal_datasets_have_time_period(self):
        """SNODAS and gridMET entries include time_period."""
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        datasets_by_name = {d["name"]: d for d in parsed["datasets"]}
        assert "time_period" in datasets_by_name["snodas"]
        assert "time_period" in datasets_by_name["gridmet"]

    def test_template_nlcd_has_year(self):
        """NLCD entries include year field."""
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        datasets_by_name = {d["name"]: d for d in parsed["datasets"]}
        assert "year" in datasets_by_name["nlcd_osn_lndcov"]
        assert "year" in datasets_by_name["nlcd_osn_fctimp"]
```

Note: The old `test_template_references_data_dirs` asserted `data/land_cover/` — that assertion is removed because NLCD OSN doesn't use local paths.

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_project.py::TestGeneratePipelineTemplate -v`
Expected: `test_template_includes_all_dataset_strategies` FAILS (only `dem_3dep_10m` in current template), `test_template_temporal_datasets_have_time_period` FAILS, `test_template_nlcd_has_year` FAILS.

**Step 3: Commit**

```bash
git add tests/test_project.py
git commit -m "test: add comprehensive pipeline template assertions"
```

---

### Task 2: Rewrite generate_pipeline_template()

**Files:**
- Modify: `src/hydro_param/project.py:122-209`

**Step 1: Replace the function body**

Replace the entire `generate_pipeline_template()` function (lines 122–209) with:

```python
def generate_pipeline_template(project_name: str) -> str:
    """Generate a well-commented pipeline YAML config template.

    Produce a starter ``pipeline.yml`` with inline comments explaining
    each section (target fabric, domain, datasets, output, processing).
    The template includes all 7 tested dataset configurations covering
    all 5 data access strategies (stac_cog, local_tiff, nhgf_stac
    static, nhgf_stac temporal, climr_cat).

    Parameters
    ----------
    project_name : str
        Project name inserted into the ``output.sir_name`` field and the
        header comment.

    Returns
    -------
    str
        YAML content suitable for writing to ``configs/pipeline.yml``.

    See Also
    --------
    hydro_param.config.PipelineConfig : Schema the generated YAML must conform to.
    """
    return f"""\
# Pipeline configuration for {project_name}
#
# This file drives the hydro-param parameterization pipeline.
# Edit the sections below, then run:
#   hydro-param run configs/pipeline.yml
#
# For dataset details run:
#   hydro-param datasets list
#   hydro-param datasets info <dataset-name>

# --- Target Fabric ---
# The polygon mesh (catchments, HRUs, grid cells) to parameterize.
# Place your GeoPackage or Parquet file in data/fabrics/.
target_fabric:
  path: "data/fabrics/catchments.gpkg"   # GeoPackage, GeoParquet, or Shapefile
  id_field: "featureid"                  # Unique ID column in the fabric
  crs: "EPSG:4326"                       # CRS of the fabric file

# --- Domain (optional) ---
# By default, the pipeline uses the full extent of the target fabric.
# Uncomment to restrict processing to a spatial subset:
# domain:
#   type: bbox
#   bbox: [-76.5, 38.5, -74.0, 42.6]    # [west, south, east, north] in EPSG:4326

# --- Datasets ---
# Each entry references a dataset from the registry by name.
# Use 'hydro-param datasets list' to see available datasets.
# Remove or comment out datasets you don't need.
datasets:
  # --- Topography ---
  # 3DEP 10m DEM — elevation, slope, aspect (stac_cog via Planetary Computer)
  - name: dem_3dep_10m
    variables: [elevation, slope, aspect]
    statistics: [mean]

  # --- Soils ---
  # gNATSGO pre-summarized soil properties (stac_cog via Planetary Computer)
  - name: gnatsgo_rasters
    variables: [aws0_100, rootznemc, rootznaws]
    statistics: [mean]

  # POLARIS soil texture properties, 30m (local_tiff via remote VRT)
  - name: polaris_30m
    variables: [sand, silt, clay, theta_s, ksat]
    statistics: [mean]

  # --- Land Cover ---
  # NLCD Land Cover via NHGF STAC OSN — categorical fractions (nhgf_stac)
  - name: nlcd_osn_lndcov
    variables: [LndCov]
    statistics: [categorical]
    year: [2021]

  # NLCD Fractional Impervious via NHGF STAC OSN (nhgf_stac)
  - name: nlcd_osn_fctimp
    variables: [FctImp]
    statistics: [mean]
    year: [2021]

  # --- Snow ---
  # SNODAS daily snow — historical SWE (nhgf_stac temporal)
  - name: snodas
    variables: [SWE]
    statistics: [mean]
    time_period: ["2020-01-01", "2021-12-31"]

  # --- Climate ---
  # gridMET daily climate via OPeNDAP (climr_cat)
  # pr/tmmx/tmmn for forcing; srad/pet/vs for radiation and PET derivation
  - name: gridmet
    variables: [pr, tmmx, tmmn, srad, pet, vs]
    statistics: [mean]
    time_period: ["2020-01-01", "2021-12-31"]

# --- Output ---
# Where and how to write results.
output:
  path: "output"                          # Output directory (created automatically)
  format: netcdf                          # netcdf or parquet
  sir_name: "{project_name}"             # Name for the output file

# --- Processing ---
# Engine and batching options.
processing:
  engine: exactextract                    # exactextract or serial
  batch_size: 500                         # Number of features per spatial batch
"""
```

**Step 2: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_project.py::TestGeneratePipelineTemplate -v`
Expected: All 6 tests PASS.

**Step 3: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass (no regressions).

**Step 4: Run pre-commit checks**

Run: `pixi run -e dev check`
Expected: lint, format, typecheck, tests all pass.

**Step 5: Commit**

```bash
git add src/hydro_param/project.py
git commit -m "feat: comprehensive pipeline template with all tested datasets"
```

---

### Task 3: Verify init end-to-end

**Step 1: Test init in a temp directory**

```bash
cd /tmp && mkdir test_init && cd test_init
pixi run -e dev python -m hydro_param init .
cat configs/pipeline.yml
```

Verify the generated `pipeline.yml` contains all 7 dataset entries.

**Step 2: Verify pywatershed_run.yml is unchanged**

```bash
cat configs/pywatershed_run.yml
```

Verify it still has the category-key `datasets:` format (topography, landcover, soils), NOT pipeline-style dataset lists.

**Step 3: Clean up**

```bash
rm -rf /tmp/test_init
```
