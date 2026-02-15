# Hydrologic Model Parameterization System — Complete Design Document

**Version:** 5.3 — Gemini 2.5 Pro Review Integration
**Date:** 2026-02-14
**Author:** Rich McDonald / Claude (AI-assisted brainstorm)
**Synthesized from:** v1.0 (Foundation), v2.0 (Compute), v3.0 (Data Strategy), v3.1 (Soils), v4.0 (Landscape), v5.0 (Comprehensive Synthesis), v5.1 (Data Category Taxonomy), v5.2 (Project-Scoped Data Staging)

---

## 1. Project Brief

### 1.1 Problem Statement

Hydrologic models (PRMS, NWM, SUMMA, VIC, HEC-HMS) require extensive spatial parameterization — translating gridded and raster datasets of topography, soils, vegetation, and climate onto a model's computational "fabric" (watershed polygons + stream segments) or structured grid. This process is currently ad-hoc: each modeler writes bespoke scripts, discovers datasets manually, and manages the weight-generation and aggregation pipeline independently.

The NHM Parameter Database (NhmParamDb) — the existing product for 109,951 CONUS HRUs — was computed using a patchwork of poorly documented GIS preprocessing steps. The actual computation code is largely internal and ad-hoc. There is no publicly available, reproducible pipeline for regenerating parameters from source data.

### 1.2 Vision

Build a **configuration-driven parameterization system** that:

1. Accepts a **configuration file** (YAML/TOML) specifying the target fabric (GFv1.1 HRUs, HUC12s, NWM catchments, NextGen divides, or a user-defined grid), which datasets to process, and processing options (time period, statistics, output format)
2. Leverages **HyTEST data catalogs** (Intake + WMA STAC) and the **ClimateR catalog** as the dataset discovery layer
3. Uses **gdptools** as the core spatial processing engine (grid→polygon, grid→line, polygon→polygon, zonal stats)
4. Runs on a **laptop, workstation, or HPC** (Hovenweep/Tallgrass) via configurable compute backends
5. Produces **model-ready parameter files** for multiple targets (NHM-PRMS, NextGen/ngen, pywatershed, generic formats)

### 1.3 Data Categories

The source data landscape for hydrologic parameterization spans eight distinct categories, each with characteristic access patterns, processing strategies, and output shapes. See §6.10 for detailed processing profiles per category.

| # | Category | Purpose | Temporality | Geometry | Example Sources |
|---|----------|---------|-------------|----------|-----------------|
| 1 | **Topography / Terrain** | Elevation, slope, aspect, curvature, TWI, flow direction | Static | Raster | 3DEP, NED, MERIT DEM |
| 2 | **Hydrography / Drainage Network** | Stream geometry, connectivity, Strahler order, contributing area | Static | Vector | NHDPlus v2, NHDPlus HR, MERIT Hydro |
| 3 | **Soils** | AWC, texture, Ksat, bulk density, porosity, HSG, drainage class | Static | Raster/Tabular | POLARIS, gSSURGO, gNATSGO, STATSGO2 |
| 4 | **Land Cover / Vegetation*** | Cover type, canopy density, impervious fraction, LAI, rooting depth | Static* | Raster | NLCD, MODIS, LANDFIRE, CDL |
| 5 | **Geology / Subsurface** | Hydrogeologic class, permeability, aquifer type, depth to bedrock | Static | Raster/Vector | Gleeson global permeability, state surveys, USGS maps |
| 6 | **Climate / Atmospheric Forcing** | Precip, temperature, solar radiation, humidity, wind, PET | Time-Varying | Raster | DAYMET, gridMET, PRISM, NLDAS-2, ERA5 |
| 7 | **Snow / Cryosphere** | SCA depletion curves, SWE, precipitation phase, albedo dynamics | Mixed | Raster | SNODAS, MODIS snow products, DAYMET snowfall |
| 8 | **Surface Water Bodies / Depressions** | Lake/reservoir extents, depression storage fraction, depth | Static | Vector/Raster | NHD waterbodies, GRanD, NID |

**Notes:**

- \*Land Cover is static per epoch (NLCD 2019, 2021) but users may select an epoch or compute change metrics across epochs.
- **Temporality:** Categories 1–5 and 8 are essentially static (updated on decadal timescales at most). Category 6 is time-varying forcing data. Category 7 straddles both — some snow parameters are static calibration coefficients, others are derived from time-series analysis.
- **Calibration/Verification** is a *role*, not a data category. A Snow dataset (category 7) can serve both as a parameter source and a calibration target (e.g., SNODAS SWE for model evaluation). Calibration datasets are tagged with their primary category.
- **Hydrographic Fabric** is the *target* geometry for parameterization, not a source dataset to be processed. It does not appear as a data category.
- **Resolution hierarchy:** These datasets span orders of magnitude in resolution: 1m (3DEP lidar), 10m (gSSURGO raster), 30m (NLCD, NED), 100m (POLARIS), 1km (DAYMET, SNODAS), 4km (gridMET/PRISM). The aggregation-to-fabric step is inherently resolution-aware.
- **Dependency chains:** Some model parameters require data from multiple categories (e.g., TWI needs terrain + hydrography; curve numbers need soils + land cover). These cross-category dependencies are handled by the parameter derivation framework (§A.8), not the core processing pipeline.

### 1.4 Key Constraints

- Must work with polygon-based fabrics AND structured grids
- Weights are expensive to compute but reusable → cache/persist them
- CONUS-scale runs demand parallel processing; regional runs should work serially on a laptop
- System should be extensible to new datasets without code changes (config-only additions)

---

## 2. Existing Landscape & Strategic Positioning

### 2.1 Nobody Has Built This — But Many Have Built Pieces

The landscape breaks into two federal ecosystems plus independent academic projects. None does exactly what we're designing.

**NOAA/NextGen Ecosystem (R-centric):**

- **NOAA hydrofabric** (Mike Johnson, Lynker) — R meta-package: climateR (112K+ federated datasets), zonal (grid→polygon summaries), hydrofab, hfsubsetR. climateR docs explicitly note gdptools as the Python counterpart. No config-driven parameterization wrapper exists in either language.
- **CIROH NGIAB_data_preprocess** — Closest in spirit: end-to-end Python data prep for NextGen. But hardcoded to NextGen v2.2, uses pre-computed attributes, no config flexibility, no HPC story.
- **ngen-forcing / ForcingProcessor** — Forcing-specific (AORC, GFS, HRRR → NextGen catchments). Uses ExactExtract. Not general parameterization.

**USGS/NHM Ecosystem (Python-centric):**

- **pywatershed** (McCreight, Langevin) — Python PRMS reimplementation. The *model* our tool feeds. Active v2.0.0.
- **NHM-Assist** — Jupyter evaluation/visualization. Downstream consumer, not parameterization.
- **pyPRMS** — PRMS file I/O. We'd use for writing PRMS output format.
- **NhmParamDb** — The existing 109,951-HRU parameter set our tool would modernize/replace.
- **gdptools** — Our core spatial engine. Recognized Python counterpart to climateR.

**Independent Academic:**

- **Watershed Workflow** (Coon, ORNL) — Most comprehensive open-source parameterization tool. Published 2022. But targets mesh-based models (ATS), raw SSURGO, no cloud-native, no HPC, no config-driven, single-watershed focus.
- **pyGSFLOW**, **pytRIBS** — Model-specific setup tools validating the same need.

### 2.2 Gap Analysis

| Feature | WatershedWF | NOAA hydrofab | NGIAB_preproc | NhmParamDb | **THIS TOOL** |
|---|---|---|---|---|---|
| Config-driven | No (code) | No (code) | Partial | No | **YES** |
| Arbitrary fabric support | Mesh only | NextGen only | NextGen only | NHM only | **YES (any)** |
| Cloud-native data | No | Yes (R) | Partial | No | **YES (Zarr)** |
| Python | Yes | No (R) | Yes | N/A | **YES** |
| Parallel/HPC | No | No | No | N/A | **YES** |
| gdptools integration | No | N/A (R) | No (ExactExt) | N/A | **YES** |
| POLARIS soils | No | TBD | Pre-computed | No | **YES** |
| Multiple model outputs | No (ATS) | No (NextGen) | No (NextGen) | No (PRMS) | **YES** |
| Dataset registry/catalog | No | Yes (climateR) | Pre-built | No | **YES** |
| Weight caching | No | No | Pre-computed | N/A | **YES** |
| Reproducible pipeline | Yes (NB) | Partial | Partial | No | **YES (config)** |
| National-scale ready | No | Yes (R) | Yes | Yes | **YES** |

### 2.3 Strategic Positioning: The "Missing Middle" Layer

| Layer | Components |
|---|---|
| Data Sources | Cloud Zarr, STAC, etc. |
| ↓ **Data Access** | gdptools, pynhd, dataretrieval |
| ↓ **► Parameterization Engine ◄** | **THIS TOOL — the gap nobody has filled** |
| ↓ **Model I/O** | pyPRMS, pywatershed, ngen configs |
| ↓ **Model Execution** | PRMS, pywatershed, ngen |

**Don't rebuild:** Data catalogs (gdptools/climateR), hydrofabric subsetting (pynhd/pygeohydro), spatial processing core (gdptools), model I/O (pyPRMS, FloPy).

**Key positioning:** Name for generality (not NHM-specific). Build ON gdptools. Support NextGen output early. Lead with curated data products. Engage Johnson and McCreight early.

**Risks:** NOAA Python tools "coming soon," pre-computed NextGen attributes reducing demand, institutional inertia.

**From Watershed Workflow — emulate:** Peer-reviewed paper, Jupyter examples, clear stage separation. **Improve on:** National scale, cloud-native, HPC, config-driven, multiple formats, POLARIS.

---

## 3. Foundation: HyTEST & gdptools Assessment

### 3.1 HyTEST

HyTEST is a community/workflow repository providing data catalogs, workflow patterns, and compute environment templates.

**Data Catalogs:** Intake (~30+ entries: CONUS404, NHM-PRMS output, NWM2.1, NWIS, GFv1.1, WBD HUC12s, GAGES-II, LCMAP), WMA STAC (gridded datasets), ClimateR (1700+ datasets via gdptools ClimRCatData). Storage: OSN (free), AWS S3 (requester-pays), USGS Caldera (HPC).

**Workflow Patterns:** Spatial aggregation tutorials, Dask cluster recipes, temporal aggregation, model evaluation (StdSuite, D-Score), chunking best practices.

**What HyTEST does NOT provide:** No unified pipeline, no config automation, no static parameter coverage, no standardized model output. Catalogs in Intake→STAC transition.

> HyTEST is the data discovery/access layer. What's missing is automation, configuration, and pipeline orchestration.

### 3.2 gdptools

The core spatial processing engine.

**Data Input Classes:** ClimRCatData (ClimateR catalog, auto-metadata), NHGFStacData (NHGF STAC), UserCatData (any xarray dataset), UserTiffData (GeoTIFF/VRT/COG).

**Processing Classes:** WeightGen (grid→polygon weights), WeightGenP2P (polygon→polygon), AggGen (area-weighted temporal aggregation), InterpGen (grid→polyline), ZonalGen (zonal stats), WeightedZonalGen (combined).

**Statistical Methods:** mean, std, median, count, min, max, masked variants.

**Compute Engines:** serial, parallel (joblib), dask, exactextract (C++ zonal).

**Output:** CSV, Parquet, NetCDF, JSON.

**What gdptools does NOT provide:** No pipeline orchestration, no config interface, no weight caching, no model output formatters, no grid-to-grid regridding, no parameterization recipes.

> gdptools handles the hard spatial computation. We need an orchestration layer above it.

---

## 4. System Architecture

### Configuration File (YAML/TOML)

| Parameter | Options |
|---|---|
| `target_fabric` | GFv1.1 \| HUC12 \| NextGen \| user grid |
| `domain` | bbox \| HUC-id \| gage-id \| shapefile |
| `datasets` | list of dataset specs |
| `output` | format, path, model_target |
| `compute` | engine, workers, scheduler |

### Parameterization Engine Pipeline

| Stage | Step | Sources / Methods |
|---|---|---|
| **1. Resolve Target Fabric** | Load target geometry | HyTEST catalog → GeoDataFrame |
| | | pynhd/WaterData → WBD, NHDPlus, NextGen |
| | | User shapefile / grid definition |
| **2. Resolve Source Datasets** | Query Dataset Registry | Native Zarr (CONUS404, GridMET, Daymet) |
| | | Virtual Zarr (MODIS, SNODAS via Kerchunk) |
| | | Curated Zarr (POLARIS, NLCD, DEM on OSN) |
| | | COGs (NLCD if not converted) |
| | | Local files (GeoTIFF, NetCDF) |
| **3. Compute/Load Weights** | Spatial intersection weights | Check weight cache (hash of src+tgt+CRS) |
| | | gdptools WeightGen / WeightGenP2P |
| | | Store weights for reuse |
| **4. Process Datasets** | Apply spatial processing | Time-varying: AggGen (grid→poly) |
| | | Static raster: ZonalGen (zonal stats) |
| | | Stream attributes: InterpGen (grid→line) |
| | | Poly→Poly: WeightGenP2P + crosswalk |
| **5. Format Output** | Write model-ready files | Generic (Parquet, NetCDF, CSV) |
| | | PRMS parameter file (via pyPRMS patterns) |
| | | NextGen realization config |
| | | pywatershed parameter set |
| | | Other model formats |

**Fabric flexibility (open question):** Treat grids as special case of polygons (rasterize to cell polygons) or separate pathway? For NHG (1km MODFLOW grid), polygon approach works. For high-res regular grids, xesmf-based regridding may be more efficient.

---

## 5. Compute & Parallelism Strategy

### 5.1 Honest Assessment of Dask

The concern about Dask dependency management is valid. **The problem is largely `distributed`, not `dask` core.** Dask.array/dask.dataframe with threaded scheduler has few dependencies and is rock-solid.

**Shines:** Lazy xarray on huge Zarr (9TB CONUS404), unmatched xarray integration, gdptools already supports `engine="dask"`, HyTEST has Hovenweep/Tallgrass configs.

**Hurts:** Dependency hell on HPC, inscrutable debugging, fragile memory management (OOM), overhead exceeds computation under 10-50GB, groupby+apply doesn't parallelize well, "works on laptop fails on HPC" syndrome.

### 5.2 Alternatives

**Joblib** — What gdptools uses internally. Zero dependency drama, dead simple mental model, works identically laptop↔HPC node. Single-node only.

**SLURM Job Arrays** — Embarrassingly parallel by spatial unit (textbook case). Zero Python dependency overhead, own memory per task, scales to thousands of nodes, built-in fault tolerance, easy resume. HPC only.

**concurrent.futures** — Stdlib, good for orchestration and I/O-bound. Not ideal for compute-heavy spatial work.

**Emerging (watch):** Cubed (bounded-memory Dask alt), Polars (Rust DataFrames for tabular post-processing), Python 3.13+ free-threading (no-GIL) — now officially supported in 3.14, single-thread overhead ~5-10%, NumPy compatible; GIL-disabled-by-default expected ~2027-2028, broader scientific stack (xarray, geopandas) still catching up.

### 5.3 The Pragmatic Hybrid: Right Tool for Each Phase

| Phase | Compute Pattern | Best Tool |
|---|---|---|
| 1. Data Discovery | I/O-bound, sequential | Standard Python |
| 2. Data Reading | I/O-bound, chunked | xarray + Dask (lazy only) |
| 3. Weight Compute | CPU-bound, parallel | joblib (laptop) / SLURM (HPC) |
| 4. Aggregation | CPU+I/O, parallel | joblib (laptop) / SLURM (HPC) |
| 5. Zonal Stats | CPU-bound, parallel | exactextract or joblib |
| 6. Assembly | I/O-bound, sequential | pandas/xarray merge |

### 5.4 Key Principle: "Chunk by Space, Not by Time"

Weight generation has no time dimension. Aggregation and zonal stats are independent per spatial unit. The natural parallelism axis is **spatial partitioning**: divide domain into batches of HRUs, process each independently, assemble.

**Spatial Partitioning Pattern:**

| Step | Batch 1 | Batch 2 | ... | Batch N |
|---|---|---|---|---|
| **Partition** | HRU 1–500 | HRU 501–1000 | ... | HRU ...–end |
| **Subset grid** | bbox for batch | bbox for batch | ... | bbox for batch |
| **Compute weights** | gdptools WeightGen | gdptools WeightGen | ... | gdptools WeightGen |
| **Aggregate** | AggGen / ZonalGen | AggGen / ZonalGen | ... | AggGen / ZonalGen |
| **Write** | partition file | partition file | ... | partition file |
| | ↘ | ↓ | | ↙ |
| **Assembly** | | Merge all partitions | | |

### 5.5 Spatial Batching: The Missing Tool

The spatial partitioning strategy above (§5.4) assumes that batches of HRUs are spatially contiguous, so each batch's bounding box is compact enough to efficiently subset the source raster. In practice, this assumption often fails. A target fabric like GFv1.1 with 110K HRUs may be ordered by Pfafstetter codes, sequential database IDs, or arbitrary shapefile row order — none of which guarantee spatial locality. Naively slicing `HRUs[0:1000]` can produce a batch scattered across CONUS, where the bounding box covers the entire domain, defeating the purpose of spatial subsetting.

**The problem, concretely:** If batch 7 contains HRUs in Maine, Texas, and Oregon, the source data read for that batch must span the entire CONUS raster. Every batch reads nearly the full dataset. The spatial chunking strategy collapses to "read everything, process a few HRUs" — the worst of both worlds.

**What we need:** A preprocessing step that reorders (or groups) the target fabric features into spatially contiguous batches, such that each batch's bounding box is minimal. This is a one-time operation per fabric, and the resulting batch assignments can be cached alongside the weight files.

#### 5.5.1 Available Approaches

**Approach 1: Hilbert Curve Sorting (dask-geopandas)**

The `dask-geopandas` library implements `spatial_shuffle()` which sorts a GeoDataFrame by Hilbert curve distance, then repartitions into spatially coherent chunks. The Hilbert curve maps 2D space to 1D while preserving spatial locality — features that are nearby in geographic space get similar Hilbert distances.

```python
import dask_geopandas

d_fabric = dask_geopandas.from_geopandas(fabric_gdf, npartitions=10)
d_sorted = d_fabric.spatial_shuffle(by="hilbert", npartitions=100)

# Each partition is now a spatially contiguous batch
for partition_id in range(d_sorted.npartitions):
    batch = d_sorted.get_partition(partition_id).compute()
    bbox = batch.total_bounds  # tight bounding box
```

This is the closest existing Python tool. The Hilbert sort is fast (~seconds for 110K polygons), produces excellent spatial locality, and integrates with the geopandas ecosystem. However, it's designed for dask's partition model, not for generating standalone batch assignment files.

**Approach 2: Hierarchical Grouping (HUC-based)**

For hydrologic fabrics with hierarchical IDs, grouping by parent HUC produces spatially contiguous batches with hydrologically meaningful boundaries. Group by HUC4 (~220 batches for CONUS) or HUC8 (~2,200 batches). Works beautifully for USGS fabrics, but fails for arbitrary research domains or non-USGS hydrofabrics.

**Approach 3: Regular Grid Overlay**

Overlay a regular grid (e.g., 2° × 2° tiles) on the fabric extent and assign each feature to its tile based on centroid location. Simple, fast, reproducible. But tiles at domain edges may have very few features, creating unbalanced batches.

**Approach 4: R's `chopin` Package (Cross-Language Reference)**

The R package `chopin` (rOpenSci, Song & Messier 2025) directly addresses this problem with `par_pad_grid()`, `par_pad_balanced()`, and `par_hierarchy()`. It generates padded grid partitions for parallel spatial computation, handles boundary objects, and supports H3/DGGRID discrete global grids. This is the most complete existing implementation of the concept, but it's R-only (`terra`/`sf`). Its design is worth studying — particularly the padding strategy for handling features near partition boundaries.

**Approach 5: Balanced KD-Tree Recursive Bisection**

Use recursive median bisection of feature centroids along alternating axes to produce balanced, spatially compact batches:

```python
def recursive_bisect(centroids, indices, depth=0, max_depth=7):
    if depth >= max_depth or len(indices) < 500:
        return [indices]
    axis = depth % 2
    coords = centroids[indices, axis]
    median = np.median(coords)
    left = indices[coords <= median]
    right = indices[coords > median]
    return (recursive_bisect(centroids, left, depth+1, max_depth) +
            recursive_bisect(centroids, right, depth+1, max_depth))
```

Guarantees balanced batch sizes (within 2x) and tight bounding boxes. Works for any geometry type. ~50 lines of pure numpy/scipy.

#### 5.5.2 Assessment: Build or Adopt?

| Approach | Spatial Locality | Balance | Generality | Dependencies |
|---|---|---|---|---|
| Hilbert sort (dask-geopandas) | Excellent | Good | Any geometry | dask-geopandas |
| HUC hierarchy | Excellent | Variable | HUC fabrics only | None |
| Regular grid overlay | Good | Variable | Any geometry | numpy |
| `chopin` (R) | Good | Good | Any geometry | R, terra, sf |
| KD-tree bisection | Very good | Excellent | Any geometry | scipy |

**Recommendation: Build a lightweight Python utility, but don't overthink it.**

The core function is ~50 lines and solves a problem encountered every time the pipeline runs on a new fabric. It belongs in the parameterization engine's spatial utilities module, not as a separate package. But it's foundational — every downstream step depends on batch quality.

**Proposed interface:**

```python
def spatial_batch(
    gdf: gpd.GeoDataFrame,
    method: str = "hilbert",       # hilbert, kdtree, grid, hierarchy
    n_batches: int = 100,
    hierarchy_field: str = None,   # for hierarchy method
    min_batch_size: int = 50,
    max_batch_size: int = 5000,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Assign spatially contiguous batch IDs to a GeoDataFrame.

    Returns:
        gdf with 'batch_id' column added
        summary DataFrame: batch_id → bbox, count, area
    """
```

**Batch assignments are cached** (same ID-based strategy as weight caching, §A.2):

```
batches/
  GFv1.1__hilbert_100/
    batch_assignments.parquet    # hru_id → batch_id
    batch_summary.parquet        # batch_id → bbox, count
    metadata.json
```

#### 5.5.3 Why This Is the Multiplier

The spatial batching step is what makes the entire chunk-by-space paradigm work:

| Without Spatial Batching | With Spatial Batching |
|---|---|
| Each batch bbox ≈ full CONUS | Each batch bbox ≈ 3° × 3° tile |
| Source read: ~26 billion cells (100m POLARIS) | Source read: ~26 million cells |
| I/O bottleneck dominates, CPU underutilized | I/O scales with CPU (1000x reduction) |
| Parallelization helps CPU only | Parallelization helps CPU **and** I/O |

Without spatial batching, parallelization provides CPU scaling but I/O stays constant. With spatial batching, both CPU and I/O scale. This is the difference between "parallel processing that feels fast" and "parallel processing that actually is fast."

### 5.6 Where Dask Still Fits

Dask remains best for **one specific thing**: efficiently reading spatial subsets from huge Zarr stores. The `.load()` call is the boundary — Dask handles smart I/O, then pure numpy/gdptools for computation.

### 5.6 Compute Backend Design

```yaml
compute:
  backend: "auto"        # auto | joblib | slurm | serial
  n_workers: -1          # -1 = all cores
  batch_size: 100        # HRUs per batch
  slurm:
    partition: "cpu"
    time: "04:00:00"
    mem_per_task: "16G"
    array_size: 50
  data_reading:
    engine: "zarr"
    chunk_strategy: "spatial_bbox"
```

```python
class ComputeBackend:
    def map_batches(self, func, batches, **kwargs):
        raise NotImplementedError

class SerialBackend(ComputeBackend):
    def map_batches(self, func, batches, **kwargs):
        return [func(batch) for batch in batches]

class JoblibBackend(ComputeBackend):
    def map_batches(self, func, batches, **kwargs):
        from joblib import Parallel, delayed
        return Parallel(n_jobs=self.n_workers)(
            delayed(func)(batch) for batch in batches)

class SlurmArrayBackend(ComputeBackend):
    def map_batches(self, func, batches, **kwargs):
        # Serialize batches → SLURM array script → submit → collect
        ...
```

**Two-level parallelism:** Level 1 (outer) partitions domain into batches via SLURM or sequential. Level 2 (inner) gdptools uses joblib for multi-core within each batch.

### 5.7 Dependency Comparison

| Approach | Core Dependencies | Risk Level |
|----------|------------------|------------|
| Serial | xarray, geopandas, gdptools | Minimal |
| Joblib | + joblib (ships w/ sklearn) | Low |
| Dask (lazy read) | + dask (core only) | Low |
| Dask (distributed) | + distributed, tornado, bokeh, msgpack... | **High** |
| SLURM array | (no additional Python deps) | Minimal |

### 5.8 Cloud Compute: An Alternative to HPC

Since many of the source datasets (CONUS404, GridMET, Daymet, NWM) already live on cloud object storage (AWS S3, OSN, Google Cloud Storage), there's a compelling argument for bringing compute to the data rather than pulling data to HPC or a desktop. This section evaluates cloud options as a third compute tier alongside laptop/workstation and USGS HPC.

#### 5.8.1 The Data Colocation Argument

The core principle is simple: **move compute to the data, not data to the compute.** When datasets are already on S3 in `us-west-2`, spinning up EC2 instances in the same region means S3→EC2 transfer is free (within the same region, with proper VPC endpoint configuration). This eliminates the biggest bottleneck in HPC workflows — transferring terabytes from cloud storage to Caldera/Hovenweep scratch space.

| Scenario | Data Transfer Cost | Latency |
|---|---|---|
| S3 → EC2 (same region) | **Free** | ~10 GB/s |
| S3 → USGS HPC (Hovenweep) | Egress: $0.09/GB first 10 TB | Hours to stage |
| S3 → Desktop (internet) | Egress: $0.09/GB first 10 TB | Variable, slow |
| OSN → anywhere | **Free** (public bucket) | Good, but not colocated |
| Cross-region S3 → EC2 | $0.02/GB | Good |

For a CONUS-scale parameterization run touching ~500 GB of source data, the difference between free colocated access and downloading to HPC ($45 in egress alone, plus hours of staging time) is significant.

#### 5.8.2 Cloud Platform Options

**Managed Dask: Coiled**

Coiled deploys Dask clusters in your own AWS/GCP account with automatic package sync, spot instance support, and auto-shutdown. This is the most natural fit for our xarray/Dask-based workflow.

| Feature | Details |
|---|---|
| Pricing | $0.05/CPU-hour (Coiled fee) + underlying AWS costs |
| Free tier | $25/month (~500 CPU-hours) |
| Spot support | Yes, with automatic fallback to on-demand |
| ARM support | Yes (20-30% cheaper) |
| Package sync | Mirrors local conda/pip environment to workers |
| Integration | Native xarray, Dask, joblib support |
| Startup time | ~2 minutes for cluster |

Example: A 50-worker Dask cluster (spot, ARM, `us-west-2`) processing CONUS-scale parameterization might cost $5-15/run including Coiled fees and AWS compute. Compared to: free on USGS HPC but with job queue waits and staging overhead, or days on a desktop.

**Serverless Functions: Modal Labs**

Modal offers a Python-native serverless platform where you decorate functions and they run in the cloud. Per-second billing, sub-second cold starts, and a $30/month free tier.

| Feature | Details |
|---|---|
| Pricing | ~$0.192/CPU-hour; $30/month free tier |
| DX | Pure Python decorators, no YAML/Docker |
| Scaling | 0 to thousands of containers automatically |
| Geospatial fit | Less natural than Coiled for xarray workflows; better for embarrassingly parallel batch jobs |
| Data access | Runs in their cloud account; S3 access via IAM roles |

Modal would be a good fit for the embarrassingly parallel batch pattern (§5.4) — decorate a `process_batch(hru_ids)` function and let Modal fan it out. But the impedance mismatch with xarray/Dask lazy evaluation makes it less natural for the data reading phase.

**Raw Cloud VMs: AWS EC2 / Batch**

Spinning up EC2 instances directly (or via AWS Batch for job arrays) offers full control and the lowest per-hour cost, but with more operational overhead.

| Instance | vCPUs | RAM | On-Demand | Spot (~70% off) |
|---|---|---|---|---|
| c7g.4xlarge (ARM) | 16 | 32 GB | ~$0.58/hr | ~$0.17/hr |
| c7g.8xlarge (ARM) | 32 | 64 GB | ~$1.16/hr | ~$0.35/hr |
| r7g.4xlarge (ARM, mem-opt) | 16 | 128 GB | ~$0.86/hr | ~$0.26/hr |
| m7g.8xlarge (ARM, general) | 32 | 128 GB | ~$1.31/hr | ~$0.39/hr |

AWS Batch with spot instances is conceptually identical to SLURM job arrays — define a job, set array size, AWS manages provisioning and queuing. Our `SlurmArrayBackend` design could have a parallel `AWSBatchBackend` with the same interface.

**USGS Cloud Hosting Solutions (CHS)**

USGS operates its own AWS-backed cloud environment (CHS) for internal use. This is relevant for the USGS institutional angle — work done under USGS auspices could potentially leverage CHS for production parameterization runs with data colocated in the same AWS infrastructure.

#### 5.8.3 Cost Comparison: Realistic CONUS Parameterization Run

Estimated costs for processing ~110K HRUs against 6 gridded datasets (soils, DEM, NLCD, climate forcings):

| Platform | Compute Cost | Data Transfer | Queue Wait | Setup Effort | Total Time |
|---|---|---|---|---|---|
| **Desktop** (32-core, 64 GB) | $0 (owned) | Egress if from S3 | None | Low | Days |
| **USGS HPC** (Hovenweep) | $0 (allocated) | Staging time | Hours–days | Medium (SLURM) | Hours (compute) |
| **Coiled + AWS Spot** (50 workers) | ~$5-15 | Free (same region) | None | Low (pip install) | 1-2 hours |
| **AWS Batch Spot** (50 c7g.4xl) | ~$8-20 | Free (same region) | Minutes | Medium (AWS config) | 1-2 hours |
| **Modal** (batch functions) | ~$10-25 | Possible egress | None | Low (decorators) | 1-2 hours |

#### 5.8.4 When Cloud Beats HPC (and Vice Versa)

**Cloud wins when:**
- Source data is already on S3/cloud storage (data colocation advantage)
- You need on-demand bursting without job queue waits
- You're a non-USGS user (contractors, university collaborators) without HPC allocations
- Development iteration — spin up, test, tear down in minutes
- Reproducibility — anyone can run the same config on the same cloud data

**HPC wins when:**
- You have active allocations with no marginal cost (Hovenweep, Tallgrass)
- Data is already staged on Caldera scratch
- Jobs are very long-running (days) where spot interruption risk matters
- Institutional requirements mandate on-prem processing
- You need hundreds of nodes (HPC scales larger per-job than most cloud budgets)

**Desktop wins when:**
- Regional/watershed-scale runs (Delaware River Basin, single HUC2)
- Development and debugging
- No cloud credentials or HPC access needed

#### 5.8.5 Implications for Architecture

The compute backend abstraction (§5.6) should be extended:

| Backend | Target | Dependencies |
|---|---|---|
| `SerialBackend` | Desktop, small domains | None |
| `JoblibBackend` | Desktop/workstation, medium domains | joblib |
| `SlurmArrayBackend` | USGS HPC (Hovenweep, Tallgrass) | SLURM CLI |
| `CoiledBackend` | Cloud, burst, collaborators | coiled, dask |
| `AWSBatchBackend` | Cloud, production runs | boto3 |
| `ModalBackend` | Cloud, embarrassingly parallel | modal |

The config file gains a cloud section:

```yaml
compute:
  backend: "coiled"    # or aws_batch, modal, slurm, joblib, serial
  cloud:
    provider: "aws"
    region: "us-west-2"    # colocate with data
    instance_type: "c7g.4xlarge"
    spot: true
    n_workers: 50
    auto_shutdown: true
```

The key design principle: **the same config file and pipeline code should run on any backend.** A user developing on their laptop with `backend: serial` should be able to switch to `backend: coiled` and process CONUS in an hour.

#### 5.8.6 Recommendation

**Lead with cloud-optional, not cloud-required.** The system should work beautifully on a laptop for regional runs and on HPC for USGS production. But cloud support — especially Coiled — should be a first-class option, not an afterthought, because:

1. **Data colocation** with S3/OSN Zarr stores is the fastest path to processing
2. **Democratizes access** — any hydrologist with a laptop and AWS credentials (or Coiled's free tier) can run CONUS-scale parameterization
3. **Reproducibility** — "run this config on this cloud data" is more reproducible than "stage data to your local HPC scratch"
4. **Non-USGS collaborators** (contractors, universities) lack HPC allocations; cloud is the natural production environment

Coiled is the recommended starting point: Pythonic, Dask-native, free tier adequate for development, spot instances for production, and the `us-west-2` colocation with most USGS cloud data is a perfect fit.

### 5.9 Containerization: Reproducible Environments Across Platforms

#### 5.9.1 The Problem

The geospatial Python stack is one of the hardest scientific computing environments to install reproducibly. GDAL, PROJ, GEOS, and their Python bindings (rasterio, pyproj, fiona/geopandas) depend on system-level C/C++ libraries with version-sensitive interactions. Adding xesmf (which requires ESMF/C++) and gdptools compounds the problem. The result: "works on my laptop but not on HPC" is the norm, not the exception.

This directly undermines two of the project's core goals — reproducibility ("this config produced these parameters") and platform-agnosticism ("same config, any backend"). If the environment itself isn't portable, config-driven reproducibility is incomplete.

#### 5.9.2 Container Strategy: Optional but First-Class

Containers should follow the same principle as cloud compute (§5.8.6): **optional but first-class.** The system should work fine in a standard conda/pixi/uv environment for development and simple runs. But for production runs — especially on HPC and cloud — a containerized environment eliminates an entire class of failure modes.

| Platform | Container Runtime | Image Format | Notes |
|---|---|---|---|
| Desktop/Workstation | Docker or Podman | Docker image | Development, testing |
| USGS HPC (Hovenweep, Tallgrass) | Apptainer (formerly Singularity) | `.sif` file | No root required, standard on USGS HPC |
| AWS Batch / ECS | Docker | Docker image | Native container platform |
| Coiled | N/A (package sync) | — | Coiled mirrors local environment; container not needed |
| Modal | Docker | Docker image | Container-native platform |

**Why Apptainer, not Docker, on HPC:** Docker requires root (daemon-based architecture). HPC admins universally prohibit this. Apptainer (the Singularity successor, adopted by the Linux Foundation) runs unprivileged, integrates with SLURM, supports bind-mounting HPC filesystems, and can import Docker images directly. It is the de facto standard for containers on federal HPC systems.

#### 5.9.3 What Goes Wrong on HPC (and How Containers Help)

Common HPC environment failures that containers eliminate:

| Failure Mode | Without Container | With Container |
|---|---|---|
| GDAL/PROJ version mismatch | Broken CRS transforms, silent wrong results | Pinned and tested |
| Module conflicts (`module load` collisions) | Unpredictable | Isolated from host modules |
| Conda env corruption after system update | Re-solve, hours lost | Image unchanged |
| Different results on different nodes | Possible (shared lib drift) | Identical environment |
| New team member onboarding | Days of environment debugging | `apptainer pull` + run |

Common HPC container difficulties that **do not apply** to hydro-param:

| Difficulty | Why It Doesn't Apply |
|---|---|
| MPI host/container version matching | hydro-param uses joblib + SLURM arrays, not MPI |
| GPU driver compatibility | No GPU compute |
| Network fabric (InfiniBand) passthrough | Independent batch jobs, not distributed |
| Interactive GUI/display | Batch processing only |

This is a favorable position — hydro-param's embarrassingly parallel, batch-oriented architecture (§5.4) is the **easy case** for HPC containers. Each SLURM array task runs independently inside its own container instance with no inter-node communication.

#### 5.9.4 Image Architecture

A single `Dockerfile` produces a multi-stage image:

```dockerfile
# Stage 1: Geospatial base (GDAL, PROJ, GEOS)
FROM condaforge/miniforge3 AS base
RUN mamba install -y gdal proj geos rasterio fiona pyproj

# Stage 2: Python environment
FROM base AS runtime
COPY pyproject.toml .
RUN pip install "hydro-param[gdp,regrid,parallel]"

# Stage 3: Slim production image
FROM runtime AS production
# Remove build tools, reduce image size
```

**Image distribution:**

| Registry | Audience | Format |
|---|---|---|
| GitHub Container Registry (ghcr.io) | Public, CI/CD | Docker |
| USGS internal registry (if available) | USGS HPC users | Docker → `.sif` |
| Dockerhub (mirror) | Broad discovery | Docker |

**Apptainer conversion** (one command, run once per release):

```bash
apptainer pull hydro-param-v0.1.0.sif docker://ghcr.io/rmcd-mscb/hydro-param:0.1.0
```

The `.sif` file is a single immutable file (~3-5 GB for the full geospatial stack) that can be placed on shared HPC storage and used by all team members.

#### 5.9.5 Integration with Compute Backends

The `container` config is orthogonal to backend selection — any backend can optionally use a container:

```yaml
compute:
  backend: "slurm"
  container:
    enabled: true
    image: "hydro-param-v0.1.0.sif"
    bind_paths: ["/caldera", "/home", "/scratch"]
  slurm:
    partition: "cpu"
    time: "04:00:00"
    mem_per_task: "16G"
    array_size: 50
```

The `SlurmArrayBackend` generates job scripts that wrap execution in `apptainer exec`:

```bash
#!/bin/bash
#SBATCH --array=1-50
apptainer exec \
    --bind /caldera:/caldera \
    --bind /scratch:/scratch \
    hydro-param-v0.1.0.sif \
    python -m hydro_param.cli process-batch \
        --config run_config.yaml \
        --batch-id $SLURM_ARRAY_TASK_ID
```

For cloud backends, the same Docker image is used natively:

```yaml
compute:
  backend: "aws_batch"
  container:
    image: "ghcr.io/rmcd-mscb/hydro-param:0.1.0"
  cloud:
    region: "us-west-2"
    spot: true
```

For desktop/development, containers are unnecessary — a standard Python environment is simpler and allows faster iteration:

```yaml
compute:
  backend: "joblib"
  # no container section — runs in local Python environment
```

#### 5.9.6 CI/CD and Versioning

Container images should be:

1. **Built automatically** via GitHub Actions on each release tag
2. **Tagged by version** (`v0.1.0`, `v0.2.0`) — never `latest` only
3. **Tested in CI** — the test suite runs inside the container as part of the release pipeline
4. **Immutable** — a given tag always produces identical results

This means the full reproducibility chain becomes: config file (what to compute) + container tag (environment) + dataset versions (source data) = deterministic output.

#### 5.9.7 Recommendation

**Phase 1:** Provide a `Dockerfile` and document the Apptainer conversion workflow. Publish images to GHCR on releases. This is low effort and immediately useful for HPC users.

**Phase 2:** Integrate container awareness into the `SlurmArrayBackend` and `AWSBatchBackend` so the config file drives container execution automatically.

**Don't do:** Don't require containers for any workflow. Don't build separate images for different backends. Don't manage Apptainer builds in CI (let users convert from Docker). One Dockerfile, multiple runtimes.

---

## 6. Curated Data Layer

### 6.1 Three-Tier Strategy

| Category | Examples | Strategy |
|----------|----------|----------|
| **A. Already cloud-optimized** | CONUS404, GridMET, Daymet, ERA5, NWM 2.1 | Use in place |
| **B. Archival but accessible** | NLCD COGs, MODIS, SNODAS, NHDPlus | Virtualize or convert |
| **C. Scattered/legacy** | POLARIS tiles, custom delineations, derived products | Convert to curated Zarr |

### 6.2 Category A: Don't Touch

HyTEST/Pangeo maintain these. Already optimized Zarr on OSN/S3. Duplicating TBs costs money and creates maintenance burden. Register in catalog and consume. Exception: temporary staging on Caldera for HPC production runs.

### 6.3 Category B: Virtualization Sweet Spot

**VirtualiZarr / Kerchunk** create lightweight reference manifests (KB-MB) making existing NetCDF/HDF5/GeoTIFF look like Zarr via `xr.open_zarr()`, without copying data. Kerchunk has 2+ years production use at NOAA/NASA. VirtualiZarr being upstreamed into Zarr spec.

**Caveat:** Can't fix bad chunking in originals. Performance depends on source layout.

**Use for:** MODIS composites, SNODAS time series, PRISM daily — many-file legacy datasets.

### 6.4 Category C: Convert and Curate (Highest Value)

**Static rasters (POLARIS, NLCD, DEM, PRISM normals):** Convert to Zarr. One-time cost, static, adds real value (harmonized CRS, CF metadata, spatial chunking). Community products.

**Complex relational (SSURGO, NHDPlus attributes):** Leave as-is, access via geopandas/pynhd.

**COG question:** NLCD as COG may suffice (single-band, gdptools reads COGs). Soils/DEM with multiple variables benefits from Zarr conversion.

### 6.5 Unified Access Architecture

**Data Access Layer** — unified `xr.open_zarr` interface:

| Tier | Strategy | Datasets |
|---|---|---|
| **Native Zarr** (in place) | Use as-is from community stores | CONUS404, GridMET, Daymet, NWM 2.1, ERA5 |
| **Virtual Zarr** (Kerchunk refs) | Lightweight reference manifests | MODIS composites, SNODAS, PRISM daily |
| **Converted Zarr** (our S3/OSN) | Curated conversion from legacy formats | POLARIS, NLCD, DEM/slope, PRISM normals, custom fabrics |

All three tiers resolve through the **Dataset Registry (YAML)** to a uniform access API.

### 6.6 Dataset Registry Schema

```yaml
datasets:
  conus404_daily:
    strategy: native_zarr
    source: "s3://hytest/conus404/conus404_daily.zarr"
    variables: [T2D, RAIN, SNOW, U10, V10, Q2D, PSFC, SWDOWN, LWDOWN]
    spatial_coords: [x, y]
    crs: "EPSG:4326"
    category: climate_forcing
    temporal: true
    time_range: ["1979-10-01", "2023-09-30"]

  polaris_100m:
    strategy: converted_zarr
    source: "s3://our-bucket/curated/polaris_100m_mean.zarr"
    original_source: "http://hydrology.cee.duke.edu/POLARIS/"
    variables: [sand_pct, silt_pct, clay_pct, ksat, theta_s, theta_r,
                vg_alpha, vg_n, bc_lambda, bc_hb, bulk_density, ph, om]
    spatial_coords: [lon, lat]
    depth_layers: [0-5cm, 5-15cm, 15-30cm, 30-60cm, 60-100cm, 100-200cm]
    crs: "EPSG:4326"
    category: soils
    temporal: false
    license: "CC BY-NC 4.0"

  nlcd_2021:
    strategy: converted_zarr
    source: "s3://our-bucket/curated/nlcd_2021.zarr"
    original_source: "https://www.mrlc.gov/data/nlcd-2021-land-cover-conus"
    variables: [land_cover]
    crs: "EPSG:5070"
    category: land_cover
    temporal: false

  modis_ndvi:
    strategy: virtual_zarr
    source: "s3://our-bucket/references/modis_ndvi.json"
    reference_format: kerchunk
    variables: [NDVI, EVI]
    category: vegetation
    temporal: true
```

### 6.7 Conversion Pipeline

Reproducible scripts, each independently valuable:

| Script | Purpose |
|---|---|
| `convert_polaris.py` | POLARIS tiles → Zarr (log10 handling) |
| `convert_nlcd.py` | NLCD GeoTIFF → Zarr (or validate COG) |
| `convert_dem.py` | 3DEP tiles → CONUS Zarr + slope/aspect |
| `convert_prism.py` | PRISM normals → Zarr |
| `virtualize_modis.py` | MODIS HDF → Kerchunk references |
| `virtualize_snodas.py` | SNODAS → Kerchunk references |
| `validate_registry.py` | Check all entries accessible |

Each: download → QC → convert (documented chunking/compression) → CF metadata → upload → update registry.

### 6.8 Chunking & Storage

**Chunking:** Static rasters `{y: 512, x: 512}` (~1 MB/chunk float32). Time-varying `{time: 1, y: 512, x: 512}`.

**Storage costs:** OSN free (USGS), S3 ~$0.023/GB/mo, Caldera free (HPC). NLCD as Zarr ~15 GB = $0.35/mo S3 or free OSN. CONUS404 at 9TB — never copy.

### 6.9 Decision Flowchart

1. Already Zarr on cloud? → Register, use in place (A)
2. Static and <100 GB? → Convert to Zarr (C)
3. Static and >100 GB? → Virtualize with Kerchunk (B)
4. Many-file time series? → Virtualize (B)
5. Available as COG? → Use via gdptools UserTiffData
6. None of above? → Convert (C)

### 6.10 Category Processing Profiles

The eight data categories (§1.3) differ in their access patterns, processing requirements, and the shape of their dataset registry entries. Rather than enforcing a rigid per-category schema, the system uses a common `DatasetEntry` model with optional fields. The `category` field on each dataset entry serves as documentation and enables category-aware defaults and validation. This section documents the processing profile for each category.

| Category | Access Pattern | Processing Strategy | Category-Specific Fields | Output Shape |
|----------|---------------|---------------------|--------------------------|--------------|
| **Topography** | STAC COG, converted Zarr | ZonalGen (continuous) + derivation (slope/aspect from DEM) | `derived_variables`, `gsd` | Per-feature scalar statistics |
| **Hydrography** | Vector (GeoPackage, NHDPlus) | Attribute join, topology extraction | `topology_fields`, `connectivity` | Per-feature attributes, network graph |
| **Soils** | Converted Zarr (depth-resolved) | ZonalGen (continuous) + depth aggregation | `depth_layers`, `depth_weights` | Per-feature per-depth or depth-weighted |
| **Land Cover** | COG, converted Zarr | ZonalGen (categorical) | `class_map`, `reclassify` | Per-feature class fractions |
| **Geology** | Mixed raster/vector | ZonalGen or polygon overlay | `classification_scheme` | Per-feature categorical or continuous |
| **Climate** | Native Zarr, virtual Zarr | AggGen (time-varying grid→polygon) | `time_range`, `temporal_resolution`, `climatology_period` | Per-feature time series or summary stats |
| **Snow** | Native Zarr / STAC COG | ZonalGen (static products) or AggGen (time-varying) | `time_range`, `sca_curve_method` | Mixed: per-feature scalar + time series |
| **Water Bodies** | Vector (NHD), converted Zarr | Polygon overlay, attribute extraction | `waterbody_type`, `storage_curve` | Per-feature attributes |

**Design considerations:**

1. **Category does NOT determine processing strategy mechanically.** The `strategy` field on `DatasetEntry` (stac_cog, native_zarr, converted_zarr, local_tiff) controls data access. The `categorical` flag on variables controls ZonalGen mode. Category is a human-readable grouping that provides defaults and documentation, not a dispatch key. For example, both Topography and Soils use ZonalGen continuous mode, but Soils entries are expected to carry `depth_layers` while Topography entries carry `derived_variables`.

2. **Category-specific fields are optional, not enforced by schema.** The `DatasetEntry` Pydantic model includes optional fields like `depth_layers`, `time_range`, and `class_map`. The `category` value tells the user and tooling which fields are relevant. A validator could warn if a soils entry lacks `depth_layers`, but it should not fail — this preserves flexibility for unconventional dataset configurations.

3. **The `category` field should be a constrained enum.** In code, the `category` field on `DatasetEntry` will become a `Literal` type constrained to the eight canonical categories: `topography`, `hydrography`, `soils`, `land_cover`, `geology`, `climate`, `snow`, `water_bodies`. This catches typos, enables IDE completion, and provides a stable vocabulary for category-aware processing logic.

4. **Cross-category dependency chains.** Some model parameters require data from multiple categories. For example, the Topographic Wetness Index (TWI) depends on terrain (slope) and hydrography (contributing area). Soil-adjusted curve numbers depend on soils (HSG) and land cover (NLCD class). `soil_moist_max` depends on soils (AWC) and land cover (rooting depth). `tmax_allsnow` depends on climate (temperatures) and snow (phase observations). These cross-category dependencies are handled by the parameter derivation framework (§A.8), not by the dataset processing pipeline.

5. **Resolution hierarchy is handled at the processing level.** Datasets span orders of magnitude in resolution (1m lidar to 4km gridMET). The ZonalGen aggregation step is inherently resolution-aware — it operates on the intersection of source grid cells and target polygons regardless of source resolution. The spatial batching system (§5.5) ensures efficient I/O at any source resolution by keeping batch bounding boxes compact.

### 6.11 Data Staging: Library-Managed Caching

> **Design revision (v5.3):** An external review (Appendix B, item 2A) identified that the original "Project Directory" pattern introduced a stateful, local filesystem dependency conflicting with the "config-driven, run-anywhere" vision. The design now adopts **library-managed transparent caching** (pooch-style) instead of user-managed project scaffolding. The pipeline config is the single source of truth — no separate `project.yaml`.

The three-tier strategy (§6.1) addresses how datasets are *served* — cloud-native Zarr, virtual references, or curated conversions. But not every dataset has a clean cloud-native access path. Many fall into access tiers that require local staging before the pipeline can consume them:

| Access Tier | Description | Examples |
|-------------|-------------|----------|
| **Tier 1 — Streamable** | Cloud-native, queryable by bbox on-the-fly | gNATSGO on Planetary Computer (STAC), DAYMET/gridMET (OPeNDAP), 3DEP (web services), NLCD via WMS/WCS |
| **Tier 2 — Downloadable** | Available programmatically but must be fetched and staged locally | gSSURGO file geodatabases, SSURGO via Soil Data Access API, Gleeson global permeability raster, SNODAS daily archives, NHDPlus HR bulk GeoPackages |
| **Tier 3 — Manual/Awkward** | Custom download logic or manual acquisition required | State geological surveys, older STATSGO products, USGS pre-processed soil property GeoTIFFs on ScienceBase |

#### Transparent Caching (pooch-style)

Rather than requiring users to manage a project directory structure, the library manages data caching transparently. The user's config requests `dataset: polaris`. The system checks a configured cache path, and if missing/stale, fetches it (or streams it). Local files are a **caching detail**, not a required scaffold.

```python
class DataCache:
    """Library-managed transparent data cache."""

    def __init__(self, cache_dir: Path | None = None):
        # Default: ~/.cache/hydro-param or XDG_CACHE_HOME
        self.cache_dir = cache_dir or _default_cache_dir()

    def get(
        self,
        dataset_id: str,
        bbox: tuple[float, float, float, float],
        registry_version: str | None = None,
        **kwargs: object,
    ) -> Path:
        """Return path to cached data, fetching if needed.

        The cache key is derived from:

        - dataset_id
        - registry_version / dataset version
        - a canonicalized bbox (west, south, east, north) with fixed rounding
        - additional byte-affecting options in **kwargs (sorted by key)
        """
        cache_key = self._cache_key(
            dataset_id,
            bbox,
            registry_version=registry_version,
            **kwargs,
        )
        cached = self.cache_dir / cache_key
        if cached.exists() and not self._is_stale(cached):
            return cached
        return self._fetch_and_cache(
            dataset_id,
            bbox,
            cached,
            registry_version=registry_version,
            **kwargs,
        )

    def _cache_key(
        self,
        dataset_id: str,
        bbox: tuple[float, float, float, float],
        *,
        registry_version: str | None = None,
        **kwargs: object,
    ) -> str:
        """Deterministic key from dataset ID, version, canonical bbox, and options.

        Notes
        -----
        - Bbox is always interpreted as (west, south, east, north).
        - Coordinates are rounded to a fixed number of decimal places so that
          equivalent floating-point values serialize to the same string.
        - Additional parameters that affect the bytes on disk are included as
          sorted key=value segments to avoid accidental cache collisions.
        - No geometry hashing is used; the key is a composite stable identifier.
        """
        west, south, east, north = bbox

        def _round_coord(value: float, ndigits: int = 6) -> str:
            return f"{value:.{ndigits}f}"

        bbox_str = "_".join(
            [
                _round_coord(west),
                _round_coord(south),
                _round_coord(east),
                _round_coord(north),
            ]
        )

        version = registry_version or "unversioned"

        # Include additional, byte-affecting options in a stable, sorted way.
        extra_parts: list[str] = []
        for key in sorted(kwargs):
            value = kwargs[key]
            extra_parts.append(f"{key}={value}")
        extra_str = "__".join(extra_parts) if extra_parts else "default"

        return f"{dataset_id}__v={version}__bbox={bbox_str}__opts={extra_str}"

    def _is_stale(self, path: Path) -> bool:
        """Check manifest for staleness."""
        ...

    def _fetch_and_cache(
        self,
        dataset_id: str,
        bbox: tuple[float, float, float, float],
        dest: Path,
        *,
        registry_version: str | None = None,
        **kwargs: object,
    ) -> Path:
        """Download, log provenance (including version + bbox), and return path."""
        ...
```

The cache directory structure is managed by the library, not the user:

```text
~/.cache/hydro-param/          # or user-configured path
├── raw/
│   ├── polaris_100m__bbox_-109.5_37.0_-105.5_40.5/
│   │   ├── data.zarr
│   │   └── manifest.json    # provenance: source URL, download time, checksum
│   ├── gleeson_permeability__bbox_-109.5_37.0_-105.5_40.5/
│   │   ├── data.tif
│   │   └── manifest.json
│   └── ...
└── weights/                   # weight cache (§A.2)
    └── ...
```

The pipeline config controls everything — no separate project config:

```yaml
# Pipeline config IS the single source of truth
target_fabric:
  path: "data/upper_colorado/catchments.gpkg"
  id_field: "featureid"
  crs: "EPSG:4326"

domain:
  type: bbox
  bbox: [-109.5, 37.0, -105.5, 40.5]

datasets:
  - name: polaris_100m       # system checks cache, fetches if needed
  - name: nlcd_2021
  - name: dem_3dep_10m

cache:
  path: "~/.cache/hydro-param"   # optional override; default is XDG_CACHE_HOME
```

#### Design Principles

1. **No staging barrier.** A user switching from desktop to cloud should not have to "stage" data to a project directory first. The cache is transparent — data flows to wherever compute runs.

2. **Provenance tracking is still critical.** Each cached file's `manifest.json` records: source URL, download timestamp, dataset version/vintage, checksum, and the bbox used. This makes results reproducible and flags stale data.

3. **Idempotency.** Requesting a dataset twice detects existing cached data and skips re-download. If the domain bbox changes, the cache key changes and a new fetch occurs. Cache keys MUST be derived from a deterministic, canonical representation of only the parameters that affect download scope (e.g., fabric stable ID, dataset ID/version, CRS, domain type/bounds, and method), and then optionally hashed with a stable cross-platform algorithm (such as SHA-256 over a JSON serialization with sorted keys). Do not use language- or runtime-dependent object hashes, and do not hash raw geometry coordinates; geometries participate in cache keys only via their stable IDs and CRS.

4. **Cache is disposable.** Deleting `~/.cache/hydro-param/` simply triggers re-downloads on next run. No user data is lost.

5. **Relationship to the dataset registry.** The dataset registry (§6.6) describes *what* data exists and how to access it. The cache is *where* local copies live. A `local_tiff` or `local_file` strategy in the registry can point to user-provided paths, bypassing the cache entirely for pre-staged data.

6. **CLI for cache management.** Simple commands for inspection, not scaffolding:

```bash
# Show what's cached
hydro-param cache --status

# Clear stale entries
hydro-param cache --clean

# Clear everything
hydro-param cache --purge
```

---

## 7. Soils Data: The POLARIS Recommendation

### 7.1 Soils Landscape

| Dataset | Resolution | Depth Layers | Hydraulic Props | Gap-Free | Format Pain |
|---------|-----------|-------------|----------------|----------|------------|
| STATSGO2 | ~1-10 km | Variable | Limited | Yes | High (relational) |
| gSSURGO | 10 m | Variable | Via pedotransfer | No | Very High (ESRI gdb) |
| gNATSGO | 10 m* | Variable | Via pedotransfer | Yes | Very High (ESRI gdb) |
| **POLARIS** | **30 m** | **6 standard** | **Direct (Ksat, VG, BC)** | **Yes** | **Moderate (tiles)** |
| SoilGrids | 250 m | 6 standard | Limited | Yes | Low (COG) |

### 7.2 Why POLARIS

30m (also 100m, 1km). 13 variables × 6 depth layers (0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm). Full CONUS, no gaps. Statistics: mean, p50, mode, p5, p95.

1. **Solves hard problems already** — SSURGO boundary discontinuities removed, gaps filled, relational structure resolved to clean rasters.
2. **Hydrologically meaningful properties** — Ksat, Van Genuchten α/n, Brooks-Corey λ/hb directly. No pedotransfer needed.
3. **USGS institutional buy-in** — NEHF/IWAAs selected POLARIS, published USGS data release at 100m/1km.
4. **Perfect Zarr conversion candidate** — ~9 billion cells in scattered 1°×1° tiles.
5. **Probabilistic estimates** — p5/p95 enable uncertainty-aware parameterization.

**Caveats:** R² ~0.41 vs in-situ. Some properties in log10 space. CC BY-NC 4.0.

**Secondary:** gNATSGO for 10m users. SoilGrids for global work.

### 7.3 POLARIS → Zarr Conversion

**Proposed `polaris_v1.zarr/` structure:**

| Variable | Dimensions | Units | Notes |
|---|---|---|---|
| `sand_pct` | (depth, y, x) — float32 | % | |
| `silt_pct` | (depth, y, x) | % | |
| `clay_pct` | (depth, y, x) | % | |
| `bulk_density` | (depth, y, x) | g/cm³ | |
| `organic_matter` | (depth, y, x) | % | converted from log10 |
| `ph` | (depth, y, x) | — | |
| `ksat` | (depth, y, x) | cm/hr | converted from log10 |
| `theta_s` | (depth, y, x) | m³/m³ | saturated water content |
| `theta_r` | (depth, y, x) | m³/m³ | residual water content |
| `vg_alpha` | (depth, y, x) | 1/kPa | converted from log10 |
| `vg_n` | (depth, y, x) | — | |
| `bc_lambda` | (depth, y, x) | — | Brooks-Corey λ |
| `bc_hb` | (depth, y, x) | kPa | converted from log10 |
| `lat`, `lon` | 1D coordinates | degrees | WGS84 |
| `depth` | 6 layers | cm | 0-5, 5-15, 15-30, 30-60, 60-100, 100-200 |

Chunking: `{depth: 6, y: 512, x: 512}`. CF-compliant attributes, WGS84 grid_mapping.

**Resolution options (with ~3-5x zstd compression):**

| Resolution | Raw Total (13 vars) | Compressed Est. |
|------------|-------------------|----------------|
| 30 m | ~1.7 TB | ~400 GB |
| 100 m | ~155 GB | ~40 GB |
| 1 km | ~1.5 GB | ~500 MB |

**Recommendation:** Start with 100m and 1km (matching NEHF). 100m adequate for HRU-scale (GFv1.1 HRUs avg ~30 km²). Add 30m later.

**Uncertainty:** Separate Zarr stores for mean, p5, p95 (simplest approach).

---

## 8. Dataset Prioritization

### Phase 1: Core Parameterization (MVP)

| # | Dataset | Action | Est. Size |
|---|---------|--------|-----------|
| 1 | GFv1.1 geofabric | Register (HyTEST) | — |
| 2 | CONUS404 daily | Register (OSN Zarr) | — |
| 3 | GridMET | Register (cloud Zarr) | — |
| 4 | **POLARIS soils (100m)** | **Convert → Zarr** | **~40 GB** |
| 5 | NLCD 2021 | Register (COG) or convert | ~15 GB |
| 6 | 3DEP DEM (30m) | Convert → slope/aspect/elev Zarr | ~50 GB |

### Phase 2: Extended Parameters

| # | Dataset | Action | Est. Size |
|---|---------|--------|-----------|
| 7 | POLARIS (30m) | Convert → Zarr | ~400 GB |
| 8 | POLARIS uncertainty | Separate Zarr (p5/p95) | ~80 GB |
| 9 | PRISM normals | Convert → Zarr | ~5 GB |
| 10 | MODIS LAI/NDVI | Virtualize (Kerchunk) | refs only |
| 11 | gNATSGO | Convert (selected props) | ~100 GB |
| 12 | Daymet | Register (existing Zarr) | — |

### Phase 3: Calibration/Verification

| # | Dataset | Action | Est. Size |
|---|---------|--------|-----------|
| 13 | NWIS streamflow | API (dataretrieval) | — |
| 14 | SNODAS SWE | Virtualize (Kerchunk) | refs only |
| 15 | MODSCAG snow cover | Virtualize | refs only |
| 16 | SMAP soil moisture | Register (cloud) | — |
| 17 | NHM-PRMS output | Register (OSN Zarr) | — |

---

## 9. USGS Institutional Alignment

- **IWAAs / NEHF:** Already published POLARIS-derived products. Our Zarr stores complement their depth-weighted averages with full depth-resolved cloud-native data.
- **NHM Infrastructure:** Needs exactly this — current PRMS parameterization is poorly documented patchwork. Config-driven system would be significant advance.
- **HyTEST:** Consume their catalogs, contribute POLARIS Zarr stores back.
- **National Hydrogeologic Grid (NHG):** 1km MODFLOW grid. Our arbitrary-grid support serves groundwater community too.
- **pywatershed:** McCreight et al. need parameterization tooling. Our tool producing pywatershed-ready parameters = immediately useful.

---

## Appendix A: Response to External Review (Gemini 3 Pro)

An independent review was conducted by GitHub Copilot (Gemini 3 Pro Preview) on the v5 synthesis document. The review returned a **GO** verdict and validated the core architecture as "superior to the existing ad-hoc scripts used in NhmParamDb."

**Validated strengths** (no action needed, confirms design direction):

- **Dask Realism** (§5.1–5.5): Reviewer called this "the strongest technical insight" — the distinction between Dask lazy I/O (essential) vs Dask distributed scheduling (fragile on HPC) was recognized as a "mature architectural decision."
- **Data Tiering** (§6): The three-tier strategy (native Zarr / virtual Zarr / converted Zarr) was validated as the correct approach to avoiding a "storage management nightmare."
- **Environmental Agnosticism** (§5.6–5.8): The `ComputeBackend` abstraction across laptop/HPC/cloud was called "excellent software design."
- **POLARIS Selection** (§7): Converting POLARIS to Zarr recognized as "a high-value community contribution that solves the SSURGO geometry mess upstream."

**Tactical recommendation from reviewer:** "If you execute the Data Layer (Tier C) transformations (POLARIS/NLCD to Zarr) first, you will deliver immediate value to the USGS before the main software is even finished." — **Adopted.** This aligns with the `hydro-param-data` separate-package strategy (§A.5) and becomes the recommended first deliverable.

Below we address each critical risk and recommendation raised.

### A.1 The "Grid as Polygon" Trap — RESOLVED

**Reviewer concern:** Converting high-resolution grids (30m DEM, 100m POLARIS) to polygons for intersection with target grids will choke gdptools due to massive vertex counts. Recommends bifurcating the pipeline.

**Response: Agree completely. This resolves open question #3.**

The pipeline needs two pathways based on target fabric type:

| Target Type | Source Type | Method | Tool |
|---|---|---|---|
| Polygon fabric (HRUs, HUC12s) | Gridded raster | Zonal stats / area-weighted | gdptools ZonalGen, AggGen |
| Polygon fabric | Polygon source | Polygon-to-polygon crosswalk | gdptools WeightGenP2P |
| Regular grid target | Gridded raster | Regridding / resampling | xesmf, rioxarray |
| Regular grid target | Polygon source | Rasterize then resample | rasterio, rioxarray |

This means the architecture diagram (§4) gains a decision node after step 1:

- **If target is polygon fabric** → gdptools pathway (WeightGen/AggGen/ZonalGen)
- **If target is regular grid** → raster pathway (xesmf conservative regridding, rioxarray reproject_match)

For the NHG 1km MODFLOW grid specifically, `xesmf` conservative regridding from POLARIS 100m → 1km is vastly more efficient than polygonizing 26 million grid cells. The `xesmf` approach also naturally handles area-weighted averaging.

**Design implication:** The `ProcessingStrategy` class needs a factory method:

```python
def get_processor(target_fabric, source_dataset):
    if target_fabric.is_regular_grid and source_dataset.is_gridded:
        return RegridProcessor(method="conservative")
    elif target_fabric.is_polygon:
        return GdptoolsProcessor(engine="zonalgen")
    ...
```

### A.2 Weight Caching Fragility — RESOLVED

**Reviewer concern:** Hashing geometry data for cache keys fails due to floating-point precision differences across platforms. Recommends hashing identifiers instead.

**Response: Agree. This resolves open question #7 (partially) and adds a new design constraint.**

The weight cache key should be a composite of stable identifiers, not geometry hashes:

```
cache_key = hash(
    target_fabric_id,      # e.g., "GFv1.1" or "HUC12_WBD_2024"
    target_fabric_version,  # version/date stamp
    source_dataset_id,      # e.g., "polaris_100m_mean"
    source_crs,             # e.g., "EPSG:4326"
    target_crs,             # e.g., "EPSG:5070"
    processing_method,      # e.g., "zonalgen_mean"
)
```

The cache directory structure becomes human-readable:

```
weights/
  GFv1.1__polaris_100m__zonalgen/
    weights.parquet
    metadata.json  # records all IDs, versions, date computed
  GFv1.1__conus404__agggen/
    weights.parquet
    metadata.json
```

The `metadata.json` records enough provenance to invalidate caches when upstream data changes. This is simpler and more robust than geometry hashing.

**Adopted:** Parquet for weight files (faster I/O, smaller, preserves dtypes) over CSV. This also resolves the format part of open question #7.

### A.3 The "Formatter" Bottleneck — ADOPTED AS PLUGIN ARCHITECTURE

**Reviewer concern:** Writing valid PRMS parameter files and NextGen realization configs is hard. Legacy formats are whitespace-sensitive and deeply nested. Recommends a plugin/adapter pattern with a standardized internal representation.

**Response: Agree strongly. This is the correct architecture.**

The core engine produces a **Standardized Internal Representation (SIR)** — an `xarray.Dataset` with CF-compliant metadata, consistent naming, and full provenance attributes.

> **Design revision (v5.3):** An external review (Appendix B, item 2D) identified that the SIR must be more than "an xarray Dataset" — it needs **strict schema validation** with enforced standard names and units. Without this, Model A might expect slope in degrees while Model B expects radians, leading to silent errors. The SIR producer (core engine) must guarantee a canonical set of variable names and units using a single naming pattern:
>
> - Variable names follow: `<base>_<unit>[_<aggregation>]`.
> - `<base>` is the physical quantity (e.g., `elevation`, `slope`, `soil_depth`).
> - `<unit>` is a canonical abbreviation in SI or accepted derived units (e.g., `m`, `deg`, `m_s`, `kg_m2`); dimensionless quantities omit the unit and use just `<base>[_<aggregation>]` (e.g., `curve_number_mean`).
> - `<aggregation>` is optional and encodes the statistic or operation (`mean`, `min`, `max`, `std`, `sum`, `pXX`, etc.) computed over space and/or time.
> - Examples: `elevation_m_mean`, `slope_deg_mean`, `soil_depth_m_min`, `lai_m2_m2_mean`, `curve_number_mean`. Older examples like `elevation_mean` and `slope_mean` are to be interpreted as `elevation_m_mean` and `slope_deg_mean` under this convention.
>
> A validation step ensures all SIR variables conform to this schema before passing to derivation plugins or formatters. See §A.8 for the complete variable list and any allowed exceptions.

Output formatters are adapters:

```python
class OutputFormatter:
    """Base class for model-specific output formatters."""
    def write(self, sir: xr.Dataset, output_path: Path, config: dict):
        raise NotImplementedError

class ParquetFormatter(OutputFormatter):
    """Generic tabular output."""
    ...

class PRMSFormatter(OutputFormatter):
    """PRMS parameter file writer. Leverages pyPRMS internally."""
    ...

class NextGenFormatter(OutputFormatter):
    """NextGen realization config writer."""
    ...

class PywatershedFormatter(OutputFormatter):
    """pywatershed parameter set writer."""
    ...
```

This also means third parties can write their own formatters (GSFLOW, VIC, SUMMA) without touching core code. The config specifies which formatter:

```yaml
output:
  format: "prms"           # or "nextgen", "pywatershed", "parquet", "netcdf"
  path: "./output/"
  model_version: "5.2.1"   # formatter may need to know model version
```

**Key insight from reviewer:** The PRMS formatter should use `pyPRMS` internally for the actual file writing. We don't reimplement PRMS file format quirks — we delegate to the existing library that already handles them.

### A.4 Config Scope: Keep It Declarative — ADOPTED

**Reviewer concern:** YAML/TOML configs can devolve into Turing-complete scripting if not disciplined. Configs should say *what*, not *how*.

**Response: Agree. This is a firm design principle.**

The config is **declarative only** — no variables, no conditionals, no loops, no templating. If a user needs conditional logic, they write a Python script using the library API:

```python
# This is fine — Python script using the library
import hydro_param as hp

config = hp.load_config("base_config.yaml")
config.datasets.append("polaris_30m")  # programmatic customization
hp.run(config)
```

```yaml
# This is fine — declarative YAML
target_fabric: "GFv1.1"
domain:
  type: "huc2"
  id: "02"            # Delaware River Basin
datasets:
  - polaris_100m
  - nlcd_2021
  - dem_30m
output:
  format: "prms"
```

```yaml
# This would NOT be supported — scripting in YAML
datasets:
  - if: "{{ resolution == 'high' }}"
    then: polaris_30m
    else: polaris_100m    # NO. Use Python for this.
```

### A.5 Dataset Registry as Separate Package — RESOLVED

**Reviewer concern:** The dataset registry should absolutely be a separate repo/package. It will be adopted by people who don't care about the parameterization tool.

**Response: Agree. This resolves open question #4.**

`hydro-param-data` (or similar name) becomes an independent package that:

1. Ships YAML registry files describing all curated datasets
2. Provides `open_dataset("polaris_100m")` → `xr.Dataset` helper functions
3. Handles storage backends (S3, OSN, local) transparently
4. Includes provenance metadata and citation helpers
5. Has zero dependency on the parameterization engine

This follows the pattern of `intake` catalogs and `pooch` for data fetching. Independent value proposition: any hydrologist or ML researcher can `pip install hydro-param-data` and immediately get clean xarray access to POLARIS, NLCD, DEM, etc. without ever using the parameterization system.

**Repository structure:**

```
hydro-param-data/          # standalone data access package
hydro-param/               # parameterization engine (depends on hydro-param-data)
```

### A.6 Partial Failure / Fault Tolerance — NEW REQUIREMENT

**Reviewer concern:** In 110K HRU runs, ~0.1% will fail (bad geometry, empty intersections, NaN data). The pipeline needs a tolerant mode with logging and patch runs.

**Response: Agree. This is a critical requirement we hadn't explicitly addressed.**

**Design:**

1. **Tolerant mode** (default for production): Failed HRUs are logged to `failed_hrus.csv` with error details (HRU ID, dataset, error type, traceback). Processing continues for all valid HRUs.

2. **Strict mode** (default for development): First failure raises an exception. Useful for debugging configs on small domains.

3. **Patch run**: A follow-up mode that reads `failed_hrus.csv` and retries only those HRUs, potentially with different settings (larger buffer, different interpolation method, relaxed NaN threshold).

```yaml
processing:
  failure_mode: "tolerant"     # or "strict"
  max_failure_pct: 1.0         # abort if >1% of HRUs fail
  failed_log: "./failed_hrus.csv"
  nan_threshold: 0.10          # allow up to 10% NaN in source data per HRU
```

**Common failure modes to handle gracefully:**
- HRU geometry too small / slivered (no source grid cells intersect)
- Source data has NaN/nodata at HRU location (coastal, border effects)
- CRS transformation edge cases (antimeridian, polar)
- Timeout on single HRU (abnormally large polygon, e.g., Great Lakes HRUs)

**Post-processing report:**

```
=== Parameterization Run Summary ===
Total HRUs:     109,951
Successful:     109,843 (99.9%)
Failed:         108 (0.1%)
  - Empty intersection:  47
  - NaN threshold:       38
  - Timeout:             23
Failed HRU log: ./failed_hrus.csv
```

### A.7 Summary of Review-Driven Resolutions

| Open Question | Resolution | Source |
|---|---|---|
| #3 Fabric flexibility (grids as polygons?) | **Bifurcate: gdptools for polygon targets, xesmf for grid targets** | Review §A.1 |
| #4 Dataset registry as separate package? | **Yes, independent `hydro-param-data` package** | Review §A.5 |
| #7 Weight file format? | **Parquet (not CSV). Cache key from IDs, not geometry hashes.** | Review §A.2 |
| (new) Output formatter architecture | **Plugin/adapter pattern with Standardized Internal Representation** | Review §A.3 |
| (new) Config philosophy | **Strictly declarative. Logic belongs in Python scripts.** | Review §A.4 |
| (new) Partial failure handling | **Tolerant mode + failed log + patch runs** | Review §A.6 |
| (new) Parameter derivation architecture | **Three-layer: SIR → model-specific derivation plugins → output formatters** | Review §A.8 |

### A.8 Parameter Derivation: SIR to Model Parameters

Section A.3 established the OutputFormatter plugin pattern for writing model-specific file formats. But there is a critical middle step between the SIR and the formatted output: **parameter derivation**. Many model parameters are not direct zonal statistics of source data — they are derived quantities that combine multiple SIR variables using model-specific formulas. This section clarifies the three-layer architecture.

```
Source Data  →  Core Pipeline (§4, §11)  →  SIR (xr.Dataset)
                                              Physical properties per feature
                                              (elevation_mean, clay_pct, nlcd_fractions...)
                                                │
                                                ▼
                                    Parameter Derivation Layer
                                    Model-specific plugin classes
                                                │
                                                ▼
                                    Derived Parameters (xr.Dataset)
                                    Model parameters per feature
                                    (carea_max, soil_moist_max, tmax_allsnow...)
                                                │
                                                ▼
                                    Output Formatter (§A.3)
                                    File format adapter
                                    (PRMS .param, NextGen YAML, Parquet...)
```

**A. The SIR contains physical properties, not model parameters.** The SIR should contain values like `elevation_mean`, `slope_mean`, `clay_pct_mean`, `nlcd_class_21_frac`. These are model-independent physical measurements of each feature in the target fabric. Anyone can use them regardless of which model they are parameterizing.

**A′. The SIR enforces a strict schema with standard names and units (v5.3).** Per external review (Appendix B, item 2D), the SIR is not just "an xarray Dataset" — it must validate against a declared schema:

- **Standard variable names:** A canonical vocabulary (e.g., `elevation_m`, `slope_deg`, `clay_pct`, `ksat_cm_hr`, `nlcd_class_21_frac`). Names encode both the physical quantity and the unit to prevent ambiguity.
- **Guaranteed units:** The SIR producer converts all source data to canonical units during processing. Slope is always in degrees, elevation in meters, Ksat in cm/hr, etc. Unit conversions happen once, at the SIR boundary.
- **Schema validation:** A `validate_sir(ds: xr.Dataset)` function checks that all variables have the expected names, units attributes, and reasonable value ranges before the SIR is passed to derivation plugins or formatters. This catches silent unit or naming errors early.
- **CF-1.8 attributes:** Every SIR variable carries `units`, `long_name`, `standard_name` (where CF conventions define one), and `valid_range` attributes.

**B. Model-specific derivation is plugin code, not config.** The relationship between SIR variables and model parameters is complex, model-specific, and sometimes requires domain judgment (e.g., choosing between Penman-Monteith and Hamon PET). This logic belongs in Python plugin classes, not in YAML configuration. Each model gets a derivation plugin:

```python
class PRMSDerivation:
    """Derive PRMS-specific parameters from SIR variables."""

    REQUIRED_SIR_VARS = [
        "elevation_mean", "slope_mean", "aspect_mean",
        "clay_pct_mean", "sand_pct_mean", "ksat_mean",
        "nlcd_class_*",  # glob pattern for class fractions
    ]

    def derive(self, sir: xr.Dataset) -> xr.Dataset:
        ds = xr.Dataset()
        # Soil parameters from POLARIS
        ds["soil_moist_max"] = self._soil_moist_max(sir)
        ds["soil_rechr_max"] = self._soil_rechr_max(sir)
        # Terrain parameters (unit conversion)
        ds["hru_slope"] = sir["slope_mean"] * DEGREES_TO_RADIANS
        # Land cover parameters
        ds["cov_type"] = self._classify_cover_type(sir)
        ds["covden_sum"] = self._summer_cover_density(sir)
        return ds

class NextGenDerivation:
    """Derive NextGen/CFE/TOPMODEL parameters from SIR variables."""
    ...
```

**C. Derivation plugins declare their SIR requirements.** The `REQUIRED_SIR_VARS` attribute enables the pipeline to validate that all needed source data is being processed before running a long computation. If the pipeline config requests `derivation: "prms"` but the datasets section does not include soils data, the pipeline can fail early with a clear error.

**D. Cross-category dependencies are explicit in derivation code.** When a PRMS parameter like `carea_max` depends on both soils HSG and land cover class, that dependency is encoded in the derivation plugin, not in the pipeline orchestrator. The pipeline simply ensures all requested datasets are processed to SIR before derivation runs.

**E. Config integration.** The pipeline config gains an optional `derivation` field:

```yaml
output:
  format: "prms"
  derivation: "prms"       # which derivation plugin to use
  model_version: "5.2.1"   # formatter may need model version
```

This connects to the existing `OutputFormatter` plugin via a two-step invocation: `DerivationPlugin.derive(sir)` then `OutputFormatter.write(derived, path)`. When no derivation is specified, the SIR passes directly to the formatter — useful for generic Parquet/NetCDF output of physical properties.

**F. This resolves open question Q16** (§10.2): "Where does transformation logic live?" Answer: model-specific derivation plugins that consume the SIR and produce derived parameter datasets. The dataset registry and pipeline config describe what physical data to collect; the derivation plugins know what to do with it for each model.

---

## 10. Open Questions & Action Items

### 10.1 Resolved

- ~~OSN access~~ → USGS contractor context provides access
- ~~Which soils product~~ → POLARIS primary, gNATSGO secondary
- ~~Similar projects~~ → Gap confirmed for config-driven, fabric-agnostic tool
- ~~Fabric flexibility (grids as polygons?)~~ → Bifurcate: gdptools for polygon targets, xesmf/rioxarray for grid targets (Appendix A.1)
- ~~Dataset registry as separate package?~~ → Yes, independent `hydro-param-data` (Appendix A.5)
- ~~Weight file format?~~ → Parquet; cache key from stable IDs not geometry hashes (Appendix A.2)
- ~~Output formatter architecture~~ → Plugin/adapter pattern with standardized internal representation (Appendix A.3)
- ~~Config philosophy~~ → Strictly declarative; logic in Python scripts (Appendix A.4)
- ~~Partial failure handling~~ → Tolerant mode + failed log + patch runs (Appendix A.6); skip/log implemented in MVP (Appendix B.2B)
- ~~Custom parameter derivations~~ → Model-specific derivation plugins consuming the SIR (Appendix A.8)
- ~~Project directory scaffolding~~ → Replaced with library-managed transparent caching, pooch-style (Appendix B.2A, §6.11 revised)
- ~~Fabric acquisition scope~~ → hydro-param does NOT fetch/subset fabrics; input is a path to a geospatial file (Appendix B.3A)
- ~~SIR schema enforcement~~ → Strict schema with standard names, guaranteed units, and validation function (Appendix B.2D, §A.8 A′)
- ~~CRS handling in exactextract~~ → Already handled by gdptools, which reprojects target geometry to source CRS by design (Appendix B.2C, §11.5)

### 10.2 Open Design Questions

1. **Config format:** YAML (more expressive nested structures) vs TOML (Python ecosystem trend)?
2. **Project name:** Signal generality. Candidates: `hydro-param`, `fabric-param`, `param-forge`, `model-fabric`.
3. **POLARIS depth weighting:** Pre-computed in Zarr, or 6 layers + compute on-the-fly? (Lean: on-the-fly)
4. **Batch sizing:** Fixed count vs adaptive (by polygon area)?
5. **Result assembly:** Concat partitions, or direct Zarr region writes?
6. **VirtualiZarr:** Adopt Kerchunk now, or full conversions initially?
7. **DEM resolution:** 3DEP 1/3" (~10m) vs 1" (~30m) for HRU-scale?
8. **Test domain:** Delaware River Basin (canonical gdptools example)?
9. **Outreach:** When/how to engage Mike Johnson and James McCreight?
10. **Grid regridding engine:** For grid-to-grid pathway (Appendix A.1), which conservative regridding method? `xesmf` vs `rioxarray.reproject_match` vs `exactextract`?
11. **SIR schema:** What CF conventions and variable naming for the Standardized Internal Representation (Appendix A.3)?
12. **Formatter prioritization:** Which output formatters to implement first? (Lean: Parquet → pywatershed → PRMS → NextGen)
13. **Spatial correctness validation:** How do we verify that computed parameters are correct? Regression tests against existing NhmParamDb values? Known-answer tests on small hand-computable domains? Spatial processing bugs can be silent — you get a number, it's just the wrong number.
14. **Output provenance and lineage:** What metadata travels with output files? When someone receives a PRMS parameter file, can they trace every value back to its source dataset, version, processing method, and config? Matters for USGS data releases and reproducibility claims. CF conventions help but don't cover the full lineage.
15. **Dataset registry versioning:** `hydro-param-data` will evolve — POLARIS Zarr stores may be reprocessed, NLCD 2023 replaces 2021, conversion bugs are found. How do we version the data products themselves, separate from code version? Affects cache invalidation strategy (§A.2).
16. ~~**Custom parameter derivations:** Many model parameters aren't a direct zonal mean — they're derived (depth-weighted soil averages, NLCD class fractions, slope-aspect combinations, seasonal climate normals from daily data). Where does transformation logic live? Dataset registry metadata, config, or Python code?~~ → Model-specific derivation plugins that consume the SIR (Appendix A.8)
17. ~~**Fabric acquisition and subsetting:** Step 1 (§4) is "resolve target fabric" but the path from "I want to model the Delaware River Basin" to a GeoDataFrame of HRUs involves non-trivial subsetting (pynhd for GFv1.1, different toolchains for NextGen, manual for custom fabrics). How much of this does hydro-param own vs. expect the user to provide?~~ → **hydro-param owns none of it.** The input must be a valid, flat file (GeoPackage/Parquet) of polygons. Let tools like pynhd or hydrodata handle subsetting. Attempting to build fabric subsetting is massive scope creep (Appendix B.3A).
18. **POLARIS licensing (CC BY-NC 4.0):** The NC (non-commercial) restriction on POLARIS — does it propagate to derived parameter values in model outputs? Matters for any operational or commercial use of parameterization results. May need legal guidance or an alternative pathway (gNATSGO) for affected users.
19. **Category enum governance:** The `category` field on `DatasetEntry` will become a `Literal` enum constrained to the 8 canonical categories (§1.3, §6.10). Decision: **Yes, Literal enum** — catches typos, enables IDE completion. Implementation deferred to code PR.
20. ~~**Project directory scaffolding:** Should the `project/` folder (§6.11) live inside the hydro-param repository as a template/example, or should hydro-param provide a scaffolding command (`hydro-param init --name upper_colorado --bbox ...`) that creates a project directory structure?~~ → **Dropped.** Replaced with library-managed transparent caching. The pipeline config is the single source of truth; no separate project config (Appendix B.2A, §6.11 revised).
21. **Dataset registry versioning:** The `hydro-param-data` registry will evolve — POLARIS Zarr stores may be reprocessed, conversion bugs fixed, new datasets added. The pipeline config should pin the registry version (e.g., `registry_version: "2024.02.1"`) to ensure re-running a config 6 months later yields identical results. This intersects with weight cache invalidation (§A.2) and data provenance (Q14). See Appendix B.3B.

### 10.3 Completed Actions

- [x] Foundation assessment (HyTEST + gdptools capabilities/gaps)
- [x] Parallel compute strategy with honest Dask assessment
- [x] Three-tier data layer strategy (native/virtual/converted)
- [x] Soils landscape analysis with POLARIS recommendation
- [x] Landscape scan of similar projects
- [x] Gap analysis and strategic positioning
- [x] Comprehensive synthesis (this document)
- [x] Cloud compute evaluation (§5.8)
- [x] External review response — Gemini 3 Pro (Appendix A)
- [x] Containerization strategy — Apptainer/Docker (§5.9)
- [x] External review response — Gemini 2.5 Pro (Appendix B)

### 10.4 Next Steps

**Phase 0: Immediate Value Deliverable** (per reviewer recommendation)
- [ ] POLARIS Zarr conversion pipeline (Tier C data transformation)
- [ ] NLCD Zarr conversion pipeline
- [ ] `hydro-param-data` package: standalone dataset registry + access helpers
- [ ] Publish converted Zarr stores (OSN or S3)

**Phase 1: Architecture & MVP** (see §11 for implementation details)

- [x] Config file schema design (declarative YAML structure) — §11.6
- [x] Test domain selection and MVP scope (Delaware River Basin) — §11.1
- [x] Config schema implementation (Pydantic models + YAML loader)
- [x] Dataset registry implementation (YAML schema + variable resolution)
- [x] Spatial batching implementation (KD-tree recursive bisection) — §11.2
- [x] STAC COG data access (Planetary Computer integration) — §11.4
- [x] gdptools ZonalGen + exactextract integration — §11.5
- [x] Pipeline orchestrator (5-stage batch loop) — §11.7
- [x] SIR output as xr.Dataset with CF-1.8 metadata
- [ ] Processing pathway bifurcation (gdptools vs xesmf routing)
- [ ] Compute backend interface (serial + joblib backends first)
- [x] Fault tolerance: skip/log in MVP (v5.3); patch runs deferred to Phase 2
- [ ] Spatial correctness validation strategy (regression tests, known-answer tests)
- [ ] SIR schema validation with standard names and units (§A.8 A′)
- [ ] Library-managed data caching (pooch-style, §6.11 v5.3 revision)
- [ ] Registry version pinning in pipeline config (§B.3B)
- [ ] Dockerfile and Apptainer conversion documentation (§5.9)

**Phase 2: Integration & Scaling**
- [ ] gdptools integration layer (wrapper API)
- [ ] Output formatter plugin system (Parquet → pywatershed → PRMS)
- [ ] Weight caching with ID-based keys and Parquet storage
- [ ] SLURM and Coiled compute backends
- [ ] Container integration in SlurmArrayBackend and AWSBatchBackend (§5.9)
- [ ] Output provenance and lineage metadata in SIR
- [ ] Parameter derivation framework (depth-weighting, class fractions, etc.)
- [ ] Project naming, repository setup, packaging

**Phase 3: Community & Outreach**
- [ ] Draft one-page summary for collaborators
- [ ] Outreach to Johnson and McCreight
- [ ] Documentation and example notebooks
- [ ] POLARIS license implications assessment (CC BY-NC 4.0 propagation)

---

## 11. MVP Implementation: Config-Driven Pipeline

This section documents the decisions made during initial implementation of the config-driven pipeline, using the Delaware River Basin as the test domain and USGS 3DEP terrain as the first dataset.

### 11.1 Scope: What the MVP Builds

The MVP implements the core pipeline loop (§4 stages 1–5) with enough infrastructure to demonstrate the config-driven pattern end-to-end. It is designed so that adding new datasets, access strategies, or output formats requires minimal code changes.

**Built in MVP:**

| Component | Implementation |
| --- | --- |
| Config schema | Pydantic models: TargetFabricConfig, DomainConfig, DatasetRequest, OutputConfig, ProcessingConfig |
| Dataset registry | YAML file mapping dataset names to access strategies, variable specs, and derivation rules |
| Spatial batching | KD-tree recursive bisection (§5.5.1 Approach 5) |
| Data access | STAC COG strategy via Planetary Computer (pystac-client + planetary-computer + rioxarray) |
| Processing | gdptools ZonalGen with exactextract engine; continuous and categorical modes |
| Output | SIR as xr.Dataset with CF-1.8 metadata; NetCDF writer |
| Test domain | Delaware River Basin, 7,268 NHDPlus catchments |
| Test dataset | 3DEP 1/3 arc-second (~10m) DEM with derived slope and aspect |

**Deferred to later phases:**

| Component | Rationale |
| --- | --- |
| Compute backends (joblib, SLURM, Coiled) | Serial is sufficient for regional-scale MVP |
| Weight caching | gdptools ZonalGen manages weights internally |
| Output formatters (PRMS, NextGen, pywatershed) | NetCDF/Parquet sufficient for validation |
| native_zarr and local_tiff access strategies | Implemented in registry schema but not in code; STAC COG is the working strategy |
| Patch runs (retry failed HRUs with different settings) | Tolerant mode with skip/log is sufficient for MVP |
| Spatial batching caching | Batching is fast enough to recompute (~seconds for 7K features) |

> **Design revision (v5.3):** An external review (Appendix B, item 2B) identified that strict-only failure mode makes the MVP frustrating for real-world testing — even at HUC2 scale, invalid geometries, self-intersections, and sliver polygons causing empty intersections are inevitable. The MVP now implements **tolerant mode with skip/log** as the default. A `try/except` around each spatial batch logs the failing batch ID and continues processing. See §A.6 for the full fault tolerance design.

### 11.2 Spatial Batching: Enabling Desktop-Scale Processing at 10m

The full Delaware basin at 10m DEM resolution is approximately 1.2 billion pixels (~5 GB). Loading this into memory on a desktop would fail. Spatial batching (§5.5) solves this by grouping catchments into spatially contiguous batches with compact bounding boxes, so each batch's data read is small.

**Implementation choice: KD-tree recursive bisection** (§5.5.1 Approach 5). Selected because:

- ~50 lines of pure numpy — no external dependencies
- Guarantees balanced batch sizes (within 2x of target)
- Produces tight bounding boxes for each batch
- Works for any geometry type (polygons, points)
- Fast: ~milliseconds for 7K features, ~seconds for 110K features

**Batch-based data flow:**

1. Assign 7,268 catchments to ~15 batches of ~500 each
2. For each batch: fetch source data clipped to batch bounding box (~0.3° × 0.3°)
3. At 10m resolution, each batch is ~1,100 × 1,100 pixels = ~5 MB per variable
4. Derive slope and aspect from the batch DEM
5. Run exactextract zonal statistics for the batch features
6. Merge batch results after all batches complete

This is the "chunk by space, not by time" principle (§5.4). The serial batch loop is trivially replaceable with joblib or SLURM array dispatch in a later phase.

### 11.3 Dataset Registry: Multi-Type Support

The dataset registry (`configs/datasets.yml`) maps human-readable dataset names to access strategies. The schema supports three dataset categories with different characteristics:

| Category | Example | Strategy | Variable Type | Key Difference |
| --- | --- | --- | --- | --- |
| Topography | 3DEP DEM | `stac_cog` | Continuous + derived | Derived variables (slope, aspect) computed from source (elevation) |
| Soils | POLARIS | `converted_zarr` | Continuous, multi-variable | Multiple independent variables; future: depth dimension |
| Land Cover | NLCD | `local_tiff` | Categorical | Class fractions, not means; `categorical: true` flag |

The MVP implements three of the eight data categories defined in §1.3. The full category processing profiles — including hydrography (vector join), climate (temporal aggregation), and cross-category parameter derivation — are described in §6.10 and Appendix A.8.

**Registry schema design decisions:**

- **`strategy` field**: Discriminator for which data access function to invoke. Extensible: adding a new access strategy (e.g., `native_zarr`) requires one new function in the data access module.
- **`categorical` flag on variables**: Flows through to `gdptools ZonalGen.calculate_zonal(categorical=True|False)`. Categorical variables produce class fraction columns instead of statistical summaries.
- **`derived_variables`**: Declared in the registry, not the pipeline config. The registry knows *how* to produce a variable (e.g., slope from elevation via Horn method). The pipeline config only says *which* variables to compute.
- **`sign` field**: Authentication method for signed URLs (e.g., `planetary_computer` for Azure blob signing). Extensible to other signing strategies.
- **CF metadata on variables**: Each variable carries `units` and `long_name` for SIR output compliance.

### 11.4 STAC COG Data Access via Planetary Computer

For the MVP, 3DEP data is accessed from the Microsoft Planetary Computer STAC catalog. This exercises a general pattern: **STAC query → COG loading → spatial clip → rioxarray DataArray**.

**Tile handling:** The 3DEP-seamless collection stores data in spatial tiles. A batch bounding box may span multiple tiles. The data access module queries STAC for all items intersecting the batch bbox, loads each via `rioxarray.open_rasterio()`, and mosaics if necessary using `rioxarray.merge.merge_arrays()`.

**URL signing:** Planetary Computer COGs require Azure blob URL signing. The `planetary_computer.sign_inplace` modifier handles this transparently via pystac-client's modifier hook.

**Generalization path:** Other STAC catalogs (HyTEST WMA STAC, Element 84, etc.) use the same pattern with different catalog URLs and potentially different signing methods. The registry's `catalog_url` and `sign` fields parameterize this.

### 11.5 Processing: gdptools ZonalGen + exactextract

The MVP replaces the placeholder point-in-polygon processor with real gdptools integration:

```text
UserTiffData(source_ds=batch_tiff, target_gdf=batch_catchments, target_id=id_field)
    → ZonalGen(user_data, zonal_engine="exactextract")
    → calculate_zonal(categorical=False)  # or True for land cover
    → pd.DataFrame of statistics per feature
```

**Why exactextract:** C++ implementation, 10–100x faster than serial Python. Handles its own memory management — reads only the raster window needed for each polygon, so memory usage is O(single polygon extent) regardless of total raster size.

**CRS alignment (v5.3):** An external review (Appendix B, item 2C) flagged that grid-based zonal stats engines can be sensitive to micro-misalignments between raster grids and vector coordinates when CRS definitions differ subtly (e.g., different WKT versions of the same EPSG code). **This is already handled by gdptools by design** — `ZonalGen` reprojects the target geometry into the source data's CRS before invoking exactextract, ensuring CRS-aligned inputs. Because hydro-param uses exactextract exclusively through `gdptools.ZonalGen(zonal_engine="exactextract")`, not as a standalone library, this class of error is mitigated at the gdptools layer. No additional reprojection logic is needed in hydro-param's batch processing.

**Continuous vs categorical:**

- `categorical=False`: Returns statistical summaries (mean, min, max, std) per polygon. Used for elevation, slope, aspect, soil properties.
- `categorical=True`: Returns fraction of each class within each polygon. Used for land cover. Output columns are class-specific (e.g., `class_11`, `class_21`, ...).

### 11.6 Pipeline Config Schema

The pipeline config is **strictly declarative** (§A.4). It says *what* to compute, not *how*:

```yaml
target_fabric:
  path: "data/delaware/catchments.gpkg"
  id_field: "featureid"
  crs: "EPSG:4326"

domain:
  type: bbox
  bbox: [-76.5, 38.5, -74.0, 42.6]

datasets:
  - name: dem_3dep_10m
    variables: [elevation, slope, aspect]
    statistics: [mean]

registry_version: "2024.02.1"    # pin registry for reproducibility (§B.3B)

output:
  path: "./output"
  format: netcdf
  sir_name: "delaware_terrain"

processing:
  engine: exactextract
  failure_mode: tolerant           # skip/log bad features, don't crash (§B.2B)
  batch_size: 500
```

**Design decisions:**

- `target_fabric` is a structured object with `path`, `id_field`, `crs` — not a bare string. The user provides the fabric as a file; fabric resolution is a separate pre-processing step (§10.2 Q17).
- `datasets` references registry names. The `variables` list selects which variables to compute; the registry resolves how.
- `batch_size` controls spatial batching granularity. Default 500 balances memory and overhead.
- `domain.type: bbox` is the only implemented domain type for MVP. Extension to HUC-based and gage-based domains is straightforward via pynhd.

### 11.7 Module Architecture

```text
src/hydro_param/
  config.py            — Pydantic config schema + YAML loader
  dataset_registry.py  — Registry schema + YAML loader + variable resolution
  data_access.py       — STAC COG fetch, terrain derivation, tile mosaic
  batching.py          — KD-tree spatial batching
  processing.py        — gdptools ZonalGen wrapper (continuous + categorical)
  pipeline.py          — 5-stage orchestrator with batch loop
```

Each module has a single responsibility and minimal coupling. The pipeline orchestrator calls the others in sequence but doesn't implement data access, batching, or processing logic itself.

### 11.8 Resolved Open Questions

This implementation resolves or partially resolves several open questions from §10.2:

| Question | Resolution |
| --- | --- |
| Q1 Config format | YAML (more expressive nested structures, gdptools precedent) |
| Q4 Batch sizing | Fixed count (500 features), configurable via `processing.batch_size` |
| Q7 DEM resolution | 10m (1/3 arc-second), enabled by spatial batching |
| Q8 Test domain | Delaware River Basin confirmed (7,268 NHDPlus catchments) |
| Q16 Custom derivations | Derived variables declared in dataset registry metadata (`derived_variables` section) |
| Q17 Fabric acquisition | User provides pre-downloaded fabric file; download is a separate script |

---

## Appendix B: Response to External Review (Gemini 2.5 Pro)

A second independent review was conducted by Gemini 2.5 Pro on the v5.2 document. The review returned a **GO** verdict: "The design document is exceptionally high-quality, demonstrating a deep understanding of the problem domain and the specific pitfalls of scaling geospatial compute." The review's summary recommendation: "The design is sound."

### B.1 Validated Strengths (No Action Needed)

The reviewer validated the following design decisions, confirming they are architecturally correct:

| Design Choice | Reviewer Assessment |
| --- | --- |
| **No dask.distributed** — shift to spatial batching + joblib/SLURM (§5.3–5.7) | "The correct architectural choice for this workload. Avoids fragile memory management issues common with distributed Dask on HPC." |
| **Strictly declarative configs** (§A.4) | "Crucial for long-term maintainability." |
| **Three-layer separation** — SIR → Derivation Plugins → Output Formatters (§A.8) | "Effectively isolates model-specific logic from the core processing engine." |
| **KD-tree recursive bisection** for spatial batching (§5.5, §11.2) | "A standout design choice. Solves the specific problem of scattered HRUs causing massive I/O amplification." |
| **Cloud vs. HPC cost/benefit analysis** (§5.8) and container strategy (§5.9) | "Shows a mature understanding of the deployment environment." |

### B.2 Critical Feedback — Responses

#### B.2A Project Directory Pattern (§6.11) — REVISED

**Concern:** The "Project Directory" concept introduced a stateful, local filesystem dependency conflicting with the "config-driven, run-anywhere" vision. Requiring a `project/data/raw` structure creates a "staging phase" barrier. Cloud users would need massive transfers to compute nodes, negating the "compute-to-data" advantage.

**Recommendation:** Treat local files as a caching detail. Move towards a transparent caching mechanism (pooch or fsspec caching) managed by the library. The pipeline config should be the single source of truth — no separate `project.yaml`.

**Response: Agree. §6.11 has been rewritten.** The explicit project directory scaffolding is replaced with library-managed transparent caching. The user's config requests datasets by name; the system checks a cache path and fetches if missing/stale. The pipeline config is the single source of truth. See revised §6.11 for the `DataCache` pattern and cache directory structure.

**Design impact:** Open question Q20 (project directory scaffolding) is now resolved. The `hydro-param init` scaffolding CLI is dropped; replaced with `hydro-param cache --status/--clean/--purge` for cache management.

#### B.2B MVP Strict Mode Viability (§11.1) — REVISED

**Concern:** The MVP specifies `failure_mode: strict`. At any meaningful scale (even a HUC2 like Delaware), invalid geometries, self-intersections, and sliver polygons cause empty intersections. Strict mode crashes the entire pipeline on the first bad feature, making the MVP frustrating for real-world testing.

**Recommendation:** Even for MVP, implement a basic "Skip and Log" behavior. A `try/except` around each spatial batch that logs the failing batch ID is sufficient. Don't let one bad polygon kill a 4-hour run.

**Response: Agree. §11.1 has been revised.** The MVP now defaults to `failure_mode: tolerant` with skip/log behavior. Each batch is wrapped in error handling that logs the failing features and continues processing. The more sophisticated patch-run mechanism (retry failed HRUs with different settings) remains deferred to Phase 2, but basic fault tolerance is now a Day 1 feature.

**Design impact:** §11.6 pipeline config example updated to show `failure_mode: tolerant`. §A.6 fault tolerance design remains the target architecture; MVP implements the core skip/log subset.

#### B.2C exactextract and CRS Precision (§11.5) — ALREADY HANDLED BY gdptools

**Concern:** Strictly grid-based zonal stats engines like exactextract can be sensitive to micro-misalignments between raster grids and vector coordinates, especially with on-the-fly reprojection. Different WKT versions of the same EPSG code can cause row/col lookup shifts, leading to silent data degradation.

**Recommendation:** Enforce explicit reprojection in the pipeline. Never rely on implicit transforms during the exactextract call. The batching step should crop the raster and reproject the vector batch to the exact raster CRS/transform before passing both to exactextract.

**Response: Valid concern, but already mitigated.** hydro-param does not call exactextract directly — it uses `gdptools.ZonalGen(zonal_engine="exactextract")`, and **gdptools reprojects the target geometry into the source data's CRS by design** before invoking exactextract. This means CRS alignment is guaranteed at the gdptools layer without additional reprojection logic in hydro-param's batch processing. §11.5 has been annotated to document this architectural safeguard.

#### B.2D The SIR Schema (§A.3, §A.8) — ADOPTED

**Concern:** The SIR design glosses over internal schema enforcement. If the SIR is just a "bag of variables," Model A might expect slope in degrees and Model B in radians. The derivation plugins mitigate this, but valid SIR state (units, naming conventions) must be strictly enforced.

**Recommendation:** The SIR must define a core set of standard names (e.g., `elevation_m` vs `elev_ft`). The SIR producer (core engine) must guarantee these units.

**Response: Agree. §A.3 and §A.8 have been strengthened.** The SIR now requires:

1. **Standard variable names** encoding both quantity and unit (e.g., `elevation_m`, `slope_deg`, `ksat_cm_hr`)
2. **Guaranteed units** — the SIR producer converts all source data to canonical units during processing
3. **Schema validation** — a `validate_sir()` function checks names, units, and value ranges
4. **CF-1.8 attributes** — every variable carries `units`, `long_name`, `standard_name`, and `valid_range`

This ensures derivation plugins receive consistently formatted data regardless of source dataset quirks.

### B.3 Missing Considerations — Responses

#### B.3A Fabric Subsetting Scope — ADOPTED

**Concern:** The document (Q17) asks "How much of fabric subsetting does hydro-param own?" The reviewer's verdict: **own none of it.** Building a tool that also subsets NHDPlus/NextGen fabrics is massive scope creep.

**Recommendation:** The input to the system must be a valid, flat file (GeoPackage/Parquet) of polygons. Let tools like pynhd or hydrodata handle the fetching/subsetting before hydro-param runs.

**Response: Agree. This is now a hard constraint.** Open question Q17 is resolved. The `target_fabric` config field accepts a `path` to a pre-existing geospatial file. hydro-param does not fetch, subset, or construct fabrics. This was already the de facto MVP behavior (§11.6 shows `target_fabric.path`), but it is now an explicit architectural boundary.

**Implication for documentation:** Example workflows should show a two-step pattern: (1) use pynhd/pygeohydro to subset the fabric, (2) point hydro-param at the resulting file.

#### B.3B Dataset Registry Versioning — ADOPTED

**Concern:** Moving the registry to `hydro-param-data` is smart, but there must be a mechanism for config reproducibility. Re-running a config 6 months later might yield different results if the `polaris` definition changed.

**Recommendation:** The pipeline config should pin the registry version: `registry_version: "2024.02.1"`.

**Response: Agree. Added to the pipeline config schema.** The `registry_version` field is now part of the pipeline config (§11.6). The `hydro-param-data` package will use calendar-based versioning (e.g., `2024.02.1`) so that pinning a version guarantees identical dataset definitions, access paths, and variable schemas.

**Implementation plan:**
- `hydro-param-data` publishes versioned registry snapshots
- The pipeline validates the installed registry version against `registry_version` in the config
- Mismatch produces a clear error: "Config requires registry 2024.02.1 but 2024.08.0 is installed"
- This is new open question Q21 in §10.2

### B.4 Summary of Review-Driven Changes

| Issue | Section(s) Modified | Change |
| --- | --- | --- |
| Project directory → library-managed caching | §6.11 (rewritten), §10.1, §10.2 Q20, §10.4 | Dropped scaffolding; adopted pooch-style transparent caching |
| MVP strict mode → tolerant with skip/log | §11.1, §11.6 | Default `failure_mode: tolerant` in MVP |
| CRS alignment via gdptools | §11.5 | Already handled — gdptools reprojects target to source CRS by design |
| SIR schema validation | §A.3, §A.8 | Standard names, guaranteed units, `validate_sir()` |
| Hard-scope fabric input | §10.1, §10.2 Q17 | hydro-param does NOT fetch/subset fabrics |
| Registry version pinning | §11.6, §10.2 Q21 | `registry_version` field in pipeline config |

---

## 12. Key Links & References

### Data Catalogs & Tools
- HyTEST: https://hytest-org.github.io/hytest/ | https://github.com/hytest-org
- WMA STAC: https://api.water.usgs.gov/gdp/pygeoapi/stac/stac-collection
- gdptools: https://gdptools.readthedocs.io/ | https://code.usgs.gov/wma/nhgf/toolsteam/gdptools
- ClimateR Catalog: https://mikejohnson51.github.io/climateR-catalogs/catalog.parquet

### USGS/NHM Ecosystem
- pywatershed: https://github.com/EC-USGS/pywatershed
- pyPRMS: https://github.com/DOI-USGS/pyPRMS
- NhmParamDb: https://www.sciencebase.gov/catalog/item/58af4f93e4b01ccd54f9f3da

### NOAA/NextGen Ecosystem
- NOAA hydrofabric: https://github.com/NOAA-OWP/hydrofabric
- climateR: https://github.com/mikejohnson51/climateR
- NGIAB_data_preprocess: https://github.com/CIROH-UA/NGIAB_data_preprocess
- ForcingProcessor: https://github.com/CIROH-UA/forcingprocessor

### Academic
- Watershed Workflow: https://github.com/environmental-modeling-workflows/watershed-workflow
- pyGSFLOW: Larsen et al. (2022), Frontiers in Earth Science
- pytRIBS: Raming et al. (2024), Env. Modelling & Software

### Data Sources
- POLARIS: http://hydrology.cee.duke.edu/POLARIS/ — Chaney et al. (2019) WRR
- CONUS404: OSN Zarr via HyTEST
- gNATSGO/gSSURGO: NRCS/USDA
- SoilGrids v2.0: https://soilgrids.org/

### Key Publications
- Regan et al. (2018) — NHM for PRMS
- Hay et al. (2023) — CONUS parameter estimation for NHM-PRMS
- Chaney et al. (2019) — POLARIS
- Coon & Shuai (2022) — Watershed Workflow

### Emerging Technologies
- VirtualiZarr: https://github.com/zarr-developers/VirtualiZarr
- Icechunk: https://github.com/earth-mover/icechunk
- Kerchunk: https://github.com/fsspec/kerchunk
- Cubed: https://github.com/cubed-dev/cubed
