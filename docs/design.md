# Hydrologic Model Parameterization System — Complete Design Document

**Version:** 5.0 — Comprehensive Synthesis
**Date:** 2025-02-13
**Author:** Rich McDonald / Claude (AI-assisted brainstorm)
**Synthesized from:** v1.0 (Foundation), v2.0 (Compute), v3.0 (Data Strategy), v3.1 (Soils), v4.0 (Landscape)

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

| Category | Purpose | Example Datasets |
|----------|---------|-----------------|
| **Static Physical** | Basin/grid characterization | DEM/slope (3DEP), soils (POLARIS, gSSURGO), geology |
| **Vegetation/Land Cover** | Land surface parameters | NLCD, MODIS LAI/NDVI, LCMAP |
| **Climate Forcings** | Model inputs (time-varying) | CONUS404, GridMET, Daymet, Livneh, PRISM, TerraClimate |
| **Calibration/Verification** | Model evaluation | NWIS streamflow, SNODAS/UA SWE, MODSCAG, ET (SSEBop/OpenET) |
| **Hydrographic Fabric** | Model discretization | GFv1.1, WBD HUC12, NHDPlus, NextGen hydrofabric |

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
- You're a non-USGS user (Connected Waters LLC, university collaborators) without HPC allocations
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
4. **Connected Waters LLC** doesn't have USGS HPC allocations; cloud is the natural production environment for contractor work

Coiled is the recommended starting point: Pythonic, Dask-native, free tier adequate for development, spot instances for production, and the `us-west-2` colocation with most USGS cloud data is a perfect fit.

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

The core engine produces a **Standardized Internal Representation (SIR)** — an `xarray.Dataset` with CF-compliant metadata, consistent naming, and full provenance attributes. Output formatters are adapters:

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
- ~~Partial failure handling~~ → Tolerant mode + failed log + patch runs (Appendix A.6)

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

### 10.4 Next Steps

**Phase 0: Immediate Value Deliverable** (per reviewer recommendation)
- [ ] POLARIS Zarr conversion pipeline (Tier C data transformation)
- [ ] NLCD Zarr conversion pipeline
- [ ] `hydro-param-data` package: standalone dataset registry + access helpers
- [ ] Publish converted Zarr stores (OSN or S3)

**Phase 1: Architecture & MVP**
- [ ] Config file schema design (declarative YAML structure)
- [ ] Processing pathway bifurcation (gdptools vs xesmf routing)
- [ ] Standardized Internal Representation (SIR) schema definition
- [ ] Compute backend interface (serial + joblib backends first)
- [ ] Fault tolerance: tolerant/strict modes, failed HRU logging, patch runs
- [ ] Test domain selection and MVP scope (Delaware River Basin)

**Phase 2: Integration & Scaling**
- [ ] gdptools integration layer (wrapper API)
- [ ] Output formatter plugin system (Parquet → pywatershed → PRMS)
- [ ] Weight caching with ID-based keys and Parquet storage
- [ ] SLURM and Coiled compute backends
- [ ] Project naming, repository setup, packaging

**Phase 3: Community & Outreach**
- [ ] Draft one-page summary for collaborators
- [ ] Outreach to Johnson and McCreight
- [ ] Documentation and example notebooks

---

## 11. Key Links & References

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
