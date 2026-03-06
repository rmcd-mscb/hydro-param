# Design: GFv1.1 ScienceBase Raster Integration

## Problem

hydro-param currently derives parameters from modern data sources (3DEP, gNATSGO, NLCD
via STAC/OSN) but cannot reproduce or validate against the GFv1.1 NHM parameters. Two
ScienceBase items contain the original CONUS-wide rasters used to parameterize NHM v1.1,
with complete provenance metadata. Integrating these rasters enables:

1. **Validation** — compare hydro-param output against the GFv1.1 reference parameters
2. **Alternative data sources** — SoilGrids250m soils, NALCMS land cover, MODIS canopy
3. **Reproducibility** — rebuild the rasters from documented upstream sources
4. **TWI access** — the TGF item includes pre-computed TWI rasters (feeds into #152)

## Architecture

The integration fits cleanly into hydro-param's existing architecture:

- **SB rasters** are consumed as `local_tiff` datasets — no new processing pathway needed
- **Download script** is standalone (`scripts/`) — not coupled to the library
- **Registry entries** follow the existing `DatasetEntry` schema
- **Rebuild scripts** are also standalone (`scripts/`) — produce the same GeoTIFFs
- **Pipeline and model plugins are untouched** — purely data access layer work

The topographic rebuild ports gfv2-params VPU-based raster merging + richdem terrain
computation into hydro-param `scripts/`, using the same libraries (rioxarray, richdem).

## ScienceBase Items

| Item | SB ID | Content | ~Size |
|------|-------|---------|-------|
| Data Layers | `5ebb182b82ce25b5136181cf` | Soils, LULC, Imperv, Canopy, Snow, Interception, Lithology, Water bodies | ~9 GB |
| TGF Topo | `5ebb17d082ce25b5136181cb` | DEM, Slope, Aspect, TWI, Flow Direction (transboundary HUC04s only) | ~2 GB |

Full provenance: `docs/reference/gfv11_raster_provenance.md`

## Phase 1: Download & Register (Implement Now)

### Issue A: Save provenance guide as reference doc
- Commit `docs/reference/gfv11_raster_provenance.md`
- No code changes

### Issue B: ScienceBase download script
- `scripts/download_gfv11_layers.py`
- CLI: `--output-dir <shared_dir>` `--items {data-layers,tgf-topo,all}`
- Downloads + unzips both SB items
- Organized subdirectories: `soils/`, `land_cover/`, `topo/`, `water_bodies/`, `misc/`
- Skip-if-exists, progress reporting
- Dependencies: `requests` (no sciencebasepy needed — SB items are public)

### Issue C: Register SB rasters in dataset catalog
New entries in existing registry YAMLs:

**`soils.yml`:**
- `gfv11_sand` — SoilGrids250m depth-weighted sand %, 250m, NAD83 Albers
- `gfv11_clay` — SoilGrids250m depth-weighted clay %, 250m
- `gfv11_silt` — SoilGrids250m silt % (derived), 250m
- `gfv11_awc` — SoilGrids250m available water capacity, 250m
- `gfv11_text_prms` — USDA texture class → PRMS soil_type codes, categorical

**`land_cover.yml`:**
- `gfv11_lulc` — NALCMS 2015 → PRMS cov_type (categorical, classes 1–7)
- `gfv11_imperv` — GMIS impervious surface %, 30m
- `gfv11_cnpy` — MODIS tree canopy cover %, resampled to 30m

**`water_bodies.yml`:**
- `gfv11_wbg` — NHD HR waterbody mask, 30m, categorical

**`topography.yml`:**
- `gfv11_tgf_dem` — SRTM 30m DEM (TGF domain only)
- `gfv11_tgf_slope` — Slope × 100 integer (TGF domain)
- `gfv11_tgf_twi` — TWI × 100 integer (TGF domain)

All use `strategy: local_tiff` with `source:` paths relative to the shared data dir.

### Issue D: Validate SB rasters against DRB reference
- Notebook or test script running ZonalGen on DRB fabric
- Compare vs current 3DEP/gNATSGO/NLCD pipeline output
- Document expected differences:
  - SoilGrids250m vs gNATSGO (different soil products entirely)
  - NALCMS vs NLCD (different classification systems)
  - GMIS vs NLCD impervious (different vintage/methodology)
  - MODIS canopy vs NLCD tree canopy

## Phase 2: Reproduce from Scratch (Roadmap — Create Issues Only)

### Issue E: Build CONUS topographic rasters from NHDPlus RPU DEMs
Port gfv2-params workflow:
- **Step a** (`process_NHD_by_vpu.py`): Download NHDPlus RPU ESRI Grid rasters, merge
  per-VPU using `rioxarray.merge.merge_arrays()`, convert cm → meters
- **Step b** (`process_slope_and_aspect.py`): Compute slope/aspect via `richdem.TerrainAttribute`
- Config-driven, VPU-parallel (Slurm-compatible)
- Output: per-VPU GeoTIFFs for elevation, slope, aspect
- Dependencies: rioxarray, richdem

### Issue F: Build soils rasters from SoilGrids250m
- Download SoilGrids250m v0.2 (Zenodo DOI: 10.5281/zenodo.2525663)
- Depth-weighted map algebra for sand/clay (weights in provenance guide)
- Silt = 100 - (sand + clay)
- TEXT_PRMS via USDA texture triangle (reuse `classification.py`)
- Validate reproduced rasters match SB downloads

### Issue G: Build land cover rasters from NALCMS/NLCD
- Download NALCMS 2015 or NLCD 2021
- Reclassify to PRMS cov_type (lookup table in provenance guide)
- Derive interception/leaf rasters (SRain, WRain, Snow, keep, loss)
- Option for NLCD 2021 as modern alternative

### Issue H: Build TWI raster from DEM (connects to #152)
- Fill DEM → D8 flow direction → flow accumulation → TWI
- Requires whitebox-tools (GDAL cannot do flow accumulation)
- VPU-parallel, same structure as Issue E
- Produces CONUS-wide TWI matching SB TGF product methodology

### Issue I: Track modern dataset upgrade opportunities
Tracking issue for legacy → modern replacements:
- SoilGrids v0.2 → v2.0 (2021)
- NALCMS 2015 → NLCD 2021
- GMIS 2010 → NLCD imperv 2021
- MOD44B → GLAD 30m canopy
- NHDPlus RPU DEMs → 3DEP 10m (via py3dep)

## Data Flow

```
ScienceBase Items                    Shared Data Directory
  ├─ Data Layers (5ebb182b)     ──→  /shared/gfv11/soils/{Sand,Clay,Silt,AWC,TEXT_PRMS}.tif
  │                                  /shared/gfv11/land_cover/{LULC,Imperv,CNPY}.tif
  │                                  /shared/gfv11/water_bodies/wbg.tif
  │                                  /shared/gfv11/misc/{Snow,SRain,WRain,keep,loss,...}.tif
  └─ TGF Topo (5ebb17d0)       ──→  /shared/gfv11/topo/{dem,slope100X,asp100X,twi100X}.tif

Dataset Registry (soils.yml, etc.)
  └─ strategy: local_tiff
     source: /shared/gfv11/soils/Sand.tif

Pipeline Config (user's project)
  └─ datasets:
       - name: gfv11_sand
         variables: [sand_pct]
         statistics: [mean]
```

## Dependencies

Phase 1 requires only `requests` for downloading (no new library deps).
Phase 2 adds `richdem` and optionally `whitebox` for terrain generation.
