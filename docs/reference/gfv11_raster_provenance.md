# GFV1.1 ScienceBase Raster Provenance & Rebuild Guide
*Extracted from FGDC metadata XML files — both ScienceBase items*
*Authors: Michael E. Wieczorek, Andrew R. Bock (USGS Northeast Region, 2020–2021)*

---

## Overview

Two ScienceBase items provide the source rasters used by gfv2-params:

| Item | ScienceBase ID | Title | Domain |
|------|---------------|-------|--------|
| Data Layers | `5ebb182b82ce25b5136181cf` | Data Layers for the National Hydrologic Model, version 1.1 | CONUS (GFv1.1) |
| TGF Topo | `5ebb17d082ce25b5136181cb` | Topographic derivative datasets for the United States-Canada transboundary Geospatial Fabric | TGF transboundary basins |

Both are children of the GFv1.1 parent item `5e29b87fe4b0a79317cf7df5`.

**Key finding: The FGDC XML metadata files contain complete, step-by-step rebuild instructions
for every raster, including exact source datasets, download sources, map algebra formulas,
and in the TGF case, the actual ArcPy Python script used to derive topographic products.**

---

## Item 1: Data Layers for NHM v1.1 (`5ebb182b82ce25b5136181cf`)

### Files Inventory

| File | Purpose | Size |
|------|---------|------|
| TEXT_PRMS.zip | Soil texture class (STATSGO → PRMS codes) | 6 MB |
| Clay.zip | Percent clay (depth-weighted) | 211 MB |
| Silt.zip | Percent silt (derived) | 218 MB |
| Sand.zip | Percent sand (depth-weighted) | 228 MB |
| AWC.zip | Available water capacity | 189 MB |
| LULC.zip | Land use/land cover (NALCMS 2015 → PRMS cov_type) | 1.0 GB |
| Imperv.zip | Percent impervious surface | 365 MB |
| Snow.zip | Snow depletion curve numbers | 680 MB |
| SRain.zip | Summer rain interception values | 470 MB |
| WRain.zip | Winter rain interception values | 584 MB |
| keep.zip | Leaf presence values | 801 MB |
| loss.zip | Leaf loss values | 600 MB |
| RootDepth.zip | Rooting depth | 766 MB |
| CNPY.zip | Tree canopy cover percent | 2.8 GB |
| CV_INT.zip | Covden lookup table (integer remap) | ~0.5 MB |
| Lithology_exp_Konly_Project.zip | GLHYMPS lithology/permeability shapefile | 456 MB |
| wbg.zip | Water bodies GIS | 11 MB |
| CrossWalk.xlsx | NLCD crosswalk table | 11 KB |
| SDC_table.csv | Snow depletion curve table | 552 bytes |
| Data Layers for the NHM Domain_Final.xml | **FGDC metadata** | 75 KB |

**Note:** No elevation, slope, or aspect rasters are in this item. Those are in the TGF item
for transboundary basins, and for CONUS they came from gfv2-params' Steps a+b using NHDPlus
RPU DEMs. This confirms that the Data Layers item covers *non-topographic* parameters only.

---

### Process Steps — Complete Rebuild Instructions

#### Process Step 1: Percent Sand, Silt, Clay

**Source:** OpenGeoHub SoilGrids250m v0.2 (Hengl 2018, Zenodo DOI: 10.5281/zenodo.2525663)

**Procedure:**
1. Download the 5 standard layer depths (0–10, 10–30, 30–60, 60–100, 100–200 cm) of clay
   and sand from OpenGeoHub (accessed 04/12/2019)
2. Using map algebra, compute depth-weighted percent sand and clay:

```
SAND_PCT = (sol_sand_b0cm * 0.025)
         + (sol_sand_b10cm * 0.075)
         + (sol_sand_b30cm * 0.125)
         + (sol_sand_b60cm * 0.175)
         + (sol_sand_b100cm * 0.25)
         + (sol_sand_b200cm * 0.35)
```

Exact filenames (SoilGrids250m v0.2):
- `sol_sand.wfraction_usda.3a1a1a_m_250m_b0..0cm_1950..2017_v0.2.tif`
- `sol_sand.wfraction_usda.3a1a1a_m_250m_b10..10cm_1950..2017_v0.2.tif`
- `sol_sand.wfraction_usda.3a1a1a_m_250m_b30..30cm_1950..2017_v0.2.tif`
- `sol_sand.wfraction_usda.3a1a1a_m_250m_b60..60cm_1950..2017_v0.2.tif`
- `sol_sand.wfraction_usda.3a1a1a_m_250m_b100..100cm_1950..2017_v0.2.tif`
- `sol_sand.wfraction_usda.3a1a1a_m_250m_b200..200cm_1950..2017_v0.2.tif`

Weights are based on fraction of each layer within the total 200 cm soil column:
- 0–10 cm = 10/200 × 0.5 = 0.025  (10 cm layer)
- 10–30 cm = 20/200 × 0.75 = 0.075
- 30–60 cm = 30/200 × ... (etc.)

3. Clip to GFv1.1 extent
4. Project from GCS_WGS_1984 → NAD83 National Albers
5. Export to GeoTIFF
6. Repeat steps 1–5 for clay
7. Create silt by map algebra: `SILT_PCT = 100 - (SAND_PCT + CLAY_PCT)`

**Process Date:** 02/23/2020
**Contact:** Michael E. Wieczorek (mewieczo@usgs.gov), USGS Northeast Region

**Hydro-param notes:**
- **NOT** STATSGO2 as assumed. Uses SoilGrids250m (OpenGeoHub global product).
- Sand.zip, Silt.zip, Clay.zip are these products.
- TEXT_PRMS.zip is a separate reclassification (see gfv2-params Step 4).
- Modern replacement: SoilGrids250m v2.0 (2021) at same OpenGeoHub, same schema.

---

#### Process Step 2: Average Water Capacity (AWC)

**Source:** OpenGeoHub SoilGrids250m — soil available water capacity product (Hengl 2018)

**Procedure:**
1. Download soil available water capacity (AWC) in mm, derived for 0–200 cm, at 250 m
2. Clip to GFv1.1 extent
3. Divide by 200 to convert from mm/200cm to mm/mm depth (unit normalization)
4. Project GCS_WGS_1984 → NAD83 National Albers
5. Export to GeoTIFF

**Hydro-param notes:**
- AWC is not directly a pywatershed parameter but feeds soil_moist_max calculation.
- Same OpenGeoHub source as sand/clay/silt.

---

#### Process Step 3: Land Use/Land Cover (LULC → cov_type)

**Source:** North American Land Change Monitoring System (NALCMS) 2015

**Procedure:**
1. Download NALCMS 2015 land cover raster
2. Remap NALCMS class codes → PRMS cov_type codes using this lookup table
   (from Viger and Leavesley 2007):

```
NALCMS  NALCMS Definition                              PRMS  PRMS Definition
0       NODATA                                         -999  NODATA
1       Temperate/sub-polar needleleaf forest            4   Coniferous
2       Sub-polar taiga needleleaf forest                4   Coniferous
3       Tropical/sub-tropical broadleaf evergreen        4   Coniferous
4       Tropical/sub-tropical broadleaf deciduous        3   Deciduous
5       Temperate/sub-polar broadleaf deciduous          3   Deciduous
6       Mixed forest                                     3   Deciduous
7       Tropical/sub-tropical shrubland                  2   Shrub
8       Temperate/sub-polar shrubland                    2   Shrub
9       Tropical/sub-tropical grassland                  1   Grass
10      Temperate/sub-polar grassland                    1   Grass
11      Sub-polar/polar shrubland-lichen-moss            1   Grass
12      Sub-polar/polar grassland-lichen-moss            1   Grass
13      Sub-polar/polar barren-lichen-moss               1   Grass
14      Wetland                                          5   Wetland
15      Cropland                                         6   Bare soil
16      Barren land                                      6   Bare soil
17      Urban and built-up                               6   Bare soil
18      Water                                            7   Water
19      Snow and ice                                     1   Grass
```

3. Clip to GFv1.1 extent
4. Project GCS_WGS_1984 → NAD83 National Albers
5. Export to GeoTIFF (this is the LULC.zip file)

**Hydro-param notes:**
- LULC.zip = the remapped cov_type raster (PRMS classes 1–7).
- NALCMS 2015 was used (not NLCD) — consistent across CONUS + Canada.
- For CONUS-only hydro-param, NLCD 2021 is a valid substitute.
- PRMS cov_type values: 1=Grass, 2=Shrub, 3=Deciduous, 4=Coniferous, 5=Wetland,
  6=Bare soil/urban/crop, 7=Water.

---

#### Process Step 4: Remapped Data (rain/snow/leaf interception parameters)

**Source:** LULC raster from Step 3 + lookup tables from Viger and Leavesley (2007)

**Procedure:**
Using the remapped LULC layer from Step 3, derive the following parameter rasters
by applying class-based lookup values from Viger and Leavesley (2007):

- **SRain.tif** — Summer rain interception weights (hundredths of inches)
- **WRain.tif** — Winter rain interception weights (hundredths of inches)
- **Snow.tif** — Snow interception values
- **keep.tif** — Leaf presence (value retained) per cover type
- **loss.tif** — Leaf loss value per cover type

All are remaps of cov_type → parameter value using published tables.
Values are GeoTIFFs at 30 m resolution, NAD83 National Albers.

**Hydro-param notes:**
- These parameters are derived entirely from land cover.
- They are outputs of a simple reclassify operation on LULC.zip.
- If you have LULC.zip (Step 3) and the Viger/Leavesley lookup tables, Step 4 is
  essentially free to reproduce.

---

#### Process Step 5: Remapped Soil Values (TEXT_PRMS)

**Source:** Sand, Clay rasters from Step 1 + STATSGO2 soil texture class boundaries

**Procedure:**
1. Take the soil layers from Step 1 (Sand_PCT and Clay_PCT)
2. Reclassify using standard USDA soil texture triangle classification to assign
   soil texture class codes compatible with PRMS (soil_type parameter)
3. Output: TEXT_PRMS.tif — integer soil texture class raster

USDA soil texture classes used:
```
1 = Sand
2 = Loamy sand
3 = Sandy loam
4 = Loam
5 = Silt loam
6 = Silt
7 = Sandy clay loam
8 = Clay loam
9 = Silty clay loam
10 = Sandy clay
11 = Silty clay
12 = Clay
```

**Hydro-param notes:**
- TEXT_PRMS.zip is the soil texture classification output (Step 5).
- Derived from SoilGrids250m (OpenGeoHub) sand/clay — NOT from STATSGO2 directly.
- hydro-param already has USDA texture triangle classification in `classification.py`.

---

#### Process Step 6: Water Bodies

**Source:** NHD High Resolution (NHDHR) waterbodies, 4-digit HUC coverage

**Procedure:**
1. Download all NHDHR waterbodies for 4-digit Hydrologic Unit Codes (HUC04s)
   covering the GFv1.1 domain
2. Merge waterbody polygons
3. Clip to GFv1.1 extent
4. Rasterize to 30 m grid
5. Export to GeoTIFF (wbg.zip)

**Hydro-param notes:**
- Water bodies mask used in parameter assignments (hru_type = 0 for lakes).
- wbg.zip is a supplementary masking layer.

---

#### Process Step 7: Tree Canopy Cover (CNPY)

**Source:** MOD44B MODIS/Terra Vegetation Continuous Fields Yearly L3 Global 250m
SIN Grid V006 (Carroll and others 2017)

**Procedure:**
1. Download all 4 Tree Canopy Cover (TCC) tiles covering the GFv1.1 domain
2. Mosaic tiles
3. Clip to GFv1.1 extent
4. Project → NAD83 National Albers
5. Resample to 30 m
6. Export to GeoTIFF (CNPY.zip)

**Additional files from same source:**
- **Imperv.zip** — Percent impervious surface from Global Man-made Impervious Surface
  (GMIS) Dataset from Landsat v1 (NASA 2010) — different source than tree canopy
- **RootDepth.zip** — Rooting depth raster (source: Viger and Leavesley 2007 defaults
  applied spatially based on cover type)
- **Snow.zip** — Snow depletion curve numbers (from Liston and others 2009, further
  processed using Sexstone and others 2020 methods)

**Hydro-param notes:**
- CNPY.zip → covden_sum / covden_win (canopy density, summer and winter)
- Imperv.zip → imperv parameter
- These are all available pre-computed in the Data Layers SB item.

---

## Item 2: TGF Topographic Derivatives (`5ebb17d082ce25b5136181cb`)

### Files Inventory

| File | Purpose |
|------|---------|
| dem.zip | Digital elevation model (30 m, SRTM-derived) |
| slope100X.zip | Slope (rise/run × 100, integer) |
| asp100X.zip | Aspect (degrees × 100, integer) |
| twi100X.zip | Topographic Wetness Index (TWI × 100, integer) |
| fdr.tif | Flow direction |
| Topographic_Derivatives_Transboundary_Domain_Final.xml | **FGDC metadata** |

**Domain:** Transboundary HUC04s: 0101, 0105, 0108, 0901, 0902, 0903, 0904, 1005,
1006, 1701, 1702, 1711

---

### Process Steps — Complete Rebuild Instructions

#### TGF Step 1: Digital Elevation Model

**Source:** NASA Shuttle Radar Topography Mission (SRTM) Global 1 arc second
(NASA JPL 2013, DOI: 10.5067/measures/srtm/srtmgl1.003)

**Procedure:**
1. Download 1,723 HGT format (SRTM) tiles covering the TGF domain
2. Import, merge, and clip tiles by NHD HR HUC04:
   0101, 0105, 0108, 0901, 0902, 0903, 0904, 1005, 1006, 1701, 1702, 1711
3. Export to GeoTIFF, NAD83

**Process Date:** 02/23/2020

**Hydro-param notes:**
- TGF uses SRTM (global, 30 m); CONUS uses NHDPlus RPU DEMs (NED, 30 m).
- For hydro-param Phase 3, replace both with 3DEP 1/3 arcsec (10 m).

---

#### TGF Step 2: Slope, Aspect, TWI, Flow Direction

**Source:** DEM from Step 1 + ArcPy (ArcGIS Spatial Analyst)

**Complete ArcPy script documented in metadata:**

```python
# Calc TWI for HUC4's — Author: mewieczo, Created: 02/20/2020

import arcpy, math
from arcpy import env
from arcpy.sa import *

arcpy.env.overwriteOutput = True
env.qualifiedFieldNames = False

# List of HUC04's to process
inputs = {'0101', '0105', '0108', '0901', '0902', '0903', '0904',
          '1005', '1006', '1701', '1702', '1711'}

for id in inputs:
    topodir = "K:/WBEEP/Testing/data_bins/HRU" + id + "/data_bin/topo"
    arcpy.env.workspace = topodir
    arcpy.env.scratchWorkspace = topodir
    arcpy.env.mask = ''
    inDEM = topodir + '/dem.tif'

    # Fill DEM (waterways not masked)
    DEM_filled = arcpy.sa.Fill(inDEM)

    # Flow direction (FORCE: edge cells flow outward)
    outFlowDirection = arcpy.sa.FlowDirection(DEM_filled, "FORCE")
    outFlowDirection.save(topodir + '/fdr.tif')

    # Flow accumulation (+1 so headwater cells are not zero)
    outFlowAccumulation = arcpy.sa.FlowAccumulation(outFlowDirection, "", "INTEGER") + 1
    outFlowAccumulation.save(topodir + '/fac.tif')

    # Slope in degrees
    slope = arcpy.sa.Slope(DEM_filled, "DEGREE")
    slope.save(topodir + '/slopedeg.tif')

    # Slope in percent rise (for TWI calculation)
    slopepct = arcpy.sa.Slope(DEM_filled, "PERCENT_RISE")
    # ... (TWI computation continues)

    # Aspect
    aspect = arcpy.sa.Aspect(DEM_filled)
    # ... (exported as asp100X.tif = aspect * 100, integer)
```

**Outputs and scaling:**
- `dem.tif` — elevation in meters (centimeters in attribute table)
- `slope100X.tif` — slope (rise/run) × 100, converted to integer to reduce file size
- `asp100X.tif` — aspect in degrees × 100, converted to integer (range 0–35996)
- `twi100X.tif` — TWI × 100, integer
- `fdr.tif` — ArcGIS flow direction codes (1,2,4,8,16,32,64,128)

**Hydro-param notes:**
- The metadata contains the actual production script. This is rare.
- The CONUS equivalent (gfv2-params Steps a+b) used GDAL instead of ArcPy.
- The TWI raster is an **additional product** not in the CONUS Data Layers item.
  It could be useful for wetland delineation or saturation-excess parameterization.
- GDAL-equivalent commands for reproducibility without ArcPy:
  ```bash
  # Fill DEM
  gdal_fillnodata.py dem.tif dem_filled.tif
  # Slope in degrees
  gdaldem slope dem_filled.tif slope.tif -alg ZevenbergenThorne
  # Aspect
  gdaldem aspect dem_filled.tif aspect.tif
  # TWI requires flow accumulation — use whitebox-tools or TauDEM
  ```

---

## Critical Discovery: Data Layers ≠ gfv2-params Inputs

**The FGDC metadata reveals that the Data Layers rasters are NOT the same as gfv2-params
input rasters for soils and land cover.** They are different products:

| Parameter | gfv2-params source (Hovenweep) | Data Layers SB item | Notes |
|-----------|-------------------------------|---------------------|-------|
| soil texture (TEXT_PRMS) | `soils_litho/TEXT_PRMS.tif` | TEXT_PRMS.zip | Same file |
| soil Ksat | `soils_litho/Ksat_cm_hr_1m.tif` | **NOT in SB item** | Separate source |
| soil porosity | `soils_litho/poros_1m.tif` | **NOT in SB item** | Separate source |
| sand % | `soils_litho/...` (unknown) | Sand.zip (OpenGeoHub) | Different source? |
| clay % | `soils_litho/...` (unknown) | Clay.zip (OpenGeoHub) | Different source? |
| land cover | NLCD (downloaded by script) | LULC.zip (NALCMS 2015) | Different source |
| imperviousness | MRLC fract imperv (downloaded) | Imperv.zip (GMIS/Landsat) | Different source |
| lithology | `soils_litho/Lithology_exp_Konly_Project.shp` | Lithology_exp_Konly_Project.zip | Same file |
| DEM/slope/aspect | NHDPlus RPU DEMs → GDAL | Not in CONUS Data Layers | Different source |

**Key implications:**
1. The ScienceBase Data Layers item has CONUS-wide SoilGrids250m-based soil products,
   while gfv2-params used what appears to be STATSGO2-based products.
2. TEXT_PRMS.zip appears to be the same file (consistent naming in gfv2-params configs).
3. Ksat and porosity rasters used in gfv2-params Step 05 (soil_moist_max) are NOT in the
   Data Layers SB item — they must come from the Hovenweep project directory directly.
4. The land cover source is NALCMS (SB Data Layers) vs NLCD (gfv2-params download script).

---

## Source Datasets Summary

| Raster Layer | Source Dataset | Source Organization | Vintage | Resolution | Access |
|-------------|---------------|---------------------|---------|------------|--------|
| Sand, Clay, Silt, AWC | SoilGrids250m v0.2 | OpenGeoHub (Hengl 2018) | 2018 | 250 m | Zenodo DOI 10.5281/zenodo.2525663 |
| TEXT_PRMS | Derived from SoilGrids250m | USGS | 2020 | 250 m → 30 m | Same as above |
| LULC (cov_type) | NALCMS 2015 | CEC/USGS | 2015 | 30 m | http://www.cec.org/tools-and-resources/map-files/land-cover-2015 |
| Imperv | GMIS Dataset from Landsat v1 | NASA 2010 | 2010 | 30 m | NASA LP DAAC |
| Tree canopy (CNPY) | MOD44B MODIS VCF V006 | NASA/Carroll et al. 2017 | 2017 | 250 m → 30 m | LP DAAC |
| Snow (depletion curve) | Liston et al. 2009 + Sexstone et al. 2020 | USGS | 2009/2020 | 30 m | Literature-derived |
| Rain/leaf (SRain, WRain, keep, loss) | Derived from LULC + Viger & Leavesley 2007 | USGS | 2020 | 30 m | Derived |
| Lithology (ssflux) | GLHYMPS permeability | Gleeson/Zenodo | — | polygon | SB item (Lithology_exp_Konly_Project.zip) |
| DEM (CONUS) | NHDPlus RPU DEMs (NED) | USGS NHDPlus v1 | pre-2010 | 30 m | gfv2-params Step a |
| DEM (TGF) | SRTM Global 1 arc second | NASA JPL | 2013 | 30 m | DOI 10.5067/measures/srtm/srtmgl1.003 |
| Slope/Aspect (CONUS) | Derived from NHDPlus RPU DEMs via GDAL | USGS | 2020 | 30 m | gfv2-params Step b |
| Slope/Aspect/TWI (TGF) | Derived from SRTM via ArcPy | USGS | 2020 | 30 m | TGF metadata ArcPy script |

---

## gfv2-params Repository Reference

The [gfv2-params](https://github.com/rmcd-mscb/gfv2-params) repository contains the
production scripts used to generate NHM v2 parameters on Hovenweep HPC. Key files:

### Slurm Batch Scripts (`slurm_batch/`)

| Script | Purpose | Dependencies |
|--------|---------|--------------|
| `a_process_NHD_by_vpu.batch` | Merge NHDPlus RPU rasters into per-VPU GeoTIFFs | rioxarray |
| `b_process_slope_aspect.batch` | Compute slope/aspect from merged DEMs | richdem |
| `01_create_elev_params.batch` | Run ZonalGen for elevation per VPU | gdptools |
| `02_create_slope_params.batch` | Run ZonalGen for slope per VPU | gdptools |
| `03_create_aspect_params.batch` | Run ZonalGen for aspect per VPU | gdptools |
| `04_create_soils_params.batch` | Run ZonalGen for soils per VPU | gdptools |
| `05_create_soilmoistmax_params.batch` | Soil moisture max per VPU | gdptools |
| `06_create_ssflux_params.batch` | Subsurface flux per VPU | gdptools |
| `07_merge_output_params.batch` | Merge per-VPU outputs into CONUS | — |

### Python Scripts (`scripts/`)

- **`process_NHD_by_vpu.py`** — Merges NHDPlus RPU ESRI Grid rasters (NEDSnapshot,
  Hydrodem, FDR, FAC) into per-VPU GeoTIFFs. Converts DEM from cm → m.
  Uses `rioxarray.merge.merge_arrays()`.
- **`process_slope_and_aspect.py`** — Computes slope (degrees) and aspect from merged
  DEM using `richdem.TerrainAttribute`. Fixes nodata before processing.
- **`1_create_dem_params.py`** — Runs gdptools `UserTiffData` + `ZonalGen` against
  target fabric (nhru GeoPackage) for elevation, slope, or aspect (config-driven).
  Uses `zonal_engine="parallel"` with 4 jobs.

### VPU Processing

All 18 CONUS VPUs: 01–18 (with sub-VPUs 03N/03S/03W and 10U/10L for rasters vs targets).

---

## Rebuild Assessment

### What can be rebuilt from metadata alone

**YES — fully reproducible from metadata:**
- Sand, Silt, Clay, AWC: SoilGrids250m v0.2 download + documented map algebra
- TEXT_PRMS: Derived from sand/clay using USDA texture triangle (documented)
- LULC/cov_type: NALCMS 2015 download + documented NALCMS→PRMS lookup table
- SRain, WRain, Snow, keep, loss: Derived from LULC + Viger/Leavesley lookup tables
- Tree canopy (CNPY): MOD44B download + documented mosaic/resample steps
- TGF DEM: SRTM download + documented merge steps
- TGF Slope/Aspect/TWI/FDR: Documented ArcPy script (translatable to GDAL/whitebox)
- CONUS Slope/Aspect: gfv2-params Step b (GDAL-based, documented in batch scripts)
- Water bodies: NHD HR download + rasterize

**PARTIAL — source identified but Ksat/porosity details unclear:**
- `Ksat_cm_hr_1m.tif` and `poros_1m.tif` (used in gfv2-params Step 05 for soil_moist_max):
  Likely STATSGO2-derived. Not in the SB Data Layers item. Source not directly documented
  in these metadata files. The SB item has AWC from SoilGrids but not Ksat/porosity.
  → **Action needed: Check if these files exist on Hovenweep; trace their provenance.**

**MISSING — no topographic rasters for CONUS in SB items:**
- CONUS DEM (VPU-by-VPU mosaics): Must use gfv2-params Step a (download NHDPlus RPU DEMs)
  or replace with 3DEP.

### Modern Dataset Replacements (Phase 3)

| Legacy Source | Modern Replacement | Notes |
|--------------|-------------------|-------|
| SoilGrids250m v0.2 (2018) | SoilGrids250m v2.0 (2021) | Same schema, better coverage, ISRIC |
| NALCMS 2015 | NALCMS 2020 OR NLCD 2021 | NLCD better for CONUS; NALCMS better for cross-border |
| GMIS Landsat imperv (2010) | NLCD 2021 percent impervious | More recent, CONUS-focused |
| MOD44B MODIS 250m canopy | NLCD 2021 tree canopy OR GLAD 30m | Higher resolution options available |
| SRTM 30m (TGF) | 3DEP 1/3 arcsec (10m) | Better resolution; 3DEP covers CONUS+AK |
| NHDPlus RPU DEMs (CONUS) | 3DEP 1/3 arcsec (10m) | via py3dep |
| Snow depletion (Liston 2009 + Sexstone 2020) | Updated Sexstone params | Check for newer publications |
