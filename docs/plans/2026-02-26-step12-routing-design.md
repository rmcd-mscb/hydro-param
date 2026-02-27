# Step 12: Routing Coefficients — Design Document

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Date:** 2026-02-26
**Related issues:** #100 (NextGen hydrofabric slopes)

## Goal

Derive Muskingum routing parameters for stream segments in the pywatershed
derivation plugin, handling both NHD flowlines (with COMIDs) and
post-processed GF/PRMS segments (without COMIDs).

## Context

pywatershed's `PRMSChannel` uses Muskingum routing with Manning's equation
to compute flow velocity and travel time through stream segments.  The key
formula chain is:

```
seg_slope = max(slope_from_nhd, 1e-7)
velocity  = (1/n) × sqrt(seg_slope) × depth^(2/3) × 3600   # ft/hr
K_coef    = seg_length / velocity                            # hours
```

The Geospatial Fabric (GF) segments used by NHM-PRMS are **derived from**
NHDPlus but are not the same features.  They may be split at points of
interest (gages, confluences, reservoir outlets), trimmed to match
HRU/catchment boundaries, or merged.  This means GF segments lack COMIDs
and cannot use a simple crosswalk to NHDPlus — a spatial join is required.

### The slope problem

NHDPlus VAA slopes are known to be problematic:

- Hydroflattened DEMs artificially flatten water surfaces
- Endpoint elevation sampling is sensitive to DEM artifacts and cell resolution
- Short segments amplify vertical resolution errors

For MVP we accept these limitations and use NHDPlus VAA slopes as the best
readily available source.  Future work will investigate better alternatives
(see Future Enhancements below).

## Parameters

Step 12 produces six parameters.  Three topology parameters are already
handled by Step 2 (`_derive_topology`):

| Parameter | Dim | Units | Source | Step 2? |
|-----------|-----|-------|--------|---------|
| `tosegment` | nsegment | index (0=outlet) | GF attribute | Done |
| `hru_segment` | nhru | index | GF attribute | Done |
| `seg_length` | nsegment | meters | Geodesic geometry | Done |
| `seg_slope` | nsegment | m/m | NHDPlus VAA | **New** |
| `K_coef` | nsegment | hours | Manning's equation | **New** |
| `x_coef` | nsegment | dimensionless (0–0.5) | Default 0.2 | **New** |
| `segment_type` | nsegment | integer flag | GF attribute or derived | **New** |
| `obsin_segment` | nsegment | index (0=none) | Default 0 | **New** |

## Architecture

A new `_derive_routing()` method on `PywatershedDerivation`, called in the
`derive()` chain after Step 8 (lookups) and before Step 13 (defaults).
Follows the same pattern as existing step methods.

### Segment type detection fork

```
segments GeoDataFrame has 'COMID' or 'comid' column?
  ├─ YES → NHD path: direct COMID → VAA slope lookup
  └─ NO  → GF path: spatial join segments to NHDPlus flowlines
                     → length-weighted VAA slopes per segment
```

Detection is column-based, not data-source-based.  Any fabric that carries
COMIDs gets the fast path; everything else goes through the spatial join.

### Data flow

1. **Detect segment type** — check for `COMID`/`comid` column in `ctx.segments`
2. **Fetch NHDPlus flowlines + VAA** — use pynhd to get flowlines with slope
   for the bounding box of all segments
3. **Get slope per segment:**
   - **NHD path:** join on COMID, read `slope` directly from VAA
   - **GF path:** spatial join (`gpd.sjoin` or overlay), compute intersection
     lengths, length-weighted mean slope per segment
4. **Compute K_coef** via Manning's equation (see Constants below)
5. **Assign segment_type** — pass through from GF column if present, else
   default 0 (channel); lake segments get K_coef = 24.0
6. **Assign x_coef** — constant 0.2
7. **Assign obsin_segment** — constant 0 (not implemented in pywatershed)
8. **Write** all parameters to output dataset on `nsegment` dimension

### Guards

| Condition | Behavior |
|-----------|----------|
| `ctx.segments is None` | Warn, return defaults (K_coef=1.0, x_coef=0.2, seg_slope=1e-4) |
| Segment has no NHDPlus match after spatial join | Warn per segment, use fallback slope (1e-4) |
| `seg_length` missing or zero for a segment | Warn, use default K_coef=1.0 |
| `seg_slope` < 1e-7 | Clamp to 1e-7 (matching pywatershed floor) |
| `K_coef` outside [0.01, 24.0] | Clamp to bounds (matching pywatershed) |
| Lake segment (segment_type = lake) | Force K_coef = 24.0 |

### Constants

```python
_MANNING_N = 0.04           # natural channel roughness coefficient
_DEFAULT_DEPTH_FT = 1.0     # bankfull depth placeholder (feet)
_MIN_SLOPE = 1e-7           # pywatershed floor for seg_slope
_FALLBACK_SLOPE = 1e-4      # for segments with no NHDPlus match
_K_COEF_MIN = 0.01          # hours
_K_COEF_MAX = 24.0          # hours
_DEFAULT_K_COEF = 1.0       # hours, used when computation not possible
_DEFAULT_X_COEF = 0.2       # standard Muskingum weighting
_LAKE_K_COEF = 24.0         # travel time for lake segments
```

### Dependencies

- **pynhd** — fetch NHDPlus flowlines and VAA for the segment bounding box.
  Not currently a project dependency; must be added.
- **geopandas** — spatial join (already a dependency)
- **pyproj** — CRS alignment for spatial join (already a dependency)

pynhd's `nhdplus_vaa()` downloads a 245 MB parquet file (cached locally)
containing slope and other attributes for all 2.7M NHDPlus flowlines.
Flowline geometries for the spatial join can be fetched via
`pynhd.NHDPlusHR()` or `pynhd.WaterData()` for a bounding box.

### segment_type values

| Value | Meaning | K_coef behavior |
|-------|---------|-----------------|
| 0 | Channel (default) | Computed from Manning's equation |
| 1 | Lake | Forced to 24.0 hours |

Detection logic for lake segments:
1. If `segment_type` column exists in segments GeoDataFrame → use it directly
2. Else → default all segments to 0 (channel)

Lake detection from waterbody overlay (Step 6) is a future enhancement —
connecting `hru_type` or `dprst_frac` back to segments requires the
HRU-to-segment mapping and a dominance threshold.

## NHDPlus VAA spatial join — detailed approach

For GF/PRMS segments without COMIDs:

1. Compute the bounding box of all segments (with buffer)
2. Fetch NHDPlus flowlines within that bbox via pynhd
3. Reproject to a common CRS if needed
4. Spatial join: for each GF segment, find all NHDPlus flowlines that
   intersect it
5. For each match, compute the length of the intersection
6. Weight VAA slopes by intersection length:
   ```
   seg_slope_i = sum(slope_j × intersection_length_j) / sum(intersection_length_j)
   ```
7. Segments with no matches → fallback slope (1e-4) + warning

This approach is fabric-agnostic: it works regardless of how segments were
derived from NHD, whether they've been split at POIs, trimmed to catchment
boundaries, or merged.

## Future Enhancements

1. **NextGen hydrofabric slopes** (issue #100) — replace NHDPlus VAA with
   ML-informed cross-section-derived slopes from NOAA OWP / Lynker-spatial
   geopackages.  These derive flowpath slopes from transects perpendicular
   to flowlines with DEM elevations + estimated bathymetry, potentially
   more accurate than NHDPlus VAA.
   - Data: https://noaa-owp.github.io/hydrofabric/
   - HydroShare: https://www.hydroshare.org/resource/129787b468aa4d55ace7b124ed27dbde/

2. **Bankfull depth from BANKFULL_CONUS** — replace constant 1.0 ft default
   with per-segment bankfull depth from regional regressions on drainage
   area by physiographic division (Bieger et al. 2015).
   - Data: https://www.sciencebase.gov/catalog/item/5cf02bdae4b0b51330e22b85
   - Contains bankfull width, depth, and cross-sectional area for every
     NHDPlus v2.1 flowline, keyed by COMID
   - R² ~ 0.85 for width, 0.69 for depth

3. **EROM velocity validation** — use NHDPlus EROM `VEMA` (gage-adjusted
   mean annual velocity) as a cross-check against Manning-derived velocity.

4. **SWORD/SWOT satellite slopes** — for larger rivers (>=30m wide), the
   SWOT River Database provides satellite-derived water surface slopes
   independent of DEMs.  Global coverage, ~10 km reaches.
   - Data: https://zenodo.org/records/10013982

5. **NWIS gage-to-segment mapping** for `obsin_segment` — spatial join of
   USGS streamgage locations to nearest segment.  Blocked on pywatershed
   implementing observed inflow injection (currently a TODO in their code).

6. **Lake segment detection from waterbody overlay** — connect Step 6
   waterbody results to segment_type via hru_segment mapping.

## References

- Hay, L.E., et al. 2023. Parameter estimation at the CONUS scale and
  streamflow routing enhancements for NHM-PRMS. USGS TM 6-B10.
- Regan, R.S., et al. 2018. Description of the NHM for use with PRMS.
  USGS TM 6-B9.
- Bieger, K., et al. 2015. Bankfull hydraulic geometry related to
  physiographic divisions. (via USGS ScienceBase 5cf02bdae4b0b51330e22b85)
- Altenau, E.H., et al. 2021. The SWOT Mission River Database (SWORD).
  Water Resources Research, 57(6).
- Frasson, R.P.M., et al. 2019. Global database of river width, slope,
  catchment area, meander wavelength, sinuosity, and discharge. Zenodo.
