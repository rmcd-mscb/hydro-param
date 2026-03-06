# Design: Add Missing Segment Spatial Parameters (#159)

## Problem

The DRB reference parameter file contains three segment-dimensioned parameters
that hydro-param does not currently produce: `seg_lat`, `seg_cum_area`, and
`seg_elev`.

## Data Source Investigation

A notebook exploration (`notebooks/segment_nhd_overlay.ipynb`) confirmed:

- **`seg_lat`**: Trivial — WGS84 centroid latitude of each segment polyline.
- **`seg_cum_area`**: NHDPlus VAA `totdasqkm` column is available via pynhd.
  Uses the same spatial-join-to-NHD pipeline already built for `seg_slope`.
- **`seg_elev`**: NHDPlus VAA elevation columns (`maxelevsmo`, `minelevsmo`)
  are **missing** from the pynhd VAA parquet. Alternative: gdptools `InterpGen`
  samples 3DEP DEM along segment polylines and returns mean elevation.

## Design

### Parameter sources

| Output parameter | Source                     | Method                          | Unit conversion        |
|-----------------|----------------------------|---------------------------------|------------------------|
| `seg_lat`       | Segment centroid           | Reproject to WGS84, `.y` coord  | None (decimal degrees) |
| `seg_cum_area`  | VAA `totdasqkm`            | Spatial join (same as slope)     | km² → acres            |
| `seg_elev`      | 3DEP DEM via `InterpGen`   | Grid-to-line interpolation, mean | meters → feet          |

### Where each parameter fits in the derivation DAG

- **`seg_lat`** belongs in **Step 2 (topology)** — it's a geometric property of
  the segment, like `seg_length`. Add to `_derive_topology()` after `seg_length`.

- **`seg_cum_area`** belongs in **Step 12 (routing)** — it requires the NHDPlus
  VAA spatial join already performed there. Expand `_fetch_vaa()` to include
  `totdasqkm`, then extract cumulative area alongside slopes.

- **`seg_elev`** is a new **Step 3b** (topo-segment). It requires a DEM raster
  + segment line geometries. Add a new `_derive_segment_elevation()` method
  using gdptools `InterpGen`. Call it from the main `derive()` method after
  Step 3 (topo) and before Step 4 (landcover).

### Implementation details

**`seg_lat` (Step 2, `_derive_topology`)**
- Same pattern as `hru_lat` in `_derive_geometry()`.
- Reproject segments to EPSG:4326 if needed, compute centroid `.y`.
- Add as DataArray on `nsegment` dim.

**`seg_cum_area` (Step 12, `_derive_routing`)**
- Expand `_fetch_vaa()` to select `["comid", "slope", "totdasqkm"]`.
- Two paths (same as slope):
  - **COMID path**: Direct lookup `vaa[totdasqkm]` by COMID.
  - **Spatial join path**: Length-weighted mean `totdasqkm` from matched NHD
    flowlines (same corridor buffer).
- Convert km² to acres (1 km² = 247.105 acres).
- Fallback: sum of upstream `hru_area` values if VAA unavailable (logged warning).

**`seg_elev` (new method `_derive_segment_elevation`)**
- Requires 3DEP DEM as a local GeoTIFF (same data already fetched for HRU
  elevation in Step 3).
- Use `gdptools.UserTiffData` with segment polylines as target geometries.
- Use `gdptools.InterpGen(user_data, pt_spacing=50, stat="mean")` to compute
  mean elevation along each segment.
- Convert meters to feet (× 3.28084).
- Fallback: if DEM unavailable, emit a warning and skip (not required).

The DEM path is provided through `DerivationContext`. The existing Step 3
(`_derive_topo`) already receives the DEM path for HRU processing. We'll
reuse that same DEM path for segment interpolation.

### Changes by file

**`_derive_topology()` — add `seg_lat`**
- Reproject segments to WGS84, compute centroid latitude.
- Add DataArray on `nsegment` dim.

**`_derive_routing()` — add `seg_cum_area`**
- Expand `_fetch_vaa()` columns to include `totdasqkm`.
- Add COMID-path and spatial-join-path extraction for `totdasqkm`.
- Convert km² → acres, assign on `nsegment`.

**New `_derive_segment_elevation()` — add `seg_elev`**
- `InterpGen` with `UserTiffData` for 3DEP DEM + segment lines.
- Mean elevation per segment, meters → feet.
- Called from `derive()` after Step 3.

**`parameter_metadata.yml`**
- `seg_lat`: nsegment, decimal_degrees, [-90, 90]
- `seg_cum_area`: nsegment, acres, [0, 1e9]
- `seg_elev`: nsegment, feet, [-1000, 30000]

**Tests**
- `seg_lat`: verify centroid latitude matches expected values
- `seg_cum_area`: verify VAA spatial join and unit conversion
- `seg_elev`: verify InterpGen integration with mock DEM raster
