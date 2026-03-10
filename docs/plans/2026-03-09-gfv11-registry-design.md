# Design: GFv1.1 Dataset Registry Integration

## Problem

The GFv1.1 ScienceBase rasters are downloaded via `hydro-param gfv11 download` but
are invisible to the pipeline — there are no dataset registry entries for them.
Users must manually configure `source` paths in their pipeline config for each
raster. The registry should know about these datasets automatically after download.

## Design Decisions

1. **Local registry overlay.** A user-local directory (`~/.hydro-param/datasets/`)
   extends the bundled registry. The registry loader merges bundled + user-local
   entries. User-local entries win on name collision (full replacement, no partial
   merge). Project-level overlays are deferred — user-global only for now.

2. **Download auto-registers.** The `gfv11 download` command writes
   `~/.hydro-param/datasets/gfv11.yml` after successful download. Each entry has
   a resolved absolute `source` path. The user never manually configures paths.

3. **All 21 rasters registered.** Every downloaded raster gets a registry entry,
   including pre-computed lookup-derived parameters (SRain, WRain, Snow, etc.)
   and topographic derivatives (aspect, flow direction). Available if needed.

4. **`scale_factor` on VariableSpec.** Integer-encoded rasters (slope100X, asp100X,
   twi100X) declare `scale_factor: 0.01` so consumers know the encoding is
   machine-readable, not just documented in a description string. Follows
   CF-conventions. The pipeline passes it through as metadata; consumers apply it.

5. **CRS is EPSG:5070.** All GFv1.1 rasters use ESRI:102039 (or EPSG:5070 for wbg).
   pyproj confirms these are identical. Registry entries use `EPSG:5070` uniformly.

## Scope

**In this issue (#170):**
- `VariableSpec` gains `scale_factor: float | None = None`
- GFv1.1 dataset definitions hardcoded in `gfv11.py` (variable names, units, CRS,
  categorical flags, scale_factor)
- Registry overlay loader in `dataset_registry.py` — `~/.hydro-param/datasets/`
  scanning, merged after bundled entries
- Auto-registration in `gfv11 download` — writes overlay YAML with resolved paths
- `datasets list`/`info` pick up overlay entries automatically
- Tests for overlay loading, auto-registration, scale_factor field

**Separate issues:**
- #182 — Themed pipeline config (datasets organized by category)
- #183 — Remove `engine` from `ProcessingConfig`
- #184 — Consolidated dataset catalog CLI command
- #185 — pywatershed plugin skip logic for pre-computed lookup-derived parameters

## Dataset Inventory

All 21 rasters from the two ScienceBase items:

### Soils (5 datasets)

| Registry Name | File | Variable | Units | Type | Notes |
|--------------|------|----------|-------|------|-------|
| gfv11_sand | Sand.tif | sand_pct | % | continuous | SoilGrids250m depth-weighted |
| gfv11_clay | Clay.tif | clay_pct | % | continuous | SoilGrids250m depth-weighted |
| gfv11_silt | Silt.tif | silt_pct | % | continuous | SoilGrids250m derived (100 - sand - clay) |
| gfv11_awc | AWC.tif | awc | mm | continuous | SoilGrids250m available water capacity |
| gfv11_text_prms | TEXT_PRMS.tif | soil_type | class | categorical | USDA texture → PRMS soil_type codes |

### Land Cover (10 datasets)

| Registry Name | File | Variable | Units | Type | Notes |
|--------------|------|----------|-------|------|-------|
| gfv11_lulc | LULC.tif | cov_type | class | categorical | NALCMS 2015 → PRMS cov_type (1–7) |
| gfv11_imperv | Imperv.tif | imperv_pct | % | continuous | GMIS impervious surface |
| gfv11_cnpy | CNPY.tif | canopy_pct | % | continuous | MODIS tree canopy cover |
| gfv11_srain | SRain.tif | srain_intcp | inches | continuous | Pre-computed summer rain interception |
| gfv11_wrain | WRain.tif | wrain_intcp | inches | continuous | Pre-computed winter rain interception |
| gfv11_snow | Snow.tif | snow_intcp | inches | continuous | Pre-computed snow interception |
| gfv11_covden_win | keep.tif | covden_win | fraction | continuous | Pre-computed winter cover density |
| gfv11_covden_loss | loss.tif | covden_loss | fraction | continuous | Pre-computed seasonal cover loss |
| gfv11_covden_sum | CV_INT.tif | covden_sum | fraction | continuous | Pre-computed summer cover density |
| gfv11_root_depth | RootDepth.tif | root_depth | inches | continuous | Pre-computed root depth |

### Water Bodies (1 dataset)

| Registry Name | File | Variable | Units | Type | Notes |
|--------------|------|----------|-------|------|-------|
| gfv11_wbg | wbg.tif | waterbody | class | categorical | NHD HR waterbody mask |

### Topography (5 datasets)

| Registry Name | File | Variable | Units | Type | Notes |
|--------------|------|----------|-------|------|-------|
| gfv11_dem | dem.tif | elevation | m | continuous | SRTM 30m DEM (TGF domain) |
| gfv11_slope | slope100X.tif | slope | degrees | continuous | scale_factor: 0.01 |
| gfv11_aspect | asp100X.tif | aspect | degrees | continuous | scale_factor: 0.01 |
| gfv11_twi | twi100X.tif | twi | unitless | continuous | scale_factor: 0.01 |
| gfv11_fdr | fdr.tif | flow_dir | class | categorical | D8 flow direction |

## Auto-Generated Overlay Example

After `hydro-param gfv11 download --output-dir /mnt/e/data/hydro_param`:

```yaml
# Auto-generated by: hydro-param gfv11 download
# Source: ScienceBase items 5ebb182b / 5ebb17d0
# Downloaded: 2026-03-09

datasets:
  gfv11_sand:
    description: "GFv1.1 SoilGrids250m sand %, 250m, CONUS"
    strategy: local_tiff
    source: /mnt/e/data/hydro_param/soils/Sand.tif
    crs: "EPSG:5070"
    x_coord: "x"
    y_coord: "y"
    category: soils
    temporal: false
    variables:
      - name: sand_pct
        band: 1
        units: "%"
        long_name: "Depth-weighted sand percentage (SoilGrids250m)"
        native_name: "sand_pct"
        categorical: false

  gfv11_slope:
    description: "GFv1.1 TGF terrain slope, 30m (integer-encoded × 100)"
    strategy: local_tiff
    source: /mnt/e/data/hydro_param/topo/slope100X.tif
    crs: "EPSG:5070"
    x_coord: "x"
    y_coord: "y"
    category: topography
    temporal: false
    variables:
      - name: slope
        band: 1
        units: "degrees"
        long_name: "Terrain slope"
        native_name: "slope"
        categorical: false
        scale_factor: 0.01

  # ... remaining 19 datasets follow the same pattern
```

## Registry Loader Changes

`dataset_registry.py` `load_registry()` currently loads only from
`importlib.resources`. The change:

```
1. Load bundled entries from src/hydro_param/data/datasets/*.yml
2. Scan ~/.hydro-param/datasets/*.yml (if directory exists)
3. Merge: user-local entries added to registry; on name collision,
   user-local replaces the bundled entry entirely
```

The `~/.hydro-param/` directory and `datasets/` subdirectory are created by
the download command on first use — no manual setup required.

## Data Flow

```
hydro-param gfv11 download --output-dir /mnt/e/data/hydro_param
  │
  ├─ Downloads + extracts 21 rasters to subdirectories
  │
  └─ Writes ~/.hydro-param/datasets/gfv11.yml
       │   (21 dataset entries with resolved source paths)
       │
       ▼
hydro-param datasets list
  │   (shows bundled + overlay entries)
  │
  ▼
Pipeline config references by name:
  datasets:
    soils:
      - name: gfv11_sand        ← resolved via overlay (no source needed)
        variables: [sand_pct]
        statistics: [mean]
      - name: polaris_30m        ← resolved via bundled registry
        variables: [sand]
        statistics: [mean]
    topography:
      - name: dem_3dep_10m       ← resolved via bundled STAC catalog
        variables: [elevation]
        statistics: [mean]
```
