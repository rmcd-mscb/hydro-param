# Datasets

hydro-param includes a built-in registry of geospatial datasets organized
by category. The registry defines how each dataset is accessed, what variables
it provides, and which processing strategy to use.

## Dataset Categories

| Category | Datasets | Description |
|----------|----------|-------------|
| topography | 3DEP 10m DEM | Elevation, slope, aspect |
| soils | POLARIS 30m, gNATSGO | Soil texture, AWC, hydraulic conductivity |
| land_cover | NLCD Annual (6 collections) | Land cover, impervious surface, change |
| climate | gridMET | Precipitation, temperature, radiation, wind |
| snow | SNODAS | Daily snow water equivalent |
| hydrography | NHDPlus | Flowlines, catchments |
| water_bodies | NHDPlus waterbodies | Lakes, reservoirs |

## Data Access Strategies

hydro-param routes each dataset through one of five processing strategies
implemented in gdptools. The strategy determines how source data is fetched
and how zonal statistics are computed against the target fabric.

| Strategy | gdptools Class | Datasets | Access Method |
|----------|---------------|----------|---------------|
| `stac_cog` | `UserTiffData` | 3DEP 10m, gNATSGO | Planetary Computer STAC |
| `local_tiff` | `UserTiffData` | POLARIS 30m, GFv1.1 rasters | Local GeoTIFF files |
| `nhgf_stac` (static) | `NHGFStacTiffData` | NLCD Annual (6 collections) | USGS NHGF STAC on OSN |
| `nhgf_stac` (temporal) | `NHGFStacData` | SNODAS, CONUS404-BA | USGS NHGF STAC |
| `climr_cat` | `ClimRCatData` | gridMET | OPeNDAP via ClimateR catalog |

**Static strategies** (`stac_cog`, `local_tiff`, `nhgf_stac` static) produce
zonal statistics through `ZonalGen`, which computes per-feature summaries
(mean, majority, fractions) in a single pass.

**Temporal strategies** (`nhgf_stac` temporal, `climr_cat`) use a two-step
workflow: `WeightGen` pre-computes spatial weights, then `AggGen` applies
those weights across timesteps to produce time series output.

## GFv1.1 National Rasters

The Geospatial Fabric version 1.1 (GFv1.1) provides 14 national-scale rasters
covering soils, land cover, snow, and topography. These are the same datasets
used to parameterize NHM v1.1.

### Downloading GFv1.1

Use the CLI to fetch all GFv1.1 rasters from ScienceBase (~15 GB total):

```bash
hydro-param gfv11 download --output-dir /path/to/data/gfv11
```

The download command:

- Fetches all 14 rasters from their ScienceBase items
- Extracts any compressed archives automatically
- Registers a user-local overlay at `~/.hydro-param/datasets/gfv11.yml`
  so that GFv1.1 datasets appear in `datasets list` immediately

Once downloaded, GFv1.1 datasets can be referenced in pipeline configs using
the `local_tiff` access strategy.

## User-Local Registry Overlays

The bundled dataset registry ships with the package as 8 YAML files in
`src/hydro_param/data/datasets/`. Users can extend this registry without
modifying the package by placing additional YAML files in
`~/.hydro-param/datasets/`.

**How overlays work:**

- At load time, the registry loader merges overlay datasets with the
  bundled registry
- Overlay entries can add new datasets or override existing ones
- The `gfv11 download` command creates an overlay automatically
- Any YAML file in `~/.hydro-param/datasets/` is picked up as an overlay

This allows local or organization-specific datasets to be registered
alongside the built-in catalog.

## Discovering Datasets

List all available datasets (including any user-local overlays):

```bash
hydro-param datasets list
```

Filter by category:

```bash
hydro-param datasets list --category soils
```

Get detailed information for a specific dataset, including its variables,
access strategy, and source metadata:

```bash
hydro-param datasets info nlcd_osn_lndcov
```

## Dataset Registry Schema

See the [API Reference](../api/registry.md) for full details on registry models:

- [`DatasetEntry`](../api/registry.md#hydro_param.dataset_registry.DatasetEntry) --- Schema for a single dataset entry, including variables, access strategy, and source URL
- [`DownloadInfo`](../api/registry.md#hydro_param.dataset_registry.DownloadInfo) --- Download provenance and URL templates
