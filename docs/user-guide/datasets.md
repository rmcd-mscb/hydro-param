# Datasets

hydro-param includes a built-in registry of geospatial datasets organized
by category.

## Categories

| Category | Description |
|---|---|
| topography | DEM, slope, aspect |
| land_cover | NLCD legacy epochs, NLCD annual products |
| soils | POLARIS, gSSURGO |
| climate | gridMET, Daymet |
| hydrography | NHDPlus |
| geology | Hydrogeologic data |
| snow | Snow products |
| water_bodies | Lakes, reservoirs |

## Discovering Datasets

List all available datasets:

```bash
hydro-param datasets list
```

Get details for a specific dataset:

```bash
hydro-param datasets info nlcd_annual
```

## Downloading Data

Some datasets require local files. Use the download command:

```bash
hydro-param datasets download nlcd_annual --years 2021 --variables LndCov
```

When inside an initialized project, files are automatically routed to the
correct `data/<category>/` subdirectory.

## Dataset Registry Schema

See the [API Reference](../api/registry.md) for full details on registry models:

- [`DatasetEntry`](../api/registry.md#hydro_param.dataset_registry.DatasetEntry) --- Schema for a single dataset
- [`DownloadInfo`](../api/registry.md#hydro_param.dataset_registry.DownloadInfo) --- Download provenance and URL templates
