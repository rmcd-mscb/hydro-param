# hydro-param Pipeline Flowchart

```mermaid
flowchart TB
    START(["Config + Registry"]) --> S1A

    subgraph Stage1["Stage 1: Resolve Fabric"]
        S1A["Load GeoPackage fabric"] --> S1B["Bbox filter"]
        S1B --> S1C["Spatial batching"]
    end

    S1C --> S2A

    subgraph Stage2["Stage 2: Resolve Datasets"]
        S2A["Match names to registry"] --> S2B["Resolve variables"]
        S2B --> S2C["Determine strategy"]
    end

    S2C --> S3

    subgraph Stage3["Stage 3: Weights"]
        S3["Computed internally by gdptools"]
    end

    S3 --> LOOP

    subgraph Stage4["Stage 4: Process Datasets + Incremental Writes"]
        LOOP{"For each dataset"}

        LOOP -->|temporal| T1["Split by calendar year"]
        LOOP -->|static| B0["Expand years"]

        T1 --> T2{"Strategy?"}
        T2 -->|nhgf_stac| T3["WeightGen + AggGen"]
        T2 -->|climr_cat| T4["WeightGen + AggGen"]
        T3 --> T5["xr.Dataset per year"]
        T4 --> T5
        T5 --> TW["Write temporal file"]

        B0 --> B1{"For each year"}
        B1 --> B2{"For each batch"}
        B2 --> B3{"Strategy?"}
        B3 -->|stac_cog| B4["fetch_stac_cog"]
        B3 -->|local_tiff| B5["fetch_local_tiff"]
        B3 -->|nhgf_stac| B6["nhgf_stac direct"]
        B4 --> B7{"Var type?"}
        B5 --> B7
        B7 -->|raw| B8["Save GeoTIFF"]
        B7 -->|derived| B9["Derive from source"]
        B9 --> B8
        B8 --> B10["ZonalGen exactextract"]
        B6 --> B11["DataFrame per batch"]
        B10 --> B11

        B11 --> MERGE["Concat batches"]
        MERGE --> VW["Write per-variable file"]
    end

    VW --> S5A
    TW --> S5A

    subgraph Stage5["Stage 5: Format Output - Combined SIR"]
        S5A["Assemble SIR xr.Dataset"] --> S5B["Write combined SIR .nc"]
    end

    S5B --> RESULT(["PipelineResult"])
```

## Data Flow Summary

| Stage | Input | Output | Key Module |
|-------|-------|--------|------------|
| 1 | GeoPackage path + bbox | GeoDataFrame with `batch_id` | `pipeline.py`, `batching.py` |
| 2 | Dataset names from config | `(DatasetEntry, DatasetRequest, [VarSpec])` tuples | `dataset_registry.py` |
| 3 | *(internal to gdptools)* | Spatial weights | gdptools |
| 4 | Fabric + resolved datasets | `Stage4Results` + per-variable/temporal files written to disk | `processing.py`, `data_access.py`, `pipeline.py` |
| 5 | Stage4Results + config | Combined SIR `.nc` → `PipelineResult` | `pipeline.py` |

## Processing Strategy Matrix

| Strategy | gdptools Class | Static/Temporal | Batched? | Path |
|----------|---------------|----------------|----------|------|
| `stac_cog` | `UserTiffData` | Static | Yes (spatial) | fetch raster - GeoTIFF - `ZonalGen` |
| `local_tiff` | `UserTiffData` | Static | Yes (spatial) | load local - GeoTIFF - `ZonalGen` |
| `nhgf_stac` (static) | `NHGFStacTiffData` | Static | Yes (spatial) | direct - `ZonalGen` |
| `nhgf_stac` (temporal) | `NHGFStacData` | Temporal | No (full fabric) | `WeightGen` - `AggGen` |
| `climr_cat` | `ClimRCatData` | Temporal | No (full fabric) | `WeightGen` - `AggGen` |
