# Design: ScienceBase GFv1.1 Raster Download CLI (#169)

## Problem

hydro-param needs access to the GFv1.1 NHM data layer rasters stored on ScienceBase
for validation and alternative parameterization. Currently there's no automated way
to download these ~15 GB of rasters.

## Design

### CLI Command

```
hydro-param gfv11 download --output-dir /shared/gfv11 [--items {data-layers,tgf-topo,all}]
```

- `--output-dir` (required): Shared data directory for downloaded rasters
- `--items` (default: `all`): Which ScienceBase item(s) to download

### Module

New `src/hydro_param/gfv11.py` containing download logic. CLI wiring in `cli.py`
adds a `gfv11` command group with `download` subcommand.

### ScienceBase Items

| Item | SB ID | Files | Size |
|------|-------|-------|------|
| Data Layers | `5ebb182b82ce25b5136181cf` | 20 | ~9.4 GB |
| TGF Topo | `5ebb17d082ce25b5136181cb` | 7 | ~5.8 GB |

### Download Mechanics

1. Query ScienceBase JSON API (`?format=json`) to get file list + download URLs
2. Stream each zip file with progress reporting (file size from API response)
3. Unzip to categorized subdirectories under `--output-dir`
4. Skip files that already exist (check extracted files, not zips)
5. Clean up zip files after successful extraction

### Directory Layout

```
{output-dir}/
  soils/          — Sand, Clay, Silt, AWC, TEXT_PRMS
  land_cover/     — LULC, Imperv, CNPY, Snow, SRain, WRain, keep, loss, RootDepth, CV_INT
  water_bodies/   — wbg
  geology/        — Lithology_exp_Konly_Project
  topo/           — dem, slope100X, asp100X, twi100X, fdr
  metadata/       — XML files, CrossWalk.xlsx, SDC_table.csv
```

### File-to-Directory Mapping

Hardcoded dict mapping each SB filename to its target subdirectory. Explicit, no magic.

### Dependencies

`requests` only — already a transitive dependency via pystac-client. No new deps needed.

### Error Handling

- Network errors: retry 3 times with backoff, then skip file with warning
- Corrupt zip: log error, skip, continue with remaining files
- Disk full: fail immediately with clear message

### No Tests

This is a CLI download utility. Validation of the downloaded rasters comes from #171.
