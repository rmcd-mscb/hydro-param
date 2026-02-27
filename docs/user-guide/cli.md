# CLI Reference

hydro-param provides a command-line interface for project management,
dataset discovery, and pipeline execution.

## Commands

### `hydro-param init`

Scaffold a new project directory.

```bash
hydro-param init [PROJECT_DIR] [--force] [--registry PATH]
```

| Argument | Description |
|---|---|
| `PROJECT_DIR` | Directory to initialize (default: current directory) |
| `--force` | Re-initialize existing project |
| `--registry` | Custom registry path |

### `hydro-param datasets list`

Display all datasets grouped by category.

```bash
hydro-param datasets list [--registry PATH]
```

### `hydro-param datasets info`

Show full details for a specific dataset.

```bash
hydro-param datasets info NAME [--registry PATH]
```

### `hydro-param datasets download`

Download dataset files to local storage.

```bash
hydro-param datasets download NAME [--dest PATH] [--years YEARS] [--variables VARS] [--registry PATH]
```

| Option | Description |
|---|---|
| `--dest` | Destination directory (auto-routes to `data/<category>/` in projects) |
| `--years` | Comma-separated years to download |
| `--variables` | Comma-separated variables/products |

!!! note

    Downloads require the [AWS CLI](https://aws.amazon.com/cli/) to be
    installed and available on your PATH.

### `hydro-param run`

Execute the parameterization pipeline.

```bash
hydro-param run CONFIG [--registry PATH]
```

### `hydro-param pywatershed run`

Generate a complete pywatershed model setup (parameters, forcing, control).

```bash
hydro-param pywatershed run CONFIG [--registry PATH]
```

Runs a two-phase workflow:

1. **Phase 1 (generic pipeline):** Fetch and process source datasets
   (topography, land cover, soils, climate) via the 5-stage pipeline.
2. **Phase 2 (pywatershed derivation):** Derive PRMS parameters from
   the raw SIR, merge temporal climate data with unit conversions, and
   write output files.

| Option | Description |
|---|---|
| `CONFIG` | Path to a pywatershed run config YAML file |
| `--registry` | Custom dataset registry path |

**Output files:**

| File | Description |
|------|-------------|
| `parameters.nc` | Static PRMS parameters (CF-1.8 NetCDF) |
| `forcing/prcp.nc` | Precipitation forcing (inches/day) |
| `forcing/tmax.nc` | Maximum temperature forcing (degrees F) |
| `forcing/tmin.nc` | Minimum temperature forcing (degrees F) |
| `soltab.nc` | Potential solar radiation tables (nhru x 366) |
| `control.yml` | Simulation time period configuration |

### `hydro-param pywatershed validate`

Validate a pywatershed parameter file against metadata constraints.

```bash
hydro-param pywatershed validate PARAM_FILE
```

Checks that required PRMS parameters are present and values fall within
valid ranges defined in the bundled `parameter_metadata.yml`.

| Argument | Description |
|---|---|
| `PARAM_FILE` | Path to a pywatershed parameter NetCDF file |
