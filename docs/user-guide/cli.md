# CLI Reference

hydro-param provides a command-line interface for project management,
dataset discovery, pipeline execution, and model-specific parameterization.

```
Usage: hydro-param COMMAND

Configuration-driven hydrologic parameterization.

Commands: datasets, gfv11, init, pywatershed, run, --help, --version
```

## Top-Level Commands

| Command | Description |
|---------|-------------|
| [`init`](#project-management) | Scaffold a new hydro-param project directory |
| [`datasets`](#dataset-discovery) | List, inspect, and download registered datasets |
| [`run`](#pipeline-execution) | Execute the generic parameterization pipeline (stages 1--5) |
| [`pywatershed`](#pywatershed-model-setup) | Derive PRMS parameters and validate output |
| [`gfv11`](#gfv11-data-layers) | Download GFv1.1 NHM data layer rasters from ScienceBase |

---

## Project Management

### `hydro-param init`

Scaffold a new hydro-param project directory.

```
Usage: hydro-param init [OPTIONS] [ARGS]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `PROJECT-DIR` | Directory to initialise. | `.` |
| `--force` / `--no-force` | Re-initialise an existing project. | `False` |
| `--registry` | Path to a custom registry YAML. | — |

**Example:**

```bash
# Create a new project in the current directory
hydro-param init

# Create a project in a specific directory, overwriting if it exists
hydro-param init my-project --force
```

---

## Dataset Discovery

The `datasets` subcommand provides tools for exploring and downloading
the bundled dataset registry.

### `hydro-param datasets list`

Display all registered datasets grouped by category.

```
Usage: hydro-param datasets list [OPTIONS]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--registry` | Path to a custom registry YAML file or directory. | — |

**Example:**

```bash
hydro-param datasets list
```

### `hydro-param datasets info`

Show full details for a single dataset.

```
Usage: hydro-param datasets info [OPTIONS] NAME
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `NAME` *(required)* | Dataset name (e.g., `dem_3dep_10m`). | — |
| `--registry` | Path to a custom registry YAML. | — |

**Example:**

```bash
hydro-param datasets info dem_3dep_10m
```

### `hydro-param datasets download`

Download dataset files via the AWS CLI.

```
Usage: hydro-param datasets download NAME [--dest PATH] [--years YEARS] [--variables VARS] [--registry PATH]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `NAME` *(required)* | Dataset name to download. | — |
| `--dest` | Destination directory for downloaded files. | — |
| `--years` | Comma-separated years to download. | — |
| `--variables` | Comma-separated variables or products. | — |
| `--registry` | Path to a custom registry YAML. | — |

!!! note

    Downloads require the [AWS CLI](https://aws.amazon.com/cli/) to be
    installed and available on your `PATH`.

**Example:**

```bash
hydro-param datasets download nlcd_annual --dest ./data/landcover --years 2019,2021
```

---

## Pipeline Execution

### `hydro-param run`

Execute the generic parameterization pipeline. Runs stages 1--5 to produce
a normalized Standardized Internal Representation (SIR).

```
Usage: hydro-param run [OPTIONS] CONFIG
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CONFIG` *(required)* | Path to pipeline YAML config file. | — |
| `--registry` | Path to a custom dataset registry. | — |
| `--resume` / `--no-resume` | Enable manifest-based resume. | `False` |

**Example:**

```bash
# Run the pipeline
hydro-param run configs/examples/drb_pipeline.yml

# Resume a previously interrupted run
hydro-param run configs/examples/drb_pipeline.yml --resume
```

---

## pywatershed Model Setup

The `pywatershed` subcommand derives PRMS parameters from pre-built SIR
output and validates the resulting parameter files.

### `hydro-param pywatershed run`

Generate a complete pywatershed model setup from existing SIR output.
Consumes pre-built SIR and derives all PRMS parameters (Phase 2 only --
run `hydro-param run` first to produce the SIR).

```
Usage: hydro-param pywatershed run CONFIG
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `CONFIG` *(required)* | Path to pywatershed run config YAML (v4.0). | — |

**Output files:**

| File | Description |
|------|-------------|
| `parameters.nc` | Static PRMS parameters (CF-1.8 NetCDF) |
| `forcing/<var>.nc` | Forcing time series (precipitation, temperature) |
| `soltab.nc` | Potential solar radiation tables (nhru x 366) |
| `control.yml` | Simulation time period configuration |

**Example:**

```bash
hydro-param pywatershed run configs/examples/drb_pywatershed.yml
```

### `hydro-param pywatershed validate`

Validate a pywatershed parameter file against metadata constraints.
Checks that required PRMS parameters are present and values fall within
valid ranges defined in the bundled `parameter_metadata.yml`.

```
Usage: hydro-param pywatershed validate PARAM-FILE
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `PARAM-FILE` *(required)* | Path to parameter NetCDF file. | — |

**Example:**

```bash
hydro-param pywatershed validate output/parameters.nc
```

---

## GFv1.1 Data Layers

The `gfv11` subcommand downloads Geospatial Fabric v1.1 NHM data layer
rasters from USGS ScienceBase.

### `hydro-param gfv11 download`

Download GFv1.1 NHM data layer rasters from ScienceBase (~15 GB total).

```
Usage: hydro-param gfv11 download --output-dir PATH [OPTIONS]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--output-dir` *(required)* | Shared data directory for downloaded rasters. | — |
| `--items` | Which ScienceBase item(s): `all`, `data-layers`, or `tgf-topo`. | `all` |

**Example:**

```bash
# Download all GFv1.1 rasters
hydro-param gfv11 download --output-dir /data/gfv11

# Download only the topographic data
hydro-param gfv11 download --output-dir /data/gfv11 --items tgf-topo
```
