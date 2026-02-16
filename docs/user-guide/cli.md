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

Download dataset files via AWS CLI.

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
