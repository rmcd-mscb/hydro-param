# Project Initialization

hydro-param uses a project directory structure to organize data, configs,
and outputs. This makes each project a reproducible modeling artifact
documenting the full provenance chain.

## Create a project

```bash
hydro-param init my-watershed
```

This creates:

```
my-watershed/
├── .hydro-param           # Project marker
├── configs/
│   └── pipeline.yml       # Template pipeline configuration
├── data/
│   ├── fabrics/           # Target polygon files
│   ├── climate/           # Climate datasets
│   ├── land_cover/        # Land cover datasets
│   ├── soils/             # Soil datasets
│   ├── topography/        # DEM and terrain datasets
│   └── ...                # Other categories
├── output/                # Pipeline results
├── models/                # Model exports
└── .gitignore
```

## Re-initialize

Use `--force` to refresh an existing project. This creates missing directories
and updates the marker, but never overwrites `pipeline.yml`:

```bash
hydro-param init --force
```

## Project detection

hydro-param detects whether you're inside a project by walking up the
directory tree looking for the `.hydro-param` marker file (similar to how
git finds `.git`). When inside a project, commands like `datasets download`
automatically route files to the correct `data/<category>/` subdirectory.
