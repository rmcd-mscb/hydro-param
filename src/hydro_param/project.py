"""Project scaffolding: init command, project root detection, template generation.

Provides the ``hydro-param init`` functionality that creates a standard project
directory structure with categorical data subfolders and a template pipeline
configuration.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

MARKER_FILE = ".hydro-param"

# Fallback categories when registry is unavailable.
# These match the per-category YAML files in configs/datasets/.
DEFAULT_CATEGORIES: list[str] = [
    "climate",
    "geology",
    "hydrography",
    "land_cover",
    "snow",
    "soils",
    "topography",
    "water_bodies",
]


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* looking for a ``.hydro-param`` marker file.

    Parameters
    ----------
    start
        Directory to begin searching from.  Defaults to the current
        working directory.

    Returns
    -------
    Path or None
        The project root directory, or ``None`` if no marker is found.
    """
    current = (start or Path.cwd()).resolve()
    while True:
        if (current / MARKER_FILE).is_file():
            return current
        parent = current.parent
        if parent == current:
            return None
        current = parent


def get_data_categories(registry_path: Path | None = None) -> list[str]:
    """Discover dataset categories, falling back to built-in defaults.

    Parameters
    ----------
    registry_path
        Path to a registry YAML file or directory.  When ``None`` or
        unloadable, :data:`DEFAULT_CATEGORIES` is returned.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of category names.
    """
    if registry_path is None:
        return list(DEFAULT_CATEGORIES)
    try:
        from hydro_param.dataset_registry import load_registry

        reg = load_registry(registry_path)
        categories = {entry.category for entry in reg.datasets.values() if entry.category}
        if categories:
            return sorted(categories | set(DEFAULT_CATEGORIES))
    except Exception:
        logger.debug("Could not load registry for category discovery; using defaults")
    return list(DEFAULT_CATEGORIES)


def generate_pipeline_template(project_name: str) -> str:
    """Generate a well-commented pipeline config template.

    Parameters
    ----------
    project_name
        Project name used in the ``output.sir_name`` field.

    Returns
    -------
    str
        YAML content for ``configs/pipeline.yml``.
    """
    return f"""\
# Pipeline configuration for {project_name}
#
# This file drives the hydro-param parameterization pipeline.
# Edit the sections below, then run:
#   hydro-param run configs/pipeline.yml
#
# For dataset details run:
#   hydro-param datasets list
#   hydro-param datasets info <dataset-name>

# --- Target Fabric ---
# The polygon mesh (catchments, HRUs, grid cells) to parameterize.
# Place your GeoPackage or Parquet file in data/fabrics/.
target_fabric:
  path: "data/fabrics/catchments.gpkg"   # GeoPackage, GeoParquet, or Shapefile
  id_field: "featureid"                  # Unique ID column in the fabric
  crs: "EPSG:4326"                       # CRS of the fabric file

# --- Domain ---
# Spatial extent for the analysis.
# Supported types: bbox, huc2, huc4, gage.
domain:
  type: bbox
  bbox: [-76.5, 38.5, -74.0, 42.6]      # [west, south, east, north] in EPSG:4326

# --- Datasets ---
# Each entry references a dataset from the registry by name.
# Use 'hydro-param datasets list' to see available datasets.
#
# For local_tiff datasets, set 'source' to the downloaded file path.
# Downloads auto-route to data/<category>/ inside an initialized project.
datasets:
  # DEM example (remote via STAC — no download needed):
  - name: dem_3dep_10m
    variables:
      - elevation
      - slope
      - aspect
    statistics:
      - mean

  # NLCD example (requires download first):
  # 1. Run: hydro-param datasets download nlcd_legacy --years 2021
  # 2. Uncomment and update the source path below:
  # - name: nlcd_legacy
  #   source: data/land_cover/nlcd_2021_land_cover_l48_20230630.tif
  #   variables:
  #     - land_cover
  #   statistics:
  #     - majority

# --- Output ---
# Where and how to write results.
output:
  path: "output"                          # Output directory (created automatically)
  format: netcdf                          # netcdf or parquet
  sir_name: "{project_name}"             # Name for the output file

# --- Processing ---
# Engine and batching options.
processing:
  engine: exactextract                    # exactextract or serial
  failure_mode: strict                    # strict (fail fast) or tolerant (log and continue)
  batch_size: 500                         # Number of features per spatial batch
"""


def generate_gitignore() -> str:
    """Generate ``.gitignore`` content for a hydro-param project."""
    return """\
# hydro-param project
#
# Track: .hydro-param marker, configs/
# Ignore: downloaded data, pipeline output, model exports

# Downloaded data (large raster/vector files)
data/topography/
data/land_cover/
data/soils/
data/geology/
data/hydrography/
data/climate/
data/snow/
data/water_bodies/
data/fabrics/*.tif
data/fabrics/*.tiff

# Pipeline output
output/

# Model exports
models/

# Common large geospatial formats (safety net)
*.nc
*.tif
*.tiff
*.zarr/
"""


def init_project(
    project_dir: Path,
    *,
    force: bool = False,
    registry_path: Path | None = None,
) -> None:
    """Scaffold a hydro-param project directory.

    Creates the standard directory structure, a ``.hydro-param`` marker
    file, a template ``configs/pipeline.yml``, and a ``.gitignore``.

    Parameters
    ----------
    project_dir
        Root directory for the new project.
    force
        If ``True``, re-initialise an existing project (refreshes the
        marker and creates missing directories, but never overwrites an
        existing ``configs/pipeline.yml``).
    registry_path
        Optional path to a dataset registry for category discovery.

    Raises
    ------
    SystemExit
        If the directory already contains a ``.hydro-param`` marker and
        *force* is ``False``.
    """
    project_dir = project_dir.resolve()
    marker_path = project_dir / MARKER_FILE
    config_path = project_dir / "configs" / "pipeline.yml"

    if marker_path.exists() and not force:
        print(
            f"Error: '{project_dir}' is already a hydro-param project.\n"
            "Use --force to re-initialize.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    project_name = project_dir.name

    # Build list of directories
    categories = get_data_categories(registry_path)
    dirs_to_create = [
        project_dir / "configs",
        project_dir / "data" / "fabrics",
        project_dir / "output",
        project_dir / "models",
    ]
    for cat in categories:
        dirs_to_create.append(project_dir / "data" / cat)

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # Marker file
    marker_content = {
        "name": project_name,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(yaml.dump(marker_content, default_flow_style=False))

    # Template pipeline config (never overwrite existing)
    if not config_path.exists():
        config_path.write_text(generate_pipeline_template(project_name))

    # .gitignore (overwrite is safe — declarative)
    (project_dir / ".gitignore").write_text(generate_gitignore())

    # Summary
    print(f"Initialized hydro-param project in {project_dir}/\n")
    print("Created:")
    print(f"  {MARKER_FILE:<28s} Project marker")
    print(f"  {'configs/pipeline.yml':<28s} Pipeline configuration template")
    print(f"  {'data/fabrics/':<28s} Target polygon files")
    for cat in categories:
        print(f"  data/{cat + '/':<22s} Dataset downloads")
    print(f"  {'output/':<28s} Pipeline results")
    print(f"  {'models/':<28s} Model exports")
    print(f"  {'.gitignore':<28s} Git ignore rules")
    print()
    print("Next steps:")
    print("  1. Copy your target fabric to data/fabrics/")
    print("  2. Edit configs/pipeline.yml")
    print("  3. Download datasets:  hydro-param datasets download <name>")
    print("  4. Run the pipeline:   hydro-param run configs/pipeline.yml")
