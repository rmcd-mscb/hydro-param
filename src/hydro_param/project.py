"""Project scaffolding: init command, project root detection, and template generation.

Provide the ``hydro-param init`` functionality that creates a standard project
directory structure with categorical data subfolders, template pipeline and
pywatershed configurations, and a ``.gitignore``.  A hidden ``.hydro-param``
marker file at the project root enables upward directory discovery so that
commands run from subdirectories can locate the project root automatically.

The scaffolding is intentionally lightweight -- hydro-param uses
library-managed transparent caching (pooch-style), not a heavyweight project
directory contract.  The templates exist to give users a working starting
point, not to enforce a rigid layout.

See Also
--------
hydro_param.cli : CLI commands that invoke :func:`init_project`.
hydro_param.config : Pipeline config schema that the generated template follows.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

MARKER_FILE = ".hydro-param"
"""str : Hidden file name placed at the project root by ``hydro-param init``."""

# Fallback categories when registry is unavailable.
# These match the per-category YAML files in hydro_param.data.datasets.
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
"""list[str] : Built-in dataset categories used when the registry is unavailable.

These names correspond to the per-category YAML files in ``hydro_param.data.datasets``
and become subdirectory names under ``data/`` in an initialized project.
"""


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* looking for a ``.hydro-param`` marker file.

    Traverse parent directories from *start* toward the filesystem root,
    returning the first directory that contains the marker file.  This
    allows CLI commands invoked from any subdirectory to locate the
    project root without requiring an explicit ``--project-dir`` flag.

    Parameters
    ----------
    start : Path or None
        Directory to begin searching from.  Defaults to the current
        working directory (``Path.cwd()``).

    Returns
    -------
    Path or None
        The resolved project root directory, or ``None`` if no marker
        is found before reaching the filesystem root.

    Notes
    -----
    The search terminates when ``parent == current``, which is the
    filesystem root on both POSIX and Windows systems.
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
    """Discover dataset categories from the registry, with built-in fallback.

    Attempt to load the dataset registry and extract the union of all
    ``category`` values.  If the registry cannot be loaded (missing path,
    parse error, etc.), fall back to :data:`DEFAULT_CATEGORIES` so that
    project scaffolding always succeeds.

    Parameters
    ----------
    registry_path : Path or None
        Path to a registry YAML file or directory of YAML files.  When
        ``None`` or unloadable, :data:`DEFAULT_CATEGORIES` is returned.

    Returns
    -------
    list[str]
        Sorted, deduplicated list of category names.  Always includes at
        least the :data:`DEFAULT_CATEGORIES`.
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
    """Generate a well-commented pipeline YAML config template.

    Produce a starter ``pipeline.yml`` with inline comments explaining
    each section (target fabric, domain, datasets, output, processing).
    The template includes all 7 tested dataset configurations covering
    all 5 data access strategies (stac_cog, local_tiff, nhgf_stac
    static, nhgf_stac temporal, climr_cat).

    Parameters
    ----------
    project_name : str
        Project name inserted into the ``output.sir_name`` field and the
        header comment.

    Returns
    -------
    str
        YAML content suitable for writing to ``configs/pipeline.yml``.

    See Also
    --------
    hydro_param.config.PipelineConfig : Schema the generated YAML must conform to.
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

# --- Domain (optional) ---
# By default, the pipeline uses the full extent of the target fabric.
# Uncomment to restrict processing to a spatial subset:
# domain:
#   type: bbox
#   bbox: [-76.5, 38.5, -74.0, 42.6]    # [west, south, east, north] in EPSG:4326

# --- Datasets ---
# Each entry references a dataset from the registry by name.
# Use 'hydro-param datasets list' to see available datasets.
# Remove or comment out datasets you don't need.
datasets:
  # --- Topography ---
  # 3DEP 10m DEM — elevation, slope, aspect (stac_cog via Planetary Computer)
  - name: dem_3dep_10m
    variables: [elevation, slope, aspect]
    statistics: [mean]

  # --- Soils ---
  # gNATSGO pre-summarized soil properties (stac_cog via Planetary Computer)
  - name: gnatsgo_rasters
    variables: [aws0_100, rootznemc, rootznaws]
    statistics: [mean]

  # POLARIS soil texture properties, 30m (local_tiff — these variables have
  # remote VRT source_overrides in the registry, so no local download needed.
  # Adding other POLARIS variables may require a 'source:' path or download.)
  - name: polaris_30m
    variables: [sand, silt, clay, theta_s, ksat]
    statistics: [mean]

  # --- Land Cover ---
  # NLCD Land Cover via NHGF STAC OSN (nhgf_stac)
  # LndCov is categorical — class fractions are computed automatically
  # based on the registry's categorical flag; statistics is not used.
  - name: nlcd_osn_lndcov
    variables: [LndCov]
    statistics: [categorical]
    year: [2021]

  # NLCD Fractional Impervious via NHGF STAC OSN (nhgf_stac)
  - name: nlcd_osn_fctimp
    variables: [FctImp]
    statistics: [mean]
    year: [2021]

  # --- Snow ---
  # SNODAS daily snow — historical SWE (nhgf_stac temporal)
  - name: snodas
    variables: [SWE]
    statistics: [mean]
    time_period: ["2020-01-01", "2021-12-31"]

  # --- Climate ---
  # gridMET daily climate via OPeNDAP (climr_cat)
  # pr/tmmx/tmmn for forcing; srad/pet/vs for radiation and PET derivation
  - name: gridmet
    variables: [pr, tmmx, tmmn, srad, pet, vs]
    statistics: [mean]
    time_period: ["2020-01-01", "2021-12-31"]

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
  batch_size: 500                         # Number of features per spatial batch
  resume: true                            # Skip already-completed datasets on re-run
"""


def generate_pywatershed_template(project_name: str) -> str:
    """Generate a well-commented pywatershed run config template.

    Produce a starter ``pywatershed_run.yml`` covering domain extraction,
    simulation period, climate forcing, dataset sources, processing,
    parameter overrides, calibration, and output format sections.  The
    template targets the Delaware River Basin as a default example.

    Parameters
    ----------
    project_name : str
        Project name inserted into the output path and header comment.

    Returns
    -------
    str
        YAML content suitable for writing to ``configs/pywatershed_run.yml``.

    References
    ----------
    - ``docs/reference/pywatershed_parameterization_guide.md``
    - ``docs/reference/pywatershed_dataset_param_map.yml``

    See Also
    --------
    hydro_param.pywatershed_config : Schema for pywatershed run configs.
    """
    return f"""\
# pywatershed run configuration for {project_name}
#
# This file specifies everything needed for hydro-param to generate
# a complete pywatershed model setup.  Edit the sections below, then run:
#   hydro-param pywatershed run configs/pywatershed_run.yml
#
# Reference:
#   docs/reference/pywatershed_parameterization_guide.md
#   docs/reference/pywatershed_dataset_param_map.yml

target_model: pywatershed
version: "2.0"

# --- Domain ---
# Spatial extent for the model.
# Provide pre-existing HRU and segment fabric files (GeoPackage or GeoParquet).
# Obtain fabrics with pynhd, pygeohydro, or similar upstream tools —
# hydro-param does NOT fetch or subset fabrics.
domain:
  source: custom
  extraction_method: bbox
  bbox: [-76.5, 38.5, -74.0, 42.6]    # [west, south, east, north] EPSG:4326
  fabric_path: "data/fabrics/nhru.gpkg"        # REQUIRED: path to HRU fabric
  segment_path: "data/fabrics/nsegment.gpkg"   # path to segment/flowline fabric
  # waterbody_path: "data/fabrics/waterbodies.gpkg"  # NHDPlus waterbody polygons (optional)
  id_field: "nhm_id"
  segment_id_field: "nhm_seg"

# --- Simulation Period ---
time:
  start: "1980-10-01"
  end: "2020-09-30"
  timestep: daily

# --- Climate Forcing ---
# Source for area-weighted forcing time series.
# Output is one-variable-per-NetCDF-file (prcp.nc, tmax.nc, tmin.nc).
climate:
  source: gridmet                       # only gridmet is currently supported
  method: area_weighted_mean
  variables: [prcp, tmax, tmin]

# --- Dataset Sources ---
# Which datasets to use for each category.
# Names reference the dataset registry (hydro-param datasets list).
datasets:
  topography: dem_3dep_10m
  landcover: nlcd_legacy
  soils: polaris_30m
  # hydrography: null                  # NHDPlus (optional, for routing)

# --- Processing ---
processing:
  zonal_method: exactextract            # exactextract or serial
  batch_size: 500
  n_workers: 1

# --- Parameter Overrides ---
# Manually override any derived parameter value.
parameter_overrides:
  values: {{}}
  # values:
  #   tmax_allsnow: 32.0
  #   den_max: 0.55
  # from_file: null                     # path to a NetCDF with override values

# --- Calibration ---
calibration:
  generate_seeds: true
  seed_method: physically_based         # physically_based or all_defaults
  preserve_from_existing: []

# --- Output ---
output:
  path: "models/pywatershed"
  format: netcdf                        # netcdf or prms_text
  parameter_file: "parameters.nc"
  forcing_dir: "forcing"                  # one-variable-per-file NetCDF (prcp.nc, tmax.nc, tmin.nc)
  control_file: "control.yml"
  soltab_file: "soltab.nc"
"""


def generate_gitignore() -> str:
    """Generate ``.gitignore`` content for a hydro-param project.

    The ignore rules track lightweight config files and the marker, while
    ignoring downloaded raster/vector data, pipeline output, and model
    exports.  Large geospatial formats (``*.nc``, ``*.tif``, ``*.zarr/``)
    are caught by a safety-net section.

    Returns
    -------
    str
        Content suitable for writing to ``.gitignore`` in the project root.
    """
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

    Create the standard directory structure expected by the pipeline:

    - ``configs/`` -- pipeline and pywatershed run config templates
    - ``data/fabrics/`` -- target polygon files
    - ``data/<category>/`` -- one subdirectory per dataset category
    - ``output/`` -- pipeline results
    - ``models/pywatershed/`` -- pywatershed model exports
    - ``.hydro-param`` -- marker file for project root discovery
    - ``.gitignore`` -- rules to keep large data out of version control

    Existing config templates (``pipeline.yml``, ``pywatershed_run.yml``)
    are never overwritten, even with ``force=True``, so user edits are
    preserved.  The ``.gitignore`` is always regenerated because it is
    declarative and safe to replace.

    Parameters
    ----------
    project_dir : Path
        Root directory for the new project.  Created if it does not exist.
    force : bool
        If ``True``, re-initialise an existing project (refreshes the
        marker and creates missing directories, but never overwrites
        existing config templates).
    registry_path : Path or None
        Optional path to a dataset registry YAML file or directory for
        category discovery.  When ``None``, :data:`DEFAULT_CATEGORIES`
        is used.

    Raises
    ------
    SystemExit
        If the directory already contains a ``.hydro-param`` marker and
        *force* is ``False``.

    Notes
    -----
    The marker file (``.hydro-param``) is a YAML file containing the
    project name and UTC creation timestamp.  It is used by
    :func:`find_project_root` for upward directory discovery.
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
        project_dir / "models" / "pywatershed",
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

    # Template pywatershed config (never overwrite existing)
    pws_config_path = project_dir / "configs" / "pywatershed_run.yml"
    if not pws_config_path.exists():
        pws_config_path.write_text(generate_pywatershed_template(project_name))

    # .gitignore (overwrite is safe — declarative)
    (project_dir / ".gitignore").write_text(generate_gitignore())

    # Summary
    print(f"Initialized hydro-param project in {project_dir}/\n")
    print("Created:")
    print(f"  {MARKER_FILE:<28s} Project marker")
    print(f"  {'configs/pipeline.yml':<28s} Pipeline configuration template")
    print(f"  {'configs/pywatershed_run.yml':<28s} pywatershed run config template")
    print(f"  {'data/fabrics/':<28s} Target polygon files")
    for cat in categories:
        print(f"  data/{cat + '/':<22s} Dataset downloads")
    print(f"  {'output/':<28s} Pipeline results")
    print(f"  {'models/':<28s} Model exports")
    print(f"  {'models/pywatershed/':<28s} pywatershed model files")
    print(f"  {'.gitignore':<28s} Git ignore rules")
    print()
    print("Next steps:")
    print("  1. Obtain HRU/segment fabrics (pynhd, pygeohydro, or USGS GF)")
    print("     and place them in data/fabrics/")
    print("  2. Edit configs/pipeline.yml or configs/pywatershed_run.yml")
    print("  3. Download datasets:  hydro-param datasets download <name>")
    print("  4. Run the pipeline:")
    print("       hydro-param run configs/pipeline.yml")
    print("     Or the pywatershed workflow:")
    print("       hydro-param pywatershed run configs/pywatershed_run.yml")
    print("       hydro-param pywatershed validate models/pywatershed/")
