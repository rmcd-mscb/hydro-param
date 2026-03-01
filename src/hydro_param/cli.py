"""CLI entry point for hydro-param.

Provide the top-level ``hydro-param`` console script and all subcommands
using the cyclopts framework.  The CLI is the primary user interface for
configuration-driven hydrologic parameterization workflows.

Commands
--------
hydro-param init [project-dir]
    Scaffold a project directory with standard layout.
hydro-param datasets list
    List available datasets grouped by category.
hydro-param datasets info <name>
    Show full details for a single dataset.
hydro-param datasets download <name>
    Download dataset files via AWS CLI.
hydro-param run <config>
    Execute the generic parameterization pipeline (stages 1--5).
hydro-param pywatershed run <config>
    Generate a complete pywatershed model setup (two-phase workflow).
hydro-param pywatershed validate <param_file>
    Validate a pywatershed parameter NetCDF file.

Notes
-----
All model-specific logic (unit conversions, variable renaming, derivation)
lives in plugin modules (``derivations/pywatershed.py``,
``formatters/pywatershed.py``).  The generic ``run`` command produces a raw
Standardized Internal Representation (SIR); the ``pywatershed run`` command
adds a second phase of model-specific post-processing.

See Also
--------
docs/design.md : Full architecture and CLI specification (§11.9).
hydro_param.pipeline : Generic 5-stage pipeline orchestrator.
hydro_param.derivations.pywatershed : pywatershed derivation plugin.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from cyclopts import App

from hydro_param.config import load_config
from hydro_param.dataset_registry import DatasetEntry, DatasetRegistry, load_registry
from hydro_param.pipeline import DEFAULT_REGISTRY, run_pipeline_from_config
from hydro_param.project import find_project_root, init_project

logger = logging.getLogger(__name__)

app = App(name="hydro-param", help="Configuration-driven hydrologic parameterization.")
datasets_app = app.command(App(name="datasets", help="Discover and download datasets."))
pws_app = app.command(App(name="pywatershed", help="pywatershed model setup."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_registry(registry: Path | None) -> DatasetRegistry:
    """Load the dataset registry, falling back to the bundled default.

    Parameters
    ----------
    registry
        Path to a custom registry YAML file or directory.  When ``None``,
        uses the default registry shipped with hydro-param.

    Returns
    -------
    DatasetRegistry
        Loaded and validated dataset registry.
    """
    path = registry if registry is not None else DEFAULT_REGISTRY
    return load_registry(path)


def _access_status(entry: DatasetEntry) -> str:
    """Return a human-readable access status string for a dataset entry.

    Classify the dataset's availability as one of ``"local"``,
    ``"download required"``, ``"not configured"``, ``"remote"``, or
    ``"not yet available"`` based on its access strategy and source
    configuration.

    Parameters
    ----------
    entry
        A dataset entry from the registry.

    Returns
    -------
    str
        One of ``"local"``, ``"download required"``, ``"not configured"``,
        ``"remote"``, ``"not yet available"``, or the raw strategy name
        as a fallback.
    """
    if entry.strategy == "local_tiff":
        if entry.source is not None:
            return "local"
        if entry.download and (
            entry.download.url or entry.download.files or entry.download.url_template
        ):
            return "download required"
        return "not configured"
    if entry.strategy in ("stac_cog", "native_zarr", "climr_cat", "nhgf_stac"):
        return "remote (no download needed)"
    if entry.strategy == "converted_zarr":
        return "not yet available"
    return entry.strategy


# ---------------------------------------------------------------------------
# datasets list
# ---------------------------------------------------------------------------


@datasets_app.command(name="list")
def datasets_list(*, registry: Path | None = None) -> None:
    """Display all registered datasets grouped by category.

    Print a formatted table of datasets to stdout, showing name,
    description, access strategy, and availability status.  Datasets
    are grouped under their category heading (e.g., Topography,
    Land Cover, Soils).

    Parameters
    ----------
    registry
        Path to a custom registry YAML file or directory.  When
        omitted, the bundled default registry is used.
    """
    reg = _load_registry(registry)

    # Group by category
    by_category: dict[str, list[tuple[str, DatasetEntry]]] = {}
    for name, entry in reg.datasets.items():
        cat = entry.category or "uncategorized"
        by_category.setdefault(cat, []).append((name, entry))

    for category, entries in sorted(by_category.items()):
        print(f"\n{category.replace('_', ' ').title()}:")
        for name, entry in sorted(entries):
            desc = entry.description or ""
            status = _access_status(entry)
            temporal_tag = ""
            if entry.temporal and entry.time_step:
                yr = f", {entry.year_range[0]}-{entry.year_range[1]}" if entry.year_range else ""
                temporal_tag = f", {entry.time_step}{yr}"
            print(f"  {name:<20s} {desc:<50s} [{entry.strategy}{temporal_tag}, {status}]")


# ---------------------------------------------------------------------------
# datasets info
# ---------------------------------------------------------------------------


@datasets_app.command(name="info")
def datasets_info(name: str, *, registry: Path | None = None) -> None:
    """Show full details for a single dataset.

    Print comprehensive information including description, strategy,
    CRS, variables (continuous and derived), download instructions,
    and a pipeline config snippet for easy copy-paste.

    Parameters
    ----------
    name
        Dataset name as it appears in the registry (e.g.,
        ``"dem_3dep_10m"``, ``"nlcd_osn_lndcov"``).
    registry
        Path to a custom registry YAML file or directory.  When
        omitted, the bundled default registry is used.

    Raises
    ------
    SystemExit
        If the dataset name is not found in the registry (exit code 1).
    """
    reg = _load_registry(registry)
    try:
        entry = reg.get(name)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    print(f"Dataset: {name}")
    if entry.description:
        print(f"Description: {entry.description}")
    print(f"Strategy: {entry.strategy} ({_access_status(entry)})")
    print(f"CRS: {entry.crs}")
    if entry.category:
        print(f"Category: {entry.category}")
    if entry.temporal:
        print(f"Time step: {entry.time_step}")
        if entry.year_range:
            print(f"Available years: {entry.year_range[0]}-{entry.year_range[1]}")

    if entry.variables:
        print("\nVariables:")
        for v in entry.variables:
            cat_tag = " (categorical)" if v.categorical else ""
            print(f"  {v.name}{cat_tag}")
            if v.long_name:
                print(f"    {v.long_name}")

    if entry.derived_variables:
        print("\nDerived Variables:")
        for dv in entry.derived_variables:
            print(f"  {dv.name} (from {dv.source}, method={dv.method})")
            if dv.long_name:
                print(f"    {dv.long_name}")

    if entry.download:
        dl = entry.download
        if dl.files:
            # Multi-file dataset
            years = sorted({f.year for f in dl.files})
            variables = sorted({f.variable for f in dl.files})
            print(f"\nDownload: {len(dl.files)} files available")
            print(f"  Years: {', '.join(str(y) for y in years)}")
            print(f"  Products: {', '.join(variables)}")
            print("\n  Files:")
            for f in sorted(dl.files, key=lambda x: (x.year, x.variable)):
                size = f"  ~{f.size_gb} GB" if f.size_gb else ""
                print(f"    {f.year} {f.variable:<20s}{size}")
                print(f"      {f.url}")
            print("\n  Download with:")
            print(f"    hydro-param datasets download {name} --years {years[-1]}")
        elif dl.url_template:
            # Template-based dataset
            start, end = dl.year_range
            total = (end - start + 1) * len(dl.variables_available)
            example_url = dl.url_template.format(variable=dl.variables_available[0], year=end)
            print(f"\nDownload: {total} files via URL template")
            print(f"  Years: {start}-{end}")
            print(f"  Products: {', '.join(dl.variables_available)}")
            print(f"  Requester-pays: {'yes' if dl.requester_pays else 'no'}")
            print(f"\n  Example URL:\n    {example_url}")
            if dl.notes:
                print(f"  {dl.notes.strip()}")
            first_var = dl.variables_available[0]
            print("\n  Download with:")
            print(f"    hydro-param datasets download {name} --years {end} --variables {first_var}")
        elif dl.url:
            # Single-file dataset
            print("\nDownload:")
            print(f"  URL: {dl.url}")
            if dl.size_gb:
                print(f"  Size: ~{dl.size_gb} GB")
            if dl.format:
                print(f"  Format: {dl.format}")
            if dl.notes:
                print(f"  {dl.notes.strip()}")
            print("\n  Download with:")
            print(f"    hydro-param datasets download {name}")

    print("\nTo use in your pipeline config:")
    print("  datasets:")
    print(f"    - name: {name}")
    if entry.strategy == "local_tiff":
        print("      source: /path/to/your/downloaded/file.tif")
    if entry.variables:
        var_names = [v.name for v in entry.variables]
        print(f"      variables: [{', '.join(var_names)}]")


# ---------------------------------------------------------------------------
# datasets download
# ---------------------------------------------------------------------------


@datasets_app.command(name="download")
def datasets_download(
    name: str,
    *,
    dest: Path | None = None,
    years: str | None = None,
    variables: str | None = None,
    registry: Path | None = None,
) -> None:
    """Download dataset files via the AWS CLI.

    Fetch remote dataset files (typically from S3) using ``aws s3 cp``.
    Supports single-file, multi-file (explicit file list), and
    template-based (URL pattern with year/variable placeholders)
    download configurations.

    When run inside an initialised hydro-param project (detected via
    ``.hydro-param`` marker), files are automatically routed to the
    ``data/<category>/`` subdirectory.

    Parameters
    ----------
    name
        Dataset name as it appears in the registry (e.g.,
        ``"polaris_30m"``, ``"nlcd_legacy"``).
    dest
        Destination directory for downloaded files.  When omitted inside
        an initialised project, files are routed to ``data/<category>/``
        automatically.  Otherwise defaults to the current directory.
    years
        Comma-separated list of years to download (multi-file datasets).
        Example: ``"2019,2020,2021"``.
    variables
        Comma-separated list of variables/products to download
        (multi-file datasets).  Example: ``"silt,sand,clay"``.
    registry
        Path to a custom registry YAML file or directory.  When
        omitted, the bundled default registry is used.

    Raises
    ------
    SystemExit
        If the dataset is not found, has no download info, AWS CLI is
        not installed, or a download fails (exit code 1).

    Notes
    -----
    Requires the AWS CLI (``aws``) to be installed and available on
    ``PATH``.  For requester-pays buckets (e.g., ``s3://usgs-landcover``),
    valid AWS credentials are needed.  Anonymous access is used otherwise.
    """
    reg = _load_registry(registry)
    try:
        entry = reg.get(name)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None

    if entry.download is None:
        print(f"Error: Dataset '{name}' has no download information.", file=sys.stderr)
        print("It may be a remote dataset that does not require downloading.", file=sys.stderr)
        raise SystemExit(1)

    # Resolve destination: auto-route when inside an initialised project
    if dest is None:
        project_root = find_project_root()
        if project_root is not None and entry.category:
            dest = project_root / "data" / entry.category
            print(f"Project detected: downloading to {dest}")
        else:
            dest = Path(".")

    # Check for AWS CLI
    if shutil.which("aws") is None:
        print(
            "Error: AWS CLI not found. Install it to download datasets.\n"
            "  https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html",
            file=sys.stderr,
        )
        raise SystemExit(1)

    dest.mkdir(parents=True, exist_ok=True)

    if entry.download.files or entry.download.url_template:
        _download_multi_file(name, entry, dest, years, variables)
    elif entry.download.url:
        _download_single_file(
            entry.download.url, dest, requester_pays=entry.download.requester_pays
        )
    else:
        print(f"Error: Dataset '{name}' has no download URLs.", file=sys.stderr)
        raise SystemExit(1)


def _download_single_file(url: str, dest: Path, *, requester_pays: bool = False) -> None:
    """Download a single file from S3 via ``aws s3 cp``.

    Parameters
    ----------
    url
        Full S3 URL (e.g., ``s3://bucket/path/to/file.tif``).
    dest
        Local destination directory.  The filename is extracted from the
        URL's last path component.
    requester_pays
        If ``True``, pass ``--request-payer=requester`` to ``aws s3 cp``.
        Otherwise, use ``--no-sign-request`` for anonymous access.

    Raises
    ------
    SystemExit
        If the ``aws s3 cp`` command exits with a non-zero return code.
    """
    filename = url.rsplit("/", 1)[-1]
    dest_path = dest / filename
    print(f"Downloading: {url}")
    print(f"        To: {dest_path}")
    cmd = ["aws", "s3", "cp"]
    if requester_pays:
        cmd.append("--request-payer=requester")
    else:
        cmd.append("--no-sign-request")
    cmd.extend([url, str(dest_path)])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Error: Download failed (exit code {result.returncode})", file=sys.stderr)
        raise SystemExit(1)
    print(f"Done: {dest_path}")


def _download_multi_file(
    name: str,
    entry: DatasetEntry,
    dest: Path,
    years_str: str | None,
    variables_str: str | None,
) -> None:
    """Download selected files from a multi-file or template dataset.

    Expand the download specification (explicit file list or URL
    template) with optional year and variable filters, then download
    each matching file via ``_download_single_file``.

    Parameters
    ----------
    name
        Dataset name (used in error messages).
    entry
        Registry entry with download configuration.
    dest
        Local destination directory for downloaded files.
    years_str
        Comma-separated year filter (e.g., ``"2019,2020"``), or
        ``None`` to download all available years.
    variables_str
        Comma-separated variable filter (e.g., ``"silt,clay"``), or
        ``None`` to download all available variables.

    Raises
    ------
    SystemExit
        If no files match the given filters or a download fails.
    """
    assert entry.download is not None  # guaranteed by caller
    dl = entry.download

    # Parse year filter
    year_set: set[int] | None = None
    if years_str is not None:
        year_set = set()
        for raw_year in years_str.split(","):
            year_token = raw_year.strip()
            if not year_token:
                continue
            try:
                year_set.add(int(year_token))
            except ValueError:
                print(f"Error: Invalid year value '{year_token}' in --years.", file=sys.stderr)
                raise SystemExit(1) from None

    # Parse variable filter
    var_set: set[str] | None = None
    if variables_str is not None:
        var_set = {v.strip() for v in variables_str.split(",") if v.strip()}

    files = dl.expand_files(years=year_set, variables=var_set)

    if not files:
        print(f"Error: No matching files for dataset '{name}'.", file=sys.stderr)
        print(f"Run 'hydro-param datasets info {name}' to see available files.", file=sys.stderr)
        raise SystemExit(1)

    print(f"Downloading {len(files)} file(s) for '{name}':")
    for f in sorted(files, key=lambda x: (x.year, x.variable)):
        _download_single_file(f.url, dest, requester_pays=dl.requester_pays)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command(name="run")
def run_cmd(config: Path, *, registry: Path | None = None, resume: bool = False) -> None:
    """Execute the generic parameterization pipeline.

    Run stages 1--5 (resolve fabric, resolve datasets, compute weights,
    process datasets, normalize output) to produce a normalized
    Standardized Internal Representation (SIR) with canonical variable
    names and converted units.  This command is model-agnostic; use ``pywatershed run`` for
    model-specific post-processing.

    Parameters
    ----------
    config
        Path to the pipeline YAML config file (e.g.,
        ``configs/examples/delaware_2yr.yml``).
    registry
        Path to a custom dataset registry YAML file or directory.
        When omitted, the bundled default registry is used.
    resume
        Enable manifest-based resume: skip datasets whose outputs are
        already complete and whose config fingerprint has not changed.
        Compares SHA-256 fingerprints of dataset request + registry
        entry + processing options.

    Raises
    ------
    SystemExit
        If the pipeline raises any exception (exit code 1).

    See Also
    --------
    hydro_param.pipeline.run_pipeline_from_config : Pipeline entry point.
    hydro_param.manifest : Manifest-based resume logic.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = load_config(config)
        if resume:
            cfg = cfg.model_copy(
                update={"processing": cfg.processing.model_copy(update={"resume": True})}
            )
        reg = _load_registry(registry)
        run_pipeline_from_config(cfg, reg)
    except Exception as exc:
        logger.exception("Pipeline failed.")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command(name="init")
def init_cmd(
    project_dir: Path = Path("."),
    *,
    force: bool = False,
    registry: Path | None = None,
) -> None:
    """Scaffold a new hydro-param project directory.

    Create a standard directory structure with a template pipeline
    configuration, data directories organised by dataset category
    (e.g., ``data/topography/``, ``data/soils/``), and a ``.gitignore``.
    A ``.hydro-param`` marker file is written to identify the project
    root for automatic path resolution in other commands (e.g.,
    ``datasets download`` auto-routes files to ``data/<category>/``).

    Parameters
    ----------
    project_dir
        Directory to initialise.  Defaults to the current directory.
    force
        Re-initialise an existing project: create missing directories
        and refresh the marker file, but preserve any existing
        ``pipeline.yml``.
    registry
        Path to a custom registry YAML for category discovery.  When
        omitted, the bundled default registry is used to determine
        which ``data/<category>/`` subdirectories to create.

    See Also
    --------
    hydro_param.project.init_project : Implementation of project scaffolding.
    """
    init_project(project_dir, force=force, registry_path=registry)


# ---------------------------------------------------------------------------
# pywatershed run
# ---------------------------------------------------------------------------


@pws_app.command(name="run")
def pws_run_cmd(config: Path) -> None:
    """Generate a complete pywatershed model setup from existing SIR output.

    Consume pre-built SIR (Standardized Internal Representation) output
    produced by ``hydro-param run`` and derive all PRMS parameters needed
    for a pywatershed (NHM-PRMS) simulation.

    This command executes Phase 2 only -- it does **not** re-run the
    generic pipeline.  Run ``hydro-param run pipeline.yml`` first to
    produce SIR output, then run this command to derive model-specific
    parameters.

    Output files produced:

    - ``parameters.nc`` -- static PRMS parameters loadable by
      ``pws.Parameters.from_netcdf()``
    - ``forcing/<var>.nc`` -- one file per climate variable (prcp,
      tmax, tmin) in PRMS units (inches, degrees F)
    - ``soltab.nc`` -- potential solar radiation tables (nhru x 366)
    - ``control.yml`` -- simulation time period configuration

    Parameters
    ----------
    config
        Path to a pywatershed run config YAML file (v3.0).  See
        ``PywatershedRunConfig`` for the expected schema.

    Raises
    ------
    SystemExit
        If config loading, SIR loading, or derivation fails (exit code 1).

    See Also
    --------
    hydro_param.pywatershed_config.PywatershedRunConfig : Config schema.
    hydro_param.derivations.pywatershed.PywatershedDerivation : Derivation plugin.
    hydro_param.formatters.pywatershed.PywatershedFormatter : Output formatter.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import geopandas as gpd
    import xarray as xr

    from hydro_param.derivations.pywatershed import PywatershedDerivation
    from hydro_param.plugins import DerivationContext, get_formatter
    from hydro_param.pywatershed_config import load_pywatershed_config
    from hydro_param.sir_accessor import SIRAccessor

    try:
        pws_config = load_pywatershed_config(config)
    except Exception as exc:
        logger.exception("Failed to load pywatershed config '%s'.", config)
        raise SystemExit(1) from exc

    logger.info("pywatershed config validated: %s", config)
    logger.info("  Time: %s to %s", pws_config.time.start, pws_config.time.end)
    logger.info("  SIR path: %s", pws_config.sir_path)
    logger.info("  Output: %s", pws_config.output.path)

    # ── Resolve SIR path ──
    sir_path = pws_config.sir_path
    if not sir_path.is_absolute():
        sir_path = config.parent / sir_path
    sir_path = sir_path.resolve()

    try:
        sir = SIRAccessor(sir_path)
    except FileNotFoundError as exc:
        logger.error("SIR output not found: %s", exc)
        logger.error("Run 'hydro-param run pipeline.yml' first to produce SIR output.")
        raise SystemExit(1) from exc

    # ── Validate config contract against SIR ──
    declared = pws_config.declared_entries()
    if declared:
        logger.info("Config declares %d parameter entries:", len(declared))
        for name, entry in declared.items():
            var = entry.variable or entry.variables
            logger.info("  %s <- %s.%s", name, entry.source, var)
    else:
        logger.warning("No parameter entries declared in config — derivation will use SIR as-is.")

    # ── Load fabric ──
    fabric_path = pws_config.domain.fabric_path
    if not fabric_path.exists():
        logger.error(
            "Fabric file not found: '%s'. Check domain.fabric_path in '%s'.",
            fabric_path,
            config,
        )
        raise SystemExit(1)
    try:
        fabric = gpd.read_file(fabric_path)
    except Exception as exc:
        logger.exception("Failed to read fabric file '%s'.", fabric_path)
        raise SystemExit(1) from exc

    # ── Load optional segments / waterbodies ──
    segments = None
    if pws_config.domain.segment_path is not None:
        seg_path = pws_config.domain.segment_path
        if not seg_path.exists():
            logger.error(
                "Segment file not found: '%s'. Check domain.segment_path in '%s'.",
                seg_path,
                config,
            )
            raise SystemExit(1)
        try:
            segments = gpd.read_file(seg_path)
        except Exception as exc:
            logger.exception(
                "Failed to read segment file '%s'. Ensure it is a valid GeoPackage or GeoParquet.",
                seg_path,
            )
            raise SystemExit(1) from exc

    waterbodies = None
    if pws_config.domain.waterbody_path is not None:
        wb_path = pws_config.domain.waterbody_path
        if not wb_path.exists():
            logger.error(
                "Waterbody file not found: '%s'. Check domain.waterbody_path in '%s'.",
                wb_path,
                config,
            )
            raise SystemExit(1)
        try:
            waterbodies = gpd.read_file(wb_path)
        except Exception as exc:
            logger.exception(
                "Failed to read waterbody file '%s'. "
                "Ensure it is a valid GeoPackage or GeoParquet.",
                wb_path,
            )
            raise SystemExit(1) from exc
        if "ftype" not in waterbodies.columns:
            logger.error(
                "Waterbody file '%s' is missing required 'ftype' column. "
                "Expected NHDPlus waterbody polygons with 'ftype' values "
                "like 'LakePond' and 'Reservoir'. Found columns: %s",
                wb_path,
                sorted(waterbodies.columns.tolist()),
            )
            raise SystemExit(1)

    # ── Load temporal data from SIR ──
    temporal: dict[str, xr.Dataset] = {}
    try:
        for name in sir.available_temporal():
            try:
                temporal[name] = sir.load_temporal(name)
            except (OSError, KeyError) as exc:
                logger.error("Failed to load temporal SIR data '%s': %s", name, exc)
                logger.error("Re-run 'hydro-param run pipeline.yml' to regenerate SIR output.")
                raise SystemExit(1) from exc

        if temporal:
            logger.info(
                "Loaded %d temporal datasets: %s",
                len(temporal),
                list(temporal.keys()),
            )
        else:
            logger.info(
                "No temporal data in SIR; forcing generation will be skipped "
                "and PET/transpiration will use scalar defaults."
            )

        # ── Derive parameters ──
        logger.info("Deriving pywatershed parameters from SIR")

        derivation_config: dict = {}
        if pws_config.parameter_overrides.values:
            derivation_config["parameter_overrides"] = {
                "values": pws_config.parameter_overrides.values,
            }

        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            waterbodies=waterbodies,
            fabric_id_field=pws_config.domain.id_field,
            segment_id_field=pws_config.domain.segment_id_field,
            config=derivation_config,
            temporal=temporal,
        )

        plugin = PywatershedDerivation()
        derived = plugin.derive(ctx)
    except SystemExit:
        raise
    except Exception as exc:
        logger.exception("Parameter derivation failed.")
        raise SystemExit(1) from exc
    finally:
        for ds in temporal.values():
            try:
                ds.close()
            except Exception:
                logger.debug("Failed to close temporal dataset", exc_info=True)

    # ── Format and write ──
    formatter = get_formatter("pywatershed")
    formatter_config = {
        "parameter_file": pws_config.output.parameter_file,
        "forcing_dir": pws_config.output.forcing_dir,
        "soltab_file": pws_config.output.soltab_file,
        "control_file": pws_config.output.control_file,
        "start": pws_config.time.start,
        "end": pws_config.time.end,
    }
    try:
        formatter.write(derived, pws_config.output.path, formatter_config)
    except Exception as exc:
        logger.exception("Failed to write pywatershed output to '%s'.", pws_config.output.path)
        raise SystemExit(1) from exc

    soltab_path = Path(pws_config.output.path) / pws_config.output.soltab_file
    if not soltab_path.exists():
        logger.info(
            "soltab.nc was not produced. Ensure the topography dataset includes "
            "elevation, slope, and aspect variables. Solar radiation tables will "
            "not be available for this run."
        )
    elif soltab_path.stat().st_size == 0:
        logger.warning(
            "soltab.nc exists but is empty (0 bytes). This may indicate a write "
            "failure. Solar radiation tables may not be usable."
        )

    logger.info("pywatershed model setup complete: %s", pws_config.output.path)


@pws_app.command(name="validate")
def pws_validate_cmd(
    param_file: Path,
) -> None:
    """Validate a pywatershed parameter file against metadata constraints.

    Check that required PRMS parameters are present and that values
    fall within the valid ranges defined in the bundled
    ``parameter_metadata.yml``.  Print a summary of issues found or
    a success message.

    Parameters
    ----------
    param_file
        Path to a pywatershed parameter NetCDF file (e.g.,
        ``output/parameters.nc``).

    Raises
    ------
    SystemExit
        If the file cannot be opened (exit code 1) or validation
        finds any issues (exit code 1).

    See Also
    --------
    PywatershedFormatter.validate : Underlying validation logic.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    import xarray as xr

    from hydro_param.formatters.pywatershed import PywatershedFormatter

    try:
        ds = xr.open_dataset(param_file)
    except Exception as exc:
        print(f"Error: Could not open '{param_file}': {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    formatter = PywatershedFormatter()
    warnings = formatter.validate(ds)
    ds.close()

    if warnings:
        print(f"Validation found {len(warnings)} issue(s):")
        for w in warnings:
            print(f"  - {w}")
        raise SystemExit(1)
    else:
        print("Validation passed: all checks OK.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Invoke the cyclopts CLI application.

    This is the entry point registered as the ``hydro-param`` console
    script in ``pyproject.toml``.  Delegates to the cyclopts ``App``
    for argument parsing and command dispatch.
    """
    app()
