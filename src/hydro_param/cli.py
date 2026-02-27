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

from hydro_param.config import PipelineConfig, load_config
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
        return "remote"
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
            print(f"  {name:<20s} {desc:<50s} [{entry.strategy}, {status}]")


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
# pywatershed helpers
# ---------------------------------------------------------------------------


def _translate_pws_to_pipeline(
    pws_config: object,
) -> PipelineConfig:
    """Translate a PywatershedRunConfig into a generic PipelineConfig.

    Map the pywatershed-specific configuration schema onto the generic
    pipeline config so stages 1--5 can produce a raw SIR and temporal
    data.  This is the bridge between the model-specific user interface
    and the model-agnostic pipeline engine.

    No model-specific transforms (derivation, variable renaming, unit
    conversion) are included in the translated config -- those happen
    in ``pws_run_cmd()`` after the pipeline completes (two-phase
    separation).

    Parameters
    ----------
    pws_config
        A validated ``PywatershedRunConfig`` instance (typed as
        ``object`` to avoid circular imports at module level).

    Returns
    -------
    PipelineConfig
        Generic pipeline configuration ready for
        ``run_pipeline_from_config()``.

    Raises
    ------
    ValueError
        If required fields (e.g., ``fabric_path``) are missing.
    NotImplementedError
        If an unsupported extraction method is requested.

    Notes
    -----
    Climate variable names are mapped from PRMS user-facing names
    (``prcp``, ``tmax``, ``tmin``) to registry/gdptools source names
    (``pr``, ``tmmx``, ``tmmn``) for the pipeline request.  The reverse
    mapping happens during pywatershed post-processing.

    See Also
    --------
    pws_run_cmd : Two-phase pywatershed workflow that calls this function.
    """
    from hydro_param.config import (
        DatasetRequest,
        DomainConfig,
        OutputConfig,
        ProcessingConfig,
        TargetFabricConfig,
    )  # noqa: F811 — local import for clarity
    from hydro_param.pywatershed_config import PywatershedRunConfig

    cfg: PywatershedRunConfig = pws_config  # type: ignore[assignment]

    # Target fabric
    if cfg.domain.fabric_path is None:
        raise ValueError("pywatershed config requires 'fabric_path' in domain")
    target_fabric = TargetFabricConfig(
        path=cfg.domain.fabric_path,
        id_field=cfg.domain.id_field,
    )

    # Domain
    if cfg.domain.extraction_method == "bbox":
        domain = DomainConfig(type="bbox", bbox=cfg.domain.bbox)
    else:
        raise NotImplementedError(
            f"Extraction method '{cfg.domain.extraction_method}' not yet supported"
        )

    # Datasets
    datasets: list[DatasetRequest] = []

    # Topography
    datasets.append(
        DatasetRequest(
            name=cfg.datasets.topography,
            variables=["elevation", "slope", "aspect"],
            statistics=["mean"],
        )
    )

    # Land cover (categorical fractions)
    lc_name = cfg.datasets.landcover
    if lc_name.startswith("nlcd_osn"):
        # Use end year of simulation period, clamped to NLCD availability (1985-2024)
        from datetime import date as _date

        _end_year = _date.fromisoformat(cfg.time.end).year
        _nlcd_year = min(_end_year, 2024)
        datasets.append(
            DatasetRequest(
                name=lc_name,
                variables=["LndCov"],
                statistics=["categorical"],
                year=_nlcd_year,
            )
        )
    else:
        datasets.append(
            DatasetRequest(
                name=lc_name,
                variables=["land_cover"],
                statistics=["majority"],
            )
        )

    # Soils
    datasets.append(
        DatasetRequest(
            name=cfg.datasets.soils,
            variables=["sand", "silt", "clay", "ksat", "theta_s", "bd"],
            statistics=["mean"],
        )
    )

    # Climate (temporal) — validate source and map variable names
    _SUPPORTED_CLIMATE_SOURCES = {"gridmet"}
    if cfg.climate.source not in _SUPPORTED_CLIMATE_SOURCES:
        raise ValueError(
            f"Climate source '{cfg.climate.source}' is not yet supported. "
            f"Available sources: {', '.join(sorted(_SUPPORTED_CLIMATE_SOURCES))}"
        )
    climate_ds_name = cfg.climate.source

    _CLIMATE_VAR_MAP = {
        "prcp": "pr",
        "tmax": "tmmx",
        "tmin": "tmmn",
    }
    climate_vars = [_CLIMATE_VAR_MAP.get(v, v) for v in cfg.climate.variables]
    datasets.append(
        DatasetRequest(
            name=climate_ds_name,
            variables=climate_vars,
            statistics=["mean"],
            time_period=[cfg.time.start, cfg.time.end],
        )
    )

    output = OutputConfig(
        path=cfg.output.path,
        format="netcdf",
        sir_name="pywatershed_sir",
    )

    processing = ProcessingConfig(
        engine=cfg.processing.zonal_method,
        batch_size=cfg.processing.batch_size,
    )

    return PipelineConfig(
        target_fabric=target_fabric,
        domain=domain,
        datasets=datasets,
        output=output,
        processing=processing,
    )


# ---------------------------------------------------------------------------
# pywatershed run
# ---------------------------------------------------------------------------


@pws_app.command(name="run")
def pws_run_cmd(config: Path, *, registry: Path | None = None) -> None:
    """Generate a complete pywatershed model setup.

    Execute the full two-phase workflow to produce all files needed
    for a pywatershed (NHM-PRMS) simulation:

    1. **Generic pipeline** (phase 1) -- run stages 1--5 to produce a
       raw SIR (source units, source variable names) and temporal
       climate data files.
    2. **pywatershed post-processing** (phase 2) -- derive PRMS
       parameters from the SIR (geometry, topology, topography,
       landcover, lookups, defaults), merge temporal data with
       variable renaming and unit conversion (K to C, mm to in),
       and write output files.

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
        Path to a pywatershed run config YAML file.  See
        ``PywatershedRunConfig`` for the expected schema.
    registry
        Path to a custom dataset registry YAML file or directory.
        When omitted, the bundled default registry is used.

    Raises
    ------
    SystemExit
        If config loading or either pipeline phase fails (exit code 1).

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

    from hydro_param.derivations.pywatershed import (
        PywatershedDerivation,
        merge_temporal_into_derived,
    )
    from hydro_param.plugins import DerivationContext, get_formatter
    from hydro_param.pywatershed_config import load_pywatershed_config

    try:
        pws_config = load_pywatershed_config(config)
    except Exception as exc:
        logger.exception("Failed to load pywatershed config '%s'.", config)
        raise SystemExit(1) from exc

    logger.info("pywatershed config validated: %s", config)
    logger.info("  Domain: %s %s", pws_config.domain.extraction_method, pws_config.domain.bbox)
    logger.info("  Time: %s to %s", pws_config.time.start, pws_config.time.end)
    logger.info("  Climate: %s", pws_config.climate.source)
    logger.info("  Output: %s", pws_config.output.path)

    # ── Phase 1: Generic pipeline (raw SIR + temporal) ──
    pipeline_config = _translate_pws_to_pipeline(pws_config)
    reg = _load_registry(registry)

    try:
        result = run_pipeline_from_config(pipeline_config, reg)
    except Exception as exc:
        logger.exception("Pipeline failed (phase 1). Check config '%s'.", config)
        raise SystemExit(1) from exc

    # ── Phase 2: pywatershed post-processing ──
    logger.info("Phase 2: pywatershed derivation + formatting")

    plugin = PywatershedDerivation()
    sir = result.load_sir()

    segments = None
    if pws_config.domain.segment_path is not None:
        segments = gpd.read_file(pws_config.domain.segment_path)

    derivation_config: dict = {}
    if pws_config.parameter_overrides.values:
        derivation_config["parameter_overrides"] = {
            "values": pws_config.parameter_overrides.values,
        }

    ctx = DerivationContext(
        sir=sir,
        fabric=result.fabric,
        segments=segments,
        fabric_id_field=pws_config.domain.id_field,
        segment_id_field=pws_config.domain.segment_id_field,
        config=derivation_config,
    )
    derived = plugin.derive(ctx)

    # Load temporal data from per-file paths and merge with model-specific transforms
    temporal = {name: xr.open_dataset(path) for name, path in result.temporal_files.items()}
    try:
        derived = merge_temporal_into_derived(
            derived,
            temporal,
            renames={"pr": "prcp", "tmmx": "tmax", "tmmn": "tmin"},
            conversions={"tmax": ("K", "C"), "tmin": ("K", "C")},
        )
    finally:
        for ds in temporal.values():
            ds.close()

    # Write using the model-specific formatter
    formatter = get_formatter("pywatershed")
    formatter_config = {
        "parameter_file": pws_config.output.parameter_file,
        "forcing_dir": pws_config.output.forcing_dir,
        "soltab_file": pws_config.output.soltab_file,
        "control_file": pws_config.output.control_file,
        "start": pws_config.time.start,
        "end": pws_config.time.end,
    }
    formatter.write(derived, pws_config.output.path, formatter_config)

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
