"""CLI entry point for hydro-param.

Commands:
    hydro-param init [project-dir]       — scaffold a project directory
    hydro-param datasets list            — list available datasets
    hydro-param datasets info <name>     — show dataset details
    hydro-param datasets download <name> — download dataset files
    hydro-param run <config>             — execute the pipeline
    hydro-param pywatershed run <config> — generate pywatershed model setup

See design.md §11.9 for the full CLI specification.
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
    """Load the dataset registry, falling back to the default."""
    path = registry if registry is not None else DEFAULT_REGISTRY
    return load_registry(path)


def _access_status(entry: DatasetEntry) -> str:
    """Return a human-readable access status string."""
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
    """Display datasets grouped by category.

    Parameters
    ----------
    registry
        Path to a custom registry YAML file or directory.
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
    """Show full details for a dataset.

    Parameters
    ----------
    name
        Dataset name as it appears in the registry.
    registry
        Path to a custom registry YAML file or directory.
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
    """Download dataset files via AWS CLI.

    Parameters
    ----------
    name
        Dataset name as it appears in the registry.
    dest
        Destination directory for downloaded files.  When omitted inside
        an initialised project, files are routed to ``data/<category>/``
        automatically.
    years
        Comma-separated list of years to download (multi-file datasets).
    variables
        Comma-separated list of variables/products to download (multi-file datasets).
    registry
        Path to a custom registry YAML file or directory.
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
    """Download a single file via aws s3 cp."""
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
    """Download selected files from a multi-file or template dataset."""
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
    """Execute the parameterization pipeline.

    Parameters
    ----------
    config
        Path to the pipeline YAML config.
    registry
        Path to a custom dataset registry YAML file or directory.
    resume
        Skip datasets whose outputs are already complete and inputs
        haven't changed (manifest-based).
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

    Creates a standard directory structure with a template pipeline
    configuration, data directories organised by dataset category,
    and a .gitignore.

    Parameters
    ----------
    project_dir
        Directory to initialise.  Defaults to the current directory.
    force
        Re-initialise an existing project (creates missing directories,
        refreshes marker, but preserves existing pipeline.yml).
    registry
        Path to a custom registry for category discovery.
    """
    init_project(project_dir, force=force, registry_path=registry)


# ---------------------------------------------------------------------------
# pywatershed helpers
# ---------------------------------------------------------------------------


def _translate_pws_to_pipeline(
    pws_config: object,
) -> PipelineConfig:
    """Translate a PywatershedRunConfig into a generic PipelineConfig.

    Maps the model-specific schema onto the generic pipeline config so
    stages 1-5 can produce a raw SIR + temporal data.  No model-specific
    transforms (derivation, renaming, unit conversion) are included —
    those happen in ``pws_run_cmd()`` after the pipeline completes.
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
        datasets.append(
            DatasetRequest(
                name=lc_name,
                variables=["LndCov"],
                statistics=["categorical"],
                year=2021,
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

    # Climate (temporal) — map user-facing PRMS names to registry/gdptools names
    _CLIMATE_SOURCE_MAP = {
        "gridmet": "gridmet",
        "daymet_v4": "daymet_v4",
        "conus404_ba": "conus404_ba",
    }
    climate_ds_name = _CLIMATE_SOURCE_MAP.get(cfg.climate.source, cfg.climate.source)

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

    Two-phase workflow:

    1. **Generic pipeline** — runs stages 1-5 to produce a raw SIR
       (source units, source variable names) + temporal data.
    2. **pywatershed post-processing** — derives PRMS parameters,
       merges temporal data with renaming/unit conversion, and writes
       output files (parameters.nc, forcing/, control.yml).

    Parameters
    ----------
    config
        Path to a pywatershed run config YAML.
    registry
        Path to a custom dataset registry YAML file or directory.
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
    from hydro_param.output import get_formatter
    from hydro_param.pywatershed_config import load_pywatershed_config

    try:
        pws_config = load_pywatershed_config(config)
    except Exception as exc:
        logger.error("Failed to load pywatershed config: %s", exc)
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
        logger.exception("Pipeline failed (phase 1).")
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

    derived = plugin.derive(
        sir,
        config=derivation_config,
        fabric=result.fabric,
        segments=segments,
        id_field=pws_config.domain.id_field,
        segment_id_field=pws_config.domain.segment_id_field,
    )

    # Load temporal data from per-file paths and merge with model-specific transforms
    temporal = {name: xr.open_dataset(path) for name, path in result.temporal_files.items()}
    derived = merge_temporal_into_derived(
        derived,
        temporal,
        renames={"pr": "prcp", "tmmx": "tmax", "tmmn": "tmin"},
        conversions={"tmax": ("K", "C"), "tmin": ("K", "C")},
    )

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
    *,
    metadata: Path | None = None,
) -> None:
    """Validate a pywatershed parameter file.

    Checks that required parameters are present and values fall
    within valid ranges.

    Parameters
    ----------
    param_file
        Path to a pywatershed parameter NetCDF file.
    metadata
        Path to parameter metadata YAML. Defaults to
        ``configs/pywatershed/parameter_metadata.yml``.
    """
    import xarray as xr

    from hydro_param.formatters.pywatershed import PywatershedFormatter

    try:
        ds = xr.open_dataset(param_file)
    except Exception as exc:
        print(f"Error: Could not open '{param_file}': {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    formatter = PywatershedFormatter(metadata_path=metadata) if metadata else PywatershedFormatter()

    if not formatter.has_metadata():
        print(
            "Warning: parameter metadata not found at "
            f"'{formatter.metadata_path}'. Validation will be incomplete.",
            file=sys.stderr,
        )

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
    """CLI entry point called by the ``hydro-param`` console script."""
    app()
