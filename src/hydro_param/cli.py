"""CLI entry point for hydro-param.

Commands:
    hydro-param datasets list          — list available datasets
    hydro-param datasets info <name>   — show dataset details
    hydro-param datasets download <name> — download dataset files
    hydro-param run <config>           — execute the pipeline

See design.md §11.9 for the full CLI specification.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

from cyclopts import App

from hydro_param.dataset_registry import DatasetEntry, DatasetRegistry, load_registry
from hydro_param.pipeline import DEFAULT_REGISTRY, run_pipeline

logger = logging.getLogger(__name__)

app = App(name="hydro-param", help="Configuration-driven hydrologic parameterization.")
datasets_app = app.command(App(name="datasets", help="Discover and download datasets."))


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
    if entry.strategy in ("stac_cog", "native_zarr", "climr_cat"):
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
        Path to a custom registry YAML file.
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
        Path to a custom registry YAML file.
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
    dest: Path = Path("."),
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
        Destination directory for downloaded files.
    years
        Comma-separated list of years to download (multi-file datasets).
    variables
        Comma-separated list of variables/products to download (multi-file datasets).
    registry
        Path to a custom registry YAML file.
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
        _download_single_file(entry.download.url, dest)
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
def run_cmd(config: Path, *, registry: Path | None = None) -> None:
    """Execute the parameterization pipeline.

    Parameters
    ----------
    config
        Path to the pipeline YAML config.
    registry
        Path to a custom dataset registry YAML file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    registry_path = str(registry) if registry is not None else None
    try:
        run_pipeline(str(config), registry_path)
    except Exception as exc:
        logger.exception("Pipeline failed.")
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point called by the ``hydro-param`` console script."""
    app()
