"""Download GFv1.1 NHM data layer rasters and topo derivatives from ScienceBase.

Provide utilities for fetching, downloading, and organizing the Geospatial
Fabric version 1.1 (GFv1.1) data layers used by the National Hydrologic Model.
Source data is hosted on USGS ScienceBase as zip archives organized into
thematic subdirectories (soils, land_cover, water_bodies, geology, topo,
metadata).

The module queries the ScienceBase JSON API to discover files, streams
downloads with retry logic, and automatically extracts zip archives into
the appropriate subdirectory structure.

Two ScienceBase items are supported:

- **Data Layers** (``5ebb182b82ce25b5136181cf``) -- soils, land cover,
  water bodies, geology, and metadata files.
- **TGF Topo** (``5ebb17d082ce25b5136181cb``) -- topographic derivatives
  (DEM, slope, aspect, TWI, flow direction).

References
----------
.. [1] Regan, R.S., et al. (2019). NHM Infrastructure, USGS.
.. [2] https://www.sciencebase.gov/catalog/item/5ebb182b82ce25b5136181cf
.. [3] https://www.sciencebase.gov/catalog/item/5ebb17d082ce25b5136181cb
"""

from __future__ import annotations

import logging
import time
import zipfile
from pathlib import Path
from typing import Literal

import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

GFv11Items = Literal["all", "data-layers", "tgf-topo"]

# ---------------------------------------------------------------------------
# ScienceBase item IDs
# ---------------------------------------------------------------------------

DATA_LAYERS_ITEM_ID: str = "5ebb182b82ce25b5136181cf"
"""ScienceBase item ID for GFv1.1 NHM data layers (soils, land cover, etc.)."""

TGF_TOPO_ITEM_ID: str = "5ebb17d082ce25b5136181cb"
"""ScienceBase item ID for GFv1.1 topographic derivatives."""

SB_API_URL: str = "https://www.sciencebase.gov/catalog/item"
"""Base URL for the ScienceBase catalog JSON API."""

# ---------------------------------------------------------------------------
# File-to-subdirectory mapping
# ---------------------------------------------------------------------------

FILE_DIRECTORY_MAP: dict[str, str] = {
    # soils
    "TEXT_PRMS.zip": "soils",
    "Clay.zip": "soils",
    "Silt.zip": "soils",
    "Sand.zip": "soils",
    "AWC.zip": "soils",
    # land_cover
    "LULC.zip": "land_cover",
    "Imperv.zip": "land_cover",
    "CNPY.zip": "land_cover",
    "Snow.zip": "land_cover",
    "SRain.zip": "land_cover",
    "WRain.zip": "land_cover",
    "keep.zip": "land_cover",
    "loss.zip": "land_cover",
    "RootDepth.zip": "land_cover",
    "CV_INT.zip": "land_cover",
    # water_bodies
    "wbg.zip": "water_bodies",
    # geology
    "Lithology_exp_Konly_Project.zip": "geology",
    # metadata
    "CrossWalk.xlsx": "metadata",
    "SDC_table.csv": "metadata",
    "Data Layers for the NHM Domain_Final.xml": "metadata",
    "Topographic_Derivatives_Transboundary_Domain_Final.xml": "metadata",
    "Geospatial Fabric for National Hydrologic Modeling, version 1.2.txt": "metadata",
    # topo
    "dem.zip": "topo",
    "slope100X.zip": "topo",
    "asp100X.zip": "topo",
    "twi100X.zip": "topo",
    "fdr.zip": "topo",
}
"""Map each ScienceBase filename to its target subdirectory under the output root."""


# ---------------------------------------------------------------------------
# ScienceBase API query
# ---------------------------------------------------------------------------


def fetch_item_files(item_id: str) -> list[tuple[str, str, int]]:
    """Query a ScienceBase item and return metadata for its downloadable files.

    Sends a GET request to the ScienceBase JSON API to retrieve the item's
    file listing.  Files that lack a download URL are silently skipped
    (logged at DEBUG level).

    Parameters
    ----------
    item_id : str
        ScienceBase item identifier
        (e.g. ``"5ebb182b82ce25b5136181cf"``).  # pragma: allowlist secret

    Returns
    -------
    list[tuple[str, str, int]]
        Each tuple contains ``(filename, download_url, size_bytes)``.

    Raises
    ------
    requests.HTTPError
        If the ScienceBase API returns a non-2xx status code.
    requests.ConnectionError
        If the ScienceBase API is unreachable.

    Examples
    --------
    >>> files = fetch_item_files("5ebb182b82ce25b5136181cf")  # doctest: +SKIP
    >>> files[0]  # doctest: +SKIP
    ('TEXT_PRMS.zip', 'https://...', 12345678)
    """
    url = f"{SB_API_URL}/{item_id}?format=json"
    logger.info("Querying ScienceBase item %s", item_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    result: list[tuple[str, str, int]] = []
    for f in data.get("files", []):
        name = f.get("name", "")
        dl_url = f.get("url", "")
        size = f.get("size", 0)
        if not dl_url:
            logger.debug("Skipping file %r (no download URL)", name)
            continue
        result.append((name, dl_url, size))

    logger.info("Found %d downloadable files for item %s", len(result), item_id)
    return result


# ---------------------------------------------------------------------------
# Single-file download with retry
# ---------------------------------------------------------------------------


def download_file(url: str, dest: Path, *, retries: int = 3) -> bool:
    """Stream-download a file from a URL to a local path.

    If the destination file already exists the download is skipped and
    ``False`` is returned.  On network errors the download is retried up
    to *retries* times with exponential back-off.  Partial files are
    removed on failure.

    Parameters
    ----------
    url : str
        Fully-qualified download URL.
    dest : Path
        Local filesystem path where the file will be written.
    retries : int, optional
        Maximum number of retry attempts on ``requests.RequestException``.
        Default is 3.

    Returns
    -------
    bool
        ``True`` if the file was downloaded, ``False`` if it already existed.

    Notes
    -----
    Uses chunked streaming (8 KiB chunks) to keep memory usage low for
    large raster archives.  Partial files are deleted when a download fails
    to avoid leaving corrupt data on disk.
    """
    if dest.exists():
        logger.debug("Skipping %s (already exists)", dest.name)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    size_mb = ""

    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", dest.name, attempt, retries)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                if total:
                    size_mb = f" ({total / 1_048_576:.1f} MB)"
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        fh.write(chunk)
            logger.info("Saved %s%s", dest.name, size_mb)
            return True
        except requests.RequestException as exc:
            # Clean up partial file
            if dest.exists():
                dest.unlink()
            if attempt < retries:
                wait = 2**attempt
                logger.warning(
                    "Download failed for %s: %s — retrying in %ds",
                    dest.name,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Download failed for %s after %d attempts: %s",
                    dest.name,
                    retries,
                    exc,
                )
    return False


# ---------------------------------------------------------------------------
# Unzip helper
# ---------------------------------------------------------------------------


def _unzip_and_clean(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip archive and delete the zip file.

    On extraction failure (``BadZipFile``, ``OSError``) the zip file is
    preserved and a warning is logged so the user can inspect or retry
    manually.

    Parameters
    ----------
    zip_path : Path
        Path to the zip archive.
    extract_dir : Path
        Directory into which the archive contents are extracted.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        zip_path.unlink()
        logger.info("Extracted and removed %s", zip_path.name)
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning("Could not extract %s: %s — preserving zip", zip_path.name, exc)


# ---------------------------------------------------------------------------
# Item-level download
# ---------------------------------------------------------------------------


def download_item(item_id: str, output_dir: Path) -> None:
    """Download all mapped files for a ScienceBase item into subdirectories.

    Queries the ScienceBase API for the item's file listing, downloads each
    file whose name appears in :data:`FILE_DIRECTORY_MAP` into the
    corresponding subdirectory under *output_dir*, and extracts any zip
    archives.  Files not present in the map are skipped with a warning.

    Parameters
    ----------
    item_id : str
        ScienceBase item identifier.
    output_dir : Path
        Root directory for downloaded data.  Subdirectories (``soils/``,
        ``land_cover/``, etc.) are created automatically.

    Notes
    -----
    Existing files are not re-downloaded (see :func:`download_file`).
    """
    files = fetch_item_files(item_id)

    for name, url, _size in files:
        subdir = FILE_DIRECTORY_MAP.get(name)
        if subdir is None:
            logger.warning("Unmapped file %r — skipping", name)
            continue

        dest = output_dir / subdir / name
        downloaded = download_file(url, dest)

        if downloaded and name.lower().endswith(".zip"):
            _unzip_and_clean(dest, dest.parent)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def download_gfv11(
    output_dir: Path,
    *,
    items: GFv11Items = "all",
) -> None:
    """Download GFv1.1 raster data from ScienceBase.

    This is the top-level entry point for downloading GFv1.1 NHM data
    layer rasters and/or topographic derivatives.  Files are organized
    into thematic subdirectories under *output_dir*.

    Parameters
    ----------
    output_dir : Path
        Root directory for downloaded data.  Created if it does not exist.
    items : {"all", "data-layers", "tgf-topo"}, optional
        Which ScienceBase items to download.  ``"all"`` downloads both data
        layers and topographic derivatives.  Default is ``"all"``.

    Examples
    --------
    >>> from pathlib import Path
    >>> download_gfv11(Path("/tmp/gfv11"), items="data-layers")  # doctest: +SKIP
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if items in ("all", "data-layers"):
        logger.info("Downloading GFv1.1 data layers")
        download_item(DATA_LAYERS_ITEM_ID, output_dir)

    if items in ("all", "tgf-topo"):
        logger.info("Downloading GFv1.1 topographic derivatives")
        download_item(TGF_TOPO_ITEM_ID, output_dir)
