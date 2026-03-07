"""Download GFv1.1 NHM data layer rasters and topo derivatives from ScienceBase.

Provide utilities for fetching, downloading, and organizing the NHM v1.1
data layers and topographic derivatives from USGS ScienceBase.  Source data
is hosted as zip archives organized into thematic subdirectories (soils,
land_cover, water_bodies, geology, topo, metadata).

The module queries the ScienceBase JSON API to discover files, streams
downloads with retry logic, and automatically extracts zip archives into
the appropriate subdirectory structure.

Two ScienceBase items are supported:

- **Data Layers** (``5ebb182b82ce25b5136181cf``) -- soils, land cover,
  water bodies, geology, and metadata files for the NHM v1.1 domain.
- **TGF Topo** (``5ebb17d082ce25b5136181cb``) -- topographic derivatives
  (DEM, slope, aspect, TWI, flow direction) for the US-Canada
  transboundary Geospatial Fabric.

References
----------
.. [1] Regan, R.S., Markstrom, S.L., Hay, L.E., Viger, R.J., Norton, P.A.,
   Driscoll, J.M., and LaFontaine, J.H., 2018, Description of the National
   Hydrologic Model for use with the Precipitation-Runoff Modeling System
   (PRMS): U.S. Geological Survey Techniques and Methods, book 6, chap. B9,
   38 p., https://doi.org/10.3133/tm6B9
.. [2] https://www.sciencebase.gov/catalog/item/5ebb182b82ce25b5136181cf
.. [3] https://www.sciencebase.gov/catalog/item/5ebb17d082ce25b5136181cb
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

GFv11Items = Literal["all", "data-layers", "tgf-topo"]

# ---------------------------------------------------------------------------
# ScienceBase item IDs
# ---------------------------------------------------------------------------

DATA_LAYERS_ITEM_ID: str = "5ebb182b82ce25b5136181cf"
"""ScienceBase item ID for NHM v1.1 data layers (soils, land cover, etc.)."""

TGF_TOPO_ITEM_ID: str = "5ebb17d082ce25b5136181cb"
"""ScienceBase item ID for transboundary GF topographic derivatives."""

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
"""Map recognized ScienceBase filenames to target subdirectories.

Files not listed here are skipped at download time with a warning.
"""


# ---------------------------------------------------------------------------
# Download result tracking
# ---------------------------------------------------------------------------


@dataclass
class DownloadSummary:
    """Accumulate download results for reporting.

    Attributes
    ----------
    downloaded : list[str]
        Filenames successfully downloaded.
    skipped : list[str]
        Filenames skipped (already existed on disk).
    failed : list[str]
        Filenames that failed after exhausting retries.
    extract_failed : list[str]
        Filenames where zip extraction failed.
    unmapped : list[str]
        Filenames not present in FILE_DIRECTORY_MAP.
    """

    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    extract_failed: list[str] = field(default_factory=list)
    unmapped: list[str] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        """Return True if any downloads or extractions failed."""
        return bool(self.failed or self.extract_failed)

    def log_summary(self) -> None:
        """Log a summary of all download results."""
        total = len(self.downloaded) + len(self.skipped) + len(self.failed)
        logger.info(
            "Download summary: %d downloaded, %d skipped, %d failed (of %d total)",
            len(self.downloaded),
            len(self.skipped),
            len(self.failed),
            total,
        )
        if self.extract_failed:
            logger.error(
                "Extraction failed for %d file(s): %s",
                len(self.extract_failed),
                ", ".join(self.extract_failed),
            )
        if self.failed:
            logger.error(
                "Download failed for %d file(s): %s",
                len(self.failed),
                ", ".join(self.failed),
            )
        if self.unmapped:
            logger.warning(
                "Skipped %d unmapped file(s): %s",
                len(self.unmapped),
                ", ".join(self.unmapped),
            )


class DownloadError(Exception):
    """A file download failed after all retry attempts."""


# ---------------------------------------------------------------------------
# ScienceBase API query
# ---------------------------------------------------------------------------


def fetch_item_files(item_id: str) -> list[tuple[str, str, int]]:
    """Query a ScienceBase item and return metadata for its downloadable files.

    Send a GET request to the ScienceBase JSON API to retrieve the item's
    file listing.  Files that lack a download URL are silently skipped
    (logged at DEBUG level).

    Parameters
    ----------
    item_id : str
        ScienceBase item identifier (24-character hex string).

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
    ValueError
        If the ScienceBase API returns a non-JSON response (e.g. during
        maintenance windows).

    Examples
    --------
    >>> files = fetch_item_files(  # doctest: +SKIP
    ...     "5ebb182b82ce25b5136181cf"  # pragma: allowlist secret
    ... )
    >>> files[0]  # doctest: +SKIP
    ('TEXT_PRMS.zip', 'https://...', 12345678)
    """
    url = f"{SB_API_URL}/{item_id}?format=json"
    logger.info("Querying ScienceBase item %s", item_id)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError as exc:
        raise ValueError(
            f"ScienceBase returned non-JSON response for item {item_id}. "
            f"The service may be in maintenance mode. Try again later."
        ) from exc

    raw_files = data.get("files")
    if raw_files is None:
        logger.warning(
            "ScienceBase item %s has no 'files' key in response — "
            "the API schema may have changed or the item ID may be wrong",
            item_id,
        )
        return []

    result: list[tuple[str, str, int]] = []
    for f in raw_files:
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
        Maximum number of download attempts.  Default is 3.

    Returns
    -------
    bool
        ``True`` if the file was downloaded, ``False`` if it already existed.

    Raises
    ------
    DownloadError
        If all retry attempts are exhausted.
    OSError
        If the filesystem is unwritable (e.g. disk full, permissions).

    Notes
    -----
    Uses chunked streaming (8 KiB chunks) to keep memory usage low for
    large raster archives.  Partial files are deleted when a download fails
    to avoid leaving corrupt data on disk.  Retry delay is ``2**attempt``
    seconds (2 s, 4 s, 8 s for the default 3 attempts).
    """
    if dest.exists():
        logger.debug("Skipping %s (already exists)", dest.name)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading %s (attempt %d/%d)", dest.name, attempt, retries)
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                size_mb = f" ({total / 1_048_576:.1f} MB)" if total else ""
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        fh.write(chunk)
            logger.info("Saved %s%s", dest.name, size_mb)
            return True
        except requests.RequestException as exc:
            last_exc = exc
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
        except OSError:
            # Filesystem error (disk full, permissions) — clean up and raise
            if dest.exists():
                dest.unlink()
            raise

    raise DownloadError(f"Download failed for {dest.name} after {retries} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Unzip helper
# ---------------------------------------------------------------------------


def _unzip_and_clean(zip_path: Path, extract_dir: Path) -> bool:
    """Extract a zip archive and delete the zip file.

    On extraction failure (``BadZipFile``, ``OSError``) the zip file is
    preserved and an error is logged so the user can inspect or retry
    manually.

    Parameters
    ----------
    zip_path : Path
        Path to the zip archive.
    extract_dir : Path
        Directory into which the archive contents are extracted.

    Returns
    -------
    bool
        ``True`` if extraction succeeded, ``False`` otherwise.

    Notes
    -----
    Validates archive member paths before extraction to guard against
    zip-slip (path traversal) attacks.  This is relevant for Python < 3.12
    which lacks built-in protections in ``extractall()``.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Guard against zip-slip (path traversal)
            resolved_dir = extract_dir.resolve()
            for member in zf.namelist():
                member_path = (extract_dir / member).resolve()
                if not str(member_path).startswith(str(resolved_dir)):
                    raise zipfile.BadZipFile(f"Zip slip detected in {zip_path.name}: {member}")
            zf.extractall(extract_dir)
        zip_path.unlink()
        logger.info("Extracted and removed %s", zip_path.name)
        return True
    except NotImplementedError:
        # Deflate64 and other compression methods unsupported by Python's
        # zipfile module — fall back to the system ``unzip`` command.
        return _unzip_with_system(zip_path, extract_dir)
    except (zipfile.BadZipFile, OSError) as exc:
        logger.error("Could not extract %s: %s — preserving zip", zip_path.name, exc)
        return False


def _unzip_with_system(zip_path: Path, extract_dir: Path) -> bool:
    """Extract a zip archive using the system ``unzip`` command.

    Falls back to ``7z`` if ``unzip`` is not available.  Used when Python's
    :mod:`zipfile` module cannot handle the archive's compression method
    (e.g., Deflate64 / method 9).

    Parameters
    ----------
    zip_path : Path
        Path to the zip archive.
    extract_dir : Path
        Directory into which archive contents are extracted.

    Returns
    -------
    bool
        ``True`` if extraction succeeded, ``False`` otherwise.
    """
    unzip_bin = shutil.which("unzip")
    if unzip_bin:
        cmd = [unzip_bin, "-o", str(zip_path), "-d", str(extract_dir)]
    else:
        sevenz_bin = shutil.which("7z")
        if sevenz_bin:
            cmd = [sevenz_bin, "x", str(zip_path), f"-o{extract_dir}", "-y"]
        else:
            logger.error(
                "Cannot extract %s: unsupported compression and neither "
                "'unzip' nor '7z' found on PATH — preserving zip",
                zip_path.name,
            )
            return False

    logger.info(
        "Falling back to system extractor for %s (unsupported compression)",
        zip_path.name,
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True)  # noqa: S603
    except subprocess.CalledProcessError as exc:
        logger.error(
            "System extraction failed for %s: %s — preserving zip",
            zip_path.name,
            exc.stderr.decode(errors="replace").strip(),
        )
        return False

    zip_path.unlink()
    logger.info("Extracted and removed %s", zip_path.name)
    return True


# ---------------------------------------------------------------------------
# Item-level download
# ---------------------------------------------------------------------------


def download_item(item_id: str, output_dir: Path) -> DownloadSummary:
    """Download all mapped files for a ScienceBase item into subdirectories.

    Query the ScienceBase API for the item's file listing, download each
    file whose name appears in :data:`FILE_DIRECTORY_MAP` into the
    corresponding subdirectory under *output_dir*, and extract any zip
    archives.  Files not present in the map are skipped with a warning.

    Parameters
    ----------
    item_id : str
        ScienceBase item identifier.
    output_dir : Path
        Root directory for downloaded data.  Subdirectories (``soils/``,
        ``land_cover/``, etc.) are created automatically.

    Returns
    -------
    DownloadSummary
        Counts of downloaded, skipped, failed, and unmapped files.

    Notes
    -----
    Existing files are not re-downloaded (see :func:`download_file`).
    Download failures are accumulated and reported in the summary rather
    than stopping the entire batch.
    """
    files = fetch_item_files(item_id)
    summary = DownloadSummary()

    for name, url, _size in files:
        subdir = FILE_DIRECTORY_MAP.get(name)
        if subdir is None:
            logger.warning("Unmapped file %r — skipping", name)
            summary.unmapped.append(name)
            continue

        dest = output_dir / subdir / name
        try:
            downloaded = download_file(url, dest)
        except DownloadError:
            summary.failed.append(name)
            continue

        if downloaded:
            summary.downloaded.append(name)
            if name.lower().endswith(".zip"):
                if not _unzip_and_clean(dest, dest.parent):
                    summary.extract_failed.append(name)
        else:
            summary.skipped.append(name)

    return summary


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def download_gfv11(
    output_dir: Path,
    *,
    items: GFv11Items = "all",
) -> DownloadSummary:
    """Download GFv1.1 raster data from ScienceBase.

    This is the top-level entry point for downloading NHM v1.1 data
    layer rasters and/or topographic derivatives.  Files are organized
    into thematic subdirectories under *output_dir*.

    Parameters
    ----------
    output_dir : Path
        Root directory for downloaded data.  Created if it does not exist.
    items : {"all", "data-layers", "tgf-topo"}, optional
        Which ScienceBase items to download.  ``"all"`` downloads both data
        layers and topographic derivatives.  Default is ``"all"``.

    Returns
    -------
    DownloadSummary
        Combined summary across all downloaded items.

    Raises
    ------
    DownloadError
        If any files failed to download or extract (raised after
        logging a summary).

    Examples
    --------
    >>> from pathlib import Path
    >>> download_gfv11(Path("/tmp/gfv11"), items="data-layers")  # doctest: +SKIP
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = DownloadSummary()

    def _merge(summary: DownloadSummary) -> None:
        combined.downloaded.extend(summary.downloaded)
        combined.skipped.extend(summary.skipped)
        combined.failed.extend(summary.failed)
        combined.extract_failed.extend(summary.extract_failed)
        combined.unmapped.extend(summary.unmapped)

    if items in ("all", "data-layers"):
        logger.info("Downloading GFv1.1 data layers")
        _merge(download_item(DATA_LAYERS_ITEM_ID, output_dir))

    if items in ("all", "tgf-topo"):
        logger.info("Downloading GFv1.1 topographic derivatives")
        _merge(download_item(TGF_TOPO_ITEM_ID, output_dir))

    combined.log_summary()

    if combined.has_failures:
        raise DownloadError(
            f"{len(combined.failed)} download(s) and "
            f"{len(combined.extract_failed)} extraction(s) failed. "
            f"See log output above for details."
        )

    return combined
