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
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import requests  # type: ignore[import-untyped]
import yaml
import zipfile_deflate64  # noqa: F401  — registers Deflate64 codec with zipfile

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
# GFv1.1 dataset metadata
# ---------------------------------------------------------------------------

GFV11_DATASETS: dict[str, dict] = {
    # --- Soils (5) ---
    "gfv11_sand": {
        "description": "GFv1.1 SoilGrids250m sand %, 250m, CONUS",
        "category": "soils",
        "filename": "Sand.tif",
        "subdir": "soils",
        "variables": [
            {
                "name": "sand_pct",
                "band": 1,
                "units": "%",
                "long_name": "Depth-weighted sand percentage (SoilGrids250m)",
                "native_name": "sand_pct",
                "categorical": False,
            }
        ],
    },
    "gfv11_clay": {
        "description": "GFv1.1 SoilGrids250m clay %, 250m, CONUS",
        "category": "soils",
        "filename": "Clay.tif",
        "subdir": "soils",
        "variables": [
            {
                "name": "clay_pct",
                "band": 1,
                "units": "%",
                "long_name": "Depth-weighted clay percentage (SoilGrids250m)",
                "native_name": "clay_pct",
                "categorical": False,
            }
        ],
    },
    "gfv11_silt": {
        "description": "GFv1.1 SoilGrids250m silt %, 250m, CONUS",
        "category": "soils",
        "filename": "Silt.tif",
        "subdir": "soils",
        "variables": [
            {
                "name": "silt_pct",
                "band": 1,
                "units": "%",
                "long_name": "Silt percentage (derived: 100 - sand - clay)",
                "native_name": "silt_pct",
                "categorical": False,
            }
        ],
    },
    "gfv11_awc": {
        "description": "GFv1.1 SoilGrids250m available water capacity, 250m, CONUS",
        "category": "soils",
        "filename": "AWC.tif",
        "subdir": "soils",
        "variables": [
            {
                "name": "awc",
                "band": 1,
                "units": "mm",
                "long_name": "Available water capacity (SoilGrids250m)",
                "native_name": "awc",
                "categorical": False,
            }
        ],
    },
    "gfv11_text_prms": {
        "description": "GFv1.1 USDA texture class -> PRMS soil_type codes, 250m, CONUS",
        "category": "soils",
        "filename": "TEXT_PRMS.tif",
        "subdir": "soils",
        "variables": [
            {
                "name": "soil_type",
                "band": 1,
                "units": "class",
                "long_name": "PRMS soil type (1=sand, 2=loam, 3=clay)",
                "native_name": "soil_type",
                "categorical": True,
            }
        ],
    },
    # --- Land Cover (10) ---
    "gfv11_lulc": {
        "description": "GFv1.1 NALCMS 2015 -> PRMS cov_type, 30m, CONUS",
        "category": "land_cover",
        "filename": "LULC.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "cov_type",
                "band": 1,
                "units": "class",
                "long_name": "PRMS cover type (0=bare, 1=grasses, 2=shrubs, 3=trees, 4=coniferous)",
                "native_name": "cov_type",
                "categorical": True,
            }
        ],
    },
    "gfv11_imperv": {
        "description": "GFv1.1 GMIS impervious surface %, 30m, CONUS",
        "category": "land_cover",
        "filename": "Imperv.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "imperv_pct",
                "band": 1,
                "units": "%",
                "long_name": "Impervious surface percentage (GMIS)",
                "native_name": "imperv_pct",
                "categorical": False,
            }
        ],
    },
    "gfv11_cnpy": {
        "description": "GFv1.1 MODIS tree canopy cover %, 30m, CONUS",
        "category": "land_cover",
        "filename": "CNPY.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "canopy_pct",
                "band": 1,
                "units": "%",
                "long_name": "Tree canopy cover percentage (MODIS MOD44B)",
                "native_name": "canopy_pct",
                "categorical": False,
            }
        ],
    },
    "gfv11_srain": {
        "description": "GFv1.1 pre-computed summer rain interception, 30m, CONUS",
        "category": "land_cover",
        "filename": "SRain.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "srain_intcp",
                "band": 1,
                "units": "inches",
                "long_name": "Summer rain interception (pre-computed from NALCMS + lookup)",
                "native_name": "srain_intcp",
                "categorical": False,
            }
        ],
    },
    "gfv11_wrain": {
        "description": "GFv1.1 pre-computed winter rain interception, 30m, CONUS",
        "category": "land_cover",
        "filename": "WRain.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "wrain_intcp",
                "band": 1,
                "units": "inches",
                "long_name": "Winter rain interception (pre-computed from NALCMS + lookup)",
                "native_name": "wrain_intcp",
                "categorical": False,
            }
        ],
    },
    "gfv11_snow_intcp": {
        "description": "GFv1.1 pre-computed snow interception, 30m, CONUS",
        "category": "land_cover",
        "filename": "Snow.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "snow_intcp",
                "band": 1,
                "units": "inches",
                "long_name": "Snow interception (pre-computed from NALCMS + lookup)",
                "native_name": "snow_intcp",
                "categorical": False,
            }
        ],
    },
    "gfv11_covden_win": {
        "description": "GFv1.1 pre-computed winter cover density, 30m, CONUS",
        "category": "land_cover",
        "filename": "keep.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "covden_win",
                "band": 1,
                "units": "fraction",
                "long_name": "Winter vegetation cover density (pre-computed from NALCMS + lookup)",
                "native_name": "covden_win",
                "categorical": False,
            }
        ],
    },
    "gfv11_covden_loss": {
        "description": "GFv1.1 pre-computed seasonal cover density loss, 30m, CONUS",
        "category": "land_cover",
        "filename": "loss.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "covden_loss",
                "band": 1,
                "units": "fraction",
                "long_name": "Seasonal cover density loss (covden_sum - covden_win)",
                "native_name": "covden_loss",
                "categorical": False,
            }
        ],
    },
    "gfv11_covden_sum": {
        "description": "GFv1.1 pre-computed summer cover density, 30m, CONUS",
        "category": "land_cover",
        "filename": "CV_INT.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "covden_sum",
                "band": 1,
                "units": "fraction",
                "long_name": "Summer vegetation cover density (pre-computed from NALCMS + lookup)",
                "native_name": "covden_sum",
                "categorical": False,
            }
        ],
    },
    "gfv11_root_depth": {
        "description": "GFv1.1 pre-computed root depth, 30m, CONUS",
        "category": "land_cover",
        "filename": "RootDepth.tif",
        "subdir": "land_cover",
        "variables": [
            {
                "name": "root_depth",
                "band": 1,
                "units": "inches",
                "long_name": "Root depth (pre-computed from NALCMS + lookup)",
                "native_name": "root_depth",
                "categorical": False,
            }
        ],
    },
    # --- Water Bodies (1) ---
    "gfv11_wbg": {
        "description": "GFv1.1 NHD HR waterbody mask, 30m, CONUS",
        "category": "water_bodies",
        "filename": "wbg.tif",
        "subdir": "water_bodies",
        "variables": [
            {
                "name": "waterbody",
                "band": 1,
                "units": "class",
                "long_name": "NHD HR waterbody presence mask",
                "native_name": "waterbody",
                "categorical": True,
            }
        ],
    },
    # --- Topography (5) ---
    "gfv11_dem": {
        "description": "GFv1.1 SRTM 30m DEM, TGF domain",
        "category": "topography",
        "filename": "dem.tif",
        "subdir": "topo",
        "variables": [
            {
                "name": "elevation",
                "band": 1,
                "units": "m",
                "long_name": "Surface elevation (SRTM 30m)",
                "native_name": "elevation",
                "categorical": False,
            }
        ],
    },
    "gfv11_slope": {
        "description": "GFv1.1 TGF terrain slope, 30m (integer-encoded x 100)",
        "category": "topography",
        "filename": "slope100X.tif",
        "subdir": "topo",
        "variables": [
            {
                "name": "slope",
                "band": 1,
                "units": "degrees",
                "long_name": "Terrain slope (stored as value x 100)",
                "native_name": "slope",
                "categorical": False,
                "scale_factor": 0.01,
            }
        ],
    },
    "gfv11_aspect": {
        "description": "GFv1.1 TGF terrain aspect, 30m (integer-encoded x 100)",
        "category": "topography",
        "filename": "asp100X.tif",
        "subdir": "topo",
        "variables": [
            {
                "name": "aspect",
                "band": 1,
                "units": "degrees",
                "long_name": "Terrain aspect clockwise from north (stored as value x 100)",
                "native_name": "aspect",
                "categorical": False,
                "scale_factor": 0.01,
            }
        ],
    },
    "gfv11_twi": {
        "description": "GFv1.1 TGF topographic wetness index, 30m (integer-encoded x 100)",
        "category": "topography",
        "filename": "twi100X.tif",
        "subdir": "topo",
        "variables": [
            {
                "name": "twi",
                "band": 1,
                "units": "unitless",
                "long_name": "Topographic wetness index (stored as value x 100)",
                "native_name": "twi",
                "categorical": False,
                "scale_factor": 0.01,
            }
        ],
    },
    "gfv11_fdr": {
        "description": "GFv1.1 TGF D8 flow direction, 30m",
        "category": "topography",
        "filename": "fdr.tif",
        "subdir": "topo",
        "variables": [
            {
                "name": "flow_dir",
                "band": 1,
                "units": "class",
                "long_name": "D8 flow direction",
                "native_name": "flow_dir",
                "categorical": True,
            }
        ],
    },
}
"""GFv1.1 dataset metadata for all 21 ScienceBase rasters.

Maps registry dataset names to their metadata (description, category,
source filename, subdirectory, and variable specifications).  Used by
:func:`write_registry_overlay` to generate user-local registry entries
with resolved ``source`` paths after download.
"""

GFV11_CRS = "EPSG:5070"
"""CRS for all GFv1.1 rasters (ESRI:102039 == EPSG:5070)."""

GFV11_OVERLAY_FILENAME = "gfv11.yml"
"""Filename for the auto-generated GFv1.1 registry overlay."""


def write_registry_overlay(
    data_dir: Path,
    overlay_path: Path | None = None,
) -> Path:
    """Write a dataset registry overlay YAML for downloaded GFv1.1 rasters.

    Generate a complete registry YAML file with one ``DatasetEntry`` per
    GFv1.1 raster, using resolved absolute ``source`` paths based on
    *data_dir*.  The output file can be loaded by the registry overlay
    mechanism in :func:`~hydro_param.dataset_registry.load_registry`.

    Parameters
    ----------
    data_dir : Path
        Root directory where GFv1.1 rasters were downloaded (the
        ``--output-dir`` passed to ``gfv11 download``).  Subdirectory
        structure (``soils/``, ``land_cover/``, etc.) is assumed.
    overlay_path : Path or None
        Where to write the overlay YAML.  Defaults to
        ``~/.hydro-param/datasets/gfv11.yml``.

    Returns
    -------
    Path
        Absolute path to the written overlay file.

    Notes
    -----
    Overwrites any existing overlay file at the same path.  The file
    includes a header comment recording the source data directory and
    generation timestamp.
    """
    from hydro_param.pipeline import USER_REGISTRY_DIR

    if overlay_path is None:
        overlay_path = USER_REGISTRY_DIR / GFV11_OVERLAY_FILENAME

    resolved_dir = data_dir.resolve()
    datasets: dict[str, dict] = {}

    for name, meta in GFV11_DATASETS.items():
        source = str(resolved_dir / meta["subdir"] / meta["filename"])
        entry: dict = {
            "description": meta["description"],
            "strategy": "local_tiff",
            "source": source,
            "crs": GFV11_CRS,
            "x_coord": "x",
            "y_coord": "y",
            "category": meta["category"],
            "temporal": False,
            "variables": meta["variables"],
        }
        datasets[name] = entry

    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    from datetime import date as _date

    header = (
        f"# Auto-generated by: hydro-param gfv11 download\n"
        f"# Source directory: {resolved_dir}\n"
        f"# Generated: {_date.today().isoformat()}\n"
        f"# ScienceBase items: {DATA_LAYERS_ITEM_ID}, {TGF_TOPO_ITEM_ID}\n\n"
    )
    body = yaml.dump({"datasets": datasets}, default_flow_style=False, sort_keys=False)

    overlay_path.write_text(header + body)
    logger.info("Registered %d GFv1.1 datasets in %s", len(datasets), overlay_path)

    return overlay_path


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
        Filenames skipped (already existed on disk or already extracted).
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
    except (zipfile.BadZipFile, OSError) as exc:
        logger.error("Could not extract %s: %s — preserving zip", zip_path.name, exc)
        return False


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
    For zip archives, a ``.zip.done`` marker file is written after
    successful extraction; on subsequent runs, files with an existing
    marker are skipped without contacting the server.
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

        # For zip files the archive is deleted after extraction, so check
        # for a marker file that records successful extraction.
        marker = dest.with_suffix(".zip.done") if name.lower().endswith(".zip") else None
        if marker and marker.exists():
            logger.info("Skipping %s (already extracted)", name)
            summary.skipped.append(name)
            continue

        try:
            downloaded = download_file(url, dest)
        except DownloadError:
            summary.failed.append(name)
            continue

        if downloaded:
            summary.downloaded.append(name)
            if name.lower().endswith(".zip"):
                assert marker is not None  # zip guard above guarantees this
                if _unzip_and_clean(dest, dest.parent):
                    try:
                        marker.touch()
                    except OSError:
                        logger.warning(
                            "Could not write marker %s — file will be re-downloaded on next run",
                            marker.name,
                        )
                else:
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
