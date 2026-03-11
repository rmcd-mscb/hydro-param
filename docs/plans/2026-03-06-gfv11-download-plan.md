# GFv1.1 ScienceBase Download CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `hydro-param gfv11 download` CLI command to download GFv1.1 NHM data layer rasters from ScienceBase into a shared data directory.

**Architecture:** New `src/hydro_param/gfv11.py` module with download logic (query SB JSON API, stream files, unzip). CLI wiring in `cli.py` adds a `gfv11` command group. File-to-directory mapping is a hardcoded dict — explicit, no magic.

**Tech Stack:** `requests` (HTTP downloads), stdlib `zipfile`, `pathlib`, `argparse`. cyclopts for CLI.

---

### Task 1: Create gfv11.py module with SB item constants and file mapping

**Files:**
- Create: `src/hydro_param/gfv11.py`
- Test: `tests/test_gfv11.py`

**Step 1: Write the failing test**

```python
# tests/test_gfv11.py
"""Tests for GFv1.1 ScienceBase download utilities."""

from hydro_param.gfv11 import (
    DATA_LAYERS_ITEM_ID,
    TGF_TOPO_ITEM_ID,
    FILE_DIRECTORY_MAP,
)


def test_sciencebase_item_ids():
    """Item IDs should be the known ScienceBase identifiers."""
    assert DATA_LAYERS_ITEM_ID == "5ebb182b82ce25b5136181cf"
    assert TGF_TOPO_ITEM_ID == "5ebb17d082ce25b5136181cb"


def test_file_directory_map_covers_all_data_files():
    """Every downloadable data file should have a directory mapping."""
    expected_files = {
        "TEXT_PRMS.zip", "Clay.zip", "Silt.zip", "Sand.zip", "AWC.zip",
        "LULC.zip", "Imperv.zip", "CNPY.zip", "Snow.zip", "SRain.zip",
        "WRain.zip", "keep.zip", "loss.zip", "RootDepth.zip", "CV_INT.zip",
        "Lithology_exp_Konly_Project.zip", "wbg.zip",
        "CrossWalk.xlsx", "SDC_table.csv",
        "Data Layers for the NHM Domain_Final.xml",
        "dem.zip", "slope100X.zip", "asp100X.zip", "twi100X.zip", "fdr.zip",
        "Topographic_Derivatives_Transboundary_Domain_Final.xml",
    }
    assert expected_files.issubset(set(FILE_DIRECTORY_MAP.keys()))


def test_file_directory_map_subdirectories():
    """Spot-check that files map to the correct subdirectories."""
    assert FILE_DIRECTORY_MAP["Sand.zip"] == "soils"
    assert FILE_DIRECTORY_MAP["LULC.zip"] == "land_cover"
    assert FILE_DIRECTORY_MAP["wbg.zip"] == "water_bodies"
    assert FILE_DIRECTORY_MAP["dem.zip"] == "topo"
    assert FILE_DIRECTORY_MAP["Lithology_exp_Konly_Project.zip"] == "geology"
    assert FILE_DIRECTORY_MAP["CrossWalk.xlsx"] == "metadata"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_gfv11.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError`

**Step 3: Write minimal implementation**

```python
# src/hydro_param/gfv11.py
"""GFv1.1 ScienceBase data layer download utilities.

Download pre-built CONUS-wide rasters from two ScienceBase items
(NHM v1.1 Data Layers and TGF Topographic Derivatives) into a shared
data directory organised by thematic category.

References
----------
Provenance: docs/reference/gfv11_raster_provenance.md
Design: docs/plans/2026-03-06-gfv11-download-design.md
"""

from __future__ import annotations

DATA_LAYERS_ITEM_ID = "5ebb182b82ce25b5136181cf"
"""ScienceBase item ID for Data Layers for NHM v1.1."""

TGF_TOPO_ITEM_ID = "5ebb17d082ce25b5136181cb"
"""ScienceBase item ID for TGF Topographic Derivatives."""

SB_API_URL = "https://www.sciencebase.gov/catalog/item"
"""Base URL for ScienceBase catalog JSON API."""

FILE_DIRECTORY_MAP: dict[str, str] = {
    # Data Layers item — soils
    "TEXT_PRMS.zip": "soils",
    "Clay.zip": "soils",
    "Silt.zip": "soils",
    "Sand.zip": "soils",
    "AWC.zip": "soils",
    # Data Layers item — land cover
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
    # Data Layers item — water bodies
    "wbg.zip": "water_bodies",
    # Data Layers item — geology
    "Lithology_exp_Konly_Project.zip": "geology",
    # Data Layers item — metadata
    "CrossWalk.xlsx": "metadata",
    "SDC_table.csv": "metadata",
    "Data Layers for the NHM Domain_Final.xml": "metadata",
    # TGF Topo item — topography
    "dem.zip": "topo",
    "slope100X.zip": "topo",
    "asp100X.zip": "topo",
    "twi100X.zip": "topo",
    "fdr.zip": "topo",
    "Topographic_Derivatives_Transboundary_Domain_Final.xml": "metadata",
    "Geospatial Fabric for National Hydrologic Modeling, version 1.2.txt": "metadata",
}
"""Map each ScienceBase filename to its target subdirectory."""
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_gfv11.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/gfv11.py tests/test_gfv11.py
git commit -m "feat(gfv11): add SB item constants and file-to-directory mapping"
```

---

### Task 2: Add fetch_item_files() to query ScienceBase API

**Files:**
- Modify: `src/hydro_param/gfv11.py`
- Test: `tests/test_gfv11.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_gfv11.py
from unittest.mock import patch, MagicMock
import json

from hydro_param.gfv11 import fetch_item_files


def _mock_sb_response(filenames: list[str]) -> dict:
    """Build a fake ScienceBase JSON API response."""
    return {
        "title": "Test Item",
        "files": [
            {"name": f, "url": f"https://example.com/{f}", "size": 1000}
            for f in filenames
        ],
    }


def test_fetch_item_files_returns_file_list():
    """fetch_item_files should return list of (name, url, size) tuples."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_sb_response(["Sand.zip", "Clay.zip"])
    mock_resp.raise_for_status = MagicMock()

    with patch("hydro_param.gfv11.requests.get", return_value=mock_resp) as mock_get:
        files = fetch_item_files("fake-id")

    mock_get.assert_called_once_with(
        "https://www.sciencebase.gov/catalog/item/fake-id",
        params={"format": "json"},
        timeout=30,
    )
    assert len(files) == 2
    assert files[0] == ("Sand.zip", "https://example.com/Sand.zip", 1000)
    assert files[1] == ("Clay.zip", "https://example.com/Clay.zip", 1000)


def test_fetch_item_files_skips_files_without_url():
    """Files missing a URL should be silently skipped."""
    response_data = {
        "title": "Test",
        "files": [
            {"name": "good.zip", "url": "https://example.com/good.zip", "size": 100},
            {"name": "bad.zip", "size": 100},  # no url
        ],
    }
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_data
    mock_resp.raise_for_status = MagicMock()

    with patch("hydro_param.gfv11.requests.get", return_value=mock_resp):
        files = fetch_item_files("fake-id")

    assert len(files) == 1
    assert files[0][0] == "good.zip"
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_gfv11.py::test_fetch_item_files_returns_file_list tests/test_gfv11.py::test_fetch_item_files_skips_files_without_url -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/hydro_param/gfv11.py`:

```python
import logging

import requests

logger = logging.getLogger(__name__)


def fetch_item_files(item_id: str) -> list[tuple[str, str, int]]:
    """Query the ScienceBase JSON API for an item's downloadable files.

    Parameters
    ----------
    item_id
        ScienceBase item identifier (e.g., ``"5ebb182b82ce25b5136181cf"``).

    Returns
    -------
    list[tuple[str, str, int]]
        List of ``(filename, download_url, size_bytes)`` tuples for each
        file attached to the item.  Files without a download URL are
        silently skipped.

    Raises
    ------
    requests.HTTPError
        If the ScienceBase API returns a non-2xx response.
    """
    url = f"{SB_API_URL}/{item_id}"
    resp = requests.get(url, params={"format": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    files = []
    for f in data.get("files", []):
        name = f.get("name")
        dl_url = f.get("url")
        size = f.get("size", 0)
        if name and dl_url:
            files.append((name, dl_url, size))
        elif name:
            logger.debug("Skipping file '%s' — no download URL", name)
    return files
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_gfv11.py -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/gfv11.py tests/test_gfv11.py
git commit -m "feat(gfv11): add fetch_item_files() to query ScienceBase API"
```

---

### Task 3: Add download_file() with streaming, progress, and skip-if-exists

**Files:**
- Modify: `src/hydro_param/gfv11.py`
- Test: `tests/test_gfv11.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_gfv11.py
from pathlib import Path

from hydro_param.gfv11 import download_file


def test_download_file_skips_existing(tmp_path: Path):
    """download_file should skip when the target file already exists."""
    target = tmp_path / "existing.zip"
    target.write_bytes(b"existing content")

    result = download_file("https://example.com/existing.zip", target)
    assert result is False  # skipped
    assert target.read_bytes() == b"existing content"  # unchanged


def test_download_file_streams_content(tmp_path: Path):
    """download_file should stream response content to disk."""
    target = tmp_path / "new.zip"
    fake_content = b"fake zip data" * 100

    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": str(len(fake_content))}
    mock_resp.iter_content = MagicMock(return_value=[fake_content])
    mock_resp.raise_for_status = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("hydro_param.gfv11.requests.get", return_value=mock_resp):
        result = download_file("https://example.com/new.zip", target)

    assert result is True
    assert target.read_bytes() == fake_content
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_gfv11.py::test_download_file_skips_existing tests/test_gfv11.py::test_download_file_streams_content -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/hydro_param/gfv11.py`:

```python
from pathlib import Path

_CHUNK_SIZE = 8192


def download_file(url: str, dest: Path, *, retries: int = 3) -> bool:
    """Download a single file from a URL with streaming and retry.

    Parameters
    ----------
    url
        Full download URL.
    dest
        Local file path to write to.
    retries
        Number of retry attempts on network failure.

    Returns
    -------
    bool
        ``True`` if the file was downloaded, ``False`` if it was skipped
        because it already exists.

    Raises
    ------
    requests.HTTPError
        If all retry attempts fail.
    """
    if dest.exists():
        logger.info("Skipping (exists): %s", dest.name)
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                total_mb = total / 1e6 if total else 0
                logger.info(
                    "Downloading: %s (%.1f MB)", dest.name, total_mb
                )
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=_CHUNK_SIZE):
                        fh.write(chunk)
            return True
        except requests.RequestException as exc:
            last_exc = exc
            logger.warning(
                "Download attempt %d/%d failed for %s: %s",
                attempt, retries, dest.name, exc,
            )
            # Clean up partial file
            if dest.exists():
                dest.unlink()

    raise last_exc  # type: ignore[misc]
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_gfv11.py -v`
Expected: PASS (7 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/gfv11.py tests/test_gfv11.py
git commit -m "feat(gfv11): add download_file() with streaming and retry"
```

---

### Task 4: Add download_item() orchestrator with unzip

**Files:**
- Modify: `src/hydro_param/gfv11.py`
- Test: `tests/test_gfv11.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_gfv11.py
import zipfile

from hydro_param.gfv11 import download_item, FILE_DIRECTORY_MAP


def test_download_item_organizes_and_unzips(tmp_path: Path):
    """download_item should download, unzip, and organise files."""
    # Create a real zip file for testing
    zip_content = b"fake tiff data"
    zip_path = tmp_path / "staging" / "Sand.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Sand.tif", zip_content)

    # Mock fetch_item_files to return just Sand.zip
    mock_files = [("Sand.zip", "https://example.com/Sand.zip", 1000)]

    output_dir = tmp_path / "output"

    def fake_download(url: str, dest: Path, **kwargs) -> bool:
        # Copy the pre-made zip to where download_item expects it
        import shutil
        shutil.copy(zip_path, dest)
        return True

    with patch("hydro_param.gfv11.fetch_item_files", return_value=mock_files), \
         patch("hydro_param.gfv11.download_file", side_effect=fake_download):
        download_item("fake-id", output_dir)

    # Verify the tif was extracted to the correct subdirectory
    extracted = output_dir / "soils" / "Sand.tif"
    assert extracted.exists()
    assert extracted.read_bytes() == zip_content


def test_download_item_skips_unmapped_files(tmp_path: Path):
    """Files not in FILE_DIRECTORY_MAP should be skipped with a warning."""
    mock_files = [("unknown_file.dat", "https://example.com/unknown.dat", 100)]
    output_dir = tmp_path / "output"

    with patch("hydro_param.gfv11.fetch_item_files", return_value=mock_files), \
         patch("hydro_param.gfv11.download_file") as mock_dl:
        download_item("fake-id", output_dir)

    mock_dl.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_gfv11.py::test_download_item_organizes_and_unzips tests/test_gfv11.py::test_download_item_skips_unmapped_files -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/hydro_param/gfv11.py`:

```python
import zipfile


def download_item(item_id: str, output_dir: Path) -> None:
    """Download and organise all files from a ScienceBase item.

    Query the ScienceBase API for the item's file list, download each
    mapped file, unzip archives into thematic subdirectories, and clean
    up the zip files.

    Parameters
    ----------
    item_id
        ScienceBase item identifier.
    output_dir
        Root output directory.  Files are organised into subdirectories
        (``soils/``, ``land_cover/``, ``topo/``, etc.) based on
        ``FILE_DIRECTORY_MAP``.

    Notes
    -----
    Files not present in ``FILE_DIRECTORY_MAP`` are skipped with a
    warning.  Zip files are deleted after successful extraction.
    Files that already exist in the target directory are skipped.
    """
    files = fetch_item_files(item_id)
    logger.info("Found %d files in ScienceBase item %s", len(files), item_id)

    for name, url, size in files:
        subdir = FILE_DIRECTORY_MAP.get(name)
        if subdir is None:
            logger.warning("Skipping unmapped file: %s", name)
            continue

        target_dir = output_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)

        dest = target_dir / name
        downloaded = download_file(url, dest)

        if downloaded and name.endswith(".zip"):
            _unzip_and_clean(dest, target_dir)


def _unzip_and_clean(zip_path: Path, extract_dir: Path) -> None:
    """Extract a zip archive and remove the zip file.

    Parameters
    ----------
    zip_path
        Path to the zip file.
    extract_dir
        Directory to extract contents into.

    Notes
    -----
    If extraction fails, the zip file is preserved and a warning is
    logged.
    """
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("Extracted: %s → %s", zip_path.name, extract_dir)
        zip_path.unlink()
    except (zipfile.BadZipFile, OSError) as exc:
        logger.warning(
            "Failed to extract %s: %s (zip preserved)", zip_path.name, exc
        )
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_gfv11.py -v`
Expected: PASS (9 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/gfv11.py tests/test_gfv11.py
git commit -m "feat(gfv11): add download_item() orchestrator with unzip"
```

---

### Task 5: Add download_gfv11() top-level function

**Files:**
- Modify: `src/hydro_param/gfv11.py`
- Test: `tests/test_gfv11.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_gfv11.py
from hydro_param.gfv11 import download_gfv11, DATA_LAYERS_ITEM_ID, TGF_TOPO_ITEM_ID


def test_download_gfv11_all_calls_both_items(tmp_path: Path):
    """download_gfv11 with items='all' should call download_item twice."""
    with patch("hydro_param.gfv11.download_item") as mock_dl:
        download_gfv11(tmp_path, items="all")

    assert mock_dl.call_count == 2
    mock_dl.assert_any_call(DATA_LAYERS_ITEM_ID, tmp_path)
    mock_dl.assert_any_call(TGF_TOPO_ITEM_ID, tmp_path)


def test_download_gfv11_data_layers_only(tmp_path: Path):
    """download_gfv11 with items='data-layers' should call only Data Layers."""
    with patch("hydro_param.gfv11.download_item") as mock_dl:
        download_gfv11(tmp_path, items="data-layers")

    mock_dl.assert_called_once_with(DATA_LAYERS_ITEM_ID, tmp_path)


def test_download_gfv11_tgf_topo_only(tmp_path: Path):
    """download_gfv11 with items='tgf-topo' should call only TGF Topo."""
    with patch("hydro_param.gfv11.download_item") as mock_dl:
        download_gfv11(tmp_path, items="tgf-topo")

    mock_dl.assert_called_once_with(TGF_TOPO_ITEM_ID, tmp_path)
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_gfv11.py::test_download_gfv11_all_calls_both_items -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Add to `src/hydro_param/gfv11.py`:

```python
from typing import Literal


def download_gfv11(
    output_dir: Path,
    *,
    items: Literal["all", "data-layers", "tgf-topo"] = "all",
) -> None:
    """Download GFv1.1 rasters from ScienceBase.

    Parameters
    ----------
    output_dir
        Root directory for downloaded data.  Files are organised into
        subdirectories (``soils/``, ``land_cover/``, ``topo/``, etc.).
    items
        Which ScienceBase item(s) to download:
        - ``"all"`` — both Data Layers and TGF Topographic Derivatives
        - ``"data-layers"`` — soils, land cover, water bodies, geology
        - ``"tgf-topo"`` — DEM, slope, aspect, TWI, flow direction

    See Also
    --------
    docs/reference/gfv11_raster_provenance.md : Full provenance guide.
    """
    output_dir = Path(output_dir)

    if items in ("all", "data-layers"):
        logger.info("Downloading Data Layers (NHM v1.1)...")
        download_item(DATA_LAYERS_ITEM_ID, output_dir)

    if items in ("all", "tgf-topo"):
        logger.info("Downloading TGF Topographic Derivatives...")
        download_item(TGF_TOPO_ITEM_ID, output_dir)

    logger.info("Download complete: %s", output_dir)
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_gfv11.py -v`
Expected: PASS (12 tests)

**Step 5: Commit**

```bash
git add src/hydro_param/gfv11.py tests/test_gfv11.py
git commit -m "feat(gfv11): add download_gfv11() top-level entry point"
```

---

### Task 6: Wire into CLI as `hydro-param gfv11 download`

**Files:**
- Modify: `src/hydro_param/cli.py`
- Test: `tests/test_cli.py` (or manual verification)

**Step 1: Write the failing test**

```python
# Add to tests/test_gfv11.py
from hydro_param.cli import app


def test_gfv11_download_cli_exists():
    """The gfv11 download command should be registered in the CLI."""
    # cyclopts introspection — verify the command group exists
    command_names = [cmd.name[0] for cmd in app._commands.values()]
    assert "gfv11" in command_names
```

**Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_gfv11.py::test_gfv11_download_cli_exists -v`
Expected: FAIL with `AssertionError` (gfv11 not in commands)

**Step 3: Write minimal implementation**

Add to `src/hydro_param/cli.py`:

After line 58 (`pws_app = ...`), add:

```python
gfv11_app = app.command(App(name="gfv11", help="GFv1.1 NHM data layer utilities."))
```

Then add the command function before the `# Entry point` section:

```python
# ---------------------------------------------------------------------------
# gfv11 download
# ---------------------------------------------------------------------------


@gfv11_app.command(name="download")
def gfv11_download_cmd(
    output_dir: Path,
    *,
    items: str = "all",
) -> None:
    """Download GFv1.1 NHM data layer rasters from ScienceBase.

    Fetch pre-built CONUS-wide rasters from two ScienceBase items into
    a shared data directory organised by thematic category (soils,
    land_cover, topo, etc.).

    Total download size is approximately 15 GB.

    Parameters
    ----------
    output_dir
        Root directory for downloaded data.  Subdirectories (``soils/``,
        ``land_cover/``, ``topo/``, ``water_bodies/``, ``geology/``,
        ``metadata/``) are created automatically.
    items
        Which ScienceBase item(s) to download.  One of ``"all"``
        (default), ``"data-layers"`` (soils + land cover + misc),
        or ``"tgf-topo"`` (DEM, slope, aspect, TWI).

    Raises
    ------
    SystemExit
        If downloads fail after retries (exit code 1).

    See Also
    --------
    docs/reference/gfv11_raster_provenance.md : Full provenance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from hydro_param.gfv11 import download_gfv11

    valid_items = ("all", "data-layers", "tgf-topo")
    if items not in valid_items:
        print(f"Error: --items must be one of {valid_items}", file=sys.stderr)
        raise SystemExit(1)

    try:
        download_gfv11(output_dir, items=items)  # type: ignore[arg-type]
    except Exception as exc:
        logger.exception("Download failed.")
        raise SystemExit(1) from exc
```

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_gfv11.py -v`
Expected: PASS (13 tests)

**Step 5: Verify CLI help works**

Run: `pixi run -e dev python -m hydro_param --help`
Expected: Shows `gfv11` in command list

Run: `pixi run -e dev python -m hydro_param gfv11 download --help`
Expected: Shows `output-dir` and `--items` parameters

**Step 6: Commit**

```bash
git add src/hydro_param/cli.py tests/test_gfv11.py
git commit -m "feat(gfv11): wire download command into CLI"
```

---

### Task 7: Run all checks and final commit

**Step 1: Run full test suite**

Run: `pixi run -e dev check`
Expected: All checks pass (lint, format, typecheck, tests)

**Step 2: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: All hooks pass

**Step 3: Fix any issues found in Steps 1-2**

If ruff/mypy/tests report issues, fix them and re-run.

**Step 4: Final commit if any fixes were needed**

```bash
git add -u
git commit -m "fix(gfv11): address lint/type issues from checks"
```
