"""Tests for hydro_param.gfv11 — GFv1.1 ScienceBase download utilities."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from hydro_param.gfv11 import (
    DATA_LAYERS_ITEM_ID,
    FILE_DIRECTORY_MAP,
    GFV11_DATASETS,
    SB_API_URL,
    TGF_TOPO_ITEM_ID,
    DownloadError,
    DownloadSummary,
    _unzip_and_clean,
    download_file,
    download_gfv11,
    download_item,
    fetch_item_files,
    write_registry_overlay,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants are correct."""

    def test_data_layers_item_id(self) -> None:
        assert DATA_LAYERS_ITEM_ID == "5ebb182b82ce25b5136181cf"

    def test_tgf_topo_item_id(self) -> None:
        assert TGF_TOPO_ITEM_ID == "5ebb17d082ce25b5136181cb"

    def test_sb_api_url(self) -> None:
        assert SB_API_URL == "https://www.sciencebase.gov/catalog/item"

    def test_file_directory_map_not_empty(self) -> None:
        assert len(FILE_DIRECTORY_MAP) > 0

    def test_file_directory_map_expected_subdirs(self) -> None:
        expected = {"soils", "land_cover", "water_bodies", "geology", "metadata", "topo"}
        actual = set(FILE_DIRECTORY_MAP.values())
        assert actual == expected

    @pytest.mark.parametrize(
        "filename, expected_subdir",
        [
            ("TEXT_PRMS.zip", "soils"),
            ("Clay.zip", "soils"),
            ("LULC.zip", "land_cover"),
            ("Imperv.zip", "land_cover"),
            ("wbg.zip", "water_bodies"),
            ("Lithology_exp_Konly_Project.zip", "geology"),
            ("CrossWalk.xlsx", "metadata"),
            ("SDC_table.csv", "metadata"),
            ("dem.zip", "topo"),
            ("slope100X.zip", "topo"),
        ],
    )
    def test_file_directory_map_spot_checks(self, filename: str, expected_subdir: str) -> None:
        assert FILE_DIRECTORY_MAP[filename] == expected_subdir

    def test_soils_files_count(self) -> None:
        soils = [k for k, v in FILE_DIRECTORY_MAP.items() if v == "soils"]
        assert len(soils) == 5

    def test_topo_files_count(self) -> None:
        topo = [k for k, v in FILE_DIRECTORY_MAP.items() if v == "topo"]
        assert len(topo) == 5

    def test_land_cover_files_count(self) -> None:
        lc = [k for k, v in FILE_DIRECTORY_MAP.items() if v == "land_cover"]
        assert len(lc) == 10


# ---------------------------------------------------------------------------
# GFV11_DATASETS metadata
# ---------------------------------------------------------------------------


class TestGfv11Datasets:
    """Verify GFV11_DATASETS metadata dict is complete and well-formed."""

    def test_total_count(self) -> None:
        """All 21 GFv1.1 rasters are registered."""
        assert len(GFV11_DATASETS) == 21

    def test_required_keys(self) -> None:
        """Every entry has the required metadata keys."""
        required = {"description", "category", "filename", "subdir", "variables"}
        for name, meta in GFV11_DATASETS.items():
            assert required.issubset(meta.keys()), f"{name} missing keys: {required - meta.keys()}"

    def test_valid_categories(self) -> None:
        """All categories are from the expected set."""
        valid = {"soils", "land_cover", "water_bodies", "topography", "snow"}
        for name, meta in GFV11_DATASETS.items():
            assert meta["category"] in valid, f"{name} has unexpected category: {meta['category']}"

    def test_category_counts(self) -> None:
        """Category counts match the expected distribution."""
        cats = [m["category"] for m in GFV11_DATASETS.values()]
        assert cats.count("soils") == 5
        assert cats.count("land_cover") == 9
        assert cats.count("water_bodies") == 1
        assert cats.count("topography") == 5
        assert cats.count("snow") == 1

    def test_variables_nonempty(self) -> None:
        """Every dataset has at least one variable."""
        for name, meta in GFV11_DATASETS.items():
            assert len(meta["variables"]) >= 1, f"{name} has no variables"

    def test_variable_required_fields(self) -> None:
        """Every variable has the required fields."""
        required = {"name", "band", "units", "long_name", "native_name", "categorical"}
        for name, meta in GFV11_DATASETS.items():
            for var in meta["variables"]:
                assert required.issubset(var.keys()), (
                    f"{name}/{var.get('name', '?')} missing: {required - var.keys()}"
                )

    def test_filenames_in_file_directory_map(self) -> None:
        """Every filename maps to an entry in FILE_DIRECTORY_MAP."""
        for name, meta in GFV11_DATASETS.items():
            # Zip filename: replace .tif with .zip
            zip_name = meta["filename"].replace(".tif", ".zip")
            assert zip_name in FILE_DIRECTORY_MAP, f"{name}: {zip_name} not in FILE_DIRECTORY_MAP"

    def test_scale_factor_on_encoded_rasters(self) -> None:
        """Integer-encoded rasters (slope, aspect, twi) have scale_factor=0.01."""
        encoded = {"gfv11_slope", "gfv11_aspect", "gfv11_twi"}
        for name in encoded:
            var = GFV11_DATASETS[name]["variables"][0]
            assert var.get("scale_factor") == 0.01, f"{name} missing scale_factor=0.01"

    def test_no_scale_factor_on_other_rasters(self) -> None:
        """Non-encoded rasters do not have scale_factor."""
        encoded = {"gfv11_slope", "gfv11_aspect", "gfv11_twi"}
        for name, meta in GFV11_DATASETS.items():
            if name not in encoded:
                var = meta["variables"][0]
                assert "scale_factor" not in var, f"{name} should not have scale_factor"

    def test_categorical_datasets(self) -> None:
        """Categorical flag is set on the correct datasets."""
        expected_categorical = {
            "gfv11_text_prms",
            "gfv11_lulc",
            "gfv11_wbg",
            "gfv11_fdr",
            "gfv11_cv_int",
        }
        for name, meta in GFV11_DATASETS.items():
            is_cat = meta["variables"][0]["categorical"]
            if name in expected_categorical:
                assert is_cat, f"{name} should be categorical"
            else:
                assert not is_cat, f"{name} should not be categorical"

    def test_cv_int_registered_correctly(self) -> None:
        """CV_INT.tif is registered as gfv11_cv_int (categorical snow CV), not gfv11_covden_sum."""
        assert "gfv11_cv_int" in GFV11_DATASETS
        assert "gfv11_covden_sum" not in GFV11_DATASETS
        entry = GFV11_DATASETS["gfv11_cv_int"]
        assert entry["filename"] == "CV_INT.tif"
        assert entry["variables"][0]["categorical"] is True
        assert entry["variables"][0]["name"] == "cv_int"
        assert entry["category"] == "snow"


# ---------------------------------------------------------------------------
# DownloadSummary
# ---------------------------------------------------------------------------


class TestDownloadSummary:
    """Tests for DownloadSummary dataclass."""

    def test_has_failures_false_when_clean(self) -> None:
        s = DownloadSummary(downloaded=["a.zip"], skipped=["b.zip"])
        assert s.has_failures is False

    def test_has_failures_true_with_download_failure(self) -> None:
        s = DownloadSummary(failed=["a.zip"])
        assert s.has_failures is True

    def test_has_failures_true_with_extract_failure(self) -> None:
        s = DownloadSummary(extract_failed=["a.zip"])
        assert s.has_failures is True


# ---------------------------------------------------------------------------
# fetch_item_files
# ---------------------------------------------------------------------------


class TestFetchItemFiles:
    """Tests for fetch_item_files()."""

    def _mock_response(self, files: list[dict]) -> MagicMock:
        """Create a mock requests.Response with the given files payload."""
        resp = MagicMock()
        resp.json.return_value = {"files": files}
        resp.raise_for_status.return_value = None
        return resp

    @patch("hydro_param.gfv11.requests.get")
    def test_returns_correct_tuples(self, mock_get: MagicMock) -> None:
        mock_get.return_value = self._mock_response(
            [
                {"name": "dem.zip", "url": "https://example.com/dem.zip", "size": 100},
                {"name": "Clay.zip", "url": "https://example.com/Clay.zip", "size": 200},
            ]
        )
        result = fetch_item_files("fake-id")
        assert result == [
            ("dem.zip", "https://example.com/dem.zip", 100),
            ("Clay.zip", "https://example.com/Clay.zip", 200),
        ]
        mock_get.assert_called_once_with(f"{SB_API_URL}/fake-id?format=json", timeout=30)

    @patch("hydro_param.gfv11.requests.get")
    def test_skips_files_without_url(self, mock_get: MagicMock) -> None:
        mock_get.return_value = self._mock_response(
            [
                {"name": "dem.zip", "url": "https://example.com/dem.zip", "size": 100},
                {"name": "nourl.zip", "url": "", "size": 50},
                {"name": "missing.zip", "size": 30},
            ]
        )
        result = fetch_item_files("fake-id")
        assert len(result) == 1
        assert result[0][0] == "dem.zip"

    @patch("hydro_param.gfv11.requests.get")
    def test_empty_files_list(self, mock_get: MagicMock) -> None:
        mock_get.return_value = self._mock_response([])
        result = fetch_item_files("fake-id")
        assert result == []

    @patch("hydro_param.gfv11.requests.get")
    def test_missing_size_defaults_to_zero(self, mock_get: MagicMock) -> None:
        mock_get.return_value = self._mock_response(
            [{"name": "f.zip", "url": "https://example.com/f.zip"}]
        )
        result = fetch_item_files("fake-id")
        assert result[0][2] == 0

    @patch("hydro_param.gfv11.requests.get")
    def test_no_files_key_returns_empty(self, mock_get: MagicMock) -> None:
        """Returns empty list when API response has no 'files' key."""
        resp = MagicMock()
        resp.json.return_value = {"title": "Some item"}
        resp.raise_for_status.return_value = None
        mock_get.return_value = resp
        result = fetch_item_files("fake-id")
        assert result == []

    @patch("hydro_param.gfv11.requests.get")
    def test_raises_on_http_error_after_retries(self, mock_get: MagicMock) -> None:
        """HTTP errors from ScienceBase API propagate after exhausting retries."""
        import requests as req

        resp = MagicMock()
        resp.raise_for_status.side_effect = req.HTTPError("404 Not Found")
        mock_get.return_value = resp

        with pytest.raises(req.HTTPError):
            fetch_item_files("fake-id", retries=1)

    @patch("hydro_param.gfv11.time.sleep")
    @patch("hydro_param.gfv11.requests.get")
    def test_retries_on_connection_error(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        """Transient connection errors are retried before giving up."""
        import requests as req

        # Fail twice, succeed on third attempt
        good_resp = self._mock_response(
            [{"name": "dem.zip", "url": "https://example.com/dem.zip", "size": 100}]
        )
        mock_get.side_effect = [
            req.ConnectionError("Network unreachable"),
            req.ConnectionError("Network unreachable"),
            good_resp,
        ]
        result = fetch_item_files("fake-id", retries=3)
        assert len(result) == 1
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("hydro_param.gfv11.time.sleep")
    @patch("hydro_param.gfv11.requests.get")
    def test_raises_after_all_retries_exhausted(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """ConnectionError raised after all retries exhausted."""
        import requests as req

        mock_get.side_effect = req.ConnectionError("Network unreachable")
        with pytest.raises(req.ConnectionError, match="Network unreachable"):
            fetch_item_files("fake-id", retries=2)
        assert mock_get.call_count == 2

    @patch("hydro_param.gfv11.requests.get")
    def test_raises_on_non_json_response(self, mock_get: MagicMock) -> None:
        """Non-JSON responses raise ValueError with descriptive message."""
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.side_effect = ValueError("No JSON")
        mock_get.return_value = resp

        with pytest.raises(ValueError, match="non-JSON response"):
            fetch_item_files("fake-id")


# ---------------------------------------------------------------------------
# download_file
# ---------------------------------------------------------------------------


class TestDownloadFile:
    """Tests for download_file()."""

    def test_skips_existing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "existing.zip"
        dest.write_text("data")
        result = download_file("https://example.com/existing.zip", dest)
        assert result is False

    @patch("hydro_param.gfv11.requests.get")
    def test_streams_content_to_disk(self, mock_get: MagicMock, tmp_path: Path) -> None:
        # Set up mock streaming response
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {"content-length": "1024"}
        mock_resp.iter_content.return_value = [b"hello", b"world"]
        mock_get.return_value = mock_resp

        dest = tmp_path / "subdir" / "new_file.tif"
        result = download_file("https://example.com/new_file.tif", dest)

        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"helloworld"
        mock_get.assert_called_once_with(
            "https://example.com/new_file.tif", stream=True, timeout=60
        )

    @patch("hydro_param.gfv11.requests.get")
    def test_creates_parent_directories(self, mock_get: MagicMock, tmp_path: Path) -> None:
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {}
        mock_resp.iter_content.return_value = [b"data"]
        mock_get.return_value = mock_resp

        dest = tmp_path / "a" / "b" / "c" / "file.dat"
        download_file("https://example.com/file.dat", dest)
        assert dest.parent.exists()

    @patch("hydro_param.gfv11.time.sleep")
    @patch("hydro_param.gfv11.requests.get")
    def test_raises_after_exhausting_retries(
        self, mock_get: MagicMock, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        import requests as req

        mock_get.side_effect = req.ConnectionError("network error")

        dest = tmp_path / "fail.zip"
        with pytest.raises(DownloadError, match="after 2 attempts"):
            download_file("https://example.com/fail.zip", dest, retries=2)

        assert not dest.exists()
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1 = 2 on first retry

    @patch("hydro_param.gfv11.time.sleep")
    @patch("hydro_param.gfv11.requests.get")
    def test_retry_succeeds_on_second_attempt(
        self, mock_get: MagicMock, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        """Download succeeds after first attempt fails."""
        import requests as req

        # Second call succeeds
        mock_resp = MagicMock()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.raise_for_status.return_value = None
        mock_resp.headers = {}
        mock_resp.iter_content.return_value = [b"recovered"]

        mock_get.side_effect = [req.ConnectionError("fail"), mock_resp]

        dest = tmp_path / "retry_ok.zip"
        result = download_file("https://example.com/retry.zip", dest, retries=3)

        assert result is True
        assert dest.read_bytes() == b"recovered"


# ---------------------------------------------------------------------------
# _unzip_and_clean
# ---------------------------------------------------------------------------


class TestUnzipAndClean:
    """Tests for _unzip_and_clean()."""

    def test_extracts_and_deletes_zip(self, tmp_path: Path) -> None:
        # Create a real zip file
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner.txt", "contents")

        result = _unzip_and_clean(zip_path, tmp_path)

        assert result is True
        assert not zip_path.exists()
        assert (tmp_path / "inner.txt").exists()
        assert (tmp_path / "inner.txt").read_text() == "contents"

    def test_preserves_bad_zip(self, tmp_path: Path) -> None:
        # Create a file that is NOT a valid zip
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"not a zip at all")

        result = _unzip_and_clean(bad_zip, tmp_path)

        assert result is False
        assert bad_zip.exists()  # preserved on failure

    def test_rejects_zip_slip(self, tmp_path: Path) -> None:
        """Archives with path traversal entries are rejected."""
        zip_path = tmp_path / "evil.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("../../etc/passwd", "pwned")

        result = _unzip_and_clean(zip_path, tmp_path)

        assert result is False
        assert zip_path.exists()  # preserved, not extracted


# ---------------------------------------------------------------------------
# download_item
# ---------------------------------------------------------------------------


class TestDownloadItem:
    """Tests for download_item()."""

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_organizes_files_into_subdirs(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
            ("Clay.zip", "https://example.com/Clay.zip", 200),
        ]
        mock_dl.return_value = True  # simulate successful download
        mock_unzip.return_value = True

        summary = download_item("fake-id", tmp_path)

        mock_dl.assert_any_call("https://example.com/dem.zip", tmp_path / "topo" / "dem.zip")
        mock_dl.assert_any_call("https://example.com/Clay.zip", tmp_path / "soils" / "Clay.zip")
        assert mock_dl.call_count == 2
        assert len(summary.downloaded) == 2

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_unzips_downloaded_zips(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
        ]
        mock_dl.return_value = True
        mock_unzip.return_value = True

        download_item("fake-id", tmp_path)

        mock_unzip.assert_called_once_with(tmp_path / "topo" / "dem.zip", tmp_path / "topo")

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_does_not_unzip_non_zip(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            ("CrossWalk.xlsx", "https://example.com/CrossWalk.xlsx", 50),
        ]
        mock_dl.return_value = True

        download_item("fake-id", tmp_path)

        mock_unzip.assert_not_called()

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_skips_unmapped_files(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            ("unknown_file.dat", "https://example.com/unknown_file.dat", 10),
        ]

        summary = download_item("fake-id", tmp_path)

        mock_dl.assert_not_called()
        assert summary.unmapped == ["unknown_file.dat"]

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_no_unzip_when_download_skipped(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
        ]
        mock_dl.return_value = False  # file already existed

        summary = download_item("fake-id", tmp_path)

        mock_unzip.assert_not_called()
        assert summary.skipped == ["dem.zip"]

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_accumulates_download_failures(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Failed downloads are accumulated, not raised immediately."""
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
            ("Clay.zip", "https://example.com/Clay.zip", 200),
        ]
        mock_dl.side_effect = [DownloadError("fail"), True]
        mock_unzip.return_value = True

        summary = download_item("fake-id", tmp_path)

        assert summary.failed == ["dem.zip"]
        assert summary.downloaded == ["Clay.zip"]

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_tracks_extraction_failures(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Failed extractions are tracked in the summary."""
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
        ]
        mock_dl.return_value = True
        mock_unzip.return_value = False  # extraction failed

        summary = download_item("fake-id", tmp_path)

        assert summary.downloaded == ["dem.zip"]
        assert summary.extract_failed == ["dem.zip"]

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_creates_marker_after_successful_extraction(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A .zip.done marker is created after successful zip extraction."""
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
        ]
        mock_dl.return_value = True
        mock_unzip.return_value = True
        # download_file is mocked, so create the parent directory manually
        (tmp_path / "topo").mkdir(parents=True)

        download_item("fake-id", tmp_path)

        marker = tmp_path / "topo" / "dem.zip.done"
        assert marker.exists()

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_no_marker_after_failed_extraction(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No marker is created when zip extraction fails."""
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
        ]
        mock_dl.return_value = True
        mock_unzip.return_value = False

        download_item("fake-id", tmp_path)

        marker = tmp_path / "topo" / "dem.zip.done"
        assert not marker.exists()

    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_skips_download_when_marker_exists(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Files with an existing .zip.done marker are skipped entirely."""
        mock_fetch.return_value = [
            ("dem.zip", "https://example.com/dem.zip", 100),
        ]
        # Create the marker file
        marker_dir = tmp_path / "topo"
        marker_dir.mkdir(parents=True)
        (marker_dir / "dem.zip.done").touch()

        summary = download_item("fake-id", tmp_path)

        mock_dl.assert_not_called()
        assert summary.skipped == ["dem.zip"]

    @patch("hydro_param.gfv11._unzip_and_clean")
    @patch("hydro_param.gfv11.download_file")
    @patch("hydro_param.gfv11.fetch_item_files")
    def test_no_marker_for_non_zip_files(
        self,
        mock_fetch: MagicMock,
        mock_dl: MagicMock,
        mock_unzip: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Non-zip files do not produce marker files."""
        mock_fetch.return_value = [
            ("CrossWalk.xlsx", "https://example.com/CrossWalk.xlsx", 50),
        ]
        mock_dl.return_value = True

        download_item("fake-id", tmp_path)

        # No .done marker for non-zip files
        assert not (tmp_path / "metadata" / "CrossWalk.xlsx.done").exists()


# ---------------------------------------------------------------------------
# download_gfv11
# ---------------------------------------------------------------------------


class TestWriteRegistryOverlay:
    """Tests for auto-registration of GFv1.1 datasets."""

    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        overlay_path = tmp_path / "gfv11.yml"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        write_registry_overlay(data_dir, overlay_path)

        assert overlay_path.exists()
        raw = yaml.safe_load(overlay_path.read_text())
        assert "datasets" in raw
        assert len(raw["datasets"]) == 21

    def test_source_paths_are_absolute(self, tmp_path: Path) -> None:
        overlay_path = tmp_path / "gfv11.yml"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        write_registry_overlay(data_dir, overlay_path)

        raw = yaml.safe_load(overlay_path.read_text())
        for name, entry in raw["datasets"].items():
            source = entry.get("source", "")
            assert Path(source).is_absolute(), f"{name}: source is not absolute: {source}"

    def test_source_paths_match_subdir_structure(self, tmp_path: Path) -> None:
        overlay_path = tmp_path / "gfv11.yml"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        write_registry_overlay(data_dir, overlay_path)

        raw = yaml.safe_load(overlay_path.read_text())
        sand = raw["datasets"]["gfv11_sand"]
        expected = str(data_dir.resolve() / "soils" / "Sand.tif")
        assert sand["source"] == expected

    def test_all_entries_have_strategy_local_tiff(self, tmp_path: Path) -> None:
        overlay_path = tmp_path / "gfv11.yml"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        write_registry_overlay(data_dir, overlay_path)

        raw = yaml.safe_load(overlay_path.read_text())
        for name, entry in raw["datasets"].items():
            assert entry["strategy"] == "local_tiff", f"{name}: strategy != local_tiff"

    def test_entries_parseable_as_dataset_entries(self, tmp_path: Path) -> None:
        """Generated YAML can be loaded by the registry loader."""
        overlay_path = tmp_path / "gfv11.yml"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        write_registry_overlay(data_dir, overlay_path)

        from hydro_param.dataset_registry import load_registry

        registry = load_registry(overlay_path)
        assert len(registry.datasets) == 21

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        overlay_path = tmp_path / "nested" / "dir" / "gfv11.yml"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        write_registry_overlay(data_dir, overlay_path)

        assert overlay_path.exists()

    def test_scale_factor_preserved_in_yaml(self, tmp_path: Path) -> None:
        overlay_path = tmp_path / "gfv11.yml"
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        write_registry_overlay(data_dir, overlay_path)

        raw = yaml.safe_load(overlay_path.read_text())
        slope_vars = raw["datasets"]["gfv11_slope"]["variables"]
        assert slope_vars[0]["scale_factor"] == 0.01


# ---------------------------------------------------------------------------
# download_gfv11
# ---------------------------------------------------------------------------


class TestDownloadGfv11:
    """Tests for download_gfv11() dispatch logic."""

    @patch("hydro_param.gfv11.download_item")
    def test_all_downloads_both_items(self, mock_di: MagicMock, tmp_path: Path) -> None:
        mock_di.return_value = DownloadSummary()
        summary = download_gfv11(tmp_path, items="all")

        assert mock_di.call_count == 2
        mock_di.assert_any_call(DATA_LAYERS_ITEM_ID, tmp_path)
        mock_di.assert_any_call(TGF_TOPO_ITEM_ID, tmp_path)
        assert not summary.has_failures

    @patch("hydro_param.gfv11.download_item")
    def test_data_layers_only(self, mock_di: MagicMock, tmp_path: Path) -> None:
        mock_di.return_value = DownloadSummary()
        download_gfv11(tmp_path, items="data-layers")

        mock_di.assert_called_once_with(DATA_LAYERS_ITEM_ID, tmp_path)

    @patch("hydro_param.gfv11.download_item")
    def test_tgf_topo_only(self, mock_di: MagicMock, tmp_path: Path) -> None:
        mock_di.return_value = DownloadSummary()
        download_gfv11(tmp_path, items="tgf-topo")

        mock_di.assert_called_once_with(TGF_TOPO_ITEM_ID, tmp_path)

    @patch("hydro_param.gfv11.download_item")
    def test_default_is_all(self, mock_di: MagicMock, tmp_path: Path) -> None:
        mock_di.return_value = DownloadSummary()
        download_gfv11(tmp_path)

        assert mock_di.call_count == 2

    @patch("hydro_param.gfv11.download_item")
    def test_creates_output_dir(self, mock_di: MagicMock, tmp_path: Path) -> None:
        mock_di.return_value = DownloadSummary()
        new_dir = tmp_path / "nested" / "output"
        download_gfv11(new_dir)

        assert new_dir.exists()

    @patch("hydro_param.gfv11.download_item")
    def test_raises_on_failures(self, mock_di: MagicMock, tmp_path: Path) -> None:
        """DownloadError raised when any files failed."""
        mock_di.return_value = DownloadSummary(failed=["dem.zip"])

        with pytest.raises(DownloadError, match="1 download"):
            download_gfv11(tmp_path, items="data-layers")

    @patch("hydro_param.gfv11.download_item")
    def test_raises_on_extract_failures(self, mock_di: MagicMock, tmp_path: Path) -> None:
        """DownloadError raised when extraction fails, but overlay still written."""
        mock_di.return_value = DownloadSummary(downloaded=["dem.zip"], extract_failed=["dem.zip"])
        overlay_path = tmp_path / "overlay" / "gfv11.yml"

        with pytest.raises(DownloadError, match="1 extraction"):
            download_gfv11(tmp_path, items="data-layers", overlay_path=overlay_path)

        # Overlay IS written even with partial failures so that successfully
        # downloaded files are registered (fault-tolerant behavior).
        assert overlay_path.exists()


# ---------------------------------------------------------------------------
# Auto-registration after download
# ---------------------------------------------------------------------------


class TestDownloadGfv11AutoRegistration:
    """Tests for auto-registration after download."""

    @patch("hydro_param.gfv11.download_item")
    def test_writes_overlay_after_successful_download(
        self, mock_di: MagicMock, tmp_path: Path
    ) -> None:
        mock_di.return_value = DownloadSummary(downloaded=["Sand.tif"])
        overlay_path = tmp_path / "overlay" / "gfv11.yml"

        download_gfv11(tmp_path, items="data-layers", overlay_path=overlay_path)

        assert overlay_path.exists()

    @patch("hydro_param.gfv11.download_item")
    def test_writes_overlay_when_all_skipped(self, mock_di: MagicMock, tmp_path: Path) -> None:
        """Overlay is written even when all files were already downloaded."""
        mock_di.return_value = DownloadSummary(skipped=["Sand.tif"])
        overlay_path = tmp_path / "overlay" / "gfv11.yml"

        download_gfv11(tmp_path, items="data-layers", overlay_path=overlay_path)

        assert overlay_path.exists()

    @patch("hydro_param.gfv11.download_item")
    def test_no_overlay_on_total_failure(self, mock_di: MagicMock, tmp_path: Path) -> None:
        """No overlay written when all downloads fail."""
        mock_di.return_value = DownloadSummary(failed=["Sand.tif"], extract_failed=["Clay.tif"])
        overlay_path = tmp_path / "overlay" / "gfv11.yml"

        with pytest.raises(DownloadError):
            download_gfv11(tmp_path, items="data-layers", overlay_path=overlay_path)

        assert not overlay_path.exists()

    @patch("hydro_param.gfv11.download_item")
    def test_no_overlay_when_nothing_happened(self, mock_di: MagicMock, tmp_path: Path) -> None:
        """No overlay when downloaded and skipped are both empty."""
        mock_di.return_value = DownloadSummary()
        overlay_path = tmp_path / "overlay" / "gfv11.yml"

        download_gfv11(tmp_path, items="data-layers", overlay_path=overlay_path)

        assert not overlay_path.exists()

    @patch("hydro_param.gfv11.write_registry_overlay")
    @patch("hydro_param.gfv11.download_item")
    def test_overlay_write_failure_does_not_crash_download(
        self, mock_di: MagicMock, mock_write: MagicMock, tmp_path: Path
    ) -> None:
        """Overlay write failure logs warning but returns summary."""
        mock_di.return_value = DownloadSummary(downloaded=["Sand.tif"])
        mock_write.side_effect = OSError("Permission denied")

        summary = download_gfv11(tmp_path, items="data-layers")

        assert summary.downloaded == ["Sand.tif"]
        assert not summary.has_failures

    @patch("hydro_param.gfv11.download_item")
    def test_api_error_on_one_item_does_not_block_other(
        self, mock_di: MagicMock, tmp_path: Path
    ) -> None:
        """When one ScienceBase item query fails, the other still succeeds."""
        import requests as req

        # First call (data-layers) succeeds with all skipped,
        # second call (tgf-topo) raises ConnectionError.
        mock_di.side_effect = [
            DownloadSummary(skipped=["Sand.tif", "Clay.tif"]),
            req.ConnectionError("Network is unreachable"),
        ]
        overlay_path = tmp_path / "overlay" / "gfv11.yml"

        # Should NOT raise — the error is logged, not fatal
        summary = download_gfv11(tmp_path, items="all", overlay_path=overlay_path)

        assert summary.skipped == ["Sand.tif", "Clay.tif"]
        # Overlay IS written because item 1 had skipped files
        assert overlay_path.exists()

    @patch("hydro_param.gfv11.download_item")
    def test_api_error_on_all_items_no_overlay(self, mock_di: MagicMock, tmp_path: Path) -> None:
        """When all ScienceBase item queries fail, no overlay is written."""
        import requests as req

        mock_di.side_effect = req.ConnectionError("Network is unreachable")
        overlay_path = tmp_path / "overlay" / "gfv11.yml"

        summary = download_gfv11(tmp_path, items="all", overlay_path=overlay_path)

        assert not summary.downloaded
        assert not summary.skipped
        assert not overlay_path.exists()


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


class TestGfv11EndToEnd:
    """Integration test: download -> register -> load."""

    @patch("hydro_param.gfv11.download_item")
    def test_download_then_load_registry(self, mock_di: MagicMock, tmp_path: Path) -> None:
        """Downloaded datasets are visible through registry overlay."""
        mock_di.return_value = DownloadSummary(downloaded=["Sand.tif"])
        overlay_dir = tmp_path / "overlay"
        overlay_path = overlay_dir / "gfv11.yml"

        download_gfv11(tmp_path, items="data-layers", overlay_path=overlay_path)

        from hydro_param.dataset_registry import load_registry
        from hydro_param.pipeline import DEFAULT_REGISTRY

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])

        # GFv1.1 datasets present
        assert "gfv11_sand" in registry.datasets
        assert "gfv11_slope" in registry.datasets
        # Bundled datasets still present
        assert "dem_3dep_10m" in registry.datasets
        # Source path is resolved
        sand = registry.get("gfv11_sand")
        assert sand.source is not None
        assert "soils" in sand.source
        assert "Sand.tif" in sand.source
