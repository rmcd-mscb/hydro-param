"""Tests for hydro_param.gfv11 — GFv1.1 ScienceBase download utilities."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hydro_param.gfv11 import (
    DATA_LAYERS_ITEM_ID,
    FILE_DIRECTORY_MAP,
    SB_API_URL,
    TGF_TOPO_ITEM_ID,
    _unzip_and_clean,
    download_file,
    download_gfv11,
    download_item,
    fetch_item_files,
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
    def test_retries_on_failure(
        self, mock_get: MagicMock, mock_sleep: MagicMock, tmp_path: Path
    ) -> None:
        import requests as req

        mock_get.side_effect = req.ConnectionError("network error")

        dest = tmp_path / "fail.zip"
        result = download_file("https://example.com/fail.zip", dest, retries=2)

        assert result is False
        assert not dest.exists()
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(2)  # 2^1 = 2 on first retry


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

        _unzip_and_clean(zip_path, tmp_path)

        assert not zip_path.exists()
        assert (tmp_path / "inner.txt").exists()
        assert (tmp_path / "inner.txt").read_text() == "contents"

    def test_preserves_bad_zip(self, tmp_path: Path) -> None:
        # Create a file that is NOT a valid zip
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"not a zip at all")

        _unzip_and_clean(bad_zip, tmp_path)

        assert bad_zip.exists()  # preserved on failure


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

        download_item("fake-id", tmp_path)

        mock_dl.assert_any_call("https://example.com/dem.zip", tmp_path / "topo" / "dem.zip")
        mock_dl.assert_any_call("https://example.com/Clay.zip", tmp_path / "soils" / "Clay.zip")
        assert mock_dl.call_count == 2

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

        download_item("fake-id", tmp_path)

        mock_dl.assert_not_called()

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

        download_item("fake-id", tmp_path)

        mock_unzip.assert_not_called()


# ---------------------------------------------------------------------------
# download_gfv11
# ---------------------------------------------------------------------------


class TestDownloadGfv11:
    """Tests for download_gfv11() dispatch logic."""

    @patch("hydro_param.gfv11.download_item")
    def test_all_downloads_both_items(self, mock_di: MagicMock, tmp_path: Path) -> None:
        download_gfv11(tmp_path, items="all")

        assert mock_di.call_count == 2
        mock_di.assert_any_call(DATA_LAYERS_ITEM_ID, tmp_path)
        mock_di.assert_any_call(TGF_TOPO_ITEM_ID, tmp_path)

    @patch("hydro_param.gfv11.download_item")
    def test_data_layers_only(self, mock_di: MagicMock, tmp_path: Path) -> None:
        download_gfv11(tmp_path, items="data-layers")

        mock_di.assert_called_once_with(DATA_LAYERS_ITEM_ID, tmp_path)

    @patch("hydro_param.gfv11.download_item")
    def test_tgf_topo_only(self, mock_di: MagicMock, tmp_path: Path) -> None:
        download_gfv11(tmp_path, items="tgf-topo")

        mock_di.assert_called_once_with(TGF_TOPO_ITEM_ID, tmp_path)

    @patch("hydro_param.gfv11.download_item")
    def test_default_is_all(self, mock_di: MagicMock, tmp_path: Path) -> None:
        download_gfv11(tmp_path)

        assert mock_di.call_count == 2

    @patch("hydro_param.gfv11.download_item")
    def test_creates_output_dir(self, mock_di: MagicMock, tmp_path: Path) -> None:
        new_dir = tmp_path / "nested" / "output"
        download_gfv11(new_dir)

        assert new_dir.exists()
