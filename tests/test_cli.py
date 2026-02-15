"""Tests for the CLI module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hydro_param.cli import app


def _run(*args: str) -> None:
    """Invoke the CLI app without sys.exit on success."""
    app(list(args), result_action="return_value")


@pytest.fixture()
def registry_yaml(tmp_path: Path) -> Path:
    """Create a test registry YAML for CLI testing."""
    raw = {
        "datasets": {
            "dem_test": {
                "description": "Test DEM dataset",
                "strategy": "stac_cog",
                "catalog_url": "https://example.com/stac/v1",
                "collection": "3dep-seamless",
                "crs": "EPSG:4326",
                "category": "topography",
                "variables": [
                    {"name": "elevation", "band": 1, "units": "m", "categorical": False},
                ],
                "derived_variables": [
                    {"name": "slope", "source": "elevation", "method": "horn"},
                ],
            },
            "nlcd_single": {
                "description": "NLCD single-file test",
                "strategy": "local_tiff",
                "crs": "EPSG:5070",
                "category": "land_cover",
                "download": {
                    "url": "s3://bucket/nlcd_2021.tif",
                    "size_gb": 1.5,
                    "format": "COG",
                },
                "variables": [
                    {"name": "land_cover", "band": 1, "categorical": True},
                ],
            },
            "nlcd_multi": {
                "description": "NLCD multi-file test",
                "strategy": "local_tiff",
                "crs": "EPSG:5070",
                "category": "land_cover",
                "download": {
                    "files": [
                        {
                            "year": 2021,
                            "variable": "land_cover",
                            "url": "s3://bucket/lc_2021.tif",
                            "size_gb": 1.5,
                        },
                        {
                            "year": 2019,
                            "variable": "land_cover",
                            "url": "s3://bucket/lc_2019.tif",
                            "size_gb": 1.5,
                        },
                        {
                            "year": 2021,
                            "variable": "impervious",
                            "url": "s3://bucket/imp_2021.tif",
                            "size_gb": 1.2,
                        },
                    ],
                },
                "variables": [
                    {"name": "land_cover", "band": 1, "categorical": True},
                    {"name": "impervious", "band": 1, "categorical": False},
                ],
            },
            "remote_ds": {
                "description": "Remote dataset (no download)",
                "strategy": "stac_cog",
                "catalog_url": "https://example.com/stac/v1",
                "collection": "test",
                "crs": "EPSG:4326",
                "category": "topography",
                "variables": [],
            },
        }
    }
    path = tmp_path / "datasets.yml"
    path.write_text(yaml.dump(raw))
    return path


# ---------------------------------------------------------------------------
# datasets list
# ---------------------------------------------------------------------------


def test_datasets_list(registry_yaml: Path, capsys: pytest.CaptureFixture[str]):
    _run("datasets", "list", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "dem_test" in out
    assert "nlcd_single" in out
    assert "nlcd_multi" in out
    assert "Topography" in out
    assert "Land Cover" in out


def test_datasets_list_shows_strategy(registry_yaml: Path, capsys: pytest.CaptureFixture[str]):
    _run("datasets", "list", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "stac_cog" in out
    assert "local_tiff" in out


# ---------------------------------------------------------------------------
# datasets info
# ---------------------------------------------------------------------------


def test_datasets_info_known(registry_yaml: Path, capsys: pytest.CaptureFixture[str]):
    _run("datasets", "info", "dem_test", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "Dataset: dem_test" in out
    assert "stac_cog" in out
    assert "EPSG:4326" in out
    assert "elevation" in out
    assert "slope" in out


def test_datasets_info_with_download(registry_yaml: Path, capsys: pytest.CaptureFixture[str]):
    _run("datasets", "info", "nlcd_single", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "s3://bucket/nlcd_2021.tif" in out
    assert "1.5 GB" in out


def test_datasets_info_multi_file(registry_yaml: Path, capsys: pytest.CaptureFixture[str]):
    _run("datasets", "info", "nlcd_multi", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "3 files available" in out
    assert "2021" in out
    assert "2019" in out
    assert "land_cover" in out
    assert "impervious" in out


def test_datasets_info_unknown(registry_yaml: Path):
    with pytest.raises(SystemExit, match="1"):
        _run("datasets", "info", "nonexistent", "--registry", str(registry_yaml))


# ---------------------------------------------------------------------------
# datasets download
# ---------------------------------------------------------------------------


def test_datasets_download_no_download_block(registry_yaml: Path):
    """Datasets without download info should error."""
    with pytest.raises(SystemExit, match="1"):
        _run("datasets", "download", "remote_ds", "--registry", str(registry_yaml))


@patch("shutil.which", return_value=None)
def test_datasets_download_no_aws_cli(mock_which, registry_yaml: Path):
    """Error when AWS CLI is not installed."""
    with pytest.raises(SystemExit, match="1"):
        _run(
            "datasets",
            "download",
            "nlcd_single",
            "--registry",
            str(registry_yaml),
        )


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_single_file(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_single",
        "--dest",
        str(tmp_path),
        "--registry",
        str(registry_yaml),
    )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "aws"
    assert "s3://bucket/nlcd_2021.tif" in cmd


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_multi_file_all(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """Download all files when no --years/--variables filters."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_multi",
        "--dest",
        str(tmp_path),
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 3


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_multi_file_filter_years(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """--years filter selects only matching files."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_multi",
        "--dest",
        str(tmp_path),
        "--years",
        "2021",
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 2  # lc_2021 + imp_2021


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_multi_file_filter_variables(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """--variables filter selects only matching files."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_multi",
        "--dest",
        str(tmp_path),
        "--variables",
        "land_cover",
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 2  # lc_2021 + lc_2019


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_multi_file_filter_both(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """--years + --variables narrows to intersection."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_multi",
        "--dest",
        str(tmp_path),
        "--years",
        "2021",
        "--variables",
        "land_cover",
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 1  # only lc_2021


@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_multi_file_no_match(
    mock_which,
    registry_yaml: Path,
    tmp_path: Path,
):
    """No matching files raises error."""
    with pytest.raises(SystemExit, match="1"):
        _run(
            "datasets",
            "download",
            "nlcd_multi",
            "--dest",
            str(tmp_path),
            "--years",
            "2000",
            "--registry",
            str(registry_yaml),
        )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@patch("hydro_param.cli.run_pipeline")
def test_run_invokes_pipeline(mock_pipeline, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy: true")
    _run("run", str(config_path))
    mock_pipeline.assert_called_once()
    args = mock_pipeline.call_args[0]
    assert args[0] == str(config_path)


@patch("hydro_param.cli.run_pipeline")
def test_run_with_registry(mock_pipeline, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy: true")
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text("datasets: {}")
    _run("run", str(config_path), "--registry", str(reg_path))
    mock_pipeline.assert_called_once()
    args = mock_pipeline.call_args[0]
    assert args[1] == str(reg_path)


@patch("hydro_param.cli.run_pipeline", side_effect=RuntimeError("boom"))
def test_run_pipeline_failure(mock_pipeline, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy: true")
    with pytest.raises(SystemExit, match="1"):
        _run("run", str(config_path))
