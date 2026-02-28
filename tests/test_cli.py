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
            "nlcd_template": {
                "description": "NLCD template test",
                "strategy": "local_tiff",
                "crs": "EPSG:5070",
                "category": "land_cover",
                "download": {
                    "url_template": "s3://bucket/{variable}_{year}.tif",
                    "year_range": [2020, 2022],
                    "variables_available": ["lc", "imp"],
                    "requester_pays": True,
                    "format": "COG",
                    "notes": "Requester-pays bucket.",
                },
                "variables": [
                    {"name": "lc", "band": 1, "categorical": True},
                    {"name": "imp", "band": 1, "categorical": False},
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
            "temporal_ds": {
                "description": "Temporal test dataset",
                "strategy": "nhgf_stac",
                "collection": "snodas",
                "crs": "EPSG:4326",
                "category": "snow",
                "temporal": True,
                "t_coord": "time",
                "time_step": "daily",
                "year_range": [2003, 2025],
                "variables": [
                    {"name": "SWE", "band": 1, "native_name": "SWE"},
                ],
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


def test_datasets_list_shows_temporal_info(registry_yaml: Path, capsys: pytest.CaptureFixture[str]):
    """datasets list shows time_step and year_range for temporal datasets."""
    _run("datasets", "list", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "daily" in out
    assert "2003" in out
    assert "2025" in out


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


def test_datasets_info_shows_temporal_metadata(
    registry_yaml: Path, capsys: pytest.CaptureFixture[str]
):
    """datasets info shows time_step and year_range for temporal datasets."""
    _run("datasets", "info", "temporal_ds", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "Time step: daily" in out
    assert "2003" in out and "2025" in out


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


@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_invalid_year(
    mock_which,
    registry_yaml: Path,
    tmp_path: Path,
):
    """Non-integer --years value raises a clean error."""
    with pytest.raises(SystemExit, match="1"):
        _run(
            "datasets",
            "download",
            "nlcd_multi",
            "--dest",
            str(tmp_path),
            "--years",
            "abc",
            "--registry",
            str(registry_yaml),
        )


# ---------------------------------------------------------------------------
# datasets info — template
# ---------------------------------------------------------------------------


def test_datasets_info_template(registry_yaml: Path, capsys: pytest.CaptureFixture[str]):
    """Template-based dataset shows year range, variables, requester-pays."""
    _run("datasets", "info", "nlcd_template", "--registry", str(registry_yaml))
    out = capsys.readouterr().out
    assert "6 files via URL template" in out
    assert "2020-2022" in out
    assert "lc" in out
    assert "imp" in out
    assert "Requester-pays: yes" in out
    assert "Requester-pays bucket." in out


# ---------------------------------------------------------------------------
# datasets download — template
# ---------------------------------------------------------------------------


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_template_all(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """Download all template-expanded files (3 years x 2 vars = 6)."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_template",
        "--dest",
        str(tmp_path),
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 6


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_template_filter_years(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """--years filter on template dataset."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_template",
        "--dest",
        str(tmp_path),
        "--years",
        "2021",
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 2  # 1 year x 2 vars


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_template_filter_variables(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """--variables filter on template dataset."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_template",
        "--dest",
        str(tmp_path),
        "--variables",
        "lc",
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 3  # 3 years x 1 var


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_requester_pays_flag(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """Requester-pays datasets use --request-payer=requester."""
    mock_run.return_value.returncode = 0
    _run(
        "datasets",
        "download",
        "nlcd_template",
        "--dest",
        str(tmp_path),
        "--years",
        "2020",
        "--variables",
        "lc",
        "--registry",
        str(registry_yaml),
    )
    assert mock_run.call_count == 1
    cmd = mock_run.call_args[0][0]
    assert "--request-payer=requester" in cmd
    assert "--no-sign-request" not in cmd


@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_datasets_download_no_sign_request_for_public(
    mock_which,
    mock_run,
    registry_yaml: Path,
    tmp_path: Path,
):
    """Public datasets use --no-sign-request."""
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
    assert mock_run.call_count == 1
    cmd = mock_run.call_args[0][0]
    assert "--no-sign-request" in cmd
    assert "--request-payer=requester" not in cmd


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_creates_project(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    project = tmp_path / "test_project"
    _run("init", str(project))
    assert (project / ".hydro-param").is_file()
    assert (project / "configs" / "pipeline.yml").is_file()
    assert (project / "data" / "fabrics").is_dir()
    assert (project / "data" / "topography").is_dir()
    assert (project / "output").is_dir()
    assert (project / "models").is_dir()
    out = capsys.readouterr().out
    assert "Initialized" in out


def test_init_refuses_existing(tmp_path: Path):
    project = tmp_path / "existing"
    _run("init", str(project))
    with pytest.raises(SystemExit):
        _run("init", str(project))


def test_init_force_reinitialises(tmp_path: Path):
    project = tmp_path / "existing"
    _run("init", str(project))
    _run("init", str(project), "--force")
    assert (project / ".hydro-param").is_file()


def test_init_default_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _run("init")
    assert (tmp_path / ".hydro-param").is_file()


def test_init_template_config_is_valid(tmp_path: Path):
    project = tmp_path / "valid_config"
    _run("init", str(project))
    content = (project / "configs" / "pipeline.yml").read_text()
    parsed = yaml.safe_load(content)
    assert parsed["target_fabric"]["path"] == "data/fabrics/catchments.gpkg"


# ---------------------------------------------------------------------------
# datasets download — auto-routing
# ---------------------------------------------------------------------------


@patch("hydro_param.cli.find_project_root")
@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_download_auto_routes_in_project(
    mock_which,
    mock_run,
    mock_find_root,
    registry_yaml: Path,
    tmp_path: Path,
):
    """When inside a project and no --dest, routes to data/<category>/."""
    mock_run.return_value.returncode = 0
    project = tmp_path / "my_project"
    project.mkdir()
    mock_find_root.return_value = project

    _run(
        "datasets",
        "download",
        "nlcd_single",
        "--registry",
        str(registry_yaml),
    )

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    dest_arg = cmd[-1]
    assert str(project / "data" / "land_cover") in dest_arg


@patch("hydro_param.cli.find_project_root")
@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_download_explicit_dest_overrides_project(
    mock_which,
    mock_run,
    mock_find_root,
    registry_yaml: Path,
    tmp_path: Path,
):
    """When --dest is given, use it even inside a project."""
    mock_run.return_value.returncode = 0
    mock_find_root.return_value = tmp_path  # project detected
    custom_dest = tmp_path / "custom"

    _run(
        "datasets",
        "download",
        "nlcd_single",
        "--dest",
        str(custom_dest),
        "--registry",
        str(registry_yaml),
    )

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    dest_arg = cmd[-1]
    assert str(custom_dest) in dest_arg


@patch("hydro_param.cli.find_project_root", return_value=None)
@patch("hydro_param.cli.subprocess.run")
@patch("shutil.which", return_value="/usr/bin/aws")
def test_download_no_project_falls_back_to_cwd(
    mock_which,
    mock_run,
    mock_find_root,
    registry_yaml: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """When not inside a project and no --dest, downloads to cwd."""
    mock_run.return_value.returncode = 0
    monkeypatch.chdir(tmp_path)

    _run(
        "datasets",
        "download",
        "nlcd_single",
        "--registry",
        str(registry_yaml),
    )

    mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@patch("hydro_param.cli.run_pipeline_from_config")
@patch("hydro_param.cli.load_config")
def test_run_invokes_pipeline(mock_load_config, mock_run_pipeline, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy: true")
    _run("run", str(config_path))
    mock_load_config.assert_called_once()
    mock_run_pipeline.assert_called_once()


@patch("hydro_param.cli.run_pipeline_from_config")
@patch("hydro_param.cli.load_config")
def test_run_with_registry(mock_load_config, mock_run_pipeline, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy: true")
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text("datasets: {}")
    _run("run", str(config_path), "--registry", str(reg_path))
    mock_run_pipeline.assert_called_once()


@patch("hydro_param.cli.run_pipeline_from_config", side_effect=RuntimeError("boom"))
@patch("hydro_param.cli.load_config")
def test_run_pipeline_failure(mock_load_config, mock_run_pipeline, tmp_path: Path):
    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy: true")
    with pytest.raises(SystemExit, match="1"):
        _run("run", str(config_path))


@patch("hydro_param.cli.run_pipeline_from_config")
@patch("hydro_param.cli.load_config")
def test_run_resume_flag_sets_config(mock_load_config, mock_run_pipeline, tmp_path: Path):
    """--resume flag causes processing.resume to be True in the config."""
    from hydro_param.config import PipelineConfig

    mock_cfg = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )
    mock_load_config.return_value = mock_cfg

    config_path = tmp_path / "config.yml"
    config_path.write_text("dummy: true")
    _run("run", str(config_path), "--resume")

    mock_run_pipeline.assert_called_once()
    actual_cfg = mock_run_pipeline.call_args[0][0]
    assert actual_cfg.processing.resume is True


# ---------------------------------------------------------------------------
# pywatershed run — Phase 2 input validation
# ---------------------------------------------------------------------------


def _setup_pws_project(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal fabric + SIR output for pywatershed CLI tests.

    Returns
    -------
    tuple[Path, Path]
        (fabric_path, sir_output_dir)
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import box

    from hydro_param.manifest import PipelineManifest, SIRManifestEntry

    # Fabric
    fabric_path = tmp_path / "nhru.gpkg"
    gdf = gpd.GeoDataFrame(
        {"nhm_id": [1, 2], "geometry": [box(0, 0, 1, 1), box(1, 1, 2, 2)]},
        crs="EPSG:4326",
    )
    gdf.to_file(fabric_path, driver="GPKG")

    # SIR output directory with one static variable
    sir_dir = tmp_path / "output" / "sir"
    sir_dir.mkdir(parents=True)
    idx = pd.Index([1, 2], name="nhm_id")
    pd.DataFrame({"elevation_m_mean": [100.0, 200.0]}, index=idx).to_csv(
        sir_dir / "elevation_m_mean.csv"
    )

    # Manifest
    sir_entry = SIRManifestEntry(
        static_files={"elevation_m_mean": "sir/elevation_m_mean.csv"},
    )
    PipelineManifest(sir=sir_entry).save(tmp_path / "output")

    return fabric_path, tmp_path / "output"


def _write_pws_config(
    tmp_path: Path,
    *,
    fabric_path: str | Path,
    sir_path: str | Path = "output",
    segment_path: str | None = None,
    waterbody_path: str | None = None,
) -> Path:
    """Write a v3.0 pywatershed run config YAML for testing."""
    cfg: dict = {
        "target_model": "pywatershed",
        "version": "3.0",
        "domain": {
            "fabric_path": str(fabric_path),
        },
        "time": {"start": "2020-10-01", "end": "2021-09-30"},
        "sir_path": str(sir_path),
    }
    if segment_path is not None:
        cfg["domain"]["segment_path"] = segment_path
    if waterbody_path is not None:
        cfg["domain"]["waterbody_path"] = waterbody_path
    path = tmp_path / "pws_config.yml"
    path.write_text(yaml.dump(cfg))
    return path


def test_pws_run_segment_path_missing_exits(tmp_path: Path) -> None:
    """pws_run_cmd exits early when segment_path file does not exist."""
    fabric_path, sir_path = _setup_pws_project(tmp_path)
    config_path = _write_pws_config(
        tmp_path,
        fabric_path=fabric_path,
        sir_path=sir_path,
        segment_path=str(tmp_path / "nonexistent.gpkg"),
    )
    with pytest.raises(SystemExit):
        _run("pywatershed", "run", str(config_path))


def test_pws_run_waterbody_path_missing_exits(tmp_path: Path) -> None:
    """pws_run_cmd exits early when waterbody_path file does not exist."""
    fabric_path, sir_path = _setup_pws_project(tmp_path)
    config_path = _write_pws_config(
        tmp_path,
        fabric_path=fabric_path,
        sir_path=sir_path,
        waterbody_path=str(tmp_path / "nonexistent.gpkg"),
    )
    with pytest.raises(SystemExit):
        _run("pywatershed", "run", str(config_path))


def test_pws_run_waterbody_missing_ftype_exits(tmp_path: Path) -> None:
    """pws_run_cmd exits early when waterbody file lacks ftype column."""
    import geopandas as gpd
    from shapely.geometry import box

    fabric_path, sir_path = _setup_pws_project(tmp_path)

    wb_path = tmp_path / "waterbodies.gpkg"
    gdf = gpd.GeoDataFrame({"name": ["lake1"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    gdf.to_file(wb_path, driver="GPKG")

    config_path = _write_pws_config(
        tmp_path,
        fabric_path=fabric_path,
        sir_path=sir_path,
        waterbody_path=str(wb_path),
    )
    with pytest.raises(SystemExit):
        _run("pywatershed", "run", str(config_path))
