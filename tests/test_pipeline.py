"""Tests for pipeline orchestration.

Tests focus on the pipeline stages that don't require gdptools
(fabric loading, dataset resolution, SIR assembly, batching).
Processing tests are integration-level and require gdptools.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import xarray as xr
import yaml
from shapely.geometry import Point, box

from hydro_param.config import PipelineConfig, load_config
from hydro_param.dataset_registry import load_registry
from hydro_param.pipeline import (
    PipelineResult,
    Stage4Results,
    _buffered_bbox,
    _process_batch,
    _process_temporal,
    _split_time_period_by_year,
    _write_temporal_file,
    _write_variable_file,
    resolve_bbox,
    stage1_resolve_fabric,
    stage2_resolve_datasets,
)
from hydro_param.processing import TemporalProcessor, ZonalProcessor, get_processor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fabric_gpkg(tmp_path: Path) -> Path:
    """Create a test fabric GeoPackage."""
    gdf = gpd.GeoDataFrame(
        {"featureid": [1, 2, 3, 4]},
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(0, 1, 1, 2),
            box(1, 1, 2, 2),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "catchments.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


@pytest.fixture()
def config_yaml(tmp_path: Path, fabric_gpkg: Path) -> Path:
    """Create a test pipeline config YAML."""
    raw = {
        "target_fabric": {
            "path": str(fabric_gpkg),
            "id_field": "featureid",
        },
        "domain": {"type": "bbox", "bbox": [-1.0, -1.0, 3.0, 3.0]},
        "datasets": [
            {"name": "dem_test", "variables": ["elevation", "slope"], "statistics": ["mean"]},
        ],
        "output": {
            "path": str(tmp_path / "output"),
            "format": "netcdf",
            "sir_name": "test_sir",
        },
        "processing": {"batch_size": 2},
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw))
    return path


@pytest.fixture()
def registry_yaml(tmp_path: Path) -> Path:
    """Create a test dataset registry YAML."""
    raw = {
        "datasets": {
            "dem_test": {
                "strategy": "stac_cog",
                "catalog_url": "https://example.com/stac/v1",
                "collection": "3dep-seamless",
                "crs": "EPSG:4326",
                "variables": [
                    {"name": "elevation", "band": 1, "units": "m", "categorical": False},
                ],
                "derived_variables": [
                    {"name": "slope", "source": "elevation", "method": "horn", "units": "degrees"},
                ],
            },
        }
    }
    path = tmp_path / "registry.yml"
    path.write_text(yaml.dump(raw))
    return path


# ---------------------------------------------------------------------------
# Stage 1: Resolve fabric
# ---------------------------------------------------------------------------


def test_stage1_loads_fabric(config_yaml: Path, fabric_gpkg: Path):
    config = load_config(config_yaml)
    fabric = stage1_resolve_fabric(config)
    assert isinstance(fabric, gpd.GeoDataFrame)
    assert len(fabric) == 4
    assert "featureid" in fabric.columns


def test_stage1_bbox_filter_clips_fabric(tmp_path: Path):
    """Bbox filter reduces feature count to only those within the bbox."""
    gdf = gpd.GeoDataFrame(
        {"featureid": [1, 2, 3, 4]},
        geometry=[
            box(0, 0, 1, 1),  # inside bbox
            box(1, 0, 2, 1),  # inside bbox
            box(5, 5, 6, 6),  # outside bbox
            box(10, 10, 11, 11),  # outside bbox
        ],
        crs="EPSG:4326",
    )
    gpkg_path = tmp_path / "catchments.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    raw = {
        "target_fabric": {"path": str(gpkg_path), "id_field": "featureid"},
        "domain": {"type": "bbox", "bbox": [-0.5, -0.5, 2.5, 1.5]},
        "datasets": [],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(raw))

    config = load_config(cfg_path)
    fabric = stage1_resolve_fabric(config)
    assert len(fabric) == 2
    assert set(fabric["featureid"]) == {1, 2}


def test_stage1_bbox_filter_empty_raises(tmp_path: Path):
    """Bbox filter raises ValueError when no features are within the bbox."""
    gdf = gpd.GeoDataFrame(
        {"featureid": [1]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    gpkg_path = tmp_path / "catchments.gpkg"
    gdf.to_file(gpkg_path, driver="GPKG")

    raw = {
        "target_fabric": {"path": str(gpkg_path), "id_field": "featureid"},
        "domain": {"type": "bbox", "bbox": [50.0, 50.0, 51.0, 51.0]},
        "datasets": [],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(raw))

    config = load_config(cfg_path)
    with pytest.raises(ValueError, match="No features found"):
        stage1_resolve_fabric(config)


def test_stage1_rejects_missing_id_field(config_yaml: Path, tmp_path: Path):
    # Create config pointing to fabric without required id field
    gdf = gpd.GeoDataFrame(
        {"wrong_id": [1]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )
    bad_gpkg = tmp_path / "bad.gpkg"
    gdf.to_file(bad_gpkg, driver="GPKG")

    raw = {
        "target_fabric": {"path": str(bad_gpkg), "id_field": "featureid"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": [],
    }
    cfg_path = tmp_path / "bad_config.yml"
    cfg_path.write_text(yaml.dump(raw))

    config = load_config(cfg_path)
    with pytest.raises(ValueError, match="ID field.*not found"):
        stage1_resolve_fabric(config)


# ---------------------------------------------------------------------------
# Stage 2: Resolve datasets
# ---------------------------------------------------------------------------


def test_stage2_resolves_datasets(config_yaml: Path, registry_yaml: Path):
    config = load_config(config_yaml)
    registry = load_registry(registry_yaml)
    resolved = stage2_resolve_datasets(config, registry)
    assert len(resolved) == 1
    entry, ds_req, var_specs = resolved[0]
    assert entry.strategy == "stac_cog"
    assert len(var_specs) == 2
    assert var_specs[0].name == "elevation"
    assert var_specs[1].name == "slope"


def test_stage2_applies_source_override(tmp_path: Path):
    """Pipeline config source overrides registry source for local_tiff."""
    reg_raw = {
        "datasets": {
            "nlcd_2021": {
                "strategy": "local_tiff",
                "source": "/registry/default.tif",
                "variables": [{"name": "land_cover", "band": 1, "categorical": True}],
            },
        }
    }
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text(yaml.dump(reg_raw))

    cfg_raw = {
        "target_fabric": {"path": "test.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": [
            {"name": "nlcd_2021", "source": "/user/override.tif", "variables": ["land_cover"]},
        ],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(cfg_raw))

    config = load_config(cfg_path)
    registry = load_registry(reg_path)
    resolved = stage2_resolve_datasets(config, registry)

    entry, ds_req, var_specs = resolved[0]
    assert entry.source == "/user/override.tif"


def test_stage2_source_override_does_not_mutate_registry(tmp_path: Path):
    """Source override creates a copy; original registry entry is unchanged."""
    reg_raw = {
        "datasets": {
            "nlcd_2021": {
                "strategy": "local_tiff",
                "source": "/registry/default.tif",
                "variables": [{"name": "land_cover", "band": 1, "categorical": True}],
            },
        }
    }
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text(yaml.dump(reg_raw))

    cfg_raw = {
        "target_fabric": {"path": "test.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": [
            {"name": "nlcd_2021", "source": "/user/override.tif", "variables": ["land_cover"]},
        ],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(cfg_raw))

    config = load_config(cfg_path)
    registry = load_registry(reg_path)
    stage2_resolve_datasets(config, registry)

    # Original registry entry must be unchanged
    assert registry.get("nlcd_2021").source == "/registry/default.tif"


def test_stage2_missing_source_for_local_tiff_raises(tmp_path: Path):
    """local_tiff with no source in registry or config raises with helpful message."""
    reg_raw = {
        "datasets": {
            "nlcd_2021": {
                "strategy": "local_tiff",
                "download": {"url": "s3://bucket/nlcd.tif", "size_gb": 1.5},
                "variables": [{"name": "land_cover", "band": 1, "categorical": True}],
            },
        }
    }
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text(yaml.dump(reg_raw))

    cfg_raw = {
        "target_fabric": {"path": "test.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": [
            {"name": "nlcd_2021", "variables": ["land_cover"]},
        ],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(cfg_raw))

    config = load_config(cfg_path)
    registry = load_registry(reg_path)

    with pytest.raises(ValueError, match="nlcd_2021") as exc_info:
        stage2_resolve_datasets(config, registry)
    msg = str(exc_info.value)
    assert "s3://bucket/nlcd.tif" in msg
    assert "~1.5 GB" in msg
    assert "source" in msg


def test_stage2_rejects_unknown_dataset(config_yaml: Path, tmp_path: Path):
    # Registry with no matching dataset
    raw = {"datasets": {"other_ds": {"strategy": "local_tiff", "source": "x.tif"}}}
    reg_path = tmp_path / "empty_reg.yml"
    reg_path.write_text(yaml.dump(raw))

    config = load_config(config_yaml)
    registry = load_registry(reg_path)
    with pytest.raises(KeyError, match="not found in registry"):
        stage2_resolve_datasets(config, registry)


# ---------------------------------------------------------------------------
# Per-variable file writes + load_sir
# ---------------------------------------------------------------------------


def test_write_variable_file_returns_path(config_yaml: Path, fabric_gpkg: Path):
    """_write_variable_file returns the written CSV file path."""
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)
    feature_ids = fabric["featureid"].values

    df = pd.DataFrame(
        {"mean": [100.0, 200.0, 300.0, 400.0]},
        index=pd.Index([1, 2, 3, 4], name="featureid"),
    )

    path = _write_variable_file("elevation", df, "topography", config, feature_ids)
    assert path.exists()
    assert path.suffix == ".csv"

    result = pd.read_csv(path, index_col=0)
    assert "elevation" in result.columns
    assert len(result) == 4
    assert result.index.name == "featureid"


def test_load_sir_merges_variable_files(config_yaml: Path, fabric_gpkg: Path):
    """PipelineResult.load_sir() merges per-variable CSV files into one Dataset."""
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)
    feature_ids = fabric["featureid"].values

    elev_df = pd.DataFrame(
        {"mean": [100.0, 200.0, 300.0, 400.0]},
        index=pd.Index([1, 2, 3, 4], name="featureid"),
    )
    slope_df = pd.DataFrame(
        {"mean": [5.0, 10.0, 15.0, 20.0]},
        index=pd.Index([1, 2, 3, 4], name="featureid"),
    )

    elev_path = _write_variable_file("elevation", elev_df, "topo", config, feature_ids)
    slope_path = _write_variable_file("slope", slope_df, "topo", config, feature_ids)

    result = PipelineResult(
        output_dir=config.output.path,
        static_files={"elevation": elev_path, "slope": slope_path},
    )
    sir = result.load_sir()

    assert isinstance(sir, xr.Dataset)
    assert "elevation" in sir.data_vars
    assert "slope" in sir.data_vars
    id_field = config.target_fabric.id_field
    assert id_field in sir.dims
    assert sir.sizes[id_field] == 4


def test_load_sir_multi_statistic_round_trip(config_yaml: Path, fabric_gpkg: Path):
    """Multi-statistic columns survive the write→load_sir() round trip."""
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)
    feature_ids = fabric["featureid"].values

    df = pd.DataFrame(
        {"mean": [100.0, 200.0, 300.0, 400.0], "min": [90.0, 180.0, 270.0, 360.0]},
        index=pd.Index([1, 2, 3, 4], name="featureid"),
    )

    path = _write_variable_file("elevation", df, "topo", config, feature_ids)
    result = PipelineResult(
        output_dir=config.output.path,
        static_files={"elevation": path},
    )
    sir = result.load_sir()

    assert "elevation" in sir.data_vars  # "mean" renamed to var_name
    assert "elevation_min" in sir.data_vars  # "min" renamed to var_name_min
    assert sir["elevation"].values.tolist() == [100.0, 200.0, 300.0, 400.0]
    assert sir["elevation_min"].values.tolist() == [90.0, 180.0, 270.0, 360.0]


def test_write_variable_file_warns_on_index_mismatch(
    config_yaml: Path, fabric_gpkg: Path, caplog: pytest.LogCaptureFixture
):
    """_write_variable_file warns when index name doesn't match id_field."""
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)
    feature_ids = fabric["featureid"].values

    # DataFrame with unnamed index (not "featureid")
    df = pd.DataFrame(
        {"mean": [100.0, 200.0, 300.0, 400.0]},
        index=pd.Index([1, 2, 3, 4]),
    )

    with caplog.at_level(logging.WARNING, logger="hydro_param.pipeline"):
        _write_variable_file("elevation", df, "topo", config, feature_ids)

    assert "Index name mismatch" in caplog.text
    assert "featureid" in caplog.text


def test_load_sir_empty_returns_empty_dataset():
    """load_sir() returns empty Dataset when no files exist."""
    result = PipelineResult(output_dir=Path("/tmp"), static_files={})
    sir = result.load_sir()
    assert isinstance(sir, xr.Dataset)
    assert len(sir.data_vars) == 0


# ---------------------------------------------------------------------------
# Per-variable file: reindex with partial features
# ---------------------------------------------------------------------------


def test_write_variable_file_reindexes_partial_features(config_yaml: Path, fabric_gpkg: Path):
    """Missing features are filled with NaN and output is sorted."""
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)
    feature_ids = fabric["featureid"].values  # [1, 2, 3, 4]

    # Only features 1 and 3 present (simulates a single batch)
    df = pd.DataFrame(
        {"mean": [100.0, 300.0]},
        index=pd.Index([1, 3], name="featureid"),
    )

    path = _write_variable_file("elevation", df, "topo", config, feature_ids)
    result = pd.read_csv(path, index_col=0)

    assert len(result) == 4
    assert result.loc[1, "elevation"] == 100.0
    assert result.loc[3, "elevation"] == 300.0
    assert pd.isna(result.loc[2, "elevation"])
    assert pd.isna(result.loc[4, "elevation"])
    # Sorted by index
    assert list(result.index) == sorted(result.index)


# ---------------------------------------------------------------------------
# Temporal file writes
# ---------------------------------------------------------------------------


def test_write_temporal_file_netcdf(tmp_path: Path):
    """_write_temporal_file writes a valid NetCDF file."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(tmp_path / "output"), "format": "netcdf"},
    )

    ds = xr.Dataset({"SWE": (["time", "hru_id"], [[1.0, 2.0], [3.0, 4.0]])})
    path = _write_temporal_file("snodas_2020", ds, "climate", config)

    assert path.exists()
    assert path.suffix == ".nc"
    loaded = xr.open_dataset(path)
    assert "SWE" in loaded.data_vars
    assert loaded["SWE"].shape == (2, 2)
    loaded.close()


def test_write_temporal_file_parquet(tmp_path: Path):
    """_write_temporal_file writes a valid Parquet file."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(tmp_path / "output"), "format": "parquet"},
    )

    ds = xr.Dataset({"SWE": (["time", "hru_id"], [[1.0, 2.0]])})
    path = _write_temporal_file("snodas_2020", ds, "climate", config)

    assert path.exists()
    assert path.suffix == ".parquet"
    result = pd.read_parquet(path)
    assert "SWE" in result.columns


# ---------------------------------------------------------------------------
# Resolve bbox
# ---------------------------------------------------------------------------


def test_resolve_bbox(config_yaml: Path):
    config = load_config(config_yaml)
    bbox = resolve_bbox(config)
    assert bbox == [-1.0, -1.0, 3.0, 3.0]


def test_resolve_bbox_unsupported_type():
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        domain={"type": "huc2", "id": "02"},
        datasets=[],
    )
    with pytest.raises(NotImplementedError, match="not yet supported"):
        resolve_bbox(config)


def test_resolve_bbox_no_domain():
    """resolve_bbox raises ValueError when domain is None."""
    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "id"},
        datasets=[],
    )
    assert config.domain is None
    with pytest.raises(ValueError, match="No domain configured"):
        resolve_bbox(config)


# ---------------------------------------------------------------------------
# _buffered_bbox
# ---------------------------------------------------------------------------


def test_buffered_bbox_geographic():
    """_buffered_bbox returns WGS84 bbox with buffer for geographic CRS."""
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(-75.0, 40.0, -74.0, 41.0)],
        crs="EPSG:4326",
    )
    result = _buffered_bbox(gdf, buffer_frac=0.1)
    assert len(result) == 4
    assert result[0] < -75.0  # west buffered
    assert result[1] < 40.0  # south buffered
    assert result[2] > -74.0  # east buffered
    assert result[3] > 41.0  # north buffered


def test_buffered_bbox_projected():
    """_buffered_bbox reprojects to WGS84 for projected CRS fabrics."""
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[box(-75.0, 40.0, -74.0, 41.0)],
        crs="EPSG:4326",
    ).to_crs("EPSG:5070")
    result = _buffered_bbox(gdf, buffer_frac=0.0)
    # Should be approximately the original WGS84 bounds
    assert -76.0 < result[0] < -74.5
    assert 39.5 < result[1] < 40.5


# ---------------------------------------------------------------------------
# get_processor (kept from original tests)
# ---------------------------------------------------------------------------


def test_get_processor_returns_zonal_for_polygons():
    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )
    proc = get_processor(fabric)
    assert isinstance(proc, ZonalProcessor)


def test_get_processor_rejects_unsupported_geometry():
    points = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[Point(0, 0)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="Unsupported geometry type"):
        get_processor(points)


def test_get_processor_accepts_mixed_polygon_multipolygon():
    """Mixed Polygon/MultiPolygon (as produced by gpd.clip) should be accepted."""
    from shapely.geometry import MultiPolygon

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[
            box(0, 0, 1, 1),  # Polygon
            MultiPolygon([box(2, 2, 3, 3), box(4, 4, 5, 5)]),  # MultiPolygon
        ],
        crs="EPSG:4326",
    )
    proc = get_processor(fabric)
    assert isinstance(proc, ZonalProcessor)


def test_get_processor_rejects_empty_fabric():
    empty = gpd.GeoDataFrame({"hru_id": []}, geometry=[], crs="EPSG:4326")
    with pytest.raises(ValueError, match="empty"):
        get_processor(empty)


# ---------------------------------------------------------------------------
# NHGF STAC dispatch
# ---------------------------------------------------------------------------


def test_process_nhgf_stac_method_exists():
    """ZonalProcessor has the process_nhgf_stac method."""
    proc = ZonalProcessor()
    assert hasattr(proc, "process_nhgf_stac")
    assert callable(proc.process_nhgf_stac)


def test_process_batch_nhgf_stac_dispatch(tmp_path: Path):
    """_process_batch dispatches to process_nhgf_stac for nhgf_stac static datasets."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        crs="EPSG:5070",
        temporal=False,
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["majority"],
        year=2021,
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
    )

    mock_df = pd.DataFrame({"majority": [11, 21]}, index=["a", "b"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        results = _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)
        mock_method.assert_called_once()
        call_kwargs = mock_method.call_args
        assert call_kwargs.kwargs["collection_id"] == "nlcd-LndCov"
        assert call_kwargs.kwargs["year"] == 2021
        assert call_kwargs.kwargs["categorical"] is True

    assert "LndCov" in results
    assert len(results["LndCov"]) == 2


def test_process_batch_nhgf_stac_rejects_derived(tmp_path: Path):
    """Derived variables raise NotImplementedError for nhgf_stac strategy."""
    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, DerivedVariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
    )
    derived = DerivedVariableSpec(name="slope", source="elevation", method="horn")
    ds_req = DatasetRequest(name="test", variables=["slope"])

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    with pytest.raises(NotImplementedError, match="Derived variables not supported"):
        _process_batch(fabric, entry, ds_req, [derived], config, tmp_path)


def test_process_batch_nhgf_stac_passes_year(tmp_path: Path):
    """Year from DatasetRequest is propagated to process_nhgf_stac."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(name="test", variables=["LndCov"], year=2019)

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    mock_df = pd.DataFrame({"majority": [11]}, index=["a"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)
        assert mock_method.call_args.kwargs["year"] == 2019


def test_process_batch_nhgf_stac_year_none(tmp_path: Path):
    """When year is None, process_nhgf_stac receives year=None."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(name="test", variables=["LndCov"])  # year=None

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    mock_df = pd.DataFrame({"majority": [11]}, index=["a"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)
        assert mock_method.call_args.kwargs["year"] is None


def test_process_batch_nhgf_stac_passes_statistics(tmp_path: Path):
    """Statistics from DatasetRequest are propagated to process_nhgf_stac."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-FctImp",
        temporal=False,
    )
    var_spec = VariableSpec(name="FctImp", band=1, categorical=False)
    ds_req = DatasetRequest(
        name="nlcd_osn_fctimp",
        variables=["FctImp"],
        statistics=["mean", "median"],
        year=2021,
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    mock_df = pd.DataFrame({"mean": [0.5], "median": [0.4]}, index=["a"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)
        assert mock_method.call_args.kwargs["statistics"] == ["mean", "median"]


def test_process_batch_temporal_nhgf_stac_raises(tmp_path: Path):
    """Temporal nhgf_stac datasets raise NotImplementedError in _fetch()."""
    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
    )
    var_spec = VariableSpec(name="SWE", band=1)
    ds_req = DatasetRequest(name="snodas", variables=["SWE"])

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    with pytest.raises(NotImplementedError, match="Temporal nhgf_stac"):
        _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)


def test_process_nhgf_stac_integration(tmp_path: Path):
    """Integration test: verify process_nhgf_stac wires gdptools classes correctly."""
    from unittest.mock import MagicMock, patch

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    mock_collection = MagicMock(name="pystac.Collection")
    mock_nhgf_data = MagicMock(name="NHGFStacTiffData")
    mock_zonal = MagicMock(name="ZonalGen")
    mock_zonal.calculate_zonal.return_value = pd.DataFrame({"majority": [11, 21]}, index=["a", "b"])

    with (
        patch("gdptools.helpers.get_stac_collection", return_value=mock_collection) as p_coll,
        patch("gdptools.NHGFStacTiffData", return_value=mock_nhgf_data) as p_nhgf,
        patch("gdptools.ZonalGen", return_value=mock_zonal) as p_zonal,
    ):
        proc = ZonalProcessor()
        result = proc.process_nhgf_stac(
            fabric=fabric,
            collection_id="nlcd-LndCov",
            variable_name="LndCov",
            id_field="hru_id",
            year=2021,
            categorical=True,
        )

        p_coll.assert_called_once_with("nlcd-LndCov")
        p_nhgf.assert_called_once()
        nhgf_kwargs = p_nhgf.call_args.kwargs
        assert nhgf_kwargs["source_collection"] is mock_collection
        assert nhgf_kwargs["source_var"] == "LndCov"
        assert nhgf_kwargs["target_id"] == "hru_id"
        assert nhgf_kwargs["source_time_period"] == ["2021-01-01", "2021-12-31"]
        assert nhgf_kwargs["band"] == 1

        p_zonal.assert_called_once()
        mock_zonal.calculate_zonal.assert_called_once_with(categorical=True)

    assert list(result.columns) == ["majority"]
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Temporal processing
# ---------------------------------------------------------------------------


def test_temporal_processor_has_nhgf_stac_method():
    proc = TemporalProcessor()
    assert hasattr(proc, "process_nhgf_stac")
    assert callable(proc.process_nhgf_stac)


def test_temporal_processor_has_climr_cat_method():
    proc = TemporalProcessor()
    assert hasattr(proc, "process_climr_cat")
    assert callable(proc.process_climr_cat)


def test_stage4_results_dataclass(tmp_path: Path):
    r = Stage4Results()
    assert r.static_files == {}
    assert r.temporal_files == {}

    elev_path = tmp_path / "elevation.nc"
    snodas_path = tmp_path / "snodas_temporal.nc"
    r2 = Stage4Results(
        static_files={"elev": elev_path},
        temporal_files={"snodas": snodas_path},
    )
    assert "elev" in r2.static_files
    assert "snodas" in r2.temporal_files


def test_temporal_requires_time_period(tmp_path: Path):
    """stage2 raises if temporal dataset has no time_period."""
    reg_raw = {
        "datasets": {
            "snodas": {
                "strategy": "nhgf_stac",
                "collection": "snodas",
                "temporal": True,
                "t_coord": "time",
                "variables": [{"name": "SWE", "band": 1}],
            },
        }
    }
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text(yaml.dump(reg_raw))

    cfg_raw = {
        "target_fabric": {"path": "test.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": [
            {"name": "snodas", "variables": ["SWE"]},  # no time_period
        ],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(cfg_raw))

    config = load_config(cfg_path)
    registry = load_registry(reg_path)

    with pytest.raises(ValueError, match="temporal but no 'time_period'"):
        stage2_resolve_datasets(config, registry)


def test_process_temporal_nhgf_stac_dispatch(tmp_path: Path):
    """_process_temporal dispatches to TemporalProcessor.process_nhgf_stac."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
    )
    var_spec = VariableSpec(name="SWE", band=1)
    ds_req = DatasetRequest(
        name="snodas",
        variables=["SWE"],
        statistics=["mean"],
        time_period=["2020-01-01", "2020-01-31"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
    )

    mock_ds = xr.Dataset({"SWE": (["time", "hru_id"], [[1.0, 2.0]])})

    with patch.object(TemporalProcessor, "process_nhgf_stac", return_value=mock_ds) as mock_method:
        result = _process_temporal(fabric, entry, ds_req, [var_spec], config)
        mock_method.assert_called_once()
        call_kwargs = mock_method.call_args.kwargs
        assert call_kwargs["collection_id"] == "snodas"
        assert call_kwargs["time_period"] == ["2020-01-01", "2020-01-31"]
        assert call_kwargs["stat_method"] == "mean"

    assert isinstance(result, xr.Dataset)


def test_process_temporal_climr_cat_dispatch(tmp_path: Path):
    """_process_temporal dispatches to TemporalProcessor.process_climr_cat."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="climr_cat",
        catalog_id="gridmet",
        temporal=True,
        t_coord="day",
    )
    var_spec = VariableSpec(name="pr", band=1)
    ds_req = DatasetRequest(
        name="gridmet",
        variables=["pr"],
        statistics=["mean"],
        time_period=["2020-01-01", "2020-01-31"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
    )

    mock_ds = xr.Dataset({"pr": (["time", "hru_id"], [[1.0, 2.0]])})

    with patch.object(TemporalProcessor, "process_climr_cat", return_value=mock_ds) as mock_method:
        result = _process_temporal(fabric, entry, ds_req, [var_spec], config)
        mock_method.assert_called_once()
        call_kwargs = mock_method.call_args.kwargs
        assert call_kwargs["catalog_id"] == "gridmet"
        assert call_kwargs["time_period"] == ["2020-01-01", "2020-01-31"]

    assert isinstance(result, xr.Dataset)


def test_process_temporal_rejects_derived():
    """_process_temporal raises for DerivedVariableSpec."""
    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, DerivedVariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
    )
    derived = DerivedVariableSpec(name="slope", source="elevation", method="horn")
    ds_req = DatasetRequest(
        name="snodas",
        variables=["slope"],
        time_period=["2020-01-01", "2020-12-31"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    with pytest.raises(NotImplementedError, match="Derived variables not supported"):
        _process_temporal(fabric, entry, ds_req, [derived], config)


def test_process_temporal_unsupported_strategy():
    """_process_temporal raises for unsupported strategy."""
    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com/stac",
        collection="test",
        temporal=True,
        t_coord="time",
    )
    var_spec = VariableSpec(name="test_var", band=1)
    ds_req = DatasetRequest(
        name="test",
        variables=["test_var"],
        time_period=["2020-01-01", "2020-12-31"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    with pytest.raises(NotImplementedError, match="Temporal processing not supported"):
        _process_temporal(fabric, entry, ds_req, [var_spec], config)


def test_split_time_period_single_year():
    """Single-year period returns one chunk."""
    result = _split_time_period_by_year(["2020-01-01", "2020-12-31"])
    assert result == [["2020-01-01", "2020-12-31"]]


def test_split_time_period_multi_year():
    """Multi-year period splits at year boundaries."""
    result = _split_time_period_by_year(["2020-01-01", "2021-12-31"])
    assert result == [
        ["2020-01-01", "2020-12-31"],
        ["2021-01-01", "2021-12-31"],
    ]


def test_split_time_period_partial_years():
    """Partial years at start/end are preserved."""
    result = _split_time_period_by_year(["2020-03-15", "2022-06-30"])
    assert result == [
        ["2020-03-15", "2020-12-31"],
        ["2021-01-01", "2021-12-31"],
        ["2022-01-01", "2022-06-30"],
    ]


def test_split_time_period_same_day():
    """Single-day period returns one chunk."""
    result = _split_time_period_by_year(["2020-06-15", "2020-06-15"])
    assert result == [["2020-06-15", "2020-06-15"]]


def test_stage4_multi_year_produces_suffixed_keys(tmp_path: Path):
    """Multi-year datasets produce year-suffixed result keys in stage4."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"], "batch_id": [0, 0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        crs="EPSG:5070",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
        year=[2020, 2021],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
        output={"path": str(tmp_path / "output")},
    )

    mock_df = pd.DataFrame({"categorical": [11, 21]}, index=["a", "b"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df):
        results = stage4_process(fabric, [(entry, ds_req, [var_spec])], config)

    assert "LndCov_2020" in results.static_files
    assert "LndCov_2021" in results.static_files
    assert "LndCov" not in results.static_files
    assert results.categories["LndCov_2020"] == "land_cover"
    assert results.categories["LndCov_2021"] == "land_cover"


def test_stage4_single_year_produces_suffixed_key(tmp_path: Path):
    """A single-int year still produces a year-suffixed result key."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
        year=2021,
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(tmp_path / "output")},
    )

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df):
        results = stage4_process(fabric, [(entry, ds_req, [var_spec])], config)

    assert "LndCov_2021" in results.static_files
    assert "LndCov" not in results.static_files


def test_stage4_no_year_produces_unsuffixed_key(tmp_path: Path):
    """When year=None, result keys have no year suffix."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(tmp_path / "output")},
    )

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df):
        results = stage4_process(fabric, [(entry, ds_req, [var_spec])], config)

    assert "LndCov" in results.static_files
    assert not any("LndCov_" in k for k in results.static_files)


def test_stage4_duplicate_var_key_raises(tmp_path: Path):
    """stage4_process raises ValueError when two datasets produce the same var_key."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req1 = DatasetRequest(name="dataset_a", variables=["LndCov"], statistics=["categorical"])
    ds_req2 = DatasetRequest(name="dataset_b", variables=["LndCov"], statistics=["categorical"])

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(tmp_path / "output")},
    )

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    with (
        patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df),
        pytest.raises(ValueError, match="Duplicate static result key 'LndCov'"),
    ):
        stage4_process(
            fabric,
            [(entry, ds_req1, [var_spec]), (entry, ds_req2, [var_spec])],
            config,
        )


def test_process_temporal_empty_statistics_raises():
    """_process_temporal raises if statistics list is empty."""
    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
    )
    var_spec = VariableSpec(name="SWE", band=1)
    ds_req = DatasetRequest(
        name="snodas",
        variables=["SWE"],
        statistics=[],
        time_period=["2020-01-01", "2020-12-31"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    with pytest.raises(ValueError, match="no statistics specified"):
        _process_temporal(fabric, entry, ds_req, [var_spec], config)


def test_process_temporal_multi_statistics_warns(caplog: pytest.LogCaptureFixture):
    """_process_temporal warns when multiple statistics are provided."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
    )
    var_spec = VariableSpec(name="SWE", band=1)
    ds_req = DatasetRequest(
        name="snodas",
        variables=["SWE"],
        statistics=["mean", "median"],
        time_period=["2020-01-01", "2020-01-31"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    mock_ds = xr.Dataset({"SWE": (["time", "hru_id"], [[1.0]])})

    with (
        patch.object(TemporalProcessor, "process_nhgf_stac", return_value=mock_ds),
        caplog.at_level(logging.WARNING, logger="hydro_param.pipeline"),
    ):
        _process_temporal(fabric, entry, ds_req, [var_spec], config)

    assert "only 'mean' will be used" in caplog.text


def test_stage4_results_empty_has_no_files():
    """Empty Stage4Results has no file paths."""
    results = Stage4Results()
    assert len(results.static_files) == 0
    assert len(results.temporal_files) == 0
    assert len(results.categories) == 0


# ---------------------------------------------------------------------------
# Per-variable source_override tests
# ---------------------------------------------------------------------------


def test_stage2_allows_local_tiff_with_per_variable_sources(tmp_path: Path):
    """local_tiff with no dataset source but all vars having source_override passes stage2."""
    reg_raw = {
        "datasets": {
            "polaris_30m": {
                "strategy": "local_tiff",
                "category": "soils",
                "variables": [
                    {
                        "name": "sand",
                        "band": 1,
                        "source_override": "http://example.com/sand.vrt",
                    },
                    {
                        "name": "clay",
                        "band": 1,
                        "source_override": "http://example.com/clay.vrt",
                    },
                ],
            },
        }
    }
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text(yaml.dump(reg_raw))

    cfg_raw = {
        "target_fabric": {"path": "test.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": [
            {"name": "polaris_30m", "variables": ["sand", "clay"], "statistics": ["mean"]},
        ],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(cfg_raw))

    config = load_config(cfg_path)
    registry = load_registry(reg_path)

    # Should not raise — all requested vars have source_override
    resolved = stage2_resolve_datasets(config, registry)
    assert len(resolved) == 1


def test_stage2_rejects_local_tiff_mixed_sources(tmp_path: Path):
    """local_tiff raises when some requested vars lack source_override and no dataset source."""
    reg_raw = {
        "datasets": {
            "polaris_30m": {
                "strategy": "local_tiff",
                "category": "soils",
                "variables": [
                    {
                        "name": "sand",
                        "band": 1,
                        "source_override": "http://example.com/sand.vrt",
                    },
                    {"name": "theta_r", "band": 1},  # no source_override
                ],
            },
        }
    }
    reg_path = tmp_path / "registry.yml"
    reg_path.write_text(yaml.dump(reg_raw))

    cfg_raw = {
        "target_fabric": {"path": "test.gpkg", "id_field": "id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 1, 1]},
        "datasets": [
            {"name": "polaris_30m", "variables": ["sand", "theta_r"]},
        ],
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.dump(cfg_raw))

    config = load_config(cfg_path)
    registry = load_registry(reg_path)

    with pytest.raises(ValueError, match="polaris_30m"):
        stage2_resolve_datasets(config, registry)


def test_process_batch_local_tiff_passes_variable_source(tmp_path: Path):
    """_process_batch passes var_spec.source_override as variable_source to fetch_local_tiff."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    vrt_url = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/sand_mean_0_5.vrt"
    entry = DatasetEntry(strategy="local_tiff", crs="EPSG:4326")
    var_spec = VariableSpec(name="sand", band=1, source_override=vrt_url)
    ds_req = DatasetRequest(name="polaris_30m", variables=["sand"], statistics=["mean"])

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
    )

    mock_df = pd.DataFrame({"mean": [50.0, 60.0]}, index=["a", "b"])

    with (
        patch("hydro_param.pipeline.fetch_local_tiff") as mock_fetch,
        patch("hydro_param.pipeline.save_to_geotiff"),
        patch.object(ZonalProcessor, "process", return_value=mock_df),
    ):
        # Make fetch return a minimal DataArray
        import numpy as np

        mock_fetch.return_value = xr.DataArray(
            np.ones((4, 4)),
            dims=["y", "x"],
            coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
        )

        results = _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)

        # Verify variable_source was passed through
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args
        assert call_kwargs.kwargs["variable_source"] == vrt_url

    assert "sand" in results


def test_process_batch_stac_cog_passes_asset_key(tmp_path: Path):
    """_process_batch passes var_spec.asset_key to fetch_stac_cog for gNATSGO-style datasets."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="gnatsgo-rasters",
        crs="EPSG:5070",
    )
    var_spec = VariableSpec(name="aws0_100", band=1, asset_key="aws0_100")
    ds_req = DatasetRequest(name="gnatsgo_rasters", variables=["aws0_100"], statistics=["mean"])

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
    )

    mock_df = pd.DataFrame({"mean": [1.0, 2.0]}, index=["a", "b"])

    with (
        patch("hydro_param.pipeline.fetch_stac_cog") as mock_fetch,
        patch("hydro_param.pipeline.query_stac_items", return_value=[]),
        patch("hydro_param.pipeline.save_to_geotiff"),
        patch.object(ZonalProcessor, "process", return_value=mock_df),
    ):
        import numpy as np

        mock_fetch.return_value = xr.DataArray(
            np.ones((4, 4)),
            dims=["y", "x"],
            coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
        )

        results = _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)

        # Verify asset_key was passed through
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args
        assert call_kwargs.kwargs["asset_key"] == "aws0_100"

    assert "aws0_100" in results


def test_process_batch_stac_cog_no_asset_key_passes_none(tmp_path: Path):
    """_process_batch passes asset_key=None when VarSpec has no asset_key (e.g. 3DEP)."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://planetarycomputer.microsoft.com/api/stac/v1",
        collection="3dep-seamless",
        crs="EPSG:4269",
    )
    var_spec = VariableSpec(name="elevation", band=1)
    ds_req = DatasetRequest(name="dem_3dep_10m", variables=["elevation"], statistics=["mean"])

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
    )

    mock_df = pd.DataFrame({"mean": [100.0, 200.0]}, index=["a", "b"])

    with (
        patch("hydro_param.pipeline.fetch_stac_cog") as mock_fetch,
        patch("hydro_param.pipeline.query_stac_items", return_value=[]),
        patch("hydro_param.pipeline.save_to_geotiff"),
        patch.object(ZonalProcessor, "process", return_value=mock_df),
    ):
        import numpy as np

        mock_fetch.return_value = xr.DataArray(
            np.ones((4, 4)),
            dims=["y", "x"],
            coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
        )

        results = _process_batch(fabric, entry, ds_req, [var_spec], config, tmp_path)

        # Verify asset_key=None when not specified (uses dataset-level default)
        mock_fetch.assert_called_once()
        call_kwargs = mock_fetch.call_args
        assert call_kwargs.kwargs["asset_key"] is None

    assert "elevation" in results


# ---------------------------------------------------------------------------
# Resume (manifest-based skip)
# ---------------------------------------------------------------------------


def test_stage4_resume_skips_completed_dataset(tmp_path: Path):
    """With resume=True and valid manifest, stage4 skips completed datasets."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.manifest import (
        ManifestEntry,
        PipelineManifest,
        dataset_fingerprint,
        fabric_fingerprint,
    )
    from hydro_param.pipeline import stage4_process

    # Create fabric with batch_id
    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a", "b"], "batch_id": [0, 0]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    # Create output dir with existing file
    output_dir = tmp_path / "output"
    lc_dir = output_dir / "land_cover"
    lc_dir.mkdir(parents=True)
    csv_file = lc_dir / "LndCov.csv"
    csv_file.write_text("hru_id,LndCov\na,11\nb,21\n")

    # Create a fabric file so fingerprint works
    gpkg_path = tmp_path / "test.gpkg"
    gpkg_path.write_text("fake")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg_path), "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 2, 2]},
        datasets=[],
        output={"path": str(output_dir)},
        processing={"resume": True},
    )

    # Compute the expected fingerprint
    ds_fp = dataset_fingerprint(ds_req, entry, [var_spec], config.processing)
    fab_fp = fabric_fingerprint(config)

    # Write a manifest
    manifest = PipelineManifest(
        fabric_fingerprint=fab_fp,
        entries={
            "nlcd_osn_lndcov": ManifestEntry(
                fingerprint=ds_fp,
                static_files={"LndCov": "land_cover/LndCov.csv"},
            ),
        },
    )
    manifest.save(output_dir)

    # process_nhgf_stac should NOT be called — dataset is skipped
    with patch.object(ZonalProcessor, "process_nhgf_stac") as mock_method:
        results = stage4_process(fabric, [(entry, ds_req, [var_spec])], config)
        mock_method.assert_not_called()

    assert "LndCov" in results.static_files
    assert results.static_files["LndCov"] == csv_file


def test_stage4_resume_reprocesses_on_config_change(tmp_path: Path):
    """With resume=True but changed fingerprint, stage4 reprocesses."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.manifest import ManifestEntry, PipelineManifest, fabric_fingerprint
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    output_dir = tmp_path / "output"
    gpkg_path = tmp_path / "test.gpkg"
    gpkg_path.write_text("fake")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg_path), "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(output_dir)},
        processing={"resume": True},
    )

    fab_fp = fabric_fingerprint(config)

    # Write manifest with STALE fingerprint
    manifest = PipelineManifest(
        fabric_fingerprint=fab_fp,
        entries={
            "nlcd_osn_lndcov": ManifestEntry(
                fingerprint="sha256:stale_fingerprint",
                static_files={"LndCov": "land_cover/LndCov.csv"},
            ),
        },
    )
    manifest.save(output_dir)

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    # process_nhgf_stac SHOULD be called — fingerprint doesn't match
    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        results = stage4_process(fabric, [(entry, ds_req, [var_spec])], config)
        mock_method.assert_called_once()

    assert "LndCov" in results.static_files


def test_stage4_resume_disabled_by_default(tmp_path: Path):
    """With resume=False (default), stage4 processes everything."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    config = PipelineConfig(
        target_fabric={"path": "test.gpkg", "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(tmp_path / "output")},
    )

    # Default resume=False
    assert config.processing.resume is False

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        results = stage4_process(fabric, [(entry, ds_req, [var_spec])], config)
        mock_method.assert_called_once()

    assert "LndCov" in results.static_files  # noqa: E501


def test_stage4_resume_reprocesses_on_fabric_change(tmp_path: Path):
    """With resume=True but changed fabric, stage4 reprocesses all datasets."""
    from unittest.mock import patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.manifest import ManifestEntry, PipelineManifest
    from hydro_param.pipeline import stage4_process

    fabric = gpd.GeoDataFrame(
        {"hru_id": ["a"], "batch_id": [0]},
        geometry=[box(0, 0, 1, 1)],
        crs="EPSG:4326",
    )

    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
        category="land_cover",
    )
    var_spec = VariableSpec(name="LndCov", band=1, categorical=True)
    ds_req = DatasetRequest(
        name="nlcd_osn_lndcov",
        variables=["LndCov"],
        statistics=["categorical"],
    )

    output_dir = tmp_path / "output"
    lc_dir = output_dir / "land_cover"
    lc_dir.mkdir(parents=True)
    (lc_dir / "LndCov.csv").write_text("data")

    gpkg_path = tmp_path / "test.gpkg"
    gpkg_path.write_text("fake")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg_path), "id_field": "hru_id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
        output={"path": str(output_dir)},
        processing={"resume": True},
    )

    # Write manifest with STALE fabric fingerprint
    manifest = PipelineManifest(
        fabric_fingerprint="old_fabric.gpkg|0.0|0",
        entries={
            "nlcd_osn_lndcov": ManifestEntry(
                fingerprint="sha256:doesnt_matter",
                static_files={"LndCov": "land_cover/LndCov.csv"},
            ),
        },
    )
    manifest.save(output_dir)

    mock_df = pd.DataFrame({"categorical": [11]}, index=["a"])

    # Should reprocess because fabric fingerprint changed
    with patch.object(ZonalProcessor, "process_nhgf_stac", return_value=mock_df) as mock_method:
        results = stage4_process(fabric, [(entry, ds_req, [var_spec])], config)
        mock_method.assert_called_once()

    assert "LndCov" in results.static_files  # noqa: E501


# ---------------------------------------------------------------------------
# PipelineResult with SIR normalization
# ---------------------------------------------------------------------------


class TestPipelineResultSIR:
    """Tests for PipelineResult with SIR normalization."""

    def test_sir_fields_default_empty(self) -> None:
        result = PipelineResult(output_dir=Path("/tmp"))
        assert result.sir_files == {}
        assert result.sir_schema == []

    def test_load_sir_from_sir_files(self, tmp_path: Path) -> None:
        """load_sir() reads from sir_files (normalized) when available."""
        df = pd.DataFrame(
            {"elevation_m_mean": [100.0, 200.0]},
            index=pd.Index([1, 2], name="nhm_id"),
        )
        sir_path = tmp_path / "elevation_m_mean.csv"
        df.to_csv(sir_path)

        result = PipelineResult(
            output_dir=tmp_path,
            sir_files={"elevation_m_mean": sir_path},
        )
        sir = result.load_sir()
        assert "elevation_m_mean" in sir.data_vars

    def test_load_sir_falls_back_to_static(self, tmp_path: Path) -> None:
        """load_sir() falls back to static_files when no sir_files."""
        df = pd.DataFrame(
            {"elevation": [100.0]},
            index=pd.Index([1], name="nhm_id"),
        )
        path = tmp_path / "elevation.csv"
        df.to_csv(path)

        result = PipelineResult(
            output_dir=tmp_path,
            static_files={"elevation": path},
        )
        sir = result.load_sir()
        assert "elevation" in sir.data_vars

    def test_load_raw_sir(self, tmp_path: Path) -> None:
        """load_raw_sir() always reads from static_files."""
        df = pd.DataFrame(
            {"elevation": [100.0]},
            index=pd.Index([1], name="nhm_id"),
        )
        path = tmp_path / "elevation.csv"
        df.to_csv(path)

        result = PipelineResult(
            output_dir=tmp_path,
            static_files={"elevation": path},
            sir_files={"elevation_m_mean": path},  # even with sir_files present
        )
        raw = result.load_raw_sir()
        assert "elevation" in raw.data_vars


# ---------------------------------------------------------------------------
# Stage 5: temporal SIR normalization integration
# ---------------------------------------------------------------------------


def test_stage5_includes_temporal_normalization(tmp_path: Path) -> None:
    """stage5_normalize_sir calls normalize_sir_temporal for temporal files."""
    from unittest.mock import MagicMock, patch

    from hydro_param.config import DatasetRequest
    from hydro_param.dataset_registry import DatasetEntry, VariableSpec
    from hydro_param.pipeline import stage5_normalize_sir

    # Create minimal config mock
    config = MagicMock()
    config.output.path = tmp_path
    config.target_fabric.id_field = "nhm_id"
    config.processing.sir_validation = "tolerant"

    # Create stage4 results with a temporal file
    temporal_path = tmp_path / "climate" / "gridmet_2020_temporal.nc"
    temporal_path.parent.mkdir(parents=True)
    temporal_path.touch()

    stage4 = Stage4Results(
        static_files={},
        temporal_files={"gridmet_2020": temporal_path},
    )

    # Minimal resolved
    entry = DatasetEntry(
        strategy="climr_cat",
        catalog_id="gridmet",
        temporal=True,
        t_coord="day",
        variables=[VariableSpec(name="tmmx", units="K", long_name="daily_maximum_temperature")],
        category="climate",
    )
    ds_req = DatasetRequest(
        name="gridmet",
        variables=["tmmx"],
        statistics=["mean"],
        time_period=["2020-01-01", "2020-12-31"],
    )
    resolved: list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec]]] = [
        (entry, ds_req, list(entry.variables))
    ]

    # Mock normalize_sir_temporal to return a known result
    mock_temporal_result = {"tmmx_C_mean": tmp_path / "sir" / "tmmx_C_mean.nc"}

    with (
        patch("hydro_param.pipeline.normalize_sir") as mock_static,
        patch("hydro_param.pipeline.normalize_sir_temporal") as mock_temporal,
        patch("hydro_param.pipeline.validate_sir") as mock_validate,
    ):
        mock_static.return_value = {}
        mock_temporal.return_value = mock_temporal_result
        mock_validate.return_value = []

        sir_files, _schema, _warnings = stage5_normalize_sir(
            stage4,
            resolved,
            config,  # type: ignore[arg-type]
        )

        # normalize_sir_temporal was called
        mock_temporal.assert_called_once()

        # Result includes temporal files
        assert "tmmx_C_mean" in sir_files


def test_stage5_skips_temporal_when_no_temporal_files(tmp_path: Path) -> None:
    """stage5_normalize_sir does not call normalize_sir_temporal when empty."""
    from unittest.mock import MagicMock, patch

    from hydro_param.pipeline import stage5_normalize_sir

    config = MagicMock()
    config.output.path = tmp_path
    config.target_fabric.id_field = "nhm_id"
    config.processing.sir_validation = "tolerant"

    stage4 = Stage4Results(static_files={}, temporal_files={})

    with (
        patch("hydro_param.pipeline.normalize_sir") as mock_static,
        patch("hydro_param.pipeline.normalize_sir_temporal") as mock_temporal,
        patch("hydro_param.pipeline.validate_sir") as mock_validate,
    ):
        mock_static.return_value = {}
        mock_validate.return_value = []

        stage5_normalize_sir(stage4, [], config)  # type: ignore[arg-type]

        mock_temporal.assert_not_called()


# ---------------------------------------------------------------------------
# _process_batch memory cleanup + STAC items caching
# ---------------------------------------------------------------------------


def test_process_batch_releases_source_cache(tmp_path: Path):
    """_process_batch releases source_cache entries for raw vars after save."""
    from unittest.mock import MagicMock, patch

    import numpy as np

    rioxarray = pytest.importorskip("rioxarray")  # noqa: F841

    # Create a mock raster
    mock_da = xr.DataArray(
        np.ones((4, 4)),
        dims=["y", "x"],
        coords={"y": [1.0, 2.0, 3.0, 4.0], "x": [1.0, 2.0, 3.0, 4.0]},
        attrs={"units": "cm"},
    )
    mock_da = mock_da.rio.set_crs("EPSG:5070")
    mock_da = mock_da.rio.set_spatial_dims(x_dim="x", y_dim="y")

    # Build minimal fixtures
    fabric = gpd.GeoDataFrame(
        {"nhm_id": [1, 2]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:4326",
    )

    from hydro_param.dataset_registry import DatasetEntry, VariableSpec

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com/stac",
        collection="test",
        crs="EPSG:5070",
    )

    config_raw = {
        "target_fabric": {"path": "dummy.gpkg", "id_field": "nhm_id"},
        "domain": {"type": "bbox", "bbox": [0, 0, 2, 2]},
        "datasets": [
            {"name": "test_ds", "variables": ["var_a", "var_b"], "statistics": ["mean"]},
        ],
    }
    cfg_path = tmp_path / "cfg.yml"
    cfg_path.write_text(yaml.dump(config_raw))
    config = load_config(cfg_path)

    ds_req = config.datasets[0]

    var_specs: list = [
        VariableSpec(name="var_a", band=1, units="cm", categorical=False, asset_key="var_a"),
        VariableSpec(name="var_b", band=1, units="cm", categorical=False, asset_key="var_b"),
    ]

    mock_zonal_df = pd.DataFrame({"mean": [1.0, 2.0]}, index=pd.Index([1, 2], name="nhm_id"))

    with (
        patch("hydro_param.pipeline.fetch_stac_cog", return_value=mock_da),
        patch("hydro_param.pipeline.query_stac_items", return_value=[MagicMock()]),
        patch("hydro_param.pipeline.save_to_geotiff", return_value=tmp_path / "mock.tif"),
        patch("hydro_param.processing.ZonalProcessor.process", return_value=mock_zonal_df),
    ):
        results = _process_batch(fabric, entry, ds_req, var_specs, config, tmp_path)

    assert "var_a" in results
    assert "var_b" in results
    assert len(results["var_a"]) == 2
    assert len(results["var_b"]) == 2
