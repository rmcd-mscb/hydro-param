"""Tests for pipeline orchestration.

Tests focus on the pipeline stages that don't require gdptools
(fabric loading, dataset resolution, SIR assembly, batching).
Processing tests are integration-level and require gdptools.
"""

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
    resolve_bbox,
    stage1_resolve_fabric,
    stage2_resolve_datasets,
    stage5_format_output,
)
from hydro_param.processing import ZonalProcessor, get_processor

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
# Stage 5: SIR assembly
# ---------------------------------------------------------------------------


def test_stage5_produces_sir(config_yaml: Path, fabric_gpkg: Path):
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)

    results = {
        "elevation": pd.DataFrame(
            {"mean": [100.0, 200.0, 300.0, 400.0]},
            index=pd.Index([1, 2, 3, 4], name="featureid"),
        ),
        "slope": pd.DataFrame(
            {"mean": [5.0, 10.0, 15.0, 20.0]},
            index=pd.Index([1, 2, 3, 4], name="featureid"),
        ),
    }

    sir = stage5_format_output(results, config, fabric)

    assert isinstance(sir, xr.Dataset)
    assert "elevation" in sir.data_vars
    assert "slope" in sir.data_vars
    assert "hru_id" in sir.dims
    assert sir.sizes["hru_id"] == 4


def test_sir_has_cf_metadata(config_yaml: Path, fabric_gpkg: Path):
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)

    results = {
        "elevation": pd.DataFrame(
            {"mean": [100.0, 200.0, 300.0, 400.0]},
            index=pd.Index([1, 2, 3, 4], name="featureid"),
        ),
    }

    sir = stage5_format_output(results, config, fabric)

    assert sir.attrs["Conventions"] == "CF-1.8"
    assert "hydro-param" in sir.attrs["source"]
    assert sir.attrs["n_features"] == 4
    assert sir.attrs["target_fabric_id_field"] == "featureid"


def test_sir_writes_netcdf(config_yaml: Path, fabric_gpkg: Path, tmp_path: Path):
    config = load_config(config_yaml)
    fabric = gpd.read_file(fabric_gpkg)

    results = {
        "elevation": pd.DataFrame(
            {"mean": [100.0, 200.0, 300.0, 400.0]},
            index=pd.Index([1, 2, 3, 4], name="featureid"),
        ),
    }

    stage5_format_output(results, config, fabric)

    out_path = Path(config.output.path) / f"{config.output.sir_name}.nc"
    assert out_path.exists()

    loaded = xr.open_dataset(out_path)
    assert "elevation" in loaded.data_vars
    loaded.close()


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
    from hydro_param.pipeline import _process_batch

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
    from hydro_param.pipeline import _process_batch

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
    from hydro_param.pipeline import _process_batch

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
    from hydro_param.pipeline import _process_batch

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
    from hydro_param.pipeline import _process_batch

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
    from hydro_param.pipeline import _process_batch

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
