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


def test_get_processor_rejects_empty_fabric():
    empty = gpd.GeoDataFrame({"hru_id": []}, geometry=[], crs="EPSG:4326")
    with pytest.raises(ValueError, match="empty"):
        get_processor(empty)
