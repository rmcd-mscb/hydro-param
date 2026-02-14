"""Test the core pipeline with synthetic data."""

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
import yaml
from shapely.geometry import box

from hydro_param.config import PipelineConfig, load_config
from hydro_param.processing import ZonalProcessor, get_processor

# -- Config tests --


def test_load_config_from_yaml():
    raw = {
        "target_fabric": "test_fabric",
        "domain": {"type": "bbox", "bbox": [0, 0, 2, 2]},
        "datasets": [{"name": "temperature"}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        yaml.dump(raw, f)
        path = f.name

    config = load_config(path)
    assert isinstance(config, PipelineConfig)
    assert config.target_fabric == "test_fabric"
    assert len(config.datasets) == 1
    assert config.datasets[0].name == "temperature"
    Path(path).unlink()


def test_config_defaults():
    config = PipelineConfig(
        target_fabric="x",
        domain={"type": "bbox"},
        datasets=[],
    )
    assert config.output_path == "./output"


# -- Processing tests --


def _make_fabric():
    """Two side-by-side polygons covering a 4x4 grid."""
    return gpd.GeoDataFrame(
        {"hru_id": ["left", "right"]},
        geometry=[box(0, 0, 2, 4), box(2, 0, 4, 4)],
        crs="EPSG:4326",
    )


def _make_dataset():
    """4x4 grid: left half = 10, right half = 20."""
    data = np.array(
        [
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
            [10, 10, 20, 20],
        ],
        dtype=float,
    )
    return xr.Dataset(
        {"temperature": (["y", "x"], data)},
        coords={"x": [0.5, 1.5, 2.5, 3.5], "y": [0.5, 1.5, 2.5, 3.5]},
    )


def test_zonal_processor():
    fabric = _make_fabric()
    dataset = _make_dataset()

    proc = ZonalProcessor()
    result = proc.process(fabric, dataset, "temperature")

    assert isinstance(result, xr.Dataset)
    assert "temperature" in result
    assert set(result.hru_id.values) == {"left", "right"}

    left_val = float(result["temperature"].sel(hru_id="left"))
    right_val = float(result["temperature"].sel(hru_id="right"))
    assert left_val == 10.0
    assert right_val == 20.0


def test_get_processor_returns_zonal_for_polygons():
    fabric = _make_fabric()
    proc = get_processor(fabric)
    assert isinstance(proc, ZonalProcessor)


def test_sir_has_provenance_attrs():
    fabric = _make_fabric()
    dataset = _make_dataset()

    result = ZonalProcessor().process(fabric, dataset, "temperature")

    assert "source_dataset" in result.attrs
    assert "processing_method" in result.attrs
    assert result.attrs["processing_method"] == "zonal_mean"
