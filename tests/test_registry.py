"""Tests for dataset registry loading and resolution."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hydro_param.dataset_registry import (
    DatasetEntry,
    DatasetRegistry,
    DerivedVariableSpec,
    VariableSpec,
    load_registry,
)


@pytest.fixture()
def registry_yaml(tmp_path: Path) -> Path:
    """Create a minimal registry YAML file for testing."""
    raw = {
        "datasets": {
            "dem_test": {
                "description": "Test DEM",
                "strategy": "stac_cog",
                "catalog_url": "https://example.com/stac/v1",
                "collection": "3dep-seamless",
                "gsd": 10,
                "crs": "EPSG:4326",
                "sign": "planetary_computer",
                "category": "topography",
                "variables": [
                    {
                        "name": "elevation",
                        "band": 1,
                        "units": "m",
                        "long_name": "Surface elevation",
                        "categorical": False,
                    },
                ],
                "derived_variables": [
                    {
                        "name": "slope",
                        "source": "elevation",
                        "method": "horn",
                        "units": "degrees",
                    },
                    {
                        "name": "aspect",
                        "source": "elevation",
                        "method": "horn",
                        "units": "degrees",
                    },
                ],
            },
            "nlcd_test": {
                "strategy": "local_tiff",
                "source": "data/nlcd.tif",
                "crs": "EPSG:5070",
                "category": "land_cover",
                "variables": [
                    {
                        "name": "land_cover",
                        "band": 1,
                        "categorical": True,
                    },
                ],
            },
        }
    }
    path = tmp_path / "datasets.yml"
    path.write_text(yaml.dump(raw))
    return path


def test_load_registry(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    assert isinstance(registry, DatasetRegistry)
    assert "dem_test" in registry.datasets
    assert "nlcd_test" in registry.datasets


def test_registry_get_known_dataset(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    entry = registry.get("dem_test")
    assert isinstance(entry, DatasetEntry)
    assert entry.strategy == "stac_cog"
    assert entry.collection == "3dep-seamless"
    assert entry.gsd == 10


def test_registry_get_unknown_dataset(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("nonexistent")


def test_registry_unknown_dataset_lists_available(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="dem_test"):
        registry.get("nonexistent")


def test_resolve_raw_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    var = registry.resolve_variable("dem_test", "elevation")
    assert isinstance(var, VariableSpec)
    assert var.name == "elevation"
    assert var.units == "m"
    assert var.categorical is False


def test_resolve_derived_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    var = registry.resolve_variable("dem_test", "slope")
    assert isinstance(var, DerivedVariableSpec)
    assert var.name == "slope"
    assert var.source == "elevation"
    assert var.method == "horn"


def test_resolve_unknown_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="not found in dataset"):
        registry.resolve_variable("dem_test", "nonexistent")


def test_resolve_unknown_variable_lists_available(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="elevation"):
        registry.resolve_variable("dem_test", "nonexistent")


def test_categorical_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    var = registry.resolve_variable("nlcd_test", "land_cover")
    assert isinstance(var, VariableSpec)
    assert var.categorical is True


def test_stac_cog_entry_fields(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    entry = registry.get("dem_test")
    assert entry.catalog_url == "https://example.com/stac/v1"
    assert entry.sign == "planetary_computer"
    assert entry.asset_key == "data"


def test_local_tiff_entry_fields(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    entry = registry.get("nlcd_test")
    assert entry.source == "data/nlcd.tif"
    assert entry.crs == "EPSG:5070"


def test_stac_cog_requires_catalog_url_and_collection():
    with pytest.raises(ValidationError, match="stac_cog strategy requires"):
        DatasetEntry(strategy="stac_cog")


def test_local_tiff_requires_source():
    with pytest.raises(ValidationError, match="local_tiff strategy requires"):
        DatasetEntry(strategy="local_tiff")


def test_load_real_registry():
    """Test loading the actual configs/datasets.yml file."""
    registry_path = Path("configs/datasets.yml")
    if not registry_path.exists():
        pytest.skip("configs/datasets.yml not found")
    registry = load_registry(registry_path)
    assert "dem_3dep_10m" in registry.datasets
    assert "polaris_100m" in registry.datasets
    assert "nlcd_2021" in registry.datasets
