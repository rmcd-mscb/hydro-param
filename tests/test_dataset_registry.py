"""Tests for dataset_registry module constants and helpers."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from hydro_param.dataset_registry import VALID_CATEGORIES


def test_valid_categories_is_frozenset():
    """VALID_CATEGORIES contains the 8 registry category names."""
    assert isinstance(VALID_CATEGORIES, frozenset)
    assert VALID_CATEGORIES == frozenset(
        {
            "climate",
            "geology",
            "hydrography",
            "land_cover",
            "snow",
            "soils",
            "topography",
            "water_bodies",
        }
    )


def test_valid_categories_matches_yaml_files():
    """VALID_CATEGORIES matches actual YAML files in data/datasets/."""
    datasets_dir = Path(str(files("hydro_param").joinpath("data/datasets")))
    yaml_categories = {p.stem for p in datasets_dir.glob("*.yml")}
    assert VALID_CATEGORIES == yaml_categories
