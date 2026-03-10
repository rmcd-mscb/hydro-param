"""Tests for dataset_registry module constants and helpers."""

from __future__ import annotations

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
