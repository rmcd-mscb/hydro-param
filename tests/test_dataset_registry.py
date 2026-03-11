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


# ---------------------------------------------------------------------------
# DerivedContinuousSpec tests
# ---------------------------------------------------------------------------

import pytest
from pydantic import ValidationError

from hydro_param.dataset_registry import DerivedContinuousSpec


class TestDerivedContinuousSpec:
    """Tests for the DerivedContinuousSpec model."""

    def test_valid_spec(self):
        spec = DerivedContinuousSpec(
            name="product",
            sources=["a", "b"],
            operation="multiply",
            align_to="a",
        )
        assert spec.name == "product"
        assert spec.operation == "multiply"
        assert spec.resampling_method == "nearest"

    def test_scale_factor_optional(self):
        spec = DerivedContinuousSpec(
            name="product",
            sources=["a", "b"],
            operation="divide",
            align_to="b",
            scale_factor=0.01,
        )
        assert spec.scale_factor == 0.01

    def test_rejects_single_source(self):
        with pytest.raises(ValidationError):
            DerivedContinuousSpec(
                name="bad",
                sources=["a"],
                operation="multiply",
                align_to="a",
            )

    def test_rejects_invalid_operation(self):
        with pytest.raises(ValidationError):
            DerivedContinuousSpec(
                name="bad",
                sources=["a", "b"],
                operation="power",
                align_to="a",
            )

    def test_align_to_must_be_in_sources(self):
        with pytest.raises(ValidationError):
            DerivedContinuousSpec(
                name="bad",
                sources=["a", "b"],
                operation="multiply",
                align_to="c",
            )

    def test_three_sources(self):
        spec = DerivedContinuousSpec(
            name="triple",
            sources=["a", "b", "c"],
            operation="add",
            align_to="b",
        )
        assert len(spec.sources) == 3
