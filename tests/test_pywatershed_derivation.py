"""Tests for pywatershed derivation plugin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydro_param.derivations.pywatershed import PywatershedDerivation

LOOKUP_TABLES_DIR = Path("configs/lookup_tables")


@pytest.fixture()
def derivation() -> PywatershedDerivation:
    """Derivation plugin with project lookup tables."""
    return PywatershedDerivation(lookup_tables_dir=LOOKUP_TABLES_DIR)


@pytest.fixture()
def sir_topography() -> xr.Dataset:
    """Synthetic SIR with topographic data (metric units)."""
    return xr.Dataset(
        {
            "elevation": ("hru_id", np.array([100.0, 500.0, 1500.0])),
            "slope": ("hru_id", np.array([5.0, 15.0, 30.0])),  # degrees
            "aspect": ("hru_id", np.array([0.0, 90.0, 270.0])),  # degrees
        },
        coords={"hru_id": [1, 2, 3]},
    )


@pytest.fixture()
def sir_landcover() -> xr.Dataset:
    """Synthetic SIR with land cover data."""
    return xr.Dataset(
        {
            "land_cover": ("hru_id", np.array([42, 71, 52])),  # Evergreen, Grass, Shrub
            "impervious": ("hru_id", np.array([5.0, 20.0, 0.0])),  # percent
            "tree_canopy": ("hru_id", np.array([80.0, 10.0, 30.0])),  # percent
        },
        coords={"hru_id": [1, 2, 3]},
    )


@pytest.fixture()
def sir_geometry() -> xr.Dataset:
    """Synthetic SIR with geometry data."""
    return xr.Dataset(
        {
            "hru_area_m2": ("hru_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            "hru_lat": ("hru_id", np.array([42.0, 41.5, 43.0])),
        },
        coords={"hru_id": [1, 2, 3]},
    )


@pytest.fixture()
def sir_full(
    sir_topography: xr.Dataset, sir_landcover: xr.Dataset, sir_geometry: xr.Dataset
) -> xr.Dataset:
    """Synthetic SIR with all foundation data."""
    return xr.merge([sir_topography, sir_landcover, sir_geometry])


class TestDeriveTopography:
    """Tests for step 3: topographic parameter derivation."""

    def test_elevation_m_to_ft(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_topography)
        assert "hru_elev" in ds
        # 100m ≈ 328.084 ft
        np.testing.assert_allclose(ds["hru_elev"].values[0], 328.084, atol=0.01)
        assert ds["hru_elev"].attrs["units"] == "feet"

    def test_slope_degrees_to_fraction(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_topography)
        assert "hru_slope" in ds
        # tan(5°) ≈ 0.0875
        np.testing.assert_allclose(ds["hru_slope"].values[0], np.tan(np.radians(5.0)), atol=1e-6)
        # tan(30°) ≈ 0.5774
        np.testing.assert_allclose(ds["hru_slope"].values[2], np.tan(np.radians(30.0)), atol=1e-4)
        assert ds["hru_slope"].attrs["units"] == "decimal_fraction"

    def test_aspect_preserved(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_topography)
        assert "hru_aspect" in ds
        np.testing.assert_array_equal(ds["hru_aspect"].values, [0.0, 90.0, 270.0])
        assert ds["hru_aspect"].attrs["units"] == "degrees"


class TestDeriveGeometry:
    """Tests for step 1: geometry extraction."""

    def test_area_m2_to_acres(
        self, derivation: PywatershedDerivation, sir_geometry: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_geometry)
        assert "hru_area" in ds
        # 4046856 m² ≈ 1000 acres
        np.testing.assert_allclose(ds["hru_area"].values[0], 1000.0, atol=1.0)
        assert ds["hru_area"].attrs["units"] == "acres"

    def test_lat_preserved(
        self, derivation: PywatershedDerivation, sir_geometry: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_geometry)
        assert "hru_lat" in ds
        np.testing.assert_array_equal(ds["hru_lat"].values, [42.0, 41.5, 43.0])


class TestDeriveLandcover:
    """Tests for step 4: land cover parameter derivation."""

    def test_nlcd_reclassification(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_landcover)
        assert "cov_type" in ds
        # NLCD 42 = Evergreen → cov_type 4 (coniferous)
        assert ds["cov_type"].values[0] == 4
        # NLCD 71 = Grassland → cov_type 1 (grasses)
        assert ds["cov_type"].values[1] == 1
        # NLCD 52 = Shrub/Scrub → cov_type 2 (shrubs)
        assert ds["cov_type"].values[2] == 2

    def test_tree_canopy_to_covden(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_landcover)
        assert "covden_sum" in ds
        np.testing.assert_allclose(ds["covden_sum"].values, [0.8, 0.1, 0.3])

    def test_impervious_to_fraction(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_landcover)
        assert "hru_percent_imperv" in ds
        np.testing.assert_allclose(ds["hru_percent_imperv"].values, [0.05, 0.20, 0.0])

    def test_covden_fallback_without_canopy(self, derivation: PywatershedDerivation) -> None:
        """When tree_canopy is absent, covden_sum uses lookup fallback."""
        sir = xr.Dataset(
            {"land_cover": ("hru_id", np.array([42, 71]))},
            coords={"hru_id": [1, 2]},
        )
        ds = derivation.derive(sir)
        assert "covden_sum" in ds
        # Coniferous (4) → 0.8, Grasses (1) → 0.3
        np.testing.assert_allclose(ds["covden_sum"].values, [0.8, 0.3])


class TestApplyLookupTables:
    """Tests for step 8: lookup table application."""

    def test_interception_values(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_landcover)
        # cov_type 4 (coniferous): srain=0.08, wrain=0.08, snow=0.06
        assert ds["srain_intcp"].values[0] == 0.08
        assert ds["wrain_intcp"].values[0] == 0.08
        assert ds["snow_intcp"].values[0] == 0.06
        # cov_type 1 (grasses): srain=0.05, wrain=0.05, snow=0.01
        assert ds["srain_intcp"].values[1] == 0.05
        assert ds["snow_intcp"].values[1] == 0.01

    def test_imperv_stor_max(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_landcover)
        assert "imperv_stor_max" in ds
        np.testing.assert_allclose(ds["imperv_stor_max"].values, 0.03)

    def test_covden_win(self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset) -> None:
        ds = derivation.derive(sir_landcover)
        assert "covden_win" in ds
        # coniferous (factor=1.0): 0.8 * 1.0 = 0.8
        np.testing.assert_allclose(ds["covden_win"].values[0], 0.8)
        # grasses (factor=0.5): 0.1 * 0.5 = 0.05
        np.testing.assert_allclose(ds["covden_win"].values[1], 0.05)
        # shrubs (factor=0.5): 0.3 * 0.5 = 0.15
        np.testing.assert_allclose(ds["covden_win"].values[2], 0.15)


class TestApplyDefaults:
    """Tests for step 13: default values."""

    def test_defaults_present(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_topography)
        assert ds["tmax_allsnow"].item() == 32.0
        assert ds["den_init"].item() == 0.10
        assert ds["gwstor_init"].item() == 2.0
        assert ds["radmax"].item() == 0.8

    def test_defaults_not_overwritten(self, derivation: PywatershedDerivation) -> None:
        """If a default param is already derived from data, it's preserved."""
        sir = xr.Dataset(
            {"elevation": ("hru_id", np.array([100.0]))},
            coords={"hru_id": [1]},
        )
        ds = derivation.derive(sir)
        # hru_elev was derived from data, not from defaults
        np.testing.assert_allclose(ds["hru_elev"].values, [328.084], atol=0.01)


class TestParameterOverrides:
    """Tests for user parameter overrides."""

    def test_override_scalar(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"tmax_allsnow": 30.0}}}
        ds = derivation.derive(sir_topography, config=config)
        assert ds["tmax_allsnow"].item() == 30.0

    def test_override_array(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"hru_elev": [100.0, 200.0, 300.0]}}}
        ds = derivation.derive(sir_topography, config=config)
        np.testing.assert_array_equal(ds["hru_elev"].values, [100.0, 200.0, 300.0])

    def test_override_new_param(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"custom_param": 42.0}}}
        ds = derivation.derive(sir_topography, config=config)
        assert ds["custom_param"].item() == 42.0


class TestHruCoordinates:
    """Tests for HRU coordinate carryover."""

    def test_nhru_coords_from_sir(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_topography)
        assert "nhru" in ds.coords
        np.testing.assert_array_equal(ds.coords["nhru"].values, [1, 2, 3])


class TestLandCoverMajorityFallback:
    """Tests for land_cover_majority variable name fallback."""

    def test_land_cover_majority_accepted(self, derivation: PywatershedDerivation) -> None:
        sir = xr.Dataset(
            {"land_cover_majority": ("hru_id", np.array([42, 71]))},
            coords={"hru_id": [1, 2]},
        )
        ds = derivation.derive(sir)
        assert "cov_type" in ds
        assert ds["cov_type"].values[0] == 4  # Evergreen → coniferous


class TestOverrideDims:
    """Tests for override dimension assignment."""

    def test_new_1d_override_gets_nhru_dim(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"new_param": [1.0, 2.0, 3.0]}}}
        ds = derivation.derive(sir_topography, config=config)
        assert ds["new_param"].dims == ("nhru",)


class TestFullDerivation:
    """Integration tests with all foundation SIR data."""

    def test_all_foundation_params_present(
        self, derivation: PywatershedDerivation, sir_full: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_full)
        expected = {
            "hru_elev",
            "hru_slope",
            "hru_aspect",
            "hru_area",
            "hru_lat",
            "cov_type",
            "covden_sum",
            "covden_win",
            "hru_percent_imperv",
            "srain_intcp",
            "wrain_intcp",
            "snow_intcp",
            "imperv_stor_max",
        }
        for param in expected:
            assert param in ds, f"Missing parameter: {param}"

    def test_defaults_included(
        self, derivation: PywatershedDerivation, sir_full: xr.Dataset
    ) -> None:
        ds = derivation.derive(sir_full)
        assert "tmax_allsnow" in ds
        assert "den_init" in ds
        assert "gwstor_init" in ds
