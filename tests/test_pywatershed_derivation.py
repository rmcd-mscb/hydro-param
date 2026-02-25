"""Tests for pywatershed derivation plugin."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from shapely.geometry import LineString, Polygon

from hydro_param.derivations.pywatershed import PywatershedDerivation
from hydro_param.plugins import DerivationContext


@pytest.fixture()
def derivation() -> PywatershedDerivation:
    """Derivation plugin instance."""
    return PywatershedDerivation()


@pytest.fixture()
def sir_topography() -> xr.Dataset:
    """Synthetic SIR with topographic data (canonical SIR names)."""
    return xr.Dataset(
        {
            "elevation_m_mean": ("nhm_id", np.array([100.0, 500.0, 1500.0])),
            "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0, 30.0])),  # degrees
            "aspect_deg_mean": ("nhm_id", np.array([0.0, 90.0, 270.0])),  # degrees
        },
        coords={"nhm_id": [1, 2, 3]},
    )


@pytest.fixture()
def sir_landcover() -> xr.Dataset:
    """Synthetic SIR with land cover data (canonical SIR names)."""
    return xr.Dataset(
        {
            "land_cover": ("nhm_id", np.array([42, 71, 52])),  # Evergreen, Grass, Shrub
            "fctimp_pct_mean": ("nhm_id", np.array([5.0, 20.0, 0.0])),  # percent
            "tree_canopy_pct_mean": ("nhm_id", np.array([80.0, 10.0, 30.0])),  # percent
        },
        coords={"nhm_id": [1, 2, 3]},
    )


@pytest.fixture()
def sir_geometry() -> xr.Dataset:
    """Synthetic SIR with geometry data."""
    return xr.Dataset(
        {
            "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
            "hru_lat": ("nhm_id", np.array([42.0, 41.5, 43.0])),
        },
        coords={"nhm_id": [1, 2, 3]},
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
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_elev" in ds
        # 100m ~ 328.084 ft
        np.testing.assert_allclose(ds["hru_elev"].values[0], 328.084, atol=0.01)
        assert ds["hru_elev"].attrs["units"] == "feet"

    def test_slope_degrees_to_fraction(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_slope" in ds
        # tan(5deg) ~ 0.0875
        np.testing.assert_allclose(ds["hru_slope"].values[0], np.tan(np.radians(5.0)), atol=1e-6)
        # tan(30deg) ~ 0.5774
        np.testing.assert_allclose(ds["hru_slope"].values[2], np.tan(np.radians(30.0)), atol=1e-4)
        assert ds["hru_slope"].attrs["units"] == "decimal_fraction"

    def test_aspect_preserved(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_aspect" in ds
        np.testing.assert_array_equal(ds["hru_aspect"].values, [0.0, 90.0, 270.0])
        assert ds["hru_aspect"].attrs["units"] == "degrees"


class TestDeriveGeometry:
    """Tests for step 1: geometry extraction."""

    def test_area_m2_to_acres(
        self, derivation: PywatershedDerivation, sir_geometry: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_geometry, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_area" in ds
        # 4046856 m2 ~ 1000 acres
        np.testing.assert_allclose(ds["hru_area"].values[0], 1000.0, atol=1.0)
        assert ds["hru_area"].attrs["units"] == "acres"

    def test_lat_preserved(
        self, derivation: PywatershedDerivation, sir_geometry: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_geometry, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_lat" in ds
        np.testing.assert_array_equal(ds["hru_lat"].values, [42.0, 41.5, 43.0])


class TestDeriveLandcover:
    """Tests for step 4: land cover parameter derivation."""

    def test_nlcd_reclassification(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_landcover, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "cov_type" in ds
        # NLCD 42 = Evergreen -> cov_type 4 (coniferous)
        assert ds["cov_type"].values[0] == 4
        # NLCD 71 = Grassland -> cov_type 1 (grasses)
        assert ds["cov_type"].values[1] == 1
        # NLCD 52 = Shrub/Scrub -> cov_type 2 (shrubs)
        assert ds["cov_type"].values[2] == 2

    def test_tree_canopy_to_covden(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_landcover, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "covden_sum" in ds
        np.testing.assert_allclose(ds["covden_sum"].values, [0.8, 0.1, 0.3])

    def test_impervious_to_fraction(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_landcover, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_percent_imperv" in ds
        np.testing.assert_allclose(ds["hru_percent_imperv"].values, [0.05, 0.20, 0.0])

    def test_covden_fallback_without_canopy(self, derivation: PywatershedDerivation) -> None:
        """When tree_canopy is absent, covden_sum uses lookup fallback."""
        sir = xr.Dataset(
            {"land_cover": ("nhm_id", np.array([42, 71]))},
            coords={"nhm_id": [1, 2]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "covden_sum" in ds
        # Coniferous (4) -> 0.8, Grasses (1) -> 0.3
        np.testing.assert_allclose(ds["covden_sum"].values, [0.8, 0.3])


class TestApplyLookupTables:
    """Tests for step 8: lookup table application."""

    def test_interception_values(
        self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_landcover, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
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
        ctx = DerivationContext(sir=sir_landcover, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "imperv_stor_max" in ds
        np.testing.assert_allclose(ds["imperv_stor_max"].values, 0.03)

    def test_covden_win(self, derivation: PywatershedDerivation, sir_landcover: xr.Dataset) -> None:
        ctx = DerivationContext(sir=sir_landcover, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
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
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["tmax_allsnow"].item() == 32.0
        assert ds["den_init"].item() == 0.10
        assert ds["gwstor_init"].item() == 2.0
        assert ds["radmax"].item() == 0.8

    def test_defaults_not_overwritten(self, derivation: PywatershedDerivation) -> None:
        """If a default param is already derived from data, it's preserved."""
        sir = xr.Dataset(
            {"elevation_m_mean": ("nhm_id", np.array([100.0]))},
            coords={"nhm_id": [1]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # hru_elev was derived from data, not from defaults
        np.testing.assert_allclose(ds["hru_elev"].values, [328.084], atol=0.01)


class TestParameterOverrides:
    """Tests for user parameter overrides."""

    def test_override_scalar(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"tmax_allsnow": 30.0}}}
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id", config=config)
        ds = derivation.derive(ctx)
        assert ds["tmax_allsnow"].item() == 30.0

    def test_override_array(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"hru_elev": [100.0, 200.0, 300.0]}}}
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id", config=config)
        ds = derivation.derive(ctx)
        np.testing.assert_array_equal(ds["hru_elev"].values, [100.0, 200.0, 300.0])

    def test_override_new_param(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"custom_param": 42.0}}}
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id", config=config)
        ds = derivation.derive(ctx)
        assert ds["custom_param"].item() == 42.0


class TestHruCoordinates:
    """Tests for HRU coordinate carryover."""

    def test_nhru_coords_from_sir(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "nhru" in ds.coords
        np.testing.assert_array_equal(ds.coords["nhru"].values, [1, 2, 3])


class TestLandCoverMajorityFallback:
    """Tests for land_cover_majority variable name fallback."""

    def test_land_cover_majority_accepted(self, derivation: PywatershedDerivation) -> None:
        sir = xr.Dataset(
            {"land_cover_majority": ("nhm_id", np.array([42, 71]))},
            coords={"nhm_id": [1, 2]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "cov_type" in ds
        assert ds["cov_type"].values[0] == 4  # Evergreen -> coniferous


class TestOverrideDims:
    """Tests for override dimension assignment."""

    def test_new_1d_override_gets_nhru_dim(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"new_param": [1.0, 2.0, 3.0]}}}
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id", config=config)
        ds = derivation.derive(ctx)
        assert ds["new_param"].dims == ("nhru",)


class TestFullDerivation:
    """Integration tests with all foundation SIR data."""

    def test_all_foundation_params_present(
        self, derivation: PywatershedDerivation, sir_full: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_full, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
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
        ctx = DerivationContext(sir=sir_full, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "tmax_allsnow" in ds
        assert "den_init" in ds
        assert "gwstor_init" in ds


# ------------------------------------------------------------------
# Topology fixtures (synthetic 3-segment network)
# ------------------------------------------------------------------

# Network topology:
#   seg1 -> seg2 -> seg3 -> outlet
#   hru1 -> seg1, hru2 -> seg2, hru3 -> seg2


@pytest.fixture()
def synthetic_fabric() -> gpd.GeoDataFrame:
    """3-HRU fabric with hru_segment attribute."""
    return gpd.GeoDataFrame(
        {
            "nhm_id": [101, 102, 103],
            "hru_segment": [1, 2, 2],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture()
def synthetic_segments() -> gpd.GeoDataFrame:
    """3-segment network with tosegment attribute."""
    return gpd.GeoDataFrame(
        {
            "nhm_seg": [201, 202, 203],
            "tosegment": [2, 3, 0],  # seg1->seg2, seg2->seg3, seg3->outlet
        },
        geometry=[
            LineString([(0.5, 0.5), (1.0, 0.5)]),
            LineString([(1.0, 0.5), (2.0, 0.5)]),
            LineString([(2.0, 0.5), (3.0, 0.5)]),
        ],
        crs="EPSG:4326",
    )


@pytest.fixture()
def sir_minimal() -> xr.Dataset:
    """Minimal SIR for topology tests (3 HRUs, no physical data)."""
    return xr.Dataset(coords={"nhm_id": [101, 102, 103]})


class TestDeriveTopology:
    """Tests for step 2: topology extraction."""

    def test_tosegment_extraction(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: xr.Dataset,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "tosegment" in ds
        np.testing.assert_array_equal(ds["tosegment"].values, [2, 3, 0])
        assert ds["tosegment"].dims == ("nsegment",)

    def test_hru_segment_extraction(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: xr.Dataset,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "hru_segment" in ds
        np.testing.assert_array_equal(ds["hru_segment"].values, [1, 2, 2])
        assert ds["hru_segment"].dims == ("nhru",)

    def test_seg_length_computed(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: xr.Dataset,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "seg_length" in ds
        assert ds["seg_length"].dims == ("nsegment",)
        # All segments span ~0.5 to ~1.0 degrees longitude at ~0.5deg lat
        # Geodesic lengths should be positive and reasonable
        assert np.all(ds["seg_length"].values > 0)
        assert ds["seg_length"].attrs["units"] == "meters"

    def test_nsegment_coordinate(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: xr.Dataset,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "nsegment" in ds.coords
        np.testing.assert_array_equal(ds.coords["nsegment"].values, [201, 202, 203])

    def test_backward_compatible_without_topology(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """derive() without fabric/segments still works (no topology)."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "tosegment" not in ds
        assert "hru_segment" not in ds
        assert "seg_length" not in ds
        # But topographic params still present
        assert "hru_elev" in ds

    def test_seg_length_longer_for_longer_segment(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """Segments spanning more distance should have longer seg_length."""
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1, 2], "tosegment": [2, 0]},
            geometry=[
                LineString([(0.5, 0.5), (0.6, 0.5)]),  # short
                LineString([(0.6, 0.5), (2.0, 0.5)]),  # long
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert ds["seg_length"].values[1] > ds["seg_length"].values[0]


class TestTopologyValidation:
    """Tests for topology validation rules."""

    def test_self_loop_raises(self, derivation: PywatershedDerivation) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1], "hru_segment": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1], "tosegment": [1]},  # self-loop!
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        with pytest.raises(ValueError, match="self-loops"):
            derivation.derive(ctx)

    def test_no_outlet_raises(self, derivation: PywatershedDerivation) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1, 2], "tosegment": [2, 1]},  # cycle, no outlet
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        with pytest.raises(ValueError, match="No outlets"):
            derivation.derive(ctx)

    def test_hru_segment_out_of_range_raises(self, derivation: PywatershedDerivation) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1], "hru_segment": [5]},  # out of range
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        with pytest.raises(ValueError, match="hru_segment values out of range"):
            derivation.derive(ctx)

    def test_hru_segment_zero_is_valid(self, derivation: PywatershedDerivation) -> None:
        """hru_segment=0 means HRU doesn't drain to any segment."""
        sir = xr.Dataset(coords={"nhm_id": [1]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1], "hru_segment": [0]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert ds["hru_segment"].values[0] == 0

    def test_missing_tosegment_raises(self, derivation: PywatershedDerivation) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1], "hru_segment": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1]},  # no tosegment column
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        with pytest.raises(ValueError, match="missing required 'tosegment'"):
            derivation.derive(ctx)

    def test_missing_hru_segment_raises(self, derivation: PywatershedDerivation) -> None:
        sir = xr.Dataset(coords={"nhm_id": [1]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1]},  # no hru_segment column
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        with pytest.raises(ValueError, match="missing required 'hru_segment'"):
            derivation.derive(ctx)


# ------------------------------------------------------------------
# Integration tests with real pywatershed GIS data
# ------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GIS_DATA_DIR = _PROJECT_ROOT / "data" / "pywatershed_gis"
_DRB_DIR = _GIS_DATA_DIR / "drb_2yr"


def _is_real_file(path: Path) -> bool:
    """Check that a file exists and is not a Git LFS pointer."""
    if not path.exists():
        return False
    # LFS pointers are small text files starting with "version https://git-lfs"
    if path.stat().st_size < 200:
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
            if text.startswith("version https://git-lfs"):
                return False
        except (UnicodeDecodeError, ValueError):
            pass  # Binary file = real data
    return True


_HAS_DRB_DATA = (
    _is_real_file(_DRB_DIR / "nhru.gpkg")
    and _is_real_file(_DRB_DIR / "nsegment.gpkg")
    and _is_real_file(_DRB_DIR / "parameters_dis_both.nc")
    and _is_real_file(_DRB_DIR / "parameters_PRMSChannel.nc")
)


@pytest.mark.skipif(not _HAS_DRB_DATA, reason="pywatershed DRB GIS data not available")
class TestTopologyIntegrationDRB:
    """Integration tests with real DRB pywatershed GIS data."""

    @pytest.fixture()
    def drb_fabric(self) -> gpd.GeoDataFrame:
        return gpd.read_file(_DRB_DIR / "nhru.gpkg")

    @pytest.fixture()
    def drb_segments(self) -> gpd.GeoDataFrame:
        return gpd.read_file(_DRB_DIR / "nsegment.gpkg")

    @pytest.fixture()
    def drb_params_dis_both(self) -> xr.Dataset:  # type: ignore[misc]
        ds = xr.open_dataset(_DRB_DIR / "parameters_dis_both.nc")
        try:
            yield ds
        finally:
            ds.close()

    @pytest.fixture()
    def drb_params_channel(self) -> xr.Dataset:  # type: ignore[misc]
        ds = xr.open_dataset(_DRB_DIR / "parameters_PRMSChannel.nc")
        try:
            yield ds
        finally:
            ds.close()

    def test_drb_domain_sizes(
        self,
        drb_fabric: gpd.GeoDataFrame,
        drb_segments: gpd.GeoDataFrame,
    ) -> None:
        assert len(drb_fabric) == 765
        assert len(drb_segments) == 456

    def test_topology_from_fabric_columns(
        self,
        derivation: PywatershedDerivation,
        drb_fabric: gpd.GeoDataFrame,
        drb_segments: gpd.GeoDataFrame,
        drb_params_dis_both: xr.Dataset,
        drb_params_channel: xr.Dataset,
    ) -> None:
        """Topology extracted from GeoPackage columns matches reference params."""
        sir = xr.Dataset(coords={"nhm_id": drb_fabric["nhm_id"].values})
        seg_id_field = "nhm_seg" if "nhm_seg" in drb_segments.columns else "nsegment_v"

        ctx = DerivationContext(
            sir=sir,
            fabric=drb_fabric,
            segments=drb_segments,
            fabric_id_field="nhm_id",
            segment_id_field=seg_id_field,
        )
        ds = derivation.derive(ctx)

        # tosegment from GeoPackage should match reference param NetCDF
        ref_tosegment = drb_params_dis_both["tosegment"].values
        np.testing.assert_array_equal(ds["tosegment"].values, ref_tosegment)

        # hru_segment from GeoPackage should match reference param NetCDF
        ref_hru_segment = drb_params_channel["hru_segment"].values
        np.testing.assert_array_equal(ds["hru_segment"].values, ref_hru_segment)

        # seg_length from GeoPackage column should match reference
        ref_seg_length = drb_params_dis_both["seg_length"].values
        np.testing.assert_allclose(ds["seg_length"].values, ref_seg_length)

    def test_geodesic_seg_length_fallback(
        self,
        derivation: PywatershedDerivation,
        drb_fabric: gpd.GeoDataFrame,
        drb_segments: gpd.GeoDataFrame,
        drb_params_dis_both: xr.Dataset,
    ) -> None:
        """Geodesic seg_length (no column) correlates with reference."""
        # Drop seg_length column to force geodesic computation
        segs_no_length = drb_segments.drop(columns=["seg_length"])
        sir = xr.Dataset(coords={"nhm_id": drb_fabric["nhm_id"].values})
        seg_id_field = "nhm_seg" if "nhm_seg" in segs_no_length.columns else "nsegment_v"

        ctx = DerivationContext(
            sir=sir,
            fabric=drb_fabric,
            segments=segs_no_length,
            fabric_id_field="nhm_id",
            segment_id_field=seg_id_field,
        )
        ds = derivation.derive(ctx)

        ref_seg_length = drb_params_dis_both["seg_length"].values
        computed = ds["seg_length"].values
        assert np.all(computed > 0), "All segment lengths should be positive"
        correlation = np.corrcoef(computed, ref_seg_length)[0, 1]
        assert correlation > 0.9, f"Correlation too low: {correlation}"


# ------------------------------------------------------------------
# Geometry from fabric GeoDataFrame
# ------------------------------------------------------------------


class TestDeriveGeometryFromFabric:
    """Tests for step 1: geometry from fabric GeoDataFrame."""

    def test_area_from_fabric(self, derivation: PywatershedDerivation) -> None:
        """hru_area computed from fabric polygon geometry."""
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        # Two 1-degree squares near equator -- area should be > 0
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_area" in ds
        assert np.all(ds["hru_area"].values > 0)
        assert ds["hru_area"].attrs["units"] == "acres"

    def test_lat_from_fabric(self, derivation: PywatershedDerivation) -> None:
        """hru_lat computed from fabric centroid latitude."""
        sir = xr.Dataset(coords={"nhm_id": [1, 2]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2]},
            geometry=[
                Polygon([(0, 40), (1, 40), (1, 41), (0, 41)]),
                Polygon([(0, 42), (1, 42), (1, 43), (0, 43)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_lat" in ds
        np.testing.assert_allclose(ds["hru_lat"].values, [40.5, 42.5], atol=0.01)

    def test_fabric_geometry_overrides_sir(self, derivation: PywatershedDerivation) -> None:
        """When fabric is provided, SIR hru_area_m2/hru_lat are ignored."""
        sir = xr.Dataset(
            {
                "hru_area_m2": ("nhm_id", np.array([1.0, 1.0])),
                "hru_lat": ("nhm_id", np.array([0.0, 0.0])),
            },
            coords={"nhm_id": [1, 2]},
        )
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2]},
            geometry=[
                Polygon([(0, 40), (1, 40), (1, 41), (0, 41)]),
                Polygon([(0, 42), (1, 42), (1, 43), (0, 43)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # Should NOT be 1.0 (from SIR) -- should be computed from fabric
        assert np.all(ds["hru_area"].values > 1.0)
        # Should NOT be 0.0 (from SIR) -- should be ~40.5, ~42.5
        assert np.all(ds["hru_lat"].values > 30.0)

    def test_fallback_without_fabric(
        self, derivation: PywatershedDerivation, sir_geometry: xr.Dataset
    ) -> None:
        """Without fabric, falls back to SIR-based geometry."""
        ctx = DerivationContext(sir=sir_geometry, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_area" in ds
        np.testing.assert_allclose(ds["hru_area"].values[0], 1000.0, atol=1.0)


# ------------------------------------------------------------------
# Categorical fraction majority
# ------------------------------------------------------------------


class TestCategoricalFractionMajority:
    """Tests for computing majority class from categorical fractions."""

    def test_majority_from_lndcov_fractions(self, derivation: PywatershedDerivation) -> None:
        """Majority class extracted from lndcov_frac_ fraction columns."""
        sir = xr.Dataset(
            {
                "lndcov_frac_11": ("nhm_id", np.array([0.1, 0.0, 0.0])),
                "lndcov_frac_41": ("nhm_id", np.array([0.8, 0.1, 0.2])),
                "lndcov_frac_42": ("nhm_id", np.array([0.05, 0.0, 0.7])),
                "lndcov_frac_71": ("nhm_id", np.array([0.05, 0.9, 0.1])),
            },
            coords={"nhm_id": [1, 2, 3]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "cov_type" in ds
        # HRU 1: LndCov_41 (Deciduous Forest) highest -> cov_type 3
        assert ds["cov_type"].values[0] == 3
        # HRU 2: LndCov_71 (Grassland) highest -> cov_type 1
        assert ds["cov_type"].values[1] == 1
        # HRU 3: LndCov_42 (Evergreen Forest) highest -> cov_type 4
        assert ds["cov_type"].values[2] == 4

    def test_falls_back_to_single_land_cover(self, derivation: PywatershedDerivation) -> None:
        """When no fraction columns exist, falls back to land_cover."""
        sir = xr.Dataset(
            {"land_cover": ("nhm_id", np.array([42, 71]))},
            coords={"nhm_id": [1, 2]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "cov_type" in ds
        assert ds["cov_type"].values[0] == 4  # Evergreen -> coniferous


# ------------------------------------------------------------------
# merge_temporal_into_derived
# ------------------------------------------------------------------


class TestMergeTemporalIntoDerived:
    """Tests for the merge_temporal_into_derived() helper."""

    def test_renames_variables(self) -> None:
        from hydro_param.derivations.pywatershed import merge_temporal_into_derived

        derived = xr.Dataset(
            {"hru_elev": ("nhru", [100.0, 200.0])},
            coords={"nhru": [1, 2]},
        )
        temporal = {
            "gridmet": xr.Dataset(
                {
                    "pr": (("nhru", "time"), np.array([[1.0, 2.0], [3.0, 4.0]])),
                    "tmmx": (("nhru", "time"), np.array([[300.0, 301.0], [302.0, 303.0]])),
                    "tmmn": (("nhru", "time"), np.array([[280.0, 281.0], [282.0, 283.0]])),
                },
                coords={"nhru": [1, 2], "time": ["2020-01-01", "2020-01-02"]},
            )
        }

        result = merge_temporal_into_derived(
            derived,
            temporal,
            renames={"pr": "prcp", "tmmx": "tmax", "tmmn": "tmin"},
        )

        assert "prcp" in result
        assert "tmax" in result
        assert "tmin" in result
        assert "pr" not in result
        assert "tmmx" not in result

    def test_converts_units(self) -> None:
        from hydro_param.derivations.pywatershed import merge_temporal_into_derived

        derived = xr.Dataset(
            {"hru_elev": ("nhru", [100.0])},
            coords={"nhru": [1]},
        )
        temporal = {
            "gridmet": xr.Dataset(
                {
                    "tmax": (("nhru", "time"), np.array([[300.0]])),
                },
                coords={"nhru": [1], "time": ["2020-01-01"]},
            )
        }

        result = merge_temporal_into_derived(
            derived,
            temporal,
            conversions={"tmax": ("K", "C")},
        )

        # 300K = 26.85 degC
        np.testing.assert_allclose(result["tmax"].values[0, 0], 26.85, atol=0.01)

    def test_aligns_dimension(self) -> None:
        from hydro_param.derivations.pywatershed import merge_temporal_into_derived

        derived = xr.Dataset(
            {"hru_elev": ("nhru", [100.0, 200.0])},
            coords={"nhru": [1, 2]},
        )
        temporal = {
            "gridmet": xr.Dataset(
                {
                    "pr": (("nhm_id", "time"), np.array([[1.0], [2.0]])),
                },
                coords={"nhm_id": [1, 2], "time": ["2020-01-01"]},
            )
        }

        result = merge_temporal_into_derived(derived, temporal)

        assert "nhru" in result["pr"].dims

    def test_empty_temporal_is_noop(self) -> None:
        from hydro_param.derivations.pywatershed import merge_temporal_into_derived

        derived = xr.Dataset(
            {"hru_elev": ("nhru", [100.0])},
            coords={"nhru": [1]},
        )

        result = merge_temporal_into_derived(derived, {})

        assert list(result.data_vars) == ["hru_elev"]

    def test_concatenates_year_chunks_sorted(self) -> None:
        """Multi-year chunks are sorted by time before concatenation."""
        from hydro_param.derivations.pywatershed import merge_temporal_into_derived

        derived = xr.Dataset(
            {"hru_elev": ("nhru", [100.0])},
            coords={"nhru": [1]},
        )

        # Provide chunks out of order to verify sorting
        temporal = {
            "gridmet_2021": xr.Dataset(
                {"pr": (("nhru", "time"), [[3.0, 4.0]])},
                coords={
                    "nhru": [1],
                    "time": np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]"),
                },
            ),
            "gridmet_2020": xr.Dataset(
                {"pr": (("nhru", "time"), [[1.0, 2.0]])},
                coords={
                    "nhru": [1],
                    "time": np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]"),
                },
            ),
        }

        result = merge_temporal_into_derived(derived, temporal)

        assert "pr" in result
        # Should have 4 time steps concatenated in chronological order
        assert result.sizes["time"] == 4
        expected_values = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_equal(result["pr"].values[0], expected_values)


# ------------------------------------------------------------------
# Error handling: segment_id_field warning + fraction suffix debug
# ------------------------------------------------------------------


class TestSegmentIdFieldWarning:
    """Tests for segment_id_field fallback warning (item 4)."""

    def test_segment_id_field_missing_warns(
        self,
        derivation: PywatershedDerivation,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning logged when segment_id_field not in segments columns."""
        import logging

        sir = xr.Dataset(coords={"nhm_id": [1]})
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1], "hru_segment": [1]},
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"other_id": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        with caplog.at_level(logging.WARNING, logger="hydro_param.derivations.pywatershed"):
            derivation.derive(ctx)
        assert any("segment_id_field" in r.message for r in caplog.records)
        assert any("sequential IDs" in r.message for r in caplog.records)


class TestFractionSuffixDebugLog:
    """Tests for non-integer fraction suffix debug log (item 5)."""

    def test_fraction_suffix_non_integer_skipped(
        self,
        derivation: PywatershedDerivation,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Debug log emitted when fraction variable has non-integer suffix."""
        import logging

        sir = xr.Dataset(
            {
                "lndcov_frac_11": ("nhm_id", np.array([0.8, 0.2])),
                "lndcov_frac_42": ("nhm_id", np.array([0.2, 0.8])),
                "lndcov_frac_meta": ("nhm_id", np.array([0.0, 0.0])),
            },
            coords={"nhm_id": [1, 2]},
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        with caplog.at_level(logging.DEBUG, logger="hydro_param.derivations.pywatershed"):
            derivation.derive(ctx)
        assert any("Skipping variable" in r.message and "meta" in r.message for r in caplog.records)
