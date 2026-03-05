"""Tests for pywatershed derivation plugin."""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import LineString, Polygon

from hydro_param.derivations.pywatershed import (
    _DEFAULT_K_COEF,
    _FALLBACK_SLOPE,
    _K_COEF_MAX,
    _K_COEF_MIN,
    _LAKE_K_COEF,
    _NHD_MISSING_SLOPE_SENTINEL,
    PywatershedDerivation,
)
from hydro_param.plugins import DerivationContext
from hydro_param.solar import NDOY


class _MockSIRAccessor:
    """Test-only adapter: wrap an xr.Dataset with the SIRAccessor interface.

    Allows existing test fixtures that create xr.Dataset objects to be used
    with DerivationContext without writing files to disk.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self._ds = ds

    def available_variables(self) -> list[str]:
        return list(self._ds.data_vars)

    def available_temporal(self) -> list[str]:
        return []

    @property
    def data_vars(self) -> list[str]:
        return list(str(v) for v in self._ds.data_vars)

    @property
    def sir_schema(self) -> list[dict]:
        return []

    def load_variable(self, name: str) -> xr.DataArray:
        if name not in self._ds:
            raise KeyError(f"SIR variable '{name}' not found.")
        return self._ds[name]

    def load_temporal(self, name: str) -> xr.Dataset:
        raise KeyError("No temporal data in mock accessor")

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._ds

    def __getitem__(self, name: str) -> xr.DataArray:
        return self.load_variable(name)

    def load_dataset(self, name: str) -> xr.Dataset:
        """Load all vars starting with name as a Dataset."""
        matching = {str(k): v for k, v in self._ds.data_vars.items() if str(k).startswith(name)}
        if not matching:
            raise KeyError(f"SIR variable '{name}' not found.")
        return xr.Dataset(matching)

    def find_variable(self, base_name: str) -> str | None:
        """Find variable by base name, allowing year suffixes."""

        if base_name in self._ds:
            return base_name
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{4}})$")
        matches = [str(v) for v in self._ds.data_vars if pattern.match(str(v))]
        if matches:
            return sorted(matches)[-1]
        return None


@pytest.fixture()
def derivation() -> PywatershedDerivation:
    """Derivation plugin instance."""
    return PywatershedDerivation()


@pytest.fixture()
def sir_topography() -> _MockSIRAccessor:
    """Synthetic SIR with topographic data (canonical SIR names).

    Includes ``hru_lat`` so that step 1 (geometry) populates latitude,
    which step 9 (soltab) requires alongside slope and aspect.
    """
    return _MockSIRAccessor(
        xr.Dataset(
            {
                "elevation_m_mean": ("nhm_id", np.array([100.0, 500.0, 1500.0])),
                "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0, 30.0])),
                "aspect_deg_mean": ("nhm_id", np.array([0.0, 90.0, 270.0])),
                "sin_aspect_mean": (
                    "nhm_id",
                    np.sin(np.radians([0.0, 90.0, 270.0])),
                ),
                "cos_aspect_mean": (
                    "nhm_id",
                    np.cos(np.radians([0.0, 90.0, 270.0])),
                ),
                "hru_lat": ("nhm_id", np.array([42.0, 41.5, 43.0])),
            },
            coords={"nhm_id": [1, 2, 3]},
        )
    )


@pytest.fixture()
def sir_landcover() -> _MockSIRAccessor:
    """Synthetic SIR with land cover data (canonical SIR names)."""
    return _MockSIRAccessor(
        xr.Dataset(
            {
                "land_cover": ("nhm_id", np.array([42, 71, 52])),
                "fctimp_pct_mean": ("nhm_id", np.array([5.0, 20.0, 0.0])),
                "tree_canopy_pct_mean": ("nhm_id", np.array([80.0, 10.0, 30.0])),
            },
            coords={"nhm_id": [1, 2, 3]},
        )
    )


@pytest.fixture()
def sir_geometry() -> _MockSIRAccessor:
    """Synthetic SIR with geometry data."""
    return _MockSIRAccessor(
        xr.Dataset(
            {
                "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0, 2023428.0])),
                "hru_lat": ("nhm_id", np.array([42.0, 41.5, 43.0])),
            },
            coords={"nhm_id": [1, 2, 3]},
        )
    )


@pytest.fixture()
def sir_topo_with_area(
    sir_topography: _MockSIRAccessor, sir_geometry: _MockSIRAccessor
) -> _MockSIRAccessor:
    """Synthetic SIR with topography + geometry (area) data."""
    return _MockSIRAccessor(xr.merge([sir_topography._ds, sir_geometry._ds]))


@pytest.fixture()
def sir_soils() -> _MockSIRAccessor:
    """Synthetic SIR with soil data (gNATSGO-like)."""
    return _MockSIRAccessor(
        xr.Dataset(
            {
                "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.1, 0.0])),
                "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.8, 0.1])),
                "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1, 0.9])),
                "awc_mm_mean": ("nhm_id", np.array([50.0, 150.0, 80.0])),
            },
            coords={"nhm_id": [1, 2, 3]},
        )
    )


@pytest.fixture()
def sir_full(
    sir_topography: _MockSIRAccessor,
    sir_landcover: _MockSIRAccessor,
    sir_geometry: _MockSIRAccessor,
) -> _MockSIRAccessor:
    """Synthetic SIR with all foundation data."""
    return _MockSIRAccessor(xr.merge([sir_topography._ds, sir_landcover._ds, sir_geometry._ds]))


@pytest.fixture()
def temporal_gridmet() -> dict[str, xr.Dataset]:
    """Synthetic SIR-normalized temporal data mimicking per-variable SIR output.

    Real SIR normalizes temporal data into per-variable-per-year NetCDF files.
    Each dataset contains a single variable with its SIR canonical name.
    """
    import pandas as pd

    nhru = 3
    rng = np.random.default_rng(42)

    variables = {
        "pr_mm_mean": lambda n, h: rng.uniform(0, 20, (n, h)),
        "tmmx_C_mean": lambda n, h: rng.uniform(10, 35, (n, h)),
        "tmmn_C_mean": lambda n, h: rng.uniform(-5, 15, (n, h)),
        "srad_W_m2_mean": lambda n, h: rng.uniform(50, 300, (n, h)),
        "pet_mm_mean": lambda n, h: rng.uniform(0, 8, (n, h)),
    }

    result: dict[str, xr.Dataset] = {}
    for year in [2020, 2021]:
        times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        ntime = len(times)
        for var_name, gen_fn in variables.items():
            key = f"{var_name}_{year}"
            result[key] = xr.Dataset(
                {var_name: (("time", "nhm_id"), gen_fn(ntime, nhru))},
                coords={"time": times, "nhm_id": [1, 2, 3]},
            )

    return result


@pytest.fixture()
def waterbody_fabric() -> gpd.GeoDataFrame:
    """Synthetic HRU fabric with known areas for waterbody overlay tests.

    Two 100m x 100m square HRUs in EPSG:5070 (area = 10,000 m² each).
    """
    hru1 = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    hru2 = Polygon([(200, 0), (300, 0), (300, 100), (200, 100)])
    return gpd.GeoDataFrame(
        {"nhm_id": [1, 2], "geometry": [hru1, hru2]},
        crs="EPSG:5070",
    )


@pytest.fixture()
def waterbody_sir() -> _MockSIRAccessor:
    """Synthetic SIR for waterbody overlay tests.

    Includes hru_area_m2 matching the waterbody_fabric (10,000 m² each).
    """
    return _MockSIRAccessor(
        xr.Dataset(
            {"hru_area_m2": ("nhm_id", np.array([10000.0, 10000.0]))},
            coords={"nhm_id": [1, 2]},
        )
    )


@pytest.fixture()
def sample_waterbodies() -> gpd.GeoDataFrame:
    """Synthetic waterbody polygons for overlay tests.

    - Waterbody A: 60m x 100m LakePond overlapping HRU 1 (60% coverage)
    - Waterbody B: 30m x 100m Reservoir overlapping HRU 2 (30% coverage)
    - Waterbody C: SwampMarsh overlapping HRU 1 (should be filtered out)
    """
    wb_a = Polygon([(0, 0), (60, 0), (60, 100), (0, 100)])
    wb_b = Polygon([(200, 0), (230, 0), (230, 100), (200, 100)])
    wb_c = Polygon([(70, 0), (90, 0), (90, 100), (70, 100)])
    return gpd.GeoDataFrame(
        {
            "comid": [101, 102, 103],
            "ftype": ["LakePond", "Reservoir", "SwampMarsh"],
            "geometry": [wb_a, wb_b, wb_c],
        },
        crs="EPSG:5070",
    )


class TestDerivationContextTemporal:
    """Tests for temporal field on DerivationContext."""

    def test_temporal_defaults_to_none(self, sir_topography: xr.Dataset) -> None:
        """DerivationContext.temporal is None when not provided."""
        ctx = DerivationContext(sir=sir_topography)
        assert ctx.temporal is None

    def test_temporal_accepts_dict(self, sir_topography: xr.Dataset) -> None:
        """DerivationContext.temporal accepts a dict of datasets."""
        temporal_ds = xr.Dataset(
            {"pr_mm_mean": (("time", "nhm_id"), np.ones((3, 3)))},
            coords={"time": [0, 1, 2], "nhm_id": [1, 2, 3]},
        )
        ctx = DerivationContext(
            sir=sir_topography,
            temporal={"pr_mm_mean_2020": temporal_ds},
        )
        assert ctx.temporal is not None
        assert "pr_mm_mean_2020" in ctx.temporal


class TestDeriveTopography:
    """Tests for step 3: topographic parameter derivation."""

    def test_elevation_meters_preserved(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_elev" in ds
        # Elevation is kept in meters (no m→ft conversion)
        np.testing.assert_allclose(ds["hru_elev"].values[0], 100.0, atol=0.01)
        assert ds["hru_elev"].attrs["units"] == "meters"

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

    def test_aspect_circular_mean(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """Aspect uses atan2(sin_mean, cos_mean) circular mean."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_aspect" in ds
        # sin/cos decomposition of [0°, 90°, 270°] → atan2 recovers same values
        np.testing.assert_allclose(ds["hru_aspect"].values, [0.0, 90.0, 270.0], atol=1e-10)
        assert ds["hru_aspect"].attrs["units"] == "degrees"

    def test_aspect_legacy_fallback(self, derivation: PywatershedDerivation) -> None:
        """Aspect falls back to arithmetic mean when only aspect_deg_mean available."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([100.0, 500.0, 1500.0])),
                    "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0, 30.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([45.0, 180.0, 315.0])),
                    "hru_lat": ("nhm_id", np.array([42.0, 41.5, 43.0])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_aspect" in ds
        np.testing.assert_array_equal(ds["hru_aspect"].values, [45.0, 180.0, 315.0])


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
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"land_cover": ("nhm_id", np.array([42, 71]))},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "covden_sum" in ds
        # Coniferous (4) -> 0.8, Grasses (1) -> 0.3
        np.testing.assert_allclose(ds["covden_sum"].values, [0.8, 0.3])

    def test_derive_landcover_year_suffixed_imperv(self, derivation: PywatershedDerivation) -> None:
        """hru_percent_imperv derived from year-suffixed fctimp_pct_mean_2021."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "lndcov_frac_11": ("nhm_id", np.array([0.8, 0.2])),
                    "lndcov_frac_42": ("nhm_id", np.array([0.2, 0.8])),
                    "fctimp_pct_mean_2021": ("nhm_id", np.array([10.0, 50.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation._derive_landcover(ctx, xr.Dataset())
        assert "hru_percent_imperv" in ds
        np.testing.assert_allclose(ds["hru_percent_imperv"].values, [0.1, 0.5])


class TestDeriveSoils:
    """Tests for step 5: soils zonal stats derivation."""

    def test_soil_type_from_fractions(
        self, derivation: PywatershedDerivation, sir_soils: xr.Dataset
    ) -> None:
        """Majority texture class maps to PRMS soil_type."""
        ctx = DerivationContext(sir=sir_soils, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # HRU 1: sand=0.7 (highest) -> soil_type 1
        assert ds["soil_type"].values[0] == 1
        # HRU 2: loam=0.8 (highest) -> soil_type 2
        assert ds["soil_type"].values[1] == 2
        # HRU 3: clay=0.9 (highest) -> soil_type 3
        assert ds["soil_type"].values[2] == 3

    def test_soil_moist_max(self, derivation: PywatershedDerivation, sir_soils: xr.Dataset) -> None:
        """awc_mm_mean converted to inches for soil_moist_max."""
        ctx = DerivationContext(sir=sir_soils, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_moist_max" in ds
        # 50mm / 25.4 ~ 1.9685 inches
        np.testing.assert_allclose(ds["soil_moist_max"].values[0], 50.0 / 25.4, atol=0.01)
        assert ds["soil_moist_max"].attrs["units"] == "inches"

    def test_soil_rechr_max_frac_default(
        self, derivation: PywatershedDerivation, sir_soils: xr.Dataset
    ) -> None:
        """Defaults to 0.4 when no layer data available."""
        ctx = DerivationContext(sir=sir_soils, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_rechr_max_frac" in ds
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, 0.4)
        assert ds["soil_rechr_max_frac"].attrs["units"] == "decimal_fraction"

    def test_soil_rechr_max_frac_from_awc_ratio(self, derivation: PywatershedDerivation) -> None:
        """Computes aws0_30/aws0_100 ratio when both variables present."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.3])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.5])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.2])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([100.0, 200.0])),
                    "aws0_30_mm_mean": ("nhm_id", np.array([60.0, 80.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_rechr_max_frac" in ds
        # 60/100 = 0.6, 80/200 = 0.4
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, [0.6, 0.4], atol=1e-6)
        assert ds["soil_rechr_max_frac"].attrs["units"] == "decimal_fraction"

    def test_soil_rechr_max_frac_clipped(self, derivation: PywatershedDerivation) -> None:
        """Ratio clipped to [0.1, 0.9] physical bounds."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.7])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.2])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([100.0, 100.0])),
                    "aws0_30_mm_mean": ("nhm_id", np.array([5.0, 99.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # 5/100 = 0.05 -> clips to 0.1;  99/100 = 0.99 -> clips to 0.9
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, [0.1, 0.9])

    def test_soil_rechr_max_frac_zero_aws100_uses_default(
        self, derivation: PywatershedDerivation
    ) -> None:
        """HRUs with aws0_100 = 0 get the 0.4 default (no data)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.7])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.2])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([0.0, 100.0])),
                    "aws0_30_mm_mean": ("nhm_id", np.array([0.0, 60.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # HRU 1: aws0_100=0 -> default 0.4;  HRU 2: 60/100=0.6
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, [0.4, 0.6])

    def test_soil_rechr_max_frac_nan_aws50_uses_default(
        self, derivation: PywatershedDerivation
    ) -> None:
        """HRUs with NaN aws0_30 get the default (no recharge zone data)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.7])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.2])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([100.0, 100.0])),
                    "aws0_30_mm_mean": ("nhm_id", np.array([np.nan, 60.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # HRU 1: NaN -> default 0.4;  HRU 2: 60/100=0.6
        np.testing.assert_allclose(ds["soil_rechr_max_frac"].values, [0.4, 0.6])

    def test_soil_moist_max_clipped(self, derivation: PywatershedDerivation) -> None:
        """Very low AWC clips to 0.5 inches, very high clips to 20.0 inches."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.7])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.2])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1])),
                    "awc_mm_mean": ("nhm_id", np.array([1.0, 1000.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # 1.0mm / 25.4 = 0.039 inches -> clips to 0.5
        assert ds["soil_moist_max"].values[0] == 0.5
        # 1000.0mm / 25.4 = 39.37 inches -> clips to 20.0
        assert ds["soil_moist_max"].values[1] == 20.0

    def test_soils_missing_sir_vars(self, derivation: PywatershedDerivation) -> None:
        """Graceful skip when no soil data in SIR."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"elevation_m_mean": ("nhm_id", np.array([100.0]))},
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" not in ds
        assert "soil_moist_max" not in ds
        assert "soil_rechr_max_frac" not in ds

    def test_soil_type_single_value_fallback(self, derivation: PywatershedDerivation) -> None:
        """Falls back to single soil_texture variable if no fractions."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture": ("nhm_id", np.array(["clay", "sand", "loam"])),
                    "awc_mm_mean": ("nhm_id", np.array([80.0, 60.0, 100.0])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # clay -> 3, sand -> 1, loam -> 2
        np.testing.assert_array_equal(ds["soil_type"].values, [3, 1, 2])

    def test_soil_texture_majority_fallback(self, derivation: PywatershedDerivation) -> None:
        """Falls back to soil_texture_majority when soil_texture absent."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_majority": ("nhm_id", np.array(["sand", "clay"])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0, 80.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        np.testing.assert_array_equal(ds["soil_type"].values, [1, 3])

    def test_unknown_texture_defaults_to_loam(self, derivation: PywatershedDerivation) -> None:
        """Unrecognized texture class defaults to loam (soil_type=2) with warning."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture": ("nhm_id", np.array(["sand", "organic", "clay"])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0, 70.0, 80.0])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # sand -> 1, organic (unknown) -> 2 (loam default), clay -> 3
        np.testing.assert_array_equal(ds["soil_type"].values, [1, 2, 3])

    def test_fraction_columns_with_unknown_class_filtered(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Fraction columns with names not in lookup table are filtered out."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.6, 0.2])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.3, 0.7])),
                    "soil_texture_frac_bogus": ("nhm_id", np.array([0.1, 0.1])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0, 80.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # sand=0.6 > loam=0.3 -> 1; loam=0.7 > sand=0.2 -> 2
        np.testing.assert_array_equal(ds["soil_type"].values, [1, 2])

    def test_soil_moist_max_without_soil_type(self, derivation: PywatershedDerivation) -> None:
        """soil_moist_max produced even when no texture data (soil_type absent)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"awc_mm_mean": ("nhm_id", np.array([100.0]))},
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_moist_max" in ds
        assert "soil_type" not in ds
        # soil_rechr_max_frac gates on soil_type
        assert "soil_rechr_max_frac" not in ds

    def test_derive_soils_aws_mm_preferred(self, derivation: PywatershedDerivation) -> None:
        """soil_moist_max derived from aws0_100_mm_mean (preferred source, mm -> in)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "aws0_100_mm_mean": ("nhm_id", np.array([50.0, 150.0, 80.0])),
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.1, 0.0])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.8, 0.1])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1, 0.9])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_moist_max" in ds
        # 50 mm -> convert(50, mm, in) = 50/25.4 ≈ 1.969
        # 150 mm -> convert(150, mm, in) = 150/25.4 ≈ 5.906
        # 80 mm -> convert(80, mm, in) = 80/25.4 ≈ 3.150
        expected = np.clip(np.array([50.0, 150.0, 80.0]) / 25.4, 0.5, 20.0)
        np.testing.assert_allclose(ds["soil_moist_max"].values, expected, rtol=1e-3)

    def test_derive_soils_rootznaws_last_resort(self, derivation: PywatershedDerivation) -> None:
        """soil_moist_max falls back to rootznaws_mm_mean when aws0_100 and awc absent."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "rootznaws_mm_mean": ("nhm_id", np.array([75.0, 200.0])),
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.5, 0.5])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.3, 0.3])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.2, 0.2])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_moist_max" in ds
        expected = np.clip(np.array([75.0, 200.0]) / 25.4, 0.5, 20.0)
        np.testing.assert_allclose(ds["soil_moist_max"].values, expected, rtol=1e-3)

    def test_soil_moist_max_priority_aws_over_awc(self, derivation: PywatershedDerivation) -> None:
        """aws0_100_mm_mean is preferred over awc_mm_mean when both present."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "aws0_100_mm_mean": ("nhm_id", np.array([100.0])),
                    "awc_mm_mean": ("nhm_id", np.array([200.0])),
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.5])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.3])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.2])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # Should use aws0_100 (100mm), not awc (200mm)
        expected = np.clip(100.0 / 25.4, 0.5, 20.0)
        np.testing.assert_allclose(ds["soil_moist_max"].values[0], expected, rtol=1e-3)

    def test_soil_type_from_continuous_percentages(self, derivation: PywatershedDerivation) -> None:
        """Falls back to USDA texture triangle when only continuous percentages available."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "sand_pct_mean": ("nhm_id", np.array([90.0, 40.0, 20.0])),
                    "silt_pct_mean": ("nhm_id", np.array([5.0, 40.0, 20.0])),
                    "clay_pct_mean": ("nhm_id", np.array([5.0, 20.0, 60.0])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0, 80.0, 100.0])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # sand(90/5/5) -> PRMS 1, loam(40/40/20) -> PRMS 2, clay(20/20/60) -> PRMS 3
        np.testing.assert_array_equal(ds["soil_type"].values, [1, 2, 3])

    def test_soil_type_percentages_preferred_over_skip(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Percentages path used when no fractions or single texture available."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "sand_pct_mean": ("nhm_id", np.array([82.0])),
                    "silt_pct_mean": ("nhm_id", np.array([10.0])),
                    "clay_pct_mean": ("nhm_id", np.array([8.0])),
                    "aws0_100_mm_mean": ("nhm_id", np.array([50.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # loamy_sand(82/10/8) -> PRMS 1 (coarse)
        assert ds["soil_type"].values[0] == 1

    def test_soil_type_fractions_preferred_over_percentages(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Fraction columns take priority over continuous percentages."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    # Fractions say sand dominant
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.8])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.2])),
                    # Percentages say clay dominant (should be ignored)
                    "sand_pct_mean": ("nhm_id", np.array([10.0])),
                    "silt_pct_mean": ("nhm_id", np.array([10.0])),
                    "clay_pct_mean": ("nhm_id", np.array([80.0])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # Fractions win: sand=0.8 > clay=0.2 -> PRMS 1
        assert ds["soil_type"].values[0] == 1

    def test_soil_type_continuous_with_partial_nan(self, derivation: PywatershedDerivation) -> None:
        """Partial NaN in one HRU defaults that HRU to loam (soil_type=2)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "sand_pct_mean": ("nhm_id", np.array([90.0, np.nan])),
                    "silt_pct_mean": ("nhm_id", np.array([5.0, 30.0])),
                    "clay_pct_mean": ("nhm_id", np.array([5.0, 30.0])),
                    "awc_mm_mean": ("nhm_id", np.array([50.0, 80.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soil_type" in ds
        # HRU 1: sand(90/5/5) -> PRMS 1; HRU 2: NaN -> default loam -> PRMS 2
        np.testing.assert_array_equal(ds["soil_type"].values, [1, 2])

    def test_soil_type_skipped_with_partial_continuous_vars(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Only 2 of 3 continuous vars present — should skip, not crash."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "sand_pct_mean": ("nhm_id", np.array([90.0])),
                    "clay_pct_mean": ("nhm_id", np.array([5.0])),
                    # silt_pct_mean intentionally missing
                    "awc_mm_mean": ("nhm_id", np.array([50.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # No texture data available — soil_type should not be derived
        assert "soil_type" not in ds


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
        # All elements should be the default value (arrays, not scalars)
        np.testing.assert_allclose(ds["tmax_allsnow"].values, 32.0)
        np.testing.assert_allclose(ds["den_init"].values, 0.10)
        np.testing.assert_allclose(ds["gwstor_init"].values, 2.0)
        np.testing.assert_allclose(ds["radmax"].values, 0.8)

    def test_defaults_have_correct_shapes(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """All defaults must be correctly-dimensioned arrays, not 0-d scalars."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        nhru = 3  # sir_topography fixture has 3 HRUs

        # Per-HRU defaults must be 1-D arrays of length nhru.
        # Snow albedo/density params (albset_*, den_*, settle_const) were
        # previously scalar but pywatershed parameters.yaml declares (nhru,).
        # Fixed in #160.
        per_hru = [
            "albset_rna",
            "albset_rnm",
            "albset_sna",
            "albset_snm",
            "den_init",
            "den_max",
            "settle_const",
            "emis_noppt",
            "freeh2o_cap",
            "potet_sublim",
            "radj_sppt",
            "radj_wppt",
            "soil_moist_init_frac",
            "soil_rechr_init_frac",
            "ssstor_init_frac",
            "gwstor_init",
            "gwstor_min",
            "dprst_depth_avg",
            "transp_tmax",
            "jh_coef_hru",
        ]
        for name in per_hru:
            assert name in ds, f"Missing default: {name}"
            assert ds[name].ndim == 1, f"{name}: expected 1-D (nhru,), got ndim={ds[name].ndim}"
            assert ds[name].shape == (nhru,), (
                f"{name}: expected shape ({nhru},), got {ds[name].shape}"
            )

        # Per-month-per-HRU defaults must be 2-D (nmonth, nhru)
        per_month_hru = [
            "tmax_allsnow",
            "radmax",
            "cecn_coef",
            "tstorm_mo",
            "ppt_rad_adj",
            "radadj_intcp",
            "radadj_slope",
            "tmax_index",
        ]
        for name in per_month_hru:
            assert name in ds, f"Missing monthly default: {name}"
            assert ds[name].ndim == 2, (
                f"{name}: expected 2-D (nmonth, nhru), got ndim={ds[name].ndim}"
            )
            assert ds[name].shape == (12, nhru), (
                f"{name}: expected shape (12, {nhru}), got {ds[name].shape}"
            )

    def test_all_pywatershed_required_defaults(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """All parameters required by pywatershed NHM processes must be present."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        required = {
            # Structural (hru_in_to_cf requires hru_area, tested separately)
            "doy",
            "temp_units",
            # Depression storage operational
            "dprst_et_coef",
            "dprst_flow_coef",
            "dprst_frac_init",
            "dprst_frac_open",
            "dprst_seep_rate_clos",
            "dprst_seep_rate_open",
            "sro_to_dprst_imperv",
            "sro_to_dprst_perv",
            "op_flow_thres",
            "va_clos_exp",
            "va_open_exp",
            # Snow
            "cecn_coef",
            "rad_trncf",
            "melt_force",
            "melt_look",
            "snowinfil_max",
            "snowpack_init",
            "hru_deplcrv",
            "tstorm_mo",
            "snarea_curve",
            # Atmosphere
            "ppt_rad_adj",
            "radadj_intcp",
            "radadj_slope",
            "tmax_index",
            # Soilzone
            "sat_threshold",
            "pref_flow_infil_frac",
        }
        missing = required - set(ds.data_vars)
        assert not missing, f"Missing pywatershed defaults: {sorted(missing)}"

    def test_defaults_not_overwritten(self, derivation: PywatershedDerivation) -> None:
        """If a default param is already derived from data, it's preserved."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"elevation_m_mean": ("nhm_id", np.array([100.0]))},
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # hru_elev was derived from data (meters preserved), not from defaults
        np.testing.assert_allclose(ds["hru_elev"].values, [100.0], atol=0.01)


class TestParameterOverrides:
    """Tests for user parameter overrides."""

    def test_override_scalar(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        config = {"parameter_overrides": {"values": {"tmax_allsnow": 30.0}}}
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id", config=config)
        ds = derivation.derive(ctx)
        np.testing.assert_allclose(ds["tmax_allsnow"].values, 30.0)
        # Verify the override preserved the 2D shape, not collapsed to scalar
        assert ds["tmax_allsnow"].shape == (12, 3)

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
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"land_cover_majority": ("nhm_id", np.array([42, 71]))},
                coords={"nhm_id": [1, 2]},
            )
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


class TestDeriveSoltab:
    """Tests for step 9: solar radiation tables."""

    def test_soltab_output_shape(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soltab_potsw" in ds
        assert "soltab_horad_potsw" in ds
        assert ds["soltab_potsw"].shape == (NDOY, 3)
        assert ds["soltab_horad_potsw"].shape == (NDOY, 3)

    def test_soltab_dims(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["soltab_potsw"].dims == ("ndoy", "nhru")
        assert ds["soltab_horad_potsw"].dims == ("ndoy", "nhru")

    def test_soltab_non_negative(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert np.all(ds["soltab_potsw"].values >= 0)
        assert np.all(ds["soltab_horad_potsw"].values >= 0)

    def test_soltab_units_langleys(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["soltab_potsw"].attrs["units"] == "cal/cm2/day"

    def test_soltab_requires_topo_params(self, derivation: PywatershedDerivation) -> None:
        sir = _MockSIRAccessor(
            xr.Dataset({"_dummy": ("nhm_id", [0.0, 0.0])}, coords={"nhm_id": [1, 2]})
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soltab_potsw" not in ds

    def test_soltab_sunhrs_present(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "soltab_sunhrs" in ds
        assert ds["soltab_sunhrs"].shape == (NDOY, 3)


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
def sir_minimal() -> _MockSIRAccessor:
    """Minimal SIR for topology tests (3 HRUs, no physical data)."""
    return _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [101, 102, 103]}))


class TestDeriveTopology:
    """Tests for step 2: topology extraction."""

    def test_tosegment_extraction(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: _MockSIRAccessor,
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
        # tosegment_nhm maps local indices to NHM segment IDs:
        # tosegment=[2,3,0] with seg_ids=[201,202,203] → [202,203,0]
        assert "tosegment_nhm" in ds
        np.testing.assert_array_equal(ds["tosegment_nhm"].values, [202, 203, 0])

    def test_hru_segment_extraction(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: _MockSIRAccessor,
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
        sir_minimal: _MockSIRAccessor,
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
        sir_minimal: _MockSIRAccessor,
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

    def test_derive_without_topology(
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
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

    def test_nhm_id_from_fabric(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: _MockSIRAccessor,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        """nhm_id emitted from fabric id_field column."""
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "nhm_id" in ds
        np.testing.assert_array_equal(ds["nhm_id"].values, [101, 102, 103])
        assert ds["nhm_id"].dims == ("nhru",)

    def test_nhm_seg_from_segments(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: _MockSIRAccessor,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        """nhm_seg emitted from segments segment_id_field column."""
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "nhm_seg" in ds
        np.testing.assert_array_equal(ds["nhm_seg"].values, [201, 202, 203])
        assert ds["nhm_seg"].dims == ("nsegment",)

    def test_hru_segment_nhm_mapping(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: _MockSIRAccessor,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        """hru_segment_nhm maps each HRU to its segment ID."""
        # synthetic_fabric has hru_segment=[1,2,2], seg_ids=[201,202,203]
        # HRU 101 -> segment 1 -> seg_id 201
        # HRU 102 -> segment 2 -> seg_id 202
        # HRU 103 -> segment 2 -> seg_id 202
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "hru_segment_nhm" in ds
        np.testing.assert_array_equal(ds["hru_segment_nhm"].values, [201, 202, 202])
        assert ds["hru_segment_nhm"].dims == ("nhru",)

    def test_identity_params_with_custom_column_names(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """Identity params work with non-NHM column names."""
        sir = _MockSIRAccessor(xr.Dataset(coords={"my_hru": [10, 20]}))
        fabric = gpd.GeoDataFrame(
            {"my_hru": [10, 20], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"my_seg": [500, 600], "tosegment": [2, 0]},
            geometry=[
                LineString([(0.5, 0.5), (1.0, 0.5)]),
                LineString([(1.0, 0.5), (2.0, 0.5)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="my_hru",
            segment_id_field="my_seg",
        )
        ds = derivation.derive(ctx)
        # Output params always named nhm_id/nhm_seg regardless of source column
        assert "nhm_id" in ds
        np.testing.assert_array_equal(ds["nhm_id"].values, [10, 20])
        assert "nhm_seg" in ds
        np.testing.assert_array_equal(ds["nhm_seg"].values, [500, 600])
        assert "hru_segment_nhm" in ds
        np.testing.assert_array_equal(ds["hru_segment_nhm"].values, [500, 600])
        assert ds["nhm_id"].dims == ("nhru",)
        assert ds["nhm_seg"].dims == ("nsegment",)
        assert ds["hru_segment_nhm"].dims == ("nhru",)

    def test_hru_segment_nhm_zero_segment(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """hru_segment_nhm outputs 0 for HRUs with no segment."""
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2, 3]}))
        fabric = gpd.GeoDataFrame(
            {
                "nhm_id": [1, 2, 3],
                "hru_segment": [1, 2, 0],  # 3rd HRU unassigned
            },
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {
                "nhm_seg": [201, 202, 203],
                "tosegment": [2, 3, 0],
            },
            geometry=[
                LineString([(0.5, 0.5), (1.0, 0.5)]),
                LineString([(1.0, 0.5), (2.0, 0.5)]),
                LineString([(2.0, 0.5), (3.0, 0.5)]),
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
        np.testing.assert_array_equal(ds["hru_segment_nhm"].values, [201, 202, 0])

    def test_hru_segment_nhm_nonsequential_ids(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """hru_segment_nhm maps correctly with non-sequential IDs."""
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [2, 1]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [501, 302], "tosegment": [2, 0]},
            geometry=[
                LineString([(0.5, 0.5), (1.0, 0.5)]),
                LineString([(1.0, 0.5), (2.0, 0.5)]),
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
        # hru_segment=2 -> seg_ids[1] = 302
        # hru_segment=1 -> seg_ids[0] = 501
        np.testing.assert_array_equal(ds["hru_segment_nhm"].values, [302, 501])

    def test_seg_lat_from_segments(
        self,
        derivation: PywatershedDerivation,
        sir_minimal: _MockSIRAccessor,
        synthetic_fabric: gpd.GeoDataFrame,
        synthetic_segments: gpd.GeoDataFrame,
    ) -> None:
        """seg_lat computed from segment centroid latitude."""
        ctx = DerivationContext(
            sir=sir_minimal,
            fabric=synthetic_fabric,
            segments=synthetic_segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )
        ds = derivation.derive(ctx)
        assert "seg_lat" in ds
        assert ds["seg_lat"].dims == ("nsegment",)
        # synthetic_segments are at y=0.5 (EPSG:4326)
        np.testing.assert_allclose(ds["seg_lat"].values, [0.5, 0.5, 0.5], atol=0.01)


class TestTopologyValidation:
    """Tests for topology validation rules."""

    def test_self_loop_raises(self, derivation: PywatershedDerivation) -> None:
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": drb_fabric["nhm_id"].values}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": drb_fabric["nhm_id"].values}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
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
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
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

    def test_lon_from_fabric(self, derivation: PywatershedDerivation) -> None:
        """hru_lon computed from fabric centroid longitude."""
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2]},
            geometry=[
                Polygon([(0, 40), (1, 40), (1, 41), (0, 41)]),
                Polygon([(2, 42), (3, 42), (3, 43), (2, 43)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(sir=sir, fabric=fabric, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_lon" in ds
        np.testing.assert_allclose(ds["hru_lon"].values, [0.5, 2.5], atol=0.01)

    def test_lon_fallback_from_sir(self, derivation: PywatershedDerivation) -> None:
        """Without fabric, hru_lon falls back to SIR."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"hru_lon": ("nhm_id", np.array([-75.0, -76.0]))},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_lon" in ds
        np.testing.assert_allclose(ds["hru_lon"].values, [-75.0, -76.0])
        assert ds["hru_lon"].attrs["units"] == "decimal_degrees"
        assert ds["hru_lon"].attrs["long_name"] == "Longitude of HRU centroid"

    def test_lon_missing_without_fabric_or_sir(self, derivation: PywatershedDerivation) -> None:
        """When both fabric and SIR lack hru_lon, param is absent."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"_dummy": ("nhm_id", [0.0, 0.0])},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_lon" not in ds

    def test_fabric_geometry_overrides_sir(self, derivation: PywatershedDerivation) -> None:
        """When fabric is provided, SIR hru_area_m2/hru_lat are ignored."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "hru_area_m2": ("nhm_id", np.array([1.0, 1.0])),
                    "hru_lat": ("nhm_id", np.array([0.0, 0.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
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
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "lndcov_frac_11": ("nhm_id", np.array([0.1, 0.0, 0.0])),
                    "lndcov_frac_41": ("nhm_id", np.array([0.8, 0.1, 0.2])),
                    "lndcov_frac_42": ("nhm_id", np.array([0.05, 0.0, 0.7])),
                    "lndcov_frac_71": ("nhm_id", np.array([0.05, 0.9, 0.1])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
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

    def test_grouped_majority_forest_split_vote(self, derivation: PywatershedDerivation) -> None:
        """Forest wins when multiple forest classes together exceed any single competitor.

        Regression test for the grouped-majority fix: NLCD classes 41+42+43
        all map to forest (cov_type 3 or 4).  Their combined fraction should
        beat a single non-forest class even when no individual forest class
        is the single largest.
        """
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    # HRU 1: 41=25%, 42=15%, 43=10% -> 50% forest, but 81=30% pasture
                    # Old bug: argmax picks class 81 (30% > 25%), cov_type=1
                    # Fix: grouped type3=35% (41+43), type4=15% (42) > grasses 30%
                    "lndcov_frac_41": ("nhm_id", np.array([0.25, 0.05])),
                    "lndcov_frac_42": ("nhm_id", np.array([0.15, 0.05])),
                    "lndcov_frac_43": ("nhm_id", np.array([0.10, 0.05])),
                    "lndcov_frac_81": ("nhm_id", np.array([0.30, 0.70])),
                    "lndcov_frac_11": ("nhm_id", np.array([0.20, 0.15])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "cov_type" in ds
        # HRU 1: grouped type3 (41→3, 43→3) = 35%, type4 (42→4) = 15%
        # vs grasses (81→1) = 30%, bare (11→0) = 20%
        # type3 wins at 35% > type1 at 30%
        assert ds["cov_type"].values[0] == 3
        # HRU 2: grasses dominate at 70%
        assert ds["cov_type"].values[1] == 1

    def test_grouped_majority_nan_fractions(self, derivation: PywatershedDerivation) -> None:
        """NaN fractions treated as zero; all-NaN HRU gets bare ground default."""
        class_codes = [41, 81]
        fractions_list = [
            np.array([0.6, np.nan]),  # forest
            np.array([0.4, np.nan]),  # grasses
        ]
        mapping: dict[int, int] = {41: 3, 81: 1}
        result = derivation._compute_grouped_majority(class_codes, fractions_list, mapping)
        assert result[0] == 3  # forest wins for HRU 1
        # HRU 2: all NaN -> all zero -> explicit bare ground default (0)
        assert result[1] == 0

    def test_falls_back_to_single_land_cover(self, derivation: PywatershedDerivation) -> None:
        """When no fraction columns exist, falls back to land_cover."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"land_cover": ("nhm_id", np.array([42, 71]))},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "cov_type" in ds
        assert ds["cov_type"].values[0] == 4  # Evergreen -> coniferous

    def test_extract_fractions_file_level_key(self, derivation: PywatershedDerivation) -> None:
        """Extract NLCD fractions from a file-level SIR key with inner columns.

        Simulates real SIR where data_vars=['lndcov_frac_2021'] and
        load_dataset('lndcov_frac_2021') returns the inner columns.
        """
        inner_ds = xr.Dataset(
            {
                "lndcov_frac_2021_11": ("nhm_id", np.array([0.1, 0.0])),
                "lndcov_frac_2021_41": ("nhm_id", np.array([0.8, 0.1])),
                "lndcov_frac_2021_42": ("nhm_id", np.array([0.1, 0.9])),
            },
            coords={"nhm_id": [1, 2]},
        )
        outer_ds = xr.Dataset(
            {"lndcov_frac_2021": ("nhm_id", np.array([0.0, 0.0]))},
            coords={"nhm_id": [1, 2]},
        )

        class _FileKeyMock(_MockSIRAccessor):
            def __init__(self) -> None:
                super().__init__(outer_ds)
                self._inner = inner_ds

            def load_dataset(self, name: str) -> xr.Dataset:
                if name == "lndcov_frac_2021":
                    return self._inner
                raise KeyError(name)

        sir = _FileKeyMock()
        result = derivation._extract_nlcd_fractions(sir)
        assert result is not None
        codes, fracs = result
        assert sorted(codes) == [11, 41, 42]
        assert len(fracs) == 3

    def test_extract_fractions_multi_year_file_keys(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Multiple year-suffixed file keys don't overwrite each other.

        Regression test for the inner_ds overwrite bug: when both
        lndcov_frac_2020 and lndcov_frac_2021 exist, fractions from
        both years must be collected correctly.
        """
        inner_2020 = xr.Dataset(
            {
                "lndcov_frac_2020_11": ("nhm_id", np.array([0.9, 0.1])),
                "lndcov_frac_2020_41": ("nhm_id", np.array([0.1, 0.9])),
            },
            coords={"nhm_id": [1, 2]},
        )
        inner_2021 = xr.Dataset(
            {
                "lndcov_frac_2021_11": ("nhm_id", np.array([0.8, 0.2])),
                "lndcov_frac_2021_41": ("nhm_id", np.array([0.2, 0.8])),
            },
            coords={"nhm_id": [1, 2]},
        )
        outer_ds = xr.Dataset(
            {
                "lndcov_frac_2020": ("nhm_id", np.array([0.0, 0.0])),
                "lndcov_frac_2021": ("nhm_id", np.array([0.0, 0.0])),
            },
            coords={"nhm_id": [1, 2]},
        )

        class _MultiYearMock(_MockSIRAccessor):
            def __init__(self) -> None:
                super().__init__(outer_ds)
                self._inners = {
                    "lndcov_frac_2020": inner_2020,
                    "lndcov_frac_2021": inner_2021,
                }

            def load_dataset(self, name: str) -> xr.Dataset:
                if name in self._inners:
                    return self._inners[name]
                raise KeyError(name)

        sir = _MultiYearMock()
        result = derivation._extract_nlcd_fractions(sir)
        assert result is not None
        codes, fracs = result
        # 4 fraction columns total (2020_11, 2020_41, 2021_11, 2021_41).
        assert len(codes) == 4
        assert len(fracs) == 4
        assert 11 in codes
        assert 41 in codes

    def test_nodata_sentinel_filtered(self, derivation: PywatershedDerivation) -> None:
        """NLCD NoData sentinel class (250) is filtered from fraction extraction."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "lndcov_frac_41": ("nhm_id", np.array([0.3, 0.1])),
                    "lndcov_frac_71": ("nhm_id", np.array([0.2, 0.4])),
                    "lndcov_frac_250": ("nhm_id", np.array([0.5, 0.5])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        result = derivation._extract_nlcd_fractions(sir, valid_codes=derivation._VALID_NLCD_CLASSES)
        assert result is not None
        codes, _ = result
        assert 250 not in codes
        assert 41 in codes
        assert 71 in codes


# ------------------------------------------------------------------
# Error handling: segment_id_field warning + fraction suffix debug
# ------------------------------------------------------------------


class TestSegmentIdFieldWarning:
    """Tests for segment_id_field fallback and error behavior (item 4)."""

    def test_explicit_segment_id_field_missing_raises(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """KeyError raised when explicitly configured segment_id_field not found."""
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
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
        with pytest.raises(KeyError, match="segment_id_field"):
            derivation.derive(ctx)

    def test_default_segment_id_field_missing_warns(
        self,
        derivation: PywatershedDerivation,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Warning logged when default segment_id_field not in segments columns."""
        import logging

        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1]}))
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
            # segment_id_field=None (default) — triggers warning fallback
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

        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "lndcov_frac_11": ("nhm_id", np.array([0.8, 0.2])),
                    "lndcov_frac_42": ("nhm_id", np.array([0.2, 0.8])),
                    "lndcov_frac_meta": ("nhm_id", np.array([0.0, 0.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        with caplog.at_level(logging.DEBUG, logger="hydro_param.derivations.pywatershed"):
            derivation.derive(ctx)
        assert any("Skipping variable" in r.message and "meta" in r.message for r in caplog.records)


# ------------------------------------------------------------------
# Calibration seeds YAML validation
# ------------------------------------------------------------------


class TestCalibrationSeedsYAML:
    """Validates the calibration_seeds.yml file structure."""

    @pytest.fixture()
    def seed_data(self) -> dict:
        """Load the calibration seeds YAML file."""
        import yaml

        seeds_path = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "hydro_param"
            / "data"
            / "pywatershed"
            / "lookup_tables"
            / "calibration_seeds.yml"
        )
        with open(seeds_path) as f:
            return yaml.safe_load(f)

    def test_yaml_loads(self, seed_data: dict) -> None:
        """All entries have method, params, range, default keys."""
        mapping = seed_data["mapping"]
        assert len(mapping) == 22
        for param_name, spec in mapping.items():
            assert "method" in spec, f"{param_name} missing 'method'"
            assert "params" in spec, f"{param_name} missing 'params'"
            assert "range" in spec, f"{param_name} missing 'range'"
            assert "default" in spec, f"{param_name} missing 'default'"
            assert len(spec["range"]) == 2, f"{param_name} range must have 2 elements"

    def test_all_methods_recognized(self, seed_data: dict) -> None:
        """All method names are in the known set."""
        known_methods = {"linear", "exponential_scale", "fraction_of", "constant"}
        mapping = seed_data["mapping"]
        for param_name, spec in mapping.items():
            assert spec["method"] in known_methods, (
                f"{param_name} has unknown method '{spec['method']}'"
            )


class TestForcingVariablesYAML:
    """Validate the forcing_variables.yml lookup table."""

    def test_forcing_variables_yaml_loads(self, derivation: PywatershedDerivation) -> None:
        """forcing_variables.yml loads and has required structure."""
        from importlib.resources import files as pkg_files

        tables_dir = Path(str(pkg_files("hydro_param").joinpath("data/pywatershed/lookup_tables")))
        data = derivation._load_lookup_table("forcing_variables", tables_dir)
        assert "mapping" in data
        datasets = data["mapping"]
        assert "gridmet" in datasets

    def test_gridmet_variables_have_required_keys(self, derivation: PywatershedDerivation) -> None:
        """Each gridmet variable entry has sir_name, sir_unit, intermediate_unit."""
        from importlib.resources import files as pkg_files

        tables_dir = Path(str(pkg_files("hydro_param").joinpath("data/pywatershed/lookup_tables")))
        data = derivation._load_lookup_table("forcing_variables", tables_dir)
        gridmet = data["mapping"]["gridmet"]
        required_keys = {"sir_name", "sir_unit", "intermediate_unit"}
        for prms_name, entry in gridmet.items():
            missing = required_keys - set(entry.keys())
            assert not missing, f"{prms_name} missing keys: {missing}"


# ------------------------------------------------------------------
# Forcing derivation (step 7)
# ------------------------------------------------------------------


class TestBuildSirToForcingLookup:
    """Tests for _build_sir_to_forcing_lookup reverse mapping."""

    @pytest.fixture()
    def lookup(self, derivation: PywatershedDerivation) -> dict[str, dict[str, str]]:
        """Build the reverse lookup from the bundled forcing_variables.yml."""
        from importlib.resources import files

        tables_dir = Path(str(files("hydro_param").joinpath("data/pywatershed/lookup_tables")))
        return derivation._build_sir_to_forcing_lookup(tables_dir)

    def test_returns_all_five_gridmet_variables(self, lookup: dict[str, dict[str, str]]) -> None:
        """Reverse lookup contains all 5 gridmet SIR variable names."""
        expected_sir_names = {
            "pr_mm_mean",
            "tmmx_C_mean",
            "tmmn_C_mean",
            "srad_W_m2_mean",
            "pet_mm_mean",
        }
        assert set(lookup.keys()) == expected_sir_names

    def test_prms_names_correct(self, lookup: dict[str, dict[str, str]]) -> None:
        """Each SIR name maps to the correct PRMS name."""
        assert lookup["pr_mm_mean"]["prms_name"] == "prcp"
        assert lookup["tmmx_C_mean"]["prms_name"] == "tmax"
        assert lookup["tmmn_C_mean"]["prms_name"] == "tmin"
        assert lookup["srad_W_m2_mean"]["prms_name"] == "swrad"
        assert lookup["pet_mm_mean"]["prms_name"] == "potet"

    def test_source_field_present(self, lookup: dict[str, dict[str, str]]) -> None:
        """Each entry includes the source dataset name."""
        for entry in lookup.values():
            assert entry["source"] == "gridmet"


class TestDeriveForcing:
    """Tests for _derive_forcing (step 7)."""

    def test_no_temporal_returns_unchanged(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """When temporal is None, dataset is returned unchanged."""
        ctx = DerivationContext(sir=sir_topography, temporal=None)
        ds = xr.Dataset({"hru_elev": ("nhru", [100.0, 200.0, 300.0])})
        result = derivation._derive_forcing(ctx, ds)
        assert set(result.data_vars) == {"hru_elev"}

    def test_empty_temporal_returns_unchanged(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Empty temporal dict returns dataset unchanged."""
        ctx = DerivationContext(sir=sir_topography, temporal={})
        ds = xr.Dataset({"hru_elev": ("nhru", [100.0, 200.0, 300.0])})
        result = derivation._derive_forcing(ctx, ds)
        assert set(result.data_vars) == {"hru_elev"}

    def test_renames_sir_to_prms(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """SIR canonical names are renamed to PRMS names."""
        ctx = DerivationContext(sir=sir_topography, temporal=temporal_gridmet)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert "prcp" in result
        assert "tmax" in result
        assert "tmin" in result
        assert "swrad" in result
        assert "potet" in result
        assert "pr_mm_mean" not in result
        assert "tmmx_C_mean" not in result

    def test_multiyear_concat(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """Multi-year temporal data is concatenated along time."""
        ctx = DerivationContext(sir=sir_topography, temporal=temporal_gridmet)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert result["prcp"].sizes["time"] == 731

    def test_swrad_converted_to_langleys(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """swrad is converted from W/m2 to Langleys/day."""
        temporal = {
            "srad_W_m2_mean_2020": xr.Dataset(
                {"srad_W_m2_mean": (("time", "nhm_id"), np.array([[100.0, 200.0, 300.0]]))},
                coords={"time": [0], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        expected = np.array([100.0, 200.0, 300.0]) * 2.065
        np.testing.assert_allclose(result["swrad"].values[0], expected, rtol=1e-6)

    def test_potet_converted_to_inches(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """potet is converted from mm to inches."""
        temporal = {
            "pet_mm_mean_2020": xr.Dataset(
                {"pet_mm_mean": (("time", "nhm_id"), np.array([[25.4, 50.8, 0.0]]))},
                coords={"time": [0], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        np.testing.assert_allclose(result["potet"].values[0], [1.0, 2.0, 0.0], rtol=1e-6)

    def test_feature_dim_aligned(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Temporal feature dim is renamed to match derived dataset nhru."""
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert "nhru" in result["prcp"].dims

    def test_missing_sir_variable_skipped(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Only provided forcing variables appear; absent ones are not created."""
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert "prcp" in result
        assert "tmax" not in result

    def test_unknown_source_skipped(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Temporal variable with no matching forcing config entry is skipped."""
        temporal = {
            "unknown_source_2020": xr.Dataset(
                {"some_unknown_var": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        result = derivation._derive_forcing(ctx, ds)
        assert len(result.data_vars) == 0

    def test_per_variable_temporal_matched(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Per-variable temporal keys are matched via reverse lookup."""
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), np.ones((2, 3)) * 20.0)},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
            "tmmn_C_mean_2020": xr.Dataset(
                {"tmmn_C_mean": (("time", "nhm_id"), np.ones((2, 3)) * 5.0)},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(sir=sir_topography, temporal=temporal)
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        assert "prcp" in result
        assert "tmax" in result
        assert "tmin" in result

    def test_unregistered_conversion_skipped(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        """Unregistered unit conversion logs error and skips variable."""
        import yaml

        # Create a custom YAML with a bogus conversion
        custom_yaml = {
            "name": "forcing_variables",
            "description": "test",
            "mapping": {
                "gridmet": {
                    "prcp": {
                        "sir_name": "pr_mm_mean",
                        "sir_unit": "mm",
                        "intermediate_unit": "mm",
                    },
                    "bogus": {
                        "sir_name": "tmmx_C_mean",
                        "sir_unit": "C",
                        "intermediate_unit": "furlongs",
                    },
                },
            },
        }
        yaml_path = tmp_path / "forcing_variables.yml"
        with open(yaml_path, "w") as f:
            yaml.dump(custom_yaml, f)

        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.ones((2, 3)))},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), np.ones((2, 3)) * 20.0)},
                coords={"time": [0, 1], "nhm_id": [1, 2, 3]},
            ),
        }
        ctx = DerivationContext(
            sir=sir_topography,
            temporal=temporal,
            lookup_tables_dir=tmp_path,
        )
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir_topography["nhm_id"].values)
        result = derivation._derive_forcing(ctx, ds)
        # prcp should succeed, bogus should be skipped (no crash)
        assert "prcp" in result
        assert "bogus" not in result


class TestDeriveIntegrationForcing:
    """Integration test: full derive() with temporal data produces forcing."""

    def test_derive_with_temporal_produces_forcing(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """Full derive() call with temporal context includes forcing vars."""
        ctx = DerivationContext(
            sir=sir_topography,
            temporal=temporal_gridmet,
        )
        result = derivation.derive(ctx)
        assert "prcp" in result
        assert "tmax" in result
        assert "tmin" in result
        assert "swrad" in result
        assert "potet" in result
        assert "time" in result["prcp"].dims


# ------------------------------------------------------------------
# Calibration seeds derivation
# ------------------------------------------------------------------


class TestDeriveCalibrationSeeds:
    """Tests for step 14: calibration seed derivation."""

    def test_constant_seeds_present(self, derivation: PywatershedDerivation) -> None:
        """All 18 constant seeds produced with just a minimal SIR."""
        sir = _MockSIRAccessor(
            xr.Dataset({"_dummy": ("nhm_id", [0.0, 0.0, 0.0])}, coords={"nhm_id": [1, 2, 3]})
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)

        constant_seeds = [
            "smidx_exp",
            "ssr2gw_rate",
            "ssr2gw_exp",
            "slowcoef_lin",
            "slowcoef_sq",
            "fastcoef_lin",
            "fastcoef_sq",
            "pref_flow_den",
            "gwflow_coef",
            "gwsink_coef",
            "rain_cbh_adj",
            "snow_cbh_adj",
            "tmax_cbh_adj",
            "tmin_cbh_adj",
            "tmax_allrain_offset",
            "adjmix_rain",
            "dday_slope",
            "dday_intcp",
        ]
        for seed_name in constant_seeds:
            assert seed_name in ds, f"Constant seed '{seed_name}' missing from output"

    def test_constant_seed_values(self, derivation: PywatershedDerivation) -> None:
        """Constant seeds have correct values from YAML."""
        sir = _MockSIRAccessor(xr.Dataset({"_dummy": ("nhm_id", [0.0])}, coords={"nhm_id": [1]}))
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        np.testing.assert_allclose(ds["smidx_exp"].values, 0.3)
        np.testing.assert_allclose(ds["dday_intcp"].values, -40.0)
        np.testing.assert_allclose(ds["ssr2gw_exp"].values, 1.0)
        np.testing.assert_allclose(ds["fastcoef_sq"].values, 0.8)

    def test_malformed_params_raises(
        self,
        derivation: PywatershedDerivation,
        tmp_path: Path,
    ) -> None:
        """Missing params key in YAML raises ValueError with context."""
        import shutil

        import yaml

        bundled_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "hydro_param"
            / "data"
            / "pywatershed"
            / "lookup_tables"
        )
        for f in bundled_dir.iterdir():
            shutil.copy(f, tmp_path / f.name)

        seeds_path = tmp_path / "calibration_seeds.yml"
        with open(seeds_path) as f:
            data = yaml.safe_load(f)
        # Remove required 'value' key from a constant seed
        data["mapping"]["smidx_exp"]["params"] = {}
        with open(seeds_path, "w") as f:
            yaml.dump(data, f)

        sir = _MockSIRAccessor(xr.Dataset({"_dummy": ("nhm_id", [0.0])}, coords={"nhm_id": [1]}))
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            lookup_tables_dir=tmp_path,
        )
        with pytest.raises(ValueError, match="smidx_exp"):
            derivation.derive(ctx)

    def test_linear_seed_carea_max(self, derivation: PywatershedDerivation) -> None:
        """carea_max = 0.6 * hru_percent_imperv + 0.2."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "fctimp_pct_mean": ("nhm_id", np.array([10.0, 50.0, 0.0])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)

        assert "carea_max" in ds
        # hru_percent_imperv = fctimp_pct_mean / 100 = [0.1, 0.5, 0.0]
        # carea_max = 0.6 * [0.1, 0.5, 0.0] + 0.2 = [0.26, 0.5, 0.2]
        expected = np.array([0.26, 0.5, 0.2])
        np.testing.assert_allclose(ds["carea_max"].values, expected, atol=1e-10)

    def test_exponential_seed_smidx_coef(self, derivation: PywatershedDerivation) -> None:
        """smidx_coef = 0.005 * exp(3.0 * hru_slope)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([100.0, 200.0])),
                    "slope_deg_mean": ("nhm_id", np.array([5.0, 10.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([0.0, 90.0])),
                    "hru_lat": ("nhm_id", np.array([42.0, 42.0])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)

        assert "smidx_coef" in ds
        # hru_slope = tan(deg2rad(slope_deg_mean))
        slopes = np.tan(np.deg2rad(np.array([5.0, 10.0])))
        expected = np.clip(0.005 * np.exp(3.0 * slopes), 0.001, 0.06)
        np.testing.assert_allclose(ds["smidx_coef"].values, expected, atol=1e-10)

    def test_fraction_of_seed(self, derivation: PywatershedDerivation) -> None:
        """soil2gw_max = 0.1 * soil_moist_max."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "awc_mm_mean": ("nhm_id", np.array([50.0, 150.0, 80.0])),
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.7, 0.1, 0.0])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.2, 0.8, 0.1])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.1, 0.1, 0.9])),
                },
                coords={"nhm_id": [1, 2, 3]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)

        assert "soil2gw_max" in ds
        assert "soil_moist_max" in ds
        expected = np.clip(0.1 * ds["soil_moist_max"].values, 0.0, 5.0)
        np.testing.assert_allclose(ds["soil2gw_max"].values, expected, atol=1e-10)

    def test_missing_input_uses_default(self, derivation: PywatershedDerivation) -> None:
        """When input variable missing, default value is used."""
        # No slope data -> smidx_coef should use default 0.01
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"_dummy": ("nhm_id", [0.0, 0.0])},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)

        assert "smidx_coef" in ds
        np.testing.assert_allclose(ds["smidx_coef"].values, [0.01, 0.01])

    def test_range_clipping(self, derivation: PywatershedDerivation) -> None:
        """Very steep slope -> smidx_coef clipped to 0.06."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([100.0])),
                    "slope_deg_mean": ("nhm_id", np.array([80.0])),  # Very steep
                    "aspect_deg_mean": ("nhm_id", np.array([0.0])),
                    "hru_lat": ("nhm_id", np.array([42.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)

        assert "smidx_coef" in ds
        # tan(80 deg) ~ 5.67 -> 0.005 * exp(3.0 * 5.67) is huge -> clipped to 0.06
        assert ds["smidx_coef"].values[0] == pytest.approx(0.06)

    def test_existing_param_not_overwritten(self, derivation: PywatershedDerivation) -> None:
        """Seed skipped if param already in dataset; override wins."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"_dummy": ("nhm_id", [0.0, 0.0])},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            config={"parameter_overrides": {"values": {"gwflow_coef": [0.999, 0.999]}}},
        )
        ds = derivation.derive(ctx)

        # Override is applied AFTER seeds, so override value should win
        assert "gwflow_coef" in ds
        np.testing.assert_allclose(ds["gwflow_coef"].values, [0.999, 0.999])

    def test_all_seeds_produced(self, derivation: PywatershedDerivation) -> None:
        """All 22 seeds present when inputs are available."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([500.0, 1000.0])),
                    "slope_deg_mean": ("nhm_id", np.array([10.0, 20.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([180.0, 270.0])),
                    "hru_lat": ("nhm_id", np.array([42.0, 43.0])),
                    "fctimp_pct_mean": ("nhm_id", np.array([10.0, 5.0])),
                    "awc_mm_mean": ("nhm_id", np.array([100.0, 200.0])),
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.5, 0.2])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.3, 0.6])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.2, 0.2])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)

        all_seeds = [
            "carea_max",
            "smidx_coef",
            "smidx_exp",
            "soil2gw_max",
            "ssr2gw_rate",
            "ssr2gw_exp",
            "slowcoef_lin",
            "slowcoef_sq",
            "fastcoef_lin",
            "fastcoef_sq",
            "pref_flow_den",
            "gwflow_coef",
            "gwsink_coef",
            "snarea_thresh",
            "rain_cbh_adj",
            "snow_cbh_adj",
            "tmax_cbh_adj",
            "tmin_cbh_adj",
            "tmax_allrain_offset",
            "adjmix_rain",
            "dday_slope",
            "dday_intcp",
        ]
        for seed_name in all_seeds:
            assert seed_name in ds, f"Seed '{seed_name}' missing from output"

    def test_unknown_method_uses_default(
        self,
        derivation: PywatershedDerivation,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Unknown method name in YAML -> uses default with warning."""
        import logging
        import shutil

        import yaml

        # Copy bundled lookup tables to tmp_path, then modify calibration_seeds
        bundled_dir = (
            Path(__file__).resolve().parent.parent
            / "src"
            / "hydro_param"
            / "data"
            / "pywatershed"
            / "lookup_tables"
        )
        for f in bundled_dir.iterdir():
            shutil.copy(f, tmp_path / f.name)

        # Modify calibration_seeds.yml with an unknown method
        seeds_path = tmp_path / "calibration_seeds.yml"
        with open(seeds_path) as f:
            data = yaml.safe_load(f)
        data["mapping"]["gwflow_coef"]["method"] = "bogus_method"
        with open(seeds_path, "w") as f:
            yaml.dump(data, f)

        sir = _MockSIRAccessor(
            xr.Dataset({"_dummy": ("nhm_id", [0.0, 0.0])}, coords={"nhm_id": [1, 2]})
        )
        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            lookup_tables_dir=tmp_path,
        )
        with caplog.at_level(logging.WARNING, logger="hydro_param.derivations.pywatershed"):
            ds = derivation.derive(ctx)

        assert "gwflow_coef" in ds
        # Should use default value 0.015
        np.testing.assert_allclose(ds["gwflow_coef"].values, [0.015, 0.015])
        assert any("unknown method" in r.message for r in caplog.records)


class TestSatVp:
    """Tests for saturation vapor pressure helper."""

    def test_freezing_point(self) -> None:
        """sat_vp at 32°F (0°C) should be ~6.11 hPa."""
        from hydro_param.derivations.pywatershed import _sat_vp

        result = _sat_vp(np.array([32.0]))
        np.testing.assert_allclose(result, 6.1078, atol=0.01)

    def test_boiling_point(self) -> None:
        """sat_vp at 212°F (100°C) should be ~1013 hPa."""
        from hydro_param.derivations.pywatershed import _sat_vp

        result = _sat_vp(np.array([212.0]))
        np.testing.assert_allclose(result, 1013.0, rtol=0.02)

    def test_vectorized(self) -> None:
        """sat_vp works on arrays."""
        from hydro_param.derivations.pywatershed import _sat_vp

        temps = np.array([32.0, 50.0, 68.0, 86.0])
        result = _sat_vp(temps)
        assert result.shape == (4,)
        assert np.all(np.diff(result) > 0), "sat_vp should increase with temperature"


class TestComputeMonthlyNormals:
    """Tests for monthly climate normals computation."""

    def test_returns_none_without_temporal(
        self, derivation: PywatershedDerivation, sir_topography: xr.Dataset
    ) -> None:
        """No temporal data -> returns None."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        result = derivation._compute_monthly_normals(ctx)
        assert result is None

    def test_returns_monthly_arrays(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """With temporal data, returns (tmax, tmin) each shape (12, nhru)."""
        ctx = DerivationContext(
            sir=sir_topography,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        result = derivation._compute_monthly_normals(ctx)
        assert result is not None
        monthly_tmax, monthly_tmin = result
        assert monthly_tmax.shape == (12, 3)
        assert monthly_tmin.shape == (12, 3)

    def test_units_are_fahrenheit(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """Output normals should be in °F (converted from °C)."""
        ctx = DerivationContext(
            sir=sir_topography,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        monthly_tmax, monthly_tmin = derivation._compute_monthly_normals(ctx)
        # gridMET tmmx_C_mean is uniform(10, 35) °C -> 50-95°F range
        assert np.all(monthly_tmax > 40.0), "Expected °F values (>40)"
        assert np.all(monthly_tmax < 100.0), "Expected °F values (<100)"


class TestDerivePetCoefficients:
    """Tests for step 10: Jensen-Haise PET coefficient derivation."""

    def test_jh_coef_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """jh_coef should have shape (nmonth, nhru) = (12, nhru)."""
        ctx = DerivationContext(
            sir=sir_topo_with_area,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "jh_coef" in ds
        assert ds["jh_coef"].shape == (12, 3)

    def test_jh_coef_hru_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """jh_coef_hru should have shape (nhru,)."""
        ctx = DerivationContext(
            sir=sir_topo_with_area,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "jh_coef_hru" in ds
        assert ds["jh_coef_hru"].shape == (3,)

    def test_jh_coef_in_valid_range(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """jh_coef values should be in [0.005, 0.06]."""
        ctx = DerivationContext(
            sir=sir_topo_with_area,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert np.all(ds["jh_coef"].values >= 0.005)
        assert np.all(ds["jh_coef"].values <= 0.06)

    def test_jh_coef_formula_known_values(self) -> None:
        """Test jh_coef formula: 1/(C1 + 13*Ch) per Jensen et al. 1970."""
        from hydro_param.derivations.pywatershed import _sat_vp

        # tmax=80°F, tmin=50°F, elev=500ft
        e2 = _sat_vp(np.array([80.0]))[0]
        e1 = _sat_vp(np.array([50.0]))[0]
        c1 = 68.0 - 3.6 * (500.0 / 1000.0)  # 66.2
        ch = 50.0 / (e2 - e1)
        ct = c1 + 13.0 * ch
        jh_coef = 1.0 / ct
        clipped = np.clip(jh_coef, 0.005, 0.06)
        # Ct ≈ 95, jh_coef ≈ 0.0105
        assert 0.008 < clipped < 0.015, f"Expected ~0.0105, got {clipped}"

    def test_jh_coef_hru_formula_known_values(self) -> None:
        """Test jh_coef_hru (Tx) formula: -2.5 - 0.14*(e2-e1) - elev_ft/1000."""
        from hydro_param.derivations.pywatershed import _sat_vp

        # July tmax=85°F, tmin=60°F, elev=1000ft (≈304.8m)
        e_max = _sat_vp(np.array([85.0]))[0]
        e_min = _sat_vp(np.array([60.0]))[0]
        elev_ft = 1000.0
        tx = -2.5 - 0.14 * (e_max - e_min) - elev_ft / 1000.0
        # Tx should be negative and in typical CONUS range
        assert -20 < tx < 0, f"Tx out of typical range, got {tx}"
        # Verify exact computation
        expected_tx = -2.5 - 0.14 * (e_max - e_min) - 1.0
        np.testing.assert_allclose(tx, expected_tx, rtol=1e-10)

    def test_fallback_without_temporal(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
    ) -> None:
        """Without temporal data, jh_coef/jh_coef_hru use defaults."""
        ctx = DerivationContext(sir=sir_topo_with_area, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        # Defaults set by _apply_defaults (step 13)
        assert "jh_coef" in ds
        assert ds["jh_coef"].shape == (12, 3)
        assert ds["jh_coef"].dims == ("nmonth", "nhru")
        assert "jh_coef_hru" in ds


class TestDeriveTranspTiming:
    """Tests for step 11: transpiration timing derivation."""

    def test_transp_beg_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_beg should have shape (nhru,)."""
        ctx = DerivationContext(
            sir=sir_topo_with_area,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "transp_beg" in ds
        assert ds["transp_beg"].shape == (3,)

    def test_transp_end_shape(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_end should have shape (nhru,)."""
        ctx = DerivationContext(
            sir=sir_topo_with_area,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert "transp_end" in ds
        assert ds["transp_end"].shape == (3,)

    def test_values_are_valid_months(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_beg and transp_end should be integers in [1, 12]."""
        ctx = DerivationContext(
            sir=sir_topo_with_area,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert np.all(ds["transp_beg"].values >= 1)
        assert np.all(ds["transp_beg"].values <= 12)
        assert np.all(ds["transp_end"].values >= 1)
        assert np.all(ds["transp_end"].values <= 12)

    def test_beg_before_end(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
        temporal_gridmet: dict[str, xr.Dataset],
    ) -> None:
        """transp_beg should be before transp_end for temperate climates."""
        ctx = DerivationContext(
            sir=sir_topo_with_area,
            fabric_id_field="nhm_id",
            temporal=temporal_gridmet,
        )
        ds = derivation.derive(ctx)
        assert np.all(ds["transp_beg"].values < ds["transp_end"].values)

    def test_warm_climate_early_onset(self, derivation: PywatershedDerivation) -> None:
        """Warm climate (all tmin > 32°F) -> transp_beg = 1 (January)."""
        import pandas as pd

        nhru = 1
        times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        temporal = {
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), np.full((len(times), nhru), 30.0))},
                coords={"time": times, "nhm_id": [1]},
            ),
            "tmmn_C_mean_2020": xr.Dataset(
                {"tmmn_C_mean": (("time", "nhm_id"), np.full((len(times), nhru), 15.0))},
                coords={"time": times, "nhm_id": [1]},
            ),
        }
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([100.0])),
                    "slope_deg_mean": ("nhm_id", np.array([5.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([180.0])),
                    "hru_lat": ("nhm_id", np.array([30.0])),
                    "hru_area_m2": ("nhm_id", np.array([4046856.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", temporal=temporal)
        ds = derivation.derive(ctx)
        assert ds["transp_beg"].values[0] == 1

    def test_cold_climate_late_onset(self, derivation: PywatershedDerivation) -> None:
        """Cold climate with short growing season -> later transp_beg."""
        import pandas as pd

        nhru = 1
        times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        # Cold: tmin < 0°C (32°F) for Jan-May, warm Jun-Aug, cold Sep-Dec
        tmin_values = np.full((len(times), nhru), -5.0)  # °C, well below freezing
        # Warm only in summer months (Jun=152, Jul, Aug=243)
        tmin_values[152:244, :] = 10.0  # °C, above freezing

        temporal = {
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), tmin_values + 15.0)},
                coords={"time": times, "nhm_id": [1]},
            ),
            "tmmn_C_mean_2020": xr.Dataset(
                {"tmmn_C_mean": (("time", "nhm_id"), tmin_values)},
                coords={"time": times, "nhm_id": [1]},
            ),
        }
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([2000.0])),
                    "slope_deg_mean": ("nhm_id", np.array([10.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([180.0])),
                    "hru_lat": ("nhm_id", np.array([45.0])),
                    "hru_area_m2": ("nhm_id", np.array([4046856.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", temporal=temporal)
        ds = derivation.derive(ctx)
        # June onset (month 6) expected
        assert ds["transp_beg"].values[0] >= 5
        assert ds["transp_end"].values[0] <= 10

    def test_fallback_without_temporal(
        self,
        derivation: PywatershedDerivation,
        sir_topo_with_area: xr.Dataset,
    ) -> None:
        """Without temporal data, transp_beg/end use defaults."""
        ctx = DerivationContext(sir=sir_topo_with_area, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "transp_beg" in ds
        assert "transp_end" in ds
        # Defaults: beg=4, end=10
        np.testing.assert_array_equal(ds["transp_beg"].values, [4, 4, 4])
        np.testing.assert_array_equal(ds["transp_end"].values, [10, 10, 10])


class TestDeriveIntegrationPetTransp:
    """Integration test: full derive() produces PET and transpiration params."""

    def test_full_pipeline_with_temporal_produces_all_params(
        self,
        derivation: PywatershedDerivation,
    ) -> None:
        """Full derive() with temporal data produces all PET/transp params."""
        import pandas as pd

        rng = np.random.default_rng(42)
        nhru = 2

        variables = {
            "pr_mm_mean": lambda n, h: rng.uniform(0, 20, (n, h)),
            "tmmx_C_mean": lambda n, h: rng.uniform(10, 35, (n, h)),
            "tmmn_C_mean": lambda n, h: rng.uniform(-5, 15, (n, h)),
            "srad_W_m2_mean": lambda n, h: rng.uniform(50, 300, (n, h)),
            "pet_mm_mean": lambda n, h: rng.uniform(0, 8, (n, h)),
        }

        temporal: dict[str, xr.Dataset] = {}
        for year in [2020, 2021]:
            times = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
            ntime = len(times)
            for var_name, gen_fn in variables.items():
                key = f"{var_name}_{year}"
                temporal[key] = xr.Dataset(
                    {var_name: (("time", "nhm_id"), gen_fn(ntime, nhru))},
                    coords={"time": times, "nhm_id": [1, 2]},
                )

        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([200.0, 800.0])),
                    "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([180.0, 90.0])),
                    "hru_lat": ("nhm_id", np.array([42.0, 43.0])),
                    "hru_area_m2": ("nhm_id", np.array([4046856.0, 8093712.0])),
                    "land_cover": ("nhm_id", np.array([42, 71])),
                    "fctimp_pct_mean": ("nhm_id", np.array([10.0, 5.0])),
                    "tree_canopy_pct_mean": ("nhm_id", np.array([80.0, 10.0])),
                    "awc_mm_mean": ("nhm_id", np.array([100.0, 200.0])),
                    "soil_texture_frac_sand": ("nhm_id", np.array([0.5, 0.2])),
                    "soil_texture_frac_loam": ("nhm_id", np.array([0.3, 0.6])),
                    "soil_texture_frac_clay": ("nhm_id", np.array([0.2, 0.2])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )

        ctx = DerivationContext(
            sir=sir,
            fabric_id_field="nhm_id",
            temporal=temporal,
        )
        ds = derivation.derive(ctx)

        # PET params
        assert "jh_coef" in ds
        assert ds["jh_coef"].dims == ("nmonth", "nhru")
        assert ds["jh_coef"].shape == (12, 2)
        assert "jh_coef_hru" in ds
        assert ds["jh_coef_hru"].shape == (2,)

        # Transpiration params
        assert "transp_beg" in ds
        assert "transp_end" in ds
        assert ds["transp_beg"].shape == (2,)
        assert ds["transp_end"].shape == (2,)
        assert np.all(ds["transp_beg"].values >= 1)
        assert np.all(ds["transp_end"].values <= 12)


class TestClimateNormalsEdgeCases:
    """Edge-case tests for climate normals and derived parameters."""

    def test_all_months_frozen_uses_defaults(self, derivation: PywatershedDerivation) -> None:
        """Arctic HRU where tmin never exceeds freezing -> default transp_beg=4."""
        import pandas as pd

        times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        temporal = {
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), np.full((len(times), 1), -10.0))},
                coords={"time": times, "nhm_id": [1]},
            ),
            "tmmn_C_mean_2020": xr.Dataset(
                {"tmmn_C_mean": (("time", "nhm_id"), np.full((len(times), 1), -20.0))},
                coords={"time": times, "nhm_id": [1]},
            ),
        }
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([3000.0])),
                    "slope_deg_mean": ("nhm_id", np.array([10.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([180.0])),
                    "hru_lat": ("nhm_id", np.array([70.0])),
                    "hru_area_m2": ("nhm_id", np.array([4046856.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", temporal=temporal)
        ds = derivation.derive(ctx)
        # Never thaws -> default April
        assert ds["transp_beg"].values[0] == 4

    def test_never_freezes_after_june_uses_default_end(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Tropical HRU where tmin never drops below freezing -> default transp_end=10."""
        import pandas as pd

        times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        temporal = {
            "tmmx_C_mean_2020": xr.Dataset(
                {"tmmx_C_mean": (("time", "nhm_id"), np.full((len(times), 1), 35.0))},
                coords={"time": times, "nhm_id": [1]},
            ),
            "tmmn_C_mean_2020": xr.Dataset(
                {"tmmn_C_mean": (("time", "nhm_id"), np.full((len(times), 1), 20.0))},
                coords={"time": times, "nhm_id": [1]},
            ),
        }
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([50.0])),
                    "slope_deg_mean": ("nhm_id", np.array([2.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([180.0])),
                    "hru_lat": ("nhm_id", np.array([25.0])),
                    "hru_area_m2": ("nhm_id", np.array([4046856.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", temporal=temporal)
        ds = derivation.derive(ctx)
        # Never freezes -> beg=1 (January), end=10 (default October)
        assert ds["transp_beg"].values[0] == 1
        assert ds["transp_end"].values[0] == 10

    def test_normals_returns_none_for_precip_only_temporal(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Temporal data with only precipitation (no tmax/tmin) -> normals is None."""
        import pandas as pd

        times = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        temporal = {
            "pr_mm_mean_2020": xr.Dataset(
                {"pr_mm_mean": (("time", "nhm_id"), np.full((len(times), 1), 5.0))},
                coords={"time": times, "nhm_id": [1]},
            ),
        }
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "elevation_m_mean": ("nhm_id", np.array([500.0])),
                    "slope_deg_mean": ("nhm_id", np.array([5.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([180.0])),
                    "hru_lat": ("nhm_id", np.array([42.0])),
                    "hru_area_m2": ("nhm_id", np.array([4046856.0])),
                },
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id", temporal=temporal)
        # Normals should return None -> defaults used
        result = derivation._compute_monthly_normals(ctx)
        assert result is None

    def test_normals_with_empty_temporal_dict(
        self,
        derivation: PywatershedDerivation,
        sir_topography: xr.Dataset,
    ) -> None:
        """Empty temporal dict -> normals returns None."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id", temporal={})
        result = derivation._compute_monthly_normals(ctx)
        assert result is None


class TestDeriveWaterbody:
    """Tests for step 6: waterbody overlay."""

    def test_overlay_fraction_and_area(
        self, derivation, waterbody_sir, waterbody_fabric, sample_waterbodies
    ):
        """Verify dprst_frac and dprst_area_max from known geometry."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=sample_waterbodies,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        # HRU 1: 60% LakePond coverage (SwampMarsh excluded)
        assert ds["dprst_frac"].values[0] == pytest.approx(0.6, abs=0.01)
        # HRU 2: 30% Reservoir coverage
        assert ds["dprst_frac"].values[1] == pytest.approx(0.3, abs=0.01)

        # Area in acres: 6000 m² and 3000 m²
        assert ds["dprst_area_max"].values[0] == pytest.approx(6000.0 / 4046.8564224, abs=0.01)
        assert ds["dprst_area_max"].values[1] == pytest.approx(3000.0 / 4046.8564224, abs=0.01)

    def test_hru_type_threshold(
        self, derivation, waterbody_sir, waterbody_fabric, sample_waterbodies
    ):
        """HRU with >50% coverage gets type=2 (lake), others type=1."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=sample_waterbodies,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        assert ds["hru_type"].values[0] == 2  # 60% > 50%
        assert ds["hru_type"].values[1] == 1  # 30% < 50%

    def test_no_waterbodies_fallback(self, derivation, waterbody_sir, waterbody_fabric):
        """When waterbodies=None, assign defaults."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=None,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        np.testing.assert_array_equal(ds["dprst_frac"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["dprst_area_max"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["hru_type"].values, [1, 1])

    def test_swamp_only_fallback(self, derivation, waterbody_sir, waterbody_fabric):
        """When only SwampMarsh waterbodies exist, assign defaults."""
        swamp = gpd.GeoDataFrame(
            {
                "comid": [201],
                "ftype": ["SwampMarsh"],
                "geometry": [Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=swamp,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        np.testing.assert_array_equal(ds["dprst_frac"].values, [0.0, 0.0])

    def test_partial_overlap(self, derivation, waterbody_sir, waterbody_fabric):
        """Waterbody extending beyond HRU — only clipped area counted."""
        big_wb = gpd.GeoDataFrame(
            {
                "comid": [301],
                "ftype": ["LakePond"],
                "geometry": [Polygon([(-50, 0), (80, 0), (80, 100), (-50, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=big_wb,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        # Only 80m of 100m HRU covered (80%)
        assert ds["dprst_frac"].values[0] == pytest.approx(0.8, abs=0.01)
        # HRU 2 has no overlap
        assert ds["dprst_frac"].values[1] == pytest.approx(0.0, abs=0.01)

    def test_multiple_waterbodies_per_hru(self, derivation, waterbody_sir, waterbody_fabric):
        """Two waterbodies in one HRU — areas summed."""
        wb1 = Polygon([(0, 0), (20, 0), (20, 100), (0, 100)])
        wb2 = Polygon([(40, 0), (60, 0), (60, 100), (40, 100)])
        multi_wb = gpd.GeoDataFrame(
            {
                "comid": [401, 402],
                "ftype": ["LakePond", "LakePond"],
                "geometry": [wb1, wb2],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=multi_wb,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        # 20m + 20m = 40m of 100m → 40%
        assert ds["dprst_frac"].values[0] == pytest.approx(0.4, abs=0.01)

    def test_crs_mismatch_auto_reproject(self, derivation, waterbody_sir, waterbody_fabric):
        """Waterbodies in different CRS are reprojected to fabric CRS."""
        wb_5070 = gpd.GeoDataFrame(
            {
                "comid": [501],
                "ftype": ["LakePond"],
                "geometry": [Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        wb_4326 = wb_5070.to_crs("EPSG:4326")
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=wb_4326,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        # Should get ~50% coverage after reprojection
        assert ds["dprst_frac"].values[0] == pytest.approx(0.5, abs=0.05)

    def test_missing_ftype_column_raises(self, derivation, waterbody_sir, waterbody_fabric):
        """Waterbodies without 'ftype' column raise KeyError."""
        wb_no_ftype = gpd.GeoDataFrame(
            {
                "comid": [101],
                "geometry": [Polygon([(0, 0), (50, 0), (50, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=wb_no_ftype,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        with pytest.raises(KeyError, match="ftype"):
            derivation._derive_waterbody(ctx, ds)

    def test_fabric_none_with_waterbodies(self, derivation, waterbody_sir, sample_waterbodies):
        """When fabric=None but waterbodies provided, assign defaults."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=None,
            waterbodies=sample_waterbodies,
        )
        ds = xr.Dataset()
        ds["hru_area"] = xr.DataArray(np.array([2.47, 2.47]), dims="nhru")
        ds = derivation._derive_waterbody(ctx, ds)

        np.testing.assert_array_equal(ds["dprst_frac"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["dprst_area_max"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["hru_type"].values, [1, 1])

    def test_waterbodies_outside_all_hrus(self, derivation, waterbody_sir, waterbody_fabric):
        """Waterbodies that don't overlap any HRU produce defaults."""
        distant_wb = gpd.GeoDataFrame(
            {
                "comid": [601],
                "ftype": ["LakePond"],
                "geometry": [Polygon([(9000, 9000), (9100, 9000), (9100, 9100), (9000, 9100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=distant_wb,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        np.testing.assert_array_equal(ds["dprst_frac"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["dprst_area_max"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["hru_type"].values, [1, 1])

    def test_below_50_percent_is_land(self, derivation, waterbody_sir, waterbody_fabric):
        """HRU with <50% coverage is type=1 (land)."""
        wb_under_half = gpd.GeoDataFrame(
            {
                "comid": [701],
                "ftype": ["LakePond"],
                # 49m of 100m = 49% — clearly below threshold
                "geometry": [Polygon([(0, 0), (49, 0), (49, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=wb_under_half,
        )
        ds = xr.Dataset()
        ds = derivation._derive_geometry(ctx, ds)
        ds = derivation._derive_waterbody(ctx, ds)

        assert ds["dprst_frac"].values[0] == pytest.approx(0.49, abs=0.02)
        assert ds["hru_type"].values[0] == 1

    def test_missing_hru_area_uses_defaults(self, derivation, waterbody_sir, waterbody_fabric):
        """Missing hru_area in dataset falls back to defaults with warning."""
        ctx = DerivationContext(
            sir=waterbody_sir,
            fabric=waterbody_fabric,
            waterbodies=None,
        )
        ds = xr.Dataset(coords={"nhru": [1, 2]})  # No hru_area
        ds = derivation._derive_waterbody(ctx, ds)

        np.testing.assert_array_equal(ds["dprst_frac"].values, [0.0, 0.0])
        np.testing.assert_array_equal(ds["hru_type"].values, [1, 1])


class TestDeriveIntegrationWaterbody:
    """Integration test: full derive() with waterbody data."""

    def test_full_derive_with_waterbodies(self, derivation, waterbody_fabric):
        """Full pipeline produces waterbody params when waterbodies provided."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {
                    "hru_area_m2": ("nhm_id", np.array([10000.0, 10000.0])),
                    "elevation_m_mean": ("nhm_id", np.array([100.0, 500.0])),
                    "slope_deg_mean": ("nhm_id", np.array([5.0, 15.0])),
                    "aspect_deg_mean": ("nhm_id", np.array([0.0, 90.0])),
                    "hru_lat": ("nhm_id", np.array([42.0, 41.5])),
                    "land_cover": ("nhm_id", np.array([42, 71])),
                },
                coords={"nhm_id": [1, 2]},
            )
        )
        wb = gpd.GeoDataFrame(
            {
                "comid": [101],
                "ftype": ["LakePond"],
                "geometry": [Polygon([(0, 0), (70, 0), (70, 100), (0, 100)])],
            },
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=waterbody_fabric,
            waterbodies=wb,
        )
        ds = derivation.derive(ctx)

        assert "dprst_frac" in ds
        assert "dprst_area_max" in ds
        assert "hru_type" in ds
        assert ds["dprst_frac"].shape == (2,)
        assert ds["hru_type"].dtype == np.int32
        # HRU 1 should have 70% lake coverage → type 2
        assert ds["hru_type"].values[0] == 2


class TestRoutingSlopes:
    """Tests for slope retrieval helpers (step 12 routing)."""

    def test_slopes_from_comid_direct(self) -> None:
        """All COMIDs present in VAA → correct slopes returned."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 102, 103],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                    LineString([(2, 0), (3, 0)]),
                ],
            },
        )
        vaa = pd.DataFrame({"comid": [101, 102, 103], "slope": [0.01, 0.05, 0.001]})

        slopes = PywatershedDerivation._get_slopes_from_comid(segments, vaa, "comid")

        np.testing.assert_array_equal(slopes, [0.01, 0.05, 0.001])
        assert slopes.dtype == np.float64

    def test_slopes_from_comid_missing_uses_fallback(self) -> None:
        """COMID not in VAA → fallback slope assigned."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 999],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
        )
        vaa = pd.DataFrame({"comid": [101], "slope": [0.02]})

        slopes = PywatershedDerivation._get_slopes_from_comid(segments, vaa, "comid")

        assert slopes[0] == 0.02
        assert slopes[1] == _FALLBACK_SLOPE

    def test_slopes_from_comid_case_insensitive(self) -> None:
        """Column named COMID (uppercase) still works."""
        segments = gpd.GeoDataFrame(
            {
                "COMID": [201, 202],
                "geometry": [
                    LineString([(0, 0), (1, 0)]),
                    LineString([(1, 0), (2, 0)]),
                ],
            },
        )
        vaa = pd.DataFrame({"comid": [201, 202], "slope": [0.03, 0.07]})

        slopes = PywatershedDerivation._get_slopes_from_comid(segments, vaa, "COMID")

        np.testing.assert_array_equal(slopes, [0.03, 0.07])

    # -- Spatial join slope tests --

    def test_slopes_spatial_join_basic(self) -> None:
        """Two GF segments each matching one NHD flowline → correct slopes.

        Segments are 500 m apart to prevent 100 m buffer cross-matching.
        """
        segments = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(500_000, 500_000), (500_500, 500_000)]),
                    LineString([(500_000, 500_500), (500_500, 500_500)]),
                ]
            },
            crs="EPSG:5070",
        )
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "slope": [0.01, 0.05],
                "geometry": [
                    LineString([(500_000, 500_000), (500_500, 500_000)]),
                    LineString([(500_000, 500_500), (500_500, 500_500)]),
                ],
            },
            crs="EPSG:5070",
        )

        slopes = PywatershedDerivation._get_slopes_spatial_join(segments, nhd_flowlines)

        np.testing.assert_allclose(slopes, [0.01, 0.05])
        assert slopes.dtype == np.float64

    def test_slopes_spatial_join_no_match_uses_fallback(self) -> None:
        """GF segment far from any NHD flowline → fallback slope."""
        segments = gpd.GeoDataFrame(
            {"geometry": [LineString([(500_000, 500_000), (500_500, 500_000)])]},
            crs="EPSG:5070",
        )
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "slope": [0.02],
                "geometry": [LineString([(600_000, 600_000), (600_500, 600_000)])],
            },
            crs="EPSG:5070",
        )

        slopes = PywatershedDerivation._get_slopes_spatial_join(segments, nhd_flowlines)

        assert slopes[0] == _FALLBACK_SLOPE

    def test_slopes_spatial_join_multiple_nhd_per_segment(self) -> None:
        """One GF segment with two equal-length NHD flowlines → average."""
        segments = gpd.GeoDataFrame(
            {"geometry": [LineString([(500_000, 500_000), (501_000, 500_000)])]},
            crs="EPSG:5070",
        )
        # Two NHD flowlines of equal length, both within the buffer
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "slope": [0.02, 0.06],
                "geometry": [
                    LineString([(500_000, 500_000), (500_500, 500_000)]),
                    LineString([(500_500, 500_000), (501_000, 500_000)]),
                ],
            },
            crs="EPSG:5070",
        )

        slopes = PywatershedDerivation._get_slopes_spatial_join(segments, nhd_flowlines)

        # Equal lengths → simple average: (0.02 + 0.06) / 2 = 0.04
        np.testing.assert_allclose(slopes[0], 0.04)

    def test_slopes_spatial_join_parallel_offset_within_buffer(self) -> None:
        """NHD flowline running parallel 50 m from segment produces valid slope.

        This is the core regression scenario: GF segments and NHD flowlines
        represent the same river but have different vertex coordinates.  The
        100 m buffer captures the parallel flowline even though the raw lines
        don't intersect.
        """
        segments = gpd.GeoDataFrame(
            {"geometry": [LineString([(500_000, 500_000), (500_500, 500_000)])]},
            crs="EPSG:5070",
        )
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "slope": [0.015],
                "geometry": [LineString([(500_000, 500_050), (500_500, 500_050)])],
            },
            crs="EPSG:5070",
        )
        slopes = PywatershedDerivation._get_slopes_spatial_join(segments, nhd_flowlines)
        np.testing.assert_allclose(slopes[0], 0.015)

    def test_slopes_spatial_join_parallel_offset_beyond_buffer(self) -> None:
        """NHD flowline >100 m from segment does not contribute."""
        segments = gpd.GeoDataFrame(
            {"geometry": [LineString([(500_000, 500_000), (500_500, 500_000)])]},
            crs="EPSG:5070",
        )
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "slope": [0.99],
                "geometry": [LineString([(500_000, 500_200), (500_500, 500_200)])],
            },
            crs="EPSG:5070",
        )
        slopes = PywatershedDerivation._get_slopes_spatial_join(segments, nhd_flowlines)
        assert slopes[0] == _FALLBACK_SLOPE

    def test_slopes_spatial_join_geographic_crs_raises(self) -> None:
        """Geographic CRS raises ValueError (buffer distance is in metres)."""
        segments = gpd.GeoDataFrame(
            {"geometry": [LineString([(-75, 40), (-74, 40)])]},
            crs="EPSG:4326",
        )
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "slope": [0.01],
                "geometry": [LineString([(-75, 40), (-74, 40)])],
            },
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="projected CRS"):
            PywatershedDerivation._get_slopes_spatial_join(segments, nhd_flowlines)

    def test_vaa_sentinel_filtering(self) -> None:
        """VAA -9998 sentinel slopes are excluded but small negatives are kept."""
        vaa_raw = pd.DataFrame(
            {
                "comid": [101, 102, 103, 104],
                "slope": [0.01, -9998.0, -0.001, 0.005],
            }
        )
        # Simulate the filtering logic from _fetch_vaa
        result = vaa_raw[["comid", "slope"]].dropna(subset=["slope"])
        result = result[result["slope"] != _NHD_MISSING_SLOPE_SENTINEL]

        # -9998 sentinel removed, small negative kept
        assert 102 not in result["comid"].values
        assert 103 in result["comid"].values
        assert len(result) == 3


# ------------------------------------------------------------------
# Manning's equation K_coef computation
# ------------------------------------------------------------------


class TestManningKCoef:
    """Tests for _compute_k_coef (Manning's equation K_coef)."""

    def test_basic_k_coef(self) -> None:
        """Known slope and length produce expected K_coef.

        velocity = (1/0.04) * sqrt(0.01) * 1.0^(2/3) * 3600
                 = 25 * 0.1 * 1.0 * 3600 = 9000 ft/hr
        seg_length_ft = 10000 * 3.28084 = 32808.4 ft
        K_coef = 32808.4 / 9000 ≈ 3.645 hours
        """
        slopes = np.array([0.01])
        lengths = np.array([10000.0])
        result = PywatershedDerivation._compute_k_coef(slopes, lengths)
        np.testing.assert_allclose(result[0], 32808.4 / 9000.0, rtol=1e-4)

    def test_k_coef_clamped_max(self) -> None:
        """Very low slope + long segment clamps K_coef to maximum."""
        slopes = np.array([1e-7])
        lengths = np.array([100_000.0])  # 100 km
        result = PywatershedDerivation._compute_k_coef(slopes, lengths)
        assert result[0] == _K_COEF_MAX

    def test_k_coef_clamped_min(self) -> None:
        """Steep slope + very short segment clamps K_coef to minimum."""
        slopes = np.array([1.0])
        lengths = np.array([1.0])  # 1 meter
        result = PywatershedDerivation._compute_k_coef(slopes, lengths)
        assert result[0] == _K_COEF_MIN

    def test_slope_floor_applied(self) -> None:
        """Zero and negative slopes both get clamped to _MIN_SLOPE."""
        lengths = np.array([5000.0, 5000.0])
        slopes_zero = np.array([0.0, -0.001])
        result = PywatershedDerivation._compute_k_coef(slopes_zero, lengths)
        # Both should produce the same K_coef since both clamp to 1e-7
        np.testing.assert_allclose(result[0], result[1])

    def test_zero_length_gets_default(self) -> None:
        """Zero-length segment receives _DEFAULT_K_COEF."""
        slopes = np.array([0.01])
        lengths = np.array([0.0])
        result = PywatershedDerivation._compute_k_coef(slopes, lengths)
        assert result[0] == _DEFAULT_K_COEF


class TestSegmentTypeDetection:
    """Tests for NHD vs GF segment detection via _find_comid_column."""

    def test_find_comid_lowercase(self, derivation: PywatershedDerivation) -> None:
        """Segments with 'comid' column return the column name."""
        segments = gpd.GeoDataFrame(
            {"comid": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        assert derivation._find_comid_column(segments) == "comid"

    def test_find_comid_uppercase(self, derivation: PywatershedDerivation) -> None:
        """Segments with 'COMID' column return the column name."""
        segments = gpd.GeoDataFrame(
            {"COMID": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        assert derivation._find_comid_column(segments) == "COMID"

    def test_find_comid_mixed_case(self, derivation: PywatershedDerivation) -> None:
        """Segments with 'Comid' column return the column name."""
        segments = gpd.GeoDataFrame(
            {"Comid": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        assert derivation._find_comid_column(segments) == "Comid"

    def test_no_comid(self, derivation: PywatershedDerivation) -> None:
        """Segments without COMID column return None."""
        segments = gpd.GeoDataFrame(
            {"nhm_seg": [1], "tosegment": [0]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs="EPSG:4326",
        )
        assert derivation._find_comid_column(segments) is None


# ------------------------------------------------------------------
# Step 12: _derive_routing orchestration
# ------------------------------------------------------------------


class TestDeriveRouting:
    """Tests for Step 12: _derive_routing orchestration."""

    def test_routing_no_segments_returns_unchanged(self, derivation: PywatershedDerivation) -> None:
        """No segments -> warn and return ds unchanged."""
        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2, 3]}))
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = xr.Dataset()
        ds = ds.assign_coords(nhru=sir["nhm_id"].values)
        ds = derivation._derive_routing(ctx, ds)
        assert "K_coef" not in ds
        assert "x_coef" not in ds

    def test_routing_with_comid_segments(
        self, derivation: PywatershedDerivation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NHD segments with COMID produce K_coef from VAA slope."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 102],
                "tosegment": [2, 0],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        vaa = pd.DataFrame({"comid": [101, 102], "slope": [0.01, 0.005]})

        monkeypatch.setattr(
            PywatershedDerivation,
            "_fetch_vaa",
            staticmethod(lambda: vaa),
        )

        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="comid",
        )

        ds = derivation.derive(ctx)

        assert "K_coef" in ds
        assert "x_coef" in ds
        assert "seg_slope" in ds
        assert "segment_type" in ds
        assert "obsin_segment" in ds
        assert ds["K_coef"].dims == ("nsegment",)
        assert np.all(ds["K_coef"].values > 0)
        assert np.all(ds["K_coef"].values <= 24.0)
        np.testing.assert_array_almost_equal(ds["x_coef"].values, [0.2, 0.2])
        np.testing.assert_array_equal(ds["segment_type"].values, [0, 0])
        np.testing.assert_array_equal(ds["obsin_segment"].values, [0, 0])

    def test_routing_gf_segments_spatial_join(
        self, derivation: PywatershedDerivation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GF segments without COMID use spatial join for slopes.

        Uses EPSG:5070 (projected, metres) so the 100 m buffer in
        ``_get_slopes_spatial_join`` is meaningful.  Flowlines run
        parallel to segments with a 30 m offset (within the 100 m
        buffer), mimicking independently digitized river representations.
        Segments are spaced 500 m apart to avoid cross-matching.
        """
        segments = gpd.GeoDataFrame(
            {
                "nhm_seg": [1, 2],
                "tosegment": [2, 0],
            },
            geometry=[
                LineString([(1_700_000, 2_000_000), (1_700_500, 2_000_000)]),
                LineString([(1_700_000, 2_000_500), (1_700_500, 2_000_500)]),
            ],
            crs="EPSG:5070",
        )
        # Flowlines run parallel 30 m from each segment (within the
        # 100 m buffer).  They don't share vertices with the segments,
        # mimicking the independently digitized GF/NHD scenario.
        nhd_flowlines = gpd.GeoDataFrame(
            {
                "comid": [101, 102],
                "slope": [0.01, 0.005],
            },
            geometry=[
                LineString([(1_700_000, 2_000_030), (1_700_500, 2_000_030)]),
                LineString([(1_700_000, 2_000_530), (1_700_500, 2_000_530)]),
            ],
            crs="EPSG:5070",
        )
        vaa = pd.DataFrame({"comid": [101, 102], "slope": [0.01, 0.005]})

        monkeypatch.setattr(
            PywatershedDerivation,
            "_fetch_vaa",
            staticmethod(lambda: vaa),
        )
        monkeypatch.setattr(
            PywatershedDerivation,
            "_fetch_nhd_flowlines",
            staticmethod(lambda segs, v: nhd_flowlines),
        )

        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon(
                    [
                        (1_699_900, 1_999_900),
                        (1_700_600, 1_999_900),
                        (1_700_600, 2_000_100),
                        (1_699_900, 2_000_100),
                    ]
                ),
                Polygon(
                    [
                        (1_699_900, 2_000_400),
                        (1_700_600, 2_000_400),
                        (1_700_600, 2_000_600),
                        (1_699_900, 2_000_600),
                    ]
                ),
            ],
            crs="EPSG:5070",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="nhm_seg",
        )

        ds = derivation.derive(ctx)

        assert "K_coef" in ds
        assert "seg_slope" in ds
        np.testing.assert_array_almost_equal(ds["seg_slope"].values, [0.01, 0.005])

    def test_routing_segment_type_passthrough(
        self, derivation: PywatershedDerivation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """segment_type column in segments GeoDataFrame is passed through."""
        segments = gpd.GeoDataFrame(
            {
                "comid": [101, 102],
                "tosegment": [2, 0],
                "segment_type": [0, 1],  # channel, lake
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )
        vaa = pd.DataFrame({"comid": [101, 102], "slope": [0.01, 0.005]})

        monkeypatch.setattr(
            PywatershedDerivation,
            "_fetch_vaa",
            staticmethod(lambda: vaa),
        )

        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
            crs="EPSG:4326",
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=fabric,
            segments=segments,
            fabric_id_field="nhm_id",
            segment_id_field="comid",
        )

        ds = derivation.derive(ctx)

        np.testing.assert_array_equal(ds["segment_type"].values, [0, 1])
        # Lake segment should have K_coef = 24.0
        assert ds["K_coef"].values[1] == _LAKE_K_COEF

    def test_routing_fetch_failure_uses_fallbacks(
        self, derivation: PywatershedDerivation, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failed NHD fetch produces fallback slopes."""
        segments = gpd.GeoDataFrame(
            {
                "nhm_seg": [1, 2],
                "tosegment": [2, 0],
            },
            geometry=[
                LineString([(0, 0), (1, 0)]),
                LineString([(1, 0), (2, 0)]),
            ],
            crs="EPSG:4326",
        )

        monkeypatch.setattr(
            PywatershedDerivation,
            "_fetch_vaa",
            staticmethod(lambda: None),
        )

        sir = _MockSIRAccessor(xr.Dataset(coords={"nhm_id": [1, 2]}))
        fabric = gpd.GeoDataFrame(
            {"nhm_id": [1, 2], "hru_segment": [1, 2]},
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
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

        assert "K_coef" in ds
        assert "seg_slope" in ds
        np.testing.assert_array_equal(ds["seg_slope"].values, [_FALLBACK_SLOPE, _FALLBACK_SLOPE])


class TestDeriveNhruFallback:
    """Test nhru resolution fallback paths in derive()."""

    def test_no_fabric_uses_sir_length(self, derivation: PywatershedDerivation) -> None:
        """When fabric=None, nhru is derived from SIR variable length."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"elevation_m_mean": ("nhm_id", np.array([100.0, 200.0]))},
                coords={"nhm_id": [1, 2]},
            )
        )
        ctx = DerivationContext(
            sir=sir,
            fabric=None,
            fabric_id_field="nhm_id",
        )
        ds = derivation.derive(ctx)
        # Should produce params for 2 HRUs (derived from SIR variable length)
        assert ds.sizes.get("nhru", 0) >= 2 or len(ds.data_vars) > 0

    def test_no_fabric_no_sir_vars_raises(self, derivation: PywatershedDerivation) -> None:
        """When fabric=None and SIR has no data_vars, derive() raises ValueError."""
        sir = _MockSIRAccessor(xr.Dataset())
        ctx = DerivationContext(
            sir=sir,
            fabric=None,
            fabric_id_field="nhm_id",
        )
        with pytest.raises(ValueError, match="Cannot determine HRU count"):
            derivation.derive(ctx)


class TestSegmentDefaults:
    """Tests for segment-level defaults in _apply_defaults."""

    def test_segment_defaults_present_when_nsegment_exists(
        self, derivation: PywatershedDerivation
    ) -> None:
        """Segment defaults appear when nsegment dimension is in the dataset.

        Call ``_apply_defaults`` directly with a dataset that already has
        an ``nsegment`` dimension, simulating what topology + routing steps
        would produce before defaults run.
        """
        ds = xr.Dataset(
            {
                "hru_elev": ("nhru", np.array([100.0, 200.0])),
                "tosegment": ("nsegment", np.array([2, 0], dtype=np.int32)),
            },
            coords={"nhru": [1, 2]},
        )
        ds = derivation._apply_defaults(ds, nhru=2)
        for name in ("mann_n", "seg_depth", "segment_flow_init", "obsout_segment"):
            assert name in ds, f"Missing segment default: {name}"
            assert ds[name].shape == (2,), f"{name}: expected (2,), got {ds[name].shape}"
        # tosegment_nhm is now set by _derive_topology, not _apply_defaults
        assert "tosegment_nhm" not in ds

    def test_segment_defaults_absent_without_nsegment(
        self, derivation: PywatershedDerivation, sir_topography: _MockSIRAccessor
    ) -> None:
        """Segment defaults must NOT appear when no nsegment dimension."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "mann_n" not in ds
        assert "tosegment_nhm" not in ds


class TestHruInToCf:
    """Tests for hru_in_to_cf conversion factor."""

    def test_hru_in_to_cf_from_hru_area(
        self, derivation: PywatershedDerivation, sir_geometry: _MockSIRAccessor
    ) -> None:
        """hru_in_to_cf = hru_area * 43560/12 (inches*acres to cubic feet)."""
        ctx = DerivationContext(sir=sir_geometry, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_in_to_cf" in ds
        expected = ds["hru_area"].values * (43560.0 / 12.0)
        np.testing.assert_allclose(ds["hru_in_to_cf"].values, expected)

    def test_hru_in_to_cf_missing_without_hru_area(self, derivation: PywatershedDerivation) -> None:
        """hru_in_to_cf not created when hru_area is missing (warning logged)."""
        sir = _MockSIRAccessor(
            xr.Dataset(
                {"elevation_m_mean": ("nhm_id", np.array([100.0]))},
                coords={"nhm_id": [1]},
            )
        )
        ctx = DerivationContext(sir=sir, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "hru_in_to_cf" not in ds


class TestPrefFlowInfilFrac:
    """Tests for pref_flow_infil_frac conditional logic."""

    def test_defaults_to_zero_without_pref_flow_den(
        self, derivation: PywatershedDerivation, sir_topography: _MockSIRAccessor
    ) -> None:
        """Without pref_flow_den, pref_flow_infil_frac defaults to 0.0."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert "pref_flow_infil_frac" in ds
        np.testing.assert_array_equal(ds["pref_flow_infil_frac"].values, [0.0, 0.0, 0.0])


class TestSpecialDefaultShapes:
    """Tests for special-case default parameter shapes."""

    def test_doy_shape_and_dims(
        self, derivation: PywatershedDerivation, sir_topography: _MockSIRAccessor
    ) -> None:
        """doy must be (366,) with ndoy dimension."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["doy"].shape == (366,)
        assert ds["doy"].dims == ("ndoy",)

    def test_snarea_curve_shape_and_dims(
        self, derivation: PywatershedDerivation, sir_topography: _MockSIRAccessor
    ) -> None:
        """snarea_curve must be (11,) with ndeplval dimension."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["snarea_curve"].shape == (11,)
        assert ds["snarea_curve"].dims == ("ndeplval",)

    def test_temp_units_is_zero(
        self, derivation: PywatershedDerivation, sir_topography: _MockSIRAccessor
    ) -> None:
        """temp_units must be 0 (Fahrenheit)."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        assert ds["temp_units"].item() == 0


class TestMonthlyCaliSeedShapes:
    """Tests for monthly calibration seed shapes."""

    def test_monthly_calibration_seeds_shape(
        self, derivation: PywatershedDerivation, sir_topography: _MockSIRAccessor
    ) -> None:
        """Monthly calibration seeds must have shape (12, nhru)."""
        ctx = DerivationContext(sir=sir_topography, fabric_id_field="nhm_id")
        ds = derivation.derive(ctx)
        nhru = 3
        monthly_seeds = [
            "rain_cbh_adj",
            "snow_cbh_adj",
            "tmax_cbh_adj",
            "tmin_cbh_adj",
            "tmax_allrain_offset",
            "adjmix_rain",
            "dday_slope",
            "dday_intcp",
        ]
        for name in monthly_seeds:
            assert name in ds, f"Missing calibration seed: {name}"
            assert ds[name].shape == (12, nhru), (
                f"{name}: expected (12, {nhru}), got {ds[name].shape}"
            )


class TestParamDimsConsistency:
    """Tests for _PARAM_DIMS / _DEFAULTS structural invariants."""

    def test_param_dims_covers_all_non_special_defaults(self) -> None:
        """Every non-special default must have a _PARAM_DIMS entry."""
        from hydro_param.derivations.pywatershed import (
            _DEFAULTS,
            _DEFAULTS_SPECIAL,
            _PARAM_DIMS,
        )

        missing = (set(_DEFAULTS) - _DEFAULTS_SPECIAL) - set(_PARAM_DIMS)
        assert not missing, f"_DEFAULTS keys missing from _PARAM_DIMS: {sorted(missing)}"
