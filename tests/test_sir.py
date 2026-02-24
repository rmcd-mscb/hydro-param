"""Tests for SIR normalization layer."""

from __future__ import annotations

from hydro_param.sir import (
    canonical_name,
    unit_abbreviation,
)


class TestUnitAbbreviation:
    """Tests for unit_abbreviation()."""

    def test_meters(self) -> None:
        assert unit_abbreviation("m") == "m"

    def test_degrees(self) -> None:
        assert unit_abbreviation("degrees") == "deg"

    def test_percent(self) -> None:
        assert unit_abbreviation("%") == "pct"

    def test_volume_fraction(self) -> None:
        assert unit_abbreviation("m3/m3") == "m3_m3"

    def test_bulk_density(self) -> None:
        assert unit_abbreviation("g/cm3") == "g_cm3"

    def test_centimeters(self) -> None:
        assert unit_abbreviation("cm") == "cm"

    def test_hydraulic_conductivity(self) -> None:
        assert unit_abbreviation("cm/hr") == "cm_hr"

    def test_log_ksat(self) -> None:
        # log10(cm/hr) normalizes to cm/hr canonical
        assert unit_abbreviation("log10(cm/hr)") == "cm_hr"

    def test_log_kpa(self) -> None:
        assert unit_abbreviation("log10(kPa)") == "kPa"

    def test_log_percent(self) -> None:
        assert unit_abbreviation("log10(%)") == "pct"

    def test_log_kpa_inverse(self) -> None:
        assert unit_abbreviation("log10(kPa^-1)") == "kPa_inv"

    def test_day_of_year(self) -> None:
        assert unit_abbreviation("day_of_year") == "doy"

    def test_empty_dimensionless(self) -> None:
        assert unit_abbreviation("") == ""

    def test_unknown_unit_passthrough(self) -> None:
        # Unknown units: slugify (replace / with _, strip special chars)
        assert unit_abbreviation("kg/m2") == "kg_m2"


class TestCanonicalName:
    """Tests for canonical_name()."""

    def test_elevation_meters_mean(self) -> None:
        assert canonical_name("elevation", "m", "mean") == "elevation_m_mean"

    def test_slope_degrees_mean(self) -> None:
        assert canonical_name("slope", "degrees", "mean") == "slope_deg_mean"

    def test_clay_percent_mean(self) -> None:
        assert canonical_name("clay", "%", "mean") == "clay_pct_mean"

    def test_dimensionless_no_unit(self) -> None:
        # Empty units -> base_stat (no unit segment)
        assert canonical_name("lambda", "", "mean") == "lambda_mean"

    def test_ksat_log_transform(self) -> None:
        # log10(cm/hr) -> canonical abbrev is cm_hr
        assert canonical_name("ksat", "log10(cm/hr)", "mean") == "ksat_cm_hr_mean"

    def test_min_statistic(self) -> None:
        assert canonical_name("elevation", "m", "min") == "elevation_m_min"

    def test_uppercase_base_lowered(self) -> None:
        assert canonical_name("FctImp", "%", "mean") == "fctimp_pct_mean"

    def test_categorical_stat(self) -> None:
        assert canonical_name("LndCov", "", "majority") == "lndcov_majority"


class TestSIRVariableSchema:
    """Tests for SIRVariableSchema dataclass."""

    def test_basic_creation(self) -> None:
        from hydro_param.sir import SIRVariableSchema

        schema = SIRVariableSchema(
            canonical_name="elevation_m_mean",
            source_name="elevation",
            source_units="m",
            canonical_units="m",
            long_name="Surface elevation above NAVD88",
            categorical=False,
            valid_range=(-500.0, 9000.0),
            conversion=None,
        )
        assert schema.canonical_name == "elevation_m_mean"
        assert schema.conversion is None

    def test_log_transform_schema(self) -> None:
        from hydro_param.sir import SIRVariableSchema

        schema = SIRVariableSchema(
            canonical_name="ksat_cm_hr_mean",
            source_name="ksat",
            source_units="log10(cm/hr)",
            canonical_units="cm/hr",
            long_name="Saturated hydraulic conductivity",
            categorical=False,
            valid_range=None,
            conversion="log10_to_linear",
        )
        assert schema.conversion == "log10_to_linear"
        assert schema.canonical_units == "cm/hr"


class TestBuildSIRSchema:
    """Tests for build_sir_schema()."""

    def _make_entry(
        self,
        strategy: str = "stac_cog",
        variables: list[VariableSpec] | None = None,
        derived_variables: list[DerivedVariableSpec] | None = None,
        category: str = "topography",
    ) -> DatasetEntry:
        """Helper to create a DatasetEntry for testing."""
        from hydro_param.dataset_registry import DatasetEntry

        return DatasetEntry(
            strategy=strategy,
            catalog_url="https://example.com/stac" if strategy == "stac_cog" else None,
            collection="test" if strategy in ("stac_cog", "nhgf_stac") else None,
            catalog_id="test" if strategy == "climr_cat" else None,
            temporal=strategy == "climr_cat",
            variables=variables or [],
            derived_variables=derived_variables or [],
            category=category,
        )

    def test_single_continuous_variable(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
        )
        ds_req = DatasetRequest(name="dem_test", variables=["elevation"], statistics=["mean"])
        var_specs: list[VariableSpec | DerivedVariableSpec] = [
            VariableSpec(name="elevation", units="m", long_name="Elevation")
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert len(schema) == 1
        assert schema[0].canonical_name == "elevation_m_mean"
        assert schema[0].source_name == "elevation"
        assert schema[0].conversion is None

    def test_multiple_statistics(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
        )
        ds_req = DatasetRequest(
            name="dem_test", variables=["elevation"], statistics=["mean", "min", "max"]
        )
        var_specs: list[VariableSpec | DerivedVariableSpec] = [
            VariableSpec(name="elevation", units="m", long_name="Elevation")
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        names = {s.canonical_name for s in schema}
        assert names == {"elevation_m_mean", "elevation_m_min", "elevation_m_max"}

    def test_derived_variable(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import DerivedVariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            derived_variables=[
                DerivedVariableSpec(
                    name="slope", source="elevation", method="horn", units="degrees"
                )
            ]
        )
        ds_req = DatasetRequest(name="dem_test", variables=["slope"], statistics=["mean"])
        var_specs: list[VariableSpec | DerivedVariableSpec] = [
            DerivedVariableSpec(name="slope", source="elevation", method="horn", units="degrees")
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert len(schema) == 1
        assert schema[0].canonical_name == "slope_deg_mean"

    def test_log_transform_variable(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            strategy="local_tiff",
            variables=[VariableSpec(name="ksat", units="log10(cm/hr)", long_name="Ksat")],
            category="soils",
        )
        ds_req = DatasetRequest(name="polaris", variables=["ksat"], statistics=["mean"])
        var_specs: list[VariableSpec | DerivedVariableSpec] = [
            VariableSpec(name="ksat", units="log10(cm/hr)", long_name="Ksat")
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert schema[0].canonical_name == "ksat_cm_hr_mean"
        assert schema[0].source_units == "log10(cm/hr)"
        assert schema[0].canonical_units == "cm/hr"
        assert schema[0].conversion == "log10_to_linear"

    def test_categorical_variable(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            strategy="nhgf_stac",
            variables=[VariableSpec(name="LndCov", long_name="Land cover", categorical=True)],
            category="land_cover",
        )
        ds_req = DatasetRequest(name="nlcd_test", variables=["LndCov"], statistics=["mean"])
        var_specs: list[VariableSpec | DerivedVariableSpec] = [
            VariableSpec(name="LndCov", long_name="Land cover", categorical=True)
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        # Categorical variables produce a single schema entry with
        # stat="frac" to indicate fraction columns will be generated
        assert len(schema) == 1
        assert schema[0].categorical is True
        assert schema[0].source_name == "LndCov"

    def test_multi_year_keys(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
        )
        ds_req = DatasetRequest(
            name="dem_test", variables=["elevation"], statistics=["mean"], year=[2020, 2021]
        )
        var_specs: list[VariableSpec | DerivedVariableSpec] = [
            VariableSpec(name="elevation", units="m", long_name="Elevation")
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        names = {s.canonical_name for s in schema}
        assert names == {"elevation_m_mean_2020", "elevation_m_mean_2021"}

    def test_dimensionless_variable(self) -> None:
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            strategy="local_tiff",
            variables=[VariableSpec(name="lambda", units="", long_name="Pore size distribution")],
            category="soils",
        )
        ds_req = DatasetRequest(name="polaris", variables=["lambda"], statistics=["mean"])
        var_specs: list[VariableSpec | DerivedVariableSpec] = [
            VariableSpec(name="lambda", units="", long_name="Pore size distribution")
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert schema[0].canonical_name == "lambda_mean"


class TestApplyConversion:
    """Tests for SIR unit conversion application."""

    def test_no_conversion(self) -> None:
        import numpy as np
        from numpy.testing import assert_allclose

        from hydro_param.sir import apply_conversion

        values = np.array([100.0, 200.0, 300.0])
        result = apply_conversion(values, conversion=None)
        assert_allclose(result, values)

    def test_log10_to_linear(self) -> None:
        import numpy as np
        from numpy.testing import assert_allclose

        from hydro_param.sir import apply_conversion

        values = np.array([0.0, 1.0, 2.0])  # log10 scale
        result = apply_conversion(values, conversion="log10_to_linear")
        assert_allclose(result, [1.0, 10.0, 100.0])

    def test_log10_to_linear_with_nan(self) -> None:
        import numpy as np
        from numpy.testing import assert_allclose

        from hydro_param.sir import apply_conversion

        values = np.array([1.0, np.nan, 2.0])
        result = apply_conversion(values, conversion="log10_to_linear")
        assert_allclose(result[0], 10.0)
        assert np.isnan(result[1])
        assert_allclose(result[2], 100.0)

    def test_unknown_conversion_raises(self) -> None:
        import numpy as np
        import pytest

        from hydro_param.sir import apply_conversion

        with pytest.raises(ValueError, match="Unknown conversion"):
            apply_conversion(np.array([1.0]), conversion="unknown_transform")
