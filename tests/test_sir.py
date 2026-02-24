"""Tests for SIR normalization layer."""

from __future__ import annotations

from pathlib import Path

from hydro_param.dataset_registry import (
    DatasetEntry,
    DerivedVariableSpec,
    VariableSpec,
)
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

    def test_kelvin(self) -> None:
        assert unit_abbreviation("K") == "C"

    def test_millimeters(self) -> None:
        assert unit_abbreviation("mm") == "mm"

    def test_watts_per_m2(self) -> None:
        assert unit_abbreviation("W/m2") == "W_m2"

    def test_kg_per_kg(self) -> None:
        assert unit_abbreviation("kg/kg") == "kg_kg"

    def test_meters_per_second(self) -> None:
        assert unit_abbreviation("m/s") == "m_s"

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

    def test_temperature_kelvin_converts_to_celsius(self) -> None:
        assert canonical_name("tmmx", "K", "mean") == "tmmx_C_mean"

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

    def test_temporal_field_default_false(self) -> None:
        from hydro_param.sir import SIRVariableSchema

        s = SIRVariableSchema(
            canonical_name="test",
            source_name="test",
            source_units="m",
            canonical_units="m",
            long_name="Test",
            categorical=False,
            valid_range=None,
            conversion=None,
        )
        assert s.temporal is False

    def test_temporal_field_explicit(self) -> None:
        from hydro_param.sir import SIRVariableSchema

        s = SIRVariableSchema(
            canonical_name="test",
            source_name="test",
            source_units="m",
            canonical_units="m",
            long_name="Test",
            categorical=False,
            valid_range=None,
            conversion=None,
            temporal=True,
        )
        assert s.temporal is True

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

    def test_temporal_dataset_marked(self) -> None:
        """Schema entries from temporal datasets have temporal=True."""
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import DatasetEntry, VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = DatasetEntry(
            strategy="climr_cat",
            catalog_id="gridmet",
            temporal=True,
            t_coord="day",
            variables=[VariableSpec(name="pr", units="mm", long_name="Precipitation")],
            category="climate",
        )
        ds_req = DatasetRequest(
            name="gridmet",
            variables=["pr"],
            statistics=["mean"],
            time_period=["2020-01-01", "2020-12-31"],
        )
        var_specs = [VariableSpec(name="pr", units="mm", long_name="Precipitation")]
        resolved = [(entry, ds_req, var_specs)]
        schema = build_sir_schema(resolved)
        assert len(schema) == 1
        assert schema[0].temporal is True

    def test_static_dataset_not_temporal(self) -> None:
        """Schema entries from static datasets have temporal=False."""
        from hydro_param.config import DatasetRequest
        from hydro_param.dataset_registry import VariableSpec
        from hydro_param.sir import build_sir_schema

        entry = self._make_entry(
            variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
        )
        ds_req = DatasetRequest(name="test", variables=["elevation"], statistics=["mean"])
        var_specs = [VariableSpec(name="elevation", units="m", long_name="Elevation")]
        resolved = [(entry, ds_req, var_specs)]
        schema = build_sir_schema(resolved)
        assert len(schema) == 1
        assert schema[0].temporal is False

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

    def test_k_to_c(self) -> None:
        import numpy as np

        from hydro_param.sir import apply_conversion

        values = np.array([273.15, 283.15, 293.15])
        result = apply_conversion(values, "K_to_C")
        np.testing.assert_allclose(result, [0.0, 10.0, 20.0])

    def test_k_to_c_with_nan(self) -> None:
        import numpy as np

        from hydro_param.sir import apply_conversion

        values = np.array([273.15, np.nan, 293.15])
        result = apply_conversion(values, "K_to_C")
        np.testing.assert_allclose(result[0], 0.0)
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[2], 20.0)

    def test_unknown_conversion_raises(self) -> None:
        import numpy as np
        import pytest

        from hydro_param.sir import apply_conversion

        with pytest.raises(ValueError, match="Unknown conversion"):
            apply_conversion(np.array([1.0]), conversion="unknown_transform")


class TestNormalizeSIR:
    """Tests for normalize_sir()."""

    def test_continuous_mean_rename(self, tmp_path: Path) -> None:
        """Single continuous variable with mean stat gets canonical name."""
        import pandas as pd
        from numpy.testing import assert_allclose

        from hydro_param.sir import SIRVariableSchema, normalize_sir

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df = pd.DataFrame(
            {"elevation": [100.0, 200.0]},
            index=pd.Index([1, 2], name="nhm_id"),
        )
        raw_path = raw_dir / "elevation.csv"
        df.to_csv(raw_path)

        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Surface elevation",
                categorical=False,
                valid_range=None,
                conversion=None,
            )
        ]

        sir_dir = tmp_path / "sir"
        sir_files = normalize_sir(
            raw_files={"elevation": raw_path},
            schema=schema,
            output_dir=sir_dir,
            id_field="nhm_id",
        )

        assert "elevation_m_mean" in sir_files
        result = pd.read_csv(sir_files["elevation_m_mean"], index_col=0)
        assert "elevation_m_mean" in result.columns
        assert_allclose(result["elevation_m_mean"].values, [100.0, 200.0])

    def test_log_transform_applied(self, tmp_path: Path) -> None:
        """Log-transformed values are converted to linear scale."""
        import pandas as pd
        from numpy.testing import assert_allclose

        from hydro_param.sir import SIRVariableSchema, normalize_sir

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df = pd.DataFrame(
            {"ksat": [0.0, 1.0, 2.0]},
            index=pd.Index([1, 2, 3], name="nhm_id"),
        )
        raw_path = raw_dir / "ksat.csv"
        df.to_csv(raw_path)

        schema = [
            SIRVariableSchema(
                canonical_name="ksat_cm_hr_mean",
                source_name="ksat",
                source_units="log10(cm/hr)",
                canonical_units="cm/hr",
                long_name="Ksat",
                categorical=False,
                valid_range=None,
                conversion="log10_to_linear",
            )
        ]

        sir_dir = tmp_path / "sir"
        sir_files = normalize_sir(
            raw_files={"ksat": raw_path},
            schema=schema,
            output_dir=sir_dir,
            id_field="nhm_id",
        )

        result = pd.read_csv(sir_files["ksat_cm_hr_mean"], index_col=0)
        assert_allclose(result["ksat_cm_hr_mean"].values, [1.0, 10.0, 100.0])

    def test_categorical_fraction_rename(self, tmp_path: Path) -> None:
        """Categorical fraction columns are renamed with _frac_ pattern."""
        import pandas as pd

        from hydro_param.sir import SIRVariableSchema, normalize_sir

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df = pd.DataFrame(
            {"LndCov_11": [0.3, 0.1], "LndCov_21": [0.5, 0.2], "LndCov_41": [0.2, 0.7]},
            index=pd.Index([1, 2], name="nhm_id"),
        )
        raw_path = raw_dir / "LndCov.csv"
        df.to_csv(raw_path)

        schema = [
            SIRVariableSchema(
                canonical_name="lndcov_frac",
                source_name="LndCov",
                source_units="",
                canonical_units="",
                long_name="Land cover",
                categorical=True,
                valid_range=(0.0, 1.0),
                conversion=None,
            )
        ]

        sir_dir = tmp_path / "sir"
        sir_files = normalize_sir(
            raw_files={"LndCov": raw_path},
            schema=schema,
            output_dir=sir_dir,
            id_field="nhm_id",
        )

        assert "lndcov_frac" in sir_files
        result = pd.read_csv(sir_files["lndcov_frac"], index_col=0)
        assert "lndcov_frac_11" in result.columns
        assert "lndcov_frac_21" in result.columns
        assert "lndcov_frac_41" in result.columns

    def test_multiple_stats_produces_multiple_files(self, tmp_path: Path) -> None:
        """Multiple statistics produce separate canonical files."""
        import pandas as pd

        from hydro_param.sir import SIRVariableSchema, normalize_sir

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df = pd.DataFrame(
            {"elevation": [100.0, 200.0], "elevation_min": [90.0, 180.0]},
            index=pd.Index([1, 2], name="nhm_id"),
        )
        raw_path = raw_dir / "elevation.csv"
        df.to_csv(raw_path)

        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=None,
                conversion=None,
            ),
            SIRVariableSchema(
                canonical_name="elevation_m_min",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=None,
                conversion=None,
            ),
        ]

        sir_dir = tmp_path / "sir"
        sir_files = normalize_sir(
            raw_files={"elevation": raw_path},
            schema=schema,
            output_dir=sir_dir,
            id_field="nhm_id",
        )

        assert "elevation_m_mean" in sir_files
        assert "elevation_m_min" in sir_files

    def test_nan_values_preserved(self, tmp_path: Path) -> None:
        """NaN values pass through normalization unchanged."""
        import numpy as np
        import pandas as pd

        from hydro_param.sir import SIRVariableSchema, normalize_sir

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        df = pd.DataFrame(
            {"elevation": [100.0, np.nan, 300.0]},
            index=pd.Index([1, 2, 3], name="nhm_id"),
        )
        raw_path = raw_dir / "elevation.csv"
        df.to_csv(raw_path)

        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=None,
                conversion=None,
            )
        ]

        sir_dir = tmp_path / "sir"
        sir_files = normalize_sir(
            raw_files={"elevation": raw_path},
            schema=schema,
            output_dir=sir_dir,
            id_field="nhm_id",
        )

        result = pd.read_csv(sir_files["elevation_m_mean"], index_col=0)
        assert np.isnan(result["elevation_m_mean"].iloc[1])


class TestValidateSIR:
    """Tests for validate_sir()."""

    def _write_csv(self, tmp_path: Path, name: str, values: list[float], index: list[int]) -> Path:
        import pandas as pd

        df = pd.DataFrame({name: values}, index=pd.Index(index, name="nhm_id"))
        path = tmp_path / f"{name}.csv"
        df.to_csv(path)
        return path

    def test_valid_sir_no_warnings(self, tmp_path: Path) -> None:
        from hydro_param.sir import SIRVariableSchema, validate_sir

        path = self._write_csv(tmp_path, "elevation_m_mean", [100.0, 200.0], [1, 2])
        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=(-500.0, 9000.0),
                conversion=None,
            )
        ]
        warnings = validate_sir({"elevation_m_mean": path}, schema)
        assert warnings == []

    def test_all_nan_warns(self, tmp_path: Path) -> None:
        import numpy as np

        from hydro_param.sir import SIRVariableSchema, validate_sir

        path = self._write_csv(tmp_path, "elevation_m_mean", [np.nan, np.nan], [1, 2])
        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=None,
                conversion=None,
            )
        ]
        warnings = validate_sir({"elevation_m_mean": path}, schema)
        assert len(warnings) == 1
        assert warnings[0].check_type == "nan_coverage"

    def test_partial_nan_no_warning(self, tmp_path: Path) -> None:
        """Partial NaN coverage is expected (e.g., gridMET edge coverage)."""
        import numpy as np

        from hydro_param.sir import SIRVariableSchema, validate_sir

        path = self._write_csv(tmp_path, "elevation_m_mean", [100.0, np.nan, 300.0], [1, 2, 3])
        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=(-500.0, 9000.0),
                conversion=None,
            )
        ]
        warnings = validate_sir({"elevation_m_mean": path}, schema)
        assert warnings == []

    def test_out_of_range_warns(self, tmp_path: Path) -> None:
        from hydro_param.sir import SIRVariableSchema, validate_sir

        path = self._write_csv(tmp_path, "elevation_m_mean", [100.0, 99999.0], [1, 2])
        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=(-500.0, 9000.0),
                conversion=None,
            )
        ]
        warnings = validate_sir({"elevation_m_mean": path}, schema)
        assert any(w.check_type == "range" for w in warnings)

    def test_strict_mode_raises(self, tmp_path: Path) -> None:
        import numpy as np
        import pytest

        from hydro_param.sir import SIRValidationError, SIRVariableSchema, validate_sir

        path = self._write_csv(tmp_path, "elevation_m_mean", [np.nan, np.nan], [1, 2])
        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=None,
                conversion=None,
            )
        ]
        with pytest.raises(SIRValidationError):
            validate_sir({"elevation_m_mean": path}, schema, strict=True)

    def test_missing_schema_variable_warns(self, tmp_path: Path) -> None:
        """Schema expects variable not in sir_files -> warning."""
        from hydro_param.sir import SIRVariableSchema, validate_sir

        schema = [
            SIRVariableSchema(
                canonical_name="elevation_m_mean",
                source_name="elevation",
                source_units="m",
                canonical_units="m",
                long_name="Elevation",
                categorical=False,
                valid_range=None,
                conversion=None,
            )
        ]
        warnings = validate_sir({}, schema)
        assert any(w.check_type == "missing" for w in warnings)

    def test_categorical_count_column_not_range_checked(self, tmp_path: Path) -> None:
        """Count columns in categorical CSVs should not trigger range warnings."""
        import pandas as pd

        from hydro_param.sir import SIRVariableSchema, validate_sir

        # Simulate NLCD categorical output with fraction + count columns
        df = pd.DataFrame(
            {
                "lndcov_frac_11": [0.3, 0.5],
                "lndcov_frac_21": [0.7, 0.5],
                "count": [1000, 2000],  # pixel counts — NOT fractions
            },
            index=pd.Index([1, 2], name="nhm_id"),
        )
        path = tmp_path / "lndcov_frac.csv"
        df.to_csv(path)

        schema = [
            SIRVariableSchema(
                canonical_name="lndcov_frac",
                source_name="LndCov",
                source_units="",
                canonical_units="",
                long_name="Land Cover",
                categorical=True,
                valid_range=(0.0, 1.0),
                conversion=None,
            )
        ]

        warnings = validate_sir({"lndcov_frac": path}, schema)
        # count column values [1000, 2000] should NOT produce range warnings
        range_warnings = [w for w in warnings if w.check_type == "range"]
        assert len(range_warnings) == 0
