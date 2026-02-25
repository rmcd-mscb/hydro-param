# SIR Normalization Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Stage 5 to the pipeline that normalizes raw gdptools output into a Standardized Internal Representation (SIR) with self-documenting variable names, canonical SI units, and schema validation.

**Architecture:** New `sir.py` module provides schema generation, normalization, and validation. The SIR schema is auto-generated from existing dataset registry metadata (no new config files). Stage 5 reads raw per-variable CSVs from Stage 4, renames/converts/validates, and writes normalized per-variable CSVs to `output_dir/sir/`. Derivation plugins then consume canonical SIR names instead of raw source names.

**Tech Stack:** Python dataclasses, pandas, numpy, xarray (attrs), existing `units.py` registry.

**Design doc:** `docs/plans/2026-02-23-sir-normalization-design.md`

---

## Task 1: Unit Abbreviation Table and `canonical_name()` Function

**Files:**
- Create: `src/hydro_param/sir.py`
- Test: `tests/test_sir.py`

**Context:** The naming convention follows `<base>_<unit_abbrev>[_<stat>]`. The unit abbreviation table maps source units to short canonical abbreviations. Dimensionless quantities omit the unit part. Categorical variables use a `_frac_` pattern.

**Step 1: Write the failing tests**

```python
"""Tests for SIR normalization layer."""

from hydro_param.sir import canonical_name, unit_abbreviation


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
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'hydro_param.sir'`

**Step 3: Write minimal implementation**

Create `src/hydro_param/sir.py`:

```python
"""SIR normalization: canonical naming, unit conversion, and schema validation.

The Standardized Internal Representation (SIR) normalizes raw gdptools output
into self-documenting variable names with canonical SI units. See
docs/plans/2026-02-23-sir-normalization-design.md for the full design.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Unit abbreviation lookup: source_units -> (abbreviation, canonical_units, conversion)
# conversion is None (passthrough) or "log10_to_linear"
_UNIT_TABLE: dict[str, tuple[str, str, str | None]] = {
    "m": ("m", "m", None),
    "degrees": ("deg", "degrees", None),
    "%": ("pct", "%", None),
    "m3/m3": ("m3_m3", "m3/m3", None),
    "g/cm3": ("g_cm3", "g/cm3", None),
    "cm": ("cm", "cm", None),
    "cm/hr": ("cm_hr", "cm/hr", None),
    "log10(cm/hr)": ("cm_hr", "cm/hr", "log10_to_linear"),
    "log10(kPa)": ("kPa", "kPa", "log10_to_linear"),
    "log10(%)": ("pct", "%", "log10_to_linear"),
    "log10(kPa^-1)": ("kPa_inv", "kPa^-1", "log10_to_linear"),
    "day_of_year": ("doy", "day_of_year", None),
}


def unit_abbreviation(units: str) -> str:
    """Return the canonical abbreviation for a unit string.

    Parameters
    ----------
    units
        Source unit string from the dataset registry (e.g., ``"m"``,
        ``"log10(cm/hr)"``, ``"%"``).

    Returns
    -------
    str
        Short canonical abbreviation. Empty string for dimensionless.
    """
    if units == "":
        return ""
    if units in _UNIT_TABLE:
        return _UNIT_TABLE[units][0]
    # Unknown units: slugify (replace / with _, strip non-alphanumeric)
    slug = re.sub(r"[^a-zA-Z0-9]", "_", units)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug


def canonical_name(name: str, units: str, stat: str) -> str:
    """Generate a canonical SIR variable name.

    Follows the pattern ``<base>_<unit_abbrev>_<stat>``. Dimensionless
    quantities omit the unit segment: ``<base>_<stat>``.

    Parameters
    ----------
    name
        Base variable name from the dataset registry (e.g., ``"elevation"``).
    units
        Source unit string (e.g., ``"m"``, ``"log10(cm/hr)"``).
    stat
        Aggregation statistic (e.g., ``"mean"``, ``"min"``, ``"majority"``).

    Returns
    -------
    str
        Canonical SIR variable name (e.g., ``"elevation_m_mean"``).
    """
    base = name.lower()
    abbrev = unit_abbreviation(units)
    if abbrev:
        return f"{base}_{abbrev}_{stat}"
    return f"{base}_{stat}"
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add canonical_name() and unit_abbreviation() for SIR naming convention"
```

---

## Task 2: `SIRVariableSchema` Dataclass and `build_sir_schema()`

**Files:**
- Modify: `src/hydro_param/sir.py`
- Test: `tests/test_sir.py`

**Context:** `build_sir_schema()` takes the resolved datasets from stage 2 (list of `(DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec])` tuples) and auto-generates a `SIRVariableSchema` for each variable. It must handle: continuous variables with statistics, categorical variables (which produce fraction columns), derived variables, and multi-year datasets (where keys are `var_name_YYYY`).

**Key file to understand:** `src/hydro_param/pipeline.py:820-829` — how `result_key` is constructed for multi-year datasets: `f"{var_name}_{year}"` if year is not None.

**Step 1: Write the failing tests**

Add to `tests/test_sir.py`:

```python
import numpy as np
import pandas as pd
import pytest

from hydro_param.sir import (
    SIRVariableSchema,
    build_sir_schema,
    canonical_name,
    unit_abbreviation,
)
from hydro_param.config import DatasetRequest, PipelineConfig
from hydro_param.dataset_registry import (
    DatasetEntry,
    DerivedVariableSpec,
    VariableSpec,
)


class TestSIRVariableSchema:
    """Tests for SIRVariableSchema dataclass."""

    def test_basic_creation(self) -> None:
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
        entry = self._make_entry(
            variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
        )
        ds_req = DatasetRequest(name="dem_test", variables=["elevation"], statistics=["mean"])
        var_specs = [VariableSpec(name="elevation", units="m", long_name="Elevation")]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert len(schema) == 1
        assert schema[0].canonical_name == "elevation_m_mean"
        assert schema[0].source_name == "elevation"
        assert schema[0].conversion is None

    def test_multiple_statistics(self) -> None:
        entry = self._make_entry(
            variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
        )
        ds_req = DatasetRequest(
            name="dem_test", variables=["elevation"], statistics=["mean", "min", "max"]
        )
        var_specs = [VariableSpec(name="elevation", units="m", long_name="Elevation")]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        names = {s.canonical_name for s in schema}
        assert names == {"elevation_m_mean", "elevation_m_min", "elevation_m_max"}

    def test_derived_variable(self) -> None:
        entry = self._make_entry(
            derived_variables=[
                DerivedVariableSpec(
                    name="slope", source="elevation", method="horn", units="degrees"
                )
            ]
        )
        ds_req = DatasetRequest(name="dem_test", variables=["slope"], statistics=["mean"])
        var_specs = [
            DerivedVariableSpec(
                name="slope", source="elevation", method="horn", units="degrees"
            )
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert len(schema) == 1
        assert schema[0].canonical_name == "slope_deg_mean"

    def test_log_transform_variable(self) -> None:
        entry = self._make_entry(
            strategy="local_tiff",
            variables=[
                VariableSpec(name="ksat", units="log10(cm/hr)", long_name="Ksat")
            ],
            category="soils",
        )
        ds_req = DatasetRequest(name="polaris", variables=["ksat"], statistics=["mean"])
        var_specs = [VariableSpec(name="ksat", units="log10(cm/hr)", long_name="Ksat")]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert schema[0].canonical_name == "ksat_cm_hr_mean"
        assert schema[0].source_units == "log10(cm/hr)"
        assert schema[0].canonical_units == "cm/hr"
        assert schema[0].conversion == "log10_to_linear"

    def test_categorical_variable(self) -> None:
        entry = self._make_entry(
            strategy="nhgf_stac",
            variables=[
                VariableSpec(name="LndCov", long_name="Land cover", categorical=True)
            ],
            category="land_cover",
        )
        ds_req = DatasetRequest(name="nlcd_test", variables=["LndCov"], statistics=["mean"])
        var_specs = [
            VariableSpec(name="LndCov", long_name="Land cover", categorical=True)
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        # Categorical variables produce a single schema entry with
        # stat="frac" to indicate fraction columns will be generated
        assert len(schema) == 1
        assert schema[0].categorical is True
        assert schema[0].source_name == "LndCov"

    def test_multi_year_keys(self) -> None:
        entry = self._make_entry(
            variables=[VariableSpec(name="elevation", units="m", long_name="Elevation")]
        )
        ds_req = DatasetRequest(
            name="dem_test", variables=["elevation"], statistics=["mean"], year=[2020, 2021]
        )
        var_specs = [VariableSpec(name="elevation", units="m", long_name="Elevation")]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        names = {s.canonical_name for s in schema}
        assert names == {"elevation_m_mean_2020", "elevation_m_mean_2021"}

    def test_dimensionless_variable(self) -> None:
        entry = self._make_entry(
            strategy="local_tiff",
            variables=[
                VariableSpec(name="lambda", units="", long_name="Pore size distribution")
            ],
            category="soils",
        )
        ds_req = DatasetRequest(name="polaris", variables=["lambda"], statistics=["mean"])
        var_specs = [
            VariableSpec(name="lambda", units="", long_name="Pore size distribution")
        ]

        schema = build_sir_schema([(entry, ds_req, var_specs)])
        assert schema[0].canonical_name == "lambda_mean"
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir.py::TestSIRVariableSchema -v && pixi run -e dev pytest tests/test_sir.py::TestBuildSIRSchema -v`
Expected: FAIL — `ImportError: cannot import name 'SIRVariableSchema'`

**Step 3: Write minimal implementation**

Add to `src/hydro_param/sir.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from hydro_param.config import DatasetRequest
from hydro_param.dataset_registry import (
    DatasetEntry,
    DerivedVariableSpec,
    VariableSpec,
)


@dataclass
class SIRVariableSchema:
    """Schema entry for a single SIR variable."""

    canonical_name: str
    source_name: str
    source_units: str
    canonical_units: str
    long_name: str
    categorical: bool
    valid_range: tuple[float, float] | None
    conversion: str | None


def _unit_info(units: str) -> tuple[str, str, str | None]:
    """Return (abbreviation, canonical_units, conversion) for a unit string."""
    if units == "":
        return ("", "", None)
    if units in _UNIT_TABLE:
        return _UNIT_TABLE[units]
    slug = re.sub(r"[^a-zA-Z0-9]", "_", units)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug, units, None)


def build_sir_schema(
    resolved: list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
) -> list[SIRVariableSchema]:
    """Auto-generate SIR schema from stage 2 resolved datasets.

    Parameters
    ----------
    resolved
        Output of ``stage2_resolve_datasets()``: list of
        ``(DatasetEntry, DatasetRequest, var_specs)`` tuples.

    Returns
    -------
    list[SIRVariableSchema]
        One schema entry per expected SIR output column.
    """
    schema: list[SIRVariableSchema] = []
    for entry, ds_req, var_specs in resolved:
        # Determine years for multi-year key suffixes
        if isinstance(ds_req.year, list):
            years: list[int | None] = list(ds_req.year)
        elif ds_req.year is not None:
            years = [ds_req.year]
        else:
            years = [None]

        for var_spec in var_specs:
            if isinstance(var_spec, DerivedVariableSpec):
                units = var_spec.units
                long_name = var_spec.long_name or var_spec.name
            else:
                units = var_spec.units
                long_name = var_spec.long_name or var_spec.name

            abbrev, canonical_units, conversion = _unit_info(units)
            categorical = isinstance(var_spec, VariableSpec) and var_spec.categorical

            if categorical:
                # Categorical variables: single schema entry, fraction columns
                # generated dynamically during normalization
                for year in years:
                    cname = canonical_name(var_spec.name, "", "frac")
                    if year is not None:
                        cname = f"{cname}_{year}"
                    schema.append(
                        SIRVariableSchema(
                            canonical_name=cname,
                            source_name=var_spec.name,
                            source_units=units,
                            canonical_units=canonical_units or "",
                            long_name=long_name,
                            categorical=True,
                            valid_range=(0.0, 1.0),
                            conversion=conversion,
                        )
                    )
            else:
                # Continuous: one schema entry per statistic per year
                for stat in ds_req.statistics:
                    for year in years:
                        cname = canonical_name(var_spec.name, units, stat)
                        if year is not None:
                            cname = f"{cname}_{year}"
                        schema.append(
                            SIRVariableSchema(
                                canonical_name=cname,
                                source_name=var_spec.name,
                                source_units=units,
                                canonical_units=canonical_units or units,
                                long_name=long_name,
                                categorical=False,
                                valid_range=None,
                                conversion=conversion,
                            )
                        )
    return schema
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add SIRVariableSchema and build_sir_schema() for auto-generated SIR schemas"
```

---

## Task 3: Unit Conversions for SIR Normalization

**Files:**
- Modify: `src/hydro_param/units.py` (register log10 conversions)
- Modify: `src/hydro_param/sir.py` (add `apply_conversion()`)
- Test: `tests/test_sir.py`

**Context:** The SIR normalization must convert log-transformed POLARIS values (e.g., `log10(cm/hr)`) to linear scale. Register these in the existing `units.py` conversion registry so they are centralized.

**Step 1: Write the failing tests**

Add to `tests/test_sir.py`:

```python
import numpy as np
from numpy.testing import assert_allclose

from hydro_param.sir import apply_conversion


class TestApplyConversion:
    """Tests for SIR unit conversion application."""

    def test_no_conversion(self) -> None:
        values = np.array([100.0, 200.0, 300.0])
        result = apply_conversion(values, conversion=None)
        assert_allclose(result, values)

    def test_log10_to_linear(self) -> None:
        values = np.array([0.0, 1.0, 2.0])  # log10 scale
        result = apply_conversion(values, conversion="log10_to_linear")
        assert_allclose(result, [1.0, 10.0, 100.0])

    def test_log10_to_linear_with_nan(self) -> None:
        values = np.array([1.0, np.nan, 2.0])
        result = apply_conversion(values, conversion="log10_to_linear")
        assert_allclose(result[0], 10.0)
        assert np.isnan(result[1])
        assert_allclose(result[2], 100.0)

    def test_unknown_conversion_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown conversion"):
            apply_conversion(np.array([1.0]), conversion="unknown_transform")
```

Also add to `tests/test_units.py`:

```python
class TestLog10Conversions:
    """Tests for log10 unit conversions registered for SIR normalization."""

    def test_log10_cm_hr_to_cm_hr(self) -> None:
        from hydro_param.units import convert
        values = np.array([0.0, 1.0, 2.0])
        result = convert(values, "log10(cm/hr)", "cm/hr")
        assert_allclose(result, [1.0, 10.0, 100.0])

    def test_log10_kpa_to_kpa(self) -> None:
        from hydro_param.units import convert
        values = np.array([0.0, 1.0, 2.0])
        result = convert(values, "log10(kPa)", "kPa")
        assert_allclose(result, [1.0, 10.0, 100.0])

    def test_log10_pct_to_pct(self) -> None:
        from hydro_param.units import convert
        values = np.array([0.0, 1.0, 2.0])
        result = convert(values, "log10(%)", "%")
        assert_allclose(result, [1.0, 10.0, 100.0])

    def test_log10_kpa_inv_to_kpa_inv(self) -> None:
        from hydro_param.units import convert
        values = np.array([0.0, -1.0, 1.0])
        result = convert(values, "log10(kPa^-1)", "kPa^-1")
        assert_allclose(result, [1.0, 0.1, 10.0])
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir.py::TestApplyConversion tests/test_units.py::TestLog10Conversions -v`
Expected: FAIL — missing `apply_conversion` and unregistered log10 conversions

**Step 3: Write minimal implementation**

Add to `src/hydro_param/units.py` (after existing registrations):

```python
# Log-transform conversions (SIR normalization: source -> canonical SI)
register("log10(cm/hr)", "cm/hr", lambda v: np.power(10.0, v), "log10 Ksat to linear cm/hr")
register("log10(kPa)", "kPa", lambda v: np.power(10.0, v), "log10 pressure to linear kPa")
register("log10(%)", "%", lambda v: np.power(10.0, v), "log10 percent to linear %")
register("log10(kPa^-1)", "kPa^-1", lambda v: np.power(10.0, v), "log10 inverse pressure to linear")
```

Add to `src/hydro_param/sir.py`:

```python
import numpy as np
from numpy.typing import NDArray


def apply_conversion(
    values: NDArray[np.floating],
    conversion: str | None,
) -> NDArray[np.floating]:
    """Apply a SIR unit conversion to an array of values.

    Parameters
    ----------
    values
        Input values in source units.
    conversion
        Conversion type: ``None`` (passthrough), ``"log10_to_linear"``
        (10^x), or raises ``ValueError`` for unknown types.

    Returns
    -------
    NDArray
        Converted values in canonical units.
    """
    if conversion is None:
        return values
    if conversion == "log10_to_linear":
        return np.power(10.0, values)
    raise ValueError(f"Unknown conversion: {conversion!r}")
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py::TestApplyConversion tests/test_units.py::TestLog10Conversions -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py src/hydro_param/units.py tests/test_sir.py tests/test_units.py
git commit -m "feat: add log10 unit conversions and apply_conversion() for SIR normalization"
```

---

## Task 4: `normalize_sir()` Function

**Files:**
- Modify: `src/hydro_param/sir.py`
- Test: `tests/test_sir.py`

**Context:** `normalize_sir()` reads the raw per-variable CSV files from stage 4, renames columns to canonical names, applies unit conversions, and writes normalized CSV files to `output_dir/sir/`. Raw CSVs have `id_field` as the index column and one or more data columns. For continuous variables with stat "mean", the column is named `var_name` (e.g., `elevation`). For other stats, columns are `var_name_stat` (e.g., `elevation_min`). For categorical variables, columns are `VarName_classCode` (e.g., `LndCov_11`, `LndCov_21`).

**Step 1: Write the failing tests**

Add to `tests/test_sir.py`:

```python
from hydro_param.sir import normalize_sir


class TestNormalizeSIR:
    """Tests for normalize_sir()."""

    def test_continuous_mean_rename(self, tmp_path: Path) -> None:
        """Single continuous variable with mean stat gets canonical name."""
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
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir.py::TestNormalizeSIR -v`
Expected: FAIL — `ImportError: cannot import name 'normalize_sir'`

**Step 3: Write minimal implementation**

Add to `src/hydro_param/sir.py`:

```python
import pandas as pd


def normalize_sir(
    raw_files: dict[str, Path],
    schema: list[SIRVariableSchema],
    output_dir: Path,
    id_field: str,
) -> dict[str, Path]:
    """Normalize raw per-variable files to canonical SIR format.

    Reads raw CSVs from stage 4, renames columns to canonical names,
    applies unit conversions, and writes normalized per-variable CSVs
    to ``output_dir/``.

    Parameters
    ----------
    raw_files
        Mapping of source variable key to raw CSV file path
        (output of stage 4).
    schema
        SIR variable schema entries (output of ``build_sir_schema()``).
    output_dir
        Directory to write normalized CSV files.
    id_field
        Feature ID column name (used as CSV index).

    Returns
    -------
    dict[str, Path]
        Mapping of canonical name to normalized CSV file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    sir_files: dict[str, Path] = {}

    # Index schema by source_name for lookup
    schema_by_source: dict[str, list[SIRVariableSchema]] = {}
    for entry in schema:
        schema_by_source.setdefault(entry.source_name, []).append(entry)

    for raw_key, raw_path in raw_files.items():
        raw_df = pd.read_csv(raw_path, index_col=0)

        # Determine the base source name: strip year suffix if present
        # (e.g., "elevation_2020" -> "elevation")
        base_source = raw_key
        year_suffix = ""
        for s in schema:
            if raw_key.startswith(s.source_name) and raw_key != s.source_name:
                candidate_suffix = raw_key[len(s.source_name) :]
                if re.match(r"^_\d{4}$", candidate_suffix):
                    base_source = s.source_name
                    year_suffix = candidate_suffix
                    break

        entries = schema_by_source.get(base_source, [])
        if not entries:
            logger.warning(
                "No SIR schema entry for raw variable '%s' — skipping", raw_key
            )
            continue

        for entry in entries:
            # Check if this schema entry matches the year suffix
            if year_suffix and not entry.canonical_name.endswith(year_suffix):
                continue
            if not year_suffix and re.search(r"_\d{4}$", entry.canonical_name):
                continue

            if entry.categorical:
                # Categorical: rename fraction columns
                rename_map = {}
                prefix = entry.source_name
                for col in raw_df.columns:
                    if col.startswith(f"{prefix}_"):
                        class_code = col[len(prefix) + 1 :]
                        new_col = f"{prefix.lower()}_frac_{class_code}"
                        rename_map[col] = new_col
                if rename_map:
                    out_df = raw_df[list(rename_map.keys())].rename(columns=rename_map)
                else:
                    out_df = raw_df.copy()
                out_path = output_dir / f"{entry.canonical_name}.csv"
                out_df.to_csv(out_path)
                sir_files[entry.canonical_name] = out_path
                logger.info("SIR normalized: %s → %s", raw_key, out_path.name)
            else:
                # Continuous: find the matching column and rename
                # Column naming from _write_variable_file:
                #   "mean" stat → column named var_name (e.g., "elevation")
                #   other stats → column named var_name_stat (e.g., "elevation_min")
                cname = entry.canonical_name
                # Extract stat from canonical name (last segment after removing year)
                cname_no_year = re.sub(r"_\d{4}$", "", cname)
                parts = cname_no_year.rsplit("_", 1)
                stat = parts[-1] if len(parts) > 1 else "mean"

                # Find source column name
                if stat == "mean":
                    source_col = base_source
                    if year_suffix:
                        source_col = raw_key
                else:
                    source_col = f"{raw_key}_{stat}"

                if source_col not in raw_df.columns:
                    # Try without year suffix
                    alt_col = f"{base_source}_{stat}" if stat != "mean" else base_source
                    if alt_col in raw_df.columns:
                        source_col = alt_col
                    else:
                        logger.warning(
                            "Column '%s' not found in %s (available: %s) — skipping",
                            source_col,
                            raw_path.name,
                            list(raw_df.columns),
                        )
                        continue

                values = raw_df[source_col].values.astype(np.float64)

                # Apply unit conversion
                if entry.conversion is not None:
                    values = apply_conversion(values, entry.conversion)

                out_df = pd.DataFrame(
                    {cname: values},
                    index=raw_df.index,
                )
                out_df.index.name = id_field
                out_path = output_dir / f"{cname}.csv"
                out_df.to_csv(out_path)
                sir_files[cname] = out_path
                logger.info("SIR normalized: %s → %s", raw_key, out_path.name)

    return sir_files
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py::TestNormalizeSIR -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add normalize_sir() for raw-to-canonical SIR file transformation"
```

---

## Task 5: `validate_sir()` Function

**Files:**
- Modify: `src/hydro_param/sir.py`
- Test: `tests/test_sir.py`

**Context:** `validate_sir()` checks normalized SIR files against the schema. It returns a list of `SIRValidationWarning` dataclasses. In strict mode, any warnings become errors.

**Step 1: Write the failing tests**

Add to `tests/test_sir.py`:

```python
from hydro_param.sir import (
    SIRValidationError,
    SIRValidationWarning,
    validate_sir,
)


class TestValidateSIR:
    """Tests for validate_sir()."""

    def _write_csv(
        self, tmp_path: Path, name: str, values: list[float], index: list[int]
    ) -> Path:
        df = pd.DataFrame(
            {name: values}, index=pd.Index(index, name="nhm_id")
        )
        path = tmp_path / f"{name}.csv"
        df.to_csv(path)
        return path

    def test_valid_sir_no_warnings(self, tmp_path: Path) -> None:
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
        path = self._write_csv(
            tmp_path, "elevation_m_mean", [100.0, np.nan, 300.0], [1, 2, 3]
        )
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
        path = self._write_csv(
            tmp_path, "elevation_m_mean", [100.0, 99999.0], [1, 2]
        )
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
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_sir.py::TestValidateSIR -v`
Expected: FAIL — `ImportError: cannot import name 'validate_sir'`

**Step 3: Write minimal implementation**

Add to `src/hydro_param/sir.py`:

```python
class SIRValidationError(Exception):
    """Raised when SIR validation fails in strict mode."""

    def __init__(self, warnings: list[SIRValidationWarning]) -> None:
        self.warnings = warnings
        messages = [f"  [{w.check_type}] {w.variable}: {w.message}" for w in warnings]
        super().__init__(f"SIR validation failed with {len(warnings)} warnings:\n" + "\n".join(messages))


@dataclass
class SIRValidationWarning:
    """A single SIR validation warning."""

    variable: str
    check_type: str
    message: str


def validate_sir(
    sir_files: dict[str, Path],
    schema: list[SIRVariableSchema],
    *,
    strict: bool = False,
) -> list[SIRValidationWarning]:
    """Validate normalized SIR files against schema.

    Parameters
    ----------
    sir_files
        Mapping of canonical name to normalized CSV file path.
    schema
        SIR variable schema entries.
    strict
        If ``True``, raise ``SIRValidationError`` on any warnings.

    Returns
    -------
    list[SIRValidationWarning]
        Validation warnings (empty list = valid).
    """
    warnings: list[SIRValidationWarning] = []

    # Check completeness: schema variables present in files
    for entry in schema:
        if entry.canonical_name not in sir_files:
            warnings.append(
                SIRValidationWarning(
                    variable=entry.canonical_name,
                    check_type="missing",
                    message=f"Expected variable '{entry.canonical_name}' not found in SIR output",
                )
            )

    # Check each file
    for cname, path in sir_files.items():
        df = pd.read_csv(path, index_col=0)

        # Find matching schema entry
        matching = [s for s in schema if s.canonical_name == cname]

        for col in df.columns:
            values = df[col].values.astype(np.float64)

            # NaN coverage: warn only if 100% NaN
            if np.all(np.isnan(values)):
                warnings.append(
                    SIRValidationWarning(
                        variable=col,
                        check_type="nan_coverage",
                        message=f"Variable '{col}' is 100% NaN — possible processing failure",
                    )
                )

            # Value range checks
            if matching:
                entry = matching[0]
                if entry.valid_range is not None:
                    vmin, vmax = entry.valid_range
                    non_nan = values[~np.isnan(values)]
                    if len(non_nan) > 0:
                        if np.any(non_nan < vmin) or np.any(non_nan > vmax):
                            actual_min = float(np.nanmin(non_nan))
                            actual_max = float(np.nanmax(non_nan))
                            warnings.append(
                                SIRValidationWarning(
                                    variable=col,
                                    check_type="range",
                                    message=(
                                        f"Values [{actual_min:.2f}, {actual_max:.2f}] "
                                        f"outside expected range [{vmin}, {vmax}]"
                                    ),
                                )
                            )

    for w in warnings:
        logger.warning("SIR validation: [%s] %s: %s", w.check_type, w.variable, w.message)

    if strict and warnings:
        raise SIRValidationError(warnings)

    return warnings
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_sir.py::TestValidateSIR -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/sir.py tests/test_sir.py
git commit -m "feat: add validate_sir() with tolerant/strict modes for SIR schema validation"
```

---

## Task 6: Config Addition and Pipeline Stage 5 Integration

**Files:**
- Modify: `src/hydro_param/config.py` — add `sir_validation` field to `ProcessingConfig`
- Modify: `src/hydro_param/pipeline.py` — add stage 5, update `PipelineResult`
- Test: `tests/test_config.py` (validation field), `tests/test_pipeline.py` (stage 5)

**Context:** Stage 5 is called after stage 4 in `run_pipeline_from_config()`. It builds the SIR schema from the resolved datasets, calls `normalize_sir()` on the raw stage 4 files, then calls `validate_sir()`. The `PipelineResult` dataclass gains `sir_files` and `sir_schema` fields. `load_sir()` now reads from `sir_files`.

**Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_sir_validation_default() -> None:
    """ProcessingConfig defaults sir_validation to 'tolerant'."""
    from hydro_param.config import ProcessingConfig
    config = ProcessingConfig()
    assert config.sir_validation == "tolerant"


def test_sir_validation_strict() -> None:
    from hydro_param.config import ProcessingConfig
    config = ProcessingConfig(sir_validation="strict")
    assert config.sir_validation == "strict"
```

Add to `tests/test_pipeline.py`:

```python
from hydro_param.sir import SIRVariableSchema


class TestPipelineResultSIR:
    """Tests for PipelineResult with SIR normalization."""

    def test_sir_fields_default_empty(self) -> None:
        result = PipelineResult(output_dir=Path("/tmp"))
        assert result.sir_files == {}
        assert result.sir_schema == []

    def test_load_sir_from_sir_files(self, tmp_path: Path) -> None:
        """load_sir() reads from sir_files (normalized) when available."""
        df = pd.DataFrame(
            {"elevation_m_mean": [100.0, 200.0]},
            index=pd.Index([1, 2], name="nhm_id"),
        )
        sir_path = tmp_path / "elevation_m_mean.csv"
        df.to_csv(sir_path)

        result = PipelineResult(
            output_dir=tmp_path,
            sir_files={"elevation_m_mean": sir_path},
        )
        sir = result.load_sir()
        assert "elevation_m_mean" in sir.data_vars

    def test_load_sir_falls_back_to_static(self, tmp_path: Path) -> None:
        """load_sir() falls back to static_files when no sir_files."""
        df = pd.DataFrame(
            {"elevation": [100.0]},
            index=pd.Index([1], name="nhm_id"),
        )
        path = tmp_path / "elevation.csv"
        df.to_csv(path)

        result = PipelineResult(
            output_dir=tmp_path,
            static_files={"elevation": path},
        )
        sir = result.load_sir()
        assert "elevation" in sir.data_vars

    def test_load_raw_sir(self, tmp_path: Path) -> None:
        """load_raw_sir() always reads from static_files."""
        df = pd.DataFrame(
            {"elevation": [100.0]},
            index=pd.Index([1], name="nhm_id"),
        )
        path = tmp_path / "elevation.csv"
        df.to_csv(path)

        result = PipelineResult(
            output_dir=tmp_path,
            static_files={"elevation": path},
            sir_files={"elevation_m_mean": path},  # even with sir_files present
        )
        raw = result.load_raw_sir()
        assert "elevation" in raw.data_vars
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_config.py::test_sir_validation_default tests/test_pipeline.py::TestPipelineResultSIR -v`
Expected: FAIL — missing `sir_validation` field and `sir_files` attribute

**Step 3: Write minimal implementation**

Modify `src/hydro_param/config.py` — add to `ProcessingConfig`:

```python
class ProcessingConfig(BaseModel):
    """Processing options."""

    engine: Literal["exactextract", "serial"] = "exactextract"
    failure_mode: Literal["strict", "tolerant"] = "strict"
    batch_size: int = Field(default=500, gt=0)
    resume: bool = False
    sir_validation: Literal["tolerant", "strict"] = "tolerant"
```

Modify `src/hydro_param/pipeline.py`:

1. Add imports at top:
```python
from hydro_param.sir import (
    SIRVariableSchema,
    build_sir_schema,
    normalize_sir,
    validate_sir,
)
```

2. Update `PipelineResult`:
```python
@dataclass
class PipelineResult:
    """Full pipeline result with file paths and lazy SIR loading."""

    output_dir: Path
    static_files: dict[str, Path] = field(default_factory=dict)
    temporal_files: dict[str, Path] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)
    fabric: gpd.GeoDataFrame | None = None
    sir_files: dict[str, Path] = field(default_factory=dict)
    sir_schema: list[SIRVariableSchema] = field(default_factory=list)

    def load_sir(self) -> xr.Dataset:
        """Load normalized SIR files into a combined xr.Dataset.

        Uses ``sir_files`` (normalized) when available, falling back
        to ``static_files`` (raw) for backward compatibility.
        """
        files = self.sir_files if self.sir_files else self.static_files
        if not files:
            return xr.Dataset()
        dfs = [pd.read_csv(p, index_col=0) for p in files.values()]
        combined = pd.concat(dfs, axis=1)
        return xr.Dataset.from_dataframe(combined)

    def load_raw_sir(self) -> xr.Dataset:
        """Load raw (pre-normalization) static files into a combined xr.Dataset."""
        if not self.static_files:
            return xr.Dataset()
        dfs = [pd.read_csv(p, index_col=0) for p in self.static_files.values()]
        combined = pd.concat(dfs, axis=1)
        return xr.Dataset.from_dataframe(combined)
```

3. Add `stage5_normalize_sir()` function:
```python
def stage5_normalize_sir(
    stage4: Stage4Results,
    resolved: list[tuple[DatasetEntry, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
    config: PipelineConfig,
) -> tuple[dict[str, Path], list[SIRVariableSchema]]:
    """Stage 5: Normalize raw stage 4 output to canonical SIR format.

    Parameters
    ----------
    stage4
        Stage 4 results with raw per-variable file paths.
    resolved
        Resolved dataset entries from stage 2.
    config
        Pipeline configuration.

    Returns
    -------
    tuple[dict[str, Path], list[SIRVariableSchema]]
        Normalized SIR file paths and the schema used.
    """
    logger.info("Stage 5: SIR normalization")
    schema = build_sir_schema(resolved)
    logger.info("  SIR schema: %d variables", len(schema))

    sir_dir = config.output.path / "sir"
    sir_files = normalize_sir(
        raw_files=stage4.static_files,
        schema=schema,
        output_dir=sir_dir,
        id_field=config.target_fabric.id_field,
    )
    logger.info("  Normalized %d SIR files → %s", len(sir_files), sir_dir)

    strict = config.processing.sir_validation == "strict"
    warnings = validate_sir(sir_files, schema, strict=strict)
    if warnings:
        logger.warning("  SIR validation: %d warnings", len(warnings))
    else:
        logger.info("  SIR validation: passed")

    return sir_files, schema
```

4. Wire stage 5 into `run_pipeline_from_config()` — after stage 4 results, before building `PipelineResult`:

```python
    # Stage 5: Normalize SIR
    t5 = time.perf_counter()
    sir_files, sir_schema = stage5_normalize_sir(results, resolved, config)
    logger.info("Stage 5 complete (%.1fs)", time.perf_counter() - t5)
```

And update the returned `PipelineResult`:
```python
    return PipelineResult(
        output_dir=config.output.path,
        static_files=results.static_files,
        temporal_files=results.temporal_files,
        categories=results.categories,
        fabric=fabric,
        sir_files=sir_files,
        sir_schema=sir_schema,
    )
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_config.py tests/test_pipeline.py tests/test_pipeline_derivation.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/config.py src/hydro_param/pipeline.py tests/test_config.py tests/test_pipeline.py
git commit -m "feat: add Stage 5 SIR normalization to pipeline with sir_validation config"
```

---

## Task 7: Update pywatershed Derivation Plugin for Canonical SIR Names

**Files:**
- Modify: `src/hydro_param/derivations/pywatershed.py`
- Modify: `src/hydro_param/cli.py` (remove `rename_sir_variables` call)
- Test: `tests/test_pywatershed_derivation.py`

**Context:** The derivation plugin currently expects raw SIR names (`elevation`, `slope`, `aspect`, `impervious`, `tree_canopy`, `LndCov_*`). After SIR normalization, these become `elevation_m_mean`, `slope_deg_mean`, `aspect_deg_mean`, `fctimp_pct_mean`, `tree_canopy_pct_mean`, `lndcov_frac_*`. The plugin needs updating to look for canonical names. The `rename_sir_variables()` method becomes unnecessary.

**Important:** Model-specific unit conversions (m→ft, deg→rad) STAY in the plugin. Only the variable name lookup changes.

**Step 1: Read the full derivation test file and plugin to understand all SIR variable references**

Read: `tests/test_pywatershed_derivation.py`
Read: `src/hydro_param/derivations/pywatershed.py` (already read above)

Identify all SIR variable names referenced in the plugin:
- `elevation` → `elevation_m_mean`
- `slope` → `slope_deg_mean`
- `aspect` → `aspect_deg_mean`
- `land_cover` / `land_cover_majority` → `lndcov_majority` (or categorical fracs `lndcov_frac_*`)
- `impervious` → `fctimp_pct_mean` (after rename_sir_variables mapped `FctImp_mean`)
- `tree_canopy` → `tree_canopy_pct_mean`
- `LndCov_*` → `lndcov_frac_*`
- `hru_area_m2` → `hru_area_m2_m_mean` (or computed from fabric — most common path)
- `hru_lat` → computed from fabric (most common path)

**Step 2: Update test fixtures to use canonical SIR names**

Update each test that constructs a SIR `xr.Dataset` to use the new names. For example, where tests use `coords={"nhm_id": [...]}` with `sir["elevation"]`, change to `sir["elevation_m_mean"]`.

**Step 3: Update derivation plugin lookups**

In `_derive_topography()`:
- `"elevation" in sir` → `"elevation_m_mean" in sir`
- `sir["elevation"].values` → `sir["elevation_m_mean"].values`
- Same for `slope`, `aspect`

In `_derive_landcover()`:
- `"impervious" in sir` → `"fctimp_pct_mean" in sir` (canonical name for fractional impervious)
- `"tree_canopy" in sir` → `"tree_canopy_pct_mean" in sir`
- `_compute_majority_from_fractions()` prefixes change: `("LndCov_", "land_cover_")` → `("lndcov_frac_",)`

In `rename_sir_variables()`:
- Keep the method but make it a no-op that logs a deprecation warning.
- Or remove entirely and update CLI call.

In `cli.py`:
- Remove or simplify the `sir_renamed = plugin.rename_sir_variables(result.load_sir())` call.
- Since `result.load_sir()` now returns normalized SIR, pass it directly.

**Step 4: Run all derivation and pipeline tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_derivation.py tests/test_pipeline_derivation.py tests/test_cli.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/hydro_param/derivations/pywatershed.py src/hydro_param/cli.py tests/test_pywatershed_derivation.py
git commit -m "refactor: update pywatershed derivation plugin for canonical SIR variable names"
```

---

## Task 8: Full Test Suite Green + Pre-push Checks

**Files:**
- All modified files from tasks 1-7

**Step 1: Run full test suite**

Run: `pixi run -e dev test`
Expected: All tests pass

**Step 2: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: All pass (ruff, mypy, detect-secrets)

**Step 3: Run full check**

Run: `pixi run -e dev check`
Expected: All pass (lint, format-check, typecheck, tests)

**Step 4: Fix any issues found**

If any check fails, fix the issue and re-run.

**Step 5: Final commit if needed**

```bash
git add -u
git commit -m "chore: fix lint/type issues from SIR normalization"
```

---

## Task 9: Update Pipeline Docstring and Module Docstrings

**Files:**
- Modify: `src/hydro_param/pipeline.py` (docstring mentions stages 1-4, needs to say 1-5)
- Modify: `src/hydro_param/sir.py` (ensure module docstring is complete)

**Step 1: Update pipeline module docstring**

Change `pipeline.py` line 1-8 from mentioning "4-stage pipeline" to "5-stage pipeline" and add stage 5 to the list.

**Step 2: Verify sir.py module docstring is complete**

Ensure it references the design doc and explains the module's role.

**Step 3: Commit**

```bash
git add src/hydro_param/pipeline.py src/hydro_param/sir.py
git commit -m "docs: update pipeline docstring for 5-stage pipeline with SIR normalization"
```
