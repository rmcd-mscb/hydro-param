"""SIR normalization: canonical naming, unit conversion, and schema validation.

The Standardized Internal Representation (SIR) normalizes raw gdptools output
into self-documenting variable names with canonical SI units. See
docs/plans/2026-02-23-sir-normalization-design.md for the full design.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hydro_param.config import DatasetRequest
from hydro_param.dataset_registry import (
    DerivedVariableSpec,
    VariableSpec,
)

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


def _unit_info(units: str) -> tuple[str, str, str | None]:
    """Return (abbreviation, canonical_units, conversion) for a unit string."""
    if units == "":
        return ("", "", None)
    if units in _UNIT_TABLE:
        return _UNIT_TABLE[units]
    slug = re.sub(r"[^a-zA-Z0-9]", "_", units)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug, units, None)


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


def build_sir_schema(
    resolved: list[tuple[object, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
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
    for _entry, ds_req, var_specs in resolved:
        # Determine years for multi-year key suffixes
        if isinstance(ds_req.year, list):
            years: list[int | None] = list(ds_req.year)
        elif ds_req.year is not None:
            years = [ds_req.year]
        else:
            years = [None]

        for var_spec in var_specs:
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
