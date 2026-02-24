"""SIR normalization: canonical naming, unit conversion, and schema validation.

The Standardized Internal Representation (SIR) normalizes raw gdptools output
into self-documenting variable names with canonical units. See
docs/plans/2026-02-23-sir-normalization-design.md for the full design.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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
    "K": ("C", "°C", "K_to_C"),
    "mm": ("mm", "mm", None),
    "W/m2": ("W_m2", "W/m2", None),
    "kg/kg": ("kg_kg", "kg/kg", None),
    "m/s": ("m_s", "m/s", None),
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
    resolved: Sequence[tuple[object, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
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
    if conversion == "K_to_C":
        return values - 273.15
    raise ValueError(f"Unknown conversion: {conversion!r}")


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
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    sir_files: dict[str, Path] = {}
    skipped_variables: list[str] = []

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
            logger.warning("No SIR schema entry for raw variable '%s' — skipping", raw_key)
            skipped_variables.append(raw_key)
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
                    logger.warning(
                        "No fraction columns matching prefix '%s_' found in %s "
                        "(available: %s) — skipping categorical variable",
                        prefix,
                        raw_path.name,
                        list(raw_df.columns),
                    )
                    skipped_variables.append(entry.canonical_name)
                    continue
                out_path = output_dir / f"{entry.canonical_name}.csv"
                out_df.to_csv(out_path)
                sir_files[entry.canonical_name] = out_path
                logger.info("SIR normalized: %s → %s", raw_key, out_path.name)
            else:
                # Continuous: find the matching column and rename
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
                        logger.debug(
                            "Column '%s' not found in %s, using fallback '%s'",
                            source_col,
                            raw_path.name,
                            alt_col,
                        )
                        source_col = alt_col
                    else:
                        logger.warning(
                            "Column '%s' not found in %s (available: %s) — skipping",
                            source_col,
                            raw_path.name,
                            list(raw_df.columns),
                        )
                        skipped_variables.append(cname)
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

    if skipped_variables:
        logger.warning(
            "SIR normalization skipped %d variable(s): %s",
            len(skipped_variables),
            skipped_variables,
        )

    return sir_files


@dataclass
class SIRValidationWarning:
    """A single SIR validation warning."""

    variable: str
    check_type: str
    message: str


class SIRValidationError(Exception):
    """Raised when SIR validation fails in strict mode."""

    def __init__(self, warnings: list[SIRValidationWarning]) -> None:
        self.warnings = warnings
        messages = [f"  [{w.check_type}] {w.variable}: {w.message}" for w in warnings]
        super().__init__(
            f"SIR validation failed with {len(warnings)} warnings:\n" + "\n".join(messages)
        )


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
    import pandas as pd

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
