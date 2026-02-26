"""SIR normalization: canonical naming, unit conversion, and schema validation.

Normalize raw gdptools output into the Standardized Internal Representation
(SIR) -- self-documenting variable names with canonical SI-like units.  The
SIR is the boundary between the generic pipeline (stages 1-5) and model
plugins: plugins consume only SIR data and never see raw gdptools output.

This module handles three concerns:

1. **Naming**: Convert raw source variable names (e.g., ``"elevation"``) into
   canonical SIR names (e.g., ``"elevation_m_mean"``).
2. **Unit conversion**: Transform source units to canonical units at the SIR
   boundary (e.g., log10(cm/hr) to cm/hr, Kelvin to degrees Celsius).
3. **Validation**: Check completeness, NaN coverage, and value ranges.

References
----------
.. [1] docs/plans/2026-02-23-sir-normalization-design.md -- Full design document.

See Also
--------
hydro_param.units : Model-specific unit conversions (SI to imperial).
hydro_param.plugins : Plugin protocols that consume SIR output.
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

    Map a source unit string to a short, filesystem-safe abbreviation for
    use in SIR variable names.  Known units are looked up in ``_UNIT_TABLE``;
    unknown units are slugified by replacing non-alphanumeric characters with
    underscores.

    Parameters
    ----------
    units : str
        Source unit string from the dataset registry (e.g., ``"m"``,
        ``"log10(cm/hr)"``, ``"%"``, ``"W/m2"``).

    Returns
    -------
    str
        Short canonical abbreviation (e.g., ``"m"``, ``"cm_hr"``,
        ``"pct"``, ``"W_m2"``).  Empty string for dimensionless
        quantities (``units=""``).

    Examples
    --------
    >>> unit_abbreviation("log10(cm/hr)")
    'cm_hr'
    >>> unit_abbreviation("%")
    'pct'
    >>> unit_abbreviation("")
    ''
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

    Assemble a self-documenting variable name following the pattern
    ``<base>_<unit_abbrev>_<stat>``.  Dimensionless quantities omit the
    unit segment: ``<base>_<stat>``.  All base names are lowercased.

    Parameters
    ----------
    name : str
        Base variable name from the dataset registry (e.g., ``"elevation"``,
        ``"land_cover"``).
    units : str
        Source unit string (e.g., ``"m"``, ``"log10(cm/hr)"``, ``""``).
    stat : str
        Aggregation statistic (e.g., ``"mean"``, ``"min"``, ``"majority"``,
        ``"frac"``).

    Returns
    -------
    str
        Canonical SIR variable name (e.g., ``"elevation_m_mean"``,
        ``"land_cover_frac"``).

    Examples
    --------
    >>> canonical_name("elevation", "m", "mean")
    'elevation_m_mean'
    >>> canonical_name("land_cover", "", "frac")
    'land_cover_frac'
    """
    base = name.lower()
    abbrev = unit_abbreviation(units)
    if abbrev:
        return f"{base}_{abbrev}_{stat}"
    return f"{base}_{stat}"


def _unit_info(units: str) -> tuple[str, str, str | None]:
    """Look up abbreviation, canonical units, and conversion type for a unit string.

    Parameters
    ----------
    units : str
        Source unit string from the dataset registry.

    Returns
    -------
    tuple[str, str, str or None]
        ``(abbreviation, canonical_units, conversion)`` where
        ``conversion`` is ``None`` for passthrough, ``"log10_to_linear"``
        for log-transformed units, or ``"K_to_C"`` for Kelvin.
    """
    if units == "":
        return ("", "", None)
    if units in _UNIT_TABLE:
        return _UNIT_TABLE[units]
    slug = re.sub(r"[^a-zA-Z0-9]", "_", units)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return (slug, units, None)


@dataclass
class SIRVariableSchema:
    """Describe the expected schema for a single SIR output variable.

    Each schema entry records the mapping from a raw source variable to its
    canonical SIR name, the unit conversion to apply, and validation
    constraints.  Used by ``normalize_sir()`` and ``validate_sir()`` to
    transform and check pipeline output.

    Attributes
    ----------
    canonical_name : str
        Canonical SIR variable name (e.g., ``"elevation_m_mean"``).
    source_name : str
        Original variable name from the dataset registry.
    source_units : str
        Units of the raw source data (e.g., ``"log10(cm/hr)"``).
    canonical_units : str
        Units after SIR normalization (e.g., ``"cm/hr"``).
    long_name : str
        Human-readable description for metadata / NetCDF attributes.
    categorical : bool
        ``True`` for land-cover or other categorical variables that
        produce fraction columns rather than continuous statistics.
    valid_range : tuple[float, float] or None
        Expected ``(min, max)`` value range for validation.  ``None``
        to skip range checks.  Categorical fractions use ``(0.0, 1.0)``.
    conversion : str or None
        Conversion type to apply: ``None`` (passthrough),
        ``"log10_to_linear"`` (10^x), or ``"K_to_C"`` (Kelvin to Celsius).
    temporal : bool
        ``True`` for time-indexed variables (e.g., gridMET climate data).
    """

    canonical_name: str
    source_name: str
    source_units: str
    canonical_units: str
    long_name: str
    categorical: bool
    valid_range: tuple[float, float] | None
    conversion: str | None
    temporal: bool = False


def build_sir_schema(
    resolved: Sequence[tuple[object, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
) -> list[SIRVariableSchema]:
    """Auto-generate the SIR schema from stage 2 resolved datasets.

    Walk the resolved dataset/variable tuples produced by
    ``stage2_resolve_datasets()`` and create one ``SIRVariableSchema`` entry
    per expected output column.  For continuous variables, one entry is
    created per ``(variable, statistic, year)`` combination.  For categorical
    variables, a single ``_frac`` entry is created per ``(variable, year)``
    -- individual fraction columns (e.g., ``land_cover_frac_42``) are
    generated dynamically during normalization.

    Parameters
    ----------
    resolved : Sequence[tuple[object, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]]
        Output of ``stage2_resolve_datasets()``.  Each tuple contains:

        - ``DatasetEntry`` -- dataset metadata from the registry.
        - ``DatasetRequest`` -- user-specified request (statistics, year).
        - ``list[VariableSpec | DerivedVariableSpec]`` -- resolved variables.

    Returns
    -------
    list[SIRVariableSchema]
        One schema entry per expected SIR output column.  Multi-year
        datasets produce year-suffixed entries (e.g.,
        ``"elevation_m_mean_2020"``).

    Notes
    -----
    The schema is deterministic for a given set of resolved datasets: same
    input always produces the same schema entries in the same order.
    """
    schema: list[SIRVariableSchema] = []
    for entry, ds_req, var_specs in resolved:
        is_temporal = hasattr(entry, "temporal") and entry.temporal

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
                            temporal=is_temporal,
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
                                temporal=is_temporal,
                            )
                        )
    return schema


def apply_conversion(
    values: NDArray[np.floating],
    conversion: str | None,
) -> NDArray[np.floating]:
    """Apply a SIR unit conversion to an array of values.

    Transform raw source values into canonical SIR units.  This handles
    conversions that occur at the pipeline/SIR boundary -- distinct from
    the model-specific conversions in ``hydro_param.units`` which occur
    inside derivation plugins.

    Parameters
    ----------
    values : NDArray[np.floating]
        Input values in source units.
    conversion : str or None
        Conversion type:

        - ``None`` -- passthrough, return input unchanged.
        - ``"log10_to_linear"`` -- apply ``10^x`` (e.g., POLARIS
          log10(cm/hr) to cm/hr).
        - ``"K_to_C"`` -- subtract 273.15 (Kelvin to degrees Celsius).

    Returns
    -------
    NDArray[np.floating]
        Converted values in canonical SIR units.

    Raises
    ------
    ValueError
        If ``conversion`` is not a recognized conversion type.

    See Also
    --------
    hydro_param.units.convert : Model-specific unit conversions (e.g.,
        metres to feet).
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
    """Normalize raw per-variable CSV files to canonical SIR format.

    Read raw CSVs produced by stage 4 (gdptools zonal statistics), rename
    columns to canonical SIR names, apply unit conversions (e.g.,
    log10-to-linear, Kelvin-to-Celsius), and write normalized per-variable
    CSVs to ``output_dir/``.

    Categorical variables (e.g., NLCD land cover) have their fraction
    columns renamed from ``<source>_<class>`` to ``<source>_frac_<class>``.
    Continuous variables are matched by statistic suffix and renamed to the
    canonical ``<base>_<unit>_<stat>`` pattern.

    Parameters
    ----------
    raw_files : dict[str, pathlib.Path]
        Mapping of source variable key (e.g., ``"elevation"``,
        ``"elevation_2020"``) to raw CSV file path produced by stage 4.
    schema : list[SIRVariableSchema]
        SIR variable schema entries from ``build_sir_schema()``.
    output_dir : pathlib.Path
        Directory to write normalized CSV files.  Created if absent.
    id_field : str
        Feature ID column name from ``target_fabric.id_field`` config
        (e.g., ``"nhm_id"``).  Used as the CSV index column.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of canonical SIR name to normalized CSV file path.
        Variables that could not be matched or were missing columns are
        omitted and logged as warnings.

    Warnings
    --------
    Logs a warning for each raw variable that has no matching schema entry
    or whose expected column is missing from the raw CSV.  A summary
    warning is logged at the end if any variables were skipped.

    Notes
    -----
    Year-suffixed keys (e.g., ``"elevation_2020"``) are matched to their
    base schema entry by stripping the ``_YYYY`` suffix.  This supports
    multi-year static datasets where each year produces a separate raw file.
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


def normalize_sir_temporal(
    temporal_files: dict[str, Path],
    schema: list[SIRVariableSchema],
    resolved: Sequence[tuple[object, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]],
    output_dir: Path,
) -> dict[str, Path]:
    """Normalize temporal NetCDF files to canonical SIR format.

    Read raw temporal NetCDFs from stage 4 (gdptools WeightGen/AggGen
    output), rename data variables from native source names (OPeNDAP/CF
    variable names like ``"daily_mean_shortwave_radiation_at_surface"``)
    to canonical SIR names, apply unit conversions, and write normalized
    per-variable NetCDFs.

    A reverse lookup table maps native variable names to their corresponding
    ``VariableSpec`` and ``SIRVariableSchema`` entries, using the
    ``native_name`` field from the dataset registry.

    Parameters
    ----------
    temporal_files : dict[str, pathlib.Path]
        Mapping of dataset key (e.g., ``"gridmet_2020"``) to raw NetCDF
        file path produced by stage 4.
    schema : list[SIRVariableSchema]
        SIR variable schema entries from ``build_sir_schema()``.
    resolved : Sequence[tuple[object, DatasetRequest, list[VariableSpec | DerivedVariableSpec]]]
        Resolved dataset entries from stage 2.  Used to build the native
        name reverse lookup.
    output_dir : pathlib.Path
        Directory to write normalized NetCDF files.  Created if absent.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping of canonical name (with year suffix if applicable, e.g.,
        ``"pr_mm_mean_2020"``) to normalized NetCDF file path.

    Warnings
    --------
    Logs a warning if a native variable name appears in multiple
    ``VariableSpec`` entries (collision), or if a data variable in the
    raw NetCDF has no matching schema entry.

    Notes
    -----
    Year suffixes are extracted from the ``temporal_files`` keys (e.g.,
    ``"gridmet_2020"`` yields ``"_2020"``) and appended to canonical
    names to prevent multi-year collisions in the output mapping.
    """
    import xarray as xr

    output_dir.mkdir(parents=True, exist_ok=True)
    sir_files: dict[str, Path] = {}

    # Build reverse lookup: native source name -> (var_spec, schema_entries)
    # Only for temporal datasets. Keys are native_name (OPeNDAP/CF variable name
    # that gdptools writes into the temporal NetCDF).
    native_name_lookup: dict[str, tuple[VariableSpec, list[SIRVariableSchema]]] = {}
    for entry_obj, _ds_req, var_specs in resolved:
        if not (hasattr(entry_obj, "temporal") and entry_obj.temporal):
            continue
        for vs in var_specs:
            if isinstance(vs, VariableSpec):
                matching = [s for s in schema if s.source_name == vs.name and s.temporal]
                if matching:
                    key = vs.native_name or vs.name
                    if key in native_name_lookup:
                        existing_vs = native_name_lookup[key][0]
                        logger.warning(
                            "Duplicate native_name_lookup key '%s': variable '%s' "
                            "collides with '%s' — check native_name settings",
                            key,
                            vs.name,
                            existing_vs.name,
                        )
                    native_name_lookup[key] = (vs, matching)

    for file_key, nc_path in temporal_files.items():
        # Extract year suffix from file_key (e.g., "gridmet_2020" → "_2020")
        year_match = re.search(r"_(\d{4})$", file_key)
        year_suffix = f"_{year_match.group(1)}" if year_match else ""

        with xr.open_dataset(nc_path) as ds:
            for data_var in list(ds.data_vars):
                lookup = native_name_lookup.get(str(data_var))
                if lookup is None:
                    logger.warning(
                        "No SIR schema match for temporal variable '%s' in %s — skipping",
                        data_var,
                        nc_path.name,
                    )
                    continue

                schema_entries = lookup[1]
                schema_entry = schema_entries[0]

                # Apply unit conversion
                values: NDArray[np.floating] = ds[data_var].values.astype(np.float64)
                if schema_entry.conversion is not None:
                    values = apply_conversion(values, schema_entry.conversion)

                cname = schema_entry.canonical_name
                # Include year suffix to avoid multi-year collisions
                out_key = f"{cname}{year_suffix}"

                out_ds = xr.Dataset(
                    {cname: (ds[data_var].dims, values)},
                    coords=ds.coords,
                )
                out_path = output_dir / f"{out_key}.nc"
                out_ds.to_netcdf(out_path)
                sir_files[out_key] = out_path
                logger.info(
                    "SIR temporal normalized: %s/%s → %s", file_key, data_var, out_path.name
                )

    return sir_files


@dataclass
class SIRValidationWarning:
    """Represent a single SIR validation warning.

    Attributes
    ----------
    variable : str
        Canonical variable name or column name that triggered the warning.
    check_type : str
        Category of the check: ``"missing"``, ``"nan_coverage"``, or
        ``"range"``.
    message : str
        Human-readable description of the issue.
    """

    variable: str
    check_type: str
    message: str


class SIRValidationError(Exception):
    """Raise when SIR validation fails in strict mode.

    Wraps one or more ``SIRValidationWarning`` instances into a single
    exception with a formatted multi-line message listing each warning.

    Attributes
    ----------
    warnings : list[SIRValidationWarning]
        All validation warnings that triggered the error.
    """

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
    """Validate normalized SIR files against the expected schema.

    Perform three categories of checks on the normalized SIR output:

    1. **Completeness** -- every schema variable has a corresponding file.
    2. **NaN coverage** -- warn if any variable is 100% NaN (likely a
       processing failure).
    3. **Value range** -- warn if values fall outside the schema's
       ``valid_range`` (e.g., fractions outside [0, 1]).

    Temporal NetCDF files (``.nc``) are checked for completeness but
    skipped for NaN/range checks (CSV-only validation).

    Parameters
    ----------
    sir_files : dict[str, pathlib.Path]
        Mapping of canonical name to normalized file path (CSV or NetCDF).
    schema : list[SIRVariableSchema]
        Expected SIR variable schema entries from ``build_sir_schema()``.
    strict : bool
        If ``True``, raise ``SIRValidationError`` when any warnings are
        produced.  Default ``False`` (tolerant mode -- log warnings and
        continue).

    Returns
    -------
    list[SIRValidationWarning]
        Validation warnings.  Empty list means all checks passed.

    Raises
    ------
    SIRValidationError
        If ``strict=True`` and any validation warnings are produced.

    Notes
    -----
    This function aligns with the project's fault-tolerance strategy:
    production runs use ``strict=False`` (warnings are logged but
    processing continues), while development/debugging uses
    ``strict=True`` to catch issues early.
    """
    import pandas as pd

    warnings: list[SIRValidationWarning] = []

    # Check completeness: schema variables present in files.
    # Temporal variables have year-suffixed keys (e.g. "pr_mm_mean_2020"),
    # so check if any key starts with the canonical name for temporal entries.
    sir_keys = set(sir_files.keys())
    for entry in schema:
        if entry.canonical_name in sir_keys:
            continue
        if entry.temporal and any(k.startswith(entry.canonical_name + "_") for k in sir_keys):
            continue
        warnings.append(
            SIRValidationWarning(
                variable=entry.canonical_name,
                check_type="missing",
                message=f"Expected variable '{entry.canonical_name}' not found in SIR output",
            )
        )

    # Check each file (skip temporal NetCDF — CSV validation only)
    for cname, path in sir_files.items():
        if path.suffix == ".nc":
            continue
        df = pd.read_csv(path, index_col=0)

        # Find matching schema entry
        matching = [s for s in schema if s.canonical_name == cname]

        for col in df.columns:
            # For categorical entries, only validate fraction columns (skip count, etc.)
            if matching and matching[0].categorical:
                is_fraction_col = "_frac_" in col and not col.endswith("_count")
                if not is_fraction_col:
                    continue

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
