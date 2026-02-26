"""Plugin protocols, context, and registries.

Define the contracts that all model plugins (derivation + formatter) must
satisfy, the typed ``DerivationContext`` input bundle, and factory functions
for plugin discovery.

This module enforces the two-phase separation between the generic pipeline
and model-specific logic.  The pipeline produces a normalized Standardized
Internal Representation (SIR); plugins consume SIR data and transform it
into model-specific parameters and output files.  This module is the single
source of truth for "what is a plugin?" and how plugins are discovered.

See Also
--------
hydro_param.derivations.pywatershed : pywatershed derivation plugin.
hydro_param.formatters.pywatershed : pywatershed output formatter plugin.

Notes
-----
Plugin discovery uses lazy imports so that heavy model-specific dependencies
are only loaded when a plugin is actually requested.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Protocol, runtime_checkable

import geopandas as gpd
import xarray as xr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DerivationContext:
    """Bundle all inputs a derivation plugin needs into a single immutable object.

    ``DerivationContext`` is the sole interface between the generic pipeline and
    model-specific derivation logic.  It packages normalized SIR output, target
    fabric geometry, segment topology, and plugin configuration so that
    derivation plugins never reach back into the pipeline internals.

    Validates on construction that ``fabric_id_field`` exists as a dimension in
    the SIR and as a column in the fabric GeoDataFrame (if provided).  This
    fail-fast validation prevents silent dimension mismatches downstream.

    Attributes
    ----------
    sir : xr.Dataset
        Normalized Standardized Internal Representation (SIR) dataset produced
        by stage 5 of the pipeline.  Variables use canonical names and SI-like
        units (metres, degrees Celsius, etc.).
    temporal : dict[str, xr.Dataset] or None
        SIR-normalized temporal datasets keyed by name (e.g., ``"gridmet_2020"``).
        Each dataset contains time-indexed climate variables.  When ``None``,
        step 7 (forcing generation) is skipped.
    fabric : geopandas.GeoDataFrame or None
        Target HRU polygon GeoDataFrame with a geometry column and an ID column
        named by ``fabric_id_field``.
    segments : geopandas.GeoDataFrame or None
        Stream segment line GeoDataFrame for routing derivations (step 12).
    waterbodies : geopandas.GeoDataFrame or None
        NHDPlus waterbody polygon GeoDataFrame for depression storage
        parameters.  When ``None``, step 6 (waterbody overlay) is skipped.
    fabric_id_field : str
        Column name for HRU identifiers in ``fabric``.  Must also exist as a
        dimension in ``sir``.  Defaults to ``"nhm_id"`` for pywatershed.
    segment_id_field : str or None
        Column name for segment identifiers in ``segments``.  When ``None``,
        the derivation plugin is responsible for determining the correct field.
    config : dict
        Plugin-specific configuration dict passed through from the pipeline
        YAML.
    lookup_tables_dir : pathlib.Path or None
        Override path to lookup table YAML files.  When ``None``, defaults
        to the package-bundled tables under ``hydro_param/data/lookup_tables/``
        via ``importlib.resources``.

    Raises
    ------
    KeyError
        If ``fabric_id_field`` is not a dimension in ``sir``, or not a column
        in ``fabric`` (when ``fabric`` is provided).

    See Also
    --------
    DerivationPlugin : Protocol that consumes this context.

    Notes
    -----
    Frozen dataclass -- all fields are immutable after construction.  This
    prevents derivation plugins from accidentally mutating shared state.
    """

    sir: xr.Dataset
    temporal: dict[str, xr.Dataset] | None = None
    fabric: gpd.GeoDataFrame | None = None
    segments: gpd.GeoDataFrame | None = None
    waterbodies: gpd.GeoDataFrame | None = None
    fabric_id_field: str = "nhm_id"
    segment_id_field: str | None = None
    config: dict = field(default_factory=dict)
    lookup_tables_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.fabric_id_field not in self.sir.dims:
            raise KeyError(
                f"Expected dimension '{self.fabric_id_field}' not found in SIR. "
                f"Available dims: {list(self.sir.dims)}"
            )
        if self.fabric is not None and self.fabric_id_field not in self.fabric.columns:
            raise KeyError(
                f"fabric_id_field '{self.fabric_id_field}' not found in fabric columns. "
                f"Available columns: {sorted(self.fabric.columns.tolist())}"
            )

    @property
    def resolved_lookup_tables_dir(self) -> Path:
        """Resolve the lookup tables directory to an absolute path.

        Return the explicit override if set, otherwise the package-bundled
        default under ``hydro_param/data/lookup_tables/`` discovered via
        ``importlib.resources``.

        Returns
        -------
        pathlib.Path
            Absolute path to a directory containing lookup table YAML files.

        Raises
        ------
        FileNotFoundError
            If ``lookup_tables_dir`` was explicitly set but does not exist
            on disk.
        """
        if self.lookup_tables_dir is not None:
            if not self.lookup_tables_dir.is_dir():
                raise FileNotFoundError(
                    f"Lookup tables directory does not exist: '{self.lookup_tables_dir}'"
                )
            return self.lookup_tables_dir
        return Path(str(files("hydro_param").joinpath("data/lookup_tables")))


@runtime_checkable
class DerivationPlugin(Protocol):
    """Define the contract for model-specific parameter derivation.

    Implementations transform a normalized SIR dataset into model-specific
    parameters.  This includes unit conversions (e.g., metres to feet, Celsius
    to Fahrenheit), variable renaming, lookup-table reclassification, majority
    extraction from categorical fractions, gap-filling, and derived math.

    All model-specific logic lives in derivation plugins -- the generic
    pipeline never performs these transforms.

    Attributes
    ----------
    name : str
        Unique plugin identifier used by ``get_derivation()`` for discovery.

    See Also
    --------
    DerivationContext : Immutable input bundle consumed by ``derive()``.
    FormatterPlugin : Companion protocol for output formatting.
    """

    name: str

    def derive(self, context: DerivationContext) -> xr.Dataset:
        """Derive model-specific parameters from the normalized SIR.

        Execute the full derivation pipeline for a target model, including
        unit conversions, reclassification, lookup-table application, and
        derived math.

        Parameters
        ----------
        context : DerivationContext
            Immutable input bundle containing the SIR dataset, target fabric
            geometry, segment topology, and plugin configuration.

        Returns
        -------
        xr.Dataset
            Model-specific parameter dataset with variables in model-native
            names and units (e.g., feet, acres, degrees Fahrenheit for PRMS).
        """
        ...


@runtime_checkable
class FormatterPlugin(Protocol):
    """Define the contract for model-specific output formatting.

    Implementations serialize derived parameters to the file format(s)
    expected by the target model (e.g., PRMS parameter files, NextGen
    configuration, or generic NetCDF/Parquet).

    Attributes
    ----------
    name : str
        Unique plugin identifier used by ``get_formatter()`` for discovery.

    See Also
    --------
    DerivationPlugin : Companion protocol for parameter derivation.
    NetCDFFormatter : Generic NetCDF implementation.
    ParquetFormatter : Generic Parquet implementation.
    """

    name: str

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """Validate derived parameters before writing output files.

        Check that required variables are present, values are within
        physically plausible ranges, and units are consistent.

        Parameters
        ----------
        parameters : xr.Dataset
            Derived model parameters to validate.

        Returns
        -------
        list[str]
            Validation warning messages.  Empty list if all checks pass.
        """
        ...

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write derived parameters to model-specific output files.

        Create the output directory if it does not exist, then serialize
        the parameter dataset into one or more files in the format expected
        by the target model.

        Parameters
        ----------
        parameters : xr.Dataset
            Derived model parameters (output of ``DerivationPlugin.derive``).
        output_path : pathlib.Path
            Directory to write output files into.  Created if absent.
        config : dict
            Formatter-specific configuration options (e.g., ``sir_name``
            for file naming).

        Returns
        -------
        list[pathlib.Path]
            Absolute paths to all files written.

        Raises
        ------
        OSError
            If file I/O fails (e.g., permission denied, disk full).
        """
        ...


class NetCDFFormatter:
    """Format parameters as a single NetCDF-4 file.

    Generic formatter that writes the full parameter dataset to one NetCDF
    file without any model-specific transformations.  Suitable for archival,
    inspection, or consumption by downstream tools that read CF-compliant
    NetCDF.

    Attributes
    ----------
    name : str
        Formatter identifier (``"netcdf"``).

    See Also
    --------
    ParquetFormatter : Alternative tabular output format.
    """

    name: str = "netcdf"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write parameters as a single NetCDF-4 file.

        Parameters
        ----------
        parameters : xr.Dataset
            Parameter dataset to serialize.
        output_path : pathlib.Path
            Directory to write into (created if absent).
        config : dict
            Options.  Recognized keys:

            - ``sir_name`` (str): Base filename (default ``"result"``).
              Output file is ``<sir_name>.nc``.

        Returns
        -------
        list[pathlib.Path]
            Single-element list containing the path to the written file.

        Raises
        ------
        OSError
            If the NetCDF write fails (wrapped with the target path for
            easier debugging).
        """
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.nc"
        try:
            parameters.to_netcdf(out_file)
        except OSError as exc:
            raise OSError(f"NetCDF write failed for '{out_file}': {exc}") from exc
        logger.info("Wrote NetCDF: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """Perform no-op validation (generic NetCDF has no schema constraints).

        Parameters
        ----------
        parameters : xr.Dataset
            Parameter dataset (unused).

        Returns
        -------
        list[str]
            Always returns an empty list.
        """
        return []


class ParquetFormatter:
    """Format parameters as a single Apache Parquet file.

    Generic formatter that converts the parameter dataset to a pandas
    DataFrame and writes it as a single Parquet file.  Parquet provides
    efficient columnar storage with compression, suitable for downstream
    analysis in pandas, Spark, or DuckDB.

    Attributes
    ----------
    name : str
        Formatter identifier (``"parquet"``).

    See Also
    --------
    NetCDFFormatter : Alternative multidimensional output format.
    """

    name: str = "parquet"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write parameters as a single Parquet file.

        Convert the xarray Dataset to a pandas DataFrame via
        ``to_dataframe()`` before serializing.  Index columns are
        preserved.

        Parameters
        ----------
        parameters : xr.Dataset
            Parameter dataset to serialize.
        output_path : pathlib.Path
            Directory to write into (created if absent).
        config : dict
            Options.  Recognized keys:

            - ``sir_name`` (str): Base filename (default ``"result"``).
              Output file is ``<sir_name>.parquet``.

        Returns
        -------
        list[pathlib.Path]
            Single-element list containing the path to the written file.

        Raises
        ------
        OSError
            If the Parquet write fails (wrapped with the target path for
            easier debugging).
        """
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.parquet"
        try:
            parameters.to_dataframe().to_parquet(out_file)
        except OSError as exc:
            raise OSError(f"Parquet write failed for '{out_file}': {exc}") from exc
        logger.info("Wrote Parquet: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """Perform no-op validation (generic Parquet has no schema constraints).

        Parameters
        ----------
        parameters : xr.Dataset
            Parameter dataset (unused).

        Returns
        -------
        list[str]
            Always returns an empty list.
        """
        return []


def get_derivation(name: str) -> DerivationPlugin:
    """Look up and instantiate a derivation plugin by name.

    Factory function that lazily imports the requested derivation plugin
    module.  This avoids loading heavy model-specific dependencies until
    they are actually needed.

    Parameters
    ----------
    name : str
        Plugin name.  Currently supported: ``"pywatershed"``.

    Returns
    -------
    DerivationPlugin
        A freshly instantiated derivation plugin.

    Raises
    ------
    ValueError
        If ``name`` does not match any registered plugin.

    See Also
    --------
    get_formatter : Companion factory for output formatters.
    """
    if name == "pywatershed":
        from hydro_param.derivations.pywatershed import PywatershedDerivation

        return PywatershedDerivation()

    available = "pywatershed"
    raise ValueError(f"Unknown derivation plugin '{name}'. Available: {available}")


def get_formatter(name: str) -> FormatterPlugin:
    """Look up and instantiate an output formatter by name.

    Factory function that returns a formatter matching the requested output
    format.  Model-specific formatters (e.g., ``"pywatershed"``) are lazily
    imported to avoid loading unused dependencies.

    Parameters
    ----------
    name : str
        Formatter name.  Currently supported: ``"netcdf"``, ``"parquet"``,
        ``"pywatershed"``.

    Returns
    -------
    FormatterPlugin
        A freshly instantiated formatter plugin.

    Raises
    ------
    ValueError
        If ``name`` does not match any registered formatter.

    See Also
    --------
    get_derivation : Companion factory for derivation plugins.
    """
    if name == "netcdf":
        return NetCDFFormatter()
    if name == "parquet":
        return ParquetFormatter()
    if name == "pywatershed":
        from hydro_param.formatters.pywatershed import PywatershedFormatter

        return PywatershedFormatter()

    available = "netcdf, parquet, pywatershed"
    raise ValueError(f"Unknown output formatter '{name}'. Available: {available}")
