"""Plugin protocols, context, and registries.

Defines the contracts that all model plugins (derivation + formatter) must
satisfy, the typed ``DerivationContext`` input bundle, and factory functions
for plugin discovery.

This module is the single source of truth for "what is a plugin?"
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
    """Everything a derivation plugin needs.

    Bundles SIR output, target fabric geometry, and configuration into
    a single immutable object.  Validates that ``fabric_id_field`` exists
    as a dimension in the SIR on construction.

    Parameters
    ----------
    sir
        Normalized Standardized Internal Representation (SIR) dataset.
    fabric
        Target HRU polygon GeoDataFrame.
    segments
        Stream segment line GeoDataFrame.
    fabric_id_field
        Column name for HRU identifiers in the fabric.  Must exist as a
        dimension in ``sir``.
    segment_id_field
        Column name for segment identifiers in the segments GeoDataFrame.
    config
        Plugin-specific configuration dict.
    lookup_tables_dir
        Override path to lookup table YAML files.  When ``None``, defaults
        to the package-bundled tables via ``importlib.resources``.
    """

    sir: xr.Dataset
    fabric: gpd.GeoDataFrame | None = None
    segments: gpd.GeoDataFrame | None = None
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

    @property
    def resolved_lookup_tables_dir(self) -> Path:
        """Resolve the lookup tables directory.

        Returns the explicit override if set, otherwise the
        package-bundled default via ``importlib.resources``.
        """
        if self.lookup_tables_dir is not None:
            return self.lookup_tables_dir
        return Path(str(files("hydro_param").joinpath("data/lookup_tables")))


@runtime_checkable
class DerivationPlugin(Protocol):
    """Contract for model-specific parameter derivation.

    Implementations transform a normalized SIR dataset into
    model-specific parameters (e.g., PRMS units, variable names,
    lookup-table reclassification).
    """

    name: str

    def derive(self, context: DerivationContext) -> xr.Dataset:
        """Derive model parameters from the SIR.

        Parameters
        ----------
        context
            Typed input bundle containing SIR, fabric, config, etc.

        Returns
        -------
        xr.Dataset
            Model-specific parameter dataset.
        """
        ...


@runtime_checkable
class FormatterPlugin(Protocol):
    """Contract for model-specific output formatting.

    Implementations write derived parameters to the file format(s)
    expected by the target model.
    """

    name: str

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """Validate parameters before writing.

        Returns
        -------
        list[str]
            Validation warnings.  Empty if all checks pass.
        """
        ...

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write model-specific output files.

        Parameters
        ----------
        parameters
            Derived model parameters.
        output_path
            Output directory.
        config
            Formatter-specific configuration options.

        Returns
        -------
        list[Path]
            Paths to all files written.
        """
        ...


class NetCDFFormatter:
    """Simple NetCDF output (wraps SIR -> NetCDF)."""

    name: str = "netcdf"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write parameters as a single NetCDF file."""
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.nc"
        parameters.to_netcdf(out_file)
        logger.info("Wrote NetCDF: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """No-op validation for generic NetCDF output."""
        return []


class ParquetFormatter:
    """Simple Parquet output (wraps SIR -> Parquet)."""

    name: str = "parquet"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write parameters as a single Parquet file."""
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.parquet"
        parameters.to_dataframe().to_parquet(out_file)
        logger.info("Wrote Parquet: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """No-op validation for generic Parquet output."""
        return []


def get_derivation(name: str) -> DerivationPlugin:
    """Select a derivation plugin by name.

    Parameters
    ----------
    name
        Plugin name: ``"pywatershed"``.

    Raises
    ------
    ValueError
        If the name is not recognized.
    """
    if name == "pywatershed":
        from hydro_param.derivations.pywatershed import PywatershedDerivation

        return PywatershedDerivation()

    available = "pywatershed"
    raise ValueError(f"Unknown derivation plugin '{name}'. Available: {available}")


def get_formatter(name: str) -> FormatterPlugin:
    """Select an output formatter by name.

    Parameters
    ----------
    name
        Formatter name: ``"netcdf"``, ``"parquet"``, or ``"pywatershed"``.

    Raises
    ------
    ValueError
        If the name is not recognized.
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
