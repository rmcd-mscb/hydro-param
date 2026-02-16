"""Output formatter protocol and factory.

Defines the ``OutputFormatter`` protocol for model-specific output
formatters and a factory function for selecting formatters by name.
Follows the same Protocol + factory pattern as ``processing.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import xarray as xr

logger = logging.getLogger(__name__)


class OutputFormatter(Protocol):
    """Protocol for model-specific output formatters.

    Implementations convert an ``xr.Dataset`` of derived model parameters
    into the file format(s) expected by the target model.
    """

    name: str

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

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """Validate parameters before writing.

        Returns
        -------
        list[str]
            List of validation warnings/errors. Empty if valid.
        """
        ...


class NetCDFFormatter:
    """Simple NetCDF output (wraps existing SIR → NetCDF logic)."""

    name: str = "netcdf"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.nc"
        parameters.to_netcdf(out_file)
        logger.info("Wrote NetCDF: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        return []


class ParquetFormatter:
    """Simple Parquet output (wraps existing SIR → Parquet logic)."""

    name: str = "parquet"

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        output_path.mkdir(parents=True, exist_ok=True)
        sir_name = config.get("sir_name", "result")
        out_file = output_path / f"{sir_name}.parquet"
        parameters.to_dataframe().to_parquet(out_file)
        logger.info("Wrote Parquet: %s", out_file)
        return [out_file]

    def validate(self, parameters: xr.Dataset) -> list[str]:
        return []


def get_formatter(name: str) -> OutputFormatter:
    """Select an output formatter by name.

    Parameters
    ----------
    name
        Formatter name: ``"netcdf"``, ``"parquet"``, or ``"pywatershed"``.

    Returns
    -------
    OutputFormatter

    Raises
    ------
    ValueError
        If the formatter name is not recognized.
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
