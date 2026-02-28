"""Lazy accessor for SIR files backed by a pipeline manifest.

Provide per-variable on-demand loading from the SIR output directory.
No data is held in memory between calls -- each ``load_variable()`` reads
from disk and the caller is responsible for releasing.

Support a Dataset-compatible API (``__contains__``, ``__getitem__``,
``data_vars``) so derivation steps can check variable availability and
load data with minimal code changes.

When the manifest is missing or corrupt, fall back to discovering SIR
files by globbing the ``sir/`` subdirectory with a warning.

See Also
--------
hydro_param.manifest : Manifest schema that tracks SIR output.
hydro_param.plugins.DerivationContext : Consumer of SIRAccessor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from hydro_param.manifest import load_manifest

logger = logging.getLogger(__name__)


class SIRAccessor:
    """Lazy accessor for SIR files backed by a pipeline manifest.

    Do not hold data in memory.  Each call to ``load_variable()``
    reads from disk; the caller releases when done.

    Parameters
    ----------
    output_dir : Path
        Pipeline output directory containing ``.manifest.yml`` and
        ``sir/`` subdirectory.

    Raises
    ------
    FileNotFoundError
        If a file referenced in the manifest does not exist on disk.

    Notes
    -----
    Implement ``__contains__`` and ``__getitem__`` for compatibility
    with derivation steps that check ``"var_name" in sir`` and access
    ``sir["var_name"]``.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._sir_schema: list[dict] = []

        manifest = load_manifest(self._output_dir)
        if manifest is not None and manifest.sir is not None:
            self._static = dict(manifest.sir.static_files)
            self._temporal = dict(manifest.sir.temporal_files)
            self._sir_schema = list(manifest.sir.sir_schema)
        else:
            logger.warning(
                "No valid manifest with SIR section at %s — discovering SIR "
                "files by scanning sir/. Schema metadata will not be "
                "available. Consider re-running 'hydro-param run pipeline.yml' "
                "to regenerate the manifest.",
                self._output_dir,
            )
            self._static = _glob_sir_static(self._output_dir / "sir")
            self._temporal = _glob_sir_temporal(self._output_dir / "sir")

        self._validate_files()

    def _validate_files(self) -> None:
        """Verify all referenced files exist on disk.

        Raises
        ------
        FileNotFoundError
            If any referenced file is missing.
        """
        for name, rel_path in self._static.items():
            full = self._output_dir / rel_path
            if not full.exists():
                raise FileNotFoundError(
                    f"SIR static file for '{name}' not found at {full}. "
                    f"Re-run 'hydro-param run pipeline.yml' to regenerate."
                )
        for name, rel_path in self._temporal.items():
            full = self._output_dir / rel_path
            if not full.exists():
                raise FileNotFoundError(
                    f"SIR temporal file for '{name}' not found at {full}. "
                    f"Re-run 'hydro-param run pipeline.yml' to regenerate."
                )

    def available_variables(self) -> list[str]:
        """List all static SIR variable names.

        Returns
        -------
        list[str]
            Variable names available for ``load_variable()``.
        """
        return list(self._static.keys())

    def available_temporal(self) -> list[str]:
        """List all temporal SIR dataset keys.

        Returns
        -------
        list[str]
            Dataset keys available for ``load_temporal()``.
        """
        return list(self._temporal.keys())

    @property
    def data_vars(self) -> list[str]:
        """Return static variable names (Dataset-compatible API).

        Returns
        -------
        list[str]
            Same as ``available_variables()``.
        """
        return self.available_variables()

    @property
    def sir_schema(self) -> list[dict]:
        """Return SIR variable schema metadata.

        Returns
        -------
        list[dict]
            Schema entries from the manifest, or empty list if
            discovered via glob fallback.
        """
        return self._sir_schema

    def load_variable(self, name: str) -> xr.DataArray:
        """Load a single SIR variable from disk.

        Parameters
        ----------
        name : str
            SIR variable name (e.g., ``"elevation_m_mean"``).

        Returns
        -------
        xr.DataArray
            Variable data with the fabric id field as dimension.

        Raises
        ------
        KeyError
            If the variable name is not in the SIR.
        """
        if name not in self._static:
            raise KeyError(
                f"SIR variable '{name}' not found. "
                f"Available: {sorted(self._static.keys())}"
            )
        path = self._output_dir / self._static[name]
        df = pd.read_csv(path, index_col=0)
        ds = xr.Dataset.from_dataframe(df)
        if name in ds:
            return ds[name]
        # Single-column CSV: return the first (only) variable
        return next(iter(ds.data_vars.values()))

    def load_temporal(self, name: str) -> xr.Dataset:
        """Load a single temporal SIR file from disk.

        Parameters
        ----------
        name : str
            Temporal dataset key (e.g., ``"gridmet_2020"``).

        Returns
        -------
        xr.Dataset
            Temporal dataset.  Caller should close when done.

        Raises
        ------
        KeyError
            If the temporal key is not in the SIR.
        """
        if name not in self._temporal:
            raise KeyError(
                f"SIR temporal dataset '{name}' not found. "
                f"Available: {sorted(self._temporal.keys())}"
            )
        path = self._output_dir / self._temporal[name]
        return xr.open_dataset(path)

    def __contains__(self, name: object) -> bool:
        """Check if a variable name is available (Dataset-compatible API)."""
        return isinstance(name, str) and name in self._static

    def __getitem__(self, name: str) -> xr.DataArray:
        """Load a variable by name (Dataset-compatible API).

        Equivalent to ``load_variable(name)``.
        """
        return self.load_variable(name)


def _glob_sir_static(sir_dir: Path) -> dict[str, str]:
    """Discover static SIR files by globbing CSV files.

    Parameters
    ----------
    sir_dir : Path
        The ``sir/`` subdirectory to scan.

    Returns
    -------
    dict[str, str]
        Mapping of variable names (stem) to paths relative to the
        parent output directory.
    """
    if not sir_dir.is_dir():
        return {}
    return {p.stem: str(p.relative_to(sir_dir.parent)) for p in sorted(sir_dir.glob("*.csv"))}


def _glob_sir_temporal(sir_dir: Path) -> dict[str, str]:
    """Discover temporal SIR files by globbing NetCDF files.

    Parameters
    ----------
    sir_dir : Path
        The ``sir/`` subdirectory to scan.

    Returns
    -------
    dict[str, str]
        Mapping of dataset keys (stem) to paths relative to the
        parent output directory.
    """
    if not sir_dir.is_dir():
        return {}
    return {p.stem: str(p.relative_to(sir_dir.parent)) for p in sorted(sir_dir.glob("*.nc"))}
