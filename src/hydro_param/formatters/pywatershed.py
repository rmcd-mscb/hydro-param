"""Format derived pywatershed parameters into model-ready output files.

Convert the xarray Dataset of derived PRMS parameters into the file
formats expected by pywatershed v2.0.  This is the output formatter
plugin for the pywatershed model -- it implements the second half of
the two-phase separation (pipeline produces raw SIR, plugin formats
model-specific output).

Output components
-----------------
1. **Parameter NetCDF** -- static parameters loadable by
   ``pws.Parameters.from_netcdf()``.  CF-1.8 compliant.
2. **Forcing NetCDF files** -- one file per climate variable
   (``prcp.nc``, ``tmax.nc``, ``tmin.nc``) with PRMS internal units
   (inches/day, degrees F).  pywatershed's ``PRMSAtmosphere`` accepts
   ``Union[str, Path, ndarray, Adapter]`` for these inputs.
3. **Soltab NetCDF** -- potential solar radiation tables with shape
   ``(nhru, 366)`` in Langleys (cal/cm2/day).
4. **Control YAML** -- simulation time period configuration.

Unit conventions
~~~~~~~~~~~~~~~~
PRMS internal units: feet, inches, degrees F, acres, Langleys.
Forcing conversions performed here: mm to inches (prcp), C to F
(tmax, tmin).

See Also
--------
docs/reference/pywatershed_parameterization_guide.md : Section 2C.
hydro_param.derivations.pywatershed : Derivation plugin that produces
    the input Dataset for this formatter.
hydro_param.units.convert : Unit conversion utility.
"""

from __future__ import annotations

import logging
from importlib.resources import files
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from hydro_param.units import convert

logger = logging.getLogger(__name__)

# Forcing variables and their required unit conversions
_FORCING_VARS: dict[str, tuple[str, str]] = {
    "prcp": ("mm", "in"),
    "tmax": ("C", "F"),
    "tmin": ("C", "F"),
    "swrad": ("Langleys/day", "Langleys/day"),
    "potet": ("in", "in"),
}

# Soltab variable names
_SOLTAB_VARS = {"soltab_potsw", "soltab_horad_potsw"}


class PywatershedFormatter:
    """Format derived parameters for pywatershed consumption.

    Produce parameter NetCDF, forcing NetCDF files (one variable per
    file), soltab arrays, and control configuration compatible with
    pywatershed v2.0.  This class implements the ``FormatterProtocol``
    defined in ``hydro_param.plugins``.

    The formatter validates parameters against bundled metadata before
    writing, logging warnings for missing required parameters or
    out-of-range values.

    Attributes
    ----------
    name : str
        Formatter identifier (``"pywatershed"``).  Used by the plugin
        factory ``get_formatter()`` for lookup.

    Notes
    -----
    Parameter metadata (valid ranges, required flags) is loaded from
    the bundled ``data/pywatershed/parameter_metadata.yml`` file and
    cached for the lifetime of the formatter instance.
    """

    name: str = "pywatershed"

    def __init__(self) -> None:
        """Initialize the formatter with an empty metadata cache."""
        self._metadata_cache: dict | None = None

    @staticmethod
    def _default_metadata_path() -> Path:
        """Return the path to the bundled parameter metadata YAML.

        Returns
        -------
        Path
            Absolute path to ``data/pywatershed/parameter_metadata.yml``
            within the installed ``hydro_param`` package.
        """
        return Path(str(files("hydro_param").joinpath("data/pywatershed/parameter_metadata.yml")))

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write all pywatershed output files to a directory.

        Orchestrate the four output components: static parameters,
        climate forcing, soltab, and control.  Validates the dataset
        before writing and logs any validation warnings.

        Parameters
        ----------
        parameters
            Derived pywatershed parameter dataset containing static
            parameters, optional forcing time series (``prcp``,
            ``tmax``, ``tmin``), and optional soltab arrays
            (``soltab_potsw``, ``soltab_horad_potsw``).
        output_path
            Root output directory.  Created if it does not exist.
        config
            Formatter configuration dict with keys:

            - ``parameter_file`` (str): filename for static params
            - ``forcing_dir`` (str): subdirectory for forcing files
            - ``soltab_file`` (str): filename for solar tables
            - ``control_file`` (str): filename for control YAML
            - ``start`` (str): simulation start date (ISO format)
            - ``end`` (str): simulation end date (ISO format)

        Returns
        -------
        list[Path]
            Absolute paths to all files written.

        Raises
        ------
        OSError
            If output directory creation or file writing fails.
        ValueError
            If ``start`` or ``end`` are missing from *config*.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        # Validate before writing
        warnings = self.validate(parameters)
        for w in warnings:
            logger.warning("Validation: %s", w)

        # 1. Parameter NetCDF
        param_path = output_path / config.get("parameter_file", "parameters.nc")
        self.write_parameters(parameters, param_path)
        if param_path.exists():
            written.append(param_path)

        # 2. Forcing NetCDF files (only if climate data present)
        forcing_dir = config.get("forcing_dir") or config.get("cbh_dir") or "forcing"
        forcing_paths = self.write_forcing_netcdf(parameters, output_path / forcing_dir)
        written.extend(forcing_paths)

        # 3. Soltab (only if soltab arrays present)
        has_soltab = any(v in parameters for v in _SOLTAB_VARS)
        if has_soltab:
            soltab_path = output_path / config.get("soltab_file", "soltab.nc")
            self.write_soltab(parameters, soltab_path)
            written.append(soltab_path)

        # 4. Control file
        control_path = output_path / config.get("control_file", "control.yml")
        self.write_control(config, control_path)
        written.append(control_path)

        logger.info("Wrote %d pywatershed output files to %s", len(written), output_path)
        return written

    def write_parameters(self, parameters: xr.Dataset, output_path: Path) -> None:
        """Write static parameter NetCDF for ``pws.Parameters.from_netcdf()``.

        Extract static parameters (excluding forcing time series and
        soltab variables) and write them to a CF-1.8 compliant NetCDF
        file with pywatershed-compatible dimension names.

        Parameters
        ----------
        parameters
            Derived parameter dataset.  Forcing variables (``prcp``,
            ``tmax``, ``tmin``, ``swrad``, ``potet``) and soltab
            variables (``soltab_potsw``, ``soltab_horad_potsw``) are
            excluded -- they are written to separate files.
        output_path
            Path for the output NetCDF file (e.g.,
            ``output/parameters.nc``).

        Notes
        -----
        If no static parameters remain after exclusion, a warning is
        logged and no file is written.
        """
        # Exclude forcing and soltab variables
        exclude = set(_FORCING_VARS.keys()) | _SOLTAB_VARS
        static_vars = [v for v in parameters.data_vars if v not in exclude]

        if not static_vars:
            logger.warning("No static parameters to write.")
            return

        param_ds = parameters[static_vars].copy()
        param_ds.attrs.update(
            {
                "title": "pywatershed parameters from hydro-param",
                "Conventions": "CF-1.8",
                "source": "hydro-param derivation pipeline",
            }
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        param_ds.to_netcdf(output_path)
        logger.info("Wrote parameter NetCDF: %s (%d variables)", output_path, len(static_vars))

    def write_forcing_netcdf(self, parameters: xr.Dataset, output_dir: Path) -> list[Path]:
        """Write forcing NetCDF files (one variable per file) in PRMS units.

        Produce one NetCDF file per forcing variable with unit
        conversions applied:

        - ``prcp.nc`` -- precipitation (mm to inches/day)
        - ``tmax.nc`` -- maximum temperature (C to F)
        - ``tmin.nc`` -- minimum temperature (C to F)
        - ``swrad.nc`` -- shortwave radiation (Langleys/day, no conversion)
        - ``potet.nc`` -- potential ET (inches, no conversion)

        Variables not present in the input dataset are silently skipped.
        pywatershed's ``PRMSAtmosphere`` accepts file paths directly
        for these inputs.

        Parameters
        ----------
        parameters
            Dataset potentially containing forcing variables (``prcp``,
            ``tmax``, ``tmin``, ``swrad``, ``potet``).  Each must have
            a time dimension.
        output_dir
            Directory for forcing output files.  Created if it does
            not exist.

        Returns
        -------
        list[Path]
            Absolute paths to forcing files written.  Empty list if no
            forcing variables are present.

        Notes
        -----
        Unit conversions use ``hydro_param.units.convert()`` with
        float64 precision to avoid truncation of precipitation values.
        """
        written: list[Path] = []
        has_forcing = any(v in parameters for v in _FORCING_VARS)
        if not has_forcing:
            logger.info("No forcing variables present; skipping forcing output.")
            return written

        output_dir.mkdir(parents=True, exist_ok=True)

        for var, (from_unit, to_unit) in _FORCING_VARS.items():
            if var not in parameters:
                continue
            da = parameters[var].copy(deep=True)
            da.values = convert(da.values.astype(np.float64), from_unit, to_unit)
            da.attrs["units"] = to_unit

            out_path = output_dir / f"{var}.nc"
            da.to_dataset(name=var).to_netcdf(out_path)
            written.append(out_path)
            logger.info("Wrote forcing: %s (%s → %s)", out_path, from_unit, to_unit)

        return written

    def write_soltab(self, parameters: xr.Dataset, output_path: Path) -> None:
        """Write potential solar radiation tables to NetCDF.

        Extract soltab variables from the derived dataset and write
        them to a standalone NetCDF file.  These tables provide
        365+1 days of potential clear-sky radiation for PRMS's
        solar geometry calculations.

        Parameters
        ----------
        parameters
            Dataset containing ``soltab_potsw`` (sloped-surface
            radiation) and/or ``soltab_horad_potsw`` (horizontal-
            surface radiation).  Expected shape: ``(nhru, 366)``.
            Units: Langleys (cal/cm2/day).
        output_path
            Path for the output NetCDF file (e.g.,
            ``output/soltab.nc``).

        Notes
        -----
        If no soltab variables are present, a warning is logged and
        no file is written.
        """
        soltab_ds = xr.Dataset()
        for var in sorted(_SOLTAB_VARS):
            if var in parameters:
                soltab_ds[var] = parameters[var]

        if not soltab_ds.data_vars:
            logger.warning("No soltab variables present.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        soltab_ds.to_netcdf(output_path)
        logger.info("Wrote soltab: %s", output_path)

    def write_control(self, config: dict, output_path: Path) -> None:
        """Write pywatershed control YAML with simulation time bounds.

        Produce a minimal control file specifying the simulation
        start time, end time, and daily timestep.  This file is
        consumed by pywatershed's control infrastructure.

        Parameters
        ----------
        config
            Configuration dict with ``start`` (str, ISO date) and
            ``end`` (str, ISO date) keys.
        output_path
            Path for the control YAML file (e.g.,
            ``output/control.yml``).

        Raises
        ------
        ValueError
            If ``start`` or ``end`` keys are missing from *config*.
        """
        start = config.get("start")
        end = config.get("end")
        if start is None or end is None:
            msg = (
                "Control configuration requires 'start' and 'end' values; "
                f"received start={start!r}, end={end!r}."
            )
            raise ValueError(msg)

        control: dict = {
            "start_time": start,
            "end_time": end,
            "time_step": "24:00:00",
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            yaml.dump(control, f, default_flow_style=False, sort_keys=False)
        logger.info("Wrote control: %s", output_path)

    def validate(self, parameters: xr.Dataset) -> list[str]:
        """Validate parameters against pywatershed metadata constraints.

        Check that all parameters marked as ``required`` in the bundled
        ``parameter_metadata.yml`` are present, and that values for
        parameters with defined ``valid_range`` fall within bounds.
        Non-finite values (NaN, inf) are excluded from range checks.

        Parameters
        ----------
        parameters
            Derived parameter dataset to validate.

        Returns
        -------
        list[str]
            Human-readable validation warning messages.  Empty list
            if all checks pass.  Returns a single-element list with
            a skip message if metadata cannot be loaded.

        Notes
        -----
        This method does not raise exceptions for invalid data --
        it collects all issues as warning strings so the caller can
        decide whether to proceed or abort.
        """
        warnings: list[str] = []
        metadata = self._load_metadata()
        if metadata is None:
            return ["Parameter metadata unavailable — validation skipped"]

        params_meta = metadata.get("parameters", {})

        # Check required parameters
        for name, meta in params_meta.items():
            if meta.get("required", False) and name not in parameters:
                warnings.append(f"Required parameter '{name}' is missing.")

        # Check value ranges
        for name, meta in params_meta.items():
            if name not in parameters:
                continue
            valid_range = meta.get("valid_range")
            if valid_range is None:
                continue
            vmin, vmax = valid_range
            values = parameters[name].values
            if np.isscalar(values):
                values = np.array([values])
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            if np.any(values < vmin):
                n_bad = int(np.sum(values < vmin))
                warnings.append(f"Parameter '{name}': {n_bad} value(s) below minimum {vmin}")
            if np.any(values > vmax):
                n_bad = int(np.sum(values > vmax))
                warnings.append(f"Parameter '{name}': {n_bad} value(s) above maximum {vmax}")

        return warnings

    def _load_metadata(self) -> dict | None:
        """Load and cache the bundled parameter metadata YAML.

        Returns
        -------
        dict or None
            Parsed metadata dict, or ``None`` if the file is missing
            or contains invalid YAML.  Result is cached for subsequent
            calls.
        """
        if self._metadata_cache is not None:
            return self._metadata_cache
        path = self._default_metadata_path()
        try:
            with open(path) as f:
                self._metadata_cache = yaml.safe_load(f)
            return self._metadata_cache
        except FileNotFoundError:
            logger.error(
                "Bundled parameter metadata not found at '%s'. "
                "This indicates a broken installation — reinstall hydro-param.",
                path,
            )
            return None
        except yaml.YAMLError as exc:
            logger.error(
                "Parameter metadata at '%s' contains invalid YAML: %s",
                path,
                exc,
            )
            return None
