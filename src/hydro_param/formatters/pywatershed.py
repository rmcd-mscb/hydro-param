"""pywatershed output formatter.

Converts derived pywatershed parameters into the file format(s)
expected by pywatershed v2.0:

1. Parameter NetCDF — loadable by ``pws.Parameters.from_netcdf()``
2. CBH NetCDF files — prcp.nc, tmax.nc, tmin.nc with PRMS units
3. Soltab NetCDF — potential solar radiation tables (nhru × 366)
4. Control YAML — simulation configuration

See docs/reference/pywatershed_parameterization_guide.md §2C.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from hydro_param.units import convert

logger = logging.getLogger(__name__)

# CBH variables and their required unit conversions
_CBH_VARS: dict[str, tuple[str, str]] = {
    "prcp": ("mm", "in"),
    "tmax": ("C", "F"),
    "tmin": ("C", "F"),
}

# Soltab variable names
_SOLTAB_VARS = {"soltab_potsw", "soltab_horad_potsw"}

# Path to parameter metadata YAML (relative to project root)
_PARAM_METADATA_PATH = Path("configs/pywatershed/parameter_metadata.yml")


class PywatershedFormatter:
    """Format derived parameters for pywatershed consumption.

    Produces parameter NetCDF, CBH NetCDF files, soltab arrays,
    and control configuration compatible with pywatershed v2.0.
    """

    name: str = "pywatershed"

    def __init__(self, metadata_path: Path | None = None) -> None:
        self._metadata_path = metadata_path or _PARAM_METADATA_PATH
        self._metadata_cache: dict | None = None

    @property
    def metadata_path(self) -> Path:
        """Path to the parameter metadata YAML."""
        return self._metadata_path

    def has_metadata(self) -> bool:
        """Return True if the parameter metadata file exists."""
        return self._metadata_path.exists()

    def write(
        self,
        parameters: xr.Dataset,
        output_path: Path,
        config: dict,
    ) -> list[Path]:
        """Write all pywatershed output files.

        Parameters
        ----------
        parameters
            Derived pywatershed parameter dataset.
        output_path
            Output directory.
        config
            Formatter configuration with keys: ``parameter_file``,
            ``cbh_dir``, ``soltab_file``, ``control_file``,
            ``start``, ``end``.

        Returns
        -------
        list[Path]
            Paths to all files written.
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

        # 2. CBH files (only if climate data present)
        cbh_paths = self.write_cbh(parameters, output_path / config.get("cbh_dir", "cbh"))
        written.extend(cbh_paths)

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
        """Write parameter NetCDF for ``pws.Parameters.from_netcdf()``.

        Excludes CBH time-series and soltab variables (written separately).
        Sets CF-1.8 attributes and pywatershed-compatible dimensions.

        Parameters
        ----------
        parameters
            Derived parameter dataset.
        output_path
            Path for the output NetCDF file.
        """
        # Exclude CBH and soltab variables
        exclude = set(_CBH_VARS.keys()) | _SOLTAB_VARS
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

    def write_cbh(self, parameters: xr.Dataset, output_dir: Path) -> list[Path]:
        """Write Climate-By-HRU NetCDF files with PRMS unit conversions.

        Produces ``prcp.nc`` (inches/day), ``tmax.nc`` (°F),
        ``tmin.nc`` (°F).  Skips variables not present in the dataset.

        Parameters
        ----------
        parameters
            Dataset potentially containing ``prcp``, ``tmax``, ``tmin``.
        output_dir
            Directory for CBH output files.

        Returns
        -------
        list[Path]
            Paths to CBH files written.
        """
        written: list[Path] = []
        has_cbh = any(v in parameters for v in _CBH_VARS)
        if not has_cbh:
            logger.info("No CBH variables present; skipping CBH output.")
            return written

        output_dir.mkdir(parents=True, exist_ok=True)

        for var, (from_unit, to_unit) in _CBH_VARS.items():
            if var not in parameters:
                continue
            da = parameters[var].copy(deep=True)
            da.values = convert(da.values.astype(np.float64), from_unit, to_unit)
            da.attrs["units"] = to_unit

            out_path = output_dir / f"{var}.nc"
            da.to_dataset(name=var).to_netcdf(out_path)
            written.append(out_path)
            logger.info("Wrote CBH: %s (%s → %s)", out_path, from_unit, to_unit)

        return written

    def write_soltab(self, parameters: xr.Dataset, output_path: Path) -> None:
        """Write potential solar radiation tables.

        Parameters
        ----------
        parameters
            Dataset containing ``soltab_potsw`` and/or
            ``soltab_horad_potsw`` (dimension: ``[nhru, 366]``).
        output_path
            Path for the output NetCDF file.
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
        """Write pywatershed control YAML.

        Parameters
        ----------
        config
            Configuration dict with ``start`` and ``end`` keys.
        output_path
            Path for the control YAML file.
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
        """Validate parameters against pywatershed requirements.

        Checks that required parameters are present and values fall
        within valid ranges defined in ``parameter_metadata.yml``.

        Parameters
        ----------
        parameters
            Derived parameter dataset.

        Returns
        -------
        list[str]
            Validation warnings. Empty if all checks pass.
        """
        warnings: list[str] = []
        metadata = self._load_metadata()
        if metadata is None:
            return warnings

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
        """Load parameter metadata YAML."""
        if self._metadata_cache is not None:
            return self._metadata_cache
        try:
            with open(self._metadata_path) as f:
                self._metadata_cache = yaml.safe_load(f)
            return self._metadata_cache
        except FileNotFoundError:
            logger.warning("Parameter metadata not found: %s", self._metadata_path)
            return None
