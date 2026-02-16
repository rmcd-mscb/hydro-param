"""pywatershed parameter derivation plugin.

Converts SIR physical properties (zonal statistics of raw geospatial
data) into PRMS/pywatershed model parameters.  Implements the
derivation pipeline from ``pywatershed_dataset_param_map.yml``.

Foundation implementation covers steps 1, 3, 4, 8, and 13.
Future PRs will add steps 2 (topology), 5 (soils), 6 (waterbody
overlay), 7 (CBH generation), 9 (soltab), 10 (PET), 11 (transp),
12 (routing), and 14 (calibration seeds).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from hydro_param.units import convert

logger = logging.getLogger(__name__)

# Default PRMS values from Regan et al. 2018 / Markstrom et al. 2015
_DEFAULTS: dict[str, float] = {
    # Snow process
    "den_init": 0.10,
    "den_max": 0.60,
    "settle_const": 0.10,
    "emis_noppt": 0.757,
    "freeh2o_cap": 0.05,
    "potet_sublim": 0.75,
    "tmax_allsnow": 32.0,  # degF
    # Albedo
    "albset_rna": 0.8,
    "albset_snm": 0.1,
    "albset_rnm": 0.6,
    "albset_sna": 0.05,
    # Radiation
    "radmax": 0.8,
    "radj_sppt": 0.44,
    "radj_wppt": 0.50,
    # Routing
    "x_coef": 0.2,
    # Initial conditions
    "soil_moist_init_frac": 0.5,
    "soil_rechr_init_frac": 0.5,
    "ssstor_init_frac": 0.0,
    "gwstor_init": 2.0,  # inches
    "gwstor_min": 0.0,
    "dprst_depth_avg": 24.0,  # inches
    # Transpiration
    "transp_tmax": 500.0,  # degree-days
}

# Default imperv_stor_max by cov_type (inches)
_IMPERV_STOR_MAX_DEFAULT = 0.03


class PywatershedDerivation:
    """Derive pywatershed/PRMS parameters from SIR physical properties.

    Implements the derivation pipeline from
    ``docs/reference/pywatershed_dataset_param_map.yml``.  The foundation
    covers pipeline steps 1 (geometry), 3 (topography), 4 (land cover),
    8 (lookup tables), and 13 (defaults).

    Parameters
    ----------
    lookup_tables_dir
        Path to the directory containing lookup table YAML files.
        Defaults to ``configs/lookup_tables``.
    """

    def __init__(self, lookup_tables_dir: Path | None = None) -> None:
        self._lookup_tables_dir = lookup_tables_dir or Path("configs/lookup_tables")
        self._lookup_cache: dict[str, dict] = {}

    def derive(
        self,
        sir: xr.Dataset,
        *,
        config: dict | None = None,
    ) -> xr.Dataset:
        """Derive all pywatershed parameters from the SIR.

        Parameters
        ----------
        sir
            Standardized Internal Representation with physical properties
            (e.g., ``elevation``, ``slope``, ``aspect``, ``land_cover``).
        config
            Optional derivation configuration.  Supports keys:
            ``generate_seeds`` (bool), ``parameter_overrides`` (dict).

        Returns
        -------
        xr.Dataset
            pywatershed parameter dataset with PRMS-convention units.
        """
        config = config or {}
        nhru = sir.sizes.get("hru_id", 0)
        ds = xr.Dataset()

        # Step 1: Geometry (hru_area, hru_lat)
        ds = self._derive_geometry(sir, ds)

        # Step 3: Topographic parameters (hru_elev, hru_slope, hru_aspect)
        ds = self._derive_topography(sir, ds)

        # Step 4: Land cover parameters (cov_type, covden_sum, hru_percent_imperv)
        ds = self._derive_landcover(sir, ds)

        # Step 8: Lookup table application
        ds = self._apply_lookup_tables(ds)

        # Step 13: Defaults and initial conditions
        ds = self._apply_defaults(ds, nhru)

        # Apply parameter overrides last
        overrides = config.get("parameter_overrides", {})
        if isinstance(overrides, dict) and "values" in overrides:
            overrides = overrides["values"]
        if overrides:
            ds = self._apply_overrides(ds, overrides)

        return ds

    # ------------------------------------------------------------------
    # Step 1: Geometry extraction
    # ------------------------------------------------------------------

    def _derive_geometry(self, sir: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
        """Step 1: Compute basic geometry from the target fabric.

        Expects the SIR to contain ``hru_area_m2`` (HRU area in square
        meters) and ``hru_lat`` (centroid latitude in decimal degrees).
        """
        if "hru_area_m2" in sir:
            ds["hru_area"] = xr.DataArray(
                convert(sir["hru_area_m2"].values, "m2", "acres"),
                dims="nhru",
                attrs={"units": "acres", "long_name": "Area of HRU"},
            )
        if "hru_lat" in sir:
            ds["hru_lat"] = xr.DataArray(
                sir["hru_lat"].values,
                dims="nhru",
                attrs={"units": "decimal_degrees", "long_name": "Latitude of HRU centroid"},
            )
        return ds

    # ------------------------------------------------------------------
    # Step 3: Topographic parameters
    # ------------------------------------------------------------------

    def _derive_topography(self, sir: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
        """Step 3: Convert DEM zonal statistics to PRMS parameters."""
        if "elevation" in sir:
            ds["hru_elev"] = xr.DataArray(
                convert(sir["elevation"].values, "m", "ft"),
                dims="nhru",
                attrs={"units": "feet", "long_name": "Mean HRU elevation"},
            )

        if "slope" in sir:
            # SIR slope is in degrees; PRMS wants decimal fraction (rise/run)
            slope_rad = convert(sir["slope"].values, "deg", "rad")
            ds["hru_slope"] = xr.DataArray(
                np.tan(slope_rad),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "Mean HRU slope"},
            )

        if "aspect" in sir:
            ds["hru_aspect"] = xr.DataArray(
                sir["aspect"].values.astype(np.float64),
                dims="nhru",
                attrs={"units": "degrees", "long_name": "Mean HRU aspect"},
            )

        return ds

    # ------------------------------------------------------------------
    # Step 4: Land cover parameters
    # ------------------------------------------------------------------

    def _derive_landcover(self, sir: xr.Dataset, ds: xr.Dataset) -> xr.Dataset:
        """Step 4: Derive vegetation and impervious parameters from NLCD.

        Expects the SIR to contain ``land_cover`` (dominant NLCD class
        per HRU), and optionally ``tree_canopy`` (percent, 0-100) and
        ``impervious`` (percent, 0-100).
        """
        if "land_cover" in sir:
            nlcd_table = self._load_lookup_table("nlcd_to_prms_cov_type")
            mapping = nlcd_table["mapping"]
            nlcd_values = sir["land_cover"].values.astype(int)
            cov_type = np.array([mapping.get(int(v), 0) for v in nlcd_values])
            ds["cov_type"] = xr.DataArray(
                cov_type,
                dims="nhru",
                attrs={"units": "integer", "long_name": "Vegetation cover type"},
            )

        if "tree_canopy" in sir:
            # Continuous canopy cover (0-100%) → fraction (0-1)
            ds["covden_sum"] = xr.DataArray(
                np.clip(sir["tree_canopy"].values / 100.0, 0.0, 1.0),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "Summer vegetation cover density"},
            )
        elif "cov_type" in ds:
            # Fallback: simple lookup-based canopy density
            _covden_lookup = {0: 0.0, 1: 0.3, 2: 0.4, 3: 0.7, 4: 0.8}
            covden = np.array([_covden_lookup.get(int(v), 0.3) for v in ds["cov_type"].values])
            ds["covden_sum"] = xr.DataArray(
                covden,
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "Summer vegetation cover density"},
            )

        if "impervious" in sir:
            # Percent (0-100) → fraction (0-1)
            ds["hru_percent_imperv"] = xr.DataArray(
                np.clip(sir["impervious"].values / 100.0, 0.0, 1.0),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "HRU impervious fraction"},
            )

        return ds

    # ------------------------------------------------------------------
    # Step 8: Lookup table application
    # ------------------------------------------------------------------

    def _apply_lookup_tables(self, ds: xr.Dataset) -> xr.Dataset:
        """Step 8: Apply lookup tables for interception and winter cover.

        Requires ``cov_type`` and ``covden_sum`` to already be in ``ds``.
        """
        if "cov_type" not in ds:
            return ds

        cov_type_vals = ds["cov_type"].values.astype(int)

        # Interception capacities
        intcp_table = self._load_lookup_table("cov_type_to_interception")
        columns = intcp_table["columns"]  # [srain_intcp, wrain_intcp, snow_intcp]
        mapping = intcp_table["mapping"]

        for i, col_name in enumerate(columns):
            values = np.array([mapping.get(int(ct), [0.0, 0.0, 0.0])[i] for ct in cov_type_vals])
            ds[col_name] = xr.DataArray(
                values,
                dims="nhru",
                attrs={"units": "inches", "long_name": f"{col_name.replace('_', ' ').title()}"},
            )

        # Imperv storage max (uniform default)
        nhru = len(cov_type_vals)
        ds["imperv_stor_max"] = xr.DataArray(
            np.full(nhru, _IMPERV_STOR_MAX_DEFAULT),
            dims="nhru",
            attrs={"units": "inches", "long_name": "Maximum impervious retention storage"},
        )

        # Winter cover density
        if "covden_sum" in ds:
            winter_table = self._load_lookup_table("cov_type_winter_reduction")
            winter_mapping = winter_table["mapping"]
            reduction = np.array([winter_mapping.get(int(ct), 0.5) for ct in cov_type_vals])
            ds["covden_win"] = xr.DataArray(
                ds["covden_sum"].values * reduction,
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Winter vegetation cover density",
                },
            )

        return ds

    # ------------------------------------------------------------------
    # Step 13: Defaults and initial conditions
    # ------------------------------------------------------------------

    def _apply_defaults(self, ds: xr.Dataset, nhru: int) -> xr.Dataset:
        """Step 13: Apply standard PRMS default values.

        Only sets parameters that are not already present in ``ds``
        (preserves any values derived from data in earlier steps).
        """
        for param_name, default_val in _DEFAULTS.items():
            if param_name not in ds:
                ds[param_name] = xr.DataArray(
                    np.float64(default_val),
                    attrs={"long_name": param_name.replace("_", " ").title()},
                )
        return ds

    # ------------------------------------------------------------------
    # Parameter overrides
    # ------------------------------------------------------------------

    def _apply_overrides(self, ds: xr.Dataset, overrides: dict) -> xr.Dataset:
        """Apply user-specified parameter value overrides.

        Overrides are applied last, after all derivation steps.
        Values can be scalars or lists matching the parameter dimension.
        """
        for param_name, value in overrides.items():
            if isinstance(value, list):
                arr: np.floating | np.ndarray = np.array(value, dtype=np.float64)
            else:
                arr = np.array(value, dtype=np.float64)

            if param_name in ds:
                ds[param_name].values = arr
                logger.info("Override: %s = %s", param_name, value)
            else:
                ds[param_name] = xr.DataArray(
                    arr,
                    attrs={"long_name": f"{param_name} (user override)"},
                )
                logger.info("Override (new): %s = %s", param_name, value)
        return ds

    # ------------------------------------------------------------------
    # Future steps (stubs)
    # ------------------------------------------------------------------

    # TODO: Step 2 — topology extraction (tosegment, hru_segment)
    # TODO: Step 5 — soils (soil_type, soil_moist_max, soil_rechr_max_frac)
    # TODO: Step 6 — waterbody overlay (dprst_frac, dprst_area_max, hru_type)
    # TODO: Step 7 — CBH generation (prcp, tmax, tmin time series)
    # TODO: Step 9 — soltab computation (Swift 1976 algorithm)
    # TODO: Step 10 — PET coefficients (jh_coef, jh_coef_hru)
    # TODO: Step 11 — climate-derived params (transp_beg, transp_end)
    # TODO: Step 12 — routing coefficients (K_coef from seg_length + velocity)
    # TODO: Step 14 — calibration seeds (physically-based initial values)

    # ------------------------------------------------------------------
    # Lookup table loader
    # ------------------------------------------------------------------

    def _load_lookup_table(self, name: str) -> dict:
        """Load a YAML lookup table from the lookup tables directory.

        Parameters
        ----------
        name
            Lookup table filename (without ``.yml`` extension).

        Returns
        -------
        dict
            Parsed YAML content with at least a ``mapping`` key.
        """
        if name not in self._lookup_cache:
            path = self._lookup_tables_dir / f"{name}.yml"
            with open(path) as f:
                self._lookup_cache[name] = yaml.safe_load(f)
        return self._lookup_cache[name]
