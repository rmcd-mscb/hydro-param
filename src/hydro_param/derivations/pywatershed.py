"""pywatershed parameter derivation plugin.

Convert SIR physical properties (zonal statistics of raw geospatial data)
into PRMS/pywatershed model parameters.  This module implements the
model-specific derivation pipeline defined in
``docs/reference/pywatershed_dataset_param_map.yml``, transforming
source-unit geospatial statistics into the internal unit system required
by PRMS (feet, inches, degrees Fahrenheit, acres).

The derivation follows a 15-step DAG.  Steps implemented here:

1. Geometry --- HRU area (acres) and latitude (decimal degrees)
2. Topology --- segment routing (tosegment, hru_segment, seg_length)
3. Topography --- elevation (feet), slope (decimal fraction), aspect (degrees)
3b. Segment elevation --- mean channel elevation from 3DEP DEM via InterpGen
4. Land cover --- NLCD reclassification to PRMS cov_type, canopy density, imperviousness
5. Soils --- gNATSGO/STATSGO2 texture classification and AWC
6. Waterbody --- NHDPlus depression storage overlay
7. Forcing --- temporal CBH merge (prcp, tmax, tmin)
8. Lookup tables --- interception capacities and winter cover density
9. Soltab --- potential solar radiation tables (Swift 1976)
10. PET --- Jensen-Haise coefficients from climate normals
11. Transpiration --- frost-free period timing from monthly tmin
12. Routing --- Muskingum routing parameters (K_coef, x_coef, seg_slope, etc.)
13. Defaults --- standard PRMS default values and initial conditions
14. Calibration seeds --- physically-based initial values for calibration parameters

References
----------
Markstrom, S. L., et al. (2015). PRMS-IV, the Precipitation-Runoff
    Modeling System, Version 4. USGS Techniques and Methods 6-B7.
Regan, R. S., et al. (2018). Description of the National Hydrologic
    Model Infrastructure. USGS Techniques and Methods 6-B9.

See Also
--------
hydro_param.plugins.DerivationContext : Typed input bundle for derivation.
hydro_param.units.convert : Unit conversion dispatch used throughout.
hydro_param.solar.compute_soltab : Solar radiation table computation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pynhd
import pyproj
import xarray as xr
import yaml
from shapely.errors import GEOSException

from hydro_param.classification import USDA_TEXTURE_CLASSES, classify_usda_texture
from hydro_param.plugins import DerivationContext
from hydro_param.sir_accessor import SIRAccessor
from hydro_param.solar import compute_soltab
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
    # Initial conditions
    "soil_moist_init_frac": 0.5,
    "soil_rechr_init_frac": 0.5,
    "ssstor_init_frac": 0.0,
    "gwstor_init": 2.0,  # inches
    "gwstor_min": 0.0,
    "dprst_depth_avg": 24.0,  # inches
    # Transpiration
    "transp_tmax": 500.0,  # degree-days
    # PET (Jensen-Haise)
    "jh_coef": 0.014,
    "jh_coef_hru": -10.0,  # temperature threshold Tx in °F
    # Transpiration timing
    "transp_beg": 4,  # April
    "transp_end": 10,  # October
    # --- Depression storage (operational) ---
    "dprst_et_coef": 1.0,
    "dprst_flow_coef": 0.05,
    "dprst_frac_init": 0.5,
    "dprst_frac_open": 1.0,
    "dprst_seep_rate_clos": 0.02,
    "dprst_seep_rate_open": 0.02,
    "sro_to_dprst_imperv": 0.2,
    "sro_to_dprst_perv": 0.2,
    "op_flow_thres": 1.0,
    "va_clos_exp": 0.001,
    "va_open_exp": 0.001,
    # --- Snow (additional) ---
    "cecn_coef": 5.0,
    "rad_trncf": 0.5,
    "melt_force": 140,  # Julian day
    "melt_look": 90,  # Julian day
    "snowinfil_max": 2.0,
    "snowpack_init": 0.0,
    "hru_deplcrv": 1,  # index into snarea_curve
    "tstorm_mo": 0,
    # --- Atmosphere ---
    "ppt_rad_adj": 0.02,
    "radadj_intcp": 1.0,
    "radadj_slope": 0.0,
    "tmax_index": 50.0,  # degF
    # --- Soilzone ---
    "sat_threshold": 999.0,
    # Depression storage — hru_type only; dprst_frac is always set by
    # _derive_waterbody (or _waterbody_defaults), so no scalar fallback
    # is needed here.
    "hru_type": 1,
}

# Parameters with non-scalar defaults handled specially in _apply_defaults
_DEFAULTS_SPECIAL: frozenset[str] = frozenset(
    {
        "jh_coef",
        "transp_beg",
        "transp_end",
        "hru_type",
        "doy",
        "hru_in_to_cf",
        "temp_units",
        "elev_units",
        "snarea_curve",
        "pref_flow_infil_frac",
    }
)

# Dimension mapping for default parameters.  Every entry in _DEFAULTS
# that is NOT in _DEFAULTS_SPECIAL must appear here.  Additionally,
# calibration seed parameters from step 14 are included for correct
# shape broadcasting in _derive_calibration_seeds.  pywatershed v2.0
# requires all parameters as correctly-dimensioned arrays.
_PARAM_DIMS: dict[str, tuple[str, ...]] = {
    # Per-HRU (nhru,)
    "albset_rna": ("nhru",),
    "albset_rnm": ("nhru",),
    "albset_sna": ("nhru",),
    "albset_snm": ("nhru",),
    "den_init": ("nhru",),
    "den_max": ("nhru",),
    "settle_const": ("nhru",),
    "emis_noppt": ("nhru",),
    "freeh2o_cap": ("nhru",),
    "potet_sublim": ("nhru",),
    "radj_sppt": ("nhru",),
    "radj_wppt": ("nhru",),
    "soil_moist_init_frac": ("nhru",),
    "soil_rechr_init_frac": ("nhru",),
    "ssstor_init_frac": ("nhru",),
    "gwstor_init": ("nhru",),
    "gwstor_min": ("nhru",),
    "dprst_depth_avg": ("nhru",),
    "transp_tmax": ("nhru",),
    "jh_coef_hru": ("nhru",),
    # Depression storage (operational)
    "dprst_et_coef": ("nhru",),
    "dprst_flow_coef": ("nhru",),
    "dprst_frac_init": ("nhru",),
    "dprst_frac_open": ("nhru",),
    "dprst_seep_rate_clos": ("nhru",),
    "dprst_seep_rate_open": ("nhru",),
    "sro_to_dprst_imperv": ("nhru",),
    "sro_to_dprst_perv": ("nhru",),
    "op_flow_thres": ("nhru",),
    "va_clos_exp": ("nhru",),
    "va_open_exp": ("nhru",),
    # Snow (additional)
    "rad_trncf": ("nhru",),
    "melt_force": ("nhru",),
    "melt_look": ("nhru",),
    "snowinfil_max": ("nhru",),
    "snowpack_init": ("nhru",),
    "hru_deplcrv": ("nhru",),
    # Soilzone
    "sat_threshold": ("nhru",),
    # Per-month-per-HRU (nmonth, nhru)
    "tmax_allsnow": ("nmonth", "nhru"),
    "radmax": ("nmonth", "nhru"),
    "cecn_coef": ("nmonth", "nhru"),
    "tstorm_mo": ("nmonth", "nhru"),
    "ppt_rad_adj": ("nmonth", "nhru"),
    "radadj_intcp": ("nmonth", "nhru"),
    "radadj_slope": ("nmonth", "nhru"),
    "tmax_index": ("nmonth", "nhru"),
    # Monthly calibration seeds (step 14)
    "rain_cbh_adj": ("nmonth", "nhru"),
    "snow_cbh_adj": ("nmonth", "nhru"),
    "tmax_cbh_adj": ("nmonth", "nhru"),
    "tmin_cbh_adj": ("nmonth", "nhru"),
    "tmax_allrain_offset": ("nmonth", "nhru"),
    "adjmix_rain": ("nmonth", "nhru"),
    "dday_slope": ("nmonth", "nhru"),
    "dday_intcp": ("nmonth", "nhru"),
}

# Validate _PARAM_DIMS covers all non-special defaults at import time.
_DEFAULTS_NEEDING_DIMS = set(_DEFAULTS) - _DEFAULTS_SPECIAL
_MISSING_DIMS = _DEFAULTS_NEEDING_DIMS - set(_PARAM_DIMS)
if _MISSING_DIMS:
    raise ImportError(
        f"Parameters in _DEFAULTS missing from _PARAM_DIMS "
        f"(would silently default to (nhru,)): {sorted(_MISSING_DIMS)}. "
        f"Add explicit dimension mappings."
    )

# Known dimension names referenced by _PARAM_DIMS.
_KNOWN_DIMS = frozenset({"nhru", "nmonth"})
_BAD_DIMS = {
    p: [d for d in dims if d not in _KNOWN_DIMS]
    for p, dims in _PARAM_DIMS.items()
    if any(d not in _KNOWN_DIMS for d in dims)
}
if _BAD_DIMS:
    raise ImportError(
        f"_PARAM_DIMS contains unknown dimension names: {_BAD_DIMS}. "
        f"Known dimensions: {sorted(_KNOWN_DIMS)}."
    )

# Default imperv_stor_max by cov_type (inches)
_IMPERV_STOR_MAX_DEFAULT = 0.03

# Routing constants (Step 12)
_MANNING_N = 0.04  # natural channel roughness coefficient
_DEFAULT_DEPTH_FT = 1.0  # bankfull depth placeholder (feet)
_MIN_SLOPE = 1e-7  # pywatershed floor for seg_slope
_FALLBACK_SLOPE = 1e-4  # for segments with no NHDPlus match
_K_COEF_MIN = 0.01  # hours
_K_COEF_MAX = 24.0  # hours
_DEFAULT_K_COEF = 1.0  # hours, when computation not possible
_DEFAULT_X_COEF = 0.2  # standard Muskingum weighting
_LAKE_K_COEF = 24.0  # travel time for lake segments
_LAKE_SEGMENT_TYPE = 1  # segment_type value for lake
_CHANNEL_SEGMENT_TYPE = 0  # segment_type value for channel
_SPATIAL_JOIN_BUFFER_M = 100.0  # metres — buffer around segments for NHD flowline clipping
_NHD_MISSING_SLOPE_SENTINEL = -9998.0  # NHDPlus missing-data sentinel for slope
_KM2_TO_ACRES = 247.10538146717  # 1 km² = 247.105 acres

# Square metres per acre (exact)
_M2_PER_ACRE = 4046.8564224

# Calibration seed method dispatch — safe lambdas only, NO eval.
_SEED_METHODS: dict[str, Callable[..., np.floating | np.ndarray]] = {
    "linear": lambda ds, p: p["scale"] * ds[p["input"]].values + p["offset"],
    "exponential_scale": lambda ds, p: p["scale"] * np.exp(p["exponent"] * ds[p["input"]].values),
    "fraction_of": lambda ds, p: p["fraction"] * ds[p["input"]].values,
    "constant": lambda ds, p: np.float64(p["value"]),
}


def _sat_vp(temp_f: np.ndarray) -> np.ndarray:
    """Compute saturation vapor pressure from temperature.

    Apply the Magnus formula (Alduchov & Eskridge 1996) to convert
    temperature in degrees Fahrenheit to saturation vapor pressure.
    Used internally by step 10 (PET coefficients) for the Jensen-Haise
    equation.

    Parameters
    ----------
    temp_f : np.ndarray
        Temperature in degrees Fahrenheit (°F).

    Returns
    -------
    np.ndarray
        Saturation vapor pressure in hectopascals (hPa).

    Notes
    -----
    Internally converts °F to °C before applying the Magnus formula:
    ``es = 6.1078 * exp(17.269 * T_c / (T_c + 237.3))``.

    References
    ----------
    Alduchov, O. A. and Eskridge, R. E. (1996). Improved Magnus Form
        Approximation of Saturation Vapor Pressure. J. Appl. Meteor.,
        35, 601-609.
    """
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    return 6.1078 * np.exp(17.269 * temp_c / (temp_c + 237.3))


class PywatershedDerivation:
    """Derive pywatershed/PRMS parameters from SIR physical properties.

    Implement the full derivation pipeline defined in
    ``docs/reference/pywatershed_dataset_param_map.yml``, converting
    source-unit zonal statistics from the Standardized Internal
    Representation (SIR) into the ~100+ parameters required by
    pywatershed (PRMS-IV in Python).

    The derivation follows a directed acyclic graph (DAG) of 15 ordered
    steps.  Each step is implemented as a private method (e.g.,
    ``_derive_geometry`` for step 1).  Steps execute in dependency order:
    later steps may read parameters produced by earlier ones (e.g.,
    step 8 reads ``cov_type`` from step 4).

    This class conforms to the ``DerivationPlugin`` protocol defined in
    ``hydro_param.plugins``.

    Attributes
    ----------
    name : str
        Plugin identifier (``"pywatershed"``).

    Notes
    -----
    PRMS internal units are: feet, inches, degrees Fahrenheit, acres.
    All unit conversions from SI source data happen within individual
    derivation steps using ``hydro_param.units.convert``.

    Lookup tables are loaded lazily from YAML files and cached in
    ``_lookup_cache`` for the lifetime of the instance.

    References
    ----------
    Markstrom, S. L., et al. (2015). PRMS-IV, the Precipitation-Runoff
        Modeling System, Version 4. USGS Techniques and Methods 6-B7.

    See Also
    --------
    hydro_param.plugins.DerivationContext : Input bundle for ``derive()``.
    hydro_param.plugins.DerivationPlugin : Protocol this class implements.
    hydro_param.formatters.pywatershed : Output formatter for pywatershed.
    """

    name: str = "pywatershed"

    def __init__(self) -> None:
        self._lookup_cache: dict[str, dict] = {}

    @staticmethod
    def _try_precomputed(
        ctx: DerivationContext,
        param_name: str,
        *,
        categorical: bool = False,
    ) -> np.ndarray | None:
        """Load a pre-computed parameter from the SIR if declared.

        Check whether ``ctx.precomputed`` declares a pre-computed source
        for ``param_name``.  If so, resolve the SIR variable name from
        the declaration and load it.  For categorical parameters, load
        the full dataset and extract the majority class via argmax.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context with ``precomputed`` map and ``sir``.
        param_name : str
            PRMS parameter name (e.g., ``"covden_sum"``).
        categorical : bool
            If ``True``, the SIR variable is a categorical fraction file
            and the majority class index is extracted via argmax.

        Returns
        -------
        np.ndarray or None
            Pre-computed values if found and loaded, ``None`` otherwise.

        Notes
        -----
        Logs at INFO level when a pre-computed value is successfully loaded.
        Logs at WARNING level when a declaration exists but the SIR variable
        cannot be found (indicating a config/pipeline mismatch).

        For categorical variables, fraction columns are identified by the
        ``_<digit>`` suffix pattern.  All-NaN rows (HRUs outside raster
        coverage) are detected and warned about; ``nanargmax`` assigns
        them to the first class index.
        """
        if ctx.precomputed is None or param_name not in ctx.precomputed:
            return None

        decl = ctx.precomputed[param_name]
        sir_var = decl["variable"]
        source = decl["source"]
        sir = ctx.sir

        if categorical:
            # Categorical fraction file: load full dataset, extract majority.
            # Try "{var}_frac" first (SIR convention), then "{var}" as-is.
            frac_key = sir.find_variable(f"{sir_var}_frac")
            if frac_key is None:
                frac_key = sir.find_variable(sir_var)
            if frac_key is None:
                logger.warning(
                    "Pre-computed '%s' declared (source=%s) but '%s_frac' "
                    "not found in SIR; falling back to derivation.",
                    param_name,
                    source,
                    sir_var,
                )
                return None
            frac_ds = sir.load_dataset(frac_key)
            # Fraction columns end with _<digit> (e.g. cov_type_frac_0).
            # Exclude _count columns and the bare anchor key.
            frac_cols = sorted(str(v) for v in frac_ds.data_vars if re.search(r"_\d+$", str(v)))
            if len(frac_cols) < 2:
                logger.warning(
                    "Pre-computed '%s': fraction file has < 2 class columns; "
                    "falling back to derivation.",
                    param_name,
                )
                return None
            # Extract class indices from column names (e.g., "soil_type_frac_1" → 1).
            # The regex filter guarantees each column ends with _<digits>.
            class_indices = np.array([int(str(col).rsplit("_", 1)[-1]) for col in frac_cols])
            fractions = np.column_stack([frac_ds[col].values for col in frac_cols])
            # Warn on all-NaN rows (HRUs outside raster coverage).
            nan_rows = np.all(np.isnan(fractions), axis=1)
            if np.any(nan_rows):
                logger.warning(
                    "Pre-computed '%s': %d/%d HRUs have all-NaN fractions; "
                    "these will be assigned class index %d (first class).",
                    param_name,
                    int(np.sum(nan_rows)),
                    len(fractions),
                    int(class_indices[0]),
                )
            majority_pos = np.nanargmax(fractions, axis=1)
            result = class_indices[majority_pos]
            logger.info(
                "Using pre-computed '%s' from %s (categorical majority, %d classes)",
                param_name,
                source,
                len(frac_cols),
            )
            return result

        # Continuous variable: try multiple SIR name patterns
        candidates = [sir_var]
        # Also try common SIR naming conventions
        stat = decl.get("statistic", "mean")
        if f"_{stat}" not in sir_var:
            candidates.append(f"{sir_var}_{stat}")

        for candidate in candidates:
            found = sir.find_variable(candidate)
            if found is not None:
                values = sir[found].values
                logger.info(
                    "Using pre-computed '%s' from %s (SIR variable: %s)",
                    param_name,
                    source,
                    found,
                )
                return values

        logger.warning(
            "Pre-computed '%s' declared (source=%s, variable=%s) but "
            "not found in SIR; falling back to derivation.",
            param_name,
            source,
            sir_var,
        )
        return None

    def derive(self, context: DerivationContext) -> xr.Dataset:
        """Derive all pywatershed parameters from the SIR.

        Execute the full derivation DAG in dependency order, producing a
        single ``xr.Dataset`` containing all derivable PRMS parameters.
        Steps that lack required input data log warnings and are skipped
        gracefully.  Parameter overrides from the config are applied last.

        Parameters
        ----------
        context : DerivationContext
            Typed input bundle containing the SIR dataset, target fabric
            GeoDataFrame, segment GeoDataFrame, waterbody GeoDataFrame,
            temporal forcing datasets, lookup table directory, and
            pipeline configuration.  ``temporal`` may be ``None`` if no
            temporal SIR data is available, in which case step 7 (forcing)
            is skipped and PET/transpiration steps use scalar defaults.

        Returns
        -------
        xr.Dataset
            Parameter dataset with PRMS-convention variable names and units.
            Dimensions are ``nhru`` (and ``nsegment`` for routing, ``ndoy``
            for soltab, ``nmonth`` for monthly parameters, ``time`` for
            forcing).

        Notes
        -----
        Step execution order: 1 (geometry) -> 2 (topology) -> 3 (topo) ->
        3b (seg_elev) -> 4 (landcover) -> 5 (soils) -> 6 (waterbody) ->
        8 (lookups) -> 12 (routing) -> 9 (soltab) -> 10 (PET) ->
        11 (transp) -> 13 (defaults) -> 14 (calibration) -> 7 (forcing) ->
        overrides.

        Step 7 (forcing) runs late because it has no downstream
        dependencies within the static parameter DAG.
        """
        sir = context.sir
        id_field = context.fabric_id_field
        fabric = context.fabric

        # Derive nhru count and IDs from fabric (authoritative source).
        # Fall back to SIR first variable length when no fabric is provided
        # (e.g., library API use without a fabric file).
        if fabric is not None and id_field in fabric.columns:
            nhru = len(fabric)
            hru_ids = fabric[id_field].values
        elif sir.data_vars:
            first_var = sir.data_vars[0]
            first_da = sir[first_var]
            nhru = len(first_da)
            # Try to recover HRU IDs from the SIR variable's coordinates.
            if id_field in first_da.dims and id_field in first_da.coords:
                hru_ids = first_da.coords[id_field].values
            else:
                hru_ids = None
        else:
            raise ValueError(
                f"Cannot determine HRU count: fabric is None or missing "
                f"'{id_field}' column, and SIR contains no static variables. "
                f"Provide a fabric GeoDataFrame with an '{id_field}' column."
            )

        ds = xr.Dataset()

        # Carry HRU coordinates so derived params retain stable indexing
        if hru_ids is not None:
            ds = ds.assign_coords(nhru=hru_ids)

        # Step 1: Geometry (hru_area, hru_lat)
        ds = self._derive_geometry(context, ds)

        # Step 2: Topology (tosegment, hru_segment, seg_length)
        ds = self._derive_topology(context, ds)

        # Step 3: Topographic parameters (hru_elev, hru_slope, hru_aspect)
        ds = self._derive_topography(context, ds)

        # Step 3b: Segment elevation (InterpGen + 3DEP DEM)
        ds = self._derive_segment_elevation(context, ds)

        # Step 4: Land cover parameters (cov_type, covden_sum, hru_percent_imperv)
        ds = self._derive_landcover(context, ds)

        # Step 5: Soils parameters (soil_type, soil_moist_max, soil_rechr_max_frac)
        ds = self._derive_soils(context, ds)

        # Step 6: Waterbody overlay (dprst_frac, hru_type)
        ds = self._derive_waterbody(context, ds)

        # Step 8: Lookup table application
        ds = self._apply_lookup_tables(context, ds)

        # Step 12: Routing parameters
        ds = self._derive_routing(context, ds)

        # Step 9: Solar radiation tables (soltab)
        ds = self._derive_soltab(context, ds)

        # Compute monthly climate normals once for steps 10 and 11
        normals = self._compute_monthly_normals(context)

        # Step 10: PET coefficients (Jensen-Haise)
        ds = self._derive_pet_coefficients(ds, normals)

        # Step 11: Transpiration timing (frost-free period)
        ds = self._derive_transp_timing(ds, normals)

        # Step 13: Defaults and initial conditions
        ds = self._apply_defaults(ds, nhru)

        # Step 14: Calibration seeds
        ds = self._derive_calibration_seeds(context, ds)

        # Step 7: Forcing (temporal merge — runs late, no downstream deps)
        ds = self._derive_forcing(context, ds)

        # Apply parameter overrides last
        overrides = context.config.get("parameter_overrides", {})
        if isinstance(overrides, dict) and "values" in overrides:
            overrides = overrides["values"]
        if overrides:
            ds = self._apply_overrides(ds, overrides)

        return ds

    # ------------------------------------------------------------------
    # Step 1: Geometry extraction
    # ------------------------------------------------------------------

    def _derive_geometry(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Compute HRU area, centroid lat/lon from the target fabric (step 1).

        Derive ``hru_area`` (acres), ``hru_lat`` (decimal degrees), and
        ``hru_lon`` (decimal degrees) from the fabric GeoDataFrame geometry.
        Area is computed in EPSG:5070 (NAD83 CONUS Albers equal-area) and
        converted from m² to acres.  Latitude and longitude are extracted
        Latitude and longitude are extracted from centroids computed in
        EPSG:5070 (equal-area) and reprojected to EPSG:4326 (WGS84) for
        accurate positions.

        Falls back to SIR variables ``hru_area_m2``, ``hru_lat``, and
        ``hru_lon`` when fabric is ``None`` or lacks the ``id_field``
        column.  Each variable is loaded only if present in the SIR;
        missing variables are skipped.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing ``sir``, ``fabric``, and
            ``fabric_id_field``.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with ``hru_area`` (acres), ``hru_lat``
            (decimal degrees), and ``hru_lon`` (decimal degrees)
            added on the ``nhru`` dimension.

        Notes
        -----
        Unit conversions: m² -> acres (factor: 1 acre = 4046.8564224 m²).
        The equal-area projection ensures accurate area computation for
        continental US fabrics; other regions may need a different CRS.
        """
        sir = ctx.sir
        fabric = ctx.fabric
        id_field = ctx.fabric_id_field

        if fabric is not None and id_field in fabric.columns:
            fab = fabric

            # Area via equal-area projection (EPSG:5070 = CONUS Albers)
            fab_5070 = fab.to_crs(epsg=5070)
            area_m2 = fab_5070.geometry.area.values
            ds["hru_area"] = xr.DataArray(
                convert(area_m2, "m2", "acres"),
                dims="nhru",
                attrs={"units": "acres", "long_name": "Area of HRU"},
            )

            # Latitude from WGS84 centroids (compute in projected CRS, reproject)
            centroids_5070 = fab_5070.geometry.centroid
            centroids_4326 = gpd.GeoSeries(centroids_5070, crs="EPSG:5070").to_crs(epsg=4326)
            lats = centroids_4326.y.values
            ds["hru_lat"] = xr.DataArray(
                lats,
                dims="nhru",
                attrs={"units": "decimal_degrees", "long_name": "Latitude of HRU centroid"},
            )
            lons = centroids_4326.x.values
            ds["hru_lon"] = xr.DataArray(
                lons,
                dims="nhru",
                attrs={"units": "decimal_degrees", "long_name": "Longitude of HRU centroid"},
            )
        else:
            # Fallback to SIR-based geometry
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
            if "hru_lon" in sir:
                ds["hru_lon"] = xr.DataArray(
                    sir["hru_lon"].values,
                    dims="nhru",
                    attrs={"units": "decimal_degrees", "long_name": "Longitude of HRU centroid"},
                )
        return ds

    # ------------------------------------------------------------------
    # Step 2: Topology extraction
    # ------------------------------------------------------------------

    def _derive_topology(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Extract routing topology from fabric and segment GeoDataFrames (step 2).

        Read ``tosegment``, ``tosegment_nhm``, ``hru_segment``,
        ``hru_segment_nhm``, ``seg_length``, ``seg_lat``, ``nhm_id``,
        and ``nhm_seg`` from the Geospatial Fabric GeoDataFrames.
        These define the stream-segment routing network
        and HRU-to-segment flow contributions used by PRMS for Muskingum
        routing.

        Segment lengths are computed geodesically (WGS84 ellipsoid) from
        line geometries when a ``seg_length`` column is not already present,
        using ``pyproj.Geod.geometry_length``.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context with ``fabric``, ``segments``,
            ``fabric_id_field``, and ``segment_id_field``.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with the following variables added (returns ``ds``
            unchanged if ``fabric`` or ``segments`` is ``None``):

            - ``tosegment`` : dimensionless index on ``nsegment``
            - ``tosegment_nhm`` : segment ID on ``nsegment``
            - ``hru_segment`` : dimensionless index on ``nhru``
            - ``seg_length`` : meters on ``nsegment``
            - ``nhm_id`` : HRU identifier on ``nhru``
            - ``nhm_seg`` : segment identifier on ``nsegment``
            - ``hru_segment_nhm`` : segment ID for each HRU on ``nhru``
            - ``seg_lat`` : decimal degrees on ``nsegment``

        Raises
        ------
        ValueError
            If ``tosegment`` column is missing from segments, ``hru_segment``
            column is missing from fabric, tosegment contains self-loops or
            out-of-range values, or no outlet segments (tosegment == 0) exist.
        KeyError
            If an explicitly configured ``segment_id_field`` is not
            found in the segments GeoDataFrame columns.  When the
            default field (``"nhm_seg"``) is absent, a warning is
            logged and sequential IDs (1..nseg) are used as fallback.

        Notes
        -----
        Topology is model-specific and comes directly from the fabric
        GeoDataFrames --- hydro-param does not normalize between
        topology conventions.  The ``tosegment`` array uses 1-based
        indexing with 0 indicating an outlet segment.

        ``nhm_id`` and ``nhm_seg`` are identity copies from the fabric
        columns named by the config fields ``id_field`` and
        ``segment_id_field`` respectively.  Output parameter names are
        always ``nhm_id`` and ``nhm_seg`` (pywatershed convention)
        regardless of the source column name.

        ``tosegment_nhm`` and ``hru_segment_nhm`` are derived by
        mapping the 1-based segment indices (``tosegment``,
        ``hru_segment``) to the corresponding segment IDs from
        ``segment_id_field``.
        """
        fabric = ctx.fabric
        segments = ctx.segments
        if fabric is None or segments is None:
            return ds

        id_field = ctx.fabric_id_field
        segment_id_field = ctx.segment_id_field or "nhm_seg"

        nseg = len(segments)

        # Add nsegment coordinate from segment IDs
        if segment_id_field in segments.columns:
            seg_ids = segments[segment_id_field].values
        elif ctx.segment_id_field is not None:
            # User explicitly configured a field that doesn't exist — raise
            raise KeyError(
                f"segment_id_field '{segment_id_field}' not found in segments columns. "
                f"Available columns: {sorted(segments.columns.tolist())}"
            )
        else:
            # Default field not found — fall back with warning
            logger.warning(
                "Default segment_id_field '%s' not found in segments columns %s; "
                "using sequential IDs (1..%d)",
                segment_id_field,
                sorted(segments.columns.tolist()),
                len(segments),
            )
            seg_ids = np.arange(1, nseg + 1)
        ds = ds.assign_coords(nsegment=seg_ids)

        # --- tosegment ---
        if "tosegment" not in segments.columns:
            raise ValueError("segments GeoDataFrame missing required 'tosegment' column")
        tosegment = segments["tosegment"].values.astype(np.int64)
        self._validate_tosegment(tosegment, nseg)
        ds["tosegment"] = xr.DataArray(
            tosegment,
            dims="nsegment",
            attrs={
                "units": "none",
                "long_name": "Index of downstream segment (0=outlet)",
            },
        )

        # --- tosegment_nhm: map local indices to NHM segment IDs ---
        # tosegment values are 1-based indices into the segment array;
        # 0 means outlet (no downstream segment).
        toseg_nhm = np.where(tosegment > 0, seg_ids[tosegment.astype(int) - 1], 0)
        ds["tosegment_nhm"] = xr.DataArray(
            toseg_nhm,
            dims="nsegment",
            attrs={
                "units": "none",
                "long_name": "NHM downstream segment ID (0=outlet)",
            },
        )

        # --- hru_segment ---
        if "hru_segment" not in fabric.columns:
            raise ValueError("fabric GeoDataFrame missing required 'hru_segment' column")

        # Align fabric rows to ds.coords['nhru'] via id_field
        if "nhru" in ds.coords and id_field in fabric.columns:
            hru_ids = ds.coords["nhru"].values
            fabric_indexed = fabric.set_index(id_field)
            if fabric_indexed.index.has_duplicates:
                raise ValueError(f"Duplicate HRU IDs in fabric column '{id_field}'")
            missing = np.setdiff1d(hru_ids, np.asarray(fabric_indexed.index))
            if missing.size > 0:
                raise ValueError(
                    f"HRU IDs in dataset missing from fabric '{id_field}': {missing.tolist()}"
                )
            hru_segment = fabric_indexed.loc[hru_ids, "hru_segment"].values.astype(np.int64)
        else:
            hru_segment = fabric["hru_segment"].values.astype(np.int64)

        self._validate_hru_segment(hru_segment, nseg)
        ds["hru_segment"] = xr.DataArray(
            hru_segment,
            dims="nhru",
            attrs={
                "units": "none",
                "long_name": "Index of segment to which HRU contributes flow",
            },
        )

        # --- seg_length ---
        seg_length = self._compute_seg_length(segments)
        ds["seg_length"] = xr.DataArray(
            seg_length,
            dims="nsegment",
            attrs={
                "units": "meters",
                "long_name": "Length of stream segment",
            },
        )

        # --- nhm_id: HRU identifier from config id_field ---
        if id_field in fabric.columns:
            if "nhru" in ds.coords:
                nhm_id_vals = ds.coords["nhru"].values
            else:
                nhm_id_vals = fabric[id_field].values
            ds["nhm_id"] = xr.DataArray(
                nhm_id_vals,
                dims="nhru",
                attrs={"units": "none", "long_name": "HRU identifier"},
            )

        # --- nhm_seg: segment identifier from config segment_id_field ---
        ds["nhm_seg"] = xr.DataArray(
            seg_ids,
            dims="nsegment",
            attrs={"units": "none", "long_name": "Segment identifier"},
        )

        # --- hru_segment_nhm: map HRU segment index to segment ID ---
        # hru_segment is 1-based index into segments; 0 means no segment.
        hru_seg_nhm = np.where(
            hru_segment > 0,
            seg_ids[hru_segment.astype(int) - 1],
            0,
        )
        ds["hru_segment_nhm"] = xr.DataArray(
            hru_seg_nhm,
            dims="nhru",
            attrs={
                "units": "none",
                "long_name": "Segment identifier for HRU contributing flow",
            },
        )

        # --- seg_lat: segment centroid latitude (WGS84) ---
        if segments.crs is None:
            logger.warning(
                "Segments have no CRS; seg_lat assumes coordinates are geographic (WGS84)"
            )
            segs_4326 = segments
        elif not segments.crs.is_geographic:
            segs_4326 = segments.to_crs(epsg=4326)
        else:
            segs_4326 = segments
        seg_centroids = segs_4326.geometry.centroid
        ds["seg_lat"] = xr.DataArray(
            seg_centroids.y.values,
            dims="nsegment",
            attrs={
                "units": "decimal_degrees",
                "long_name": "Latitude of segment centroid",
            },
        )

        return ds

    @staticmethod
    def _compute_seg_length(segments: gpd.GeoDataFrame) -> np.ndarray:
        """Compute segment lengths from column or geodesic calculation.

        Return segment lengths in meters.  Prefer the ``seg_length`` column
        if present in the GeoDataFrame; otherwise compute geodesic length
        from line geometries using the WGS84 ellipsoid.  Handles LineString
        and MultiLineString geometries.  Projected CRS data is reprojected
        to EPSG:4326 before geodesic computation.

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame with line geometries and optional
            ``seg_length`` column.

        Returns
        -------
        np.ndarray
            Segment lengths in meters, shape ``(nseg,)``.  Empty or null
            geometries yield a length of 0.0.

        Notes
        -----
        PRMS expects ``seg_length`` in meters.  No unit conversion is
        needed because ``pyproj.Geod.geometry_length`` returns meters
        on the WGS84 ellipsoid.
        """
        if "seg_length" in segments.columns:
            return segments["seg_length"].values.astype(np.float64)

        # Reproject to geographic CRS if needed for geodesic calculation
        if segments.crs is not None and not segments.crs.is_geographic:
            segments_geo = segments.to_crs(epsg=4326)
        else:
            segments_geo = segments

        geod = pyproj.Geod(ellps="WGS84")
        lengths = np.empty(len(segments_geo), dtype=np.float64)
        for i, geom in enumerate(segments_geo.geometry):
            if geom is None or geom.is_empty:
                lengths[i] = 0.0
            else:
                lengths[i] = geod.geometry_length(geom)
        return lengths

    @staticmethod
    def _validate_tosegment(tosegment: np.ndarray, nseg: int) -> None:
        """Validate the tosegment routing array for topological consistency.

        Check that no segment routes to itself (self-loop), all values are
        within the valid range ``[0, nseg]``, and at least one outlet
        (value == 0) exists.

        Parameters
        ----------
        tosegment : np.ndarray
            Downstream segment index array, 1-based with 0 = outlet.
        nseg : int
            Total number of segments (maximum valid index).

        Raises
        ------
        ValueError
            If self-loops exist, values are out of range ``[0, nseg]``,
            or no outlet segments (tosegment == 0) are found.
        """
        indices = np.arange(1, nseg + 1)

        # Check for self-loops
        self_loops = np.where(tosegment == indices)[0]
        if len(self_loops) > 0:
            raise ValueError(
                f"tosegment contains self-loops at 1-based indices: {indices[self_loops].tolist()}"
            )

        # Check value range: 0..nseg
        if np.any(tosegment < 0) or np.any(tosegment > nseg):
            bad = tosegment[(tosegment < 0) | (tosegment > nseg)]
            raise ValueError(f"tosegment values out of range [0, {nseg}]: {bad.tolist()}")

        # At least one outlet
        if not np.any(tosegment == 0):
            raise ValueError("No outlets found (tosegment == 0)")

    @staticmethod
    def _validate_hru_segment(hru_segment: np.ndarray, nseg: int) -> None:
        """Validate the hru_segment array for valid segment references.

        Ensure all HRU-to-segment assignments reference existing segments
        (values in ``[0, nseg]``, where 0 means the HRU is unassigned).

        Parameters
        ----------
        hru_segment : np.ndarray
            HRU-to-segment assignment array (1-based segment indices,
            0 = unassigned).
        nseg : int
            Total number of segments (maximum valid index).

        Raises
        ------
        ValueError
            If any values are outside the range ``[0, nseg]``.
        """
        if np.any(hru_segment < 0) or np.any(hru_segment > nseg):
            bad = hru_segment[(hru_segment < 0) | (hru_segment > nseg)]
            raise ValueError(f"hru_segment values out of range [0, {nseg}]: {bad.tolist()}")

    @staticmethod
    def _get_slopes_from_comid(
        segments: gpd.GeoDataFrame,
        vaa: pd.DataFrame,
        comid_col: str,
    ) -> np.ndarray:
        """Look up NHDPlus VAA slope by COMID (direct join).

        Retrieve channel slopes from the NHDPlus Value Added Attributes
        (VAA) table by matching segment COMIDs.  This is the primary
        slope source for step 12 routing coefficient computation.

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame with a COMID column.
        vaa : pd.DataFrame
            NHDPlus VAA table with ``comid`` and ``slope`` columns.
            Slope is in dimensionless rise/run (decimal fraction).
        comid_col : str
            Name of the COMID column in ``segments`` (as returned by
            ``_find_comid_column``).

        Returns
        -------
        np.ndarray
            Slope values (decimal fraction) aligned to segment order.
            Segments with no matching COMID in the VAA get
            ``_FALLBACK_SLOPE`` (1e-4).

        See Also
        --------
        _find_comid_column : Locate the COMID column name.
        _FALLBACK_SLOPE : Default slope for unmatched segments.
        """
        comids = segments[comid_col].values

        vaa_comids = set(vaa["comid"].values)
        vaa_slopes = dict(zip(vaa["comid"].values, vaa["slope"].values, strict=True))

        matched = np.array([c in vaa_comids for c in comids])
        slopes = np.where(
            matched,
            [vaa_slopes.get(c, 0.0) for c in comids],
            _FALLBACK_SLOPE,
        ).astype(np.float64)

        n_missing = int(np.sum(~matched))
        if n_missing > 0:
            logger.warning(
                "%d of %d segments have no matching COMID in VAA; using fallback slope %.1e",
                n_missing,
                len(comids),
                _FALLBACK_SLOPE,
            )
        return slopes

    @staticmethod
    def _get_slopes_spatial_join(
        segments: gpd.GeoDataFrame,
        nhd_flowlines: gpd.GeoDataFrame,
    ) -> np.ndarray:
        """Get slopes via spatial join to NHDPlus flowlines (length-weighted).

        For each segment, find all NHD flowlines within a buffer corridor,
        clip each flowline to the corridor, and return a length-weighted
        mean slope.  This handles GF/PRMS segments that have been
        post-processed from NHD (split at POIs, trimmed to catchment
        boundaries) and therefore lack COMIDs for a direct VAA join.

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame (no COMID column).  Must have line
            geometries and a projected CRS (units in metres).
        nhd_flowlines : gpd.GeoDataFrame
            NHDPlus flowlines with ``slope`` column (dimensionless
            rise/run, decimal fraction) and line geometries.

        Returns
        -------
        np.ndarray
            Length-weighted mean slope per segment.  Segments with no
            NHD flowlines within the buffer receive ``_FALLBACK_SLOPE``.

        Raises
        ------
        ValueError
            If ``segments`` has a geographic (non-projected) CRS.  The
            buffer distance is in metres and requires a projected CRS.

        Notes
        -----
        GF/PRMS segments and NHD flowlines are independently digitized
        representations of the same river network.  They may run parallel
        with offsets of tens of metres and share no vertices.  A raw
        ``gpd.sjoin`` with ``predicate="intersects"`` on the line
        geometries would miss parallel-but-offset flowlines entirely, and
        any crossing matches would produce only Point intersections (zero
        length, unusable as weights).

        The solution is two-fold:

        1. **Candidate finding:** buffer each segment by
           ``_SPATIAL_JOIN_BUFFER_M`` (100 m) to create a polygon
           corridor, then spatial-join NHD flowlines against the
           corridors.  This captures nearby flowlines regardless of
           whether they cross the segment centreline.
        2. **Length weighting:** clip each matched flowline to the
           corridor polygon and use the clipped length as the weight
           for slope averaging.

        CRS alignment is enforced: if the two GeoDataFrames differ in
        CRS, the NHD flowlines are reprojected to match the segments.

        See Also
        --------
        _get_slopes_from_comid : Primary slope lookup via COMID join.
        _FALLBACK_SLOPE : Default slope for unmatched segments.
        _SPATIAL_JOIN_BUFFER_M : Buffer distance for flowline clipping.
        """
        if segments.crs is not None and not segments.crs.is_projected:
            raise ValueError(
                f"_get_slopes_spatial_join requires a projected CRS (units in metres), "
                f"got {segments.crs}. Reproject segments to e.g. EPSG:5070."
            )

        # Ensure same CRS
        if segments.crs != nhd_flowlines.crs:
            nhd_flowlines = nhd_flowlines.to_crs(segments.crs)

        # Reset index to use positional alignment
        segs = segments.reset_index(drop=True)
        nhd = nhd_flowlines.reset_index(drop=True)

        # Buffer segments into polygon corridors for candidate finding.
        # This captures NHD flowlines that run parallel but offset from
        # the segment centreline (see docstring Notes).
        seg_buffers = segs.copy()
        seg_buffers["geometry"] = segs.geometry.buffer(_SPATIAL_JOIN_BUFFER_M)

        # Spatial join — find NHD flowlines within each segment corridor
        joined = gpd.sjoin(seg_buffers, nhd, how="left", predicate="intersects")

        slopes = np.full(len(segs), _FALLBACK_SLOPE, dtype=np.float64)
        matched = np.zeros(len(segs), dtype=bool)

        if "slope" not in joined.columns:
            logger.warning(
                "NHD flowlines lack 'slope' column after spatial join; "
                "using fallback slope %.1e for all %d segments",
                _FALLBACK_SLOPE,
                len(segs),
            )
            return slopes

        for seg_idx in range(len(segs)):
            matches = joined[joined.index == seg_idx]
            matches = matches.dropna(subset=["slope"])
            if matches.empty:
                continue

            # Clip each matched flowline to the buffer corridor and use
            # clipped length as the weight for slope averaging.
            seg_buffer = seg_buffers.geometry.iloc[seg_idx]
            weights: list[float] = []
            match_slopes: list[float] = []
            for _, row in matches.iterrows():
                nhd_idx = int(row["index_right"])
                nhd_geom = nhd.geometry.iloc[nhd_idx]
                try:
                    clipped = nhd_geom.intersection(seg_buffer)
                except GEOSException as exc:
                    logger.warning(
                        "Geometry intersection failed for segment %d with NHD index %d: %s; "
                        "skipping",
                        seg_idx,
                        nhd_idx,
                        exc,
                    )
                    continue
                if clipped.is_empty:
                    continue
                weight = clipped.length
                if weight > 0:
                    weights.append(weight)
                    match_slopes.append(row["slope"])

            if weights:
                weights_arr = np.array(weights)
                match_slopes_arr = np.array(match_slopes)
                slopes[seg_idx] = np.average(match_slopes_arr, weights=weights_arr)
                matched[seg_idx] = True

        n_fallback = int(np.sum(~matched))
        if n_fallback > 0:
            logger.warning(
                "%d of %d segments have no NHDPlus flowlines within %d m buffer; "
                "using fallback slope %.1e",
                n_fallback,
                len(segs),
                int(_SPATIAL_JOIN_BUFFER_M),
                _FALLBACK_SLOPE,
            )
        return slopes

    @staticmethod
    def _get_cum_area_from_comid(
        segments: gpd.GeoDataFrame,
        vaa: pd.DataFrame,
        comid_col: str,
    ) -> np.ndarray:
        """Look up cumulative drainage area from VAA by COMID.

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame with a COMID column.
        vaa : pd.DataFrame
            VAA table with ``comid`` and ``totdasqkm`` columns.
        comid_col : str
            Name of the COMID column in *segments*.

        Returns
        -------
        np.ndarray
            Cumulative drainage area in km² per segment.  Unmatched
            segments get 0.0.

        See Also
        --------
        _get_slopes_from_comid : Analogous slope lookup by COMID.
        """
        comids = segments[comid_col].values
        vaa_comid_set = set(vaa["comid"].values)
        vaa_areas = dict(zip(vaa["comid"].values, vaa["totdasqkm"].values, strict=True))

        matched = np.array([c in vaa_comid_set for c in comids])
        n_missing = int(np.sum(~matched))
        if n_missing > 0:
            logger.warning(
                "%d of %d segments have no matching COMID in VAA for "
                "cumulative area; using 0.0 km²",
                n_missing,
                len(comids),
            )
        return np.array([vaa_areas.get(c, 0.0) for c in comids], dtype=np.float64)

    @staticmethod
    def _get_cum_area_spatial_join(
        segments: gpd.GeoDataFrame,
        nhd_flowlines: gpd.GeoDataFrame | None,
        vaa: pd.DataFrame,
    ) -> np.ndarray:
        """Get cumulative area via spatial join to NHDPlus flowlines.

        For each segment, find all NHD flowlines that intersect a
        ``_SPATIAL_JOIN_BUFFER_M`` (100 m) buffer corridor around the
        segment, then take the maximum ``totdasqkm`` among matched
        flowlines.  The largest matched flowline's cumulative area is
        the best proxy for the segment's upstream drainage (unlike
        slopes, which use length-weighted averaging, cumulative area
        is a property of the most downstream matched flowline).

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame (no COMID column).  Must have a
            projected CRS (units in metres) for correct buffering.
        nhd_flowlines : gpd.GeoDataFrame or None
            NHDPlus flowlines with COMID.  If ``None``, returns zeros
            with a warning.
        vaa : pd.DataFrame
            VAA table with ``comid`` and ``totdasqkm`` columns.

        Returns
        -------
        np.ndarray
            Cumulative drainage area in km² per segment.  Unmatched
            segments get 0.0.

        Raises
        ------
        ValueError
            If ``segments`` has a geographic (non-projected) CRS.  The
            buffer distance is in metres and requires a projected CRS.

        See Also
        --------
        _get_slopes_spatial_join : Analogous slope extraction via spatial join.
        """
        nseg = len(segments)
        if nhd_flowlines is None:
            logger.warning(
                "NHD flowlines unavailable; returning 0.0 km² for all %d segments",
                nseg,
            )
            return np.zeros(nseg, dtype=np.float64)

        if segments.crs is not None and not segments.crs.is_projected:
            raise ValueError(
                f"_get_cum_area_spatial_join requires a projected CRS "
                f"(units in metres), got {segments.crs}. "
                f"Reproject segments to e.g. EPSG:5070."
            )

        # Ensure same CRS
        if segments.crs != nhd_flowlines.crs:
            nhd_flowlines = nhd_flowlines.to_crs(segments.crs)

        segs = segments.reset_index(drop=True)
        nhd = nhd_flowlines.reset_index(drop=True)

        # Buffer segments into corridors
        seg_buffers = segs.copy()
        seg_buffers["geometry"] = segs.geometry.buffer(_SPATIAL_JOIN_BUFFER_M)

        joined = gpd.sjoin(seg_buffers, nhd, how="left", predicate="intersects")

        # Find COMID column in flowlines
        fl_comid_col = next((c for c in nhd.columns if c.lower() == "comid"), None)
        if fl_comid_col is None:
            logger.warning(
                "NHD flowlines lack COMID column; returning 0.0 km² for all %d segments",
                nseg,
            )
            return np.zeros(nseg, dtype=np.float64)

        # Build COMID -> totdasqkm lookup
        vaa_areas = dict(zip(vaa["comid"].values, vaa["totdasqkm"].values, strict=True))

        # For each segment, take max totdasqkm among matched flowlines
        cum_area = np.zeros(nseg, dtype=np.float64)
        for seg_idx in range(nseg):
            matches = joined[joined.index == seg_idx]
            if matches.empty or matches[fl_comid_col].isna().all():
                continue
            areas = [vaa_areas.get(int(c), 0.0) for c in matches[fl_comid_col].dropna()]
            if areas:
                cum_area[seg_idx] = max(areas)

        return cum_area

    @staticmethod
    def _compute_k_coef(
        slopes: np.ndarray,
        seg_lengths_m: np.ndarray,
    ) -> np.ndarray:
        """Compute Muskingum K_coef via Manning's equation.

        Velocity is computed from Manning's formula in PRMS internal units
        (feet, hours):

            velocity = (1 / n) * sqrt(slope) * depth^(2/3) * 3600

        Travel time is then:

            K_coef = seg_length_ft / velocity

        Parameters
        ----------
        slopes : np.ndarray
            Channel slope (m/m) per segment.
        seg_lengths_m : np.ndarray
            Segment lengths in meters.

        Returns
        -------
        np.ndarray
            K_coef in hours, clamped to ``[_K_COEF_MIN, _K_COEF_MAX]``.
            Zero-length segments receive ``_DEFAULT_K_COEF``.

        Notes
        -----
        Manning's n defaults to 0.04 (natural channel) and bankfull depth
        to 1.0 ft.  Both are module-level constants that can be refined
        in future work (e.g., BANKFULL_CONUS dataset for per-segment depth).

        The formula follows pywatershed's ``muskingum_mann`` implementation,
        which uses the SI form of Manning's equation (1/n rather than the
        US customary 1.49/n).  Because the default depth is 1.0, the
        depth term evaluates to unity and the unit system choice is
        numerically irrelevant.  If ``_DEFAULT_DEPTH_FT`` is changed to a
        non-unity value, the formula must be reconciled to a consistent
        unit system.

        References
        ----------
        Hay, L.E., et al. (2023). USGS TM 6-B10, muskingum_mann module.
        """
        k_coef = np.full(len(slopes), _DEFAULT_K_COEF, dtype=np.float64)

        # Mask valid segments (nonzero length)
        valid = seg_lengths_m > 0
        if not np.any(valid):
            return k_coef

        # Clamp slopes
        s = np.clip(slopes[valid], _MIN_SLOPE, None)

        # Manning's equation: velocity in ft/hr
        velocity = (1.0 / _MANNING_N) * np.sqrt(s) * (_DEFAULT_DEPTH_FT ** (2.0 / 3.0)) * 3600.0

        # Convert segment length from meters to feet
        seg_length_ft = seg_lengths_m[valid] * 3.28084

        # K = length / velocity (hours)
        k_coef[valid] = seg_length_ft / velocity

        # Clamp to valid range
        k_coef = np.clip(k_coef, _K_COEF_MIN, _K_COEF_MAX)

        return k_coef

    @staticmethod
    def _find_comid_column(segments: gpd.GeoDataFrame) -> str | None:
        """Find the COMID column name in a segment GeoDataFrame.

        When segments originate from NHDPlus they include a ``comid``
        (Common Identifier) column that can be used for a direct join
        against the NHDPlus VAA table.  Segments from the Geospatial
        Fabric (GF) lack this column and require a spatial join instead.

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame to inspect.

        Returns
        -------
        str or None
            The actual column name (preserving original case) if found,
            or ``None`` if no COMID column exists.  The check is fully
            case-insensitive.

        See Also
        --------
        _get_slopes_from_comid : Slope lookup when COMIDs are present.
        _get_slopes_spatial_join : Spatial join fallback for GF segments.
        """
        for col in segments.columns:
            if col.lower() == "comid":
                return col
        return None

    @staticmethod
    def _fetch_vaa() -> pd.DataFrame | None:
        """Fetch the NHDPlus Value Added Attributes (VAA) table.

        Downloads a cached ~245 MB parquet file on first call, then
        returns it from the local pynhd cache on subsequent calls.

        Returns
        -------
        pd.DataFrame or None
            VAA table with ``comid``, ``slope``, and (when available)
            ``totdasqkm`` columns.  NaN slopes and
            ``_NHD_MISSING_SLOPE_SENTINEL`` values are removed.
            Returns ``None`` if the download fails.

        References
        ----------
        McKay, L., et al. (2012). NHDPlus Version 2: User Guide.
            https://www.epa.gov/waterdata/nhdplus-national-data
        """
        try:
            vaa = pynhd.nhdplus_vaa()
            cols = ["comid", "slope"]
            if "totdasqkm" in vaa.columns:
                cols.append("totdasqkm")
            result = vaa[cols].dropna(subset=["slope"])
            # NHDPlus uses -9998 as a sentinel for missing slope values.
            result = result[result["slope"] != _NHD_MISSING_SLOPE_SENTINEL]
            return result
        except (OSError, KeyError) as exc:
            logger.error(
                "Failed to fetch or parse NHDPlus VAA table: %s",
                exc,
            )
            return None

    @staticmethod
    def _fetch_nhd_flowlines(
        segments: gpd.GeoDataFrame,
        vaa: pd.DataFrame,
    ) -> gpd.GeoDataFrame | None:
        """Fetch NHDPlus flowline geometries and join VAA slopes.

        Queries the NHDPlus flowline network for the bounding box of
        the given segments, then merges VAA slope values onto the
        flowlines by COMID so they are ready for spatial join.

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame used to determine the bounding box.
        vaa : pd.DataFrame
            VAA table with ``comid`` and ``slope`` columns (from
            ``_fetch_vaa``).

        Returns
        -------
        gpd.GeoDataFrame or None
            NHDPlus flowlines with a ``slope`` column, or ``None``
            if the fetch fails.
        """
        try:
            bbox = segments.to_crs("EPSG:4326").total_bounds
            wd = pynhd.WaterData("nhdflowline_network")
            flowlines = wd.bybox(tuple(bbox))

            # Join VAA slopes onto flowlines by COMID
            fl_comid_col = next(
                (c for c in flowlines.columns if c.lower() == "comid"),
                None,
            )
            if fl_comid_col is None:
                logger.warning("NHDPlus flowlines have no COMID column; cannot join VAA slopes")
                return None

            # Drop any existing slope column to avoid suffix collision
            # (the NHDPlus service may include VAA attributes)
            if "slope" in flowlines.columns:
                flowlines = gpd.GeoDataFrame(flowlines.drop(columns=["slope"]))

            merged = flowlines.merge(
                vaa[["comid", "slope"]],
                left_on=fl_comid_col,
                right_on="comid",
                how="left",
            )
            return gpd.GeoDataFrame(merged)

        except (OSError, ValueError) as exc:
            logger.warning(
                "Failed to fetch NHDPlus flowlines for spatial join: %s",
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Step 12: Routing coefficients
    # ------------------------------------------------------------------

    def _derive_routing(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Derive Muskingum routing parameters from channel geometry (step 12).

        Compute ``K_coef`` (travel time) via Manning's equation using
        NHDPlus VAA slopes, and assign ``x_coef``, ``seg_slope``,
        ``seg_cum_area``, ``segment_type``, and ``obsin_segment``.

        Supports two segment types:

        - **NHD segments** (have ``comid``/``COMID`` column): direct
          COMID-to-VAA slope lookup.
        - **GF/PRMS segments** (no COMID): spatial join to NHDPlus
          flowlines with length-weighted slope averaging.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing ``segments`` GeoDataFrame.
        ds : xr.Dataset
            In-progress parameter dataset.  Should contain ``seg_length``
            on ``nsegment`` from Step 2; if absent, all ``K_coef`` values
            will be ``_DEFAULT_K_COEF``.

        Returns
        -------
        xr.Dataset
            Dataset with ``K_coef`` (hours), ``x_coef`` (dimensionless),
            ``seg_slope`` (m/m), ``seg_cum_area`` (acres, when VAA
            ``totdasqkm`` is available), ``segment_type`` (integer),
            and ``obsin_segment`` (integer) added on ``nsegment``.
            Returns ``ds`` unchanged if ``ctx.segments`` is ``None``.

        Notes
        -----
        Step 12 of the derivation DAG.  Runs after Step 8 (lookups) and
        before Step 9 (soltab).  See the design document at
        ``docs/plans/2026-02-26-step12-routing-design.md``.

        References
        ----------
        Hay, L.E., et al. 2023. Parameter estimation at the CONUS scale.
            USGS TM 6-B10.
        """
        segments = ctx.segments
        if segments is None:
            logger.warning("No segments provided; skipping routing derivation")
            return ds

        nseg = len(segments)
        if "nsegment" not in ds.dims:
            logger.warning("nsegment dimension not in dataset; skipping routing")
            return ds

        # --- Fetch NHDPlus slopes ---
        comid_col = self._find_comid_column(segments)
        vaa = self._fetch_vaa()
        nhd_flowlines = None  # populated by GF spatial-join branch below

        if vaa is not None and comid_col is not None:
            # NHD path: direct COMID lookup (no flowline fetch needed)
            slopes = self._get_slopes_from_comid(segments, vaa, comid_col)
        elif vaa is not None:
            # GF path: fetch flowlines for spatial join
            nhd_flowlines = self._fetch_nhd_flowlines(segments, vaa)
            if nhd_flowlines is not None:
                slopes = self._get_slopes_spatial_join(segments, nhd_flowlines)
            else:
                logger.warning(
                    "NHDPlus flowlines unavailable; using fallback slope %.1e for all %d segments",
                    _FALLBACK_SLOPE,
                    nseg,
                )
                slopes = np.full(nseg, _FALLBACK_SLOPE, dtype=np.float64)
        else:
            # VAA fetch failed — use fallback slopes everywhere
            logger.warning(
                "NHDPlus data unavailable; using fallback slope %.1e for all %d segments",
                _FALLBACK_SLOPE,
                nseg,
            )
            slopes = np.full(nseg, _FALLBACK_SLOPE, dtype=np.float64)

        # --- seg_slope ---
        ds["seg_slope"] = xr.DataArray(
            slopes,
            dims="nsegment",
            attrs={"units": "m/m", "long_name": "Channel slope"},
        )

        # --- seg_cum_area: cumulative drainage area from VAA ---
        if vaa is not None and "totdasqkm" in vaa.columns:
            if comid_col is not None:
                cum_area = self._get_cum_area_from_comid(segments, vaa, comid_col)
            else:
                cum_area = self._get_cum_area_spatial_join(segments, nhd_flowlines, vaa)
            ds["seg_cum_area"] = xr.DataArray(
                cum_area * _KM2_TO_ACRES,
                dims="nsegment",
                attrs={
                    "units": "acres",
                    "long_name": "Cumulative drainage area of segment",
                },
            )
        else:
            logger.warning("VAA totdasqkm unavailable; skipping seg_cum_area")

        # --- K_coef ---
        if "seg_length" in ds:
            seg_lengths = ds["seg_length"].values
        else:
            logger.warning(
                "seg_length not found in dataset; K_coef will use default "
                "%.1f hours for all %d segments. Ensure Step 2 (topology) "
                "completed successfully.",
                _DEFAULT_K_COEF,
                nseg,
            )
            seg_lengths = np.zeros(nseg)
        k_coef = self._compute_k_coef(slopes, seg_lengths)

        # --- segment_type ---
        if "segment_type" in segments.columns:
            seg_type = segments["segment_type"].values.astype(np.int32)
        else:
            seg_type = np.full(nseg, _CHANNEL_SEGMENT_TYPE, dtype=np.int32)

        # Force lake segments to max K_coef
        lake_mask = seg_type == _LAKE_SEGMENT_TYPE
        k_coef[lake_mask] = _LAKE_K_COEF

        ds["K_coef"] = xr.DataArray(
            k_coef,
            dims="nsegment",
            attrs={"units": "hr", "long_name": "Muskingum storage time coefficient"},
        )

        # --- x_coef ---
        ds["x_coef"] = xr.DataArray(
            np.full(nseg, _DEFAULT_X_COEF, dtype=np.float64),
            dims="nsegment",
            attrs={"units": "none", "long_name": "Muskingum routing weighting factor"},
        )

        # --- segment_type ---
        ds["segment_type"] = xr.DataArray(
            seg_type,
            dims="nsegment",
            attrs={"units": "none", "long_name": "Segment type (0=channel, 1=lake)"},
        )

        # --- obsin_segment ---
        ds["obsin_segment"] = xr.DataArray(
            np.zeros(nseg, dtype=np.int32),
            dims="nsegment",
            attrs={"units": "none", "long_name": "Observed inflow segment (0=none)"},
        )

        return ds

    # ------------------------------------------------------------------
    # Step 3: Topographic parameters
    # ------------------------------------------------------------------

    def _derive_topography(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Convert DEM zonal statistics to PRMS topographic parameters (step 3).

        Transform 3DEP-derived zonal statistics to PRMS conventions:

        - ``elevation_m_mean`` (meters) -> ``hru_elev`` (meters, preserved)
        - ``slope_deg_mean`` (degrees) -> ``hru_slope`` (decimal fraction = tan(slope))
        - ``sin_aspect_mean`` + ``cos_aspect_mean`` -> ``hru_aspect`` (degrees,
          circular mean via atan2)
        - ``aspect_deg_mean`` -> ``hru_aspect`` (degrees, legacy fallback)

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing the SIR dataset.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with ``hru_elev`` (meters), ``hru_slope`` (decimal
            fraction), and ``hru_aspect`` (degrees) on the ``nhru``
            dimension.  Only variables with corresponding SIR input are
            added.

        Notes
        -----
        Elevation is kept in meters (``elev_units=1``).  pywatershed v2
        does not use ``elev_units`` internally — the reference DRB data
        uses meters throughout.

        PRMS slope is rise/run (dimensionless), not an angle.  The
        conversion is ``tan(slope_rad)`` where ``slope_rad`` is the
        slope angle in radians.

        Aspect uses circular mean via sin/cos decomposition: the pipeline
        computes per-pixel ``sin(aspect)`` and ``cos(aspect)``, then
        arithmetic zonal mean is correct for those components.  The
        derivation recombines with ``atan2(mean_sin, mean_cos)`` to get
        a proper circular mean.  Falls back to arithmetic mean of raw
        aspect if only ``aspect_deg_mean`` is available (legacy SIR).
        """
        sir = ctx.sir
        if "elevation_m_mean" in sir:
            ds["hru_elev"] = xr.DataArray(
                sir["elevation_m_mean"].values.astype(np.float64),
                dims="nhru",
                attrs={"units": "meters", "long_name": "Mean HRU elevation"},
            )

        if "slope_deg_mean" in sir:
            # SIR slope is in degrees; PRMS wants decimal fraction (rise/run)
            slope_rad = convert(sir["slope_deg_mean"].values, "deg", "rad")
            ds["hru_slope"] = xr.DataArray(
                np.tan(slope_rad),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "Mean HRU slope"},
            )

        # Aspect: prefer circular mean via sin/cos decomposition
        if "sin_aspect_mean" in sir and "cos_aspect_mean" in sir:
            sin_mean = sir["sin_aspect_mean"].values.astype(np.float64)
            cos_mean = sir["cos_aspect_mean"].values.astype(np.float64)
            aspect = (np.degrees(np.arctan2(sin_mean, cos_mean)) + 360) % 360
            ds["hru_aspect"] = xr.DataArray(
                aspect,
                dims="nhru",
                attrs={
                    "units": "degrees",
                    "long_name": "Mean HRU aspect (circular mean)",
                },
            )
        elif "aspect_deg_mean" in sir:
            logger.warning(
                "Using arithmetic mean of aspect (legacy SIR). Circular mean via "
                "sin_aspect_mean/cos_aspect_mean is preferred. Consider regenerating "
                "SIR with the sin_aspect and cos_aspect terrain derivations."
            )
            ds["hru_aspect"] = xr.DataArray(
                sir["aspect_deg_mean"].values.astype(np.float64),
                dims="nhru",
                attrs={"units": "degrees", "long_name": "Mean HRU aspect"},
            )

        return ds

    # ------------------------------------------------------------------
    # Step 3b: Segment elevation (InterpGen + DEM)
    # ------------------------------------------------------------------

    def _derive_segment_elevation(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Derive mean segment elevation from a DEM raster (step 3b).

        Sample a 3DEP DEM along each segment polyline using gdptools
        ``InterpGen`` (grid-to-line interpolation) and compute the mean
        elevation.  Convert from meters to feet.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context.  Must have ``segments`` and a
            ``dem_path`` key in ``config`` pointing to a local
            GeoTIFF DEM.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with ``seg_elev`` (feet) on ``nsegment``, or
            ``ds`` unchanged if DEM path is unavailable or segments
            are ``None``.

        Notes
        -----
        Step 3b of the derivation DAG (runs after step 3, before step 4).
        The DEM path is provided via ``config["dem_path"]``.  When absent,
        this step is skipped with a debug log message.

        ``InterpGen`` samples the raster at 50 m intervals along each
        segment polyline and returns per-segment statistics.  The
        ``"mean"`` statistic gives the average elevation along the
        stream channel.

        Unit conversion: DEM values are assumed to be in meters; the
        output ``seg_elev`` is in feet (1 m = 3.28084 ft).  Note that
        ``hru_elev`` (step 3) is kept in meters (``elev_units=1``),
        while ``seg_elev`` follows the PRMS convention of feet for
        segment-level parameters.

        The DEM CRS is read from the GeoTIFF file metadata via
        rioxarray.  3DEP DEMs are natively EPSG:4269 (NAD83).

        References
        ----------
        gdptools InterpGen: grid-to-line interpolation for polyline
        geometries.
        """
        segments = ctx.segments
        dem_path_str = ctx.config.get("dem_path")

        if segments is None or dem_path_str is None:
            if dem_path_str is None:
                logger.debug("No dem_path in config; skipping seg_elev derivation")
            return ds

        dem_path = Path(dem_path_str)
        if not dem_path.exists():
            logger.warning("DEM path %s does not exist; skipping seg_elev", dem_path)
            return ds

        try:
            from gdptools import InterpGen, UserTiffData
        except ImportError:
            logger.warning("gdptools not available; skipping seg_elev")
            return ds

        segment_id_field = ctx.segment_id_field or "nhm_seg"

        # Prepare segments in geographic CRS for InterpGen
        if segments.crs is not None and not segments.crs.is_geographic:
            segs_geo = segments.to_crs(epsg=4326)
        else:
            segs_geo = segments.copy()

        # Build a target_id column for InterpGen
        if segment_id_field in segs_geo.columns:
            target_id = segment_id_field
        else:
            segs_geo = segs_geo.copy()
            segs_geo["_seg_idx"] = range(len(segs_geo))
            target_id = "_seg_idx"

        try:
            from typing import cast

            import rioxarray as _rio  # noqa: F811

            dem_da = cast(xr.DataArray, _rio.open_rasterio(str(dem_path)))
            source_crs = dem_da.rio.crs.to_epsg() if dem_da.rio.crs else 4326
            dem_da.close()

            user_data = UserTiffData(
                source_ds=str(dem_path),
                source_crs=source_crs,
                source_x_coord="x",
                source_y_coord="y",
                target_gdf=segs_geo,
                target_id=target_id,
            )

            interp = InterpGen(
                user_data,
                pt_spacing=50,
                stat="mean",
                interp_method="linear",
            )

            # calc_interp always returns (stats_df, points_gdf) tuple
            stats_df, _points_gdf = interp.calc_interp()
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning(
                "InterpGen failed for seg_elev: %s; skipping",
                exc,
            )
            return ds

        # Extract mean elevation per segment, maintaining order
        _m_to_ft = 3.28084
        nseg = len(segments)
        seg_elev = np.full(nseg, np.nan, dtype=np.float64)

        # The stats_df has a column named after target_id (e.g. "nhm_seg")
        # and a "mean" column with the interpolated values
        if target_id not in stats_df.columns:
            logger.error(
                "InterpGen stats_df missing expected column '%s' "
                "(found columns: %s); seg_elev will be set to 0.0",
                target_id,
                list(stats_df.columns),
            )
        elif "mean" not in stats_df.columns:
            logger.error(
                "InterpGen stats_df missing 'mean' column "
                "(found columns: %s); seg_elev will be set to 0.0",
                list(stats_df.columns),
            )
        else:
            id_vals = segs_geo[target_id].values
            for i, sid in enumerate(id_vals):
                row = stats_df[stats_df[target_id] == sid]
                if not row.empty:
                    seg_elev[i] = row["mean"].iloc[0] * _m_to_ft

        # Fill NaN with 0 and warn
        nan_count = int(np.isnan(seg_elev).sum())
        if nan_count > 0:
            logger.warning(
                "%d of %d segments have no DEM elevation; using 0.0 feet",
                nan_count,
                nseg,
            )
            seg_elev = np.nan_to_num(seg_elev, nan=0.0)

        ds["seg_elev"] = xr.DataArray(
            seg_elev,
            dims="nsegment",
            attrs={
                "units": "feet",
                "long_name": "Mean segment channel elevation",
            },
        )

        return ds

    # ------------------------------------------------------------------
    # Step 4: Land cover parameters
    # ------------------------------------------------------------------

    def _derive_landcover(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Derive vegetation cover and impervious parameters from NLCD (step 4).

        Reclassify NLCD land cover classes to PRMS vegetation cover type
        (``cov_type``) and derive canopy density (``covden_sum``) and
        impervious fraction (``hru_percent_imperv``).

        Supports four input modes:

        1. **Pre-computed pass-through** (highest priority): when the
           consumer config declares a pre-computed source for ``cov_type``,
           ``covden_sum``, or ``hru_percent_imperv``, the SIR value is
           loaded directly via ``_try_precomputed()`` and no NLCD
           derivation is performed for that parameter.
        2. **Categorical fractions** (preferred NLCD path): SIR contains
           columns like ``lndcov_frac_11``, ``lndcov_frac_21``, etc. from
           normalized categorical zonal output.  Fractions are grouped
           by target category before selecting the majority class to
           avoid split-vote problems.
        3. **Single majority value**: ``land_cover`` or
           ``land_cover_majority`` variable containing the dominant
           NLCD class code per HRU.
        4. **Continuous auxiliary layers**: ``fctimp_pct_mean`` (0--100%)
           for impervious fraction and ``tree_canopy_pct_mean``
           (0--100%) for canopy cover density.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing the SIR dataset and lookup
            tables directory.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with ``cov_type`` (integer PRMS code on ``nhru``),
            ``covden_sum`` (decimal fraction 0--1 on ``nhru``), and
            ``hru_percent_imperv`` (decimal fraction 0--1 on ``nhru``)
            added where input data is available.

        Notes
        -----
        PRMS ``cov_type`` codes: 0 = bare, 1 = grasses, 2 = shrubs,
        3 = deciduous trees, 4 = coniferous trees.  The NLCD-to-PRMS
        mapping is defined in ``nlcd_to_prms_cov_type.yml``.

        When ``tree_canopy_pct_mean`` is unavailable, ``covden_sum`` is
        estimated from ``cov_type`` using a simple lookup (0=0.0,
        1=0.3, 2=0.4, 3=0.7, 4=0.8).

        See Also
        --------
        _extract_nlcd_fractions : Extract class codes and fractions from SIR.
        _compute_grouped_majority : Group fractions by target category.
        """
        sir = ctx.sir

        # --- cov_type: try pre-computed first, then NLCD derivation ---
        cov_type: np.ndarray | None = self._try_precomputed(ctx, "cov_type", categorical=True)

        if cov_type is None:
            # NLCD derivation path
            nlcd_table = self._load_lookup_table(
                "nlcd_to_prms_cov_type", ctx.resolved_lookup_tables_dir
            )
            mapping: dict[int, int] = nlcd_table["mapping"]

            extracted = self._extract_nlcd_fractions(sir, valid_codes=self._VALID_NLCD_CLASSES)
            if extracted is not None:
                class_codes, fractions_list = extracted
                cov_type = self._compute_grouped_majority(class_codes, fractions_list, mapping)
            else:
                for candidate in ("land_cover", "land_cover_majority"):
                    if candidate in sir:
                        nlcd_values_lc = sir[candidate].values.astype(int)
                        cov_type = np.array([mapping.get(int(v), 0) for v in nlcd_values_lc])
                        break

        if cov_type is not None:
            ds["cov_type"] = xr.DataArray(
                cov_type,
                dims="nhru",
                attrs={"units": "integer", "long_name": "Vegetation cover type"},
            )

        # --- covden_sum: try pre-computed, then canopy %, then lookup ---
        covden_sum = self._try_precomputed(ctx, "covden_sum")
        if covden_sum is not None:
            ds["covden_sum"] = xr.DataArray(
                np.clip(covden_sum, 0.0, 1.0),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "Summer vegetation cover density"},
            )
        else:
            canopy_key = sir.find_variable("tree_canopy_pct_mean")
            if canopy_key is not None:
                ds["covden_sum"] = xr.DataArray(
                    np.clip(sir[canopy_key].values / 100.0, 0.0, 1.0),
                    dims="nhru",
                    attrs={
                        "units": "decimal_fraction",
                        "long_name": "Summer vegetation cover density",
                    },
                )
            elif "cov_type" in ds:
                _covden_lookup = {0: 0.0, 1: 0.3, 2: 0.4, 3: 0.7, 4: 0.8}
                covden = np.array([_covden_lookup.get(int(v), 0.3) for v in ds["cov_type"].values])
                ds["covden_sum"] = xr.DataArray(
                    covden,
                    dims="nhru",
                    attrs={
                        "units": "decimal_fraction",
                        "long_name": "Summer vegetation cover density",
                    },
                )

        # --- hru_percent_imperv: try pre-computed, then NLCD imperviousness ---
        imperv = self._try_precomputed(ctx, "hru_percent_imperv")
        if imperv is not None:
            # Determine if values are percentages (0-100) or fractions (0-1)
            # by checking the SIR variable name suffix: "_pct_" → percentage.
            imperv_decl = (ctx.precomputed or {}).get("hru_percent_imperv", {})
            imperv_var = imperv_decl.get("variable", "")
            if "_pct" in imperv_var or np.nanmax(imperv) > 1.0:
                if np.nanmax(imperv) > 1.0:
                    logger.info(
                        "Pre-computed 'hru_percent_imperv' max=%.2f; "
                        "converting from percent to fraction.",
                        float(np.nanmax(imperv)),
                    )
                imperv = imperv / 100.0
            ds["hru_percent_imperv"] = xr.DataArray(
                np.clip(imperv, 0.0, 1.0),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "HRU impervious fraction"},
            )
        else:
            fctimp_key = sir.find_variable("fctimp_pct_mean")
            if fctimp_key is not None:
                ds["hru_percent_imperv"] = xr.DataArray(
                    np.clip(sir[fctimp_key].values / 100.0, 0.0, 1.0),
                    dims="nhru",
                    attrs={"units": "decimal_fraction", "long_name": "HRU impervious fraction"},
                )

        return ds

    # Valid NLCD Anderson Level II class codes (11–95).
    _VALID_NLCD_CLASSES: frozenset[int] = frozenset(
        {11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95}
    )

    @staticmethod
    def _extract_nlcd_fractions(
        sir: SIRAccessor,
        prefixes: tuple[str, ...] = ("lndcov_frac_",),
        valid_codes: frozenset[int] | None = None,
    ) -> tuple[list[int], list[np.ndarray]] | None:
        """Extract NLCD class codes and fraction arrays from SIR.

        Scan SIR variables for columns matching ``{prefix}{class_code}``
        and return parallel lists of class codes and fraction arrays.
        Filter out NoData sentinels (class codes not in ``valid_codes``).

        Support two SIR layouts:

        1. **Column-level keys** — each fraction is a separate SIR variable
           (e.g., ``lndcov_frac_11``, ``lndcov_frac_41``).
        2. **File-level keys** — ``data_vars`` contains a year-suffixed
           entry like ``lndcov_frac_2021``.  The individual fraction
           columns are inside the backing file and accessed via
           ``sir.load_dataset()``.

        Parameters
        ----------
        sir : SIRAccessor
            SIR dataset containing fraction columns.
        prefixes : tuple[str, ...]
            Variable name prefixes to search for.
        valid_codes : frozenset[int] or None
            Set of valid NLCD class codes.  Codes not in this set are
            filtered out (e.g., NoData sentinel 250).  If ``None``,
            no code filtering is applied.  Codes exceeding 95 are
            treated as year-suffixed file-level keys regardless.

        Returns
        -------
        tuple[list[int], list[np.ndarray]] or None
            ``(class_codes, fractions_list)`` if at least 2 valid
            fraction columns are found, else ``None``.

        Notes
        -----
        When a suffix parses as an integer but exceeds 95 (the maximum
        NLCD class code), it is treated as a year-suffixed file-level
        key.  Inner columns within that file are individually checked
        against ``valid_codes`` to filter NoData sentinels (e.g.,
        class 250).
        """
        for prefix in prefixes:
            fraction_vars = sorted(v for v in sir.data_vars if v.startswith(prefix))
            if not fraction_vars:
                continue

            class_codes: list[int] = []
            fractions_list: list[np.ndarray] = []
            for v in fraction_vars:
                suffix = v[len(prefix) :]
                try:
                    code = int(suffix)
                except ValueError:
                    logger.debug(
                        "Skipping variable '%s': suffix '%s' is not an integer class code",
                        v,
                        suffix,
                    )
                    continue

                if code > 95:
                    # Suffix looks like a year (e.g. 2021), not an NLCD class.
                    try:
                        inner_ds = sir.load_dataset(v)
                    except KeyError:
                        logger.warning(
                            "SIR inconsistency: variable '%s' found in data_vars but "
                            "load_dataset() raised KeyError. Skipping file-level expansion.",
                            v,
                        )
                        continue

                    inner_prefix = f"{v}_"
                    for inner_name in sorted(str(v_) for v_ in inner_ds.data_vars):
                        if not inner_name.startswith(inner_prefix):
                            continue
                        inner_suffix = inner_name[len(inner_prefix) :]
                        try:
                            inner_code = int(inner_suffix)
                        except ValueError:
                            logger.debug(
                                "Skipping inner variable '%s': suffix '%s' is not an int",
                                inner_name,
                                inner_suffix,
                            )
                            continue
                        if valid_codes is not None and inner_code not in valid_codes:
                            logger.debug(
                                "Filtering inner variable '%s': code %d not in valid NLCD classes",
                                inner_name,
                                inner_code,
                            )
                            continue
                        class_codes.append(inner_code)
                        fractions_list.append(inner_ds[inner_name].values)
                else:
                    if valid_codes is not None and code not in valid_codes:
                        logger.debug(
                            "Filtering variable '%s': code %d not in valid NLCD classes",
                            v,
                            code,
                        )
                        continue
                    class_codes.append(code)
                    fractions_list.append(sir[v].values)

            if len(class_codes) < 2:
                if fraction_vars:
                    logger.debug(
                        "Found %d variable(s) matching prefix '%s' but only %d "
                        "valid class codes extracted; need at least 2.",
                        len(fraction_vars),
                        prefix,
                        len(class_codes),
                    )
                continue

            return class_codes, fractions_list

        return None

    @staticmethod
    def _compute_grouped_majority(
        class_codes: list[int],
        fractions_list: list[np.ndarray],
        mapping: dict[int, int],
    ) -> np.ndarray:
        """Compute majority category by grouping source fractions.

        Sum NLCD class fractions that map to the same target category
        (e.g., cov_type), then return the category with the highest
        total fraction per HRU.  This avoids the "split vote" problem
        where multiple source classes mapping to the same category
        individually lose to a single competitor class.

        Parameters
        ----------
        class_codes : list[int]
            NLCD class codes (parallel with ``fractions_list``).
        fractions_list : list[np.ndarray]
            Fraction arrays, each shape ``(nhru,)``.
        mapping : dict[int, int]
            Source class code → target category code (e.g., NLCD →
            cov_type).  Unmapped codes default to 0.

        Returns
        -------
        np.ndarray
            Majority target category code per HRU, shape ``(nhru,)``.

        Notes
        -----
        For HRUs where all grouped fractions are NaN or zero, cov_type
        0 (bare ground) is explicitly assigned and a warning is logged.

        Raises
        ------
        ValueError
            If ``fractions_list`` is empty or ``mapping`` is empty.

        References
        ----------
        Regan et al. 2018 (USGS TM6-B9), Table A1-3 for NLCD → PRMS
        cov_type mapping.
        """
        if not fractions_list:
            raise ValueError(
                "_compute_grouped_majority requires at least one fraction array; "
                "received an empty list."
            )
        if not mapping:
            raise ValueError("_compute_grouped_majority requires a non-empty mapping dict.")

        nhru = len(fractions_list[0])

        # Identify unique target categories (sorted for determinism).
        target_codes = sorted(set(mapping.values()))
        code_to_idx = {c: i for i, c in enumerate(target_codes)}
        n_groups = len(target_codes)

        # Accumulate fractions by target group.
        grouped = np.zeros((nhru, n_groups), dtype=np.float64)
        for src_code, frac_arr in zip(class_codes, fractions_list, strict=True):
            tgt = mapping.get(src_code, 0)
            idx = code_to_idx[tgt]
            frac_clean = np.where(np.isfinite(frac_arr), frac_arr, 0.0)
            grouped[:, idx] += frac_clean

        # Majority = argmax over grouped fractions.
        with np.errstate(invalid="ignore"):
            majority_idx = np.argmax(grouped, axis=1)
        result = np.array([target_codes[i] for i in majority_idx])

        # Explicitly assign bare ground (0) for HRUs with no valid data.
        _DEFAULT_CATEGORY = 0
        all_zero = np.all(grouped == 0, axis=1)
        if np.any(all_zero):
            result[all_zero] = _DEFAULT_CATEGORY
            logger.warning(
                "%d of %d HRUs have all-zero grouped fractions; "
                "majority category set to %d (bare ground) for those HRUs",
                int(np.sum(all_zero)),
                nhru,
                _DEFAULT_CATEGORY,
            )

        logger.info(
            "Computed grouped majority from %d source classes into %d target categories",
            len(class_codes),
            n_groups,
        )
        return result

    # ------------------------------------------------------------------
    # Step 5: Soils zonal stats
    # ------------------------------------------------------------------

    _SOIL_RECHR_MAX_FRAC_DEFAULT: float = 0.4

    def _derive_soils(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Derive soil parameters from gNATSGO/STATSGO2 zonal statistics (step 5).

        Classify soil texture into PRMS ``soil_type`` and derive
        ``soil_moist_max`` (maximum soil moisture capacity) from
        available water storage (``aws0_100_mm_mean``), available water
        capacity (``awc_mm_mean``), or root-zone AWS
        (``rootznaws_mm_mean``), in that priority order.

        Supports three input modes for soil texture:

        1. **Soil texture fractions** (preferred): SIR contains columns
           like ``soil_texture_frac_sand``, ``soil_texture_frac_loam``,
           etc.  Majority class determined via argmax, then reclassified
           to PRMS soil_type using ``soil_texture_to_prms_type.yml``.
        2. **Single texture class**: ``soil_texture`` or
           ``soil_texture_majority`` variable containing the dominant
           USDA texture class name per HRU.
        3. **Continuous percentages** (fallback): SIR contains
           ``sand_pct_mean``, ``silt_pct_mean``, ``clay_pct_mean``.
           Each HRU's mean percentages are classified via the USDA
           soil texture triangle, then mapped to PRMS soil_type.
           This is an aggregate-then-classify approach.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing the SIR dataset and lookup
            tables directory.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with ``soil_type`` (integer: 1=sand, 2=loam, 3=clay
            on ``nhru``), ``soil_moist_max`` (inches on ``nhru``), and
            ``soil_rechr_max_frac`` (decimal fraction on ``nhru``) added
            where input data is available.

        Notes
        -----
        Unit conversions for ``soil_moist_max``:

        - ``aws0_100_mm_mean`` (preferred): mm -> inches via ``convert()``.
        - ``awc_mm_mean``: mm -> inches via ``convert()``.
        - ``rootznaws_mm_mean`` (last resort): mm -> inches.

        gNATSGO ``aws0_100`` (fixed 0-100 cm column) is preferred over
        ``rootznaws`` (variable root-zone depth) because ``rootznaws``
        uses dominant-component-only rasterization that can produce
        spatial patterns inconsistent with the underlying SSURGO
        polygon database.  NHM historically derived ``soil_moist_max``
        from SSURGO component-weighted AWC with a 60-inch depth cap,
        which ``aws0_100`` (fixed depth) better approximates.

        All paths clip to ``[0.5, 20.0]`` inches.

        ``soil_rechr_max_frac`` is derived from the ratio
        ``aws0_30_mm / aws0_100_mm`` when both variables are present
        in the SIR.  ``aws0_30`` (0–30 cm, ~12 inches) is the closest
        gNATSGO depth interval to the classic PRMS recharge zone
        (upper 18 inches of soil); ``aws0_100`` (0–100 cm) approximates
        the full root zone.  HRUs with zero or NaN ``aws0_100`` receive
        the default 0.4.  The ratio is clipped to ``[0.1, 0.9]``.
        Falls back to a uniform 0.4 when ``aws0_30_mm_mean`` is absent
        from the SIR.

        See Also
        --------
        _compute_soil_type : Texture classification and PRMS mapping.
        hydro_param.classification.classify_usda_texture : USDA texture triangle.
        """
        sir = ctx.sir

        # --- soil_type: try pre-computed first, then texture derivation ---
        soil_type = self._try_precomputed(ctx, "soil_type", categorical=True)
        if soil_type is None:
            soil_type = self._compute_soil_type(sir, ctx)
        if soil_type is not None:
            ds["soil_type"] = xr.DataArray(
                soil_type,
                dims="nhru",
                attrs={"units": "integer", "long_name": "PRMS soil type (1=sand, 2=loam, 3=clay)"},
            )
        else:
            logger.warning(
                "Skipping soil_type derivation (step 5): no soil texture data "
                "found in SIR. Expected soil_texture_frac_* columns, "
                "soil_texture/soil_texture_majority variable, or continuous "
                "sand_pct_mean/silt_pct_mean/clay_pct_mean percentages."
            )

        # --- soil_moist_max ---
        # Prefer fixed-depth aws0_100 (0-100cm column) over variable-depth
        # rootznaws (root zone).  gNATSGO rootznaws uses dominant-component-
        # only rasterization that can produce spatial patterns inconsistent
        # with the underlying SSURGO polygon database.  aws0_100 is a
        # fixed-depth product that better approximates NHM's SSURGO-derived
        # component-weighted AWC with a 60-inch depth cap.
        #
        # All gNATSGO water storage variables (rootznaws, aws0_100) are in mm
        # (per Planetary Computer STAC metadata).  awc_mm_mean is also in mm.
        # All paths convert mm → inches via convert().
        aws_key = sir.find_variable("aws0_100_mm_mean")
        awc_key = sir.find_variable("awc_mm_mean")
        rzaws_key = sir.find_variable("rootznaws_mm_mean")

        if aws_key is not None:
            aws_mm = sir[aws_key].values.astype(np.float64)
            soil_moist_max = convert(aws_mm, "mm", "in")
            soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
            ds["soil_moist_max"] = xr.DataArray(
                soil_moist_max,
                dims="nhru",
                attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
            )
            logger.info("Used aws0_100_mm_mean (0-100cm AWS, mm -> in) for soil_moist_max")
        elif awc_key is not None:
            logger.debug("aws0_100_mm_mean not found in SIR; falling back to awc_mm_mean")
            awc_mm = sir[awc_key].values.astype(np.float64)
            soil_moist_max = convert(awc_mm, "mm", "in")
            soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
            ds["soil_moist_max"] = xr.DataArray(
                soil_moist_max,
                dims="nhru",
                attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
            )
            logger.info("Used awc_mm_mean (mm -> in) for soil_moist_max")
        elif rzaws_key is not None:
            logger.debug(
                "aws0_100_mm_mean and awc_mm_mean not found in SIR; "
                "falling back to rootznaws_mm_mean (lower quality)"
            )
            rzaws_mm = sir[rzaws_key].values.astype(np.float64)
            soil_moist_max = convert(rzaws_mm, "mm", "in")
            soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
            ds["soil_moist_max"] = xr.DataArray(
                soil_moist_max,
                dims="nhru",
                attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
            )
            logger.info("Used rootznaws_mm_mean (root zone AWS, mm -> in) for soil_moist_max")
        else:
            logger.warning(
                "Skipping soil_moist_max derivation (step 5): no rootznaws_mm_mean, "
                "awc_mm_mean, or aws0_100_mm_mean found in SIR."
            )

        # --- soil_rechr_max_frac ---
        # Prefer derived ratio: aws0_30 (0-30cm, ~12 inches) / aws0_100
        # (0-100cm, full root zone).  aws0_30 is the closest gNATSGO depth
        # interval to the PRMS recharge zone (~18 inches).  Falls back to
        # constant default when either variable is missing from the SIR.
        # Note: aws_key (aws0_100_mm_mean) was resolved above for soil_moist_max.
        aws30_key = sir.find_variable("aws0_30_mm_mean")
        if aws30_key is not None and aws_key is not None:
            aws30_mm = sir[aws30_key].values.astype(np.float64)
            aws100_mm = sir[aws_key].values.astype(np.float64)
            # Guard division by zero and NaN: invalid HRUs get default
            valid = (aws100_mm > 0) & ~np.isnan(aws100_mm) & ~np.isnan(aws30_mm)
            ratio = np.full_like(aws100_mm, self._SOIL_RECHR_MAX_FRAC_DEFAULT)
            ratio[valid] = aws30_mm[valid] / aws100_mm[valid]
            ratio = np.clip(ratio, 0.1, 0.9)
            ds["soil_rechr_max_frac"] = xr.DataArray(
                ratio,
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Fraction of soil moisture in recharge zone",
                },
            )
            n_invalid = int(np.sum(~valid))
            n_nan = int(np.sum(np.isnan(sir[aws30_key].values) | np.isnan(sir[aws_key].values)))
            logger.info(
                "soil_rechr_max_frac derived from aws0_30/aws0_100 ratio for %d HRUs "
                "(%d non-positive/NaN inputs set to default %.2f)",
                len(ratio),
                n_invalid,
                self._SOIL_RECHR_MAX_FRAC_DEFAULT,
            )
            if n_nan > 0:
                logger.warning(
                    "%d HRUs have NaN in aws0_30 or aws0_100 input data; "
                    "set to default soil_rechr_max_frac=%.2f",
                    n_nan,
                    self._SOIL_RECHR_MAX_FRAC_DEFAULT,
                )
        elif "soil_type" in ds:
            nhru = len(ds["soil_type"])
            ds["soil_rechr_max_frac"] = xr.DataArray(
                np.full(nhru, self._SOIL_RECHR_MAX_FRAC_DEFAULT),
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Fraction of soil moisture in recharge zone",
                },
            )
            logger.info(
                "soil_rechr_max_frac set to default %.2f for %d HRUs "
                "(aws0_30_mm_mean and/or aws0_100_mm_mean not available in SIR)",
                self._SOIL_RECHR_MAX_FRAC_DEFAULT,
                nhru,
            )

        return ds

    def _compute_soil_type(self, sir: SIRAccessor, ctx: DerivationContext) -> np.ndarray | None:
        """Compute PRMS soil_type from SIR soil texture data.

        Try fraction columns first (argmax across texture classes), then
        fall back to a single texture class variable, then to continuous
        sand/silt/clay percentages via USDA texture triangle classification.
        Unrecognized texture names default to loam (soil_type=2).

        Parameters
        ----------
        sir : SIRAccessor
            SIR dataset with soil texture variables.
        ctx : DerivationContext
            Derivation context providing the lookup tables directory.

        Returns
        -------
        np.ndarray or None
            Array of PRMS soil type codes (1=sand, 2=loam, 3=clay) with
            shape ``(nhru,)``, or ``None`` if no soil texture data is
            found in the SIR.

        Notes
        -----
        Requires ``soil_texture_to_prms_type.yml`` lookup table from
        ``ctx.resolved_lookup_tables_dir``.  The table maps USDA texture
        class names to PRMS soil_type integers (1=coarse, 2=medium,
        3=fine).

        See Also
        --------
        hydro_param.classification.classify_usda_texture : USDA texture triangle.
        """
        # Check data availability before loading lookup table
        prefix = "soil_texture_frac_"
        fraction_vars = sorted(v for v in sir.data_vars if v.startswith(prefix))
        has_single = any(c in sir for c in ("soil_texture", "soil_texture_majority"))
        has_continuous = all(v in sir for v in ("sand_pct_mean", "silt_pct_mean", "clay_pct_mean"))

        if len(fraction_vars) < 2 and not has_single and not has_continuous:
            return None

        tables_dir = ctx.resolved_lookup_tables_dir
        soil_table = self._load_lookup_table("soil_texture_to_prms_type", tables_dir)
        mapping = soil_table["mapping"]

        # Try fraction columns first
        if len(fraction_vars) >= 2:
            class_names: list[str] = []
            valid_vars: list[str] = []
            for v in fraction_vars:
                name = v[len(prefix) :]
                if name in mapping:
                    class_names.append(name)
                    valid_vars.append(v)
                else:
                    logger.debug("Skipping soil fraction '%s': class '%s' not in lookup", v, name)

            if len(valid_vars) >= 2:
                fractions = np.column_stack([sir[v].values for v in valid_vars])
                nan_mask = np.any(np.isnan(fractions), axis=1)
                if np.any(nan_mask):
                    logger.warning(
                        "soil_type: %d/%d HRU(s) have NaN soil texture fractions; "
                        "argmax result may be unreliable for those HRUs",
                        int(np.sum(nan_mask)),
                        len(fractions),
                    )
                majority_idx = np.argmax(fractions, axis=1)
                majority_names = [class_names[i] for i in majority_idx]
                return np.array([mapping[name] for name in majority_names])

        # Fallback: single texture class variable
        for candidate in ("soil_texture", "soil_texture_majority"):
            if candidate in sir:
                texture_values = sir[candidate].values
                result = []
                unknown_values: set[str] = set()
                for v in texture_values:
                    key = str(v)
                    if key in mapping:
                        result.append(mapping[key])
                    else:
                        unknown_values.add(key)
                        result.append(2)  # default to loam
                if unknown_values:
                    logger.warning(
                        "soil_type: %d HRU(s) have unrecognized texture class(es) "
                        "%s in '%s'; defaulting to loam (soil_type=2)",
                        sum(1 for val in texture_values if str(val) in unknown_values),
                        sorted(unknown_values),
                        candidate,
                    )
                return np.array(result)

        # Fallback: classify continuous sand/silt/clay percentages via
        # the USDA texture triangle.  This is an aggregate-then-classify
        # approach — HRU-mean percentages are classified directly, which
        # may differ from pixel-level classification.
        if has_continuous:
            sand = sir["sand_pct_mean"].values.astype(np.float64)
            silt = sir["silt_pct_mean"].values.astype(np.float64)
            clay = sir["clay_pct_mean"].values.astype(np.float64)

            # Normalize HRU-mean percentages to sum=100%.  POLARIS
            # estimates each fraction independently so HRU-level means
            # (from zonal aggregation of independently-estimated rasters)
            # rarely sum to exactly 100%.  Same rationale as the pixel-
            # level normalization in classify_usda_texture_raster().
            total = sand + silt + clay
            valid = ~(np.isnan(sand) | np.isnan(silt) | np.isnan(clay)) & (total > 0.1)
            need_norm = valid & (np.abs(total - 100.0) > 0.01)
            n_norm = int(np.sum(need_norm))
            if n_norm > 0:
                logger.info(
                    "soil_type: normalizing %d/%d HRU(s) sand+silt+clay "
                    "to sum=100%% before texture classification",
                    n_norm,
                    int(np.sum(valid)),
                )
                scale = 100.0 / total[need_norm]
                sand[need_norm] *= scale
                silt[need_norm] *= scale
                clay[need_norm] *= scale

            codes = classify_usda_texture(sand, silt, clay)
            nan_mask = np.isnan(codes)
            nan_count = int(np.sum(nan_mask))
            if nan_count > 0:
                logger.warning(
                    "soil_type: %d/%d HRU(s) have NaN sand/silt/clay "
                    "percentages; defaulting to loam (soil_type=2). "
                    "Check source data coverage.",
                    nan_count,
                    len(codes),
                )
            texture_names = np.array(
                [
                    USDA_TEXTURE_CLASSES.get(int(c), "loam") if not np.isnan(c) else "loam"
                    for c in codes
                ]
            )
            logger.info(
                "soil_type: classified %d HRUs from continuous sand/silt/clay "
                "percentages via USDA texture triangle (aggregate-then-classify)",
                len(texture_names),
            )
            result_arr = []
            for name in texture_names:
                if name in mapping:
                    result_arr.append(mapping[name])
                else:
                    logger.warning(
                        "soil_type: unrecognized texture name '%s' from USDA "
                        "triangle; defaulting to loam (soil_type=2)",
                        name,
                    )
                    result_arr.append(mapping.get("loam", 2))
            return np.array(result_arr)

        return None

    # ------------------------------------------------------------------
    # Step 6: Waterbody overlay (depression storage)
    # ------------------------------------------------------------------

    def _waterbody_defaults(self, ds: xr.Dataset, nhru: int) -> xr.Dataset:
        """Assign zero/default waterbody parameters when no overlay data exists.

        Set ``dprst_frac`` to zero and ``hru_type`` to 1 (land) for all
        HRUs.  Used as a fallback when waterbody data, fabric, or
        ``hru_area`` is unavailable.

        Parameters
        ----------
        ds : xr.Dataset
            In-progress parameter dataset to augment.
        nhru : int
            Number of HRUs for array dimensioning.

        Returns
        -------
        xr.Dataset
            Dataset with ``dprst_frac`` (fraction) and ``hru_type``
            (dimensionless integer) set to defaults on the ``nhru``
            dimension.
        """
        ds["dprst_frac"] = xr.DataArray(
            np.zeros(nhru),
            dims="nhru",
            attrs={"units": "fraction", "long_name": "Depression storage fraction of HRU area"},
        )
        ds["hru_type"] = xr.DataArray(
            np.ones(nhru, dtype=np.int32),
            dims="nhru",
            attrs={"units": "none", "long_name": "HRU type (1=land, 2=lake)"},
        )
        return ds

    def _derive_waterbody(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Derive depression storage from NHDPlus waterbody overlay (step 6).

        Perform polygon-on-polygon overlay of NHDPlus waterbody polygons
        (LakePond and Reservoir feature types) against the HRU fabric to
        compute depression storage fraction and HRU type classification
        (land vs. lake).

        The overlay uses ``geopandas.overlay(how="intersection")`` to clip
        waterbody polygons to HRU boundaries, then sums clipped areas per
        HRU.  HRUs with >50% waterbody coverage are classified as lake-type
        (``hru_type=2``).

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing ``fabric``, ``waterbodies``,
            ``fabric_id_field``, and ``sir``.
        ds : xr.Dataset
            In-progress parameter dataset (must contain ``hru_area`` from
            step 1 for fraction computation).

        Returns
        -------
        xr.Dataset
            Dataset with ``dprst_frac`` (decimal fraction 0--1 on ``nhru``)
            and ``hru_type`` (integer: 1=land, 2=lake on ``nhru``).  Falls
            back to zero defaults if any prerequisite data is missing.

        Raises
        ------
        KeyError
            If ``id_field`` is missing from the fabric columns, or if the
            waterbody GeoDataFrame lacks an ``ftype`` column.

        Notes
        -----
        Unit conversions: intersection area from m² -> acres (factor:
        1 acre = 4046.8564224 m²).  The fabric CRS must be projected
        for accurate area computation; a warning is logged if geographic.

        Waterbodies are reprojected to match the fabric CRS if they differ.
        Only ``ftype`` values ``"LakePond"`` and ``"Reservoir"`` are
        included; other feature types (e.g., SwampMarsh) are excluded.

        See Also
        --------
        _waterbody_defaults : Zero-value fallback when overlay is skipped.
        """
        nhru = ds.sizes.get("nhru", 0)
        id_field = ctx.fabric_id_field

        # Guard: hru_area must exist from step 1
        if "hru_area" not in ds:
            logger.warning("Step 6 requires 'hru_area' from step 1 (not found); using defaults")
            return self._waterbody_defaults(ds, nhru)

        # Guard: fabric required for overlay
        fabric = ctx.fabric
        if fabric is None:
            logger.warning("No fabric provided; using defaults for step 6")
            return self._waterbody_defaults(ds, nhru)

        # Guard: id_field must exist in fabric
        if id_field not in fabric.columns:
            raise KeyError(
                f"Fabric GeoDataFrame missing id_field '{id_field}' "
                f"(found: {sorted(fabric.columns.tolist())})"
            )

        # Fallback: no waterbody data
        if ctx.waterbodies is None:
            logger.warning("No waterbody data provided; using defaults for step 6")
            return self._waterbody_defaults(ds, nhru)

        if "ftype" not in ctx.waterbodies.columns:
            raise KeyError(
                "Waterbody GeoDataFrame must contain an 'ftype' column "
                f"(found: {sorted(ctx.waterbodies.columns.tolist())})"
            )

        # Filter to LakePond and Reservoir only
        wb = ctx.waterbodies[ctx.waterbodies["ftype"].isin({"LakePond", "Reservoir"})]
        if wb.empty:
            logger.info("No LakePond/Reservoir waterbodies found; using defaults for step 6")
            return self._waterbody_defaults(ds, nhru)

        # Ensure matching CRS
        if wb.crs != fabric.crs:
            logger.info("Reprojecting waterbodies from %s to %s", wb.crs, fabric.crs)
            wb = wb.to_crs(fabric.crs)

        # Warn if CRS is geographic (area computation will be wrong)
        if fabric.crs is not None and not fabric.crs.is_projected:
            logger.warning(
                "Fabric CRS %s is geographic — area computations may be inaccurate. "
                "Use a projected CRS (e.g. EPSG:5070) for reliable results.",
                fabric.crs,
            )

        # Polygon overlay: intersection of fabric x waterbodies
        try:
            intersections = gpd.overlay(
                fabric[[id_field, "geometry"]],
                wb[["geometry"]],
                how="intersection",
            )
        except MemoryError:
            raise
        except Exception:
            logger.warning(
                "gpd.overlay failed in step 6 (waterbody); using zero defaults "
                "for depression storage parameters (dprst_frac). "
                "Check that your waterbody file has valid geometries and a "
                "compatible CRS.",
                exc_info=True,
            )
            return self._waterbody_defaults(ds, nhru)

        if intersections.empty:
            logger.info("No waterbody-HRU intersections found; using defaults for step 6")
            return self._waterbody_defaults(ds, nhru)

        # Compute clipped areas and group by HRU
        intersections["_clip_area_m2"] = intersections.geometry.area
        area_by_hru = intersections.groupby(id_field)["_clip_area_m2"].sum()

        # Vectorized alignment to fabric HRU order
        hru_ids = fabric[id_field].values
        clipped_acres = area_by_hru.reindex(hru_ids, fill_value=0.0).values / _M2_PER_ACRE

        # Compute fraction from hru_area (already in acres from step 1)
        hru_area_acres = ds["hru_area"].values
        dprst_frac = np.where(hru_area_acres > 0, clipped_acres / hru_area_acres, 0.0)
        dprst_frac = np.clip(dprst_frac, 0.0, 1.0)

        # HRU type: 2 (lake) if >50% waterbody, else 1 (land)
        hru_type = np.where(dprst_frac > 0.5, 2, 1).astype(np.int32)

        ds["dprst_frac"] = xr.DataArray(
            dprst_frac,
            dims="nhru",
            attrs={"units": "fraction", "long_name": "Depression storage fraction of HRU area"},
        )
        ds["hru_type"] = xr.DataArray(
            hru_type,
            dims="nhru",
            attrs={"units": "none", "long_name": "HRU type (1=land, 2=lake)"},
        )

        n_lake = int((hru_type == 2).sum())
        n_with_water = int((dprst_frac > 0).sum())
        logger.info(
            "Step 6 waterbody overlay: %d/%d HRUs with waterbodies, %d lake-type",
            n_with_water,
            nhru,
            n_lake,
        )

        return ds

    # ------------------------------------------------------------------
    # Step 8: Lookup table application
    # ------------------------------------------------------------------

    def _apply_lookup_tables(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Apply lookup tables for interception and winter cover density (step 8).

        For each interception parameter (``srain_intcp``, ``wrain_intcp``,
        ``snow_intcp``) and ``covden_win``, check for a pre-computed value
        via ``_try_precomputed()`` before falling back to lookup-table
        derivation from ``cov_type``.  This allows GFv1.1 (or other
        pre-computed sources) to supply these values directly.

        When no pre-computed value is available, use ``cov_type`` (from
        step 4) to look up per-HRU interception capacities and compute
        winter cover density by applying a cov_type-dependent reduction
        factor to ``covden_sum``.

        Also sets ``imperv_stor_max`` to a uniform default of 0.03 inches.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing the lookup tables directory.
        ds : xr.Dataset
            In-progress parameter dataset (must contain ``cov_type`` from
            step 4; ``covden_sum`` needed for winter density).

        Returns
        -------
        xr.Dataset
            Dataset with ``srain_intcp`` (inches), ``wrain_intcp``
            (inches), ``snow_intcp`` (inches), ``imperv_stor_max``
            (inches), and ``covden_win`` (decimal fraction) added on
            the ``nhru`` dimension.  Returns ``ds`` unchanged if
            ``cov_type`` is not present.

        Notes
        -----
        Lookup tables used:

        - ``cov_type_to_interception.yml`` --- maps PRMS cov_type to
          seasonal interception capacities (inches).
        - ``cov_type_winter_reduction.yml`` --- maps PRMS cov_type to a
          multiplicative reduction factor for converting summer to winter
          cover density.

        See Also
        --------
        _load_lookup_table : YAML lookup table loader with caching.
        """
        if "cov_type" not in ds:
            return ds

        tables_dir = ctx.resolved_lookup_tables_dir
        cov_type_vals = ds["cov_type"].values.astype(int)
        nhru = len(cov_type_vals)

        # --- Interception capacities: try pre-computed, then lookup ---
        intcp_params = ("srain_intcp", "wrain_intcp", "snow_intcp")
        intcp_table = self._load_lookup_table("cov_type_to_interception", tables_dir)
        columns = intcp_table["columns"]
        mapping_intcp = intcp_table["mapping"]

        for i, col_name in enumerate(intcp_params):
            precomputed_val = self._try_precomputed(ctx, col_name)
            if precomputed_val is not None:
                ds[col_name] = xr.DataArray(
                    precomputed_val,
                    dims="nhru",
                    attrs={
                        "units": "inches",
                        "long_name": f"{col_name.replace('_', ' ').title()}",
                    },
                )
            else:
                col_idx = columns.index(col_name) if col_name in columns else i
                values = np.array(
                    [mapping_intcp.get(int(ct), [0.0, 0.0, 0.0])[col_idx] for ct in cov_type_vals]
                )
                ds[col_name] = xr.DataArray(
                    values,
                    dims="nhru",
                    attrs={
                        "units": "inches",
                        "long_name": f"{col_name.replace('_', ' ').title()}",
                    },
                )

        # Imperv storage max (uniform default)
        ds["imperv_stor_max"] = xr.DataArray(
            np.full(nhru, _IMPERV_STOR_MAX_DEFAULT),
            dims="nhru",
            attrs={"units": "inches", "long_name": "Maximum impervious retention storage"},
        )

        # --- Winter cover density: try pre-computed, then reduction factor ---
        covden_win = self._try_precomputed(ctx, "covden_win")
        if covden_win is not None:
            ds["covden_win"] = xr.DataArray(
                np.clip(covden_win, 0.0, 1.0),
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Winter vegetation cover density",
                },
            )
        elif "covden_sum" in ds:
            winter_table = self._load_lookup_table("cov_type_winter_reduction", tables_dir)
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
    # Step 9: Solar radiation tables
    # ------------------------------------------------------------------

    def _derive_soltab(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Compute potential solar radiation tables using Swift (1976) (step 9).

        Generate day-of-year-resolved potential shortwave radiation for
        sloped and horizontal surfaces, plus hours of direct sunlight,
        for each HRU.  These tables drive the ``soltab`` module in
        pywatershed/PRMS for daily solar radiation distribution.

        Requires ``hru_lat`` (decimal degrees), ``hru_slope`` (decimal
        fraction), and ``hru_aspect`` (degrees) from step 3.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context (used for logging context only).
        ds : xr.Dataset
            In-progress parameter dataset containing step 3 outputs.

        Returns
        -------
        xr.Dataset
            Dataset with three 2-D arrays on dimensions ``(ndoy, nhru)``:

            - ``soltab_potsw`` --- potential SW radiation on slope
              (cal/cm²/day)
            - ``soltab_horad_potsw`` --- potential SW radiation on
              horizontal surface (cal/cm²/day)
            - ``soltab_sunhrs`` --- hours of direct sunlight (hours)

            Returns ``ds`` unchanged (with warning) if required inputs
            are missing.

        Notes
        -----
        ``ndoy`` has 366 entries (includes leap day).  The ``nhru``
        dimension aligns with the ``id_field`` coordinate on the
        existing dataset.

        NaN values in the output indicate NaN in the upstream zonal
        statistics (slope, aspect, or latitude) and are logged as
        warnings.

        References
        ----------
        Swift, L. W. (1976). Algorithm for Solar Radiation on Mountain
            Slopes. Water Resources Research, 12(1), 108-112.

        See Also
        --------
        hydro_param.solar.compute_soltab : Core solar table computation.
        """
        required = ("hru_lat", "hru_slope", "hru_aspect")
        if not all(v in ds for v in required):
            missing = [v for v in required if v not in ds]
            logger.warning(
                "Skipping soltab derivation (step 9): missing required variables %s. "
                "Ensure step 3 (topography) completed successfully. "
                "Output will NOT contain soltab_potsw, soltab_horad_potsw, or soltab_sunhrs.",
                missing,
            )
            return ds

        potsw, horad, sunhrs = compute_soltab(
            slopes=ds["hru_slope"].values,
            aspects=ds["hru_aspect"].values,
            lats=ds["hru_lat"].values,
        )

        # Check for NaN in output (can occur if upstream zonal stats had gaps)
        for name, arr in [
            ("soltab_potsw", potsw),
            ("soltab_horad_potsw", horad),
            ("soltab_sunhrs", sunhrs),
        ]:
            nan_count = np.count_nonzero(np.isnan(arr))
            if nan_count:
                logger.warning(
                    "%s contains %d/%d NaN values — likely caused by NaN in "
                    "hru_slope, hru_aspect, or hru_lat inputs",
                    name,
                    nan_count,
                    arr.size,
                )

        ds["soltab_potsw"] = xr.DataArray(
            potsw,
            dims=("ndoy", "nhru"),
            attrs={"units": "cal/cm2/day", "long_name": "Potential SW radiation on slope"},
        )
        ds["soltab_horad_potsw"] = xr.DataArray(
            horad,
            dims=("ndoy", "nhru"),
            attrs={"units": "cal/cm2/day", "long_name": "Potential SW radiation on horizontal"},
        )
        ds["soltab_sunhrs"] = xr.DataArray(
            sunhrs,
            dims=("ndoy", "nhru"),
            attrs={"units": "hr", "long_name": "Hours of direct sunlight"},
        )
        return ds

    # ------------------------------------------------------------------
    # Step 13: Defaults and initial conditions
    # ------------------------------------------------------------------

    def _apply_defaults(self, ds: xr.Dataset, nhru: int) -> xr.Dataset:
        """Apply standard PRMS default values and initial conditions (step 13).

        Fill parameters that were not derived from data in earlier steps
        with literature-standard defaults from Regan et al. (2018) and
        Markstrom et al. (2015).  Only sets parameters that are **not**
        already present in ``ds``, preserving data-derived values.

        Parameters
        ----------
        ds : xr.Dataset
            In-progress parameter dataset to augment.
        nhru : int
            Number of HRUs for array dimensioning of per-HRU defaults.

        Returns
        -------
        xr.Dataset
            Dataset with default values added for any missing parameters
            from the ``_DEFAULTS`` dictionary.

        Notes
        -----
        Special-case parameters that require unique shapes or derivations:

        - ``jh_coef``: shape ``(nmonth, nhru)`` i.e. ``(12, nhru)`` --- per_degF_per_day
        - ``transp_beg``, ``transp_end``: shape ``(nhru,)`` --- integer month
        - ``hru_type``: shape ``(nhru,)`` --- integer (1=land, 2=lake)
        - ``doy``: shape ``(366,)`` --- day-of-year coordinate
        - ``hru_in_to_cf``: shape ``(nhru,)`` --- derived from ``hru_area``
        - ``temp_units``: scalar --- 0 for Fahrenheit
        - ``snarea_curve``: shape ``(11,)`` --- snow depletion curve
        - ``pref_flow_infil_frac``: shape ``(nhru,)`` --- derived from
          ``pref_flow_den`` or defaults to 0.0

        All other defaults are broadcast to ``(nhru,)`` or ``(nmonth, nhru)``
        using the ``_PARAM_DIMS`` mapping.  Default units match PRMS
        conventions: inches for storage depths, degree-days for
        ``transp_tmax``, etc.  Segment-level defaults are added only when
        routing topology (``nsegment`` dimension) is present.

        References
        ----------
        Regan, R. S., et al. (2018). USGS Techniques and Methods 6-B9.
        Markstrom, S. L., et al. (2015). USGS Techniques and Methods 6-B7.
        """
        # Special handling for 2D jh_coef default (nmonth, nhru)
        if "jh_coef" not in ds:
            ds["jh_coef"] = xr.DataArray(
                np.full((12, nhru), _DEFAULTS["jh_coef"]),
                dims=("nmonth", "nhru"),
                attrs={
                    "units": "per_degF_per_day",
                    "long_name": "Jensen-Haise PET coefficient (default)",
                },
            )

        # Special handling for transp_beg/transp_end (integer, per-HRU)
        for param in ("transp_beg", "transp_end"):
            if param not in ds:
                ds[param] = xr.DataArray(
                    np.full(nhru, int(_DEFAULTS[param]), dtype=np.int32),
                    dims=("nhru",),
                    attrs={"units": "integer_month", "long_name": f"{param} (default)"},
                )

        # Special handling for hru_type (integer, per-HRU)
        if "hru_type" not in ds:
            ds["hru_type"] = xr.DataArray(
                np.full(nhru, int(_DEFAULTS["hru_type"]), dtype=np.int32),
                dims=("nhru",),
                attrs={"units": "none", "long_name": "HRU type (default)"},
            )

        # doy: coordinate array 1-366
        if "doy" not in ds:
            ds["doy"] = xr.DataArray(
                np.arange(1, 367, dtype=np.int32),
                dims=("ndoy",),
                attrs={"long_name": "Day of year"},
            )

        # hru_in_to_cf: unit conversion factor (inches*acres → cubic feet)
        # 1 inch over 1 acre = 43560/12 = 3630 ft³
        if "hru_in_to_cf" not in ds:
            if "hru_area" in ds:
                ds["hru_in_to_cf"] = xr.DataArray(
                    ds["hru_area"].values * (43560.0 / 12.0),
                    dims=("nhru",),
                    attrs={
                        "units": "cubic_feet_per_inch_acre",
                        "long_name": "Inches to cubic feet conversion factor",
                    },
                )
            else:
                logger.warning(
                    "Cannot compute 'hru_in_to_cf': 'hru_area' not in dataset. "
                    "Ensure geometry derivation (step 1) runs before defaults. "
                    "pywatershed may fail at runtime without this parameter."
                )

        # temp_units: 0 = Fahrenheit (PRMS convention)
        if "temp_units" not in ds:
            ds["temp_units"] = xr.DataArray(
                np.int32(0),
                attrs={"long_name": "Temperature units (0=F, 1=C)"},
            )

        # elev_units: 1 = meters (elevations kept in meters, not converted)
        if "elev_units" not in ds:
            ds["elev_units"] = xr.DataArray(
                np.int32(1),
                attrs={"long_name": "Elevation units (0=feet, 1=meters)"},
            )

        # snarea_curve: snow depletion curve (11 values, default all 1.0)
        if "snarea_curve" not in ds:
            ds["snarea_curve"] = xr.DataArray(
                np.ones(11, dtype=np.float64),
                dims=("ndeplval",),
                attrs={"long_name": "Snow area depletion curve"},
            )

        # pref_flow_infil_frac: pywatershed v2.0 requires values in [0, 1].
        # Use pref_flow_den if available, otherwise default to 0.0.
        if "pref_flow_infil_frac" not in ds:
            if "pref_flow_den" in ds:
                ds["pref_flow_infil_frac"] = xr.DataArray(
                    ds["pref_flow_den"].values.copy(),
                    dims=("nhru",),
                    attrs={"long_name": "Preferential flow infiltration fraction"},
                )
            else:
                ds["pref_flow_infil_frac"] = xr.DataArray(
                    np.zeros(nhru, dtype=np.float64),
                    dims=("nhru",),
                    attrs={"long_name": "Preferential flow infiltration fraction"},
                )

        # Dimension sizes for broadcasting
        dim_sizes: dict[str, int] = {
            "nhru": nhru,
            "nmonth": 12,
        }

        for param_name, default_val in _DEFAULTS.items():
            if param_name in _DEFAULTS_SPECIAL:
                continue  # handled above
            if param_name in ds:
                continue  # data-derived value takes precedence
            dims = _PARAM_DIMS[param_name]
            shape = tuple(dim_sizes[d] for d in dims)
            dtype = np.int32 if isinstance(default_val, int) else np.float64
            ds[param_name] = xr.DataArray(
                np.full(shape, default_val, dtype=dtype),
                dims=dims,
                attrs={"long_name": param_name.replace("_", " ").title()},
            )

        # Segment-level defaults (only when routing topology is present)
        nseg_vars = [v for v in ds.data_vars if "nsegment" in (ds[v].dims or ())]
        if nseg_vars:
            nsegment = ds[nseg_vars[0]].sizes.get("nsegment", 0)
            if nsegment > 0:
                seg_defaults: dict[str, tuple[float | int, str]] = {
                    "mann_n": (0.04, "Manning's roughness coefficient"),
                    "seg_depth": (1.0, "Bankfull water depth"),
                    "segment_flow_init": (0.0, "Initial flow in segment"),
                    "obsout_segment": (0, "Observed streamflow segment index"),
                }
                for name, (val, desc) in seg_defaults.items():
                    if name not in ds:
                        dtype = np.int32 if isinstance(val, int) else np.float64
                        ds[name] = xr.DataArray(
                            np.full(nsegment, val, dtype=dtype),
                            dims=("nsegment",),
                            attrs={"long_name": desc},
                        )

        return ds

    # ------------------------------------------------------------------
    # Step 14: Calibration seeds
    # ------------------------------------------------------------------

    def _derive_calibration_seeds(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Apply YAML-driven calibration seed values (step 14).

        Compute physically-based initial values for calibration parameters
        from the ``calibration_seeds.yml`` lookup table.  Seeds provide
        reasonable starting points for calibration, derived from existing
        dataset variables using simple mathematical relationships.

        Each seed specification includes a method (``linear``,
        ``exponential_scale``, ``fraction_of``, ``constant``), input
        variable reference, method-specific parameters, valid range
        for clipping, and a fallback default value.

        Seeds are **skipped** when the parameter already exists in ``ds``
        (e.g., set by an earlier derivation step or user override).

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing the lookup tables directory.
        ds : xr.Dataset
            In-progress parameter dataset to augment.  Existing parameters
            take precedence over seed values.

        Returns
        -------
        xr.Dataset
            Dataset with calibration seed parameters added.  Each seed
            variable carries ``attrs["source"] = "calibration_seed"`` for
            provenance tracking.

        Raises
        ------
        ValueError
            If a seed method requires a parameter key not present in the
            YAML ``params`` dict, or if ``range`` does not have exactly
            2 elements ``[min, max]``.

        Notes
        -----
        Supported seed methods (defined in ``_SEED_METHODS``):

        - ``linear``: ``scale * input + offset``
        - ``exponential_scale``: ``scale * exp(exponent * input)``
        - ``fraction_of``: ``fraction * input``
        - ``constant``: fixed scalar value

        Unknown methods and missing input variables trigger a warning
        and fall back to the specified default value.

        See Also
        --------
        _SEED_METHODS : Method dispatch dictionary.
        """
        tables_dir = ctx.resolved_lookup_tables_dir
        seed_table = self._load_lookup_table("calibration_seeds", tables_dir)
        mapping = seed_table["mapping"]

        nhru = ds.sizes.get("nhru", 0)

        for param_name, spec in mapping.items():
            # Skip if already derived or set by an earlier step
            if param_name in ds:
                logger.debug(
                    "Calibration seed '%s' skipped: already present in dataset",
                    param_name,
                )
                continue

            method_name = spec.get("method", "")
            params = spec.get("params", {})
            val_range = spec.get("range", [None, None])
            default = spec.get("default")
            default_val = default if default is not None else 0.0

            # Check method is known
            method_fn = _SEED_METHODS.get(method_name)
            value: np.floating | np.ndarray
            if method_fn is None:
                logger.warning(
                    "Calibration seed '%s': unknown method '%s'; using default %.4g",
                    param_name,
                    method_name,
                    default_val,
                )
                value = np.float64(default_val)
            else:
                # Check if required input variable exists for non-constant methods
                input_var = params.get("input")
                if input_var is not None and input_var not in ds:
                    logger.warning(
                        "Calibration seed '%s': input '%s' not in dataset; using default %.4g",
                        param_name,
                        input_var,
                        default_val,
                    )
                    value = np.float64(default_val)
                else:
                    try:
                        value = method_fn(ds, params)
                    except KeyError as exc:
                        raise ValueError(
                            f"Calibration seed '{param_name}': method '{method_name}' "
                            f"requires parameter {exc} missing from 'params' dict. "
                            f"Available keys: {sorted(params.keys())}. "
                            f"Check calibration_seeds.yml entry for '{param_name}'."
                        ) from exc

            # Expand scalar to correct shape.  Monthly parameters need
            # (nmonth, nhru); all others need (nhru,).
            dims_for_seed = _PARAM_DIMS.get(param_name, ("nhru",))
            dim_sizes_seed: dict[str, int] = {"nhru": nhru, "nmonth": 12}
            if nhru > 0 and np.ndim(value) == 0:
                shape = tuple(dim_sizes_seed[d] for d in dims_for_seed)
                value = np.full(shape, value, dtype=np.float64)

            # Clip to range
            if len(val_range) != 2:
                raise ValueError(
                    f"Calibration seed '{param_name}': 'range' must have exactly "
                    f"2 elements [min, max], got {val_range}"
                )
            rmin, rmax = val_range
            if rmin is not None or rmax is not None:
                value = np.clip(value, rmin, rmax)

            # Check for NaN in computed values
            if np.ndim(value) >= 1:
                nan_count = int(np.count_nonzero(np.isnan(value)))
                if nan_count > 0:
                    logger.warning(
                        "Calibration seed '%s': %d/%d HRU(s) have NaN values "
                        "(likely from NaN in input '%s')",
                        param_name,
                        nan_count,
                        int(np.size(value)),
                        params.get("input", "N/A"),
                    )

            # Add to dataset
            dims: tuple[str, ...] = dims_for_seed if np.ndim(value) >= 1 else ()
            ds[param_name] = xr.DataArray(
                value,
                dims=dims if dims else None,
                attrs={
                    "long_name": param_name.replace("_", " ").title(),
                    "source": "calibration_seed",
                },
            )

        logger.info(
            "Step 14: applied %d calibration seeds",
            sum(1 for p in mapping if p in ds and ds[p].attrs.get("source") == "calibration_seed"),
        )
        return ds

    # ------------------------------------------------------------------
    # Step 7: Forcing generation (temporal merge)
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_temporal_chunks(chunks: list[xr.Dataset]) -> xr.Dataset:
        """Sort and concatenate multi-year temporal chunks along time.

        Parameters
        ----------
        chunks : list[xr.Dataset]
            One or more single-year temporal datasets to concatenate.

        Returns
        -------
        xr.Dataset
            Concatenated dataset sorted by time.
        """
        if len(chunks) > 1:
            chunks.sort(key=lambda c: c["time"].values[0])
            return xr.concat(chunks, dim="time")
        return chunks[0]

    def _build_sir_to_forcing_lookup(
        self,
        tables_dir: Path,
    ) -> dict[str, dict[str, str]]:
        """Build reverse lookup from SIR variable names to forcing config.

        Invert the ``forcing_variables.yml`` mapping so that each SIR
        canonical name maps to its PRMS name, source, and unit config.
        This allows per-variable temporal data to be matched to forcing
        config without requiring all variables in a single dataset.

        Parameters
        ----------
        tables_dir : pathlib.Path
            Directory containing ``forcing_variables.yml``.

        Returns
        -------
        dict[str, dict[str, str]]
            Mapping from SIR name to config dict with keys:
            ``prms_name``, ``sir_unit``, ``intermediate_unit``, ``source``.

        Examples
        --------
        >>> lookup = deriv._build_sir_to_forcing_lookup(tables_dir)
        >>> lookup["pr_mm_mean"]
        {'prms_name': 'prcp', 'sir_unit': 'mm', 'intermediate_unit': 'mm', 'source': 'gridmet'}
        """
        config = self._load_lookup_table("forcing_variables", tables_dir)
        datasets_config = config["mapping"]

        lookup: dict[str, dict[str, str]] = {}
        for source_name, variables in datasets_config.items():
            for prms_name, var_cfg in variables.items():
                for required_key in ("sir_name", "sir_unit", "intermediate_unit"):
                    if required_key not in var_cfg:
                        raise ValueError(
                            f"forcing_variables.yml: source '{source_name}', "
                            f"variable '{prms_name}' is missing required key "
                            f"'{required_key}'. Available keys: "
                            f"{list(var_cfg.keys())}"
                        )
                sir_name = var_cfg["sir_name"]
                if sir_name in lookup:
                    logger.warning(
                        "Duplicate SIR name '%s' in forcing_variables.yml: "
                        "source '%s' (prms_name='%s') overwrites source '%s' "
                        "(prms_name='%s').",
                        sir_name,
                        source_name,
                        prms_name,
                        lookup[sir_name]["source"],
                        lookup[sir_name]["prms_name"],
                    )
                lookup[sir_name] = {
                    "prms_name": prms_name,
                    "sir_unit": var_cfg["sir_unit"],
                    "intermediate_unit": var_cfg["intermediate_unit"],
                    "source": source_name,
                }
        return lookup

    def _derive_forcing(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Merge SIR-normalized temporal forcing into derived dataset (step 7).

        Look up each per-variable temporal dataset in the reverse mapping
        from ``forcing_variables.yml``, rename to PRMS conventions
        (e.g., ``tmmx_C_mean`` -> ``tmax``), apply unit conversions
        (e.g., W/m² -> Langleys/day, mm -> inches), and merge time-series
        arrays into the parameter dataset.

        Multi-year temporal chunks (keyed with ``_YYYY`` suffixes like
        ``"pr_mm_mean_2020"``, ``"pr_mm_mean_2021"``) are concatenated
        along the time dimension before processing.

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing ``temporal`` datasets and the
            lookup tables directory.
        ds : xr.Dataset
            In-progress parameter dataset to augment with forcing
            variables.

        Returns
        -------
        xr.Dataset
            Dataset with temporal forcing variables (e.g., ``prcp``,
            ``tmax``, ``tmin``) merged in on dimensions ``(time, nhru)``.
            Returns ``ds`` unchanged if no temporal data is available.

        Notes
        -----
        This step runs late in the DAG (after step 14) because forcing
        variables have no downstream dependencies within the static
        parameter derivation.

        Unit conversions are defined in ``forcing_variables.yml`` per
        variable as ``sir_unit`` -> ``intermediate_unit`` pairs and
        dispatched through ``hydro_param.units.convert``.

        See Also
        --------
        _build_sir_to_forcing_lookup : Reverse mapping from SIR names.
        """
        if ctx.temporal is None or len(ctx.temporal) == 0:
            logger.info("No temporal data provided; skipping forcing generation.")
            return ds

        tables_dir = ctx.resolved_lookup_tables_dir
        sir_lookup = self._build_sir_to_forcing_lookup(tables_dir)

        # Group per-variable multi-year chunks: strip _YYYY suffix
        chunks_by_var: dict[str, list[xr.Dataset]] = {}
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            chunks_by_var.setdefault(base_name, []).append(tds)

        forced_count = 0
        for var_base, chunks in chunks_by_var.items():
            # Look up config for this SIR variable
            var_cfg = sir_lookup.get(var_base)
            if var_cfg is None:
                logger.debug(
                    "Temporal variable '%s' is not a forcing variable; skipping.",
                    var_base,
                )
                continue

            prms_name = var_cfg["prms_name"]
            sir_unit = var_cfg["sir_unit"]
            intermediate_unit = var_cfg["intermediate_unit"]

            merged = self._concat_temporal_chunks(chunks)

            if var_base not in merged:
                logger.warning(
                    "Forcing variable '%s' (SIR name '%s') not found in "
                    "temporal data after concat; skipping.",
                    prms_name,
                    var_base,
                )
                continue

            da = merged[var_base]

            # Unit conversion (SIR unit → intermediate unit)
            if sir_unit != intermediate_unit:
                try:
                    converted = convert(da.values.astype(np.float64), sir_unit, intermediate_unit)
                except KeyError:
                    logger.error(
                        "No unit conversion registered for '%s' → '%s' "
                        "(forcing variable '%s'). Register the conversion "
                        "in units.py or fix forcing_variables.yml.",
                        sir_unit,
                        intermediate_unit,
                        prms_name,
                    )
                    continue
                da = da.copy(data=converted)
                da.attrs["units"] = intermediate_unit

            # Align feature dimension to derived dataset
            target_dim = "nhru"
            feat_dims = [d for d in da.dims if d != "time"]
            if feat_dims and target_dim in ds.dims and feat_dims[0] != target_dim:
                da = da.rename({feat_dims[0]: target_dim})

            ds[prms_name] = da
            forced_count += 1

        if forced_count > 0:
            logger.info("Step 7: merged %d forcing variables.", forced_count)
        else:
            logger.warning(
                "Step 7: temporal data contained %d variable(s) but none "
                "matched forcing config. Expected SIR names: %s. "
                "Received: %s.",
                len(chunks_by_var),
                sorted(sir_lookup.keys()),
                sorted(chunks_by_var.keys()),
            )

        return ds

    # ------------------------------------------------------------------
    # Climate normals helpers (steps 10, 11)
    # ------------------------------------------------------------------

    def _compute_monthly_normals(
        self,
        ctx: DerivationContext,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute monthly mean tmax and tmin from temporal forcing data.

        Aggregate multi-year daily temperature data into 12 monthly means
        per HRU, converting from the SIR unit (°C) to PRMS internal
        units (°F).  Used by steps 10 (PET coefficients) and 11
        (transpiration timing).

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing ``temporal`` datasets and the
            lookup tables directory for forcing variable configuration.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            Tuple of ``(monthly_tmax, monthly_tmin)`` each with shape
            ``(12, nhru)`` in degrees Fahrenheit (°F), or ``None`` if
            temporal data is unavailable or lacks tmax/tmin variables.

        Notes
        -----
        Temperature conversion: °C -> °F via ``T_f = T_c * 9/5 + 32``.

        Requires full 12-month coverage in both tmax and tmin temporal
        data to produce reliable normals.  Returns ``None`` with a
        warning if either variable covers fewer than 12 months.

        For single-HRU datasets, the output is reshaped from ``(12,)``
        to ``(12, 1)`` to maintain consistent 2-D array shape.
        """
        if ctx.temporal is None or len(ctx.temporal) == 0:
            return None

        tables_dir = ctx.resolved_lookup_tables_dir
        sir_lookup = self._build_sir_to_forcing_lookup(tables_dir)

        # Find tmax and tmin SIR names from config
        prms_to_sir = {cfg["prms_name"]: sn for sn, cfg in sir_lookup.items()}
        tmax_sir = prms_to_sir.get("tmax")
        tmin_sir = prms_to_sir.get("tmin")

        if tmax_sir is None or tmin_sir is None:
            logger.warning(
                "Forcing config is missing tmax and/or tmin entries; "
                "cannot compute climate normals."
            )
            return None

        # Collect multi-year chunks for tmax and tmin independently
        tmax_chunks: list[xr.Dataset] = []
        tmin_chunks: list[xr.Dataset] = []
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            if base_name == tmax_sir:
                tmax_chunks.append(tds)
            elif base_name == tmin_sir:
                tmin_chunks.append(tds)

        if not tmax_chunks or not tmin_chunks:
            logger.warning("No tmax/tmin variables found in temporal data for climate normals.")
            return None

        tmax_merged = self._concat_temporal_chunks(tmax_chunks)
        tmin_merged = self._concat_temporal_chunks(tmin_chunks)

        if tmax_sir not in tmax_merged or tmin_sir not in tmin_merged:
            logger.warning(
                "Temporal data missing tmax='%s' or tmin='%s' after concat.",
                tmax_sir,
                tmin_sir,
            )
            return None

        # Group by month and compute mean
        tmax_monthly = tmax_merged[tmax_sir].groupby("time.month").mean(dim="time")
        tmin_monthly = tmin_merged[tmin_sir].groupby("time.month").mean(dim="time")

        # Validate full 12-month coverage for both variables
        n_tmax_months = tmax_monthly.sizes.get("month", 0)
        n_tmin_months = tmin_monthly.sizes.get("month", 0)
        if n_tmax_months != 12 or n_tmin_months != 12:
            logger.warning(
                "Temporal data covers only %d (tmax) / %d (tmin) of "
                "12 months; cannot compute reliable monthly normals. "
                "Skipping.",
                n_tmax_months,
                n_tmin_months,
            )
            return None

        tmax_f = tmax_monthly.values * 9.0 / 5.0 + 32.0
        tmin_f = tmin_monthly.values * 9.0 / 5.0 + 32.0

        # Ensure 2-D shape (12, nhru) for single-HRU case
        if tmax_f.ndim == 1:
            tmax_f = tmax_f[:, np.newaxis]
            tmin_f = tmin_f[:, np.newaxis]

        logger.info(
            "Computed monthly climate normals from tmax='%s', tmin='%s' "
            "(%d + %d timesteps, %d HRUs).",
            tmax_sir,
            tmin_sir,
            tmax_merged.sizes.get("time", 0),
            tmin_merged.sizes.get("time", 0),
            tmax_f.shape[1],
        )
        return tmax_f, tmin_f

    # ------------------------------------------------------------------
    # Step 10: PET coefficients (Jensen-Haise)
    # ------------------------------------------------------------------

    def _derive_pet_coefficients(
        self,
        ds: xr.Dataset,
        normals: tuple[np.ndarray, np.ndarray] | None,
    ) -> xr.Dataset:
        """Derive Jensen-Haise PET coefficients from climate normals (step 10).

        Compute monthly ``jh_coef`` (= 1/Ct) and per-HRU ``jh_coef_hru``
        (= Tx, temperature threshold in °F) using the PRMS-IV equation 1-26
        (Markstrom et al. 2015).

        ``jh_coef = 1 / Ct`` where ``Ct = C1 + 13 * Ch``
        (see Notes for full derivation).

        ``jh_coef_hru = Tx = -2.5 - 0.14 * (e_july_max - e_july_min) - elev_ft / 1000``

        The Jensen-Haise PET equation uses these as:
        ``PET = jh_coef * (T_avg_F - jh_coef_hru) * swrad / elh``

        Falls back to step 13 scalar defaults when no temporal data is
        available (normals is ``None``).

        Parameters
        ----------
        ds : xr.Dataset
            In-progress parameter dataset.  ``hru_elev`` (meters) is used
            for elevation adjustment of ``jh_coef_hru`` if available.
        normals : tuple[np.ndarray, np.ndarray] or None
            Monthly climate normals ``(monthly_tmax, monthly_tmin)`` each
            with shape ``(12, nhru)`` in degrees Fahrenheit (°F), as
            returned by ``_compute_monthly_normals``.

        Returns
        -------
        xr.Dataset
            Dataset with ``jh_coef`` (per_degF_per_day, shape
            ``(nmonth, nhru)``) and ``jh_coef_hru``
            (degrees_fahrenheit, shape ``(nhru,)``) added.

        Notes
        -----
        The Jensen-Haise coefficient (Jensen et al. 1970, PRMS-IV eq. 1-26):

        ``C1 = 68 - 3.6 * (elev_ft / 1000)``
        ``Ch = 50 / (e2 - e1)``
        ``Ct = C1 + 13 * Ch``
        ``jh_coef = 1 / Ct``

        where ``e2`` and ``e1`` are saturation vapor pressures (mb) at
        monthly mean tmax and tmin respectively.  ``C1`` is the elevation
        correction term.  Computed per month to capture seasonal variation.
        Values are clipped to ``[0.005, 0.06]``.

        ``jh_coef_hru`` (Tx) is the temperature threshold in the PET
        equation, computed from July vapor pressure extremes and elevation:
        ``Tx = -2.5 - 0.14 * (e_july_max - e_july_min) - elev_ft / 1000``.

        References
        ----------
        Markstrom, S. L., et al. (2015). PRMS-IV. USGS TM 6-B7, eq. 1-26.
        Jensen, M. E., Haise, H. R. (1963). Estimating evapotranspiration
            from solar radiation. J. Irrig. Drain. Div., 89, 15-41.
        Jensen, M. E., Robb, D. C. N., Franzoy, C. E. (1970). Scheduling
            irrigations using climate-crop-soil data. J. Irrig. Drain.
            Div., 96(1), 25-38.

        See Also
        --------
        _sat_vp : Saturation vapor pressure helper.
        _compute_monthly_normals : Climate normals computation.
        """
        if normals is None:
            logger.info("No temporal data for PET coefficients; deferring to defaults.")
            return ds

        monthly_tmax, monthly_tmin = normals  # (12, nhru) in °F
        nhru = monthly_tmax.shape[1]

        # --- Elevation for both jh_coef and jh_coef_hru ---
        if "hru_elev" in ds:
            elev_m = ds["hru_elev"].values
            elev_ft = elev_m / 0.3048
        else:
            logger.warning(
                "hru_elev not in dataset; PET coefficients will be computed assuming "
                "sea-level elevation (0 ft). This affects both jh_coef and jh_coef_hru. "
                "Ensure step 3 (topography) ran successfully.",
            )
            elev_ft = np.zeros(nhru)

        # --- jh_coef: Jensen-Haise (Jensen et al. 1970, PRMS-IV eq. 1-26) ---
        # Ct = C1 + 13*Ch  where:
        #   C1 = 68 - 3.6 * (elev_ft / 1000)  [elevation correction]
        #   Ch = 50 / (e2 - e1)                [humidity index, mb]
        # e2 = SVP at monthly mean tmax, e1 = SVP at monthly mean tmin
        # jh_coef = 1/Ct
        c1 = 68.0 - 3.6 * (elev_ft / 1000.0)  # (nhru,)

        jh_coef = np.zeros((12, nhru))
        for m in range(12):
            e2 = _sat_vp(monthly_tmax[m, :])  # SVP at tmax (hPa ≈ mb)
            e1 = _sat_vp(monthly_tmin[m, :])  # SVP at tmin
            de = np.maximum(e2 - e1, 0.01)  # guard against zero
            ch = 50.0 / de
            ct = c1 + 13.0 * ch
            ct_safe = np.maximum(ct, 1e-6)
            jh_coef[m, :] = 1.0 / ct_safe

        jh_coef = np.clip(jh_coef, 0.005, 0.06)

        ds["jh_coef"] = xr.DataArray(
            jh_coef,
            dims=("nmonth", "nhru"),
            attrs={"units": "per_degF_per_day", "long_name": "Jensen-Haise PET coefficient"},
        )

        # --- jh_coef_hru: Jensen-Haise temperature threshold Tx (°F) ---
        # Tx = -2.5 - 0.14*(e_july_max - e_july_min) - elev_ft/1000
        # Used in PET equation: PET = jh_coef * (T_avg_F - jh_coef_hru) * swrad / elh
        svp_july_max = _sat_vp(monthly_tmax[6, :])  # (nhru,)
        svp_july_min = _sat_vp(monthly_tmin[6, :])  # (nhru,)

        jh_coef_hru = -2.5 - 0.14 * (svp_july_max - svp_july_min) - elev_ft / 1000.0
        ds["jh_coef_hru"] = xr.DataArray(
            jh_coef_hru,
            dims=("nhru",),
            attrs={
                "units": "degrees_fahrenheit",
                "long_name": "Jensen-Haise temperature threshold (Tx)",
            },
        )

        logger.info(
            "Step 10: derived jh_coef (%d HRUs x 12 months) and jh_coef_hru.",
            nhru,
        )
        return ds

    # ------------------------------------------------------------------
    # Step 11: Transpiration timing (frost-free period)
    # ------------------------------------------------------------------

    def _derive_transp_timing(
        self,
        ds: xr.Dataset,
        normals: tuple[np.ndarray, np.ndarray] | None,
    ) -> xr.Dataset:
        """Derive transpiration onset and offset from monthly tmin (step 11).

        Compute ``transp_beg`` (month transpiration begins) and
        ``transp_end`` (month transpiration ends) by detecting the
        frost-free period from monthly minimum temperature normals.

        ``transp_beg`` is the first month (1-indexed, Jan=1) where
        monthly mean tmin exceeds 32 °F.  ``transp_end`` is the first
        month from July onward where monthly mean tmin drops below 32 °F.

        Falls back to step 13 defaults (``transp_beg=4``,
        ``transp_end=10``) when no temporal data is available.

        Parameters
        ----------
        ds : xr.Dataset
            In-progress parameter dataset.
        normals : tuple[np.ndarray, np.ndarray] or None
            Monthly climate normals ``(monthly_tmax, monthly_tmin)`` each
            with shape ``(12, nhru)`` in degrees Fahrenheit (°F).

        Returns
        -------
        xr.Dataset
            Dataset with ``transp_beg`` (integer month, 1--12 on
            ``nhru``) and ``transp_end`` (integer month, 7--12 on
            ``nhru``) added.

        Notes
        -----
        HRUs that never exceed freezing in any month default to
        ``transp_beg=4`` (April).  HRUs that never drop below freezing
        from July onward default to ``transp_end=10`` (October).
        Both conditions are logged as warnings.

        The freezing threshold is 32 °F (0 °C).  This is a simplified
        approach; more sophisticated phenology models could improve
        accuracy in transitional climates.
        """
        if normals is None:
            logger.info("No temporal data for transpiration timing; deferring to defaults.")
            return ds

        _monthly_tmax, monthly_tmin = normals  # (12, nhru) in °F
        nhru = monthly_tmin.shape[1]
        freezing = 32.0  # °F

        # Check for NaN in monthly tmin
        nan_count = int(np.count_nonzero(np.isnan(monthly_tmin)))
        if nan_count:
            logger.warning(
                "monthly_tmin contains %d NaN values across %d HRUs x 12 months; "
                "affected HRUs will use default transp_beg/transp_end values.",
                nan_count,
                nhru,
            )

        # transp_beg: first month (1-indexed) where tmin > freezing (vectorized)
        above_freezing = monthly_tmin > freezing  # (12, nhru) boolean
        has_warm_month = above_freezing.any(axis=0)
        transp_beg = np.where(has_warm_month, np.argmax(above_freezing, axis=0) + 1, 4).astype(
            np.int32
        )

        fallback_beg = int(nhru - np.count_nonzero(has_warm_month))
        if fallback_beg > 0:
            logger.warning(
                "transp_beg: %d of %d HRUs never exceeded freezing in any "
                "month; using default transp_beg=4 for those HRUs.",
                fallback_beg,
                nhru,
            )

        # transp_end: first month from July onward where tmin < freezing
        below_freezing_late = monthly_tmin[6:, :] < freezing  # (6, nhru)
        has_cold_month = below_freezing_late.any(axis=0)
        transp_end = np.where(
            has_cold_month, np.argmax(below_freezing_late, axis=0) + 7, 10
        ).astype(np.int32)

        fallback_end = int(nhru - np.count_nonzero(has_cold_month))
        if fallback_end > 0:
            logger.warning(
                "transp_end: %d of %d HRUs never dropped below freezing "
                "Jul-Dec; using default transp_end=10 for those HRUs.",
                fallback_end,
                nhru,
            )

        ds["transp_beg"] = xr.DataArray(
            transp_beg,
            dims=("nhru",),
            attrs={"units": "integer_month", "long_name": "Month transpiration begins"},
        )
        ds["transp_end"] = xr.DataArray(
            transp_end,
            dims=("nhru",),
            attrs={"units": "integer_month", "long_name": "Month transpiration ends"},
        )

        logger.info(
            "Step 11: derived transp_beg (range %d-%d) and transp_end (range %d-%d) for %d HRUs.",
            int(transp_beg.min()),
            int(transp_beg.max()),
            int(transp_end.min()),
            int(transp_end.max()),
            nhru,
        )
        return ds

    # ------------------------------------------------------------------
    # Parameter overrides
    # ------------------------------------------------------------------

    def _apply_overrides(self, ds: xr.Dataset, overrides: dict) -> xr.Dataset:
        """Apply user-specified parameter value overrides.

        Replace or add parameter values from the ``parameter_overrides``
        section of the pipeline configuration.  Overrides are applied
        **last**, after all derivation steps and calibration seeds, giving
        users final control over any parameter value.

        Parameters
        ----------
        ds : xr.Dataset
            In-progress parameter dataset.
        overrides : dict
            Mapping of parameter names to override values.  Values can be
            scalars (broadcast) or lists matching the parameter dimension
            length.

        Returns
        -------
        xr.Dataset
            Dataset with overridden parameter values applied.

        Notes
        -----
        Existing parameters are updated in-place; scalar overrides are
        broadcast to match the target variable's shape (e.g., a scalar
        override for a ``(nmonth, nhru)`` parameter fills all elements).
        New parameters are created with an ``nhru`` dimension if the
        value is a list, or as a scalar otherwise.
        """
        for param_name, value in overrides.items():
            if isinstance(value, list):
                arr: np.floating | np.ndarray = np.array(value, dtype=np.float64)
            else:
                arr = np.array(value, dtype=np.float64)

            if param_name in ds:
                # Broadcast scalar to match existing variable shape
                if arr.ndim == 0 and ds[param_name].ndim > 0:
                    arr = np.full(ds[param_name].shape, arr, dtype=arr.dtype)
                try:
                    ds[param_name].values = arr
                except ValueError as exc:
                    raise ValueError(
                        f"Override for '{param_name}': array shape {arr.shape} "
                        f"does not match existing parameter shape "
                        f"{ds[param_name].shape}. Provide either a scalar or "
                        f"an array with the correct shape."
                    ) from exc
                logger.info("Override: %s = %s", param_name, value)
            else:
                dims = ("nhru",) if arr.ndim == 1 else ()
                ds[param_name] = xr.DataArray(
                    arr,
                    dims=dims if dims else None,
                    attrs={"long_name": f"{param_name} (user override)"},
                )
                logger.info("Override (new): %s = %s", param_name, value)
        return ds

    # ------------------------------------------------------------------
    # Lookup table loader
    # ------------------------------------------------------------------

    def _load_lookup_table(self, name: str, tables_dir: Path) -> dict:
        """Load and cache a YAML lookup table from the lookup tables directory.

        Load the named YAML file, validate that it contains a ``mapping``
        key, and cache the result for subsequent calls within the same
        ``PywatershedDerivation`` instance.

        Parameters
        ----------
        name : str
            Lookup table filename without the ``.yml`` extension
            (e.g., ``"nlcd_to_prms_cov_type"``).
        tables_dir : Path
            Directory containing lookup table YAML files.

        Returns
        -------
        dict
            Parsed YAML content with at least a ``mapping`` key.

        Raises
        ------
        FileNotFoundError
            If ``{name}.yml`` does not exist in ``tables_dir``.
        ValueError
            If the YAML file is malformed or missing the required
            ``mapping`` key.

        Notes
        -----
        Results are cached in ``self._lookup_cache`` keyed by ``name``.
        The cache persists for the lifetime of the instance, so lookup
        tables are read from disk at most once per derivation run.
        """
        if name not in self._lookup_cache:
            path = tables_dir / f"{name}.yml"
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Lookup table '{name}.yml' not found at '{path}'. "
                    f"Verify lookup_tables_dir is correct or use the default bundled tables."
                ) from None
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Lookup table '{name}.yml' at '{path}' contains invalid YAML: {exc}"
                ) from exc
            if not isinstance(data, dict) or "mapping" not in data:
                raise ValueError(
                    f"Lookup table '{name}.yml' at '{path}' is missing required 'mapping' key."
                )
            self._lookup_cache[name] = data
        return self._lookup_cache[name]
