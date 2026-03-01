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
import pyproj
import xarray as xr
import yaml

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
    "jh_coef_hru": 0.014,
    # Transpiration timing
    "transp_beg": 4,  # April
    "transp_end": 10,  # October
    # Depression storage — hru_type only; dprst_frac and dprst_area_max are
    # always set by _derive_waterbody (or _waterbody_defaults), so no scalar
    # fallback is needed here.
    "hru_type": 1,
}

# Parameters with non-scalar defaults handled specially in _apply_defaults
_DEFAULTS_SPECIAL: frozenset[str] = frozenset({"jh_coef", "transp_beg", "transp_end", "hru_type"})

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
            for soltab, ``nmonths`` for monthly parameters, ``time`` for
            forcing).

        Notes
        -----
        Step execution order: 1 (geometry) -> 2 (topology) -> 3 (topo) ->
        4 (landcover) -> 5 (soils) -> 6 (waterbody) -> 8 (lookups) ->
        12 (routing) -> 9 (soltab) -> 10 (PET) -> 11 (transp) ->
        13 (defaults) -> 14 (calibration) -> 7 (forcing) -> overrides.

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

        # Step 4: Land cover parameters (cov_type, covden_sum, hru_percent_imperv)
        ds = self._derive_landcover(context, ds)

        # Step 5: Soils parameters (soil_type, soil_moist_max, soil_rechr_max_frac)
        ds = self._derive_soils(context, ds)

        # Step 6: Waterbody overlay (dprst_frac, dprst_area_max, hru_type)
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
        """Compute HRU area and centroid latitude from the target fabric (step 1).

        Derive ``hru_area`` (acres) and ``hru_lat`` (decimal degrees) from
        the fabric GeoDataFrame geometry.  Area is computed in EPSG:5070
        (NAD83 CONUS Albers equal-area) and converted from m² to acres.
        Latitude is extracted from EPSG:4326 (WGS84) centroids.

        Falls back to SIR variables ``hru_area_m2`` and ``hru_lat`` when
        fabric is ``None`` or lacks the ``id_field`` column.

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
            Dataset with ``hru_area`` (acres) and ``hru_lat``
            (decimal degrees) added on the ``nhru`` dimension.

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

        Read ``tosegment``, ``hru_segment``, and ``seg_length`` from the
        Geospatial Fabric GeoDataFrames.  These define the stream-segment
        routing network and HRU-to-segment flow contributions used by
        PRMS for Muskingum routing.

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
            Dataset with ``tosegment`` (dimensionless index on ``nsegment``),
            ``hru_segment`` (dimensionless index on ``nhru``), and
            ``seg_length`` (meters on ``nsegment``) added.  Returns ``ds``
            unchanged if ``fabric`` or ``segments`` is ``None``.

        Raises
        ------
        ValueError
            If ``tosegment`` column is missing from segments, ``hru_segment``
            column is missing from fabric, tosegment contains self-loops or
            out-of-range values, or no outlet segments (tosegment == 0) exist.
        KeyError
            If an explicitly configured ``segment_id_field`` is not found in
            the segments GeoDataFrame columns.

        Notes
        -----
        Topology is model-specific and comes directly from the fabric
        GeoDataFrames --- hydro-param does not normalize between topology
        conventions.  The ``tosegment`` array uses 1-based indexing with
        0 indicating an outlet segment.
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

        For each segment, find all NHDPlus flowlines that intersect it,
        compute the intersection length, and return a length-weighted
        mean slope.  This handles GF/PRMS segments that have been
        post-processed from NHD (split at POIs, trimmed to catchment
        boundaries) and therefore lack COMIDs for a direct VAA join.

        Parameters
        ----------
        segments : gpd.GeoDataFrame
            Segment GeoDataFrame (no COMID column).  Must have line
            geometries and a CRS set.
        nhd_flowlines : gpd.GeoDataFrame
            NHDPlus flowlines with ``slope`` column (dimensionless
            rise/run, decimal fraction) and line geometries.

        Returns
        -------
        np.ndarray
            Length-weighted mean slope per segment.  Segments with no
            intersecting NHD flowlines receive ``_FALLBACK_SLOPE``.

        Notes
        -----
        The weighting uses actual intersection geometry length rather
        than total flowline length.  This correctly handles partial
        overlaps where a segment only intersects part of a longer NHD
        reach.

        CRS alignment is enforced: if the two GeoDataFrames differ in
        CRS, the NHD flowlines are reprojected to match the segments.

        See Also
        --------
        _get_slopes_from_comid : Primary slope lookup via COMID join.
        _FALLBACK_SLOPE : Default slope for unmatched segments.
        """
        # Ensure same CRS
        if segments.crs != nhd_flowlines.crs:
            nhd_flowlines = nhd_flowlines.to_crs(segments.crs)

        # Reset index to use positional alignment
        segs = segments.reset_index(drop=True)
        nhd = nhd_flowlines.reset_index(drop=True)

        # Spatial join — find all NHD flowlines intersecting each segment
        joined = gpd.sjoin(segs, nhd, how="left", predicate="intersects")

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

            # Compute intersection lengths for weighting
            seg_geom = segs.geometry.iloc[seg_idx]
            weights: list[float] = []
            match_slopes: list[float] = []
            for _, row in matches.iterrows():
                nhd_idx = int(row["index_right"])
                nhd_geom = nhd.geometry.iloc[nhd_idx]
                try:
                    intersection = seg_geom.intersection(nhd_geom)
                except Exception:
                    logger.warning(
                        "Geometry intersection failed for segment %d with NHD index %d; skipping",
                        seg_idx,
                        nhd_idx,
                    )
                    continue
                if intersection.is_empty:
                    continue
                weights.append(intersection.length)
                match_slopes.append(row["slope"])

            if weights:
                weights_arr = np.array(weights)
                match_slopes_arr = np.array(match_slopes)
                total_weight = weights_arr.sum()
                if total_weight > 0:
                    slopes[seg_idx] = np.average(match_slopes_arr, weights=weights_arr)
                    matched[seg_idx] = True

        n_fallback = int(np.sum(~matched))
        if n_fallback > 0:
            logger.warning(
                "%d of %d segments have no intersecting NHDPlus flowlines; "
                "using fallback slope %.1e",
                n_fallback,
                len(segs),
                _FALLBACK_SLOPE,
            )
        return slopes

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
            VAA table with ``comid`` and ``slope`` columns (NaN slopes
            removed), or ``None`` if pynhd is not installed or the
            download fails.

        Notes
        -----
        pynhd is an optional dependency.  When it is not installed this
        method logs a warning and returns ``None`` so that the caller
        can fall back to default slope values.

        References
        ----------
        McKay, L., et al. (2012). NHDPlus Version 2: User Guide.
            https://www.epa.gov/waterdata/nhdplus-national-data
        """
        try:
            import pynhd as nhd
        except ImportError:
            logger.warning("pynhd not installed; cannot fetch NHDPlus slopes")
            return None

        try:
            vaa = nhd.nhdplus_vaa()
            return vaa[["comid", "slope"]].dropna(subset=["slope"])
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
            import pynhd as nhd
        except ImportError:
            return None

        try:
            bbox = segments.to_crs("EPSG:4326").total_bounds
            wd = nhd.WaterData("nhdflowline_network")
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
        ``segment_type``, and ``obsin_segment``.

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
            ``seg_slope`` (m/m), ``segment_type`` (integer), and
            ``obsin_segment`` (integer) added on ``nsegment``.
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
            attrs={"units": "hours", "long_name": "Muskingum storage time coefficient"},
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

        Transform 3DEP-derived zonal statistics from SI units to PRMS
        conventions:

        - ``elevation_m_mean`` (meters) -> ``hru_elev`` (feet)
        - ``slope_deg_mean`` (degrees) -> ``hru_slope`` (decimal fraction = tan(slope))
        - ``aspect_deg_mean`` (degrees) -> ``hru_aspect`` (degrees, unchanged)

        Parameters
        ----------
        ctx : DerivationContext
            Derivation context providing the SIR dataset.
        ds : xr.Dataset
            In-progress parameter dataset to augment.

        Returns
        -------
        xr.Dataset
            Dataset with ``hru_elev`` (feet), ``hru_slope`` (decimal
            fraction), and ``hru_aspect`` (degrees) on the ``nhru``
            dimension.  Only variables with corresponding SIR input are
            added.

        Notes
        -----
        Unit conversions: meters -> feet (``convert(m, ft)``),
        degrees -> radians -> tan() for slope.

        PRMS slope is rise/run (dimensionless), not an angle.  The
        conversion is ``tan(slope_rad)`` where ``slope_rad`` is the
        slope angle in radians.
        """
        sir = ctx.sir
        if "elevation_m_mean" in sir:
            ds["hru_elev"] = xr.DataArray(
                convert(sir["elevation_m_mean"].values, "m", "ft"),
                dims="nhru",
                attrs={"units": "feet", "long_name": "Mean HRU elevation"},
            )

        if "slope_deg_mean" in sir:
            # SIR slope is in degrees; PRMS wants decimal fraction (rise/run)
            slope_rad = convert(sir["slope_deg_mean"].values, "deg", "rad")
            ds["hru_slope"] = xr.DataArray(
                np.tan(slope_rad),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "Mean HRU slope"},
            )

        if "aspect_deg_mean" in sir:
            ds["hru_aspect"] = xr.DataArray(
                sir["aspect_deg_mean"].values.astype(np.float64),
                dims="nhru",
                attrs={"units": "degrees", "long_name": "Mean HRU aspect"},
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

        Supports three input modes:

        1. **Categorical fractions** (preferred): SIR contains columns
           like ``lndcov_frac_11``, ``lndcov_frac_21``, etc. from
           normalized categorical zonal output.  The majority class is
           computed via argmax across fraction columns.
        2. **Single majority value**: ``land_cover`` or
           ``land_cover_majority`` variable containing the dominant
           NLCD class code per HRU.
        3. **Continuous auxiliary layers**: ``fctimp_pct_mean`` (0--100%)
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
        _compute_majority_from_fractions : Argmax over fraction columns.
        """
        sir = ctx.sir
        # Try categorical fractions first (e.g., LndCov_11, LndCov_21, ...)
        lc_var = None
        nlcd_values = self._compute_majority_from_fractions(sir)

        if nlcd_values is None:
            # Fallback to single majority value
            for candidate in ("land_cover", "land_cover_majority"):
                if candidate in sir:
                    lc_var = candidate
                    break

        has_lc = nlcd_values is not None or lc_var is not None
        if nlcd_values is not None:
            nlcd_table = self._load_lookup_table(
                "nlcd_to_prms_cov_type", ctx.resolved_lookup_tables_dir
            )
            mapping = nlcd_table["mapping"]
            cov_type = np.array([mapping.get(int(v), 0) for v in nlcd_values])
        elif lc_var is not None:
            nlcd_table = self._load_lookup_table(
                "nlcd_to_prms_cov_type", ctx.resolved_lookup_tables_dir
            )
            mapping = nlcd_table["mapping"]
            nlcd_values_lc = sir[lc_var].values.astype(int)
            cov_type = np.array([mapping.get(int(v), 0) for v in nlcd_values_lc])

        if has_lc:
            ds["cov_type"] = xr.DataArray(
                cov_type,
                dims="nhru",
                attrs={"units": "integer", "long_name": "Vegetation cover type"},
            )

        if "tree_canopy_pct_mean" in sir:
            # Continuous canopy cover (0-100%) -> fraction (0-1)
            ds["covden_sum"] = xr.DataArray(
                np.clip(sir["tree_canopy_pct_mean"].values / 100.0, 0.0, 1.0),
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

        fctimp_key = sir.find_variable("fctimp_pct_mean")
        if fctimp_key is not None:
            # Percent (0-100) -> fraction (0-1)
            ds["hru_percent_imperv"] = xr.DataArray(
                np.clip(sir[fctimp_key].values / 100.0, 0.0, 1.0),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "HRU impervious fraction"},
            )

        return ds

    @staticmethod
    def _compute_majority_from_fractions(
        sir: SIRAccessor,
        prefixes: tuple[str, ...] = ("lndcov_frac_",),
    ) -> np.ndarray | None:
        """Compute majority NLCD class from categorical fraction columns.

        Scan SIR variables for columns matching ``{prefix}{class_code}``
        (e.g., ``lndcov_frac_11``, ``lndcov_frac_41``).  For each HRU,
        return the class code with the highest fraction via ``np.argmax``.

        This method supports two SIR layouts:

        1. **Column-level keys** — each fraction is a separate SIR variable
           (e.g., ``lndcov_frac_11``, ``lndcov_frac_41``).  This is the
           normalized categorical output from gdptools ``ZonalGen``.
        2. **File-level keys** — ``data_vars`` contains a year-suffixed
           entry like ``lndcov_frac_2021``.  The individual fraction
           columns (``lndcov_frac_2021_11``, ``lndcov_frac_2021_41``,
           etc.) are inside the backing file and accessed via
           ``sir.load_dataset()``.

        Parameters
        ----------
        sir : SIRAccessor
            SIR dataset potentially containing fraction columns.
        prefixes : tuple[str, ...]
            Variable name prefixes to search for (default:
            ``("lndcov_frac_",)``).

        Returns
        -------
        np.ndarray or None
            Array of majority NLCD class codes (int) with shape
            ``(nhru,)``, or ``None`` if fewer than 2 fraction columns
            are found for any prefix.

        Notes
        -----
        Suffixes that cannot be parsed as integers are silently skipped
        with a debug log message.  At least 2 valid fraction columns
        are required to compute a meaningful majority.

        When a suffix parses as an integer but exceeds 95 (the maximum
        NLCD class code), it is treated as a year-suffixed file-level
        key.  The method calls ``sir.load_dataset()`` to retrieve the
        inner columns and searches them for ``{file_key}_{class_code}``
        patterns.
        """
        for prefix in prefixes:
            fraction_vars = sorted(v for v in sir.data_vars if v.startswith(prefix))
            if not fraction_vars:
                continue

            # Extract class codes from suffixes.  Fraction values are
            # collected eagerly so that file-level datasets can be released
            # immediately and multiple year-suffixed files don't overwrite
            # each other.
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
                    # Load the backing file and extract inner fraction columns.
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
                        class_codes.append(inner_code)
                        fractions_list.append(inner_ds[inner_name].values)
                else:
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

            fractions = np.column_stack(fractions_list)
            codes = np.array(class_codes)
            majority_idx = np.argmax(fractions, axis=1)
            majority_class = codes[majority_idx]

            logger.info(
                "Computed majority class from %d categorical fraction columns (prefix=%r)",
                len(class_codes),
                prefix,
            )
            return majority_class

        return None

    # ------------------------------------------------------------------
    # Step 5: Soils zonal stats
    # ------------------------------------------------------------------

    _SOIL_RECHR_MAX_FRAC_DEFAULT: float = 0.4

    def _derive_soils(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Derive soil parameters from gNATSGO/STATSGO2 zonal statistics (step 5).

        Classify soil texture into PRMS ``soil_type`` and derive
        ``soil_moist_max`` (maximum soil moisture capacity) from
        available water capacity (``awc_mm_mean``) or, as a fallback,
        available water storage (``aws0_100_cm_mean``).

        Supports two input modes for soil texture:

        1. **Soil texture fractions** (preferred): SIR contains columns
           like ``soil_texture_frac_sand``, ``soil_texture_frac_loam``,
           etc.  Majority class determined via argmax, then reclassified
           to PRMS soil_type using ``soil_texture_to_prms_type.yml``.
        2. **Single texture class**: ``soil_texture`` or
           ``soil_texture_majority`` variable containing the dominant
           USDA texture class name per HRU.

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

        - ``awc_mm_mean``: mm -> inches via ``convert(mm, in)``.
        - ``aws0_100_cm_mean`` (fallback): cm -> mm (* 10) -> inches via
          ``convert(mm, in)``.

        ``soil_moist_max`` is clipped to ``[0.5, 20.0]`` inches in both cases.

        ``soil_rechr_max_frac`` is set to a constant default of 0.4
        (no soil layer depth data is currently available from the SIR
        to compute it from first principles).

        See Also
        --------
        _compute_soil_type : Texture classification helper.
        """
        sir = ctx.sir

        # --- soil_type ---
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
                "found in SIR. Expected soil_texture_frac_* columns or "
                "soil_texture/soil_texture_majority variable."
            )

        # --- soil_moist_max ---
        if "awc_mm_mean" in sir:
            awc_mm = sir["awc_mm_mean"].values.astype(np.float64)
            soil_moist_max = convert(awc_mm, "mm", "in")
            soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
            ds["soil_moist_max"] = xr.DataArray(
                soil_moist_max,
                dims="nhru",
                attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
            )
        elif "aws0_100_cm_mean" in sir:
            # Available water storage in cm — convert to mm first, then to inches.
            aws_cm = sir["aws0_100_cm_mean"].values.astype(np.float64)
            awc_mm = aws_cm * 10.0  # cm -> mm
            soil_moist_max = convert(awc_mm, "mm", "in")
            soil_moist_max = np.clip(soil_moist_max, 0.5, 20.0)
            ds["soil_moist_max"] = xr.DataArray(
                soil_moist_max,
                dims="nhru",
                attrs={"units": "inches", "long_name": "Maximum soil moisture capacity"},
            )
            logger.info("Used aws0_100_cm_mean (cm -> mm -> in) for soil_moist_max")
        else:
            logger.warning(
                "Skipping soil_moist_max derivation (step 5): neither 'awc_mm_mean' "
                "nor 'aws0_100_cm_mean' found in SIR."
            )

        # --- soil_rechr_max_frac ---
        if "soil_type" in ds:
            nhru = len(ds["soil_type"])
            ds["soil_rechr_max_frac"] = xr.DataArray(
                np.full(nhru, self._SOIL_RECHR_MAX_FRAC_DEFAULT),
                dims="nhru",
                attrs={
                    "units": "decimal_fraction",
                    "long_name": "Fraction of soil moisture in recharge zone",
                },
            )
            logger.debug(
                "soil_rechr_max_frac set to default %.2f for %d HRUs "
                "(no soil layer data available)",
                self._SOIL_RECHR_MAX_FRAC_DEFAULT,
                nhru,
            )

        return ds

    def _compute_soil_type(self, sir: SIRAccessor, ctx: DerivationContext) -> np.ndarray | None:
        """Compute PRMS soil_type from SIR soil texture data.

        Try fraction columns first (argmax across texture classes), then
        fall back to a single texture class variable.  Unrecognized
        texture names default to loam (soil_type=2).

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
        """
        # Check data availability before loading lookup table
        prefix = "soil_texture_frac_"
        fraction_vars = sorted(v for v in sir.data_vars if v.startswith(prefix))
        has_single = any(c in sir for c in ("soil_texture", "soil_texture_majority"))

        if len(fraction_vars) < 2 and not has_single:
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

        return None

    # ------------------------------------------------------------------
    # Step 6: Waterbody overlay (depression storage)
    # ------------------------------------------------------------------

    def _waterbody_defaults(self, ds: xr.Dataset, nhru: int) -> xr.Dataset:
        """Assign zero/default waterbody parameters when no overlay data exists.

        Set ``dprst_frac`` and ``dprst_area_max`` to zero, and
        ``hru_type`` to 1 (land) for all HRUs.  Used as a fallback
        when waterbody data, fabric, or ``hru_area`` is unavailable.

        Parameters
        ----------
        ds : xr.Dataset
            In-progress parameter dataset to augment.
        nhru : int
            Number of HRUs for array dimensioning.

        Returns
        -------
        xr.Dataset
            Dataset with ``dprst_frac`` (fraction), ``dprst_area_max``
            (acres), and ``hru_type`` (dimensionless integer) set to
            defaults on the ``nhru`` dimension.
        """
        ds["dprst_frac"] = xr.DataArray(
            np.zeros(nhru),
            dims="nhru",
            attrs={"units": "fraction", "long_name": "Depression storage fraction of HRU area"},
        )
        ds["dprst_area_max"] = xr.DataArray(
            np.zeros(nhru),
            dims="nhru",
            attrs={"units": "acres", "long_name": "Maximum depression storage area"},
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
        compute depression storage fraction, maximum depression area, and
        HRU type classification (land vs. lake).

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
            Dataset with ``dprst_frac`` (decimal fraction 0--1 on ``nhru``),
            ``dprst_area_max`` (acres on ``nhru``), and ``hru_type``
            (integer: 1=land, 2=lake on ``nhru``).  Falls back to zero
            defaults if any prerequisite data is missing.

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
                "for depression storage parameters (dprst_frac, dprst_area_max). "
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
        ds["dprst_area_max"] = xr.DataArray(
            clipped_acres,
            dims="nhru",
            attrs={"units": "acres", "long_name": "Maximum depression storage area"},
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

        Use ``cov_type`` (from step 4) to look up per-HRU interception
        capacities (``srain_intcp``, ``wrain_intcp``, ``snow_intcp``) and
        compute winter cover density (``covden_win``) by applying a
        cov_type-dependent reduction factor to ``covden_sum``.

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

        # Interception capacities
        intcp_table = self._load_lookup_table("cov_type_to_interception", tables_dir)
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
            attrs={"units": "hours", "long_name": "Hours of direct sunlight"},
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
        Special-case parameters that require array shapes:

        - ``jh_coef``: shape ``(nhru, 12)`` --- per_degF_per_day
        - ``transp_beg``, ``transp_end``: shape ``(nhru,)`` --- integer month
        - ``hru_type``: shape ``(nhru,)`` --- integer (1=land, 2=lake)

        All other defaults are scalar values broadcast by xarray as needed.
        Default units match PRMS conventions: inches for storage depths,
        degree-days for ``transp_tmax``, etc.

        References
        ----------
        Regan, R. S., et al. (2018). USGS Techniques and Methods 6-B9.
        Markstrom, S. L., et al. (2015). USGS Techniques and Methods 6-B7.
        """
        # Special handling for 2D jh_coef default (nhru, 12)
        if "jh_coef" not in ds:
            ds["jh_coef"] = xr.DataArray(
                np.full((nhru, 12), _DEFAULTS["jh_coef"]),
                dims=("nhru", "nmonths"),
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

        for param_name, default_val in _DEFAULTS.items():
            if param_name in _DEFAULTS_SPECIAL:
                continue  # handled above
            if param_name not in ds:
                ds[param_name] = xr.DataArray(
                    np.float64(default_val),
                    attrs={"long_name": param_name.replace("_", " ").title()},
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

            # Expand scalar to array if nhru dimension exists
            if nhru > 0 and np.ndim(value) == 0:
                value = np.full(nhru, value, dtype=np.float64)

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
            dims: tuple[str, ...] = ("nhru",) if np.ndim(value) >= 1 else ()
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
                sir_name = var_cfg["sir_name"]
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

        Load variable mappings from ``forcing_variables.yml``, auto-detect
        the source dataset by matching SIR variable names, rename to PRMS
        conventions (e.g., ``tmmx`` -> ``tmax``), apply unit conversions
        (e.g., K -> °F, mm -> inches), and merge time-series arrays into
        the parameter dataset.

        Multi-year temporal chunks (keyed with ``_YYYY`` suffixes like
        ``"gridmet_2020"``, ``"gridmet_2021"``) are concatenated along the
        time dimension before processing.

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
        _detect_forcing_dataset : Source dataset matching logic.
        """
        if ctx.temporal is None or len(ctx.temporal) == 0:
            logger.info("No temporal data provided; skipping forcing generation.")
            return ds

        tables_dir = ctx.resolved_lookup_tables_dir
        config = self._load_lookup_table("forcing_variables", tables_dir)
        datasets_config = config["mapping"]

        # Concat multi-year chunks by base name (strip _YYYY suffix)
        chunks_by_source: dict[str, list[xr.Dataset]] = {}
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            chunks_by_source.setdefault(base_name, []).append(tds)

        for source_name, chunks in chunks_by_source.items():
            if len(chunks) > 1:
                chunks.sort(key=lambda c: c["time"].values[0])
                merged_temporal = xr.concat(chunks, dim="time")
            else:
                merged_temporal = chunks[0]

            # Detect dataset config by matching source name or SIR variable names
            dataset_cfg = self._detect_forcing_dataset(
                source_name, merged_temporal, datasets_config
            )
            if dataset_cfg is None:
                logger.warning(
                    "Could not match temporal source '%s' to any forcing dataset config; skipping.",
                    source_name,
                )
                continue

            # Process each mapped variable
            for prms_name, var_cfg in dataset_cfg.items():
                sir_name = var_cfg["sir_name"]
                sir_unit = var_cfg["sir_unit"]
                intermediate_unit = var_cfg["intermediate_unit"]

                if sir_name not in merged_temporal:
                    logger.warning(
                        "Forcing variable '%s' (SIR name '%s') not found in "
                        "temporal data; skipping.",
                        prms_name,
                        sir_name,
                    )
                    continue

                da = merged_temporal[sir_name]

                # Unit conversion (SIR unit → intermediate unit)
                if sir_unit != intermediate_unit:
                    try:
                        converted = convert(
                            da.values.astype(np.float64), sir_unit, intermediate_unit
                        )
                    except KeyError:
                        logger.error(
                            "No unit conversion registered for '%s' → '%s' "
                            "(forcing variable '%s' from source '%s'). "
                            "Register the conversion in units.py or fix "
                            "forcing_variables.yml.",
                            sir_unit,
                            intermediate_unit,
                            prms_name,
                            source_name,
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

            n_vars = sum(1 for p in dataset_cfg if p in ds)
            logger.info(
                "Step 7: merged %d forcing variables from '%s' (%d timesteps).",
                n_vars,
                source_name,
                merged_temporal.sizes.get("time", 0),
            )

        return ds

    @staticmethod
    def _detect_forcing_dataset(
        source_name: str,
        temporal: xr.Dataset,
        datasets_config: dict,
    ) -> dict | None:
        """Match a temporal dataset to its forcing config section.

        Try exact name match on ``source_name`` first, then fall back to
        a fuzzy match by counting how many SIR variable names from each
        config section appear in the temporal dataset.  The section with
        the most variable name hits is selected.

        Parameters
        ----------
        source_name : str
            Base name of the temporal source (e.g., ``"gridmet"``).
        temporal : xr.Dataset
            Temporal dataset to match against config sections.
        datasets_config : dict
            Forcing dataset configurations from ``forcing_variables.yml``,
            keyed by dataset name.

        Returns
        -------
        dict or None
            The matched forcing config section (mapping PRMS names to
            variable specs), or ``None`` if no match is found.
        """
        # Exact match on source name
        if source_name in datasets_config:
            return datasets_config[source_name]

        # Fuzzy match: pick config with most SIR variable name hits
        best_match: str | None = None
        best_count = 0
        temporal_vars = set(temporal.data_vars)
        for cfg_name, cfg_vars in datasets_config.items():
            sir_names = {v["sir_name"] for v in cfg_vars.values()}
            count = len(sir_names & temporal_vars)
            if count > best_count:
                best_count = count
                best_match = cfg_name

        if best_match is not None and best_count > 0:
            logger.info(
                "Matched temporal source '%s' to forcing config '%s' (%d/%d variables matched).",
                source_name,
                best_match,
                best_count,
                len(datasets_config[best_match]),
            )
            return datasets_config[best_match]

        return None

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

        Requires full 12-month coverage in the temporal data to produce
        reliable normals.  Sources with fewer than 12 months are skipped
        with a warning.

        For single-HRU datasets, the output is reshaped from ``(12,)``
        to ``(12, 1)`` to maintain consistent 2-D array shape.
        """
        if ctx.temporal is None or len(ctx.temporal) == 0:
            return None

        tables_dir = ctx.resolved_lookup_tables_dir
        config = self._load_lookup_table("forcing_variables", tables_dir)
        datasets_config = config["mapping"]

        # Concat multi-year chunks by base name
        chunks_by_source: dict[str, list[xr.Dataset]] = {}
        for ds_name, tds in ctx.temporal.items():
            base_name = re.sub(r"_\d{4}$", "", ds_name)
            chunks_by_source.setdefault(base_name, []).append(tds)

        for source_name, chunks in chunks_by_source.items():
            if len(chunks) > 1:
                chunks.sort(key=lambda c: c["time"].values[0])
                merged = xr.concat(chunks, dim="time")
            else:
                merged = chunks[0]

            dataset_cfg = self._detect_forcing_dataset(source_name, merged, datasets_config)
            if dataset_cfg is None:
                logger.warning(
                    "Could not match temporal source '%s' to any forcing "
                    "dataset config for climate normals; skipping.",
                    source_name,
                )
                continue

            # Find tmax and tmin SIR variable names
            tmax_sir = dataset_cfg.get("tmax", {}).get("sir_name")
            tmin_sir = dataset_cfg.get("tmin", {}).get("sir_name")

            if tmax_sir is None or tmin_sir is None:
                logger.warning(
                    "Forcing config for source '%s' is missing tmax and/or "
                    "tmin sir_name entries; cannot compute climate normals.",
                    source_name,
                )
                continue
            if tmax_sir not in merged or tmin_sir not in merged:
                logger.warning(
                    "Temporal source '%s' missing required variables: "
                    "tmax='%s' (present=%s), tmin='%s' (present=%s).",
                    source_name,
                    tmax_sir,
                    tmax_sir in merged,
                    tmin_sir,
                    tmin_sir in merged,
                )
                continue

            # Group by month, compute mean, convert C -> F
            tmax_monthly = merged[tmax_sir].groupby("time.month").mean(dim="time")
            tmin_monthly = merged[tmin_sir].groupby("time.month").mean(dim="time")

            # Validate full 12-month coverage
            n_months = tmax_monthly.sizes.get("month", 0)
            if n_months != 12:
                logger.warning(
                    "Temporal source '%s' covers only %d of 12 months; "
                    "cannot compute reliable monthly normals. Skipping.",
                    source_name,
                    n_months,
                )
                continue

            tmax_f = tmax_monthly.values * 9.0 / 5.0 + 32.0
            tmin_f = tmin_monthly.values * 9.0 / 5.0 + 32.0

            # Ensure 2-D shape (12, nhru) for single-HRU case
            if tmax_f.ndim == 1:
                tmax_f = tmax_f[:, np.newaxis]
                tmin_f = tmin_f[:, np.newaxis]

            logger.info(
                "Computed monthly climate normals from '%s' (%d timesteps, %d HRUs).",
                source_name,
                merged.sizes.get("time", 0),
                tmax_f.shape[1],
            )
            return tmax_f, tmin_f

        logger.warning("No tmax/tmin variables found in temporal data for climate normals.")
        return None

    # ------------------------------------------------------------------
    # Step 10: PET coefficients (Jensen-Haise)
    # ------------------------------------------------------------------

    def _derive_pet_coefficients(
        self,
        ds: xr.Dataset,
        normals: tuple[np.ndarray, np.ndarray] | None,
    ) -> xr.Dataset:
        """Derive Jensen-Haise PET coefficients from climate normals (step 10).

        Compute monthly ``jh_coef`` and per-HRU ``jh_coef_hru`` using
        the PRMS-IV equation 1-26 (Markstrom et al. 2015).  The
        Jensen-Haise method estimates potential evapotranspiration from
        temperature and vapor pressure deficit.

        Falls back to step 13 scalar defaults when no temporal data is
        available (normals is ``None``).

        Parameters
        ----------
        ds : xr.Dataset
            In-progress parameter dataset.  ``hru_elev`` (feet) is used
            for elevation adjustment of ``jh_coef_hru`` if available.
        normals : tuple[np.ndarray, np.ndarray] or None
            Monthly climate normals ``(monthly_tmax, monthly_tmin)`` each
            with shape ``(12, nhru)`` in degrees Fahrenheit (°F), as
            returned by ``_compute_monthly_normals``.

        Returns
        -------
        xr.Dataset
            Dataset with ``jh_coef`` (per_degF_per_day, shape
            ``(nhru, 12)``) and ``jh_coef_hru`` (per_degF_per_day,
            shape ``(nhru,)``) added.

        Notes
        -----
        The PRMS-IV equation 1-26 for ``jh_coef``:
        ``jh = 27.5 - 0.25 * (es_max - es_min) / es_max``
        where ``es_max`` and ``es_min`` are monthly saturation vapor
        pressures computed from tmax and tmin respectively.  Values are
        clipped to ``[0.005, 0.06]``.

        ``jh_coef_hru`` uses the July (warmest month) coefficient with
        a linear elevation adjustment: ``+0.00001`` per foot above sea
        level, reflecting the increased vapor pressure deficit at higher
        elevations due to lower atmospheric pressure.

        References
        ----------
        Markstrom, S. L., et al. (2015). PRMS-IV. USGS TM 6-B7, eq. 1-26.
        Jensen, M. E. and Haise, H. R. (1963). Estimating evapotranspiration
            from solar radiation. J. Irrig. Drain. Div., 89, 15-41.

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

        # --- jh_coef: PRMS-IV eq. 1-26 ---
        svp_max = _sat_vp(monthly_tmax)  # (12, nhru)
        svp_min = _sat_vp(monthly_tmin)  # (12, nhru)

        # Guard against division by zero
        svp_max_safe = np.maximum(svp_max, 1e-6)
        jh_coef = 27.5 - 0.25 * (svp_max - svp_min) / svp_max_safe
        jh_coef = np.clip(jh_coef, 0.005, 0.06)

        # Transpose to (nhru, 12) for output convention
        ds["jh_coef"] = xr.DataArray(
            jh_coef.T,
            dims=("nhru", "nmonths"),
            attrs={"units": "per_degF_per_day", "long_name": "Jensen-Haise PET coefficient"},
        )

        # --- jh_coef_hru: elevation-adjusted coefficient ---
        # Use July (index 6) as warmest month for base coefficient
        july_jh = jh_coef[6, :]  # (nhru,)

        # Elevation adjustment: higher elevations have lower boiling point,
        # increasing vapor pressure deficit -> slightly higher coefficients.
        # Linear approximation: +0.00001 per foot above sea level.
        if "hru_elev" in ds:
            elev_ft = ds["hru_elev"].values
            jh_coef_hru = july_jh + 0.00001 * elev_ft
        else:
            logger.info(
                "hru_elev not in dataset; computing jh_coef_hru without elevation adjustment.",
            )
            jh_coef_hru = july_jh

        jh_coef_hru = np.clip(jh_coef_hru, 0.005, 0.06)
        ds["jh_coef_hru"] = xr.DataArray(
            jh_coef_hru,
            dims=("nhru",),
            attrs={"units": "per_degF_per_day", "long_name": "Per-HRU Jensen-Haise coefficient"},
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
        Existing parameters are updated in-place; new parameters are
        created with an ``nhru`` dimension if the value is a list, or
        as a scalar otherwise.
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
