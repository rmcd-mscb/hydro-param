"""pywatershed parameter derivation plugin.

Converts SIR physical properties (zonal statistics of raw geospatial
data) into PRMS/pywatershed model parameters.  Implements the
derivation pipeline from ``pywatershed_dataset_param_map.yml``.

Foundation implementation covers steps 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, and 14.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
import xarray as xr
import yaml

from hydro_param.plugins import DerivationContext
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
    # PET (Jensen-Haise)
    "jh_coef": 0.014,
    "jh_coef_hru": 0.014,
    # Transpiration timing
    "transp_beg": 4,  # April
    "transp_end": 10,  # October
}

# Default imperv_stor_max by cov_type (inches)
_IMPERV_STOR_MAX_DEFAULT = 0.03

# Calibration seed method dispatch — safe lambdas only, NO eval.
_SEED_METHODS: dict[str, Callable[..., np.floating | np.ndarray]] = {
    "linear": lambda ds, p: p["scale"] * ds[p["input"]].values + p["offset"],
    "exponential_scale": lambda ds, p: p["scale"] * np.exp(p["exponent"] * ds[p["input"]].values),
    "fraction_of": lambda ds, p: p["fraction"] * ds[p["input"]].values,
    "constant": lambda ds, p: np.float64(p["value"]),
}


def _sat_vp(temp_f: np.ndarray) -> np.ndarray:
    """Saturation vapor pressure (hPa) from temperature in °F.

    Uses the Magnus formula (Alduchov & Eskridge 1996).

    Parameters
    ----------
    temp_f
        Temperature in degrees Fahrenheit.

    Returns
    -------
    np.ndarray
        Saturation vapor pressure in hectopascals (hPa).
    """
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    return 6.1078 * np.exp(17.269 * temp_c / (temp_c + 237.3))


def merge_temporal_into_derived(
    derived: xr.Dataset,
    temporal: dict[str, xr.Dataset],
    renames: dict[str, str] | None = None,
    conversions: dict[str, tuple[str, str]] | None = None,
    id_field: str = "nhru",
) -> xr.Dataset:
    """Merge temporal data into derived dataset with renaming and unit conversion.

    Parameters
    ----------
    derived
        Derived parameter dataset (output of ``PywatershedDerivation.derive()``).
    temporal
        Temporal datasets keyed by dataset name from ``PipelineResult.temporal``.
    renames
        Variable name mapping ``{source_name: target_name}``.
    conversions
        Unit conversions ``{variable_name: (from_unit, to_unit)}``.
        Applied **after** renames (use target names).
    id_field
        Feature dimension name used in the derived dataset.

    Returns
    -------
    xr.Dataset
        Derived dataset with temporal variables merged in.
    """
    import warnings

    warnings.warn(
        "merge_temporal_into_derived() is deprecated. "
        "Use PywatershedDerivation._derive_forcing() via DerivationContext.temporal instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    logger.warning(
        "merge_temporal_into_derived() is deprecated; "
        "use DerivationContext.temporal with _derive_forcing() instead."
    )
    renames = renames or {}
    conversions = conversions or {}

    # Collect per-year chunks by base dataset name, then concatenate.
    # Keys like "gridmet_2020", "gridmet_2021" share the base name "gridmet";
    # we concatenate along time before merging into derived.
    chunks_by_source: dict[str, list[xr.Dataset]] = {}
    for ds_name, ds in temporal.items():
        base_name = re.sub(r"_\d{4}$", "", ds_name)
        chunks_by_source.setdefault(base_name, []).append(ds)

    for _source, chunks in chunks_by_source.items():
        if len(chunks) > 1:
            chunks.sort(key=lambda c: c["time"].values[0])
            ds = xr.concat(chunks, dim="time")
        else:
            ds = chunks[0]

        # Rename temporal variables (e.g., pr->prcp, tmmx->tmax)
        actual_renames = {old: new for old, new in renames.items() if old in ds}
        if actual_renames:
            ds = ds.rename(actual_renames)

        # Apply unit conversions (e.g., K->C for temperature)
        for var_name, (from_unit, to_unit) in conversions.items():
            if var_name in ds:
                da = ds[var_name]
                converted = convert(da.values.astype(np.float64), from_unit, to_unit)
                ds[var_name] = da.copy(data=converted)

        # Align temporal feature dimension to derived dataset's id_field
        for var in ds.data_vars:
            da = ds[str(var)]
            feat_dims = [d for d in da.dims if d != "time"]
            if feat_dims and id_field in derived.dims and feat_dims[0] != id_field:
                da = da.rename({feat_dims[0]: id_field})
            derived[str(var)] = da

    return derived


class PywatershedDerivation:
    """Derive pywatershed/PRMS parameters from SIR physical properties.

    Implements the derivation pipeline from
    ``docs/reference/pywatershed_dataset_param_map.yml``.  Covers pipeline
    steps 1 (geometry), 2 (topology), 3 (topography), 4 (land cover),
    5 (soils), 7 (forcing), 8 (lookup tables), 9 (soltab), 10 (PET
    coefficients), 11 (transpiration timing), 13 (defaults), and
    14 (calibration seeds).
    """

    name: str = "pywatershed"

    def __init__(self) -> None:
        self._lookup_cache: dict[str, dict] = {}

    def derive(self, context: DerivationContext) -> xr.Dataset:
        """Derive all pywatershed parameters from the SIR.

        Parameters
        ----------
        context
            Typed input bundle containing SIR, fabric, config, etc.

        Returns
        -------
        xr.Dataset
            pywatershed parameter dataset with PRMS-convention units.
        """
        sir = context.sir
        id_field = context.fabric_id_field
        nhru = sir.sizes.get(id_field, 0)
        ds = xr.Dataset()

        # Carry HRU coordinates so derived params retain stable indexing
        if id_field in sir.coords:
            ds = ds.assign_coords(nhru=sir[id_field].values)

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

        # Step 8: Lookup table application
        ds = self._apply_lookup_tables(context, ds)

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
        """Step 1: Compute basic geometry from the target fabric.

        When *fabric* is provided, computes area via EPSG:5070 equal-area
        projection and latitude from EPSG:4326 centroids.  Falls back to
        SIR variables ``hru_area_m2`` and ``hru_lat`` when fabric is not
        available.
        """
        sir = ctx.sir
        fabric = ctx.fabric
        id_field = ctx.fabric_id_field

        if fabric is not None and id_field in fabric.columns:
            # Align fabric rows to SIR HRU ordering
            hru_ids = sir[id_field].values if id_field in sir.coords else None
            if hru_ids is not None:
                fab = fabric.set_index(id_field).loc[hru_ids].reset_index()
            else:
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
        """Step 2: Extract routing topology from GeoDataFrames.

        Extracts ``tosegment``, ``hru_segment``, and ``seg_length``
        directly from GeoDataFrame columns.  The input fabric should
        carry these as attributes (e.g., from the Geospatial Fabric).

        Parameters
        ----------
        ctx
            Derivation context with fabric, segments, and field names.
        ds
            Output dataset being constructed.

        Raises
        ------
        ValueError
            If required columns (``tosegment``, ``hru_segment``) are
            missing from the GeoDataFrames.
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
        """Compute segment length from column or geodesic calculation.

        Uses ``seg_length`` column if present, otherwise computes
        geodesic length from segment line geometries using
        ``pyproj.Geod.geometry_length``.  Handles LineString,
        MultiLineString, and projected CRS (auto-reprojects to WGS84).
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
        """Validate tosegment array.

        Raises
        ------
        ValueError
            If self-loops exist, values out of range, or no outlets found.
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
        """Validate hru_segment array.

        Raises
        ------
        ValueError
            If values are out of range [0, nseg].
        """
        if np.any(hru_segment < 0) or np.any(hru_segment > nseg):
            bad = hru_segment[(hru_segment < 0) | (hru_segment > nseg)]
            raise ValueError(f"hru_segment values out of range [0, {nseg}]: {bad.tolist()}")

    # ------------------------------------------------------------------
    # Step 3: Topographic parameters
    # ------------------------------------------------------------------

    def _derive_topography(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Step 3: Convert DEM zonal statistics to PRMS parameters."""
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
        """Step 4: Derive vegetation and impervious parameters from NLCD.

        Supports three input modes:

        1. **Categorical fractions** (preferred): SIR contains columns
           like ``lndcov_frac_11``, ``lndcov_frac_21``, etc. from
           normalized categorical output.  The majority class is
           computed via argmax.
        2. **Single majority value**: ``land_cover`` or
           ``land_cover_majority`` containing the dominant NLCD class.
        3. **Impervious/canopy**: ``fctimp_pct_mean`` (0-100%) and
           ``tree_canopy_pct_mean`` (0-100%).
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

        if "fctimp_pct_mean" in sir:
            # Percent (0-100) -> fraction (0-1)
            ds["hru_percent_imperv"] = xr.DataArray(
                np.clip(sir["fctimp_pct_mean"].values / 100.0, 0.0, 1.0),
                dims="nhru",
                attrs={"units": "decimal_fraction", "long_name": "HRU impervious fraction"},
            )

        return ds

    @staticmethod
    def _compute_majority_from_fractions(
        sir: xr.Dataset,
        prefixes: tuple[str, ...] = ("lndcov_frac_",),
    ) -> np.ndarray | None:
        """Compute majority NLCD class from categorical fraction columns.

        Scans SIR variables for columns matching ``{prefix}{class_code}``
        (e.g., ``lndcov_frac_11``, ``lndcov_frac_41``).  For each HRU, returns the
        class code with the highest fraction.

        Parameters
        ----------
        sir
            SIR dataset potentially containing fraction columns.
        prefixes
            Variable name prefixes to search for.

        Returns
        -------
        np.ndarray or None
            Array of majority NLCD class codes (int), or ``None`` if no
            fraction columns found.
        """
        for prefix in prefixes:
            fraction_vars = sorted(str(v) for v in sir.data_vars if str(v).startswith(prefix))
            if len(fraction_vars) < 2:
                continue

            # Extract class codes from suffixes
            class_codes: list[int] = []
            valid_vars: list[str] = []
            for v in fraction_vars:
                suffix = v[len(prefix) :]
                try:
                    class_codes.append(int(suffix))
                    valid_vars.append(v)
                except ValueError:
                    logger.debug(
                        "Skipping variable '%s': suffix '%s' is not an integer class code",
                        v,
                        suffix,
                    )
                    continue

            if len(class_codes) < 2:
                continue

            # Stack fractions into (nhru, n_classes) array
            fractions = np.column_stack([sir[v].values for v in valid_vars])
            codes = np.array(class_codes)
            majority_idx = np.argmax(fractions, axis=1)
            majority_class = codes[majority_idx]

            logger.info(
                "Computed majority class from %d categorical fraction columns (prefix=%r)",
                len(valid_vars),
                prefix,
            )
            return majority_class

        return None

    # ------------------------------------------------------------------
    # Step 5: Soils zonal stats
    # ------------------------------------------------------------------

    _SOIL_RECHR_MAX_FRAC_DEFAULT: float = 0.4

    def _derive_soils(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Step 5: Derive soil parameters from gNATSGO/STATSGO2 zonal stats.

        Supports two input modes:

        1. **Soil texture fractions** (preferred): SIR contains columns
           like ``soil_texture_frac_sand``, ``soil_texture_frac_loam``, etc.
           Majority class via argmax, then reclassify to PRMS soil_type.
        2. **Single texture class**: ``soil_texture`` or
           ``soil_texture_majority`` containing the dominant USDA class name.

        Also derives ``soil_moist_max`` from available water capacity (AWC).
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
        else:
            logger.warning(
                "Skipping soil_moist_max derivation (step 5): 'awc_mm_mean' not found in SIR."
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

    def _compute_soil_type(self, sir: xr.Dataset, ctx: DerivationContext) -> np.ndarray | None:
        """Compute PRMS soil_type from SIR soil texture data."""
        # Check data availability before loading lookup table
        prefix = "soil_texture_frac_"
        fraction_vars = sorted(str(v) for v in sir.data_vars if str(v).startswith(prefix))
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
    # Step 8: Lookup table application
    # ------------------------------------------------------------------

    def _apply_lookup_tables(self, ctx: DerivationContext, ds: xr.Dataset) -> xr.Dataset:
        """Step 8: Apply lookup tables for interception and winter cover.

        Requires ``cov_type`` and ``covden_sum`` to already be in ``ds``.
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
        """Step 9: Compute potential solar radiation tables (Swift 1976).

        Requires ``hru_lat``, ``hru_slope``, and ``hru_aspect`` from step 3.
        Produces 2-D arrays of shape ``(ndoy=366, nhru)`` for potential solar
        radiation on sloped and horizontal surfaces.  Output dimensions are
        named ``ndoy`` and ``nhru`` to match pywatershed's internal convention;
        ``nhru`` aligns with the id_field coordinate on the existing dataset.
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
        """Step 13: Apply standard PRMS default values.

        Only sets parameters that are not already present in ``ds``
        (preserves any values derived from data in earlier steps).
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

        _SKIP_IN_LOOP = {"jh_coef", "transp_beg", "transp_end"}
        for param_name, default_val in _DEFAULTS.items():
            if param_name in _SKIP_IN_LOOP:
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
        """Step 14: Apply YAML-driven calibration seed values.

        Seeds provide physically-based initial values for calibration
        parameters.  Each seed is computed from a method (linear,
        exponential_scale, fraction_of, constant) with range clipping.
        Seeds are skipped when the parameter already exists in ``ds``.
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

    def _derive_forcing(
        self,
        ctx: DerivationContext,
        ds: xr.Dataset,
    ) -> xr.Dataset:
        """Step 7: Merge SIR-normalized temporal forcing into derived dataset.

        Loads variable mappings from ``forcing_variables.yml``, detects the
        source dataset by matching SIR variable names, renames to PRMS names,
        applies unit conversions, and merges into *ds*.

        Skips gracefully when no temporal data is available.
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

        Tries exact name match first, then falls back to counting SIR
        variable name matches.
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

        Aggregates multi-year daily data into 12 monthly means per HRU,
        converting from °C to °F.

        Returns
        -------
        tuple of (monthly_tmax, monthly_tmin) each shape (12, nhru) in °F,
        or None if temporal data is unavailable.
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
        """Step 10: Derive Jensen-Haise PET coefficients.

        Computes ``jh_coef`` (nhru, 12) and ``jh_coef_hru`` (nhru,) from
        monthly climate normals using PRMS-IV equation 1-26.

        Falls back to step 13 defaults when no temporal data is available.
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
        svp_max_safe = np.where(svp_max < 1e-6, 1e-6, svp_max)
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
        """Step 11: Derive transpiration onset/offset from monthly tmin.

        Computes ``transp_beg`` and ``transp_end`` (integer months) by
        detecting the frost-free period from monthly minimum temperature
        normals.  Falls back to step 13 defaults when no temporal data
        is available.
        """
        if normals is None:
            logger.info("No temporal data for transpiration timing; deferring to defaults.")
            return ds

        _monthly_tmax, monthly_tmin = normals  # (12, nhru) in °F
        nhru = monthly_tmin.shape[1]
        freezing = 32.0  # °F

        # transp_beg: first month (1-indexed) where tmin > freezing
        transp_beg = np.full(nhru, 4, dtype=np.int32)  # default April
        for hru in range(nhru):
            for month_idx in range(12):
                if monthly_tmin[month_idx, hru] > freezing:
                    transp_beg[hru] = month_idx + 1  # 1-indexed
                    break

        # transp_end: first month after June (7+) where tmin < freezing
        transp_end = np.full(nhru, 10, dtype=np.int32)  # default October
        for hru in range(nhru):
            for month_idx in range(6, 12):  # July onward
                if monthly_tmin[month_idx, hru] < freezing:
                    transp_end[hru] = month_idx + 1  # 1-indexed
                    break

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
        """Load a YAML lookup table from the lookup tables directory.

        Parameters
        ----------
        name
            Lookup table filename (without ``.yml`` extension).
        tables_dir
            Directory containing lookup table YAML files.

        Returns
        -------
        dict
            Parsed YAML content with at least a ``mapping`` key.
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
