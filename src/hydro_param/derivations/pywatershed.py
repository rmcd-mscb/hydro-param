"""pywatershed parameter derivation plugin.

Converts SIR physical properties (zonal statistics of raw geospatial
data) into PRMS/pywatershed model parameters.  Implements the
derivation pipeline from ``pywatershed_dataset_param_map.yml``.

Foundation implementation covers steps 1, 2, 3, 4, 8, 9, and 13.
"""

from __future__ import annotations

import logging
import re
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
}

# Default imperv_stor_max by cov_type (inches)
_IMPERV_STOR_MAX_DEFAULT = 0.03


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
    8 (lookup tables), 9 (soltab), and 13 (defaults).
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

        # Step 8: Lookup table application
        ds = self._apply_lookup_tables(context, ds)

        # Step 9: Solar radiation tables (soltab)
        ds = self._derive_soltab(context, ds)

        # Step 13: Defaults and initial conditions
        ds = self._apply_defaults(ds, nhru)

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
        Produces 2-D arrays of shape (ndoy=366, nhru) for potential solar
        radiation on sloped and horizontal surfaces.
        """
        required = ("hru_lat", "hru_slope", "hru_aspect")
        if not all(v in ds for v in required):
            missing = [v for v in required if v not in ds]
            logger.info("Skipping soltab: missing %s", missing)
            return ds

        potsw, horad, sunhrs = compute_soltab(
            slopes=ds["hru_slope"].values,
            aspects=ds["hru_aspect"].values,
            lats=ds["hru_lat"].values,
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
