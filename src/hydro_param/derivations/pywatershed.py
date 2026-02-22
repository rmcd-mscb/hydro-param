"""pywatershed parameter derivation plugin.

Converts SIR physical properties (zonal statistics of raw geospatial
data) into PRMS/pywatershed model parameters.  Implements the
derivation pipeline from ``pywatershed_dataset_param_map.yml``.

Foundation implementation covers steps 1, 2, 3, 4, 8, and 13.
Future PRs will add steps 5 (soils), 6 (waterbody overlay),
7 (forcing generation), 9 (soltab), 10 (PET), 11 (transp),
12 (routing), and 14 (calibration seeds).
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pyproj
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
    ``docs/reference/pywatershed_dataset_param_map.yml``.  Covers pipeline
    steps 1 (geometry), 2 (topology), 3 (topography), 4 (land cover),
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

    def rename_sir_variables(
        self,
        sir: xr.Dataset,
        renames: dict[str, str] | None = None,
    ) -> xr.Dataset:
        """Rename SIR variables for derivation compatibility.

        Applies a mapping from pipeline SIR names to the names expected
        by this derivation plugin.  If no explicit renames are provided,
        applies the default mapping for common NHGF STAC / gdptools
        variable names.

        Parameters
        ----------
        sir
            Input SIR dataset.
        renames
            Explicit variable name mapping ``{old: new}``.  Merged with
            (and overrides) the built-in defaults.

        Returns
        -------
        xr.Dataset
            SIR with renamed variables.
        """
        defaults: dict[str, str] = {
            "FctImp_mean": "impervious",
        }
        mapping = {**defaults, **(renames or {})}
        actual = {old: new for old, new in mapping.items() if old in sir}
        if actual:
            sir = sir.rename(actual)
            logger.info("SIR variable renames: %s", actual)
        return sir

    def derive(
        self,
        sir: xr.Dataset,
        *,
        config: dict | None = None,
        fabric: gpd.GeoDataFrame | None = None,
        segments: gpd.GeoDataFrame | None = None,
        id_field: str = "nhm_id",
        segment_id_field: str = "nhm_seg",
    ) -> xr.Dataset:
        """Derive all pywatershed parameters from the SIR.

        Parameters
        ----------
        sir
            Standardized Internal Representation with physical properties
            (e.g., ``elevation``, ``slope``, ``aspect``, ``land_cover``).
        config
            Optional derivation configuration.  Supports key:
            ``parameter_overrides`` (dict).
        fabric
            HRU polygon GeoDataFrame with topology attributes.
            Required for step 2 (topology extraction).
        segments
            Stream segment GeoDataFrame with line geometries.
            Required for step 2 (topology extraction).
        id_field
            Column name for HRU identifiers in the fabric.
        segment_id_field
            Column name for segment identifiers in the segments
            GeoDataFrame.

        Returns
        -------
        xr.Dataset
            pywatershed parameter dataset with PRMS-convention units.
        """
        config = config or {}
        nhru = sir.sizes.get("hru_id", 0)
        ds = xr.Dataset()

        # Carry HRU coordinates so derived params retain stable indexing
        if "hru_id" in sir.coords:
            ds = ds.assign_coords(nhru=sir["hru_id"].values)

        # Step 1: Geometry (hru_area, hru_lat)
        ds = self._derive_geometry(sir, ds, fabric=fabric, id_field=id_field)

        # Step 2: Topology (tosegment, hru_segment, seg_length)
        if fabric is not None and segments is not None:
            ds = self._derive_topology(
                ds,
                fabric=fabric,
                segments=segments,
                id_field=id_field,
                segment_id_field=segment_id_field,
            )

        # Step 2: Topology (tosegment, hru_segment, seg_length)
        if fabric is not None and segments is not None:
            ds = self._derive_topology(
                ds,
                fabric=fabric,
                segments=segments,
                id_field=id_field,
                segment_id_field=segment_id_field,
            )

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

    def _derive_geometry(
        self,
        sir: xr.Dataset,
        ds: xr.Dataset,
        *,
        fabric: gpd.GeoDataFrame | None = None,
        id_field: str = "nhm_id",
    ) -> xr.Dataset:
        """Step 1: Compute basic geometry from the target fabric.

        When *fabric* is provided, computes area via EPSG:5070 equal-area
        projection and latitude from EPSG:4326 centroids.  Falls back to
        SIR variables ``hru_area_m2`` and ``hru_lat`` when fabric is not
        available.
        """
        if fabric is not None and id_field in fabric.columns:
            # Align fabric rows to SIR HRU ordering
            hru_ids = sir["hru_id"].values if "hru_id" in sir.coords else None
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
        ds: xr.Dataset,
        *,
        fabric: gpd.GeoDataFrame,
        segments: gpd.GeoDataFrame,
        id_field: str,
        segment_id_field: str,
    ) -> xr.Dataset:
        """Step 2: Extract routing topology from GeoDataFrames.

        Extracts ``tosegment``, ``hru_segment``, and ``seg_length``
        directly from GeoDataFrame columns.  The input fabric should
        carry these as attributes (e.g., from the Geospatial Fabric).

        Parameters
        ----------
        ds
            Output dataset being constructed.
        fabric
            HRU polygon GeoDataFrame.  Required columns: ``hru_segment``
            and the column named by ``id_field``.
        segments
            Stream segment line GeoDataFrame.  Required column:
            ``tosegment``.  Optional: ``seg_length``.
        id_field
            Column name for HRU identifiers in the fabric.  Used to
            align fabric rows to ``ds.coords['nhru']``.
        segment_id_field
            Column name for segment identifiers.

        Raises
        ------
        ValueError
            If required columns (``tosegment``, ``hru_segment``) are
            missing from the GeoDataFrames.
        """
        nseg = len(segments)

        # Add nsegment coordinate from segment IDs
        if segment_id_field in segments.columns:
            seg_ids = segments[segment_id_field].values
        else:
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

        Supports three input modes:

        1. **Categorical fractions** (preferred): SIR contains columns
           like ``LndCov_11``, ``LndCov_21``, etc. from gdptools
           ``ZonalGen(categorical=True)``.  The majority class is
           computed via argmax.
        2. **Single majority value**: ``land_cover`` or
           ``land_cover_majority`` containing the dominant NLCD class.
        3. **Impervious/canopy**: ``impervious`` (0-100%) and
           ``tree_canopy`` (0-100%).
        """
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
            nlcd_table = self._load_lookup_table("nlcd_to_prms_cov_type")
            mapping = nlcd_table["mapping"]
            cov_type = np.array([mapping.get(int(v), 0) for v in nlcd_values])
        elif lc_var is not None:
            nlcd_table = self._load_lookup_table("nlcd_to_prms_cov_type")
            mapping = nlcd_table["mapping"]
            nlcd_values_lc = sir[lc_var].values.astype(int)
            cov_type = np.array([mapping.get(int(v), 0) for v in nlcd_values_lc])

        if has_lc:
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

    @staticmethod
    def _compute_majority_from_fractions(
        sir: xr.Dataset,
        prefixes: tuple[str, ...] = ("LndCov_", "land_cover_"),
    ) -> np.ndarray | None:
        """Compute majority NLCD class from categorical fraction columns.

        Scans SIR variables for columns matching ``{prefix}{class_code}``
        (e.g., ``LndCov_11``, ``LndCov_41``).  For each HRU, returns the
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
                dims = ("nhru",) if arr.ndim == 1 else ()
                ds[param_name] = xr.DataArray(
                    arr,
                    dims=dims if dims else None,
                    attrs={"long_name": f"{param_name} (user override)"},
                )
                logger.info("Override (new): %s = %s", param_name, value)
        return ds

    # ------------------------------------------------------------------
    # Future steps (stubs)
    # ------------------------------------------------------------------

    # TODO: Step 5 — soils (soil_type, soil_moist_max, soil_rechr_max_frac)
    # TODO: Step 6 — waterbody overlay (dprst_frac, dprst_area_max, hru_type)
    # TODO: Step 7 — forcing generation (prcp, tmax, tmin time series)
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
