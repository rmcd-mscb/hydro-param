import { useState } from "react";

/*
  Complete mapping of pywatershed/PRMS parameters to source datasets
  and derivation methods, organized for hydro-param YML data catalog design.

  Based on: Regan et al. 2018 (TM6-B9 Appendix 1), Hay et al. 2023 (TM6-B10),
  Markstrom et al. 2015 (TM6-B7), Viger & Bock 2014 (GF), and pywatershed source.
*/

const dataCategories = {
  topography: {
    name: "Topography / DEM",
    icon: "⛰️",
    color: "#8B6914",
    datasets: [
      {
        name: "NED / 3DEP",
        desc: "National Elevation Dataset / 3D Elevation Program",
        resolution: "10m–30m",
        source: "USGS",
        accessor: "pynhd / pygeohydro / direct S3",
        format: "GeoTIFF / COG",
      },
    ],
    parameters: [
      {
        name: "hru_elev",
        desc: "Mean HRU elevation (feet or meters)",
        method: "Zonal mean of DEM over HRU polygon",
        type: "gis_zonal",
      },
      {
        name: "hru_slope",
        desc: "Mean HRU slope (decimal fraction)",
        method: "Compute slope raster from DEM, then zonal mean over HRU",
        type: "gis_zonal",
      },
      {
        name: "hru_aspect",
        desc: "Mean HRU aspect (degrees)",
        method: "Compute aspect raster from DEM, then circular mean over HRU",
        type: "gis_zonal",
      },
      {
        name: "hru_lat",
        desc: "HRU centroid latitude",
        method: "Centroid of HRU polygon geometry",
        type: "gis_geometry",
      },
      {
        name: "hru_area",
        desc: "HRU area (acres)",
        method: "Geodesic area of HRU polygon, convert to acres",
        type: "gis_geometry",
      },
    ],
  },

  landcover: {
    name: "Land Cover / Vegetation",
    icon: "🌲",
    color: "#228B22",
    datasets: [
      {
        name: "NLCD",
        desc: "National Land Cover Database",
        resolution: "30m, multi-year (2001–2021)",
        source: "USGS MRLC",
        accessor: "pygeohydro / MRLC API",
        format: "GeoTIFF / COG",
      },
      {
        name: "NLCD Tree Canopy",
        desc: "NLCD Tree Canopy Cover",
        resolution: "30m",
        source: "USGS MRLC",
        accessor: "pygeohydro",
        format: "GeoTIFF",
      },
      {
        name: "MODIS LAI/FPAR",
        desc: "Leaf Area Index / FPAR (optional, for enhanced veg params)",
        resolution: "500m, 8-day",
        source: "NASA LP DAAC",
        accessor: "AppEEARS / earthaccess",
        format: "HDF / GeoTIFF",
      },
    ],
    parameters: [
      {
        name: "cov_type",
        desc: "Land cover type (0=bare, 1=grass, 2=shrub, 3=tree, 4=conifer)",
        method: "Dominant NLCD class reclassified to PRMS vegetation types via lookup table",
        type: "gis_reclassify",
      },
      {
        name: "covden_sum",
        desc: "Summer vegetation cover density (0–1)",
        method: "From NLCD canopy cover or from NLCD class lookup table (summer fraction)",
        type: "lookup_table",
      },
      {
        name: "covden_win",
        desc: "Winter vegetation cover density (0–1)",
        method: "Summer density × deciduous reduction factor by cov_type; conifers retain ~full",
        type: "derived_formula",
      },
      {
        name: "srain_intcp",
        desc: "Summer rain interception storage capacity (inches)",
        method: "Lookup by cov_type: bare=0, grass=0.05, shrub=0.05, deciduous=0.08, conifer=0.08",
        type: "lookup_table",
      },
      {
        name: "wrain_intcp",
        desc: "Winter rain interception storage capacity (inches)",
        method: "Lookup by cov_type, reduced for deciduous",
        type: "lookup_table",
      },
      {
        name: "snow_intcp",
        desc: "Snow interception storage capacity (inches)",
        method: "Lookup by cov_type: highest for conifers (~0.06), zero for bare",
        type: "lookup_table",
      },
      {
        name: "hru_percent_imperv",
        desc: "Fraction of HRU that is impervious",
        method: "Zonal mean of NLCD impervious surface dataset over HRU",
        type: "gis_zonal",
      },
      {
        name: "hru_type",
        desc: "HRU type (0=inactive, 1=land, 2=lake, 3=swale)",
        method: "From GF/NHDPlus waterbody overlay + land/swale classification",
        type: "gis_overlay",
      },
      {
        name: "transp_beg",
        desc: "Month transpiration begins",
        method: "Derived from growing season / frost-free period using climate normals + latitude",
        type: "derived_climate",
      },
      {
        name: "transp_end",
        desc: "Month transpiration ends",
        method: "Derived from killing frost date using climate normals + latitude",
        type: "derived_climate",
      },
      {
        name: "transp_tmax",
        desc: "Temperature threshold for transpiration onset (°F)",
        method: "Default value (500°F accumulated, or set via sensitivity analysis)",
        type: "default_value",
      },
    ],
  },

  soils: {
    name: "Soils",
    icon: "🟤",
    color: "#8B4513",
    datasets: [
      {
        name: "STATSGO2",
        desc: "U.S. General Soil Map (State Soil Geographic)",
        resolution: "~1:250,000",
        source: "USDA NRCS",
        accessor: "pygeohydro / NRCS Web Soil Survey API",
        format: "Shapefile / GDB",
      },
      {
        name: "SSURGO",
        desc: "Soil Survey Geographic Database (higher resolution alternative)",
        resolution: "~1:12,000–1:63,360",
        source: "USDA NRCS",
        accessor: "pygeohydro / SoilsDB / NRCS WFS",
        format: "Shapefile / GDB / SQLite",
      },
      {
        name: "SoilGrids 250m",
        desc: "Global gridded soil properties (alternative/gap-fill)",
        resolution: "250m",
        source: "ISRIC",
        accessor: "REST API / COG",
        format: "GeoTIFF / COG",
      },
    ],
    parameters: [
      {
        name: "soil_type",
        desc: "Soil type (1=sand, 2=loam, 3=clay)",
        method: "Dominant soil texture class from STATSGO/SSURGO reclassified to 3 types",
        type: "gis_reclassify",
      },
      {
        name: "soil_moist_max",
        desc: "Maximum available water-holding capacity of capillary reservoir (inches)",
        method: "AWC (available water capacity) × soil depth from STATSGO/SSURGO, per HRU zonal stats",
        type: "derived_formula",
      },
      {
        name: "soil_moist_init_frac",
        desc: "Initial soil moisture as fraction of soil_moist_max",
        method: "Typically set to 0.5 or from spin-up / restart file",
        type: "default_value",
      },
      {
        name: "soil_rechr_max_frac",
        desc: "Maximum recharge zone storage as fraction of soil_moist_max",
        method: "Estimated from soil depth ratios; NHM uses AWC of upper layers / total AWC",
        type: "derived_formula",
      },
      {
        name: "soil_rechr_init_frac",
        desc: "Initial recharge zone moisture fraction",
        method: "Default 0.5 or from restart",
        type: "default_value",
      },
      {
        name: "soil2gw_max",
        desc: "Maximum rate of soil water excess moving to GW (inches/day)",
        method: "Calibration parameter; initial from Ksat/hydraulic conductivity",
        type: "calibration",
      },
      {
        name: "ssr2gw_rate",
        desc: "Coefficient for gravity drainage to GW reservoir",
        method: "Calibration parameter; initial estimate from Ksat",
        type: "calibration",
      },
      {
        name: "ssr2gw_exp",
        desc: "Exponent for gravity drainage to GW",
        method: "Calibration; default ~1.0",
        type: "calibration",
      },
      {
        name: "slowcoef_lin",
        desc: "Linear slow interflow routing coefficient",
        method: "Calibration parameter (sensitive); initial from Ksat × slope relationship",
        type: "calibration",
      },
      {
        name: "slowcoef_sq",
        desc: "Non-linear slow interflow routing coefficient",
        method: "Calibration; typically small",
        type: "calibration",
      },
      {
        name: "fastcoef_lin",
        desc: "Linear fast interflow (preferential flow) coefficient",
        method: "Calibration; initial from macropore / Ksat estimates",
        type: "calibration",
      },
      {
        name: "fastcoef_sq",
        desc: "Non-linear fast interflow coefficient",
        method: "Calibration",
        type: "calibration",
      },
      {
        name: "pref_flow_den",
        desc: "Fraction of soil zone with preferential flow paths",
        method: "Calibration; initial from soil macroporosity estimates or set to 0",
        type: "calibration",
      },
      {
        name: "sat_threshold",
        desc: "Water-holding capacity of gravity + preferential reservoirs (inches)",
        method: "Derived: soil_moist_max × soil depth ratio; adjusted in calibration",
        type: "derived_formula",
      },
      {
        name: "ssstor_init_frac",
        desc: "Initial subsurface storage fraction",
        method: "Default 0.0 or from restart",
        type: "default_value",
      },
    ],
  },

  climate: {
    name: "Climate Forcing / Meteorological",
    icon: "🌡️",
    color: "#DC143C",
    datasets: [
      {
        name: "Daymet v4",
        desc: "Daily surface weather data on 1km grid",
        resolution: "1km, daily",
        source: "ORNL DAAC / NASA",
        accessor: "pydaymet / gdptools",
        format: "NetCDF / Zarr (via Thredds or S3)",
      },
      {
        name: "GridMET",
        desc: "Gridded meteorological data (extended vars)",
        resolution: "~4km, daily",
        source: "U of Idaho / climatologylab",
        accessor: "pygridmet / gdptools",
        format: "NetCDF / OPENDAP",
      },
      {
        name: "PRISM",
        desc: "Parameter-elevation Regressions on Independent Slopes Model",
        resolution: "800m–4km, daily/monthly",
        source: "Oregon State / PRISM Climate Group",
        accessor: "prism API",
        format: "BIL / GeoTIFF",
      },
      {
        name: "NLDAS-2",
        desc: "North American Land Data Assimilation System",
        resolution: "~12km, hourly → daily agg",
        source: "NASA GES DISC",
        accessor: "pygeohydro / earthaccess",
        format: "GRIB / NetCDF",
      },
      {
        name: "CONUS404-BA",
        desc: "Bias-adjusted high-res WRF atmospheric simulation",
        resolution: "4km, hourly → daily",
        source: "USGS (Zhang et al. 2024)",
        accessor: "S3 / Zarr / gdptools",
        format: "Zarr / NetCDF",
      },
    ],
    parameters: [
      {
        name: "prcp (CBH)",
        desc: "Daily precipitation per HRU (inches)",
        method: "Area-weighted spatial average of gridded precip over HRU polygons",
        type: "gis_zonal",
      },
      {
        name: "tmax (CBH)",
        desc: "Daily maximum temperature per HRU",
        method: "Area-weighted spatial average of gridded tmax over HRU polygons",
        type: "gis_zonal",
      },
      {
        name: "tmin (CBH)",
        desc: "Daily minimum temperature per HRU",
        method: "Area-weighted spatial average of gridded tmin over HRU polygons",
        type: "gis_zonal",
      },
      {
        name: "rain_cbh_adj",
        desc: "Monthly rain adjustment factor per HRU (12 values)",
        method: "Calibration; initial from PRISM/Daymet ratio or set to 1.0",
        type: "calibration",
      },
      {
        name: "snow_cbh_adj",
        desc: "Monthly snow adjustment factor per HRU",
        method: "Calibration; initial 1.0",
        type: "calibration",
      },
      {
        name: "tmax_cbh_adj",
        desc: "Monthly max temp lapse rate adjustment per HRU",
        method: "Calibration or from elevation-temperature regression by month",
        type: "calibration",
      },
      {
        name: "tmin_cbh_adj",
        desc: "Monthly min temp lapse rate adjustment per HRU",
        method: "Calibration or from elevation-temperature regression by month",
        type: "calibration",
      },
      {
        name: "tmax_allsnow",
        desc: "Maximum temp (°F) when all precip is snow",
        method: "Default 32°F; regional sensitivity analysis can refine",
        type: "default_value",
      },
      {
        name: "tmax_allrain_offset",
        desc: "Offset from tmax_allsnow for all-rain threshold",
        method: "Calibration; default ~6°F",
        type: "calibration",
      },
      {
        name: "adjmix_rain",
        desc: "Monthly adjustment factor for mixed precipitation events",
        method: "Calibration; initial 1.0",
        type: "calibration",
      },
    ],
  },

  solar_et: {
    name: "Solar Radiation / ET",
    icon: "☀️",
    color: "#FF8C00",
    datasets: [
      {
        name: "Computed from DEM/Aspect/Latitude",
        desc: "Potential clear-sky solar radiation (soltab tables)",
        resolution: "Per HRU",
        source: "Derived (pywatershed built-in or external tool)",
        accessor: "pywatershed soltab utility",
        format: "NumPy arrays / NetCDF",
      },
    ],
    parameters: [
      {
        name: "soltab_potsw",
        desc: "Potential shortwave radiation table (nhru × 366 days)",
        method: "Computed from hru_lat, hru_slope, hru_aspect using Swift (1976) / soltab algorithm",
        type: "derived_formula",
      },
      {
        name: "soltab_horad_potsw",
        desc: "Horizontal potential shortwave radiation (nhru × 366)",
        method: "Same soltab algorithm with slope=0",
        type: "derived_formula",
      },
      {
        name: "jh_coef",
        desc: "Monthly Jensen-Haise PET coefficient (12 values)",
        method: "Empirical: derived from mean monthly tmin, elevation, & vapor pressure relationship",
        type: "derived_formula",
      },
      {
        name: "jh_coef_hru",
        desc: "Per-HRU Jensen-Haise PET air temp coefficient",
        method: "Derived from elevation-dependent saturation vapor pressure curve",
        type: "derived_formula",
      },
      {
        name: "radmax",
        desc: "Maximum fraction of potential SW reaching ground",
        method: "Default 0.8; can be adjusted for atmospheric clarity",
        type: "default_value",
      },
      {
        name: "dday_slope",
        desc: "Monthly slope of degree-day vs radiation relationship",
        method: "Regression of obs radiation on tmax; or from calibration",
        type: "calibration",
      },
      {
        name: "dday_intcp",
        desc: "Monthly intercept of degree-day vs radiation",
        method: "Regression or calibration",
        type: "calibration",
      },
      {
        name: "ppt_rad_adj",
        desc: "Radiation adjustment on precipitation days (monthly)",
        method: "Default or from obs radiation data analysis",
        type: "default_value",
      },
      {
        name: "radj_sppt / radj_wppt",
        desc: "Summer/winter precipitation radiation adjustment",
        method: "Default 0.44 / 0.5",
        type: "default_value",
      },
      {
        name: "epan_coef",
        desc: "Monthly evaporation pan coefficient (12 values)",
        method: "Literature values by region, typically 0.6–0.8",
        type: "lookup_table",
      },
      {
        name: "potet_sublim",
        desc: "Fraction of PET that can sublimate from snow",
        method: "Default 0.75",
        type: "default_value",
      },
    ],
  },

  snow: {
    name: "Snow",
    icon: "❄️",
    color: "#4682B4",
    datasets: [
      {
        name: "SNODAS",
        desc: "Snow Data Assimilation System (calibration target, not input)",
        resolution: "1km, daily",
        source: "NOAA NSIDC",
        accessor: "pygeohydro",
        format: "NetCDF / GeoTIFF",
      },
    ],
    parameters: [
      {
        name: "snarea_curve",
        desc: "Snow depletion curve (11 values × ndeplval)",
        method: "Empirical curves from literature; Regan et al. used CV-based approach",
        type: "lookup_table",
      },
      {
        name: "snarea_thresh",
        desc: "Max SWE threshold for snow depletion curve (inches)",
        method: "Set from historical max SWE or calibration",
        type: "calibration",
      },
      {
        name: "hru_deplcrv",
        desc: "Index of snow depletion curve per HRU",
        method: "Assign based on vegetation type or elevation band",
        type: "lookup_table",
      },
      {
        name: "den_init",
        desc: "Initial snow density (fraction of water)",
        method: "Default 0.10",
        type: "default_value",
      },
      {
        name: "den_max",
        desc: "Maximum snow density",
        method: "Default 0.60",
        type: "default_value",
      },
      {
        name: "settle_const",
        desc: "Snowpack settlement time constant",
        method: "Default 0.10",
        type: "default_value",
      },
      {
        name: "emis_noppt",
        desc: "Emissivity of air on non-precip days",
        method: "Default 0.757",
        type: "default_value",
      },
      {
        name: "cecn_coef",
        desc: "Monthly convection-condensation energy coefficient",
        method: "Default (often monthly values ~2–15)",
        type: "default_value",
      },
      {
        name: "freeh2o_cap",
        desc: "Free water holding capacity of snowpack",
        method: "Default 0.05",
        type: "default_value",
      },
      {
        name: "albset_rna / albset_snm / albset_rnm / albset_sna",
        desc: "Albedo reset fractions (4 params)",
        method: "Default values from Anderson (1973)",
        type: "default_value",
      },
      {
        name: "tstorm_mo",
        desc: "Monthly flag for convective storms (12 values, 0 or 1)",
        method: "Set based on regional climate: 1 for summer months in continental areas",
        type: "lookup_table",
      },
    ],
  },

  runoff_depression: {
    name: "Runoff / Impervious / Depression Storage",
    icon: "💧",
    color: "#1E90FF",
    datasets: [
      {
        name: "NLCD Impervious",
        desc: "NLCD Impervious Surface",
        resolution: "30m",
        source: "USGS MRLC",
        accessor: "pygeohydro",
        format: "GeoTIFF",
      },
      {
        name: "NHDPlus HR / Waterbodies",
        desc: "High-res hydrography for depression/lake features",
        resolution: "1:24,000",
        source: "USGS",
        accessor: "pynhd",
        format: "GDB / GPKG",
      },
    ],
    parameters: [
      {
        name: "carea_max",
        desc: "Maximum contributing area for surface runoff",
        method: "Calibration; initial from soil/slope/imperv relationship",
        type: "calibration",
      },
      {
        name: "smidx_coef",
        desc: "Coefficient in non-linear surface runoff equation",
        method: "Calibration",
        type: "calibration",
      },
      {
        name: "smidx_exp",
        desc: "Exponent in non-linear surface runoff equation",
        method: "Calibration",
        type: "calibration",
      },
      {
        name: "imperv_stor_max",
        desc: "Maximum impervious retention storage (inches)",
        method: "Lookup by land use: typically 0.01–0.05 inches",
        type: "lookup_table",
      },
      {
        name: "dprst_frac",
        desc: "Fraction of HRU area with surface depressions",
        method: "GIS analysis of NHDPlus waterbodies / wetlands overlay with HRU",
        type: "gis_overlay",
      },
      {
        name: "dprst_area_max",
        desc: "Maximum area of surface depressions (acres)",
        method: "From NHDPlus waterbody areas clipped to HRU",
        type: "gis_overlay",
      },
      {
        name: "dprst_depth_avg",
        desc: "Average depth of surface depressions (inches)",
        method: "Regional estimates or DEM-derived; default ~20–48 inches",
        type: "default_value",
      },
      {
        name: "dprst_flow_coef / dprst_seep_rate_open",
        desc: "Depression outflow and seepage coefficients",
        method: "Calibration",
        type: "calibration",
      },
    ],
  },

  groundwater: {
    name: "Groundwater",
    icon: "🪨",
    color: "#2F4F4F",
    datasets: [
      {
        name: "GLHYMPS",
        desc: "Global Hydrogeology Maps (permeability/porosity)",
        resolution: "~1km polygons",
        source: "Gleeson et al.",
        accessor: "Download / pygeohydro",
        format: "Shapefile / GeoTIFF",
      },
      {
        name: "USGS Surficial Geology / Aquifer Maps",
        desc: "Aquifer characteristics",
        resolution: "Varies",
        source: "USGS",
        accessor: "ScienceBase / NWIS",
        format: "Shapefile",
      },
    ],
    parameters: [
      {
        name: "gwflow_coef",
        desc: "GW reservoir linear discharge coefficient",
        method: "Calibration; initial from baseflow recession analysis",
        type: "calibration",
      },
      {
        name: "gwsink_coef",
        desc: "GW reservoir sink (loss) coefficient",
        method: "Calibration or set to 0 (no deep losses)",
        type: "calibration",
      },
      {
        name: "gwstor_init",
        desc: "Initial GW storage (inches)",
        method: "Spin-up / restart or default",
        type: "default_value",
      },
      {
        name: "gwstor_min",
        desc: "Minimum GW storage for flow to occur (inches)",
        method: "Default 0.0",
        type: "default_value",
      },
    ],
  },

  routing: {
    name: "Channel Routing / Hydrography",
    icon: "🌊",
    color: "#0077B6",
    datasets: [
      {
        name: "NHDPlus v2.1 / HR",
        desc: "National Hydrography Dataset Plus (flowlines, VPU, comids)",
        resolution: "1:100,000 / 1:24,000",
        source: "USGS / EPA",
        accessor: "pynhd",
        format: "GDB / GPKG / Parquet",
      },
      {
        name: "Geospatial Fabric v1.1",
        desc: "NHM discretization with HRU-segment topology",
        resolution: "~115K HRUs, ~57K segments (CONUS)",
        source: "USGS (Viger & Bock 2014)",
        accessor: "ScienceBase / onhm-fetcher",
        format: "GDB / Shapefile / NetCDF",
      },
    ],
    parameters: [
      {
        name: "tosegment",
        desc: "Downstream segment ID for each segment",
        method: "From GF / NHDPlus flowline topology (from-to node table)",
        type: "gis_topology",
      },
      {
        name: "hru_segment",
        desc: "Segment ID that each HRU drains to",
        method: "From GF / spatial join of HRU to flowline",
        type: "gis_topology",
      },
      {
        name: "K_coef",
        desc: "Muskingum storage coefficient (hours)",
        method: "Estimated from segment length, velocity, slope: K = seg_length / velocity",
        type: "derived_formula",
      },
      {
        name: "x_coef",
        desc: "Muskingum weighting factor (0–0.5)",
        method: "Default 0.2; or from channel geometry analysis",
        type: "default_value",
      },
      {
        name: "seg_length",
        desc: "Length of each stream segment (meters)",
        method: "Computed from GF/NHDPlus flowline geometry (geodesic length)",
        type: "gis_geometry",
      },
      {
        name: "obsin_segment",
        desc: "Segment ID receiving observed inflow (for calibration)",
        method: "Mapped from NWIS gage location to nearest segment",
        type: "gis_overlay",
      },
      {
        name: "segment_type",
        desc: "Segment type flag",
        method: "From GF attributes",
        type: "gis_topology",
      },
    ],
  },
};

const derivationMethods = {
  gis_zonal: {
    label: "GIS Zonal Statistics",
    desc: "Area-weighted mean/dominant of raster over HRU polygons",
    color: "#3498db",
    hydroparam: "Core parameterization op — exactextract / rasterstats on polygon batches",
  },
  gis_reclassify: {
    label: "GIS Reclassify + Zonal",
    desc: "Reclassify raster values via lookup, then zonal stats",
    color: "#2ecc71",
    hydroparam: "Two-step: reclassify raster in-memory, then zonal dominant class",
  },
  gis_geometry: {
    label: "Geometry Computation",
    desc: "Computed directly from HRU/segment polygon/line geometry",
    color: "#9b59b6",
    hydroparam: "GeoPandas/Shapely operations on GF geometry",
  },
  gis_overlay: {
    label: "GIS Overlay / Spatial Join",
    desc: "Intersect HRU polygons with another feature layer",
    color: "#e67e22",
    hydroparam: "GeoPandas overlay/sjoin operations",
  },
  gis_topology: {
    label: "Network Topology",
    desc: "From hydrographic network connectivity tables",
    color: "#1abc9c",
    hydroparam: "Direct from GF/NHDPlus attribute tables — pynhd network trace",
  },
  lookup_table: {
    label: "Lookup Table / Literature",
    desc: "Values assigned from published tables keyed on land class, soil type, etc.",
    color: "#f39c12",
    hydroparam: "Plugin lookup tables in YAML config; user-customizable",
  },
  derived_formula: {
    label: "Derived via Formula",
    desc: "Computed from other parameters using documented equations",
    color: "#e74c3c",
    hydroparam: "Post-processing step after base params exist; chain in DAG",
  },
  derived_climate: {
    label: "Derived from Climate Stats",
    desc: "Computed from climate normals, growing season, frost dates",
    color: "#c0392b",
    hydroparam: "Requires long-term climate stats (e.g., monthly normals from Daymet/PRISM)",
  },
  calibration: {
    label: "Calibration Parameter",
    desc: "Optimized during calibration; initial guess from data or defaults",
    color: "#8e44ad",
    hydroparam: "Provide physically-based initial values; final via external calibration tool",
  },
  default_value: {
    label: "Default / Constant",
    desc: "Standard PRMS default values, well-established in literature",
    color: "#7f8c8d",
    hydroparam: "Ship as defaults in process config; user can override",
  },
};

function DatasetCard({ ds }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.03)",
      border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: "8px",
      padding: "10px 12px",
      marginBottom: "6px",
    }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: "8px", flexWrap: "wrap" }}>
        <span style={{ fontFamily: "monospace", fontSize: "13px", fontWeight: 700, color: "#f1c40f" }}>
          {ds.name}
        </span>
        <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.3)", background: "rgba(255,255,255,0.05)", padding: "2px 6px", borderRadius: "3px" }}>
          {ds.resolution}
        </span>
        <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.25)" }}>
          {ds.source}
        </span>
      </div>
      <p style={{ fontSize: "11px", color: "rgba(255,255,255,0.45)", margin: "4px 0 2px" }}>{ds.desc}</p>
      <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.3)" }}>
        <span style={{ color: "#48c9b0" }}>accessor:</span> {ds.accessor} &nbsp;|&nbsp; 
        <span style={{ color: "#48c9b0" }}>format:</span> {ds.format}
      </div>
    </div>
  );
}

function ParamRow({ p }) {
  const m = derivationMethods[p.type] || {};
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "170px 1fr 100px",
      gap: "8px",
      padding: "6px 4px",
      borderBottom: "1px solid rgba(255,255,255,0.03)",
      alignItems: "baseline",
    }}>
      <span style={{ fontFamily: "monospace", fontSize: "11px", color: "#e6e6e6", fontWeight: 600 }}>
        {p.name}
      </span>
      <div>
        <span style={{ fontSize: "11px", color: "rgba(255,255,255,0.5)" }}>{p.desc}</span>
        <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.35)", marginTop: "2px" }}>
          → {p.method}
        </div>
      </div>
      <span style={{
        fontSize: "9px",
        fontWeight: 600,
        color: m.color || "#aaa",
        textTransform: "uppercase",
        letterSpacing: "0.04em",
        textAlign: "right",
      }}>
        {m.label || p.type}
      </span>
    </div>
  );
}

export default function DatasetParamMap() {
  const [selectedCat, setSelectedCat] = useState("topography");
  const [viewMode, setViewMode] = useState("categories");

  const catKeys = Object.keys(dataCategories);
  const cat = dataCategories[selectedCat];

  // Aggregate stats
  const allParams = catKeys.flatMap(k => dataCategories[k].parameters);
  const typeCounts = {};
  allParams.forEach(p => {
    typeCounts[p.type] = (typeCounts[p.type] || 0) + 1;
  });

  return (
    <div style={{
      background: "#0d1117",
      color: "#e6e6e6",
      minHeight: "100vh",
      fontFamily: "'Segoe UI', system-ui, sans-serif",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
        @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{ padding: "24px 28px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: "10px", marginBottom: "4px" }}>
          <h1 style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "20px", fontWeight: 700, margin: 0 }}>
            hydro-param
          </h1>
          <span style={{ fontSize: "11px", color: "rgba(255,255,255,0.3)", background: "rgba(255,255,255,0.05)", padding: "2px 8px", borderRadius: "4px" }}>
            Dataset → Parameter Mapping for pywatershed/NHM-PRMS
          </span>
        </div>
        <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.35)", margin: "4px 0 14px" }}>
          Curated datasets by category, target parameters, and derivation methods
        </p>

        <div style={{ display: "flex", gap: "4px" }}>
          {[
            { key: "categories", label: "By Data Category" },
            { key: "methods", label: "By Derivation Method" },
            { key: "yml", label: "YML Catalog Outline" },
          ].map(tab => (
            <button key={tab.key} onClick={() => setViewMode(tab.key)} style={{
              background: viewMode === tab.key ? "rgba(255,255,255,0.1)" : "transparent",
              border: `1px solid ${viewMode === tab.key ? "rgba(255,255,255,0.15)" : "rgba(255,255,255,0.05)"}`,
              borderRadius: "6px", padding: "5px 12px",
              color: viewMode === tab.key ? "#fff" : "rgba(255,255,255,0.4)",
              fontSize: "11px", fontWeight: viewMode === tab.key ? 600 : 400, cursor: "pointer",
            }}>
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "20px 28px" }}>

        {viewMode === "categories" && (
          <div style={{ display: "flex", gap: "20px" }}>
            {/* Category sidebar */}
            <div style={{ width: "240px", flexShrink: 0, display: "flex", flexDirection: "column", gap: "4px" }}>
              {catKeys.map(k => {
                const c = dataCategories[k];
                const sel = selectedCat === k;
                return (
                  <div key={k} onClick={() => setSelectedCat(k)} style={{
                    background: sel ? `rgba(255,255,255,0.06)` : "transparent",
                    border: `1px solid ${sel ? c.color + "66" : "rgba(255,255,255,0.04)"}`,
                    borderRadius: "8px", padding: "10px 12px", cursor: "pointer",
                    borderLeft: sel ? `3px solid ${c.color}` : "3px solid transparent",
                    transition: "all 0.2s",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                      <span style={{ fontSize: "16px" }}>{c.icon}</span>
                      <span style={{
                        fontSize: "12px", fontWeight: sel ? 700 : 500,
                        color: sel ? "#fff" : "rgba(255,255,255,0.5)",
                      }}>
                        {c.name}
                      </span>
                    </div>
                    <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.25)", marginTop: "2px", paddingLeft: "24px" }}>
                      {c.datasets.length} dataset{c.datasets.length > 1 ? "s" : ""} · {c.parameters.length} params
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Detail panel */}
            <div style={{ flex: 1, minWidth: 0, animation: "fadeIn 0.25s ease" }} key={selectedCat}>
              <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                <span style={{ fontSize: "24px" }}>{cat.icon}</span>
                <h2 style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: "18px", fontWeight: 700, color: cat.color, margin: 0 }}>
                  {cat.name}
                </h2>
              </div>

              {/* Datasets */}
              <div style={{ marginBottom: "20px" }}>
                <h3 style={{ fontSize: "11px", fontWeight: 600, color: "#f1c40f", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" }}>
                  📊 Recommended Datasets for data.yml
                </h3>
                {cat.datasets.map((ds, i) => <DatasetCard key={i} ds={ds} />)}
              </div>

              {/* Parameters derived from these datasets */}
              <div>
                <h3 style={{ fontSize: "11px", fontWeight: 600, color: "#48c9b0", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" }}>
                  🔧 Parameters Derived ({cat.parameters.length})
                </h3>
                {cat.parameters.map((p, i) => <ParamRow key={i} p={p} />)}
              </div>
            </div>
          </div>
        )}

        {viewMode === "methods" && (
          <div>
            <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.4)", marginBottom: "16px" }}>
              All ~{allParams.length} parameters grouped by how they are derived. This informs the hydro-param processing pipeline architecture.
            </p>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginBottom: "20px" }}>
              {Object.entries(derivationMethods).map(([key, m]) => (
                <div key={key} style={{
                  background: "rgba(255,255,255,0.02)",
                  border: `1px solid ${m.color}33`,
                  borderRadius: "10px", padding: "14px",
                  borderLeft: `3px solid ${m.color}`,
                }}>
                  <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginBottom: "6px" }}>
                    <span style={{ fontFamily: "monospace", fontSize: "13px", fontWeight: 700, color: m.color }}>
                      {m.label}
                    </span>
                    <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.3)" }}>
                      ({typeCounts[key] || 0} params)
                    </span>
                  </div>
                  <p style={{ fontSize: "11px", color: "rgba(255,255,255,0.45)", margin: "0 0 6px" }}>{m.desc}</p>
                  <p style={{ fontSize: "10px", color: "#48c9b0", margin: "0 0 8px", fontStyle: "italic" }}>
                    hydro-param: {m.hydroparam}
                  </p>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "4px" }}>
                    {allParams.filter(p => p.type === key).map((p, i) => (
                      <span key={i} style={{
                        fontFamily: "monospace", fontSize: "9px",
                        background: `${m.color}15`, border: `1px solid ${m.color}25`,
                        borderRadius: "3px", padding: "2px 5px",
                        color: "rgba(255,255,255,0.6)",
                      }}>
                        {p.name}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* Pipeline summary */}
            <div style={{
              background: "rgba(72,201,176,0.05)", border: "1px solid rgba(72,201,176,0.2)",
              borderRadius: "10px", padding: "16px",
            }}>
              <h3 style={{ fontFamily: "monospace", fontSize: "14px", color: "#48c9b0", margin: "0 0 8px" }}>
                Suggested hydro-param Pipeline Order
              </h3>
              <div style={{ fontSize: "12px", color: "rgba(255,255,255,0.55)", lineHeight: 1.8 }}>
                <div><strong style={{ color: "#9b59b6" }}>1.</strong> Geometry computation → hru_area, hru_lat, seg_length</div>
                <div><strong style={{ color: "#1abc9c" }}>2.</strong> Topology extraction → tosegment, hru_segment</div>
                <div><strong style={{ color: "#3498db" }}>3.</strong> Zonal stats (DEM) → hru_elev, hru_slope, hru_aspect</div>
                <div><strong style={{ color: "#2ecc71" }}>4.</strong> Zonal stats (land cover) → cov_type, hru_percent_imperv, covden_sum</div>
                <div><strong style={{ color: "#e67e22" }}>5.</strong> Zonal stats (soils) → soil_type, soil_moist_max, soil_rechr_max_frac</div>
                <div><strong style={{ color: "#3498db" }}>6.</strong> Zonal stats (climate) → prcp, tmax, tmin CBH time series</div>
                <div><strong style={{ color: "#f39c12" }}>7.</strong> Lookup tables → srain_intcp, wrain_intcp, snow_intcp, imperv_stor_max, etc.</div>
                <div><strong style={{ color: "#e74c3c" }}>8.</strong> Formula derivation → covden_win, soltab tables, jh_coef, K_coef, sat_threshold</div>
                <div><strong style={{ color: "#c0392b" }}>9.</strong> Climate-derived → transp_beg, transp_end</div>
                <div><strong style={{ color: "#7f8c8d" }}>10.</strong> Defaults → snow params, albedo, emissivity, init fractions</div>
                <div><strong style={{ color: "#8e44ad" }}>11.</strong> Calibration seed values → all *_coef, *_adj, *_exp params</div>
              </div>
            </div>
          </div>
        )}

        {viewMode === "yml" && (
          <div>
            <p style={{ fontSize: "12px", color: "rgba(255,255,255,0.4)", marginBottom: "16px" }}>
              Proposed structure for hydro-param data.yml catalog files organized by category.
            </p>

            <pre style={{
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: "10px",
              padding: "20px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: "11px",
              color: "rgba(255,255,255,0.7)",
              lineHeight: 1.7,
              overflowX: "auto",
              whiteSpace: "pre",
            }}>
{`# hydro-param data catalog
# Source datasets for pywatershed/NHM-PRMS parameterization

topography:
  ned_3dep:
    source: "USGS 3DEP"
    resolution: "10m"
    accessor: "pygeohydro.get_3dep_dem"
    format: "COG/GeoTIFF"
    variables: [elevation]
    derives:
      - hru_elev    # zonal_mean(dem, hru_poly)
      - hru_slope   # zonal_mean(slope(dem), hru_poly)
      - hru_aspect  # circular_mean(aspect(dem), hru_poly)

landcover:
  nlcd:
    source: "USGS MRLC"
    resolution: "30m"
    accessor: "pygeohydro.get_nlcd"
    format: "GeoTIFF"
    variables: [land_cover, impervious, canopy]
    derives:
      - cov_type            # reclassify + zonal_dominant
      - covden_sum          # zonal_mean(canopy) or lookup(cov_type)
      - hru_percent_imperv  # zonal_mean(impervious)
      - hru_type            # overlay with waterbodies

soils:
  statsgo2:
    source: "USDA NRCS"
    accessor: "pygeohydro.get_ssurgo"
    format: "Shapefile/GDB"
    variables: [texture, awc, ksat, depth, porosity]
    derives:
      - soil_type           # reclassify(texture → sand/loam/clay)
      - soil_moist_max      # awc × depth
      - soil_rechr_max_frac # awc_upper / awc_total
      - sat_threshold       # soil_moist_max × depth_ratio
  ssurgo:  # higher-res alternative
    source: "USDA NRCS"
    accessor: "pygeohydro.get_ssurgo"
    format: "SQLite/GDB"
    variables: [texture, awc, ksat, depth, porosity]

climate_forcing:
  daymet_v4:
    source: "ORNL DAAC"
    resolution: "1km daily"
    accessor: "pydaymet.get_bygeom / gdptools"
    format: "NetCDF/Zarr"
    variables: [prcp, tmax, tmin, srad, dayl, vp]
    derives:
      - prcp_cbh     # area_weighted_mean(prcp, hru_poly)
      - tmax_cbh     # area_weighted_mean(tmax, hru_poly)
      - tmin_cbh     # area_weighted_mean(tmin, hru_poly)
      - transp_beg   # from frost-free period
      - transp_end   # from killing frost date
  gridmet:
    source: "U of Idaho"
    resolution: "4km daily"
    accessor: "pygridmet / gdptools"
    format: "NetCDF"
    variables: [pr, tmmx, tmmn, srad, pet, vs]
  conus404_ba:  # new high-res option
    source: "USGS"
    resolution: "4km hourly→daily"
    accessor: "S3 Zarr / gdptools"
    format: "Zarr"
    variables: [T2, RAIN, SNOW, SWDOWN]

hydrography:
  geospatial_fabric:
    source: "USGS (Viger & Bock 2014)"
    accessor: "ScienceBase / onhm-fetcher"
    format: "GDB/GPKG"
    variables: [hru_poly, seg_line, topology]
    derives:
      - hru_area       # geodesic_area(hru_poly)
      - hru_lat        # centroid(hru_poly).y
      - tosegment      # attribute: tosegment
      - hru_segment    # attribute: hru_segment
      - seg_length     # geodesic_length(seg_line)
  nhdplus_hr:
    source: "USGS"
    accessor: "pynhd"
    format: "GPKG/Parquet"
    variables: [flowlines, catchments, waterbodies]
    derives:
      - dprst_frac       # waterbody_area / hru_area
      - dprst_area_max   # clip(waterbody, hru).area

solar_radiation:
  soltab:
    source: "Derived from topography"
    accessor: "pywatershed.utils (soltab algorithm)"
    requires: [hru_lat, hru_slope, hru_aspect]
    derives:
      - soltab_potsw        # Swift 1976 algorithm
      - soltab_horad_potsw  # flat surface variant

groundwater:
  glhymps:
    source: "Gleeson et al."
    resolution: "~1km"
    accessor: "download / pygeohydro"
    format: "Shapefile/GeoTIFF"
    variables: [logK_ice, porosity]

# ------ Lookup Tables (ship with hydro-param) ------
lookup_tables:
  nlcd_to_prms_cov_type:
    maps: "NLCD class → cov_type (0-4)"
  cov_type_to_interception:
    maps: "cov_type → srain_intcp, wrain_intcp, snow_intcp"
  cov_type_to_winter_density:
    maps: "cov_type → covden_win reduction factor"
  imperv_stor_lookup:
    maps: "land_use → imperv_stor_max"
  snow_depletion_curves:
    maps: "curve_index → 11 SCA values"

# ------ Defaults (PRMS literature) ------
defaults:
  snow:
    den_init: 0.10
    den_max: 0.60
    settle_const: 0.10
    emis_noppt: 0.757
    freeh2o_cap: 0.05
    potet_sublim: 0.75
  albedo:
    albset_rna: 0.8
    albset_snm: 0.1
    albset_rnm: 0.6
    albset_sna: 0.05
  initial_conditions:
    soil_moist_init_frac: 0.5
    soil_rechr_init_frac: 0.5
    ssstor_init_frac: 0.0
    gwstor_init: 2.0
  routing:
    x_coef: 0.2
    tmax_allsnow: 32.0
    radmax: 0.8

# ------ Calibration Parameters (initial seeds) ------
calibration:
  runoff:
    - carea_max        # init: f(soil_type, slope, imperv)
    - smidx_coef       # init: 0.01
    - smidx_exp        # init: 0.3
  soil_drainage:
    - soil2gw_max      # init: f(ksat)
    - ssr2gw_rate      # init: f(ksat)
    - ssr2gw_exp       # init: 1.0
    - slowcoef_lin     # init: f(ksat, slope)
    - slowcoef_sq      # init: 0.1
    - fastcoef_lin     # init: f(ksat)
    - fastcoef_sq      # init: 0.0
    - pref_flow_den    # init: 0.0
  groundwater:
    - gwflow_coef      # init: recession analysis
    - gwsink_coef      # init: 0.0
  snow:
    - snarea_thresh    # init: from historical SWE
  climate_adj:
    - rain_cbh_adj     # init: 1.0 (12 monthly)
    - snow_cbh_adj     # init: 1.0
    - tmax_cbh_adj     # init: 0.0
    - tmin_cbh_adj     # init: 0.0
    - tmax_allrain_offset # init: 6.0
    - adjmix_rain      # init: 1.0
  radiation:
    - dday_slope       # init: regression or regional
    - dday_intcp       # init: regression or regional`}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
