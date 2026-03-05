# hydro-param pywatershed Parameter Inventory & Classification

## Context

A complete audit of all pywatershed/PRMS parameters produced by hydro-param, cross-referenced
against:
- What is actively mapped in `pw-check/configs/pywatershed_run.yml` (user-configurable source)
- What is automatically derived by the plugin (no user config needed)
- What is currently a default/placeholder awaiting data-driven derivation
- The full parameter_metadata.yml spec (95 parameters + 5 forcing = 100 total)

Used as a living reference for prioritizing derivation improvements.

---

## Category Definitions

| Code | Category | Description |
|------|----------|-------------|
| **DS** | Derived — Spatial | Direct output of gdptools ZonalGen zonal stats from a source dataset |
| **DF** | Derived — Formula | Computed from spatial inputs via a physics/math formula |
| **DR** | Derived — Reclassify | Category code mapped to PRMS integer via lookup table |
| **DL** | Derived — Lookup Table | Value assigned from a class-indexed lookup (not a formula) |
| **DA** | Derived — Algorithm | Computed via a multi-step algorithm (Swift 1976, Jensen-Haise, Manning's) |
| **DT** | Derived — Topology | Extracted from fabric routing network structure |
| **DC** | Derived — Climate | Derived from temporal climate data aggregated to normals |
| **CS-F** | Calib Seed — Formula | Physically-based first guess from data (will be calibrated) |
| **CS-C** | Calib Seed — Constant | Domain-wide constant first guess (will be calibrated) |
| **DEF** | Default — Physical | Literature/reference constant; spatially uniform; physically grounded |
| **PH** | Placeholder | Uniform value chosen for runtime stability only; known to need improvement |
| **IC** | Initial Condition | Model state variable at t=0; not a process parameter |
| **FRC** | Forcing | Time-varying climate input |
| **STR** | Structural | Dimensional index, unit flag, or conversion constant |

---

## Full Parameter Table

**Column headers:**
- **Step** — derivation step in pywatershed.py
- **Cat** — category code (see above)
- **Source Dataset** — primary data source
- **Dim** — dimension (nhru, nseg, nmo×nhru, ndoy×nhru, scalar)
- **In yml?** — actively mapped in pywatershed_run.yml (✓) or auto-derived (—) or N/A
- **Issue** — open improvement issue
- **Notes** — current value or key detail

### STEP 1 — Geometry

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `hru_area` | 1 | DF | geospatial fabric | nhru | — | — | m²→acres via EPSG:5070 |
| `hru_lat` | 1 | DF | geospatial fabric | nhru | — | — | WGS84 centroid latitude |

### STEP 2 — Topology

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `tosegment` | 2 | DT | geospatial fabric | nseg | — | — | Downstream segment index (0=outlet) |
| `tosegment_nhm` | 2 | DT | geospatial fabric | nseg | — | #147 | NHM segment ID mapping |
| `hru_segment` | 2 | DT | geospatial fabric | nhru | — | — | Segment each HRU drains to |
| `seg_length` | 2 | DF | geospatial fabric | nseg | — | — | Geodesic polyline length (m) |

### STEP 3 — Topography (3DEP 10m DEM)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `hru_elev` | 3 | DS | 3DEP 10m | nhru | ✓ | — | Zonal mean elevation (m) |
| `hru_slope` | 3 | DF | 3DEP 10m | nhru | ✓ | — | tan(zonal mean slope°) → rise/run fraction |
| `hru_aspect` | 3 | DF | 3DEP 10m | nhru | ✓ | — | Circular mean via sin/cos components (°) |

### STEP 4 — Land Cover (NLCD)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `cov_type` | 4 | DR | NLCD | nhru | ✓ | — | NLCD→PRMS 5-class (0=bare…4=conifer) |
| `covden_sum` | 4 | DS | NLCD tree canopy | nhru | — | — | Zonal mean canopy %, /100 |
| `hru_percent_imperv` | 4 | DS | NLCD impervious | nhru | ✓ | — | Zonal mean impervious %, /100 |

### STEP 5 — Soils (gNATSGO / POLARIS)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `soil_type` | 5 | DR | gNATSGO / POLARIS | nhru | ✓ | — | USDA texture→PRMS 3-class (1=sand,2=loam,3=clay) |
| `soil_moist_max` | 5 | DF | gNATSGO aws0_100 | nhru | ✓ | — | AWC 0–100cm mm→inches, clipped [0.5,20] |
| `soil_rechr_max_frac` | 5 | PH | — | nhru | ✓ | [#151](https://github.com/rmcd-mscb/hydro-param/issues/151) | Fixed 0.4 default; needs aws0_50/aws0_100 ratio |
| `sat_threshold` | 5 | PH | POLARIS θ_s | nhru | ✓ | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | θ_s×depth; currently 999.0 placeholder |

### STEP 6 — Waterbody (NHDPlus)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `dprst_frac` | 6 | DF | NHDPlus waterbodies | nhru | — | — | Clipped waterbody area / hru_area |
| `hru_type` | 6 | DF | NHDPlus waterbodies | nhru | — | — | 2 if dprst_frac>0.5 else 1 |

### STEP 7 — Forcing (Temporal Climate)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `prcp` | 7 | FRC | gridMET / SNODAS | (time, nhru) | — | — | mm→inches/day |
| `tmax` | 7 | FRC | gridMET | (time, nhru) | — | — | °C→°F |
| `tmin` | 7 | FRC | gridMET | (time, nhru) | — | — | °C→°F |
| `swrad` | 7 | FRC | gridMET | (time, nhru) | — | — | W/m²→Langleys/day |
| `potet` | 7 | FRC | gridMET | (time, nhru) | — | — | Written as forcing file |

### STEP 8 — Lookup Tables (from cov_type)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `srain_intcp` | 8 | DL | cov_type_to_interception.yml | nhru | — | — | Summer rain interception (inches) |
| `wrain_intcp` | 8 | DL | cov_type_to_interception.yml | nhru | — | — | Winter rain interception (inches) |
| `snow_intcp` | 8 | DL | cov_type_to_interception.yml | nhru | — | — | Snow interception (inches) |
| `imperv_stor_max` | 8 | DL | constant | nhru | — | — | 0.03 inches (uniform) |
| `covden_win` | 8 | DF | cov_type_winter_reduction.yml | nhru | — | — | covden_sum × class reduction factor |

### STEP 9 — Solar Radiation Tables (Swift 1976)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `soltab_potsw` | 9 | DA | hru_lat, hru_slope, hru_aspect | (ndoy, nhru) | — | [#156](https://github.com/rmcd-mscb/hydro-param/issues/156) | Potential SW on slope (cal/cm²/day) |
| `soltab_horad_potsw` | 9 | DA | hru_lat, hru_slope, hru_aspect | (ndoy, nhru) | — | [#156](https://github.com/rmcd-mscb/hydro-param/issues/156) | Potential SW horizontal (cal/cm²/day) |
| `soltab_sunhrs` | 9 | DA | hru_lat, hru_slope, hru_aspect | (ndoy, nhru) | — | [#156](https://github.com/rmcd-mscb/hydro-param/issues/156) | Hours of direct sunlight |

### STEP 10 — PET Coefficients (Jensen-Haise)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `jh_coef` | 10 | DC | gridMET monthly normals | (nmo, nhru) | ✓ | [#153](https://github.com/rmcd-mscb/hydro-param/issues/153) | 1/Ct from monthly SVP; upgrade to PRISM |
| `jh_coef_hru` | 10 | DA | gridMET + hru_elev | nhru | — | — | Tx = -2.5 - 0.14*(emax-emin) - elev_ft/1000 |

### STEP 11 — Transpiration Timing

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `transp_beg` | 11 | DC | gridMET monthly normals | nhru | ✓ | [#153](https://github.com/rmcd-mscb/hydro-param/issues/153) | First month tmin>32°F; upgrade to PRISM |
| `transp_end` | 11 | DC | gridMET monthly normals | nhru | ✓ | [#153](https://github.com/rmcd-mscb/hydro-param/issues/153) | Last month (Jul+) tmin<32°F; upgrade to PRISM |

### STEP 12 — Routing (Muskingum / NHDPlus)

| Parameter | Step | Cat | Source Dataset | Dim | In yml? | Issue | Notes |
|-----------|------|-----|---------------|-----|---------|-------|-------|
| `K_coef` | 12 | DA | NHDPlus + fabric | nseg | — | — | Manning's eqn: seg_len / velocity (hours) |
| `x_coef` | 12 | CS-C | — | nseg | — | — | Fixed 0.2 Muskingum weighting factor |
| `seg_slope` | 12 | DS | NHDPlus VAA | nseg | — | — | Direct NHDPlus slope (m/m) |
| `segment_type` | 12 | DT | geospatial fabric | nseg | — | — | 0=channel, 1=lake |
| `obsin_segment` | 12 | STR | — | nseg | — | — | 0 (no observed inflow station) |
| `mann_n` | 12 | DEF | — | nseg | — | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | 0.04 uniform; needs stream order lookup |
| `seg_depth` | 12 | DEF | — | nseg | — | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | 1.0 ft uniform; needs hydraulic geometry |

### STEP 13 — Defaults & Initial Conditions

#### Snow Process

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `den_init` | 13 | DEF | scalar | 0.10 | — | Initial snowpack density |
| `den_max` | 13 | DEF | scalar | 0.60 | — | Maximum snowpack density |
| `settle_const` | 13 | DEF | scalar | 0.10 | — | Snowpack settle constant |
| `emis_noppt` | 13 | DEF | nhru | 0.757 | — | Emissivity, no precip |
| `freeh2o_cap` | 13 | DEF | nhru | 0.05 | — | Free water holding capacity |
| `potet_sublim` | 13 | DEF | nhru | 0.75 | — | Sublimation fraction of PET |
| `tmax_allsnow` | 13 | DEF | (nmo, nhru) | 32.0°F | — | All-snow temperature threshold |
| `cecn_coef` | 13 | DEF | (nmo, nhru) | 5.0 | — | Convection-condensation energy coef |
| `melt_force` | 13 | PH | nhru | 140 (DOY) | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Force melt start; needs climate normals |
| `melt_look` | 13 | PH | nhru | 90 (DOY) | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Melt lookback window; needs climate normals |
| `snowinfil_max` | 13 | PH | nhru | 2.0 in/day | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs soil_type class lookup |
| `snowpack_init` | 13 | IC | nhru | 0.0 | — | Initial SWE |
| `hru_deplcrv` | 13 | DEF | nhru | 1 | — | Snow depletion curve index |
| `snarea_curve` | 13 | PH | ndeplval(11) | all 1.0 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147)/[#155](https://github.com/rmcd-mscb/hydro-param/issues/155) | **Wrong — needs MODIS+SNODAS** |
| `rad_trncf` | 13 | PH | nhru | 0.5 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Should be 1-covden_win |

#### Albedo

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `albset_rna` | 13 | DEF | scalar | 0.8 | — | Rain-on-snow albedo threshold (new) |
| `albset_rnm` | 13 | DEF | scalar | 0.6 | — | Rain-on-snow albedo threshold (melt) |
| `albset_sna` | 13 | DEF | scalar | 0.05 | — | Snow albedo threshold (new) |
| `albset_snm` | 13 | DEF | scalar | 0.1 | — | Snow albedo threshold (melt) |

#### Radiation

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `radmax` | 13 | DEF | (nmo, nhru) | 0.8 | — | Max fraction of clear-sky radiation |
| `radj_sppt` | 13 | DEF | nhru | 0.44 | — | Spring precip radiation adjustment |
| `radj_wppt` | 13 | DEF | nhru | 0.50 | — | Winter precip radiation adjustment |
| `ppt_rad_adj` | 13 | DEF | (nmo, nhru) | 0.02 in | — | Precip radiation adjustment threshold |
| `radadj_intcp` | 13 | PH | (nmo, nhru) | 1.0 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs regression from observed srad |
| `radadj_slope` | 13 | PH | (nmo, nhru) | 0.0 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs regression from observed srad |
| `tmax_index` | 13 | PH | (nmo, nhru) | 50°F | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs 90th-pct tmax from normals |

#### Atmosphere / Precip Phase

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `tstorm_mo` | 13 | PH | (nmo, nhru) | 0 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs NOAA thunderstorm climatology |

#### Initial Conditions

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `soil_moist_init_frac` | 13 | IC | nhru | 0.5 | — | Initial soil moisture fraction |
| `soil_rechr_init_frac` | 13 | IC | nhru | 0.5 | — | Initial recharge zone moisture fraction |
| `ssstor_init_frac` | 13 | IC | nhru | 0.0 | — | Initial subsurface storage fraction |
| `gwstor_init` | 13 | IC | nhru | 2.0 in | — | Initial groundwater storage |
| `gwstor_min` | 13 | IC | nhru | 0.0 in | — | Minimum groundwater storage |
| `dprst_frac_init` | 13 | IC | nhru | 0.5 | — | Initial depression fraction filled |
| `segment_flow_init` | 13 | IC | nseg | 0.0 | — | Initial channel flow |

#### Depression Storage Process

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `dprst_depth_avg` | 13 | DEF | nhru | 24.0 in | — | Mean depression depth |
| `dprst_et_coef` | 13 | DEF | nhru | 1.0 | — | ET from depression fraction |
| `dprst_flow_coef` | 13 | DEF | nhru | 0.05 | — | Depression outflow coefficient |
| `dprst_frac_open` | 13 | PH | nhru | 1.0 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs NHDPlus ftype classification |
| `dprst_seep_rate_clos` | 13 | PH | nhru | 0.02 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs Ksat from gNATSGO |
| `dprst_seep_rate_open` | 13 | PH | nhru | 0.02 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) | Needs Ksat from gNATSGO |
| `sro_to_dprst_imperv` | 13 | DEF | nhru | 0.2 | — | Runoff to depression (impervious) |
| `sro_to_dprst_perv` | 13 | DEF | nhru | 0.2 | — | Runoff to depression (pervious) |
| `op_flow_thres` | 13 | DEF | nhru | 1.0 | — | Open flow threshold fraction |
| `va_clos_exp` | 13 | DEF | nhru | 0.001 | — | Volume-area closed exponent |
| `va_open_exp` | 13 | DEF | nhru | 0.001 | — | Volume-area open exponent |

#### Soilzone / Groundwater Process

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `transp_tmax` | 13 | DEF | nhru | 500.0 dd | — | Transpiration degree-day threshold |

#### Structural

| Parameter | Step | Cat | Dim | Value | Issue | Notes |
|-----------|------|-----|-----|-------|-------|-------|
| `doy` | 13 | STR | ndoy | 1–366 | — | Day-of-year sequence |
| `hru_in_to_cf` | 13 | DF | nhru | area×3630 | — | hru_area (acres) × 43560/12 |
| `temp_units` | 13 | STR | scalar | 0 | — | 0=°F |
| `elev_units` | 13 | STR | scalar | 1 | — | 1=meters |
| `pref_flow_infil_frac` | 13 | DEF | nhru | 0.0 | — | Preferential flow infiltration fraction |
| `obsout_segment` | 13 | STR | nseg | 0 | — | Observed outflow segment (0=none) |

### STEP 14 — Calibration Seeds

#### Formula-Based Seeds (spatially variable)

| Parameter | Step | Cat | Method | Input | Range | Default | Issue |
|-----------|------|-----|--------|-------|-------|---------|-------|
| `carea_max` | 14 | CS-F | linear | hru_percent_imperv | [0,1] | 0.4 | [#152](https://github.com/rmcd-mscb/hydro-param/issues/152) |
| `smidx_coef` | 14 | CS-F | exp scale | hru_slope | [0.001,0.06] | 0.01 | [#152](https://github.com/rmcd-mscb/hydro-param/issues/152) |
| `soil2gw_max` | 14 | CS-F | fraction_of | soil_moist_max | [0,5] | 0.1 | [#154](https://github.com/rmcd-mscb/hydro-param/issues/154) |
| `snarea_thresh` | 14 | CS-F | fraction_of | soil_moist_max | [0,200] | 50.0 | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147)/[#155](https://github.com/rmcd-mscb/hydro-param/issues/155) |

#### Constant Seeds (uniform — will be calibrated)

| Parameter | Step | Cat | Dim | Value | Range | Issue |
|-----------|------|-----|-----|-------|-------|-------|
| `smidx_exp` | 14 | CS-C | nhru | 0.3 | [0.1,0.8] | [#152](https://github.com/rmcd-mscb/hydro-param/issues/152) |
| `ssr2gw_rate` | 14 | CS-C | nhru | 0.1 | [0,1] | [#154](https://github.com/rmcd-mscb/hydro-param/issues/154) |
| `ssr2gw_exp` | 14 | CS-C | nhru | 1.0 | [0,3] | — |
| `slowcoef_lin` | 14 | CS-C | nhru | 0.015 | [0.001,0.5] | — |
| `slowcoef_sq` | 14 | CS-C | nhru | 0.1 | [0,1] | — |
| `fastcoef_lin` | 14 | CS-C | nhru | 0.09 | [0.001,0.8] | — |
| `fastcoef_sq` | 14 | CS-C | nhru | 0.8 | [0,1] | — |
| `pref_flow_den` | 14 | CS-C | nhru | 0.0 | [0,0.1] | — |
| `gwflow_coef` | 14 | CS-C | nhru | 0.015 | [0.001,0.5] | [#154](https://github.com/rmcd-mscb/hydro-param/issues/154) |
| `gwsink_coef` | 14 | CS-C | nhru | 0.0 | [0,1] | — |
| `rain_cbh_adj` | 14 | CS-C | (nmo,nhru) | 1.0 | [0.5,2] | — |
| `snow_cbh_adj` | 14 | CS-C | (nmo,nhru) | 1.0 | [0.5,2] | — |
| `tmax_cbh_adj` | 14 | CS-C | (nmo,nhru) | 0.0 | [-10,10] | — |
| `tmin_cbh_adj` | 14 | CS-C | (nmo,nhru) | 0.0 | [-10,10] | — |
| `tmax_allrain_offset` | 14 | CS-C | (nmo,nhru) | 1.0 | [0,10] | — |
| `adjmix_rain` | 14 | CS-C | (nmo,nhru) | 1.0 | [0.6,1.4] | — |
| `dday_slope` | 14 | CS-C | (nmo,nhru) | 0.4 | [0.2,0.9] | — |
| `dday_intcp` | 14 | CS-C | (nmo,nhru) | -40.0 | [-60,10] | — |

---

## Summary Statistics

| Category | Count | % of total | Notes |
|----------|-------|-----------|-------|
| DS — Derived Spatial | 7 | 7% | Direct zonal stats from datasets |
| DF — Derived Formula | 11 | 11% | Math/physics applied to spatial inputs |
| DR — Derived Reclassify | 2 | 2% | NLCD→PRMS, USDA→PRMS |
| DL — Derived Lookup Table | 5 | 5% | Interception, imperv storage |
| DA — Derived Algorithm | 5 | 5% | Swift 1976, Jensen-Haise Tx, Manning's |
| DT — Derived Topology | 5 | 5% | Fabric routing network |
| DC — Derived Climate | 4 | 4% | Monthly normals → PET, transp timing |
| **Total Derived** | **39** | **39%** | Spatially variable from data |
| CS-F — Calib Seed Formula | 4 | 4% | Variable; physically-based; calibration start |
| CS-C — Calib Seed Constant | 18 | 18% | Uniform; reasonable start; calibration target |
| **Total Calib Seeds** | **22** | **22%** | |
| DEF — Default Physical | 21 | 21% | Spatially uniform; literature-grounded |
| PH — Placeholder | 13 | 13% | **Known improvement targets** |
| IC — Initial Condition | 7 | 7% | Model state at t=0 |
| FRC — Forcing | 5 | 5% | Time-varying; separate files |
| STR — Structural | 5 | 5% | Indices, flags, conversions |
| **Total Non-Derived** | **61** | **61%** | |
| **GRAND TOTAL** | **100** | | 95 static + 5 forcing |

---

## pywatershed_run.yml: Actively Configured vs Auto-Derived

Of the 39 derived parameters, **13 are explicitly mapped in pywatershed_run.yml**
(the user can override the source dataset). The other 26 are auto-derived by the plugin.

**Mapped in pywatershed_run.yml (13):**

| Parameter | Category | Source in yml |
|-----------|----------|--------------|
| `hru_elev` | DS | dem_3dep_10m / elevation |
| `hru_slope` | DF | dem_3dep_10m / slope |
| `hru_aspect` | DF | dem_3dep_10m / aspect |
| `soil_type` | DR | polaris_30m / sand,silt,clay |
| `sat_threshold` | PH | polaris_30m / theta_s |
| `soil_moist_max` | DF | gnatsgo_rasters / aws0_100 |
| `soil_rechr_max_frac` | PH | gnatsgo_rasters / rootznemc,rootznaws |
| `cov_type` | DR | nlcd_osn_lndcov / LndCov |
| `hru_percent_imperv` | DS | nlcd_osn_fctimp / FctImp |
| `snarea_thresh` | CS-F | snodas / SWE |
| `jh_coef` | DC | gridmet / tmmx,tmmn |
| `transp_beg` | DC | gridmet / tmmn |
| `transp_end` | DC | gridmet / tmmn |

**Auto-derived by plugin (25 derived, not in yml):**
`hru_area`, `hru_lat`, `tosegment`, `tosegment_nhm`, `hru_segment`, `seg_length`,
`covden_sum`, `dprst_frac`, `hru_type`, `covden_win`, `srain_intcp`,
`wrain_intcp`, `snow_intcp`, `imperv_stor_max`, `soltab_potsw`, `soltab_horad_potsw`,
`soltab_sunhrs`, `jh_coef_hru`, `K_coef`, `seg_slope`, `segment_type`, `mann_n`,
`seg_depth`, `hru_in_to_cf`, `x_coef`

---

## Parameter Count vs DRB Reference

Reference: [`test_data/drb_2yr/myparam.param`](https://raw.githubusercontent.com/DOI-USGS/pywatershed/refs/heads/develop/test_data/drb_2yr/myparam.param)

### Count comparison

| Source | Raw Count | Notes |
|--------|-----------|-------|
| DRB reference myparam.param | 157 | Includes dimension vars, control, POI, stream temp |
| — minus dimension vars | −10 | nhru, nsegment, nmonths, nssr, ngw, ndepl, ndeplval, nobs, npoigages, one |
| — minus control/output flags | −6 | print_freq, print_type, precip_units, runoff_units, ppt_zero_thresh, outlet_sta |
| — minus observation/POI | −3 | poi_gage_id, poi_gage_segment, poi_type |
| **DRB true hydrologic parameters** | **138** | |
| — minus stream temp process (PRMSStreamTemp) | −13 | gw_tau, lat_temp_adj, maxiter_sntemp, melt_temp, seg_cum_area, seg_elev, seg_humidity, seg_lat, seg_width, ss_tau, stream_tave_init, width_alpha, width_m |
| — minus solar geometry inputs | −13 | alte, altw, azrh, vce, vcw, vdemn, vdemx, vdwmn, vdwmx, vhe, vhw, voe, vow |
| **DRB "core" parameters** | **112** | Non-stream-temp, non-solar-geometry |
| hydro-param parameter_metadata.yml | 95 | Required: 13, Optional: 82 |
| + Forcing (prcp, tmax, tmin, swrad, potet) | +5 | Separate output files |
| **hydro-param total** | **100** | |

### In DRB reference but NOT in our spec

#### Solar geometry inputs (13) — see [#156](https://github.com/rmcd-mscb/hydro-param/issues/156)

The reference uses raw geometry arrays; we pre-compute the soltab tables via Swift (1976).
Needs reconciliation with pywatershed v2.0 `PRMSSolarGeometry` process class expectations.

| Parameter | Description |
|-----------|-------------|
| `alte` | Altitude of east horizon (degrees) |
| `altw` | Altitude of west horizon (degrees) |
| `azrh` | Azimuth of slope (degrees) |
| `vce` / `vcw` | View factor, clear east/west |
| `vdemn` / `vdemx` | View factor, diffuse east min/max |
| `vdwmn` / `vdwmx` | View factor, diffuse west min/max |
| `vhe` / `vhw` | View factor, horizon east/west |
| `voe` / `vow` | View factor, overhead east/west |

#### Topology / identification (4) — see [#157](https://github.com/rmcd-mscb/hydro-param/issues/157)

| Parameter | Description | Category |
|-----------|-------------|----------|
| `hru_lon` | HRU centroid longitude | DF — centroid from fabric |
| `nhm_id` | NHM HRU identifier | DT — from fabric attribute |
| `nhm_seg` | NHM segment identifier | DT — from fabric attribute |
| `hru_segment_nhm` | NHM segment ID per HRU | DT — from fabric attribute |

#### Segment spatial params (3) — see [#158](https://github.com/rmcd-mscb/hydro-param/issues/158)

| Parameter | Description | Category |
|-----------|-------------|----------|
| `seg_cum_area` | Cumulative drainage area per segment | DS — from NHDPlus |
| `seg_elev` | Segment mean elevation | DS — from DEM/NHDPlus |
| `seg_lat` | Segment centroid latitude | DF — from fabric |

#### Other missing params (4)

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `albedo` | Initial snow surface albedo | IC — add as default 0.8 |
| `epan_coef` | Pan evaporation coefficient | DEF — for epan PET method only |
| `melt_temp` | Temperature threshold for melt | DEF — physical constant (32°F) |
| `stream_tave_init` | Initial stream temperature | IC — needed with PRMSStreamTemp |

#### Stream temperature process — PRMSStreamTemp (low priority)

Only needed when running the stream temperature module:
`gw_tau`, `lat_temp_adj`, `maxiter_sntemp`, `seg_humidity`, `seg_width`, `ss_tau`,
`width_alpha`, `width_m`

### In our spec but NOT in DRB reference

| Parameter | Notes |
|-----------|-------|
| `soltab_potsw` | We pre-compute; reference uses solar geometry inputs instead |
| `soltab_horad_potsw` | Same — pre-computed vs geometry-based |
| `soltab_sunhrs` | Same |
| `doy` | Structural array we add; reference uses it as a dimension |
| `hru_in_to_cf` | Our convenience conversion constant |

---

## Priority Placeholder Parameters (Quick Reference)

These 13 **PH** parameters have the most significant model impact and are tracked in open issues:

| Parameter | Current Value | Correct Derivation | Issue |
|-----------|--------------|-------------------|-------|
| `snarea_curve` | all 1.0 (wrong) | MODIS SCA + SNODAS SWE curves | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147), [#155](https://github.com/rmcd-mscb/hydro-param/issues/155) |
| `soil_rechr_max_frac` | 0.4 (fixed) | aws0_50 / aws0_100 ratio | [#151](https://github.com/rmcd-mscb/hydro-param/issues/151) |
| `rad_trncf` | 0.5 (uniform) | 1 − covden_win | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `sat_threshold` | 999.0 (placeholder) | (porosity − field_cap) × root depth | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `radadj_intcp` | 1.0 (uniform) | Regression vs potential srad | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `radadj_slope` | 0.0 (uniform) | Regression vs potential srad | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `tmax_index` | 50°F (uniform) | 90th-pct monthly tmax from normals | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `melt_force` | 140 (uniform DOY) | Last spring frost date from climate | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `melt_look` | 90 (uniform DOY) | Spring climate normals | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `snowinfil_max` | 2.0 in/day | soil_type class lookup | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `tstorm_mo` | 0 (uniform) | NOAA thunderstorm climatology | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `dprst_frac_open` | 1.0 (uniform) | NHDPlus ftype classification | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
| `dprst_seep_rate_*` | 0.02 (uniform) | Ksat from gNATSGO | [#147](https://github.com/rmcd-mscb/hydro-param/issues/147) |
