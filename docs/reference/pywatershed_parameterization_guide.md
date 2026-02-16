# hydro-param: pywatershed Parameterization Guide

## Context

This document provides comprehensive guidance for developing hydro-param's
pywatershed/NHM-PRMS support. It was derived from extensive research of:

- pywatershed v2.0 source code (EC-USGS/pywatershed)
- Regan et al. 2018 (TM6-B9) — NHM parameter derivation methods
- Hay et al. 2023 (TM6-B10) — Continental-scale calibration
- Markstrom et al. 2015 (TM6-B7) — PRMS-IV documentation
- pywatershed API docs (pywatershed.readthedocs.io)

The companion file `pywatershed_dataset_param_map.yml` contains the complete
dataset-to-parameter mapping with derivation methods, lookup tables, defaults,
calibration seeds, and a 15-step derivation pipeline DAG.

## How to Use This Guide

Reference the YAML mapping file as the authoritative source for:
1. Which datasets to include in data.yml catalog files
2. How each parameter is derived (method, dependencies, type)
3. The processing order (derivation_pipeline section)
4. Default values and calibration initial seeds

---

## Task 1: Fill Out Curated Dataset Catalog (data.yml files)

### Goal
Create data.yml files organized by category that define the source datasets
hydro-param needs to generate pywatershed parameters.

### Dataset Categories to Implement

The YAML mapping file defines 9 categories under `curated_datasets`. Each
data.yml entry should include:

```yaml
dataset_name:
  description: "Human-readable description"
  source_org: "Provider organization"
  resolution: "Spatial and temporal resolution"
  coverage: "Geographic extent"
  accessor: "Python function/package to retrieve the data"
  format: "File format (COG, NetCDF, Shapefile, etc.)"
  variables:
    - variable_name_1
    - variable_name_2
  derives_parameters:
    - param_name_1  # which PRMS params this dataset feeds
    - param_name_2
```

### Priority Order for Implementation

1. **geospatial_fabric** — Must come first; defines the HRU/segment geometry
   that everything else operates on. Accessor via ScienceBase or onhm-fetcher.

2. **ned_3dep** — Topography is needed early because soltab, jh_coef, and many
   other derived params depend on elevation, slope, and aspect.
   Accessor: `pygeohydro.get_3dep_dem`

3. **nlcd** — Land cover drives cov_type which cascades into interception,
   canopy density, imperviousness, and hru_type.
   Accessor: `pygeohydro.nlcd_bygeom` or `pygeohydro.get_nlcd`

4. **statsgo2 / ssurgo** — Soil properties are needed for soil_moist_max and
   the soil zone parameters.
   Accessor: `pygeohydro.ssurgo_bygeom`

5. **daymet_v4** (or gridmet) — Climate forcing time series.
   This is the most compute-intensive data retrieval.
   Accessor: `pydaymet.get_bygeom` or `gdptools`

6. **nhdplus** — Waterbodies for depression storage, velocity/slope for
   routing coefficients.
   Accessor: `pynhd`

### Notes on Accessors

Most of these are already accessible through the HyRiver suite that you're
familiar with (pynhd, pygeohydro, pydaymet, pygridmet). For large-domain
processing, gdptools is the better choice for area-weighted climate averaging.
Consider whether hydro-param should use these directly or provide an
abstraction layer.

---

## Task 2: pywatershed Output Plugin and Config File

### 2A: pywatershed Run Configuration

A pywatershed model run needs these components assembled:

```python
import pywatershed as pws

# 1. Control — simulation settings
control = pws.Control.load_prms(control_file)
# Or build programmatically:
control = pws.Control(
    start_time=np.datetime64("1980-10-01"),
    end_time=np.datetime64("2020-09-30"),
    time_step=np.timedelta64(1, "D"),
)

# 2. Discretization — from parameter file
discretization = pws.Parameters.from_netcdf(param_file)

# 3. Parameters — the bulk of what hydro-param produces
parameters = pws.Parameters.from_netcdf(param_file)

# 4. Model — assembled from process classes
model = pws.Model(
    [
        pws.PRMSAtmosphere,
        pws.PRMSCanopy,
        pws.PRMSSnow,
        pws.PRMSRunoff,
        pws.PRMSSoilzone,
        pws.PRMSGroundwater,
        pws.PRMSChannel,
    ],
    control=control,
    discretization=discretization,
    parameters=parameters,
)

# 5. Run
model.run(finalize=True)
```

### 2B: Config File Structure for hydro-param

The hydro-param config for a pywatershed run should specify:

```yaml
# hydro-param run configuration
target_model: pywatershed
version: "2.0"

# Spatial domain
domain:
  source: geospatial_fabric  # or custom
  # For GF extraction:
  gf_version: "1.1"
  extraction_method: bandit  # or bbox, huc, pour_point
  extraction_params:
    huc_id: "01013500"  # example: upstream of a gage
    # OR:
    # bbox: [-74.5, 42.0, -73.5, 43.0]
    # pour_point: [-73.95, 42.45]

# Simulation period
time:
  start: "1980-10-01"
  end: "2020-09-30"
  timestep: daily

# Climate forcing source
# pywatershed accepts one-variable-per-NetCDF-file for forcing inputs
# (prcp.nc, tmax.nc, tmin.nc).  PRMSAtmosphere accepts file paths,
# numpy arrays, or Adapter objects for prcp, tmax, tmin, soltab_potsw,
# and soltab_horad_potsw.
climate:
  source: daymet_v4   # or gridmet, conus404_ba
  method: area_weighted_mean  # via gdptools or exactextract
  variables: [prcp, tmax, tmin]
  unit_conversion:
    prcp: "mm_to_inches"
    tmax: "C_to_F"
    tmin: "C_to_F"

# Dataset sources (references data.yml catalog)
datasets:
  topography: ned_3dep
  landcover: nlcd
  soils: statsgo2     # or ssurgo
  hydrography: nhdplus
  # Optional overrides for any dataset

# Processing options
processing:
  zonal_method: exactextract  # or rasterstats
  batch_strategy: hilbert     # spatial batching via Hilbert curve sorting
  n_workers: 4
  chunk_size: 1000            # HRUs per batch

# Parameter overrides (optional — override any derived value)
parameter_overrides:
  tmax_allsnow: 32.0
  den_max: 0.55
  # Can also point to a file:
  # from_file: "my_custom_params.nc"

# Calibration settings
calibration:
  generate_seeds: true        # produce initial calibration param values
  seed_method: physically_based  # or all_defaults
  # Parameters that should NOT be overwritten by hydro-param:
  preserve_from_existing: []

# Output
output:
  format: netcdf              # or prms_text
  parameter_file: "params.nc"
  forcing_dir: "./forcing/"   # one-variable-per-file NetCDF (prcp.nc, tmax.nc, tmin.nc)
  control_file: "control.yml"
  soltab_file: "soltab.nc"
```

### 2C: pywatershed Output Plugin Design

The output plugin converts hydro-param's internal parameter representation
into pywatershed-compatible files. Here's the proposed interface:

```python
class PywatershedOutputPlugin:
    """
    Formats hydro-param results for pywatershed consumption.

    Produces:
    1. Parameter file (NetCDF) — all static/slow-varying parameters
    2. Forcing NetCDF files — one variable per file (prcp.nc, tmax.nc, tmin.nc)
    3. Soltab arrays — potential solar radiation lookup tables
    4. Control file — simulation configuration

    pywatershed's PRMSAtmosphere constructor accepts Union[str, Path,
    ndarray, Adapter] for prcp, tmax, tmin, soltab_potsw, and
    soltab_horad_potsw — so forcing inputs can be NetCDF file paths,
    numpy arrays, or Adapter objects (not limited to legacy CBH text).
    """

    name = "pywatershed"
    target_versions = ["1.0", "2.0"]

    def __init__(self, config: dict):
        self.config = config
        self.unit_system = "prms"  # PRMS uses feet, inches, °F internally

    def write_parameters(self, params: dict, discretization: dict,
                         output_path: Path):
        """
        Write parameter NetCDF file compatible with pywatershed.

        The file must contain:
        - Dimensions: nhru, nsegment, nmonths(12), ndeplval, one
        - Coordinates: nhm_id, nhm_seg
        - All parameters from the parameter_registry

        pywatershed loads this via:
            pws.Parameters.from_netcdf(output_path)
        """
        pass

    def write_forcing_netcdf(self, climate_data: dict, output_dir: Path):
        """
        Write forcing NetCDF files (one variable per file).

        Produces separate files that pywatershed can load directly:
        - prcp.nc (nhru × ntime, units: inches/day)
        - tmax.nc (nhru × ntime, units: °F)
        - tmin.nc (nhru × ntime, units: °F)

        pywatershed's PRMSAtmosphere accepts file paths for these
        inputs via ``pws.Parameters.from_netcdf()`` or Adapter objects.

        For legacy PRMS text CBH format, use write_cbh_text() instead.
        """
        pass

    def write_soltab(self, soltab_potsw, soltab_horad_potsw, output_path):
        """Write potential solar radiation tables."""
        pass

    def write_control(self, config: dict, output_path: Path):
        """
        Write pywatershed control configuration.

        Can be YAML (for pywatershed Python API) or PRMS-format text.
        """
        pass

    def validate(self, output_dir: Path) -> list[str]:
        """
        Validate that all required parameters are present and within
        valid ranges. Returns list of warnings/errors.

        Uses parameter_registry ranges and constraints.
        """
        pass

    # --- Unit conversion helpers ---
    @staticmethod
    def mm_to_inches(values):
        return values / 25.4

    @staticmethod
    def celsius_to_fahrenheit(values):
        return values * 9.0 / 5.0 + 32.0

    @staticmethod
    def meters_to_feet(values):
        return values * 3.28084

    @staticmethod
    def sq_meters_to_acres(values):
        return values * 0.000247105
```

### 2D: Key Implementation Considerations

**Forcing Input Format**: pywatershed's ``PRMSAtmosphere`` constructor accepts
``Union[str, Path, ndarray, Adapter]`` for ``prcp``, ``tmax``, ``tmin``,
``soltab_potsw``, and ``soltab_horad_potsw``. The preferred modern approach is
one-variable-per-NetCDF-file (e.g. ``prcp.nc``, ``tmax.nc``, ``tmin.nc``).
Parameters are loaded via ``pws.Parameters.from_netcdf()``. Legacy PRMS CBH
text format is supported only as an optional secondary output.

**Unit Conversions**: PRMS internally uses feet, inches, Fahrenheit, and acres.
Most source data comes in metric (meters, mm, °C). The output plugin must
handle all conversions. This is a common source of bugs — keep a conversion
registry.

**Parameter Dimensions**: Parameters have different dimensions:
- `nhru` — one value per HRU (most parameters)
- `nsegment` — one value per stream segment (routing params)
- `[nhru, nmonths]` — monthly values per HRU (adjustment factors)
- `[nhru, 366]` — daily-of-year per HRU (soltab)
- `[ndeplval, 11]` — snow depletion curves
- `one` — scalar (basin-wide values)

**Parameter Interdependencies**: The derivation pipeline has a clear DAG.
For example:
- `covden_win` depends on `covden_sum` and `cov_type`
- `soltab_potsw` depends on `hru_lat`, `hru_slope`, `hru_aspect`
- `jh_coef_hru` depends on `hru_elev`
- `soil_rechr_max_frac` depends on soil layer data
- `K_coef` depends on `seg_length` and velocity estimates

**Validation**: Key constraints to check:
- All fractions in [0, 1]
- Areas > 0
- Temperatures in valid range for units
- Routing topology is a valid DAG (no cycles)
- Sum of hru_percent_imperv + pervious ≤ 1.0
- soil_rechr_max_frac × soil_moist_max ≤ soil_moist_max
- K_coef appropriate for daily timestep

---

## Process Class Input/Output Summary

For quick reference, here are the 8 pywatershed process classes and their
key I/O signatures:

### PRMSAtmosphere(control, discretization, parameters, prcp, tmax, tmin, soltab_potsw, soltab_horad_potsw)

- **External inputs**: prcp, tmax, tmin (forcing NetCDF or Adapter), soltab tables
  - Each forcing input accepts ``Union[str, Path, ndarray, Adapter]``
  - Preferred modern format: one-variable-per-NetCDF-file (prcp.nc, tmax.nc, tmin.nc)
- **Key parameters**: *_cbh_adj, tmax_allsnow, tmax_allrain_offset, jh_coef, jh_coef_hru, dday_slope, dday_intcp, transp_beg, transp_end
- **Outputs → downstream**: tmaxf, tminf, prmx, hru_ppt, hru_rain, hru_snow, swrad, potet, transp_on

### PRMSCanopy
- **From Atmosphere**: hru_ppt, hru_rain, hru_snow, potet, transp_on
- **Key parameters**: cov_type, covden_sum, covden_win, srain_intcp, wrain_intcp, snow_intcp
- **Outputs → downstream**: net_ppt, net_rain, net_snow, intcp_evap

### PRMSSnow
- **From Atmosphere**: swrad, tmaxf, tminf, prmx, potet, transp_on, hru_ppt
- **From Canopy**: net_ppt, net_rain, net_snow
- **Key parameters**: snarea_curve, snarea_thresh, den_init, den_max, cecn_coef, emis_noppt, freeh2o_cap, potet_sublim, albset_*
- **Outputs → downstream**: snowmelt, snow_evap, snowcov_area, pkwater_equiv, pptmix_nopack

### PRMSRunoff
- **From Canopy**: net_ppt, net_rain, net_snow
- **From Snow**: snowmelt, snowcov_area, pkwater_equiv, pptmix_nopack
- **From Atmosphere**: potet, hru_ppt, transp_on
- **Key parameters**: carea_max, smidx_coef, smidx_exp, hru_percent_imperv, imperv_stor_max, soil_moist_max, soil_rechr_max, dprst_*
- **Outputs → downstream**: sroff, infil, hru_impervevap, dprst_seep_hru

### PRMSSoilzone
- **From Runoff**: infil, sroff, hru_impervevap
- **From Snow**: snow_evap, snowcov_area
- **From Canopy**: intcp_evap
- **From Atmosphere**: potet, hru_ppt, transp_on
- **Key parameters**: soil_moist_max, soil_rechr_max_frac, soil_type, soil2gw_max, ssr2gw_rate, ssr2gw_exp, slowcoef_lin, slowcoef_sq, fastcoef_lin, fastcoef_sq, pref_flow_den, sat_threshold
- **Outputs → downstream**: ssr_to_gw, soil_to_gw, ssres_flow

### PRMSGroundwater
- **From Soilzone**: ssr_to_gw, soil_to_gw
- **From Runoff**: dprst_seep_hru
- **Key parameters**: gwflow_coef, gwsink_coef, gwstor_init, gwstor_min
- **Outputs → downstream**: gwres_flow

### PRMSChannel
- **From Runoff**: sroff
- **From Soilzone**: ssres_flow
- **From Groundwater**: gwres_flow
- **Key parameters**: tosegment, hru_segment, K_coef, x_coef, seg_length
- **Outputs**: seg_outflow (streamflow!)

---

## References

- Markstrom et al. 2015: PRMS-IV documentation (TM6-B7)
- Regan et al. 2018: NHM description (TM6-B9), Appendix 1 = parameter derivation
- Hay et al. 2023: Continental calibration + muskingum_mann (TM6-B10)
- Viger & Bock 2014: Geospatial Fabric
- pywatershed docs: https://pywatershed.readthedocs.io
- pywatershed repo: https://github.com/EC-USGS/pywatershed
