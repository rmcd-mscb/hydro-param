# Development Roadmap

This page summarizes the design decisions that shaped hydro-param during its initial development sprint (February--March 2026). Each theme groups related design documents by area of concern. The full design documents are in `docs/plans/`, and each entry in the tables below links to the corresponding document.

---

## Core Pipeline

The core pipeline is the model-agnostic engine at the center of hydro-param. It resolves target fabrics, fetches source datasets, computes intersection weights, runs zonal statistics, and writes a Standardized Internal Representation (SIR) to disk. Work in this theme established the SIR contract---normalized variable names, units, and file naming conventions---so that downstream model plugins can consume pipeline output without knowledge of how it was produced. Later additions introduced memory optimization, manifest-based resume, and specialized processing pathways for derived categorical variables that require pixel-level classification before zonal aggregation.

| Date | Design | Summary |
|------|--------|---------|
| 2026-02-23 | [SIR Normalization](2026-02-23-sir-normalization-design.md) | Standardized variable names and units at pipeline output boundary |
| 2026-02-23 | [Pipeline Memory Optimization](2026-02-23-pipeline-memory-optimization-design.md) | STAC query reuse and memory-efficient batch processing |
| 2026-02-24 | [SIR Temporal Normalization](2026-02-24-sir-temporal-normalization-design.md) | Extended SIR normalization to temporal (multi-year) datasets |
| 2026-02-24 | [Pipeline Resilience](2026-02-24-pipeline-resilience-optimization-plan.md) | Manifest-based resume, pre-fetch, network timeout handling |
| 2026-02-28 | [SIR Variable Naming Fix](2026-02-28-sir-variable-naming-fix.md) | Year-suffixed SIR variable name resolution |
| 2026-02-28 | [SIR Dataset Prefix](2026-02-28-sir-dataset-prefix-design.md) | Dataset name prefix in SIR filenames for disambiguation |
| 2026-03-02 | [Shared Classification Module](2026-03-02-shared-classification-design.md) | USDA texture triangle as shared classification module |
| 2026-03-02 | [Derived Categorical Pipeline](2026-03-02-derived-categorical-design.md) | Pixel-level multi-source classification before zonal stats |

## pywatershed Plugin

The pywatershed plugin is hydro-param's primary consumer of pipeline output. It implements all 14 derivation steps required to produce the ~100 static parameters and 3 forcing time series that pywatershed (USGS NHM-PRMS in Python) needs to run. This theme covers the plugin architecture itself---the `DerivationContext` protocol, formatter separation, and standalone Phase 2 execution---as well as individual derivation step designs for soils, soltab, routing, waterbody overlay, PET, transpiration, forcing generation, and soil texture classification. The config schema evolved through four major versions to reach the current consumer-oriented layout with explicit `static_datasets`, `forcing`, and `climate_normals` sections.

| Date | Design | Summary |
|------|--------|---------|
| 2026-02-25 | [Plugin Architecture](2026-02-25-pywatershed-plugin-architecture-design.md) | Plugin protocol, DerivationContext, formatter separation |
| 2026-02-25 | [Steps 5, 9, 14](2026-02-25-derivation-steps-5-9-14-design.md) | Soils (step 5), soltab (step 9), calibration seeds (step 14) |
| 2026-02-25 | [Forcing Generation](2026-02-25-forcing-generation-design.md) | Step 7 temporal forcing: per-variable SIR, unit conversion, CBH format |
| 2026-02-25 | [PET and Transpiration](2026-02-25-pet-transpiration-design.md) | Steps 10--11: Jensen-Haise PET and transpiration timing from climate normals |
| 2026-02-26 | [Waterbody Overlay](2026-02-26-waterbody-overlay-design.md) | Step 6: NHDPlus waterbody spatial overlay for hru_type and dprst_frac |
| 2026-02-26 | [Routing Parameters](2026-02-26-step12-routing-design.md) | Step 12: Muskingum routing coefficients from segment geometry |
| 2026-02-28 | [Decouple pywatershed Run](2026-02-28-decouple-pywatershed-design.md) | Decouple pywatershed run from Phase 1 pipeline (Phase 2 standalone) |
| 2026-02-28 | [Temporal DerivationContext](2026-02-28-temporal-derivation-context-design.md) | Wire temporal SIR data into DerivationContext |
| 2026-02-28 | [Forcing Regrouping](2026-02-28-temporal-forcing-regrouping-design.md) | Per-variable forcing detection with reverse SIR lookup |
| 2026-03-01 | [Config Redesign v4.0](2026-03-01-pywatershed-config-redesign-design.md) | Consumer-oriented config with static_datasets, forcing, climate_normals |
| 2026-03-02 | [Soil Texture Triangle](2026-03-02-soil-texture-triangle-design.md) | USDA soil texture triangle classifier for soil_type derivation |
| 2026-03-02 | [pywatershed Compatibility](2026-03-02-pywatershed-compat-design.md) | pywatershed v2.0 runtime compatibility layer |
| 2026-03-04 | [soil_rechr_max_frac](2026-03-04-soil-rechr-max-frac-design.md) | soil_rechr_max_frac from gNATSGO AWC ratio (aws0_30 / aws0_100) |

## Data Access

hydro-param supports five data access strategies spanning STAC catalogs, local GeoTIFFs, and OPeNDAP endpoints. This theme covers the integration of the Geospatial Fabric v1.1 (GFv1.1) dataset, which required a dedicated download CLI for its ~15 GB of ScienceBase-hosted rasters, a local_tiff processing pathway, and a user-local dataset registry overlay so that site-specific data paths do not pollute the bundled registry.

| Date | Design | Summary |
|------|--------|---------|
| 2026-03-06 | [GFv1.1 Download CLI](2026-03-06-gfv11-download-design.md) | GFv1.1 ScienceBase download CLI (~15 GB, fault-tolerant) |
| 2026-03-06 | [GFv1.1 Raster Integration](2026-03-06-gfv11-raster-integration-design.md) | GFv1.1 raster integration via local_tiff strategy |
| 2026-03-09 | [GFv1.1 Registry Overlay](2026-03-09-gfv11-registry-design.md) | User-local dataset registry overlay for GFv1.1 |

## Validation and QA

Validation work ensures that hydro-param's derived parameters match authoritative reference values. The soltab valid-range fix corrected a bounds error that clipped solar radiation tables. The parameter audit cross-referenced all ~100 PRMS parameters against source code to verify derivation categories. The NHM cross-check compared Delaware River Basin output against the National Hydrologic Model reference parameterization, leading to fixes for elevation statistics, centroid computation, canopy density units, and snow depletion curve assignment.

| Date | Design | Summary |
|------|--------|---------|
| 2026-03-02 | [Soltab Valid Range](2026-03-02-soltab-valid-range-design.md) | Fix soltab valid_range from [0, 1000] to [0, 2000] Langleys |
| 2026-03-05 | [Parameter Audit Design](2026-03-05-parameter-audit-design.md) | Source-code cross-reference of all ~100 parameters |
| 2026-03-05 | [Parameter Audit Findings](parameter_audit_2026-03-05.md) | Audit findings: parameter inventory with derivation categories |
| 2026-03-10 | [GFv1.1 Validation Plan](2026-03-10-gfv11-validation-plan.md) | GFv1.1 static parameter validation against NHM reference |
| 2026-03-10 | [NHM Cross-Check](2026-03-10-nhm-crosscheck-design.md) | NHM reference cross-check: elevation median, representative_point, CV_INT fix |

## Infrastructure

Infrastructure work improved the developer and user experience without changing parameterization logic. The UX audit addressed 15 gaps in CLI messages, error handling, and input validation. Registry YAMLs were bundled into the Python package so that installations are self-contained. A config schema audit relocated lookup tables and added waterbody_path support. Stale backward-compatibility code was removed, saving 167 lines. The themed datasets design introduced category-grouped pipeline configs for clearer organization.

| Date | Design | Summary |
|------|--------|---------|
| 2026-02-27 | [Pre-Release UX Audit](2026-02-27-pre-release-ux-audit-design.md) | 15-gap UX audit: CLI messages, error handling, validation |
| 2026-02-27 | [Pipeline Template](2026-02-27-pipeline-template-comprehensive-design.md) | Comprehensive pipeline template with all dataset categories |
| 2026-02-27 | [Bundle Registry in Package](2026-02-27-bundle-registry-in-package-design.md) | Bundle dataset registry YAMLs in package via importlib.resources |
| 2026-02-28 | [Config Schema Audit](2026-02-28-config-schema-audit-design.md) | Config schema audit: waterbody_path, lookup table relocation |
| 2026-03-01 | [Stale Code Cleanup](2026-03-01-stale-code-cleanup-design.md) | Remove dead backward compatibility code (-167 lines) |
| 2026-03-10 | [Themed Datasets Config](2026-03-10-themed-datasets-design.md) | Themed pipeline config: datasets grouped by category dict |

## Open and Planned Work

The following items are designed but not yet implemented, or are tracked as open issues:

- **Grid processing pathway** --- polygon targets use gdptools; grid targets will use xesmf/rioxarray for raster-on-raster operations
- **Transparent data caching** --- pooch-style, library-managed transparent cache to replace manual download management
- **Derived-raster pathway** ([#200](https://github.com/rmcd-mscb/hydro-param/issues/200)) --- pixel-level raster math before zonal stats (DerivedContinuousSpec)
- **PRMS legacy formatter** ([#92](https://github.com/rmcd-mscb/hydro-param/issues/92)) --- pyPRMS-based text output format for classic PRMS input files
- **NextGen hydrofabric slopes** ([#100](https://github.com/rmcd-mscb/hydro-param/issues/100)) --- flowpath slopes from NextGen fabric for routing parameters
- **Subsurface flux rescaling** ([#154](https://github.com/rmcd-mscb/hydro-param/issues/154)) --- needs GLHYMPS data source for bedrock permeability
- **Nearest-neighbor gap-fill** ([#73](https://github.com/rmcd-mscb/hydro-param/issues/73)) --- temporal SIR features missing grid coverage

---

*Last updated: 2026-03-11*
