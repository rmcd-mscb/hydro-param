#!/usr/bin/env python
"""Download NHDPlusV2 data for the Delaware River Basin (MVP test domain).

Downloads catchment polygons and flowlines from the NHDPlus V2 database
via USGS web services using pynhd. Saves to GeoPackage files for offline
use as the project's test fabric.

Usage:
    pixi run -e download download-delaware
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAGE_ID = "01463500"  # Delaware River at Trenton, NJ
OUTPUT_DIR = Path("data/delaware")
BASIN_FILE = OUTPUT_DIR / "basin_boundary.gpkg"
CATCHMENTS_FILE = OUTPUT_DIR / "catchments.gpkg"
FLOWLINES_FILE = OUTPUT_DIR / "flowlines.gpkg"
MAX_DISTANCE_KM = 9999  # Max NLDI navigation distance


def get_basin_boundary() -> gpd.GeoDataFrame:
    """Retrieve the watershed boundary for the outlet gage."""
    from pynhd import NLDI

    nldi = NLDI()
    logger.info("Fetching basin boundary for USGS-%s ...", GAGE_ID)
    basin = nldi.get_basins(GAGE_ID)
    if basin.empty:
        logger.error(
            "No basin boundary returned for USGS-%s. Check gage ID and network connectivity.",
            GAGE_ID,
        )
        sys.exit(1)
    area_km2 = basin.to_crs("EPSG:5070").geometry.area.sum() / 1e6
    logger.info("Basin boundary: %.0f km2, CRS=%s", area_km2, basin.crs)
    return basin


def get_upstream_comids() -> list[int]:
    """Navigate upstream to get all tributary COMIDs."""
    from pynhd import NLDI

    nldi = NLDI()
    logger.info(
        "Navigating upstream tributaries from USGS-%s (distance=%d km) ...",
        GAGE_ID,
        MAX_DISTANCE_KM,
    )
    flw = nldi.navigate_byid(
        fsource="nwissite",
        fid=f"USGS-{GAGE_ID}",
        navigation="upstreamTributaries",
        source="flowlines",
        distance=MAX_DISTANCE_KM,
    )
    if len(flw) == 0:
        logger.error("No flowlines returned. Check gage ID and network connectivity.")
        sys.exit(1)

    comids = sorted({int(c) for c in flw.nhdplus_comid.to_list()})
    logger.info("Found %d unique COMIDs via upstream navigation.", len(comids))
    return comids


def get_catchments(comids: list[int]) -> gpd.GeoDataFrame:
    """Fetch catchment polygons for the given COMIDs."""
    from pynhd import WaterData

    logger.info("Fetching %d catchments from WaterData ...", len(comids))
    wd = WaterData("catchmentsp")
    catchments = wd.byid("featureid", comids)

    missing = set(comids) - set(catchments["featureid"].astype(int))
    if missing:
        logger.warning(
            "%d COMIDs have no catchment polygon (likely coastal/divergence): %s...",
            len(missing),
            sorted(missing)[:5],
        )

    logger.info("Retrieved %d catchment polygons.", len(catchments))
    return catchments


def get_flowlines(comids: list[int]) -> gpd.GeoDataFrame:
    """Fetch full NHDPlus flowline records with all attributes."""
    from pynhd import WaterData

    logger.info("Fetching %d detailed flowlines from WaterData ...", len(comids))
    wd = WaterData("nhdflowline_network")
    flowlines = wd.byid("comid", comids)
    logger.info("Retrieved %d flowline records.", len(flowlines))
    return flowlines


def print_summary(
    basin: gpd.GeoDataFrame,
    catchments: gpd.GeoDataFrame,
    flowlines: gpd.GeoDataFrame,
) -> None:
    """Print summary statistics."""
    catch_proj = catchments.to_crs("EPSG:5070")
    area_km2 = catch_proj.geometry.area.sum() / 1e6

    logger.info("=" * 60)
    logger.info("Delaware River Basin NHDPlus Data Summary")
    logger.info("=" * 60)
    logger.info("Outlet gage:        USGS-%s (Delaware at Trenton)", GAGE_ID)
    logger.info("Catchments:         %d", len(catchments))
    logger.info("Flowlines:          %d", len(flowlines))
    logger.info("Total area:         %.0f km2", area_km2)
    logger.info("CRS:                %s", catchments.crs)
    if "streamorde" in flowlines.columns:
        logger.info(
            "Stream order range: %d - %d",
            flowlines["streamorde"].min(),
            flowlines["streamorde"].max(),
        )
    if "areasqkm" in catchments.columns:
        logger.info(
            "Catchment area:     %.2f - %.2f km2",
            catchments["areasqkm"].min(),
            catchments["areasqkm"].max(),
        )
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info("  Basin:      %s", BASIN_FILE)
    logger.info("  Catchments: %s", CATCHMENTS_FILE)
    logger.info("  Flowlines:  %s", FLOWLINES_FILE)


def main() -> int:
    """Download NHDPlusV2 data for the Delaware River Basin."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    try:
        # Step 1: Basin boundary
        basin = get_basin_boundary()
        basin.to_file(BASIN_FILE, driver="GPKG")
        logger.info("Saved basin boundary → %s", BASIN_FILE)

        # Step 2: Upstream COMIDs via NLDI navigation
        comids = get_upstream_comids()

        # Step 3: Catchment polygons
        catchments = get_catchments(comids)
        catchments.to_file(CATCHMENTS_FILE, driver="GPKG")
        logger.info("Saved catchments → %s", CATCHMENTS_FILE)

        # Step 4: Detailed flowlines
        flowlines = get_flowlines(comids)
        flowlines.to_file(FLOWLINES_FILE, driver="GPKG")
        logger.info("Saved flowlines → %s", FLOWLINES_FILE)

        # Summary
        print_summary(basin, catchments, flowlines)

    except Exception:
        logger.exception("Download failed.")
        return 1

    elapsed = time.perf_counter() - t0
    logger.info("Total elapsed time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
