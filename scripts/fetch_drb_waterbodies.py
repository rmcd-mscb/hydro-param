"""Fetch NHDPlus waterbody polygons for the DRB test fabric.

Uses pynhd to query the NHDPlus WaterData service for waterbodies
within the bounding box of the DRB nhru fabric, then reprojects
to match the fabric CRS and saves as a GeoPackage.

Usage:
    pixi run -e dev python scripts/fetch_drb_waterbodies.py
"""

import logging
from pathlib import Path

import geopandas as gpd
from pynhd import WaterData

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    data_dir = Path("data/pywatershed_gis/drb_2yr")
    fabric_path = data_dir / "nhru.gpkg"
    output_path = data_dir / "waterbodies.gpkg"

    # Load fabric and get bounding box in WGS84
    fabric = gpd.read_file(fabric_path)
    fabric_4326 = fabric.to_crs("EPSG:4326")
    bounds = tuple(fabric_4326.total_bounds)
    logger.info("Fabric: %d HRUs, CRS=%s", len(fabric), fabric.crs)
    logger.info("Query bbox (WGS84): %s", bounds)

    # Fetch NHDPlus waterbodies
    wd = WaterData("nhdwaterbody")
    wb = wd.bybox(bounds)
    logger.info("Fetched %d waterbodies", len(wb))

    # Keep useful columns for step 6 derivation
    keep_cols = [
        "geometry",
        "comid",
        "gnis_name",
        "areasqkm",
        "ftype",
        "fcode",
        "meandepth",
        "lakevolume",
        "maxdepth",
        "lakearea",
    ]
    wb = wb[[c for c in keep_cols if c in wb.columns]]

    # Reproject to match fabric CRS
    wb = wb.to_crs(fabric.crs)
    logger.info("Reprojected to %s", wb.crs)

    # Summary stats
    logger.info("Waterbody types (ftype):\n%s", wb["ftype"].value_counts().to_string())
    logger.info("Area stats (km²):\n%s", wb["areasqkm"].describe().to_string())

    # Save
    wb.to_file(output_path, driver="GPKG")
    logger.info("Saved %d waterbodies to %s", len(wb), output_path)


if __name__ == "__main__":
    main()
