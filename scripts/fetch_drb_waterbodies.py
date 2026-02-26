"""Fetch NHDPlus waterbody polygons for the DRB test fabric.

Uses pynhd to query the NHDPlus WaterData service for waterbodies
within the bounding box of the DRB nhru fabric, then reprojects
to match the fabric CRS and saves as a GeoPackage.

Usage:
    pixi run -e dev python scripts/fetch_drb_waterbodies.py
"""

from pathlib import Path

import geopandas as gpd
from pynhd import WaterData


def main() -> None:
    data_dir = Path("data/pywatershed_gis/drb_2yr")
    fabric_path = data_dir / "nhru.gpkg"
    output_path = data_dir / "waterbodies.gpkg"

    # Load fabric and get bounding box in WGS84
    fabric = gpd.read_file(fabric_path)
    fabric_4326 = fabric.to_crs("EPSG:4326")
    bounds = tuple(fabric_4326.total_bounds)
    print(f"Fabric: {len(fabric)} HRUs, CRS={fabric.crs}")
    print(f"Query bbox (WGS84): {bounds}")

    # Fetch NHDPlus waterbodies
    wd = WaterData("nhdwaterbody")
    wb = wd.bybox(bounds)
    print(f"Fetched {len(wb)} waterbodies")

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
    print(f"Reprojected to {wb.crs}")

    # Summary stats
    print(f"\nWaterbody types (ftype):\n{wb['ftype'].value_counts().to_string()}")
    print(f"\nArea stats (km²):\n{wb['areasqkm'].describe().to_string()}")

    # Save
    wb.to_file(output_path, driver="GPKG")
    print(f"\nSaved {len(wb)} waterbodies to {output_path}")


if __name__ == "__main__":
    main()
