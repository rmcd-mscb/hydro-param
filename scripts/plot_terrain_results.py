"""Validation plots for terrain pipeline results.

Joins SIR NetCDF output to catchment polygons and produces:
  1. Choropleth maps (elevation, slope, aspect)
  2. Histograms of variable distributions
  3. Summary statistics printed to terminal

Usage:
    python scripts/plot_terrain_results.py <output.nc> <catchments.gpkg>

Example:
    pixi run -e full python scripts/plot_terrain_results.py \
        output/delaware_terrain.nc data/delaware/catchments.gpkg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

VARIABLES = ["elevation", "slope", "aspect"]

PLOT_CONFIG = {
    "elevation": {
        "cmap": "terrain",
        "label": "Elevation (m)",
        "title": "Mean Elevation",
    },
    "slope": {
        "cmap": "YlOrRd",
        "label": "Slope (degrees)",
        "title": "Mean Slope",
    },
    "aspect": {
        "cmap": "hsv",
        "label": "Aspect (degrees)",
        "title": "Mean Aspect",
    },
}


def load_data(nc_path: Path, gpkg_path: Path) -> gpd.GeoDataFrame:
    """Load SIR NetCDF and join to catchment polygons.

    Parameters
    ----------
    nc_path : Path
        Path to the SIR NetCDF file.
    gpkg_path : Path
        Path to the catchments GeoPackage.

    Returns
    -------
    gpd.GeoDataFrame
        Catchments with terrain variables joined.
    """
    ds = xr.open_dataset(nc_path)
    df = ds.to_dataframe().reset_index()
    ds.close()

    catchments = gpd.read_file(gpkg_path)

    # Join on featureid = hru_id
    merged = catchments.merge(df, left_on="featureid", right_on="hru_id", how="inner")
    print(f"Joined {len(merged)} of {len(catchments)} catchments to SIR data")

    return merged


def print_summary(gdf: gpd.GeoDataFrame) -> None:
    """Print summary statistics for each variable."""
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    for var in VARIABLES:
        if var not in gdf.columns:
            print(f"\n  {var}: NOT FOUND in data")
            continue

        values = gdf[var]
        n_total = len(values)
        n_valid = values.notna().sum()
        n_missing = n_total - n_valid

        print(f"\n  {var}:")
        print(f"    count:   {n_valid}")
        print(f"    missing: {n_missing} ({100 * n_missing / n_total:.1f}%)")
        if n_valid > 0:
            print(f"    min:     {values.min():.2f}")
            print(f"    max:     {values.max():.2f}")
            print(f"    mean:    {values.mean():.2f}")
            print(f"    std:     {values.std():.2f}")
            print(f"    median:  {values.median():.2f}")

    print("\n" + "=" * 60)


def plot_choropleths(gdf: gpd.GeoDataFrame, output_dir: Path) -> None:
    """Create 1x3 choropleth map panel."""
    available = [v for v in VARIABLES if v in gdf.columns]
    n = len(available)
    if n == 0:
        print("No variables found for choropleth maps.")
        return

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, available, strict=False):
        cfg = PLOT_CONFIG[var]
        vmin = gdf[var].quantile(0.02)
        vmax = gdf[var].quantile(0.98)

        gdf.plot(
            column=var,
            ax=ax,
            cmap=cfg["cmap"],
            legend=True,
            legend_kwds={"label": cfg["label"], "shrink": 0.6},
            vmin=vmin,
            vmax=vmax,
            edgecolor="none",
            linewidth=0.1,
        )
        ax.set_title(cfg["title"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    fig.suptitle("Delaware River Basin — Terrain Parameters", fontsize=14, y=1.02)
    fig.tight_layout()

    out_path = output_dir / "choropleth_maps.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved choropleth maps → {out_path}")


def plot_histograms(gdf: gpd.GeoDataFrame, output_dir: Path) -> None:
    """Create 1x3 histogram panel."""
    available = [v for v in VARIABLES if v in gdf.columns]
    n = len(available)
    if n == 0:
        print("No variables found for histograms.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, var in zip(axes, available, strict=False):
        cfg = PLOT_CONFIG[var]
        values = gdf[var].dropna()

        ax.hist(values, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(cfg["label"])
        ax.set_ylabel("Count")
        ax.set_title(cfg["title"])

        # Add vertical line for mean
        mean_val = np.mean(values)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1, label=f"mean={mean_val:.1f}")
        ax.legend(fontsize=8)

    fig.suptitle("Distribution of Terrain Parameters", fontsize=14)
    fig.tight_layout()

    out_path = output_dir / "histograms.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histograms → {out_path}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Plot validation results for terrain pipeline output.",
    )
    parser.add_argument("nc_path", type=Path, help="Path to SIR NetCDF file")
    parser.add_argument("gpkg_path", type=Path, help="Path to catchments GeoPackage")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for plot output (default: same directory as NC file + /plots)",
    )
    args = parser.parse_args()

    if not args.nc_path.exists():
        print(f"Error: NetCDF file not found: {args.nc_path}", file=sys.stderr)
        return 1
    if not args.gpkg_path.exists():
        print(f"Error: GeoPackage not found: {args.gpkg_path}", file=sys.stderr)
        return 1

    output_dir = args.output_dir or (args.nc_path.parent / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_data(args.nc_path, args.gpkg_path)
    print_summary(gdf)
    plot_choropleths(gdf, output_dir)
    plot_histograms(gdf, output_dir)

    print(f"\nAll plots saved to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
