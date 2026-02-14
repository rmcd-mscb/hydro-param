"""Core processing: zonal statistics over polygon fabrics."""

from __future__ import annotations

from typing import Protocol

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import Point


class Processor(Protocol):
    def process(
        self, fabric: gpd.GeoDataFrame, dataset: xr.Dataset, variable: str
    ) -> xr.Dataset: ...


class ZonalProcessor:
    """Point-in-polygon zonal mean. Placeholder for gdptools ZonalGen."""

    def process(self, fabric: gpd.GeoDataFrame, dataset: xr.Dataset, variable: str) -> xr.Dataset:
        # Convert grid cells to centroid points
        lon, lat = np.meshgrid(dataset.coords["x"].values, dataset.coords["y"].values)
        points = gpd.GeoDataFrame(
            {variable: dataset[variable].values.ravel()},
            geometry=[Point(x, y) for x, y in zip(lon.ravel(), lat.ravel(), strict=True)],
            crs=fabric.crs,
        )

        # Spatial join: assign each point to a polygon
        joined = gpd.sjoin(points, fabric[["geometry", "hru_id"]], predicate="within")
        means = joined.groupby("hru_id")[variable].mean()

        return xr.Dataset(
            {variable: ("hru_id", means.values)},
            coords={"hru_id": means.index.values},
            attrs={
                "source_dataset": variable,
                "processing_method": "zonal_mean",
            },
        )


def get_processor(fabric: gpd.GeoDataFrame) -> Processor:
    geom_type = fabric.geometry.geom_type.iloc[0]
    if geom_type in ("Polygon", "MultiPolygon"):
        return ZonalProcessor()
    raise ValueError(f"Unsupported geometry type: {geom_type}")
