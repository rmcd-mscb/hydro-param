"""Dataset registry: load and resolve dataset definitions from YAML.

The registry maps human-readable dataset names to access strategies,
variable specifications, and derivation rules. See design.md section 6.6
and section 11.3 for the schema design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class VariableSpec(BaseModel):
    """A variable available directly in a dataset."""

    name: str
    band: int = 1
    units: str = ""
    long_name: str = ""
    categorical: bool = False


class DerivedVariableSpec(BaseModel):
    """A variable derived from another variable (e.g., slope from elevation)."""

    name: str
    source: str
    method: str
    units: str = ""
    long_name: str = ""


class DatasetEntry(BaseModel):
    """A single dataset in the registry."""

    description: str = ""
    strategy: Literal["stac_cog", "native_zarr", "converted_zarr", "local_tiff"]
    # STAC COG fields
    catalog_url: str | None = None
    collection: str | None = None
    asset_key: str = "data"
    gsd: int | None = None
    sign: str | None = None
    # Zarr / local fields
    source: str | None = None
    # Common fields
    crs: str = "EPSG:4326"
    variables: list[VariableSpec] = []
    derived_variables: list[DerivedVariableSpec] = []
    category: str = ""
    temporal: bool = False


class DatasetRegistry(BaseModel):
    """Container for all registered datasets."""

    datasets: dict[str, DatasetEntry]

    def get(self, name: str) -> DatasetEntry:
        """Look up a dataset by name.

        Parameters
        ----------
        name : str
            Dataset name as it appears in the registry YAML.

        Returns
        -------
        DatasetEntry

        Raises
        ------
        KeyError
            If the dataset is not found.
        """
        if name not in self.datasets:
            available = ", ".join(sorted(self.datasets.keys()))
            raise KeyError(f"Dataset '{name}' not found in registry. Available: {available}")
        return self.datasets[name]

    def resolve_variable(
        self, dataset_name: str, variable_name: str
    ) -> VariableSpec | DerivedVariableSpec:
        """Resolve a variable name to its spec within a dataset.

        Parameters
        ----------
        dataset_name : str
            Dataset name in the registry.
        variable_name : str
            Variable name to look up.

        Returns
        -------
        VariableSpec or DerivedVariableSpec

        Raises
        ------
        KeyError
            If the dataset or variable is not found.
        """
        entry = self.get(dataset_name)
        for v in entry.variables:
            if v.name == variable_name:
                return v
        for dv in entry.derived_variables:
            if dv.name == variable_name:
                return dv
        available = [v.name for v in entry.variables] + [dv.name for dv in entry.derived_variables]
        raise KeyError(
            f"Variable '{variable_name}' not found in dataset '{dataset_name}'. "
            f"Available: {', '.join(available)}"
        )


def load_registry(path: str | Path) -> DatasetRegistry:
    """Load a dataset registry from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the registry YAML file.

    Returns
    -------
    DatasetRegistry
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DatasetRegistry(**raw)
