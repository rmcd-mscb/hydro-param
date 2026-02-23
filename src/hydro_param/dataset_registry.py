"""Dataset registry: load and resolve dataset definitions from YAML.

The registry maps human-readable dataset names to access strategies,
variable specifications, and derivation rules. See design.md section 6.6
and section 11.3 for the schema design.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class VariableSpec(BaseModel):
    """A variable available directly in a dataset."""

    name: str
    band: int = 1
    units: str = ""
    long_name: str = ""
    categorical: bool = False
    asset_key: str | None = None  # per-variable STAC asset override (e.g. gNATSGO)


class DerivedVariableSpec(BaseModel):
    """A variable derived from another variable (e.g., slope from elevation)."""

    name: str
    source: str
    method: str
    units: str = ""
    long_name: str = ""


class DownloadFile(BaseModel):
    """A single downloadable file in a multi-file dataset."""

    year: int
    variable: str
    url: str
    size_gb: float | None = None


class DownloadInfo(BaseModel):
    """Download provenance for datasets requiring local staging."""

    url: str = ""
    size_gb: float | None = None
    format: str = ""
    notes: str = ""
    files: list[DownloadFile] = []
    url_template: str = ""
    year_range: list[int] = []
    variables_available: list[str] = []
    requester_pays: bool = False

    @model_validator(mode="after")
    def _require_download_source(self) -> DownloadInfo:
        if not self.url and not self.files and not self.url_template:
            raise ValueError("DownloadInfo requires at least 'url', 'files', or 'url_template'")
        if self.url_template:
            if len(self.year_range) != 2 or self.year_range[0] > self.year_range[1]:
                raise ValueError(
                    "url_template requires 'year_range' as [start, end] with start <= end"
                )
            if not self.variables_available:
                raise ValueError("url_template requires a non-empty 'variables_available' list")
        return self

    def expand_files(
        self,
        *,
        years: set[int] | None = None,
        variables: set[str] | None = None,
    ) -> list[DownloadFile]:
        """Expand download sources into a list of files, with optional filtering.

        For template mode, iterates year_range x variables_available and
        formats the URL template. For explicit files mode, returns the
        files list with optional year/variable filtering.

        Parameters
        ----------
        years
            If given, only include files matching these years.
        variables
            If given, only include files matching these variables.

        Returns
        -------
        list[DownloadFile]
        """
        if self.url_template:
            start, end = self.year_range
            result = []
            for yr in range(start, end + 1):
                if years is not None and yr not in years:
                    continue
                for var in self.variables_available:
                    if variables is not None and var not in variables:
                        continue
                    url = self.url_template.format(variable=var, year=yr)
                    result.append(DownloadFile(year=yr, variable=var, url=url))
            return result

        result = list(self.files)
        if years is not None:
            result = [f for f in result if f.year in years]
        if variables is not None:
            result = [f for f in result if f.variable in variables]
        return result


class DatasetEntry(BaseModel):
    """A single dataset in the registry."""

    description: str = ""
    strategy: Literal[
        "stac_cog", "native_zarr", "converted_zarr", "local_tiff", "climr_cat", "nhgf_stac"
    ]
    # STAC COG fields
    catalog_url: str | None = None
    collection: str | None = None
    asset_key: str = "data"
    gsd: int | None = None
    sign: str | None = None
    # Zarr / local fields
    source: str | None = None
    # Download provenance (local_tiff datasets that require user download)
    download: DownloadInfo | None = None
    # ClimateR-Catalog identifier (climr_cat strategy)
    catalog_id: str | None = None
    # Common fields
    crs: str = "EPSG:4326"
    x_coord: str = "x"
    y_coord: str = "y"
    t_coord: str | None = None
    variables: list[VariableSpec] = []
    derived_variables: list[DerivedVariableSpec] = []
    category: str = ""
    temporal: bool = False
    year_range: list[int] | None = None

    @model_validator(mode="after")
    def _validate_strategy_fields(self) -> DatasetEntry:
        if self.strategy == "stac_cog":
            if not self.catalog_url or not self.collection:
                raise ValueError("stac_cog strategy requires 'catalog_url' and 'collection'")
        if self.strategy == "climr_cat":
            if not self.catalog_id:
                raise ValueError("climr_cat strategy requires 'catalog_id'")
            if not self.temporal:
                raise ValueError("climr_cat strategy requires 'temporal: true'")
        if self.strategy == "nhgf_stac":
            if not self.collection:
                raise ValueError("nhgf_stac strategy requires 'collection'")
        if self.temporal and not self.t_coord:
            raise ValueError("Temporal datasets require 't_coord'")
        if self.year_range is not None:
            if len(self.year_range) != 2:
                raise ValueError("year_range must be a 2-element list [start, end]")
            if self.year_range[0] > self.year_range[1]:
                raise ValueError(
                    "year_range start must be <= end: "
                    f"got [{self.year_range[0]}, {self.year_range[1]}]"
                )
        return self


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
    """Load a dataset registry from a YAML file or directory of YAML files.

    Parameters
    ----------
    path : str or Path
        Path to a single registry YAML file, or a directory containing
        per-category YAML files.  Each file must have a top-level
        ``datasets:`` key mapping dataset names to entries.

    Returns
    -------
    DatasetRegistry

    Raises
    ------
    FileNotFoundError
        If the path does not exist or contains no datasets.
    ValueError
        If dataset names collide across files.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry path does not exist: {path}")
    if path.is_file():
        return _load_registry_file(path)
    if path.is_dir():
        return _load_registry_dir(path)
    raise FileNotFoundError(f"Registry path is neither a file nor directory: {path}")


def _load_registry_file(path: Path) -> DatasetRegistry:
    """Load a single registry YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return DatasetRegistry(**raw)


def _load_registry_dir(directory: Path) -> DatasetRegistry:
    """Load and merge all YAML files in a registry directory.

    Parameters
    ----------
    directory : Path
        Directory containing ``*.yml`` and/or ``*.yaml`` files, each with
        a ``datasets:`` root key.

    Returns
    -------
    DatasetRegistry

    Raises
    ------
    FileNotFoundError
        If the directory contains no YAML files or no datasets.
    ValueError
        If a dataset name appears in more than one file.
    """
    yaml_files = sorted(list(directory.glob("*.yml")) + list(directory.glob("*.yaml")))
    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML files (*.yml, *.yaml) found in registry directory: {directory}"
        )

    merged: dict[str, DatasetEntry] = {}
    source_files: dict[str, str] = {}
    for yaml_file in yaml_files:
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
        if raw is None or "datasets" not in raw:
            continue
        partial = DatasetRegistry(**raw)
        for name, entry in partial.datasets.items():
            if name in merged:
                raise ValueError(
                    f"Duplicate dataset name '{name}': found in "
                    f"'{yaml_file.name}' and '{source_files[name]}'. "
                    f"Dataset names must be unique across all registry files."
                )
            merged[name] = entry
            source_files[name] = yaml_file.name

    if not merged:
        raise FileNotFoundError(f"No datasets found in any YAML file in: {directory}")

    return DatasetRegistry(datasets=merged)
