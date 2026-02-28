"""Manage the pipeline manifest for incremental resume support.

Record which configuration produced each output file so that re-runs
with ``resume: true`` can skip datasets whose outputs are already
complete and whose inputs (config, registry entry, processing options)
have not changed.

The manifest is stored as ``.manifest.yml`` in the output directory
and tracks per-dataset SHA-256 fingerprints, output file paths, and
completion timestamps.  A separate fabric fingerprint detects when
the target fabric file has changed, invalidating all cached results.

Notes
-----
The manifest uses cheap file-metadata proxies (filename, mtime, size)
for fabric identity and content-based SHA-256 hashing for dataset
configuration identity.  This avoids hashing large GeoPackage files
while still detecting config changes reliably.

See Also
--------
hydro_param.pipeline : Pipeline orchestrator that reads/writes manifests.
hydro_param.cli.run_cmd : CLI ``--resume`` flag that enables manifest use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from hydro_param.config import DatasetRequest, PipelineConfig, ProcessingConfig
from hydro_param.dataset_registry import (
    DatasetEntry,
    DerivedVariableSpec,
    VariableSpec,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".manifest.yml"

_SUPPORTED_VERSIONS = {1, 2}
_CURRENT_VERSION = 2


class ManifestEntry(BaseModel):
    """Represent a single dataset's record in the pipeline manifest.

    Track the configuration fingerprint, output file paths, and
    completion timestamp for one dataset.  Used to determine whether
    a dataset can be skipped during resume runs.

    Attributes
    ----------
    fingerprint : str
        SHA-256 fingerprint of the dataset request, registry entry,
        variable specs, and processing config.  Format:
        ``"sha256:<hex>"``.
    static_files : dict[str, str]
        Mapping of variable/result names to output file paths
        relative to the output directory.
    temporal_files : dict[str, str]
        Mapping of temporal variable names to output file paths
        relative to the output directory.
    completed_at : datetime
        UTC timestamp when processing completed.  Defaults to
        ``datetime.min`` for incomplete or legacy entries.
    """

    fingerprint: str
    static_files: dict[str, str] = {}
    temporal_files: dict[str, str] = {}
    completed_at: datetime = datetime.min.replace(tzinfo=timezone.utc)

    @field_validator("completed_at", mode="before")
    @classmethod
    def _parse_completed_at(cls, v: object) -> object:
        """Parse ISO date strings and accept empty strings for legacy manifests."""
        if isinstance(v, str):
            return datetime.fromisoformat(v) if v else datetime.min.replace(tzinfo=timezone.utc)
        return v


class SIRSchemaEntry(TypedDict):
    """Schema metadata for a single SIR variable.

    Attributes
    ----------
    name : str
        Canonical SIR variable name (e.g., ``"elevation_m_mean"``).
    units : str
        Physical units of the variable (e.g., ``"m"``, ``"fraction"``).
    statistic : str
        Zonal statistic used (e.g., ``"mean"``, ``"categorical"``).
    """

    name: str
    units: str
    statistic: str


class SIRManifestEntry(BaseModel):
    """Track normalized SIR output from stage 5.

    Record the file paths, schema metadata, and completion time for
    the SIR normalization step.  Used by Phase 2 (model plugins) to
    discover what the pipeline produced without re-running it.

    Attributes
    ----------
    static_files : dict[str, str]
        Mapping of SIR variable names to file paths relative to the
        output directory (e.g., ``{"elevation_m_mean": "sir/elevation_m_mean.csv"}``).
    temporal_files : dict[str, str]
        Mapping of temporal dataset keys to file paths relative to the
        output directory (e.g., ``{"gridmet_2020": "sir/gridmet_2020.nc"}``).
    sir_schema : list[SIRSchemaEntry]
        SIR variable schema entries from ``build_sir_schema()``.
        Each entry contains ``name``, ``units``, and ``statistic`` keys.
    completed_at : datetime
        UTC timestamp when SIR normalization completed.

    See Also
    --------
    SIRSchemaEntry : TypedDict defining the schema entry structure.

    Notes
    -----
    This entry is the contract between Phase 1 (pipeline) and Phase 2
    (model plugins).  ``SIRAccessor`` reads these file paths to discover
    available SIR variables without re-running the pipeline.
    """

    static_files: dict[str, str] = {}
    temporal_files: dict[str, str] = {}
    sir_schema: list[SIRSchemaEntry] = []
    completed_at: datetime = datetime.min.replace(tzinfo=timezone.utc)

    @field_validator("completed_at", mode="before")
    @classmethod
    def _parse_completed_at(cls, v: object) -> object:
        """Parse ISO date strings and accept empty strings for legacy entries."""
        if isinstance(v, str):
            return datetime.fromisoformat(v) if v else datetime.min.replace(tzinfo=timezone.utc)
        return v


class PipelineManifest(BaseModel):
    """Record what configuration produced each output file.

    The manifest is the top-level structure persisted as
    ``.manifest.yml`` in the output directory.  It contains a fabric
    fingerprint (to detect fabric changes) and per-dataset entries
    (to detect config changes and verify output completeness).

    Attributes
    ----------
    version : int
        Manifest schema version.  Must be one of ``_SUPPORTED_VERSIONS``
        (currently {1, 2}).  Incompatible versions cause a validation error.
    fabric_fingerprint : str
        Fingerprint of the target fabric file (format:
        ``"{filename}|{mtime}|{size}"``).  Empty string for new
        manifests.
    entries : dict[str, ManifestEntry]
        Per-dataset manifest entries, keyed by dataset name.
    sir : SIRManifestEntry or None
        SIR output tracking for Phase 2 consumers.  ``None`` for v1
        manifests created before SIR normalization was implemented,
        and also valid for v2 manifests that have not yet run SIR
        normalization.
    """

    version: int = _CURRENT_VERSION
    fabric_fingerprint: str = ""
    entries: dict[str, ManifestEntry] = {}
    sir: SIRManifestEntry | None = None

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        """Reject manifest versions that don't match the current schema."""
        if v not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported manifest version {v} (expected one of {sorted(_SUPPORTED_VERSIONS)})"
            )
        return v

    def save(self, output_dir: Path) -> None:
        """Write the manifest atomically to ``{output_dir}/.manifest.yml``.

        Write to a temporary file first, then atomically rename.
        This prevents corrupt manifests from partial writes (e.g.,
        disk-full or interrupted process).

        Parameters
        ----------
        output_dir
            Output directory.  Created if it does not exist.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / MANIFEST_FILENAME
        tmp_path = output_dir / f"{MANIFEST_FILENAME}.tmp"
        data = self.model_dump(mode="json")
        try:
            tmp_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
            tmp_path.replace(manifest_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise

    def is_fabric_current(self, expected_fingerprint: str) -> bool:
        """Check whether the stored fabric fingerprint matches the expected value.

        Parameters
        ----------
        expected_fingerprint
            Fingerprint computed from the current fabric file via
            ``fabric_fingerprint()``.

        Returns
        -------
        bool
            ``True`` if the fingerprints match (fabric unchanged).
        """
        return self.fabric_fingerprint == expected_fingerprint

    def is_dataset_current(
        self,
        ds_name: str,
        fingerprint: str,
        output_dir: Path,
    ) -> bool:
        """Check whether a dataset's outputs are still valid for reuse.

        A dataset is considered current when all three conditions hold:

        1. The dataset name exists in the manifest.
        2. The stored fingerprint matches the computed fingerprint
           (no config changes).
        3. All listed output files (static and temporal) exist on disk.

        Parameters
        ----------
        ds_name
            Dataset name as it appears in the pipeline config.
        fingerprint
            SHA-256 fingerprint computed from the current dataset
            request, registry entry, variable specs, and processing
            config via ``dataset_fingerprint()``.
        output_dir
            Output directory used to resolve relative file paths.

        Returns
        -------
        bool
            ``True`` if the dataset can be skipped (outputs are
            current); ``False`` if it needs reprocessing.
        """
        if ds_name not in self.entries:
            return False
        entry = self.entries[ds_name]
        if entry.fingerprint != fingerprint:
            return False
        for rel_path in entry.static_files.values():
            if not (output_dir / rel_path).exists():
                return False
        for rel_path in entry.temporal_files.values():
            if not (output_dir / rel_path).exists():
                return False
        return True


def load_manifest(output_dir: Path) -> PipelineManifest | None:
    """Load a manifest from disk, returning ``None`` if absent or corrupt.

    Attempt to read and parse ``.manifest.yml`` from the output
    directory.  Filesystem errors (permissions, I/O) propagate to the
    caller for actionable diagnostics.  YAML parse errors and Pydantic
    validation failures are caught, logged as warnings, and result in
    a ``None`` return (triggering full reprocessing).

    Parameters
    ----------
    output_dir
        Directory containing ``.manifest.yml``.

    Returns
    -------
    PipelineManifest or None
        Loaded manifest, or ``None`` if the file does not exist or
        fails to parse/validate.

    Raises
    ------
    OSError
        If the file exists but cannot be read (permissions, disk
        errors).
    """
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    # Read file first — let OSError/PermissionError propagate
    text = manifest_path.read_text()
    try:
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            raise ValueError(f"Expected YAML mapping, got {type(raw).__name__}")
        return PipelineManifest(**raw)
    except (yaml.YAMLError, ValueError, ValidationError) as exc:
        logger.warning(
            "Corrupt manifest at %s — will reprocess all datasets. Error: %s",
            manifest_path,
            exc,
        )
        return None


def fabric_fingerprint(config: PipelineConfig) -> str:
    """Compute a fingerprint for the target fabric file.

    Return ``"{filename}|{mtime}|{size}"`` as a cheap proxy for
    content identity without hashing large GeoPackage files.

    Parameters
    ----------
    config
        Pipeline configuration containing the ``target_fabric.path``.

    Returns
    -------
    str
        Fingerprint string in the format ``"{filename}|{mtime}|{size}"``.

    Raises
    ------
    FileNotFoundError
        If the fabric file does not exist at the configured path.

    Notes
    -----
    The mtime-based fingerprint changes when a file is copied or
    restored from backup, causing unnecessary reprocessing even if
    content is identical.  This is acceptable for MVP but may warrant
    content-based hashing for large production workflows.

    This function deliberately avoids hashing geometry coordinates,
    consistent with the project's cache-by-stable-ID principle (see
    CLAUDE.md architectural decision 7).
    """
    path = config.target_fabric.path
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot compute fabric fingerprint: file not found at {path}. "
            f"Ensure target_fabric.path is correct in your pipeline config."
        ) from None
    return f"{path.name}|{stat.st_mtime}|{stat.st_size}"


def dataset_fingerprint(
    ds_req: DatasetRequest,
    entry: DatasetEntry,
    var_specs: list[VariableSpec | DerivedVariableSpec],
    processing: ProcessingConfig,
) -> str:
    """Compute a SHA-256 fingerprint for a dataset processing request.

    Serialize all fields that affect processing output into a canonical
    JSON representation and hash it with SHA-256.  Any change to the
    dataset request, registry entry metadata, variable specifications,
    or processing options will produce a different fingerprint,
    triggering reprocessing on resume.

    Parameters
    ----------
    ds_req
        Dataset request from the pipeline config (name, variables,
        statistics, year, time_period, source override).
    entry
        Registry entry for the dataset (strategy, source paths, CRS,
        STAC collection, etc.).
    var_specs
        Resolved variable specifications (band numbers, categorical
        flags) and derived variable specifications (source, method).
    processing
        Processing config (engine type, batch size).

    Returns
    -------
    str
        Fingerprint in the format ``"sha256:<64-char-hex>"``.

    Notes
    -----
    Deliberately excluded fields (do not affect output content):
    ``resume``, ``description``, ``download``,
    ``year_range`` (informational only).

    The JSON serialization uses sorted keys and compact separators
    to ensure deterministic output across Python versions.
    """
    canonical: dict[str, object] = {
        "ds_req": {
            "name": ds_req.name,
            "variables": ds_req.variables,
            "statistics": ds_req.statistics,
            "year": ds_req.year,
            "time_period": ds_req.time_period,
            "source": str(ds_req.source) if ds_req.source is not None else None,
        },
        "entry": {
            "strategy": entry.strategy,
            "source": entry.source,
            "crs": entry.crs,
            "collection": entry.collection,
            "catalog_url": entry.catalog_url,
            "catalog_id": entry.catalog_id,
            "asset_key": entry.asset_key,
            "gsd": entry.gsd,
            "sign": entry.sign,
            "x_coord": entry.x_coord,
            "y_coord": entry.y_coord,
        },
        "var_specs": [
            (
                {
                    "name": v.name,
                    "band": v.band,
                    "categorical": v.categorical,
                    "source_override": v.source_override,
                }
                if isinstance(v, VariableSpec)
                else {
                    "name": v.name,
                    "source": v.source,
                    "method": v.method,
                }
            )
            for v in var_specs
        ],
        "processing": {
            "engine": processing.engine,
            "batch_size": processing.batch_size,
        },
    }

    json_bytes = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(json_bytes).hexdigest()}"


def make_manifest_entry(
    fingerprint: str,
    static_files: dict[str, Path],
    temporal_files: dict[str, Path],
    output_dir: Path,
) -> ManifestEntry:
    """Create a ManifestEntry with paths stored relative to the output directory.

    Convert absolute file paths to relative paths for portability
    (the output directory can be moved without invalidating the
    manifest) and stamp the entry with the current UTC time.

    Parameters
    ----------
    fingerprint
        SHA-256 fingerprint for the dataset configuration (from
        ``dataset_fingerprint()``).
    static_files
        Mapping of variable names to absolute paths for static
        output files.
    temporal_files
        Mapping of variable names to absolute paths for temporal
        output files.
    output_dir
        Root output directory used to compute relative paths.

    Returns
    -------
    ManifestEntry
        Entry ready for insertion into ``PipelineManifest.entries``.
    """
    return ManifestEntry(
        fingerprint=fingerprint,
        static_files={k: str(v.relative_to(output_dir)) for k, v in static_files.items()},
        temporal_files={k: str(v.relative_to(output_dir)) for k, v in temporal_files.items()},
        completed_at=datetime.now(timezone.utc),
    )
