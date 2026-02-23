"""Pipeline manifest for resume support.

Records what config produced each output file so that re-runs with
``resume: true`` can skip datasets whose outputs are already complete
and inputs haven't changed.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

from hydro_param.config import DatasetRequest, PipelineConfig, ProcessingConfig
from hydro_param.dataset_registry import (
    DatasetEntry,
    DerivedVariableSpec,
    VariableSpec,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".manifest.yml"

_SUPPORTED_VERSION = 1


class ManifestEntry(BaseModel):
    """A single dataset's entry in the pipeline manifest.

    Keys in ``static_files`` and ``temporal_files`` are variable/result
    names; values are paths relative to the output directory.
    """

    fingerprint: str
    static_files: dict[str, str] = {}
    temporal_files: dict[str, str] = {}
    completed_at: datetime = datetime.min

    @field_validator("completed_at", mode="before")
    @classmethod
    def _parse_completed_at(cls, v: object) -> object:
        """Accept ISO strings and empty strings (legacy/partial manifests)."""
        if isinstance(v, str):
            return datetime.fromisoformat(v) if v else datetime.min
        return v


class PipelineManifest(BaseModel):
    """Pipeline manifest recording what config produced each output file."""

    version: int = _SUPPORTED_VERSION
    fabric_fingerprint: str = ""
    entries: dict[str, ManifestEntry] = {}

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != _SUPPORTED_VERSION:
            raise ValueError(f"Unsupported manifest version {v} (expected {_SUPPORTED_VERSION})")
        return v

    def save(self, output_dir: Path) -> None:
        """Write manifest to ``{output_dir}/.manifest.yml``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / MANIFEST_FILENAME
        data = self.model_dump(mode="json")
        manifest_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def is_fabric_current(self, expected_fingerprint: str) -> bool:
        """Check whether the fabric fingerprint matches."""
        return self.fabric_fingerprint == expected_fingerprint

    def is_dataset_current(
        self,
        ds_name: str,
        fingerprint: str,
        output_dir: Path,
    ) -> bool:
        """Check whether a dataset's outputs are still valid.

        Returns ``True`` when the dataset is present in the manifest,
        the fingerprint matches, AND all listed output files exist on
        disk.  Returns ``False`` for unknown datasets.
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
    """Load a manifest from disk, returning ``None`` if missing or corrupt.

    I/O and permission errors are raised (not swallowed) so the caller
    can surface actionable diagnostics.  Only YAML parse errors and
    schema validation errors result in a ``None`` return.
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
    except (yaml.YAMLError, ValueError, Exception) as exc:
        logger.warning(
            "Corrupt manifest at %s — will reprocess all datasets. Error: %s",
            manifest_path,
            exc,
        )
        return None


def fabric_fingerprint(config: PipelineConfig) -> str:
    """Compute a fingerprint for the target fabric file.

    Returns ``"{filename}|{mtime}|{size}"`` as a cheap proxy for
    content identity without hashing large GeoPackages.

    Note: mtime changes when a file is copied or restored from backup,
    causing unnecessary reprocessing even if content is identical.
    This is acceptable for MVP but may warrant content-based hashing
    for large production workflows.
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
    """Compute a SHA-256 fingerprint for a dataset request.

    Captures the fields that affect processing output so that any
    config change triggers reprocessing.

    Deliberately excluded (do not affect output content):
    ``resume``, ``failure_mode``, ``description``, ``download``,
    ``year_range`` (informational).
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
    """Create a ManifestEntry with paths relative to output_dir."""
    return ManifestEntry(
        fingerprint=fingerprint,
        static_files={k: str(v.relative_to(output_dir)) for k, v in static_files.items()},
        temporal_files={k: str(v.relative_to(output_dir)) for k, v in temporal_files.items()},
        completed_at=datetime.now(timezone.utc),
    )
