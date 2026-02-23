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
from pydantic import BaseModel

from hydro_param.config import PipelineConfig, ProcessingConfig
from hydro_param.dataset_registry import (
    DatasetEntry,
    DerivedVariableSpec,
    VariableSpec,
)

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".manifest.yml"


class ManifestEntry(BaseModel):
    """A single dataset's entry in the pipeline manifest."""

    fingerprint: str
    static_files: dict[str, str] = {}
    temporal_files: dict[str, str] = {}
    completed_at: str = ""


class PipelineManifest(BaseModel):
    """Pipeline manifest recording what config produced each output file."""

    version: int = 1
    fabric_fingerprint: str = ""
    entries: dict[str, ManifestEntry] = {}

    def save(self, output_dir: Path) -> None:
        """Write manifest to ``{output_dir}/.manifest.yml``."""
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / MANIFEST_FILENAME
        data = self.model_dump()
        manifest_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    def is_dataset_current(
        self,
        ds_name: str,
        fingerprint: str,
        output_dir: Path,
    ) -> bool:
        """Check whether a dataset's outputs are still valid.

        Returns ``True`` when the fingerprint matches AND all listed
        output files exist on disk.
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
    """Load a manifest from disk, returning ``None`` if missing or corrupt."""
    manifest_path = output_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        raw = yaml.safe_load(manifest_path.read_text())
        return PipelineManifest(**raw)
    except Exception:
        logger.warning("Corrupt manifest at %s — will reprocess all datasets", manifest_path)
        return None


def fabric_fingerprint(config: PipelineConfig) -> str:
    """Compute a fingerprint for the target fabric file.

    Returns ``"{filename}|{mtime}|{size}"`` as a cheap proxy for
    content identity without hashing large GeoPackages.
    """
    path = config.target_fabric.path
    stat = path.stat()
    return f"{path.name}|{stat.st_mtime}|{stat.st_size}"


def dataset_fingerprint(
    ds_req: object,
    entry: DatasetEntry,
    var_specs: list[VariableSpec | DerivedVariableSpec],
    processing: ProcessingConfig,
) -> str:
    """Compute a SHA-256 fingerprint for a dataset request.

    Captures all fields that affect processing output so that any
    config change triggers reprocessing.
    """
    from hydro_param.config import DatasetRequest

    req: DatasetRequest = ds_req  # type: ignore[assignment]

    canonical: dict[str, object] = {
        "ds_req": {
            "name": req.name,
            "variables": req.variables,
            "statistics": req.statistics,
            "year": req.year,
            "time_period": req.time_period,
            "source": str(req.source) if req.source is not None else None,
        },
        "entry": {
            "strategy": entry.strategy,
            "source": entry.source,
            "crs": entry.crs,
            "collection": entry.collection,
            "catalog_url": entry.catalog_url,
            "catalog_id": entry.catalog_id,
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
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
