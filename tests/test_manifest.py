"""Tests for pipeline manifest (resume support)."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from hydro_param.config import PipelineConfig, ProcessingConfig
from hydro_param.dataset_registry import DatasetEntry, DerivedVariableSpec, VariableSpec
from hydro_param.manifest import (
    ManifestEntry,
    PipelineManifest,
    dataset_fingerprint,
    fabric_fingerprint,
    load_manifest,
    make_manifest_entry,
)

# ---------------------------------------------------------------------------
# fabric_fingerprint
# ---------------------------------------------------------------------------


def test_fabric_fingerprint(tmp_path: Path):
    """fabric_fingerprint returns '{filename}|{mtime}|{size}' format."""
    gpkg = tmp_path / "test.gpkg"
    gpkg.write_text("fake data")

    config = PipelineConfig(
        target_fabric={"path": str(gpkg), "id_field": "id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    fp = fabric_fingerprint(config)
    parts = fp.split("|")
    assert len(parts) == 3
    assert parts[0] == "test.gpkg"
    # mtime and size should be numeric
    float(parts[1])
    int(parts[2])


def test_fabric_fingerprint_missing_file(tmp_path: Path):
    """fabric_fingerprint raises FileNotFoundError with actionable message."""
    config = PipelineConfig(
        target_fabric={"path": str(tmp_path / "nonexistent.gpkg"), "id_field": "id"},
        domain={"type": "bbox", "bbox": [0, 0, 1, 1]},
        datasets=[],
    )

    with pytest.raises(FileNotFoundError, match="Cannot compute fabric fingerprint"):
        fabric_fingerprint(config)


# ---------------------------------------------------------------------------
# dataset_fingerprint
# ---------------------------------------------------------------------------


def test_dataset_fingerprint_stable():
    """Same inputs produce the same hash."""
    from hydro_param.config import DatasetRequest

    ds_req = DatasetRequest(name="dem", variables=["elevation"], statistics=["mean"])
    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com",
        collection="3dep",
        crs="EPSG:4326",
    )
    var_specs: list[VariableSpec | DerivedVariableSpec] = [VariableSpec(name="elevation", band=1)]
    processing = ProcessingConfig()

    fp1 = dataset_fingerprint(ds_req, entry, var_specs, processing)
    fp2 = dataset_fingerprint(ds_req, entry, var_specs, processing)
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_dataset_fingerprint_changes_on_variable_diff():
    """Different variables produce different hashes."""
    from hydro_param.config import DatasetRequest

    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com",
        collection="3dep",
    )
    processing = ProcessingConfig()

    ds_req1 = DatasetRequest(name="dem", variables=["elevation"], statistics=["mean"])
    vars1: list[VariableSpec | DerivedVariableSpec] = [VariableSpec(name="elevation", band=1)]

    ds_req2 = DatasetRequest(name="dem", variables=["slope"], statistics=["mean"])
    vars2: list[VariableSpec | DerivedVariableSpec] = [
        DerivedVariableSpec(name="slope", source="elevation", method="horn")
    ]

    fp1 = dataset_fingerprint(ds_req1, entry, vars1, processing)
    fp2 = dataset_fingerprint(ds_req2, entry, vars2, processing)
    assert fp1 != fp2


def test_dataset_fingerprint_changes_on_batch_size():
    """Different batch_size produces different hash."""
    from hydro_param.config import DatasetRequest

    ds_req = DatasetRequest(name="dem", variables=["elevation"])
    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com",
        collection="3dep",
    )
    var_specs: list[VariableSpec | DerivedVariableSpec] = [VariableSpec(name="elevation", band=1)]

    fp1 = dataset_fingerprint(ds_req, entry, var_specs, ProcessingConfig(batch_size=500))
    fp2 = dataset_fingerprint(ds_req, entry, var_specs, ProcessingConfig(batch_size=100))
    assert fp1 != fp2


def test_dataset_fingerprint_changes_on_source_override():
    """Different source_override produces different hash."""
    from hydro_param.config import DatasetRequest

    ds_req = DatasetRequest(name="polaris", variables=["sand"])
    entry = DatasetEntry(strategy="local_tiff")
    processing = ProcessingConfig()

    vars1: list[VariableSpec | DerivedVariableSpec] = [
        VariableSpec(name="sand", band=1, source_override="http://example.com/v1/sand.vrt")
    ]
    vars2: list[VariableSpec | DerivedVariableSpec] = [
        VariableSpec(name="sand", band=1, source_override="http://example.com/v2/sand.vrt")
    ]

    fp1 = dataset_fingerprint(ds_req, entry, vars1, processing)
    fp2 = dataset_fingerprint(ds_req, entry, vars2, processing)
    assert fp1 != fp2


def test_dataset_fingerprint_changes_on_gsd():
    """Different gsd produces different hash."""
    from hydro_param.config import DatasetRequest

    ds_req = DatasetRequest(name="dem", variables=["elevation"])
    var_specs: list[VariableSpec | DerivedVariableSpec] = [VariableSpec(name="elevation", band=1)]
    processing = ProcessingConfig()

    entry1 = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com",
        collection="3dep",
        gsd=10,
    )
    entry2 = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com",
        collection="3dep",
        gsd=30,
    )

    fp1 = dataset_fingerprint(ds_req, entry1, var_specs, processing)
    fp2 = dataset_fingerprint(ds_req, entry2, var_specs, processing)
    assert fp1 != fp2


# ---------------------------------------------------------------------------
# PipelineManifest save/load
# ---------------------------------------------------------------------------


def test_manifest_save_load_roundtrip(tmp_path: Path):
    """Write + read back preserves all fields."""
    manifest = PipelineManifest(
        fabric_fingerprint="test.gpkg|1234.0|5678",
        entries={
            "dem": ManifestEntry(
                fingerprint="sha256:abc123",
                static_files={"elevation": "topo/elevation.csv"},
                completed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
        },
    )

    manifest.save(tmp_path)
    loaded = load_manifest(tmp_path)

    assert loaded is not None
    assert loaded.fabric_fingerprint == "test.gpkg|1234.0|5678"
    assert "dem" in loaded.entries
    assert loaded.entries["dem"].fingerprint == "sha256:abc123"
    assert loaded.entries["dem"].static_files == {"elevation": "topo/elevation.csv"}


def test_manifest_load_missing(tmp_path: Path):
    """Returns None when no manifest exists."""
    assert load_manifest(tmp_path) is None


def test_manifest_load_corrupt(tmp_path: Path):
    """Returns None for invalid YAML."""
    manifest_path = tmp_path / ".manifest.yml"
    manifest_path.write_text(": : : invalid yaml [[[")
    assert load_manifest(tmp_path) is None


def test_manifest_load_invalid_schema(tmp_path: Path):
    """Returns None for valid YAML with wrong schema."""
    manifest_path = tmp_path / ".manifest.yml"
    manifest_path.write_text("version: 2\nentries: wrong_type\n")
    assert load_manifest(tmp_path) is None


def test_manifest_load_unsupported_version(tmp_path: Path):
    """Returns None for manifest with unsupported version."""
    manifest_path = tmp_path / ".manifest.yml"
    manifest_path.write_text("version: 99\nfabric_fingerprint: test\nentries: {}\n")
    assert load_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# is_dataset_current / is_fabric_current
# ---------------------------------------------------------------------------


def test_is_fabric_current_matches():
    """is_fabric_current returns True when fingerprint matches."""
    manifest = PipelineManifest(fabric_fingerprint="test.gpkg|1234.0|5678")
    assert manifest.is_fabric_current("test.gpkg|1234.0|5678")


def test_is_fabric_current_mismatch():
    """is_fabric_current returns False when fingerprint differs."""
    manifest = PipelineManifest(fabric_fingerprint="test.gpkg|1234.0|5678")
    assert not manifest.is_fabric_current("test.gpkg|9999.0|9999")


def test_is_dataset_current_valid(tmp_path: Path):
    """Returns True when fingerprint matches and files exist."""
    # Create the output file
    topo_dir = tmp_path / "topo"
    topo_dir.mkdir()
    (topo_dir / "elevation.csv").write_text("data")

    manifest = PipelineManifest(
        fabric_fingerprint="test.gpkg|1234.0|5678",
        entries={
            "dem": ManifestEntry(
                fingerprint="sha256:abc123",
                static_files={"elevation": "topo/elevation.csv"},
            ),
        },
    )

    assert manifest.is_dataset_current("dem", "sha256:abc123", tmp_path)


def test_is_dataset_current_stale_fingerprint(tmp_path: Path):
    """Returns False on fingerprint mismatch."""
    topo_dir = tmp_path / "topo"
    topo_dir.mkdir()
    (topo_dir / "elevation.csv").write_text("data")

    manifest = PipelineManifest(
        entries={
            "dem": ManifestEntry(
                fingerprint="sha256:abc123",
                static_files={"elevation": "topo/elevation.csv"},
            ),
        },
    )

    assert not manifest.is_dataset_current("dem", "sha256:different", tmp_path)


def test_is_dataset_current_missing_file(tmp_path: Path):
    """Returns False when output file is deleted."""
    manifest = PipelineManifest(
        entries={
            "dem": ManifestEntry(
                fingerprint="sha256:abc123",
                static_files={"elevation": "topo/elevation.csv"},
            ),
        },
    )

    # File doesn't exist on disk
    assert not manifest.is_dataset_current("dem", "sha256:abc123", tmp_path)


def test_is_dataset_current_missing_dataset(tmp_path: Path):
    """Returns False when dataset is not in manifest."""
    manifest = PipelineManifest()
    assert not manifest.is_dataset_current("dem", "sha256:abc123", tmp_path)


def test_is_dataset_current_temporal_files(tmp_path: Path):
    """Returns True when temporal files exist and fingerprint matches."""
    climate_dir = tmp_path / "climate"
    climate_dir.mkdir()
    (climate_dir / "gridmet_2020_temporal.nc").write_text("data")

    manifest = PipelineManifest(
        entries={
            "gridmet": ManifestEntry(
                fingerprint="sha256:def456",
                temporal_files={"gridmet_2020": "climate/gridmet_2020_temporal.nc"},
            ),
        },
    )

    assert manifest.is_dataset_current("gridmet", "sha256:def456", tmp_path)


# ---------------------------------------------------------------------------
# make_manifest_entry
# ---------------------------------------------------------------------------


def test_make_manifest_entry_relative_paths(tmp_path: Path):
    """Paths in manifest entry are relative to output_dir."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    static_files = {"elevation": output_dir / "topo" / "elevation.csv"}
    entry = make_manifest_entry("sha256:abc", static_files, {}, output_dir)

    assert entry.static_files["elevation"] == "topo/elevation.csv"
    assert entry.fingerprint == "sha256:abc"
    assert entry.completed_at > datetime.min.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# SIRManifestEntry and version 2
# ---------------------------------------------------------------------------


class TestSIRManifestEntry:
    """Test the SIR section of the pipeline manifest."""

    def test_sir_manifest_entry_defaults(self):
        from hydro_param.manifest import SIRManifestEntry

        entry = SIRManifestEntry()
        assert entry.static_files == {}
        assert entry.temporal_files == {}
        assert entry.sir_schema == []

    def test_sir_manifest_entry_roundtrip(self):
        from hydro_param.manifest import SIRManifestEntry

        entry = SIRManifestEntry(
            static_files={"elevation_m_mean": "sir/elevation_m_mean.csv"},
            temporal_files={"gridmet_2020": "sir/gridmet_2020.nc"},
            sir_schema=[
                {
                    "name": "elevation_m_mean",
                    "units": "m",
                    "statistic": "mean",
                    "source_dataset": "dem_3dep_10m",
                }
            ],
        )
        data = entry.model_dump(mode="json")
        restored = SIRManifestEntry(**data)
        assert restored.static_files == entry.static_files
        assert restored.temporal_files == entry.temporal_files
        assert restored.sir_schema == entry.sir_schema

    def test_manifest_version_2_with_sir(self, tmp_path):
        from hydro_param.manifest import SIRManifestEntry

        sir = SIRManifestEntry(
            static_files={"elevation_m_mean": "sir/elevation_m_mean.csv"},
        )
        manifest = PipelineManifest(version=2, sir=sir)
        manifest.save(tmp_path)

        loaded = load_manifest(tmp_path)
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.sir is not None
        assert loaded.sir.static_files == {"elevation_m_mean": "sir/elevation_m_mean.csv"}

    def test_manifest_version_1_rejected(self, tmp_path):
        """Version 1 manifests are no longer supported."""
        manifest_path = tmp_path / ".manifest.yml"
        manifest_path.write_text("version: 1\nfabric_fingerprint: abc\nentries: {}\n")
        loaded = load_manifest(tmp_path)
        assert loaded is None  # load_manifest returns None on validation error

    def test_manifest_atomic_write(self, tmp_path):
        """Manifest save should be atomic (no partial writes)."""
        from hydro_param.manifest import SIRManifestEntry

        sir = SIRManifestEntry(static_files={"a": "sir/a.csv"})
        manifest = PipelineManifest(sir=sir)
        manifest.save(tmp_path)
        # File should exist and be valid
        assert load_manifest(tmp_path) is not None
        # No .tmp file should remain
        assert not (tmp_path / ".manifest.yml.tmp").exists()

    def test_manifest_atomic_write_cleanup_on_failure(self, tmp_path):
        """If rename fails, tmp file should be cleaned up."""
        from unittest.mock import patch

        from hydro_param.manifest import SIRManifestEntry

        sir = SIRManifestEntry(static_files={"a": "sir/a.csv"})
        manifest = PipelineManifest(sir=sir)
        # First, save a good manifest
        manifest.save(tmp_path)
        original = load_manifest(tmp_path)
        assert original is not None

        # Now simulate a replace failure
        with patch("pathlib.Path.replace", side_effect=OSError("mock replace error")):
            with pytest.raises(OSError, match="mock replace error"):
                manifest.save(tmp_path)

        # .tmp file should be cleaned up
        assert not (tmp_path / ".manifest.yml.tmp").exists()
        # Original manifest should be preserved
        assert load_manifest(tmp_path) is not None
