"""Tests for the project scaffolding module."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hydro_param.project import (
    DEFAULT_CATEGORIES,
    MARKER_FILE,
    find_project_root,
    generate_gitignore,
    generate_pipeline_template,
    get_data_categories,
    init_project,
)

# ---------------------------------------------------------------------------
# find_project_root
# ---------------------------------------------------------------------------


class TestFindProjectRoot:
    def test_finds_marker_in_current_dir(self, tmp_path: Path):
        (tmp_path / MARKER_FILE).write_text("name: test\n")
        assert find_project_root(start=tmp_path) == tmp_path

    def test_finds_marker_in_parent(self, tmp_path: Path):
        (tmp_path / MARKER_FILE).write_text("name: test\n")
        child = tmp_path / "data" / "topography"
        child.mkdir(parents=True)
        assert find_project_root(start=child) == tmp_path

    def test_returns_none_when_no_marker(self, tmp_path: Path):
        assert find_project_root(start=tmp_path) is None

    def test_stops_at_filesystem_root(self):
        assert find_project_root(start=Path("/")) is None


# ---------------------------------------------------------------------------
# get_data_categories
# ---------------------------------------------------------------------------


class TestGetDataCategories:
    def test_returns_defaults_when_no_registry(self):
        assert get_data_categories(None) == DEFAULT_CATEGORIES

    def test_returns_defaults_for_missing_path(self, tmp_path: Path):
        assert get_data_categories(tmp_path / "nonexistent") == DEFAULT_CATEGORIES

    def test_merges_registry_categories(self, tmp_path: Path):
        reg_file = tmp_path / "datasets.yml"
        raw = {
            "datasets": {
                "test_ds": {
                    "description": "Test",
                    "strategy": "stac_cog",
                    "catalog_url": "https://example.com",
                    "collection": "test",
                    "category": "topography",
                },
            },
        }
        reg_file.write_text(yaml.dump(raw))
        result = get_data_categories(reg_file)
        assert "topography" in result
        assert "climate" in result  # from defaults


# ---------------------------------------------------------------------------
# init_project
# ---------------------------------------------------------------------------


class TestInitProject:
    def test_creates_full_structure(self, tmp_path: Path):
        project = tmp_path / "my_project"
        init_project(project)

        # Marker file
        assert (project / MARKER_FILE).is_file()
        marker = yaml.safe_load((project / MARKER_FILE).read_text())
        assert marker["name"] == "my_project"
        assert "created" in marker

        # Config template
        assert (project / "configs" / "pipeline.yml").is_file()

        # Data directories
        assert (project / "data" / "fabrics").is_dir()
        for cat in DEFAULT_CATEGORIES:
            assert (project / "data" / cat).is_dir()

        # Output and models
        assert (project / "output").is_dir()
        assert (project / "models").is_dir()

        # Gitignore
        assert (project / ".gitignore").is_file()

    def test_refuses_existing_project(self, tmp_path: Path):
        project = tmp_path / "existing"
        project.mkdir()
        (project / MARKER_FILE).write_text("name: existing\n")
        with pytest.raises(SystemExit):
            init_project(project)

    def test_force_reinitialises(self, tmp_path: Path):
        project = tmp_path / "existing"
        init_project(project)
        # Modify pipeline.yml
        config = project / "configs" / "pipeline.yml"
        config.write_text("custom: true\n")
        # Re-init with force
        init_project(project, force=True)
        # Marker refreshed
        assert (project / MARKER_FILE).is_file()
        # pipeline.yml NOT overwritten
        assert config.read_text() == "custom: true\n"

    def test_init_current_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        project = tmp_path / "workspace"
        project.mkdir()
        monkeypatch.chdir(project)
        init_project(Path("."))
        assert (project / MARKER_FILE).is_file()


# ---------------------------------------------------------------------------
# generate_pipeline_template
# ---------------------------------------------------------------------------


class TestGeneratePipelineTemplate:
    def test_template_is_valid_yaml(self):
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        assert parsed is not None
        assert "target_fabric" in parsed
        assert "datasets" in parsed
        assert "output" in parsed
        assert "processing" in parsed

    def test_template_references_data_dirs(self):
        content = generate_pipeline_template("test_project")
        assert "data/fabrics/" in content

    def test_template_uses_project_name(self):
        content = generate_pipeline_template("my_watershed")
        parsed = yaml.safe_load(content)
        assert parsed["output"]["sir_name"] == "my_watershed"

    def test_template_includes_all_dataset_strategies(self):
        """Template covers all 5 access strategies via 7 dataset entries."""
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        dataset_names = [d["name"] for d in parsed["datasets"]]
        expected = [
            "dem_3dep_10m",  # stac_cog
            "gnatsgo_rasters",  # stac_cog
            "polaris_30m",  # local_tiff
            "nlcd_osn_lndcov",  # nhgf_stac static
            "nlcd_osn_fctimp",  # nhgf_stac static
            "snodas",  # nhgf_stac temporal
            "gridmet",  # climr_cat
        ]
        assert dataset_names == expected

    def test_template_temporal_datasets_have_time_period(self):
        """SNODAS and gridMET entries include time_period."""
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        datasets_by_name = {d["name"]: d for d in parsed["datasets"]}
        assert "time_period" in datasets_by_name["snodas"]
        assert "time_period" in datasets_by_name["gridmet"]

    def test_template_nlcd_has_year(self):
        """NLCD entries include year field."""
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        datasets_by_name = {d["name"]: d for d in parsed["datasets"]}
        assert "year" in datasets_by_name["nlcd_osn_lndcov"]
        assert "year" in datasets_by_name["nlcd_osn_fctimp"]

    def test_template_enables_resume_by_default(self):
        """Generated template includes resume: true in processing section."""
        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        assert parsed["processing"]["resume"] is True

    def test_template_conforms_to_pipeline_config_schema(self):
        """Generated template must parse against PipelineConfig without error."""
        from hydro_param.config import PipelineConfig

        content = generate_pipeline_template("test_project")
        parsed = yaml.safe_load(content)
        config = PipelineConfig(**parsed)
        assert config.processing.engine == "exactextract"
        assert config.processing.batch_size == 500
        assert config.processing.resume is True
        assert config.output.format == "netcdf"


# ---------------------------------------------------------------------------
# generate_gitignore
# ---------------------------------------------------------------------------


class TestGenerateGitignore:
    def test_ignores_data_and_output(self):
        content = generate_gitignore()
        assert "output/" in content
        assert "models/" in content
        assert "data/topography/" in content
