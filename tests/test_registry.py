"""Tests for dataset registry loading and resolution."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hydro_param.dataset_registry import (
    DatasetEntry,
    DatasetRegistry,
    DerivedVariableSpec,
    DownloadInfo,
    VariableSpec,
    load_registry,
)
from hydro_param.pipeline import DEFAULT_REGISTRY


@pytest.fixture()
def registry_yaml(tmp_path: Path) -> Path:
    """Create a minimal registry YAML file for testing."""
    raw = {
        "datasets": {
            "dem_test": {
                "description": "Test DEM",
                "strategy": "stac_cog",
                "catalog_url": "https://example.com/stac/v1",
                "collection": "3dep-seamless",
                "gsd": 10,
                "crs": "EPSG:4326",
                "sign": "planetary_computer",
                "category": "topography",
                "variables": [
                    {
                        "name": "elevation",
                        "band": 1,
                        "units": "m",
                        "long_name": "Surface elevation",
                        "categorical": False,
                    },
                ],
                "derived_variables": [
                    {
                        "name": "slope",
                        "source": "elevation",
                        "method": "horn",
                        "units": "degrees",
                    },
                    {
                        "name": "aspect",
                        "source": "elevation",
                        "method": "horn",
                        "units": "degrees",
                    },
                ],
            },
            "nlcd_test": {
                "strategy": "local_tiff",
                "crs": "EPSG:5070",
                "category": "land_cover",
                "download": {
                    "url": "s3://usgs-landcover/nlcd_2021.tif",
                    "size_gb": 1.5,
                    "format": "COG",
                    "notes": "aws s3 cp --no-sign-request <url> .",
                },
                "variables": [
                    {
                        "name": "land_cover",
                        "band": 1,
                        "categorical": True,
                    },
                ],
            },
            "nlcd_with_source": {
                "strategy": "local_tiff",
                "source": "data/nlcd.tif",
                "crs": "EPSG:5070",
                "category": "land_cover",
                "variables": [
                    {
                        "name": "land_cover",
                        "band": 1,
                        "categorical": True,
                    },
                ],
            },
        }
    }
    path = tmp_path / "datasets.yml"
    path.write_text(yaml.dump(raw))
    return path


def test_load_registry(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    assert isinstance(registry, DatasetRegistry)
    assert "dem_test" in registry.datasets
    assert "nlcd_test" in registry.datasets


def test_registry_get_known_dataset(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    entry = registry.get("dem_test")
    assert isinstance(entry, DatasetEntry)
    assert entry.strategy == "stac_cog"
    assert entry.collection == "3dep-seamless"
    assert entry.gsd == 10


def test_registry_get_unknown_dataset(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="not found in registry"):
        registry.get("nonexistent")


def test_registry_unknown_dataset_lists_available(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="dem_test"):
        registry.get("nonexistent")


def test_resolve_raw_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    var = registry.resolve_variable("dem_test", "elevation")
    assert isinstance(var, VariableSpec)
    assert var.name == "elevation"
    assert var.units == "m"
    assert var.categorical is False


def test_resolve_derived_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    var = registry.resolve_variable("dem_test", "slope")
    assert isinstance(var, DerivedVariableSpec)
    assert var.name == "slope"
    assert var.source == "elevation"
    assert var.method == "horn"


def test_resolve_unknown_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="not found in dataset"):
        registry.resolve_variable("dem_test", "nonexistent")


def test_resolve_unknown_variable_lists_available(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    with pytest.raises(KeyError, match="elevation"):
        registry.resolve_variable("dem_test", "nonexistent")


def test_categorical_variable(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    var = registry.resolve_variable("nlcd_test", "land_cover")
    assert isinstance(var, VariableSpec)
    assert var.categorical is True


def test_stac_cog_entry_fields(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    entry = registry.get("dem_test")
    assert entry.catalog_url == "https://example.com/stac/v1"
    assert entry.sign == "planetary_computer"
    assert entry.asset_key == "data"


def test_local_tiff_entry_with_download(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    entry = registry.get("nlcd_test")
    assert entry.source is None
    assert entry.download is not None
    assert entry.download.url == "s3://usgs-landcover/nlcd_2021.tif"
    assert entry.download.size_gb == 1.5
    assert entry.download.format == "COG"
    assert entry.crs == "EPSG:5070"


def test_local_tiff_entry_with_source(registry_yaml: Path):
    registry = load_registry(registry_yaml)
    entry = registry.get("nlcd_with_source")
    assert entry.source == "data/nlcd.tif"
    assert entry.download is None
    assert entry.crs == "EPSG:5070"


def test_stac_cog_requires_catalog_url_and_collection():
    with pytest.raises(ValidationError, match="stac_cog strategy requires"):
        DatasetEntry(strategy="stac_cog")


def test_local_tiff_valid_without_source():
    """local_tiff strategy no longer requires source at schema level."""
    entry = DatasetEntry(strategy="local_tiff")
    assert entry.source is None
    assert entry.download is None


def test_coordinate_defaults():
    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com/stac/v1",
        collection="test",
    )
    assert entry.x_coord == "x"
    assert entry.y_coord == "y"
    assert entry.t_coord is None


def test_custom_coordinate_names():
    entry = DatasetEntry(
        strategy="converted_zarr",
        source="s3://bucket/data.zarr",
        x_coord="lon",
        y_coord="lat",
    )
    assert entry.x_coord == "lon"
    assert entry.y_coord == "lat"


def test_temporal_requires_t_coord():
    with pytest.raises(ValidationError, match="Temporal datasets require 't_coord'"):
        DatasetEntry(
            strategy="native_zarr",
            source="s3://bucket/data.zarr",
            temporal=True,
        )


def test_temporal_with_t_coord_valid():
    entry = DatasetEntry(
        strategy="native_zarr",
        source="s3://bucket/data.zarr",
        temporal=True,
        t_coord="time",
        time_step="daily",
    )
    assert entry.t_coord == "time"
    assert entry.temporal is True


def test_download_info_model():
    info = DownloadInfo(url="s3://bucket/file.tif", size_gb=2.0, format="COG", notes="Use aws cli")
    assert info.url == "s3://bucket/file.tif"
    assert info.size_gb == 2.0
    assert info.format == "COG"
    assert info.notes == "Use aws cli"


def test_download_info_defaults():
    info = DownloadInfo(url="s3://bucket/file.tif")
    assert info.size_gb is None
    assert info.format == ""
    assert info.notes == ""


def test_download_info_requires_url_or_files_or_template():
    """DownloadInfo with neither url, files, nor url_template is rejected."""
    with pytest.raises(ValidationError, match="requires at least"):
        DownloadInfo()


def test_download_file_model():
    from hydro_param.dataset_registry import DownloadFile

    f = DownloadFile(year=2021, variable="land_cover", url="s3://bucket/lc.tif", size_gb=1.5)
    assert f.year == 2021
    assert f.variable == "land_cover"
    assert f.url == "s3://bucket/lc.tif"
    assert f.size_gb == 1.5


def test_download_info_with_files():
    from hydro_param.dataset_registry import DownloadFile

    info = DownloadInfo(
        files=[
            DownloadFile(year=2021, variable="land_cover", url="s3://bucket/lc_2021.tif"),
            DownloadFile(year=2019, variable="land_cover", url="s3://bucket/lc_2019.tif"),
        ]
    )
    assert len(info.files) == 2
    assert info.url == ""  # No single URL for multi-file


def test_local_tiff_with_download_block():
    entry = DatasetEntry(
        strategy="local_tiff",
        download={"url": "s3://bucket/data.tif", "size_gb": 1.5},
    )
    assert entry.download is not None
    assert entry.download.url == "s3://bucket/data.tif"
    assert entry.download.size_gb == 1.5
    assert entry.source is None


def test_local_tiff_with_both_source_and_download():
    entry = DatasetEntry(
        strategy="local_tiff",
        source="data/local.tif",
        download={"url": "s3://bucket/data.tif"},
    )
    assert entry.source == "data/local.tif"
    assert entry.download is not None


def test_download_info_with_url_template():
    info = DownloadInfo(
        url_template="s3://bucket/{variable}_{year}.tif",
        year_range=[2020, 2022],
        variables_available=["lc", "imp"],
    )
    assert info.url_template == "s3://bucket/{variable}_{year}.tif"
    assert info.year_range == [2020, 2022]
    assert info.variables_available == ["lc", "imp"]


def test_download_info_url_template_requires_year_range():
    with pytest.raises(ValidationError, match="year_range"):
        DownloadInfo(
            url_template="s3://bucket/{variable}_{year}.tif",
            variables_available=["lc"],
        )


def test_download_info_url_template_requires_variables_available():
    with pytest.raises(ValidationError, match="variables_available"):
        DownloadInfo(
            url_template="s3://bucket/{variable}_{year}.tif",
            year_range=[2020, 2022],
        )


def test_download_info_url_template_invalid_year_range():
    with pytest.raises(ValidationError, match="year_range"):
        DownloadInfo(
            url_template="s3://bucket/{variable}_{year}.tif",
            year_range=[2022, 2020],  # start > end
            variables_available=["lc"],
        )


def test_expand_files_from_template():
    info = DownloadInfo(
        url_template="s3://bucket/{variable}_{year}.tif",
        year_range=[2020, 2022],
        variables_available=["lc", "imp"],
    )
    files = info.expand_files()
    assert len(files) == 6  # 3 years x 2 variables
    urls = [f.url for f in files]
    assert "s3://bucket/lc_2020.tif" in urls
    assert "s3://bucket/imp_2022.tif" in urls


def test_expand_files_with_year_filter():
    info = DownloadInfo(
        url_template="s3://bucket/{variable}_{year}.tif",
        year_range=[2020, 2022],
        variables_available=["lc", "imp"],
    )
    files = info.expand_files(years={2021})
    assert len(files) == 2  # 1 year x 2 variables
    assert all(f.year == 2021 for f in files)


def test_expand_files_with_variable_filter():
    info = DownloadInfo(
        url_template="s3://bucket/{variable}_{year}.tif",
        year_range=[2020, 2022],
        variables_available=["lc", "imp"],
    )
    files = info.expand_files(variables={"lc"})
    assert len(files) == 3  # 3 years x 1 variable
    assert all(f.variable == "lc" for f in files)


def test_expand_files_with_both_filters():
    info = DownloadInfo(
        url_template="s3://bucket/{variable}_{year}.tif",
        year_range=[2020, 2022],
        variables_available=["lc", "imp"],
    )
    files = info.expand_files(years={2021}, variables={"imp"})
    assert len(files) == 1
    assert files[0].year == 2021
    assert files[0].variable == "imp"
    assert files[0].url == "s3://bucket/imp_2021.tif"


def test_expand_files_from_explicit_files():
    from hydro_param.dataset_registry import DownloadFile

    info = DownloadInfo(
        files=[
            DownloadFile(year=2021, variable="lc", url="s3://bucket/lc_2021.tif"),
            DownloadFile(year=2019, variable="lc", url="s3://bucket/lc_2019.tif"),
            DownloadFile(year=2021, variable="imp", url="s3://bucket/imp_2021.tif"),
        ]
    )
    # No filter
    assert len(info.expand_files()) == 3
    # Year filter
    assert len(info.expand_files(years={2021})) == 2
    # Variable filter
    assert len(info.expand_files(variables={"lc"})) == 2
    # Both
    assert len(info.expand_files(years={2021}, variables={"lc"})) == 1


def test_requester_pays_default_false():
    info = DownloadInfo(url="s3://bucket/file.tif")
    assert info.requester_pays is False


# ---------------------------------------------------------------------------
# Directory-based registry loading
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry_dir(tmp_path: Path) -> Path:
    """Create a registry directory with two category files."""
    reg_dir = tmp_path / "datasets"
    reg_dir.mkdir()

    topo = {
        "datasets": {
            "dem_test": {
                "strategy": "stac_cog",
                "catalog_url": "https://example.com/stac/v1",
                "collection": "3dep-seamless",
                "category": "topography",
                "variables": [{"name": "elevation", "band": 1, "categorical": False}],
            },
        }
    }
    (reg_dir / "topography.yml").write_text(yaml.dump(topo))

    lc = {
        "datasets": {
            "nlcd_test": {
                "strategy": "local_tiff",
                "crs": "EPSG:5070",
                "category": "land_cover",
                "download": {"url": "s3://bucket/nlcd.tif"},
                "variables": [{"name": "land_cover", "band": 1, "categorical": True}],
            },
        }
    }
    (reg_dir / "land_cover.yml").write_text(yaml.dump(lc))
    return reg_dir


def test_load_registry_from_directory(registry_dir: Path):
    """Loading from a directory merges all category files."""
    registry = load_registry(registry_dir)
    assert "dem_test" in registry.datasets
    assert "nlcd_test" in registry.datasets
    assert len(registry.datasets) == 2


def test_load_registry_dir_duplicate_name_raises(tmp_path: Path):
    """Duplicate dataset name across files raises ValueError."""
    reg_dir = tmp_path / "datasets"
    reg_dir.mkdir()

    file_a = {
        "datasets": {
            "dup_name": {
                "strategy": "local_tiff",
                "variables": [{"name": "v", "band": 1, "categorical": False}],
            }
        }
    }
    (reg_dir / "a.yml").write_text(yaml.dump(file_a))
    (reg_dir / "b.yml").write_text(yaml.dump(file_a))

    with pytest.raises(ValueError, match="Duplicate dataset name 'dup_name'"):
        load_registry(reg_dir)


def test_load_registry_dir_empty_raises(tmp_path: Path):
    """Directory with no YAML files raises FileNotFoundError."""
    reg_dir = tmp_path / "empty"
    reg_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No YAML files"):
        load_registry(reg_dir)


def test_load_registry_dir_no_datasets_raises(tmp_path: Path):
    """Directory with YAML files but no datasets raises FileNotFoundError."""
    reg_dir = tmp_path / "datasets"
    reg_dir.mkdir()
    (reg_dir / "empty.yml").write_text("datasets: {}")
    with pytest.raises(FileNotFoundError, match="No datasets found"):
        load_registry(reg_dir)


def test_load_registry_dir_skips_non_registry_files(registry_dir: Path):
    """Files without a datasets key are silently skipped."""
    (registry_dir / "readme.yml").write_text("notes: just a note")
    registry = load_registry(registry_dir)
    assert len(registry.datasets) == 2


def test_load_registry_nonexistent_path_raises(tmp_path: Path):
    """Non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_registry(tmp_path / "nonexistent")


def test_load_registry_file_still_works(registry_yaml: Path):
    """Single file mode continues to work via auto-detection."""
    registry = load_registry(registry_yaml)
    assert "dem_test" in registry.datasets


# ---------------------------------------------------------------------------
# Real registry integration tests
# ---------------------------------------------------------------------------


def test_load_real_registry():
    """Test loading the bundled dataset registry directory."""
    registry = load_registry(DEFAULT_REGISTRY)
    assert "dem_3dep_10m" in registry.datasets
    assert "polaris_30m" in registry.datasets
    assert "gnatsgo_rasters" in registry.datasets
    assert "nlcd_legacy" in registry.datasets
    assert "nlcd_annual" in registry.datasets
    assert "gridmet" in registry.datasets
    assert "snodas" in registry.datasets


def test_real_registry_nlcd_legacy_has_download():
    """Verify NLCD legacy entry in real registry has multi-file download block."""
    registry = load_registry(DEFAULT_REGISTRY)
    nlcd = registry.get("nlcd_legacy")
    assert nlcd.strategy == "local_tiff"
    assert nlcd.source is None
    assert nlcd.download is not None
    assert len(nlcd.download.files) > 0
    # Verify multi-file structure
    years = {f.year for f in nlcd.download.files}
    variables = {f.variable for f in nlcd.download.files}
    assert 2021 in years
    assert "land_cover" in variables
    assert "impervious" in variables


def test_real_registry_nlcd_annual_has_template():
    """Verify NLCD annual entry in real registry has template download."""
    registry = load_registry(DEFAULT_REGISTRY)
    nlcd = registry.get("nlcd_annual")
    assert nlcd.strategy == "local_tiff"
    assert nlcd.download is not None
    assert nlcd.download.url_template != ""
    assert nlcd.download.year_range == [1985, 2024]
    assert "LndCov" in nlcd.download.variables_available
    assert nlcd.download.requester_pays is True
    # Verify expand_files works
    files = nlcd.download.expand_files(years={2020}, variables={"LndCov"})
    assert len(files) == 1
    assert "2020" in files[0].url
    assert "LndCov" in files[0].url


def test_real_registry_gridmet():
    """Verify gridMET entry in real registry uses climr_cat strategy."""
    registry = load_registry(DEFAULT_REGISTRY)
    gridmet = registry.get("gridmet")
    assert gridmet.strategy == "climr_cat"
    assert gridmet.catalog_id == "gridmet"
    assert gridmet.temporal is True
    assert gridmet.t_coord == "day"
    var_names = [v.name for v in gridmet.variables]
    assert "pr" in var_names
    assert "tmmx" in var_names
    assert "tmmn" in var_names


def test_real_registry_snodas():
    """Verify SNODAS entry in real registry uses nhgf_stac strategy."""
    registry = load_registry(DEFAULT_REGISTRY)
    snodas = registry.get("snodas")
    assert snodas.strategy == "nhgf_stac"
    assert snodas.collection == "snodas"
    assert snodas.temporal is True
    assert snodas.t_coord == "time"
    var_names = [v.name for v in snodas.variables]
    assert "SWE" in var_names
    assert "SDP" in var_names


def test_real_registry_gnatsgo():
    """Verify gNATSGO entry with per-variable asset keys."""
    registry = load_registry(DEFAULT_REGISTRY)
    gnatsgo = registry.get("gnatsgo_rasters")
    assert gnatsgo.strategy == "stac_cog"
    assert gnatsgo.collection == "gnatsgo-rasters"
    # Verify per-variable asset_key
    aws_var = registry.resolve_variable("gnatsgo_rasters", "aws0_100")
    assert isinstance(aws_var, VariableSpec)
    assert aws_var.asset_key == "aws0_100"


def test_real_registry_polaris_30m():
    """Verify fixed POLARIS entry has correct metadata."""
    registry = load_registry(DEFAULT_REGISTRY)
    polaris = registry.get("polaris_30m")
    assert polaris.strategy == "local_tiff"
    assert polaris.crs == "EPSG:4326"
    assert polaris.download is not None
    assert len(polaris.download.files) > 0
    var_names = [v.name for v in polaris.variables]
    assert "sand" in var_names
    assert "clay" in var_names
    assert "ksat" in var_names
    assert "theta_s" in var_names


# ---------------------------------------------------------------------------
# climr_cat and nhgf_stac strategy validation
# ---------------------------------------------------------------------------


def test_climr_cat_valid():
    """climr_cat strategy with required fields validates."""
    entry = DatasetEntry(
        strategy="climr_cat",
        catalog_id="gridmet",
        temporal=True,
        t_coord="day",
        time_step="daily",
    )
    assert entry.strategy == "climr_cat"
    assert entry.catalog_id == "gridmet"


def test_climr_cat_requires_catalog_id():
    """climr_cat without catalog_id is rejected."""
    with pytest.raises(ValidationError, match="catalog_id"):
        DatasetEntry(
            strategy="climr_cat",
            temporal=True,
            t_coord="day",
            time_step="daily",
        )


def test_climr_cat_requires_temporal():
    """climr_cat with temporal=false is rejected."""
    with pytest.raises(ValidationError, match="temporal"):
        DatasetEntry(
            strategy="climr_cat",
            catalog_id="gridmet",
            temporal=False,
        )


def test_temporal_requires_native_name_on_variables():
    """Temporal datasets with variables must have native_name set."""
    with pytest.raises(ValidationError, match="native_name"):
        DatasetEntry(
            strategy="climr_cat",
            catalog_id="gridmet",
            temporal=True,
            t_coord="day",
            time_step="daily",
            variables=[
                VariableSpec(name="pr", units="mm"),  # missing native_name
            ],
        )


def test_temporal_native_name_ok_when_set():
    """Temporal datasets with native_name set on all variables pass validation."""
    entry = DatasetEntry(
        strategy="climr_cat",
        catalog_id="gridmet",
        temporal=True,
        t_coord="day",
        time_step="daily",
        variables=[
            VariableSpec(name="pr", units="mm", native_name="precipitation_amount"),
        ],
    )
    assert entry.variables[0].native_name == "precipitation_amount"


def test_nhgf_stac_valid():
    """nhgf_stac strategy with required fields validates."""
    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
        time_step="daily",
    )
    assert entry.strategy == "nhgf_stac"
    assert entry.collection == "snodas"


def test_nhgf_stac_requires_collection():
    """nhgf_stac without collection is rejected."""
    with pytest.raises(ValidationError, match="collection"):
        DatasetEntry(
            strategy="nhgf_stac",
            temporal=True,
            t_coord="time",
            time_step="daily",
        )


def test_nhgf_stac_static_valid():
    """nhgf_stac with temporal=false (static dataset like NLCD) validates OK."""
    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="nlcd-LndCov",
        temporal=False,
    )
    assert entry.strategy == "nhgf_stac"
    assert entry.collection == "nlcd-LndCov"
    assert entry.temporal is False


def test_nhgf_stac_temporal_still_valid():
    """Temporal nhgf_stac (e.g. SNODAS) still validates as before."""
    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
        time_step="daily",
    )
    assert entry.temporal is True
    assert entry.t_coord == "time"


def test_real_registry_nlcd_osn_entries():
    """Verify all 6 NLCD OSN entries in the real registry."""
    registry = load_registry(DEFAULT_REGISTRY)

    expected = {
        "nlcd_osn_lndcov": ("nlcd-LndCov", "LndCov", True),
        "nlcd_osn_fctimp": ("nlcd-FctImp", "FctImp", False),
        "nlcd_osn_impdsc": ("nlcd-ImpDsc", "ImpDsc", True),
        "nlcd_osn_lndchg": ("nlcd-LndChg", "LndChg", True),
        "nlcd_osn_lndcnf": ("nlcd-LndCnf", "LndCnf", False),
        "nlcd_osn_spcchg": ("nlcd-SpcChg", "SpcChg", False),
    }
    for name, (collection, var_name, categorical) in expected.items():
        entry = registry.get(name)
        assert entry.strategy == "nhgf_stac"
        assert entry.collection == collection
        assert entry.temporal is False
        assert entry.year_range == [1985, 2024]
        assert entry.crs == "EPSG:5070"
        assert entry.category == "land_cover"
        assert len(entry.variables) == 1
        assert entry.variables[0].name == var_name
        assert entry.variables[0].categorical is categorical


# ---------------------------------------------------------------------------
# year_range field tests
# ---------------------------------------------------------------------------


def test_dataset_entry_year_range_default_none():
    """year_range defaults to None."""
    entry = DatasetEntry(strategy="local_tiff")
    assert entry.year_range is None


def test_dataset_entry_year_range_valid():
    """year_range accepts a valid 2-element list."""
    entry = DatasetEntry(strategy="local_tiff", year_range=[1985, 2024])
    assert entry.year_range == [1985, 2024]


def test_dataset_entry_year_range_invalid_order():
    """year_range rejects start > end."""
    with pytest.raises(ValidationError, match="year_range start must be <= end"):
        DatasetEntry(strategy="local_tiff", year_range=[2024, 1985])


class TestVariableSpecScaleFactor:
    """Tests for VariableSpec.scale_factor field."""

    def test_scale_factor_defaults_to_none(self) -> None:
        v = VariableSpec(name="elev", band=1)
        assert v.scale_factor is None

    def test_scale_factor_accepts_float(self) -> None:
        v = VariableSpec(name="slope", band=1, scale_factor=0.01)
        assert v.scale_factor == 0.01

    def test_scale_factor_in_dict_output(self) -> None:
        v = VariableSpec(name="slope", band=1, scale_factor=0.01)
        d = v.model_dump()
        assert d["scale_factor"] == 0.01

    def test_scale_factor_none_excluded_from_yaml(self) -> None:
        v = VariableSpec(name="elev", band=1)
        d = v.model_dump(exclude_none=True)
        assert "scale_factor" not in d


def test_dataset_entry_year_range_wrong_length():
    """year_range rejects lists that are not exactly 2 elements."""
    with pytest.raises(ValidationError, match="2-element list"):
        DatasetEntry(strategy="local_tiff", year_range=[1985])


def test_variable_spec_asset_key():
    """VariableSpec accepts optional asset_key override."""
    var = VariableSpec(name="aws0_100", band=1, asset_key="aws0_100")
    assert var.asset_key == "aws0_100"

    var_default = VariableSpec(name="elevation", band=1)
    assert var_default.asset_key is None


def test_variable_spec_source_override_field():
    """VariableSpec accepts optional source_override URL."""
    url = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0/vrt/sand_mean_0_5.vrt"
    var = VariableSpec(name="sand", band=1, source_override=url)
    assert var.source_override == url


def test_variable_spec_source_override_default_none():
    """VariableSpec.source_override defaults to None."""
    var = VariableSpec(name="elevation", band=1)
    assert var.source_override is None


def test_real_registry_polaris_has_variable_sources():
    """Verify key POLARIS variables have per-variable source_override URLs."""
    registry = load_registry(DEFAULT_REGISTRY)
    polaris = registry.get("polaris_30m")

    # These variables should have source_override URLs for remote VRT access
    expected_with_source = {"sand", "silt", "clay", "theta_s", "ksat", "bd"}
    for var in polaris.variables:
        if var.name in expected_with_source:
            assert var.source_override is not None, f"{var.name} should have a source URL"
            assert var.source_override.startswith("http://"), (
                f"{var.name} source should be HTTP URL"
            )
        elif var.name in {"theta_r", "ph", "om", "lambda", "hb", "n", "alpha"}:
            assert var.source_override is None, f"{var.name} should not have a source URL"


def test_default_registry_resolves_to_existing_directory():
    """DEFAULT_REGISTRY must resolve to the bundled registry directory."""
    assert DEFAULT_REGISTRY.is_dir(), f"DEFAULT_REGISTRY does not exist: {DEFAULT_REGISTRY}"
    yamls = sorted(p.name for p in DEFAULT_REGISTRY.glob("*.yml"))
    expected = [
        "climate.yml",
        "geology.yml",
        "hydrography.yml",
        "land_cover.yml",
        "snow.yml",
        "soils.yml",
        "topography.yml",
        "water_bodies.yml",
    ]
    assert yamls == expected


def test_default_registry_is_absolute():
    """DEFAULT_REGISTRY must be an absolute path to work regardless of CWD."""
    assert DEFAULT_REGISTRY.is_absolute()


# ---------------------------------------------------------------------------
# time_step field
# ---------------------------------------------------------------------------


def test_time_step_required_for_temporal():
    """Temporal datasets must specify time_step."""
    with pytest.raises(ValueError, match="time_step"):
        DatasetEntry(
            strategy="nhgf_stac",
            collection="snodas",
            temporal=True,
            t_coord="time",
            time_step=None,
            variables=[VariableSpec(name="SWE", band=1, native_name="SWE")],
        )


def test_time_step_accepted_for_temporal():
    """Temporal datasets with time_step set pass validation."""
    entry = DatasetEntry(
        strategy="nhgf_stac",
        collection="snodas",
        temporal=True,
        t_coord="time",
        time_step="daily",
        variables=[VariableSpec(name="SWE", band=1, native_name="SWE")],
    )
    assert entry.time_step == "daily"


def test_time_step_none_for_static():
    """Static datasets can omit time_step (None is valid)."""
    entry = DatasetEntry(
        strategy="stac_cog",
        catalog_url="https://example.com",
        collection="test",
        temporal=False,
    )
    assert entry.time_step is None


def test_time_step_rejects_invalid():
    """time_step only accepts 'daily' or 'monthly'."""
    with pytest.raises(ValueError):
        DatasetEntry(
            strategy="nhgf_stac",
            collection="snodas",
            temporal=True,
            t_coord="time",
            time_step="hourly",
            variables=[VariableSpec(name="SWE", band=1, native_name="SWE")],
        )


def test_bundled_temporal_datasets_have_year_range_and_time_step():
    """All temporal datasets in the bundled registry have year_range and time_step."""
    registry = load_registry(DEFAULT_REGISTRY)
    for name, entry in registry.datasets.items():
        if entry.temporal:
            assert entry.year_range is not None, f"Temporal dataset '{name}' missing year_range"
            assert entry.time_step is not None, f"Temporal dataset '{name}' missing time_step"


class TestDerivedCategoricalSpec:
    """Tests for DerivedCategoricalSpec model."""

    def test_basic_creation(self) -> None:
        from hydro_param.dataset_registry import DerivedCategoricalSpec

        spec = DerivedCategoricalSpec(
            name="soil_texture",
            sources=["sand", "silt", "clay"],
            method="usda_texture_triangle",
            units="class",
            long_name="USDA soil texture classification",
        )
        assert spec.name == "soil_texture"
        assert spec.sources == ["sand", "silt", "clay"]
        assert spec.method == "usda_texture_triangle"

    def test_sources_must_have_at_least_two(self) -> None:
        from hydro_param.dataset_registry import DerivedCategoricalSpec

        with pytest.raises(ValidationError):
            DerivedCategoricalSpec(
                name="bad",
                sources=["single"],
                method="test",
            )


class TestDerivedCategoricalParsing:
    """Tests for parsing derived_categorical_variables from YAML."""

    def test_dataset_entry_has_derived_categorical(self) -> None:
        from hydro_param.dataset_registry import DerivedCategoricalSpec

        entry = DatasetEntry(
            strategy="local_tiff",
            variables=[],
            derived_categorical_variables=[
                DerivedCategoricalSpec(
                    name="soil_texture",
                    sources=["sand", "silt", "clay"],
                    method="usda_texture_triangle",
                )
            ],
        )
        assert len(entry.derived_categorical_variables) == 1
        assert entry.derived_categorical_variables[0].name == "soil_texture"

    def test_defaults_to_empty_list(self) -> None:
        entry = DatasetEntry(strategy="local_tiff", variables=[])
        assert entry.derived_categorical_variables == []

    def test_resolve_derived_categorical_variable(self) -> None:
        from hydro_param.dataset_registry import DerivedCategoricalSpec

        entry = DatasetEntry(
            strategy="local_tiff",
            variables=[
                VariableSpec(name="sand"),
                VariableSpec(name="silt"),
                VariableSpec(name="clay"),
            ],
            derived_categorical_variables=[
                DerivedCategoricalSpec(
                    name="soil_texture",
                    sources=["sand", "silt", "clay"],
                    method="usda_texture_triangle",
                )
            ],
        )
        registry = DatasetRegistry(datasets={"test": entry})
        spec = registry.resolve_variable("test", "soil_texture")
        assert isinstance(spec, DerivedCategoricalSpec)
        assert spec.sources == ["sand", "silt", "clay"]


# ---------------------------------------------------------------------------
# Registry overlay loading
# ---------------------------------------------------------------------------


class TestRegistryOverlay:
    """Tests for user-local registry overlay loading."""

    def test_load_registry_with_overlay_dir(self, tmp_path: Path) -> None:
        """Overlay entries appear in merged registry."""
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()
        overlay_yaml = {
            "datasets": {
                "gfv11_sand": {
                    "description": "GFv1.1 sand",
                    "strategy": "local_tiff",
                    "source": str(tmp_path / "Sand.tif"),
                    "crs": "EPSG:5070",
                    "category": "soils",
                    "variables": [
                        {"name": "sand_pct", "band": 1, "units": "%"},
                    ],
                }
            }
        }
        (overlay_dir / "gfv11.yml").write_text(yaml.dump(overlay_yaml))

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])
        assert "gfv11_sand" in registry.datasets
        assert registry.get("gfv11_sand").source == str(tmp_path / "Sand.tif")
        # Bundled entries still present
        assert "dem_3dep_10m" in registry.datasets

    def test_overlay_replaces_bundled_on_collision(self, tmp_path: Path) -> None:
        """User-local entry replaces bundled entry with same name."""
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()
        overlay_yaml = {
            "datasets": {
                "dem_3dep_10m": {
                    "description": "OVERRIDDEN DEM",
                    "strategy": "local_tiff",
                    "source": str(tmp_path / "dem.tif"),
                    "crs": "EPSG:5070",
                    "category": "topography",
                    "variables": [
                        {"name": "elevation", "band": 1, "units": "m"},
                    ],
                }
            }
        }
        (overlay_dir / "override.yml").write_text(yaml.dump(overlay_yaml))

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])
        assert registry.get("dem_3dep_10m").description == "OVERRIDDEN DEM"

    def test_overlay_dir_missing_is_ignored(self) -> None:
        """Non-existent overlay dir is silently skipped."""
        registry = load_registry(
            DEFAULT_REGISTRY,
            overlay_dirs=[Path("/nonexistent/overlay")],
        )
        assert "dem_3dep_10m" in registry.datasets

    def test_overlay_empty_dir_is_ignored(self, tmp_path: Path) -> None:
        """Empty overlay dir is silently skipped."""
        overlay_dir = tmp_path / "empty_overlay"
        overlay_dir.mkdir()

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])
        assert "dem_3dep_10m" in registry.datasets

    def test_no_overlay_dirs_backward_compatible(self) -> None:
        """Calling without overlay_dirs works as before."""
        registry = load_registry(DEFAULT_REGISTRY)
        assert "dem_3dep_10m" in registry.datasets

    def test_malformed_yaml_skipped_with_warning(self, tmp_path: Path) -> None:
        """Malformed YAML overlay file is skipped, bundled registry still loads."""
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()
        (overlay_dir / "bad.yml").write_text("{{invalid yaml content")

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])
        assert "dem_3dep_10m" in registry.datasets

    def test_overlay_missing_datasets_key_skipped(self, tmp_path: Path) -> None:
        """Overlay file without 'datasets' key is skipped with warning."""
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()
        (overlay_dir / "no_datasets.yml").write_text(yaml.dump({"other_key": "value"}))

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])
        assert "dem_3dep_10m" in registry.datasets

    def test_overlay_empty_file_skipped(self, tmp_path: Path) -> None:
        """Empty YAML overlay file is skipped with warning."""
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()
        (overlay_dir / "empty.yml").write_text("")

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])
        assert "dem_3dep_10m" in registry.datasets

    def test_overlay_invalid_entries_skipped(self, tmp_path: Path) -> None:
        """Overlay with invalid Pydantic entries is skipped, bundled registry loads."""
        overlay_dir = tmp_path / "overlay"
        overlay_dir.mkdir()
        bad_yaml = {
            "datasets": {
                "bad_entry": {
                    "description": "Bad",
                    "strategy": "nonexistent_strategy",
                    "variables": [],
                }
            }
        }
        (overlay_dir / "invalid.yml").write_text(yaml.dump(bad_yaml))

        registry = load_registry(DEFAULT_REGISTRY, overlay_dirs=[overlay_dir])
        assert "dem_3dep_10m" in registry.datasets
        assert "bad_entry" not in registry.datasets
