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


def test_load_real_registry():
    """Test loading the actual configs/datasets.yml file."""
    registry_path = Path("configs/datasets.yml")
    if not registry_path.exists():
        pytest.skip("configs/datasets.yml not found")
    registry = load_registry(registry_path)
    assert "dem_3dep_10m" in registry.datasets
    assert "polaris_100m" in registry.datasets
    assert "nlcd_legacy" in registry.datasets
    assert "nlcd_annual" in registry.datasets


def test_real_registry_nlcd_legacy_has_download():
    """Verify NLCD legacy entry in real registry has multi-file download block."""
    registry_path = Path("configs/datasets.yml")
    if not registry_path.exists():
        pytest.skip("configs/datasets.yml not found")
    registry = load_registry(registry_path)
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
    registry_path = Path("configs/datasets.yml")
    if not registry_path.exists():
        pytest.skip("configs/datasets.yml not found")
    registry = load_registry(registry_path)
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
