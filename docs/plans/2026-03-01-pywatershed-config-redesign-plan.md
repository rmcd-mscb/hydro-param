# pywatershed_run.yml Config Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign `pywatershed_run.yml` as a consumer-oriented, self-documenting contract between the Phase 1 pipeline and the Phase 2 pywatershed derivation plugin.

**Architecture:** Three new data sections (`static_datasets`, `forcing`, `climate_normals`) keyed by pywatershed parameter names, with explicit Pydantic fields per domain category. Each entry declares the pipeline dataset `source`, source `variable(s)`, and `statistic` used. An `available` field per category surfaces registry choices and is validated at runtime. The derivation plugin validates the SIR against config declarations at startup and uses `source` for dataset-specific routing. Issue #124 (SIR dataset prefix) is a prerequisite — this plan assumes it is implemented first or concurrently.

**Tech Stack:** Pydantic v2, PyYAML, pytest, cyclopts (CLI)

**Design doc:** `docs/plans/2026-03-01-pywatershed-config-redesign-design.md`

---

### Task 1: Add ParameterEntry and category Pydantic models

**Files:**
- Modify: `src/hydro_param/pywatershed_config.py:1-35` (imports and new models before existing classes)
- Test: `tests/test_pywatershed_config.py`

**Step 1: Write the failing tests**

Add a new test class in `tests/test_pywatershed_config.py`:

```python
from hydro_param.pywatershed_config import (
    ParameterEntry,
    TopographyDatasets,
    SoilsDatasets,
    LandcoverDatasets,
    SnowDatasets,
    WaterbodyDatasets,
    ForcingConfig,
    ClimateNormalsConfig,
    StaticDatasetsConfig,
)


class TestParameterEntry:
    """Tests for ParameterEntry schema."""

    def test_minimal_entry(self) -> None:
        entry = ParameterEntry(
            source="dem_3dep_10m",
            variable="elevation",
            statistic="mean",
            description="Mean HRU elevation",
        )
        assert entry.source == "dem_3dep_10m"
        assert entry.variable == "elevation"
        assert entry.statistic == "mean"

    def test_multi_variable_entry(self) -> None:
        entry = ParameterEntry(
            source="polaris_30m",
            variables=["sand", "silt", "clay"],
            statistic="mean",
            description="Soil type classification",
        )
        assert entry.variables == ["sand", "silt", "clay"]

    def test_temporal_entry_with_time_period(self) -> None:
        entry = ParameterEntry(
            source="snodas",
            variable="SWE",
            statistic="mean",
            time_period=["2020-01-01", "2021-12-31"],
            description="Snow depletion threshold",
        )
        assert entry.time_period == ["2020-01-01", "2021-12-31"]

    def test_entry_with_year(self) -> None:
        entry = ParameterEntry(
            source="nlcd_osn_lndcov",
            variable="LndCov",
            statistic="categorical",
            year=[2021],
            description="Vegetation cover type",
        )
        assert entry.year == [2021]

    def test_description_required(self) -> None:
        with pytest.raises(ValidationError, match="description"):
            ParameterEntry(source="dem_3dep_10m", variable="elevation", statistic="mean")  # type: ignore[call-arg]

    def test_source_required(self) -> None:
        with pytest.raises(ValidationError, match="source"):
            ParameterEntry(variable="elevation", statistic="mean", description="test")  # type: ignore[call-arg]


class TestCategoryModels:
    """Tests for domain category Pydantic models."""

    def test_topography_defaults_empty(self) -> None:
        topo = TopographyDatasets()
        assert topo.available == []
        assert topo.hru_elev is None
        assert topo.hru_slope is None
        assert topo.hru_aspect is None

    def test_topography_with_entries(self) -> None:
        topo = TopographyDatasets(
            available=["dem_3dep_10m"],
            hru_elev=ParameterEntry(
                source="dem_3dep_10m",
                variable="elevation",
                statistic="mean",
                description="Mean HRU elevation",
            ),
        )
        assert topo.hru_elev is not None
        assert topo.hru_elev.source == "dem_3dep_10m"
        assert topo.hru_slope is None

    def test_soils_with_mixed_sources(self) -> None:
        soils = SoilsDatasets(
            available=["polaris_30m", "gnatsgo_rasters"],
            soil_type=ParameterEntry(
                source="polaris_30m",
                variables=["sand", "silt", "clay"],
                statistic="mean",
                description="Soil type classification",
            ),
            soil_moist_max=ParameterEntry(
                source="gnatsgo_rasters",
                variable="aws0_100",
                statistic="mean",
                description="Max available water-holding capacity",
            ),
        )
        assert soils.soil_type.source == "polaris_30m"
        assert soils.soil_moist_max.source == "gnatsgo_rasters"

    def test_forcing_defaults_empty(self) -> None:
        forcing = ForcingConfig()
        assert forcing.prcp is None
        assert forcing.tmax is None
        assert forcing.tmin is None

    def test_climate_normals_defaults_empty(self) -> None:
        cn = ClimateNormalsConfig()
        assert cn.jh_coef is None
        assert cn.transp_beg is None
        assert cn.transp_end is None

    def test_static_datasets_nests_all_categories(self) -> None:
        sd = StaticDatasetsConfig()
        assert isinstance(sd.topography, TopographyDatasets)
        assert isinstance(sd.soils, SoilsDatasets)
        assert isinstance(sd.landcover, LandcoverDatasets)
        assert isinstance(sd.snow, SnowDatasets)
        assert isinstance(sd.waterbodies, WaterbodyDatasets)
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestParameterEntry -v`
Expected: FAIL with `ImportError` (classes don't exist yet)

**Step 3: Write the implementation**

Add the following models to `src/hydro_param/pywatershed_config.py` after the imports, before `PwsDomainConfig`:

```python
class ParameterEntry(BaseModel):
    """Declare the SIR data source for a single pywatershed parameter.

    Each entry maps a pywatershed parameter to the pipeline dataset, source
    variable(s), and zonal statistic that produced the SIR data.

    Attributes
    ----------
    source : str
        Pipeline dataset registry name (e.g., ``"dem_3dep_10m"``).
    variable : str or None
        Source variable name when a single variable is used.
    variables : list[str] or None
        Source variable names when multiple variables contribute
        (e.g., ``["sand", "silt", "clay"]`` for soil_type).
    statistic : str or None
        Zonal statistic applied (``"mean"``, ``"categorical"``).
    year : int or list[int] or None
        NLCD year(s) for multi-epoch land cover.
    time_period : list[str] or None
        Temporal range ``[start, end]`` in ISO format for temporal datasets.
    description : str
        Human-readable description of what this parameter represents.
    """

    source: str
    variable: str | None = None
    variables: list[str] | None = None
    statistic: str | None = None
    year: int | list[int] | None = None
    time_period: list[str] | None = None
    description: str


class TopographyDatasets(BaseModel):
    """Topography parameters derived from DEM zonal statistics.

    Attributes
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    hru_elev : ParameterEntry or None
        Mean HRU elevation.
    hru_slope : ParameterEntry or None
        Mean HRU land surface slope.
    hru_aspect : ParameterEntry or None
        Mean HRU aspect.
    """

    available: list[str] = Field(default_factory=list)
    hru_elev: ParameterEntry | None = None
    hru_slope: ParameterEntry | None = None
    hru_aspect: ParameterEntry | None = None


class SoilsDatasets(BaseModel):
    """Soil parameters derived from soil property datasets.

    Attributes
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    soil_type : ParameterEntry or None
        Soil type classification (1=sand, 2=loam, 3=clay).
    sat_threshold : ParameterEntry or None
        Gravity reservoir storage capacity (from porosity).
    soil_moist_max : ParameterEntry or None
        Maximum available water-holding capacity.
    soil_rechr_max_frac : ParameterEntry or None
        Recharge zone storage as fraction of soil_moist_max.
    """

    available: list[str] = Field(default_factory=list)
    soil_type: ParameterEntry | None = None
    sat_threshold: ParameterEntry | None = None
    soil_moist_max: ParameterEntry | None = None
    soil_rechr_max_frac: ParameterEntry | None = None


class LandcoverDatasets(BaseModel):
    """Land cover parameters derived from NLCD.

    Attributes
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    cov_type : ParameterEntry or None
        Vegetation cover type.
    hru_percent_imperv : ParameterEntry or None
        Impervious surface fraction.
    """

    available: list[str] = Field(default_factory=list)
    cov_type: ParameterEntry | None = None
    hru_percent_imperv: ParameterEntry | None = None


class SnowDatasets(BaseModel):
    """Snow parameters derived from historical SWE data.

    Attributes
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    snarea_thresh : ParameterEntry or None
        Snow depletion threshold (calibration seed from historical max SWE).
    """

    available: list[str] = Field(default_factory=list)
    snarea_thresh: ParameterEntry | None = None


class WaterbodyDatasets(BaseModel):
    """Depression storage and HRU type from waterbody overlay.

    Attributes
    ----------
    available : list[str]
        Curated datasets available in the registry for this category.
    hru_type : ParameterEntry or None
        HRU type (0=inactive, 1=land, 2=lake, 3=swale).
    dprst_frac : ParameterEntry or None
        Fraction of HRU with surface depressions.
    dprst_area_max : ParameterEntry or None
        Maximum surface depression area.
    """

    available: list[str] = Field(default_factory=list)
    hru_type: ParameterEntry | None = None
    dprst_frac: ParameterEntry | None = None
    dprst_area_max: ParameterEntry | None = None


class StaticDatasetsConfig(BaseModel):
    """Static dataset declarations grouped by domain category.

    Each category contains explicit parameter fields that map to SIR data
    produced by the Phase 1 pipeline.

    Attributes
    ----------
    topography : TopographyDatasets
        DEM-derived parameters (elevation, slope, aspect).
    soils : SoilsDatasets
        Soil property parameters.
    landcover : LandcoverDatasets
        Land cover and impervious surface parameters.
    snow : SnowDatasets
        Historical snow parameters.
    waterbodies : WaterbodyDatasets
        Depression storage and HRU type.
    """

    topography: TopographyDatasets = Field(default_factory=TopographyDatasets)
    soils: SoilsDatasets = Field(default_factory=SoilsDatasets)
    landcover: LandcoverDatasets = Field(default_factory=LandcoverDatasets)
    snow: SnowDatasets = Field(default_factory=SnowDatasets)
    waterbodies: WaterbodyDatasets = Field(default_factory=WaterbodyDatasets)


class ForcingConfig(BaseModel):
    """Temporal forcing time series declarations.

    pywatershed expects one-variable-per-NetCDF in PRMS units (inches, degF).
    Only the three required CBH variables appear here.

    Attributes
    ----------
    available : list[str]
        Temporal-capable datasets available in the registry.
    prcp : ParameterEntry or None
        Daily precipitation.
    tmax : ParameterEntry or None
        Daily maximum temperature.
    tmin : ParameterEntry or None
        Daily minimum temperature.
    """

    available: list[str] = Field(default_factory=list)
    prcp: ParameterEntry | None = None
    tmax: ParameterEntry | None = None
    tmin: ParameterEntry | None = None


class ClimateNormalsConfig(BaseModel):
    """Long-term climate statistics for derived parameters.

    Can use the same source as forcing, or a different one (e.g.,
    forcing from CONUS404-BA but normals from gridMET).

    Attributes
    ----------
    available : list[str]
        Temporal-capable datasets available in the registry.
    jh_coef : ParameterEntry or None
        Jensen-Haise PET coefficient (monthly, from tmax/tmin normals).
    transp_beg : ParameterEntry or None
        Month transpiration begins (from last spring frost).
    transp_end : ParameterEntry or None
        Month transpiration ends (from first fall killing frost).
    """

    available: list[str] = Field(default_factory=list)
    jh_coef: ParameterEntry | None = None
    transp_beg: ParameterEntry | None = None
    transp_end: ParameterEntry | None = None
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestParameterEntry tests/test_pywatershed_config.py::TestCategoryModels -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/pywatershed_config.py tests/test_pywatershed_config.py
git commit -m "feat: add ParameterEntry and category Pydantic models for config redesign"
```

---

### Task 2: Wire new sections into PywatershedRunConfig

**Files:**
- Modify: `src/hydro_param/pywatershed_config.py:221-266` (PywatershedRunConfig class)
- Test: `tests/test_pywatershed_config.py`

**Step 1: Write the failing tests**

```python
class TestPywatershedRunConfigV4:
    """Tests for v4.0 config with static_datasets, forcing, climate_normals."""

    def test_v4_accepts_new_sections(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(
                    available=["dem_3dep_10m"],
                    hru_elev=ParameterEntry(
                        source="dem_3dep_10m",
                        variable="elevation",
                        statistic="mean",
                        description="Mean HRU elevation",
                    ),
                ),
            ),
        )
        assert cfg.version == "4.0"
        assert cfg.static_datasets.topography.hru_elev is not None

    def test_v4_defaults_all_sections_empty(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
        )
        assert cfg.static_datasets.topography.hru_elev is None
        assert cfg.forcing.prcp is None
        assert cfg.climate_normals.jh_coef is None

    def test_v4_full_config_from_yaml(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        config_dict = {
            "target_model": "pywatershed",
            "version": "4.0",
            "sir_path": "output",
            "domain": {"fabric_path": str(fabric), "id_field": "nhm_id"},
            "time": {"start": "2020-01-01", "end": "2020-12-31"},
            "static_datasets": {
                "topography": {
                    "available": ["dem_3dep_10m"],
                    "hru_elev": {
                        "source": "dem_3dep_10m",
                        "variable": "elevation",
                        "statistic": "mean",
                        "description": "Mean HRU elevation",
                    },
                },
                "soils": {
                    "available": ["polaris_30m", "gnatsgo_rasters"],
                    "soil_type": {
                        "source": "polaris_30m",
                        "variables": ["sand", "silt", "clay"],
                        "statistic": "mean",
                        "description": "Soil type classification",
                    },
                },
            },
            "forcing": {
                "available": ["gridmet"],
                "prcp": {
                    "source": "gridmet",
                    "variable": "pr",
                    "statistic": "mean",
                    "description": "Daily precipitation",
                },
                "tmax": {
                    "source": "gridmet",
                    "variable": "tmmx",
                    "statistic": "mean",
                    "description": "Daily maximum temperature",
                },
                "tmin": {
                    "source": "gridmet",
                    "variable": "tmmn",
                    "statistic": "mean",
                    "description": "Daily minimum temperature",
                },
            },
            "climate_normals": {
                "available": ["gridmet"],
                "jh_coef": {
                    "source": "gridmet",
                    "variables": ["tmmx", "tmmn"],
                    "description": "Jensen-Haise PET coefficient",
                },
                "transp_beg": {
                    "source": "gridmet",
                    "variable": "tmmn",
                    "description": "Month transpiration begins",
                },
                "transp_end": {
                    "source": "gridmet",
                    "variable": "tmmn",
                    "description": "Month transpiration ends",
                },
            },
        }
        config_path = tmp_path / "pws.yml"
        config_path.write_text(yaml.dump(config_dict))
        cfg = load_pywatershed_config(config_path)
        assert cfg.static_datasets.topography.hru_elev.source == "dem_3dep_10m"
        assert cfg.forcing.prcp.source == "gridmet"
        assert cfg.climate_normals.transp_beg.source == "gridmet"

    def test_v4_rejects_unknown_top_level_field(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        with pytest.raises(ValidationError, match="extra"):
            PywatershedRunConfig(
                version="4.0",
                domain=PwsDomainConfig(fabric_path=fabric),
                time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
                bogus_field="oops",
            )
```

**Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestPywatershedRunConfigV4 -v`
Expected: FAIL (version literal doesn't accept "4.0", new fields don't exist)

**Step 3: Write the implementation**

Modify `PywatershedRunConfig` in `src/hydro_param/pywatershed_config.py`:

```python
class PywatershedRunConfig(BaseModel):
    """Define the top-level configuration for pywatershed model setup.

    A consumer-oriented, self-documenting contract between the Phase 1
    pipeline and the Phase 2 pywatershed derivation plugin.  Three data
    sections (``static_datasets``, ``forcing``, ``climate_normals``)
    declare which pipeline datasets provide each pywatershed parameter.

    Attributes
    ----------
    target_model : {"pywatershed"}
        Target model identifier (fixed to ``"pywatershed"``).
    version : str
        Config schema version (``"4.0"``).
    domain : PwsDomainConfig
        Domain fabric file paths and ID field names.
    time : PwsTimeConfig
        Simulation time period.
    sir_path : Path
        Path to the Phase 1 pipeline output directory containing
        ``.manifest.yml`` and ``sir/`` subdirectory.  Relative paths
        are resolved against the config file's parent directory.
    static_datasets : StaticDatasetsConfig
        Static dataset declarations grouped by domain category.
    forcing : ForcingConfig
        Temporal forcing time series declarations.
    climate_normals : ClimateNormalsConfig
        Long-term climate statistics for derived parameters.
    parameter_overrides : PwsParameterOverrides
        Manual parameter value overrides.
    calibration : PwsCalibrationConfig
        Calibration seed generation options.
    output : PwsOutputConfig
        Output directory structure and filenames.

    See Also
    --------
    load_pywatershed_config : YAML loader for this schema.
    hydro_param.cli.pws_run_cmd : Two-phase workflow consumer.
    """

    model_config = ConfigDict(extra="forbid")

    target_model: Literal["pywatershed"] = "pywatershed"
    version: Literal["4.0"] = "4.0"
    domain: PwsDomainConfig
    time: PwsTimeConfig
    sir_path: Path = Path("output")
    static_datasets: StaticDatasetsConfig = Field(default_factory=StaticDatasetsConfig)
    forcing: ForcingConfig = Field(default_factory=ForcingConfig)
    climate_normals: ClimateNormalsConfig = Field(default_factory=ClimateNormalsConfig)
    parameter_overrides: PwsParameterOverrides = PwsParameterOverrides()
    calibration: PwsCalibrationConfig = PwsCalibrationConfig()
    output: PwsOutputConfig = PwsOutputConfig()
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestPywatershedRunConfigV4 -v`
Expected: PASS

**Step 5: Fix existing tests for v3.0 → v4.0 version change**

Update `minimal_config_dict` fixture and any tests that reference `version: "3.0"` to use `"4.0"`.

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add src/hydro_param/pywatershed_config.py tests/test_pywatershed_config.py
git commit -m "feat: wire static_datasets, forcing, climate_normals into PywatershedRunConfig v4.0"
```

---

### Task 3: Update init template to generate v4.0 config

**Files:**
- Modify: `src/hydro_param/project.py:247-333` (generate_pywatershed_template)
- Test: `tests/test_project.py` (or wherever init template tests live)

**Step 1: Write the failing test**

```python
def test_pywatershed_template_v4(self) -> None:
    """Generated template should be valid v4.0 config."""
    template = generate_pywatershed_template("test_project")
    assert 'version: "4.0"' in template
    assert "static_datasets:" in template
    assert "forcing:" in template
    assert "climate_normals:" in template
    assert "available:" in template
    assert "hru_elev:" in template
    assert "description:" in template
```

**Step 2: Run test to verify it fails**

Expected: FAIL (template still generates v3.0)

**Step 3: Rewrite `generate_pywatershed_template`**

Replace the template string in `src/hydro_param/project.py` with the v4.0 format matching the example config from the design doc. Include `available:` fields populated with the currently registered dataset names for each category. The template should read from the dataset registry to populate `available:` values.

**Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_project.py -k pywatershed_template -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/hydro_param/project.py tests/test_project.py
git commit -m "feat: update init template to generate v4.0 pywatershed config"
```

---

### Task 4: Update example config and DRB e2e config

**Files:**
- Modify: `configs/examples/drb_2yr_pywatershed.yml` (replace with v4.0 format)

**Step 1: Rewrite the example config**

Replace `configs/examples/drb_2yr_pywatershed.yml` with the v4.0 format from the design doc mockup, using the DRB 2-year dataset names.

**Step 2: Validate the example config loads**

```python
def test_example_drb_2yr_pywatershed_loads() -> None:
    """Example config should parse without validation errors."""
    config_path = Path("configs/examples/drb_2yr_pywatershed.yml")
    if not config_path.exists():
        pytest.skip("Example config not found")
    cfg = load_pywatershed_config(config_path)
    assert cfg.version == "4.0"
    assert cfg.static_datasets.topography.hru_elev is not None
```

**Step 3: Commit**

```bash
git add configs/examples/drb_2yr_pywatershed.yml tests/test_pywatershed_config.py
git commit -m "docs: update example pywatershed config to v4.0 format"
```

---

### Task 5: Add SIR validation in CLI startup

**Files:**
- Modify: `src/hydro_param/cli.py:581-770` (pws_run_cmd)
- Modify: `src/hydro_param/pywatershed_config.py` (add helper to collect declared entries)
- Test: `tests/test_pywatershed_config.py`

**Step 1: Write the failing test**

Add a helper method on `PywatershedRunConfig` that collects all declared `ParameterEntry` objects:

```python
class TestConfigDeclaredEntries:
    """Tests for collecting declared parameter entries from config."""

    def test_collect_static_entries(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(
                    hru_elev=ParameterEntry(
                        source="dem_3dep_10m",
                        variable="elevation",
                        statistic="mean",
                        description="Mean HRU elevation",
                    ),
                ),
            ),
        )
        entries = cfg.declared_entries()
        assert "hru_elev" in entries
        assert entries["hru_elev"].source == "dem_3dep_10m"

    def test_collect_forcing_entries(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            forcing=ForcingConfig(
                prcp=ParameterEntry(
                    source="gridmet",
                    variable="pr",
                    statistic="mean",
                    description="Daily precipitation",
                ),
            ),
        )
        entries = cfg.declared_entries()
        assert "prcp" in entries

    def test_empty_config_returns_no_entries(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
        )
        entries = cfg.declared_entries()
        assert len(entries) == 0
```

**Step 2: Run tests to verify they fail**

Expected: FAIL (`declared_entries` doesn't exist)

**Step 3: Implement `declared_entries` method**

Add to `PywatershedRunConfig`:

```python
def declared_entries(self) -> dict[str, ParameterEntry]:
    """Collect all declared ParameterEntry objects from the config.

    Walk ``static_datasets``, ``forcing``, and ``climate_normals``
    sections and return a flat dictionary keyed by parameter name.

    Returns
    -------
    dict[str, ParameterEntry]
        Parameter name to entry mapping for all non-None entries.
    """
    entries: dict[str, ParameterEntry] = {}

    # Static datasets: walk each category
    for category in (
        self.static_datasets.topography,
        self.static_datasets.soils,
        self.static_datasets.landcover,
        self.static_datasets.snow,
        self.static_datasets.waterbodies,
    ):
        for field_name, field_info in category.model_fields.items():
            if field_name == "available":
                continue
            value = getattr(category, field_name)
            if value is not None:
                entries[field_name] = value

    # Forcing
    for field_name in ("prcp", "tmax", "tmin"):
        value = getattr(self.forcing, field_name)
        if value is not None:
            entries[field_name] = value

    # Climate normals
    for field_name in ("jh_coef", "transp_beg", "transp_end"):
        value = getattr(self.climate_normals, field_name)
        if value is not None:
            entries[field_name] = value

    return entries
```

**Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py::TestConfigDeclaredEntries -v`
Expected: PASS

**Step 5: Add SIR validation to CLI**

In `src/hydro_param/cli.py:pws_run_cmd`, after constructing the `SIRAccessor`, add validation that logs declared entries and checks they exist in the SIR. This is a logging-only first pass — full validate-and-route with #124 provenance comes after #124 is implemented.

```python
# ── Validate config contract against SIR ──
declared = pws_config.declared_entries()
if declared:
    logger.info("Config declares %d parameter entries:", len(declared))
    for name, entry in declared.items():
        logger.info("  %s ← %s.%s", name, entry.source, entry.variable or entry.variables)
else:
    logger.warning("No parameter entries declared in config — derivation will use SIR as-is.")
```

**Step 6: Commit**

```bash
git add src/hydro_param/pywatershed_config.py src/hydro_param/cli.py tests/test_pywatershed_config.py
git commit -m "feat: add declared_entries() and SIR contract validation at CLI startup"
```

---

### Task 6: Update `available:` field validation against dataset registry

**Files:**
- Modify: `src/hydro_param/pywatershed_config.py` (add registry validation)
- Modify: `src/hydro_param/dataset_registry.py` (expose category→dataset mapping)
- Test: `tests/test_pywatershed_config.py`

**Step 1: Write the failing test**

```python
class TestAvailableFieldValidation:
    """Tests for validating 'available' fields against the dataset registry."""

    def test_valid_available_passes(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(available=["dem_3dep_10m"]),
            ),
        )
        # Should not raise
        cfg.validate_available_fields()

    def test_unknown_available_warns(self, tmp_path: Path) -> None:
        fabric = tmp_path / "nhru.gpkg"
        fabric.touch()
        cfg = PywatershedRunConfig(
            version="4.0",
            domain=PwsDomainConfig(fabric_path=fabric),
            time=PwsTimeConfig(start="2020-01-01", end="2020-12-31"),
            static_datasets=StaticDatasetsConfig(
                topography=TopographyDatasets(available=["nonexistent_dataset"]),
            ),
        )
        with pytest.warns(UserWarning, match="nonexistent_dataset"):
            cfg.validate_available_fields()
```

**Step 2: Run tests to verify they fail**

Expected: FAIL (`validate_available_fields` doesn't exist)

**Step 3: Implement**

Add `validate_available_fields()` method to `PywatershedRunConfig` that loads the dataset registry and checks each `available` list entry exists. Unknown entries emit a `UserWarning` (not an error — the registry may have grown since `init` ran).

Check `dataset_registry.py` for the existing API to load and query datasets by name. Expose a `get_all_dataset_names() -> set[str]` if one doesn't exist.

**Step 4: Run all tests**

Run: `pixi run -e dev pytest tests/test_pywatershed_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/hydro_param/pywatershed_config.py src/hydro_param/dataset_registry.py tests/test_pywatershed_config.py
git commit -m "feat: validate available fields against dataset registry"
```

---

### Task 7: Run full test suite and pre-push checks

**Files:** None (validation only)

**Step 1: Run full test suite**

Run: `pixi run -e dev test`
Expected: ALL PASS

**Step 2: Run pre-commit hooks**

Run: `pixi run -e dev pre-commit`
Expected: ALL PASS

**Step 3: Run check suite**

Run: `pixi run -e dev check`
Expected: ALL PASS

---

### Task 8: Create GitHub issue and feature branch

**Step 1: Create the GitHub issue**

```bash
gh issue create \
  --title "feat: redesign pywatershed_run.yml as consumer-oriented data contract" \
  --body "## Problem

The current pywatershed_run.yml is a black box — it declares sir_path and trusts the derivation plugin to figure out which datasets produced which SIR variables.

## Solution

Redesign as a consumer-oriented, self-documenting contract with three data sections (static_datasets, forcing, climate_normals) keyed by pywatershed parameter names. Each entry declares the pipeline dataset source, variable(s), and statistic.

## Design

See docs/plans/2026-03-01-pywatershed-config-redesign-design.md

## Dependencies

- #124 (SIR dataset prefix) is a prerequisite for full validate-and-route behavior

## Scope

- New Pydantic models (ParameterEntry, category models)
- PywatershedRunConfig v4.0 with three new sections
- Init template generation with available fields
- CLI startup validation (declared entries vs SIR)
- Available field validation against dataset registry
- Example config and DRB e2e config migration"
```

**Step 2: Create the feature branch**

```bash
git checkout -b feat/<issue-number>-pywatershed-config-redesign
```

---

## Execution Order and Dependencies

```
Task 1 (ParameterEntry + category models)
  └─→ Task 2 (wire into PywatershedRunConfig)
        ├─→ Task 3 (init template)
        ├─→ Task 4 (example configs)
        └─→ Task 5 (declared_entries + CLI validation)
              └─→ Task 6 (available field registry validation)
                    └─→ Task 7 (full test suite)
                          └─→ Task 8 (issue + branch)
```

Tasks 3 and 4 can run in parallel after Task 2. Task 5 depends on Task 2. Task 6 depends on Task 5.

## Notes

- **Issue #124 dependency:** Full validate-and-route (matching `source:` to SIR dataset-prefixed filenames) requires #124. Task 5 implements a logging-only validation pass; the full SIR↔config contract enforcement should be a follow-up after #124 lands.
- **No breaking change concern:** Pre-alpha, no external users. Version bump from 3.0 → 4.0 is sufficient.
- **Existing tests:** The `minimal_config_dict` fixture and several existing tests reference `version: "3.0"`. These must be updated in Task 2.
