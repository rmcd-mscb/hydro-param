# Copilot Code Review Guidelines — hydro-param

When reviewing pull requests for this project, apply the following guidelines.

## Architecture Rules (MUST enforce)

- **No `dask.distributed`**: Dask is for lazy I/O only (Zarr subsetting). Flag any import of `dask.distributed` or use of `Client()`.
- **No logic in YAML configs**: Config files are declarative. No variables, conditionals, or templating. Flag any suggestion to add these.
- **No geometry hashing for cache keys**: Cache keys use composite stable IDs (fabric version + dataset ID + CRS + method). Flag any use of hash on geometry coordinates.
- **No model-specific logic in core**: Output formatting for specific models (PRMS, NextGen) belongs in adapter classes, not the processing engine.
- **Polygon vs grid routing**: Polygon targets use gdptools, grid targets use xesmf/rioxarray. Flag if these are mixed incorrectly.

## Code Style

- Python ≥ 3.10, target 3.11+
- Type hints on all public functions
- NumPy-style docstrings on public API
- `logging` module only (no `print()` for status output)
- Line length: 100 characters
- Pydantic models for configuration validation

## Testing

- All new public functions need tests
- Use pytest fixtures, avoid unittest-style classes
- Test edge cases: empty geometries, single-feature fabrics, missing data

## Security

- No credentials or API keys in code
- No hardcoded file paths outside of test fixtures
- Validate all external input (config files, user-supplied paths)

## Conventional Commits

Commit messages must follow the format: `<type>: <description>`
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`
