# Contributing to hydro-param

## Development Workflow

This project uses **GitHub Flow**: all changes are made on feature branches and merged to `main` via pull requests.

### 1. Start with an Issue

Every code change begins with a GitHub issue. Use the appropriate template:

- **Bug Report** — for unexpected behavior or errors
- **Feature Request** — for new functionality or enhancements
- **Design Decision** — for architectural choices that affect the project

### 2. Create a Branch

Branch from `main` using this naming convention:

```
<type>/<issue-number>-<short-description>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:
```bash
git checkout -b feat/12-config-schema
git checkout -b fix/15-zarr-encoding
git checkout -b docs/20-api-reference
```

### 3. Write Code

Follow the project's coding conventions:

- **Type hints** on all public functions
- **NumPy-style docstrings** on public API
- **Pydantic models** for configuration validation
- **`logging`** module for status output (not `print()`)
- **Line length**: 100 characters
- **Tests** for all new public functions (pytest)

### 4. Commit with Conventional Commits

Use conventional commit messages:

```
<type>: <short description>

<optional body explaining why>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `ci`

Examples:
```
feat: add POLARIS dataset accessor
fix: handle empty geometry in spatial batch
docs: add configuration guide
refactor: extract weight cache into separate module
test: add edge cases for single-feature fabrics
chore: update ruff to v0.9
ci: add Python 3.12 to test matrix
```

### 5. Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com) to run checks before each commit:

```bash
# Install pre-commit (one-time setup)
pip install pre-commit
pre-commit install

# Run manually against all files
pre-commit run --all-files
```

Hooks include:
- **ruff** — linting and formatting
- **mypy** — type checking
- **detect-secrets** — prevents accidental credential commits
- **trailing-whitespace**, **end-of-file-fixer**, **check-yaml**, **check-toml**

### 6. Open a Pull Request

Push your branch and open a PR against `main`:

```bash
git push -u origin feat/12-config-schema
gh pr create
```

PR requirements:
- Fill out the PR template (summary, related issue, test plan, checklist)
- Link the issue with `Closes #<number>`
- All CI checks must pass (lint, type check, tests, pre-commit)
- Copilot code review will run automatically

### 7. Review and Merge

- Address Copilot review comments
- Get approval from a maintainer (or self-approve for solo work)
- Squash-merge to keep `main` history clean

## Setting Up Your Development Environment

```bash
# Clone the repository
git clone https://github.com/rmcd-mscb/hydro-param.git
cd hydro-param

# Create a virtual environment and install
python -m venv .venv
source .venv/bin/activate  # or .venv/Scripts/activate on Windows
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Verify everything works
pytest tests/
ruff check src/ tests/
mypy src/
```

## Architecture Guidelines

Before contributing, read [docs/design.md](docs/design.md) for the full architecture. Key rules:

1. **Dask is for I/O only** — do not introduce `dask.distributed`
2. **Config is declarative** — no logic in YAML files
3. **Cache by stable IDs** — never hash geometry coordinates
4. **Output formatters are plugins** — model-specific logic goes in adapters
5. **Spatial batching is foundational** — never assume row order is spatial
