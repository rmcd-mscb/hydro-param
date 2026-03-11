# Contributing

Please see the full [Contributing Guide](https://github.com/rmcd-mscb/hydro-param/blob/main/CONTRIBUTING.md)
on GitHub for detailed instructions.

## Quick Reference

### Development Setup

```bash
git clone https://github.com/rmcd-mscb/hydro-param.git
cd hydro-param
pixi install
pixi run -e dev pre-commit
pixi run -e dev check
```

### Available Tasks

| Command | What it does |
|---|---|
| `pixi run -e dev test` | Run tests with coverage |
| `pixi run -e dev lint` | Lint with ruff |
| `pixi run -e dev format` | Format with ruff |
| `pixi run -e dev typecheck` | Type check with mypy |
| `pixi run -e dev check` | Run all quality checks |
| `pixi run -e docs docs-serve` | Serve docs locally |
| `pixi run -e docs docs-build` | Build docs site |

## Design Documents

Design decisions and implementation plans are archived in `docs/plans/`,
organized by date. Each significant feature or change starts with a design
document that captures the problem, proposed approaches, and the chosen
solution.

For a themed summary of all design work to date, see the
[Development Roadmap](plans/development-roadmap.md).
