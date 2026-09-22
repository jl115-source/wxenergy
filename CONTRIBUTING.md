# Contributing

Contributions are welcome. `wxenergy` aims to keep a small, well-tested public API for reusable weather and climate processing.

## Development setup

```bash
git clone https://github.com/jl115-source/wxenergy.git
cd wxenergy
python -m pip install -e ".[dev,docs]"
pre-commit install
```

## Before opening a pull request

Run the same core checks used in CI:

```bash
ruff check src tests docs examples
ruff format --check src tests docs examples
mypy src/wxenergy
pytest --cov=wxenergy --cov-report=term-missing
python -m build
python -m twine check dist/*
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Design guidelines

- Prefer small functions that accept and return xarray objects.
- Preserve coordinates and useful metadata where practical.
- Use standard Python exceptions with actionable error messages.
- Keep data retrieval, forecasting logic, and trading-specific logic outside the core package.
- Add tests for numerical behavior, missing data, invalid inputs, and labeled-coordinate alignment.
- Public API additions should include documentation and an example of intended use.

## Versioning

The package is currently pre-1.0. New backwards-compatible functionality increments the minor development series; fixes increment the patch version once releases begin.
