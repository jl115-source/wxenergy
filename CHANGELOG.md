# Changelog

All notable changes to `wxenergy` will be documented here.

The project follows semantic versioning once public releases begin.

## Unreleased

### Added

- Sphinx documentation with the PyData theme.
- Temperature conversion between Kelvin, Celsius, and Fahrenheit.
- Wind-speed magnitude from vector components.
- Daily mean, minimum, and maximum aggregation.
- Cosine latitude weights for area-weighted gridded analysis.
- Dask and NetCDF integration tests.
- Property-based numerical tests with Hypothesis.
- Distribution build and installed-wheel smoke tests.
- Python 3.10–3.13 CI coverage.

### Changed

- Hardened validation for datetime coordinates, dimensions, and spatial weights.
- Degree-day calculations now preserve missing values and metadata.
- The development version is now `0.2.0.dev0`.

### Fixed

- Wind-speed results no longer inherit misleading component units when `u` and `v` use different unit metadata.

## 0.1.0.dev0

Initial package scaffold.
