# wxenergy

[![CI](https://github.com/jl115-source/wxenergy/actions/workflows/ci.yml/badge.svg)](https://github.com/jl115-source/wxenergy/actions/workflows/ci.yml)

Weather and climate utilities for energy analysis, built around xarray.

`wxenergy` is a small scientific-Python package for the processing layer between raw meteorological data and downstream energy analysis. It provides labeled-array utilities for coordinate normalization, temporal aggregation, climatologies, ensemble statistics, spatial weighting, unit conversion, wind diagnostics, and general weather-derived energy metrics.

The package does not provide data-vendor clients, forecasting systems, trading signals, or market-specific decision logic.

## Installation

The package is not yet published on PyPI. Install the development version directly from GitHub:

```bash
python -m pip install git+https://github.com/jl115-source/wxenergy.git
```

For development:

```bash
git clone https://github.com/jl115-source/wxenergy.git
cd wxenergy
python -m pip install -e ".[dev,docs]"
```

## Quick example

```python
import xarray as xr
import wxenergy as wx

ds = xr.open_dataset("weather.nc")
ds = wx.standardize(ds, longitude_target="-180_180")

temperature = wx.convert_temperature(ds.t2m, "degC")
daily = wx.daily_mean(temperature)
weights = wx.cosine_latitude_weights(daily.latitude)
regional = wx.weighted_mean(daily, weights, dim=("latitude", "longitude"))
cdd = wx.cdd(regional, base=18.0)
```

Ensemble data use the same xarray-native API:

```python
mean = wx.ensemble_mean(temperature_ensemble)
spread = wx.ensemble_spread(temperature_ensemble)
prob = wx.ensemble_probability(temperature_ensemble, 35.0, comparison="ge")
```

## Public API

| Area | Functions |
| --- | --- |
| Standardization | `standardize`, `normalize_longitude` |
| Climate | `climatology`, `anomaly` |
| Energy | `hdd`, `cdd`, `degree_hours` |
| Temporal | `daily_mean`, `daily_min`, `daily_max` |
| Spatial | `cosine_latitude_weights`, `weighted_mean` |
| Ensemble | `ensemble_mean`, `ensemble_spread`, `ensemble_quantile`, `ensemble_probability` |
| Units | `convert_temperature` |
| Wind | `wind_speed` |

All public functions operate on xarray objects, preserve labeled coordinates, and retain metadata where its meaning remains valid. Dask-backed inputs remain lazy for the core array-processing operations.

## Engineering and compatibility

The repository checks:

- Python 3.10–3.14
- minimum supported NumPy/xarray dependencies
- Ruff linting and formatting
- static type checking with mypy
- branch-aware test coverage with a 90% CI floor
- property-based numerical tests with Hypothesis
- Dask-backed lazy arrays
- realistic gridded-weather integration workflows
- NetCDF round trips
- executable examples
- wheel and source-distribution builds
- installation and import from the built wheel
- package metadata with `twine check`
- Sphinx documentation with warnings treated as errors

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [CHANGELOG.md](CHANGELOG.md) for notable changes.

## Documentation

https://jl115-source.github.io/wxenergy/

## Status

Pre-release development. The current development series is `0.2.x`; the public API may change before `1.0`.

## License

MIT
