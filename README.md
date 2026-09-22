# wxenergy

[![CI](https://github.com/jl115-source/wxenergy/actions/workflows/ci.yml/badge.svg)](https://github.com/jl115-source/wxenergy/actions/workflows/ci.yml)

Weather and climate utilities for energy analysis, built around xarray.

`wxenergy` provides a small set of composable functions for common processing steps between raw meteorological data and downstream analysis. The package is intentionally focused on data handling and general weather-derived metrics rather than data retrieval, forecasting systems, or trading logic.

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
ds = wx.normalize_longitude(ds)

temperature = wx.convert_temperature(ds.t2m, "degC")
daily = wx.daily_mean(temperature)
weights = wx.cosine_latitude_weights(daily.latitude)
regional = wx.weighted_mean(daily, weights, dim=("latitude", "longitude"))
cdd = wx.cdd(regional, base=18.0)
```

## Public API

| Area | Functions |
| --- | --- |
| Climate | `climatology`, `anomaly` |
| Energy | `hdd`, `cdd` |
| Temporal | `daily_mean`, `daily_min`, `daily_max` |
| Spatial | `cosine_latitude_weights`, `weighted_mean` |
| Coordinates | `normalize_longitude` |
| Units | `convert_temperature` |
| Wind | `wind_speed` |

All public functions operate on xarray objects and preserve labeled coordinates. Metadata is retained where its meaning remains valid.

## Development standards

The repository tests:

- Python 3.10–3.13
- numerical and edge-case behavior
- Dask-backed arrays
- NetCDF round trips
- property-based numerical invariants
- package wheel and source-distribution builds
- installation from the built wheel
- documentation builds with warnings treated as errors

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and [CHANGELOG.md](CHANGELOG.md) for notable changes.

## Documentation

https://jl115-source.github.io/wxenergy/

## Status

Pre-release development. The current development series is `0.2.x`; the public API may change before `1.0`.

## License

MIT
