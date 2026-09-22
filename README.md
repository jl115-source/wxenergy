# wxenergy

Weather and climate utilities for energy analysis.

`wxenergy` is a small, xarray-native Python package for reusable weather and climate data processing.

## Install

Development install from GitHub:

```bash
pip install git+https://github.com/jl115-source/wxenergy.git
```

For local development:

```bash
git clone https://github.com/jl115-source/wxenergy.git
cd wxenergy
pip install -e ".[dev]"
```

## Quickstart

```python
import xarray as xr
import wxenergy as wx

ds = xr.open_dataset("temperature.nc")

climo = wx.climatology(ds.t2m)
anom = wx.anomaly(ds.t2m)
hdd = wx.hdd(ds.t2m, base=18)
```

## Initial API

- `wx.climatology`
- `wx.anomaly`
- `wx.hdd`
- `wx.cdd`
- `wx.weighted_mean`
- `wx.normalize_longitude`

## Documentation

https://jl115-source.github.io/wxenergy/

## Status

Early development. The public API will stay deliberately small while the package structure, tests, and documentation mature.

## License

MIT
