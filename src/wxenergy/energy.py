"""General weather-derived energy metrics."""

from __future__ import annotations

import xarray as xr


def hdd(data: xr.DataArray, *, base: float = 18.0) -> xr.DataArray:
    """Return heating degree days relative to ``base``.

    ``base`` must use the same temperature units as ``data``.
    """
    result = xr.where(data < base, base - data, 0)
    result.attrs = data.attrs.copy()
    result.attrs["long_name"] = "heating degree days"
    return result


def cdd(data: xr.DataArray, *, base: float = 18.0) -> xr.DataArray:
    """Return cooling degree days relative to ``base``.

    ``base`` must use the same temperature units as ``data``.
    """
    result = xr.where(data > base, data - base, 0)
    result.attrs = data.attrs.copy()
    result.attrs["long_name"] = "cooling degree days"
    return result
