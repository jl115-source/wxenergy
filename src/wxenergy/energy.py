"""General weather-derived energy metrics."""

from __future__ import annotations

import math

import xarray as xr


def _validate_base(base: float) -> None:
    if not math.isfinite(base):
        raise ValueError("base must be a finite number")


def _finish_degree_day_result(
    result: xr.DataArray,
    source: xr.DataArray,
    *,
    metric: str,
    base: float,
) -> xr.DataArray:
    result.attrs = source.attrs.copy()
    result.attrs["long_name"] = f"{metric} degree days"
    result.attrs["base_temperature"] = base
    if source.name:
        suffix = "hdd" if metric == "heating" else "cdd"
        result.name = f"{source.name}_{suffix}"
    return result


def hdd(data: xr.DataArray, *, base: float = 18.0) -> xr.DataArray:
    """Calculate pointwise heating degree-day values.

    The calculation is ``max(base - temperature, 0)``. Missing input values remain
    missing in the result. ``base`` must use the same temperature units as ``data``.

    Parameters
    ----------
    data
        Temperature data.
    base
        Base temperature in the same units as ``data``.
    """
    _validate_base(base)
    result = (base - data).clip(min=0)
    return _finish_degree_day_result(result, data, metric="heating", base=base)


def cdd(data: xr.DataArray, *, base: float = 18.0) -> xr.DataArray:
    """Calculate pointwise cooling degree-day values.

    The calculation is ``max(temperature - base, 0)``. Missing input values remain
    missing in the result. ``base`` must use the same temperature units as ``data``.

    Parameters
    ----------
    data
        Temperature data.
    base
        Base temperature in the same units as ``data``.
    """
    _validate_base(base)
    result = (data - base).clip(min=0)
    return _finish_degree_day_result(result, data, metric="cooling", base=base)
