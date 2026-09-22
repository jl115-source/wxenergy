"""General weather-derived energy metrics."""

from __future__ import annotations

import math
from typing import Literal

import xarray as xr

DegreeHourMode = Literal["heating", "cooling"]


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
    """
    _validate_base(base)
    result = (base - data).clip(min=0)
    return _finish_degree_day_result(result, data, metric="heating", base=base)


def cdd(data: xr.DataArray, *, base: float = 18.0) -> xr.DataArray:
    """Calculate pointwise cooling degree-day values.

    The calculation is ``max(temperature - base, 0)``. Missing input values remain
    missing in the result. ``base`` must use the same temperature units as ``data``.
    """
    _validate_base(base)
    result = (data - base).clip(min=0)
    return _finish_degree_day_result(result, data, metric="cooling", base=base)


def degree_hours(
    data: xr.DataArray,
    *,
    base: float = 18.0,
    mode: DegreeHourMode = "heating",
) -> xr.DataArray:
    """Calculate pointwise heating or cooling degree-hour values.

    This function returns the instantaneous departure from ``base`` appropriate for
    regularly sampled hourly data. It does not integrate or infer sampling intervals.

    Parameters
    ----------
    data
        Temperature data.
    base
        Base temperature in the same units as ``data``.
    mode
        ``"heating"`` computes ``max(base - temperature, 0)``; ``"cooling"`` computes
        ``max(temperature - base, 0)``.
    """
    _validate_base(base)
    if mode == "heating":
        result = (base - data).clip(min=0)
        suffix = "hdh"
    elif mode == "cooling":
        result = (data - base).clip(min=0)
        suffix = "cdh"
    else:
        raise ValueError("mode must be 'heating' or 'cooling'")

    result.attrs = data.attrs.copy()
    result.attrs["long_name"] = f"{mode} degree hours"
    result.attrs["base_temperature"] = base
    if data.name:
        result.name = f"{data.name}_{suffix}"
    return result
