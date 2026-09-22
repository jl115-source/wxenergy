"""Temporal aggregation helpers."""

from __future__ import annotations

from collections.abc import Callable

import xarray as xr

from ._validation import require_datetime_coordinate


def _daily_reduce(
    data: xr.DataArray,
    reducer: Callable[..., xr.DataArray],
    *,
    time_dim: str,
    skipna: bool | None,
    suffix: str,
) -> xr.DataArray:
    require_datetime_coordinate(data, time_dim)
    result = reducer(skipna=skipna, keep_attrs=True)
    if data.name:
        result.name = f"{data.name}_{suffix}"
    return result


def daily_mean(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Resample a DataArray to daily means."""
    require_datetime_coordinate(data, time_dim)
    resampled = data.resample({time_dim: "1D"})
    return _daily_reduce(
        data,
        resampled.mean,
        time_dim=time_dim,
        skipna=skipna,
        suffix="daily_mean",
    )


def daily_max(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Resample a DataArray to daily maxima."""
    require_datetime_coordinate(data, time_dim)
    resampled = data.resample({time_dim: "1D"})
    return _daily_reduce(
        data,
        resampled.max,
        time_dim=time_dim,
        skipna=skipna,
        suffix="daily_max",
    )


def daily_min(
    data: xr.DataArray,
    *,
    time_dim: str = "time",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Resample a DataArray to daily minima."""
    require_datetime_coordinate(data, time_dim)
    resampled = data.resample({time_dim: "1D"})
    return _daily_reduce(
        data,
        resampled.min,
        time_dim=time_dim,
        skipna=skipna,
        suffix="daily_min",
    )
