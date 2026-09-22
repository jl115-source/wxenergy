"""Temporal aggregation helpers."""

from __future__ import annotations

from collections.abc import Callable

import xarray as xr


def _validate_time(data: xr.DataArray, time_dim: str) -> None:
    if time_dim not in data.dims:
        raise ValueError(f"Time dimension {time_dim!r} was not found in data dimensions.")
    if time_dim not in data.coords:
        raise ValueError(f"Time coordinate {time_dim!r} was not found.")
    try:
        _ = data[time_dim].dt
    except AttributeError as exc:
        raise TypeError(f"Coordinate {time_dim!r} must be datetime-like.") from exc


def _daily_reduce(
    data: xr.DataArray,
    reducer: Callable[..., xr.DataArray],
    *,
    time_dim: str,
    skipna: bool | None,
    suffix: str,
) -> xr.DataArray:
    _validate_time(data, time_dim)
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
    resampled = data.resample({time_dim: "1D"})
    return _daily_reduce(
        data,
        resampled.min,
        time_dim=time_dim,
        skipna=skipna,
        suffix="daily_min",
    )
