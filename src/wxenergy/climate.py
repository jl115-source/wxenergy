"""Climate statistics for xarray objects."""

from __future__ import annotations

import xarray as xr


def _validate_time_coordinate(data: xr.DataArray, time_dim: str) -> None:
    if time_dim not in data.dims:
        raise ValueError(f"Time dimension {time_dim!r} was not found in data dimensions.")
    if time_dim not in data.coords:
        raise ValueError(f"Time coordinate {time_dim!r} was not found.")
    try:
        _ = data[time_dim].dt
    except AttributeError as exc:
        raise TypeError(f"Coordinate {time_dim!r} must be datetime-like.") from exc


def _validate_groupby(data: xr.DataArray, time_dim: str, groupby: str) -> None:
    _validate_time_coordinate(data, time_dim)
    try:
        getattr(data[time_dim].dt, groupby)
    except AttributeError as exc:
        raise ValueError(
            f"Unsupported datetime grouping {groupby!r} for coordinate {time_dim!r}."
        ) from exc


def climatology(
    data: xr.DataArray,
    *,
    groupby: str = "dayofyear",
    time_dim: str = "time",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate a grouped climatological mean.

    Parameters
    ----------
    data
        Input data containing a datetime-like coordinate named by ``time_dim``.
    groupby
        Datetime accessor used for grouping, such as ``"dayofyear"`` or ``"month"``.
    time_dim
        Name of the time dimension and coordinate.
    skipna
        Whether missing values should be skipped by the mean. ``None`` uses xarray's
        dtype-dependent default.

    Returns
    -------
    xarray.DataArray
        The climatological mean with the grouping coordinate replacing ``time_dim``.
    """
    _validate_groupby(data, time_dim, groupby)
    key = f"{time_dim}.{groupby}"
    result = data.groupby(key).mean(time_dim, skipna=skipna, keep_attrs=True)
    if data.name:
        result.name = f"{data.name}_climatology"
    return result


def anomaly(
    data: xr.DataArray,
    *,
    climatology_data: xr.DataArray | None = None,
    groupby: str = "dayofyear",
    time_dim: str = "time",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate anomalies relative to a grouped climatology.

    Parameters
    ----------
    data
        Input data containing a datetime-like coordinate named by ``time_dim``.
    climatology_data
        Optional precomputed climatology. If omitted, one is calculated from ``data``.
    groupby
        Datetime accessor used for grouping, such as ``"dayofyear"`` or ``"month"``.
    time_dim
        Name of the time dimension and coordinate.
    skipna
        Passed to :func:`climatology` when ``climatology_data`` is not supplied.

    Returns
    -------
    xarray.DataArray
        Anomalies aligned to the original input coordinates.
    """
    _validate_groupby(data, time_dim, groupby)
    key = f"{time_dim}.{groupby}"

    if climatology_data is None:
        climatology_data = climatology(
            data,
            groupby=groupby,
            time_dim=time_dim,
            skipna=skipna,
        )
    elif groupby not in climatology_data.dims and groupby not in climatology_data.coords:
        raise ValueError(
            f"Provided climatology must contain the grouping coordinate {groupby!r}."
        )

    result = data.groupby(key) - climatology_data
    result.attrs = data.attrs.copy()
    if data.name:
        result.name = f"{data.name}_anomaly"
    return result
