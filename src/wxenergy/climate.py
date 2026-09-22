"""Climate statistics for xarray objects."""

from __future__ import annotations

import xarray as xr


def climatology(
    data: xr.DataArray,
    *,
    groupby: str = "dayofyear",
    time_dim: str = "time",
) -> xr.DataArray:
    """Return a grouped climatology along a time dimension.

    Parameters
    ----------
    data:
        Input data containing ``time_dim``.
    groupby:
        Datetime accessor used for grouping, for example ``"dayofyear"`` or ``"month"``.
    time_dim:
        Name of the time dimension.
    """
    key = f"{time_dim}.{groupby}"
    return data.groupby(key).mean(time_dim, keep_attrs=True)


def anomaly(
    data: xr.DataArray,
    *,
    climatology_data: xr.DataArray | None = None,
    groupby: str = "dayofyear",
    time_dim: str = "time",
) -> xr.DataArray:
    """Return anomalies from a grouped climatology.

    If ``climatology_data`` is omitted, it is calculated from ``data``.
    """
    key = f"{time_dim}.{groupby}"
    if climatology_data is None:
        climatology_data = climatology(data, groupby=groupby, time_dim=time_dim)

    result = data.groupby(key) - climatology_data
    result.attrs = data.attrs.copy()
    if data.name:
        result.name = f"{data.name}_anomaly"
    return result
