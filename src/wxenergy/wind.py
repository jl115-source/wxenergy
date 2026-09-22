"""Wind vector utilities."""

from __future__ import annotations

import xarray as xr


def wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Return wind-speed magnitude from orthogonal vector components.

    The component arrays must have exactly aligned coordinates. Broadcasting
    across additional dimensions is supported by xarray.

    Parameters
    ----------
    u, v
        Orthogonal wind components, typically zonal and meridional wind.

    Returns
    -------
    xarray.DataArray
        ``sqrt(u**2 + v**2)`` with aligned coordinates.

    Raises
    ------
    ValueError
        If coordinates shared by the two arrays are not aligned.
    """
    try:
        u_aligned, v_aligned = xr.align(u, v, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("u and v coordinates must align exactly.") from exc

    result = (u_aligned**2 + v_aligned**2) ** 0.5
    result.name = "wind_speed"
    result.attrs = {"long_name": "wind speed"}

    u_units = u.attrs.get("units")
    v_units = v.attrs.get("units")
    if u_units and u_units == v_units:
        result.attrs["units"] = u_units

    return result
