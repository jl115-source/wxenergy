"""Helpers for common weather and climate data conventions."""

from __future__ import annotations

from typing import Literal

import xarray as xr

LongitudeTarget = Literal["-180_180", "0_360"]


def normalize_longitude(
    data: xr.Dataset | xr.DataArray,
    *,
    lon_name: str = "longitude",
    target: LongitudeTarget = "-180_180",
) -> xr.Dataset | xr.DataArray:
    """Normalize and sort a one-dimensional longitude coordinate.

    Parameters
    ----------
    data
        Input xarray object.
    lon_name
        Name of the longitude coordinate.
    target
        Target convention: ``"-180_180"`` or ``"0_360"``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        A new object with normalized, ascending longitudes.

    Raises
    ------
    KeyError
        If ``lon_name`` is not a coordinate.
    ValueError
        If the longitude coordinate is not one-dimensional or ``target`` is invalid.
    """
    if lon_name not in data.coords:
        raise KeyError(f"Longitude coordinate {lon_name!r} was not found.")

    lon = data[lon_name]
    if lon.ndim != 1:
        raise ValueError("normalize_longitude currently supports one-dimensional longitude only")

    if target == "-180_180":
        normalized = ((lon + 180) % 360) - 180
    elif target == "0_360":
        normalized = lon % 360
    else:
        raise ValueError("target must be '-180_180' or '0_360'")

    normalized.attrs = lon.attrs.copy()
    return data.assign_coords({lon_name: normalized}).sortby(lon_name)
