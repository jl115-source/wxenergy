"""Helpers for common weather and climate data conventions."""

from __future__ import annotations

from typing import Literal

import xarray as xr

LongitudeTarget = Literal["-180_180", "0_360"]


def standardize(
    data: xr.Dataset | xr.DataArray,
    *,
    longitude_target: LongitudeTarget | None = None,
) -> xr.Dataset | xr.DataArray:
    """Standardize common latitude and longitude coordinate names.

    The function intentionally performs only conservative, unambiguous renaming:
    ``lat`` becomes ``latitude`` and ``lon`` becomes ``longitude``. Existing canonical
    names are left unchanged. If both an alias and its canonical name are present, a
    ``ValueError`` is raised rather than guessing which coordinate should win.

    Parameters
    ----------
    data
        Input xarray object.
    longitude_target
        Optional longitude convention. If supplied, longitude is normalized after
        coordinate renaming.
    """
    rename: dict[str, str] = {}
    for alias, canonical in (("lat", "latitude"), ("lon", "longitude")):
        if alias in data.coords or alias in data.dims:
            if canonical in data.coords or canonical in data.dims:
                raise ValueError(
                    f"Cannot rename {alias!r} to {canonical!r}: both names are present."
                )
            rename[alias] = canonical

    result = data.rename(rename) if rename else data.copy(deep=False)
    if longitude_target is not None:
        result = normalize_longitude(result, target=longitude_target)
    return result


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
