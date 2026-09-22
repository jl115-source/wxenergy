"""Internal validation helpers for wxenergy."""

from __future__ import annotations

import xarray as xr


def require_dimension(data: xr.DataArray, dim: str) -> None:
    """Raise ``ValueError`` when *dim* is not present on *data*."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} was not found in data dimensions {data.dims!r}.")


def require_coordinate(data: xr.DataArray | xr.Dataset, name: str) -> None:
    """Raise ``KeyError`` when a coordinate is missing."""
    if name not in data.coords:
        raise KeyError(f"Coordinate {name!r} was not found.")


def require_datetime_coordinate(data: xr.DataArray, name: str = "time") -> None:
    """Require a one-dimensional datetime-like coordinate and matching dimension."""
    require_dimension(data, name)
    require_coordinate(data, name)

    coord = data[name]
    if coord.ndim != 1:
        raise ValueError(f"Coordinate {name!r} must be one-dimensional.")

    try:
        _ = coord.dt.year
    except (AttributeError, TypeError) as exc:
        raise TypeError(f"Coordinate {name!r} must be datetime-like.") from exc


def require_datetime_component(data: xr.DataArray, time_dim: str, component: str) -> None:
    """Validate a datetime accessor component such as ``month`` or ``dayofyear``."""
    require_datetime_coordinate(data, time_dim)
    try:
        getattr(data[time_dim].dt, component)
    except AttributeError as exc:
        raise ValueError(f"Unsupported datetime grouping component {component!r}.") from exc
