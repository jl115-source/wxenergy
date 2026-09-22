"""Spatial utilities for gridded weather and climate data."""

from __future__ import annotations

from collections.abc import Hashable, Iterable

import numpy as np
import xarray as xr


def cosine_latitude_weights(latitude: xr.DataArray) -> xr.DataArray:
    """Return cosine-of-latitude area weights.

    Parameters
    ----------
    latitude
        One-dimensional latitude coordinate in degrees north.

    Returns
    -------
    xarray.DataArray
        Dimensionless weights with the same coordinate and dimension as ``latitude``.

    Raises
    ------
    ValueError
        If latitude is not one-dimensional or contains values outside [-90, 90].
    """
    if latitude.ndim != 1:
        raise ValueError("latitude must be one-dimensional")
    if bool(((latitude < -90) | (latitude > 90)).any()):
        raise ValueError("latitude values must lie within [-90, 90] degrees")

    values = np.cos(np.deg2rad(np.asarray(latitude.values)))
    weights = latitude.copy(data=values)
    weights.name = "latitude_weight"
    weights.attrs = {
        "long_name": "cosine latitude area weight",
        "units": "1",
    }
    return weights


def weighted_mean(
    data: xr.DataArray,
    weights: xr.DataArray,
    *,
    dim: Hashable | Iterable[Hashable] | None = None,
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate a weighted mean with exact labeled-coordinate alignment.

    Parameters
    ----------
    data
        Data to aggregate.
    weights
        Weights whose dimensions must be a subset of ``data`` dimensions. Coordinates
        on shared dimensions must match exactly.
    dim
        Dimension or dimensions to reduce. If omitted, all dimensions present in
        ``weights`` are reduced.
    skipna
        Whether missing values in ``data`` should be skipped. ``None`` uses xarray's
        dtype-dependent default.

    Returns
    -------
    xarray.DataArray
        Weighted mean with unreduced dimensions preserved.

    Raises
    ------
    ValueError
        If weight dimensions are not present in ``data``, shared coordinates do not
        align exactly, or a requested reduction dimension is absent.
    """
    unknown_weight_dims = set(weights.dims) - set(data.dims)
    if unknown_weight_dims:
        dims = ", ".join(sorted(map(str, unknown_weight_dims)))
        raise ValueError(f"Weight dimensions are not present in data: {dims}")

    try:
        _, aligned_weights = xr.align(data, weights, join="exact", copy=False)
    except ValueError as exc:
        raise ValueError("Weight coordinates must align exactly with data.") from exc

    if dim is None:
        reduce_dims: list[Hashable] = list(aligned_weights.dims)
    elif isinstance(dim, Iterable) and not isinstance(dim, (str, bytes)):
        reduce_dims = list(dim)
    else:
        reduce_dims = [dim]

    unknown_reduce_dims = set(reduce_dims) - set(data.dims)
    if unknown_reduce_dims:
        dims = ", ".join(sorted(map(str, unknown_reduce_dims)))
        raise ValueError(f"Reduction dimensions are not present in data: {dims}")

    result = data.weighted(aligned_weights).mean(
        dim=reduce_dims,
        skipna=skipna,
        keep_attrs=True,
    )
    if data.name:
        result.name = f"{data.name}_weighted_mean"
    return result
