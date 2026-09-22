"""Spatial utilities for gridded weather and climate data."""

from __future__ import annotations

from collections.abc import Iterable

import xarray as xr


def weighted_mean(
    data: xr.DataArray,
    weights: xr.DataArray,
    *,
    dim: str | Iterable[str] | None = None,
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate a weighted mean using xarray's labeled alignment.

    Parameters
    ----------
    data
        Data to aggregate.
    weights
        Weights whose dimensions must be a subset of ``data`` dimensions. Missing
        weights are rejected by xarray.
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
    """
    unknown_weight_dims = set(weights.dims) - set(data.dims)
    if unknown_weight_dims:
        dims = ", ".join(sorted(unknown_weight_dims))
        raise ValueError(f"Weight dimensions are not present in data: {dims}")

    if dim is None:
        reduce_dims: str | list[str] = list(weights.dims)
    elif isinstance(dim, str):
        reduce_dims = dim
    else:
        reduce_dims = list(dim)

    requested_dims = {reduce_dims} if isinstance(reduce_dims, str) else set(reduce_dims)
    unknown_reduce_dims = requested_dims - set(data.dims)
    if unknown_reduce_dims:
        dims = ", ".join(sorted(unknown_reduce_dims))
        raise ValueError(f"Reduction dimensions are not present in data: {dims}")

    result = data.weighted(weights).mean(
        dim=reduce_dims,
        skipna=skipna,
        keep_attrs=True,
    )
    if data.name:
        result.name = f"{data.name}_weighted_mean"
    return result
