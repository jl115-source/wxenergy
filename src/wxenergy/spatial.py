"""Spatial utilities for gridded weather and climate data."""

from __future__ import annotations

from collections.abc import Iterable

import xarray as xr


def weighted_mean(
    data: xr.DataArray,
    weights: xr.DataArray,
    *,
    dim: str | Iterable[str] | None = None,
) -> xr.DataArray:
    """Return an xarray weighted mean.

    If ``dim`` is omitted, the dimensions present in ``weights`` are reduced.
    """
    reduce_dims = list(weights.dims) if dim is None else dim
    return data.weighted(weights).mean(dim=reduce_dims, keep_attrs=True)
