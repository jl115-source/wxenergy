"""Ensemble forecast statistics."""

from __future__ import annotations

from collections.abc import Iterable

import xarray as xr


def _validate_member_dim(data: xr.DataArray, member_dim: str) -> None:
    if member_dim not in data.dims:
        raise ValueError(f"Ensemble member dimension {member_dim!r} was not found.")


def ensemble_mean(
    data: xr.DataArray,
    *,
    member_dim: str = "member",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate the ensemble mean.

    Parameters
    ----------
    data
        Ensemble data containing ``member_dim``.
    member_dim
        Name of the ensemble-member dimension.
    skipna
        Whether missing values should be skipped. ``None`` uses xarray's default.

    Returns
    -------
    xarray.DataArray
        Mean across ensemble members with remaining coordinates preserved.
    """
    _validate_member_dim(data, member_dim)
    result = data.mean(member_dim, skipna=skipna, keep_attrs=True)
    if data.name:
        result.name = f"{data.name}_ensemble_mean"
    return result


def ensemble_spread(
    data: xr.DataArray,
    *,
    member_dim: str = "member",
    ddof: int = 0,
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate ensemble standard deviation.

    Parameters
    ----------
    data
        Ensemble data containing ``member_dim``.
    member_dim
        Name of the ensemble-member dimension.
    ddof
        Delta degrees of freedom passed to :meth:`xarray.DataArray.std`.
    skipna
        Whether missing values should be skipped. ``None`` uses xarray's default.
    """
    _validate_member_dim(data, member_dim)
    if ddof < 0:
        raise ValueError("ddof must be non-negative")

    result = data.std(member_dim, ddof=ddof, skipna=skipna, keep_attrs=True)
    if data.name:
        result.name = f"{data.name}_ensemble_spread"
    return result


def ensemble_quantile(
    data: xr.DataArray,
    q: float | Iterable[float],
    *,
    member_dim: str = "member",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate ensemble quantiles.

    ``q`` follows xarray/NumPy convention and must contain probabilities in the
    closed interval ``[0, 1]``.
    """
    _validate_member_dim(data, member_dim)
    quantiles = [float(q)] if isinstance(q, (int, float)) else [float(value) for value in q]
    if not quantiles or any(value < 0.0 or value > 1.0 for value in quantiles):
        raise ValueError("q must contain one or more values between 0 and 1")

    q_arg: float | list[float] = quantiles[0] if isinstance(q, (int, float)) else quantiles
    result = data.quantile(q_arg, dim=member_dim, skipna=skipna, keep_attrs=True)
    if data.name:
        result.name = f"{data.name}_ensemble_quantile"
    return result
