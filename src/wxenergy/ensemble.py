"""Ensemble forecast statistics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import xarray as xr

Comparison = Literal["gt", "ge", "lt", "le"]


def _validate_member_dim(data: xr.DataArray, member_dim: str) -> None:
    if member_dim not in data.dims:
        raise ValueError(f"Ensemble member dimension {member_dim!r} was not found.")


def ensemble_mean(
    data: xr.DataArray,
    *,
    member_dim: str = "member",
    skipna: bool | None = None,
) -> xr.DataArray:
    """Calculate the ensemble mean."""
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
    """Calculate ensemble standard deviation."""
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


def ensemble_probability(
    data: xr.DataArray,
    threshold: float,
    *,
    comparison: Comparison = "ge",
    member_dim: str = "member",
) -> xr.DataArray:
    """Calculate the fraction of ensemble members meeting a threshold condition.

    Parameters
    ----------
    data
        Ensemble data containing ``member_dim``.
    threshold
        Threshold expressed in the same units as ``data``.
    comparison
        One of ``"gt"``, ``"ge"``, ``"lt"``, or ``"le"``.
    member_dim
        Name of the ensemble-member dimension.

    Returns
    -------
    xarray.DataArray
        Probability in the closed interval ``[0, 1]``.
    """
    _validate_member_dim(data, member_dim)
    operators = {
        "gt": data > threshold,
        "ge": data >= threshold,
        "lt": data < threshold,
        "le": data <= threshold,
    }
    try:
        condition = operators[comparison]
    except KeyError as exc:
        raise ValueError("comparison must be one of 'gt', 'ge', 'lt', or 'le'") from exc

    result = condition.mean(member_dim)
    result.attrs = {
        "long_name": "ensemble threshold probability",
        "threshold": threshold,
        "comparison": comparison,
        "units": "1",
    }
    if data.name:
        result.name = f"{data.name}_ensemble_probability"
    return result
