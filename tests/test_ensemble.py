"""Tests for ensemble forecast statistics."""

from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

import wxenergy as wx


def _ensemble_cube() -> xr.DataArray:
    member = np.arange(5)
    time = xr.date_range("2026-01-01", periods=3, freq="6h")
    latitude = [35.0, 45.0]
    values = np.arange(30.0).reshape(5, 3, 2)
    return xr.DataArray(
        values,
        dims=("member", "time", "latitude"),
        coords={"member": member, "time": time, "latitude": latitude},
        name="t2m",
        attrs={"units": "K"},
    )


def test_ensemble_statistics_match_numpy() -> None:
    data = _ensemble_cube()

    mean = wx.ensemble_mean(data)
    spread = wx.ensemble_spread(data)
    quantile = wx.ensemble_quantile(data, [0.1, 0.5, 0.9])

    np.testing.assert_allclose(mean.values, np.mean(data.values, axis=0))
    np.testing.assert_allclose(spread.values, np.std(data.values, axis=0))
    np.testing.assert_allclose(
        quantile.values,
        np.quantile(data.values, [0.1, 0.5, 0.9], axis=0),
    )
    assert mean.dims == ("time", "latitude")
    assert quantile.dims == ("quantile", "time", "latitude")
    assert mean.attrs["units"] == "K"


def test_ensemble_statistics_support_custom_member_dimension() -> None:
    data = _ensemble_cube().rename(member="number")

    result = wx.ensemble_mean(data, member_dim="number")

    assert "number" not in result.dims
    assert result.name == "t2m_ensemble_mean"


def test_ensemble_statistics_remain_lazy_with_dask() -> None:
    data = _ensemble_cube().chunk({"member": 2, "time": 1})

    mean = wx.ensemble_mean(data)
    spread = wx.ensemble_spread(data)

    assert isinstance(mean.data, da.Array)
    assert isinstance(spread.data, da.Array)
    np.testing.assert_allclose(mean.compute().values, np.mean(data.compute().values, axis=0))


def test_ensemble_mean_respects_skipna() -> None:
    data = _ensemble_cube().copy()
    data[0, 0, 0] = np.nan

    skipped = wx.ensemble_mean(data, skipna=True)
    propagated = wx.ensemble_mean(data, skipna=False)

    assert np.isfinite(skipped.isel(time=0, latitude=0))
    assert np.isnan(propagated.isel(time=0, latitude=0))


def test_ensemble_functions_validate_member_dimension() -> None:
    data = xr.DataArray([1.0, 2.0], dims="time")

    with pytest.raises(ValueError, match="member"):
        wx.ensemble_mean(data)


def test_ensemble_spread_validates_ddof() -> None:
    with pytest.raises(ValueError, match="ddof"):
        wx.ensemble_spread(_ensemble_cube(), ddof=-1)


@pytest.mark.parametrize("q", [-0.1, 1.1, [], [0.5, 2.0]])
def test_ensemble_quantile_validates_probability(q) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        wx.ensemble_quantile(_ensemble_cube(), q)
