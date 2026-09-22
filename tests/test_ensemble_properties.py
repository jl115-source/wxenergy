"""Property-based tests for ensemble statistics."""

from __future__ import annotations

import numpy as np
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

import wxenergy as wx

FINITE = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@settings(max_examples=75)
@given(values=st.lists(FINITE, min_size=2, max_size=30))
def test_ensemble_mean_is_bounded_by_member_extrema(values: list[float]) -> None:
    data = xr.DataArray(values, dims="member")

    result = float(wx.ensemble_mean(data))

    assert min(values) <= result <= max(values)


@settings(max_examples=75)
@given(values=st.lists(FINITE, min_size=2, max_size=30))
def test_ensemble_spread_is_non_negative(values: list[float]) -> None:
    data = xr.DataArray(values, dims="member")

    result = float(wx.ensemble_spread(data))

    assert result >= 0.0


@settings(max_examples=50)
@given(values=st.lists(FINITE, min_size=2, max_size=30))
def test_ensemble_quantiles_are_monotonic(values: list[float]) -> None:
    data = xr.DataArray(values, dims="member")

    quantiles = wx.ensemble_quantile(data, [0.1, 0.5, 0.9]).values

    assert np.all(np.diff(quantiles) >= 0.0)
