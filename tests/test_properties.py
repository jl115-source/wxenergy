"""Property-based tests for numerical invariants."""

from __future__ import annotations

import numpy as np
import xarray as xr
from hypothesis import given, settings, strategies as st

import wxenergy as wx


FINITE_TEMPERATURE = st.floats(
    min_value=-100.0,
    max_value=100.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
FINITE_LONGITUDE = st.floats(
    min_value=-1080.0,
    max_value=1080.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@settings(max_examples=50)
@given(
    values=st.lists(FINITE_TEMPERATURE, min_size=1, max_size=20),
    base=st.floats(
        min_value=-40.0,
        max_value=40.0,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
def test_hdd_cdd_partition_absolute_temperature_distance(values: list[float], base: float) -> None:
    data = xr.DataArray(values, dims="time")

    total = wx.hdd(data, base=base) + wx.cdd(data, base=base)

    np.testing.assert_allclose(total.values, np.abs(data.values - base), rtol=1e-12, atol=1e-12)


@settings(max_examples=50)
@given(values=st.lists(FINITE_TEMPERATURE, min_size=1, max_size=20))
def test_temperature_conversion_round_trip(values: list[float]) -> None:
    data = xr.DataArray(values, dims="time", attrs={"units": "degC"})

    round_trip = wx.convert_temperature(wx.convert_temperature(data, "K"), "degC")

    np.testing.assert_allclose(round_trip.values, data.values, rtol=1e-12, atol=1e-10)
    assert round_trip.attrs["units"] == "degC"


@settings(max_examples=50)
@given(values=st.lists(FINITE_LONGITUDE, min_size=1, max_size=20))
def test_normalized_longitudes_are_sorted_and_bounded(values: list[float]) -> None:
    data = xr.DataArray(
        np.arange(len(values)),
        dims="longitude",
        coords={"longitude": values},
    )

    result = wx.normalize_longitude(data, target="-180_180")
    longitude = result.longitude.values

    assert np.all(longitude >= -180.0)
    assert np.all(longitude < 180.0)
    assert np.all(np.diff(longitude) >= 0.0)


@settings(max_examples=30)
@given(latitude=st.floats(min_value=0.0, max_value=90.0, allow_nan=False, allow_infinity=False))
def test_cosine_latitude_weights_are_hemispherically_symmetric(latitude: float) -> None:
    coordinate = xr.DataArray([-latitude, latitude], dims="latitude")

    weights = wx.cosine_latitude_weights(coordinate)

    np.testing.assert_allclose(weights.values[0], weights.values[1], rtol=1e-12, atol=1e-12)
    assert np.all(weights.values >= 0.0)
