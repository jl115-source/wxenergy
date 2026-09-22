import numpy as np
import pytest
import xarray as xr

import wxenergy as wx


def test_climatology_rejects_unknown_datetime_component():
    time = xr.date_range("2024-01-01", periods=3, freq="D")
    data = xr.DataArray([1.0, 2.0, 3.0], dims="time", coords={"time": time})

    with pytest.raises(ValueError, match="Unsupported datetime grouping component"):
        wx.climatology(data, groupby="not_a_component")


def test_climatology_supports_custom_time_dimension():
    valid_time = xr.date_range("2024-01-01", periods=4, freq="D")
    data = xr.DataArray(
        [1.0, 3.0, 5.0, 7.0],
        dims="valid_time",
        coords={"valid_time": valid_time},
    )

    result = wx.climatology(data, time_dim="valid_time", groupby="month")

    np.testing.assert_allclose(result.values, [4.0])


def test_daily_mean_skipna_behavior():
    time = xr.date_range("2024-01-01", periods=4, freq="6h")
    data = xr.DataArray([1.0, np.nan, 3.0, 4.0], dims="time", coords={"time": time})

    skipped = wx.daily_mean(data, skipna=True)
    strict = wx.daily_mean(data, skipna=False)

    np.testing.assert_allclose(skipped.values, [8.0 / 3.0])
    assert np.isnan(strict.values[0])


def test_daily_mean_rejects_missing_time_dimension():
    data = xr.DataArray([1.0, 2.0], dims="sample")

    with pytest.raises(ValueError, match="Dimension 'time'"):
        wx.daily_mean(data)


def test_weighted_mean_explicit_dimension():
    data = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=("y", "x"),
    )
    weights = xr.DataArray([1.0, 3.0], dims="y")

    result = wx.weighted_mean(data, weights, dim="y")

    np.testing.assert_allclose(result.values, [2.5, 3.5])
    assert result.dims == ("x",)


def test_weighted_mean_rejects_unknown_reduction_dimension():
    data = xr.DataArray([1.0, 2.0], dims="x")
    weights = xr.DataArray([1.0, 1.0], dims="x")

    with pytest.raises(ValueError, match="Reduction dimensions"):
        wx.weighted_mean(data, weights, dim="y")


def test_weighted_mean_rejects_misaligned_shared_coordinates():
    data = xr.DataArray(
        [10.0, 20.0],
        dims="latitude",
        coords={"latitude": [40.0, 50.0]},
    )
    weights = xr.DataArray(
        [1.0, 2.0],
        dims="latitude",
        coords={"latitude": [40.0, 55.0]},
    )

    with pytest.raises(ValueError, match="align exactly"):
        wx.weighted_mean(data, weights)


def test_cosine_latitude_weights_reject_out_of_range_values():
    latitude = xr.DataArray([0.0, 91.0], dims="latitude")

    with pytest.raises(ValueError, match="within"):
        wx.cosine_latitude_weights(latitude)


def test_cosine_latitude_weights_reject_multidimensional_input():
    latitude = xr.DataArray([[10.0, 20.0]], dims=("y", "x"))

    with pytest.raises(ValueError, match="one-dimensional"):
        wx.cosine_latitude_weights(latitude)


def test_normalize_longitude_requires_coordinate():
    data = xr.DataArray([1.0, 2.0], dims="x")

    with pytest.raises(KeyError, match="Longitude coordinate"):
        wx.normalize_longitude(data)


def test_normalize_longitude_rejects_bad_target():
    data = xr.DataArray([1.0], dims="longitude", coords={"longitude": [0.0]})

    with pytest.raises(ValueError, match="target must be"):
        wx.normalize_longitude(data, target="bad")  # type: ignore[arg-type]
