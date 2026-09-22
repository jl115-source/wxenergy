import numpy as np
import pytest
import xarray as xr

import wxenergy as wx


def test_climatology_and_anomaly_by_month():
    time = np.array(
        ["2023-01-15", "2023-02-15", "2024-01-15", "2024-02-15"],
        dtype="datetime64[ns]",
    )
    data = xr.DataArray(
        [1.0, 2.0, 3.0, 6.0],
        dims="time",
        coords={"time": time},
        name="t2m",
        attrs={"units": "degC"},
    )

    climo = wx.climatology(data, groupby="month")
    anom = wx.anomaly(data, climatology_data=climo, groupby="month")

    np.testing.assert_allclose(climo.values, [2.0, 4.0])
    np.testing.assert_allclose(anom.values, [-1.0, -2.0, 1.0, 2.0])
    assert climo.name == "t2m_climatology"
    assert anom.name == "t2m_anomaly"
    assert anom.attrs["units"] == "degC"


def test_climatology_rejects_non_datetime_coordinate():
    data = xr.DataArray([1.0, 2.0], dims="time", coords={"time": [0, 1]})

    with pytest.raises(TypeError, match="datetime-like"):
        wx.climatology(data)


def test_anomaly_rejects_incompatible_climatology():
    time = xr.date_range("2024-01-01", periods=2, freq="D")
    data = xr.DataArray([1.0, 2.0], dims="time", coords={"time": time})
    bad_climo = xr.DataArray([1.0], dims="season", coords={"season": ["DJF"]})

    with pytest.raises(ValueError, match="grouping coordinate"):
        wx.anomaly(data, climatology_data=bad_climo)


def test_degree_days_preserve_missing_values_and_metadata():
    data = xr.DataArray(
        [10.0, np.nan, 25.0],
        dims="time",
        name="t2m",
        attrs={"units": "degC"},
    )

    hdd = wx.hdd(data, base=18)
    cdd = wx.cdd(data, base=18)

    np.testing.assert_allclose(hdd.values, [8.0, np.nan, 0.0], equal_nan=True)
    np.testing.assert_allclose(cdd.values, [0.0, np.nan, 7.0], equal_nan=True)
    assert hdd.name == "t2m_hdd"
    assert cdd.name == "t2m_cdd"
    assert hdd.attrs["units"] == "degC"
    assert hdd.attrs["base_temperature"] == 18


def test_degree_days_reject_nonfinite_base():
    data = xr.DataArray([10.0], dims="time")

    with pytest.raises(ValueError, match="finite"):
        wx.hdd(data, base=np.inf)


def test_weighted_mean_reduces_weight_dimensions_by_default():
    data = xr.DataArray(
        [[10.0, 20.0], [30.0, 40.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [40.0, 50.0], "longitude": [-80.0, -70.0]},
        name="t2m",
    )
    weights = xr.DataArray([1.0, 3.0], dims="latitude", coords={"latitude": [40.0, 50.0]})

    result = wx.weighted_mean(data, weights)

    np.testing.assert_allclose(result.values, [25.0, 35.0])
    assert result.dims == ("longitude",)
    assert result.name == "t2m_weighted_mean"


def test_weighted_mean_rejects_unknown_dimensions():
    data = xr.DataArray([10.0, 20.0], dims="location")
    weights = xr.DataArray([1.0, 3.0], dims="station")

    with pytest.raises(ValueError, match="Weight dimensions"):
        wx.weighted_mean(data, weights)


def test_normalize_longitude_to_minus180_180_preserves_attrs():
    data = xr.DataArray(
        [1, 2, 3],
        dims="longitude",
        coords={"longitude": [0.0, 180.0, 270.0]},
    )
    data.longitude.attrs["units"] = "degrees_east"

    result = wx.normalize_longitude(data)

    np.testing.assert_allclose(result.longitude.values, [-180.0, -90.0, 0.0])
    assert result.longitude.attrs["units"] == "degrees_east"


def test_normalize_longitude_to_0_360():
    data = xr.DataArray(
        [1, 2, 3],
        dims="longitude",
        coords={"longitude": [-180.0, -90.0, 0.0]},
    )

    result = wx.normalize_longitude(data, target="0_360")

    np.testing.assert_allclose(result.longitude.values, [0.0, 180.0, 270.0])


def test_daily_aggregations():
    time = xr.date_range("2024-01-01", periods=8, freq="6h")
    data = xr.DataArray(
        np.arange(1.0, 9.0),
        dims="time",
        coords={"time": time},
        name="load_weather",
    )

    mean = wx.daily_mean(data)
    maximum = wx.daily_max(data)
    minimum = wx.daily_min(data)

    np.testing.assert_allclose(mean.values, [2.5, 6.5])
    np.testing.assert_allclose(maximum.values, [4.0, 8.0])
    np.testing.assert_allclose(minimum.values, [1.0, 5.0])
    assert mean.name == "load_weather_daily_mean"


def test_daily_aggregation_requires_datetime_coordinate():
    data = xr.DataArray([1.0, 2.0], dims="time", coords={"time": [0, 1]})

    with pytest.raises(TypeError, match="datetime-like"):
        wx.daily_mean(data)
