import numpy as np
import xarray as xr

import wxenergy as wx


def test_climatology_and_anomaly():
    time = xr.date_range("2024-01-01", periods=4, freq="D")
    data = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time", coords={"time": time}, name="t2m")

    climo = wx.climatology(data)
    anom = wx.anomaly(data)

    assert climo.sizes["dayofyear"] == 4
    np.testing.assert_allclose(anom.values, 0.0)
    assert anom.name == "t2m_anomaly"


def test_degree_days():
    data = xr.DataArray([10.0, 18.0, 25.0], dims="time")

    np.testing.assert_allclose(wx.hdd(data, base=18).values, [8.0, 0.0, 0.0])
    np.testing.assert_allclose(wx.cdd(data, base=18).values, [0.0, 0.0, 7.0])


def test_weighted_mean():
    data = xr.DataArray([10.0, 20.0], dims="location")
    weights = xr.DataArray([1.0, 3.0], dims="location")

    result = wx.weighted_mean(data, weights)

    assert float(result) == 17.5


def test_normalize_longitude():
    data = xr.DataArray(
        [1, 2, 3],
        dims="longitude",
        coords={"longitude": [0.0, 180.0, 270.0]},
    )

    result = wx.normalize_longitude(data)

    np.testing.assert_allclose(result.longitude.values, [-180.0, -90.0, 0.0])
