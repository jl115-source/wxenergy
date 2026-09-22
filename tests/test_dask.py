import dask.array as da
import numpy as np
import xarray as xr

import wxenergy as wx


def test_weighted_mean_remains_lazy_with_dask():
    data = xr.DataArray(
        da.from_array(np.arange(12.0).reshape(3, 4), chunks=(3, 2)),
        dims=("latitude", "longitude"),
        coords={"latitude": [30.0, 40.0, 50.0], "longitude": [0.0, 10.0, 20.0, 30.0]},
    )
    weights = wx.cosine_latitude_weights(data.latitude)

    result = wx.weighted_mean(data, weights, dim="latitude")

    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(result.compute().values, data.weighted(weights).mean("latitude").values)


def test_climatology_remains_lazy_with_dask():
    time = xr.date_range("2024-01-01", periods=48, freq="h")
    data = xr.DataArray(
        da.from_array(np.arange(48.0), chunks=12),
        dims="time",
        coords={"time": time},
    )

    result = wx.climatology(data, groupby="hour")

    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(result.compute().values, np.arange(24.0) + 12.0)


def test_ensemble_probability_remains_lazy_with_dask():
    data = xr.DataArray(
        da.from_array(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]), chunks=(1, 3)),
        dims=("member", "time"),
    )

    result = wx.ensemble_probability(data, 3.0, comparison="ge")

    assert isinstance(result.data, da.Array)
    np.testing.assert_allclose(result.compute().values, [0.5, 0.5, 1.0])
