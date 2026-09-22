"""Integration tests that exercise realistic weather-processing workflows."""

from __future__ import annotations

import dask.array as da
import numpy as np
import xarray as xr

import wxenergy as wx


def _sample_weather_dataset() -> xr.Dataset:
    time = xr.date_range("2024-07-01", periods=16, freq="6h")
    latitude = xr.DataArray([30.0, 40.0, 50.0], dims="latitude", name="latitude")
    longitude = xr.DataArray([0.0, 90.0, 180.0, 270.0], dims="longitude", name="longitude")

    hour_index = np.arange(time.size, dtype=float)[:, None, None]
    lat_term = (latitude.values[None, :, None] - 40.0) * -0.15
    lon_term = np.cos(np.deg2rad(longitude.values))[None, None, :]

    temperature = 295.0 + 4.0 * np.sin(2.0 * np.pi * hour_index / 4.0) + lat_term + lon_term
    u10 = np.broadcast_to(3.0 + 0.1 * hour_index, temperature.shape)
    v10 = np.broadcast_to(4.0 + 0.05 * hour_index, temperature.shape)

    return xr.Dataset(
        {
            "t2m": xr.DataArray(
                temperature,
                dims=("time", "latitude", "longitude"),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                attrs={"units": "K", "long_name": "2 metre temperature"},
            ),
            "u10": xr.DataArray(
                u10,
                dims=("time", "latitude", "longitude"),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                attrs={"units": "m s-1"},
            ),
            "v10": xr.DataArray(
                v10,
                dims=("time", "latitude", "longitude"),
                coords={"time": time, "latitude": latitude, "longitude": longitude},
                attrs={"units": "m s-1"},
            ),
        }
    )


def test_end_to_end_gridded_weather_workflow_is_lazy_until_compute() -> None:
    ds = _sample_weather_dataset().chunk({"time": 4, "latitude": 2, "longitude": 2})
    ds = wx.normalize_longitude(ds)

    temperature_c = wx.convert_temperature(ds.t2m, "degC")
    daily_temperature = wx.daily_mean(temperature_c)
    weights = wx.cosine_latitude_weights(daily_temperature.latitude)
    regional_temperature = wx.weighted_mean(
        daily_temperature,
        weights,
        dim=("latitude", "longitude"),
    )
    cooling = wx.cdd(regional_temperature, base=18.0)
    wind = wx.wind_speed(ds.u10, ds.v10)

    assert isinstance(regional_temperature.data, da.Array)
    assert isinstance(cooling.data, da.Array)
    assert isinstance(wind.data, da.Array)
    assert regional_temperature.dims == ("time",)
    assert regional_temperature.sizes["time"] == 4
    assert cooling.attrs["base_temperature"] == 18.0
    assert wind.attrs["units"] == "m s-1"

    computed = regional_temperature.compute()
    assert np.isfinite(computed.values).all()
    assert computed.attrs["units"] == "degC"


def test_netcdf_round_trip_and_processing(tmp_path) -> None:
    source = _sample_weather_dataset()
    path = tmp_path / "sample-weather.nc"
    source.to_netcdf(path, engine="scipy")

    with xr.open_dataset(path, engine="scipy") as reopened:
        normalized = wx.normalize_longitude(reopened)
        temperature_c = wx.convert_temperature(normalized.t2m, "degC")
        daily = wx.daily_mean(temperature_c)

        assert daily.sizes["time"] == 4
        assert daily.attrs["units"] == "degC"
        np.testing.assert_allclose(
            normalized.longitude.values,
            [-180.0, -90.0, 0.0, 90.0],
        )


def test_climatology_anomaly_pipeline_on_two_year_dataset() -> None:
    time = xr.date_range("2023-01-01", periods=730, freq="D")
    seasonal_cycle = 10.0 + 12.0 * np.sin(2.0 * np.pi * np.arange(time.size) / 365.0)
    offset = np.where(time.year == 2024, 2.0, 0.0)
    data = xr.DataArray(
        seasonal_cycle + offset,
        dims="time",
        coords={"time": time},
        name="t2m",
        attrs={"units": "degC"},
    )

    climatology = wx.climatology(data, groupby="dayofyear")
    anomaly = wx.anomaly(data, climatology_data=climatology, groupby="dayofyear")

    assert climatology.sizes["dayofyear"] in {365, 366}
    assert anomaly.dims == ("time",)
    assert anomaly.attrs["units"] == "degC"
    assert np.isfinite(anomaly.values).all()
