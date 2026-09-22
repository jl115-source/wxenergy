"""Small end-to-end wxenergy example using synthetic gridded weather data."""

from __future__ import annotations

import numpy as np
import wxenergy as wx
import xarray as xr


time = xr.date_range("2024-07-01", periods=8, freq="6h")
latitude = [35.0, 45.0, 55.0]
longitude = [0.0, 90.0, 180.0, 270.0]

values = 293.15 + np.arange(8)[:, None, None] * 0.25
values = np.broadcast_to(values, (8, 3, 4)).copy()

temperature = xr.DataArray(
    values,
    dims=("time", "latitude", "longitude"),
    coords={"time": time, "latitude": latitude, "longitude": longitude},
    name="t2m",
    attrs={"units": "K"},
)

temperature = wx.normalize_longitude(temperature)
temperature_c = wx.convert_temperature(temperature, "degC")
daily = wx.daily_mean(temperature_c)
weights = wx.cosine_latitude_weights(daily.latitude)
regional = wx.weighted_mean(daily, weights, dim=("latitude", "longitude"))
cdd = wx.cdd(regional, base=18.0)

print(xr.Dataset({"temperature": regional, "cdd": cdd}))
