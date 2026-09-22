"""Small ensemble-weather workflow using synthetic data."""

from __future__ import annotations

import numpy as np
import xarray as xr

import wxenergy as wx

member = np.arange(10)
time = xr.date_range("2026-07-01", periods=12, freq="6h")
latitude = np.array([32.0, 38.0, 44.0])
longitude = np.array([250.0, 260.0, 270.0, 280.0])

rng = np.random.default_rng(7)
shape = (member.size, time.size, latitude.size, longitude.size)
temperature = 298.0 + rng.normal(0.0, 3.0, size=shape)

forecast = xr.DataArray(
    temperature,
    dims=("member", "time", "latitude", "longitude"),
    coords={
        "member": member,
        "time": time,
        "latitude": latitude,
        "longitude": longitude,
    },
    name="t2m",
    attrs={"units": "K"},
)

forecast = wx.normalize_longitude(forecast)
forecast_c = wx.convert_temperature(forecast, "degC")
mean = wx.ensemble_mean(forecast_c)
spread = wx.ensemble_spread(forecast_c)
p10_p50_p90 = wx.ensemble_quantile(forecast_c, [0.1, 0.5, 0.9])

daily_mean = wx.daily_mean(mean)
weights = wx.cosine_latitude_weights(daily_mean.latitude)
regional_daily_mean = wx.weighted_mean(
    daily_mean,
    weights,
    dim=("latitude", "longitude"),
)

print(regional_daily_mean)
print(spread.mean().item())
print(p10_p50_p90.quantile.values)
