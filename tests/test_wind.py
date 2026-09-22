import numpy as np
import pytest
import xarray as xr

import wxenergy as wx


def test_wind_speed_magnitude_and_metadata():
    u = xr.DataArray(
        [3.0, 5.0],
        dims="time",
        coords={"time": [0, 1]},
        attrs={"units": "m s-1"},
    )
    v = xr.DataArray(
        [4.0, 12.0],
        dims="time",
        coords={"time": [0, 1]},
        attrs={"units": "m s-1"},
    )

    result = wx.wind_speed(u, v)

    np.testing.assert_allclose(result.values, [5.0, 13.0])
    assert result.name == "wind_speed"
    assert result.attrs["long_name"] == "wind speed"
    assert result.attrs["units"] == "m s-1"


def test_wind_speed_preserves_missing_values():
    u = xr.DataArray([3.0, np.nan], dims="time")
    v = xr.DataArray([4.0, 2.0], dims="time")

    result = wx.wind_speed(u, v)

    np.testing.assert_allclose(result.values, [5.0, np.nan], equal_nan=True)


def test_wind_speed_requires_exact_shared_coordinates():
    u = xr.DataArray([3.0, 4.0], dims="time", coords={"time": [0, 1]})
    v = xr.DataArray([4.0, 3.0], dims="time", coords={"time": [1, 2]})

    with pytest.raises(ValueError, match="align exactly"):
        wx.wind_speed(u, v)


def test_wind_speed_does_not_claim_units_when_components_disagree():
    u = xr.DataArray([3.0], dims="time", attrs={"units": "m s-1"})
    v = xr.DataArray([4.0], dims="time", attrs={"units": "knots"})

    result = wx.wind_speed(u, v)

    assert "units" not in result.attrs
