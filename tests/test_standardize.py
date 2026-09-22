import numpy as np
import pytest
import xarray as xr

import wxenergy as wx


def test_standardize_renames_lat_lon():
    data = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=("lat", "lon"),
        coords={"lat": [40.0, 50.0], "lon": [0.0, 180.0, 270.0]},
    )

    result = wx.standardize(data)

    assert result.dims == ("latitude", "longitude")
    np.testing.assert_allclose(result.latitude, [40.0, 50.0])
    np.testing.assert_allclose(result.longitude, [0.0, 180.0, 270.0])


def test_standardize_can_normalize_longitude():
    data = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims="lon",
        coords={"lon": [0.0, 180.0, 270.0]},
    )

    result = wx.standardize(data, longitude_target="-180_180")

    np.testing.assert_allclose(result.longitude, [-180.0, -90.0, 0.0])


def test_standardize_rejects_ambiguous_coordinate_names():
    data = xr.Dataset(coords={"lat": [1.0], "latitude": [1.0]})

    with pytest.raises(ValueError, match="both names are present"):
        wx.standardize(data)


def test_standardize_does_not_mutate_input():
    data = xr.DataArray([1.0], dims="lat", coords={"lat": [40.0]})

    result = wx.standardize(data)

    assert data.dims == ("lat",)
    assert result.dims == ("latitude",)
