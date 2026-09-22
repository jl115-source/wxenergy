import numpy as np
import pytest
import xarray as xr

import wxenergy as wx


def test_convert_temperature_celsius_to_kelvin_preserves_structure():
    data = xr.DataArray(
        [0.0, 100.0, np.nan],
        dims="time",
        coords={"time": [0, 1, 2]},
        name="t2m",
        attrs={"units": "degC", "source": "test"},
    )

    result = wx.convert_temperature(data, "K")

    np.testing.assert_allclose(result.values[:2], [273.15, 373.15])
    assert np.isnan(result.values[2])
    assert result.identical(result.assign_attrs(result.attrs))
    assert result.name == "t2m"
    assert result.attrs == {"units": "K", "source": "test"}


def test_convert_temperature_fahrenheit_to_celsius():
    data = xr.DataArray([32.0, 212.0], dims="sample", attrs={"units": "degF"})

    result = wx.convert_temperature(data, "degC")

    np.testing.assert_allclose(result.values, [0.0, 100.0])
    assert result.attrs["units"] == "degC"


def test_convert_temperature_accepts_common_aliases():
    data = xr.DataArray([273.15], dims="sample", attrs={"units": "kelvin"})

    result = wx.convert_temperature(data, "degF")

    np.testing.assert_allclose(result.values, [32.0])


def test_convert_temperature_allows_explicit_source_unit():
    data = xr.DataArray([0.0], dims="sample")

    result = wx.convert_temperature(data, "degF", from_unit="Celsius")

    np.testing.assert_allclose(result.values, [32.0])


def test_convert_temperature_requires_source_units():
    data = xr.DataArray([0.0], dims="sample")

    with pytest.raises(ValueError, match="Temperature units are required"):
        wx.convert_temperature(data, "K")


def test_convert_temperature_rejects_unknown_unit():
    data = xr.DataArray([0.0], dims="sample", attrs={"units": "degC"})

    with pytest.raises(ValueError, match="Unsupported temperature unit"):
        wx.convert_temperature(data, "rankine")  # type: ignore[arg-type]
