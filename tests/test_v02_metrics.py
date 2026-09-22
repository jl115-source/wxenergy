import numpy as np
import pytest
import xarray as xr

import wxenergy as wx


def test_degree_hours_heating_and_cooling():
    data = xr.DataArray([10.0, 18.0, 22.0, np.nan], dims="time", name="t2m")

    heating = wx.degree_hours(data, base=18.0, mode="heating")
    cooling = wx.degree_hours(data, base=18.0, mode="cooling")

    np.testing.assert_allclose(heating.values, [8.0, 0.0, 0.0, np.nan], equal_nan=True)
    np.testing.assert_allclose(cooling.values, [0.0, 0.0, 4.0, np.nan], equal_nan=True)
    assert heating.name == "t2m_hdh"
    assert cooling.name == "t2m_cdh"


def test_degree_hours_rejects_unknown_mode():
    data = xr.DataArray([10.0], dims="time")

    with pytest.raises(ValueError, match="mode must be"):
        wx.degree_hours(data, mode="invalid")  # type: ignore[arg-type]


def test_ensemble_probability_comparisons():
    data = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="member", name="t2m")

    assert float(wx.ensemble_probability(data, 2.0, comparison="gt")) == 0.5
    assert float(wx.ensemble_probability(data, 2.0, comparison="ge")) == 0.75
    assert float(wx.ensemble_probability(data, 2.0, comparison="lt")) == 0.25
    assert float(wx.ensemble_probability(data, 2.0, comparison="le")) == 0.5


def test_ensemble_probability_metadata():
    data = xr.DataArray([1.0, 2.0, 3.0], dims="member", name="wind")

    result = wx.ensemble_probability(data, 2.5)

    assert result.name == "wind_ensemble_probability"
    assert result.attrs["units"] == "1"
    assert result.attrs["threshold"] == 2.5
    assert result.attrs["comparison"] == "ge"


def test_ensemble_probability_rejects_invalid_comparison():
    data = xr.DataArray([1.0, 2.0], dims="member")

    with pytest.raises(ValueError, match="comparison must be"):
        wx.ensemble_probability(data, 1.0, comparison="eq")  # type: ignore[arg-type]
