Examples
========

Gridded temperature workflow
----------------------------

The following pattern is representative of the package's intended use: keep data in
xarray, normalize conventions once, then compose small transformations.

.. code-block:: python

   import xarray as xr
   import wxenergy as wx

   ds = xr.open_dataset("weather.nc")
   ds = wx.normalize_longitude(ds)

   temperature = wx.convert_temperature(ds.t2m, "degC")
   daily = wx.daily_mean(temperature)
   weights = wx.cosine_latitude_weights(daily.latitude)
   regional = wx.weighted_mean(
       daily,
       weights,
       dim=("latitude", "longitude"),
   )
   cdd = wx.cdd(regional, base=18.0)

Dask-backed data
----------------

The core functions use xarray operations and preserve lazy Dask arrays where the
underlying xarray operation is lazy.

.. code-block:: python

   ds = xr.open_dataset("weather.nc", chunks={"time": 24})
   temperature = wx.convert_temperature(ds.t2m, "degC")
   daily = wx.daily_mean(temperature)

   # No explicit computation is required until a result is needed.
   result = daily.compute()

Wind magnitude
--------------

.. code-block:: python

   wind = wx.wind_speed(ds.u10, ds.v10)

``u10`` and ``v10`` must have exactly aligned coordinates. Unit metadata is only
attached to the result when both components report the same units.

Climate anomalies
-----------------

.. code-block:: python

   climatology = wx.climatology(daily, groupby="dayofyear")
   anomaly = wx.anomaly(
       daily,
       climatology_data=climatology,
       groupby="dayofyear",
   )

A complete executable example is available in ``examples/basic_workflow.py`` in the
source repository.
