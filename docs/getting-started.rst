Getting started
===============

Installation
------------

During development, install directly from GitHub::

   pip install git+https://github.com/jl115-source/wxenergy.git

For local development::

   git clone https://github.com/jl115-source/wxenergy.git
   cd wxenergy
   pip install -e ".[dev,docs]"

Quickstart
----------

.. code-block:: python

   import xarray as xr
   import wxenergy as wx

   ds = xr.open_dataset("temperature.nc")

   ds = wx.normalize_longitude(ds)
   daily = wx.daily_mean(ds.t2m)
   climo = wx.climatology(daily, groupby="dayofyear")
   anom = wx.anomaly(daily, climatology_data=climo)
   hdd = wx.hdd(daily, base=18.0)

Design principles
-----------------

* Accept and return standard xarray objects.
* Preserve coordinates and metadata where practical.
* Prefer small functions with explicit behavior over large abstractions.
* Raise standard Python exceptions with actionable messages.
* Keep weather-processing infrastructure separate from forecasting and trading logic.
