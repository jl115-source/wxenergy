wxenergy
========

Weather and climate utilities for energy analysis.

``wxenergy`` provides small, composable functions for common weather and climate
processing tasks built around xarray. The package focuses on reusable data-processing
infrastructure rather than forecasting systems, data-vendor clients, or trading logic.

The public API is intentionally small. Functions accept standard xarray objects,
preserve labeled coordinates, and retain metadata where its meaning remains valid.

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting-started
   examples
   api

Current development series
--------------------------

``0.2.x`` adds stronger validation, lazy-array testing, unit conversion, wind-vector
utilities, spatial weighting, package-build validation, and generated API documentation.
The API may still change before a stable ``1.0`` release.

Project links
-------------

* `Source <https://github.com/jl115-source/wxenergy>`_
* `Issue tracker <https://github.com/jl115-source/wxenergy/issues>`_
* `Changelog <https://github.com/jl115-source/wxenergy/blob/main/CHANGELOG.md>`_
* `Contributing <https://github.com/jl115-source/wxenergy/blob/main/CONTRIBUTING.md>`_
