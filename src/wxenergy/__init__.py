"""Weather and climate utilities for energy analysis."""

from importlib.metadata import version

from .climate import anomaly, climatology
from .energy import cdd, degree_hours, hdd
from .ensemble import ensemble_mean, ensemble_probability, ensemble_quantile, ensemble_spread
from .spatial import cosine_latitude_weights, weighted_mean
from .standardize import normalize_longitude
from .temporal import daily_max, daily_mean, daily_min
from .units import convert_temperature
from .wind import wind_speed

__all__ = [
    "anomaly",
    "cdd",
    "climatology",
    "convert_temperature",
    "cosine_latitude_weights",
    "daily_max",
    "daily_mean",
    "daily_min",
    "degree_hours",
    "ensemble_mean",
    "ensemble_probability",
    "ensemble_quantile",
    "ensemble_spread",
    "hdd",
    "normalize_longitude",
    "weighted_mean",
    "wind_speed",
]

__version__ = version("wxenergy")
