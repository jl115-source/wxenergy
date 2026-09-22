"""Unit conversion helpers."""

from __future__ import annotations

from typing import Literal

import xarray as xr

TemperatureUnit = Literal["K", "degC", "degF"]

_UNIT_ALIASES = {
    "k": "K",
    "kelvin": "K",
    "c": "degC",
    "degc": "degC",
    "celsius": "degC",
    "°c": "degC",
    "f": "degF",
    "degf": "degF",
    "fahrenheit": "degF",
    "°f": "degF",
}


def _canonical_temperature_unit(unit: str) -> TemperatureUnit:
    key = unit.strip().lower()
    try:
        return _UNIT_ALIASES[key]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported temperature unit {unit!r}; expected K, degC, or degF."
        ) from exc


def convert_temperature(
    data: xr.DataArray,
    to: TemperatureUnit,
    *,
    from_unit: str | None = None,
) -> xr.DataArray:
    """Convert temperature values between kelvin, Celsius, and Fahrenheit.

    Parameters
    ----------
    data:
        Temperature values.
    to:
        Target unit: ``"K"``, ``"degC"``, or ``"degF"``.
    from_unit:
        Source unit. If omitted, ``data.attrs["units"]`` is used.

    Returns
    -------
    xarray.DataArray
        Converted values with coordinates, name, and metadata preserved.

    Raises
    ------
    ValueError
        If the source unit is missing or either unit is unsupported.
    """
    source_raw = from_unit or data.attrs.get("units")
    if not source_raw:
        raise ValueError("Temperature units are required; set attrs['units'] or pass from_unit.")

    source = _canonical_temperature_unit(str(source_raw))
    target = _canonical_temperature_unit(to)

    if source == target:
        result = data.copy(deep=False)
    else:
        if source == "K":
            celsius = data - 273.15
        elif source == "degF":
            celsius = (data - 32.0) * (5.0 / 9.0)
        else:
            celsius = data

        if target == "K":
            result = celsius + 273.15
        elif target == "degF":
            result = celsius * (9.0 / 5.0) + 32.0
        else:
            result = celsius

    result.attrs = data.attrs.copy()
    result.attrs["units"] = target
    result.name = data.name
    return result
