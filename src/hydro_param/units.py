"""Unit conversion registry for hydrologic parameterization.

Centralizes all unit conversions (m→ft, mm→in, C→F, etc.) to prevent
bugs from scattered inline magic numbers. Used by both derivation
plugins and output formatters.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UnitConversion:
    """A registered unit conversion."""

    from_unit: str
    to_unit: str
    fn: Callable[[NDArray[np.floating]], NDArray[np.floating]]
    description: str = ""


# Module-level registry: (from_unit, to_unit) → UnitConversion
_CONVERSIONS: dict[tuple[str, str], UnitConversion] = {}


def register(
    from_unit: str,
    to_unit: str,
    fn: Callable[[NDArray[np.floating]], NDArray[np.floating]],
    description: str = "",
) -> None:
    """Register a unit conversion function.

    Parameters
    ----------
    from_unit
        Source unit identifier (e.g., ``"m"``, ``"C"``).
    to_unit
        Target unit identifier (e.g., ``"ft"``, ``"F"``).
    fn
        Vectorized conversion function accepting and returning NumPy arrays.
    description
        Human-readable description of the conversion.
    """
    _CONVERSIONS[(from_unit, to_unit)] = UnitConversion(
        from_unit=from_unit, to_unit=to_unit, fn=fn, description=description
    )


def convert(
    values: NDArray[np.floating],
    from_unit: str,
    to_unit: str,
) -> NDArray[np.floating]:
    """Apply a registered unit conversion.

    Parameters
    ----------
    values
        Array of values to convert.
    from_unit
        Source unit identifier.
    to_unit
        Target unit identifier.

    Returns
    -------
    NDArray
        Converted values.

    Raises
    ------
    KeyError
        If no conversion is registered for the given unit pair.
    """
    if from_unit == to_unit:
        return values
    key = (from_unit, to_unit)
    if key not in _CONVERSIONS:
        raise KeyError(f"No conversion registered: {from_unit} -> {to_unit}")
    return _CONVERSIONS[key].fn(values)


def list_conversions() -> list[tuple[str, str, str]]:
    """List all registered conversions.

    Returns
    -------
    list[tuple[str, str, str]]
        List of ``(from_unit, to_unit, description)`` tuples.
    """
    return [(c.from_unit, c.to_unit, c.description) for c in _CONVERSIONS.values()]


# ---------------------------------------------------------------------------
# Standard conversions
# ---------------------------------------------------------------------------

# Length
register("m", "ft", lambda v: v * 3.28084, "meters to feet")
register("ft", "m", lambda v: v / 3.28084, "feet to meters")

# Precipitation / depth
register("mm", "in", lambda v: v / 25.4, "millimeters to inches")
register("in", "mm", lambda v: v * 25.4, "inches to millimeters")

# Temperature
register("C", "F", lambda v: v * 9.0 / 5.0 + 32.0, "Celsius to Fahrenheit")
register("F", "C", lambda v: (v - 32.0) * 5.0 / 9.0, "Fahrenheit to Celsius")
register("K", "F", lambda v: (v - 273.15) * 9.0 / 5.0 + 32.0, "Kelvin to Fahrenheit")
register("K", "C", lambda v: v - 273.15, "Kelvin to Celsius")

# Area
register("m2", "acres", lambda v: v * 0.000247105, "square meters to acres")
register("acres", "m2", lambda v: v / 0.000247105, "acres to square meters")

# Angular
register("deg", "rad", lambda v: np.radians(v), "degrees to radians")
register("rad", "deg", lambda v: np.degrees(v), "radians to degrees")

# Irradiance
register("W/m2", "Langleys/day", lambda v: v * 2.065, "watts per square meter to Langleys per day")

# Log-transform conversions (SIR normalization: source -> canonical SI)
register("log10(cm/hr)", "cm/hr", lambda v: np.power(10.0, v), "log10 Ksat to linear cm/hr")
register("log10(kPa)", "kPa", lambda v: np.power(10.0, v), "log10 pressure to linear kPa")
register("log10(%)", "%", lambda v: np.power(10.0, v), "log10 percent to linear %")
register("log10(kPa^-1)", "kPa^-1", lambda v: np.power(10.0, v), "log10 inverse pressure to linear")
