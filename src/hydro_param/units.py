"""Unit conversion registry for hydrologic parameterization.

Centralize all unit conversions (m to ft, mm to in, C to F, etc.) in a
single registry to prevent bugs from scattered inline magic numbers.
Used by both derivation plugins and output formatters.

Hydrologic models often require specific unit systems.  For example,
PRMS/pywatershed uses imperial units internally (feet, inches, degrees
Fahrenheit, acres), while source datasets typically provide data in SI
units (metres, millimetres, degrees Celsius, square metres).  This
module provides a declarative registry so that conversion logic is defined
once and applied consistently.

Notes
-----
Conversions are registered at module import time.  The ``convert()``
function is the primary public interface -- callers should never need to
access the internal registry dict directly.

All conversion functions are vectorized (they accept and return NumPy
arrays), enabling efficient batch application over large feature sets.

See Also
--------
hydro_param.sir : SIR normalization, which uses a separate conversion
    mechanism for source-to-canonical transforms (log10, Kelvin).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class UnitConversion:
    """Represent a single registered unit conversion.

    Immutable record binding a source/target unit pair to a vectorized
    conversion function.

    Parameters
    ----------
    from_unit : str
        Source unit identifier (e.g., ``"m"``, ``"C"``, ``"mm"``).
    to_unit : str
        Target unit identifier (e.g., ``"ft"``, ``"F"``, ``"in"``).
    fn : Callable[[NDArray], NDArray]
        Vectorized conversion function.  Must accept a NumPy array of
        float values and return a NumPy array of the same shape.
    description : str
        Human-readable description of the conversion (e.g.,
        ``"meters to feet"``).
    """

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
    """Register a unit conversion function in the module-level registry.

    Add a new ``(from_unit, to_unit) -> fn`` mapping.  If the pair already
    exists, the previous registration is silently overwritten.

    Parameters
    ----------
    from_unit : str
        Source unit identifier (e.g., ``"m"``, ``"C"``, ``"W/m2"``).
    to_unit : str
        Target unit identifier (e.g., ``"ft"``, ``"F"``, ``"Langleys/day"``).
    fn : Callable[[NDArray], NDArray]
        Vectorized conversion function.  Must accept a NumPy float array
        and return a NumPy float array of the same shape.
    description : str
        Human-readable description (e.g., ``"meters to feet"``).  Used by
        ``list_conversions()`` for introspection.

    Examples
    --------
    >>> register("km", "mi", lambda v: v * 0.621371, "kilometers to miles")
    """
    _CONVERSIONS[(from_unit, to_unit)] = UnitConversion(
        from_unit=from_unit, to_unit=to_unit, fn=fn, description=description
    )


def convert(
    values: NDArray[np.floating],
    from_unit: str,
    to_unit: str,
) -> NDArray[np.floating]:
    """Apply a registered unit conversion to an array of values.

    If ``from_unit == to_unit``, return the input array unchanged (no-op
    passthrough).  Otherwise, look up the registered conversion function
    and apply it element-wise.

    Parameters
    ----------
    values : NDArray[np.floating]
        Array of values to convert, in ``from_unit``.
    from_unit : str
        Source unit identifier (e.g., ``"m"``, ``"C"``).
    to_unit : str
        Target unit identifier (e.g., ``"ft"``, ``"F"``).

    Returns
    -------
    NDArray[np.floating]
        Converted values in ``to_unit``.  Same shape as input.

    Raises
    ------
    KeyError
        If no conversion is registered for the ``(from_unit, to_unit)`` pair.

    Examples
    --------
    >>> import numpy as np
    >>> convert(np.array([100.0]), "m", "ft")  # 100 m -> 328.084 ft
    """
    if from_unit == to_unit:
        return values
    key = (from_unit, to_unit)
    if key not in _CONVERSIONS:
        raise KeyError(f"No conversion registered: {from_unit} -> {to_unit}")
    return _CONVERSIONS[key].fn(values)


def list_conversions() -> list[tuple[str, str, str]]:
    """List all registered unit conversions for introspection.

    Useful for CLI ``datasets info`` output, debugging, and verifying that
    expected conversions are available before running a pipeline.

    Returns
    -------
    list[tuple[str, str, str]]
        List of ``(from_unit, to_unit, description)`` tuples, one per
        registered conversion, in insertion order.
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
