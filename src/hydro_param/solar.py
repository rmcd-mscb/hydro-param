"""Swift (1976) clear-sky potential solar radiation on sloped surfaces.

Ports pywatershed's ``PRMSSolarGeometry.compute_soltab()`` into a standalone
pure-function module.  The algorithm computes potential clear-sky solar
radiation on sloped and horizontal surfaces for every day of the year (1--366).

Output units are Langleys (cal/cm2/day) for radiation and hours for sunlight
duration.

References
----------
Swift, L.W., 1976, Algorithm for solar radiation on mountain slopes:
    Water Resources Research, v. 12, no. 1, p. 108--112.
Lee, R., 1963, Evaluation of solar beam irradiation as a climatic parameter
    of mountain watersheds: Colorado State University Hydrology Papers, no. 2.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_DNEARZERO: float = 1.0e-12
_PI: float = np.pi
_TWO_PI: float = 2.0 * np.pi
# 1 radian of hour angle = 12/pi hours (since 2*pi rad = 24 h).
# Also scales the radiation integral in _func3.  Name follows pywatershed.
_RAD_TO_HOURS: float = 12.0 / np.pi
_RAD_DAY: float = _TWO_PI / 365.242
_ECCENTRICITY: float = 0.01671

# ---------------------------------------------------------------------------
# Public constants — precomputed for days 1..366
# ---------------------------------------------------------------------------
NDOY: int = 366
"""Number of days used in the solar table (includes leap day)."""

_julian_days = np.arange(NDOY) + 1
_obliquity = 1.0 - (_ECCENTRICITY * np.cos((_julian_days - 3) * _RAD_DAY))
_yy = (_julian_days - 1) * _RAD_DAY

solar_declination: NDArray[np.floating] = (
    0.006918
    - 0.399912 * np.cos(_yy)
    + 0.070257 * np.sin(_yy)
    - 0.006758 * np.cos(2 * _yy)
    + 0.000907 * np.sin(2 * _yy)
    - 0.002697 * np.cos(3 * _yy)
    + 0.00148 * np.sin(3 * _yy)
)
"""Solar declination angle (radians) for each day of the year, shape (366,)."""

r1: NDArray[np.floating] = (60.0 * 2.0) / (_obliquity**2)
"""Extraterrestrial irradiance corrected for Earth-Sun distance (Langleys/hr), shape (366,)."""

# Clean up module namespace
del _julian_days, _obliquity, _yy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _compute_t(
    lats: NDArray[np.floating],
    decl: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute sunrise/sunset hour angles.

    Parameters
    ----------
    lats : ndarray, shape (nhru,)
        Latitude-like angles in radians (may be equivalent slope latitude).
    decl : ndarray, shape (ndoy,)
        Solar declination for each day of the year in radians.

    Returns
    -------
    ndarray, shape (ndoy, nhru)
        Half-day hour angle (radians).  Sunrise occurs at ``-t``,
        sunset at ``+t``.  Clamped to *pi* for perpetual daylight
        and *0* for polar night.
    """
    nhru = len(lats)
    lats_mat = np.tile(-1.0 * np.tan(lats), (NDOY, 1))
    sol_dec_mat = np.transpose(np.tile(np.tan(decl), (nhru, 1)))
    tx = lats_mat * sol_dec_mat

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"invalid value encountered in arccos")
        result = np.arccos(np.copy(tx))

    result[np.where(tx < -1.0)] = _PI
    result[np.where(tx > 1.0)] = 0.0

    nan_count = np.count_nonzero(np.isnan(result))
    if nan_count > 0:
        logger.warning(
            "%d/%d hour-angle values are NaN after arccos correction "
            "(likely NaN in latitude input)",
            nan_count,
            result.size,
        )
    return result


def _func3(
    v: NDArray[np.floating],
    w: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Swift (1976) equation 6 — potential radiation integral.

    Integrates extraterrestrial direct-beam radiation from hour angle *y*
    to *x* on an equivalent tilted surface defined by latitude *w* and
    longitude offset *v* (Lee 1963, eqs. 12--13).

    Parameters
    ----------
    v : ndarray, shape (nhru,)
        Longitude offset of equivalent slope.
    w : ndarray, shape (nhru,)
        Equivalent slope latitude.
    x : ndarray, shape (ndoy, nhru)
        Upper hour-angle bound.
    y : ndarray, shape (ndoy, nhru)
        Lower hour-angle bound.

    Returns
    -------
    ndarray, shape (ndoy, nhru)
        Potential solar radiation between hour angles *y* and *x*.
    """
    nhru = len(v)
    vv = np.tile(v, (NDOY, 1))
    ww = np.tile(w, (NDOY, 1))
    rr = np.transpose(np.tile(r1, (nhru, 1)))
    dd = np.transpose(np.tile(solar_declination, (nhru, 1)))

    f3 = (
        rr
        * _RAD_TO_HOURS
        * (
            np.sin(dd) * np.sin(ww) * (x - y)
            + np.cos(dd) * np.cos(ww) * (np.sin(x + vv) - np.sin(y + vv))
        )
    )
    return f3


def _compute_soltab_core(
    slopes: NDArray[np.floating],
    aspects: NDArray[np.floating],
    lats: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Core soltab computation for a single surface configuration.

    This is a faithful port of ``PRMSSolarGeometry.compute_soltab`` from
    pywatershed.

    Parameters
    ----------
    slopes : ndarray, shape (nhru,)
        Slope as decimal fraction (rise/run).
    aspects : ndarray, shape (nhru,)
        Aspect in degrees (0 = north, 180 = south).
    lats : ndarray, shape (nhru,)
        Latitude in decimal degrees.

    Returns
    -------
    solt : ndarray, shape (ndoy, nhru)
        Potential clear-sky solar radiation (Langleys, cal/cm2/day).
    sunh : ndarray, shape (ndoy, nhru)
        Hours of direct sunlight.
    """
    nhru = len(slopes)

    # Slope geometry
    sl = np.arctan(slopes)
    sl_sin = np.sin(sl)
    sl_cos = np.cos(sl)
    aspects_rad = np.radians(aspects)
    aspects_cos = np.cos(aspects_rad)

    # Equivalent slope latitude (Lee 1963, eq. 13)
    x0 = np.radians(lats)
    x0_cos = np.cos(x0)
    x1 = np.arcsin(sl_cos * np.sin(x0) + sl_sin * x0_cos * aspects_cos)

    # Longitude offset (Lee 1963, eq. 12)
    d1 = sl_cos * x0_cos - sl_sin * np.sin(x0) * aspects_cos
    d1 = np.where(np.abs(d1) < _DNEARZERO, _DNEARZERO, d1)
    x2 = np.arctan(sl_sin * np.sin(aspects_rad) / d1)

    wh_d1_lt_zero = np.where(d1 < 0.0)
    if len(wh_d1_lt_zero[0]) > 0:
        x2[wh_d1_lt_zero] = x2[wh_d1_lt_zero] + _PI

    # Hour angles for equivalent slope and horizontal
    tt = _compute_t(x1, solar_declination)  # (ndoy, nhru)
    t6 = (-1.0 * tt) - x2  # broadcasts x2 (nhru,) across ndoy
    t7 = tt - x2

    tt = _compute_t(x0, solar_declination)  # (ndoy, nhru)
    t0 = -1.0 * tt
    t1 = tt

    # Clip slope angles to horizontal bounds.
    # pywatershed uses aliasing here: t3 = t7 (same array), so mutations
    # to t3 also mutate t7.  We replicate this by assigning aliases and
    # then creating the shifted versions from the (now clipped) originals.
    t3 = t7  # alias — t3[i]=val also sets t7[i]=val
    wh_t7_gt_t1 = np.where(t7 > t1)
    if len(wh_t7_gt_t1[0]) > 0:
        t3[wh_t7_gt_t1] = t1[wh_t7_gt_t1]

    t2 = t6  # alias — t2[i]=val also sets t6[i]=val
    wh_t6_lt_t0 = np.where(t6 < t0)
    if len(wh_t6_lt_t0[0]) > 0:
        t2[wh_t6_lt_t0] = t0[wh_t6_lt_t0]

    # Wrap-around shifts (operate on clipped values via aliasing)
    t6 = t6 + _TWO_PI
    t7 = t7 - _TWO_PI

    # Handle t3 < t2
    wh_t3_lt_t2 = np.where(t3 < t2)
    if len(wh_t3_lt_t2[0]):
        t2[wh_t3_lt_t2] = 0.0
        t3[wh_t3_lt_t2] = 0.0

    # Base computation
    solt = _func3(x2, x1, t3, t2)
    sunh = (t3 - t2) * _RAD_TO_HOURS

    # Wrap-around: t7 > t0
    wh_t7_gt_t0 = np.where(t7 > t0)
    if len(wh_t7_gt_t0[0]):
        solt[wh_t7_gt_t0] = (
            _func3(x2, x1, t3, t2)[wh_t7_gt_t0] + _func3(x2, x1, t7, t0)[wh_t7_gt_t0]
        )
        sunh[wh_t7_gt_t0] = (t3 - t2 + t7 - t0)[wh_t7_gt_t0] * _RAD_TO_HOURS

    # Wrap-around: t6 < t1
    wh_t6_lt_t1 = np.where(t6 < t1)
    if len(wh_t6_lt_t1[0]):
        solt[wh_t6_lt_t1] = (
            _func3(x2, x1, t3, t2)[wh_t6_lt_t1] + _func3(x2, x1, t1, t6)[wh_t6_lt_t1]
        )
        sunh[wh_t6_lt_t1] = (t3 - t2 + t1 - t6)[wh_t6_lt_t1] * _RAD_TO_HOURS

    # Override flat surfaces with horizontal computation
    mask_sl_lt_dnearzero = np.tile(np.abs(sl), (NDOY, 1)) < _DNEARZERO
    solt = np.where(
        mask_sl_lt_dnearzero,
        _func3(np.zeros(nhru), x0, t1, t0),
        solt,
    )
    sunh = np.where(mask_sl_lt_dnearzero, (t1 - t0) * _RAD_TO_HOURS, sunh)

    # Clamp negatives
    mask_sunh_lt_dnearzero = sunh < _DNEARZERO
    sunh = np.where(mask_sunh_lt_dnearzero, 0.0, sunh)

    wh_solt_lt_zero = np.where(solt < 0.0)
    if len(wh_solt_lt_zero[0]):
        n_neg = len(wh_solt_lt_zero[0])
        n_total = int(np.prod(solt.shape))
        pct = 100.0 * n_neg / n_total
        solt[wh_solt_lt_zero] = 0.0
        if pct > 1.0:
            logger.warning(
                "Clamped %d/%d (%.1f%%) negative potential solar radiation "
                "values to zero — may indicate incorrect input units or CRS",
                n_neg,
                n_total,
                pct,
            )
        else:
            logger.info(
                "Clamped %d/%d (%.2f%%) negative potential solar radiation "
                "values to zero (numerical edge cases)",
                n_neg,
                n_total,
                pct,
            )

    return solt, sunh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_soltab(
    slopes: NDArray[np.floating],
    aspects: NDArray[np.floating],
    lats: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Compute potential clear-sky solar radiation tables.

    Implements the Swift (1976) algorithm for potential solar radiation on
    sloped and horizontal surfaces for every day of the year (1--366).

    Parameters
    ----------
    slopes : ndarray, shape (nhru,)
        Slope as decimal fraction (rise / run), non-negative.
    aspects : ndarray, shape (nhru,)
        Aspect in degrees, 0--360 clockwise from north (0 = north,
        180 = south).
    lats : ndarray, shape (nhru,)
        Latitude in decimal degrees, range [-90, 90].

    Returns
    -------
    soltab_potsw : ndarray, shape (ndoy, nhru)
        Potential solar radiation on the sloped surface (Langleys).
    soltab_horad_potsw : ndarray, shape (ndoy, nhru)
        Potential solar radiation on a horizontal surface (Langleys).
    soltab_sunhrs : ndarray, shape (ndoy, nhru)
        Hours of direct sunlight on the sloped surface.

    Raises
    ------
    ValueError
        If input arrays have mismatched lengths, are empty, contain NaN,
        or have out-of-range values.
    """
    # --- Input validation ---
    if len(slopes) != len(aspects) or len(slopes) != len(lats):
        raise ValueError(
            f"Input arrays must have equal length: "
            f"slopes={len(slopes)}, aspects={len(aspects)}, lats={len(lats)}"
        )
    if len(slopes) == 0:
        raise ValueError("Input arrays must not be empty")
    if np.any(np.isnan(slopes)) or np.any(np.isnan(aspects)) or np.any(np.isnan(lats)):
        nan_counts = (
            np.count_nonzero(np.isnan(slopes)),
            np.count_nonzero(np.isnan(aspects)),
            np.count_nonzero(np.isnan(lats)),
        )
        raise ValueError(
            f"NaN values in input arrays: slopes={nan_counts[0]}, "
            f"aspects={nan_counts[1]}, lats={nan_counts[2]}"
        )
    if np.any(np.abs(lats) > 90):
        raise ValueError(
            f"Latitude values must be in [-90, 90], got range [{lats.min():.2f}, {lats.max():.2f}]"
        )
    if np.any(slopes < 0):
        raise ValueError("Slope values must be non-negative (decimal fraction rise/run)")
    if np.any(slopes > 10):
        logger.warning(
            "Slopes > 10 detected (max=%.1f). Slopes must be decimal fraction "
            "(rise/run), not degrees. Verify input units.",
            float(slopes.max()),
        )

    nhru = len(slopes)

    # Horizontal surface (zero slope, zero aspect)
    horad_potsw, _horad_sunh = _compute_soltab_core(np.zeros(nhru), np.zeros(nhru), lats)

    # Sloped surface
    sloped_potsw, sloped_sunh = _compute_soltab_core(slopes, aspects, lats)

    return sloped_potsw, horad_potsw, sloped_sunh
