"""USDA soil texture triangle classification.

Vectorized implementation of the standard USDA soil texture triangle
decision tree.  Used by both the generic pipeline (pixel-level raster
classification for categorical zonal stats) and the pywatershed plugin
(HRU-level aggregate-then-classify fallback).

References
----------
Gerakis, A. and B. Baer, 1999. A computer program for soil textural
classification. Soil Science Society of America Journal, 63:807-808.

USDA-NRCS Soil Texture Calculator:
https://www.nrcs.usda.gov/resources/education-and-teaching-materials/soil-texture-calculator
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

USDA_TEXTURE_CLASSES: dict[int, str] = {
    1: "sand",
    2: "loamy_sand",
    3: "sandy_loam",
    4: "sandy_clay_loam",
    5: "loam",
    6: "silt_loam",
    7: "sandy_clay",
    8: "silt",
    9: "clay_loam",
    10: "silty_clay_loam",
    11: "silty_clay",
    12: "clay",
}


def classify_usda_texture(
    sand: np.ndarray,
    silt: np.ndarray,
    clay: np.ndarray,
) -> np.ndarray:
    """Classify sand/silt/clay percentages into USDA texture class codes.

    Apply the standard USDA soil texture triangle decision tree using
    vectorized boolean masking.  Each element is assigned one of 12
    integer class codes defined in ``USDA_TEXTURE_CLASSES``.

    Parameters
    ----------
    sand : np.ndarray
        Sand content as percentage (0--100), shape ``(n,)``.
    silt : np.ndarray
        Silt content as percentage (0--100), shape ``(n,)``.
    clay : np.ndarray
        Clay content as percentage (0--100), shape ``(n,)``.

    Returns
    -------
    np.ndarray
        Float64 array of USDA texture class codes (1--12), shape
        ``(n,)``.  Elements where any input is NaN are NaN.

    Notes
    -----
    The decision tree evaluates conditions using line-equation
    boundaries from the USDA Soil Survey Manual (Ch. 3).  Evaluation
    order matters: later assignments overwrite earlier ones where
    conditions overlap, so clay-dominated classes must be evaluated
    last.

    Inputs must satisfy ``sand + silt + clay ≈ 100``.  A warning is
    logged if inputs appear to be fractions (0--1) rather than
    percentages, or if the sum deviates from 100 by more than 5%.

    References
    ----------
    Gerakis, A. and B. Baer, 1999. A computer program for soil
    textural classification. Soil Science Society of America
    Journal, 63:807-808.
    """
    s = np.asarray(sand, dtype=np.float64)
    si = np.asarray(silt, dtype=np.float64)
    c = np.asarray(clay, dtype=np.float64)

    valid = ~(np.isnan(s) | np.isnan(si) | np.isnan(c))

    # Input validation on valid elements only
    if valid.any():
        totals = (s + si + c)[valid]
        if np.all(totals < 2.0):
            logger.warning(
                "classify_usda_texture: values appear to be fractions "
                "(0-1) rather than percentages (0-100); classification "
                "results will be incorrect. Check source data units."
            )
        far_from_100 = np.abs(totals - 100.0) > 5.0
        if np.any(far_from_100):
            logger.warning(
                "classify_usda_texture: %d/%d element(s) have "
                "sand+silt+clay summing outside 95-105%% range; "
                "texture classification may be unreliable",
                int(np.sum(far_from_100)),
                len(totals),
            )

    # Initialize: NaN everywhere, then fill valid elements
    result = np.full(len(s), np.nan)
    sv, siv, cv = s[valid], si[valid], c[valid]
    codes = np.full(len(sv), 5, dtype=np.float64)  # default loam

    # Line-equation conditions matching the USDA Soil Survey Manual
    # (Ch. 3) texture triangle boundaries.  Order matters: later
    # assignments overwrite earlier ones for overlapping regions.
    codes[siv + 1.5 * cv < 15] = 1  # sand
    codes[(siv + 1.5 * cv >= 15) & (siv + 2 * cv < 30)] = 2  # loamy_sand
    codes[
        ((cv >= 7) & (cv < 20) & (sv > 52) & (siv + 2 * cv >= 30))
        | ((cv < 7) & (siv < 50) & (siv + 2 * cv >= 30))
    ] = 3  # sandy_loam
    codes[(cv >= 20) & (cv < 35) & (siv < 28) & (sv > 45)] = 4  # sandy_clay_loam
    codes[(cv >= 7) & (cv < 27) & (siv >= 28) & (siv < 50) & (sv <= 52)] = 5  # loam
    codes[((siv >= 50) & (cv >= 12) & (cv < 27)) | ((siv >= 50) & (siv < 80) & (cv < 12))] = (
        6  # silt_loam
    )
    codes[(siv >= 80) & (cv < 12)] = 8  # silt
    codes[(cv >= 27) & (cv < 40) & (sv > 20) & (sv <= 45)] = 9  # clay_loam
    codes[(cv >= 27) & (cv < 40) & (sv <= 20)] = 10  # silty_clay_loam
    codes[cv >= 40] = 12  # clay (broadest — overwritten by specific subcategories)
    codes[(cv >= 35) & (sv > 45)] = 7  # sandy_clay
    codes[(cv >= 40) & (siv >= 40)] = 11  # silty_clay

    result[valid] = codes
    return result
