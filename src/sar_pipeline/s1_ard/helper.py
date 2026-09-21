"""Small helpers shared by the ARD steps: polarisation bands, linear <-> dB, output encoding.

Adapted from gee_s1_ard/python-api/helper.py
(https://github.com/adugnag/gee_s1_ard, MIT License, (c) 2021 Adugna Mullissa;
see LICENSE_gee_s1_ard.txt). Changes: band lists are explicit instead of "all bands except
angle", and non-positive linear values are masked before log10 (log10 of <= 0 is -inf/NaN).
"""
from __future__ import annotations

import math

import ee

# Canonical polarisation order. Band order in every output is VV then VH (interfaces.md §6.3).
CANONICAL_POLS = ("VV", "VH")


def ordered_pols(pols) -> list[str]:
    """Return the configured polarisations in canonical order (VV before VH)."""
    pols = list(pols)
    unknown = [p for p in pols if p not in CANONICAL_POLS]
    if unknown or not pols:
        raise ValueError(f"s1.pols must be a non-empty subset of {list(CANONICAL_POLS)}, got {pols}")
    return [p for p in CANONICAL_POLS if p in pols]


def lin_to_db(image: ee.Image, bands: list[str]) -> ee.Image:
    """10*log10(power). Pixels with power <= 0 are masked (their dB value is undefined)."""
    lin = image.select(bands)
    return lin.updateMask(lin.gt(0)).log10().multiply(10).rename(bands)


def db_to_lin(image: ee.Image, bands: list[str]) -> ee.Image:
    return ee.Image.constant(10).pow(image.select(bands).divide(10)).rename(bands)


# ---------------------------------------------------------------- pure-python mirrors (unit-testable)
INT16_MIN_VALID = -32767
INT16_MAX_VALID = 32767


def encode_int16_value(db: float | None, scale: float, nodata: int) -> int:
    """Pure-python mirror of the Earth Engine int16 encoding in pipeline.to_output.

    EE expression: floor(dB * scale + 0.5), clamp to [-32767, 32767], toInt16, masked -> nodata.
    The same "add 0.5 then floor" rounding is used on both sides so they agree exactly,
    including at .5 ties. -32768 is reserved for nodata, hence the clamp starts at -32767.
    """
    if db is None or (isinstance(db, float) and (math.isnan(db) or math.isinf(db))):
        return int(nodata)
    v = math.floor(db * scale + 0.5)
    return int(min(INT16_MAX_VALID, max(INT16_MIN_VALID, v)))


def decode_int16_value(value: int, scale: float, nodata: int) -> float:
    """Inverse of encode_int16_value (nodata -> NaN)."""
    return float("nan") if value == nodata else value / scale
