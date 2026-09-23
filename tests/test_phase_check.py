"""Tests for the SAR-vs-optical phase diagnostic.

Only the pure timing helper is tested here; the curve builders need real rasters and are covered by
running them on an AOI.
"""
import numpy as np

from sar_pipeline.analysis.phase_check import half_fall_date


def _dates(n):
    import datetime as dt

    return [dt.date(2025, 5, 1) + dt.timedelta(days=10 * i) for i in range(n)]


def test_half_fall_is_the_first_crossing_below_the_midpoint():
    values = [0.2, 0.5, 0.8, 0.7, 0.4, 0.2, 0.2]  # peak 0.8 at index 2, floor 0.2 -> midpoint 0.5
    dates = _dates(len(values))
    assert half_fall_date(dates, values, 2) == dates[4]


def test_half_fall_ignores_a_pre_peak_dip():
    values = [0.1, 0.9, 0.3, 0.85, 0.8, 0.2]
    dates = _dates(len(values))
    assert half_fall_date(dates, values, 3) == dates[5]


def test_half_fall_is_none_when_the_curve_never_falls():
    values = [0.2, 0.5, 0.8, 0.79, 0.8]
    assert half_fall_date(_dates(len(values)), values, 2) is None


def test_half_fall_is_none_at_the_last_point():
    values = [0.2, 0.5, 0.8]
    assert half_fall_date(_dates(len(values)), values, 2) is None


def test_half_fall_tolerates_nan_gaps():
    values = [0.2, 0.8, np.nan, 0.3, 0.2]
    dates = _dates(len(values))
    assert half_fall_date(dates, values, 1) == dates[3]
