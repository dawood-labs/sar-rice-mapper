"""Tests for the whole-AOI date summary. No network, no files."""
from __future__ import annotations

import numpy as np

from sar_pipeline.analysis import aoi_profile as ap


def test_summary_counts_canopy_and_water_only_over_usable_pixels():
    ndvi = np.array([0.8, 0.7, 0.1, -0.2, 0.9])
    lswi = np.array([0.3, 0.2, 0.3, 0.1, 0.0])
    usable = np.array([True, True, True, True, False])      # the last one is cloud
    s = ap.summarise(ndvi, lswi, usable, canopy_ndvi=0.5)
    assert s["usable_pct"] == 80.0
    assert s["canopy_pct"] == 50.0                           # 0.8 and 0.7
    assert s["water_pct"] == 50.0                            # 0.1<0.3 and -0.2<0.1


def test_summary_of_a_fully_clouded_date_says_so():
    assert ap.summarise(np.ones(3), np.ones(3), np.zeros(3, bool)) == {"usable_pct": 0.0}
