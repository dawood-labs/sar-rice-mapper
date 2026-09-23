"""Tests for the current-season rule on synthetic pixel series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import monsoon_rule as mr


def _series():
    windows = pd.date_range("2025-09-01", periods=78, freq="5D")
    t = np.arange(78, dtype=float)
    season = (windows >= "2026-06-01")
    rice = np.full(78, 0.6)
    rice[season] = 0.05 + 0.75 / (1 + np.exp(-(t[season] - t[season][0] - 8) / 1.5))   # trough, then climb
    young = np.full(78, 0.6)
    young[season] = 0.1 + 0.3 / (1 + np.exp(-(t[season] - t[season][-1] + 2) / 1.5))   # climb just starting, ~0.4
    trees = np.full(78, 0.75)                                                          # never at a trough
    bare = np.full(78, 0.2)                                                            # never climbs
    flood = np.full(78, 0.2)
    flood[season] = np.where(t[season] < t[season][10], -0.35, 0.2)                    # water, then soil
    ndvi = np.stack([rice, young, trees, bare, flood], axis=1)
    lswi = np.full_like(ndvi, 0.2)
    # the young pixel dries out after a harvest (-0.2) and is then wetted to ~0.05: relative wetting only
    dry = season & (t < t[season][0] + 3)
    lswi[dry, 1] = -0.2
    lswi[season & ~dry, 1] = 0.05
    return ndvi, lswi, windows


def test_classes_follow_trough_and_rise():
    ndvi, lswi, windows = _series()
    ev = mr.pixel_events(ndvi, lswi, windows)
    assert mr.classify(ev).tolist() == [1, 2, 0, 0, 0]   # water re-emerging as soil is not a canopy
    assert ev.loc[0, "wet_open"] and ev.loc[0, "trough_ndvi"] < 0.1
    assert not ev.loc[1, "wet_open"] and ev.loc[1, "wet_relative"] and ev.loc[1, "wet_at_trough"]
    assert pd.notna(ev.loc[0, "climb_date"]) and pd.isna(ev.loc[1, "climb_date"])


def test_missing_values_in_the_season_are_no_data():
    ndvi, lswi, windows = _series()
    ndvi[70, 0] = np.nan
    assert mr.classify(mr.pixel_events(ndvi, lswi, windows))[0] == 255
