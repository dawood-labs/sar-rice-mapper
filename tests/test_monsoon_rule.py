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
    first = int(np.flatnonzero(season)[0])
    rice[first - 12:first] = 0.2                                                       # harvested, bare for weeks
    rice[season] = 0.05 + 0.75 / (1 + np.exp(-(t[season] - t[season][0] - 8) / 1.5))   # trough, then climb
    young = np.full(78, 0.6)
    young[season] = 0.1 + 0.3 / (1 + np.exp(-(t[season] - t[season][-1] + 2) / 1.5))   # climb just starting, ~0.4
    trees = np.full(78, 0.75)                                                          # never at a trough
    bare = np.full(78, 0.2)                                                            # never climbs
    flood = np.full(78, 0.2)
    flood[season] = np.where(t[season] < t[season][10], -0.35, 0.2)                    # water, then soil
    cut = np.full(78, 0.6)
    cut[first - 12:first] = 0.2
    cut[season] = 0.05 + 0.75 / (1 + np.exp(-(t[season] - t[season][0] - 6) / 1.5))  # climbed early...
    cut[-6:] = 0.2                                                                     # ...and harvested
    # an orchard with one haze dip that the light mask let through: rice-like to the optical alone
    haze = np.full(78, 0.8)
    haze[first + 3] = 0.3
    ndvi = np.stack([rice, young, trees, bare, flood, cut, haze], axis=1)
    lswi = np.full_like(ndvi, 0.2)
    # the young pixel dries out after a harvest (-0.2) and is then wetted to ~0.05: relative wetting only
    dry = season & (t < t[season][0] + 3)
    lswi[dry, 1] = -0.2
    lswi[season & ~dry, 1] = 0.05
    lswi[:, 5] = 0.2
    lswi[:, 6] = 0.3
    return ndvi, lswi, windows


def test_classes_follow_trough_and_rise():
    ndvi, lswi, windows = _series()
    ev = mr.pixel_events(ndvi, lswi, windows)
    assert mr.classify(ev).tolist() == [3, 2, 0, 0, 0, 4, 0]   # no radar: standing rice unconfirmed; water->soil is no canopy; cut crop is 4; haze dip on a canopy is nothing
    assert mr.classify(ev, radar_wet=[True] * 7).tolist() == [1, 2, 0, 0, 0, 4, 0]
    assert ev.loc[0, "low_windows"] >= mr.LOW_WINDOWS_MIN and ev.loc[6, "low_windows"] < mr.LOW_WINDOWS_MIN
    assert ev.loc[0, "standing"] and not ev.loc[5, "standing"]
    assert ev.loc[0, "wet_open"] and ev.loc[0, "trough_ndvi"] < 0.1
    assert not ev.loc[1, "wet_open"] and ev.loc[1, "wet_relative"] and ev.loc[1, "wet_at_trough"]
    assert pd.notna(ev.loc[0, "climb_date"]) and pd.isna(ev.loc[1, "climb_date"])


def test_missing_values_in_the_season_are_no_data():
    ndvi, lswi, windows = _series()
    ndvi[70, 0] = np.nan
    assert mr.classify(mr.pixel_events(ndvi, lswi, windows))[0] == 255


def test_classify_sends_never_bare_unconfirmed_pixels_to_class_5_only():
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    ev = pd.DataFrame({"valid": [True] * 3, "trough_ndvi": [0.2] * 3, "rise": [0.5] * 3,
                       "peak_after": [0.8] * 3, "standing": [True] * 3, "low_windows": [10] * 3})
    wet = np.array([True, False, False])
    never = np.array([True, True, False])
    out = mr.classify(ev, radar_wet=wet, never_bare=never)
    # water confirmed wins over never-bare; never-bare only relabels the unconfirmed
    assert out.tolist() == [1, 5, 3]
