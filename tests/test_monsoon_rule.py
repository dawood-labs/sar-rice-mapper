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
    assert mr.classify(ev, radar_wet=[True] * 7).tolist() == [1, 6, 0, 0, 0, 4, 0]   # the young crop with water is young rice
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
    # never-bare ground cannot have been a flooded field: a v1 dip there is rain on a garden, so
    # class 1 falls to 5 as well (fix plan, issues 2/5/9); only a confirmed flood (flood_ok) wins
    assert out.tolist() == [5, 5, 3]


def test_young_with_water_and_visible_canopy_is_young_rice():
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    # three young pixels (rise 0.2, peak 0.4): wet & visible, wet & too low, dry & visible
    ev = pd.DataFrame({"valid": [True] * 3, "trough_ndvi": [0.1] * 3, "rise": [0.2] * 3,
                       "peak_after": [0.4] * 3, "standing": [False] * 3, "low_windows": [10] * 3,
                       "last_ndvi": [0.35, 0.25, 0.35]})
    out = mr.classify(ev, radar_wet=np.array([True, True, False]))
    assert out.tolist() == [6, 2, 2]


def test_radar_trough_rescues_a_paddy_whose_optical_trough_was_hidden():
    """A field hidden by cloud around transplanting: the fit never falls below 0.40, but the radar
    saw the flood; with the radar trough it is rice, without it not rice (issue 4)."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    # pixel 0: straight rise 0.30 -> 0.85 (gap interpolated; above 0.40 inside the 110-day window); pixel 1: same, but no radar flood;
    # pixel 2: a real optical trough (ordinary rice); pixel 3: flood then no climb (not rice)
    ndvi = np.stack([np.linspace(0.30, 0.85, n)] * 2 + [np.r_[np.full(10, 0.1), np.linspace(0.1, 0.8, n - 10)]]
                    + [np.full(n, 0.45)], axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-05-25", "NaT", "2026-05-25", "2026-05-25"])
    ev["flood_ok"] = [True, False, True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    wet = ev["flood_ok"].to_numpy()
    assert mr.classify(ev, radar_wet=wet).tolist() == [0, 0, 1, 0]
    out = mr.classify(ev, radar_wet=wet, radar_trough=True)
    assert out.tolist() == [1, 0, 1, 0]
    assert ev.loc[0, "rise_from_flood"] > 0.3 and np.isnan(ev.loc[1, "rise_from_flood"])
    # harvested after a radar trough: the canopy fell away at the end
    ndvi2 = ndvi.copy()
    ndvi2[-4:, 0] = 0.2
    ev2 = mr.pixel_events(ndvi2, ndvi2 * 0, windows)
    ev2["flood_date"], ev2["flood_ok"] = ev["flood_date"], ev["flood_ok"]
    ev2 = pd.concat([ev2, mr.radar_trough_events(ev2, ndvi2, windows)], axis=1)
    assert mr.classify(ev2, radar_wet=wet, radar_trough=True)[0] == 4


def test_never_bare_removes_every_rice_like_class_unless_a_flood_was_confirmed():
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    rice = np.r_[np.full(10, 0.1), np.linspace(0.1, 0.8, n - 10)]               # standing rice curve
    cut = rice.copy(); cut[-5:] = 0.2                                              # harvested curve
    young = np.r_[np.full(n - 6, 0.1), np.linspace(0.1, 0.35, 6)]                 # young curve
    ndvi = np.stack([rice, rice, cut, young, young], axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_ok"] = [False, True, False, False, False]
    wet = np.array([True, True, False, True, False])
    nb = np.array([True, True, True, True, True])
    out = mr.classify(ev, radar_wet=wet, never_bare=nb)
    # class 1 (v1 water only), 4, 6 and 2 all fall to 5; the confirmed flood keeps its class 1
    assert out.tolist() == [5, 1, 5, 5, 5]
    assert mr.classify(ev, radar_wet=wet, never_bare=~nb).tolist() == [1, 1, 4, 6, 2]
