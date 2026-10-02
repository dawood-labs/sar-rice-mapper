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
    assert mr.classify(ev).tolist() == [3, 2, 0, 0, 0, 8, 0]   # no radar: standing rice unconfirmed; water->soil is no canopy; a cut crop without water is 8; haze dip on a canopy is nothing
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
    assert out.tolist() == [1, 0, 1, 6]        # pixel 3: a confirmed flood, a 0.45 canopy visible now = young rice
    ev["bare_near_flood"] = [False, True, True, True]        # pixel 0 never seen bare: the radar alone may not decide
    assert mr.classify(ev, radar_wet=wet, radar_trough=True).tolist() == [0, 0, 1, 6]
    ev = ev.drop(columns="bare_near_flood")
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
    out = mr.classify(ev, radar_wet=wet, never_bare=pd.Series(nb))      # a Series, as run_aoi passes it
    # class 1 (v1 water only), 4, 6 and 2 all fall to 5; the confirmed flood keeps its class 1
    assert out.tolist() == [5, 1, 5, 5, 5]
    assert mr.classify(ev, radar_wet=wet, never_bare=~nb).tolist() == [1, 1, 8, 6, 2]   # the cut pixel had no water: 8


def test_report_classes_flooded_not_green_and_cut_unconfirmed():
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    cut = np.r_[np.full(10, 0.1), np.linspace(0.1, 0.8, n - 15), np.full(5, 0.2)]     # cycle cut at the end
    water = np.r_[np.full(n - 4, 0.15), np.full(4, 0.05)]                             # bare, then open water
    ndvi = np.stack([cut, cut, water, water], axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_ok"] = [True, False, True, True]
    ev["flood_date"] = pd.to_datetime(["2026-06-05", "NaT", "2026-09-10", "2026-06-05"])
    wet = ev["flood_ok"].to_numpy()
    out = mr.classify(ev, radar_wet=wet, map_date=windows[-1])
    # cut with water -> 4, cut without -> 8; open water with a flood 11 days ago -> 7; an old flood -> stays 0
    assert out.tolist() == [4, 8, 7, 0]
    assert mr.classify(ev, radar_wet=wet).tolist() == [4, 8, 0, 0]       # no map date: no class 7


def test_radar_canopy_rise_makes_young_rice_when_the_optical_end_is_stale():
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    stale = np.r_[np.full(n - 6, 0.12), np.full(6, 0.12)]     # bare, last clear view before the transplanting
    ndvi = np.stack([stale, stale, stale], axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_ok"] = [True, True, False]
    # pixel 0 flooded in mid-June (more than 60 days before the map date): a radar canopy on it is a
    # crop, not a fresh flood; pixel 1 flooded in mid-August and still dark
    ev["flood_date"] = pd.to_datetime(["2026-06-15", "2026-08-15", "NaT"])
    ev["bare_near_flood"] = [True, True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    ev["radar_canopy_rise"] = [6.0, 2.0, 6.0]                  # dB from the water to the season end
    ev["vh_end"] = [-15.5, -22.0, -15.5]
    wet = ev["flood_ok"].to_numpy()
    out = mr.classify(ev, radar_wet=wet, radar_trough=True, map_date=windows[-1])
    assert out[0] == 6 and out[1] == 7 and out[2] == 0     # canopy in the radar; still water; no flood
    ev["radar_canopy_rise"], ev["vh_end"] = [11.0, 2.0, 6.0], [-14.0, -22.0, -15.5]
    assert mr.classify(ev, radar_wet=wet, radar_trough=True, map_date=windows[-1])[0] == 1   # full-grown in the radar
    # a radar-only young rice whose optical value is still open water (0.12 here) is class 7, not 6
    ev["radar_canopy_rise"], ev["vh_end"] = [6.0, 2.0, 6.0], [-15.5, -22.0, -15.5]
    ev["flood_date"] = pd.to_datetime(["2026-08-25", "2026-08-15", "NaT"])
    assert mr.classify(ev, radar_wet=wet, radar_trough=True, map_date=windows[-1])[0] == 7
    ev["vh_end"] = [-24.0, -22.0, -15.0]                       # a pond: from -32 to -24 dB is still water
    assert mr.classify(ev, radar_wet=wet, radar_trough=True, map_date=windows[-1])[0] == 7


def test_a_ripening_crop_is_still_standing():
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ripening = np.r_[np.full(10, 0.1), np.linspace(0.1, 0.78, n - 16), np.linspace(0.75, 0.45, 6)]
    cut = ripening.copy(); cut[-3:] = 0.2
    ndvi = np.stack([ripening, cut], axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    assert mr.classify(ev, radar_wet=np.array([True, True])).tolist() == [1, 4]


def test_a_recent_clear_view_without_canopy_beats_the_radar_canopy():
    """Stage 6: open water / bare mud (raw NDVI -0.1 on a clear view 10 days before the map date)
    beside bright bunds showed a 12 dB radar rise; the radar canopy may only stand in for the
    optical one where the optical end is stale."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    base = dict(valid=True, trough_ndvi=0.0, low_windows=12, rise=0.0, peak_after=0.1, standing=False,
                last_ndvi=0.1, flood_ok=True, flood_date=pd.Timestamp("2026-07-15"), bare_near_flood=True,
                peak_after_flood=0.1, rise_from_flood=0.1, standing_after_flood=False,
                radar_canopy_rise=12.0, vh_end=-12.0)
    ev = pd.DataFrame([dict(base, last_clear_ndvi=-0.1, last_clear_age=2),     # clear 10 days ago: water
                       dict(base, last_clear_ndvi=0.2, last_clear_age=12),     # last seen 60 days ago: stale
                       dict(base, last_clear_ndvi=0.6, last_clear_age=1)])     # clear and green
    out = mr.classify(ev, radar_wet=[True] * 3, radar_trough=True, map_date="2026-09-21")
    assert out[0] not in (1, 6)
    assert out[1] == 1 and out[2] == 1
    raw = np.full((5, 2), np.nan)
    raw[1, 0], raw[3, 0] = 0.5, -0.1
    view = mr.last_clear_view(raw, pd.date_range("2026-09-01", periods=5, freq="5D"))
    assert view.loc[0, "last_clear_ndvi"] == -0.1 and view.loc[0, "last_clear_age"] == 1
    assert np.isnan(view.loc[1, "last_clear_ndvi"]) and np.isinf(view.loc[1, "last_clear_age"])


def test_classify_does_not_change_its_input():
    """numpy 2 + pandas 2.2: np.array(series) is a view; classify must copy before any in-place step (issue 35)."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.stack([np.linspace(0.30, 0.85, n)] * 2, axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-05-25", "2026-05-25"])
    ev["flood_ok"] = [True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    ev["bare_near_flood"] = [False, True]
    ev["never_bare"] = [True, False]
    before = ev.copy()
    mr.classify(ev, radar_wet=ev["flood_ok"], never_bare=ev["never_bare"], radar_trough=True, map_date=windows[-1])
    pd.testing.assert_frame_equal(ev, before)


def test_median_day_survives_an_empty_column():
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    assert mr.median_day(pd.Series([np.nan, np.nan])) is None
    assert mr.median_day(pd.to_datetime(["2026-06-01", "2026-06-11"])) == "06 Jun"


def test_radar_path_with_own_levels_needs_no_db_cut_off():
    """Water test v3 columns (round 3): cloud hides the optical end, the radar decides against each pixel's own levels."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.full((n, 4), 0.1)                       # open water all along: the optical shows no crop
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-08-05"] * 3 + ["NaT"])
    ev["flood_ok"] = [True, True, True, False]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    ev["rise_z_all"] = [5.0, 3.0, 1.0, np.nan]
    ev["flood_vh"] = [-23.0, -23.0, -23.0, np.nan]
    ev["vh_premonsoon"] = [-16.0, -16.0, -16.0, np.nan]
    ev["vh_end_track"] = [-15.5, -19.5, -22.5, np.nan]
    out = mr.classify(ev, radar_wet=ev["flood_ok"].to_numpy(), radar_trough=True, map_date=windows[-1])
    assert out.tolist() == [1, 6, 7, 0]


def test_a_pond_is_not_a_flooded_field():
    """Round 3: a pond's water and its ripples in the radar must not become rice (median clear NDVI below 0 all year)."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.full((n, 2), 0.1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-07-10"] * 2)
    ev["flood_ok"] = [True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    ev["rise_z_all"] = [5.0, 5.0]
    ev["flood_vh"] = [-23.0, -23.0]
    ev["vh_premonsoon"] = [-16.0, -23.5]
    ev["vh_end_track"] = [-15.5, -20.5]
    ev["open_water_year"] = [False, True]
    out = mr.classify(ev, radar_wet=ev["flood_ok"].to_numpy(), radar_trough=True, map_date=windows[-1])
    assert out.tolist() == [1, 0]


def test_a_pattern_flood_counts_only_when_the_crop_rises():
    """Round 3 (aoi13_000598): shallow water that only the pixel's own dry season shows is water when it lies at the
    pixel's own sowing and the crop rises after it (young rice); without the rise, or on a tree line with no sowing, nothing."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.full((n, 3), 0.1)
    ndvi[:, 2] = 0.8                                  # a tree line: always green, no sowing
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-08-05"] * 3)
    ev["flood_ok"] = [True, True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    ev["flood_by_pattern"] = [True, True, True]
    ev["ndvi_noise_own"] = [0.03, 0.03, 0.03]
    ev["ndvi_max_own"] = [0.5, 0.5, 0.82]             # the fields had a crop before; the tree line never dips
    ev["rise_z_all"] = [4.5, 1.0, 4.5]
    ev["flood_vh"] = [-18.2, -18.2, -16.0]
    ev["vh_premonsoon"] = [-14.1, -14.1, -13.0]
    ev["vh_end_track"] = [-16.0, -17.9, -13.5]
    out = mr.classify(ev, radar_wet=ev["flood_ok"].to_numpy(), radar_trough=True, map_date=windows[-1])
    assert out.tolist() == [6, 0, 0]



def test_young_is_young_rice_unless_everything_is_low_and_dry():
    """User decision (30 Sep): no class 2. A young crop with water at sowing or a radar canopy is young rice; only a young
    crop with a very low optical AND radar canopy AND no water sign is not rice."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    base = np.r_[np.full(19, 0.2), [0.1], np.linspace(0.1, 0.35, n - 20)]   # sown 4 Aug, a small crop, no water seen
    ndvi = np.stack([base, base, np.full(n, 0.1)], axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["NaT"] * 3)
    ev["flood_ok"] = [False, False, False]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    ev["flood_by_pattern"] = [False, False, False]
    ev["ndvi_noise_own"] = [0.02, 0.02, 0.02]
    ev["rise_z_all"] = [np.nan, np.nan, np.nan]
    ev["flood_vh"] = ev["vh_premonsoon"] = ev["vh_end_track"] = np.nan
    before = mr.classify(ev.drop(columns=["rise_z_all"]), radar_wet=np.zeros(3, bool), radar_trough=True, map_date=windows[-1])
    assert before[0] == 2
    out = mr.classify(ev, radar_wet=np.zeros(3, bool), radar_trough=True, map_date=windows[-1])
    assert out[0] == 6 and out[1] == 6 and 2 not in out.tolist()   # a visible optical rise above its own noise


def test_open_water_on_the_latest_clear_view_after_the_flood_is_class_7():
    """User decision (30 Sep, "b"): flooded, a slight radar rise, but the latest clear view (after the flood) is open
    water -> flooded, not yet green (7); a latest clear view from before the flood says nothing -> young rice."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.stack([np.r_[np.full(8, 0.5), np.full(n - 8, 0.05)]] * 2, axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-08-05"] * 2)
    ev["flood_ok"] = [True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    ev["flood_by_pattern"] = [False, False]
    ev["ndvi_noise_own"] = [0.02, 0.02]
    ev["ndvi_max_own"] = [0.5, 0.5]
    ev["rise_z_all"] = [3.0, 3.0]
    ev["flood_vh"] = [-25.0, -25.0]
    ev["vh_premonsoon"] = [-16.0, -16.0]
    ev["vh_end_track"] = [-21.0, -21.0]
    ev["last_clear_ndvi"] = [-0.3, -0.3]
    ev["last_clear_age"] = [2.0, 12.0]            # 11 Sep (after the flood) / 1 Aug (before it)
    out = mr.classify(ev, radar_wet=ev["flood_ok"].to_numpy(), radar_trough=True, map_date=windows[-1])
    assert out.tolist() == [7, 6]


def test_a_tree_line_with_a_haze_dip_is_not_a_paddy():
    """Round 3: a tree line beside paddies shares their radar flood and has a haze-dip trough (0.6); its low point is far
    above the AOI's rice at sowing, so the radar path must not make it rice. The paddies themselves stay rice."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    rng = np.random.default_rng(0)
    trough = np.r_[rng.normal(0.05, 0.03, 60), 0.6]
    ev = pd.DataFrame({"trough_ndvi": trough, "flood_ok": True, "rise_z_all": 4.0, "flood_by_pattern": False,
                       "ndvi_noise_own": 0.03})
    z = mr.trough_vs_rice(ev)
    assert (z[:60] < 2).mean() > 0.95 and z[60] > 2


def test_a_radar_rise_after_the_last_clear_view_overrules_its_water():
    """aoi13_000737 / 000471 (user, 30 Sep): the latest clear view (17 Aug) shows water, the radar rises clearly after it:
    young rice, not flooded-not-green. Without that later rise the "b" rule still gives 7."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.stack([np.r_[np.full(8, 0.5), np.full(n - 8, 0.0)]] * 2, axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-06-20"] * 2)
    ev["flood_ok"] = [True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    for c, v in (("flood_by_pattern", False), ("ndvi_noise_own", 0.03), ("ndvi_max_own", 0.8), ("rise_z_all", 8.0),
                 ("flood_vh", -24.0), ("vh_premonsoon", -17.0), ("vh_end_track", -17.0), ("last_clear_ndvi", -0.02),
                 ("last_clear_age", 7.0)):
        ev[c] = v
    ev["rise_since_last_clear_z"] = [6.0, 0.5]
    out = mr.classify(ev, radar_wet=ev["flood_ok"].to_numpy(), radar_trough=True, map_date=windows[-1])
    assert out[0] in (1, 6) and out[1] == 7


def test_young_rice_sown_before_the_window_end_is_rice():
    """User decision (30 Sep): young rice sown before YOUNG_SOWN_AFTER is rice; sown after it stays young rice."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.stack([np.r_[np.full(8, 0.5), np.full(n - 8, 0.05)]] * 2, axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-07-08", "2026-08-10"])
    ev["flood_ok"] = [True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    for c, v in (("flood_by_pattern", False), ("ndvi_noise_own", 0.03), ("ndvi_max_own", 0.5), ("rise_z_all", 3.0),
                 ("flood_vh", -23.0), ("vh_premonsoon", -16.0), ("vh_end_track", -19.5)):
        ev[c] = v
    out = mr.classify(ev, radar_wet=ev["flood_ok"].to_numpy(), radar_trough=True, map_date=windows[-1])
    assert out.tolist() == [1, 6]


def test_trough_is_the_last_bare_moment_of_the_season_not_a_haze_dip():
    """aoi160_003471 (user, 30 Sep): bare in May, green by 3 Jun, one hazy view 0.41 on 23 Jun, green again. With the
    AOI's rice-at-sowing level the trough is in May (the whole season is searched), not on the haze dip."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    d = windows.to_numpy()
    ndvi = np.where(d < np.datetime64("2026-05-20"), 0.12,
                    np.where(d < np.datetime64("2026-06-20"), 0.70,
                             np.where(d < np.datetime64("2026-06-28"), 0.41, 0.76)))[:, None]
    old = mr.pixel_events(ndvi, ndvi * 0, windows)                                 # the 110-day lookback
    new = mr.pixel_events(ndvi, ndvi * 0, windows, lookback_days=None, trough_level=np.array([0.2]))
    assert pd.Timestamp(old.loc[0, "trough_date"]) >= pd.Timestamp("2026-06-01")
    assert pd.Timestamp(new.loc[0, "trough_date"]) < pd.Timestamp("2026-05-20")
    assert new.loc[0, "trough_ndvi"] == 0.12


def test_a_young_crop_still_rising_is_not_harvested():
    """Plan O5 (round 3, plot 2971): cut = the latest value back at the field's own bare level; a crop at 0.30 two
    weeks after its trough (0.18) stands; one back at 0.12 (its bare level 0.10) after a peak of 0.8 is cut."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    mr.HARVEST_BACK_TO_BARE = True                     # the switched-off option, tested so it stays correct
    windows = pd.date_range("2026-05-01", "2026-09-21", freq="5D")
    n = len(windows)
    rising = np.r_[np.full(n - 6, 0.5), [0.18, 0.18, 0.2, 0.24, 0.27, 0.30]]
    cut = np.r_[np.full(n - 20, 0.5), np.full(4, 0.1), np.linspace(0.1, 0.8, 10), np.full(6, 0.12)]
    ndvi = np.stack([rising, cut], axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["flood_date"] = pd.to_datetime(["2026-08-25", "2026-06-10"])
    ev["flood_ok"] = [True, True]
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    for c, v in (("flood_by_pattern", False), ("ndvi_noise_own", 0.02), ("ndvi_max_own", 0.8), ("rise_z_all", 3.0),
                 ("flood_vh", -23.0), ("vh_premonsoon", -16.0), ("vh_end_track", -19.0)):
        ev[c] = v
    out = mr.classify(ev, radar_wet=ev["flood_ok"].to_numpy(), radar_trough=True, map_date=windows[-1])
    mr.HARVEST_BACK_TO_BARE = False
    assert out[0] in (1, 6) and out[1] == 4


def test_season_end_follows_a_longer_series_only():
    ndvi, lswi, windows = _series()                                    # last window 21 Sep: season unchanged
    assert mr.season_to_series(mr.SEASON, windows) == mr.SEASON
    late = windows.append(pd.DatetimeIndex(["2026-09-26"]))           # one more window (26 and 28 Sep images)
    assert mr.season_to_series(mr.SEASON, late) == (mr.SEASON[0], "2026-09-27")
    # a field whose lowest point is in the new window: found only with the moved end
    cut = np.concatenate([ndvi[:, [0]], [[0.02]]])
    old = mr.pixel_events(cut, cut * 0, late)
    new = mr.pixel_events(cut, cut * 0, late, mr.season_to_series(mr.SEASON, late))
    assert old.loc[0, "trough_date"] < pd.Timestamp("2026-09-24") and new.loc[0, "trough_date"] == pd.Timestamp("2026-09-26")


def test_latest_view_at_the_aois_sowing_level_is_no_crop_yet():
    """Issue 37 (user, 30 Sep): a field flooded in June whose latest clear view reads 0.11 has nothing growing: it lies at
    this AOI's paddies-at-sowing level (60 reference paddies with troughs ~0.05), so rule "b" gives class 7 although the
    view is not open water. A clear view well above that level (0.45) keeps the crop."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    rng = np.random.default_rng(0)
    ev = pd.DataFrame({"trough_ndvi": np.r_[rng.normal(0.05, 0.03, 60), 0.05, 0.05], "flood_ok": True,
                       "rise_z_all": 4.0, "flood_by_pattern": False, "ndvi_noise_own": 0.03})
    z = mr.level_vs_rice_sowing(ev, np.r_[np.full(60, 0.05), 0.11, 0.45])
    assert z[60] < 2 <= z[61]
    assert np.allclose(mr.trough_vs_rice(ev), mr.level_vs_rice_sowing(ev, ev["trough_ndvi"]), equal_nan=True)
    # a wide spread of the AOI's troughs does not hide a young crop from the own-noise reading
    wide = ev.assign(trough_ndvi=np.r_[rng.uniform(0.0, 0.5, 60), 0.05, 0.05])
    z_own = mr.level_vs_rice_sowing(wide, np.r_[np.full(60, 0.05), 0.11, 0.45], with_spread=False)
    assert mr.level_vs_rice_sowing(wide, np.r_[np.full(60, 0.05), 0.11, 0.45])[61] < 2 <= z_own[61]


def test_a_climb_seen_on_one_last_view_only_is_not_a_crop():
    """aoi160_007735 (user, 30 Sep): bare all season, one clear view on the last date jumps 0.22 -> 0.53, the radar flat.
    With two clear views above the trough, or a VH rise since it, the same curve stays rice."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-26", freq="5D")
    n = len(windows)
    fit = np.r_[np.full(n - 4, 0.2), 0.3, 0.4, 0.48, 0.55]
    ndvi = np.stack([fit] * 3, axis=1)
    ev = mr.pixel_events(ndvi, ndvi * 0, windows)
    ev["low_windows"] = 12
    for c, v in (("flood_ok", False), ("flood_by_pattern", False), ("ndvi_noise_own", 0.03), ("ndvi_max_own", 0.55),
                 ("rise_z_all", np.nan), ("flood_vh", np.nan), ("vh_premonsoon", -17.0), ("vh_end_track", np.nan),
                 ("flood_date", pd.NaT)):
        ev[c] = v
    ev = pd.concat([ev, mr.radar_trough_events(ev, ndvi, windows)], axis=1)
    raw = np.full((n, 3), np.nan)
    raw[:-3] = 0.2
    raw[-1] = 0.53                                     # pixel 0: one clear view up
    raw[-3, 1] = 0.35                                  # pixel 1: two clear views up
    ev["views_above_trough"] = mr.views_above_trough(raw, windows, ev["trough_date"], ev["trough_ndvi"],
                                                     ev["ndvi_noise_own"])
    ev["rise_since_trough_z"] = [0.5, 0.5, 4.0]         # pixel 2: one view, but the VH rose
    assert ev["views_above_trough"].tolist() == [1, 2, 1]
    out = mr.classify(ev, radar_wet=np.zeros(3, bool), radar_trough=True, map_date=windows[-1])
    assert out[0] == 0 and out[1] != 0 and out[2] == out[1]


def test_views_above_flood_counts_after_the_flood_date():
    """The radar path's climb is counted from the flood (June), not from a post-harvest trough at the series end."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    windows = pd.date_range("2026-05-01", "2026-09-26", freq="5D")
    raw = np.full((len(windows), 1), np.nan)
    d = windows.to_numpy()
    raw[(d > np.datetime64("2026-07-01")) & (d < np.datetime64("2026-08-20")), 0] = 0.7     # the crop after the flood
    raw[d >= np.datetime64("2026-09-10"), 0] = 0.1                                             # cut
    assert mr.views_above_trough(raw, windows, pd.to_datetime(["2026-06-15"]), [0.1], [0.03])[0] >= 2
    assert mr.views_above_trough(raw, windows, pd.to_datetime(["2026-09-15"]), [0.1], [0.03])[0] == 0
    assert mr.views_above_trough(raw, windows, pd.to_datetime([pd.NaT]), [0.1], [0.03])[0] == 0


def test_one_view_at_the_aois_rice_canopy_is_enough():
    """aoi20 plots (user rule, 30 Sep): one clear view of 0.75 in a cloudy AOI whose two-view crops peak at ~0.8 is a crop;
    one view of 0.53 (aoi160_007735) is not."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import monsoon_rule as mr

    rng = np.random.default_rng(0)
    ev = pd.DataFrame({"peak_after": np.r_[rng.normal(0.8, 0.03, 60), 0.75, 0.53], "ndvi_noise_own": 0.03})
    ref = np.r_[np.ones(60, bool), False, False]
    ok = mr.at_rice_canopy(ev, ev["peak_after"], ref)
    assert ok[60] and not ok[61]
    assert not mr.at_rice_canopy(ev, ev["peak_after"], np.zeros(62, bool)).any()      # no reference: no shortcut
    windows = pd.date_range("2026-05-01", "2026-09-26", freq="5D")
    raw = np.full((len(windows), 1), np.nan)
    raw[-4, 0], raw[-1, 0] = 0.75, 0.2
    assert mr.max_view_after(raw, windows, pd.to_datetime(["2026-08-01"]))[0] == 0.75
