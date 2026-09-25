"""Tests for per-pixel radar dips at each pixel's own trough date. No files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import radar_water as rw


def test_dip_is_read_at_each_pixels_own_trough():
    dates = pd.date_range("2026-03-15", "2026-09-20", freq="12D")
    troughs = np.array(["2026-06-15", "2026-07-27", "NaT"], dtype="datetime64[D]")
    cube = np.full((len(dates), 3), -8.0)
    for p, t in enumerate(troughs[:2]):
        days = (dates.to_numpy().astype("datetime64[D]") - t) / np.timedelta64(1, "D")
        cube[(days >= -6) & (days <= 12), p] = -8.0 - (6.0 if p == 0 else 1.0)
    dry, flood, dip = rw.dips_for_track(dates, cube, troughs)
    assert abs(dip[0] - 6.0) < 1e-9 and abs(dip[1] - 1.0) < 1e-9
    assert np.isnan(dip[2])                       # no trough: nothing to line up on
    assert dry[0] == -8.0 and flood[0] == -14.0


def test_trough_at_the_start_of_the_run_is_not_checkable():
    dates = pd.date_range("2026-05-01", "2026-09-20", freq="12D")
    troughs = np.array(["2026-05-04"], dtype="datetime64[D]")
    _, _, dip = rw.dips_for_track(dates, np.full((len(dates), 1), -8.0), troughs)
    assert np.isnan(dip[0])                       # the dry window lies before the first date


def test_flood_search_finds_a_flood_weeks_before_the_anchor():
    import numpy as np

    from sar_pipeline.analysis import radar_water as rw

    dates = np.arange(np.datetime64("2026-04-01"), np.datetime64("2026-09-01"), 6)
    v = np.full(len(dates), -9.0)
    flood = (dates >= np.datetime64("2026-06-10")) & (dates <= np.datetime64("2026-06-25"))
    v[flood] = -15.0
    cube = v[:, None]
    anchor = np.array(["2026-07-20"], dtype="datetime64[D]")
    best, when, persist = rw.flood_search(dates, cube, anchor)
    assert best[0] == 6.0 and np.datetime64("2026-06-10") <= when[0] <= np.datetime64("2026-06-25")
    assert persist[0] >= 2
    # a steadily rising field (a dry-land crop growing) never drops below its own earlier level
    rising = np.linspace(-12, -5, len(dates))[:, None]
    best, _, persist = rw.flood_search(dates, rising, anchor)
    assert best[0] < 0 and persist[0] == 0


def _two_pol_series(n_pixels=3):
    """One track, 6-day passes; pixel 0 a real flood (VV and VH fall), pixel 1 a broken pass (VH
    falls 8 dB while VV rises 2 dB), pixel 2 dry all season."""
    dates = pd.DatetimeIndex(np.arange(np.datetime64("2026-04-01"), np.datetime64("2026-09-20"), 6))
    day = dates.to_numpy().astype("datetime64[D]")
    vv = np.full((len(dates), n_pixels), -9.0)
    vh = np.full((len(dates), n_pixels), -16.0)
    flood = (day >= np.datetime64("2026-06-10")) & (day <= np.datetime64("2026-06-28"))
    vv[flood, 0] = -16.0
    vh[flood, 0] = -23.0
    one = day == np.datetime64("2026-06-18")      # a pass date (6-day steps from 1 Apr)
    vh[one, 1] = -24.0
    vv[one, 1] = -7.0
    return [(dates, {"VV": vv, "VH": vh})]


def test_water_evidence_rejects_a_pass_where_only_one_polarisation_fell():
    series = _two_pol_series()
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 3), 0.2)
    trough = np.array(["2026-06-20"] * 3, dtype="datetime64[D]")
    climb = np.array(["2026-07-25"] * 3, dtype="datetime64[D]")
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series)
    assert ev.loc[0, "flood_ok"] and ev.loc[0, "flood_other_drop"] > 0        # both fell: water
    assert ev.loc[1, "flood_drop"] >= 4 and ev.loc[1, "flood_vh"] <= -19       # looks like water in VH...
    assert ev.loc[1, "flood_other_drop"] < rw.OTHER_POL_DROP_MIN and not ev.loc[1, "flood_ok"]
    assert not ev.loc[2, "flood_ok"]


def test_flood_support_counts_either_polarisation_and_a_deep_single_pass_needs_none():
    dates = pd.DatetimeIndex(np.arange(np.datetime64("2026-04-01"), np.datetime64("2026-09-20"), 6))
    day = dates.to_numpy().astype("datetime64[D]")
    vv = np.full((len(dates), 3), -9.0)
    vh = np.full((len(dates), 3), -16.0)
    d1, d2 = day == np.datetime64("2026-06-12"), day == np.datetime64("2026-06-18")
    # pixel 0: VH flood on one pass, the next pass drops in VV only (3 dB) -> supported by VV
    vh[d1, 0], vv[d1, 0] = -23.0, -14.0
    vv[d2, 0] = -12.0
    # pixel 1: one very dark, very deep pass, nothing else -> deep flood, no support needed
    vh[d1, 1], vv[d1, 1] = -26.0, -17.0
    # pixel 2: one moderate pass only -> not enough
    vh[d1, 2], vv[d1, 2] = -21.0, -13.0
    series = [(dates, {"VV": vv, "VH": vh})]
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 3), 0.2)
    trough = np.array(["2026-06-15"] * 3, dtype="datetime64[D]")
    climb = np.array(["2026-07-25"] * 3, dtype="datetime64[D]")
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series)
    assert ev.loc[0, "support"] >= 2 and ev.loc[0, "flood_ok"]
    assert ev.loc[1, "support"] == 1 and ev.loc[1, "flood_ok"]
    assert ev.loc[2, "support"] == 1 and not ev.loc[2, "flood_ok"]


def test_canopy_at_flood_is_judged_on_observations_not_on_the_fit():
    series = _two_pol_series()
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    # the fit interpolates 0.6 across a gap on the flood date for every pixel
    ndvi = np.full((len(windows), 3), 0.6)
    trough = np.array(["2026-06-20"] * 3, dtype="datetime64[D]")
    climb = np.array(["2026-07-25"] * 3, dtype="datetime64[D]")
    assert not rw.water_evidence(0, trough, climb, ndvi, windows, series=series).loc[0, "flood_ok"]
    raw = np.full((len(windows), 3), np.nan)                 # nothing observed near the flood: pixel 0 passes
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series, ndvi_raw=raw)
    assert ev.loc[0, "flood_ok"] and np.isnan(ev.loc[0, "ndvi_at_flood"])
    before = ((windows - pd.Timestamp("2026-06-12")).days >= -10) & ((windows - pd.Timestamp("2026-06-12")).days <= -3)
    raw[before, 0] = 0.7                                     # a canopy seen in the 10 days before the drop: a harvest, not water
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series, ndvi_raw=raw)
    assert not ev.loc[0, "flood_ok"] and ev.loc[0, "ndvi_at_flood"] == 0.7
    raw[:] = np.nan
    after = ((windows - pd.Timestamp("2026-06-12")).days >= 5) & ((windows - pd.Timestamp("2026-06-12")).days <= 30)
    raw[after, 0] = 0.7                                      # a canopy after the flood is the rice itself, however fast
    assert rw.water_evidence(0, trough, climb, ndvi, windows, series=series, ndvi_raw=raw).loc[0, "flood_ok"]


def test_bare_near_flood_needs_one_clear_view_without_canopy():
    series = _two_pol_series()
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 3), 0.6)
    trough = np.array(["2026-06-20"] * 3, dtype="datetime64[D]")
    climb = np.array(["2026-07-25"] * 3, dtype="datetime64[D]")
    raw = np.full((len(windows), 3), np.nan)
    days = (windows - pd.Timestamp("2026-06-12")).days
    raw[(days >= -60) & (days <= -35), 0] = 0.15                 # bare 5-8 weeks before the flood: a field
    raw[(days >= -60) & (days <= -35), 1] = 0.75                 # green then, and nothing else seen: trees
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series, ndvi_raw=raw)
    assert ev.loc[0, "bare_near_flood"] and not ev.loc[1, "bare_near_flood"]
    assert ev.loc[2, "bare_near_flood"]                          # never observed: unknown, not a canopy


def test_shallow_water_counts_with_a_larger_drop_and_more_passes():
    dates = pd.DatetimeIndex(np.arange(np.datetime64("2026-04-01"), np.datetime64("2026-09-20"), 6))
    day = dates.to_numpy().astype("datetime64[D]")
    vv = np.full((len(dates), 2), -9.0)
    vh = np.full((len(dates), 2), -12.5)
    flood = (day >= np.datetime64("2026-06-06")) & (day <= np.datetime64("2026-06-30"))   # five passes
    vh[flood, 0], vv[flood, 0] = -18.2, -13.0          # shallow: VH short of -19 but a 5.7 dB drop, 5 passes
    one = day == np.datetime64("2026-06-06")
    vh[one, 1], vv[one, 1] = -18.2, -13.0              # the same level on one pass per track only: not enough
    # a second track three days behind sees the same ground: two tracks, as every AOI has
    series = [(dates, {"VV": vv, "VH": vh}), (dates + pd.Timedelta(days=3), {"VV": vv, "VH": vh})]
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 2), 0.2)
    trough = np.array(["2026-06-15"] * 2, dtype="datetime64[D]")
    climb = np.array(["2026-07-25"] * 2, dtype="datetime64[D]")
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series)
    assert ev.loc[0, "support"] >= 4 and ev.loc[0, "flood_ok"]
    assert not ev.loc[1, "flood_ok"]


def test_a_summer_crop_cut_three_weeks_before_the_flood_does_not_block_it():
    series = _two_pol_series()
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 3), 0.6)
    trough = np.array(["2026-06-20"] * 3, dtype="datetime64[D]")
    climb = np.array(["2026-07-25"] * 3, dtype="datetime64[D]")
    raw = np.full((len(windows), 3), np.nan)
    days = (windows - pd.Timestamp("2026-06-12")).days
    raw[(days >= -25) & (days <= -15), 0] = 0.65             # the summer rice, green three weeks before the flood
    raw[(days >= -60) & (days <= -35), 0] = 0.2              # ... and bare before that (a field)
    assert rw.water_evidence(0, trough, climb, ndvi, windows, series=series, ndvi_raw=raw).loc[0, "flood_ok"]
