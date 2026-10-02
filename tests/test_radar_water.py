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


def test_water_evidence_reports_the_other_polarisation_and_finds_the_real_flood():
    series = _two_pol_series()
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 3), 0.2)
    trough = np.array(["2026-06-20"] * 3, dtype="datetime64[D]")
    climb = np.array(["2026-07-25"] * 3, dtype="datetime64[D]")
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series)
    assert ev.loc[0, "flood_ok"] and ev.loc[0, "flood_other_drop"] > 0        # both fell: water
    # the v2 flood reports the other polarisation but no longer judges on it (a transplanted paddy
    # raises VV while VH falls); broken passes are removed by the pass screening instead
    assert ev.loc[1, "flood_drop"] >= 4 and ev.loc[1, "flood_vh"] <= -19
    assert ev.loc[1, "flood_other_drop"] < rw.OTHER_POL_DROP_MIN
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
    # a canopy seen in the 10 days before every dark pass (12-28 Jun): a harvest, not water
    before = ((windows - pd.Timestamp("2026-06-12")).days >= -10) & ((windows - pd.Timestamp("2026-06-28")).days <= -3)
    raw[before, 0] = 0.7
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series, ndvi_raw=raw)
    assert not ev.loc[0, "flood_ok"] and ev.loc[0, "ndvi_at_flood"] == 0.7
    # every pass is judged on its own: a canopy cut just before the first dark pass refuses that
    # pass, but the field still under water 12 days later is a flood (a double crop's transplanting)
    raw[:] = np.nan
    raw[((windows - pd.Timestamp("2026-06-12")).days >= -10) & ((windows - pd.Timestamp("2026-06-12")).days <= -3), 0] = 0.7
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series, ndvi_raw=raw)
    assert ev.loc[0, "flood_ok"] and ev.loc[0, "flood_date"] >= np.datetime64("2026-06-24")
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


def test_flood_is_picked_among_dark_passes_not_the_seasons_largest_drop():
    """Double-crop plain (fix plan, round 2, aoi110): the summer crop's harvest is the season's
    largest radar drop (bright canopy -> -17.5 dB, not water); the real transplanting flood in
    August is a smaller drop from the already dry field. The flood must be found anyway."""
    dates = pd.DatetimeIndex(np.arange(np.datetime64("2026-04-01"), np.datetime64("2026-09-20"), 6))
    day = dates.to_numpy().astype("datetime64[D]")
    vh = np.full((len(dates), 1), -12.0)                    # summer canopy
    vv = np.full((len(dates), 1), -6.0)
    after_harvest = day >= np.datetime64("2026-06-24")
    vh[after_harvest], vv[after_harvest] = -17.5, -10.0     # cut and drying: a 5.5 dB drop, VH -17.5
    flood = (day >= np.datetime64("2026-08-15")) & (day <= np.datetime64("2026-08-27"))
    vh[flood], vv[flood] = -22.0, -14.0                     # transplanting water: a 4.5 dB drop, VH -22
    series = [(dates, {"VV": vv, "VH": vh})]
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 1), 0.2)
    trough = np.array(["2026-08-10"], dtype="datetime64[D]")
    climb = np.array(["2026-09-15"], dtype="datetime64[D]")
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=series)
    assert ev.loc[0, "flood_vh"] <= -19 and ev.loc[0, "flood_date"] >= np.datetime64("2026-08-15")
    assert 4.0 <= ev.loc[0, "flood_drop"] < 5.5                                  # not the harvest drop
    assert ev.loc[0, "flood_ok"]


def test_recovery_and_rise_z_are_measured_against_the_pixels_own_levels():
    """Stage 2.2 (round 3): no dB cut-off; the rise is read against the pixel's own flood level, own dry level and
    own pass-to-pass noise. Pixel 0 is half-way back from the water, pixel 1 is back at its own dry level."""
    dates = pd.DatetimeIndex(np.arange(np.datetime64("2026-04-01"), np.datetime64("2026-09-22"), 6))
    day = dates.to_numpy().astype("datetime64[D]")
    wiggle = np.where(np.arange(len(dates)) % 2 == 0, 0.3, -0.3)[:, None]     # +-0.3 dB pass-to-pass noise
    vh = np.full((len(dates), 2), -15.0) + wiggle
    vv = np.full((len(dates), 2), -8.0) + wiggle
    flood = (day >= np.datetime64("2026-07-01")) & (day <= np.datetime64("2026-07-20"))
    vh[flood], vv[flood] = -23.0, -16.0
    after = day > np.datetime64("2026-07-20")
    ramp = np.linspace(0, 1, after.sum())
    vh[after, 0] = -23.0 + 4.0 * ramp            # ends at -19: half-way back
    vh[after, 1] = -23.0 + 8.0 * ramp            # ends at -15: back at its own dry level
    vv[after] = -12.0
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    ndvi = np.full((len(windows), 2), 0.2)
    trough = np.array(["2026-07-10"] * 2, dtype="datetime64[D]")
    climb = np.array(["2026-08-20"] * 2, dtype="datetime64[D]")
    ev = rw.water_evidence(0, trough, climb, ndvi, windows, series=[(dates, {"VV": vv, "VH": vh})])
    assert ev["flood_ok"].all()
    assert np.allclose(ev["vh_dry_own"], -15.0, atol=0.6)
    assert 0.35 < ev.loc[0, "recovery"] < 0.65 and 0.85 < ev.loc[1, "recovery"] < 1.15
    assert 0.2 < ev.loc[0, "vh_noise_own"] < 1.0                 # its own wiggle, not the flood or the climb
    assert ev.loc[1, "rise_z"] > ev.loc[0, "rise_z"] > 3


def test_own_levels_are_nan_without_a_flood_track():
    day = pd.DatetimeIndex(pd.date_range("2026-06-01", periods=5, freq="12D")).to_numpy().astype("datetime64[D]")
    flat = {"VH": np.full((5, 2), -15.0), "VV": np.full((5, 2), -8.0)}
    out = rw.own_levels([(day, None, flat)], np.array([0, -1]), day[-1])
    assert out["end"][0] == -15.0 and out["noise"][0] == 0.0 and np.isnan(out["end"][1]) and np.isnan(out["noise"][1])


def _v3_track(start, vh_fn, vv_fn, n_pix=1, teachers: int = 60, seed: int = 0):
    """One track: pixel 0 follows ``vh_fn`` / ``vv_fn``; ``teachers`` more pixels are clear paddies (July water on both
    polarisations, then a crop) from which the AOI's water level is learned, as in a real AOI."""
    dates = pd.DatetimeIndex(np.arange(np.datetime64(start), np.datetime64("2026-09-22"), 12))
    day = dates.to_numpy().astype("datetime64[D]")
    rng = np.random.default_rng(seed)
    wig = np.where(np.arange(len(dates)) % 2 == 0, 0.25, -0.25)[:, None] * np.ones((1, n_pix))
    vh, vv = vh_fn(day)[:, None] + wig, vv_fn(day)[:, None] + wig
    if teachers:
        jul = (day >= np.datetime64("2026-07-01")) & (day <= np.datetime64("2026-07-20"))
        after = day > np.datetime64("2026-07-20")
        t_vh = np.full((len(day), teachers), -16.0)
        t_vh[jul] = -23.0
        t_vh[after] = np.linspace(-20.0, -14.0, after.sum())[:, None]
        t_vh += rng.normal(0, 0.3, t_vh.shape)
        vh, vv = np.hstack([vh, t_vh]), np.hstack([vv, t_vh + 7.0])
    return dates, {"VH": vh, "VV": vv}


def test_flood_v3_one_track_is_enough_and_vv_alone_counts():
    """Water test v3: track A floods (VH, 2 passes), track B shows nothing there: water. A VV-only flood also counts."""
    flood = lambda d: (d >= np.datetime64("2026-07-01")) & (d <= np.datetime64("2026-07-20"))  # noqa: E731
    a = _v3_track("2026-03-15", lambda d: np.where(flood(d), -23.0, -16.0), lambda d: np.where(flood(d), -16.0, -9.0))
    b = _v3_track("2026-03-18", lambda d: np.full(len(d), -16.0), lambda d: np.full(len(d), -9.0), seed=1)
    t = rw.flood_v3([a, b], np.datetime64("2026-09-21"))
    assert t.loc[1:, "flood_ok"].mean() > 0.95                        # the teachers themselves
    assert t.loc[0, "flood_ok"] and t.loc[0, "support"] in (1, 3)
    assert np.datetime64("2026-07-01") <= t.loc[0, "flood_date"] <= np.datetime64("2026-07-20")
    assert t.loc[0, "flood_vh"] < -22
    vv_only = _v3_track("2026-03-15", lambda d: np.full(len(d), -16.0), lambda d: np.where(flood(d), -16.0, -9.0))
    t = rw.flood_v3([vv_only, b], np.datetime64("2026-09-21"))
    assert t.loc[0, "flood_ok"] and t.loc[0, "flood_pol"] == "VV"


def test_flood_v3_rejects_a_single_speckled_pass_and_trees():
    one = lambda d: d == d[np.searchsorted(d, np.datetime64("2026-07-01"))]  # noqa: E731
    a = _v3_track("2026-03-15", lambda d: np.where(one(d), -23.0, -16.0), lambda d: np.full(len(d), -9.0))
    b = _v3_track("2026-03-18", lambda d: np.full(len(d), -16.0), lambda d: np.full(len(d), -9.0))
    assert not rw.flood_v3([a, b], np.datetime64("2026-09-21")).loc[0, "flood_ok"]        # no second look
    trees = _v3_track("2026-03-15", lambda d: np.full(len(d), -12.0), lambda d: np.full(len(d), -6.0))
    assert not rw.flood_v3([trees], np.datetime64("2026-09-21")).loc[0, "flood_ok"]


def test_flood_v3_needs_darker_than_its_own_dry_season():
    """A monsoon fall that only returns to the pixel's own dry-season level is not water (e.g. rain-wetted weeds cut)."""
    grow = lambda d: np.where((d >= np.datetime64("2026-05-20")) & (d < np.datetime64("2026-07-01")), -12.0, -16.0)  # noqa: E731
    a = _v3_track("2026-03-15", grow, lambda d: np.full(len(d), -9.0))
    assert not rw.flood_v3([a], np.datetime64("2026-09-21")).loc[0, "flood_ok"]


def test_flood_v3_picks_the_transplanting_water_not_a_late_flood():
    """July water then a crop, then a late-August flood on the grown crop: the July water is the transplanting flood."""
    def vh(d):
        v = np.full(len(d), -16.0)
        jul = (d >= np.datetime64("2026-07-01")) & (d <= np.datetime64("2026-07-20"))
        v[jul] = -22.0
        after = d > np.datetime64("2026-07-20")
        v[after] = np.linspace(-20.0, -12.0, after.sum())
        late = (d >= np.datetime64("2026-08-20")) & (d <= np.datetime64("2026-09-02"))
        v[late] = -24.0
        return v
    a = _v3_track("2026-03-15", vh, lambda d: np.full(len(d), -9.0))
    t = rw.flood_v3([a], np.datetime64("2026-09-21"))
    assert t.loc[0, "flood_ok"] and t.loc[0, "flood_date"] <= np.datetime64("2026-07-20")


def test_flood_v3_learns_the_aois_water_level_from_its_clear_fields():
    """Round 3: a summer-rice field has water in its own dry season, so 'darker than its own dry season' fails; it is
    water because it is as dark as the AOI's clearly flooded fields on the same pass. A harvest to bare soil is not."""
    dates = pd.DatetimeIndex(np.arange(np.datetime64("2026-03-15"), np.datetime64("2026-09-22"), 12))
    day = dates.to_numpy().astype("datetime64[D]")
    n = 62
    rng = np.random.default_rng(0)
    vh = np.full((len(day), n), -16.0) + rng.normal(0, 0.3, (len(day), n))
    jul = (day >= np.datetime64("2026-07-01")) & (day <= np.datetime64("2026-07-20"))
    after = day > np.datetime64("2026-07-20")
    vh[np.ix_(jul, range(60))] = -23.0 + rng.normal(0, 0.3, (jul.sum(), 60))      # 60 clear paddies
    vh[np.ix_(after, range(60))] = np.linspace(-20, -14, after.sum())[:, None]
    mar = day < np.datetime64("2026-05-01")
    vh[mar, 60], vh[~mar, 60] = -23.0, -13.0                                      # summer paddy: water in March, crop
    vh[jul, 60] = -23.0
    vh[after, 60] = np.linspace(-20, -14, after.sum())
    vh[:, 61] = -13.0                                                              # a crop cut in July to bare soil
    vh[jul | after, 61] = -17.0
    vh[after, 61] = np.linspace(-17, -13, after.sum())
    vh[:, 60:] += rng.normal(0, 0.3, (len(day), 2))                              # real radar always has some noise
    vv = vh + 7.0
    t = rw.flood_v3([(dates, {"VH": vh, "VV": vv})], np.datetime64("2026-09-21"))
    assert t.loc[:59, "flood_ok"].mean() > 0.95 and not t.loc[:59, "flood_by_peers"].any()
    assert t.loc[60, "flood_ok"] and t.loc[60, "flood_by_peers"] and not t.loc[60, "flood_by_pattern"]
    # the cut crop is not water by the AOI's level; it can only be a pattern flood, which the rule accepts only when a
    # crop rises after it (the known risk of a dry-sown crop after a harvest, checked on the negatives)
    assert not t.loc[61, "flood_ok"] or t.loc[61, "flood_by_pattern"]


def test_flood_v3_shallow_water_confirmed_by_a_clear_optical_view():
    """aoi13_000594 (round 3): a July fall to only -18.8 dB VH (the AOI's floods reach -23) is still water when a clear
    optical view between the passes shows open water (NDVI below 0); without that view it is not."""
    flood = lambda d: (d >= np.datetime64("2026-07-01")) & (d <= np.datetime64("2026-07-20"))  # noqa: E731
    a = _v3_track("2026-03-15", lambda d: np.where(flood(d), -18.8, -14.0), lambda d: np.where(flood(d), -11.5, -7.5))
    b = _v3_track("2026-03-18", lambda d: np.where(flood(d), -18.5, -14.0), lambda d: np.where(flood(d), -11.5, -7.5), seed=1)
    t = rw.flood_v3([a, b], np.datetime64("2026-09-21"))
    assert not t.loc[0, "flood_ok"] or t.loc[0, "flood_by_pattern"]         # not water by the AOI's level
    win = pd.date_range("2026-03-01", "2026-09-21", freq="5D").to_numpy().astype("datetime64[D]")
    seen = np.zeros((len(win), a[1]["VH"].shape[1]), dtype=bool)
    seen[(win >= np.datetime64("2026-07-10")) & (win <= np.datetime64("2026-07-15")), 0] = True   # one clear view of water
    t = rw.flood_v3([a, b], np.datetime64("2026-09-21"), water_seen=(win, seen))
    assert t.loc[0, "flood_ok"] and not t.loc[0, "flood_by_pattern"]


def test_rise_since_counts_only_what_came_after_the_day():
    """aoi13_000737: water until late August, the crop rises after it; a clear view on 17 Aug is followed by a clear rise,
    a clear view on 10 Sep (after most of the rise) by little."""
    flood = lambda d: d <= np.datetime64("2026-08-20")  # noqa: E731
    rise = lambda d: np.where(flood(d), -24.0, -24.0 + (d - np.datetime64("2026-08-20")).astype(int) * 0.3)  # noqa: E731
    a = _v3_track("2026-03-15", rise, lambda d: rise(d) + 8.0, teachers=0)
    z_aug = rw.rise_since([a], np.array(["2026-08-17"], dtype="datetime64[D]"), np.datetime64("2026-09-21"))[0]
    z_late = rw.rise_since([a], np.array(["2026-09-20"], dtype="datetime64[D]"), np.datetime64("2026-09-21"))[0]
    assert z_aug > 2 > z_late
    assert np.isnan(rw.rise_since([a], np.array(["NaT"], dtype="datetime64[D]"), np.datetime64("2026-09-21"))[0])


def test_brightened_at_sowing_separates_dry_sowing_from_flooding():
    """aoi160_004992 (user 30 Sep): VH jumps from -18.7 to -12 right after sowing (dry-sown crop) -> True; a paddy that
    goes dark under the water after sowing -> False."""
    dry = _v3_track("2026-03-15", lambda d: np.where(d > np.datetime64("2026-05-25"), -12.5, -18.5),
                    lambda d: np.full(len(d), -8.0), teachers=0)
    wet = _v3_track("2026-03-15", lambda d: np.where((d > np.datetime64("2026-05-25")) & (d < np.datetime64("2026-06-25")),
                                                     -24.0, -17.0), lambda d: np.full(len(d), -9.0), teachers=0)
    day = np.array(["2026-05-26"], dtype="datetime64[D]")
    assert rw.brightened_at([dry], day)[0] and not rw.brightened_at([wet], day)[0]


def test_single_dip_counts_only_when_deepest_and_bare(monkeypatch):
    """aoi160_003251: one dark pass (21 May) drained before the next pass; with SINGLE_DIP_WATER it is water when it is
    the pixel's darkest pass and the optical shows the field at its own bare low then; not when the field was green."""
    one = lambda d: d == d[np.searchsorted(d, np.datetime64("2026-05-21"))]  # noqa: E731
    a = _v3_track("2026-03-15", lambda d: np.where(one(d), -21.5, -16.0), lambda d: np.where(one(d), -11.0, -8.0))
    win = pd.date_range("2026-03-01", "2026-09-21", freq="5D").to_numpy().astype("datetime64[D]")
    n = a[1]["VH"].shape[1]
    bare = np.zeros((len(win), n), dtype=bool)
    bare[(win >= np.datetime64("2026-05-01")) & (win <= np.datetime64("2026-05-31")), 0] = True
    monkeypatch.setattr(rw, "SINGLE_DIP_WATER", True)
    t = rw.flood_v3([a], np.datetime64("2026-09-21"), bare_seen=(win, bare))
    assert t.loc[0, "flood_ok"]
    t = rw.flood_v3([a], np.datetime64("2026-09-21"), bare_seen=(win, np.zeros_like(bare)))
    assert not t.loc[0, "flood_ok"]


def test_brightened_in_the_sowing_spell_not_only_at_the_trough_date():
    """aoi160_003930: last bare view 4 May, first green view 3 Jun (cloud between); the radar brightens on 30 May. The
    check must look over the whole spell."""
    dry = _v3_track("2026-03-15", lambda d: np.where(d > np.datetime64("2026-05-28"), -12.9, -17.5),
                    lambda d: np.full(len(d), -8.0), teachers=0)
    day = np.array(["2026-05-04"], dtype="datetime64[D]")
    assert not rw.brightened_at([dry], day)[0]
    assert rw.brightened_at([dry], day, until=np.array(["2026-06-03"], dtype="datetime64[D]"))[0]


def test_flood_v3_counts_water_only_around_sowing():
    """Issue 36: a field sown in May whose largest later rise follows an August dip under its canopy. Without the sowing
    gate the August dip wins; with it (the optical says the field stood above half-way to its peak from June on) the
    flood is the May water, and a pixel whose only dip is under the canopy has no flood at all."""
    def vh(d):
        v = np.full(len(d), -17.0)
        v[(d >= np.datetime64("2026-05-20")) & (d <= np.datetime64("2026-06-10"))] = -23.0
        v[d > np.datetime64("2026-06-10")] = -16.5
        v[(d >= np.datetime64("2026-08-01")) & (d <= np.datetime64("2026-08-25"))] = -24.0
        v[d > np.datetime64("2026-08-25")] = -12.0
        return v
    def aug_only(d):
        v = np.full(len(d), -16.5)
        v[(d >= np.datetime64("2026-08-01")) & (d <= np.datetime64("2026-08-25"))] = -24.0
        v[d > np.datetime64("2026-08-25")] = -12.0
        return v
    # a long dry season, so the steps of the two dips do not dominate the pixel's own pass-to-pass noise
    a = _v3_track("2025-11-03", vh, lambda d: np.full(len(d), -9.0))
    b = _v3_track("2025-11-03", aug_only, lambda d: np.full(len(d), -9.0), seed=1)
    free = rw.flood_v3([a], np.datetime64("2026-09-21"))
    assert free.loc[0, "flood_date"] >= np.datetime64("2026-08-01")          # the old behaviour: the August dip wins
    win = np.arange(np.datetime64("2025-11-01"), np.datetime64("2026-09-26"), 5)
    sow = np.ones((len(win), a[1]["VH"].shape[1]), dtype=bool)
    sow[win >= np.datetime64("2026-06-15"), 0] = False                       # pixel 0 under its canopy from mid June
    sow[win >= np.datetime64("2026-06-15"), 1:] = True                       # the teachers are sown in July
    gated = rw.flood_v3([a], np.datetime64("2026-09-21"), sowing_seen=(win, sow))
    assert gated.loc[0, "flood_ok"]
    assert np.datetime64("2026-05-15") <= gated.loc[0, "flood_date"] <= np.datetime64("2026-06-10")
    assert not rw.flood_v3([b], np.datetime64("2026-09-21"), sowing_seen=(win, sow)).loc[0, "flood_ok"]
