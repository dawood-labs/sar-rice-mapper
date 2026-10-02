"""Fresh start, step 2: sowing = the last trough of the smoothed curve; after mid-July only with the radar's agreement."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import sowing_fresh as sf

WIN = pd.date_range("2026-05-01", "2026-09-26", freq="5D")


def curve(points):
    """A smoothed curve through (date, value) points, linear in between."""
    x = [pd.Timestamp(d).value for d, _ in points]
    return np.interp(WIN.as_unit("ns").asi8, x, [v for _, v in points])


def test_troughs_are_the_valleys_followed_by_a_real_rise():
    f = np.array([0.5, 0.3, 0.2, 0.2, 0.4, 0.7, 0.69, 0.70, 0.6])[:, None]     # a flat bottom, a wiggle, a fall
    t = sf.troughs(f, np.array([0.02]))[:, 0]
    assert t.tolist() == [False, False, False, True, False, False, False, False, False]


def test_a_second_crop_trough_at_bare_level_is_the_sowing_and_one_above_it_is_not():
    """A late trough counts when the field is bare again (0.15 in May, 0.60 in June, back to 0.17 in late July, the
    September crop after it: sowing late July, the radar agrees). aoi160 pixel 59890 (0.69 -> 0.33 in July, user:
    "a trough is usually around 0.2") keeps its May sowing."""
    day = pd.date_range("2026-05-03", "2026-09-24", freq="6D")
    d = day.to_numpy()
    vh = np.full((len(day), 2), -15.0)
    vh[(d >= np.datetime64("2026-07-15")) & (d <= np.datetime64("2026-07-28"))] = -21.0
    vh += np.where(np.arange(len(day)) % 2 == 0, 0.2, -0.2)[:, None]
    bare_again = curve([("2026-05-01", 0.35), ("2026-05-16", 0.15), ("2026-06-15", 0.60), ("2026-07-21", 0.17),
                        ("2026-09-01", 0.57)])
    p59890 = curve([("2026-05-01", 0.35), ("2026-05-16", 0.15), ("2026-06-15", 0.69), ("2026-07-21", 0.33),
                    ("2026-09-01", 0.57)])
    fit = np.stack([bare_again, p59890], axis=1)
    s = sf.sowing(fit, fit.copy(), WIN, np.array([True, True]), series=[(day, {"VH": vh})], level=np.array([0.22, 0.22]))
    assert pd.Timestamp(s.loc[0, "sowing_date"]) == pd.Timestamp("2026-07-20") and s.loc[0, "period"] == 4
    assert pd.Timestamp(s.loc[1, "sowing_date"]) == pd.Timestamp("2026-05-16") and s.loc[1, "period"] == 1


def test_bare_level_comes_from_the_aois_crop_ground():
    rng = np.random.default_rng(0)
    lows = rng.normal(0.18, 0.03, 300)
    fit = np.vstack([np.full(300, 0.7), lows, lows, np.full(300, 0.7)])
    level = sf.bare_level(fit, np.full(300, 0.02), np.ones(300, bool))
    assert 0.2 < np.median(level) < 0.3


def test_a_late_trough_without_the_radar_falls_back_to_the_last_trough_before_mid_july():
    fit = curve([("2026-05-01", 0.35), ("2026-05-16", 0.15), ("2026-06-15", 0.69), ("2026-07-21", 0.33),
                 ("2026-09-01", 0.57)])[:, None]
    day = pd.date_range("2026-05-03", "2026-09-24", freq="6D")
    vh = np.full((len(day), 1), -15.0) + np.where(np.arange(len(day)) % 2 == 0, 0.2, -0.2)[:, None]
    s = sf.sowing(fit, fit.copy(), WIN, np.array([True]), series=[(day, {"VH": vh})])
    assert pd.Timestamp(s.loc[0, "sowing_date"]) == pd.Timestamp("2026-05-16") and s.loc[0, "period"] == 1


def test_a_flat_bottom_takes_its_last_window_and_a_rising_curve_its_lowest():
    """aoi160 pixel 81532: 0.14 on 14 and 19 May, rising after: sowing 19 May (the last window of the flat bottom)."""
    fit = curve([("2026-05-01", 0.18), ("2026-05-11", 0.14), ("2026-05-21", 0.14), ("2026-08-01", 0.7)])[:, None]
    assert pd.Timestamp(sf.sowing(fit, fit.copy(), WIN, np.array([True])).loc[0, "sowing_date"]) == pd.Timestamp("2026-05-21")
    up = curve([("2026-05-01", 0.2), ("2026-08-01", 0.8)])[:, None]
    assert pd.Timestamp(sf.sowing(up, up.copy(), WIN, np.array([True])).loc[0, "sowing_date"]) == pd.Timestamp("2026-05-01")


def test_not_crop_ground_gets_no_sowing():
    fit = curve([("2026-05-01", 0.5), ("2026-06-01", 0.1), ("2026-08-01", 0.8)])[:, None]
    assert sf.sowing(fit, fit.copy(), WIN, np.array([False])).loc[0, "period"] == 0


def test_a_dip_of_a_standing_canopy_is_not_a_sowing():
    """aoi160 pixel 62535: bare 0.11 in mid May, 0.72 in June, a dip to 0.57 in mid July, 0.86 in August. The dip lies
    above half-way between its low and high: the sowing is the May low."""
    fit = curve([("2026-05-01", 0.25), ("2026-05-21", 0.11), ("2026-06-20", 0.72), ("2026-07-13", 0.57),
                 ("2026-08-15", 0.86), ("2026-09-26", 0.58)])[:, None]
    s = sf.sowing(fit, fit.copy(), WIN, np.array([True]))
    assert pd.Timestamp(s.loc[0, "sowing_date"]) == pd.Timestamp("2026-05-21") and s.loc[0, "period"] == 1


def test_a_late_trough_seen_bare_on_two_clear_views_counts_without_the_radar():
    """aoi160 pixel 68729: bare on 17 Aug, 6 Sep and 16 Sep (clear views), trough 1 Sep, the radar does not agree."""
    fit = curve([("2026-05-01", 0.35), ("2026-06-20", 0.17), ("2026-07-20", 0.39), ("2026-09-01", 0.17),
                 ("2026-09-26", 0.53)])[:, None]
    raw = np.full_like(fit, np.nan)
    for d, v in (("2026-08-19", 0.21), ("2026-09-03", 0.19), ("2026-09-13", 0.23)):        # window dates
        raw[WIN == d, 0] = v
    day = pd.date_range("2026-05-03", "2026-09-24", freq="6D")
    vh = np.full((len(day), 1), -15.0) + np.where(np.arange(len(day)) % 2 == 0, 0.2, -0.2)[:, None]
    s = sf.sowing(fit, raw, WIN, np.array([True]), series=[(day, {"VH": vh})], level=np.array([0.3]))
    assert pd.Timestamp(s.loc[0, "sowing_date"]) >= pd.Timestamp("2026-08-25")


def test_radar_sowing_is_the_last_low_before_the_rise_on_every_track():
    day = pd.date_range("2026-05-02", periods=12, freq="12D")
    vh = np.array([-15, -16, -22, -23, -22, -21, -23, -19, -18, -16, -14, -12], dtype="float32")[:, None]
    vv = np.array([-9, -10, -12, -13, -10, -9, -12, -8, -6, -5, -4, -3], dtype="float32")[:, None]
    flat_tree = np.full((12, 1), -15.0, dtype="float32") + np.array([0.3, -0.3] * 6, dtype="float32")[:, None]
    got = sf.radar_sowing([(day, {"VH": np.hstack([vh, flat_tree]), "VV": np.hstack([vv, flat_tree])})])
    assert got[0] == np.datetime64(day[6].date()) and np.isnat(got[1])
