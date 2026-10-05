"""Tests of the per-pass radar images and the season summary (pure helpers; no network)."""
import numpy as np
import pandas as pd
import pytest

from sar_pipeline import s1_season_images as si


def _acq(rows):
    return pd.DataFrame(rows, columns=["track_id", "date_utc", "datetime_utc", "aoi_coverage_pct"])


def test_folder_name_uses_newest_date():
    assert si.folder_name("2026-09-30") == "s1_may_to_latest_2026-09-30"


def test_season_rows_keeps_context_before_start_and_drops_after_end():
    dates = pd.date_range("2026-03-01", "2026-06-30", freq="12D")
    acq = _acq([("T1_DSC", d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%dT23:30:00Z"), 100) for d in dates])
    out = si.season_rows(acq, "2026-05-01", "2026-06-15", half_window=3)
    ctx = out[~out["in_season"]]
    assert len(ctx) == 3 and (pd.to_datetime(ctx["date_utc"]) < "2026-05-01").all()
    season = out[out["in_season"]]
    assert pd.to_datetime(season["date_utc"]).min() >= pd.Timestamp("2026-05-01")
    assert pd.to_datetime(season["date_utc"]).max() <= pd.Timestamp("2026-06-15")


def test_choose_descending_prefers_configured_then_most_passes():
    d = ["2026-05-03", "2026-05-15", "2026-05-27"]
    acq = _acq([("T1_DSC", x, x, 100) for x in d] + [("T2_DSC", x, x, 100) for x in d[:2]]
               + [("T3_ASC", x, x, 100) for x in d])
    assert si.choose_descending(acq, ["T3_ASC", "T2_DSC"], "2026-05-01")[0] == "T2_DSC"
    assert si.choose_descending(acq, ["T3_ASC", "T1_DSC", "T2_DSC"], "2026-05-01")[0] == "T1_DSC"
    # nothing configured: a descending track offered by the archive that covers the AOI
    track, why = si.choose_descending(acq, ["T3_ASC"], "2026-05-01")
    assert track == "T1_DSC" and "not configured" in why


def test_choose_descending_never_substitutes_ascending():
    acq = _acq([("T3_ASC", "2026-05-03", "2026-05-03", 100), ("T1_DSC", "2026-05-04", "2026-05-04", 20)])
    track, why = si.choose_descending(acq, ["T3_ASC"], "2026-05-01")
    assert track is None and "none" in why


def test_encode_decode_round_trip_and_nodata():
    a = np.array([[-12.345, np.nan], [0.0, -30.0]])
    enc = si.encode_db(a)
    assert enc.dtype == np.int16 and enc[0, 1] == si.NODATA and enc[0, 0] in (-1234, -1235)
    dec = si.decode_db(enc)
    assert np.isnan(dec[0, 1]) and abs(dec[1, 1] + 30) < 1e-6


def test_date_bands_difference_is_vv_minus_vh():
    vv = np.full((2, 2), -8.0)
    vh = np.full((2, 2), -15.0)
    vh[0, 0] = np.nan
    b = si.date_bands(vv, vh)
    assert b.shape == (3, 2, 2)
    assert b[2, 1, 1] == 700 and b[2, 0, 0] == si.NODATA and b[0, 0, 0] == -800


def test_place_clips_to_target():
    t = np.full((1, 3, 3), np.nan, dtype="float32")
    si.place(t, np.ones((1, 2, 2), dtype="float32"), 2, 2)
    assert np.nansum(t) == 1 and t[0, 2, 2] == 1


def test_season_summary_paddy_story():
    dates = pd.to_datetime(["2026-05-05", "2026-06-01", "2026-07-01", "2026-08-01", "2026-09-01"])
    # pixel 0: a paddy - VH high early (weeds), water in June, crop peak in September
    vv0 = [-9, -18, -12, -10, -9]
    vh0 = [-14, -25, -20, -15, -13]
    # pixel 1: VH peak BEFORE the water must not be taken
    vv1 = [-9, -10, -20, -15, -14]
    vh1 = [-10, -18, -24, -19, -17]
    vv = np.array([vv0, vv1], dtype="float32").T[:, :, None]      # (t, 2, 1)
    vh = np.array([vh0, vh1], dtype="float32").T[:, :, None]
    s = si.season_summary(dates, vv, vh)
    assert s["vvmin"][0, 0] == -18 and s["vvmin_day"][0, 0] == np.datetime64("2026-06-01")
    assert s["vhmax"][0, 0] == -13 and s["vhmax_day"][0, 0] == np.datetime64("2026-09-01")
    assert s["base"][0, 0] == -25 and s["vh_range"][0, 0] == 12
    assert s["vvmin_day"][1, 0] == np.datetime64("2026-07-01")
    assert s["vhmax"][1, 0] == -17 and s["base"][1, 0] == -24        # not the early -10
    assert (s["vh_range"] >= 0).all()


def test_season_summary_nan_and_ties():
    dates = pd.to_datetime(["2026-05-05", "2026-05-17", "2026-05-29"])
    vv = np.array([[np.nan], [-15], [-15]], dtype="float32")
    vh = np.array([[-20], [np.nan], [-18]], dtype="float32")
    s = si.season_summary(dates, vv, vh)
    assert s["vvmin_day"][0] == np.datetime64("2026-05-17")          # first of the tie
    assert s["vhmax"][0] == -18 and s["base"][0] == -20               # base looks back to the start
    empty = si.season_summary(dates, np.full((3, 1), np.nan), np.full((3, 1), -20.0))
    assert np.isnan(empty["vvmin"][0]) and np.isnan(empty["vh_range"][0]) and np.isnat(empty["vhmax_day"][0])


def test_day_of_year():
    d = np.array(["2026-01-01", "2026-05-01", "NaT"], dtype="datetime64[D]")
    assert list(si.day_of_year(d)) == [1, 121, si.DOY_NODATA]


def test_token_date_handles_second_pass_suffix():
    assert si.token_date("20260705_2") == pd.Timestamp("2026-07-05")


def test_known_bad_selects_aoi_track_pol():
    t = pd.DataFrame({"aoi": ["aoi1", "aoi1", "aoi2"], "track": ["T1_DSC"] * 3, "pol": ["VH", "VV", "VH"],
                      "date": ["2026-06-11"] * 3, "bad": [True, False, True]})
    assert si.known_bad(t, "aoi1", "T1_DSC", "VH") == {pd.Timestamp("2026-06-11")}
    assert si.known_bad(t, "aoi1", "T1_DSC", "VV") == set()


def test_known_bad_applies_track_wide_passes_to_unjudged_aois():
    t = pd.DataFrame({"aoi": ["aoi1", "aoi2"], "track": ["T1_DSC"] * 2, "pol": ["VH", "VH"],
                      "date": ["2026-06-11", "2026-07-01"], "bad": [True, True], "reason": ["track_wide", "jump"]})
    assert si.known_bad(t, "aoi9", "T1_DSC", "VH") == {pd.Timestamp("2026-06-11")}


def test_smooth_db_averages_in_power():
    cube = np.full((1, 5, 5), -20.0, dtype="float32")
    cube[0, 2, 2] = -10.0
    out = si.smooth_db(cube, 5)
    expected = 10 * np.log10((24 * 10 ** -2 + 10 ** -1) / 25)
    assert out[0, 2, 2] == pytest.approx(expected, abs=1e-4)
