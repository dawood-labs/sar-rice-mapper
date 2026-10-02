"""Calendar-free traits of a curve and a small tree learned from labelled plots."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import rice_features as rf


def test_traits_of_a_rice_curve_and_a_tree_curve():
    t = np.arange(40)
    rice = np.where(t < 12, 0.15, np.minimum(0.15 + (t - 12) * 0.05, 0.8))     # bare, then +0.05 per window
    tree = np.full(40, 0.7)
    vh = np.vstack([np.where((t >= 10) & (t <= 13), -22.0, np.where(t > 13, -20 + (t - 13) * 0.3, -16.0)),
                    np.full(40, -14.0)])
    tr = rf.traits(np.vstack([rice, tree]), vh, np.array([11, 0]), np.array([0.02, 0.02]))
    r = tr.iloc[0]
    assert abs(r["trough"] - 0.15) < 1e-9 and abs(r["rise"] - 0.65) < 1e-9
    assert r["days_half"] == 40 and r["low_days"] >= 50                          # 7 windows to half of the rise
    assert r["vh_drop"] > 4 and r["vh_rise"] > 5
    assert tr.iloc[1]["rise"] == 0 and abs(tr.iloc[1]["year_low"] - 0.7) < 1e-6


def test_tree_learns_and_the_check_leaves_a_region_out():
    rng = np.random.default_rng(0)
    n = 200
    t = pd.DataFrame({f: rng.normal(0, 1, n) for f in rf.FEATURES})
    t["rise"] = np.r_[rng.normal(0.6, 0.05, 100), rng.normal(0.15, 0.05, 100)]
    t["is_rice"] = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    t["region"] = np.tile(["A", "B"], 100)
    c = rf.check(t)
    assert (c.loc[c["plots"] == "rice groups", "called rice %"] > 90).all()
    assert (c.loc[c["plots"] == "not-rice groups", "called rice %"] < 10).all()
    tree, _ = rf.fit_tree(t)
    assert "rise" in rf.rules(tree)


def test_special_cases_late_rice_drowned_and_weak_greenery():
    rng = np.random.default_rng(0)
    n = 100
    tr = pd.DataFrame({"top": np.r_[rng.normal(0.8, 0.03, n), 0.8, 0.45, 0.8],
                       "trough": np.r_[np.full(n, 0.15), 0.15, 0.15, -0.2],
                       "rise": np.r_[np.full(n, 0.65), 0.65, 0.3, 0.01]})
    pred = np.r_[np.ones(n, bool), True, True, True]
    sowing = pd.Series(pd.to_datetime(["2026-06-01"] * n + ["2026-08-10", "2026-06-01", "2026-06-01"]))
    rf.WEAK_RULE = True
    try:
        out = rf.special_cases(tr, pred, sowing, np.full(n + 3, 0.02))
    finally:
        rf.WEAK_RULE = False
    assert (out[:n] == 1).mean() > 0.95
    assert out[n:].tolist() == [6, 8, 7]                       # late rice, weak greenery, drowned
    assert rf.special_cases(tr, pred, sowing, np.full(n + 3, 0.02))[n + 1] == 1       # the weak rule is off by default


def test_late_rice_is_found_from_the_last_rise_even_with_a_may_sowing():
    """aoi160 groups 17-20: step 2 kept a May sowing, but the crop standing now began to rise in August."""
    dates = pd.date_range("2026-03-05", "2026-09-21", freq="5D")
    x = [pd.Timestamp(d).value for d in ("2026-03-05", "2026-05-01", "2026-06-10", "2026-08-05", "2026-09-21")]
    late = np.interp(dates.as_unit("ns").asi8, x, [0.3, 0.15, 0.25, 0.12, 0.5])
    start = rf.last_rise_start(late[None, :], dates, np.array([0.02]))
    assert start.iloc[0] >= pd.Timestamp("2026-07-20")
    tr = pd.DataFrame({"top": [0.5], "trough": [0.15], "rise": [0.35]})
    out = rf.special_cases(tr, np.array([True]), pd.Series(pd.to_datetime(["2026-05-01"])), np.array([0.02]),
                           rise_start=start)
    assert out.tolist() == [6]


def test_a_dip_of_a_standing_crop_is_not_a_late_start():
    dates = pd.date_range("2026-03-05", "2026-09-21", freq="5D")
    x = [pd.Timestamp(d).value for d in ("2026-03-05", "2026-05-01", "2026-07-01", "2026-08-10", "2026-09-21")]
    dip = np.interp(dates.as_unit("ns").asi8, x, [0.3, 0.15, 0.75, 0.47, 0.75])            # group 10: a standing crop dips
    assert pd.isna(rf.last_rise_start(dip[None, :], dates, np.array([0.02]), level=np.array([0.3])).iloc[0]) or \
        rf.last_rise_start(dip[None, :], dates, np.array([0.02]), level=np.array([0.3])).iloc[0] < pd.Timestamp("2026-07-20")


def test_radar_traits_and_a_radar_only_tree():
    t = np.arange(30)
    vh = np.vstack([np.where(t < 10, -24.0, np.minimum(-24 + (t - 10) * 0.8, -16.0)),     # water, then a crop
                    np.full(30, -17.0)])                                                   # flat
    fit = np.vstack([np.full(30, 0.2), np.full(30, 0.2)])
    tr = rf.traits(fit, vh, np.array([10, 10]), np.array([0.02, 0.02]))
    assert tr.loc[0, "vh_rise"] == 8 and tr.loc[0, "vh_half_days"] == 25 and tr.loc[0, "vh_sow"] == -24
    assert tr.loc[1, "vh_rise"] == 0 and np.isnan(tr.loc[1, "vh_half_days"])
    assert tr.loc[0, "vh_end_vs_dry"] == 8 and tr.loc[0, "vh_sow_vs_dry"] == 0        # dry level -24 (the first windows)
    rng = np.random.default_rng(0)
    d = pd.DataFrame({f: rng.normal(0, 1, 200) for f in rf.RADAR_FEATURES})
    d["vh_rise"] = np.r_[rng.normal(6, 1, 100), rng.normal(1, 1, 100)]
    d["is_rice"] = np.r_[np.ones(100, bool), np.zeros(100, bool)]
    tree, fill = rf.fit_tree(d, rf.RADAR_FEATURES)
    assert tree.predict(d[rf.RADAR_FEATURES]).mean() > 0.4 and "vh_rise" in rf.rules(tree, rf.RADAR_FEATURES)


def test_a_clear_low_at_sowing_lowers_the_trough():
    """aoi160 pixel 106113: the smooth curve reads 0.30 on the sowing day, the clear view 0.17; the rise counts from 0.17."""
    t = np.arange(20)
    fit = np.where(t < 5, 0.30, np.minimum(0.30 + (t - 5) * 0.05, 0.78))[None, :]
    raw = np.full_like(fit, np.nan)
    raw[0, 5] = 0.17
    tr = rf.traits(fit, np.full_like(fit, -15.0), np.array([5]), np.array([0.02]), raw=raw)
    assert abs(tr.loc[0, "trough"] - 0.17) < 1e-9 and abs(tr.loc[0, "rise"] - 0.61) < 1e-9


def test_a_young_late_crop_is_late_rice_even_when_the_tree_says_no():
    tr = pd.DataFrame({"top": [0.45, 0.2], "trough": [0.15, 0.15], "rise": [0.3, 0.01]})
    start = pd.Series(pd.to_datetime(["2026-08-12", "2026-08-12"]))
    out = rf.special_cases(tr, np.array([False, False]), start, np.array([0.02, 0.02]), rise_start=start)
    assert out.tolist() == [6, 2]                    # greening since August: late rice; flat: not rice


def test_vv_traits_follow_the_double_bounce():
    t = np.arange(20)
    vv = np.where(t < 8, -14.0, -5.0)[None, :]                     # aoi13 pixel 10120: VV up ~10 dB
    tr = rf.traits(np.full((1, 20), 0.2), np.full((1, 20), -18.0), np.array([6]), np.array([0.02]), vv=vv)
    assert tr.loc[0, "vv_sow"] == -14 and tr.loc[0, "vv_rise"] == 9
    assert np.isnan(rf.traits(np.full((1, 20), 0.2), np.full((1, 20), -18.0), np.array([6]), np.array([0.02]))
                    .loc[0, "vv_rise"])


def test_series_windows_run_to_the_newest_window():
    w = pd.date_range("2025-09-01", "2026-09-26", freq="5D")
    idx, dates = rf.series_windows(w)
    assert dates[0] >= pd.Timestamp(rf.CURVE_START) and dates[-1] == w[-1] and len(idx) == len(dates)


def test_a_crop_still_greening_at_the_end_is_rice_or_late_rice_whatever_the_tree_says():
    tr = pd.DataFrame({"rise": [0.4, 0.4, 0.4], "top": [0.52, 0.53, 0.8], "trough": [0.15, 0.11, 0.1],
                       "end_climb": [0.05, 0.11, 0.0], "fall_after_top": [0.0, 0.0, 0.0]})
    sowing = pd.to_datetime(["2026-07-10", "2026-08-05", "2026-06-01"])
    out = rf.special_cases(tr, [False, False, False], sowing, np.full(3, 0.02))
    assert out.tolist() == [1, 6, 2]                    # grown curve (no climb at the end) stays with the tree


def test_label_audit_flags_wrong_pixels_and_group_conflicts(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    a = tmp_path / "aoi1"
    a.mkdir()
    prof = {"driver": "GTiff", "width": 4, "height": 1, "count": 1, "dtype": "uint8", "crs": "EPSG:32633",
            "transform": from_origin(500000, 0, 10, 10)}
    for name, vals in (("aoi1_step3t_class.tif", [1, 2, 6, 1]), ("aoi1_groups_k20.tif", [1, 1, 2, 2])):
        with rasterio.open(a / name, "w", **prof) as ds:
            ds.write(np.array([vals], dtype="uint8")[None])
    pd.DataFrame({"group": [1, 2], "rice": [True, False]}).to_csv(a / "aoi1_group_verdicts_k20.csv", index=False)
    pd.DataFrame({"pixel": [0, 1, 2], "label": ["rice", "rice", "rice"], "rice": [True, True, True]}).to_csv(
        a / "aoi1_pixel_verdicts.csv", index=False)
    pix, grp = rf.label_audit(fresh=str(tmp_path))
    assert pix["agrees"].tolist() == ["yes", "NO", "near (rice vs late rice)"]
    assert pix["pixel vs group"].tolist() == ["same", "same", "CONFLICT"]
    assert grp["agrees %"].tolist() == [50.0, 0.0]
