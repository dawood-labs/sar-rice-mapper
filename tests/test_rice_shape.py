"""Fresh start, step 3: rice by the shape of the curve after sowing, learned from surveyed rice."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import rice_shape as rs


def rice_curve(sow, n=30, low=0.15, high=0.8, days_to_top=60):
    t = (np.arange(n) - sow) * rs.STEP
    return np.where(t < 0, low, low + (high - low) * np.clip(t / days_to_top, 0, 1))


def test_shapes_put_every_pixel_on_its_own_scale():
    fit = np.stack([rice_curve(4), rice_curve(4, high=0.5)], axis=1)          # a bright field and a dim edge pixel
    sh, age, amp = rs.shapes(fit, np.array([4, 4]))
    np.testing.assert_allclose(sh[:, 0], sh[:, 1], equal_nan=True, atol=1e-6)
    assert sh[0, 0] == 0 and np.nanmax(sh[:, 0]) == 1
    assert age.tolist() == [25, 25] and abs(amp[1] - 0.35) < 1e-6


def table_of(curves, region="A"):
    fit = np.stack(curves, axis=1)
    sh, age, amp = rs.shapes(fit, np.full(fit.shape[1], 2))
    t = pd.DataFrame(sh.T, columns=[f"s{i}" for i in range(rs.STEPS)])
    return t.assign(region=region, age_steps=age, amplitude=amp, noise=0.02)


def test_judge_rice_other_undecided_and_old():
    rng = np.random.default_rng(0)
    ref = rs.reference(table_of([rice_curve(2, days_to_top=d) for d in rng.uniform(50, 70, 200)]))
    fit = np.stack([rice_curve(2, days_to_top=60, high=0.5),                 # rice, dim edge pixel
                    rice_curve(2, days_to_top=15),                            # greens up in two weeks: grass
                    rice_curve(26, days_to_top=60),                           # sown three windows ago
                    rice_curve(2)], axis=1)
    sow = np.array([2, 2, 26, 2])
    sh, age, amp = rs.shapes(fit, sow)
    age = age.copy()
    age[3] = 35                                                                 # 175 days old
    cls, gap = rs.judge(sh, age, amp, np.full(4, 0.02), ref)
    assert cls.tolist() == [1, 2, 4, 3]


def test_reference_leaves_a_region_out_and_needs_enough_pixels():
    a = table_of([rice_curve(2)] * 60, "A")
    b = table_of([rice_curve(2, days_to_top=20)] * 10, "B")
    ref = rs.reference(pd.concat([a, b]), exclude_region="A")
    assert ref["median"].isna().all()                                          # only 10 pixels left


def test_judge_gap_is_zero_on_the_reference_itself():
    rng = np.random.default_rng(1)
    t = table_of([rice_curve(2, days_to_top=d) for d in rng.uniform(55, 65, 100)])
    ref = rs.reference(t)
    sh = ref["median"].to_numpy()[:, None]
    cls, gap = rs.judge(sh, np.array([20]), np.array([0.6]), np.array([0.0]), ref)
    assert cls[0] == 1 and gap[0] < 1e-6


def test_a_cloudy_month_dip_is_left_out_unless_the_radar_falls_too():
    """aoi160 pixel 50746: sown 4 May, 0.64 in June, a dip to 0.43 in early August from a hazy view, 0.85 in September.
    Pixel 0: VH flat (the dip is haze, left out, rice). Pixel 1: VH falls in August too (the dip counts)."""
    w = pd.date_range("2026-05-04", "2026-09-26", freq="5D")
    x = [pd.Timestamp(d).value for d in ("2026-05-04", "2026-06-25", "2026-08-05", "2026-09-15", "2026-09-26")]
    one = np.interp(w.as_unit("ns").asi8, x, [0.24, 0.64, 0.43, 0.85, 0.85])
    fit = np.stack([one, one], axis=1)
    day = pd.date_range("2026-05-03", "2026-09-24", freq="6D")
    d = day.to_numpy()
    vh = np.full((len(day), 2), -16.0) + np.where(np.arange(len(day)) % 2 == 0, 0.2, -0.2)[:, None]
    vh[(d >= np.datetime64("2026-07-25")) & (d <= np.datetime64("2026-08-15")), 1] = -22.0
    ign = rs.radar_unconfirmed_dips(fit, np.array([0, 0]), w, [(day, {"VH": vh})])
    aug = (w[:rs.STEPS] >= "2026-07-25") & (w[:rs.STEPS] <= "2026-08-10")
    assert ign[:len(aug)][aug, 0].all() and not ign[:, 1].any()
    assert not ign[:len(w)][(w[:rs.STEPS] < "2026-07-01")].any()          # June is not a cloudy month here


def test_a_dip_resting_on_one_clear_view_is_left_out_even_when_the_radar_falls():
    """aoi160 pixel 50746 (user): the only low view of the dip (7 Aug) was hazy; the radar was dark in early July from
    water in the standing paddy. One view does not make a dip; two views and a radar fall do."""
    w = pd.date_range("2026-05-04", "2026-09-26", freq="5D")
    x = [pd.Timestamp(d).value for d in ("2026-05-04", "2026-06-25", "2026-08-05", "2026-09-15", "2026-09-26")]
    one = np.interp(w.as_unit("ns").asi8, x, [0.24, 0.64, 0.43, 0.85, 0.85])
    fit = np.stack([one, one], axis=1)
    raw = np.full_like(fit, np.nan)
    raw[w == "2026-06-03", :] = 0.55                               # the crop, clear
    raw[w == "2026-07-18", 0] = 0.55                               # pixel 0: clear, NOT lower than June
    raw[w == "2026-08-07", :] = 0.41                               # the one low view
    raw[w == "2026-07-28", 1] = 0.45                               # pixel 1: a second low view
    day = pd.date_range("2026-05-03", "2026-09-24", freq="6D")
    d = day.to_numpy()
    vh = np.full((len(day), 2), -16.0) + np.where(np.arange(len(day)) % 2 == 0, 0.2, -0.2)[:, None]
    vh[(d >= np.datetime64("2026-07-25")) & (d <= np.datetime64("2026-08-15")), :] = -22.0    # the radar falls for both
    ign = rs.radar_unconfirmed_dips(fit, np.array([0, 0]), w, [(day, {"VH": vh})], raw=raw, noise=np.full(2, 0.04))
    aug = (w[:rs.STEPS] >= "2026-07-25") & (w[:rs.STEPS] <= "2026-08-10")
    assert ign[:len(aug)][aug, 0].all() and not ign[:, 1].any()


def test_max_shift_runs_to_the_first_clearly_green_view():
    fit = np.array([0.27, 0.35, 0.45, 0.55, 0.65, 0.75])[:, None]
    raw = np.array([0.27, np.nan, np.nan, 0.57, np.nan, 0.75])[:, None]         # 4 May empty, 20 May green
    assert rs.max_shift(fit, raw, np.array([0]), np.array([0.03])).tolist() == [2]


def test_a_sowing_seen_late_and_a_ripening_crop_are_still_rice():
    """one region pixel 2036 (user: rice): the last empty view came two windows before the real planting, and the crop
    ripens after its top. Sliding the start and comparing only up to the top keep it rice; a two-week green-up stays
    other vegetation."""
    rng = np.random.default_rng(0)
    ref = rs.reference(table_of([rice_curve(2, days_to_top=d) for d in rng.uniform(50, 70, 200)]))
    real = rice_curve(4, days_to_top=60)                         # planted two windows after the chosen sowing
    real[20:] = real[20] - 0.02 * np.arange(len(real) - 20)      # ripening after the top
    grass = rice_curve(2, days_to_top=15)
    sh, age, amp = rs.shapes(np.stack([real, grass], axis=1), np.array([2, 2]))
    plain, _ = rs.judge(sh, age, amp, np.full(2, 0.02), ref, upto_peak=False)
    rs.SLIDE_SOWING = True
    try:
        cls, _ = rs.judge(sh, age, amp, np.full(2, 0.02), ref, shift=np.array([2, 2]), upto_peak=True)
    finally:
        rs.SLIDE_SOWING = False
    assert plain[0] == 2 and cls.tolist() == [1, 2]


def test_reference_uses_only_the_rice_groups_and_verdicts_attach(tmp_path):
    rice = table_of([rice_curve(2)] * 60).assign(is_rice=True)
    other = table_of([rice_curve(2, days_to_top=15)] * 60).assign(is_rice=False)
    ref = rs.reference(pd.concat([rice, other]))
    np.testing.assert_allclose(ref["median"], rs.reference(rice.drop(columns="is_rice"))["median"], equal_nan=True)
    groups = tmp_path / "g.csv"
    verdicts = tmp_path / "v.csv"
    pd.DataFrame({"plot_id": [1, 2], "aoi": ["aoi1", "aoi1"], "group": [3, 5]}).to_csv(groups, index=False)
    pd.DataFrame({"group": [3, 5], "rice": [True, False]}).to_csv(verdicts, index=False)
    t = rs.attach_verdicts(pd.DataFrame({"plot_id": [1, 2, 9], "aoi": "aoi1"}), str(groups), str(verdicts))
    assert t["is_rice"].tolist() == [True, False, False]
