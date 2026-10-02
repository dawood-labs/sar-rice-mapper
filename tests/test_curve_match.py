"""Fresh start, step 3 by similarity search: aligned at the sowing, level kept, nearest labelled pattern."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import curve_match as cm


def _rise(start, top, days_to_top, n=cm.STEPS):
    s = np.arange(n) * cm.STEP
    return start + (top - start) * np.clip(s / days_to_top, 0, 1)


def _table(curves, names, rice, region):
    t = pd.DataFrame({"pattern": names, "region": region, "is_rice": rice, "pixel": -1})
    for c in cm.CHANNELS:
        for s in range(cm.STEPS):
            t[f"{c}_{s}"] = np.nan
        t[f"noise_{c}"] = 0.02
    for s in range(cm.STEPS):
        t[f"ndvi_{s}"] = curves[:, s]
    return t


def test_aligned_reads_each_curve_from_its_own_sowing_and_radar_as_change():
    c = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.1, 0.3, 0.6]])
    out = cm.aligned(c, [1, 0], steps=3)
    assert np.allclose(out[0], [0.2, 0.3, 0.4]) and np.allclose(out[1], [0.5, 0.1, 0.3])
    rel = cm.aligned(np.array([[-15.0, -22.0, -18.0, -14.0]]), [1], steps=3, relative=True)
    assert np.allclose(rel[0], [0.0, 4.0, 8.0])


def test_a_late_sown_rice_matches_the_rice_pattern_and_a_weak_crop_the_weak_pattern():
    rng = np.random.default_rng(0)
    rice = np.stack([_rise(0.15, 0.8, 60) + rng.normal(0, 0.01, cm.STEPS) for _ in range(20)])
    weak = np.stack([_rise(0.25, 0.45, 40) + rng.normal(0, 0.01, cm.STEPS) for _ in range(20)])
    t = _table(np.vstack([rice, weak]), ["rice"] * 20 + ["weak"] * 20, [True] * 20 + [False] * 20, "r1")
    pat = cm.patterns(t)
    # a pixel sown in August: only 50 days seen, growing a little slower than the pattern
    young = np.full((1, cm.STEPS), np.nan, dtype="float32")
    young[0, :11] = _rise(0.15, 0.8, 70)[:11]
    weakpx = _rise(0.25, 0.45, 40)[None].astype("float32")
    ch = {"ndvi": np.vstack([young, weakpx]), "vh": np.full((2, cm.STEPS), np.nan), "vv": np.full((2, cm.STEPS), np.nan)}
    noise = {c: np.full(2, 0.02) for c in cm.CHANNELS}
    cls, best, _ = cm.decide(cm.distances(ch, noise, pat), pat)
    assert cls.tolist() == [1, 0]


def test_a_curve_unlike_every_pattern_is_flagged_not_guessed():
    rng = np.random.default_rng(1)
    rice = np.stack([_rise(0.15, 0.8, 60) + rng.normal(0, 0.01, cm.STEPS) for _ in range(20)])
    pat = cm.patterns(_table(rice, ["rice"] * 20, [True] * 20, "r1"))
    water = np.full((1, cm.STEPS), -0.3, dtype="float32")
    ch = {"ndvi": water, "vh": np.full((1, cm.STEPS), np.nan), "vv": np.full((1, cm.STEPS), np.nan)}
    cls, _, d = cm.decide(cm.distances(ch, {c: np.full(1, 0.02) for c in cm.CHANNELS}, pat), pat)
    assert cls.tolist() == [3] and d[0] > 2


def test_a_stretched_pattern_reads_the_same_curve_slower():
    m = np.arange(cm.STEPS, dtype="float32")[None]
    assert np.allclose(cm._stretch(m, 2.0)[0, :5], [0, 0.5, 1, 1.5, 2])


def test_hold_last_fills_the_tail_after_the_series_end():
    x = np.array([[0.1, 0.5, np.nan, np.nan]])
    assert np.allclose(cm._hold_last(x)[0], [0.1, 0.5, 0.5, 0.5])


def test_radar_only_pixels_match_only_radar_only_patterns():
    rng = np.random.default_rng(2)
    opt = _table(np.stack([_rise(0.15, 0.8, 60) + rng.normal(0, 0.01, cm.STEPS) for _ in range(10)]),
                 ["opt"] * 10, [True] * 10, "r1")
    for s in range(cm.STEPS):
        opt[f"vh_{s}"] = min(s, 8) * 1.0
    rad = opt.copy()
    rad["pattern"], rad["is_rice"] = "radar", False
    for s in range(cm.STEPS):
        rad[f"ndvi_{s}"] = np.nan
    pat = cm.patterns(pd.concat([opt, rad], ignore_index=True))
    vh = np.array([[min(s, 8) * 1.0 for s in range(cm.STEPS)]], dtype="float32")
    ch = {"ndvi": np.full((1, cm.STEPS), np.nan, dtype="float32"), "vh": vh, "vv": np.full((1, cm.STEPS), np.nan)}
    cls, best, _ = cm.decide(cm.distances(ch, {c: np.full(1, 0.5) for c in cm.CHANNELS}, pat), pat)
    assert np.array(pat["name"])[best][0] == "radar" and cls.tolist() == [0]


def test_a_near_tie_goes_to_rice():
    pat = {"is_rice": np.array([False, True])}
    d = np.array([[0.88 ** 2, 0.94 ** 2], [0.5 ** 2, 3.0 ** 2]], dtype="float32")
    cls, best, _ = cm.decide(d, pat)
    assert cls.tolist() == [1, 0] and best.tolist() == [1, 0]
