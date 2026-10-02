"""First clear view per pixel and the image composed from those views."""
import numpy as np

from sar_pipeline.analysis import first_clear as fc


def test_first_clear_takes_the_earliest_clear_date_per_pixel():
    ok = np.array([[[False, True], [False, False]],
                   [[True, True], [False, False]],
                   [[True, False], [True, False]]])
    assert fc.first_clear(ok).tolist() == [[1, 0], [2, -1]]


def test_compose_takes_each_pixels_bands_from_its_date_and_zero_when_never_clear():
    stack = np.arange(3 * 2 * 2 * 2, dtype="uint16").reshape(3, 2, 2, 2)      # dates, bands, rows, cols
    index = np.array([[1, 0], [2, -1]])
    out = fc.compose(stack, index)
    assert out.shape == (2, 2, 2)
    assert out[:, 0, 0].tolist() == stack[1, :, 0, 0].tolist()
    assert out[:, 0, 1].tolist() == stack[0, :, 0, 1].tolist()
    assert out[:, 1, 0].tolist() == stack[2, :, 1, 0].tolist()
    assert out[:, 1, 1].tolist() == [0, 0]


def test_date_colours_are_distinct():
    c = fc.date_colours(8)
    assert len(set(c)) == 8


def test_relative_clear_rejects_a_hazy_view_when_the_pixel_was_seen_clearly_later():
    """User rule (30 Sep): a score of 50 is not clear where the same pixel reached ~90 on another date; a pixel whose
    best is 50 still takes that view."""
    rng = np.random.default_rng(0)
    n = 400
    cs = np.zeros((3, 1, n), dtype="float32")
    cs[0, 0] = 50.0                                    # hazy first date
    cs[1, 0] = rng.normal(90, 2, n)                    # clear later date
    cs[2, 0] = 0.0                                     # cloud
    cs[1, 0, -1] = 40.0                                # the last pixel never had a good view: best = the hazy 50
    base = np.ones_like(cs, dtype=bool)
    ok = fc.relative_clear(cs, base, np.array([False, False, False]), np.ones((1, n), bool))
    idx = fc.first_clear(ok)[0]
    assert (idx[:-1] == 1).mean() > 0.95
    assert idx[-1] == 0
    missing = fc.relative_clear(cs, base, np.array([False, True, False]), np.ones((1, n), bool))
    assert not missing[1].any()                        # a scene without a score is never taken


def test_otsu_splits_bare_and_green_between_the_two_groups():
    rng = np.random.default_rng(0)
    v = np.r_[rng.normal(0.15, 0.04, 3000), rng.normal(0.75, 0.05, 5000)]
    t = fc.otsu(v)
    assert 0.3 < t < 0.6


def test_radar_vegetation_sees_a_crop_that_grew_since_its_low():
    """aoi13 (user, 1 Oct): no clear September view, the radar decides. Pixel 0: flooded in July, canopy by September.
    Pixel 1: bare and flat all season. Pixel 2: a tree, high and flat (no rise)."""
    import pandas as pd

    day = pd.date_range("2026-05-03", "2026-09-28", freq="6D")
    d = day.to_numpy()
    vh = np.full((len(day), 3), -19.0)
    vh[:, 2] = -14.0
    vh[(d >= np.datetime64("2026-07-01")) & (d < np.datetime64("2026-07-20")), 0] = -24.0
    vh[d >= np.datetime64("2026-08-15"), 0] = -15.0
    vh += np.where(np.arange(len(day)) % 2 == 0, 0.3, -0.3)[:, None]
    assert fc.radar_vegetation([(day, {"VH": vh})]).tolist() == [True, False, False]


def test_radar_vegetation_also_reads_vv():
    import pandas as pd

    day = pd.date_range("2026-05-03", "2026-09-28", freq="6D")
    d = day.to_numpy()
    rng = np.random.default_rng(0)
    vh = -20 + rng.normal(0, 3, (len(day), 1))                       # VH too noisy to show a rise
    vv = np.where(d >= np.datetime64("2026-09-01"), -5.0, -14.0)[:, None] + rng.normal(0, 0.3, (len(day), 1))
    assert fc.radar_vegetation([(day, {"VH": vh, "VV": vv})]).tolist() == [True]


def test_radar_overrules_a_low_view_but_never_open_water():
    seen = np.array([True, True, True, False])
    veg = np.array([False, False, True, False])
    ndvi = np.array([0.10, -0.37, 0.6, np.nan])        # young paddy, open water, green, never seen
    out = fc.radar_overrule(seen, veg, np.ones(4, bool), np.ones(4, bool), ndvi)
    assert out.tolist() == [True, False, False, False]
