import numpy as np

from sar_pipeline.analysis import water_investigation as wi


def test_align_picks_the_nearest_date_and_blanks_far_ones():
    dates = np.array(["2026-06-01", "2026-06-13", "2026-06-25"], dtype="datetime64[D]")
    cube = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    out = wi.align(dates, cube, np.array([1]), np.array(["2026-06-13"], dtype="datetime64[D]"),
                   rel_days=np.array([-12, 0, 5, 40]), max_gap=7)
    assert out[0, :3].tolist() == [10.0, 20.0, 20.0]   # 18 Jun is nearer 13 Jun (5 d) than 25 Jun (7 d)
    assert np.isnan(out[0, 3])


def test_group_centers_are_deep_inside_one_class():
    c = np.zeros((20, 20), dtype=int)
    c[2:15, 2:15] = 3
    centers = wi.group_centers(c, 3, 50, context=9)
    rows, cols = np.divmod(centers, 20)
    assert len(centers) and (rows >= 6).all() and (rows <= 10).all() and (cols >= 6).all() and (cols <= 10).all()


def test_block_pids_is_a_square_around_the_centre():
    pids = wi.block_pids(5 * 10 + 5, 10, 10, block=3)
    assert sorted(pids.tolist()) == [44, 45, 46, 54, 55, 56, 64, 65, 66]


def test_ndvi_axis_never_hides_open_water():
    import numpy as np

    from sar_pipeline.analysis import water_investigation as wi

    assert wi.ndvi_ylim(np.array([0.2, 0.8])) == (-0.3, 1.05)
    low, top = wi.ndvi_ylim(np.array([0.1, -0.44, np.nan]), np.array([-0.2]))
    assert low <= -0.49 and top == 1.05


def test_view_window_runs_to_the_newest_image_by_default():
    """aoi160_005366 (user, 30 Sep): a clear 26 Sep image must reach the sheets; a fixed end still cuts."""
    import pandas as pd

    from sar_pipeline.analysis import water_investigation as wi

    dates = pd.to_datetime(["2026-09-16", "2026-09-26"])
    t0, t1 = wi.view_window(("2026-03-15", None), dates)
    assert t0 == pd.Timestamp("2026-03-15") and dates[-1] < t1
    assert wi.view_window(("2026-03-15", "2026-09-24"), dates)[1] == pd.Timestamp("2026-09-24")
