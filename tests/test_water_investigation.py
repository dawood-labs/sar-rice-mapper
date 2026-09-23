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
