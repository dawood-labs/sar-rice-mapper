import numpy as np

from sar_pipeline.analysis import signal_agreement as sa


def test_longest_gap_counts_from_the_period_edges():
    dates = ["2026-06-11", "2026-07-01", "2026-09-20"]
    ok = np.array([[True, False], [True, False], [True, False]])
    gap = sa.longest_gap_days(dates, ok)
    assert gap[0] == 81          # 1 Jul -> 20 Sep
    assert gap[1] == 121         # never clear: the whole Jun-Sep period


def test_pass_wobble_is_the_median_pass_to_pass_change():
    cube = np.array([[-10.0, -15.0], [-12.0, -15.0], [-10.0, -15.0], [-11.0, np.nan]])
    assert sa.pass_wobble(cube).tolist() == [2.0, 0.0]


def test_rank_corr_sees_moving_together_and_apart():
    a = np.arange(6.0)[:, None].repeat(3, axis=1)
    b = np.c_[np.arange(6.0) ** 2, -np.arange(6.0), [1, np.nan, np.nan, np.nan, 2, 3]]
    r = sa.rank_corr(a, b)
    assert r[0] == 1.0 and r[1] == -1.0 and np.isnan(r[2])
