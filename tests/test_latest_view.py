import numpy as np

from sar_pipeline.analysis import latest_view as lv


def test_newest_clear_index_walks_back_past_clouds():
    ok = np.array([[True, True, False], [True, False, False], [False, False, False]])
    # pixel 0: clear on date 1 (date 2 cloudy); pixel 1: only date 0; pixel 2: never clear
    assert lv.newest_clear_index(ok).tolist() == [1, 0, -1]


def test_tone_is_relative_to_the_pixels_own_range():
    ndvi = np.array([0.80, 0.30, 0.05, 0.40, np.nan])
    lswi = np.array([0.30, 0.10, 0.40, 0.10, np.nan])
    lo = np.array([0.10, 0.10, -0.20, 0.35, 0.0])
    hi = np.array([0.85, 0.85, 0.80, 0.45, 1.0])
    # green high in its range; empty low in its range; water (LSWI above NDVI); 0.40 is green for a pixel whose
    # whole range is 0.35-0.45 (no fixed cut); no view -> ""
    assert lv.tone(ndvi, lswi, lo, hi).tolist() == ["green", "empty", "water", "green", ""]


def test_core_pixels_keep_field_interiors_only():
    c = np.zeros((7, 7), dtype=int)
    c[1:6, 1:6] = 3
    assert lv.core_pixels(c.ravel(), (7, 7), 3, 10).tolist() == [24]     # only the centre has a full 5x5 of class 3
