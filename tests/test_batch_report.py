import numpy as np

from sar_pipeline.analysis import batch_report as br


def test_bare_end_is_the_last_bare_window_before_the_climb():
    # one pixel: bare from window 0 to 5, climbs at 6
    ndvi = np.array([[.2], [.25], [.2], [.3], [.35], [.4], [.7], [.8]])
    assert br.bare_end(ndvi, np.array([0]), np.array([6]))[0] == 5
    # no climb: the search runs to the end
    assert br.bare_end(ndvi, np.array([0]), np.array([-1]))[0] == 5


def test_bare_end_falls_back_to_the_trough():
    ndvi = np.array([[.6], [.3], [.7]])
    assert br.bare_end(ndvi, np.array([1]), np.array([2]), level=.2)[0] == 1


def test_drop_before_climb_uses_only_the_span():
    dates = np.array(["2026-06-01", "2026-06-13", "2026-06-25", "2026-07-07", "2026-07-19"], dtype="datetime64[D]")
    cube = np.array([[-8.0], [-15.0], [-9.0], [-8.0], [-2.0]])   # a flood on 13 Jun
    inside = br.drop_before_climb(dates, cube, np.array(["2026-06-05"], dtype="datetime64[D]"),
                                  np.array(["2026-07-01"], dtype="datetime64[D]"))
    outside = br.drop_before_climb(dates, cube, np.array(["2026-06-20"], dtype="datetime64[D]"),
                                   np.array(["2026-07-10"], dtype="datetime64[D]"))
    assert inside[0] == 7.0 and outside[0] == 1.0


def test_not_rice_parts_are_exclusive_and_in_order():
    # four class-0 pixels: water now, never green, evergreen, other; one rice pixel ignored
    ndvi = np.array([[.1, .2, .7, .2, .2],
                     [.3, .3, .8, .8, .8],
                     [.1, .3, .7, .6, .7]])
    lswi = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [.3, 0, 0, 0, 0]])
    parts = br.not_rice_parts(np.array([0, 0, 0, 0, 1]), ndvi, lswi, np.array([True, True, True]))
    one = br.mr.acres(1)
    assert parts == {"water_now": round(one, 1), "never_canopy": round(one, 1),
                     "evergreen": round(one, 1), "other": round(one, 1)}
