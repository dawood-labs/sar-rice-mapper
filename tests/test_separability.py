"""Tests for the separability measures. Synthetic arrays only."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import separability as sep


def test_auc_is_one_when_the_classes_do_not_overlap():
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert sep.auc(values, labels) == 1.0


def test_auc_is_zero_when_they_are_separated_the_other_way():
    values = np.array([10.0, 11.0, 12.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert sep.auc(values, labels) == 0.0
    assert sep.separation(0.0) == 1.0          # just as informative


def test_auc_is_a_half_for_a_feature_that_carries_nothing():
    rng = np.random.default_rng(0)
    values = rng.normal(size=4000)
    labels = rng.random(4000) < 0.5
    assert abs(sep.auc(values, labels) - 0.5) < 0.03


def test_ties_are_given_their_average_rank():
    """A constant feature separates nothing; without tie handling it would score 1.0."""
    values = np.full(100, 3.0)
    labels = np.array([True] * 50 + [False] * 50)
    assert sep.auc(values, labels) == 0.5


def test_auc_ignores_nan_values():
    values = np.array([1.0, np.nan, 3.0, 10.0, np.nan, 12.0])
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    assert sep.auc(values, labels) == 1.0


def test_auc_is_nan_when_one_class_is_empty():
    assert np.isnan(sep.auc(np.arange(5.0), np.zeros(5, dtype=bool)))


def test_per_step_finds_the_step_where_the_classes_differ():
    labels = np.array([True] * 50 + [False] * 50)
    cube = np.random.default_rng(1).normal(size=(5, 100))
    cube[3, :50] += 8.0                              # only step 3 carries the difference
    dates = pd.date_range("2025-05-01", periods=5, freq="5D")
    frame = sep.per_step(dates, {"VH": cube}, labels)
    assert frame["VH_sep"].idxmax() == 3
    assert frame.loc[3, "VH_sep"] > 0.9 and frame.loc[0, "VH_sep"] < 0.3


def test_class_curves_report_both_classes_with_a_band():
    labels = np.array([True] * 30 + [False] * 30)
    cube = np.zeros((2, 60))
    cube[:, :30] = 5.0
    frame = sep.class_curves(pd.date_range("2025-05-01", periods=2, freq="5D"),
                             {"VH": cube}, labels)
    assert frame.loc[0, "VH_rice_med"] == 5.0 and frame.loc[0, "VH_other_med"] == 0.0
    assert frame.loc[0, "VH_rice_lo"] <= frame.loc[0, "VH_rice_hi"]


def test_summary_features_describe_the_whole_season():
    cube = np.array([[1.0, 5.0], [3.0, 9.0], [2.0, 7.0]])
    out = sep.summary_features({"VH": cube})
    assert list(out["VH_min"]) == [1.0, 5.0]
    assert list(out["VH_max"]) == [3.0, 9.0]
    assert list(out["VH_range"]) == [2.0, 4.0]


def test_rank_features_puts_the_informative_one_first():
    labels = np.array([True] * 40 + [False] * 40)
    good = np.concatenate([np.full(40, 9.0), np.full(40, 1.0)])
    useless = np.tile([1.0, 2.0], 40)
    frame = sep.rank_features({"useless": useless, "good": good}, labels)
    assert frame.loc[0, "feature"] == "good"
    assert frame.loc[0, "separation"] > frame.loc[1, "separation"]


def test_align_cube_puts_each_pixel_on_its_own_clock():
    """Two identical crops sown five weeks apart must align to the same relative column."""
    dates = pd.date_range("2025-03-01", periods=40, freq="5D")
    cube = np.zeros((40, 2))
    cube[10, 0] = 9.0          # pixel 0 spikes at index 10
    cube[17, 1] = 9.0          # pixel 1 spikes 35 days later
    events = pd.to_datetime([dates[10], dates[17]])
    out = sep.align_cube(cube, dates, events, rel_days=[-10, -5, 0, 5])
    assert out[2, 0] == 9.0 and out[2, 1] == 9.0
    assert out[0, 0] == 0.0 and out[3, 1] == 0.0


def test_align_cube_returns_nan_outside_the_observed_period():
    dates = pd.date_range("2025-03-01", periods=10, freq="5D")
    cube = np.ones((10, 1))
    out = sep.align_cube(cube, dates, pd.to_datetime([dates[0]]), rel_days=[-20, 0, 20])
    assert np.isnan(out[0, 0]) and out[1, 0] == 1.0 and out[2, 0] == 1.0


def test_earliest_call_finds_the_first_day_past_the_target():
    frame = pd.DataFrame({"rel_day": [0, 10, 20, 30], "VH_sep": [0.2, 0.5, 0.75, 0.9]})
    assert sep.earliest_call(frame, "VH", 0.7)["rel_day"] == 20
    assert sep.earliest_call(frame, "VH", 0.95) is None
