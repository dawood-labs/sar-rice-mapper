"""Tests for the label-free seasonal analysis modules: smoothing, seasonal_stats, optical_context.

Everything here is synthetic and offline. The Earth Engine-facing functions are thin wrappers and
are exercised by running them; what is tested is the arithmetic that turns numbers into decisions.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from sar_pipeline.analysis import optical_context as oc
from sar_pipeline.analysis import seasonal_stats as ss
from sar_pipeline.analysis import smoothing as sm


# --------------------------------------------------------------------------- smoothing
def test_whittaker_with_zero_lambda_returns_the_input():
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    assert np.allclose(sm.whittaker(y, 0.0), y)


def test_whittaker_leaves_a_straight_line_untouched():
    """A second-difference penalty is zero on a line, so any lambda must leave it alone."""
    y = np.linspace(0, 10, 20)
    assert np.allclose(sm.whittaker(y, 1e6), y, atol=1e-6)


def test_whittaker_fills_a_missing_sample_from_its_neighbours():
    y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    assert sm.whittaker(y, 1.0)[2] == pytest.approx(3.0, abs=1e-6)


def test_curvature_matrix_matches_the_even_grid_operator():
    t = np.arange(0, 60, 6, dtype=float)
    uneven = sm.curvature_matrix(t).toarray()
    even = sm.difference_matrix(len(t), 2).toarray() / 36.0
    assert np.allclose(uneven, even)


def test_curvature_matrix_refuses_unsorted_times():
    with pytest.raises(ValueError):
        sm.curvature_matrix(np.array([0.0, 6.0, 3.0]))


def test_noise_sigma_recovers_white_noise_on_a_smooth_curve():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 4000)
    y = np.sin(2 * np.pi * t) + rng.normal(0, 0.5, t.size)
    assert sm.noise_sigma(y) == pytest.approx(0.5, rel=0.1)


def test_noise_sigma_ignores_a_single_sharp_step():
    """A real transition (a flood, a harvest) must not be counted as noise."""
    y = np.r_[np.zeros(200), np.full(200, 10.0)]
    assert sm.noise_sigma(y) < 0.01


# --------------------------------------------------------------------------- seasonal_stats
def test_db_and_linear_round_trip():
    db = np.array([-25.0, -15.0, -5.0], dtype="float32")
    assert np.allclose(ss.to_db(ss.to_linear(db)), db, atol=1e-4)


def test_to_db_treats_non_positive_power_as_missing():
    assert np.isnan(ss.to_db(np.array([0.0, -1.0]))).all()


def test_spatial_mean_averages_in_linear_power_not_db():
    """Mean of -10 and -20 dB is -12.6 dB in power, not -15 dB."""
    cube = np.full((1, 5, 5), -10.0, dtype="float32")
    cube[0, :, ::2] = -20.0
    out = ss.spatial_mean(cube, size=5)[0, 2, 2]
    lin = np.mean(ss.to_linear(cube[0]))
    assert out == pytest.approx(10 * np.log10(lin), abs=1e-3)
    assert out > -15.0


def test_spatial_mean_masks_windows_that_are_mostly_missing():
    cube = np.full((1, 5, 5), np.nan, dtype="float32")
    cube[0, 2, 2] = -15.0
    assert np.isnan(ss.spatial_mean(cube, size=5)[0, 2, 2])


def test_seasonal_features_on_a_flood_then_canopy_curve():
    series = np.array([-17, -26, -24, -15, -12, -18], dtype="float32")
    cube = series[:, None, None]
    f = ss.seasonal_features(cube)
    assert f["min"][0, 0] == -26 and f["max"][0, 0] == -12
    assert f["amplitude"][0, 0] == 14 and f["argmin"][0, 0] == 1


def test_seasonal_features_all_missing_pixel_is_nan_with_argmin_minus_one():
    f = ss.seasonal_features(np.full((4, 1, 1), np.nan, dtype="float32"))
    assert np.isnan(f["amplitude"][0, 0]) and f["argmin"][0, 0] == -1


def test_window_indices_is_inclusive_and_refuses_an_empty_window():
    dates = [dt.date(2025, m, 1) for m in (5, 6, 7, 8)]
    assert list(ss.window_indices(dates, "2025-06-01", "2025-07-01")) == [1, 2]
    with pytest.raises(ValueError):
        ss.window_indices(dates, "2026-01-01", "2026-02-01")


def test_half_month_bins_split_on_day_15():
    labels = ss.half_month_bins([dt.date(2025, 6, 15), dt.date(2025, 6, 16)])
    assert labels[0] != labels[1]


def test_track_correlation_high_for_a_shared_signal_low_for_noise():
    rng = np.random.default_rng(1)
    dates_a = [dt.date(2025, 5, 1) + dt.timedelta(days=6 * i) for i in range(30)]
    dates_b = [dt.date(2025, 5, 4) + dt.timedelta(days=6 * i) for i in range(30)]

    def cycle(dates):
        doy = np.array([(d - dates_a[0]).days for d in dates], dtype="float32")
        return -18 + 7 * np.sin(2 * np.pi * doy / 180)

    shared_a = (cycle(dates_a)[:, None, None] + rng.normal(0, 0.3, (30, 1, 1))).astype("float32")
    shared_b = (cycle(dates_b)[:, None, None] + rng.normal(0, 0.3, (30, 1, 1))).astype("float32")
    noise_a = rng.normal(-18, 1, (30, 1, 1)).astype("float32")
    noise_b = rng.normal(-18, 1, (30, 1, 1)).astype("float32")

    high, _ = ss.track_correlation(shared_a, dates_a, shared_b, dates_b)
    low, _ = ss.track_correlation(noise_a, dates_a, noise_b, dates_b)
    assert high[0, 0] > 0.9
    assert abs(low[0, 0]) < 0.6


# --------------------------------------------------------------------------- optical_context
def test_usable_months_needs_a_scene_under_the_threshold():
    rows = [{"month": "2025-08", "under_20pct": 0}, {"month": "2026-01", "under_20pct": 5}]
    assert oc.usable_months(rows) == ["2026-01"]


def test_compare_groups_reports_the_gap_and_ignores_cloudy_samples():
    values = {"2026-01": np.array([0.8, 0.9, np.nan, 0.2, 0.3], dtype="float32")}
    labels = np.array([True, True, True, False, False])
    row = oc.compare_groups(values, labels, ("crop", "other"))[0]
    assert row["crop_n"] == 2 and row["other_n"] == 2
    assert row["gap"] == pytest.approx(0.6, abs=1e-3)
