"""Tests for the gap-free 5-day NDVI series. No network, no files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import ndvi_5day as nd
from sar_pipeline.analysis import optical_phenology as op


def test_window_starts_step_five_days_and_exclude_the_end():
    starts = nd.window_starts("2025-09-01", "2025-09-16")
    assert [d.strftime("%m-%d") for d in starts] == ["09-01", "09-06", "09-11"]


def test_composite_keeps_the_highest_clear_ndvi_and_takes_others_from_that_date():
    starts = nd.window_starts("2025-09-01", "2025-09-11")
    dates = pd.to_datetime(["2025-09-01", "2025-09-03", "2025-09-04", "2025-09-08"])
    ndvi = np.array([[0.3], [0.7], [0.9], [0.5]])
    ok = np.array([[True], [True], [False], [True]])       # 0.9 is under QA60 cloud
    lswi = np.array([[0.1], [0.2], [0.3], [0.4]])
    best, (lswi_w,), observed = nd.max_value_composite(dates, ndvi, ok, (lswi,), starts)
    assert best[:, 0].tolist() == [0.7, 0.5]
    assert lswi_w[:, 0].tolist() == [0.2, 0.4]             # same date as the chosen NDVI
    assert observed.all()


def test_composite_leaves_a_window_without_clear_dates_empty():
    starts = nd.window_starts("2025-09-01", "2025-09-11")
    best, _, observed = nd.max_value_composite(
        pd.to_datetime(["2025-09-02"]), np.array([[0.6]]), np.array([[True]]), (), starts)
    assert best[0, 0] == 0.6 and np.isnan(best[1, 0])
    assert observed[:, 0].tolist() == [True, False]


def test_upper_envelope_ignores_cloud_dips_that_a_plain_fit_follows():
    days = np.arange(40)
    truth = 0.2 + 0.6 * np.exp(-((days - 20) / 7.0) ** 2)
    y = truth.copy()
    y[[12, 18, 25]] -= 0.35                                  # haze: only ever lowers NDVI
    y[[5, 30, 31, 32]] = np.nan                              # cloud: missing
    plain = op.whittaker(y, 0.5)
    fit, weights = nd.upper_envelope(y, 0.5)
    assert np.abs(fit - truth).max() < np.abs(plain - truth).max() / 2
    assert weights[[12, 18, 25]].max() < 0.5
    assert np.isfinite(fit).all()                            # every step has a value


def test_whittaker_without_weights_is_unchanged_by_the_new_argument():
    y = np.array([0.2, np.nan, 0.4, 0.5, np.nan, 0.3, 0.2])
    assert np.allclose(op.whittaker(y, 0.5), op.whittaker(y, 0.5, weights=np.ones(len(y))))


def test_gap_days_counts_to_the_nearest_observed_window():
    observed = np.array([True, False, False, False, True, False])[:, None]
    assert nd.gap_days(observed)[:, 0].tolist() == [0, 5, 10, 5, 0, 5]


def test_hold_edges_stops_the_fit_running_on_past_the_last_observation():
    seen = np.array([False, True, True, True, False, False])
    fit = nd.hold_edges(np.array([9.0, 0.2, 0.4, 0.6, 0.8, 1.0]), seen)
    assert fit.tolist() == [0.2, 0.2, 0.4, 0.6, 0.6, 0.6]


def test_upper_envelope_does_not_extrapolate_a_rise_into_an_unobserved_end():
    y = np.array([0.8, 0.6, 0.4, 0.2, 0.0] + [np.nan] * 6)
    fit, _ = nd.upper_envelope(y, 0.5)
    assert np.isfinite(fit).all() and fit[5:].max() <= fit[4] + 1e-9


def test_choose_lambda_prefers_more_smoothing_for_noisy_data():
    rng = np.random.default_rng(1)
    t = np.arange(60)
    truth = 0.2 + 0.6 * np.exp(-((t - 30) / 9.0) ** 2)
    raw = truth[:, None] + rng.normal(0, 0.08, (60, 50))
    chosen, table = nd.choose_lambda(raw, lambdas=(0.01, 5), n_pixels=50)
    assert chosen == 5 and list(table["lambda"]) == [0.01, 5]


def test_clear_mask_keeps_dark_water_under_a_strict_cloud_score():
    import numpy as np

    from sar_pipeline.analysis import ndvi_5day as nd

    data = np.array([True, True, True, True, True])
    qa = np.array([0, 1 << 10, 1 << 11, 0, 0])            # clear, opaque, cirrus, clear, clear
    cs = np.array([90, 90, 90, 20, 20])                    # ... Cloud Score+ cloudy on the last two
    b8 = np.array([3000, 3000, 3000, 600, 3200])           # last two: dark water, bright haze
    ndvi = np.array([0.6, 0.1, 0.6, 0.05, 0.4])
    both = (1 << 10) | (1 << 11)
    assert nd.clear_mask(data, qa, cs, b8, ndvi, None, both).tolist() == [True, False, False, True, True]
    assert nd.clear_mask(data, qa, cs, b8, ndvi, 60, 1 << 10).tolist() == [True, False, True, False, False]
    assert nd.clear_mask(data, qa, cs, b8, ndvi, 60, 1 << 10, keep_dark=True).tolist() == [True, False, True, True, False]


def test_upper_envelope_never_dives_below_the_observations_in_a_gap():
    import numpy as np

    from sar_pipeline.analysis import ndvi_5day as nd

    y = np.full(40, np.nan)
    y[:6] = [0.75, 0.8, 0.78, 0.6, 0.35, 0.2]      # a crop cut in spring ...
    y[30:] = np.linspace(0.6, 0.85, 10)            # ... and a new canopy in autumn; nothing between
    fit, _ = nd.upper_envelope(y, 1.0)
    assert np.nanmin(fit) >= 0.2 - nd.FIT_FLOOR_MARGIN - 1e-9


def test_clear_mask_drops_a_blue_bright_canopy_as_haze():
    import numpy as np

    from sar_pipeline.analysis import ndvi_5day as nd

    data = np.array([True, True, True])
    qa = np.zeros(3, dtype=int)
    cs = np.array([95, 95, 95])                          # Cloud Score+ calls all three clear
    b8 = np.array([3500, 3500, 2500])
    b2 = np.array([400, 1200, 1300])                     # clear canopy, hazy canopy, bright bare soil
    ndvi = np.array([0.7, 0.5, 0.15])
    assert nd.clear_mask(data, qa, cs, b8, ndvi, 60, 1 << 10, True, True, b2).tolist() == [True, False, True]
