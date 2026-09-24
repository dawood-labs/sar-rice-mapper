"""Tests for the merged, smoothed radar curve. Synthetic arrays only; no rasters, no network."""
import numpy as np
import pandas as pd
import pytest

from sar_pipeline.analysis import sar_curve as sc


def dates(start, n, step):
    return pd.DatetimeIndex([pd.Timestamp(start) + pd.Timedelta(days=step * i) for i in range(n)])


def test_pair_offsets_recovers_a_known_shift():
    a = dates("2025-01-01", 20, 6)
    b = dates("2025-01-03", 20, 6)          # two days behind, so every pass pairs
    cube_a = np.full((20, 4, 4), -15.0)
    cube_b = cube_a - 2.5                   # the second track reads 2.5 dB low
    offset, n_pairs = sc.pair_offsets(a, cube_a, b, cube_b)
    assert n_pairs == 20
    assert np.allclose(offset, 2.5)


def test_pair_offsets_ignores_passes_that_are_too_far_apart():
    a = dates("2025-01-01", 5, 12)
    b = dates("2025-01-07", 5, 12)          # always six days away
    cube = np.full((5, 2, 2), -15.0)
    with pytest.raises(ValueError, match="pairing window"):
        sc.pair_offsets(a, cube, b, cube, pair_days=3)


def test_pair_offsets_is_per_pixel():
    a, b = dates("2025-01-01", 10, 6), dates("2025-01-02", 10, 6)
    cube_a = np.full((10, 2, 2), -15.0)
    cube_b = cube_a.copy()
    cube_b[:, 0, 0] -= 3.0
    offset, _ = sc.pair_offsets(a, cube_a, b, cube_b)
    assert np.isclose(offset[0, 0], 3.0) and np.isclose(offset[1, 1], 0.0)


def test_windows_average_in_power_not_in_decibels():
    """The mean of decibels sits below the decibel of the mean; the difference is the whole point."""
    acq = dates("2025-03-01", 2, 1)
    cube = np.array([[[-20.0]], [[-10.0]]])
    grid = pd.DatetimeIndex([pd.Timestamp("2025-03-01")])
    out = sc.to_windows(acq, cube, grid, step_days=5)
    in_db_mean = -15.0
    expected = 10 * np.log10((10 ** -2.0 + 10 ** -1.0) / 2)
    assert np.isclose(out[0, 0, 0], expected)
    assert out[0, 0, 0] > in_db_mean


def test_a_window_with_no_acquisition_stays_empty():
    acq = dates("2025-03-01", 1, 1)
    grid = pd.DatetimeIndex([pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-06")])
    out = sc.to_windows(acq, np.array([[[-15.0]]]), grid, step_days=5)
    assert np.isfinite(out[0, 0, 0]) and np.isnan(out[1, 0, 0])


def test_whittaker_db_fits_in_power_and_comes_back_in_decibels():
    values = np.tile([-20.0, -10.0], 15)          # long enough that the fit's ends do not dominate
    fitted = sc.whittaker_db(values, lmbd=100.0)
    middle = float(np.mean(fitted[5:-5]))
    flat_in_power = 10 * np.log10((10 ** -2.0 + 10 ** -1.0) / 2)   # -12.6 dB
    assert abs(middle - flat_in_power) < 0.3
    assert middle > -15.0                          # the mean of the decibels would land here


def test_choose_lambda_prefers_smoothing_on_a_noisy_flat_series():
    rng = np.random.default_rng(0)
    series = -15.0 + rng.normal(0, 1.5, size=(40, 30))
    chosen, table = sc.choose_lambda(series, candidates=(0.05, 1.0, 20.0), max_pixels=10)
    assert chosen == 20.0
    assert table["rmse_db"].idxmin() == 2


def test_choose_lambda_prefers_a_light_touch_on_a_clean_shape():
    days = np.arange(40)
    shape = -20 + 8 * np.exp(-((days - 20) ** 2) / 50)
    series = np.tile(shape[:, None], (1, 20))
    chosen, _ = sc.choose_lambda(series, candidates=(0.05, 1.0, 20.0), max_pixels=10)
    assert chosen == 0.05


def test_the_one_se_rule_never_picks_a_larger_lambda_than_the_minimum():
    rng = np.random.default_rng(1)
    series = -15.0 + rng.normal(0, 1.2, size=(40, 30))
    lenient, table = sc.choose_lambda(series, candidates=(0.25, 1.0, 2.0, 5.0), max_pixels=10)
    strict, _ = sc.choose_lambda(series, candidates=(0.25, 1.0, 2.0, 5.0), max_pixels=10, rule="min")
    assert lenient <= strict


def test_a_fit_that_overshoots_below_zero_power_returns_nan_not_a_floor():
    """Clipping to a tiny power produced -80 dB spikes that a minimum-based feature latched onto."""
    values = np.array([-30.0, -5.0, -30.0, -5.0, -30.0, -5.0, -30.0, -5.0, -30.0])
    fitted = sc.whittaker_db(values, lmbd=0.001)
    assert not np.any(fitted < -60)
    assert np.isfinite(fitted).any()


def test_box_mean_does_not_spread_one_missing_pixel():
    """One NaN used to blank its whole row and column (issue 20); now the window just skips it."""
    from sar_pipeline.analysis.sar_curve import box_mean

    rng = np.random.default_rng(0)
    img = rng.uniform(0.01, 0.1, size=(30, 30)).astype("float32")
    plain = box_mean(img, 5)
    img[10, 12] = np.nan
    out = box_mean(img, 5)
    assert np.isfinite(out).all()
    # far from the hole the mean is the plain filter; around it, the mean of the 24 valid neighbours
    assert np.allclose(out[0:5, 0:5], plain[0:5, 0:5])
    assert abs(out[10, 12] - np.nanmean(img[8:13, 10:15])) < 1e-6
    # a 3-column strip of missing data: every column inside it (at most 2 of 5 valid) stays
    # missing, the columns beside it (3 of 5 valid) keep a value from their valid neighbours
    img[:, 20:23] = np.nan
    out = box_mean(img, 5)
    assert np.isnan(out[:, 20:23]).all()
    assert np.isfinite(out[:, [19, 23]]).all()
    assert abs(out[5, 19] - np.nanmean(img[3:8, 17:22])) < 1e-6
