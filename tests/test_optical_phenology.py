"""Tests for the optical-only phenology descriptors.

The property that matters most here is that the same crop measured in two regions, sown weeks
apart, produces the same shape numbers and only different dates. Everything is synthetic; no files
and no network.
"""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import optical_phenology as op

STEP_DAYS = 5


def curve(breakpoints, n_steps: int = 73):
    """A piecewise-linear NDVI series on a 5-day grid, from (day, value) breakpoints."""
    days = np.arange(n_steps) * STEP_DAYS
    xs, ys = zip(*breakpoints)
    return days, np.interp(days, xs, ys)


def grid(days):
    return pd.DatetimeIndex(pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D"))


def one_cycle(shift_days: int = 0):
    return curve([(0, 0.15), (60 + shift_days, 0.15), (80 + shift_days, 0.22),
                  (140 + shift_days, 0.80), (170 + shift_days, 0.25), (200 + shift_days, 0.18),
                  (360, 0.18)])


def test_a_single_cycle_is_found_and_timed():
    days, ndvi = one_cycle()
    c = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5))
    assert len(c) == 1
    assert c[0]["peak_ndvi"] == 0.80
    assert c[0]["harvest_date"] > c[0]["peak_date"] > c[0]["greenup_onset"]


def test_shifting_the_season_moves_the_dates_but_not_the_shape():
    """A crop sown 40 days later must measure as the same crop, only later."""
    shape_keys = ("amplitude", "rise_days", "fall_days", "fwhm_days",
                  "start_to_harvest_days", "greenup_to_harvest_days")
    early = op.pixel_cycles(*(grid(one_cycle(0)[0]),), one_cycle(0)[1], np.full(73, -0.5))[0]
    late = op.pixel_cycles(*(grid(one_cycle(40)[0]),), one_cycle(40)[1], np.full(73, -0.5))[0]
    for key in shape_keys:
        assert early[key] == late[key], key
    assert (late["start_date"] - early["start_date"]).days == 40
    assert (late["harvest_date"] - early["harvest_date"]).days == 40


def test_two_cycles_in_one_year_are_reported_separately():
    days, ndvi = curve([(0, 0.15), (40, 0.15), (100, 0.75), (140, 0.15),
                        (200, 0.70), (250, 0.15), (360, 0.15)])
    c = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5))
    assert len(c) == 2
    assert c[0]["peak_date"] < c[1]["peak_date"]
    assert all(x["complete"] for x in c)


def test_a_cycle_still_falling_at_the_last_observation_is_marked_incomplete():
    days, ndvi = curve([(0, 0.15), (60, 0.15), (200, 0.80), (360, 0.15)])
    c = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5))
    assert len(c) == 1 and not c[0]["complete"]


def test_flat_ground_produces_no_cycle():
    days, ndvi = curve([(0, 0.2), (360, 0.22)])
    assert op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5)) == []


def test_flood_margin_uses_lswi_against_ndvi():
    days, ndvi = one_cycle()
    lswi = np.full_like(ndvi, -0.2)
    lswi[:16] = 0.30  # wet at the start of the cycle
    c = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5), lswi)[0]
    assert c["flood_margin"] > 0
    dry = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5), np.full_like(ndvi, -0.2))[0]
    assert dry["flood_margin"] < c["flood_margin"]


def test_acceleration_onset_marks_where_the_rise_takes_off():
    days, ndvi = one_cycle()
    onset = op.acceleration_onset(ndvi, days, 0, int(np.argmax(ndvi)))
    assert 55 <= days[onset] <= 95


def test_regrid_leaves_a_wide_gap_empty_instead_of_interpolating_it():
    days = np.array([0.0, 10.0, 90.0, 100.0])
    ok = np.ones(4, dtype=bool)
    out = op.regrid(days, np.array([0.2, 0.3, 0.7, 0.8]), ok, np.array([50.0]), max_gap_days=35)
    assert np.isnan(out[0])


def test_whittaker_fills_missing_samples_without_moving_the_observed_ones():
    y = np.array([0.2, 0.3, np.nan, np.nan, 0.6, 0.7])
    z = op.whittaker(y, lmbd=0.5)
    assert np.isfinite(z).all()
    assert abs(z[0] - 0.2) < 0.05 and abs(z[-1] - 0.7) < 0.05
    assert 0.3 < z[2] < 0.6


def test_a_brief_mid_season_dip_does_not_split_one_crop_into_two():
    """Regression: a pixel running 0.06 in May to 0.80 in August was reported as a 40-day cycle
    starting at the end of August, because one contaminated observation dipped 0.24 mid-season."""
    days, ndvi = curve([(0, 0.06), (40, 0.06), (150, 0.80), (165, 0.56), (180, 0.76),
                        (230, 0.20), (360, 0.18)])
    c = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5))
    assert len(c) == 1, [str(x["peak_date"].date()) for x in c]
    assert c[0]["start_date"] < grid(days)[int(50 / STEP_DAYS)]
    assert c[0]["start_to_harvest_days"] > 120


def test_a_dip_that_stays_down_still_separates_two_crops():
    """The guard must not merge genuine double cropping: a real harvest keeps the field low."""
    days, ndvi = curve([(0, 0.15), (40, 0.15), (100, 0.75), (140, 0.15), (185, 0.15),
                        (240, 0.70), (290, 0.15), (360, 0.15)])
    c = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5))
    assert len(c) == 2


def test_min_low_days_is_measured_in_days_not_samples():
    """The same dip must survive or fail identically whatever the grid spacing is."""
    days, ndvi = curve([(0, 0.15), (40, 0.15), (100, 0.75), (140, 0.15), (185, 0.15),
                        (240, 0.70), (290, 0.15), (360, 0.15)])
    coarse_days = days[::2]
    coarse = ndvi[::2]
    fine = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5))
    rough = op.pixel_cycles(grid(coarse_days), coarse, np.full_like(coarse, -0.5))
    assert len(fine) == len(rough) == 2


def test_a_flat_pixel_yields_no_cycle_despite_smoothing_noise():
    """A constant series is not exactly constant after a fit; prominence must not chase that noise."""
    days = np.arange(73) * STEP_DAYS
    flat = op.whittaker(np.full(73, 0.22), lmbd=2.0)
    assert op.pixel_cycles(grid(days), flat, np.full(73, -0.5)) == []


def test_blank_long_gaps_empties_only_the_stretches_nobody_saw():
    days = np.arange(20) * 5.0
    ok = np.ones(20, dtype=bool)
    ok[4:14] = False                      # 10 windows = 50 days unobserved
    fitted = np.linspace(0.2, 0.8, 20)
    out = op.blank_long_gaps(fitted, days, ok, max_gap_days=35)
    assert np.isnan(out[4:14]).all()
    assert np.isfinite(out[:4]).all() and np.isfinite(out[14:]).all()


def test_a_short_gap_is_kept_because_the_fit_can_bridge_it():
    days = np.arange(20) * 5.0
    ok = np.ones(20, dtype=bool)
    ok[8:10] = False                      # 2 windows: the surrounding clear samples are 15 days apart
    out = op.blank_long_gaps(np.linspace(0.2, 0.8, 20), days, ok, max_gap_days=35)
    assert np.isfinite(out).all()


def test_blank_long_gaps_handles_a_gap_running_off_the_end():
    days = np.arange(20) * 5.0
    ok = np.ones(20, dtype=bool)
    ok[12:] = False
    out = op.blank_long_gaps(np.linspace(0.2, 0.8, 20), days, ok, max_gap_days=35)
    assert np.isnan(out[12:]).all() and np.isfinite(out[:12]).all()


def test_blank_long_gaps_on_an_entirely_unobserved_pixel():
    days = np.arange(5) * 5.0
    out = op.blank_long_gaps(np.zeros(5), days, np.zeros(5, dtype=bool))
    assert np.isnan(out).all()


def test_the_default_smoothing_is_the_gentler_one_for_window_composites():
    assert op.LMBD == 0.5


def test_a_cycle_whose_rise_falls_in_a_blanked_gap_reports_no_rate_quietly():
    """A blanked gap can cover the whole rise; that is a missing measurement, not a warning."""
    import warnings

    days, ndvi = one_cycle()
    ndvi[6:20] = np.nan                    # blank the rise
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # any RuntimeWarning fails this test
        c = op.pixel_cycles(grid(days), ndvi, np.full_like(ndvi, -0.5))
    assert all(np.isnan(x["max_rise_per_day"]) or np.isfinite(x["max_rise_per_day"]) for x in c)


def test_transplant_is_dated_from_greenup_when_the_field_sits_low_for_weeks():
    """A long flat low stretch before the crop must not drag the sowing date weeks early."""
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import optical_phenology as op

    days = np.arange(0, 300, 5, dtype=float)
    ndvi = np.where(days < 150, 0.1, 0.1 + 0.8 * np.exp(-((days - 210) / 30.0) ** 2))
    ndvi = np.where(days >= 150, np.maximum(ndvi, 0.1 + 0.8 * np.clip((days - 150) / 60, 0, 1) *
                                            np.exp(-np.clip(days - 210, 0, None) / 25)), ndvi)
    ndvi[0] = 0.08                                    # the trough sits at the very start
    dates = pd.date_range("2025-09-01", periods=len(days), freq="5D")
    c = op.pixel_cycles(dates, ndvi, None, None)[0]
    assert c["transplant_from"] == "greenup"
    assert c["transplant_date"] == c["greenup_onset"] - pd.Timedelta(days=op.GREENUP_LAG_DAYS)
    assert (c["transplant_date"] - c["start_date"]).days > 60
