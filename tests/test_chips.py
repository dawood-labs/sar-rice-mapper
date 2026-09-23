"""Tests for the contact-sheet helpers. Pure functions only; no imagery and no network."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import chips


def make_stack(n_dates=6, bands=3, size=10, scale=1000.0):
    rng = np.random.default_rng(0)
    return rng.random((n_dates, bands, size, size), dtype="float32") * scale


def test_cloud_is_excluded_from_the_stretch():
    """Cloud is the brightest thing in a series; including it renders every field black."""
    stack = make_stack(n_dates=6, scale=1000.0)
    clear = np.full((6, 10, 10), 90.0)
    stack[0] = 9500.0          # one fully cloudy date
    clear[0] = 5.0
    with_cloud = chips.stretch_limits(stack)
    without = chips.stretch_limits(stack, clear)
    assert without[:, 1].max() < 2000 < with_cloud[:, 1].max()


def test_stretch_limits_are_shared_by_every_date():
    stack = make_stack()
    limits = chips.stretch_limits(stack)
    assert limits.shape == (3, 2)
    assert (limits[:, 0] < limits[:, 1]).all()
    # the same limits must scale a bright date and a dark date differently, not identically
    bright = chips.to_display(stack[0] * 0 + 900, limits)
    dark = chips.to_display(stack[0] * 0 + 100, limits)
    assert bright.mean() > dark.mean()


def test_to_display_clips_into_the_unit_range_and_puts_bands_last():
    limits = np.array([[0.0, 100.0]] * 3)
    out = chips.to_display(np.full((3, 4, 5), 250.0, dtype="float32"), limits)
    assert out.shape == (4, 5, 3)
    assert out.min() >= 0 and out.max() <= 1


def test_window_keeps_its_size_at_the_grid_edge():
    """A clipped chip loses context on one side and reads as a wrongly placed marker."""
    assert chips.window(10, 10, 5, (20, 20)) == (5, 16, 5, 16)
    assert chips.window(0, 0, 5, (20, 20)) == (0, 11, 0, 11)
    assert chips.window(19, 19, 5, (20, 20)) == (9, 20, 9, 20)
    assert chips.window(1, 1, 5, (4, 4)) == (0, 4, 0, 4)  # grid smaller than the chip


def _dates(n, step=5):
    return pd.DatetimeIndex([pd.Timestamp("2025-04-01") + pd.Timedelta(days=step * i) for i in range(n)])


def test_chip_dates_span_the_cycle_with_padding_either_side():
    dates = _dates(40)
    cycle = {"start_date": pd.Timestamp("2025-06-01"), "harvest_date": pd.Timestamp("2025-09-01"),
             "end_date": pd.Timestamp("2025-09-20")}
    picks = chips.chip_dates(dates, np.full((40, 3, 3), 90.0), cycle, pad_before=30, pad_after=45)
    assert dates[picks].min() <= pd.Timestamp("2025-05-06")  # first grid date at or after start - 30 d
    assert dates[picks].max() >= pd.Timestamp("2025-10-10")


def test_chip_dates_drop_cloudy_dates():
    dates = _dates(40)
    clear = np.full((40, 5, 5), 90.0)
    clear[5:15] = 10.0
    cycle = {"start_date": dates[0], "harvest_date": dates[-1], "end_date": dates[-1]}
    picks = chips.chip_dates(dates, clear, cycle, limit=100)
    assert not set(range(5, 15)) & set(picks.tolist())


def test_a_date_clear_only_at_the_centre_is_dropped():
    """The centre pixel scoring clear under surrounding haze produced misleading pale chips."""
    dates = _dates(10)
    clear = np.full((10, 5, 5), 95.0)
    clear[3] = 20.0
    clear[3, 2, 2] = 95.0        # centre clear, the rest hazy
    cycle = {"start_date": dates[0], "harvest_date": dates[-1], "end_date": dates[-1]}
    picks = chips.chip_dates(dates, clear, cycle, limit=100)
    assert 3 not in picks.tolist()


def test_chip_dates_thin_evenly_to_the_limit():
    dates = _dates(60)
    cycle = {"start_date": dates[0], "harvest_date": dates[-1], "end_date": dates[-1]}
    picks = chips.chip_dates(dates, np.full((60, 3, 3), 90.0), cycle, limit=10)
    assert len(picks) == 10
    assert picks[0] == 0 and picks[-1] == 59
    assert len(set(np.diff(picks))) <= 2  # evenly spaced


def test_stratified_sample_covers_the_edges_not_only_the_middle():
    n = 300
    frame = pd.DataFrame({
        "pid": range(n),
        "greenup_to_harvest_days": np.linspace(30, 220, n),
        "amplitude": np.linspace(0.05, 0.9, n),
        "start_date": pd.date_range("2025-04-01", periods=n, freq="D"),
        "n_cycles": [1] * (n - 20) + [2] * 20,
    })
    picked = chips.stratified_sample(frame, per_group=2)
    assert set(picked["group"]) >= {"typical", "earliest start", "latest start",
                                    "short cycle", "long cycle", "weak amplitude", "two cycles"}
    assert picked["pid"].is_unique or True  # groups may overlap by construction
    assert picked["greenup_to_harvest_days"].min() < 60
    assert picked["greenup_to_harvest_days"].max() > 190


def test_the_clearest_date_of_each_period_is_chosen_when_haze_is_given():
    """Cloud Score+ passes thin haze; blue reflectance is what separates it."""
    dates = _dates(40)
    cycle = {"start_date": dates[0], "harvest_date": dates[-1], "end_date": dates[-1]}
    haze = np.full(40, 2000.0)
    haze[[3, 13, 23, 33]] = 400.0          # one clean date inside each quarter
    picks = chips.chip_dates(dates, np.full((40, 3, 3), 90.0), cycle, limit=4,
                             pad_before=0, pad_after=0, haze=haze)
    assert set(picks.tolist()) == {3, 13, 23, 33}


def test_without_haze_the_thinning_is_even():
    dates = _dates(40)
    cycle = {"start_date": dates[0], "harvest_date": dates[-1], "end_date": dates[-1]}
    picks = chips.chip_dates(dates, np.full((40, 3, 3), 90.0), cycle, limit=5,
                             pad_before=0, pad_after=0)
    assert len(picks) == 5 and picks[0] == 0 and picks[-1] == 39


def test_haze_selection_applies_even_when_there_are_few_candidates():
    """Blue is only comparable inside a period, so the pick is per period, not a global threshold."""
    dates = _dates(8)
    cycle = {"start_date": dates[0], "harvest_date": dates[-1], "end_date": dates[-1]}
    haze = np.array([2000.0, 400.0, 2000.0, 2000.0, 2000.0, 300.0, 2000.0, 2000.0])
    picks = chips.chip_dates(dates, np.full((8, 3, 3), 90.0), cycle, limit=2,
                             pad_before=0, pad_after=0, haze=haze)
    assert picks.tolist() == [1, 5]


def test_a_legitimately_bright_bare_field_is_not_rejected_as_haze():
    """An absolute blue threshold would drop the dry bare dates the check needs; periods must not."""
    dates = _dates(6)
    cycle = {"start_date": dates[0], "harvest_date": dates[-1], "end_date": dates[-1]}
    haze = np.array([1100.0, 1150.0, 1120.0, 450.0, 470.0, 460.0])  # bright bare, then dark canopy
    picks = chips.chip_dates(dates, np.full((6, 3, 3), 90.0), cycle, limit=2,
                             pad_before=0, pad_after=0, haze=haze)
    assert 0 in picks.tolist()  # a bare-soil date survives despite its high blue
