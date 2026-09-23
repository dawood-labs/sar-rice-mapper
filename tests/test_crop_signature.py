"""Tests for the season-cycle descriptors used to compare candidates with known rice."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import crop_signature as cs


def _series(peak_day: int, width: float):
    windows = pd.date_range("2025-09-01", periods=78, freq="5D")
    t = (windows - windows[0]).days.to_numpy().astype(float)
    ndvi = 0.15 + 0.7 * np.exp(-((t - peak_day) / width) ** 2)
    lswi = np.full(78, 0.1)
    return windows, ndvi, lswi


def test_season_cycle_is_the_one_starting_after_may():
    windows, ndvi, lswi = _series(peak_day=350, width=35)      # peaks mid-August 2026
    ndvi += 0.6 * np.exp(-(((windows - windows[0]).days.to_numpy() - 120) / 30) ** 2)  # a winter crop too
    c = cs.cycle_from_series(windows, ndvi, lswi)
    assert c is not None and c["greenup_onset"] >= pd.Timestamp("2026-05-01")   # the trough may sit earlier
    assert c["peak_date"].month == 8


def test_no_cycle_in_season_gives_none():
    windows, ndvi, lswi = _series(peak_day=120, width=30)      # only a winter crop
    assert cs.cycle_from_series(windows, ndvi, lswi) is None


def test_compare_lays_the_two_groups_side_by_side():
    ref = pd.DataFrame({"greenup_to_harvest_days": [90, 95], "peak_ndvi": [0.8, 0.85],
                        "harvest_date": [None, None], "transplant_date": pd.to_datetime(["2026-07-01"] * 2),
                        "wet_optical": [True, True], "radar_wet": [True, False]})
    cand = pd.DataFrame({"greenup_to_harvest_days": [50, 55], "peak_ndvi": [0.9, 0.9],
                         "harvest_date": pd.to_datetime(["2026-09-06", "2026-09-06"]),
                         "transplant_date": pd.to_datetime(["2026-07-03"] * 2)})
    table = cs.compare(ref, cand)
    assert table.loc["harvested before last date %", "known rice plots"] == 0.0
    assert table.loc["harvested before last date %", "candidate pixels"] == 100.0
    assert table.loc["radar wet %", "known rice plots"] == 50.0
