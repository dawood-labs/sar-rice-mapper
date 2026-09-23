"""Tests for the radar water check on synthetic plot series. No files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import sar_water_check as sw


def _series(dip_db: float):
    dates = pd.date_range("2026-05-01", "2026-09-20", freq="12D")
    trough = pd.Timestamp("2026-07-10")
    days = (dates - trough).days
    vv = np.where((days >= -6) & (days <= 12), -18.0 - dip_db, -18.0) + 0.01 * days
    vh = vv - 7.0
    s = pd.DataFrame({"plot_id": 1, "track": "T1", "date": dates, "n_pixels": 9, "VH": vh, "VV": vv})
    s["VHmVV"] = s["VH"] - s["VV"]
    events = pd.DataFrame({"plot_id": [1], "aoi": ["aoiX"], "trough_date": [trough]})
    return s, events


def test_dip_is_read_relative_to_the_plots_own_dry_level():
    s, ev = _series(dip_db=8.0)
    table = sw.dips(sw.align(s, ev))
    row = table.iloc[0]
    assert 7.0 < row["VV_dip"] < 9.5
    assert -10 <= row["VV_flood_day"] <= 15
    assert sw.summary(table).loc[0, "VV_flooded_pct"] == 100


def test_no_dip_means_not_flooded():
    s, ev = _series(dip_db=0.0)
    table = sw.dips(sw.align(s, ev))
    assert abs(table.iloc[0]["VV_dip"]) < 1.0
    assert sw.summary(table).loc[0, "VV_flooded_pct"] == 0
