"""Tests for the one-pixel report helpers. No network, no files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import pixel_2026 as px


def test_window_keeps_the_pixel_in_the_centre_even_at_the_grid_edge():
    w = px.centered_window(row=2, col=0, half=30)
    assert (w.row_off, w.col_off, w.height, w.width) == (-28, -30, 61, 61)
    assert w.row_off + 30 == 2 and w.col_off + 30 == 0


def test_pick_chips_takes_the_clearest_dates_of_every_month():
    table = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-02", "2026-01-10", "2026-01-20", "2026-02-05", "2026-02-15"]),
        "chip_clear_pct": [40.0, 100.0, 90.0, 0.0, 5.0],
        "chip_data_pct": [100.0] * 5,
        "pixel_clear": [50.0, 90.0, 80.0, 0.0, 10.0],
    })
    picked = px.pick_chips(table, per_month=2)
    assert [d.strftime("%m-%d") for d in picked["date"]] == ["01-10", "01-20", "02-05", "02-15"]


def test_pick_chips_skips_dates_with_little_data():
    table = pd.DataFrame({"date": pd.to_datetime(["2026-03-01", "2026-03-06"]),
                          "chip_clear_pct": [100.0, 50.0], "chip_data_pct": [10.0, 100.0],
                          "pixel_clear": [100.0, 50.0]})
    assert px.pick_chips(table, per_month=2)["date"].dt.day.tolist() == [6]


def test_stretch_ignores_cloud_pixels():
    chips = np.zeros((1, 3, 10, 10))
    chips[0, :, :5] = 1000                      # land
    chips[0, :, 5:] = 9000                      # cloud
    clear = np.zeros((1, 10, 10))
    clear[0, :5] = 100
    limits = px.stretch_limits(chips, clear)
    assert (limits[:, 1] <= 1000).all()


def test_sowing_is_estimated_from_emergence_not_the_trough():
    found = pd.DataFrame({"start_date": pd.to_datetime(["2025-10-06", "2026-05-01"]),
                          "emergence_date": [pd.Timestamp("2025-12-10"), None]})
    out = px.estimate_sowing(found, lead_days=10)
    assert out["sowing_date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-11-30", "2026-04-21"]
    assert (out["trough_date"] == found["start_date"]).all()
