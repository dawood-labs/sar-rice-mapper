"""Tests for per-plot pixel selection. No files, neutral grid (UTM 33N)."""
from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from sar_pipeline.analysis import plot_curves as pc

GRID = {"crs": "EPSG:32633", "x0": 500000.0, "y0": 5540000.0, "res": 10.0, "width": 10, "height": 10}


def _plots(*boxes):
    return gpd.GeoDataFrame({"plot_id": list(range(len(boxes)))}, geometry=list(boxes), crs=GRID["crs"])


def test_only_pixels_whose_centre_is_inside_are_taken():
    x, y = GRID["x0"], GRID["y0"]
    plots = _plots(box(x, y - 30, x + 30, y),            # 3 x 3 pixels, top-left corner
                   box(x + 51, y - 54, x + 54, y - 51))  # smaller than a pixel, misses every centre
    px = pc.plot_pixels(plots, GRID)
    assert sorted(px[0].tolist()) == [0, 1, 2, 10, 11, 12, 20, 21, 22]
    assert 1 not in px


def test_inner_buffer_drops_the_edge_pixels():
    x, y = GRID["x0"], GRID["y0"]
    px = pc.plot_pixels(_plots(box(x, y - 50, x + 50, y)), GRID, inner_buffer_m=6)
    assert sorted(px[0].tolist()) == [11, 12, 13, 21, 22, 23, 31, 32, 33]


def test_season_events_find_the_trough_the_water_and_the_onset():
    import numpy as np
    import pandas as pd

    windows = pd.date_range("2026-06-01", "2026-09-19", freq="5D")
    t = np.arange(len(windows), dtype=float)
    ndvi = 0.1 + 0.7 / (1 + np.exp(-(t - 12) / 1.5))          # flat low, then a climb from ~day 60
    ndvi[:4] = [0.3, 0.2, 0.1, 0.1]                             # falling into the trough first
    lswi = np.full(len(t), 0.2)                                  # wetter than NDVI at the trough
    c = pd.DataFrame({"plot_id": 7, "aoi": "aoiX", "window": windows, "n_pixels": 9,
                      "ndvi": ndvi, "lswi": lswi, "gap_days": 0})
    ev = pc.season_events(c).iloc[0]
    assert ev["trough_ndvi"] == 0.1 and ev["wet_at_trough"]
    assert 30 <= ev["trough_to_onset_days"] <= 60
    assert ev["last_ndvi"] > 0.7
    summary = pc.aoi_event_summary(pc.season_events(c))
    assert summary.loc[0, "wet_pct"] == 100
