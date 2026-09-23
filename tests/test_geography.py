"""Tests for the north-south / east-west shift table on synthetic inputs. No files."""
from __future__ import annotations

import pandas as pd
from shapely.geometry import box, mapping

from sar_pipeline.analysis import geography as ge


def test_centroids_and_latitude_slope():
    polys = {"aoiA": mapping(box(15.0, 50.0, 15.1, 50.1)), "aoiB": mapping(box(15.0, 51.0, 15.1, 51.1)),
             "aoiC": mapping(box(15.0, 52.0, 15.1, 52.1))}
    cal = pd.DataFrame({"aoi": ["aoiA", "aoiB", "aoiC"], "plots": [3, 3, 3], "trough": ["01 Jun", "11 Jun", "21 Jun"]})
    t = ge.shift_table(ge.aoi_centroids(polys), cal)
    assert t["aoi"].tolist() == ["aoiC", "aoiB", "aoiA"]                 # north first
    assert abs(t.attrs["trough_days_per_degree_latitude"] - 10.0) < 1e-6


def test_previous_harvest_reads_the_half_fall_before_the_season():
    windows = pd.date_range("2025-09-01", periods=78, freq="5D")
    ndvi = [0.2] * 78
    for i in range(20, 40):
        ndvi[i] = 0.8                                                    # a winter crop
    for i in range(40, 44):
        ndvi[i] = 0.3                                                    # cut in late March
    curves = pd.DataFrame({"aoi": "aoiA", "plot_id": 1, "window": windows, "n_pixels": 9, "ndvi": ndvi, "lswi": 0.1})
    h = ge.previous_harvest(curves)
    assert h.loc[0, "previous_harvest_month"] == 3
