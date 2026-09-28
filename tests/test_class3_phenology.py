"""Synthetic check of the class-3 phenology descriptors (no real data)."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import class3_phenology as cp


def test_descriptors_find_greenup_peak_and_leaf_water(monkeypatch):
    windows = pd.date_range("2026-03-01", "2026-09-21", freq="5D")
    n = len(windows)
    ndvi = np.full((n, 2), 0.2, dtype="float32")
    lswi = np.full((n, 2), -0.1, dtype="float32")
    rise = windows >= "2026-07-05"
    ndvi[rise, 0] = np.linspace(0.3, 0.8, rise.sum())            # pixel 0: greens up in July, peaks at the end
    lswi[rise, 0] = 0.35
    ndvi[:, 1] = 0.25                                             # pixel 1: never a canopy
    monkeypatch.setattr(cp.nd, "load", lambda aoi_id, **kw: {"ndvi5d": ndvi, "lswi5d": lswi, "windows": windows})
    d = cp.descriptors(0, np.array([0, 1]))
    assert 190 < d.loc[0, "greenup_doy"] <= 225 and d.loc[0, "peak_ndvi"] > 0.79 and abs(d.loc[0, "lswi_at_peak"] - 0.35) < 1e-6
    assert np.isnan(d.loc[1, "greenup_doy"]) and d.loc[1, "peak_ndvi"] == 0.25
