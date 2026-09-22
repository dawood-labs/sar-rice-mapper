"""Tests for the one-pixel investigation helpers. Synthetic files, no network."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sar_pipeline.analysis import pixel_report as pr


def test_window_is_clipped_at_the_grid_edge():
    w = pr._window(0, 1, 5, (10, 10))
    assert (w.row_off, w.col_off, w.height, w.width) == (0, 0, 3, 4)


def write_s2(path, b3, b4, b8, clear):
    bands = np.array([100, b3, b4, 200, b8, clear], dtype="uint16").reshape(6, 1, 1)
    with rasterio.open(path, "w", driver="GTiff", width=1, height=1, count=6, dtype="uint16",
                       crs="EPSG:32632", transform=from_origin(500000, 5500000, 10, 10)) as ds:
        ds.write(bands)


def test_s2_series_drops_cloudy_dates_and_flags_water(tmp_path, monkeypatch):
    write_s2(tmp_path / "aoiX_2025-07_S2_2025-07-01.tif", b3=800, b4=500, b8=3500, clear=95)  # green crop
    write_s2(tmp_path / "aoiX_2025-07_S2_2025-07-06.tif", b3=900, b4=700, b8=400, clear=90)   # water
    write_s2(tmp_path / "aoiX_2025-07_S2_2025-07-11.tif", b3=4000, b4=4000, b8=4200, clear=10) # cloud
    monkeypatch.setattr(pr, "sync_s2", lambda loc, cache_root=None: tmp_path)
    s2 = pr.s2_series({"row": 0, "col": 0}, clear_min=60)
    assert list(s2["kept"]) == [True, True, False]
    assert s2.loc[0, "ndvi"] > 0.6 and not s2.loc[0, "flooded"]
    assert s2.loc[1, "flooded"] and s2.loc[1, "ndwi"] > 0
