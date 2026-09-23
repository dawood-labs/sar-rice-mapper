"""Tests for the one-pixel investigation helpers. Synthetic files, no network."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sar_pipeline.analysis import pixel_report as pr


def test_window_is_clipped_at_the_grid_edge():
    w = pr._window(0, 1, 5, (10, 10))
    assert (w.row_off, w.col_off, w.height, w.width) == (0, 0, 3, 4)


def write_s2(path, b3, b4, b8, clear, b11=None):
    """A one-pixel export. Band descriptions are written because readers look bands up by name."""
    names = ["B2", "B3", "B4", "B5", "B8"] + (["B11", "B12"] if b11 is not None else []) + ["clear"]
    values = [100, b3, b4, 200, b8] + ([b11, b11] if b11 is not None else []) + [clear]
    bands = np.array(values, dtype="uint16").reshape(len(names), 1, 1)
    with rasterio.open(path, "w", driver="GTiff", width=1, height=1, count=len(names), dtype="uint16",
                       crs="EPSG:32632", transform=from_origin(500000, 5500000, 10, 10)) as ds:
        ds.write(bands)
        for i, name in enumerate(names, start=1):
            ds.set_band_description(i, name)


def test_s2_series_drops_cloudy_dates(tmp_path, monkeypatch):
    write_s2(tmp_path / "aoiX_2025-07_S2_2025-07-01.tif", b3=800, b4=500, b8=3500, clear=95)  # green crop
    write_s2(tmp_path / "aoiX_2025-07_S2_2025-07-06.tif", b3=900, b4=700, b8=400, clear=90)   # bare, wet
    write_s2(tmp_path / "aoiX_2025-07_S2_2025-07-11.tif", b3=4000, b4=4000, b8=4200, clear=10) # cloud
    monkeypatch.setattr(pr, "sync_s2", lambda loc, cache_root=None, folder=None: tmp_path)
    s2 = pr.s2_series({"row": 0, "col": 0}, clear_min=60)
    assert list(s2["kept"]) == [True, True, False]
    assert s2.loc[0, "ndvi"] > 0.6


def test_without_swir_no_date_is_called_flooded(tmp_path, monkeypatch):
    """NDWI cannot see water here, so a file with no B11 must not claim any flooding."""
    write_s2(tmp_path / "aoiX_2025-07_S2_2025-07-06.tif", b3=900, b4=700, b8=400, clear=90)
    monkeypatch.setattr(pr, "sync_s2", lambda loc, cache_root=None, folder=None: tmp_path)
    s2 = pr.s2_series({"row": 0, "col": 0})
    assert np.isnan(s2.loc[0, "lswi"]) and not s2.loc[0, "flooded"]


def test_with_swir_flooding_follows_the_lswi_rule(tmp_path, monkeypatch):
    # Flooded: NIR low, SWIR lower still -> LSWI above NDVI. Dry crop: NIR high, SWIR moderate.
    write_s2(tmp_path / "aoiX_2025-06_S2_2025-06-01.tif", b3=900, b4=700, b8=900, clear=90, b11=300)
    write_s2(tmp_path / "aoiX_2025-08_S2_2025-08-01.tif", b3=800, b4=500, b8=3500, clear=95, b11=1800)
    monkeypatch.setattr(pr, "sync_s2", lambda loc, cache_root=None, folder=None: tmp_path)
    s2 = pr.s2_series({"row": 0, "col": 0})
    assert s2.loc[0, "flooded"] and not s2.loc[1, "flooded"]
    assert s2.loc[0, "lswi"] > s2.loc[1, "lswi"]


def test_band_lookup_survives_the_clear_band_moving(tmp_path, monkeypatch):
    """``clear`` is band 6 without SWIR and band 8 with it; both must read the same value."""
    write_s2(tmp_path / "a_2025-07_S2_2025-07-01.tif", b3=800, b4=500, b8=3500, clear=77)
    write_s2(tmp_path / "b_2025-07_S2_2025-07-02.tif", b3=800, b4=500, b8=3500, clear=77, b11=1800)
    monkeypatch.setattr(pr, "sync_s2", lambda loc, cache_root=None, folder=None: tmp_path)
    s2 = pr.s2_series({"row": 0, "col": 0})
    assert list(s2["clear"]) == [77.0, 77.0]
