"""Tests for the per-AOI QGIS time-series package."""
from __future__ import annotations

import datetime as dt

import numpy as np
import rasterio

from sar_pipeline.analysis import qgis_package as qp

GRID = {"crs": "EPSG:32632", "res": 10.0, "x0": 500000.0, "y0": 5500000.0}


def test_bin_labels_map_to_half_month_start_dates():
    assert qp.bin_label_to_date(2025100) == dt.date(2025, 10, 1)
    assert qp.bin_label_to_date(2025101) == dt.date(2025, 10, 16)


def test_write_stack_places_each_pixel_series_in_its_bands(tmp_path):
    values = np.array([[-20.0, -15.0, -12.0], [-25.0, -24.0, -23.0]], dtype="float32")
    path = qp.write_stack(tmp_path / "s.tif", (2, 3), np.array([0, 1]), np.array([2, 0]), values, GRID,
                          ["VH 2025-05-01", "VH 2025-05-16", "VH 2025-06-01"])
    with rasterio.open(path) as ds:
        cube = ds.read()
        assert ds.count == 3 and ds.descriptions[1] == "VH 2025-05-16"
    assert np.allclose(cube[:, 0, 2], values[0]) and np.allclose(cube[:, 1, 0], values[1])
    assert cube[0, 0, 0] == -9999.0
