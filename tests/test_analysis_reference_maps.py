"""Tests for sampling published rice maps at our pixels. A tiny synthetic raster, no downloads."""
from __future__ import annotations

import numpy as np
import rasterio
from rasterio.transform import from_origin

from sar_pipeline.analysis import reference_maps as rm


def make_raster(path, x0, y0, values, res=0.01):
    values = np.asarray(values, dtype="uint8")
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(x0, y0, res, res)) as ds:
        ds.write(values, 1)
    return path


def test_sample_raster_reads_the_right_cell(tmp_path):
    grid = np.arange(16).reshape(4, 4)
    path = make_raster(tmp_path / "a.tif", 10.0, 50.0, grid)
    # cell (row 1, col 2) spans lon 10.02-10.03, lat 49.98-49.99
    assert rm.sample_raster(path, [10.025], [49.985])[0] == grid[1, 2]


def test_sample_raster_groups_give_the_same_answer_as_one_window(tmp_path):
    grid = np.arange(100).reshape(10, 10)
    path = make_raster(tmp_path / "a.tif", 10.0, 50.0, grid)
    lons = np.array([10.005, 10.095, 10.045]); lats = np.array([49.995, 49.905, 49.955])
    one = rm.sample_raster(path, lons, lats)
    grouped = rm.sample_raster(path, lons, lats, groups=np.array([0, 1, 0]))
    assert np.array_equal(one, grouped)


def test_points_outside_the_raster_come_back_as_zero_fill_not_errors(tmp_path):
    path = make_raster(tmp_path / "a.tif", 10.0, 50.0, np.ones((2, 2)))
    assert rm.sample_raster(path, [20.0], [40.0])[0] == 0


def test_sample_tiles_takes_each_point_from_the_tile_that_contains_it(tmp_path):
    west = make_raster(tmp_path / "w.tif", 10.0, 50.0, np.full((5, 5), 1))
    east = make_raster(tmp_path / "e.tif", 10.05, 50.0, np.full((5, 5), 2))
    vals = rm.sample_tiles([west, east], [10.01, 10.07, 30.0], [49.99, 49.99, 49.99])
    assert vals[0] == 1 and vals[1] == 2 and np.isnan(vals[2])
