"""Tests for the cross-AOI rice model helpers. Synthetic and offline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio

from sar_pipeline.analysis import rice_model as rm


def test_interpolate_bins_fills_inside_and_holds_ends():
    cube = np.array([np.nan, 1.0, np.nan, 3.0, np.nan], dtype="float32")[:, None]
    assert np.allclose(rm.interpolate_bins(cube)[:, 0], [1, 1, 2, 3, 3])


def test_interpolate_bins_leaves_an_all_missing_pixel_missing():
    cube = np.full((4, 2), np.nan, dtype="float32")
    cube[:, 0] = [1, np.nan, np.nan, 4]
    out = rm.interpolate_bins(cube)
    assert np.allclose(out[:, 0], [1, 2, 3, 4]) and np.isnan(out[:, 1]).all()


def table(rows):
    return pd.DataFrame(rows, columns=["cluster", "centre_dist", "zhao2024", "opensea2021", "sun2019"])


def test_consensus_needs_every_map_to_agree():
    t = rm.consensus_labels(table([(7, 1.0, 1, 1, 255), (7, 1.0, 1, 0, 255)]), centre_quantile=1.0)
    assert t["label"].tolist() == [1, 0]


def test_consensus_maps_groups_to_classes():
    t = rm.consensus_labels(table([(7, 1, 1, 1, 1), (6, 1, 2, 2, 1), (9, 1, 1, 1, 1), (3, 1, 0, 0, 0)]),
                            centre_quantile=1.0)
    assert t["label"].tolist() == [1, 2, 3, 4]


def test_consensus_rejects_non_rice_where_a_map_says_rice():
    t = rm.consensus_labels(table([(3, 1, 0, 1, 0)]), centre_quantile=1.0)
    assert t["label"].tolist() == [0]


def test_consensus_drops_pixels_far_from_their_group_centre():
    rows = [(7, d, 1, 1, 1) for d in (1, 2, 3, 100)]
    t = rm.consensus_labels(table(rows), centre_quantile=0.75)
    assert t["label"].tolist() == [1, 1, 1, 0]


def test_probable_groups_are_never_labelled():
    t = rm.consensus_labels(table([(1, 1, 1, 1, 1), (2, 1, 1, 1, 1)]), centre_quantile=1.0)
    assert t["label"].tolist() == [0, 0]


def test_write_class_raster_places_values_on_the_grid(tmp_path):
    grid = {"x0": 500000.0, "y0": 2000000.0, "res": 10.0, "crs": "EPSG:32632"}
    path = rm.write_class_raster(tmp_path / "c.tif", (3, 4), np.array([0, 2]), np.array([1, 3]),
                                 np.array([2, 4]), grid)
    with rasterio.open(path) as ds:
        arr = ds.read(1)
        assert ds.nodata == 0 and ds.crs.to_epsg() == 32632
    assert arr[0, 1] == 2 and arr[2, 3] == 4 and arr.sum() == 6


def test_balanced_sample_caps_per_aoi_and_class_and_keeps_columns():
    t = pd.DataFrame({"aoi": ["a"] * 50 + ["b"] * 5, "label": [1] * 55, "x": range(55)})
    out = rm.balanced_sample(t, per_class=20, per_group_cap=10)
    assert set(out.columns) == {"aoi", "label", "x"}
    assert (out["aoi"] == "a").sum() == 10 and (out["aoi"] == "b").sum() == 5


def test_balanced_sample_ignores_unlabelled_rows():
    t = pd.DataFrame({"aoi": ["a"] * 4, "label": [0, 0, 1, 1]})
    assert len(rm.balanced_sample(t, per_class=10, per_group_cap=10)) == 2


def test_v2_mapping_labels_probable_groups_as_monsoon_rice():
    t = rm.consensus_labels(table([(1, 1, 1, 1, 1), (2, 1, 1, 1, 1)]), centre_quantile=1.0,
                            group_to_class=rm.GROUP_TO_CLASS_V2)
    assert t["label"].tolist() == [1, 1]
