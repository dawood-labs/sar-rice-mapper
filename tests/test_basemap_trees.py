import geopandas as gpd
import numpy as np
from affine import Affine
from shapely.geometry import box

from sar_pipeline.analysis import basemap_trees as bt


def test_tidy_mask_fills_small_holes_and_drops_lone_pixels():
    m = np.zeros((40, 40), bool)
    m[5:20, 5:20] = True
    m[10, 10] = False                              # a hole inside a crown
    m[30, 30] = True                               # a lone pixel
    t = bt.tidy_mask(m)
    assert t[10, 10] and not t[30, 30]


def test_only_tree_blobs_on_the_edge_are_cut_off_a_rice_field():
    tr = Affine(1, 0, 0, 0, -1, 100)               # 1 m pixels, top-left at (0, 100); planar, no real place
    tree = np.zeros((100, 100), bool)
    tree[0:15, 0:15] = True                        # a crown on the field's corner (225 m2)
    tree[45:55, 45:55] = True                      # a crown inside the field (100 m2), away from the edge
    f = gpd.GeoDataFrame({"field_id": ["f"], "major_class": ["rice"], "sub_class": ["rice standing transplanted"],
                          "acres": [2.5]}, geometry=[box(0, 0, 100, 100)])
    out, stats = bt.cut_edge_trees(f, tree, tr, min_tree_m2=50)
    cut = out[out["field_id"].str.contains("_t")]
    assert len(cut) == 1 and abs(cut.area.iloc[0] - 225) < 30            # the corner crown only
    assert out.loc[out["field_id"] == "f", "major_class"].iloc[0] == "rice"
    assert abs(out.area.sum() - 10000) < 1                               # nothing lost, nothing doubled
