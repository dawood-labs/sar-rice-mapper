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


def test_slivers_left_by_a_cut_join_the_other_side():
    from shapely.geometry import box as b
    rice = b(0, 0, 100, 3)                         # a 3 m strip of rice left along a crown: a sliver
    tree = b(0, 3, 100, 60)
    r, t = bt.merge_slivers(rice, tree)
    assert r.is_empty and abs(t.area - 6000) < 1
    r, t = bt.merge_slivers(b(0, 0, 100, 50), b(0, 50, 4, 54))      # a 16 m2 tree crumb on a rice edge
    assert t.is_empty and abs(r.area - 5016) < 1


def test_despike_removes_a_tail_and_keeps_the_corners():
    from shapely.geometry import Polygon
    tail = Polygon([(0, 0), (40, 0), (40, 40), (20, 40), (20, 70), (20, 40), (0, 40)])   # a zero-width line on top
    d = bt.despike(tail)
    assert d.bounds == (0, 0, 40, 40) and abs(d.area - 1600) < 1
