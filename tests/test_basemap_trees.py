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


def test_straighten_turns_pixel_steps_into_straight_edges():
    import shapely
    from shapely.geometry import box as b
    steps = shapely.union_all([b(i, 0, i + 1, 20 + (i % 2)) for i in range(30)])   # a 1 m saw-tooth top edge
    s = bt.straighten(steps)
    assert len(s.exterior.coords) < 12 and abs(s.area - steps.area) < 0.05 * steps.area


def test_open_shape_drops_fingers_and_keeps_the_field():
    import shapely
    from shapely.geometry import box as b
    field = shapely.union_all([b(0, 0, 40, 40), b(40, 18, 60, 20)])      # a 2 m wide, 20 m long finger
    o = bt.open_shape(field)
    assert o.bounds == (0, 0, 40, 40) and abs(o.area - 1600) < 1
    rice, other = bt.field_pieces(b(0, 0, 40, 40), shapely.union_all([b(0, 0, 25, 40), b(25, 10, 40, 11)]))
    assert rice.bounds[2] == 25 and abs(other.area - 15 * 40) < 50     # the 1 m hook went to the other piece


def test_no_cut_fields_still_give_a_table_with_columns():
    tr = Affine(1, 0, 0, 0, -1, 100)
    f = gpd.GeoDataFrame({"field_id": ["x"], "major_class": ["non-rice"], "sub_class": ["tree/orchard"], "acres": [1.0]},
                         geometry=[box(0, 0, 50, 50)])
    out, stats = bt.cut_edge_trees(f, np.zeros((100, 100), bool), tr)
    assert list(stats.columns) == ["field_id", "cut_m2", "geometry_error"] and len(out) == 1
