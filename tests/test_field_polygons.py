"""Fields from the class map: cut a polygon holding two fields, keep a clean one, trace ground with no polygon, merge
same-label neighbours (ported from the sugarcane project's field labelling)."""
import geopandas as gpd
import numpy as np
from rasterio.transform import from_origin
from shapely.geometry import box

from sar_pipeline.analysis import field_polygons as fl

T = from_origin(500000, 1000, 10, 10)       # 10 m pixels, 100 x 100 grid
NAMES = {1: "rice", 4: "flooded"}


def _run(classes, polys):
    f = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32633")
    return fl.run(f, classes, T, "EPSG:32633", NAMES)


def test_a_polygon_holding_two_fields_is_cut_and_a_clean_one_is_not():
    c = np.full((100, 100), 1, dtype="uint8")
    c[:, 0:30] = 4                           # inside field A: left 30 px flooded, right 30 px rice
    a = box(500000, 0, 500600, 1000)          # columns 0-59
    b = box(500600, 0, 501000, 1000)          # columns 60-99, all rice
    pieces, finals = _run(c, [a, b])
    merged = finals[0.10]
    cut = pieces[pieces["origin"] == "split"]
    assert set(cut["class_name"]) == {"rice", "flooded"}
    assert (pieces.loc[pieces["origin"] == "delineation", "decision"] == "clean").all()
    # different delineated fields are never dissolved together (no monster polygons): rice half of A and B stay two
    assert (merged["class_name"] == "rice").sum() == 2 and (merged["class_name"] == "flooded").sum() == 1
    assert fl.overlap_acres(merged) == 0


def test_a_sliver_joins_the_neighbour_with_the_longest_shared_edge_inside_its_field():
    import geopandas as gpd
    from shapely.geometry import box

    big = box(0, 0, 100, 100)                 # 2.47 ac, field 1
    sliver = box(100, 0, 105, 100)            # 0.12 ac, field 1, other label
    other = box(105, 0, 205, 100)             # field 2
    f = gpd.GeoDataFrame({"parent": [1, 1, 2], "label": [1, 4, 4], "origin": "delineation", "pixels": [100, 5, 100],
                          "label_share": 1.0, "decision": "x"}, geometry=[big, sliver, other], crs="EPSG:32633")
    out = fl.absorb_slivers(f, 0.15)
    assert len(out) == 2 and fl.overlap_acres(out) == 0
    # it went to its own field (label 1), not to the same-label field next door
    assert out.loc[out["parent"] == 1, "label"].tolist() == [1] and out.area.max() < 2.7 * 4046.86


def test_ground_with_no_polygon_gets_a_field_shaped_polygon_and_ribbons_do_not():
    c = np.full((100, 100), 255, dtype="uint8")
    c[10:40, 10:40] = 1                       # a 300 m x 300 m block, no polygon over it
    c[60, 0:100] = 4                          # a one-pixel ribbon
    pieces, _ = _run(c, [box(501500, 0, 501600, 100)])
    der = pieces[pieces["origin"] == "derived from map"]
    assert list(der["class_name"]) == ["rice"] and der.area.iloc[0] > 0.8 * 300 * 300


def test_best_cut_finds_the_split_between_classes():
    xs = np.repeat(np.arange(40.0), 5)
    ys = np.tile(np.arange(5.0), 40)
    cls = np.where(xs < 25, 1, 4)
    angle, offset, purity = fl.best_cut(xs, ys, cls, (0.0, np.pi / 2), 8)
    assert purity == 1.0 and 24 < offset < 25


def test_tidy_closes_slits_fills_small_holes_and_cuts_spurs():
    import shapely
    from shapely.geometry import Polygon, box

    field = box(0, 0, 100, 50)
    slit = box(40, 20, 40.5, 50)                       # a 0.5 m slit cut into the field from its top edge
    hole = box(70, 20, 72, 22)                         # a 4 m2 hole
    spur = box(100, 24, 130, 26)                       # a 2 m wide, 30 m long tail
    g = shapely.union_all([field.difference(slit).difference(hole), spur])
    out = fl.tidy([g], hole_sqm=0.10 * 4046.86)[0]
    assert len(out.interiors) == 0
    assert abs(out.area - field.area) < 0.02 * field.area   # slit closed, hole filled, spur gone
    assert out.bounds[2] <= 101
    assert out.buffer(-0.01).contains(Polygon([(40, 30), (40.5, 30), (40.5, 31), (40, 31)]))
