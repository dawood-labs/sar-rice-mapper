"""Synthetic geometries in a neutral UTM zone (no real location)."""
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from sar_pipeline.analysis import field_refine as fr

UTM = "EPSG:32630"


def _fields(polys, **cols):
    g = gpd.GeoDataFrame({"area_acres": [p.area / fr.SQM_PER_ACRE * 1.09 for p in polys], **cols},
                         geometry=polys, crs=UTM)
    return g.to_crs(4326)


def test_geometry_stage_cuts_monsters_flags_strips_and_recomputes_areas():
    small = [box(500000 + 60 * i, 4000000, 500000 + 60 * i + 50, 4000050) for i in range(4)]   # four 50x50 m fields
    monster = box(500000 - 5, 4000000 - 5, 500000 + 60 * 3 + 55, 4000055)                       # one outline around them
    strip = box(500400, 4000000, 500410, 4000300)                                                # 10 m x 300 m road
    normal = box(500500, 4000000, 500580, 4000080)
    # a field with a 4 m tail running 120 m along a road: the tail goes, the field stays
    tailed = box(500700, 4000000, 500780, 4000080).union(box(500780, 4000038, 500900, 4000042))
    g = _fields(small + [monster, strip, normal, tailed])
    out = fr.geometry_stage(g, aoi_id=7)
    assert out["field_id"].tolist()[0] == "aoi7_000000"
    flags = out["refine_flag"].tolist()
    assert flags[:4] == ["", "", "", ""] and out["is_field"].tolist()[:4] == [True] * 4
    # the monster lost the four fields' area (10,000 of ~15,000 m2 -> 67 %): cut, remainder kept as bunds
    assert flags[4] in ("monster_cut", "monster_dropped") and out.loc[4, "overlap_lost_share"] > 0.6
    assert flags[5] == "strip" and not out.loc[5, "is_field"]
    assert flags[6] == "" and out.loc[6, "is_field"]
    assert flags[7] == "" and out.loc[7, "is_field"] and out.loc[7, "tail_removed_share"] > 0.05
    assert abs(out.loc[7, "area_acres"] - 6400 / fr.SQM_PER_ACRE) < 1e-3
    # an irregular but wide field (a 95 x 98 m outline with a wavy edge) is not a strip
    import shapely
    wavy = shapely.Polygon([(0, 0), (95, 0), (95, 98)] + [(x, 98 + (3 if i % 2 else -3)) for i, x in enumerate(range(90, 0, -5))] + [(0, 98)])
    assert not fr.strip_like(gpd.GeoSeries([wavy, box(0, 0, 8, 400)], crs=UTM)).tolist()[0]
    assert fr.strip_like(gpd.GeoSeries([wavy, box(0, 0, 8, 400)], crs=UTM)).tolist()[1]
    # areas: UTM, the delivered attribute was 9 % high
    assert abs(out.loc[6, "area_acres"] - 6400 / fr.SQM_PER_ACRE) < 1e-3
    assert out.loc[6, "area_acres_delivered"] > out.loc[6, "area_acres"]


def test_merge_tiny_joins_a_sliver_to_its_same_label_neighbour():
    big = box(500000, 4000000, 500100, 4000100)
    sliver = box(500100, 4000040, 500110, 4000050)          # 100 m2 = 1 pixel, touching the big one
    other = box(500200, 4000000, 500300, 4000100)
    g = gpd.GeoDataFrame({"field_id": ["a", "b", "c"], "label": [1, 1, 3], "is_field": [True] * 3,
                          "refine_flag": ["", "", ""]}, geometry=[big, sliver, other], crs=UTM)
    m, merged = fr.merge_tiny(g)
    assert merged.tolist() == ["", "a", ""]
    assert not m.loc[1, "is_field"] and m.loc[1, "refine_flag"] == "merged"
    assert abs(m.loc[0].geometry.area - 10100) < 1e-6
    # a crumb under 4 pixels joins its neighbour whatever the label (1-3 pixels carry no label)
    g2 = g.copy(); g2.loc[0, "label"] = 3
    _, merged2 = fr.merge_tiny(g2)
    assert merged2.tolist() == ["", "a", ""]
    # a crumb touching nothing is dropped as a crumb, not delivered
    g3 = g.copy(); g3.loc[1, "geometry"] = box(500150, 4000040, 500160, 4000050)
    m3, merged3 = fr.merge_tiny(g3)
    assert merged3.tolist() == ["", "", ""] and m3.loc[1, "refine_flag"] == "crumb" and not m3.loc[1, "is_field"]


def test_merge_pieces_joins_a_small_same_label_piece_enclosed_by_its_field():
    field = box(500000, 4000000, 500100, 4000100).difference(box(500060, 4000060, 500100, 4000100))   # an L: 8,400 m2
    corner = box(500060, 4000060, 500100, 4000100)                                                       # the cut-off corner, 1,600 m2
    across = box(500100, 4000000, 500200, 4000100)                                                       # the field beyond, same label
    g = gpd.GeoDataFrame({"field_id": ["a", "b", "c"], "label": [1, 1, 1], "is_field": [True] * 3,
                          "refine_flag": ["", "", ""]}, geometry=[field, corner, across], crs=UTM)
    m, merged = fr.merge_pieces(g)
    # the corner shares 80 of its 160 m outline with the L (50 %) and 40 m with the field beyond: it joins the L
    assert merged.tolist() == ["", "a", ""]
    assert abs(m.loc[0].geometry.area - 10000) < 1.0 and m.loc[0].geometry.geom_type == "Polygon"
    assert m.loc[0].geometry.intersection(m.loc[2].geometry).area < 1.0      # never enters a third polygon
    # a different label, or a neighbour of similar size, does not merge
    g2 = g.copy(); g2.loc[1, "label"] = 3
    assert fr.merge_pieces(g2)[1].tolist() == ["", "", ""]
    g3 = gpd.GeoDataFrame({"field_id": ["a", "b"], "label": [1, 1], "is_field": [True, True], "refine_flag": ["", ""]},
                          geometry=[box(0, 0, 40, 40), box(40, 0, 80, 40)], crs=UTM)
    assert fr.merge_pieces(g3)[1].tolist() == ["", ""]


def test_fill_small_holes_and_split_parts_and_polygonal():
    import shapely

    outer = box(0, 0, 100, 100)
    line_hole = box(20, 10, 21, 90)                  # a 1 m wide line cut out of the field
    speck = box(50, 50, 55, 55)                      # 25 m2
    pond = box(70, 70, 95, 95)                       # 625 m2, 25 m wide: stays
    holed = outer.difference(line_hole).difference(speck).difference(pond)
    filled, n = fr.fill_small_holes(np.array([holed, outer], dtype=object))
    assert n.tolist() == [2, 0] and len(filled[0].interiors) == 1 and abs(filled[0].area - (10000 - 625)) < 1e-6
    # a geometry collection with a line loses the line; an empty stays empty
    gc = shapely.GeometryCollection([box(0, 0, 10, 10), shapely.LineString([(0, 0), (5, 5)])])
    out = fr.polygonal(np.array([gc, shapely.Polygon()], dtype=object))
    assert out[0].geom_type == "Polygon" and out[1].is_empty
    # a polygon cut in three: the largest piece keeps the id, a >= 4 px piece becomes "idb", a crumb is dropped
    pieces = shapely.MultiPolygon([box(0, 0, 100, 100), box(200, 0, 230, 30), box(300, 0, 305, 5)])
    g = gpd.GeoDataFrame({"field_id": ["aoi1_000007", "aoi1_000008"], "is_field": [True, True], "refine_flag": ["", ""]},
                         geometry=[pieces, box(500, 0, 600, 100)], crs=UTM)
    m = fr.split_parts(g)
    assert m["field_id"].tolist() == ["aoi1_000007", "aoi1_000008", "aoi1_000007b"]
    assert m["refine_flag"].tolist() == ["", "", "split_part"] and abs(m.loc[0].geometry.area - 10000) < 1e-6
    assert 0 < m.loc[0, "crumbs_dropped_share"] < 0.01 and m.loc[2].geometry.area == 900


def test_cut_overlaps_drops_a_duplicate_and_the_opening_stays_inside_the_outline():
    a = box(0, 0, 100, 100)
    g = gpd.GeoDataFrame({"field_id": ["a", "b"]}, geometry=[a, box(0, 0, 100, 100)], crs=UTM)
    geoms, lost = fr.cut_overlaps(g)
    assert lost.round(3).tolist() == [1.0, 0.0] and geoms.iloc[0].is_empty and abs(geoms.iloc[1].area - 10000) < 1e-6
    # a sharp 30-degree corner: a mitred dilation would spike beyond it; the opening is clipped back
    import shapely
    wedge = shapely.Polygon([(0, 0), (200, 0), (200, 60), (0, 60), (-40, 30)])
    opened, _ = fr.remove_tails(gpd.GeoSeries([wedge], crs=UTM))
    assert opened[0].difference(wedge).area < 1e-6


def test_flag_ponds_uses_the_water_share():
    g = gpd.GeoDataFrame({"field_id": ["a", "b"], "is_field": [True, True], "refine_flag": ["", ""]},
                         geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)], crs=UTM)
    out = fr.flag_ponds(g, np.array([0.9, 0.1]))
    assert out["refine_flag"].tolist() == ["pond", ""] and out["is_field"].tolist() == [False, True]
