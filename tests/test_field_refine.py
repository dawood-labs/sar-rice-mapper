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
    g = _fields(small + [monster, strip, normal])
    out = fr.geometry_stage(g, aoi_id=7)
    assert out["field_id"].tolist()[0] == "aoi7_000000"
    flags = out["refine_flag"].tolist()
    assert flags[:4] == ["", "", "", ""] and out["is_field"].tolist()[:4] == [True] * 4
    # the monster lost the four fields' area (10,000 of ~15,000 m2 -> 67 %): cut, remainder kept as bunds
    assert flags[4] in ("monster_cut", "monster_dropped") and out.loc[4, "overlap_lost_share"] > 0.6
    assert flags[5] == "strip" and not out.loc[5, "is_field"]
    assert flags[6] == "" and out.loc[6, "is_field"]
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
    # a sliver whose neighbour has another label stays
    g2 = g.copy(); g2.loc[0, "label"] = 3
    _, merged2 = fr.merge_tiny(g2)
    assert merged2.tolist() == ["", "", ""]


def test_flag_ponds_uses_the_water_share():
    g = gpd.GeoDataFrame({"field_id": ["a", "b"], "is_field": [True, True], "refine_flag": ["", ""]},
                         geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)], crs=UTM)
    out = fr.flag_ponds(g, np.array([0.9, 0.1]))
    assert out["refine_flag"].tolist() == ["pond", ""] and out["is_field"].tolist() == [False, True]
