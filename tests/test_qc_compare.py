import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from sar_pipeline.analysis import qc_compare as q


def _fields(rows):
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=[r.pop("geometry") for r in rows])                     # planar, no real place


def test_diff_tells_deleted_split_reshaped_kept_and_new():
    o = _fields([dict(field_id=f, sub_class="rice standing transplanted", acres=1.0, geometry=box(i * 100, 0, i * 100 + 60, 60))
                 for i, f in enumerate(["a", "b", "c", "d"])])
    qc = _fields([dict(field_id="a", sub_class="rice standing transplanted", geometry=box(0, 0, 60, 60)),       # kept
                  dict(field_id="b", sub_class="rice standing transplanted", geometry=box(100, 0, 130, 60)),   # split
                  dict(field_id="b", sub_class="rice standing transplanted", geometry=box(130, 0, 160, 60)),
                  dict(field_id="c", sub_class="rice standing transplanted", geometry=box(200, 0, 240, 60)),   # trimmed
                  dict(field_id=None, sub_class=None, geometry=box(500, 0, 520, 20))])                        # new; d deleted
    d = q.diff(o, qc).set_index("field_id")["change"]
    assert d.to_dict() == {"a": "kept", "b": "split", "c": "reshaped", "d": "deleted", "new_0": "new"}


def test_separation_ranks_the_feature_that_splits_deleted_from_kept_first():
    t = pd.DataFrame({"deleted": [True] * 6 + [False] * 6, "water": [5, 6, 7, 8, 9, 10, 0, 1, 2, 1, 0, 2],
                      "noise": [1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1]})
    s = q.separation(t)
    assert s.iloc[0]["feature"] == "water" and s.iloc[0]["auc"] == 1.0


def test_too_young_and_tree_switches_only_touch_rice_and_are_off_by_default(monkeypatch):
    import numpy as np

    from sar_pipeline.analysis import curve_rules as cr

    f = pd.DataFrame({"ndvi_low_peak": [0.1, 0.4, 0.1, 0.1, 0.4], "vh_range_rel": [5, 2, 5, 5, 2],
                      "last_view_open_water": [0, 0, 1, 0, 0], "days_since_open_water_view": [90, 90, 5, 30, np.nan]})
    names = ["rice standing transplanted", "rice standing transplanted", "young rice", "rice standing direct seeded",
             "flooded / bare"]
    assert cr.delivery_age_and_trees(f, names) == names                       # both switches off
    monkeypatch.setattr(cr, "TOO_YOUNG_OPEN_WATER_DAYS", 40)
    monkeypatch.setattr(cr, "TREE_IF_NEVER_EMPTIED", (0.30, 3.5))
    assert cr.delivery_age_and_trees(f, names) == [
        "rice standing transplanted",          # emptied, old water: stays rice
        "tree/orchard",                        # never emptied, flat radar
        "too young (not delivered)",           # open water on the newest view
        "too young (not delivered)",           # open water seen 30 days ago
        "flooded / bare"]                      # not rice: untouched


def test_strip_width_is_the_short_side_of_the_rotated_box():
    import numpy as np

    from sar_pipeline.analysis import field_polygons as fl

    g = gpd.GeoDataFrame({"label": [7, 7, 4]}, geometry=[box(0, 0, 200, 10), box(0, 50, 60, 110), box(0, 200, 200, 210)])
    w = fl.short_side_m(g)
    assert list(np.round(w)) == [10, 60, 10]


def test_rice_group_majority_keeps_a_field_with_a_small_tree_as_rice(monkeypatch):
    import numpy as np
    from shapely.geometry import box as sbox

    from sar_pipeline.analysis import field_polygons as fl

    xs, ys = np.meshgrid(np.arange(5, 100, 10.0), np.arange(5, 100, 10.0))
    xs, ys = xs.ravel(), ys.ravel()
    cls = np.array([1] * 33 + [7] * 32 + [6] * 35)              # 35 % tree, rice split over two classes
    rng = np.random.default_rng(0)
    cls = cls[rng.permutation(len(cls))]                        # mixed, so no clean cut helps
    g = sbox(0, 0, 100, 100)
    plain = fl._label_one(g, xs, ys, cls, 10, 0, "delineation")
    assert all(r["label"] == 6 for r, _ in plain)              # before: tree wins on its own
    monkeypatch.setattr(fl, "RICE_GROUP_MAJORITY", True)
    grouped = fl._label_one(g, xs, ys, cls, 10, 0, "delineation")
    assert all(r["label"] in (1, 7) for r, _ in grouped)       # rice together (65 %) beats tree


def test_own_without_turns_each_switch_off_in_its_own_set(monkeypatch):
    from sar_pipeline.analysis import curve_rules as cr

    monkeypatch.setitem(cr.AOI_OVERRIDES, -7, {"YOUNG_WHILE_RADAR_LOW": True, "YOUNG_MIN_RISE_DAYS": 40})
    s = q.own_without(-7, ["YOUNG_WHILE_RADAR_LOW", "YOUNG_MIN_RISE_DAYS"])
    assert s["own without YOUNG_WHILE_RADAR_LOW"]["YOUNG_WHILE_RADAR_LOW"] is False
    assert s["own without YOUNG_MIN_RISE_DAYS"]["YOUNG_MIN_RISE_DAYS"] is None
    assert s["own without all of them"] == {"YOUNG_WHILE_RADAR_LOW": False, "YOUNG_MIN_RISE_DAYS": None}
