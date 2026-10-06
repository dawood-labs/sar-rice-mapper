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


def test_clean_keeps_rice_makes_drawn_rice_and_leaves_no_overlap_or_touch():
    qc = _fields([
        dict(field_id="a", major_class="rice", sub_class="rice standing transplanted", origin="x", geometry=box(0, 0, 50, 50)),
        dict(field_id="b", major_class="rice", sub_class="young rice", origin="x", geometry=box(50, 0, 100, 50)),  # touches a
        dict(field_id="c", major_class="non-rice", sub_class="tree/orchard", origin="x", geometry=box(200, 0, 250, 50)),
        dict(field_id=None, major_class=None, sub_class=None, origin=None, geometry=box(40, 40, 80, 90))])     # drawn, overlaps
    out, rep = q.clean_qc(9, qc)
    assert rep["overlapping_pairs"] == 0 and rep["touching_pairs"] == 0 and rep["invalid"] == 0
    assert set(out["field_id"]) == {"a", "b", "aoi9_qc001"}                       # non-rice gone, drawn got an id
    assert out.set_index("field_id").loc["aoi9_qc001", "sub_class"] == q.DRAWN_SUB_CLASS
    a = out.set_index("field_id").geometry
    assert a["a"].area > 2490 and a["b"].area > 2490                             # delineated keep their shape (minus the gap)
    assert a["aoi9_qc001"].area < 40 * 50 - 2 * 10 * 10 + 1                      # drawn lost what a and b cover


def test_latest_clear_date_is_the_newest_date_marked_clear(tmp_path):
    d = tmp_path / "aoi5"
    d.mkdir()
    pd.DataFrame({"date": ["2026-09-03", "2026-09-13", "2026-09-18"], "clear_share": [0.0, 0.99, 0.2],
                  "delivered": [False, True, False]}).to_csv(d / "aoi5_s2_dates.csv", index=False)
    assert q.latest_clear_date(5, root=str(tmp_path)) == "2026-09-13"


def test_three_way_too_young_rule_thresholds_are_in_order():
    assert 0 < q.REMOVE_BELOW < q.CUT_FROM < 1 and q.MIN_CUT_PIXELS >= 1


def test_compare_verdicts_counts_exact_merged_and_rice(tmp_path, monkeypatch):
    import json as js

    from sar_pipeline.analysis import field_review as fr

    monkeypatch.setattr(fr, "review_dir", lambda a, fresh=None: tmp_path)
    for f, (x, y) in {"f1": ("young rice", "rice standing transplanted"), "f2": ("tree/orchard", "tree/orchard"),
                      "f3": ("flooded / bare", "young rice")}.items():
        for d, c in (("verdicts", x), ("verdicts_sonnet", y)):
            (tmp_path / d).mkdir(exist_ok=True)
            (tmp_path / d / f"{f}.json").write_text(js.dumps({"field_id": f, "class": c}))
    r = fr.compare_verdicts(1)
    assert r["fields"] == 3 and r["exact_pct"] == 33.3 and r["rice_merged_pct"] == 66.7 and r["rice_vs_non_rice_pct"] == 66.7
