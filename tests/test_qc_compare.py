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
