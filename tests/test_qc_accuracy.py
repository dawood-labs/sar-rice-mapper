"""Tests for the map-vs-QC confusion counts (tiny made-up layers)."""
import geopandas as gpd
from shapely.geometry import box

from sar_pipeline.analysis import qc_accuracy as qa

ACRE = 4046.8564224


def _g(rows):
    return gpd.GeoDataFrame([{"field_id": f, "major_class": c} for f, c, _ in rows],
                            geometry=[box(x, 0, x + ACRE ** 0.5, ACRE ** 0.5) for *_, x in rows], crs=32646)


def test_confusion_one_counts_each_case():
    base = _g([("a", "rice", 0), ("b", "rice", 100), ("c", "non-rice", 200), ("d", "non-rice", 300)])
    qc = _g([("a", "rice", 0), ("c", "rice", 200), ("d", "non-rice", 300), (None, None, 400)])  # b deleted, one drawn
    r = qa.confusion_one(base, qc)
    assert (r["tp_fields"], r["fp_fields"], r["fn_fields"], r["tn_fields"]) == (1, 1, 2, 1)
    assert round(r["fn_ac"], 3) == 2.0 and round(r["tp_ac"], 3) == 1.0


def test_metrics():
    m = qa.metrics(tp=90, fp=10, fn=10, tn=90)
    assert m == {"precision": 90.0, "recall": 90.0, "f1": 90.0, "accuracy": 90.0}
