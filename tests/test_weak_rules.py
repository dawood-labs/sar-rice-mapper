"""weak_rules: the fixed halves and the delivered score."""
import pandas as pd

from sar_pipeline.analysis import weak_rules as wr


def test_half_is_fixed_and_splits_about_evenly():
    ids = [f"aoi1_{i:06d}" for i in range(1000)]
    a = [wr.half(i) for i in ids]
    assert a == [wr.half(i) for i in ids]
    assert 400 < a.count("pick") < 600


def test_scores_count_only_rice_vs_not():
    o = pd.DataFrame({"field_id": ["a", "b", "c", "d"],
                      "truth": ["rice standing transplanted", "young rice", "other vegetation", "tree/orchard"],
                      "x": ["young rice", "rice harvested", "tree/orchard", "rice standing direct seeded"]})
    s = wr._scores(o, ["x"])
    assert s.at["x", "all"] == 50.0      # a right (rice/rice), b wrong, c right (not/not), d wrong
