"""The cross table's label mapping and summary."""
import pandas as pd

from sar_pipeline.analysis import rule_generalization as rg


def test_truth_and_coarse_classes():
    assert rg.truth("rice", "standing", "transplanted (water)") == "rice standing transplanted"
    assert rg.truth("rice", "standing", "direct seeded") == "rice standing direct seeded"
    assert rg.truth("rice", "harvested", "") == "rice harvested"
    assert rg.truth("bare") == "flooded / bare" and rg.coarse("rice standing transplanted") == "rice standing"


def test_summary_scores_a_rule_set_on_the_other_aois():
    s = pd.DataFrame([{"labels_of": "aoi72", "rule_set": "aoi72 rules", "n": 10, "full_right": 10, "coarse_right": 10},
                      {"labels_of": "aoi39", "rule_set": "aoi72 rules", "n": 10, "full_right": 5, "coarse_right": 8}])
    r = rg.summary(s).iloc[0]
    assert r["full_%"] == 75.0 and r["other_full_%"] == 50.0 and r["other_coarse_%"] == 80.0
