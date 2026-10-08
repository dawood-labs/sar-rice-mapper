"""weak_mistakes: the AUC used to rank features."""
import numpy as np

from sar_pipeline.analysis.weak_mistakes import auc


def test_auc_clean_split_and_no_split():
    assert auc(np.array([5.0, 6.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0, 2.0]), np.array([5.0, 6.0])) == 0.0
    assert auc(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.5
    assert np.isnan(auc(np.array([np.nan]), np.array([1.0])))


def test_weak_aoi_switches_off_by_default_and_each_one_alone(monkeypatch):
    import pandas as pd

    from sar_pipeline.analysis import curve_rules as cr

    f = pd.DataFrame({"vh_pos_end": [0.97, 0.5, 0.5], "fit_low_peak": [0.1, -0.6, 0.6],
                      "water_spell": [1, 0, 0], "water_depth": [0.0, 0.0, 0.0]})
    out = ["rice harvested", "young rice", "rice standing direct seeded"]
    assert cr.weak_aoi_switches(f, out) == out
    monkeypatch.setattr(cr, "HARVEST_NOT_IF_RADAR_TOP", 0.94)
    assert cr.weak_aoi_switches(f, out)[0] == "rice standing transplanted"
    monkeypatch.setattr(cr, "YOUNG_NOT_IF_DEEP_LOW", -0.43)
    assert cr.weak_aoi_switches(f, out)[1] == "flooded / bare"
    monkeypatch.setattr(cr, "RICE_NEEDS_EMPTY_FIELD", 0.435)
    assert cr.weak_aoi_switches(f, out)[2] == "other vegetation"
