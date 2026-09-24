import pandas as pd

from sar_pipeline.analysis import sar_curve


def test_bad_pass_indices_masks_only_the_recorded_pass(monkeypatch):
    table = pd.DataFrame({"aoi": ["aoi7", "aoi7"], "track": ["T1", "T1"], "pol": ["VH", "VV"],
                          "date": ["2026-06-11", "2026-06-11"], "bad": [True, False]})
    monkeypatch.setitem(sar_curve._BAD_CACHE, "table", table)
    dates = pd.to_datetime(["2026-05-30", "2026-06-11", "2026-06-23"]).date
    assert sar_curve.bad_pass_indices({"aoi": "aoi7"}, "T1", "VH", list(dates)) == [1]
    assert sar_curve.bad_pass_indices({"aoi": "aoi7"}, "T1", "VV", list(dates)) == []
    assert sar_curve.bad_pass_indices({"aoi": "aoi8"}, "T1", "VH", list(dates)) == []
