import json

import numpy as np
import pandas as pd

from sar_pipeline.analysis import field_review as fr


def test_majority_ignores_no_data():
    names = {1: "a", 7: "b"}
    assert fr._majority(np.array([1, 7, 7, 255, 255, 255]), names) == ("b", 2 / 3)
    assert fr._majority(np.array([255, 255]), names) == ("no data", 0.0)


def test_score_joins_verdicts_and_agreement_counts_each_rule_set(tmp_path):
    d = fr.review_dir(83, fresh=str(tmp_path))
    (d / "verdicts").mkdir(parents=True)
    pd.DataFrame({"field_id": ["f1", "f2"], "acres": [1, 1], "map": ["x", "y"], "map_share": [1, 1],
                  "aoi116": ["x", "x"], "aoi116_share": [1, 1], "answers": [1, 2]}).to_csv(d / "fields.csv", index=False)
    for fid, c in (("f1", "x"), ("f2", "x")):
        (d / "verdicts" / f"{fid}.json").write_text(json.dumps(
            {"field_id": fid, "class": c, "sowing": "", "confidence": "high", "reason": "r"}))
    s = fr.score(83, fresh=str(tmp_path))
    assert list(s["verdict"]) == ["x", "x"]
    assert fr.agreement(s).to_dict() == {"map": 50.0, "aoi116": 100.0}


def test_user_check_overrides_the_reviewer_and_an_empty_label_drops_the_field(tmp_path):
    d = fr.review_dir(83, fresh=str(tmp_path))
    (d / "verdicts").mkdir(parents=True)
    for fid, c in (("f1", "young rice"), ("f2", "tree/orchard"), ("f3", "young rice")):
        (d / "verdicts" / f"{fid}.json").write_text(json.dumps(
            {"field_id": fid, "class": c, "sowing": "", "confidence": "low", "reason": "r"}))
    fr.check(83, "f2", "rice standing transplanted", "user", fresh=str(tmp_path))
    fr.check(83, "f3", "", "user cannot tell", fresh=str(tmp_path))
    fr.check(83, "f2", "other vegetation", "user, second look", fresh=str(tmp_path))
    assert fr.truth(83, fresh=str(tmp_path)).to_dict() == {"f1": "young rice", "f2": "other vegetation"}


def test_clean_class_drops_a_reviewer_note():
    assert fr.clean_class("other vegetation (not a field: strip along a canal)") == "other vegetation"
    assert fr.clean_class("Young rice") == "young rice" and fr.clean_class("weird") == "weird"


def test_track_cache_slices_the_same_values(monkeypatch):
    import numpy as np
    import pandas as pd

    from sar_pipeline.analysis import sar_curve
    dates = pd.DatetimeIndex(["2026-06-01", "2026-06-13"])
    cube = {"VV": np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4), "VH": np.zeros((2, 3, 4), "float32")}
    monkeypatch.setattr(sar_curve, "TRACK_CACHE", {("run", "T1", 5, True): (dates, cube)})
    d, c = sar_curve.read_track_pixels({"run": "run"}, "T1", [0, 5, 11], width=4, height=3)
    assert list(d) == list(dates) and c["VV"].tolist() == [[0, 5, 11], [12, 17, 23]]
