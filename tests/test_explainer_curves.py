"""Tests for the explainer example picker (no data on disk needed)."""
import pandas as pd

from sar_pipeline.analysis import explainer_curves as ec


def test_pick_takes_first_agreeing_field_high_confidence_first(monkeypatch):
    rule = {"f1": "young rice", "f2": "tree/orchard", "f3": "tree/orchard"}

    def fake(aoi, field_id):
        if field_id == "bad":
            raise KeyError("no pixels")
        return {"aoi": aoi, "field_id": field_id, "rule_class": rule[field_id]}

    monkeypatch.setattr(ec, "field_curves", fake)
    v = pd.DataFrame([
        {"aoi": 1, "field_id": "f3", "class": "tree/orchard", "confidence": "medium", "sowing": ""},
        {"aoi": 1, "field_id": "bad", "class": "tree/orchard", "confidence": "high", "sowing": ""},
        {"aoi": 1, "field_id": "f1", "class": "tree/orchard", "confidence": "high", "sowing": ""},
        {"aoi": 1, "field_id": "f2", "class": "tree/orchard", "confidence": "high", "sowing": ""},
        {"aoi": 2, "field_id": "f1", "class": "young rice", "confidence": "high", "sowing": "2026-08-25"},
    ])
    out = ec.pick([1], v)
    # f1 disagrees with the reviewer and "bad" fails, so the first agreeing high-confidence field wins over f3
    assert [(o["class"], o["field_id"]) for o in out] == [("tree/orchard", "f2")]
    assert out[0]["review_confidence"] == "high"
