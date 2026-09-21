"""Final models and the best-available-model map: small synthetic checks, no real data."""
from __future__ import annotations

import joblib
import numpy as np
import pytest

from sar_pipeline.analysis import final_models as fm

COLS = [f"{t}__hm_w5_s0501__{p}__{d}" for t in ("RO1_DSC", "RO2_ASC") for d in ("0501", "0801") for p in ("VH", "VV")]
TRACKS = [{"track_id": "RO1_DSC", "role": "primary"}, {"track_id": "RO2_ASC", "role": "secondary"}]


def test_variants_full_primary_only_and_without_bin():
    v = fm.variant_columns(COLS, TRACKS, ["RO2_ASC:0801"])
    assert list(v) == ["full", "only_RO1_DSC", "no_RO2_ASC_0801"]
    assert v["full"] == COLS
    assert all(c.startswith("RO1_DSC") for c in v["only_RO1_DSC"]) and len(v["only_RO1_DSC"]) == 4
    assert len(v["no_RO2_ASC_0801"]) == 6 and not any(c.startswith("RO2_ASC") and c.endswith("0801") for c in v["no_RO2_ASC_0801"])


def test_single_track_has_no_primary_only_variant():
    assert list(fm.variant_columns(COLS[:4], TRACKS[:1], [])) == ["full"]


def test_without_bin_that_matches_nothing_is_an_error():
    with pytest.raises(ValueError, match="no feature column"):
        fm.variant_columns(COLS, TRACKS, ["RO2_ASC:0915"])


def test_each_pixel_gets_the_model_with_the_most_available_features():
    bundles = [{"columns": COLS}, {"columns": [c for c in COLS if not (c.startswith("RO2") and "0801" in c)]},
               {"columns": COLS[:4]}]
    n = 4
    feats = {c: np.ones(n) for c in COLS}
    feats[COLS[6]] = np.array([1, np.nan, 1, 1])            # pixel 1: one ASC 0801 value missing
    for c in COLS[4:]:
        feats[c] = feats[c].copy(); feats[c][2] = np.nan        # pixel 2: no ASC at all
    feats[COLS[0]] = np.array([1, 1, 1, np.nan])            # pixel 3: a primary value missing -> nothing fits
    assert fm.pick_models(feats, n, bundles).tolist() == [1, 2, 3, 0]


def test_missing_columns_count_as_missing_data():
    bundles = [{"columns": COLS}, {"columns": COLS[:4]}]
    feats = {c: np.ones(3) for c in COLS[:4]}                 # the secondary track never covered this chunk
    assert fm.pick_models(feats, 3, bundles).tolist() == [2, 2, 2]


def test_load_pixel_models_orders_by_features_and_skips_field_level(tmp_path):
    base = {"model": None, "labels": ["A", "B"], "threshold": 0.5}
    joblib.dump({**base, "columns": COLS[:4]}, tmp_path / "xgb_only.joblib")
    joblib.dump({**base, "columns": COLS}, tmp_path / "xgb_full.joblib")
    joblib.dump({**base, "columns": COLS, "stat": "mean"}, tmp_path / "xgb_field_level.joblib")
    b = fm.load_pixel_models(tmp_path)
    assert [x["file"] for x in b] == ["xgb_full.joblib", "xgb_only.joblib"]
    assert b[0]["threshold"] == 0.5


def test_models_with_different_thresholds_are_refused(tmp_path):
    joblib.dump({"model": None, "labels": ["A"], "threshold": 0.5, "columns": COLS}, tmp_path / "a.joblib")
    joblib.dump({"model": None, "labels": ["A"], "threshold": 0.6, "columns": COLS[:4]}, tmp_path / "b.joblib")
    with pytest.raises(ValueError, match="threshold"):
        fm.load_pixel_models(tmp_path)
