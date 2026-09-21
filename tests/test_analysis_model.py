"""Model phase: small synthetic checks, no real data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sar_pipeline.analysis import model as m


def _fields(seed=1, n_blocks=60, classes=("A", "B", "C")):
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        for _ in range(rng.integers(1, 6)):
            rows.append({"gcp_id": len(rows), "label": classes[rng.integers(len(classes))], "group": f"g{b}",
                         "n_pixels": int(rng.integers(20, 200))})
    return pd.DataFrame(rows)


def test_holdout_split_keeps_blocks_whole_and_meets_class_minimum():
    f = _fields()
    split = m.split_holdout(f, seed=0, min_fields=8, trials=300)
    assert split.groupby("group")["split"].nunique().max() == 1      # a block is never split
    hold = f[f.gcp_id.isin(split.loc[split.split == "holdout", "gcp_id"])]
    assert (hold.label.value_counts() >= 8).all() and hold.label.nunique() == 3
    assert 0.15 <= len(hold) / len(f) <= 0.3
    assert split.equals(m.split_holdout(f, seed=0, min_fields=8, trials=300))  # deterministic


def test_holdout_split_raises_when_minimum_is_impossible():
    with pytest.raises(ValueError):
        m.split_holdout(_fields(n_blocks=10), seed=0, min_fields=50, trials=20)


def test_threshold_rule():
    proba = np.array([[0.5, 0.3, 0.2],    # target B at 0.3 < t -> best other class A
                      [0.2, 0.35, 0.45],  # B at 0.35 >= t -> B although C is larger
                      [0.1, 0.1, 0.8]])
    assert m.threshold_index(proba, t_idx=1, t=0.35).tolist() == [0, 1, 2]
    assert m.threshold_index(proba, t_idx=1, t=0.9).tolist() == [0, 2, 2]


def test_tuning_rejects_holdout_rows(tmp_path):
    X = np.zeros((4, 2), dtype="float32")
    y = np.array(["A", "B", "A", "B"], dtype=object)
    with pytest.raises(ValueError, match="holdout"):
        m.tune(X, y, np.array(["g1", "g1", "g2", "g2"]), np.array([1, 1, 2, 2]), holdout_ids=[2], labels=["A", "B"],
               target="A", out_csv=tmp_path / "t.csv", n_folds=2, n_jobs=1, seed=0)
    assert not (tmp_path / "t.csv").exists()


def _xy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(["Maize", "Rice", "Sugarcane"], dtype=object)[rng.integers(0, 3, n)]
    X = rng.normal(size=(n, 4)).astype("float32") + (y == "Maize")[:, None] * 1.5
    return X, y


def test_xgb_label_encoding_round_trip():
    X, y = _xy()
    model = m.XGBLabels(balanced=True, n_estimators=5, max_depth=2, n_jobs=1).fit(X, y)
    assert list(model.classes_) == ["Maize", "Rice", "Sugarcane"]
    pred = model.classes_[model.predict_proba(X).argmax(axis=1)]
    assert set(pred) <= set(y) and (pred[y == "Maize"] == "Maize").mean() > 0.5


@pytest.mark.parametrize("kind,params", [("rf", {"max_features": "sqrt", "min_samples_leaf": 2, "class_weight": None}),
                                         ("xgb", {"max_depth": 3, "learning_rate": 0.1, "colsample_bytree": 0.9, "balanced": False})])
def test_first_n_trees_equal_a_smaller_model(kind, params):
    X, y = _xy()
    big = m.build({"model": kind, **params, "n_estimators": 20}, n_jobs=1, seed=0).fit(X, y)
    small = m.build({"model": kind, **params, "n_estimators": 8}, n_jobs=1, seed=0).fit(X, y)
    np.testing.assert_allclose(m.proba_first(big, X, 8), small.predict_proba(X), rtol=1e-6, atol=1e-6)


def test_bootstrap_resamples_fields_and_brackets_the_estimate():
    ids = np.repeat(np.arange(40), 5)
    y = np.where(ids % 2 == 0, "Maize", "Rice").astype(object)
    pred = y.copy(); pred[ids % 8 == 0] = "Rice"     # every 4th maize field missed
    ci = m.bootstrap_target(ids, y, pred, "Maize", seed=0, n_boot=200)
    f1 = m.scores(y, pred, ["Maize", "Rice"], "Maize")["target_f1"]
    assert ci["f1"][0] <= f1 <= ci["f1"][1] and ci["precision"] == [1.0, 1.0]


def test_rank_uses_target_f1_then_macro_f1():
    board = pd.DataFrame({"spec": ["a", "b", "c"], "target_f1": [0.9001, 0.9004, 0.8], "macro_f1": [0.7, 0.8, 0.99]})
    assert m.rank(board)["spec"].tolist() == ["b", "a", "c"]
