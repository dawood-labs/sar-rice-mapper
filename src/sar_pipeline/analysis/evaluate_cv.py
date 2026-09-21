"""Spatial cross-validation of feature sets with one fixed Random Forest.

Why a fixed model: these runs compare FEATURES (bins, window, start date, tracks). Keeping the model and its
settings identical means any difference comes from the features. It is a diagnostic baseline, not a tuned
production model (see docs/08_analysis.md).

Why grouped CV: nearby fields look alike. Fields are grouped in `group_cell_m` cells and a whole cell is either
training or test in each fold, so scores are not inflated by neighbours.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import resources
from ..run_meta import atomic_write_csv
from .pixel_features import columns_for_set, sets_in

log = logging.getLogger(__name__)


def make_model(cfg: dict):
    from sklearn.ensemble import RandomForestClassifier

    a = cfg.get("analysis") or {}
    rf = dict(a.get("rf") or {})
    rf.setdefault("n_estimators", 200)
    rf.setdefault("min_samples_leaf", 2)
    rf.setdefault("class_weight", "balanced")
    rf.setdefault("max_features", "sqrt")
    return RandomForestClassifier(random_state=int(a.get("seed", 0)),
                                  n_jobs=resources.cpu_workers(resources.detect_resources(cfg), cfg), **rf)


def majority(labels: pd.Series) -> str:
    return labels.value_counts().index[0]


def evaluate(cfg: dict, features_path: Path, set_names: list[str] | None = None, save_oof: bool = False) -> pd.DataFrame:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    from sklearn.model_selection import GroupKFold, cross_val_predict

    a = cfg.get("analysis") or {}
    target = a.get("target_class")
    X = pd.read_parquet(features_path)
    y = np.asarray(X["label"].astype(str).tolist(), dtype=object)
    groups = np.asarray(X["group"].astype(str).tolist(), dtype=object)
    labels = sorted(set(y))
    if target not in labels:
        raise ValueError(f"analysis.target_class {target!r} is not one of the labels {labels}")
    out_dir = Path(features_path).parent
    set_names = set_names or sets_in(X.columns)
    rows = []
    for name in set_names:
        cols = columns_for_set(X.columns, name)
        if not cols:
            raise ValueError(f"no feature columns for set {name!r}; available: {sets_in(X.columns)}")
        proba = cross_val_predict(make_model(cfg), X[cols].to_numpy(dtype="float32"), y, groups=groups,
                                  cv=GroupKFold(n_splits=int(a.get("cv_folds", 5))), method="predict_proba")
        pred = np.array(labels, dtype=object)[proba.argmax(axis=1)]
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
        per = {lab: i for i, lab in enumerate(labels)}
        vote = (pd.DataFrame({"gcp_id": X.gcp_id.to_numpy(), "true": y, "pred": pred})
                  .groupby("gcp_id").agg(true=("true", "first"), pred=("pred", majority)))
        row = {"set": name, "n_features": len(cols), "n_pixels": len(X),
               "target_precision": p[per[target]], "target_recall": r[per[target]], "target_f1": f[per[target]],
               "target_f1_field_vote": f1_score(vote.true, vote.pred, labels=[target], average=None, zero_division=0)[0],
               "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0)}
        row.update({f"f1_{lab}": f[per[lab]] for lab in labels})
        rows.append(row)
        cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"true {l}" for l in labels],
                          columns=[f"pred {l}" for l in labels])
        atomic_write_csv(cm.reset_index(names=""), out_dir / f"confusion_{name}.csv")
        if save_oof:
            oof = pd.DataFrame({"pid": X.pid.to_numpy(), "gcp_id": X.gcp_id.to_numpy(), "label": y, "pred": pred,
                                "target_prob": proba[:, per[target]]})
            oof.to_parquet(out_dir / f"oof_{name}.parquet", index=False)
        log.info("%s: %s F1 %.3f (P %.3f, R %.3f), field vote %.3f, accuracy %.3f", name, target, row["target_f1"],
                 row["target_precision"], row["target_recall"], row["target_f1_field_vote"], row["accuracy"])
    summary = pd.DataFrame(rows).round(3)
    atomic_write_csv(summary, out_dir / f"cv_summary_{Path(features_path).stem}.csv")
    return summary
