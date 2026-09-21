"""Model phase: spatial holdout, RF / XGBoost tuning, target-class threshold, one holdout test, validated map.

Order (each stage resumes from its output file):
  1. holdout   ~20 % of the FIELDS, chosen as whole `group` blocks, is locked away before any tuning.
  2. tune      RF and XGBoost grids, GroupKFold (by block) on the remaining dev fields only.
               Winner = best target-class F1, tie-break macro F1 (both rounded to 3 decimals).
  3. threshold from the winner's dev out-of-fold probabilities: predict the target class when its probability
               is >= t, otherwise the most likely other class; t maximises target F1 on dev.
  4. holdout   winner fitted on all dev fields, tested ONCE on the holdout (argmax and threshold rule),
               with a 95 % bootstrap interval that resamples fields.
  5. map       winner refitted on ALL fields, applied to a map run with the threshold rule.

Holdout rows never reach stages 2 and 3: `check_no_holdout` raises if they do.
"""
from __future__ import annotations

import copy
import itertools
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.transform import Affine

from .. import resources, run_meta
from . import gcp_qc
from .evaluate_cv import majority
from .maps import class_shares, predict_aoi, slug, write_class_rasters, write_qml
from .pixel_features import FeatureSet, analysis_dir, columns_for_set, extract, set_from_config

log = logging.getLogger(__name__)

# n_estimators is listed last: models that differ only in it are fitted once with the largest value and scored
# with the first N trees / boosting rounds, which gives exactly the same predictions as a separate smaller model.
RF_GRID = {"max_features": ["sqrt", 0.3], "min_samples_leaf": [1, 2, 5], "class_weight": ["balanced", None],
           "n_estimators": [200, 500]}
XGB_GRID = {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "colsample_bytree": [0.6, 0.9],
            "balanced": [True, False], "n_estimators": [300, 600]}
XGB_FIXED = {"tree_method": "hist", "subsample": 0.8}
THRESHOLDS = np.round(np.arange(0.20, 0.8001, 0.05), 2)
HOLDOUT_FRACTION, MIN_HOLDOUT_FIELDS, SEARCH_TRIALS, N_BOOT = 0.2, 15, 2000, 1000


# ---------------------------------------------------------------- 1. holdout
def split_holdout(fields: pd.DataFrame, seed: int, frac: float = HOLDOUT_FRACTION, min_fields: int = MIN_HOLDOUT_FIELDS,
                  trials: int = SEARCH_TRIALS) -> pd.DataFrame:
    """Pick whole blocks as holdout. `fields`: one row per field with gcp_id, label, group, n_pixels.

    Each trial adds blocks in a random order until ~`frac` of the fields are in; a trial is valid when every
    class has >= `min_fields` holdout fields. The valid trial whose field share and pixel class shares are
    closest to the full data wins. Same data + seed -> same split.
    """
    rng = np.random.default_rng(seed)
    blocks = np.array(sorted(fields["group"].unique()))
    labels = sorted(fields["label"].unique())
    all_share = fields.groupby("label")["n_pixels"].sum().reindex(labels) / fields["n_pixels"].sum()
    target_n = frac * len(fields)
    best, best_score = None, np.inf
    for _ in range(trials):
        chosen, n = [], 0
        for b in rng.permutation(blocks):
            if n >= target_n:
                break
            chosen.append(b)
            n += int((fields["group"] == b).sum())
        hold = fields[fields["group"].isin(chosen)]
        if (hold["label"].value_counts().reindex(labels, fill_value=0) < min_fields).any():
            continue
        share = hold.groupby("label")["n_pixels"].sum().reindex(labels, fill_value=0) / hold["n_pixels"].sum()
        score = float((share - all_share).abs().sum() + abs(len(hold) / len(fields) - frac))
        if score < best_score:
            best, best_score = set(chosen), score
    if best is None:
        raise ValueError(f"no block split gives >= {min_fields} holdout fields per class in {trials} trials")
    out = fields[["gcp_id", "group"]].copy()
    out["split"] = np.where(out["group"].isin(best), "holdout", "dev")
    return out.sort_values("gcp_id").reset_index(drop=True)


def check_no_holdout(gcp_ids, holdout_ids) -> None:
    leaked = set(np.asarray(gcp_ids).tolist()) & set(np.asarray(holdout_ids).tolist())
    if leaked:
        raise ValueError(f"{len(leaked)} holdout fields reached a tuning/threshold step; the holdout must stay unseen")


# ---------------------------------------------------------------- models
class XGBLabels:
    """XGBClassifier with class-name labels (XGBoost needs integers 0..K-1) and optional balanced weights."""

    def __init__(self, balanced: bool, **params):
        self.balanced, self.params = balanced, params

    def fit(self, X, y):
        from sklearn.utils.class_weight import compute_sample_weight
        from xgboost import XGBClassifier

        self.classes_, codes = np.unique(np.asarray(y, dtype=object), return_inverse=True)
        weights = compute_sample_weight("balanced", codes) if self.balanced else None
        self.model = XGBClassifier(**self.params).fit(X, codes, sample_weight=weights)
        return self

    def predict_proba(self, X, n_rounds: int | None = None):
        return self.model.predict_proba(X, iteration_range=(0, n_rounds) if n_rounds else None)


def grid_specs() -> list[dict]:
    """Every grid point as {"model": "rf"|"xgb", **params}, in a fixed order."""
    out = []
    for kind, grid in (("rf", RF_GRID), ("xgb", XGB_GRID)):
        for values in itertools.product(*grid.values()):
            out.append({"model": kind, **dict(zip(grid, values))})
    return out


def spec_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True)


def build(spec: dict, n_jobs: int, seed: int):
    params = {k: v for k, v in spec.items() if k != "model"}
    if spec["model"] == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **params)
    balanced = params.pop("balanced")
    return XGBLabels(balanced, random_state=seed, n_jobs=n_jobs, **XGB_FIXED, **params)


def proba_first(model, X, n: int) -> np.ndarray:
    """Probabilities from the first `n` trees (RF) or boosting rounds (XGBoost) of a fitted model."""
    if isinstance(model, XGBLabels):
        return model.predict_proba(X, n_rounds=n)
    sub = copy.copy(model)
    sub.estimators_, sub.n_estimators = model.estimators_[:n], n
    return sub.predict_proba(X)


# ---------------------------------------------------------------- metrics and decision rules
def threshold_index(proba: np.ndarray, t_idx: int, t: float) -> np.ndarray:
    """Target class where its probability >= t, otherwise the most likely other class."""
    others = proba.copy()
    others[:, t_idx] = -np.inf
    return np.where(proba[:, t_idx] >= t, t_idx, others.argmax(axis=1))


def scores(y, pred, labels: list[str], target: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[target], zero_division=0)
    return {"target_precision": float(p[0]), "target_recall": float(r[0]), "target_f1": float(f[0]),
            "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, pred))}


def field_vote(gcp_ids, y, pred) -> pd.DataFrame:
    return (pd.DataFrame({"gcp_id": np.asarray(gcp_ids), "true": y, "pred": pred})
              .groupby("gcp_id").agg(true=("true", "first"), pred=("pred", majority)))


def bootstrap_target(gcp_ids, y, pred, target: str, seed: int, n_boot: int = N_BOOT) -> dict:
    """95 % intervals of target precision / recall / F1, resampling whole fields with replacement."""
    d = pd.DataFrame({"gcp_id": np.asarray(gcp_ids), "tp": (y == target) & (pred == target),
                      "fp": (y != target) & (pred == target), "fn": (y == target) & (pred != target)})
    per = d.groupby("gcp_id")[["tp", "fp", "fn"]].sum().to_numpy(dtype="float64")
    idx = np.random.default_rng(seed).integers(0, len(per), size=(n_boot, len(per)))
    tp, fp, fn = (per[idx].sum(axis=1)).T
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = {"precision": tp / (tp + fp), "recall": tp / (tp + fn), "f1": 2 * tp / (2 * tp + fp + fn)}
    return {k: [round(float(np.nanpercentile(v, 2.5)), 3), round(float(np.nanpercentile(v, 97.5)), 3)] for k, v in stats.items()}


# ---------------------------------------------------------------- 2. tuning
def group_folds(groups, n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=groups))


def oof_proba(spec_family: list[dict], X, y, folds, labels, n_jobs: int, seed: int) -> dict[int, np.ndarray]:
    """Out-of-fold probabilities for specs that differ only in n_estimators: {n_estimators: proba}."""
    n_list = [s["n_estimators"] for s in spec_family]
    out = {n: np.zeros((len(y), len(labels))) for n in n_list}
    for tr, te in folds:
        if sorted(set(y[tr])) != labels:
            raise ValueError("a training fold lacks a class; use fewer folds")
        model = build({**spec_family[0], "n_estimators": max(n_list)}, n_jobs, seed).fit(X[tr], y[tr])
        for n in n_list:
            out[n][te] = proba_first(model, X[te], n)
    return out


def tune(X, y, groups, gcp_ids, holdout_ids, labels, target: str, out_csv: Path, n_folds: int, n_jobs: int, seed: int) -> pd.DataFrame:
    """Score every grid point with grouped CV on dev rows; rows are appended to `out_csv` (resume skips done points)."""
    check_no_holdout(gcp_ids, holdout_ids)
    folds = group_folds(groups, n_folds)
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["spec"])
    done_keys = set(done["spec"])
    families: dict[str, list[dict]] = {}
    for spec in grid_specs():
        families.setdefault(spec_key({k: v for k, v in spec.items() if k != "n_estimators"}), []).append(spec)
    t0, rows = time.time(), done.to_dict("records")
    todo = [fam for fam in families.values() if any(spec_key(s) not in done_keys for s in fam)]
    log.info("tuning: %d grid points, %d still to do, %d folds, n_jobs %d", len(grid_specs()),
             sum(len(f) for f in todo), n_folds, n_jobs)
    for k, fam in enumerate(todo, 1):
        t1 = time.time()
        probas = oof_proba(fam, X, y, folds, labels, n_jobs, seed)
        for spec in fam:
            pred = np.asarray(labels, dtype=object)[probas[spec["n_estimators"]].argmax(axis=1)]
            rows.append({"spec": spec_key(spec), "model": spec["model"], **scores(y, pred, labels, target),
                         "fit_seconds": round((time.time() - t1) / len(fam), 1)})
        run_meta.atomic_write_csv(pd.DataFrame(rows), out_csv)
        log.info("  %d/%d %s F1 %s (%.0f s elapsed)", k, len(todo), fam[0]["model"],
                 [round(r["target_f1"], 3) for r in rows[-len(fam):]], time.time() - t0)
    return rank(pd.DataFrame(rows))


def rank(board: pd.DataFrame) -> pd.DataFrame:
    """Best first: target F1, then macro F1 (3 decimals, so tiny float noise does not decide)."""
    b = board.assign(_f1=board["target_f1"].round(3), _macro=board["macro_f1"].round(3))
    return b.sort_values(["_f1", "_macro"], ascending=False, kind="stable").drop(columns=["_f1", "_macro"]).reset_index(drop=True)


# ---------------------------------------------------------------- 3. threshold
def threshold_curve(y, proba, labels: list[str], target: str) -> pd.DataFrame:
    rows = []
    t_idx = labels.index(target)
    for t in THRESHOLDS:
        pred = np.asarray(labels, dtype=object)[threshold_index(proba, t_idx, t)]
        s = scores(y, pred, labels, target)
        rows.append({"t": float(t), "precision": s["target_precision"], "recall": s["target_recall"], "f1": s["target_f1"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. holdout
def evaluate_rule(y, pred, gcp_ids, labels: list[str], target: str, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": n})
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=[f"true {l}" for l in labels],
                      columns=[f"pred {l}" for l in labels]).reset_index(names="")
    vote = field_vote(gcp_ids, y, pred)
    summary = {**scores(y, pred, labels, target), "target_ci95": bootstrap_target(gcp_ids, y, pred, target, seed),
               "field_vote_accuracy": float((vote["true"] == vote["pred"]).mean()),
               "field_vote_target_f1": float(f1_score(vote["true"], vote["pred"], labels=[target], average=None, zero_division=0)[0]),
               "n_fields": int(len(vote)), "n_pixels": int(len(y))}
    return summary, per_class, cm


# ---------------------------------------------------------------- orchestration
def load_features(cfg: dict, train_run: Path, fs: FeatureSet, name: str) -> tuple[pd.DataFrame, list[str]]:
    """Features of set `fs` from `<name>.parquet` when its metadata matches, else extracted once into features_<set>."""
    out = analysis_dir(train_run)
    for stem in (name, f"features_{fs.name}"):
        meta_path = out / f"{stem}.json"
        if meta_path.exists() and json.loads(meta_path.read_text())["sets"].get(fs.name) == asdict(fs):
            log.info("features: reusing %s.parquet", stem)
            break
    else:
        stem = f"features_{fs.name}"
        extract(cfg, train_run, [fs], name=stem)
    X = pd.read_parquet(out / f"{stem}.parquet")
    return X, columns_for_set(X.columns, fs.name)


def composition(X: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    d = X[["gcp_id", "label"]].merge(split[["gcp_id", "split"]], on="gcp_id")
    px = d.groupby(["label", "split"]).size().unstack(fill_value=0)
    fl = d.drop_duplicates("gcp_id").groupby(["label", "split"]).size().unstack(fill_value=0)
    out = pd.DataFrame({"fields_dev": fl["dev"], "fields_holdout": fl["holdout"], "pixels_dev": px["dev"],
                        "pixels_holdout": px["holdout"]})
    out["pixel_share_dev"] = (out.pixels_dev / out.pixels_dev.sum()).round(3)
    out["pixel_share_holdout"] = (out.pixels_holdout / out.pixels_holdout.sum()).round(3)
    return out.reset_index(names="class")


def run(cfg: dict, train_run: Path, map_run: Path | None, name: str = "features") -> Path:
    t0 = time.time()
    a = cfg["analysis"]
    target, seed, n_folds = a["target_class"], int(a.get("seed", 0)), int(a.get("cv_folds", 5))
    n_jobs = resources.cpu_workers(resources.detect_resources(cfg), cfg)
    fs = set_from_config(cfg)
    X, cols = load_features(cfg, train_run, fs, name)
    out = analysis_dir(train_run) / f"model_{fs.name}"
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(X["label"].astype(str).tolist(), dtype=object)
    F = X[cols].to_numpy(dtype="float32")
    labels = sorted(set(y))
    if target not in labels:
        raise ValueError(f"analysis.target_class {target!r} is not one of the labels {labels}")

    # 1. holdout (recomputed every time; an existing file must match, so later stages stay valid)
    fields = X.groupby("gcp_id").agg(label=("label", "first"), group=("group", "first"), n_pixels=("pid", "size")).reset_index()
    split = split_holdout(fields, seed)
    split_path = out / "holdout_split.csv"
    if split_path.exists():
        old = pd.read_csv(split_path, dtype={"group": str})
        if not old[["gcp_id", "split"]].equals(split[["gcp_id", "split"]]):
            raise ValueError(f"{split_path} differs from the split of the current data: move the {out.name} folder away and rerun")
    run_meta.atomic_write_csv(split, split_path)
    comp = composition(X, split)
    run_meta.atomic_write_csv(comp, out / "holdout_composition.csv")
    log.info("holdout composition:\n%s", comp.to_string(index=False))
    hold_ids = split.loc[split.split == "holdout", "gcp_id"].to_numpy()
    is_hold = X["gcp_id"].isin(hold_ids).to_numpy()
    dev = ~is_hold
    g_dev, id_dev = X["group"].astype(str).to_numpy()[dev], X["gcp_id"].to_numpy()[dev]

    # 2. tuning on dev
    board = tune(F[dev], y[dev], g_dev, id_dev, hold_ids, labels, target, out / "tuning_results.csv", n_folds, n_jobs, seed)
    best = board.iloc[0]
    spec = json.loads(best["spec"])
    top = pd.concat([board.head(5), board[board.model == "rf"].head(1), board[board.model == "xgb"].head(1)]).drop_duplicates("spec")
    log.info("leaderboard (top 5 + best of each model):\n%s", top.round(3).to_string(index=False))

    # 3. threshold from the winner's dev out-of-fold probabilities
    winner_path = out / "winner.json"
    winner = json.loads(winner_path.read_text()) if winner_path.exists() else {}
    if winner.get("spec") != spec:
        check_no_holdout(id_dev, hold_ids)
        proba = oof_proba([spec], F[dev], y[dev], group_folds(g_dev, n_folds), labels, n_jobs, seed)[spec["n_estimators"]]
        curve = threshold_curve(y[dev], proba, labels, target)
        run_meta.atomic_write_csv(curve.round(4), out / "threshold_curve.csv")
        b = curve.loc[curve["f1"].idxmax()]
        winner = {"spec": spec, "threshold": float(b["t"]), "dev_oof_at_threshold": {k: round(float(b[k]), 4) for k in ("precision", "recall", "f1")},
                  "dev_oof_argmax": {k: round(float(best[k]), 4) for k in ("target_precision", "target_recall", "target_f1", "macro_f1", "accuracy")}}
        run_meta.atomic_write_json(winner_path, winner)
    t = winner["threshold"]
    t_idx = labels.index(target)
    rules = {"argmax": lambda p: p.argmax(axis=1), "threshold": lambda p: threshold_index(p, t_idx, t)}
    log.info("winner %s, threshold %.2f, dev OOF at threshold %s", spec, t, winner["dev_oof_at_threshold"])

    # 4. holdout, once
    metrics_path = out / "holdout_metrics.json"
    if metrics_path.exists():
        log.info("holdout already evaluated: %s (not re-run)", metrics_path)
    else:
        model = build(spec, n_jobs, seed).fit(F[dev], y[dev])
        proba = model.predict_proba(F[is_hold])
        report = {"spec": spec, "threshold": t}
        per_class = []
        for rule_name, rule in rules.items():
            pred = np.asarray(labels, dtype=object)[rule(proba)]
            summary, pc, cm = evaluate_rule(y[is_hold], pred, X["gcp_id"].to_numpy()[is_hold], labels, target, seed)
            report[rule_name] = summary
            per_class.append(pc.assign(rule=rule_name))
            run_meta.atomic_write_csv(cm, out / f"holdout_confusion_{rule_name}.csv")
        run_meta.atomic_write_csv(pd.concat(per_class).round(4), out / "holdout_per_class.csv")
        run_meta.atomic_write_json(metrics_path, report)
    log.info("holdout: %s", json.loads(metrics_path.read_text()))

    # 5. validated map: refit on all fields, threshold rule
    if map_run is not None:
        map_dir = analysis_dir(map_run)
        stem = f"model_{spec['model']}_{fs.name}"
        map_json = map_dir / f"{stem}_classes.json"
        if map_json.exists() and json.loads(map_json.read_text()).get("spec") == spec and json.loads(map_json.read_text()).get("threshold") == t:
            log.info("map already written: %s", map_json)
        else:
            model = build(spec, n_jobs, seed).fit(F, y)
            cls, prob, gd = predict_aoi(cfg, train_run, fs, cols, map_run, model, rules)
            codes = gcp_qc.codes_from_config(cfg)
            map_dir.mkdir(parents=True, exist_ok=True)
            cls_path = map_dir / f"{stem}_classes.tif"
            write_class_rasters(cls_path, map_dir / f"{stem}_{slug(target)}_prob.tif", cls["threshold"], prob,
                                gd["crs"], Affine(*gd["transform"]))
            write_qml(cls_path.with_suffix(".qml"), codes)
            run_meta.atomic_write_json(map_json, {"trained_on": train_run.name, "fields": int(X.gcp_id.nunique()), "set": fs.name,
                                                  "spec": spec, "threshold": t, "class_raster_rule": "threshold",
                                                  "class_share_percent": {r: class_shares(c, codes) for r, c in cls.items()}})
        log.info("map shares: %s", json.loads(map_json.read_text())["class_share_percent"])
    log.info("model phase done (%.0f s) -> %s", time.time() - t0, out)
    return out
