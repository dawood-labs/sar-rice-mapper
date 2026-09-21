"""Final models for production maps, and the class map of a large AOI with them.

`save_final_models` runs after the `model` step. It takes the tuned winner and its threshold (`winner.json`) and
fits it on ALL fields as a small set of variants:

  full                  every feature
  only_<primary track>  only the primary track's features, for pixels a secondary track never covered
  no_<track>_<MMDD>     every feature except one time bin of one track (`--without-bin TRACK:MMDD`), for pixels
                        where that one acquisition is missing (for example a single missing slice)
  field_level           the same model on the MEAN of each field's pixel features (used by `field-labels`)

Each variant gets a grouped 5-fold CV score on all fields, written to `model_card.json`.

`predict` classifies every AOI pixel of a map run. Per pixel it uses the variant with the most features whose
features are all present, and writes which one it used, so every pixel of a map is classified by the best model
its data allows.

Outputs of `predict` in the map run's analysis folder:
  <name>_classes.tif          class code, threshold rule (nodata 0) + .qml
  <name>_classes_argmax.tif   class code, highest probability (nodata 0) + .qml
  <name>_<target>_prob.tif    target probability in percent (nodata 255)
  <name>_model_used.tif       1..n = variant used (legend in the summary), 0 = not classified
  <name>_summary.json         acres per class and per variant
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from .. import config as config_mod
from .. import manifest, resources, run_meta
from . import gcp_qc
from . import model as M
from .maps import check_same_layout, class_shares, slug, write_qml
from .pixel_features import (OK_STATES, SEP, FeatureSet, analysis_dir, plans_for, read_chunk_array, run_context,
                             set_from_config, track_features, wet_flags)

log = logging.getLogger(__name__)
SQM_PER_ACRE = 4046.8564224
MEM_PER_WORKER = 1024 ** 3    # one chunk's bands, window means and features, plus the loaded models


# ---------------------------------------------------------------- variants
def variant_columns(cols: list[str], tracks: list[dict], without_bins: list[str]) -> dict[str, list[str]]:
    """{variant name: columns}. `tracks` = config s1.tracks; `without_bins` = ["TRACK:MMDD", ...]."""
    out = {"full": list(cols)}
    if len(tracks) > 1:
        for t in tracks:
            if t.get("role") == "primary":
                out[f"only_{t['track_id']}"] = [c for c in cols if c.split(SEP)[0] == t["track_id"]]
    for item in without_bins:
        track, _, mmdd = item.partition(":")
        keep = [c for c in cols if not (c.split(SEP)[0] == track and c.split(SEP)[3] == mmdd)]
        if len(keep) == len(cols):
            raise ValueError(f"--without-bin {item}: no feature column has track {track!r} and bin {mmdd!r}")
        out[f"no_{track}_{mmdd}"] = keep
    return out


def cv_scores(X: pd.DataFrame, cols: list[str], y, labels, target: str, spec: dict, t: float, n_folds: int,
              n_jobs: int, seed: int) -> dict:
    """Grouped K-fold scores of one variant, with the argmax and the threshold rule."""
    proba = np.zeros((len(X), len(labels)))
    F = X[cols].to_numpy("float32")
    for tr, te in M.group_folds(X["group"].astype(str).to_numpy(), n_folds):
        proba[te] = M.build(spec, n_jobs, seed).fit(F[tr], y[tr]).predict_proba(F[te])
    ti = labels.index(target)
    res = {}
    for rule, idx in (("argmax", proba.argmax(1)), ("threshold", M.threshold_index(proba, ti, t))):
        pred = np.asarray(labels, dtype=object)[idx]
        s = M.scores(y, pred, labels, target)
        vote = M.field_vote(X["gcp_id"].to_numpy(), y, pred)
        s["field_vote_accuracy"] = float((vote["true"] == vote["pred"]).mean())
        res[rule] = {k: round(float(v), 4) for k, v in s.items()}
    return res


def save_final_models(cfg: dict, train_run: Path, name: str = "features", without_bins: list[str] | None = None) -> Path:
    t0 = time.time()
    a = cfg["analysis"]
    target, seed, n_folds = a["target_class"], int(a.get("seed", 0)), int(a.get("cv_folds", 5))
    n_jobs = resources.cpu_workers(resources.detect_resources(cfg), cfg)
    fs = set_from_config(cfg)
    model_dir = analysis_dir(train_run) / f"model_{fs.name}"
    winner_path = model_dir / "winner.json"
    if not winner_path.exists():
        raise FileNotFoundError(f"{winner_path} is missing: run the `model` step first")
    out = model_dir / "final_models"
    if out.exists() and any(out.glob("*.joblib")):
        raise FileExistsError(f"{out} already holds models: move that folder away to save new ones")
    out.mkdir(parents=True, exist_ok=True)
    winner = json.loads(winner_path.read_text())
    spec, t = winner["spec"], float(winner["threshold"])
    X, cols = M.load_features(cfg, train_run, fs, name)
    X = X.reset_index(drop=True)
    y = np.asarray(X["label"].astype(str).tolist(), dtype=object)
    labels = sorted(set(y))
    run_cfg = config_mod.load_run_config(train_run)
    card = {"trained_on": train_run.name, "feature_set": asdict(fs), "spec": spec, "target": target, "threshold": t,
            "labels": labels, "class_codes": gcp_qc.codes_from_config(cfg), "fields": int(X.gcp_id.nunique()),
            "pixels": int(len(X)), "decision_rule": f"{target} if P({target}) >= {t}, else the most likely other class",
            "variants": {}}
    stem = spec["model"]
    for variant, vc in variant_columns(cols, run_cfg["s1"]["tracks"], without_bins or []).items():
        t1 = time.time()
        scores = cv_scores(X, vc, y, labels, target, spec, t, n_folds, n_jobs, seed)
        final = M.build(spec, n_jobs, seed).fit(X[vc].to_numpy("float32"), y)
        path = out / f"{stem}_{variant}.joblib"
        _dump({"model": final, "columns": vc, "labels": labels, "target": target, "threshold": t, "spec": spec,
               "feature_set": asdict(fs), "variant": variant}, path)
        card["variants"][variant] = {"file": path.name, "n_features": len(vc), f"cv{n_folds}_grouped_all_fields": scores}
        log.info("%s: %d features, %s F1 %.3f (threshold rule) (%.0f s)", variant, len(vc), target,
                 scores["threshold"]["target_f1"], time.time() - t1)

    # field level: one row per field, the mean of its pixel features; its own threshold from out-of-fold probabilities
    t1 = time.time()
    fields = X.groupby("gcp_id").agg(label=("label", "first"), group=("group", "first"))
    means = X.groupby("gcp_id")[cols].mean().loc[fields.index]
    fy = np.asarray(fields["label"].astype(str).tolist(), dtype=object)
    proba = np.zeros((len(fields), len(labels)))
    for tr, te in M.group_folds(fields["group"].astype(str).to_numpy(), n_folds):
        proba[te] = M.build(spec, n_jobs, seed).fit(means.to_numpy("float32")[tr], fy[tr]).predict_proba(means.to_numpy("float32")[te])
    curve = M.threshold_curve(fy, proba, labels, target)
    best = curve.loc[curve["f1"].idxmax()]
    ft = float(best["t"])
    final = M.build(spec, n_jobs, seed).fit(means.to_numpy("float32"), fy)
    path = out / f"{stem}_field_level.joblib"
    _dump({"model": final, "columns": cols, "labels": labels, "target": target, "threshold": ft, "spec": spec,
           "feature_set": asdict(fs), "variant": "field_level", "stat": "mean"}, path)
    card["field_level"] = {"file": path.name, "statistic": "mean", "fields": int(len(fields)), "threshold": ft,
                           f"cv{n_folds}_grouped_at_threshold": {k: round(float(best[k]), 4) for k in ("precision", "recall", "f1")}}
    log.info("field_level: %d fields, threshold %.2f, %s F1 %.3f (%.0f s)", len(fields), ft, target, best["f1"], time.time() - t1)
    run_meta.atomic_write_json(out / "model_card.json", card)
    log.info("final models saved in %s (%.0f s)", out, time.time() - t0)
    return out


def _dump(obj: dict, path: Path) -> None:
    import joblib

    tmp = path.with_name(f".{path.name}.tmp")
    joblib.dump(obj, tmp)
    run_meta.replace_with_retry(tmp, path)


# ---------------------------------------------------------------- predict
def load_pixel_models(models_dir: Path) -> list[dict]:
    """Pixel-level model bundles of a folder, most features first (field-level bundles are skipped)."""
    import joblib

    bundles = []
    for p in sorted(Path(models_dir).glob("*.joblib")):
        b = joblib.load(p)
        if "stat" in b:
            continue
        b["file"] = p.name
        bundles.append(b)
    if not bundles:
        raise FileNotFoundError(f"no pixel-level .joblib models in {models_dir}")
    bundles.sort(key=lambda b: (-len(b["columns"]), b["file"]))
    for b in bundles[1:]:
        if b["labels"] != bundles[0]["labels"] or b["threshold"] != bundles[0]["threshold"]:
            raise ValueError(f"{b['file']} has other labels or another threshold than {bundles[0]['file']}")
    return bundles


def pick_models(features: dict[str, np.ndarray], n: int, bundles: list[dict]) -> np.ndarray:
    """Per pixel: 1-based index of the first bundle whose columns are all finite, 0 when none is."""
    finite = {}
    pick = np.zeros(n, dtype="uint8")
    for i, b in enumerate(bundles, 1):
        ok = np.ones(n, dtype=bool)
        for c in b["columns"]:
            if c not in finite:
                finite[c] = np.isfinite(features[c]) if c in features else np.zeros(n, dtype=bool)
            ok &= finite[c]
        pick[(pick == 0) & ok] = i
    return pick


_W: dict = {}


def _init_worker(bundles: list[dict], ctx: dict) -> None:
    for b in bundles:            # one thread per model inside a worker: the workers already use every CPU
        inner = getattr(b["model"], "model", b["model"])
        if hasattr(inner, "set_params"):
            inner.set_params(n_jobs=1)
    _W.update(bundles=bundles, **ctx)


def _classify_chunk(chunk: dict):
    bundles, gd, tracks, plans, man, codes = _W["bundles"], _W["gd"], _W["tracks"], _W["plans"], _W["man"], _W["code_of"]
    h, w = chunk["height"], chunk["width"]
    n = h * w
    rr, cc = np.indices((h, w)).reshape(2, -1)
    with rasterio.open(_W["index_path"]) as idx:
        inside = idx.read(3, window=Window(chunk["col_off"], chunk["row_off"], w, h)).reshape(-1) == 1
    feats = {}
    for tr in tracks:
        row = man[(man.track_id == tr) & (man.chunk_name == chunk["name"]) & man.state.isin(OK_STATES)]
        if row.empty:
            continue                                   # this track never covered the chunk: its features stay missing
        paths = [_W["map_run"] / p for p in row.local_paths.iloc[0].split(";") if p]
        feats.update(track_features(10 ** (read_chunk_array(paths, chunk, gd) / 10), rr, cc, tr, plans[tr]))
    pick = np.where(inside, pick_models(feats, n, bundles), 0).astype("uint8")
    cls_t, cls_a = np.zeros(n, "uint8"), np.zeros(n, "uint8")
    prob = np.full(n, 255, "uint8")
    ti, t = _W["t_idx"], bundles[0]["threshold"]
    for i, b in enumerate(bundles, 1):
        m = pick == i
        if not m.any():
            continue
        p = b["model"].predict_proba(np.column_stack([feats[c][m] for c in b["columns"]]).astype("float32"))
        cls_t[m] = codes[M.threshold_index(p, ti, t)]
        cls_a[m] = codes[p.argmax(axis=1)]
        prob[m] = np.round(100 * p[:, ti]).astype("uint8")
    return chunk, cls_t.reshape(h, w), cls_a.reshape(h, w), prob.reshape(h, w), pick.reshape(h, w)


def predict(cfg: dict, train_run: Path, map_run: Path, models_dir: Path | None = None, name: str | None = None) -> Path:
    t0 = time.time()
    a = cfg["analysis"]
    target = a["target_class"]
    fs = set_from_config(cfg)
    models_dir = Path(models_dir or analysis_dir(train_run) / f"model_{fs.name}" / "final_models")
    bundles = load_pixel_models(models_dir)
    if "feature_set" in bundles[0]:
        fs = FeatureSet(**bundles[0]["feature_set"])
    labels = bundles[0]["labels"]
    codes = gcp_qc.codes_from_config(cfg)
    check_same_layout(train_run, map_run)
    _, gd, tracks, season_start, end = run_context(map_run)
    ctx = {"gd": gd, "tracks": tracks, "man": manifest.read_manifest(map_run), "map_run": Path(map_run),
           "code_of": np.array([codes[c] for c in labels], dtype="uint8"), "t_idx": labels.index(target),
           "index_path": config_mod.grid_dir(config_mod.load_run_config(map_run)) / "pixel_index.tif",
           "plans": {tr: plans_for(map_run, tr, [fs], wet_flags(train_run, tr, float(a.get("rain_mm_24h", 5))),
                                   season_start, end)[0] for tr in tracks}}
    H, W = gd["height"], gd["width"]
    grids = {k: np.zeros((H, W), "uint8") for k in ("classes", "classes_argmax", "model_used")}
    prob = np.full((H, W), 255, "uint8")
    workers = resources.cpu_workers(resources.detect_resources(cfg), cfg, mem_per_worker_bytes=MEM_PER_WORKER)
    log.info("models (most features first): %s", ", ".join(f"{i} {b['file']} ({len(b['columns'])})" for i, b in enumerate(bundles, 1)))
    log.info("%d chunks, %d processes", len(gd["chunks"]), workers)

    def place(result):
        ch, ct, ca, pr, used = result
        sl = (slice(ch["row_off"], ch["row_off"] + ch["height"]), slice(ch["col_off"], ch["col_off"] + ch["width"]))
        grids["classes"][sl], grids["classes_argmax"][sl], grids["model_used"][sl], prob[sl] = ct, ca, used, pr

    if workers == 1:
        _init_worker(bundles, ctx)
        results = map(_classify_chunk, gd["chunks"])
    else:
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(bundles, ctx))
        results = pool.map(_classify_chunk, gd["chunks"])
    try:
        for k, res in enumerate(results, 1):
            place(res)
            if k % 20 == 0 or k == len(gd["chunks"]):
                log.info("  chunk %d/%d (%.0f s)", k, len(gd["chunks"]), time.time() - t0)
    finally:
        if workers > 1:
            pool.shutdown()

    out = analysis_dir(map_run)
    out.mkdir(parents=True, exist_ok=True)
    name = name or f"final_{bundles[0]['spec']['model']}_{fs.name}"
    base = dict(driver="GTiff", width=W, height=H, count=1, dtype="uint8", crs=gd["crs"], transform=Affine(*gd["transform"]),
                tiled=True, blockxsize=512, blockysize=512, compress="deflate")
    rasters = ((f"{name}_classes", grids["classes"], 0, f"class code, {target} if its probability >= threshold"),
               (f"{name}_classes_argmax", grids["classes_argmax"], 0, "class code, highest probability"),
               (f"{name}_{slug(target)}_prob", prob, 255, f"{target} probability, percent"),
               (f"{name}_model_used", grids["model_used"], 0, "model used, see the summary json"))
    for stem, arr, nodata, desc in rasters:
        tmp = out / f".{stem}.tmp.tif"
        with rasterio.open(tmp, "w", nodata=nodata, **base) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, desc)
        run_meta.replace_with_retry(tmp, out / f"{stem}.tif")
    write_qml(out / f"{name}_classes.qml", codes)
    write_qml(out / f"{name}_classes_argmax.qml", codes)
    px_acres = gd["res"] ** 2 / SQM_PER_ACRE
    names = {v: k for k, v in codes.items()}
    summary = {"map_run": Path(map_run).name, "trained_on": Path(train_run).name, "models_dir": str(models_dir),
               "feature_set": fs.name, "threshold": bundles[0]["threshold"],
               "classified_acres": round(float((grids["model_used"] > 0).sum()) * px_acres),
               "model_used": {str(i): {"file": b["file"], "n_features": len(b["columns"]),
                                       "acres": round(float((grids["model_used"] == i).sum()) * px_acres)}
                              for i, b in enumerate(bundles, 1)},
               "class_acres": {rule: {names[int(v)]: round(float(c) * px_acres) for v, c in zip(*np.unique(g[g > 0], return_counts=True))}
                               for rule, g in (("threshold", grids["classes"]), ("argmax", grids["classes_argmax"]))},
               "class_share_percent": {"threshold": class_shares(grids["classes"], codes),
                                       "argmax": class_shares(grids["classes_argmax"], codes)},
               "seconds": round(time.time() - t0)}
    run_meta.atomic_write_json(out / f"{name}_summary.json", summary)
    log.info("map -> %s_classes.tif | %s acres classified | class share %%: %s (%.0f s)", out / name,
             f"{summary['classified_acres']:,}", summary["class_share_percent"]["threshold"], time.time() - t0)
    return out / f"{name}_classes.tif"
