"""Cross-AOI rice classification: features, consensus labels, a spatially validated model, maps.

Where this sits
---------------
The per-AOI `analysis` workflow expects labelled ground-truth polygons. This project has none, so
labels are built from **consensus**, and the model is trained across many AOIs at once:

1. **Features** (:func:`aoi_features`): for every pixel inside an AOI, the 5x5 linear-power mean of
   VH and VV, averaged into calendar half-months, plus VH - VV. The same bins on every AOI make
   pixels from different AOIs and acquisition days comparable. Missing bins are linearly
   interpolated in time.
2. **Consensus labels** (:func:`consensus_labels`): a pixel is labelled only when independent
   sources agree. It must belong to a seasonal-pattern group already judged on SAR and Sentinel-2
   evidence, **and** all three published rice maps must agree with that judgement, **and** it must
   sit near its group's centre rather than on a boundary with another group.
3. **Model** (:func:`train_and_validate`): gradient-boosted trees, validated by leaving whole
   **AOIs** out. Pixels from one AOI are correlated, so a random split would test on near-copies of
   training pixels and flatter the score.
4. **Maps** (:func:`write_class_raster`): one class raster and one rice-probability raster per AOI.

The honest limitation
---------------------
The labels come partly from SAR-derived groups, and the model is trained on SAR. It will therefore
tend to reproduce the grouping. Two things limit that circularity: labels also require agreement
from three published maps built by other teams from other data and years, and the model is judged
on AOIs it never saw. Scores here measure **consistency with the consensus**, not accuracy against
field truth, and must be reported that way.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import patterns as pt
from . import seasonal_stats as ss

#: Output classes. 0 is reserved for "no data / outside the AOI" in rasters.
CLASSES = {
    1: "rice_monsoon",      # monsoon rice, no second rice crop
    2: "rice_double",       # monsoon rice plus a dry-season rice crop
    3: "rice_dry_only",     # rice only in the dry season, after long monsoon flooding
    4: "non_rice",          # trees, villages, water, other land with no rice
}
#: Classes that carry a monsoon rice crop (the project's stated target).
MONSOON_RICE = (1, 2)

#: Which seasonal-pattern groups (from the k=10 clustering) feed which class, and what the
#: published maps must say for a pixel to be kept. Documented in docs phase 4c.
GROUP_TO_CLASS = {7: 1, 4: 1, 5: 2, 6: 2, 8: 3, 9: 3}
NON_RICE_GROUPS = (3, 0)

#: Version 2 also labels the "probable" groups where all three maps agree on rice: group 1 (rainfed
#: monsoon rice, mostly in the north) and group 2 (monsoon rice followed by an unflooded dry-season
#: crop). Both get the monsoon-only class. Reason: version 1 left them unlabelled, and the model then
#: forced group 1 into "double" rice, for which there is no evidence (no dry-season flooding).
GROUP_TO_CLASS_V2 = {**GROUP_TO_CLASS, 1: 1, 2: 1}


def interpolate_bins(cube):
    """Fill NaN time bins by linear interpolation along axis 0, holding the ends; vectorised.

    Works on ``(n_bins, ...)`` arrays of any trailing shape. Pixels with no finite bin stay NaN.
    """
    cube = np.array(cube, dtype="float32", copy=True)
    n = cube.shape[0]
    flat = cube.reshape(n, -1)
    valid = np.isfinite(flat)
    idx = np.arange(n)[:, None]
    prev = np.where(valid, idx, -1)
    prev = np.maximum.accumulate(prev, axis=0)
    nxt = np.where(valid, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1]
    cols = np.arange(flat.shape[1])
    has_prev, has_next = prev >= 0, nxt < n
    p = np.clip(prev, 0, n - 1)
    q = np.clip(nxt, 0, n - 1)
    vp = flat[p, cols[None, :].repeat(n, 0)]
    vq = flat[q, cols[None, :].repeat(n, 0)]
    span = np.where(q > p, q - p, 1)
    w = (idx - p) / span
    interp = np.where(has_prev & has_next, vp + w * (vq - vp),
                      np.where(has_prev, vp, np.where(has_next, vq, np.nan)))
    flat[:] = np.where(valid, flat, interp)
    return cube


def _binned(cube_db, dates, grid_bins):
    labels, binned = ss.bin_in_linear(cube_db, dates)
    out = np.full((grid_bins.size,) + binned.shape[1:], np.nan, dtype="float32")
    for i, label in enumerate(grid_bins):
        hit = np.where(labels == label)[0]
        if hit.size:
            out[i] = binned[hit[0]]
    return out


def aoi_features(run_dir, track, aoi_path, start, end, window=5, thin=False):
    """Per-pixel features for every pixel inside one AOI.

    Returns a dict with ``X`` (n_pixels, 3 * n_bins) as VH bins, VV bins and VH-VV bins, the pixel
    ``rows``/``cols`` on the grid, the grid ``shape``, the half-month ``bins`` and the feature
    ``names``. With ``thin=True`` every other acquisition is dropped first, which imitates the
    12-day cadence of a single satellite and is used to test robustness for years where only one
    satellite flies a track.
    """
    run_dir = Path(run_dir)
    grid_bins = pt.half_month_edges(start, end)
    stacks = {}
    for pol in ("VH", "VV"):
        dates, cube = ss.read_stack(run_dir / "stack" / f"track_{track}" / f"stack_{pol}.vrt")
        keep = ss.window_indices(dates, start, end)
        if thin:
            keep = keep[::2]
        dates = [dates[i] for i in keep]
        stacks[pol] = interpolate_bins(_binned(ss.spatial_mean(cube[keep], size=window), dates, grid_bins))
    vrt = run_dir / "stack" / f"track_{track}" / "stack_VH.vrt"
    mask = ss.aoi_mask(aoi_path, vrt)
    ok = mask & np.isfinite(stacks["VH"]).all(axis=0) & np.isfinite(stacks["VV"]).all(axis=0)
    rows, cols = np.nonzero(ok)
    vh = stacks["VH"][:, rows, cols].T
    vv = stacks["VV"][:, rows, cols].T
    X = np.concatenate([vh, vv, vh - vv], axis=1).astype("float32")
    names = ([f"VH_{b}" for b in grid_bins] + [f"VV_{b}" for b in grid_bins]
             + [f"VHmVV_{b}" for b in grid_bins])
    return {"X": X, "rows": rows, "cols": cols, "shape": mask.shape, "bins": grid_bins,
            "names": names, "n_aoi_pixels": int(mask.sum())}


def consensus_labels(table, centre_quantile=0.75, map_cols=("zhao2024", "opensea2021", "sun2019"),
                     group_to_class=None, non_rice_groups=None):
    """Assign training labels only where independent evidence agrees. Unlabelled rows get 0.

    ``table`` needs ``cluster``, ``centre_dist`` (distance to its cluster centre) and one column per
    published map (0 = not rice, >0 = rice). Rules:

    * rice classes: the pixel's group maps to that class (:data:`GROUP_TO_CLASS`), **all** maps say
      rice, and the pixel is within the ``centre_quantile`` of its group's distances to centre;
    * ``non_rice``: the group is one judged non-rice or uncertain (:data:`NON_RICE_GROUPS`), **all**
      maps say not rice, and the same centrality rule holds.
    """
    group_to_class = GROUP_TO_CLASS if group_to_class is None else group_to_class
    non_rice_groups = NON_RICE_GROUPS if non_rice_groups is None else non_rice_groups
    t = table.copy()
    all_rice = np.logical_and.reduce([t[c] > 0 for c in map_cols])
    no_rice = np.logical_and.reduce([t[c] == 0 for c in map_cols])
    limit = t.groupby("cluster")["centre_dist"].transform(lambda s: s.quantile(centre_quantile))
    central = t["centre_dist"] <= limit
    label = np.zeros(len(t), dtype=int)
    for group, cls in group_to_class.items():
        label[(t["cluster"] == group) & all_rice & central] = cls
    label[t["cluster"].isin(non_rice_groups) & no_rice & central] = 4
    t["label"] = label
    return t


def balanced_sample(table, per_class, per_group_cap, group_col="aoi", seed=0):
    """Cap each class, and each AOI within a class, so no single AOI or class dominates training."""
    parts = []
    for _, sub in table[table["label"] > 0].groupby("label"):
        # shuffle, then keep the first `per_group_cap` rows of each AOI: keeps every column
        # (groupby().apply would drop the grouping column in current pandas)
        capped = sub.sample(frac=1.0, random_state=seed).groupby(group_col).head(per_group_cap)
        parts.append(capped.sample(min(len(capped), per_class), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def make_model(seed=0):
    from xgboost import XGBClassifier

    return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8,
                         colsample_bytree=0.8, tree_method="hist", random_state=seed, n_jobs=-1,
                         eval_metric="mlogloss")


def train_and_validate(X, y, groups, n_folds=5, seed=0):
    """Leave-AOIs-out cross-validation, then a final model on all labelled data.

    Returns ``(model, report)``. ``report`` has out-of-fold predictions, per-class precision, recall
    and F1, the confusion matrix and the binary "monsoon rice yes/no" scores.
    """
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold

    y = np.asarray(y)
    classes = np.unique(y)
    enc = {c: i for i, c in enumerate(classes)}
    yi = np.array([enc[v] for v in y])
    oof = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=n_folds).split(X, yi, groups):
        m = make_model(seed).fit(X[tr], yi[tr])
        oof[te] = m.predict(X[te])
    pred = classes[oof]
    names = [CLASSES[c] for c in classes]
    report = {
        "classes": [int(c) for c in classes],
        "per_class": classification_report(y, pred, labels=classes, target_names=names,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, pred, labels=classes).tolist(),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "monsoon_rice_f1": float(f1_score(np.isin(y, MONSOON_RICE), np.isin(pred, MONSOON_RICE))),
        "oof_pred": pred,
    }
    model = make_model(seed).fit(X, yi)
    model.class_values_ = classes
    return model, report


def predict(model, X, batch=500_000):
    """Class codes and per-class probabilities for a feature matrix, in batches."""
    probs = np.concatenate([model.predict_proba(X[i:i + batch]) for i in range(0, len(X), batch)])
    return model.class_values_[probs.argmax(axis=1)], probs


def write_class_raster(path, shape, rows, cols, values, grid_def, dtype="uint8", nodata=0):
    """Write per-pixel values onto the AOI's grid as a GeoTIFF (compressed, tiled)."""
    import rasterio
    from rasterio.transform import from_origin

    arr = np.full(shape, nodata, dtype=dtype)
    arr[rows, cols] = values
    transform = from_origin(grid_def["x0"], grid_def["y0"], grid_def["res"], grid_def["res"])
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype=dtype,
                   crs=grid_def["crs"], transform=transform, nodata=nodata, compress="deflate",
                   tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr, 1)
    return path


def load_grid_def(run_dir):
    """The ``grid_def.json`` of the season folder a run belongs to."""
    return json.loads((Path(run_dir).parents[1] / "grid" / "grid_def.json").read_text())


def features_for_config(config_path, out_dir, start, end, samples=None):
    """Compute and save one AOI's features to ``<out_dir>/<aoi>.npz`` (float16 to halve the size).

    If ``samples`` (a table with ``aoi``, ``row``, ``col``) is given, the thinned (12-day-like)
    features of this AOI's sampled pixels are saved alongside, for the robustness test.
    """
    from .. import config as config_mod

    cfg = config_mod.load_config(config_path)
    aoi = cfg["aoi"]["key"]
    run = config_mod.run_dir(cfg)
    track = pt.primary_track(cfg)
    full = aoi_features(run, track, config_mod.aoi_path(cfg), start, end)
    extra = {}
    if samples is not None:
        mine = samples[samples["aoi"] == aoi]
        if len(mine):
            thin = aoi_features(run, track, config_mod.aoi_path(cfg), start, end, thin=True)
            key = {(r, c): i for i, (r, c) in enumerate(zip(thin["rows"], thin["cols"]))}
            idx = np.array([key.get((r, c), -1) for r, c in zip(mine["row"], mine["col"])])
            ok = idx >= 0
            extra = {"thin_rows": mine["row"].to_numpy()[ok], "thin_cols": mine["col"].to_numpy()[ok],
                     "thin_X": thin["X"][idx[ok]].astype("float16")}
    out = Path(out_dir) / f"{aoi}.npz"
    np.savez_compressed(out, X=full["X"].astype("float16"), rows=full["rows"], cols=full["cols"],
                        shape=np.array(full["shape"]), bins=full["bins"], names=np.array(full["names"]),
                        track=track, run=str(run), n_aoi_pixels=full["n_aoi_pixels"], **extra)
    return out


def gather_training(labels, feature_dir):
    """Pull the feature rows (full and thinned) of every labelled pixel out of the per-AOI files."""
    rows_full, rows_thin, keep = [], [], []
    for aoi, sub in labels[labels["label"] > 0].groupby("aoi"):
        z = np.load(Path(feature_dir) / f"{aoi}.npz")
        where = {(r, c): i for i, (r, c) in enumerate(zip(z["rows"], z["cols"]))}
        idx = np.array([where.get((r, c), -1) for r, c in zip(sub["row"], sub["col"])])
        thin_where = ({(r, c): i for i, (r, c) in enumerate(zip(z["thin_rows"], z["thin_cols"]))}
                      if "thin_rows" in z.files else {})
        tidx = np.array([thin_where.get((r, c), -1) for r, c in zip(sub["row"], sub["col"])])
        ok = (idx >= 0) & (tidx >= 0)
        rows_full.append(z["X"][idx[ok]].astype("float32"))
        rows_thin.append(z["thin_X"][tidx[ok]].astype("float32"))
        keep.append(sub[ok])
    names = list(np.load(Path(feature_dir) / f"{labels['aoi'].iloc[0]}.npz")["names"])
    return pd.concat(keep, ignore_index=True), np.vstack(rows_full), np.vstack(rows_thin), names


def thinned_cadence_scores(X_full, X_thin, y, groups, n_folds=5, seed=0):
    """Train on full-cadence features, test on held-out AOIs' thinned (every-other-date) features.

    This asks the question that matters for a year flown by one satellite per track (12-day
    revisit): does a model trained on the denser year still work when half the dates are missing?
    """
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupKFold

    classes = np.unique(y)
    enc = {c: i for i, c in enumerate(classes)}
    yi = np.array([enc[v] for v in y])
    pred_full = np.zeros(len(y), dtype=int)
    pred_thin = np.zeros(len(y), dtype=int)
    for tr, te in GroupKFold(n_splits=n_folds).split(X_full, yi, groups):
        m = make_model(seed).fit(X_full[tr], yi[tr])
        pred_full[te] = m.predict(X_full[te])
        pred_thin[te] = m.predict(X_thin[te])
    pf, pth = classes[pred_full], classes[pred_thin]
    return {"macro_f1_full": float(f1_score(y, pf, average="macro")),
            "macro_f1_thinned": float(f1_score(y, pth, average="macro")),
            "monsoon_f1_full": float(f1_score(np.isin(y, MONSOON_RICE), np.isin(pf, MONSOON_RICE))),
            "monsoon_f1_thinned": float(f1_score(np.isin(y, MONSOON_RICE), np.isin(pth, MONSOON_RICE)))}


def map_aoi(model, npz_path, out_dir, min_confidence=0.5):
    """Predict every pixel of one AOI and write its class, rice-probability and confidence rasters.

    Pixels whose top class probability is below ``min_confidence`` are written as class 255
    ("uncertain") rather than forced into a class. Returns per-class pixel counts.
    """
    z = np.load(npz_path)
    grid = load_grid_def(str(z["run"]))
    cls, probs = predict(model, z["X"].astype("float32"))
    top = probs.max(axis=1)
    out_cls = np.where(top >= min_confidence, cls, 255).astype("uint8")
    col = {c: i for i, c in enumerate(model.class_values_)}
    rice = sum(probs[:, col[c]] for c in (1, 2, 3) if c in col)
    monsoon = sum(probs[:, col[c]] for c in MONSOON_RICE if c in col)
    aoi = Path(npz_path).stem
    shape = tuple(int(v) for v in z["shape"])
    out_dir = Path(out_dir)
    write_class_raster(out_dir / f"{aoi}_class.tif", shape, z["rows"], z["cols"], out_cls, grid)
    write_class_raster(out_dir / f"{aoi}_rice_prob.tif", shape, z["rows"], z["cols"],
                       np.round(rice * 100).astype("uint8"), grid, nodata=255)
    write_class_raster(out_dir / f"{aoi}_monsoon_rice_prob.tif", shape, z["rows"], z["cols"],
                       np.round(monsoon * 100).astype("uint8"), grid, nodata=255)
    counts = {int(k): int(v) for k, v in zip(*np.unique(out_cls, return_counts=True))}
    return {"aoi": aoi, "pixels": int(len(out_cls)), **{f"class_{k}": v for k, v in counts.items()},
            "mean_rice_prob": float(rice.mean()), "mean_monsoon_prob": float(monsoon.mean())}


#: Map colours. The three rice classes take the three categorical slots that stay distinguishable
#: under colour-vision deficiency even on a map (all pairs); non-rice and uncertain are neutrals,
#: so they read as background rather than as a fourth and fifth category.
CLASS_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#b9b7b0", 255: "#ecebe7"}
CLASS_LABELS = {1: "rice: monsoon only", 2: "rice: monsoon + dry season", 3: "rice: dry season only",
                4: "not rice", 255: "uncertain (< 50% confidence)"}


def plot_class_maps(class_rasters, out_path, titles=None, title=None, subtitle=None, cols=3):
    """Small multiples of per-AOI class rasters with one shared legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from .curves import INK, INK_SECONDARY, SURFACE, _title_band

    codes = [1, 2, 3, 4, 255]
    lut = np.zeros(256, dtype=int)
    for i, c in enumerate(codes):
        lut[c] = i + 1
    cmap = ListedColormap([SURFACE] + [CLASS_COLOURS[c] for c in codes])
    n = len(class_rasters)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows), squeeze=False,
                             facecolor=SURFACE)
    for k, (ax, path) in enumerate(zip(axes.ravel(), class_rasters)):
        with rasterio.open(path) as ds:
            arr = ds.read(1)
        ax.imshow(lut[arr], cmap=cmap, vmin=0, vmax=len(codes), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        label = titles[k] if titles else Path(path).stem.replace("_class", "")
        ax.set_title(label, fontsize=10, color=INK_SECONDARY, loc="left", pad=4)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    handles = [Patch(facecolor=CLASS_COLOURS[c], edgecolor="none", label=CLASS_LABELS[c]) for c in codes]
    fig.legend(handles=handles, loc="lower center", ncol=len(codes), frameon=False, fontsize=9,
               labelcolor=INK)
    fig.tight_layout(rect=(0, 0.05, 1, _title_band(fig, title, subtitle)))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run(labels_path, feature_dir, out_dir, per_class=20000, per_aoi_cap=800, n_folds=5, seed=0,
        log=print):
    """End to end: balanced consensus sample -> leave-AOIs-out validation -> final model -> maps.

    Writes to ``out_dir``: ``model.joblib``, ``validation.json``, ``oof_predictions.parquet``,
    ``feature_importance.csv``, ``maps/<aoi>_{class,rice_prob,monsoon_rice_prob}.tif`` and
    ``aoi_summary.csv``. Nothing leaves the local disk.
    """
    import joblib

    out_dir = Path(out_dir)
    (out_dir / "maps").mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(labels_path)
    chosen = balanced_sample(labels, per_class, per_aoi_cap, seed=seed)
    log(f"training sample: {len(chosen)} pixels, {chosen['aoi'].nunique()} AOIs; per class "
        f"{chosen['label'].value_counts().sort_index().to_dict()}")
    table, X, X_thin, names = gather_training(chosen, feature_dir)
    y, groups = table["label"].to_numpy(), table["aoi"].to_numpy()

    model, report = train_and_validate(X, y, groups, n_folds=n_folds, seed=seed)
    log(f"leave-AOIs-out macro F1 {report['macro_f1']:.3f}, monsoon-rice F1 {report['monsoon_rice_f1']:.3f}")
    cadence = thinned_cadence_scores(X, X_thin, y, groups, n_folds=n_folds, seed=seed)
    log(f"cadence test: {cadence}")

    table = table.assign(oof_pred=report.pop("oof_pred"))
    table.to_parquet(out_dir / "oof_predictions.parquet")
    (out_dir / "validation.json").write_text(json.dumps({**report, "cadence": cadence,
                                                         "n_train": int(len(y))}, indent=2))
    pd.DataFrame({"feature": names, "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)
    joblib.dump(model, out_dir / "model.joblib")

    rows = []
    files = sorted(Path(feature_dir).glob("*.npz"))
    for i, f in enumerate(files):
        rows.append(map_aoi(model, f, out_dir / "maps"))
        if (i + 1) % 20 == 0:
            log(f"mapped {i + 1}/{len(files)}")
    summary = pd.DataFrame(rows).fillna(0)
    summary.to_csv(out_dir / "aoi_summary.csv", index=False)
    log("done")
    return summary
