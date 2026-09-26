"""Rice from Sentinel-1 alone: features, a radar rule, a spatially validated model, scores, maps.

Why
---
Some AOIs lie under cloud so heavy that no optical time series can be built there, and the
validated method (docs/17) needs one. The question is how far the radar gets on its own. It is
tested where the answer can be checked: the surveyed rice plots, the optical-derived negatives
(evergreen, water, bare / built, cut) and the validated optical+radar map of 132 AOIs, which is
used as a *teacher* for training and as the agreement reference on AOIs a model never saw.
Every score is reported as plot recall, negative false-positive rate and agreement with the
teacher on held-out AOIs; cross-validation leaves whole AOIs out, never random pixels (pixels
of one AOI are near-copies of each other). Plan and log: ``docs/sar_only_plan_2026.md`` (local).

What the radar can see (docs/01, docs/09, docs/11)
--------------------------------------------------
* a transplanting flood: VH (and VV) collapse to -19 dB and below for two to six weeks, from a
  field that was bare or stubble before (the v2 flood test of ``radar_water``);
* the canopy closing: VH climbs 5-10 dB from the water within four to six weeks, VV rises by
  double bounce off the stems;
* the pre-monsoon state: harvested cropland is bare and smooth in March-April (VH low), trees
  and settlements stay high all year;
* the end of the season: a harvest is a fall of a few dB, weak on its own.

Two tracks (ascending and descending) are merged into one series by date, so every AOI gets a
pass every ~6 days; the tracks' incidence angles differ by a dB or two, which the summaries
tolerate and a tree model learns.

Steps (``python -m sar_pipeline.analysis.sar_only <step>``)::

    features --ids ...          # per-AOI feature tables -> processed/_batch/s2_2026/sar_only/aoi<N>_features.npz
    rule --ids ...              # E1: radar rule classes per pixel, scored on plots / negatives / teacher
    train --ids ... --by aoi    # E2: XGBoost on teacher labels, leave-AOIs-out CV, scores -> report
    map --ids ... --model ...   # class + probability rasters from a saved model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from . import pixel_report as pr
from . import radar_water as rw
from . import sar_curve

SRC = "processed/_batch/s2_2026"
OUT = f"{SRC}/sar_only"
SEASON_KEY = "monsoon2026"
#: The fixed time grid every AOI's merged series is resampled to (a model needs the same columns
#: everywhere; two tracks give a pass every ~6 days).
GRID_START, GRID_END, GRID_STEP_DAYS = "2026-03-15", "2026-09-21", 6
#: Monsoon water only: floods are searched from here (docs/11).
MONSOON_FROM = "2026-05-15"
#: Pre-monsoon window: the state of the ground before any monsoon crop.
PRE_MONSOON = ("2026-03-15", "2026-05-10")
#: Late-season window for the end level and the fall from the season's canopy.
LATE = ("2026-08-01", "2026-09-22")
END_DAYS = 20
#: Radar rule thresholds, the same numbers the validated rule uses (docs/11, docs/17).
FLOOD_DROP_MIN = rw.FLOOD_DROP_MIN          # 4 dB below the pass's own previous weeks
FLOOD_VH_MAX = rw.FLOOD_VH_MAX              # -19 dB on the flood pass
SUPPORT_MIN = rw.SUPPORT_MIN                # a second pass within 14 days also down
CANOPY_VH_MIN = rw.RADAR_CANOPY_VH_MIN      # -18 dB: VH back at a canopy level by the end
#: VH up from the water by at least this, with the end at a canopy level: a flood confirms bare wet
#: ground, so the climb back need not be large (region N: shallow -21 dB floods end at -16 dB, 4-5 dB).
CANOPY_RISE_MIN = 3.0
#: End-of-season VH between the water and the canopy level: a young crop.
YOUNG_VH_MIN = -19.0
#: Pre-monsoon VH above this on every pass: permanent cover (trees, houses), never a field.
COVER_VH_MIN = -14.0
#: A harvest at the end: both polarisations fall by at least this from their late-season maximum.
HARVEST_FALL_MIN = 3.0
#: Classes of the radar rule and of the model (the teacher's classes folded to what radar can name).
RULE_CLASSES = {0: "no rice cycle", 1: "rice, standing", 2: "young (flooded, small rise)", 4: "harvested",
                7: "flooded, not green", 255: "no data"}
#: Teacher classes -> training classes: 1 rice, 2 young, 0 not rice; None = not used for training
#: (the teacher itself is unsure there: water not confirmed, never bare, cut without water).
TEACHER_TO_TRAIN = {0: 0, 1: 1, 2: 2, 3: None, 4: 0, 5: None, 6: 1, 7: 2, 8: None, 255: None}
TRAIN_CLASSES = {0: "not rice", 1: "rice (teacher 1 + 6)", 2: "young / flooded (teacher 2 + 7)"}


# --------------------------------------------------------------------------- series

def merged_series(aoi_id: int, window: int = 5) -> dict:
    """Both tracks' VH and VV of one AOI, in dB, merged by date: ``dates`` (sorted), ``VH`` / ``VV``
    (n_dates, n_pixels), ``track`` (track index per date), ``drop_VH`` / ``drop_VV`` (the drop of
    every pass below its own track's level of the previous 12-45 days, ``radar_water.local_drops``,
    computed per track so the tracks' offsets never look like a drop), ``shape``."""
    loc = pr.locate(aoi_id, 0, season_key=SEASON_KEY)
    parts = []
    for k, t in enumerate(loc["cfg"]["s1"]["tracks"]):
        dates, cubes = sar_curve.read_track(loc, t["track_id"], window)
        flat = {p: cubes[p].reshape(len(dates), -1) for p in ("VV", "VH")}
        drops = {p: rw.local_drops(dates, flat[p]) for p in ("VV", "VH")}
        parts.append((dates, flat, drops, k, cubes["VH"].shape[1:]))
    dates = np.concatenate([p[0].to_numpy() for p in parts])
    order = np.argsort(dates, kind="stable")
    out = {"dates": pd.DatetimeIndex(dates[order]), "shape": parts[0][4],
           "track": np.concatenate([np.full(len(p[0]), p[3]) for p in parts])[order]}
    for p in ("VH", "VV"):
        out[p] = np.concatenate([q[1][p] for q in parts])[order]
        out[f"drop_{p}"] = np.concatenate([q[2][p] for q in parts])[order]
    return out


def fill_gaps(arr: np.ndarray) -> np.ndarray:
    """Forward- then back-fill NaN along time (axis 0): a screened pass or a nodata edge must not
    poison a whole pixel; the gap is at most one pass, so the nearest value is honest enough."""
    a = np.array(arr, dtype="float32", copy=True)
    t = a.shape[0]
    idx = np.where(np.isfinite(a), np.arange(t)[:, None], -1)
    idx = np.maximum.accumulate(idx, axis=0)
    filled = np.where(idx >= 0, np.take_along_axis(a, np.clip(idx, 0, t - 1), axis=0), np.nan)
    idx_b = np.where(np.isfinite(filled), np.arange(t)[:, None], t)
    idx_b = np.minimum.accumulate(idx_b[::-1], axis=0)[::-1]
    return np.where(idx_b < t, np.take_along_axis(filled, np.clip(idx_b, 0, t - 1), axis=0), np.nan)


def resample(dates, values: np.ndarray, grid) -> np.ndarray:
    """Linear interpolation of (n_dates, n) values onto the ``grid`` dates; held at the ends."""
    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]").astype("int64")
    g = pd.DatetimeIndex(grid).to_numpy().astype("datetime64[D]").astype("int64")
    v = fill_gaps(values)
    out = np.empty((len(g), v.shape[1]), dtype="float32")
    for k, t in enumerate(g):
        i = int(np.searchsorted(day, t))
        if i <= 0:
            out[k] = v[0]
        elif i >= len(day):
            out[k] = v[-1]
        else:
            w = (t - day[i - 1]) / max(day[i] - day[i - 1], 1)
            out[k] = (1 - w) * v[i - 1] + w * v[i]
    return out


def time_grid():
    return pd.date_range(GRID_START, GRID_END, freq=f"{GRID_STEP_DAYS}D")


# --------------------------------------------------------------------------- features

def _doy(dates) -> np.ndarray:
    return pd.DatetimeIndex(dates).dayofyear.to_numpy().astype("float32")


def pixel_features(aoi_id: int, window: int = 5, inside_only: bool = True) -> dict:
    """Per-pixel radar features of one AOI: the merged series on the fixed grid (VH, VV, VH-VV)
    and season summaries. Returns ``X`` (n, F) float32, ``names``, ``pixels`` (flat grid indices),
    ``shape``. Only pixels inside the AOI with a complete series are kept."""
    from . import ndvi_5day as nd

    s = merged_series(aoi_id, window)
    dates, vh, vv = s["dates"], s["VH"], s["VV"]
    n = vh.shape[1]
    keep = np.isfinite(vh).sum(axis=0) >= 0.7 * len(dates)
    if inside_only:
        keep &= nd.inside_aoi(aoi_id)
    pix = np.flatnonzero(keep)
    vh, vv = vh[:, pix], vv[:, pix]
    d_vh, d_vv = s["drop_VH"][:, pix], s["drop_VV"][:, pix]
    grid = time_grid()
    g_vh, g_vv = resample(dates, vh, grid), resample(dates, vv, grid)
    cols, names = [g_vh.T, g_vv.T, (g_vh - g_vv).T], []
    stamp = [d.strftime("%m%d") for d in grid]
    names += [f"VH_{t}" for t in stamp] + [f"VV_{t}" for t in stamp] + [f"VHmVV_{t}" for t in stamp]

    day = dates.to_numpy().astype("datetime64[D]")
    doy = _doy(dates)
    pre = (day >= np.datetime64(PRE_MONSOON[0])) & (day <= np.datetime64(PRE_MONSOON[1]))
    late = (day >= np.datetime64(LATE[0])) & (day <= np.datetime64(LATE[1]))
    monsoon = day >= np.datetime64(MONSOON_FROM)
    end = day >= day[-1] - np.timedelta64(END_DAYS, "D")
    f_vh, f_vv = fill_gaps(vh), fill_gaps(vv)
    summ = {}
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pol, f in (("VH", f_vh), ("VV", f_vv)):
            summ[f"{pol}_pre_mean"] = np.nanmean(f[pre], axis=0)
            summ[f"{pol}_pre_min"] = np.nanmin(f[pre], axis=0)
            i_min = np.nanargmin(np.where(monsoon[:, None], f, np.inf), axis=0)
            summ[f"{pol}_min"] = f[i_min, np.arange(len(pix))]
            summ[f"{pol}_doy_min"] = doy[i_min]
            i_max = np.nanargmax(np.where(np.isfinite(f), f, -np.inf), axis=0)
            summ[f"{pol}_max"] = f[i_max, np.arange(len(pix))]
            summ[f"{pol}_doy_max"] = doy[i_max]
            summ[f"{pol}_std"] = np.nanstd(f, axis=0)
            after = np.where(np.arange(len(day))[:, None] >= i_min[None, :], f, -np.inf)
            summ[f"{pol}_rise_after_min"] = after.max(axis=0) - summ[f"{pol}_min"]
            summ[f"{pol}_end"] = np.nanmedian(f[end], axis=0)
            summ[f"{pol}_late_max"] = np.nanmax(f[late], axis=0)
            summ[f"{pol}_end_fall"] = summ[f"{pol}_late_max"] - summ[f"{pol}_end"]
        summ["VH_n_dark19"] = (np.where(monsoon[:, None], f_vh, 0) <= FLOOD_VH_MAX).sum(axis=0)
        summ["VH_n_dark18"] = (np.where(monsoon[:, None], f_vh, 0) <= rw.SHALLOW_VH_MAX).sum(axis=0)
        # the flood by the per-pass test of the validated rule, radar only (no anchor date): among
        # the monsoon passes dark enough for water, the largest drop below the pass's own earlier
        # level, with the number of passes within 14 days that also dropped >= 2 dB (either pol)
        hit = ((d_vh >= 2.0) | (d_vv >= 2.0)).astype("float32")
        near = (np.abs(day[:, None] - day[None, :]) <= np.timedelta64(14, "D")).astype("float32")
        support = near @ hit
        # water: the deep test (-19 dB, drop >= 4) or the shallow one of the validated rule (-18 dB,
        # drop >= 5, seen on >= 4 passes: region N's paddies are shallow and never read -19)
        deep_c = (vh <= FLOOD_VH_MAX) & (d_vh >= FLOOD_DROP_MIN)
        shallow_c = (vh <= rw.SHALLOW_VH_MAX) & (d_vh >= rw.SHALLOW_DROP_MIN) & (support >= rw.SHALLOW_SUPPORT_MIN)
        cand = monsoon[:, None] & (deep_c | shallow_c)
        m = np.where(cand, d_vh, -np.inf)
        i_fl = m.argmax(axis=0)
        has = np.isfinite(m.max(axis=0)) & (m.max(axis=0) > -np.inf)
        rows = np.arange(len(pix))
        summ["flood_drop"] = np.where(has, d_vh[i_fl, rows], 0.0)
        summ["flood_doy"] = np.where(has, doy[i_fl], 0.0)
        summ["flood_vh"] = np.where(has, vh[i_fl, rows], np.nan)
        summ["flood_vv_drop"] = np.where(has, d_vv[i_fl, rows], np.nan)
        summ["flood_support"] = np.where(has, support[i_fl, rows], 0.0)
        summ["flood_ok"] = has.astype("float32")
        summ["flood_shallow"] = np.where(has, (vh[i_fl, rows] > FLOOD_VH_MAX).astype("float32"), 0.0)
        summ["flood_vv"] = np.where(has, vv[i_fl, rows], np.nan)
        summ["VV_rise_from_flood"] = np.where(has, summ["VV_end"] - summ["flood_vv"], np.nan)
        summ["canopy_rise"] = np.where(has, summ["VH_end"] - summ["flood_vh"], np.nan)
        # any flooded stretch: longest run of consecutive monsoon passes at or below -19 dB
        dark = (np.where(monsoon[:, None], f_vh, 0) <= FLOOD_VH_MAX)
        run = np.zeros(len(pix), dtype="float32")
        best = np.zeros(len(pix), dtype="float32")
        for row in dark:
            run = np.where(row, run + 1, 0)
            best = np.maximum(best, run)
        summ["VH_dark_run"] = best
    for k, v in summ.items():
        cols.append(np.asarray(v, dtype="float32")[:, None])
        names.append(k)
    X = np.concatenate(cols, axis=1)
    bad = ~np.isfinite(X)
    X[bad] = 0.0
    return {"X": X, "names": names, "pixels": pix, "shape": s["shape"], "n_filled": int(bad.sum())}


def feature_path(aoi_id: int, out_dir=OUT) -> Path:
    return Path(out_dir) / f"aoi{aoi_id}_features.npz"


def build_features(aoi_ids, out_dir=OUT, overwrite: bool = False) -> pd.DataFrame:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for a in aoi_ids:
        p = feature_path(a, out_dir)
        if p.exists() and not overwrite:
            continue
        try:
            f = pixel_features(a)
        except Exception as exc:  # keep the batch going
            rows.append({"aoi": f"aoi{a}", "error": f"{type(exc).__name__}: {exc}"[:160]})
            print(f"aoi{a}: FAILED {type(exc).__name__}: {exc}"[:200])
            continue
        np.savez_compressed(p, X=f["X"], pixels=f["pixels"], shape=np.asarray(f["shape"]), names=np.asarray(f["names"]))
        rows.append({"aoi": f"aoi{a}", "pixels": len(f["pixels"]), "features": f["X"].shape[1], "filled": f["n_filled"]})
    return pd.DataFrame(rows)


def load_features(aoi_id: int, out_dir=OUT) -> dict:
    z = np.load(feature_path(aoi_id, out_dir), allow_pickle=False)
    return {"X": z["X"], "pixels": z["pixels"], "shape": tuple(int(v) for v in z["shape"]), "names": list(z["names"])}


def frame(f: dict) -> pd.DataFrame:
    return pd.DataFrame(f["X"], columns=f["names"])


# --------------------------------------------------------------------------- E1: radar rule

def radar_rule(F: pd.DataFrame) -> np.ndarray:
    """Classes of :data:`RULE_CLASSES` from the feature table, radar only.

    rice standing = a field (not permanent cover before the monsoon) + a transplanting flood
    (the per-pass test, deep or shallow) + a canopy at the end (VH back to >= -18 dB, >= 3 dB
    above the water) + no harvest fall at the end; young = the flood and an end between the
    water and the canopy level; flooded = still under water; harvested = canopy grown, then
    both polarisations fell >= 3 dB from the late maximum.
    """
    with np.errstate(invalid="ignore"):
        cover = F["VH_pre_min"].to_numpy() > COVER_VH_MIN
        flood = (F["flood_ok"].to_numpy() > 0) & (F["flood_support"].to_numpy() >= SUPPORT_MIN) & ~cover
        rise = F["canopy_rise"].to_numpy()
        end = F["VH_end"].to_numpy()
        canopy = flood & (end >= CANOPY_VH_MIN) & (rise >= CANOPY_RISE_MIN)
        harvested = canopy & (F["VH_end_fall"].to_numpy() >= HARVEST_FALL_MIN) & (F["VV_end_fall"].to_numpy() >= HARVEST_FALL_MIN)
        young = flood & ~canopy & (end > YOUNG_VH_MIN)
        flooded = flood & ~canopy & ~young
    out = np.zeros(len(F), dtype="uint8")
    out[flooded] = 7
    out[young] = 2
    out[canopy] = 1
    out[harvested] = 4
    return out


# --------------------------------------------------------------------------- labels and references

def teacher_classes(aoi_id: int, pixels: np.ndarray, map_suffix: str = "_final") -> np.ndarray:
    import rasterio

    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_{SEASON_KEY}{map_suffix}.tif") as ds:
        return ds.read(1).ravel()[pixels]


def reference_pixels(aoi_id: int, plots) -> pd.DataFrame:
    """The AOI's reference pixels (``validation.reference_sets``): plot interior / edge, negatives."""
    from . import validation as va

    return va.reference_sets(aoi_id, plots)


def training_table(aoi_ids, per_class: int = 3000, seed: int = 0, out_dir=OUT) -> pd.DataFrame:
    """A stratified pixel sample per AOI with the teacher's training class (:data:`TEACHER_TO_TRAIN`)."""
    rng = np.random.default_rng(seed)
    parts = []
    for a in aoi_ids:
        if not feature_path(a, out_dir).exists():
            continue
        f = load_features(a, out_dir)
        t = teacher_classes(a, f["pixels"])
        y = np.array([TEACHER_TO_TRAIN.get(int(c), None) for c in np.unique(t)])
        lut = {int(c): TEACHER_TO_TRAIN.get(int(c)) for c in np.unique(t)}
        ymap = np.array([lut[int(c)] if lut[int(c)] is not None else -1 for c in t])
        keep = []
        for k in TRAIN_CLASSES:
            idx = np.flatnonzero(ymap == k)
            if len(idx):
                keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
        if not keep:
            continue
        sel = np.concatenate(keep)
        d = frame({"X": f["X"][sel], "names": f["names"]})
        d["y"] = ymap[sel]
        d["teacher"] = t[sel]
        d["aoi"] = f"aoi{a}"
        d["pixel"] = f["pixels"][sel]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- E2: model + spatial CV

def make_model(seed: int = 0, n_jobs: int | None = None):
    import xgboost as xgb

    from .. import resources

    return xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, tree_method="hist",
                             n_jobs=n_jobs or resources.detect_resources().cpus, random_state=seed, objective="multi:softprob")


def folds_by(table: pd.DataFrame, by: str = "aoi", n_folds: int = 5, seed: int = 0):
    """Whole-AOI folds (``aoi``) or leave-one-region-out folds (``region``, plot AOIs only:
    the closest thing to a new, unseen area)."""
    from . import validation as va
    from .compare_runs import PLOT_AOIS

    rng = np.random.default_rng(seed)
    aois = np.array(sorted(table["aoi"].unique()))
    if by == "region":
        # the regions are the ones the surveyed plots lie in (names come from a local file)
        region = {a: va.REGION.get(a, a) for a in aois}
        named = sorted({region[a] for a in aois if int(a[3:]) in PLOT_AOIS})
        return [(r, [a for a in aois if region[a] == r]) for r in named]
    rng.shuffle(aois)
    return [(f"fold{k}", list(aois[k::n_folds])) for k in range(n_folds)]


def cross_validate(table: pd.DataFrame, by: str = "aoi", n_folds: int = 5, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Out-of-fold class and rice probability for every row of ``table``; returns (table with
    ``pred`` / ``p_rice``, feature importances averaged over folds)."""
    feats = [c for c in table.columns if c not in ("y", "teacher", "aoi", "pixel", "pixels", "field_id")]
    X, y = table[feats].to_numpy(dtype="float32"), table["y"].to_numpy()
    pred = np.full(len(table), -1)
    prob = np.full(len(table), np.nan, dtype="float32")
    imp = np.zeros(len(feats))
    folds = folds_by(table, by, n_folds, seed)
    for name, held in folds:
        test = table["aoi"].isin(held).to_numpy()
        if test.sum() == 0 or (~test).sum() == 0:
            continue
        m = make_model(seed)
        m.fit(X[~test], y[~test])
        p = m.predict_proba(X[test])
        pred[test] = m.classes_[p.argmax(axis=1)]
        prob[test] = p[:, list(m.classes_).index(1)] if 1 in m.classes_ else 0.0
        imp += m.feature_importances_
        print(f"  {name}: {len(held)} AOIs held out, {int(test.sum())} rows, agreement {np.mean(pred[test] == y[test]):.3f}")
    out = table.copy()
    out["pred"], out["p_rice"] = pred, prob
    return out, pd.Series(imp / max(len(folds), 1), index=feats).sort_values(ascending=False)


def agreement_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Confusion of teacher training class (rows) against the out-of-fold prediction (columns)."""
    t = pd.crosstab(oof["y"].map(TRAIN_CLASSES), oof["pred"].map(TRAIN_CLASSES), rownames=["teacher"], colnames=["model"])
    t["recall"] = np.round(100 * np.diag(t.reindex(columns=t.index, fill_value=0).to_numpy()) / t.sum(axis=1).to_numpy(), 1)
    return t


def score_reference(predict, aoi_ids, plots, out_dir=OUT, delivered=(1,)) -> pd.DataFrame:
    """Share (%) of every reference set called rice by ``predict(F) -> classes`` on the AOI's
    feature table: plot interior / edge per region and each negative set."""
    from . import validation as va

    rows = []
    for a in aoi_ids:
        if not feature_path(a, out_dir).exists():
            continue
        f = load_features(a, out_dir)
        refs = reference_pixels(a, plots)
        if refs.empty:
            continue
        cls = predict(frame(f), a)
        lookup = pd.Series(cls, index=f["pixels"])
        got = refs[refs["pixel"].isin(lookup.index)].copy()
        got["cls"] = lookup.loc[got["pixel"].to_numpy()].to_numpy()
        got["delivered"] = got["cls"].isin(delivered)
        for (s, region), g in got.groupby(["set", "region"]):
            rows.append({"aoi": f"aoi{a}", "set": s, "region": region, "pixels": len(g),
                         "delivered_pct": round(100 * g["delivered"].mean(), 2),
                         **{f"class_{k}_pct": round(100 * (g["cls"] == k).mean(), 1) for k in sorted(got["cls"].unique())}})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    summary = (t.assign(hit=t["pixels"] * t["delivered_pct"] / 100).groupby(["set", "region"])
               .agg(pixels=("pixels", "sum"), hit=("hit", "sum")).reset_index())
    summary["delivered_pct"] = np.round(100 * summary["hit"] / summary["pixels"], 1)
    return summary.drop(columns="hit")


def teacher_agreement(predict, aoi_ids, out_dir=OUT) -> pd.DataFrame:
    """Acres per (teacher class, predicted class) over whole AOIs: what a map of an unseen AOI
    would look like next to the validated map."""
    rows = []
    for a in aoi_ids:
        if not feature_path(a, out_dir).exists():
            continue
        f = load_features(a, out_dir)
        t = teacher_classes(a, f["pixels"])
        cls = predict(frame(f), a)
        ct = pd.crosstab(t, cls)
        for ti, row in ct.iterrows():
            for ci, n in row.items():
                if n:
                    rows.append({"aoi": f"aoi{a}", "teacher": int(ti), "pred": int(ci), "acres": mr.acres(int(n))})
    t = pd.DataFrame(rows)
    return t.pivot_table(index="teacher", columns="pred", values="acres", aggfunc="sum", fill_value=0.0).round(0)


# --------------------------------------------------------------------------- E3: field level

def field_table(aoi_id: int, out_dir=OUT, fields_dir=f"{SRC}/delivery/fields") -> pd.DataFrame:
    """One row per delivered field polygon: the mean of the pixel features over the field's
    pixels (speckle falls with the square root of the pixel count) and the teacher's field label
    (``label`` of the delivered fields file). Fields under 4 owned pixels are dropped."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    files = sorted(Path(fields_dir).glob(f"aoi{aoi_id}_fields_*.gpkg"))
    if not files or not feature_path(aoi_id, out_dir).exists():
        return pd.DataFrame()
    f = load_features(aoi_id, out_dir)
    g = gpd.read_file(files[0])
    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_{SEASON_KEY}_final.tif") as ds:
        g = g.to_crs(ds.crs)
        order = np.argsort(-g.geometry.area.to_numpy())
        idx = rasterize(((geom, int(i)) for i, geom in zip(order, g.geometry.to_numpy()[order])),
                        out_shape=ds.shape, transform=ds.transform, fill=-1, dtype="int32").ravel()
    owner = idx[f["pixels"]]
    ok = owner >= 0
    n = np.bincount(owner[ok], minlength=len(g))
    sums = np.zeros((len(g), f["X"].shape[1]), dtype="float64")
    np.add.at(sums, owner[ok], f["X"][ok])
    keep = n >= 4
    X = (sums[keep] / n[keep, None]).astype("float32")
    t = pd.DataFrame(X, columns=f["names"])
    t["pixels"] = n[keep]
    t["teacher"] = g["label"].to_numpy()[keep]
    t["y"] = [TEACHER_TO_TRAIN.get(int(c), -1) if TEACHER_TO_TRAIN.get(int(c)) is not None else -1 for c in t["teacher"]]
    t["field_id"] = g["field_id"].to_numpy()[keep]
    t["aoi"] = f"aoi{aoi_id}"
    return t


def field_training_table(aoi_ids, per_class: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for a in aoi_ids:
        t = field_table(a)
        if t.empty:
            continue
        keep = []
        for k in TRAIN_CLASSES:
            idx = np.flatnonzero(t["y"].to_numpy() == k)
            if len(idx):
                keep.append(rng.choice(idx, size=min(per_class, len(idx)), replace=False))
        if keep:
            parts.append(t.iloc[np.concatenate(keep)])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- maps

def write_map(aoi_id: int, classes: np.ndarray, pixels: np.ndarray, shape, path, prob: np.ndarray | None = None):
    import rasterio

    src = Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_{SEASON_KEY}_final.tif"
    with rasterio.open(src) as ds:
        profile = ds.profile.copy()
    full = np.full(shape[0] * shape[1], 255, dtype="uint8")
    full[pixels] = classes
    profile.update(dtype="uint8", nodata=255, count=1, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(full.reshape(shape), 1)
    if prob is not None:
        pfull = np.full(shape[0] * shape[1], 255, dtype="uint8")
        pfull[pixels] = np.clip(np.round(prob * 100), 0, 100).astype("uint8")
        with rasterio.open(str(path).replace(".tif", "_prob.tif"), "w", **profile) as dst:
            dst.write(pfull.reshape(shape), 1)


# --------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    import argparse

    from .compare_runs import PLOT_AOIS, load_plots

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.sar_only", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["features", "rule", "train", "map"])
    p.add_argument("--level", default="pixel", choices=["pixel", "field"], help="train: pixel features or field means (E3)")
    p.add_argument("--ids", nargs="*", type=int, default=None, help="default: every AOI with a final map")
    p.add_argument("--by", default="aoi", choices=["aoi", "region"])
    p.add_argument("--per-class", type=int, default=3000)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--model", default=f"{OUT}/model_xgb.json")
    p.add_argument("--tag", default="")
    args = p.parse_args(argv)
    ids = args.ids or sorted(int(q.parent.name[3:]) for q in Path(SRC).glob(f"aoi*/aoi*_{SEASON_KEY}_final.tif"))
    Path(OUT).mkdir(parents=True, exist_ok=True)
    if args.step == "features":
        print(build_features(ids).to_string(index=False))
        return 0
    plots = load_plots()
    plot_ids = [a for a in PLOT_AOIS if a in ids] or list(PLOT_AOIS)
    if args.step == "rule":
        pred = lambda F, a: radar_rule(F)
        print("== reference sets (radar rule; delivered = class 1)")
        print(score_reference(pred, plot_ids, plots).to_string(index=False))
        print("== agreement with the teacher map, acres (rows teacher, columns rule)")
        print(teacher_agreement(pred, ids).to_string())
        return 0
    if args.step == "train":
        table = field_training_table(ids, args.per_class) if args.level == "field" else training_table(ids, args.per_class)
        drop_cols = ("y", "teacher", "aoi", "pixel", "pixels", "field_id")
        print(f"training rows {len(table)} from {table['aoi'].nunique()} AOIs; classes {table['y'].value_counts().to_dict()}")
        oof, imp = cross_validate(table, args.by, args.folds)
        print("== out-of-fold agreement with the teacher (whole AOIs held out)")
        print(agreement_table(oof).to_string())
        print("== top features"); print(imp.head(25).round(4).to_string())
        # reference sets, each plot AOI scored by the fold that held it out
        folds = folds_by(table, args.by, args.folds)
        feats = [c for c in table.columns if c not in drop_cols]
        models = {}
        for name, held in folds:
            train = ~table["aoi"].isin(held).to_numpy()
            m = make_model()
            m.fit(table.loc[train, feats].to_numpy(dtype="float32"), table.loc[train, "y"].to_numpy())
            for a in held:
                models[a] = m
        def pred(F, a):
            m = models.get(f"aoi{a}")
            if m is None:
                return np.zeros(len(F), dtype="uint8")
            return m.classes_[m.predict_proba(F[feats].to_numpy(dtype="float32")).argmax(axis=1)].astype("uint8")
        if args.level == "pixel":
            print("== reference sets (model, held-out fold; delivered = class 1)")
            print(score_reference(pred, plot_ids, plots).to_string(index=False))
            print("== agreement with the teacher map on held-out AOIs, acres (rows teacher, columns model)")
            print(teacher_agreement(pred, ids).to_string())
        else:
            # field level: agreement in acres over the held-out fields themselves
            oof["acres"] = oof["pixels"] * 0.0247105
            print("== held-out fields, acres (rows teacher class, columns model)")
            print(oof.pivot_table(index="y", columns="pred", values="acres", aggfunc="sum", fill_value=0).round(0).to_string())
        full = make_model()
        full.fit(table[feats].to_numpy(dtype="float32"), table["y"].to_numpy())
        full.save_model(args.model)
        json.dump({"features": feats, "classes": [int(c) for c in full.classes_], "by": args.by, "rows": len(table)},
                  open(str(args.model).replace(".json", "_meta.json"), "w"))
        oof.to_parquet(Path(OUT) / f"oof_{args.by}{args.tag}.parquet")
        return 0
    if args.step == "map":
        import xgboost as xgb

        meta = json.load(open(str(args.model).replace(".json", "_meta.json")))
        m = xgb.XGBClassifier()
        m.load_model(args.model)
        for a in ids:
            f = load_features(a)
            pr_ = m.predict_proba(frame(f)[meta["features"]].to_numpy(dtype="float32"))
            cls = np.array(meta["classes"])[pr_.argmax(axis=1)].astype("uint8")
            write_map(a, cls, f["pixels"], f["shape"], Path(OUT) / f"aoi{a}_sar_only.tif", pr_[:, meta["classes"].index(1)])
            print(f"aoi{a}: rice {mr.acres(int((cls == 1).sum())):.0f} ac, young {mr.acres(int((cls == 2).sum())):.0f} ac")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
