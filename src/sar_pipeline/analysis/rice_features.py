"""Fresh start, step 3 (third try): learn WHAT makes a curve rice from the user-labelled plot groups, not the curves.

Why
---
User (30 Sep): "don't memorise the curves, learn from them: the pattern, the trend". Matching a field to the labelled
curves (``plot_clusters``) failed across regions because every region has its own calendar; allowing a time shift
recovered rice only by letting non-rice in. So each curve is described by calendar-free traits, and a small decision
tree learns from the user's labels (15 plot groups rice, 5 not) which traits separate rice. The tree's rules are
printed so the user can judge them.

Traits (per plot curve or per pixel; ``sowing`` = step 2, the last trough at bare level):

* ``trough``: NDVI at the sowing; ``low_days``: how long the curve stayed near it (within 2 x the wobble) before;
* ``rise``: highest NDVI after the sowing minus ``trough``; ``top``: that highest value;
* ``days_half`` / ``days_90``: days from the sowing to half / 90 % of the rise (the green-up speed);
* ``fall_after_top``: top minus the last value (ripening or loss); ``age``: days from the sowing to the series end;
* ``cycles``: troughs at bare level since March (one crop or two); ``year_low``: the lowest NDVI held for two windows
  since March (never bare = trees); ``mean_season``: mean NDVI since the sowing;
* ``vh_drop``: VH at the sowing (lowest within one revisit) below its median of the month before (water darkens);
  ``vh_rise``: VH at the end above VH at the sowing (the crop grows).

Use::

    python -m sar_pipeline.analysis.rice_features check          # leave one region out, and the learned rules
    python -m sar_pipeline.analysis.rice_features run --aoi 160  # the map of one AOI (steps 1 and 2 must exist)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd

CLUSTERS = "processed/_batch/s2_2026/rice_fresh/plot_clusters"
FRESH = "processed/_batch/s2_2026/rice_fresh"
STEP = 5
#: Only the curve from the sowing to the last date decides (user, 30 Sep: "our concern is only the curve that starts at
#: the sowing and runs to date, not what came before"). ``low_days``, ``cycles``, ``year_low`` and ``vh_drop`` look
#: before the sowing and are computed for reading only. ``mean_season`` (a level, not a shape) dropped 1 Oct: the tree
#: used "mean greenness above 0.59 -> not rice" and lost rice that greens early (aoi160 pixels 106113, 93201).
FEATURES = ["trough", "rise", "top", "days_half", "days_90", "fall_after_top", "age", "vh_rise"]
#: A small tree whose rules can be read: at most this deep, and every leaf holds at least this share of the plots.
#: Every example weighs the same (no "balanced" classes, 1 Oct): the few not-rice examples, each weighed up, turned
#: mixed leaves into "not rice" (aoi160 pixels 77194 / 76845, ~600 rice vs ~70 not-rice look-alikes); the AOIs are
#: mostly rice and the user leans to rice. Depth 4 (was 3) lets the tree tell those apart.
MAX_DEPTH = 4
MIN_LEAF_SHARE = 0.02


def traits(fit, vh, sow_idx, noise, k: float = 2.0, raw=None, vv=None) -> pd.DataFrame:
    """The traits of every curve. ``fit``/``vh``: (n, T) on the same windows; ``sow_idx``: window of the sowing (-1:
    none); ``noise``: each curve's wobble."""
    from .sowing_fresh import troughs
    from .vegetation_types import sustained_low

    f = np.asarray(fit, dtype=float)
    v = np.asarray(vh, dtype=float)
    n, t = f.shape
    sow = np.asarray(sow_idx, dtype=int)
    noise = np.nan_to_num(np.asarray(noise, dtype=float))
    cols = np.arange(n)
    ok = sow >= 0
    sc = np.clip(sow, 0, t - 1)
    step = np.arange(t)[None, :]
    trough = np.where(ok, f[cols, sc], np.nan)
    if raw is not None:
        # the smooth (upper-envelope) curve lifts a low: a clear view within one window of the sowing that is lower is
        # the field's real low (user, 1 Oct, aoi160 pixel 106113: curve 0.30, clear view 0.17 on the sowing day)
        r = np.asarray(raw, dtype=float)
        near = np.abs(np.arange(t)[None, :] - sc[:, None]) <= 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            seen_low = np.nanmin(np.where(near, r, np.nan), axis=1)
        trough = np.where(ok & np.isfinite(seen_low), np.fmin(trough, seen_low), trough)
    after = step >= sc[:, None]
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.nanmax(np.where(after, f, np.nan), axis=1)
        rise = top - trough
        # days to half and to 90 % of the rise
        def reach(share):
            hit = after & (f >= (trough + share * rise)[:, None])
            first = np.where(hit.any(axis=1), hit.argmax(axis=1), t)
            return np.where(ok, (first - sc) * STEP, np.nan)
        near = np.abs(f - trough[:, None]) <= (k * noise)[:, None]
        before = step <= sc[:, None]
        # consecutive windows near the trough up to the sowing
        low_days = np.zeros(n)
        run = np.ones(n, dtype=bool)
        for j in range(t):
            idx = sc - j
            inside = (idx >= 0) & ok & run
            still = inside & near[cols, np.clip(idx, 0, t - 1)] & before[cols, np.clip(idx, 0, t - 1)]
            low_days += still * STEP
            run &= still
        last = f[:, -1]
        fall = top - last
        # climb over the last three windows (15 days): a crop still greening when the series ends
        end_climb = last - f[:, max(t - 4, 0)]
        age = np.where(ok, (t - 1 - sc) * STEP, np.nan)
        level = sustained_low(f.T, pd.date_range("2000-01-01", periods=t, freq="5D"), start="2000-01-01")
        cycles = troughs(f.T, noise, level=level + k * noise).sum(axis=0)
        mean_season = np.nanmean(np.where(after, f, np.nan), axis=1)
        # radar: water at the sowing, crop after it
        win = (step >= (sc - 2)[:, None]) & (step <= (sc + 2)[:, None])
        vh_at = np.nanmin(np.where(win, v, np.nan), axis=1)
        month_before = (step >= (sc - 8)[:, None]) & (step <= (sc - 3)[:, None])
        vh_before = np.nanmedian(np.where(month_before, v, np.nan), axis=1)
        vh_end = np.nanmedian(v[:, -3:], axis=1)
        # radar green-up speed: days from the sowing until VH has made half of its rise to the end
        half = vh_at + 0.5 * (vh_end - vh_at)
        hit = after & (v >= half[:, None])
        first = np.where(hit.any(axis=1), hit.argmax(axis=1), t)
        vh_half_days = np.where(ok & (vh_end > vh_at), (first - sc) * STEP, np.nan)
        # the pixel's own radar scale: its pass-to-pass wobble and its dry-season level (the first two months of the
        # series), so the track's angle and the soil drop out (user, 1 Oct, choice "a")
        from .monsoon_rule import optical_noise

        vh_noise = optical_noise(v.T)
        dry = np.nanmedian(v[:, :12], axis=1)
    return pd.DataFrame({"trough": trough, "low_days": np.where(ok, low_days, np.nan), "rise": rise, "top": top,
                         "days_half": reach(0.5), "days_90": reach(0.9), "fall_after_top": fall, "age": age,
                         "end_climb": end_climb,
                         "cycles": cycles, "year_low": level, "mean_season": mean_season,
                         "vh_drop": vh_before - vh_at, "vh_rise": vh_end - vh_at, "vh_sow": vh_at, "vh_end": vh_end,
                         "vh_half_days": vh_half_days,
                         "vh_rise_z": (vh_end - vh_at) / np.where(vh_noise > 0, vh_noise, np.nan),
                         "vh_end_vs_dry": vh_end - dry, "vh_sow_vs_dry": vh_at - dry,
                         **_vv_traits(vv, sc, ok, step, t)})


def _vv_traits(vv, sc, ok, step, t) -> dict:
    """VV at the sowing (lowest within two windows), at the end, and its rise (user, 1 Oct: VV rises 10 dB in a
    transplanted paddy, aoi13 pixel 10120). NaN without VV."""
    n = len(sc)
    if vv is None:
        return {"vv_sow": np.full(n, np.nan), "vv_end": np.full(n, np.nan), "vv_rise": np.full(n, np.nan)}
    v = np.asarray(vv, dtype=float)
    win = (step >= (sc - 2)[:, None]) & (step <= (sc + 2)[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        at = np.where(ok, np.nanmin(np.where(win, v, np.nan), axis=1), np.nan)
        end = np.nanmedian(v[:, -3:], axis=1)
    return {"vv_sow": at, "vv_end": end, "vv_rise": end - at}


def plot_table(k: int = 20, fresh: str = FRESH, clusters: str = CLUSTERS) -> pd.DataFrame:
    """The plots with their group, the user's verdict (``is_rice``) and their traits."""
    both = pd.read_parquet(Path(clusters) / "plot_curves.parquet")
    info = both[["plot_id", "aoi", "region", "pixels"]].copy()
    fit = both.filter(like="ndvi_").to_numpy(dtype=float)
    vh = both.filter(like="vh_").to_numpy(dtype=float)
    dates = pd.to_datetime([c[5:] for c in both.filter(like="ndvi_").columns])
    groups = pd.read_csv(Path(clusters) / f"plot_groups_k{k}.csv")[["plot_id", "aoi", "group"]]
    verdicts = pd.read_csv(Path(clusters) / f"group_verdicts_k{k}.csv")
    px = pd.read_csv(Path(fresh) / "reference_shapes.csv", usecols=["aoi", "plot_id", "sowing_date", "noise"],
                     parse_dates=["sowing_date"])
    per_plot = px.groupby(["aoi", "plot_id"]).agg(sowing=("sowing_date", "median"), noise=("noise", "median")).reset_index()
    info = info.merge(per_plot, on=["aoi", "plot_id"], how="left").merge(groups, on=["plot_id", "aoi"], how="left")
    info = info.merge(verdicts, on="group", how="left")
    sow = np.where(info["sowing"].notna(), np.searchsorted(dates, pd.DatetimeIndex(info["sowing"])), -1)
    sow = np.where(sow >= len(dates), -1, sow)
    tr = traits(fit, vh, sow, info["noise"].fillna(0.03).to_numpy())
    out = pd.concat([info.reset_index(drop=True), tr], axis=1)
    out["is_rice"] = out["rice"]
    return out[out["group"].notna() & (out["group"] > 0) & out["is_rice"].notna()]


#: Final classes of the fresh-start step 3.
CLASSES = {0: "not vegetation in September", 1: "monsoon rice", 2: "not rice (learned traits)",
           5: "trees / orchards (step 1)", 6: "late rice (sown in August or later, harvest Nov-Dec; not delivered now)",
           7: "flooded, no crop yet", 8: "weak greenery (never near this AOI's rice canopy)"}
COLOURS = {0: "#d9d9d9", 1: "#1a9850", 2: "#e3a21a", 5: "#1b5e20", 6: "#9ecae1", 7: "#2166ac", 8: "#c2a5cf"}


def last_rise_start(fit, dates, noise, k: float = 2.0, level=None) -> pd.Series:
    """Per curve (rows of ``fit``): the date of its last trough (a low followed by a rise of ``k`` x its wobble), the
    start of the crop it carries at the end; NaT without one. No bare level and no radar are asked: only WHEN the current
    crop began to rise matters (user, 30 Sep: aoi160 groups 17-20 are rice planted in August for a Nov-Dec harvest).
    ``level`` (per curve, ``sowing_fresh.bare_level``): the trough must be at bare level; without it a dip of a standing
    crop (aoi160 group 10: 0.47 in August) counted as a new start."""
    from .sowing_fresh import last_true, troughs

    tr = troughs(np.asarray(fit, dtype=float).T, noise, k, level=level)
    last = last_true(tr)
    d = pd.DatetimeIndex(dates)
    return pd.Series(np.where(last >= 0, d[np.clip(last, 0, len(d) - 1)], pd.NaT)).astype("datetime64[ns]")


#: User decision (30 Sep): rice sown in August or later is the late crop (harvest November-December), not the monsoon
#: crop. The sowing is the start of the crop standing at the end (``last_rise_start``).
LATE_SOWN_FROM = "2026-08-01"
#: The weak-greenery rule (class 8). OFF (30 Sep): on aoi160 it took 26-46 % of groups 6, 12 and 17, which the user
#: labelled rice ("a canopy may stay at 0.5; the shape matters most").
WEAK_RULE = False
#: Young-crop rule (user, 1 Oct, aoi28 pixels 7181 / 7784): a crop still greening when the series ends has no grown
#: curve yet, and the tree learned on grown curves reads it as "not rice". It bypasses the tree: rice if sown before
#: ``LATE_SOWN_FROM``, late rice if on or after. ON.
YOUNG_RULE = True


def special_cases(tr: pd.DataFrame, pred, sowing, noise, k: float = 2.0, rise_start=None) -> np.ndarray:
    """Classes after the tree, with the user's simple special rules (30 Sep):

    * 6 late rice: a rice curve whose current crop began to rise (``rise_start``, :func:`last_rise_start`; else the
      step-2 sowing) on or after ``LATE_SOWN_FROM`` (1 August, user decision): a
      crop harvested in November-December, not the monsoon crop (aoi160 groups 17, 18, 19, 20); checked before the
      weak-greenery rule, since such a crop is still young;
    * young crop (``YOUNG_RULE``): a curve still climbing at the end and not past its top is rice (1) or, sown on or
      after ``LATE_SOWN_FROM``, late rice (6), whatever the tree says;
    * 7 flooded, no crop yet: the trough is open water (NDVI below 0, a physical boundary) and the curve has not risen
      by ``k`` x its wobble since (drowned fields, plot groups 15, 17, 18);
    * 8 weak greenery: a rice curve whose top stays more than ``k`` x sqrt(spread^2 + wobble^2) below the tops of this
      AOI's other rice curves (plot group 5: never a real canopy).
    Else 1 (tree says rice) or 2."""
    pred = np.asarray(pred, dtype=bool)
    noise = np.nan_to_num(np.asarray(noise, dtype=float))
    out = np.where(pred, 1, 2).astype("uint8")
    start = pd.to_datetime(pd.Series(sowing)).to_numpy() if rise_start is None else \
        pd.to_datetime(pd.Series(rise_start)).fillna(pd.Series(pd.to_datetime(pd.Series(sowing)))).to_numpy()
    # sown in August or later and greening since: late rice, whatever the tree says (it learned on grown curves and
    # reads a young late crop as "not rice"; user, 1 Oct, aoi160 pixels 66530 / 35726 / 35368)
    with np.errstate(invalid="ignore"):
        greening = tr["rise"].to_numpy(dtype=float) >= k * noise
    late = ~pd.isna(start) & (start >= np.datetime64(LATE_SOWN_FROM)) & greening
    out[late] = 6
    if YOUNG_RULE and "end_climb" in tr:
        # still greening at the end: risen by k x wobble since the sowing, still climbing k x wobble over the last
        # 15 days and the top is the last window (no fall yet)
        with np.errstate(invalid="ignore"):
            young = greening & (tr["end_climb"].to_numpy(dtype=float) >= k * noise) & \
                ~(tr["fall_after_top"].to_numpy(dtype=float) > k * noise) & ~pd.isna(start)
        out[young & ~late] = 1
    top = tr["top"].to_numpy(dtype=float)
    ref = top[pred & ~late & np.isfinite(top)]
    if WEAK_RULE and len(ref) >= 50:
        med = np.median(ref)
        spread = 1.4826 * np.median(np.abs(ref - med))
        with np.errstate(invalid="ignore"):
            weak = pred & ~late & (top < med - k * np.sqrt(spread ** 2 + noise ** 2))
        out[weak] = 8
    with np.errstate(invalid="ignore"):
        drowned = (tr["trough"].to_numpy(dtype=float) < 0) & ~(tr["rise"].to_numpy(dtype=float) >= k * noise)
    out[drowned] = 7
    return out


#: Radar-only traits (user, 1 Oct): pixels with no clear optical view in September are judged by the radar curve from
#: the sowing on (aoi13: no optical view after July, the radar rises from -24 to -16 dB).
#: Levels in dB. Tried 1 Oct (user's choice "a"): the same on the pixel's own scale (rise in own-wobble units, levels
#: against its own dry season), alone and together with dB: rice recall fell in most regions (aoi13 rice 89 -> 48-51 ac,
#: its rice groups 15-70 % instead of 46-81 %), so dB stays. The own-scale traits are still computed for reading.
RADAR_FEATURES = ["vh_sow", "vh_end", "vh_rise", "vh_half_days", "vv_sow", "vv_end", "vv_rise"]


def fit_tree(t: pd.DataFrame, features=None):
    """A small decision tree on the traits (missing values filled with the median of the training plots)."""
    from sklearn.tree import DecisionTreeClassifier

    features = FEATURES if features is None else features
    x = t[features]
    fill = x.median()
    tree = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=max(1, int(MIN_LEAF_SHARE * len(t))),
                                  class_weight=None, random_state=0)
    tree.fit(x.fillna(fill), t["is_rice"].astype(bool))
    return tree, fill


def check(t: pd.DataFrame, features=None) -> pd.DataFrame:
    """Leave one region out: the tree learns on the other regions' plots."""
    features = FEATURES if features is None else features
    rows = []
    for region in sorted(t["region"].unique()):
        train, test = t[t["region"] != region], t[t["region"] == region]
        if train["is_rice"].nunique() < 2 or len(test) < 5:
            continue
        tree, fill = fit_tree(train, features)
        pred = tree.predict(test[features].fillna(fill))
        for kind, flag in (("rice groups", True), ("not-rice groups", False)):
            m = test["is_rice"].astype(bool).to_numpy() == flag
            if m.sum():
                rows.append({"region": region, "plots": kind, "n": int(m.sum()),
                             "called rice %": round(100 * float(pred[m].mean()), 1)})
    return pd.DataFrame(rows)


def rules(tree, features=None) -> str:
    from sklearn.tree import export_text

    return export_text(tree, feature_names=FEATURES if features is None else features, decimals=2)


def pixel_vh(aoi_id: int, dates, pol: str = "VH") -> np.ndarray:
    """(pixels, T): per window the median over tracks of the pass nearest it (within 6 days), as for the plots."""
    from . import radar_water as rw

    wd = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    parts = []
    for d, flat in rw.read_series(aoi_id):
        dd = pd.DatetimeIndex(d).to_numpy().astype("datetime64[D]")
        j = np.abs(dd[:, None] - wd[None, :]).argmin(axis=0)
        near = np.abs(dd[j] - wd) <= np.timedelta64(6, "D")
        parts.append(np.where(near[:, None], np.asarray(flat[pol], dtype="float32")[j], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(np.stack(parts), axis=0).T


from .sowing_fresh import bare_level  # noqa: E402  (the AOI's bare level, step 2)


#: First window of the curves the traits read (the plot curves start here too).
CURVE_START = "2026-03-05"


def series_windows(windows) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """(indices, dates) of the AOI's own series windows from ``CURVE_START`` to its LAST window. Why (user, 1 Oct): the
    traits must read the newest image; taking the plot curves' dates cut aoi28 at 21 Sep and lost its 26 Sep view."""
    w = pd.DatetimeIndex(windows)
    idx = np.flatnonzero(w >= pd.Timestamp(CURVE_START))
    return idx, w[idx]


def pixel_traits(aoi_id: int, pixels, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
                 fresh: str = FRESH) -> pd.DataFrame:
    """Traits of the given pixels of one AOI, on its own series up to its newest window (sowing from step 2)."""
    import rasterio

    from .monsoon_rule import optical_noise

    pixels = np.asarray(pixels, dtype=int)
    with rasterio.open(Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step2_sowing.tif") as ds:
        doy = ds.read(1).ravel()[pixels]
    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    idx, dates = series_windows(w)
    fit = d["ndvi5d"].reshape(len(w), -1)[idx][:, pixels].T
    raw = d["ndvi5d_raw"].reshape(len(w), -1)[idx][:, pixels].T
    noise = optical_noise(d["ndvi5d_raw"].reshape(len(w), -1)[:, pixels])
    nd.forget()
    sow_day = pd.Timestamp("2026-01-01") + pd.to_timedelta(doy.astype(int) - 1, "D")
    sow = np.where(doy > 0, np.searchsorted(dates, sow_day), -1)
    sow = np.where(sow >= len(dates), len(dates) - 1, sow)
    return traits(fit, pixel_vh(aoi_id, dates)[pixels], sow, noise, raw=raw, vv=pixel_vh(aoi_id, dates, "VV")[pixels])


#: Pixels taken per labelled AOI group into the training set (a sample: the groups are large and alike inside).
PER_GROUP = 150


def aoi_labelled(k: int = 20, fresh: str = FRESH, seed: int = 0) -> pd.DataFrame:
    """Training rows from AOIs whose own pixel groups the user labelled (``aoi<N>_group_verdicts_k<k>.csv`` beside
    ``aoi<N>_groups_k<k>.tif``): ``PER_GROUP`` pixels per group, the AOI as its own region. The user judges a group by
    its MEDIAN curve (30 Sep), so single pixels of a group may differ from its label. Why (user, 30 Sep): all 20
    aoi160 groups are rice, while the tree learned on the plots alone called late-sown groups not rice."""
    import rasterio

    rows = []
    rng = np.random.default_rng(seed)
    for pfile in sorted(Path(fresh).glob("aoi*/aoi*_pixel_verdicts.csv")):
        # single pixels the user judged (user, 1 Oct: "treat its curve like a rice cluster median"): each counts as much
        # as one labelled group (repeated PER_GROUP times)
        aoi = pfile.parent.name
        pv = pd.read_csv(pfile)
        tr = pixel_traits(int(aoi[3:]), pv["pixel"].to_numpy())
        frame = pd.DataFrame({"plot_id": pv["pixel"], "aoi": aoi, "region": aoi, "group": -2, "rice": pv["rice"]})
        frame["is_rice"] = frame["rice"]
        one = pd.concat([frame.reset_index(drop=True), tr], axis=1)
        rows.append(one.loc[one.index.repeat(PER_GROUP)].reset_index(drop=True))
    for vfile in sorted(Path(fresh).glob(f"aoi*/aoi*_group_verdicts_k{k}.csv")):
        aoi = vfile.parent.name
        aoi_id = int(aoi[3:])
        verdicts = pd.read_csv(vfile)
        with rasterio.open(vfile.parent / f"{aoi}_groups_k{k}.tif") as ds:
            grp = ds.read(1).ravel()
        pick, gid = [], []
        for g in verdicts["group"]:
            idx = np.flatnonzero(grp == g)
            if len(idx):
                sel = rng.choice(idx, min(PER_GROUP, len(idx)), replace=False)
                pick.append(sel)
                gid.append(np.full(len(sel), g))
        if not pick:
            continue
        pix = np.concatenate(pick)
        tr = pixel_traits(aoi_id, pix)
        frame = pd.DataFrame({"plot_id": pix, "aoi": aoi, "region": aoi, "group": np.concatenate(gid)})
        frame = frame.merge(verdicts, on="group", how="left")
        frame["is_rice"] = frame["rice"]
        rows.append(pd.concat([frame.reset_index(drop=True), tr], axis=1))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _labels_key(k: int = 20) -> str:
    """A fingerprint of every label file (name, size, time): the cached training table is rebuilt when one changes."""
    files = sorted(Path(FRESH).glob("aoi*/aoi*_pixel_verdicts.csv")) + \
        sorted(Path(FRESH).glob(f"aoi*/aoi*_group_verdicts_k{k}.csv")) + \
        [Path(CLUSTERS) / f"group_verdicts_k{k}.csv", Path(CLUSTERS) / f"plot_groups_k{k}.csv"]
    return TRAITS_VERSION + "|" + "|".join(f"{f}:{f.stat().st_size}:{int(f.stat().st_mtime)}" for f in files if f.exists())


#: Bump when the traits change, so a cached training table is rebuilt (2: VV traits, 1 Oct).
TRAITS_VERSION = "4"


def training_table(k: int = 20) -> pd.DataFrame:
    """The labelled plots plus the labelled AOI pixel groups, cached in ``rice_fresh/training_table.parquet`` until a
    label file changes. Why (1 Oct): building it reads every labelled AOI's series (minutes); explaining eight pixels
    rebuilt it eight times."""
    cache = Path(FRESH) / "training_table.parquet"
    key_file = cache.with_suffix(".key")
    key = _labels_key(k)
    if cache.exists() and key_file.exists() and key_file.read_text() == key:
        return pd.read_parquet(cache)
    t = _training_table(k)
    t.to_parquet(cache)
    key_file.write_text(key)
    return t


def _training_table(k: int = 20) -> pd.DataFrame:
    extra = aoi_labelled(k)
    t = plot_table(k)
    t = pd.concat([t, extra], ignore_index=True) if len(extra) else t
    # groups the user left open (optical and radar disagree) teach nothing either way
    return t[t["is_rice"].notna()].assign(is_rice=lambda x: x["is_rice"].astype(bool))


def aoi_inputs(aoi_id: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late", fresh: str = FRESH) -> dict:
    """Everything step 3 reads for one AOI's crop ground (step 1 class 1), on the AOI's own series up to its newest
    window: ``fit`` / ``raw`` NDVI (pixels, T), ``vh`` / ``vv`` (pixels, T, dB), own optical ``noise``, ``sow`` (window
    index of the step-2 sowing, -1 none), ``sow_day``, ``dates``, ``crop`` (flat indices), ``cover``, ``profile`` and
    ``by_radar`` (no clear September view: the radar decides). Shared by the tree (``run``) and the similarity search
    (``curve_match``), so both judge exactly the same curves."""
    import rasterio

    from .monsoon_rule import optical_noise

    stem = Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step"
    with rasterio.open(f"{stem}1_cover.tif") as ds:
        cover = ds.read(1).ravel()
        profile = ds.profile
    with rasterio.open(f"{stem}2_sowing.tif") as ds:
        doy = ds.read(1).ravel()
    crop = np.flatnonzero(cover == 1)
    sow_source = np.ones(len(crop), dtype="uint8")
    if Path(f"{stem}2_source.tif").exists():
        with rasterio.open(f"{stem}2_source.tif") as ds:
            sow_source = ds.read(1).ravel()[crop]
    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    idx, dates = series_windows(w)
    fit = d["ndvi5d"].reshape(len(w), -1)[idx][:, crop].T
    raw = d["ndvi5d_raw"].reshape(len(w), -1)[idx][:, crop].T
    noise = optical_noise(d["ndvi5d_raw"].reshape(len(w), -1)[:, crop])
    nd.forget()
    sow_day = pd.Timestamp("2026-01-01") + pd.to_timedelta(doy[crop].astype(int) - 1, "D")
    sow = np.where(doy[crop] > 0, np.searchsorted(dates, sow_day), -1)
    sow = np.where(sow >= len(dates), len(dates) - 1, sow)
    src_path = Path(fresh).parent / "first_clear" / f"aoi{aoi_id}" / \
        f"aoi{aoi_id}_first_clear_2026-09-01_2026-09-30_vegetation_source.tif"
    by_radar = np.zeros(len(crop), dtype=bool)
    if src_path.exists():
        with rasterio.open(src_path) as ds:
            by_radar = np.isin(ds.read(1).ravel()[crop], (2, 3))
    return {"fit": fit, "raw": raw, "vh": pixel_vh(aoi_id, dates)[crop], "vv": pixel_vh(aoi_id, dates, "VV")[crop],
            "noise": noise, "sow": sow, "sow_day": pd.Series(sow_day).where(doy[crop] > 0), "dates": dates,
            "crop": crop, "cover": cover, "profile": profile, "by_radar": by_radar, "stem": stem,
            "sow_source": sow_source}


def finish(aoi_id: int, inp: dict, tr: pd.DataFrame, pred, tag: str = "3t", radar_pred=None,
           names: dict | None = None) -> pd.DataFrame:
    """The special rules on top of a rice / not-rice decision ``pred`` (and ``radar_pred`` for the radar-only pixels;
    an int array may carry other classes, e.g. 3 = unlike any pattern), then ``aoi<N>_step<tag>_class.tif`` + ``.qml``,
    ``_step<tag>_basis.tif`` (1 optical, 2 radar only) and the acre table ``_step<tag>.csv``."""
    import rasterio

    from .optical_phenology import acres

    crop, cover, profile, by_radar, fit = inp["crop"], inp["cover"], inp["profile"], inp["by_radar"], inp["fit"]
    noise, dates, stem = inp["noise"], inp["dates"], inp["stem"]
    pred = np.asarray(pred)
    if radar_pred is not None and by_radar.any():
        pred = np.where(by_radar, radar_pred, pred)
    extra = (pred != 0) & (pred != 1)                 # classes the decision gave itself (kept unless a rule fires)
    rise_start = last_rise_start(fit, dates, noise, level=bare_level(fit.T, noise, np.ones(len(crop), bool)))
    # where step 2 read the sowing from the radar the optical trough is a filled curve, not a start (aoi39 pixel 12617:
    # a trough on 7 Aug made it "late rice"; the radar says mid-July)
    radar_sown = np.asarray(inp.get("sow_source", np.ones(len(crop)))) == 2
    rise_start = rise_start.where(~radar_sown, pd.to_datetime(inp["sow_day"]).reset_index(drop=True))
    cls = special_cases(tr, pred == 1, inp["sow_day"], noise, rise_start=rise_start)
    keep = extra & (cls == 2)
    cls = np.where(keep, pred, cls)
    # "flooded, no crop" reads the optical rise, which is a held flat line where the optical saw nothing
    cls = np.where(by_radar & (cls == 7) & (pred == 1), 1, cls).astype("uint8")
    out = np.where(cover == 255, 255, np.where(cover == 2, 5, 0)).astype("uint8")
    out[crop] = cls
    basis = np.zeros(cover.size, dtype="uint8")
    basis[crop] = np.where(by_radar, 2, 1)
    with rasterio.open(f"{stem}{tag}_basis.tif", "w", **dict(profile, dtype="uint8", nodata=0, count=1)) as ds:
        # 1 = decided on the optical curve, 2 = radar only (no clear September view): lower confidence (user, 1 Oct)
        ds.write(basis.reshape((profile["height"], profile["width"]))[None])
    shape = (profile["height"], profile["width"])
    with rasterio.open(f"{stem}{tag}_class.tif", "w", **dict(profile, dtype="uint8", nodata=255, count=1)) as ds:
        ds.write(out.reshape(shape)[None])
    names = CLASSES if names is None else names
    colours = {c: col for c, col in {**COLOURS, 3: "#d73027"}.items() if c in names}
    entries = "\n".join(f'        <paletteEntry value="{c}" color="{col}" alpha="255" label="{c} {names[c]}"/>'
                        for c, col in colours.items())
    Path(f"{stem}{tag}_class.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
""")
    res = pd.DataFrame([{"class": c, "name": names[c], "acres": round(acres(int((out == c).sum())), 1)} for c in colours])
    res.to_csv(f"{stem}{tag}.csv", index=False)
    return res


def run(aoi_id: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late", fresh: str = FRESH) -> pd.DataFrame:
    """The tree (learned on all labelled plots) applied to the crop ground of one AOI; writes
    ``aoi<N>_step3t_class.tif`` (0 not vegetation, 1 rice, 2 not rice, 5 trees) + ``.qml`` and the acre table."""
    t = training_table()
    tree, fill = fit_tree(t)
    rtree, rfill = fit_tree(t, RADAR_FEATURES)
    inp = aoi_inputs(aoi_id, series_root, fresh)
    tr = traits(inp["fit"], inp["vh"], inp["sow"], inp["noise"], raw=inp["raw"], vv=inp["vv"])
    pred = tree.predict(tr[FEATURES].fillna(fill)).astype(int)
    # no clear optical view in September: the radar curve decides (user, 1 Oct)
    rpred = rtree.predict(tr[RADAR_FEATURES].fillna(rfill)).astype(int) if inp["by_radar"].any() else None
    return finish(aoi_id, inp, tr, pred, "3t", rpred)


def explain(aoi_id: int, pid: int) -> dict:
    """Why one pixel got its step-3 class: its traits, the tree's path (each question and the pixel's answer) and the
    special rules. Why (user, 1 Oct, aoi160 pixel 106113: "looks like rice, why not?")."""
    import rasterio

    t = training_table()
    tree, fill = fit_tree(t)
    tr = pixel_traits(aoi_id, [pid])
    x = tr[FEATURES].fillna(fill)
    node_ids = tree.decision_path(x).indices
    steps = []
    for n in node_ids:
        f = tree.tree_.feature[n]
        if f < 0:
            continue
        name, thr = FEATURES[f], tree.tree_.threshold[n]
        val = float(x.iloc[0][name])
        steps.append(f"{name} = {val:.2f} {'<=' if val <= thr else '>'} {thr:.2f}")
    with rasterio.open(Path(FRESH) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step3t_class.tif") as ds:
        cls = int(ds.read(1).ravel()[pid])
    with rasterio.open(Path(FRESH) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step2_sowing.tif") as ds:
        doy = int(ds.read(1).ravel()[pid])
    return {"class": CLASSES.get(cls, cls), "sowing": str((pd.Timestamp("2026-01-01") + pd.Timedelta(days=doy - 1)).date())
            if doy else None, "tree_says_rice": bool(tree.predict(x)[0]), "path": steps,
            "traits": {k: round(float(v), 2) for k, v in tr.iloc[0].items() if pd.notna(v)}}


#: What a user label expects of a step-3 map class (late rice is rice; 7 flooded and 0 / 5 are not rice).
_MAP_KIND = {0: "not rice", 1: "rice", 2: "not rice", 5: "not rice", 6: "late rice", 7: "not rice", 8: "not rice"}


def label_audit(fresh: str = FRESH, k: int = 20, tag: str = "3t") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every label the user gave against the current step-3 maps (user, 1 Oct: "fix the labels of every pixel I told
    you and show where the groups conflict"). Returns:

    * pixels: aoi, pixel, user label, map class, agrees (late rice vs monsoon rice counts as a near-miss), the pixel's
      AOI group and that group's label, and whether the pixel's label contradicts its group's label;
    * groups: per labelled AOI group, its label, acres and the share of its pixels per map class, and the share that
      agrees with the label."""
    import rasterio

    from .optical_phenology import acres

    def read(path):
        with rasterio.open(path) as ds:
            return ds.read(1).ravel()

    prow, grow = [], []
    for adir in sorted(Path(fresh).glob("aoi*")):
        aoi = adir.name
        mpath = adir / f"{aoi}_step{tag}_class.tif"
        if not mpath.exists():
            continue
        cls = read(mpath)
        gpath = adir / f"{aoi}_groups_k{k}.tif"
        grp = read(gpath) if gpath.exists() else None
        vpath = adir / f"{aoi}_group_verdicts_k{k}.csv"
        gv = pd.read_csv(vpath) if vpath.exists() else pd.DataFrame(columns=["group", "rice"])
        glabel = dict(zip(gv["group"], gv["rice"]))
        ppath = adir / f"{aoi}_pixel_verdicts.csv"
        if ppath.exists():
            for r in pd.read_csv(ppath).itertuples():
                c = int(cls[r.pixel])
                got = _MAP_KIND.get(c, "?")
                want = r.label
                agrees = "yes" if got == want else ("near (rice vs late rice)" if {got, want} == {"rice", "late rice"}
                                                    else "NO")
                g = int(grp[r.pixel]) if grp is not None else 0
                gl = glabel.get(g)
                prow.append({"aoi": aoi, "pixel": r.pixel, "user label": want, "map class": f"{c} {CLASSES.get(c, c)}",
                             "agrees": agrees, "group": g or "-",
                             "group label": "-" if gl is None else ("rice" if gl else "not rice"),
                             "pixel vs group": "-" if gl is None else
                             ("CONFLICT" if bool(gl) != (want != "not rice") else "same")})
        if grp is None:
            continue
        for g, rice in glabel.items():
            m = grp == g
            if not m.any():
                continue
            n = m.sum()
            share = {c: 100 * float((cls[m] == c).sum()) / n for c in (1, 6, 2, 7, 5, 0)}
            ok = share[1] + share[6] if rice else 100 - share[1] - share[6]
            grow.append({"aoi": aoi, "group": int(g), "user label": "rice" if rice else "not rice",
                         "acres": round(acres(int(n)), 1), "rice %": round(share[1], 0), "late %": round(share[6], 0),
                         "not rice %": round(share[2] + share[7], 0), "trees %": round(share[5], 0),
                         "no veg %": round(share[0], 0), "agrees %": round(ok, 0)})
    return pd.DataFrame(prow), pd.DataFrame(grow)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["check", "run", "audit"])
    p.add_argument("--aoi", type=int)
    p.add_argument("--explain", type=int, nargs="+", metavar="PIXEL_ID", help="with --aoi: why these pixels got their class")
    args = p.parse_args(argv)
    import os

    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")      # one AOI uses every core while reading (user, 1 Oct)
    if args.step == "audit":
        pix, grp = label_audit()
        pd.set_option("display.width", 250)
        print(pix.to_string(index=False))
        print(grp.to_string(index=False))
        return 0
    if args.explain is not None:
        for pid in args.explain:
            r = explain(args.aoi, pid)
            print(f"== pixel {pid}: " + ", ".join(f"{k}: {v}" for k, v in r.items() if k in ("class", "sowing",
                                                                                         "tree_says_rice", "path")))
        return 0
    if args.step == "check":
        t = training_table()
        c = check(t)
        c.to_csv(Path(FRESH) / "traits_check.csv", index=False)
        print(c.to_string(index=False))
        tree, _ = fit_tree(t)
        print("\nRules learned from all labelled plots:\n" + rules(tree))
        print(t.groupby("is_rice")[FEATURES + RADAR_FEATURES].median().round(2).T.to_string())
        r = check(t, RADAR_FEATURES)
        r.to_csv(Path(FRESH) / "traits_check_radar.csv", index=False)
        print("\nRadar-only tree (for pixels without a clear September view), leave one region out:\n" + r.to_string(index=False))
        rtree, _ = fit_tree(t, RADAR_FEATURES)
        print(rules(rtree, RADAR_FEATURES))
    else:
        print(run(args.aoi).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
