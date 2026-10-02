"""Fresh start, step 3 (similarity search): is the crop rice? Find the labelled pattern its curve resembles most.

Why
---
User (1 Oct 2026): "go with some robust similarity search approach". The decision tree (``rice_features``) learned from
10 rice examples per not-rice one and called almost everything with a canopy above 0.59 rice, so the groups the user
labelled "not rice" (aoi39 groups 1, 7, 8, 17, 18, 20; aoi13 groups 1, 2, 14, 15, 19, 20) came out rice. Two earlier
similarity tries failed for clear reasons, which this module avoids:

* matching on calendar dates (``plot_clusters.check_patterns``) missed a crop sown two weeks later than its pattern;
* matching on a 0-1 scale with only a rice reference (``rice_shape``) lost the NDVI level and called any crop that rose
  like rice "rice".

How
---
1. **Aligned at the sowing.** Every curve is read from its own sowing (step 2, ``sowing_fresh``) in 5-day steps up to
   ``MAX_AGE_DAYS``: NDVI as it is (the level is kept: a weak canopy is not a strong one), radar VH and VV as the change
   since the sowing in dB (the track's look angle and the soil's own brightness drop out). Only the curve from the
   sowing to the newest image counts (user, 30 Sep).
2. **Patterns.** Every group the user labelled by its median curve (the surveyed plots' 20 groups, each AOI's own 20
   groups, single pixels the user judged) and every group of alike pixels of a map the user accepted (``MAP_AOIS``,
   :func:`map_curves`) becomes one pattern: per step, the median of its members' aligned curves and
   their spread (1.4826 x MAD). Rice AND not-rice patterns, each counting once whatever its size: the 10-to-1 imbalance
   of the training pixels no longer tilts the answer.
3. **Growth speed.** Each pattern is also stretched in time (``STRETCH``, 0.8x-1.25x): a crop that greens a little
   faster or slower still matches its pattern; no fixed number of days.
4. **Distance.** Per step, the gap between the pixel and the pattern in units of sqrt(pattern spread^2 + the pixel's
   own wobble^2) (its own noise: own-scale, no hard-coded threshold), squared and capped at (2k)^2 so one hazy window
   cannot decide alone; the mean over the steps the pixel has. NDVI and radar steps count alike; pixels with no clear
   September view (decided by the radar in the vegetation step) are matched on VH and VV only.
5. **Decision.** Rice when any rice pattern is within k (a near tie with a not-rice pattern counts as rice: the AOIs
   are mostly rice, user 1 Oct); else the label of the nearest pattern. Farther than k (``WATER_K`` = 2) from EVERY pattern:
   class 3 "unlike any labelled pattern", shown to the user instead of being guessed.
6. The user's special rules then apply as with the tree (``rice_features.special_cases``: late rice sown from 1 August,
   a crop still greening at the end, flooded fields with no crop).

Check: leave one region out (``check``): the curves of one region are matched only with patterns from the other regions.

Use::

    python -m sar_pipeline.analysis.curve_match check            # leave-one-region-out on all labelled curves
    python -m sar_pipeline.analysis.curve_match run --aoi 39     # writes aoi39_step3s_class.tif / .qml / .csv
    python -m sar_pipeline.analysis.curve_match run --aoi 39 --explain 7636   # the nearest patterns of one pixel
    python -m sar_pipeline.analysis.curve_match run --aoi 39 --learn-from aoi160 aoi13 aoi28   # patterns of 3 AOIs
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import rice_features as rf

FRESH = rf.FRESH
STEP = 5
#: A crop older than 5 months is not rice (user, 30 Sep): the comparison never reads further than this.
MAX_AGE_DAYS = 150
STEPS = MAX_AGE_DAYS // STEP + 1
#: Time stretches tried per pattern (growth faster / slower than the pattern).
STRETCH = (0.8, 0.9, 1.0, 1.1, 1.25)
#: A pattern step needs this many members, else it has no median (a median of two curves is not a pattern).
MIN_MEMBERS = 5
#: Fewest steps a pixel must share with a pattern (15 days): a crop seen for less has no shape yet.
MIN_STEPS = 3
CLASSES = {**rf.CLASSES, 2: "not rice (nearest labelled pattern)", 3: "unlike any labelled pattern"}
CHANNELS = ("ndvi", "vh", "vv")


def aligned(curve, sow_idx, steps: int = STEPS, relative: bool = False) -> np.ndarray:
    """(n, steps): row i = ``curve[i]`` from its window ``sow_idx[i]`` on (NaN past the series end or without a sowing).
    ``relative``: minus the value at the sowing (radar: the change since the sowing; the sowing value is the lowest
    pass within one window, as the flood or bare soil is short)."""
    c = np.asarray(curve, dtype="float32")
    n, t = c.shape
    sow = np.asarray(sow_idx, dtype=int)
    j = sow[:, None] + np.arange(steps)[None, :]
    ok = (sow[:, None] >= 0) & (j < t)
    out = np.where(ok, c[np.arange(n)[:, None], np.clip(j, 0, t - 1)], np.nan)
    if relative:
        near = np.abs(np.arange(t)[None, :] - sow[:, None]) <= 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            base = np.nanmin(np.where(near, c, np.nan), axis=1)
        out = out - base[:, None]
    return out


def channels(fit, vh, vv, sow_idx, by_radar=None) -> dict:
    """The aligned NDVI, VH and VV curves of a set of pixels; NDVI is left out (NaN) where the radar decides."""
    out = {"ndvi": aligned(fit, sow_idx), "vh": aligned(vh, sow_idx, relative=True),
           "vv": aligned(vv, sow_idx, relative=True) if vv is not None else np.full((len(sow_idx), STEPS), np.nan)}
    if by_radar is not None:
        out["ndvi"][np.asarray(by_radar, dtype=bool)] = np.nan
    return out


def own_noise(fit_raw, vh, vv) -> dict:
    """Per pixel, its own wobble per channel (``monsoon_rule.optical_noise``: robust second difference)."""
    from .monsoon_rule import optical_noise

    return {"ndvi": optical_noise(np.asarray(fit_raw, dtype=float).T), "vh": optical_noise(np.asarray(vh, dtype=float).T),
            "vv": optical_noise(np.asarray(vv, dtype=float).T) if vv is not None else np.full(len(fit_raw), np.nan)}


# --------------------------------------------------------------------------------------------- labelled curves
def _pixel_rows(aoi_id: int, pixels, series_root: str, fresh: str) -> tuple[dict, dict]:
    """Aligned channels and own noise of given pixels of a labelled AOI (its own series to its newest window)."""
    import rasterio

    from . import ndvi_5day as nd

    pixels = np.asarray(pixels, dtype=int)
    with rasterio.open(Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step2_sowing.tif") as ds:
        doy = ds.read(1).ravel()[pixels]
    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    idx, dates = rf.series_windows(w)
    fit = d["ndvi5d"].reshape(len(w), -1)[idx][:, pixels].T
    raw = d["ndvi5d_raw"].reshape(len(w), -1)[:, pixels].T
    nd.forget()
    sow_day = pd.Timestamp("2026-01-01") + pd.to_timedelta(doy.astype(int) - 1, "D")
    sow = np.where(doy > 0, np.searchsorted(dates, sow_day), -1)
    sow = np.where(sow >= len(dates), len(dates) - 1, sow)
    vh, vv = rf.pixel_vh(aoi_id, dates)[pixels], rf.pixel_vh(aoi_id, dates, "VV")[pixels]
    src = Path(fresh).parent / "first_clear" / f"aoi{aoi_id}" / \
        f"aoi{aoi_id}_first_clear_2026-09-01_2026-09-30_vegetation_source.tif"
    by_radar = None
    if src.exists():
        with rasterio.open(src) as ds:
            by_radar = np.isin(ds.read(1).ravel()[pixels], (2, 3))
    return channels(fit, vh, vv, sow, by_radar), own_noise(raw, vh, vv)


def labelled_curves(k: int = 20, fresh: str = FRESH, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
                    seed: int = 0) -> pd.DataFrame:
    """Every labelled curve, aligned at its sowing: the plots (pattern = plot group), ``rf.PER_GROUP`` pixels of each
    labelled AOI group (pattern = AOI + group) and the single pixels the user judged (pattern = the pixel). Columns:
    ``pattern``, ``region``, ``is_rice``, ``ndvi_<s>`` / ``vh_<s>`` / ``vv_<s>`` and ``noise_<channel>``."""
    import rasterio

    frames = []
    # plots: their curves end where the plot series ends (21 Sep); no VV was read for them
    both = pd.read_parquet(Path(rf.CLUSTERS) / "plot_curves.parquet")
    t = rf.plot_table(k)
    both = both.merge(t[["plot_id", "aoi", "group", "is_rice", "sowing", "noise"]], on=["plot_id", "aoi"])
    dates = pd.to_datetime([c[5:] for c in both.filter(like="ndvi_").columns])
    sow = np.where(both["sowing"].notna(), np.searchsorted(dates, pd.DatetimeIndex(both["sowing"])), -1)
    sow = np.where(sow >= len(dates), -1, sow)
    vh = both.filter(like="vh_").to_numpy(dtype=float)
    ch = channels(both.filter(like="ndvi_").to_numpy(dtype=float), vh, None, sow)
    from .monsoon_rule import optical_noise

    noise = {"ndvi": both["noise"].fillna(0.03).to_numpy(), "vh": optical_noise(vh.T), "vv": np.full(len(both), np.nan)}
    frames.append(_frame(ch, noise, pattern="plots:g" + both["group"].astype(int).astype(str),
                         region=both["region"], is_rice=both["is_rice"]))
    rng = np.random.default_rng(seed)
    for vfile in sorted(Path(fresh).glob(f"aoi*/aoi*_group_verdicts_k{k}.csv")):
        aoi = vfile.parent.name
        if aoi in MAP_AOIS:
            # the accepted map replaces the group labels (user, 1 Oct: aoi13 groups 1, 2, 14, 15, 19, 20 were labelled
            # "not rice" but the accepted map shows them rice; the map wins)
            continue
        verdicts = pd.read_csv(vfile)
        verdicts = verdicts[verdicts["rice"].notna()]
        with rasterio.open(vfile.parent / f"{aoi}_groups_k{k}.tif") as ds:
            grp = ds.read(1).ravel()
        pick, gid = [], []
        for g in verdicts["group"]:
            idx = np.flatnonzero(grp == g)
            if len(idx):
                sel = rng.choice(idx, min(rf.PER_GROUP, len(idx)), replace=False)
                pick.append(sel)
                gid.append(np.full(len(sel), g))
        if not pick:
            continue
        pix = np.concatenate(pick)
        ch, noise = _pixel_rows(int(aoi[3:]), pix, series_root, fresh)
        g = np.concatenate(gid)
        rice = pd.Series(g).map(dict(zip(verdicts["group"], verdicts["rice"].astype(bool))))
        frames.append(_frame(ch, noise, pattern=pd.Series([f"{aoi}:g{x}" for x in g]), region=aoi, is_rice=rice,
                             pixel=pix))
    for pfile in sorted(Path(fresh).glob("aoi*/aoi*_pixel_verdicts.csv")):
        aoi = pfile.parent.name
        pv = pd.read_csv(pfile)
        ch, noise = _pixel_rows(int(aoi[3:]), pv["pixel"].to_numpy(), series_root, fresh)
        frames.append(_frame(ch, noise, pattern=aoi + ":p" + pv["pixel"].astype(str), region=aoi,
                             is_rice=pv["rice"].astype(bool), pixel=pv["pixel"].to_numpy()))
    for aoi in MAP_AOIS:
        frames.append(map_curves(aoi, fresh=fresh, series_root=series_root, seed=seed))
    return pd.concat(frames, ignore_index=True)


#: AOIs whose last step-3 map the user checked pixel by pixel and accepted (user, 1 Oct: "aoi160, 13, 28 ... the last
#: classified map I looked at myself, it was right; add those to the median curves database").
MAP_AOIS = ("aoi160", "aoi13", "aoi28")
#: Map classes read as rice / not rice (late rice is rice; "flooded, no crop yet" is not).
MAP_RICE, MAP_NOT_RICE = (1, 6), (2, 7)
#: Pixels sampled per map class and AOI, and their pixels per pattern (one pattern per this many, at most 20).
MAP_SAMPLE, MAP_PER_PATTERN = 4000, 50


def map_curves(aoi: str, fresh: str = FRESH, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
               seed: int = 0, tag: str = "3t") -> pd.DataFrame:
    """Labelled curves from an accepted map (``aoi<N>_step<tag>_class.tif``): up to ``MAP_SAMPLE`` crop pixels per
    class (rice, not rice), split into patterns of alike curves (KMeans on the aligned curves, about
    ``MAP_PER_PATTERN`` pixels each, at most 20), separately for pixels with and without an optical curve. Each pattern
    becomes one median curve, like a group the user labelled. Why: the user checked these maps pixel by pixel (1 Oct);
    a class split into alike curves keeps a May crop and an August crop apart instead of one blurred median."""
    import rasterio
    from sklearn.cluster import KMeans

    with rasterio.open(Path(fresh) / aoi / f"{aoi}_step{tag}_class.tif") as ds:
        cls = ds.read(1).ravel()
    rng = np.random.default_rng(seed)
    out = []
    for name, codes, rice in (("rice", MAP_RICE, True), ("notrice", MAP_NOT_RICE, False)):
        idx = np.flatnonzero(np.isin(cls, codes))
        if len(idx) < MIN_MEMBERS:
            continue
        idx = np.sort(rng.choice(idx, min(MAP_SAMPLE, len(idx)), replace=False))
        ch, noise = _pixel_rows(int(aoi[3:]), idx, series_root, fresh)
        optical = np.isfinite(ch["ndvi"]).sum(axis=1) >= MIN_STEPS
        labels = np.full(len(idx), "", dtype=object)
        for kind, sel in (("opt", optical), ("radar", ~optical)):
            if sel.sum() < MIN_MEMBERS:
                continue
            use = ("ndvi",) if kind == "opt" else ("vh", "vv")
            x = np.hstack([_hold_last(ch[c][sel]) for c in use])
            x = np.nan_to_num(x)
            n_pat = int(np.clip(sel.sum() // MAP_PER_PATTERN, 1, 20))
            km = KMeans(n_pat, n_init=4, random_state=seed).fit(x)
            labels[sel] = [f"{aoi}:map-{name}-{kind}{g + 1}" for g in km.labels_]
        keep = labels != ""
        out.append(_frame({c: v[keep] for c, v in ch.items()}, {c: v[keep] for c, v in noise.items()},
                          pattern=labels[keep], region=aoi, is_rice=np.full(keep.sum(), rice), pixel=idx[keep]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _hold_last(x) -> np.ndarray:
    """Each row's NaN tail (the series ended) filled with its last value, so curves of different ages can be grouped."""
    x = np.array(x, dtype="float32")
    for j in range(1, x.shape[1]):
        gap = np.isnan(x[:, j])
        x[gap, j] = x[gap, j - 1]
    return x


def _frame(ch, noise, pattern, region, is_rice, pixel=None) -> pd.DataFrame:
    n = len(ch["ndvi"])
    cols = {"pattern": np.asarray(pattern), "region": np.broadcast_to(np.asarray(region), (n,)),
            "is_rice": np.asarray(is_rice, dtype=bool), "pixel": -1 if pixel is None else pixel}
    for c in CHANNELS:
        for s in range(STEPS):
            cols[f"{c}_{s}"] = ch[c][:, s]
        cols[f"noise_{c}"] = noise[c]
    return pd.DataFrame(cols)


#: Bump when the alignment changes, so the cached labelled curves are rebuilt.
CURVES_VERSION = "3"


def labelled_table(k: int = 20) -> pd.DataFrame:
    """``labelled_curves`` cached in ``rice_fresh/labelled_curves.parquet`` until a label file changes (same fingerprint
    as the tree's training table)."""
    cache = Path(FRESH) / "labelled_curves.parquet"
    key_file = cache.with_suffix(".key")
    maps = [Path(FRESH) / a / f"{a}_step3t_class.tif" for a in MAP_AOIS]
    key = CURVES_VERSION + "|" + rf._labels_key(k) + "|" + "|".join(
        f"{m}:{int(m.stat().st_mtime)}" for m in maps if m.exists())
    if cache.exists() and key_file.exists() and key_file.read_text() == key:
        return pd.read_parquet(cache)
    t = labelled_curves(k)
    t.to_parquet(cache)
    key_file.write_text(key)
    return t


def _get(t: pd.DataFrame, c: str) -> np.ndarray:
    return t[[f"{c}_{s}" for s in range(STEPS)]].to_numpy(dtype="float32")


# --------------------------------------------------------------------------------------------------- patterns
def patterns(t: pd.DataFrame, exclude_region: str | None = None, only_regions=None) -> dict:
    """The labelled patterns: per pattern and channel, the median of its members per step and their spread, each also
    stretched in time (``STRETCH``). Returns ``{"name", "is_rice", "stretch", "<c>_med", "<c>_spread"}`` with one row
    per (pattern, stretch). Single-pixel patterns have no spread of their own: they take the median spread of all
    patterns at that step."""
    if exclude_region is not None:
        t = t[t["region"] != exclude_region]
    if only_regions:
        t = t[t["region"].isin(list(only_regions))]
    names, rice, med, spr = [], [], {c: [] for c in CHANNELS}, {c: [] for c in CHANNELS}
    for name, g in t.groupby("pattern", sort=True):
        names.append(name)
        rice.append(bool(g["is_rice"].iloc[0]))
        single = len(g) == 1
        for c in CHANNELS:
            x = _get(g, c)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(x, axis=0)
                s = 1.4826 * np.nanmedian(np.abs(x - m), axis=0)
            if not single:
                few = np.isfinite(x).sum(axis=0) < MIN_MEMBERS
                m[few] = np.nan
                s[few] = np.nan
            med[c].append(m)
            spr[c].append(np.full(STEPS, np.nan) if single else s)
    out = {"name": [], "is_rice": [], "stretch": []}
    for c in CHANNELS:
        m, s = np.array(med[c]), np.array(spr[c])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            typical = np.nanmedian(s, axis=0)
        s = np.where(np.isnan(s) & np.isfinite(m), typical[None, :], s)
        sm, ss = [], []
        for a in STRETCH:
            sm.append(_stretch(m, a))
            ss.append(_stretch(s, a))
        out[f"{c}_med"] = np.concatenate(sm)
        out[f"{c}_spread"] = np.concatenate(ss)
    for a in STRETCH:
        out["name"] += names
        out["is_rice"] += rice
        out["stretch"] += [a] * len(names)
    out["is_rice"] = np.array(out["is_rice"])
    return out


def _stretch(m, a: float) -> np.ndarray:
    """Each row read at step s / a (a > 1: the crop grows slower than the pattern); NaN past the pattern's end."""
    m = np.asarray(m, dtype="float32")
    pos = np.arange(STEPS) / a
    lo = np.minimum(np.floor(pos).astype(int), STEPS - 1)
    hi = np.minimum(lo + 1, STEPS - 1)
    f = (pos - lo).astype("float32")
    out = m[:, lo] * (1 - f) + m[:, hi] * f
    out[:, pos > STEPS - 1] = np.nan
    return out


# --------------------------------------------------------------------------------------------------- distance
def distances(ch: dict, noise: dict, pat: dict, k: float = 2.0, chunk: int = 512) -> np.ndarray:
    """(pixels, patterns): mean over the shared steps of min(z^2, (2k)^2), z = (pixel - pattern median) /
    sqrt(pattern spread^2 + pixel wobble^2); NaN where fewer than ``MIN_STEPS`` steps are shared. Chunks of pixels run
    on every core (numpy releases the lock)."""
    from ..resources import detect_resources

    n = len(ch["ndvi"])
    out = np.full((n, len(pat["name"])), np.nan, dtype="float32")
    cap = (2 * k) ** 2
    optical_pattern = np.isfinite(pat["ndvi_med"]).sum(axis=1) >= MIN_STEPS

    def one(lo):
        hi = min(lo + chunk, n)
        tot = np.zeros((hi - lo, len(pat["name"])), dtype="float32")
        cnt = np.zeros_like(tot)
        for c in CHANNELS:
            x = ch[c][lo:hi, None, :]
            m = pat[f"{c}_med"][None]
            var = pat[f"{c}_spread"][None] ** 2 + (np.nan_to_num(noise[c][lo:hi]).astype("float32") ** 2)[:, None, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                z2 = np.minimum((x - m) ** 2 / var, cap)
            ok = np.isfinite(z2)
            tot += np.where(ok, z2, 0).sum(axis=2)
            cnt += ok.sum(axis=2)
            if c == "ndvi":
                optical_pixel = np.isfinite(ch[c][lo:hi]).sum(axis=1) >= MIN_STEPS
                shared_ndvi = ok.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(cnt >= MIN_STEPS, tot / cnt, np.nan)
            # a pixel with an optical curve is matched only with patterns that have one: a radar-only pattern (aoi13,
            # no clear view after July) has a wide spread and "fits" almost any curve on the radar alone (1 Oct: 5,107
            # of 9,814 aoi39 pixels went to aoi13 patterns, 45 ac of its rice groups became "not rice")
            d = np.where(optical_pixel[:, None] & (shared_ndvi < MIN_STEPS), np.nan, d)
            # ...and a radar-only pixel only with radar-only patterns (user, 1 Oct: "radar-only AOIs' curves for
            # radar-only pixels, NDVI ones for NDVI pixels")
            out[lo:hi] = np.where(~optical_pixel[:, None] & optical_pattern[None, :], np.nan, d)

    with ThreadPoolExecutor(max(1, int(detect_resources().cpus))) as pool:
        list(pool.map(one, range(0, n, chunk)))
    return out


def decide(dist: np.ndarray, pat: dict, k: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per pixel: class (1 rice, 0 not rice, 3 unlike any pattern, -1 too few steps), the deciding pattern's index and
    its distance (root of the mean capped z^2: within k = as close as the pattern's own members).

    Lean to rice (user, 1 Oct, aoi39 pixel 36317: not-rice pattern 0.88 vs rice pattern 0.94): when a rice pattern is
    within k the pixel is rice, even if a not-rice pattern is a little nearer; the AOIs are mostly rice and a near tie
    is no evidence against it. Not rice only when no rice pattern fits and a not-rice pattern is the nearest."""
    d = np.where(np.isnan(dist), np.inf, dist)
    rows = np.arange(len(d))
    best = d.argmin(axis=1)
    dbest = np.sqrt(d[rows, best])
    rice = np.asarray(pat["is_rice"], dtype=bool)
    d_rice = np.where(rice[None, :], d, np.inf)
    best_rice = d_rice.argmin(axis=1)
    rice_fits = np.sqrt(d_rice[rows, best_rice]) <= k
    best = np.where(rice_fits, best_rice, best)
    dbest = np.sqrt(d[rows, best])
    cls = np.where(rice[best], 1, 0)
    cls = np.where(dbest > k, 3, cls)
    cls = np.where(np.isinf(dbest), -1, cls)
    return cls, best, dbest


# --------------------------------------------------------------------------------------------------------- check
def check(t: pd.DataFrame | None = None, k: float = 2.0) -> pd.DataFrame:
    """Leave one region out: each region's labelled curves matched only with the other regions' patterns."""
    t = labelled_table() if t is None else t
    rows = []
    for region in sorted(t["region"].unique()):
        test = t[t["region"] == region]
        pat = patterns(t, exclude_region=region)
        ch = {c: _get(test, c) for c in CHANNELS}
        noise = {c: test[f"noise_{c}"].to_numpy(dtype="float32") for c in CHANNELS}
        cls, _, _ = decide(distances(ch, noise, pat, k), pat, k)
        for kind, flag in (("rice groups", True), ("not-rice groups", False)):
            m = test["is_rice"].to_numpy() == flag
            if m.sum():
                rows.append({"region": region, "curves": kind, "n": int(m.sum()),
                             "called rice %": round(100 * float((cls[m] == 1).mean()), 1),
                             "unlike any %": round(100 * float((cls[m] == 3).mean()), 1)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- run
def run(aoi_id: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late", fresh: str = FRESH,
        k: float = 2.0, learn_from=None) -> pd.DataFrame:
    """The similarity search on one AOI's crop ground; writes ``aoi<N>_step3s_class.tif`` / ``.qml`` / ``.csv`` (the
    tree's ``_step3t`` files stay for comparison) and ``aoi<N>_step3s_nearest.csv`` (each crop pixel's nearest
    pattern and distance, for ``--explain``). ``learn_from``: the regions whose patterns are used (e.g. ``["aoi160",
    "aoi13", "aoi28"]``, user 1 Oct: "consider aoi160, 13, 28 as right"); default all labelled regions."""
    pat = patterns(labelled_table(), only_regions=learn_from)
    inp = rf.aoi_inputs(aoi_id, series_root, fresh)
    ch = channels(inp["fit"], inp["vh"], inp["vv"], inp["sow"], inp["by_radar"])
    noise = own_noise(inp["raw"], inp["vh"], inp["vv"])
    noise["ndvi"] = inp["noise"]
    dist = distances(ch, noise, pat, k)
    cls, best, dbest = decide(dist, pat, k)
    tr = rf.traits(inp["fit"], inp["vh"], inp["sow"], inp["noise"], raw=inp["raw"], vv=inp["vv"])
    # too few steps since the sowing to have a shape: leave it to the young / late rules, else not rice
    pred = np.where(cls == -1, 0, cls)
    pd.DataFrame({"pixel": inp["crop"], "class": cls, "pattern": np.array(pat["name"])[best],
                  "stretch": np.array(pat["stretch"])[best], "distance": np.round(dbest, 2)}).to_csv(
        f"{inp['stem']}3s_nearest.csv", index=False)
    return rf.finish(aoi_id, inp, tr, pred, "3s", names=CLASSES)


def explain(aoi_id: int, pid: int, top: int = 5, k: float = 2.0, learn_from=None) -> pd.DataFrame:
    """The ``top`` nearest patterns of one pixel (best stretch each), with their label and distance."""
    import rasterio

    pat = patterns(labelled_table(), only_regions=learn_from)
    inp = rf.aoi_inputs(aoi_id)
    i = np.searchsorted(inp["crop"], pid)
    if i >= len(inp["crop"]) or inp["crop"][i] != pid:
        with rasterio.open(Path(FRESH) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step3s_class.tif") as ds:
            return pd.DataFrame([{"pixel": pid, "note": f"not crop ground; class {int(ds.read(1).ravel()[pid])}"}])
    sl = slice(i, i + 1)
    ch = {c: v[sl] for c, v in channels(inp["fit"], inp["vh"], inp["vv"], inp["sow"], inp["by_radar"]).items()}
    noise = own_noise(inp["raw"][sl], inp["vh"][sl], inp["vv"][sl])
    noise["ndvi"] = inp["noise"][sl]
    d = np.sqrt(distances(ch, noise, pat, k)[0])
    r = pd.DataFrame({"pattern": pat["name"], "rice": pat["is_rice"], "stretch": pat["stretch"], "distance": d})
    r = r.sort_values("distance").drop_duplicates("pattern").head(top)
    r.insert(0, "pixel", pid)
    r["sowing"] = str(inp["sow_day"].iloc[i].date()) if pd.notna(inp["sow_day"].iloc[i]) else None
    r["radar_only"] = bool(inp["by_radar"][i])
    return r.round(2)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["check", "run"])
    p.add_argument("--aoi", type=int)
    p.add_argument("--explain", type=int, nargs="+", metavar="PIXEL_ID")
    p.add_argument("--learn-from", nargs="+", metavar="REGION",
                   help="use only these regions' patterns, e.g. aoi160 aoi13 aoi28 (default: all labelled)")
    args = p.parse_args(argv)
    import os

    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    pd.set_option("display.width", 200)
    if args.step == "check":
        print(check().to_string(index=False))
        return 0
    if args.explain:
        for pid in args.explain:
            print(explain(args.aoi, pid, learn_from=args.learn_from).to_string(index=False))
        return 0
    print(run(args.aoi, learn_from=args.learn_from).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
