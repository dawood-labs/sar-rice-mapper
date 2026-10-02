"""Fresh start, step 3: is the crop rice? Compare the SHAPE of each field's curve with the shape of surveyed rice.

Why
---
User rules (30 Sep 2026): rice is known from the ground data, not re-learned from each AOI; no fixed number of days
(50-60 days to a full canopy is only a reference); a field may be rice although its canopy stays at 0.5 or lower (edge
pixels): "the shape of the curve matters the most"; a crop older than 5 months is not rice; water is evidence, not a
condition.

How
---
1. **Shape of a curve.** From the pixel's sowing (step 2, ``sowing_fresh``) on, its smoothed NDVI is put on its own
   scale: 0 = the value at sowing, 1 = its own highest value since the sowing. Days count from the sowing, so May, June
   and July crops line up; a bright field and a dim edge pixel of the same crop get the same shape.
2. **The rice shape** is learned from the surveyed plots (reported standing rice on 2 Sep 2026): per 5-day step after
   the sowing, the median shape of the plot-interior pixels and its spread (1.4826 x MAD). A step with fewer than
   ``radar_water.WATER_PEERS_MIN`` plot pixels has no reference.
3. **The judgement.** For every step from the sowing to the pixel's own top (ripening after it varies), the gap to the
   rice median in units of sqrt(rice spread^2 + the pixel's own wobble on its scale^2); the sowing may slide up to the
   first clearly green view (``max_shift``: it was not seen between the last empty and the first green view) and the
   best start is used. Rice when the root-mean-square gap is within ``WATER_K``; older
   than 5 months (``MAX_AGE_DAYS``) is not rice; "undecided" when the pixel has fewer steps than the rice needs to get
   half-way up (read from the reference itself). Everything else: other vegetation.
4. **Cloudy months** (``CLOUDY_MONTHS``, July-August): a fall of the smoothed curve there is left out of the
   comparison unless at least two clear views show it AND the radar's VH fell too (``radar_unconfirmed_dips``): a hazy
   view dips the optical curve, a real loss of canopy shows on two views and lowers VH. The same rule shapes the
   reference.
5. **Check without circularity**: the plots of one region are judged with a shape learned from the other regions only.

Use::

    python -m sar_pipeline.analysis.rice_shape reference          # plot shapes -> rice_fresh/reference_shapes.csv
    python -m sar_pipeline.analysis.rice_shape check              # leave-one-region-out on the plots
    python -m sar_pipeline.analysis.rice_shape run --aoi 160      # the step-3 map of one AOI
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from .optical_phenology import acres

FRESH = "processed/_batch/s2_2026/rice_fresh"
#: User decision (30 Sep): a crop older than 5 months on the last date is not rice.
MAX_AGE_DAYS = 150
STEP = 5
STEPS = MAX_AGE_DAYS // STEP + 1
CLASSES = {0: "not vegetation in September", 1: "rice", 2: "other vegetation", 3: "older than 5 months",
           4: "undecided (too young to judge)", 5: "trees / orchards (step 1)", 255: "outside"}
COLOURS = {0: "#d9d9d9", 1: "#1a9850", 2: "#e3a21a", 3: "#8c510a", 4: "#9ecae1", 5: "#1b5e20"}


def shapes(fit, sow_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(shape (STEPS, pixels), age_steps, amplitude)``: the smoothed curve from the sowing window on, on the pixel's
    own 0-1 scale (0 = at sowing, 1 = its highest value since), NaN after the last window; ``age_steps`` = windows since
    the sowing; ``amplitude`` = highest minus sowing value. ``sow_idx`` -1 = no sowing (all NaN)."""
    f = np.asarray(fit, dtype="float32")
    n_win, n = f.shape
    sow = np.asarray(sow_idx)
    idx = sow[None, :] + np.arange(STEPS)[:, None]
    ok = (sow[None, :] >= 0) & (idx < n_win)
    vals = np.where(ok, f[np.clip(idx, 0, n_win - 1), np.arange(n)[None, :]], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        after = np.where((np.arange(n_win)[:, None] >= sow[None, :]) & (sow[None, :] >= 0), f, np.nan)
        top = np.nanmax(after, axis=0)
    base = vals[0]
    amp = top - base
    with np.errstate(invalid="ignore", divide="ignore"):
        shape = (vals - base[None, :]) / amp[None, :]
    age = np.where(sow >= 0, n_win - 1 - sow, -1)
    return shape, age, amp


#: Sliding sowing + comparing only up to the top (``max_shift``, ``judge(upto_peak=)``): tried 30 Sep for one region pixels
#: 2036 / 3842 / 3398 (user: rice). OFF: those three stayed "other vegetation", two non-rice plot pixels became rice and
#: aoi160 rice rose 1,196 -> 1,257 ac (a looser test, not a better one).
SLIDE_SOWING = False
UPTO_PEAK = False
#: The cloudiest months (user, 30 Sep): an optical fall in them counts only when the radar falls too.
CLOUDY_MONTHS = (7, 8)
REVISIT_DAYS = 12


def radar_unconfirmed_dips(fit, sow_idx, windows, series, k: float | None = None, raw=None, noise=None) -> np.ndarray:
    """(STEPS, pixels) True where the smoothed NDVI, in July or August, lies below its own highest value since the
    sowing (it has fallen) while the radar says the canopy did not: the pixel's VH on the nearest pass (any track,
    within one revisit) has NOT fallen by ``k`` x its pass-to-pass wobble below its VH on the pass nearest the time the
    optical curve was at its top (not the highest VH since sowing: a bright pass before the transplanting flood made
    every later VH look "fallen", pixel 50746). With ``raw`` (the clear views, same windows): a dip that rests on ONE low
    clear view is left out whatever the radar says (user, 30 Sep, pixel 50746: its only low view, 7 Aug, was hazy; the
    radar was dark in early July from water in the standing paddy). One view is often haze; a dip needs two. A low view
    is a clear view ``k`` x ``noise`` (the pixel's own wobble) below the highest clear view since the sowing, not below
    the smooth curve, whose top can overshoot between two views (pixel 50746: 0.55 on 3 Jun and 18 Jul, curve 0.64). Such steps are left out of the shape comparison (user, 30 Sep, aoi160 pixel 50746: a hazy 7 Aug view pulled
    the curve from 0.64 to 0.43 while VH stayed at -15 / -16). No radar pass nearby: the step is kept (the optical
    decides). ``fit``: (windows, pixels) from the season start, ``windows`` their dates."""
    from .radar_water import WATER_K, pass_noise

    k = WATER_K if k is None else k
    f = np.asarray(fit, dtype="float32")
    n_win, n = f.shape
    w = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    sow = np.asarray(sow_idx)
    cols = np.arange(n)
    out = np.zeros((STEPS, n), dtype=bool)
    tracks = []
    for dates, flat in series or []:
        d = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        x = np.asarray(flat["VH"], dtype="float32")
        tracks.append((d, x, pass_noise(x)))
    if not tracks:
        return out
    running = np.full(n, -np.inf, dtype="float32")
    peak_day = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    fell_opt = np.zeros((STEPS, n), dtype=bool)
    seen_r = np.zeros((STEPS, n), dtype=bool)
    fell_r = np.zeros((STEPS, n), dtype=bool)
    low_view = np.zeros((STEPS, n), dtype=bool)
    best_view = np.full(n, -np.inf, dtype="float32")
    wobble = np.zeros(n) if noise is None else np.nan_to_num(np.asarray(noise, dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for step in range(STEPS):
            t = sow + step
            inside = (sow >= 0) & (t < n_win)
            tc = np.clip(t, 0, n_win - 1)
            now = np.where(inside, f[tc, cols], np.nan)
            day = np.where(inside, w[tc], np.datetime64("NaT"))
            if raw is not None:
                rv = np.where(inside, np.asarray(raw)[tc, cols], np.nan)
                best_view = np.fmax(best_view, np.where(np.isfinite(rv), rv, -np.inf))
            higher = np.isfinite(now) & (now >= running)
            peak_day = np.where(higher, day, peak_day)          # when the optical curve was at its highest so far
            running = np.fmax(running, np.where(np.isfinite(now), now, -np.inf))
            month = pd.DatetimeIndex(day).month.to_numpy()
            cloudy = inside & np.isin(month, CLOUDY_MONTHS)
            if not cloudy.any():
                continue
            fell_optical = cloudy & (now < running)
            seen = np.zeros(n, dtype=bool)
            fell = np.zeros(n, dtype=bool)
            for d, x, noise in tracks:
                dd = np.where(np.isnat(day), d[0], day)
                j = np.abs(d[:, None] - dd[None, :]).argmin(axis=0)
                near = np.abs(d[j] - dd) <= np.timedelta64(REVISIT_DAYS, "D")
                # the radar when the optical curve was at its top: a bright pass before the flood must not count
                pd_ = np.where(np.isnat(peak_day), d[0], peak_day)
                jp = np.abs(d[:, None] - pd_[None, :]).argmin(axis=0)
                near_p = ~np.isnat(peak_day) & (np.abs(d[jp] - pd_) <= np.timedelta64(REVISIT_DAYS, "D"))
                peak = np.where(near_p, x[jp, cols], np.nan)
                val = x[j, cols]
                ok = near & np.isfinite(val) & np.isfinite(peak)
                seen |= ok
                fell |= ok & (val <= peak - k * noise)
            fell_opt[step], seen_r[step], fell_r[step] = fell_optical, seen, fell
            if raw is not None:
                with np.errstate(invalid="ignore"):
                    low_view[step] = fell_optical & np.isfinite(rv) & (rv < best_view - k * wobble)
    # one dip is one event: it is real when the radar fell at any step of it (the two rarely fall on the same day)
    run_id = np.cumsum(np.vstack([np.ones((1, n), bool), fell_opt[1:] & ~fell_opt[:-1]]), axis=0) * fell_opt
    for p in np.flatnonzero(fell_opt.any(axis=0)):
        ids = run_id[:, p]
        for r in np.unique(ids[ids > 0]):
            steps = ids == r
            one_view = raw is not None and low_view[steps, p].sum() <= 1
            if one_view or (seen_r[steps, p].any() and not fell_r[steps, p].any()):
                out[steps, p] = True
    return out


def max_shift(fit, raw, sow_idx, noise, k: float | None = None) -> np.ndarray:
    """Per pixel: how many windows the sowing may lie after the one step 2 chose: up to (not including) the first clear
    view after it that is clearly green (``k`` x the pixel's wobble above the value at sowing). Why (user, 30 Sep, one region
    pixel 2036): the last empty view was 4 May and the first green one 20 May; the crop was planted somewhere between, so
    counting from 4 May made it look 10-15 days too fast."""
    from .radar_water import WATER_K

    k = WATER_K if k is None else k
    f = np.asarray(fit, dtype="float32")
    r = np.asarray(raw)
    n_win, n = f.shape
    sow = np.asarray(sow_idx)
    step = np.arange(n_win)[:, None]
    base = f[np.clip(sow, 0, n_win - 1), np.arange(n)]
    with np.errstate(invalid="ignore"):
        green = (step > sow[None, :]) & (r > (base + k * np.nan_to_num(np.asarray(noise, dtype=float)))[None, :])
    first = np.where(green.any(axis=0), green.argmax(axis=0), sow)
    return np.where(sow >= 0, np.clip(first - sow - 1, 0, None), 0)


def reference(shape_table: pd.DataFrame, exclude_region: str | None = None) -> pd.DataFrame:
    """Per step: median and spread of the plot shapes (``s0 .. sN`` columns); steps with too few pixels are NaN."""
    from .radar_water import WATER_PEERS_MIN

    t = shape_table if exclude_region is None else shape_table[shape_table["region"] != exclude_region]
    if "is_rice" in t:
        t = t[t["is_rice"].astype(bool)]          # only the plot groups the user confirmed as rice
    cols = [f"s{i}" for i in range(STEPS)]
    v = t[cols].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=0)
        spread = 1.4826 * np.nanmedian(np.abs(v - med[None, :]), axis=0)
    count = np.isfinite(v).sum(axis=0)
    few = count < WATER_PEERS_MIN
    return pd.DataFrame({"step": np.arange(STEPS), "days": np.arange(STEPS) * STEP,
                         "median": np.where(few, np.nan, med), "spread": np.where(few, np.nan, spread), "pixels": count})


def judge(shape, age, amp, noise, ref: pd.DataFrame, k: float | None = None, ignore=None, shift=None,
          upto_peak: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(class, gap)`` per pixel: 1 rice, 2 other vegetation, 3 older than 5 months, 4 undecided (see module docs);
    ``gap`` = the root-mean-square gap to the rice shape in units of the combined spread.

    ``shift`` (per pixel, windows, :func:`max_shift`): the sowing may lie up to that many windows later; the best-fitting
    start is used (the shape re-scaled from it). ``upto_peak``: only the rise from the sowing to the pixel's own top is
    compared; ripening after the top varies from field to field (user, 30 Sep: May rice ripening in September)."""
    from .radar_water import WATER_K

    k = WATER_K if k is None else k
    upto_peak = UPTO_PEAK if upto_peak is None else upto_peak
    if not SLIDE_SOWING:
        shift = None
    sh = np.asarray(shape, dtype=float)
    n_steps, n = sh.shape
    med = ref["median"].to_numpy()[:, None]
    spread = ref["spread"].to_numpy()[:, None]
    shift = np.zeros(n, dtype=int) if shift is None else np.asarray(shift, dtype=int)
    ign = np.zeros_like(sh, dtype=bool) if ignore is None else np.asarray(ignore, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        own = np.nan_to_num(np.asarray(noise, dtype=float)) / np.asarray(amp, dtype=float)
    best = np.full(n, np.inf)
    best_s = np.zeros(n, dtype=int)
    rows = np.arange(n_steps)[:, None]
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        top_step = np.where(np.isfinite(sh).any(axis=0), np.nanargmax(np.where(np.isfinite(sh), sh, -np.inf), axis=0), 0)
        for s in range(int(shift.max()) + 1 if n else 1):
            idx = rows + s
            inside = idx < n_steps
            ic = np.clip(idx, 0, n_steps - 1)
            moved = np.where(inside, sh[ic, np.arange(n)[None, :]], np.nan)
            start = sh[min(s, n_steps - 1)]
            scaled = (moved - start[None, :]) / (1 - start[None, :])
            drop = np.where(inside, ign[ic, np.arange(n)[None, :]], True)
            if upto_peak:
                drop |= idx > top_step[None, :]
            scaled = np.where(drop, np.nan, scaled)
            z = (scaled - med) / np.sqrt(spread ** 2 + own[None, :] ** 2)
            g = np.sqrt(np.nanmean(z ** 2, axis=0))
            better = (s <= shift) & np.isfinite(g) & (g < best)
            best = np.where(better, g, best)
            best_s = np.where(better, s, best_s)
    gap = np.where(np.isfinite(best), best, np.nan)
    # the rice needs this many steps to get half-way up (from the reference itself): a younger pixel cannot be judged
    half = ref.loc[ref["median"] >= 0.5, "step"]
    need = int(half.iloc[0]) if len(half) else STEPS
    age = np.asarray(age) - best_s
    out = np.full(len(age), 2, dtype="uint8")
    out[gap <= k] = 1
    out[(np.asarray(age) + best_s >= 0) & (age < need)] = 4
    out[age * STEP > MAX_AGE_DAYS] = 3
    out[np.asarray(age) + best_s < 0] = 0
    return out, gap


#: The user's verdict per plot group (``plot_clusters``, 30 Sep: groups 1, 5, 15, 17, 18 are not rice, the other 15 are
#: variations of rice). Local file (processed/), with the group of every plot beside it.
GROUPS = "processed/_batch/s2_2026/rice_fresh/plot_clusters/plot_groups_k20.csv"
VERDICTS = "processed/_batch/s2_2026/rice_fresh/plot_clusters/group_verdicts_k20.csv"


def attach_verdicts(table: pd.DataFrame, groups: str = GROUPS, verdicts: str = VERDICTS) -> pd.DataFrame:
    """Add ``group`` and ``is_rice`` (the user's verdict on the plot's curve group) to the plot-pixel table."""
    g = pd.read_csv(groups)[["plot_id", "aoi", "group"]].merge(pd.read_csv(verdicts), on="group", how="left")
    out = table.drop(columns=[c for c in ("group", "rice", "is_rice") if c in table]).merge(
        g, on=["plot_id", "aoi"], how="left")
    out["is_rice"] = out["rice"].fillna(False).astype(bool)
    return out.drop(columns=["rice"])


def plot_shapes(plots=None, series_root: str = "processed/_batch/s2_2026") -> pd.DataFrame:
    """One row per plot-interior pixel of every plot AOI: region, aoi, sowing date, age, amplitude and shape ``s0..``."""
    from . import radar_water as rw
    from . import sowing_fresh as sf
    from . import validation as va
    from .compare_runs import load_plots
    from .monsoon_rule import optical_noise
    from .plot_curves import interior_pixels, plot_pixels

    plots = load_plots() if plots is None else plots
    rows = []
    for aoi in sorted(plots["aoi"].unique(), key=lambda a: int(a[3:])):
        aoi_id = int(aoi[3:])
        d = nd.load(aoi_id, out_root=series_root)
        g = d["loc"]["grid"]
        pix = plot_pixels(plots[plots["aoi"] == aoi], g)
        inner = interior_pixels(pix, int(g["width"]))
        owner = {int(q): k for k, qs in pix.items() for q in qs[inner[k]]}
        p = np.unique(np.concatenate([q[inner[k]] for k, q in pix.items()] or [np.array([], int)])).astype(int)
        if len(p) == 0:
            continue
        w = pd.DatetimeIndex(d["windows"])
        fit = d["ndvi5d"].reshape(len(w), -1)[:, p]
        raw = d["ndvi5d_raw"].reshape(len(w), -1)[:, p]
        series = [(dd, {q: v[:, p] for q, v in flat.items()}) for dd, flat in rw.read_series(aoi_id)]
        s = sf.sowing(fit, raw, w, np.ones(len(p), bool), series)
        use = w >= pd.Timestamp(sf.START)
        sow_idx = np.where(s["period"] > 0, np.searchsorted(w[use], pd.DatetimeIndex(s["sowing_date"])), -1)
        sh, age, amp = shapes(fit[use], sow_idx)
        # the same rule as for the judged pixels: cloudy-month dips the radar does not confirm are not part of the shape
        noise = optical_noise(raw)
        sh = np.where(radar_unconfirmed_dips(fit[use], sow_idx, w[use], series, raw=raw[use], noise=noise), np.nan, sh)
        shift = max_shift(fit[use], raw[use], sow_idx, noise)
        frame = pd.DataFrame(sh.T, columns=[f"s{i}" for i in range(STEPS)])
        frame.insert(0, "max_shift", shift)
        frame.insert(0, "noise", noise)
        frame.insert(0, "amplitude", amp)
        frame.insert(0, "age_steps", age)
        frame.insert(0, "sowing_date", s["sowing_date"].to_numpy())
        frame.insert(0, "pixel", p)
        frame.insert(0, "plot_id", [owner.get(int(q)) for q in p])
        frame.insert(0, "region", va.REGION.get(aoi, aoi))
        frame.insert(0, "aoi", aoi)
        rows.append(frame)
        nd.forget()
        print(f"{aoi}: {len(p)} plot-interior pixels", flush=True)
    return pd.concat(rows, ignore_index=True)


def check(table: pd.DataFrame) -> pd.DataFrame:
    """Leave one region out: its plot pixels judged with the shape of the other regions. With the user's verdicts
    (``is_rice``) the rice groups and the not-rice groups are scored apart (recall and false rice)."""
    out = []
    kinds = [("rice groups", True), ("not-rice groups", False)] if "is_rice" in table else [("all plots", None)]
    for region in sorted(table["region"].unique()):
        ref = reference(table, exclude_region=region)
        for kind, flag in kinds:
            t = table[table["region"] == region]
            if flag is not None:
                t = t[t["is_rice"].astype(bool) == flag]
            if not len(t):
                continue
            cls, gap = judge(t[[f"s{i}" for i in range(STEPS)]].to_numpy().T, t["age_steps"].to_numpy(),
                             t["amplitude"].to_numpy(), t["noise"].to_numpy(), ref,
                             shift=t["max_shift"].to_numpy() if "max_shift" in t else None)
            n = len(t)
            out.append({"region": region, "plots": kind, "pixels": n,
                        **{CLASSES[c]: round(100 * float((cls == c).sum()) / n, 1) for c in (1, 2, 3, 4)},
                        "median_gap": round(float(np.nanmedian(gap)), 2)})
    return pd.DataFrame(out)


def run(aoi_id: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late", fresh: str = FRESH) -> pd.DataFrame:
    """The step-3 map of one AOI (steps 1 and 2 and the reference table must exist); returns the acre table."""
    import rasterio

    from .monsoon_rule import optical_noise

    table = pd.read_csv(Path(fresh) / "reference_shapes.csv")
    ref = reference(table)
    stem = Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step"
    with rasterio.open(f"{stem}1_cover.tif") as ds:
        cover = ds.read(1).ravel()
        profile = ds.profile
    with rasterio.open(f"{stem}2_sowing.tif") as ds:
        doy = ds.read(1).ravel()
    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    use = w >= pd.Timestamp("2026-05-01")
    fit = d["ndvi5d"].reshape(len(w), -1)[use]
    raw_use = d["ndvi5d_raw"].reshape(len(w), -1)[use]
    noise = optical_noise(d["ndvi5d_raw"].reshape(len(w), -1))
    nd.forget()
    sow_day = pd.Timestamp("2026-01-01") + pd.to_timedelta(doy.astype(int) - 1, "D")
    sow_idx = np.where(doy > 0, np.searchsorted(w[use], sow_day), -1)
    sh, age, amp = shapes(fit, sow_idx)
    from . import radar_water as rw

    ignore = radar_unconfirmed_dips(fit, sow_idx, w[use], rw.read_series(aoi_id), raw=raw_use, noise=noise)
    cls, gap = judge(sh, age, amp, noise, ref, ignore=ignore, shift=max_shift(fit, raw_use, sow_idx, noise))
    out = np.where(cover == 255, 255, np.where(cover == 2, 5, np.where(cover == 1, cls, 0))).astype("uint8")
    shape = (profile["height"], profile["width"])
    with rasterio.open(f"{stem}3_class.tif", "w", **dict(profile, dtype="uint8", nodata=255, count=1)) as ds:
        ds.write(out.reshape(shape)[None])
    with rasterio.open(f"{stem}3_gap.tif", "w", **dict(profile, dtype="float32", nodata=np.nan, count=1)) as ds:
        ds.write(np.where(cover == 1, gap, np.nan).astype("float32").reshape(shape)[None])
    entries = "\n".join(f'        <paletteEntry value="{c}" color="{col}" alpha="255" label="{c} {CLASSES[c]}"/>'
                        for c, col in COLOURS.items())
    Path(f"{stem}3_class.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
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
    acres_t = pd.DataFrame([{"class": c, "name": CLASSES[c], "acres": round(acres(int((out == c).sum())), 1)}
                            for c in COLOURS])
    acres_t.to_csv(f"{stem}3.csv", index=False)
    preview(out.reshape(shape), ref, table, sh, cls, Path(f"{stem}3.png"), f"aoi{aoi_id}: step 3, rice by curve shape")
    return acres_t


def preview(classes, ref, table, sh, cls, path: Path, title: str) -> None:
    """The class map, and the rice shape band with a few pixels of each class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lut = np.ones((256, 3))
    for c, col in COLOURS.items():
        lut[c] = matplotlib.colors.to_rgb(col)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].imshow(lut[classes], interpolation="nearest")
    axes[0].axis("off")
    axes[0].legend(handles=[Patch(color=COLOURS[c], label=CLASSES[c]) for c in COLOURS], loc="lower left", fontsize=8)
    days = ref["days"].to_numpy()
    m, s = ref["median"].to_numpy(), ref["spread"].to_numpy()
    axes[1].fill_between(days, m - 2 * s, m + 2 * s, color="#1a9850", alpha=0.2, label="surveyed rice: median +- 2 spread")
    axes[1].plot(days, m, color="#1a9850", lw=2.5, label="surveyed rice: median shape")
    rng = np.random.default_rng(0)
    for c, col in ((1, "#1a9850"), (2, "#e3a21a")):
        idx = np.flatnonzero(cls == c)
        for i in rng.choice(idx, min(15, len(idx)), replace=False) if len(idx) else []:
            axes[1].plot(days, sh[:, i], color=col, lw=0.6, alpha=0.6)
    axes[1].plot([], [], color="#e3a21a", lw=1, label="other vegetation (sample pixels)")
    axes[1].set_xlabel("days since sowing")
    axes[1].set_ylabel("NDVI on the pixel's own scale (0 = sowing, 1 = its highest)")
    axes[1].set_ylim(-0.5, 1.3)
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def explain(aoi_id: int, pid: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late", fresh: str = FRESH):
    """One pixel: its sowing, and step by step its shape, the rice shape and the gap (in combined spreads). Why: to
    answer "why is this pixel other vegetation?" from the numbers (user, 30 Sep)."""
    import rasterio

    from .monsoon_rule import optical_noise
    from .radar_water import WATER_K

    ref = reference(pd.read_csv(Path(fresh) / "reference_shapes.csv"))
    with rasterio.open(Path(fresh) / f"aoi{aoi_id}" / f"aoi{aoi_id}_step2_sowing.tif") as ds:
        doy = int(ds.read(1).ravel()[pid])
    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    use = w >= pd.Timestamp("2026-05-01")
    fit = d["ndvi5d"].reshape(len(w), -1)[use][:, [pid]]
    noise = optical_noise(d["ndvi5d_raw"].reshape(len(w), -1)[:, [pid]])
    sow_day = pd.Timestamp("2026-01-01") + pd.Timedelta(days=doy - 1)
    sow_idx = np.array([np.searchsorted(w[use], sow_day)]) if doy > 0 else np.array([-1])
    sh, age, amp = shapes(fit, sow_idx)
    from . import radar_water as rw

    series = [(dd, {q: v[:, [pid]] for q, v in flat.items()}) for dd, flat in rw.read_series(aoi_id)]
    ignore = radar_unconfirmed_dips(fit, sow_idx, w[use], series,
                                    raw=d["ndvi5d_raw"].reshape(len(w), -1)[use][:, [pid]], noise=noise)
    shift = max_shift(fit, d["ndvi5d_raw"].reshape(len(w), -1)[use][:, [pid]], sow_idx, noise)
    cls, gap = judge(sh, age, amp, noise, ref, ignore=ignore, shift=shift)
    own = float(noise[0] / amp[0]) if amp[0] else np.nan
    z = (sh[:, 0] - ref["median"].to_numpy()) / np.sqrt(ref["spread"].to_numpy() ** 2 + own ** 2)
    ndvi = np.full(STEPS, np.nan)
    k = min(STEPS, fit.shape[0] - int(sow_idx[0])) if sow_idx[0] >= 0 else 0
    ndvi[:k] = fit[int(sow_idx[0]):int(sow_idx[0]) + k, 0]
    t = pd.DataFrame({"days": ref["days"], "ndvi": ndvi.round(2), "shape": sh[:, 0].round(2),
                      "rice_shape": ref["median"].round(2), "rice_spread": ref["spread"].round(2), "gap": z.round(1),
                      "left_out": ignore[:, 0]})
    t = t[t["shape"].notna()]
    t.attrs.update(sowing=str(sow_day.date()) if doy > 0 else None, sowing_may_shift_windows=int(shift[0]),
                   amplitude=round(float(amp[0]), 2),
                   rms_gap=round(float(gap[0]), 2), limit=WATER_K, result=CLASSES[int(cls[0])])
    return t


def region_curves(region: str, n: int = 12, series_root: str = "processed/_batch/s2_2026", fresh: str = FRESH,
                  seed: int = 0) -> Path:
    """A sheet of ``n`` plot-interior pixels of one region (spread over its AOIs): smoothed NDVI and clear views, VH of
    every track, the sowing step 2 found and the step-3 result (shape learned without this region). Why (user, 30 Sep):
    to look at the region whose plots fail (late-August transplanting) before changing any rule."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import radar_water as rw
    from . import sar_curve
    from .monsoon_rule import optical_noise
    from .pixel_report import locate

    table = pd.read_csv(Path(fresh) / "reference_shapes.csv", parse_dates=["sowing_date"])
    ref = reference(table, exclude_region=region)
    t = table[table["region"] == region]
    rng = np.random.default_rng(seed)
    per_aoi = max(1, n // max(t["aoi"].nunique(), 1))
    pick = pd.concat([g.iloc[rng.choice(len(g), min(per_aoi, len(g)), replace=False)] for _, g in t.groupby("aoi")])
    pick = pick.head(n)
    cls, _ = judge(pick[[f"s{i}" for i in range(STEPS)]].to_numpy().T, pick["age_steps"].to_numpy(),
                   pick["amplitude"].to_numpy(), pick["noise"].to_numpy(), ref,
                   shift=pick["max_shift"].to_numpy() if "max_shift" in pick else None)
    cols = 3
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.6 * rows), squeeze=False)
    for ax, (_, r), c in zip(axes.ravel(), pick.iterrows(), cls):
        aoi_id = int(r["aoi"][3:])
        d = nd.load(aoi_id, out_root=series_root)
        w = pd.DatetimeIndex(d["windows"])
        pid = int(r["pixel"])
        keep = w >= pd.Timestamp("2026-03-01")
        ax.plot(w[keep], d["ndvi5d"].reshape(len(w), -1)[keep, pid], color="#2a78d6", lw=2)
        raw = d["ndvi5d_raw"].reshape(len(w), -1)[keep, pid]
        ax.scatter(w[keep], raw, color="#1f7a3a", s=14, zorder=3)
        if pd.notna(r["sowing_date"]):
            ax.axvline(r["sowing_date"], color="#b45f06", ls="--", lw=1.2)
        ax.set_ylim(-0.3, 1.0)
        ax.set_title(f"{r['aoi']} px {pid}: sowing {pd.Timestamp(r['sowing_date']).date() if pd.notna(r['sowing_date']) else '-'}"
                     f", {CLASSES[int(c)]}", fontsize=9)
        a2 = ax.twinx()
        loc = locate(aoi_id, pid, season_key="monsoon2026")
        g = d["loc"]["grid"]
        for k, tr in enumerate([x["track_id"] for x in loc["cfg"]["s1"]["tracks"]]):
            rd, cubes = sar_curve.read_track_pixels(loc, tr, [pid], int(g["width"]), int(g["height"]))
            sel = rd >= pd.Timestamp("2026-03-01")
            a2.plot(rd[sel], cubes["VH"][sel, 0], ":", marker=".", ms=4, color=["#7a3ab4", "#b45f06", "#555555"][k % 3])
        a2.set_ylim(-30, -8)
        a2.tick_params(labelsize=7)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{region}: plot-interior pixels (surveyed standing rice, 2 Sep). Blue = smoothed NDVI, green = clear "
                 f"views, dotted = VH per track (right axis), dashed = sowing found", fontsize=10)
    fig.tight_layout()
    out = Path(fresh) / f"curves_{region}.png"
    fig.savefig(out, dpi=90)
    plt.close(fig)
    nd.forget()
    return out


def pixel_steps(aoi: str, pixel: int, exclude_region: str | None = None, fresh: str = FRESH) -> pd.DataFrame:
    """One plot pixel of the reference table, step by step: its shape, the rice shape (learned without
    ``exclude_region``) and the gap. Why: to see WHERE along the season a surveyed pixel leaves the rice shape."""
    from .radar_water import WATER_K

    table = pd.read_csv(Path(fresh) / "reference_shapes.csv")
    ref = reference(table, exclude_region=exclude_region)
    r = table[(table["aoi"] == aoi) & (table["pixel"] == pixel)].iloc[0]
    sh = r[[f"s{i}" for i in range(STEPS)]].to_numpy(dtype=float)
    own = r["noise"] / r["amplitude"] if r["amplitude"] else np.nan
    z = (sh - ref["median"].to_numpy()) / np.sqrt(ref["spread"].to_numpy() ** 2 + own ** 2)
    t = pd.DataFrame({"days": ref["days"], "shape": sh.round(2), "rice_shape": ref["median"].round(2),
                      "rice_spread": ref["spread"].round(2), "gap": z.round(1)})
    t = t[np.isfinite(sh)]
    t.attrs.update(sowing=r["sowing_date"], rms_gap=round(float(np.sqrt(np.nanmean(z ** 2))), 2), limit=WATER_K)
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["reference", "check", "run", "curves", "steps"])
    p.add_argument("--region", help="curves: the region to draw")
    p.add_argument("-n", type=int, default=12, help="curves: how many pixels")
    p.add_argument("--plot-pixel", nargs=2, metavar=("AOI", "PIXEL"), help="steps: one plot pixel of the reference")
    p.add_argument("--aoi", type=int)
    p.add_argument("--series-root", default=None)
    p.add_argument("--explain", type=int, metavar="PIXEL_ID", help="with run: print one pixel's steps instead")
    args = p.parse_args(argv)
    if args.explain is not None:
        t = explain(args.aoi, args.explain, args.series_root or "processed/_batch/s2_2026_hyb40m1late")
        print(t.to_string(index=False))
        print(t.attrs)
        return 0
    out = Path(FRESH)
    out.mkdir(parents=True, exist_ok=True)
    if args.step == "reference":
        t = plot_shapes(series_root=args.series_root or "processed/_batch/s2_2026")
        if Path(VERDICTS).exists() and Path(GROUPS).exists():
            t = attach_verdicts(t)
        t.to_csv(out / "reference_shapes.csv", index=False)
        print(reference(t).round(3).to_string(index=False))
    elif args.step == "steps":
        t = pixel_steps(args.plot_pixel[0], int(args.plot_pixel[1]), args.region)
        print(t.to_string(index=False))
        print(t.attrs)
    elif args.step == "curves":
        print(region_curves(args.region, args.n))
    elif args.step == "check":
        t = check(pd.read_csv(out / "reference_shapes.csv"))
        t.to_csv(out / "reference_check.csv", index=False)
        print(t.to_string(index=False))
    else:
        print(run(args.aoi, args.series_root or "processed/_batch/s2_2026_hyb40m1late").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
