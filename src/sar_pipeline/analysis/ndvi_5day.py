"""A gap-free NDVI value every five days for every pixel, from lightly masked Sentinel-2 dates.

Why this exists
---------------
The 5-day composites (``s2_windows``) mask cloud inside Earth Engine with s2cloudless, its projected
shadow and a 50 m buffer. The NDVI curves showed that recipe discarding a large share of clear
observations — and the observations it discards are concentrated in exactly the months that matter,
sowing and the monsoon. The curve was then blanked across long gaps, so a pixel could end up with no
value at the moment the classifier needed one.

This module starts from the per-date exports that carry the product's own mask bands
(``optical_export.BANDS_WITH_MASKS``) and does three things, in order:

1. **Light mask.** Only QA60 bits 10 (opaque cloud) and 11 (cirrus) are removed. There is no shadow
   mask on purpose: a flooded field at transplanting is dark, and shadow masks tend to delete it.
   SCL's cloud classes are read too, but only reported, never applied.
2. **Maximum-value composite per 5-day window.** When a window holds more than one clear date, the
   one with the highest NDVI is kept, and its NDWI and LSWI come from *that same date*. Cloud and
   haze only ever pull NDVI down, so the maximum is the least contaminated value — the standard
   choice for vegetation composites (Holben 1986).
3. **Upper-envelope Whittaker fit** (the weighting of Chen et al. 2004, used in TIMESAT). A light
   mask lets haze through, and haze shows up as dips *below* the true curve. The fit is repeated:
   after each pass, an observation above the curve keeps weight 1 and one below it gets
   ``1 - (fit - value) / largest such gap``, so the curve settles onto the upper envelope instead of
   averaging the dips in. Missing windows get weight 0 and are filled by the smoother. **Every
   pixel gets a value in every window**, however long the gap.

Filling a long gap is a guess, and a guess must not look like a measurement, so alongside every
value the module records ``gap_days``: the distance, in days, from that window to the nearest window
that had a real observation (0 where the window itself was observed). A reader can then decide how
much a value in the middle of a 30-day monsoon gap deserves to be trusted.

Outputs (per AOI, in ``out_root/<aoi>/``), each one band per window, band descriptions = window
start date, all on the AOI's own 10 m grid:

* ``<aoi>_ndvi5d.tif`` — fitted NDVI;
* ``<aoi>_lswi5d.tif`` — fitted LSWI, (B8 - B11) / (B8 + B11), the water signal;
* ``<aoi>_ndvi5d_raw.tif`` — the composited observation before fitting (nodata where none);
* ``<aoi>_gapdays5d.tif`` — days to the nearest observed window.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import optical_phenology as op
from . import pixel_report as pr

#: GCS folder of the per-date exports with QA60 and SCL.
FOLDER = "s2_dates_masks"
STEP_DAYS = 5
#: Upper-envelope passes. Chen et al. use a small fixed number; the weights stop changing after 3.
ENVELOPE_ITERATIONS = 3
#: Smoothing strength on the 5-day grid, chosen by :func:`choose_lambda` on the first AOI: 5-fold
#: cross-validation over 400 pixels gave a median absolute error of 0.0440 at 0.5, 0.0428 at 1,
#: 0.0432 at 2 and 0.0451 at 5, rising to 0.061 at 50. The curve is flat between 0.5 and 2, so the
#: choice matters little there; re-run the check on a new region rather than trusting this value.
LMBD = 1.0


def read_dates(aoi_id: int, folder: str = FOLDER, cache_root: str | None = None, cs_min: float | None = None):
    """NDVI, NDWI, LSWI, QA60 cloud and SCL cloud for every exported date, shaped (dates, rows, cols).

    ``ok`` is True where the pixel has data and QA60 does not flag cloud or cirrus. With ``cs_min``
    the Cloud Score+ ``clear`` band (0-100, per 10 m pixel) must also reach that value: the stricter
    mask used to test how much haze the light one lets through. ``scl_cloud`` is returned for
    comparison only.
    """
    import rasterio

    from ..optical_export import SCL_CLOUD_CLASSES, band_index, qa60_cloud

    loc = pr.locate(aoi_id, 0)
    paths = sorted(pr.sync_s2(loc, cache_root or f"data/{folder}", folder).glob("*.tif"))
    if not paths:
        raise FileNotFoundError(f"no files for aoi{aoi_id} in {folder}")
    dates, ndvi, ndwi, lswi, ok, scl = [], [], [], [], [], []
    for path in paths:
        dates.append(pd.to_datetime(path.stem.rsplit("_S2_", 1)[1]))
        with rasterio.open(path) as ds:
            read = lambda n: ds.read(band_index(ds, n)).astype("float32")  # noqa: E731
            b3, b4, b8, b11, qa, sc = (read(n) for n in ("B3", "B4", "B8", "B11", "QA60", "SCL"))
            cs = read("clear") if cs_min is not None else None
        data = (b4 > 0) & (b8 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi.append(np.where(data, (b8 - b4) / (b8 + b4), np.nan))
            ndwi.append(np.where(data & (b3 + b8 > 0), (b3 - b8) / (b3 + b8), np.nan))
            lswi.append(np.where(data & (b8 + b11 > 0), (b8 - b11) / (b8 + b11), np.nan))
        ok.append(data & ~qa60_cloud(qa) & (cs >= cs_min if cs is not None else True))
        scl.append(data & np.isin(sc.astype("int64"), SCL_CLOUD_CLASSES))
    order = np.argsort(dates)
    pick = lambda xs: np.stack(xs)[order]  # noqa: E731
    return (pd.DatetimeIndex(np.array(dates)[order]), pick(ndvi), pick(ndwi), pick(lswi), pick(ok),
            pick(scl), loc)


def window_starts(start: str, end: str, step: int = STEP_DAYS) -> pd.DatetimeIndex:
    """First day of every ``step``-day window from ``start`` up to (not including) ``end``."""
    return pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq=f"{step}D")


def max_value_composite(dates, ndvi, ok, others=(), starts=None, step: int = STEP_DAYS):
    """Per window and pixel, the clear date with the highest NDVI; ``others`` taken from that date.

    Arrays are (dates, pixels). Returns ``(ndvi_w, [other_w, ...], observed)`` shaped
    (windows, pixels), NaN where a window had no clear date.
    """
    dates = pd.DatetimeIndex(dates)
    n_win, n_pix = len(starts), ndvi.shape[1]
    best = np.full((n_win, n_pix), np.nan)
    rest = [np.full((n_win, n_pix), np.nan) for _ in others]
    offset = (dates - starts[0]).days.to_numpy()
    which = np.floor_divide(offset, step)
    for w in range(n_win):
        rows = np.flatnonzero(which == w)
        if not len(rows):
            continue
        cand = np.where(ok[rows], ndvi[rows], -np.inf)
        cand = np.where(np.isfinite(ndvi[rows]), cand, -np.inf)
        top = np.argmax(cand, axis=0)
        seen = np.isfinite(cand[top, np.arange(n_pix)])
        best[w] = np.where(seen, ndvi[rows][top, np.arange(n_pix)], np.nan)
        for out, arr in zip(rest, others):
            out[w] = np.where(seen, arr[rows][top, np.arange(n_pix)], np.nan)
    return best, rest, np.isfinite(best)


def hold_edges(fit, seen):
    """Hold the fit flat before the first and after the last observation.

    Why: with a second-difference penalty the smoother carries the last slope straight on into an
    unobserved end of the series. At the end of the current season that manufactured a rise of 1.0
    NDVI in one month out of nothing — exactly what a newly emerging crop looks like. Flat is the
    honest continuation of "not seen since", and ``gap_days`` says how long that has been.
    """
    fit = np.array(fit, dtype="float64")
    idx = np.flatnonzero(seen)
    if len(idx):
        fit[:idx[0]] = fit[idx[0]]
        fit[idx[-1] + 1:] = fit[idx[-1]]
    return fit


def upper_envelope(y, lmbd: float = LMBD, iterations: int = ENVELOPE_ITERATIONS, dtd=None):
    """Whittaker fit that follows the upper envelope of ``y``. Returns ``(fit, weights)``.

    Observations above the current fit keep weight 1; those below get ``1 - gap / largest gap``
    (Chen et al. 2004). NaNs have weight 0 throughout. The ends are held flat (:func:`hold_edges`).
    """
    y = np.asarray(y, dtype="float64")
    seen = np.isfinite(y)
    weights = seen.astype(float)
    if seen.sum() < 3:
        return np.full(len(y), np.nanmean(y) if seen.any() else np.nan), weights
    fit = op.whittaker(y, lmbd, dtd=dtd, weights=weights)
    for _ in range(iterations):
        below = np.where(seen, fit - y, 0.0)
        worst = below.max()
        if worst <= 0:
            break
        weights = np.where(seen, np.where(below > 0, 1.0 - below / worst, 1.0), 0.0)
        fit = op.whittaker(y, lmbd, dtd=dtd, weights=weights)
    return hold_edges(fit, seen), weights


def choose_lambda(raw, lambdas=(0.5, 1, 2, 5, 10, 20, 50), folds: int = 5, n_pixels: int = 400,
                  seed: int = 0):
    """Pick the smoothing strength by k-fold cross-validation over observed windows.

    Why: the right lambda for single lightly-masked dates is not the one tuned for s2cloudless
    composites, and guessing it is how earlier mistakes were made. For a sample of pixels, each fold
    of observed windows is hidden in turn, the envelope fit is made without it, and the hidden values
    are predicted. The score is the **median** absolute error, because haze that slipped past the
    mask produces large one-sided errors at every lambda and a mean would be driven by them.
    ``raw`` is (windows, pixels). Returns ``(chosen, table)``.
    """
    rng = np.random.default_rng(seed)
    usable = np.flatnonzero(np.isfinite(raw).sum(axis=0) >= 10)
    sample = rng.choice(usable, size=min(n_pixels, len(usable)), replace=False)
    D = np.diff(np.eye(raw.shape[0]), n=2, axis=0)
    dtd = D.T @ D
    rows = []
    for lmbd in lambdas:
        errors = []
        for p in sample:
            y = raw[:, p]
            obs = np.flatnonzero(np.isfinite(y))
            fold = rng.permutation(len(obs)) % folds
            for k in range(folds):
                hidden = obs[fold == k]
                train = y.copy()
                train[hidden] = np.nan
                fit, _ = upper_envelope(train, lmbd, dtd=dtd)
                errors.append(np.abs(fit[hidden] - y[hidden]))
        err = np.concatenate(errors)
        rows.append({"lambda": lmbd, "median_abs_error": float(np.median(err)),
                     "p75_abs_error": float(np.percentile(err, 75))})
    table = pd.DataFrame(rows)
    return float(table.loc[table["median_abs_error"].idxmin(), "lambda"]), table


def gap_days(observed, step: int = STEP_DAYS):
    """Days from each window to the nearest observed window, per pixel; arrays are (windows, pixels).

    Where a pixel was never observed the value is the length of the whole series.
    """
    observed = np.asarray(observed, dtype=bool)
    n = observed.shape[0]
    idx = np.arange(n)[:, None]
    big = n + 1
    last = np.where(observed, idx, -big)
    last = np.maximum.accumulate(last, axis=0)
    nxt = np.where(observed, idx, 2 * big)
    nxt = np.minimum.accumulate(nxt[::-1], axis=0)[::-1]
    dist = np.minimum(idx - last, nxt - idx)
    return np.minimum(dist, n) * step


def build(aoi_id: int, start: str = "2025-09-01", end: str = "2026-09-24", lmbd: float = LMBD,
          out_root="processed/_batch/s2_2026", folder: str = FOLDER, log=print, cs_min: float | None = None) -> dict:
    """Read, composite, fit and write the 5-day series of one AOI. Returns a summary dict.

    ``cs_min`` adds the Cloud Score+ requirement to the mask (see :func:`read_dates`); write such a
    series to its own ``out_root`` so it is never confused with the light-mask one.
    """
    t0 = time.time()
    dates, ndvi, ndwi, lswi, ok, scl, loc = read_dates(aoi_id, folder, cs_min=cs_min)
    h, w = ndvi.shape[1:]
    starts = window_starts(start, end)
    keep = (dates >= starts[0]) & (dates < pd.Timestamp(end))
    flat = lambda a: a[keep].reshape(int(keep.sum()), -1)  # noqa: E731
    t1 = time.time()
    raw, (lswi_w, ndwi_w), observed = max_value_composite(
        dates[keep], flat(ndvi), flat(ok), (flat(lswi), flat(ndwi)), starts)
    D = np.diff(np.eye(len(starts)), n=2, axis=0)
    dtd = D.T @ D
    fit_ndvi = np.full(raw.shape, np.nan)
    fit_lswi = np.full(raw.shape, np.nan)
    for p in range(raw.shape[1]):
        fit_ndvi[:, p], weights = upper_envelope(raw[:, p], lmbd, dtd=dtd)
        if np.isfinite(lswi_w[:, p]).sum() >= 3:
            # LSWI takes the NDVI weights: a date that looked hazy in NDVI is hazy in LSWI too.
            fit_lswi[:, p] = hold_edges(op.whittaker(lswi_w[:, p], lmbd, dtd=dtd, weights=weights),
                                        np.isfinite(lswi_w[:, p]))
    gaps = gap_days(observed)
    t2 = time.time()
    folder_out = Path(out_root) / loc["aoi"]
    names = [d.strftime("%Y-%m-%d") for d in starts]
    written = {}
    for key, cube, dtype in (("ndvi5d", fit_ndvi, "float32"), ("lswi5d", fit_lswi, "float32"),
                             ("ndvi5d_raw", raw, "float32"), ("gapdays5d", gaps, "int16")):
        written[key] = write_stack(cube.reshape(len(starts), h, w), names, loc["grid"],
                                   folder_out / f"{loc['aoi']}_{key}.tif", dtype)
    summary = {
        "aoi": loc["aoi"], "dates_read": int(keep.sum()), "windows": len(starts),
        "observed_window_pct": round(100 * float(observed.mean()), 1),
        # share of all pixel-windows whose value is more than 15 days from any real observation
        "gap_over_15d_pct": round(100 * float((gaps > 15).mean()), 1),
        "max_gap_days": int(gaps.max()),
        "qa60_clear_pct": round(100 * float(flat(ok).mean()), 1),
        "scl_cloud_pct_of_qa60_clear": round(100 * float((flat(scl) & flat(ok)).sum()
                                                         / max(flat(ok).sum(), 1)), 1),
        "seconds_read": round(t1 - t0), "seconds_fit": round(t2 - t1), "seconds_write": round(time.time() - t2),
        **{f"path_{k}": str(v) for k, v in written.items()},
    }
    log(f"{loc['aoi']}: {summary['dates_read']} dates -> {summary['windows']} windows, "
        f"{summary['observed_window_pct']}% of pixel-windows observed, longest gap "
        f"{summary['max_gap_days']} days (read {summary['seconds_read']} s, fit {summary['seconds_fit']} s)")
    return summary


def write_stack(cube, band_names, grid: dict, out_path, dtype="float32"):
    """Write a (bands, rows, cols) cube on the AOI grid, one named band per window."""
    import rasterio
    from rasterio.transform import from_origin

    nodata = -9999 if dtype != "float32" else -9999.0
    data = np.where(np.isfinite(cube), cube, nodata).astype(dtype)
    profile = dict(driver="GTiff", width=cube.shape[2], height=cube.shape[1], count=cube.shape[0],
                   dtype=dtype, crs=grid["crs"], nodata=nodata, compress="deflate", tiled=True,
                   transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(data)
        ds.descriptions = tuple(band_names)
    return out_path


# ---------------------------------------------------------------------------------------------
# Looking at the result
#
# Why: the whole point of the light mask and the envelope fit is a better curve, and that can only
# be judged by looking at curves. These helpers read what ``build`` wrote plus the raw dates, so the
# picture shows exactly what the classifier will see, next to what the satellite actually recorded.
# ---------------------------------------------------------------------------------------------

_CACHE: dict = {}


def load(aoi_id: int, out_root="processed/_batch/s2_2026", folder: str = FOLDER) -> dict:
    """Raw dates and the written stacks of one AOI, read once and kept in memory."""
    import rasterio

    key = (aoi_id, str(out_root))
    if key in _CACHE:
        return _CACHE[key]
    dates, ndvi, _, lswi, ok, scl, loc = read_dates(aoi_id, folder)
    stacks = {}
    for key in ("ndvi5d", "lswi5d", "ndvi5d_raw", "gapdays5d"):
        with rasterio.open(Path(out_root) / loc["aoi"] / f"{loc['aoi']}_{key}.tif") as ds:
            arr = ds.read().astype("float64")
            arr[arr == ds.nodata] = np.nan
            stacks[key] = arr
            windows = pd.to_datetime(list(ds.descriptions))
    _CACHE[key] = dict(dates=dates, ndvi=ndvi, lswi=lswi, ok=ok, scl=scl, loc=loc,
                       windows=windows, **stacks)
    return _CACHE[key]


def forget():
    """Drop cached AOIs, e.g. after running ``build`` again."""
    _CACHE.clear()


def inside_aoi(aoi_id: int) -> np.ndarray:
    """Flat boolean mask of the grid pixels inside the AOI polygons (the grid carries a buffer)."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from .. import config as config_mod

    loc = load(aoi_id)["loc"]
    grid = loc["grid"]
    shapes = gpd.read_file(config_mod.aoi_path(loc["cfg"])).to_crs(grid["crs"]).geometry
    mask = rasterize(((g, 1) for g in shapes), out_shape=(int(grid["height"]), int(grid["width"])),
                     transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]),
                     fill=0, dtype="uint8")
    return mask.astype(bool).ravel()


def pixel(aoi_id: int, pid: int) -> dict:
    """One pixel: every raw date (with its mask flags) and the 5-day series, as two DataFrames."""
    d = load(aoi_id)
    flat = lambda a: a.reshape(a.shape[0], -1)[:, pid]  # noqa: E731
    raw = pd.DataFrame({"date": d["dates"], "ndvi": flat(d["ndvi"]), "lswi": flat(d["lswi"]),
                        "qa60_clear": flat(d["ok"]), "scl_cloud": flat(d["scl"])})
    series = pd.DataFrame({"window": d["windows"], "ndvi_fit": flat(d["ndvi5d"]),
                           "ndvi_composite": flat(d["ndvi5d_raw"]), "lswi_fit": flat(d["lswi5d"]),
                           "gap_days": flat(d["gapdays5d"])})
    return {"aoi": d["loc"]["aoi"], "pid": pid, "raw": raw, "series": series}


def sample_pids(aoi_id: int, per_group: int = 3, groups: int = 4, seed: int = 0) -> list[int]:
    """Pixels inside the AOI from every quarter of the NDVI range, not only the typical ones.

    The range is max minus min of the fitted NDVI over the year: low-range pixels are water,
    settlement or trees, high-range ones carry at least one crop cycle.
    """
    d = load(aoi_id)
    fit = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    spread = np.nanmax(fit, axis=0) - np.nanmin(fit, axis=0)
    candidates = np.flatnonzero(inside_aoi(aoi_id) & np.isfinite(spread))
    edges = np.quantile(spread[candidates], np.linspace(0, 1, groups + 1))
    rng = np.random.default_rng(seed)
    picks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = candidates[(spread[candidates] >= lo) & (spread[candidates] <= hi)]
        picks += sorted(rng.choice(pool, size=min(per_group, len(pool)), replace=False).tolist())
    return picks


def sheet(aoi_id: int, pids, out_path=None, cols: int = 3, lswi: bool = True):
    """Small multiples: one panel per pixel with raw dates, the composite, the fit and gap shading.

    Raw dates are drawn by mask outcome: kept (QA60 clear), kept but SCL would have removed it, and
    removed by QA60. Windows filled from more than 15 days away are shaded, so a filled stretch is
    never mistaken for a measured one.
    """
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, INK_SECONDARY, SERIES_COLORS, SURFACE

    pids = list(pids)
    rows = int(np.ceil(len(pids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 3.3 * rows), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax, pid in zip(axes.ravel(), pids):
        p = pixel(aoi_id, pid)
        raw, s = p["raw"], p["series"]
        ax.set_facecolor(SURFACE)
        long_gap = s["gap_days"] > 15
        for start, filled in zip(s["window"], long_gap):
            if filled:
                ax.axvspan(start, start + pd.Timedelta(days=STEP_DAYS), color=INK_MUTED, alpha=.12, lw=0)
        kept = raw["qa60_clear"] & ~raw["scl_cloud"]
        scl_only = raw["qa60_clear"] & raw["scl_cloud"]
        ax.plot(raw["date"][~raw["qa60_clear"]], raw["ndvi"][~raw["qa60_clear"]], "x", ms=4,
                color=INK_MUTED, label="removed by QA60")
        ax.plot(raw["date"][scl_only], raw["ndvi"][scl_only], "o", ms=4, mfc="none",
                color=SERIES_COLORS[1], label="kept, SCL would remove")
        ax.plot(raw["date"][kept], raw["ndvi"][kept], "o", ms=3.5, color=INK_SECONDARY, label="kept")
        ax.plot(s["window"], s["ndvi_fit"], lw=2.2, color=SERIES_COLORS[2], label="NDVI fit")
        if lswi:
            ax.plot(s["window"], s["lswi_fit"], lw=1.4, color=SERIES_COLORS[0], label="LSWI fit")
            ax.axhline(0, lw=.8, color=INK_MUTED, alpha=.5)
        ax.set_title(f"pid {pid}", fontsize=9, color=INK, loc="left")
        ax.set_ylim(-0.6, 1.0)
        ax.grid(True, alpha=.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[len(pids):]:
        ax.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, frameon=False,
               fontsize=9)
    fig.suptitle(f"{load(aoi_id)['loc']['aoi']}: QA60-masked dates, 5-day max composite, upper-envelope "
                 f"Whittaker (λ={LMBD}). Grey band = filled from > 15 days away.",
                 y=0.995, fontsize=10.5, color=INK)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    return fig
