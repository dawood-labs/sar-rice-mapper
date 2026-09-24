"""One smoothed radar curve per pixel, on the same grid as the optical one.

Why
---
Everything learned about this AOI so far is phrased in terms of a smoothed curve read off a regular
5-day grid. To ask whether the radar sees what the optical sees, the radar has to be brought onto
exactly that footing: same grid, same kind of fit, same viewer. Three things have to be handled that
NDVI did not need.

**Two tracks, not one.** Only the primary track was ever used, which threw away nearly half the
acquisitions. Over this AOI the ascending track carries 44 acquisitions at a 6-day revisit and the
descending one 34 at 12 days; **together they give 78 dates at a 3-day median gap** — a better
cadence than the optical composites, and never interrupted by cloud. They cannot simply be
concatenated: their incidence angles differ by 5.6 degrees (39.1 vs 33.5) and they look from
opposite sides, so each carries its own offset and a naive merge produces a 12-day sawtooth that is
geometry, not crop. Because the two tracks pass within a few days of each other many times a year,
the offset is measured **per pixel from those near-coincident pairs** and removed, which is a paired
estimate and so cannot be skewed by one track sampling a different part of the season.

**Average in power, not in decibels.** Backscatter is a power; the mean of decibels is not the
decibel of the mean and sits systematically low. Every average and every fit here happens in linear
power and is converted back at the end.

**The smoothing strength is measured, not chosen.** Nobody knows the right Whittaker lambda for a
radar series, and guessing it is how the earlier mistakes were made. :func:`choose_lambda` holds out
each observation in turn, fits the rest, and scores the prediction; the lambda with the lowest
held-out error wins. ``d = 2`` stays as it is for NDVI.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import pixel_report as pr
from . import seasonal_stats as ss

POLS = ("VH", "VV")
#: Two acquisitions this many days apart or less count as near-coincident for measuring the offset.
PAIR_DAYS = 3
#: Lambdas tried by the cross-validation, log-spaced from "barely any" to "a lot".
LAMBDA_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def read_track(loc: dict, track: str, window: int = 5):
    """One track's VH and VV over the whole AOI, in dB, as ``window`` x ``window`` power means.

    The spatial mean is taken in linear power over the same 5x5 box the classifier uses, which cuts
    speckle from about 2 dB on a single pixel to about 1 dB and is what makes a per-pixel curve
    readable at all.
    """
    import rasterio
    from scipy.ndimage import uniform_filter

    stack = Path(loc["run"]) / "stack" / f"track_{track}"
    out, dates = {}, None
    for pol in POLS:
        with rasterio.open(stack / f"stack_{pol}.vrt") as ds:
            cube = ds.read().astype("float32")
            if ds.nodata is not None:
                cube[cube == ds.nodata] = np.nan
            dates = [dt.datetime.strptime(d.rsplit("_", 1)[1], "%Y%m%d").date() for d in ds.descriptions]
        for i in bad_pass_indices(loc, track, pol, dates):
            cube[i] = np.nan                  # an artefact pass (docs/15): unusable, not a field event
        power = ss.to_linear(cube)
        smoothed = np.stack([uniform_filter(p, size=window, mode="nearest") for p in power])
        out[pol] = ss.to_db(smoothed)
    return pd.DatetimeIndex(pd.to_datetime(dates)), out


#: Passes found to be artefacts (``final_audit.bad_passes``: a jump of 3 dB or more on ground that
#: cannot change within a week). Local file, gitignored with everything under processed/.
BAD_PASSES = "processed/_batch/s2_2026/report/final_audit/bad_passes_all.csv"
_BAD_CACHE: dict = {}


def bad_pass_indices(loc: dict, track: str, pol: str, dates) -> list[int]:
    """Indices of ``dates`` that are recorded as artefact passes for this AOI, track and polarisation."""
    from .. import config as config_mod

    path = config_mod.repo_root() / BAD_PASSES
    if "table" not in _BAD_CACHE:
        _BAD_CACHE["table"] = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["aoi", "track", "pol", "date", "bad"])
    b = _BAD_CACHE["table"]
    if b.empty:
        return []
    sel = b[(b["bad"]) & (b["aoi"] == loc.get("aoi")) & (b["track"] == track) & (b["pol"] == pol)]
    bad = set(pd.to_datetime(sel["date"]).dt.date)
    return [i for i, d in enumerate(dates) if d in bad]


def pair_offsets(dates_a, cube_a, dates_b, cube_b, pair_days: int = PAIR_DAYS):
    """Per-pixel dB offset of track B relative to track A, from their near-coincident passes.

    Returns ``(offset, n_pairs)``. Only dates within ``pair_days`` of each other are used, so the
    estimate compares the same ground in the same condition and is not biased by either track
    sampling a different part of the season.
    """
    a_days = dates_a.values.astype("datetime64[D]").astype(int)
    b_days = dates_b.values.astype("datetime64[D]").astype(int)
    pairs = [(i, int(np.argmin(np.abs(b_days - d))))
             for i, d in enumerate(a_days)
             if np.min(np.abs(b_days - d)) <= pair_days]
    if not pairs:
        raise ValueError("the two tracks never pass within the pairing window")
    diffs = np.stack([cube_a[i] - cube_b[j] for i, j in pairs])
    with np.errstate(invalid="ignore"):
        return np.nanmedian(diffs, axis=0), len(pairs)


_MERGED: dict[tuple, dict] = {}


def merge_tracks(loc: dict, window: int = 5, pair_days: int = PAIR_DAYS):
    """Both tracks as one series in dB, the secondary shifted onto the primary. Returns a dict."""
    key = (loc["aoi"], window, pair_days)
    if key in _MERGED:
        return _MERGED[key]
    primary, secondary = loc["primary"], loc["secondary"]
    dates_p, cube_p = read_track(loc, primary, window)
    if not secondary:
        order = np.argsort(dates_p)
        return {"dates": dates_p[order], "cubes": {p: cube_p[p][order] for p in POLS},
                "tracks": np.array([primary] * len(dates_p))[order], "offsets": {}, "n_pairs": 0}
    dates_s, cube_s = read_track(loc, secondary, window)
    offsets, n_pairs = {}, 0
    shifted = {}
    for pol in POLS:
        offsets[pol], n_pairs = pair_offsets(dates_p, cube_p[pol], dates_s, cube_s[pol], pair_days)
        shifted[pol] = cube_s[pol] + offsets[pol][None, :, :]
    dates = dates_p.append(dates_s)
    tracks = np.array([primary] * len(dates_p) + [secondary] * len(dates_s))
    order = np.argsort(dates.values)
    _MERGED[key] = {"dates": dates[order], "tracks": tracks[order], "offsets": offsets,
                    "n_pairs": n_pairs,
                    "cubes": {p: np.concatenate([cube_p[p], shifted[p]])[order] for p in POLS}}
    return _MERGED[key]


def to_windows(dates, cube, window_starts, step_days: int = 5):
    """Average the acquisitions falling in each fixed window, in linear power. NaN where none do."""
    day = dates.values.astype("datetime64[D]").astype(int)
    edges = window_starts.values.astype("datetime64[D]").astype(int)
    out = np.full((len(edges),) + cube.shape[1:], np.nan, dtype="float32")
    power = ss.to_linear(cube)
    for k, start in enumerate(edges):
        inside = (day >= start) & (day < start + step_days)
        if inside.any():
            with np.errstate(invalid="ignore"):
                out[k] = ss.to_db(np.nanmean(power[inside], axis=0))
    return out


def whittaker_db(values, lmbd: float, dtd=None):
    """Whittaker fit of a dB series, carried out in linear power and returned in dB.

    The fit is not constrained to stay positive, and where it overshoots below zero power there is
    no decibel value to report. Those samples come back as **NaN**, not as a floor: clipping them to
    a tiny power produced -80 dB spikes, which a median ignored but a minimum picked up, and a
    feature built on the minimum then collapsed onto a handful of pixels.
    """
    from . import optical_phenology as op

    power = ss.to_linear(np.asarray(values, dtype="float64"))
    fitted = op.whittaker(power, lmbd, dtd=dtd)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(fitted > 0, ss.to_db(np.where(fitted > 0, fitted, np.nan)), np.nan)


def choose_lambda(series, candidates=LAMBDA_GRID, max_pixels: int = 200, seed: int = 0,
                  rule: str = "one_se"):
    """Leave-one-out cross-validation over observed windows; returns ``(chosen, table)``.

    ``series`` is (windows, pixels) in dB with NaN where a window holds no acquisition. Each observed
    value is hidden in turn, the curve refitted, and the hidden value predicted; the error is scored
    in dB. Nothing about the radar's noise level is assumed in advance.

    Two rules are offered because they answer different questions. ``"min"`` takes the lambda with
    the lowest held-out error — the best *predictor* of a noisy observation, which for a noisy series
    always leans towards heavy smoothing and rounds off the very edges the crop calendar is read
    from. ``"one_se"`` (the default) takes the **smallest** lambda whose error is still within one
    standard error of that minimum: statistically no worse, and it keeps the steepest part of the
    harvest fall. The table is returned either way so the choice stays visible.
    """
    from . import optical_phenology as op

    series = np.asarray(series, dtype="float64")
    rng = np.random.default_rng(seed)
    usable = np.where(np.isfinite(series).sum(axis=0) >= 10)[0]
    if not len(usable):
        raise ValueError("no pixel has enough observed windows to cross-validate")
    picks = rng.choice(usable, size=min(max_pixels, len(usable)), replace=False)
    n = series.shape[0]
    D = np.diff(np.eye(n), n=2, axis=0)
    dtd = D.T @ D
    rows = []
    for lmbd in candidates:
        errors = []
        for p in picks:
            column = series[:, p]
            seen = np.where(np.isfinite(column))[0]
            for i in seen:
                held = column.copy()
                held[i] = np.nan
                fit = whittaker_db(held, lmbd, dtd=dtd)
                errors.append(fit[i] - column[i])
        errors = np.asarray(errors)
        squared = errors ** 2
        rows.append({"lambda": lmbd, "rmse_db": float(np.sqrt(squared.mean())),
                     "se_db": float(squared.std(ddof=1) / np.sqrt(len(squared))
                                    / (2 * np.sqrt(squared.mean()))),
                     "bias_db": float(errors.mean()), "n": int(len(errors))})
    table = pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
    best_row = table.loc[table["rmse_db"].idxmin()]
    if rule == "min":
        return float(best_row["lambda"]), table
    within = table[table["rmse_db"] <= best_row["rmse_db"] + best_row["se_db"]]
    return float(within["lambda"].min()), table


def aoi_curves(aoi_id: int, window_starts=None, step_days: int = 5, window: int = 5,
               lmbd: float | None = None, start="2025-03-01", end="2026-01-31"):
    """Smoothed VH, VV and VH-VV for every pixel of an AOI, on the optical 5-day grid.

    ``lmbd`` defaults to whatever :func:`choose_lambda` measures for this AOI.
    """
    from . import optical_phenology as op

    loc = pr.locate(aoi_id, 0)
    merged = merge_tracks(loc, window)
    if window_starts is None:
        window_starts = pd.date_range(start, end, freq=f"{step_days}D")
    shape = merged["cubes"]["VH"].shape[1:]
    binned = {p: to_windows(merged["dates"], merged["cubes"][p], window_starts, step_days)
              for p in POLS}
    flat = {p: binned[p].reshape(len(window_starts), -1) for p in POLS}
    if lmbd is None:
        lmbd, table = choose_lambda(flat["VH"])
    else:
        table = None
    D = np.diff(np.eye(len(window_starts)), n=2, axis=0)
    dtd = D.T @ D
    smooth = {p: np.full_like(flat[p], np.nan) for p in POLS}
    for p in POLS:
        for i in range(flat[p].shape[1]):
            column = flat[p][:, i]
            if np.isfinite(column).sum() >= 5:
                smooth[p][:, i] = whittaker_db(column, lmbd, dtd=dtd)
    return {"dates": window_starts, "shape": shape, "aoi": loc["aoi"], "lmbd": lmbd,
            "cv_table": table, "n_pairs": merged["n_pairs"], "offsets": merged["offsets"],
            "observed": {p: np.isfinite(flat[p]) for p in POLS},
            "raw": flat, "smooth": smooth,
            "acquisitions": {"dates": merged["dates"], "tracks": merged["tracks"]}}


def raw_pixel(loc_or_aoi, pid: int, window: int = 5, pair_days: int = PAIR_DAYS) -> pd.DataFrame:
    """Every individual acquisition at one pixel: VH, VV, VH - VV in dB, and which track it came from.

    This is the radar equivalent of the optical "observed" points, and it is deliberately *not* the
    5-day window means: those are already an average, and plotting them as observations hides both
    the real scatter and which track each point came from.
    """
    loc = pr.locate(loc_or_aoi, 0) if not isinstance(loc_or_aoi, dict) else loc_or_aoi
    merged = merge_tracks(loc, window, pair_days)
    row, col = divmod(int(pid), int(loc["grid"]["width"]))
    out = pd.DataFrame({"date": merged["dates"], "track": merged["tracks"]})
    for pol in POLS:
        out[pol] = merged["cubes"][pol][:, row, col]
    out["VHmVV"] = out["VH"] - out["VV"]
    return out.reset_index(drop=True)


#: AOI radar curves already built, keyed by the settings used. Cleared with :func:`forget`.
_CACHE: dict[tuple, dict] = {}


def load(aoi_id: int, window_starts=None, step_days: int = 5, window: int = 5,
         lmbd: float | None = None, start="2025-03-01", end="2026-01-31"):
    """:func:`aoi_curves`, built once per AOI and kept in memory so pixels can be flicked through."""
    key = (aoi_id, step_days, window, lmbd, start, end,
           None if window_starts is None else (window_starts[0], window_starts[-1], len(window_starts)))
    if key not in _CACHE:
        _CACHE[key] = aoi_curves(aoi_id, window_starts, step_days, window, lmbd, start, end)
    return _CACHE[key]


def forget():
    """Drop the cached radar curves and merged cubes, e.g. after re-running the stack."""
    _CACHE.clear()
    _MERGED.clear()


def pixel(aoi_id: int, pid: int, **kw) -> pd.DataFrame:
    """One pixel's radar windows: observed and smoothed VH, VV and VH - VV, all in dB."""
    curves = load(aoi_id, **kw)
    out = pd.DataFrame({"date": curves["dates"]})
    for pol in POLS:
        out[f"{pol}_obs"] = curves["raw"][pol][:, pid]
        out[pol] = curves["smooth"][pol][:, pid]
    out["VHmVV_obs"] = out["VH_obs"] - out["VV_obs"]
    out["VHmVV"] = out["VH"] - out["VV"]
    out.attrs.update(lmbd=curves["lmbd"], n_pairs=curves["n_pairs"])
    return out
