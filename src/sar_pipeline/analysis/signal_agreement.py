"""Do the optical (NDVI) and the radar (VV / VH) tell the same story over one AOI?

Why (user, 5 Oct 2026, aoi63): "over the whole AOI the NDVI keeps rising while VV and VH swing a lot; the NDVI and the
radar do not justify each other". Before changing any rule we need to know which side is weak in that AOI:

* the NDVI curve is a fit between clear views. With few clear views in the monsoon its rise is an interpolation, not
  an observation (``longest_gap_days``, ``clear_views``);
* the radar has two tracks (ascending / descending) with different look angles; each track is read on its own, and
  its pass-to-pass swing (``pass_wobble``) is compared with an accepted AOI's, so "a lot" becomes a number;
* agreement: in a growing rice field VH rises with the NDVI. The rank correlation between the fitted NDVI on the radar
  dates and VH (per pixel, per track) shows how often the two move together (``rank_corr``).

Nothing here changes a map; ``compare`` prints the AOI side by side with accepted AOIs.

    python -m sar_pipeline.analysis.signal_agreement --aoi 63 --against 39 116
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

#: The monsoon part of the season, where the rice is grown and the clouds are.
START, END = "2026-06-01", "2026-09-30"


def longest_gap_days(dates, ok) -> np.ndarray:
    """Per pixel, the longest run of days without a clear view between ``START`` and the newest date (``ok``:
    (dates, pixels) bool). A pixel with no clear view gets the whole period."""
    d = pd.DatetimeIndex(dates)
    keep = (d >= pd.Timestamp(START)) & (d <= pd.Timestamp(END))
    d, ok = d[keep], np.asarray(ok)[keep]
    day = lambda x: (pd.DatetimeIndex(x) - pd.Timestamp("1970-01-01")).days.to_numpy()   # noqa: E731
    edges = np.r_[day([START]), day(d), day([END])]
    out = np.zeros(ok.shape[1])
    last = np.full(ok.shape[1], edges[0], dtype="int64")
    for i in range(len(d)):
        out = np.maximum(out, np.where(ok[i], edges[i + 1] - last, 0))
        last = np.where(ok[i], edges[i + 1], last)
    return np.maximum(out, edges[-1] - last)


def pass_wobble(cube) -> np.ndarray:
    """Per pixel, the median absolute change (dB) from one pass to the next of the same track ((passes, pixels))."""
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.abs(np.diff(np.asarray(cube, dtype="float64"), axis=0)), axis=0)


def rank_corr(a, b) -> np.ndarray:
    """Per pixel Spearman correlation between two (times, pixels) arrays; NaN where fewer than 5 common times."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    both = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(both, a, np.nan), np.where(both, b, np.nan)
    ra = pd.DataFrame(a).rank(axis=0).to_numpy(copy=True)
    rb = pd.DataFrame(b).rank(axis=0).to_numpy(copy=True)
    with np.errstate(all="ignore"):
        ra -= np.nanmean(ra, axis=0)
        rb -= np.nanmean(rb, axis=0)
        r = np.nansum(ra * rb, axis=0) / np.sqrt(np.nansum(ra ** 2, axis=0) * np.nansum(rb ** 2, axis=0))
    return np.where(both.sum(axis=0) >= 5, r, np.nan)


def aoi_summary(aoi: int) -> dict:
    """The AOI's numbers (medians over its pixels that have data; ``rice_only``: only pixels the AOI's current rule map
    calls rice, so trees and water do not dilute the agreement)."""
    from pathlib import Path

    import rasterio

    from . import ndvi_5day as nd
    from . import radar_water as rw

    d = nd.load(aoi, out_root=nd.analysis_series_root(aoi))
    dates = pd.DatetimeIndex(d["dates"])
    ok = d["ok"].reshape(len(dates), -1)
    w = pd.DatetimeIndex(d["windows"])
    fit = d["ndvi5d"].reshape(len(w), -1)
    inside = np.isfinite(fit).any(axis=0)
    cls = Path("processed/_batch/s2_2026/rice_fresh") / f"aoi{aoi}" / f"aoi{aoi}_rel_class.tif"
    rice = inside.copy()
    if cls.exists():
        with rasterio.open(cls) as ds:
            c = ds.read(1).ravel()
        inside &= (c != 255) & (c != 0)                                   # the AOI polygon only, not its bounding box
        rice = np.isin(c, (1, 3, 7)) & inside                             # direct seeded, young, transplanted
    mon = (dates >= pd.Timestamp(START)) & (dates <= pd.Timestamp(END))
    clear = ok[mon][:, inside].mean(axis=1)
    out = {"aoi": aoi, "pixels": int(inside.sum()),
           "dates Jun-Sep >= 50 % clear": ", ".join(f"{x:%d %b}" for x, s in zip(dates[mon], clear) if s >= 0.5),
           "clear views Jun-Sep (median px)": float(np.median(ok[mon][:, inside].sum(axis=0))),
           "longest gap days (median px)": float(np.median(longest_gap_days(dates, ok[:, inside])))}
    nd.forget()
    wd = (w - pd.Timestamp("1970-01-01")).days.to_numpy()
    for j, (rd, flat) in enumerate(rw.read_series(aoi)):
        rd = pd.DatetimeIndex(rd)
        m = (rd >= pd.Timestamp(START)) & (rd <= pd.Timestamp(END))
        rdd = (rd[m] - pd.Timestamp("1970-01-01")).days.to_numpy()
        # fitted NDVI on each radar date (linear between 5-day windows)
        idx = np.clip(np.searchsorted(wd, rdd), 1, len(wd) - 1)
        t = ((rdd - wd[idx - 1]) / (wd[idx] - wd[idx - 1]))[:, None]
        nd_at = fit[idx - 1] * (1 - t) + fit[idx] * t
        for pol in ("VV", "VH"):
            cube = flat[pol][m]
            out[f"track {j + 1} {pol} wobble dB"] = round(float(np.nanmedian(pass_wobble(cube)[inside])), 2)
        r = rank_corr(nd_at[:, rice], flat["VH"][m][:, rice])
        out[f"track {j + 1} NDVI~VH corr (rice px)"] = round(float(np.nanmedian(r)), 2)
        out[f"track {j + 1} share corr>0.3"] = round(float(np.nanmean(r > 0.3)), 2)
    return out


def compare(aoi: int, against=(39,)) -> pd.DataFrame:
    """One column per AOI: the AOI under review next to accepted ones."""
    return pd.DataFrame([aoi_summary(a) for a in (aoi, *against)]).set_index("aoi").T


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aoi", type=int, required=True)
    ap.add_argument("--against", type=int, nargs="*", default=[39])
    args = ap.parse_args(argv)
    print(compare(args.aoi, args.against).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
