"""Second investigation of class 3 (standing, rice-like, water unconfirmed): every test we can run without field data.

Why
---
After the v2 water test about 12,000 acres stayed in class 3. The AOIs were chosen as rice areas, so
that much "rice-like but dry" ground needs more than one explanation ruled out. Each test here looks
at the same pixels from a different side, and each is also run on class 1 (rice, water confirmed)
and, where available, on the field plots, so a difference means something:

* **radar window** — the rule reads 5 x 5 means (50 m). Small fields mix with bunds and neighbours;
  the flood is searched again on 3 x 3 and single-pixel series (``flood_by_window``);
* **dark without a drop** — a field wet from weeks of rain is already dark, and transplanting then
  adds no further drop; so: at least two monsoon passes with VH <= -20 dB while the field is bare
  (``dark_passes``);
* **optical water** — any Cloud Score+ clear observation in the bare period with NDWI > 0 or LSWI
  above NDVI (open water or a flooded field), independent of the radar (``optical_water``);
* **shape of the crop cycle** — how long the field stayed bare, how fast and how high the canopy
  rose, when it peaked (``cycle_shape``), compared with class 1 and the plots;
* **field context** — how much class 1 surrounds each pixel (``context``): a class-3 pixel inside
  a confirmed rice field is an edge or a weak-signal part of that field.

Nothing here changes the map.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from . import ndvi_5day as nd
from . import pixel_report as pr
from . import radar_water as rw
from . import sar_curve

SRC = "processed/_batch/s2_2026"


def classes_of(aoi_id: int, suffix: str = "") -> np.ndarray:
    import rasterio

    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{suffix}.tif") as ds:
        return ds.read(1)


def sample(aoi_id: int, n: int = 2000, codes=(1, 3), seed: int = 0) -> pd.DataFrame:
    """Random pixels of each class (all positions), with their field-interior share."""
    from .water_investigation import same_class_share

    c = classes_of(aoi_id)
    share = same_class_share(c).ravel()
    flat = c.ravel()
    rng = np.random.default_rng(seed)
    parts = []
    for code in codes:
        pool = np.flatnonzero(flat == code)
        if len(pool):
            sel = rng.choice(pool, size=min(n, len(pool)), replace=False)
            parts.append(pd.DataFrame({"pixel": sel, "class": code, "same_class_share": share[sel]}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def context(classes_2d, pixels, size: int = 9) -> np.ndarray:
    """Share of class-1 pixels in the ``size`` x ``size`` neighbourhood of each pixel."""
    from scipy.ndimage import uniform_filter

    share = uniform_filter((np.asarray(classes_2d) == 1).astype("float32"), size=size, mode="constant")
    return share.ravel()[np.asarray(pixels)]


def flood_by_window(aoi_id: int, pixels, climb, ndvi, windows, sizes=(5, 3, 1)) -> pd.DataFrame:
    """The v2 flood test on 5x5, 3x3 and single-pixel radar series (thresholds of ``radar_water``).

    ``water_evidence`` works on the whole grid; here the same conditions are applied to a pixel
    subset so the window can change without recomputing an AOI.
    """
    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    pix = np.asarray(pixels).astype(int)
    climb = np.asarray(climb).astype("datetime64[D]")
    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    out = {}
    for size in sizes:
        best = np.full(len(pix), -np.inf)
        when = np.full(len(pix), np.datetime64("NaT"), dtype="datetime64[D]")
        vh_at = np.full(len(pix), np.nan)
        pol_of = np.full(len(pix), "", dtype=object)
        tracks = []
        for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
            dates, cubes = sar_curve.read_track(loc, track, size)
            day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
            flat = {p: cubes[p].reshape(len(dates), -1)[:, pix] for p in ("VV", "VH")}
            drops = {p: rw.local_drops(dates, flat[p]) for p in ("VV", "VH")}
            tracks.append((day, drops))
            rel = (day[:, None] - climb[None, :]) / np.timedelta64(1, "D")
            with np.errstate(invalid="ignore"):
                span = (rel >= rw.LONG_SPAN[0]) & (rel <= rw.LONG_SPAN[1]) & (day >= np.datetime64(rw.FLOOD_EARLIEST))[:, None]
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for p in ("VV", "VH"):
                    m = np.where(span, drops[p], np.nan)
                    b = np.nanmax(m, axis=0)
                    arg = np.nanargmax(np.where(np.isfinite(m), m, -np.inf), axis=0)
                    better = np.isfinite(b) & (b > best)
                    best = np.where(better, b, best)
                    when = np.where(better, day[arg], when)
                    pol_of = np.where(better, p, pol_of)
                    vh_at = np.where(better, flat["VH"][arg, np.arange(len(pix))], vh_at)
        support = np.zeros(len(pix), dtype=int)
        for day, drops in tracks:
            near = np.abs((day[:, None] - when[None, :]) / np.timedelta64(1, "D")) <= 14
            for p in ("VV", "VH"):
                with np.errstate(invalid="ignore"):
                    support += np.where(pol_of == p, (near & (drops[p] >= 2.0)).sum(axis=0), 0)
        has = ~np.isnat(when)
        idx = np.clip(np.searchsorted(win, np.where(has, when, win[0])), 0, len(win) - 1)
        ndvi_at = np.where(has, ndvi[idx, np.arange(len(pix))], np.nan)
        with np.errstate(invalid="ignore"):
            ok = (best >= rw.FLOOD_DROP_MIN) & (vh_at <= rw.FLOOD_VH_MAX) & (ndvi_at <= rw.FLOOD_NDVI_MAX) \
                 & (support >= rw.SUPPORT_MIN)
        out[f"flood_ok_w{size}"] = ok
        out[f"flood_drop_w{size}"] = np.where(np.isfinite(best), best, np.nan)
    return pd.DataFrame(out)


def dark_passes(aoi_id: int, pixels, trough, climb, ndvi, windows, vh_max: float = -20.0,
                window: int = 5) -> pd.DataFrame:
    """Monsoon passes (from ``radar_water.FLOOD_EARLIEST``) between 45 days before the trough and the
    climb with VH <= ``vh_max`` while the fitted NDVI is at most 0.5; count over all tracks."""
    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    pix = np.asarray(pixels).astype(int)
    trough = np.asarray(trough).astype("datetime64[D]")
    climb = np.asarray(climb).astype("datetime64[D]")
    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    count = np.zeros(len(pix), dtype=int)
    darkest = np.full(len(pix), np.inf)
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        vh = cubes["VH"].reshape(len(dates), -1)[:, pix]
        idx = np.clip(np.searchsorted(win, day), 0, len(win) - 1)
        nd_at = ndvi[idx]                                   # (dates, pixels)
        with np.errstate(invalid="ignore"):
            span = ((day[:, None] >= trough[None, :] - np.timedelta64(45, "D")) & (day[:, None] <= climb[None, :])
                    & (day >= np.datetime64(rw.FLOOD_EARLIEST))[:, None] & (nd_at <= 0.5))
            count += (span & (vh <= vh_max)).sum(axis=0)
            darkest = np.fmin(darkest, np.where(span, vh, np.inf).min(axis=0))
    return pd.DataFrame({"dark_passes": count, "darkest_vh_bare": np.where(np.isfinite(darkest), darkest, np.nan)})


def optical_water(aoi_id: int, pixels, trough, climb, cs_min: float = 60) -> pd.DataFrame:
    """Clear Sentinel-2 observations between 45 days before the trough and the climb, and whether
    any of them saw water (NDWI > 0, or LSWI above NDVI with NDVI below 0.3)."""
    dates, ndvi, ndwi, lswi, ok, _, _ = nd.read_dates(aoi_id, cs_min=cs_min)
    pix = np.asarray(pixels).astype(int)
    f = lambda a: a.reshape(len(dates), -1)[:, pix]  # noqa: E731
    ndvi, ndwi, lswi, ok = f(ndvi), f(ndwi), f(lswi), f(ok)
    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    t = np.asarray(trough).astype("datetime64[D]")
    c = np.asarray(climb).astype("datetime64[D]")
    with np.errstate(invalid="ignore"):
        span = (day[:, None] >= t[None, :] - np.timedelta64(45, "D")) & (day[:, None] <= c[None, :])
        seen = span & ok
        water = seen & ((ndwi > 0) | ((lswi > ndvi) & (ndvi < 0.3)))
    return pd.DataFrame({"clear_obs_bare": seen.sum(axis=0), "optical_water_obs": water.sum(axis=0)})


def cycle_shape(ev: pd.DataFrame, ndvi, windows) -> pd.DataFrame:
    """Bare days (the rule's low run x 5), trough-to-climb days, climb rate (NDVI per 10 days over
    the 20 days after the climb), peak NDVI and the peak's day of year."""
    win = pd.DatetimeIndex(windows)
    t = pd.to_datetime(ev["trough_date"]).to_numpy()
    c = pd.to_datetime(ev["climb_date"]).to_numpy()
    ci = np.clip(np.searchsorted(win.to_numpy(), c), 0, len(win) - 1)
    after = np.clip(ci + 4, 0, len(win) - 1)
    cols = np.arange(ndvi.shape[1])
    rate = (ndvi[after, cols] - ndvi[ci, cols]) / 2.0
    peak_i = np.nanargmax(np.where(np.arange(len(win))[:, None] >= ci[None, :], ndvi, -np.inf), axis=0)
    return pd.DataFrame({
        "bare_days": ev["low_windows"].to_numpy() * 5,
        "trough_to_climb_days": (c - t) / np.timedelta64(1, "D"),
        "climb_rate_per_10d": rate,
        "peak_ndvi": ev["peak_after"].to_numpy(),
        "peak_doy": win[peak_i].dayofyear.to_numpy(),
        "climb_doy": pd.DatetimeIndex(c).dayofyear.to_numpy(),
    })


def audit_aoi(aoi_id: int, n: int = 2000, seed: int = 0, out_dir=f"{SRC}/report/class3_audit") -> pd.DataFrame:
    """All tests for a class 1 / class 3 sample of one AOI; written to ``<out_dir>/aoi<N>.parquet``."""
    d, ev, radar = mr.aoi_events(aoi_id)
    c2 = classes_of(aoi_id)
    s = sample(aoi_id, n, seed=seed)
    if s.empty:
        return s
    pix = s["pixel"].to_numpy()
    e = ev.loc[pix].reset_index(drop=True)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)[:, pix]
    trough = e["trough_date"].to_numpy().astype("datetime64[D]")
    climb = pd.to_datetime(e["climb_date"]).to_numpy().astype("datetime64[D]")
    parts = [s, e[["trough_date", "climb_date", "trough_ndvi", "peak_after", "last_ndvi", "low_windows",
                   "VV_dip", "VH_dip", "flood_drop", "flood_vh", "support", "vh_at_trough", "flood_vh_2nd"]]]
    parts.append(pd.DataFrame({"class1_share_9x9": context(c2, pix)}))
    parts.append(flood_by_window(aoi_id, pix, climb, ndvi, d["windows"]))
    parts.append(dark_passes(aoi_id, pix, trough, climb, ndvi, d["windows"]))
    parts.append(optical_water(aoi_id, pix, trough, climb))
    parts.append(cycle_shape(e, ndvi, d["windows"]))
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    out["aoi"] = d["loc"]["aoi"]
    nd.forget()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(out_dir) / f"{d['loc']['aoi']}.parquet")
    return out


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per AOI and class: the share of pixels each test calls wet, and the medians of the cycle shape."""
    rows = []
    for (aoi, code), g in frame.groupby(["aoi", "class"]):
        rows.append({
            "aoi": aoi, "class": code, "n": len(g),
            "interior_pct": round(100 * float((g["same_class_share"] >= 0.999).mean()), 1),
            "in_rice_field_pct": round(100 * float((g["class1_share_9x9"] >= 0.5).mean()), 1),
            "flood_w5_pct": round(100 * float(g["flood_ok_w5"].mean()), 1),
            "flood_w3_pct": round(100 * float(g["flood_ok_w3"].mean()), 1),
            "flood_w1_pct": round(100 * float(g["flood_ok_w1"].mean()), 1),
            "dark2_pct": round(100 * float((g["dark_passes"] >= 2).mean()), 1),
            "optical_water_pct": round(100 * float((g["optical_water_obs"] >= 1).mean()), 1),
            "clear_obs_bare_med": float(g["clear_obs_bare"].median()),
            "bare_days_med": float(g["bare_days"].median()),
            "trough_to_climb_med": float(g["trough_to_climb_days"].median()),
            "climb_rate_med": round(float(g["climb_rate_per_10d"].median()), 3),
            "peak_ndvi_med": round(float(g["peak_ndvi"].median()), 2),
            "climb_doy_med": float(g["climb_doy"].median()),
        })
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.class3_audit",
                                description="Every no-field-data test on class 1 / class 3 samples, per AOI.")
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args(argv)
    for aoi_id in args.ids:
        out = audit_aoi(aoi_id, n=args.n)
        print(f"aoi{aoi_id}: {len(out)} pixels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def alos2_by_class(aoi_id: int, n: int = 300, seed: int = 0, out_dir=f"{SRC}/report/alos2") -> pd.DataFrame:
    """ALOS-2 HH/HV per date for field-interior class 1 and class 3 pixels of one AOI (read-only EE)."""
    from .external_checks import alos2_sample, pixels_to_lonlat
    from .water_investigation import interior_pids

    c = classes_of(aoi_id)
    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    parts = []
    for code in (1, 3):
        pids = interior_pids(c, code, n, seed=seed)
        if not len(pids):
            continue
        lon, lat = pixels_to_lonlat(loc["grid"], pids)
        t = alos2_sample(lon, lat)
        t["class"] = code
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["aoi"] = f"aoi{aoi_id}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out.to_parquet(Path(out_dir) / f"aoi{aoi_id}.parquet")
    return out


def null_tests(aoi_id: int, pixels, seed: int = 0, span=("2026-07-01", "2026-09-15")) -> pd.DataFrame:
    """The same extra tests on pixels that are never flooded paddies, with random climb dates.

    Why: a test that is loose enough to find water on class 3 must be shown not to find water
    everywhere. Negative pixels (evergreen, bare/built, cut; ``validation.reference_sets``) with a
    random climb (trough 40 days earlier) give its false-alarm rate.
    """
    d = nd.load(aoi_id)
    pix = np.asarray(pixels).astype(int)
    rng = np.random.default_rng(seed)
    days = pd.date_range(*span, freq="D").to_numpy().astype("datetime64[D]")
    climb = rng.choice(days, size=len(pix))
    trough = climb - np.timedelta64(40, "D")
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)[:, pix]
    parts = [flood_by_window(aoi_id, pix, climb, ndvi, d["windows"]),
             dark_passes(aoi_id, pix, trough, climb, ndvi, d["windows"]),
             optical_water(aoi_id, pix, trough, climb)]
    nd.forget()
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1).assign(pixel=pix)
