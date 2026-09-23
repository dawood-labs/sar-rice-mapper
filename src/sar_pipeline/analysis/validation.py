"""An independent validation of the standing-rice rule against the field plots.

Why
---
Before the rule is run over every AOI, it has to be shown — with numbers, not confidence — that it
maps rice the way the ground says rice is. Three things were never tested properly before:

* **False positives.** Only rice was ever measured (the field plots). A rule that calls everything
  rice would score 100 % on them. So negative reference sets are built from the same AOIs: pixels
  that are certainly *not* standing rice on the map date — evergreen cover, permanent water, bare or
  built ground, and fields whose crop was cut before the map date.
* **Circularity.** The thresholds were read off the plots and the recall was then measured on the
  same plots. Here they are re-derived on two regions and tested on the third
  (``leave_one_region_out``), and every threshold gets a sensitivity curve.
* **The radar dip's null.** A 3 dB dip was called "water". How often does a 3 dB dip happen at a
  random date on ground that is not being flooded? The null distribution on the negative sets
  answers that and sets the threshold from data.

Two checks on the ground data itself: **registration** (are the plot polygons on the fields they
mean? shifting them 10-20 m must make their inside less coherent than the real outline), and the
**delivery date** (what the plots looked like on the day the survey said "standing rice").

Everything is read from the same fitted 5-day series and season radar runs the rule uses; nothing
new is exported. Areas in acres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from . import ndvi_5day as nd
from . import pixel_report as pr
from . import radar_water as rw
from . import sar_curve
from .plot_curves import interior_pixels, plot_pixels



def region_names(path="config/regions_monsoon2026.yaml") -> dict:
    """``{"aoi20": "<region>", ...}`` from a local, gitignored file; the AOI key itself when absent.

    Region names are client data and never enter the repository; the code only needs *some* grouping
    label, and falls back to the AOI key, which groups nothing but still works.
    """
    import yaml

    from .. import config as config_mod

    file = config_mod.repo_root() / path
    if not file.exists():
        return {}
    return {str(k): str(v) for k, v in (yaml.safe_load(file.read_text()) or {}).items()}


REGION = region_names()

#: Negative references, from the whole series (Sep 2025 - Sep 2026): the share of windows a pixel
#: must spend above / below a level.
EVERGREEN_MIN_SHARE, EVERGREEN_NDVI = 0.90, 0.50
WATER_MIN_SHARE, WATER_NDVI = 0.80, 0.10
BARE_MAX_NDVI = 0.30
#: "Cut before the map date": a canopy in June-August, gone on the last window.
CUT_PEAK_MIN, CUT_LAST_MAX = 0.50, 0.30


def reference_sets(aoi_id: int, plots=None, out_root="processed/_batch/s2_2026") -> pd.DataFrame:
    """Pixels of one AOI with a reference label: rice plot interior / edge, or a negative class.

    Negatives are defined from the series alone (no rule involved), inside the AOI polygon, and
    never on a plot pixel. Returns ``pixel``, ``set``, ``plot_id`` (plots only).
    """
    d = nd.load(aoi_id, out_root=out_root)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    windows = pd.DatetimeIndex(d["windows"])
    inside = nd.inside_aoi(aoi_id)
    rows = []
    on_plot = np.zeros(ndvi.shape[1], dtype=bool)
    if plots is not None and len(plots):
        pix = plot_pixels(plots, d["loc"]["grid"])
        interior = interior_pixels(pix, int(d["loc"]["grid"]["width"]))
        for pid, p in pix.items():
            on_plot[p] = True
            for q, inner in zip(p, interior[pid]):
                rows.append({"pixel": int(q), "set": "rice_plot_interior" if inner else "rice_plot_edge", "plot_id": pid})
    with np.errstate(invalid="ignore"):
        share_green = np.nanmean(ndvi >= EVERGREEN_NDVI, axis=0)
        share_water = np.nanmean(ndvi < WATER_NDVI, axis=0)
        peak_all = np.nanmax(ndvi, axis=0)
        summer = (windows >= "2026-06-01") & (windows < "2026-09-01")
        peak_summer = np.nanmax(ndvi[summer], axis=0)
    last = ndvi[-1]
    candidates = inside & ~on_plot & np.isfinite(last)
    for name, mask in (("evergreen", share_green >= EVERGREEN_MIN_SHARE),
                       ("water", share_water >= WATER_MIN_SHARE),
                       ("bare_or_built", peak_all < BARE_MAX_NDVI),
                       ("cut_before_map_date", (peak_summer >= CUT_PEAK_MIN) & (last < CUT_LAST_MAX))):
        for q in np.flatnonzero(candidates & mask):
            rows.append({"pixel": int(q), "set": name, "plot_id": None})
    out = pd.DataFrame(rows)
    out["aoi"] = d["loc"]["aoi"]
    out["region"] = REGION.get(d["loc"]["aoi"], d["loc"]["aoi"])
    return out


def event_table(aoi_id: int, refs: pd.DataFrame, season=mr.SEASON, radar_season="monsoon2026",
                out_root="processed/_batch/s2_2026", window: int = 5) -> pd.DataFrame:
    """The rule's per-pixel events plus per-track radar dips, for the reference pixels of one AOI.

    Adds what the rule does not keep: the raw (un-fitted) NDVI minimum within 10 days of the fitted
    trough, whether the trough window was observed, and the dip on every track separately.
    """
    d = nd.load(aoi_id, out_root=out_root)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    raw = d["ndvi5d_raw"].reshape(d["ndvi5d_raw"].shape[0], -1)
    gaps = d["gapdays5d"].reshape(d["gapdays5d"].shape[0], -1)
    windows = pd.DatetimeIndex(d["windows"])
    pix = refs["pixel"].to_numpy()
    ev = mr.pixel_events(ndvi[:, pix], lswi[:, pix], windows, season)
    tidx = np.searchsorted(windows.to_numpy(), ev["trough_date"].to_numpy().astype("datetime64[ns]"))
    tidx = np.clip(tidx, 0, len(windows) - 1)
    ev["trough_gap_days"] = gaps[:, pix][tidx, np.arange(len(pix))]
    near = np.abs(np.arange(len(windows))[:, None] - tidx[None, :]) <= 3
    with np.errstate(all="ignore"):
        ev["raw_min_near_trough"] = np.nanmin(np.where(near, raw[:, pix], np.nan), axis=0)
    # How long the field stayed low around the trough, on the fit, and how many real (un-filled)
    # observations within 15 days were low too. A puddled field sits low for weeks and is seen low;
    # a haze dip is one window, often with no low observation behind it.
    sub_n, cols = ndvi[:, pix], np.arange(len(pix))
    low_fit = sub_n <= mr.TROUGH_MAX + 0.05
    run = np.ones(len(pix), dtype=int)
    for direction in (-1, 1):
        still = np.ones(len(pix), dtype=bool)
        for k in range(1, 9):
            j = np.clip(tidx + direction * k, 0, len(windows) - 1)
            still &= (tidx + direction * k >= 0) & (tidx + direction * k < len(windows)) & low_fit[j, cols]
            run += still
    ev["low_windows"] = run
    ev["raw_obs_low"] = np.nansum(near & (raw[:, pix] <= mr.TROUGH_MAX + 0.05), axis=0)
    ev["raw_obs_near"] = np.isfinite(np.where(near, raw[:, pix], np.nan)).sum(axis=0)
    # radar, per track
    trough = np.where(ev["valid"].to_numpy(), ev["trough_date"].to_numpy().astype("datetime64[D]"), np.datetime64("NaT"))
    if rw.season_run_exists(aoi_id, radar_season):
        loc = pr.locate(aoi_id, 0, season_key=radar_season)
        tracks = [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]
        for track in tracks:
            dates, cubes = sar_curve.read_track(loc, track, window)
            day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
            gap = (day[:, None] - trough[None, :]) / np.timedelta64(1, "D")
            with np.errstate(invalid="ignore"):
                later = (gap >= 25) & (gap <= 70)
            for pol in ("VV", "VH"):
                cube = cubes[pol].reshape(len(dates), -1)[:, pix]
                dry, flood, dip = rw.dips_for_track(dates, cube, trough)
                ev[f"{pol}_dip_{track}"] = dip
                ev[f"{pol}_dry_{track}"] = dry
                # the canopy climbing out of the water: the V-shape a harvest alone never makes
                with np.errstate(all="ignore"):
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        after = np.nanmax(np.where(later, cube, np.nan), axis=0)
                ev[f"{pol}_rise_after_{track}"] = after - flood
        for pol in ("VV", "VH"):
            cols_ = [f"{pol}_dip_{t}" for t in tracks]
            ev[f"{pol}_dip"] = ev[cols_].max(axis=1)
            ev[f"{pol}_dip_min_track"] = ev[cols_].min(axis=1)
            ev[f"{pol}_rise_after"] = ev[[f"{pol}_rise_after_{t}" for t in tracks]].max(axis=1)
        ev["n_tracks"] = len(tracks)
    out = pd.concat([refs.reset_index(drop=True), ev.reset_index(drop=True)], axis=1)
    return out


def classify_with(ev: pd.DataFrame, trough_max=mr.TROUGH_MAX, rise_min=mr.RISE_MIN, canopy_min=mr.CANOPY_MIN,
                  dip_min=rw.DIP_MIN_DB, fall_max=mr.STANDING_FALL_MAX, require_water: bool = True,
                  both_tracks: bool = False, low_windows_min: int = mr.LOW_WINDOWS_MIN) -> np.ndarray:
    """True where the rule with these parameters calls the pixel standing rice."""
    grown = (ev["trough_ndvi"] <= trough_max) & (ev["rise"] >= rise_min) & (ev["peak_after"] >= canopy_min)
    if "low_windows" in ev:
        grown &= ev["low_windows"] >= low_windows_min
    standing = (ev["last_ndvi"] >= canopy_min) & (ev["last_ndvi"] >= ev["peak_after"] - fall_max)
    if not require_water:
        return (grown & standing & ev["valid"]).to_numpy()
    if both_tracks and "VV_dip_min_track" in ev:
        wet = (ev["VV_dip_min_track"] >= dip_min) | (ev["VH_dip_min_track"] >= dip_min)
    else:
        wet = (ev["VV_dip"] >= dip_min) | (ev["VH_dip"] >= dip_min)
    return (grown & standing & ev["valid"] & wet.fillna(False)).to_numpy()


def score(ev: pd.DataFrame, called: np.ndarray) -> pd.DataFrame:
    """Share of each reference set (and region) the rule calls standing rice."""
    frame = ev[["set", "region"]].copy()
    frame["called"] = called
    return (frame.groupby(["set", "region"]).agg(pixels=("called", "size"), called_pct=("called", lambda v: round(100 * v.mean(), 1)))
            .reset_index())


def sensitivity(ev: pd.DataFrame, name: str, values, **fixed) -> pd.DataFrame:
    """Recall on plot-interior pixels and false-positive rate on negatives as one threshold varies."""
    pos = ev["set"] == "rice_plot_interior"
    neg = ev["set"].isin(("evergreen", "water", "bare_or_built", "cut_before_map_date"))
    rows = []
    for v in values:
        called = classify_with(ev, **{name: v}, **fixed)
        rows.append({name: v, "recall_pct": round(100 * called[pos].mean(), 1),
                     "false_positive_pct": round(100 * called[neg].mean(), 1) if neg.any() else None})
    return pd.DataFrame(rows)


def derive_thresholds(ev: pd.DataFrame, low_pct: float = 10, high_pct: float = 90) -> dict:
    """Thresholds read off the plot-interior pixels of ``ev`` alone (percentiles, no tuning)."""
    r = ev[ev["set"] == "rice_plot_interior"]
    return {"trough_max": round(float(np.nanpercentile(r["trough_ndvi"], high_pct)), 2),
            "rise_min": round(float(np.nanpercentile(r["rise"], low_pct)), 2),
            "canopy_min": round(float(np.nanpercentile(r["peak_after"], low_pct)), 2),
            "dip_min": round(float(np.nanpercentile(np.fmax(r["VV_dip"], r["VH_dip"]), low_pct)), 1)}


def leave_one_region_out(ev: pd.DataFrame, regions=None) -> pd.DataFrame:
    """Derive thresholds on all but one region, test on that region: recall and false positives.

    ``regions`` defaults to every region that has plot-interior pixels.
    """
    rows = []
    if regions is None:
        regions = sorted(ev.loc[ev["set"] == "rice_plot_interior", "region"].unique())
    for held in regions:
        train = ev[ev["region"].isin([r for r in regions if r != held])]
        params = derive_thresholds(train)
        test = ev[ev["region"] == held]
        called = classify_with(test, **params)
        pos = (test["set"] == "rice_plot_interior").to_numpy()
        neg = test["set"].isin(("evergreen", "water", "bare_or_built", "cut_before_map_date")).to_numpy()
        rows.append({"held_out": held, **params, "recall_pct": round(100 * called[pos].mean(), 1) if pos.any() else None,
                     "false_positive_pct": round(100 * called[neg].mean(), 1) if neg.any() else None,
                     "n_pos": int(pos.sum()), "n_neg": int(neg.sum())})
    return pd.DataFrame(rows)


def radar_null(aoi_id: int, pixels, n_dates: int = 3, seed: int = 0, radar_season="monsoon2026",
               window: int = 5, span=("2026-05-15", "2026-09-05")) -> pd.DataFrame:
    """Dips measured at random dates on the given pixels: what a dip looks like when nothing floods."""
    loc = pr.locate(aoi_id, 0, season_key=radar_season)
    rng = np.random.default_rng(seed)
    pixels = np.asarray(pixels)
    days = pd.date_range(*span, freq="D").to_numpy().astype("datetime64[D]")
    frames = []
    for _ in range(n_dates):
        when = rng.choice(days, size=len(pixels))
        best = {"VV": np.full(len(pixels), np.nan), "VH": np.full(len(pixels), np.nan)}
        for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
            dates, cubes = sar_curve.read_track(loc, track, window)
            for pol in ("VV", "VH"):
                _, _, dip = rw.dips_for_track(dates, cubes[pol].reshape(len(dates), -1)[:, pixels], when)
                best[pol] = np.fmax(best[pol], dip)
        frames.append(pd.DataFrame({"pixel": pixels, "VV_dip": best["VV"], "VH_dip": best["VH"]}))
    return pd.concat(frames, ignore_index=True)


def registration_test(aoi_id: int, plots, shifts_m=(0, 10, 20, 30), out_root="processed/_batch/s2_2026") -> pd.DataFrame:
    """Within-plot spread of the trough date for the real outlines and for outlines shifted sideways.

    Neighbouring fields are transplanted days to weeks apart, so a polygon that sits on its field
    holds pixels with one trough date, and a shifted copy straddles two fields and holds two. If the
    real outline is not the most coherent, the plots are not where they say.
    """
    d = nd.load(aoi_id, out_root=out_root)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    ev = mr.pixel_events(ndvi, lswi, d["windows"])
    tday = (ev["trough_date"] - ev["trough_date"].min()).dt.days.to_numpy().astype(float)
    tday[~ev["valid"].to_numpy()] = np.nan
    local = plots.to_crs(d["loc"]["grid"]["crs"])
    rows = []
    for s in shifts_m:
        for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)) if s else ((0, 0),):
            shifted = local.assign(geometry=local.geometry.translate(dx, dy))
            pix = plot_pixels(shifted, d["loc"]["grid"])
            spreads = [np.nanstd(tday[p]) for p in pix.values() if len(p) >= 6 and np.isfinite(tday[p]).sum() >= 4]
            rows.append({"shift_m": s, "dx": dx, "dy": dy, "plots": len(spreads),
                         "median_within_plot_std_days": round(float(np.median(spreads)), 1) if spreads else None})
            if s == 0:
                break
    return pd.DataFrame(rows)


def cloud_leakage(aoi_id: int, pixels, cs_min: float = 60, max_pixels: int = 600, seed: int = 0) -> dict:
    """How much haze the QA60 mask lets through on these pixels, measured against Cloud Score+.

    For every observation QA60 kept, Cloud Score+ (scored per 10 m pixel) says whether it was clear
    (``clear >= cs_min``). The check reports the share of kept observations Cloud Score+ calls
    cloudy, and for each such observation the NDVI difference to the median of the clear kept
    observations of the same pixel within 15 days: haze that slipped through shows as a negative
    difference. Reads the per-date files directly, so nothing depends on the fit.
    """
    import rasterio

    from ..optical_export import band_index, qa60_cloud

    rng = np.random.default_rng(seed)
    pixels = np.asarray(pixels)
    if len(pixels) > max_pixels:
        pixels = rng.choice(pixels, max_pixels, replace=False)
    loc = pr.locate(aoi_id, 0)
    paths = sorted(pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER).glob("*.tif"))
    dates, ndvi, kept, clear = [], [], [], []
    for path in paths:
        dates.append(pd.to_datetime(path.stem.rsplit("_S2_", 1)[1]))
        with rasterio.open(path) as ds:
            read = lambda n: ds.read(band_index(ds, n)).astype("float32").ravel()[pixels]  # noqa: E731
            b4, b8, qa, cs = (read(n) for n in ("B4", "B4", "QA60", "clear")) if False else (read("B4"), read("B8"), read("QA60"), read("clear"))
        data = (b4 > 0) & (b8 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi.append(np.where(data, (b8 - b4) / (b8 + b4), np.nan))
        kept.append(data & ~qa60_cloud(qa))
        clear.append(cs)
    dates = pd.DatetimeIndex(dates)
    order = np.argsort(dates)
    dates, ndvi, kept, clear = dates[order], np.stack(ndvi)[order], np.stack(kept)[order], np.stack(clear)[order]
    suspect = kept & (clear < cs_min)
    good = kept & (clear >= cs_min)
    diffs = []
    days = (dates - dates[0]).days.to_numpy()
    for j in range(len(pixels)):
        for i in np.flatnonzero(suspect[:, j]):
            window = good[:, j] & (np.abs(days - days[i]) <= 15)
            if window.sum() >= 1:
                diffs.append(ndvi[i, j] - np.nanmedian(ndvi[window, j]))
    diffs = np.asarray(diffs)
    monsoon = (dates >= "2026-06-01") & (dates < "2026-09-24")
    return {
        "pixels": int(len(pixels)), "kept_obs": int(kept.sum()),
        "kept_but_cs_cloudy_pct": round(100 * float(suspect.sum() / max(kept.sum(), 1)), 1),
        "kept_but_cs_cloudy_pct_monsoon": round(100 * float(suspect[monsoon].sum() / max(kept[monsoon].sum(), 1)), 1),
        "ndvi_diff_of_suspect_obs_p10_p50_p90": tuple(round(float(x), 2) for x in np.nanpercentile(diffs, [10, 50, 90])) if len(diffs) else None,
        "suspect_obs_lower_by_more_than_0.2_pct": round(100 * float((diffs < -0.2).mean()), 1) if len(diffs) else None,
        "n_paired": int(len(diffs)),
    }


def compare_versions(aoi_ids, plots, suffixes=("", "_v2"), out_root="processed/_batch/s2_2026") -> pd.DataFrame:
    """Class shares of every reference set under each map version, per AOI (docs/11).

    Reads the written maps ``<aoi>_monsoon2026<suffix>.tif`` (nothing is re-classified here), so
    what is scored is exactly what would be delivered. Reference sets as :func:`reference_sets`:
    field-plot interiors and edges, and the negatives built from the series alone.
    """
    import rasterio
    from pathlib import Path

    rows = []
    for aoi_id in aoi_ids:
        refs = reference_sets(aoi_id, plots[plots["aoi"] == f"aoi{aoi_id}"] if "aoi" in plots else plots,
                              out_root=out_root)
        for suffix in suffixes:
            path = Path(out_root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{suffix}.tif"
            if not path.exists():
                continue
            with rasterio.open(path) as ds:
                classes = ds.read(1).ravel()
            c = classes[refs["pixel"].to_numpy()]
            frame = refs.assign(cls=c)
            for (name, region), g in frame.groupby(["set", "region"]):
                row = {"aoi": f"aoi{aoi_id}", "version": suffix or "v1", "set": name, "region": region,
                       "pixels": len(g)}
                for code, label in mr.CLASSES.items():
                    if code != 255:
                        row[f"{label}_pct"] = round(100 * float((g["cls"] == code).mean()), 1)
                rows.append(row)
        nd.forget()
    return pd.DataFrame(rows)
