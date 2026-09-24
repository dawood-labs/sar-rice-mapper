"""The last end-to-end audit (docs/15): from the downloaded data to the field labels.

Why
---
The map passed through many stages (Earth Engine exports, downloads, stacks, 5-day series, the
rule, relabels, sieve, field labels). A fault in an early stage would pass silently into every
later one. This module re-checks the stages in order with numbers that can be compared across
AOIs:

* ``s1_inventory`` — per AOI and track: dates, the longest gap, valid pixels per date, failed chunks;
* ``s1_date_anomalies`` — per date, how far the AOI-median VH sits from the median of the dates
  around it; a jump of several dB on one date over a whole AOI is an acquisition or processing
  artefact, not a field event;
* ``s2_inventory`` — per AOI: dates on disk against the export manifest, dates in the monsoon, and
  the share of the AOI seen clear (QA60) per date;
* ``grids_match`` — the optical and the radar grids of an AOI must be identical (every radar value is
  read at the optical pixel index).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as config_mod
from . import pixel_report as pr

SRC = "processed/_batch/s2_2026"
SEASON_KEY = "monsoon2026"


def s1_inventory(aoi_id: int) -> list[dict]:
    loc = pr.locate(aoi_id, 0, season_key=SEASON_KEY)
    run = Path(loc["run"])
    failed = pd.read_csv(run / "failed_chunks.csv") if (run / "failed_chunks.csv").exists() else pd.DataFrame()
    rows = []
    for t in loc["cfg"]["s1"]["tracks"]:
        track = t["track_id"]
        d = pd.read_csv(run / "stack" / f"track_{track}" / "dates.csv", parse_dates=["date_utc"])
        gaps = d["date_utc"].sort_values().diff().dt.days
        rows.append({"aoi": f"aoi{aoi_id}", "track": track, "role": t["role"], "dates": len(d),
                     "first": d["date_utc"].min().date(), "last": d["date_utc"].max().date(),
                     "longest_gap_days": int(gaps.max()) if len(d) > 1 else None,
                     "min_valid_pct_VH": float(d["valid_pct_VH"].min()),
                     "dates_below_90pct_valid": int((d["valid_pct_VH"] < 90).sum()),
                     "failed_chunks": int((failed.get("track_id", pd.Series(dtype=str)) == track).sum())})
    return rows


def s1_date_anomalies(aoi_id: int, window: int = 5, inside=None) -> pd.DataFrame:
    """Per track and date: AOI-median VV/VH, and the jump against the median of the 2 dates each side."""
    from . import ndvi_5day as nd
    from . import sar_curve

    loc = pr.locate(aoi_id, 0, season_key=SEASON_KEY)
    mask = nd.inside_aoi(aoi_id) if inside is None else inside
    rows = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        for pol in ("VV", "VH"):
            med = np.nanmedian(cubes[pol].reshape(len(dates), -1)[:, mask], axis=1)
            s = pd.Series(med, index=dates)
            ref = s.rolling(5, center=True, min_periods=3).median()
            others = [(s.shift(k)) for k in (-2, -1, 1, 2)]
            ref_excl = pd.concat(others, axis=1).median(axis=1)
            for dte, v, r in zip(dates, med, ref_excl):
                rows.append({"aoi": f"aoi{aoi_id}", "track": track, "pol": pol, "date": dte.date(),
                             "median_db": round(float(v), 2), "jump_db": round(float(v - r), 2) if np.isfinite(r) else None})
    return pd.DataFrame(rows)


def s2_inventory(aoi_id: int, monsoon=("2026-05-15", "2026-09-24")) -> dict:
    from . import ndvi_5day as nd

    d = nd.load(aoi_id)
    dates = pd.DatetimeIndex(d["dates"])
    inside = nd.inside_aoi(aoi_id)
    ok = d["ok"].reshape(len(dates), -1)[:, inside]
    clear_share = ok.mean(axis=1)
    man = Path(SRC) / f"aoi{aoi_id}_export_manifest.csv"
    n_man = len(pd.read_csv(man)) if man.exists() else None
    m = (dates >= monsoon[0]) & (dates < monsoon[1])
    gaps = d["gapdays5d"].reshape(d["gapdays5d"].shape[0], -1)[:, inside]
    win = pd.DatetimeIndex(d["windows"])
    last = gaps[-1]
    out = {"aoi": f"aoi{aoi_id}", "dates_on_disk": len(dates), "dates_in_manifest": n_man,
           "monsoon_dates": int(m.sum()),
           "monsoon_dates_over_50pct_clear": int((clear_share[m] > 0.5).sum()),
           "monsoon_mean_clear_pct": round(100 * float(clear_share[m].mean()), 1),
           "map_window": str(win[-1].date()),
           "last_window_gap_days_median": float(np.nanmedian(last)),
           "last_window_gap_over_20d_pct": round(100 * float((last > 20).mean()), 1),
           "last_window_gap_over_30d_pct": round(100 * float((last > 30).mean()), 1)}
    nd.forget()
    return out


def grids_match(aoi_id: int) -> bool:
    from . import ndvi_5day as nd

    g_opt = nd.load(aoi_id)["loc"]["grid"]
    g_rad = pr.locate(aoi_id, 0, season_key=SEASON_KEY)["grid"]
    nd.forget()
    return all(g_opt[k] == g_rad[k] for k in ("crs", "x0", "y0", "width", "height", "res"))


def acquisition(aoi_id: int, out_dir=f"{SRC}/report/final_audit") -> dict:
    """All acquisition checks for one AOI; tables written under ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(s1_inventory(aoi_id)).to_csv(out / f"aoi{aoi_id}_s1_inventory.csv", index=False)
    s1_date_anomalies(aoi_id).to_csv(out / f"aoi{aoi_id}_s1_dates.csv", index=False)
    s2 = s2_inventory(aoi_id)
    s2["grids_match"] = grids_match(aoi_id)
    pd.DataFrame([s2]).to_csv(out / f"aoi{aoi_id}_s2_inventory.csv", index=False)
    return s2


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.final_audit")
    p.add_argument("step", choices=["acquisition", "bad-passes", "missed-rice"])
    p.add_argument("--ids", nargs="+", type=int, required=True)
    args = p.parse_args(argv)
    for a in args.ids:
        if args.step == "acquisition":
            print(acquisition(a))
        elif args.step == "missed-rice":
            out = Path(SRC) / "report" / "final_audit"
            out.mkdir(parents=True, exist_ok=True)
            r = missed_rice(a)
            pd.DataFrame([r]).to_csv(out / f"aoi{a}_missed_rice.csv", index=False)
            print(r)
        else:
            out = Path(SRC) / "report" / "final_audit"
            out.mkdir(parents=True, exist_ok=True)
            t = bad_passes(a)
            t.to_csv(out / f"aoi{a}_bad_passes.csv", index=False)
            print(f"aoi{a}: {int(t['bad'].sum()) if len(t) else 0} bad passes")
    return 0




def date_on_stable_ground(aoi_id: int, track: str, dates_to_check, window: int = 5) -> pd.DataFrame:
    """Jump on the given dates measured on ground that cannot flood or change in a week: evergreen
    pixels (``validation.reference_sets``). If trees drop by several dB on one pass the pass is an
    artefact (calibration, processing), not a field event. Also reports the audit's rain columns."""
    from . import sar_curve
    from . import validation as va

    refs = va.reference_sets(aoi_id)
    if refs.empty or "set" not in refs:
        return pd.DataFrame()
    ever = refs.loc[refs["set"] == "evergreen", "pixel"].to_numpy()
    if len(ever) < 20:
        return pd.DataFrame()
    loc = pr.locate(aoi_id, 0, season_key=SEASON_KEY)
    run = Path(loc["run"])
    meta = pd.read_csv(run / "stack" / f"track_{track}" / "dates.csv", parse_dates=["date_utc"])
    dates, cubes = sar_curve.read_track(loc, track, window)
    rows = []
    for pol in ("VV", "VH"):
        s = pd.Series(np.nanmedian(cubes[pol].reshape(len(dates), -1)[:, ever], axis=1), index=dates) if len(ever) else None
        if s is None:
            continue
        ref = pd.concat([s.shift(k) for k in (-2, -1, 1, 2)], axis=1).median(axis=1)
        for dte in dates_to_check:
            dte = pd.Timestamp(dte)
            if dte in s.index:
                m = meta[meta["date_utc"] == dte]
                rows.append({"aoi": f"aoi{aoi_id}", "track": track, "pol": pol, "date": dte.date(),
                             "evergreen_pixels": len(ever), "evergreen_jump_db": round(float(s[dte] - ref[dte]), 2),
                             "rain_24h_mm": float(m["rain_24h_mm"].iloc[0]) if len(m) else None})
    return pd.DataFrame(rows)


BAD_PASS_DB = 3.0     # a jump this large on ground that does not change marks the pass as unusable


def stable_pixels(aoi_id: int, min_px: int = 20) -> np.ndarray:
    """Pixels that should not change within a week: evergreen all year (fitted NDVI >= 0.5 in 90 % of
    windows) or class 5 (never bare) of the final map, inside the AOI."""
    import rasterio

    from . import ndvi_5day as nd

    d = nd.load(aoi_id)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    with np.errstate(invalid="ignore"):
        ever = np.nanmean(ndvi >= 0.5, axis=0) >= 0.9
    path = Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026_final.tif"
    c5 = np.zeros_like(ever)
    if path.exists():
        with rasterio.open(path) as ds:
            c5 = ds.read(1).ravel() == 5
    pix = np.flatnonzero((ever | c5) & nd.inside_aoi(aoi_id))
    return pix if len(pix) >= min_px else np.array([], dtype=int)


def bad_passes(aoi_id: int, window: int = 5, limit: float = BAD_PASS_DB) -> pd.DataFrame:
    """Every pass of every track: the jump of the stable ground's median against the 2 passes each side.
    ``bad`` where the jump is ``limit`` dB or more in that polarisation."""
    from . import sar_curve

    pix = stable_pixels(aoi_id)
    loc = pr.locate(aoi_id, 0, season_key=SEASON_KEY)
    rows = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        for pol in ("VV", "VH"):
            if not len(pix):
                continue
            s = pd.Series(np.nanmedian(cubes[pol].reshape(len(dates), -1)[:, pix], axis=1), index=dates)
            ref = pd.concat([s.shift(k) for k in (-2, -1, 1, 2)], axis=1).median(axis=1)
            for dte in dates:
                j = s[dte] - ref[dte]
                rows.append({"aoi": f"aoi{aoi_id}", "track": track, "pol": pol, "date": dte.date(),
                             "stable_pixels": len(pix), "stable_jump_db": round(float(j), 2) if np.isfinite(j) else None,
                             "bad": bool(np.isfinite(j) and abs(j) >= limit)})
    from . import ndvi_5day as nd
    nd.forget()
    return pd.DataFrame(rows)




def missed_rice(aoi_id: int) -> dict:
    """Could rice hide in "not rice" (0) or "harvested" (4)? Acres of pixels there whose radar confirms
    water (v1 dip or v2 flood) and whose canopy stands on the map date (last NDVI >= 0.5), split by
    the first rule condition they fail. Also: the share of harvested acres with water (early rice
    already cut is monsoon rice, just not standing)."""
    import rasterio

    from . import monsoon_rule as mr
    from . import ndvi_5day as nd

    d, ev, radar = mr.aoi_events(aoi_id)
    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
        c = ds.read(1).ravel()
    wet = ev["radar_wet"].fillna(False).to_numpy().astype(bool) if radar else np.zeros(len(c), bool)
    standing_canopy = (ev["last_ndvi"] >= mr.CANOPY_MIN).to_numpy()
    cand = np.isin(c, (0, 4)) & wet & standing_canopy & ev["valid"].to_numpy()
    reasons = {
        "trough_above_0.40": (ev["trough_ndvi"] > mr.TROUGH_MAX).to_numpy(),
        "low_period_under_40_days": (ev["low_windows"] < mr.LOW_WINDOWS_MIN).to_numpy(),
        "climb_under_0.30": (ev["rise"] < mr.RISE_MIN).to_numpy(),
        "fell_from_peak": ~ev["standing"].to_numpy(),
    }
    out = {"aoi": f"aoi{aoi_id}", "candidate_acres": round(mr.acres(int(cand.sum())), 1)}
    left = cand.copy()
    for name, m in reasons.items():
        out[f"{name}_acres"] = round(mr.acres(int((left & m).sum())), 1)
        left &= ~m
    out["other_acres"] = round(mr.acres(int(left.sum())), 1)
    h = c == 4
    out["harvested_acres"] = round(mr.acres(int(h.sum())), 1)
    out["harvested_with_water_pct"] = round(100 * float(wet[h].mean()), 1) if h.any() else None
    nd.forget()
    return out


if __name__ == "__main__":
    raise SystemExit(main())
