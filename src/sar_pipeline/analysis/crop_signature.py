"""Compare the season signature of known-rice plots with the phenology-rice pixels of an AOI.

Why
---
When the rule calls a large area "rice by phenology, water unconfirmed" there are two explanations,
and they need different fixes: either the analysis looked for the water at the wrong moment (the
season's lowest NDVI is not always the transplanting), or the crop is not rice. The field plots are
the only place where rice is certain, so the honest test is to measure the **same descriptors** on
both — the known-rice plots and the AOI's candidate pixels — and put the distributions side by side:

* the crop cycle found by the cycle detector (``optical_phenology.pixel_cycles``) that starts in the
  season: transplant date (green-up onset minus the measured lag), peak NDVI, green-up to harvest
  in days, whether the crop was cut before the last date;
* the water at **that** transplant date, from the optical (LSWI at the trough) and from the radar
  (the dip below the field's own dry level, ``radar_water.dips_for_track``).

Rice's cycle is long: from green-up to harvest about three months. A crop that is cut seven weeks
after green-up is something else, whatever its NDVI peak. And a field that shows the radar dip at
its cycle's own transplant date, but not at the season minimum, is a rule bug, not a non-rice field.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from . import optical_phenology as op
from . import pixel_report as pr
from . import radar_water as rw
from . import sar_curve

SEASON_START = "2026-05-01"


def cycle_from_series(windows, ndvi, lswi, start=SEASON_START) -> dict | None:
    """The season's crop: the cycle whose green-up (or peak) falls on or after ``start``, or None.

    The trough is not used for the cut: on a field that lies flat for months the detector's trough
    can sit weeks before the crop, while the green-up onset is the crop's own start.
    """
    found = op.pixel_cycles(pd.DatetimeIndex(windows), np.asarray(ndvi, float), None,
                            np.asarray(lswi, float))
    start = pd.Timestamp(start)
    cycles = [c for c in found
              if pd.Timestamp(c["greenup_onset"] if c["greenup_onset"] is not None else c["peak_date"]) >= start]
    if not cycles:
        return None
    # the crop that grew most: a weed flush before the real crop must not win
    c = max(cycles, key=lambda c: c["growth_amplitude"] if np.isfinite(c["growth_amplitude"]) else -1)
    return {
        "trough_date": pd.Timestamp(c["start_date"]), "transplant_date": pd.Timestamp(c["transplant_date"]),
        "greenup_onset": None if c["greenup_onset"] is None else pd.Timestamp(c["greenup_onset"]),
        "peak_date": pd.Timestamp(c["peak_date"]), "peak_ndvi": c["peak_ndvi"],
        "harvest_date": None if c["harvest_date"] is None else pd.Timestamp(c["harvest_date"]),
        "greenup_to_harvest_days": c["greenup_to_harvest_days"], "complete": c["complete"],
        "growth_amplitude": c["growth_amplitude"], "lswi_at_start": c["lswi_at_start"],
        "wet_optical": c["wet_at_transplant"],
    }


def plot_signatures(curves: pd.DataFrame, min_pixels: int = 3, start=SEASON_START) -> pd.DataFrame:
    """One row per field plot: its season cycle from the plot's median curve (``plot_curves`` output)."""
    rows = []
    for pid, g in curves[curves["n_pixels"] >= min_pixels].groupby("plot_id"):
        g = g.sort_values("window")
        c = cycle_from_series(g["window"], g["ndvi"], g["lswi"], start)
        if c:
            rows.append({"plot_id": pid, "aoi": g["aoi"].iloc[0] if "aoi" in g else None, **c})
    return pd.DataFrame(rows)


def pixel_signatures(aoi_id: int, pixels, start=SEASON_START, out_root="processed/_batch/s2_2026") -> pd.DataFrame:
    """One row per pixel id in ``pixels``: its season cycle from the fitted 5-day series."""
    d = nd.load(aoi_id, out_root=out_root)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    rows = []
    for p in pixels:
        c = cycle_from_series(d["windows"], ndvi[:, p], lswi[:, p], start)
        if c:
            rows.append({"pixel": int(p), "aoi": d["loc"]["aoi"], **c})
    return pd.DataFrame(rows)


def radar_dip_at(aoi_id: int, pixels, dates, grid: dict, season_key: str = "monsoon2026") -> pd.DataFrame:
    """Best VV and VH dip over tracks for the given pixels at the given per-pixel dates."""
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    pixels = np.asarray(pixels)
    when = np.asarray(pd.to_datetime(dates).values).astype("datetime64[D]")
    best = {"VV": np.full(len(pixels), np.nan), "VH": np.full(len(pixels), np.nan)}
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dts, cubes = sar_curve.read_track(loc, track, 5)
        for pol in ("VV", "VH"):
            cube = cubes[pol].reshape(len(dts), -1)[:, pixels]
            _, _, dip = rw.dips_for_track(dts, cube, when)
            best[pol] = np.fmax(best[pol], dip)
    out = pd.DataFrame({"VV_dip": best["VV"], "VH_dip": best["VH"]})
    out["radar_wet"] = (out["VV_dip"] >= rw.DIP_MIN_DB) | (out["VH_dip"] >= rw.DIP_MIN_DB)
    return out


DESCRIPTORS = ("greenup_to_harvest_days", "peak_ndvi", "growth_amplitude", "lswi_at_start", "VH_dip", "VV_dip")


def compare(reference: pd.DataFrame, candidate: pd.DataFrame, labels=("known rice plots", "candidate pixels")) -> pd.DataFrame:
    """Percentiles of each descriptor for both groups, side by side, plus the share cut before the end."""
    rows = []
    for name, frame in zip(labels, (reference, candidate)):
        row = {"group": name, "n": len(frame)}
        for col in DESCRIPTORS:
            if col in frame:
                v = pd.to_numeric(frame[col], errors="coerce").dropna()
                row[f"{col} p25/p50/p75"] = (f"{v.quantile(.25):.2f} / {v.quantile(.5):.2f} / {v.quantile(.75):.2f}"
                                             if len(v) else "-")
        row["harvested before last date %"] = round(100 * frame["harvest_date"].notna().mean(), 1) if "harvest_date" in frame else None
        row["transplant median"] = frame["transplant_date"].median().strftime("%d %b") if len(frame) else None
        if "wet_optical" in frame:
            row["optical wet %"] = round(100 * frame["wet_optical"].fillna(False).astype(bool).mean(), 1)
        if "radar_wet" in frame:
            row["radar wet %"] = round(100 * frame["radar_wet"].fillna(False).astype(bool).mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows).set_index("group").T
