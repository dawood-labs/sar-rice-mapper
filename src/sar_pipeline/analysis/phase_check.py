"""Does the radar season sit on the same phase as the optical crop cycle?

Why
---
The draft model treats a late-season rise in VH as the rice canopy. A pixel-level look at AOI 146
suggested something else: Sentinel-2 NDVI collapses (harvest) *weeks before* VH reaches its seasonal
maximum. If that holds for a whole class and not just one pixel, the feature the classifier leans on
is not the canopy at all, and the class definitions ("monsoon" vs "dry season") are anchored to the
wrong months.

This module answers that with class-mean curves rather than single pixels, because speckle (~2 dB
per pixel) and cloud gaps make single-pixel timing unreliable:

* **NDVI** per Sentinel-2 date, averaged over the pixels of one model class, using only pixels whose
  Cloud Score+ clear value passes ``clear_min`` on that date. Dates with too few clear pixels are
  dropped rather than plotted from noise.
* **VH and VV** per acquisition for the same pixels, averaged in linear power and converted back to
  dB (averaging dB would bias the mean low).
* A **timing summary**: NDVI green-up, NDVI peak, NDVI half-fall ("harvest"), VH maximum, and the lag
  between harvest and the VH maximum. A large positive lag means the radar maximum happens on a
  harvested field.

The lag is the number that matters. For a real canopy peak it should be near zero or negative.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import seasonal_stats as ss
from . import pixel_report as pr


def half_fall_date(dates, values, peak_idx: int, min_drop: float = 0.15):
    """First date after the peak where the curve drops halfway to its post-peak minimum.

    Why a half-fall and not the steepest drop: cloud gaps make the steepest single step land on
    whichever date happens to follow a gap, while the halfway crossing is stable as long as one
    clear observation exists on each side of it.

    ``min_drop`` is the fall the curve must actually make (peak minus post-peak minimum) before a
    crossing counts. Without it a flat curve still "crosses" its own midpoint and reports a harvest
    that never happened; for NDVI a real rice harvest drops far more than 0.15.
    """
    values = np.asarray(values, dtype="float64")
    after = values[peak_idx:]
    if len(after) < 2:
        return None
    floor = np.nanmin(after)
    if not np.isfinite(floor) or values[peak_idx] - floor < min_drop:
        return None
    midpoint = (values[peak_idx] + floor) / 2.0
    below = np.where(after <= midpoint)[0]
    return dates[peak_idx + below[0]] if len(below) else None


def class_ndvi(aoi_id: int, class_value: int, clear_min: float = 60, min_pixels: int = 200,
               cache_root="data/s2_reference", maps_dir="processed/_batch/model_v2/maps") -> pd.DataFrame:
    """Mean NDVI per Sentinel-2 date over the pixels the model put in ``class_value``."""
    import rasterio

    loc = pr.locate(aoi_id, 0)
    with rasterio.open(Path(maps_dir) / f"{loc['aoi']}_class.tif") as ds:
        mask = ds.read(1) == class_value
    rows = []
    for path in sorted(pr.sync_s2(loc, cache_root).glob("*.tif")):
        date = pd.to_datetime(path.stem.rsplit("_S2_", 1)[1])
        with rasterio.open(path) as ds:
            b3, b4, b8, clear = (ds.read(i).astype("float32") for i in (2, 3, 5, 6))
        use = mask & (clear >= clear_min) & (b8 + b4 > 0)
        if use.sum() < min_pixels:
            continue
        rows.append({"date": date, "n": int(use.sum()),
                     "ndvi": float(np.mean((b8[use] - b4[use]) / (b8[use] + b4[use]))),
                     "ndwi": float(np.mean((b3[use] - b8[use]) / (b3[use] + b8[use])))})
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def class_sar(aoi_id: int, class_value: int, track: str | None = None,
              maps_dir="processed/_batch/model_v2/maps") -> pd.DataFrame:
    """Mean VH and VV per acquisition (linear-power mean, reported in dB) for one model class."""
    import rasterio

    loc = pr.locate(aoi_id, 0)
    with rasterio.open(Path(maps_dir) / f"{loc['aoi']}_class.tif") as ds:
        mask = ds.read(1) == class_value
    stack = Path(loc["run"]) / "stack" / f"track_{track or loc['primary']}"
    out = {}
    for pol in ("VH", "VV"):
        with rasterio.open(stack / f"stack_{pol}.vrt") as ds:
            cube = ds.read().astype("float32")
            if ds.nodata is not None:
                cube[cube == ds.nodata] = np.nan
            dates = [dt.datetime.strptime(d.rsplit("_", 1)[1], "%Y%m%d").date() for d in ds.descriptions]
        with np.errstate(invalid="ignore"):
            out[pol] = ss.to_db(np.nanmean(ss.to_linear(cube[:, mask]), axis=1))
    frame = pd.DataFrame({"date": pd.to_datetime(dates), **out})
    frame["VHmVV"] = frame["VH"] - frame["VV"]
    info = pd.read_csv(stack / "dates.csv")
    frame["rain_24h_mm"] = pd.to_numeric(info["rain_24h_mm"], errors="coerce").to_numpy()[:len(frame)]
    return frame


def timing(ndvi: pd.DataFrame, sar: pd.DataFrame, season_start="2025-05-01", season_end="2026-01-31") -> dict:
    """Green-up, peak, harvest and VH maximum dates, and the harvest -> VH-max lag in days."""
    n = ndvi[(ndvi["date"] >= season_start) & (ndvi["date"] <= season_end)].reset_index(drop=True)
    s = sar[(sar["date"] >= season_start) & (sar["date"] <= season_end)].reset_index(drop=True)
    peak = int(n["ndvi"].idxmax())
    harvest = half_fall_date(list(n["date"]), n["ndvi"].to_numpy(), peak)
    vh_max = s.loc[s["VH"].idxmax(), "date"]
    return {"NDVI peak": n.loc[peak, "date"], "NDVI peak value": round(float(n.loc[peak, "ndvi"]), 2),
            "NDVI half-fall (harvest)": harvest, "VH max": vh_max,
            "VH max value dB": round(float(s["VH"].max()), 1),
            "VH at NDVI peak dB": round(float(s.loc[(s["date"] - n.loc[peak, "date"]).abs().idxmin(), "VH"]), 1),
            "harvest -> VH max lag (days)": None if harvest is None else int((vh_max - harvest).days)}
