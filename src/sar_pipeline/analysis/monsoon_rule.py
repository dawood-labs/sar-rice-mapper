"""The current-season rice rule, per pixel, checked against field plots.

Why this rule and not the earlier one
-------------------------------------
The field survey (plots reported as standing rice in September 2026) showed what the monsoon crop
looks like in the gap-free 5-day series, region by region:

* the field goes to a **trough** — bare or flooded, NDVI 0 to 0.3 — somewhere between late June and
  late August depending on the region (two months apart between the delta and the capital region);
* the canopy then **climbs** by 0.5 to 0.9 NDVI within six to nine weeks;
* at the trough the field is often wet (LSWI above NDVI) — but not everywhere: in one region rice
  was certain and LSWI never rose above NDVI. So water is **evidence, not a condition**.

So the rule is calendar-free within a season window, and needs no AOI-level season: per pixel, the
trough inside the window and the climb after it. Two thresholds, both read off the plots:

* ``trough_max`` = 0.40: the 90th percentile of plot troughs was below 0.2 in most regions and
  0.36-0.41 in the two latest ones;
* ``rise_min`` = 0.30 for **rice**: the plots' rises were 0.5-0.9 wherever the crop was more than
  six weeks old. Where it was younger the rise was 0.15-0.35, so a rise between ``young_min`` = 0.15
  and ``rise_min`` is called **young**: a crop has started, it cannot be confirmed yet.

A third threshold is an absolute floor on the canopy reached, ``canopy_min`` = 0.50 for rice and
``young_canopy_min`` = 0.30 for young. It exists because of a control AOI with no monsoon crop at
all: there the fields went under water (NDVI -0.35) and re-emerged as bare soil (0.2), a "rise" of
0.55 that is not a canopy. NDVI 0.2 is soil; the plots' canopies were 0.65 and above.

On the plots this gives 88-100 % recall where the crop was established and 30-60 % where it was
still young — which is the honest limit of optical data at that date, not a flaw to tune away.

Classes written: 0 not rice, 1 rice, 2 young, 255 no data. Areas in acres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from .optical_phenology import acres

TROUGH_MAX = 0.40
RISE_MIN = 0.30
YOUNG_MIN = 0.15
CANOPY_MIN = 0.50
YOUNG_CANOPY_MIN = 0.30
SEASON = ("2026-06-01", "2026-09-24")
CLASSES = {0: "not rice", 1: "rice", 2: "young", 255: "no data"}


def pixel_events(ndvi, lswi, windows, season=SEASON) -> pd.DataFrame:
    """Per pixel: trough date and value inside the season, LSWI there, and the climb after it.

    ``ndvi``/``lswi`` are (windows, pixels) fitted series. Vectorised, so an AOI of half a million
    pixels takes seconds. ``climb_date`` is the first window after the trough where NDVI has risen by
    ``RISE_MIN`` — the moment the crop became a confirmed canopy.
    """
    windows = pd.DatetimeIndex(windows)
    inside = (windows >= season[0]) & (windows < season[1])
    idx = np.flatnonzero(inside)
    sub = ndvi[idx]
    n_pix = ndvi.shape[1]
    valid = np.isfinite(sub).all(axis=0)
    trough = np.where(valid, np.nanargmin(np.where(np.isfinite(sub), sub, np.inf), axis=0), 0)
    cols = np.arange(n_pix)
    trough_ndvi = sub[trough, cols]
    lswi_at = lswi[idx][trough, cols]
    # highest value after the trough, and the first window where the climb reaches RISE_MIN
    after = np.where(np.arange(len(idx))[:, None] >= trough[None, :], sub, -np.inf)
    peak_after = after.max(axis=0)
    reached = after >= (trough_ndvi + RISE_MIN)[None, :]
    first = np.where(reached.any(axis=0), reached.argmax(axis=0), -1)
    return pd.DataFrame({
        "valid": valid,
        "trough_date": windows[idx][trough],
        "trough_ndvi": trough_ndvi, "lswi_at_trough": lswi_at,
        "wet_at_trough": lswi_at > trough_ndvi,
        "rise": peak_after - trough_ndvi, "peak_after": peak_after,
        "climb_date": pd.Series(np.where(first >= 0, windows[idx][np.clip(first, 0, None)], pd.NaT)),
        "last_ndvi": ndvi[-1],
    })


def classify(events: pd.DataFrame, trough_max=TROUGH_MAX, rise_min=RISE_MIN, young_min=YOUNG_MIN,
             canopy_min=CANOPY_MIN, young_canopy_min=YOUNG_CANOPY_MIN):
    """0 not rice, 1 rice, 2 young, 255 no data — from the per-pixel events."""
    out = np.zeros(len(events), dtype="uint8")
    low = events["trough_ndvi"] <= trough_max
    rice = low & (events["rise"] >= rise_min) & (events["peak_after"] >= canopy_min)
    young = low & ~rice & (events["rise"] >= young_min) & (events["peak_after"] >= young_canopy_min)
    out[rice.to_numpy()] = 1
    out[young.to_numpy()] = 2
    out[~events["valid"].to_numpy()] = 255
    return out


def run_aoi(aoi_id: int, out_root="processed/_batch/s2_2026", season=SEASON, inside_only: bool = True) -> dict:
    """Classify one AOI, write ``<aoi>_monsoon2026.tif``, return the acres per class."""
    import rasterio
    from rasterio.transform import from_origin

    d = nd.load(aoi_id, out_root=out_root)
    shape = d["ndvi5d"].shape[1:]
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    events = pixel_events(ndvi, lswi, d["windows"], season)
    classes = classify(events)
    if inside_only:
        classes[~nd.inside_aoi(aoi_id)] = 255
    grid = d["loc"]["grid"]
    path = Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026.tif"
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint8", crs=grid["crs"],
                   nodata=255, compress="deflate", tiled=True,
                   transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]))
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(classes.reshape(shape), 1)
    counts = {name: int((classes == code).sum()) for code, name in CLASSES.items() if code != 255}
    inside = classes != 255
    wet = events["wet_at_trough"].to_numpy()
    return {"aoi": d["loc"]["aoi"], **{f"{k}_acres": round(acres(v), 1) for k, v in counts.items()},
            "rice_pct_of_decided": round(100 * counts["rice"] / max(inside.sum(), 1), 1),
            "wet_pct_of_rice": round(100 * float(wet[classes == 1].mean()) if counts["rice"] else 0.0, 1),
            "trough_median": events.loc[classes == 1, "trough_date"].median().strftime("%d %b")
            if counts["rice"] else None,
            "climb_median": events.loc[classes == 1, "climb_date"].dropna().median().strftime("%d %b")
            if counts["rice"] else None,
            "path": str(path)}


def plot_recall(aoi_id: int, plots, out_root="processed/_batch/s2_2026") -> dict:
    """Share of field-plot pixels the map calls rice / young / not rice, for one AOI."""
    import rasterio

    from .plot_curves import plot_pixels

    d = nd.load(aoi_id, out_root=out_root)
    with rasterio.open(Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026.tif") as ds:
        classes = ds.read(1).ravel()
    pix = np.concatenate(list(plot_pixels(plots, d["loc"]["grid"]).values()))
    c = classes[pix]
    n = max(int((c != 255).sum()), 1)
    return {"aoi": d["loc"]["aoi"], "plot_pixels": int(len(pix)),
            **{f"{name}_pct": round(100 * float((c == code).sum()) / n, 1)
               for code, name in CLASSES.items() if code != 255}}
