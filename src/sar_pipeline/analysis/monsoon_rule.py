"""The current-season rice rule, per pixel, checked against field plots.

Why this rule and not the earlier one
-------------------------------------
The field survey (plots reported as standing rice in September 2026) showed what the monsoon crop
looks like in the gap-free 5-day series, region by region:

* the field goes to a **trough** — bare or flooded, NDVI 0 to 0.3 — somewhere between late June and
  late August depending on the region (two months apart between the delta and the capital region);
* the canopy then **climbs** by 0.5 to 0.9 NDVI within six to nine weeks;
* at the trough the field is wet. In the delta that is open water (LSWI above NDVI, 79-100 % of
  plots). In the capital region it is not: the field dried out after the summer harvest (LSWI -0.2)
  and was then wetted for transplanting (LSWI up by 0.2-0.3 to about zero) under a shallow, turbid
  layer with dense seedlings, so LSWI never rose above NDVI (0-13 % of plots) — yet the rise from
  the field's own dry level was there in 71-86 % of them. Water is therefore read both ways
  (``wet_open`` or ``wet_relative``), and is **evidence, not a condition**: it is reported per
  pixel, and the rule does not fail a pixel for lacking it.

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

Water is confirmed by radar (``radar_water``), not by the optical LSWI: for every rice pixel the
Sentinel-1 backscatter is lined up on that pixel's own trough and the dip below its dry level is
read. Where the dip is there the pixel is **rice** (1); where the radar could look and saw no dip,
or could not look at all, it is **rice by phenology, water unconfirmed** (3) — kept apart so that
"no water" and "could not check" are never mixed with confirmed rice.

The target is rice **standing on the map date** (decided with the client's manager): the canopy
must be there on the last window (``CANOPY_MIN``) and must not have fallen from its peak by more
than ``STANDING_FALL_MAX`` — a field already cut is not standing rice, however rice-like its
season was. Because a standing crop was transplanted within roughly one rice season of the map date,
the trough is looked for only in the last ``LOOKBACK_DAYS``; this also stops the bare, dry field of
May from being taken for the transplanting of a crop that started in July. Crops still too young
to show a canopy are reported as ``young`` and crops already cut as ``harvested``, as information:
neither is delivered as rice, and the map is re-run with new imagery when the client asks again.

Classes written: 0 not rice, 1 rice standing (water confirmed by radar), 2 young, 3 rice standing
by phenology with water unconfirmed, 4 rice-like but harvested / not standing, 255 no data.
Areas in acres.
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
#: The season: rice planted in May 2026 or later. The dry-season crop is out of scope.
SEASON = ("2026-05-01", "2026-09-24")
#: A crop standing on the last date was transplanted within about one rice season of it
#: (transplanting to harvest is 100-120 days), so its trough is looked for in this many days
#: before the last window and not earlier.
LOOKBACK_DAYS = 110
#: A standing crop may have lost this much NDVI from its peak (haze, early senescence) and still be
#: standing; a harvested field has lost far more.
STANDING_FALL_MAX = 0.25
#: The field must have been without a canopy (fitted NDVI at or below ``TROUGH_MAX + 0.05``) for at
#: least this many consecutive 5-day windows around its trough — 8 windows, about 40 days. A field
#: really starting a crop has weeks of bare, puddled and seedling-covered ground; a haze dip that
#: the light cloud mask let through on an orchard or a tree line lasts one or two windows. On the
#: validation set this took the evergreen false-positive rate from 7 % to 0 % without touching the
#: recall on the field plots, whose low period is 55 days at the 5th percentile. Counted up to 8
#: windows on each side of the trough.
LOW_WINDOWS_MIN = 8
#: LSWI must rise by this much from the field's own driest level in the 60 days before the trough
#: (and end at or above zero) to count as wetting for transplanting.
WET_RISE_MIN = 0.15
WET_LOOKBACK_DAYS, WET_LOOKAHEAD_DAYS = 60, 25
CLASSES = {0: "not rice", 1: "rice", 2: "young", 3: "rice_unconfirmed", 4: "harvested",
           5: "never_bare", 255: "no data"}
#: Water test versions: "v1" = dip at the optical trough only; "v2" (docs/11) = v1 OR the flood
#: searched over the whole bare period, plus class 5 for "rice-like" curves on ground the radar
#: shows was never bare (trees, houses: the optical trough there is haze).
WATER_DEFAULT = "v2"


def pixel_events(ndvi, lswi, windows, season=SEASON, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Per pixel: trough date and value inside the search window, LSWI there, and the climb after it.

    ``ndvi``/``lswi`` are (windows, pixels) fitted series. Vectorised, so an AOI of half a million
    pixels takes seconds. The search window is the season cut to the last ``lookback_days`` before
    the last window (see ``LOOKBACK_DAYS``). ``climb_date`` is the first window after the trough
    where NDVI has risen by ``RISE_MIN`` — the moment the crop became a confirmed canopy.
    ``standing`` says whether that canopy is still there on the last window.
    """
    windows = pd.DatetimeIndex(windows)
    earliest = max(pd.Timestamp(season[0]), windows[-1] - pd.Timedelta(days=lookback_days))
    inside = (windows >= earliest) & (windows < season[1])
    idx = np.flatnonzero(inside)
    sub = ndvi[idx]
    n_pix = ndvi.shape[1]
    valid = np.isfinite(sub).all(axis=0)
    trough = np.where(valid, np.nanargmin(np.where(np.isfinite(sub), sub, np.inf), axis=0), 0)
    cols = np.arange(n_pix)
    trough_ndvi = sub[trough, cols]
    lswi_sub = lswi[idx]
    lswi_at = lswi_sub[trough, cols]
    # wetting relative to the field's own dry level: driest LSWI in the weeks before the trough
    # against the wettest in the weeks after it (the water is applied at or just after the trough)
    step = np.arange(len(idx))[:, None]
    when = windows[idx].to_numpy()
    day_gap = (when[:, None] - when[trough][None, :]) / np.timedelta64(1, "D")
    before = (day_gap >= -WET_LOOKBACK_DAYS) & (day_gap <= 0)
    after_w = (day_gap >= 0) & (day_gap <= WET_LOOKAHEAD_DAYS)
    lswi_dry = np.where(before, lswi_sub, np.inf).min(axis=0)
    lswi_wet = np.where(after_w, lswi_sub, -np.inf).max(axis=0)
    # highest value after the trough, and the first window where the climb reaches RISE_MIN
    after = np.where(step >= trough[None, :], sub, -np.inf)
    peak_after = after.max(axis=0)
    reached = after >= (trough_ndvi + RISE_MIN)[None, :]
    first = np.where(reached.any(axis=0), reached.argmax(axis=0), -1)
    return pd.DataFrame({
        "valid": valid,
        "trough_date": windows[idx][trough],
        "trough_ndvi": trough_ndvi, "lswi_at_trough": lswi_at,
        "wet_open": lswi_at > trough_ndvi,
        "wet_relative": (lswi_wet - lswi_dry >= WET_RISE_MIN) & (lswi_wet >= 0),
        "wet_at_trough": (lswi_at > trough_ndvi) | ((lswi_wet - lswi_dry >= WET_RISE_MIN) & (lswi_wet >= 0)),
        "rise": peak_after - trough_ndvi, "peak_after": peak_after,
        "climb_date": pd.Series(np.where(first >= 0, windows[idx][np.clip(first, 0, None)], pd.NaT)),
        "last_ndvi": ndvi[-1],
        "standing": (ndvi[-1] >= CANOPY_MIN) & (ndvi[-1] >= peak_after - STANDING_FALL_MAX),
        "low_windows": low_run(ndvi, idx[trough], TROUGH_MAX + 0.05),
    })


def low_run(ndvi, trough_idx, level: float, reach: int = 8) -> np.ndarray:
    """Consecutive windows at or below ``level`` around each pixel's trough index, up to ``reach`` per side.

    ``ndvi`` is the full (windows, pixels) series; ``trough_idx`` indexes into it. The trough window
    itself counts as one.
    """
    n_win, n_pix = ndvi.shape
    cols = np.arange(n_pix)
    low = ndvi <= level
    run = np.ones(n_pix, dtype=int)
    for direction in (-1, 1):
        still = np.ones(n_pix, dtype=bool)
        for k in range(1, reach + 1):
            j = trough_idx + direction * k
            inside = (j >= 0) & (j < n_win)
            still &= inside & low[np.clip(j, 0, n_win - 1), cols]
            run += still
    return run


def classify(events: pd.DataFrame, trough_max=TROUGH_MAX, rise_min=RISE_MIN, young_min=YOUNG_MIN,
             canopy_min=CANOPY_MIN, young_canopy_min=YOUNG_CANOPY_MIN, radar_wet=None,
             low_windows_min: int = LOW_WINDOWS_MIN, never_bare=None):
    """0 not rice, 1 rice standing, 2 young, 3 standing with water unconfirmed, 4 harvested,
    5 rice-like curve on ground that was never bare (radar), 255 no data.

    ``radar_wet`` (bool per pixel) splits the standing phenology-rice pixels into confirmed (1) and
    unconfirmed (3). Without it every such pixel is unconfirmed. ``never_bare`` (bool per pixel,
    ``radar_water.water_evidence``) moves unconfirmed pixels whose radar shows a canopy or buildings
    all season to class 5.
    """
    out = np.zeros(len(events), dtype="uint8")
    low = events["trough_ndvi"] <= trough_max
    if "low_windows" in events:
        low &= events["low_windows"] >= low_windows_min
    grown = (low & (events["rise"] >= rise_min) & (events["peak_after"] >= canopy_min)).to_numpy()
    standing = events["standing"].to_numpy() if "standing" in events else np.ones(len(events), dtype=bool)
    rice = grown & standing
    young = (low & ~grown & (events["rise"] >= young_min) & (events["peak_after"] >= young_canopy_min)).to_numpy()
    wet = np.zeros(len(events), dtype=bool) if radar_wet is None else np.asarray(radar_wet, dtype=bool)
    out[rice & wet] = 1
    out[rice & ~wet] = 3
    if never_bare is not None:
        out[rice & ~wet & np.asarray(never_bare, dtype=bool)] = 5
    out[grown & ~standing] = 4
    out[young] = 2
    out[~events["valid"].to_numpy()] = 255
    return out


def aoi_events(aoi_id: int, out_root="processed/_batch/s2_2026", season=SEASON,
               radar_season: str | None = "monsoon2026", water: str = WATER_DEFAULT):
    """The per-pixel evidence the rule decides on: ``(series, events, radar)``.

    ``events`` holds, per grid pixel, the optical events of :func:`pixel_events` and, when the
    Sentinel-1 season run exists (``radar`` True), the radar dips of ``radar_water.pixel_dips``.
    Kept separate from :func:`run_aoi` so reports can look at the evidence behind a class, not only
    the class.
    """
    from . import radar_water

    d = nd.load(aoi_id, out_root=out_root)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    events = pixel_events(ndvi, lswi, d["windows"], season)
    radar = bool(radar_season) and radar_water.season_run_exists(aoi_id, radar_season)
    if radar:
        trough = np.where(events["valid"].to_numpy(), events["trough_date"].to_numpy().astype("datetime64[D]"),
                          np.datetime64("NaT"))
        dips = radar_water.pixel_dips(aoi_id, trough, d["loc"]["grid"], radar_season)
        events = pd.concat([events, dips], axis=1)
        if water == "v2":
            climb = pd.to_datetime(events["climb_date"]).to_numpy().astype("datetime64[D]")
            climb = np.where(events["valid"].to_numpy(), climb, np.datetime64("NaT"))
            v2 = radar_water.water_evidence(aoi_id, trough, climb, ndvi, d["windows"], radar_season)
            events = pd.concat([events, v2], axis=1)
            events["radar_wet_v1"] = events["radar_wet"]
            events["radar_wet"] = events["radar_wet"] | events["flood_ok"]
    return d, events, radar


def run_aoi(aoi_id: int, out_root="processed/_batch/s2_2026", season=SEASON, inside_only: bool = True,
            radar_season: str | None = "monsoon2026", water: str = WATER_DEFAULT, suffix: str = "") -> dict:
    """Classify one AOI, write ``<aoi>_monsoon2026.tif``, return the acres per class.

    ``radar_season`` names the Sentinel-1 season run used to confirm the water; when that run does
    not exist for the AOI every phenology-rice pixel lands in class 3 and ``radar`` says ``False``.
    """
    import rasterio
    from rasterio.transform import from_origin

    d, events, radar = aoi_events(aoi_id, out_root, season, radar_season, water)
    shape = d["ndvi5d"].shape[1:]
    classes = classify(events, radar_wet=events["radar_wet"] if radar else None,
                       never_bare=events["never_bare"] if radar and "never_bare" in events else None)
    if inside_only:
        classes[~nd.inside_aoi(aoi_id)] = 255
    grid = d["loc"]["grid"]
    path = Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026{suffix}.tif"
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint8", crs=grid["crs"],
                   nodata=255, compress="deflate", tiled=True,
                   transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]))
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(classes.reshape(shape), 1)
    counts = {name: int((classes == code).sum()) for code, name in CLASSES.items() if code != 255}
    inside = classes != 255
    wet = events["wet_at_trough"].to_numpy()
    n_rice = counts["rice"] + counts["rice_unconfirmed"]
    return {"aoi": d["loc"]["aoi"], "radar": radar, "water": water if radar else None,
            **{f"{k}_acres": round(acres(v), 1) for k, v in counts.items()},
            "rice_pct_of_decided": round(100 * n_rice / max(inside.sum(), 1), 1),
            "radar_wet_pct_of_rice": round(100 * counts["rice"] / max(n_rice, 1), 1),
            "radar_checkable_pct_of_rice": (round(100 * float(events["radar_checkable"].to_numpy()[
                np.isin(classes, (1, 3))].mean()), 1) if radar and n_rice else None),
            "optical_wet_pct_of_rice": round(100 * float(wet[np.isin(classes, (1, 3))].mean()) if n_rice else 0.0, 1),
            "trough_median": events.loc[np.isin(classes, (1, 3)), "trough_date"].median().strftime("%d %b")
            if n_rice else None,
            "climb_median": events.loc[np.isin(classes, (1, 3)), "climb_date"].dropna().median().strftime("%d %b")
            if n_rice else None,
            "path": str(path)}


def plot_recall(aoi_id: int, plots, out_root="processed/_batch/s2_2026", suffix: str = "") -> dict:
    """Share of field-plot pixels the map calls rice / young / not rice, for one AOI."""
    import rasterio

    from .plot_curves import plot_pixels

    d = nd.load(aoi_id, out_root=out_root)
    with rasterio.open(Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026{suffix}.tif") as ds:
        classes = ds.read(1).ravel()
    pix = np.concatenate(list(plot_pixels(plots, d["loc"]["grid"]).values()))
    c = classes[pix]
    n = max(int((c != 255).sum()), 1)
    return {"aoi": d["loc"]["aoi"], "plot_pixels": int(len(pix)),
            **{f"{name}_pct": round(100 * float((c == code).sum()) / n, 1)
               for code, name in CLASSES.items() if code != 255}}


def plot_recall_by_position(aoi_id: int, plots, out_root="processed/_batch/s2_2026") -> pd.DataFrame:
    """Class shares of field-plot pixels split into plot interior and plot edge, plus per-plot shares.

    Returns a frame with rows ``interior`` and ``edge`` (class percentages) and the attribute
    ``per_plot``: per plot, the share of its pixels called rice, to find whole plots that disagree.
    """
    import rasterio

    from .plot_curves import interior_pixels, plot_pixels

    d = nd.load(aoi_id, out_root=out_root)
    with rasterio.open(Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026.tif") as ds:
        classes = ds.read(1).ravel()
    pixels = plot_pixels(plots, d["loc"]["grid"])
    interior = interior_pixels(pixels, int(d["loc"]["grid"]["width"]))
    rows, per_plot = {}, []
    for name, want in (("interior", True), ("edge", False)):
        pix = np.concatenate([p[interior[k] == want] for k, p in pixels.items()]) if pixels else np.array([], int)
        c = classes[pix.astype(int)] if len(pix) else np.array([], "uint8")
        n = max(int((c != 255).sum()), 1)
        rows[name] = {"pixels": int(len(pix)), **{f"{label}_pct": round(100 * float((c == code).sum()) / n, 1)
                                                 for code, label in CLASSES.items() if code != 255}}
    for k, p in pixels.items():
        c = classes[p]
        per_plot.append({"plot_id": k, "n_pixels": len(p), "rice_pct": round(100 * float(np.isin(c, (1, 3)).mean()), 1),
                         "confirmed_pct": round(100 * float((c == 1).mean()), 1)})
    frame = pd.DataFrame(rows).T
    frame.attrs["per_plot"] = pd.DataFrame(per_plot)
    return frame
