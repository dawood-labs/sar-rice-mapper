"""How the whole AOI looks on every date: NDVI spread, canopy share and water share.

Why
---
A handful of sample pixels can miss a crop that covers only part of an AOI, and a single cloud mask
can hide a season altogether. The question "is there a monsoon crop here at all?" is answered better
by summarising **every pixel inside the AOI on every date**, under **two independent masks**:

* ``qa60`` — the product's own cloud bits (10, 11), the mask the 5-day series uses;
* ``cs`` — Cloud Score+ ``clear`` at or above a threshold. Scored per 10 m pixel, so on monsoon days
  where QA60 throws the whole scene away it can still keep the parts that are clear.

Per date and mask it reports how many AOI pixels were usable, the 10th / 50th / 90th percentile of
NDVI, the share with a canopy (NDVI above ``canopy_ndvi``) and the share that looks flooded
(LSWI above NDVI: open water and puddled fields reflect more in NIR than SWIR while carrying almost
no chlorophyll). A crop that covers even a tenth of the AOI moves the 90th percentile and the canopy
share well before it moves the median.

These thresholds only summarise; they classify nothing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from . import pixel_report as pr

#: NDVI above which a pixel counts as carrying a canopy in the summary.
CANOPY_NDVI = 0.5
#: Cloud Score+ clear value (0-100) for the ``cs`` mask.
CS_CLEAR_MIN = 60
#: Share of the AOI that must be usable for a date to be summarised at all.
MIN_USABLE_PCT = 5.0


def summarise(ndvi, lswi, usable, canopy_ndvi: float = CANOPY_NDVI) -> dict:
    """One date's summary over the pixels marked ``usable`` (1-D arrays of AOI pixels)."""
    v, w = ndvi[usable], lswi[usable]
    n = int(usable.sum())
    if n == 0:
        return {"usable_pct": 0.0}
    p10, p50, p90 = np.nanpercentile(v, [10, 50, 90])
    return {"usable_pct": round(100 * n / len(ndvi), 1),
            "ndvi_p10": round(float(p10), 3), "ndvi_p50": round(float(p50), 3), "ndvi_p90": round(float(p90), 3),
            "canopy_pct": round(100 * float((v > canopy_ndvi).mean()), 1),
            "water_pct": round(100 * float((w > v).mean()), 1)}


def date_profile(aoi_id: int, cs_clear_min: float = CS_CLEAR_MIN, canopy_ndvi: float = CANOPY_NDVI) -> pd.DataFrame:
    """One row per date and mask with :func:`summarise` over the AOI's pixels."""
    import rasterio

    from ..optical_export import band_index, qa60_cloud

    loc = pr.locate(aoi_id, 0)
    inside = nd.inside_aoi(aoi_id)
    paths = sorted(pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER).glob("*.tif"))
    rows = []
    for path in paths:
        date = pd.to_datetime(path.stem.rsplit("_S2_", 1)[1])
        with rasterio.open(path) as ds:
            read = lambda n: ds.read(band_index(ds, n)).astype("float32").ravel()[inside]  # noqa: E731
            b4, b8, b11, qa, clear = (read(n) for n in ("B4", "B8", "B11", "QA60", "clear"))
        data = (b4 > 0) & (b8 > 0) & (b11 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi = np.where(data, (b8 - b4) / (b8 + b4), np.nan)
            lswi = np.where(data, (b8 - b11) / (b8 + b11), np.nan)
        for mask, usable in (("qa60", data & ~qa60_cloud(qa)), ("cs", data & (clear >= cs_clear_min))):
            rows.append({"date": date, "mask": mask, **summarise(ndvi, lswi, usable, canopy_ndvi)})
    return pd.DataFrame(rows).sort_values(["date", "mask"]).reset_index(drop=True)


def plot(profile: pd.DataFrame, title: str = "", min_usable_pct: float = MIN_USABLE_PCT,
         shade=(("2026-05-01", "2026-09-30", "monsoon 2026"),), out_path=None):
    """NDVI percentiles, canopy share and water share over time, one colour per mask."""
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, SERIES_COLORS, SURFACE

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, facecolor=SURFACE)
    for (mask, g), colour in zip(profile.groupby("mask"), (SERIES_COLORS[0], SERIES_COLORS[2])):
        g = g[g["usable_pct"] >= min_usable_pct]
        label = "Cloud Score+ clear" if mask == "cs" else "QA60 clear"
        a1.fill_between(g["date"], g["ndvi_p10"], g["ndvi_p90"], color=colour, alpha=.15, lw=0)
        a1.plot(g["date"], g["ndvi_p50"], "o-", ms=4, color=colour, label=f"{label}: median NDVI (band p10-p90)")
        a1.plot(g["date"], g["ndvi_p90"], ":", color=colour, lw=1)
        a2.plot(g["date"], g["canopy_pct"], "o-", ms=4, color=colour, label=f"{label}: % canopy (NDVI>{CANOPY_NDVI})")
        a2.plot(g["date"], g["water_pct"], "s--", ms=4, color=colour, alpha=.7, label=f"{label}: % water (LSWI>NDVI)")
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        for start, end, label in shade:
            a.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=INK_MUTED, alpha=.08, lw=0)
        a.grid(True, alpha=.2)
        a.legend(loc="upper right", fontsize=8.5, frameon=False)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    a1.set_ylabel("NDVI over AOI pixels")
    a2.set_ylabel("% of usable AOI pixels")
    fig.suptitle(title or f"dates with at least {min_usable_pct:.0f}% of the AOI usable", x=0.01, ha="left",
                 color=INK)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    return fig
