"""Everything about one pixel of the current season on one page: curve, crop dates and 5-3-2 chips.

Why
---
Every decision about what is rice comes down to looking at pixels: does the smoothed curve follow the
observations, are the detected transplanting, green-up, peak and harvest dates where the field visibly
changes, and what does the field actually look like on those dates. Doing that by hand means
re-reading the exports, re-running the smoother and hunting for clear images each time — slow, and
easy to do slightly differently from the pipeline. This module does it once, the pipeline's way:

* the **curve** comes from ``ndvi_5day`` (QA60-masked dates, 5-day maximum composite, upper-envelope
  Whittaker), so it is exactly what the classifier sees;
* the **crop dates** come from ``optical_phenology.pixel_cycles`` run on that fitted curve — the same
  detector the maps use, so what is drawn is what is measured;
* the **chips** are 5-3-2 (red edge, green, blue) cut from the same per-date files, **centred on the
  pixel** with enough surroundings to see the whole field. By default **every date on which the pixel
  itself is clear** is shown (Cloud Score+ ``clear`` at the pixel >= ``CLEAR_MIN``; scored per 10 m
  pixel, so it is finer than QA60), because each of those dates is an observation the curve rests on.
  ``select="per_month"`` shows instead the ``per_month`` clearest dates of every month, ranked over
  the whole chip, which keeps one image per month even through the cloudy season.

Each chip is stretched **on its own**, 2nd-98th percentile per band over that chip's clear pixels
(``stretch="chip"``, the default): the sharpest contrast on every date. One stretch shared by all
chips (``stretch="series"``) keeps colours comparable between dates instead, but tried first it let
the bright bare soil of May and the haze of September set the range and rendered the canopy months
almost black. Cloud is excluded from the percentiles either way, or it would squash the fields into
the dark end.

**Transplanting, not sowing.** Rice here is transplanted (or direct-seeded) into a puddled field;
the nursery sowing that precedes it happens elsewhere and cannot be seen in the pixel. So the
field's start is the **trough** — the field as a pool of water with small seedlings, NDVI near zero —
and ``transplant_date`` is that start. It is dated from the green-up onset minus the measured lag
(``GREENUP_LAG_DAYS``) rather than from the trough directly, because on a field that sits low and
flat for weeks the detector's trough can land far from the real start; the trough is shown alongside.

Use it (``notebooks/07_pixel_2026.ipynb`` wraps this):

>>> from sar_pipeline.analysis import pixel_2026 as px
>>> r = px.report(143, 4484)
>>> r["info"]; r["cycles"]; r["curve"]; r["chips"]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from . import optical_phenology as op
from . import pixel_report as pr

#: Chip half-width in pixels: 30 gives 61 x 61 pixels, about 610 m across — a few fields of context.
HALF = 30
#: Clear dates shown per month.
PER_MONTH = 2
#: Cloud Score+ value (0-100) a chip pixel must reach to count as clear.
CLEAR_MIN = 60
#: Display gamma applied after the stretch. Below 1 lifts the dark end: a dense canopy is dark in the
#: red edge, and at 1.0 the peak-season chips stayed nearly black even with a per-chip stretch.
GAMMA = 0.7
#: Display bands for 5-3-2, in red, green, blue order.
RGB = ("B5", "B3", "B2")
#: Where the 2025 optical map lives, shown for context when it exists.
MAP_2025 = "processed/_batch/optical_v3"


def centered_window(row: int, col: int, half: int = HALF):
    """A (2*half+1)-pixel square window with the pixel exactly in the middle.

    Read with ``boundless=True``: near the grid edge the missing part is filled with nodata rather
    than the window sliding, so the pixel never moves off centre.
    """
    from rasterio.windows import Window

    return Window(col - half, row - half, 2 * half + 1, 2 * half + 1)


def chip_stack(aoi_id: int, pid: int, half: int = HALF):
    """Every date's 5-3-2 chip around the pixel, plus a per-date table of how clear the chip is.

    Returns ``(table, chips, clear)`` where ``chips`` is (dates, 3, size, size) reflectance x 10000
    and ``clear`` is (dates, size, size) Cloud Score+ 0-100 (0 where there is no data).
    """
    import rasterio

    from ..optical_export import band_index, qa60_cloud

    d = nd.load(aoi_id)
    loc = pr.locate(aoi_id, pid)
    folder = pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER)
    paths = sorted(folder.glob("*.tif"))
    win = centered_window(loc["row"], loc["col"], half)
    rows, chips, clears = [], [], []
    for path in paths:
        with rasterio.open(path) as ds:
            read = lambda n: ds.read(band_index(ds, n), window=win, boundless=True,  # noqa: E731
                                     fill_value=0).astype("float32")
            rgb = np.stack([read(b) for b in RGB])
            clear, qa = read("clear"), read("QA60")
        data = rgb.min(axis=0) > 0
        clear = np.where(data, clear, 0)
        rows.append({
            "date": pd.to_datetime(path.stem.rsplit("_S2_", 1)[1]),
            "chip_clear_pct": round(100 * float((clear >= CLEAR_MIN).sum()) / max(data.sum(), 1), 1),
            "chip_data_pct": round(100 * float(data.mean()), 1),
            "chip_qa60_cloud_pct": round(100 * float((qa60_cloud(qa) & data).sum()) / max(data.sum(), 1), 1),
            "pixel_clear": float(clear[half, half]),
        })
        chips.append(rgb)
        clears.append(clear)
    table = pd.DataFrame(rows)
    order = table.sort_values("date").index.to_numpy()
    return table.loc[order].reset_index(drop=True), np.stack(chips)[order], np.stack(clears)[order]


def pick_chips(table: pd.DataFrame, per_month: int = PER_MONTH, min_data_pct: float = 50) -> pd.DataFrame:
    """The ``per_month`` clearest dates of every month, in date order.

    Ranked by the clear share of the whole chip, then by the clear score at the pixel itself. Every
    month keeps its best dates even when they are cloudy, so a gap in the strip is never silent: the
    clear percentage is printed on each chip. Dates where most of the chip has no data at all (the
    AOI on the edge of a swath) are skipped.
    """
    usable = table[table["chip_data_pct"] >= min_data_pct].copy()
    usable["month"] = usable["date"].dt.to_period("M")
    best = (usable.sort_values(["chip_clear_pct", "pixel_clear"], ascending=False)
            .groupby("month", group_keys=False).head(per_month))
    return best.sort_values("date").drop(columns="month")


def pixel_clear_chips(table: pd.DataFrame, clear_min: float = CLEAR_MIN) -> pd.DataFrame:
    """Every date on which the target pixel itself is clear, in date order."""
    return table[table["pixel_clear"] >= clear_min].sort_values("date")


def stretch_limits(chips, clear, low: float = 2.0, high: float = 98.0):
    """Per-band display limits from the clear pixels of the given chips; see the module docstring."""
    good = clear >= CLEAR_MIN
    if not good.any():
        good = clear > 0
    limits = []
    for b in range(chips.shape[1]):
        values = chips[:, b][good]
        limits.append(np.percentile(values, [low, high]) if values.size else (0.0, 3000.0))
    return np.asarray(limits, dtype="float64")


def estimate_transplant(found: pd.DataFrame, lead_days: int = op.GREENUP_LAG_DAYS) -> pd.DataFrame:
    """Transplant date = green-up onset (acceleration) minus ``lead_days``; the trough only as a fallback.

    Keeps the detector's trough as ``trough_date`` so both can be compared; see the module docstring.
    """
    found = found.copy()
    if found.empty:
        return found
    found["trough_date"] = found["start_date"]
    lead = pd.Timedelta(days=lead_days)
    onset = pd.to_datetime(found["greenup_onset"])
    found["transplant_date"] = (onset - lead).where(onset.notna(),
                                                    pd.to_datetime(found["start_date"]) - lead)
    return found


def cycles(aoi_id: int, pid: int) -> tuple[pd.DataFrame, dict]:
    """Crop cycles measured on the pixel's fitted 5-day curve, and the pixel's series."""
    p = nd.pixel(aoi_id, pid)
    s = p["series"]
    found = op.pixel_cycles(pd.DatetimeIndex(s["window"]), s["ndvi_fit"].to_numpy(), None,
                            s["lswi_fit"].to_numpy(), observed=s["ndvi_composite"].notna().to_numpy())
    return estimate_transplant(pd.DataFrame(found)), p


def info(aoi_id: int, pid: int, p: dict, found: pd.DataFrame) -> pd.Series:
    """Everything known about the pixel that fits in one column."""
    loc = pr.locate(aoi_id, pid)
    raw, s = p["raw"], p["series"]
    inside = bool(nd.inside_aoi(aoi_id)[pid])
    out = {
        "aoi": loc["aoi"], "pid": pid, "row": loc["row"], "col": loc["col"],
        "lon": round(loc["lon"], 6), "lat": round(loc["lat"], 6), "inside AOI polygon": inside,
        "dates exported": len(raw),
        "dates kept (QA60 clear)": int(raw["qa60_clear"].sum()),
        "   of which SCL would remove": int((raw["qa60_clear"] & raw["scl_cloud"]).sum()),
        "5-day windows observed": f"{int(s['ndvi_composite'].notna().sum())} of {len(s)}",
        "longest gap (days)": int(s["gap_days"].max()),
        "NDVI fit min / max": f"{s['ndvi_fit'].min():.2f} / {s['ndvi_fit'].max():.2f}",
        "crop cycles found": len(found),
    }
    class_2025 = _class_2025(loc)
    if class_2025 is not None:
        out["2025 optical map (1 rice, 0 not, 255 undecided)"] = class_2025
    return pd.Series(out, name="value")


def _class_2025(loc):
    import rasterio

    path = Path(MAP_2025) / loc["aoi"] / f"{loc['aoi']}_rice_optical.tif"
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        return int(ds.read(1)[loc["row"], loc["col"]])


#: Crop-date columns shown in the cycles table and drawn on the curve, with their colours.
DATE_MARKS = (("transplant_date", "transplant", "#6b4f2a"), ("greenup_onset", "green-up", "#1baf7a"),
              ("peak_date", "peak", "#2a78d6"), ("harvest_date", "harvest", "#eb6834"))


def cycle_table(found: pd.DataFrame) -> pd.DataFrame:
    """The columns a person reads, dates as dates, lengths in days."""
    if found.empty:
        return found
    cols = ["cycle", "complete", "trough_date", "transplant_date", "greenup_onset", "greenup_date", "peak_date",
            "peak_ndvi", "harvest_date", "end_date", "greenup_to_harvest_days", "growth_amplitude",
            "lswi_at_start", "prewet_ndvi_drop", "prewet_lswi_rise", "wet_at_transplant"]
    table = found[[c for c in cols if c in found]].copy()
    for c in table.columns:
        if c.endswith("_date"):
            table[c] = pd.to_datetime(table[c]).dt.strftime("%d %b %Y")
    return table.round(3)


def plot_curve(p: dict, found: pd.DataFrame, chip_dates=(), width: float = 16, height: float = 7.5):
    """NDVI (raw dates, composite, fit) and LSWI, with the detected crop dates and the chip dates."""
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, INK_SECONDARY, SERIES_COLORS, SURFACE

    raw, s = p["raw"], p["series"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(width, height), sharex=True, facecolor=SURFACE,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for start, g in zip(s["window"], s["gap_days"]):
            if g > 15:
                a.axvspan(start, start + pd.Timedelta(days=nd.STEP_DAYS), color=INK_MUTED, alpha=.12, lw=0)
    kept = raw["qa60_clear"] & ~raw["scl_cloud"]
    scl_only = raw["qa60_clear"] & raw["scl_cloud"]
    ax.plot(raw["date"][~raw["qa60_clear"]], raw["ndvi"][~raw["qa60_clear"]], "x", ms=5,
            color=INK_MUTED, label="date removed by QA60 (cloud)")
    ax.plot(raw["date"][scl_only], raw["ndvi"][scl_only], "o", ms=5, mfc="none",
            color=SERIES_COLORS[1], label="date kept (SCL would remove)")
    ax.plot(raw["date"][kept], raw["ndvi"][kept], "o", ms=4.5, color=INK_SECONDARY, label="date kept")
    ax.plot(s["window"], s["ndvi_composite"], "s", ms=3, color=SERIES_COLORS[2], alpha=.45,
            label="5-day max composite")
    ax.plot(s["window"], s["ndvi_fit"], lw=2.6, color=SERIES_COLORS[2], label="NDVI fit (interpolated)")
    ax2.plot(raw["date"][raw["qa60_clear"]], raw["lswi"][raw["qa60_clear"]], "o", ms=3.5,
             color=INK_SECONDARY, alpha=.6, label="LSWI, kept dates")
    ax2.plot(s["window"], s["lswi_fit"], lw=2, color=SERIES_COLORS[0], label="LSWI fit (water)")
    ax2.axhline(0, lw=.8, color=INK_MUTED)
    for _, c in found.iterrows():
        for key, label, colour in DATE_MARKS:
            if pd.notna(c.get(key)):
                when = pd.Timestamp(c[key])
                for a in (ax, ax2):
                    a.axvline(when, ls="--", lw=1.6, color=colour)
                ax.annotate(f"{label}\n{when:%d %b}", (when, 1.02), xycoords=("data", "axes fraction"),
                            ha="center", va="bottom", fontsize=8, color=colour)
    for when in chip_dates:
        ax.plot([when], [-0.55], marker="^", ms=6, color=INK, clip_on=False)
    ax.set_ylim(-0.6, 1.0)
    ax.set_ylabel("NDVI")
    ax2.set_ylabel("LSWI")
    for a in (ax, ax2):
        a.grid(True, alpha=.2)
        a.legend(loc="upper right", fontsize=8, frameon=False, ncol=3)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    ax.text(0.0, -0.08, "▲ = date shown as a chip below   ·   grey band = filled from > 15 days away",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    fig.suptitle(f"{p['aoi']}  pid {p['pid']}", x=0.01, ha="left", y=1.0, fontsize=12, color=INK)
    fig.tight_layout()
    return fig


def plot_chips(picked: pd.DataFrame, chips, clear, dates, found: pd.DataFrame, half: int = HALF,
               stretch: str = "chip", cols: int = 7, size: float = 2.7, gamma: float = GAMMA, box: int = 3,
               outline=None):
    """The picked chips in date order, pixel marked, crop-date chips framed in the date's colour."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from .chips import to_display
    from .curves import INK, SURFACE

    idx = [int(np.flatnonzero(dates == d)[0]) for d in picked["date"]]
    limits = stretch_limits(chips[idx], clear[idx])
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, size * rows + 0.4), facecolor=SURFACE,
                             squeeze=False)
    crop_dates = {}
    for _, c in found.iterrows():
        for key, label, colour in DATE_MARKS:
            if pd.notna(c.get(key)):
                crop_dates[pd.Timestamp(c[key])] = (label, colour)
    for ax, i, (_, row) in zip(axes.ravel(), idx, picked.iterrows()):
        lim = stretch_limits(chips[i:i + 1], clear[i:i + 1]) if stretch == "chip" else limits
        ax.imshow(to_display(chips[i], lim) ** gamma, interpolation="nearest")
        if outline is not None:                       # (xs, ys) in chip pixel coordinates, e.g. a field
            ax.plot(outline[0], outline[1], color="#ff00ff", lw=1.2)
        else:
            ax.add_patch(Rectangle((half - box / 2, half - box / 2), box, box, fill=False, ec="#ff00ff", lw=1.4))
        label = f"{row['date']:%d %b %Y}\nclear {row['chip_clear_pct']:.0f}%"
        near = [v for k, v in crop_dates.items() if abs((k - row["date"]).days) <= 7]
        if near:
            label += "  · " + ", ".join(n[0] for n in near)
            for side in ax.spines.values():
                side.set_edgecolor(near[0][1])
                side.set_linewidth(3)
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(idx):]:
        ax.set_visible(False)
    fig.suptitle(f"5-3-2 (red edge, green, blue), {2 * half + 1} x {2 * half + 1} px "
                 f"(~{(2 * half + 1) * 10} m), pixel in the magenta box · stretch: {stretch}, gamma {gamma}",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def report(aoi_id: int, pid: int, half: int = HALF, per_month: int = PER_MONTH, stretch: str = "chip",
           gamma: float = GAMMA, select: str = "pixel_clear", out_dir=None) -> dict:
    """Info, crop cycles, the curve figure and the chip figure for one pixel.

    With ``out_dir`` both figures are also saved there as PNG.
    """
    found, p = cycles(aoi_id, pid)
    table, chips, clear = chip_stack(aoi_id, pid, half)
    picked = pixel_clear_chips(table) if select == "pixel_clear" else pick_chips(table, per_month)
    curve = plot_curve(p, found, chip_dates=list(picked["date"]))
    sheet = plot_chips(picked, chips, clear, table["date"].to_numpy(), found, half, stretch, gamma=gamma)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        curve.savefig(out / f"{p['aoi']}_pid{pid}_curve.png", dpi=110, bbox_inches="tight")
        sheet.savefig(out / f"{p['aoi']}_pid{pid}_chips.png", dpi=110, bbox_inches="tight")
    return {"info": info(aoi_id, pid, p, found), "cycles": cycle_table(found), "curve": curve,
            "chips": sheet, "chip_dates": table, "series": p["series"], "raw": p["raw"]}
