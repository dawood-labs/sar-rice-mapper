"""Figures for public posts about the method: plain labels, no place, no area id, no satellite track.

Why
---
The project is explained to a general audience (managers, recruiters, other engineers) in short posts. The figures
must show the real method on real data, but nothing that could locate the fields or name the client: no area or field
id, no coordinates, no track numbers, no satellite chips or map outlines (a field shape can be found again). Only
month names, the measured curves and plain words ("flooded", "growing") appear on the image.

Two figures:

* :func:`radar_story`: one field's season from the real series: optical greenness (clear views only) and radar
  backscatter, with the rice phases marked (dry-season crop, harvest, flooded for transplanting, new crop).
* :func:`clouds_diagram`: a drawn sketch (no data): monsoon clouds stop the optical satellite, the radar sees through.

Use::

    python -m sar_pipeline.outreach_figures radar-story <field_id> --out processed/_batch/s2_2026/outreach/radar_story.png
    python -m sar_pipeline.outreach_figures clouds --out processed/_batch/s2_2026/outreach/clouds.png

Outputs go under ``processed/`` (not tracked). Check every image by eye before posting it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INK = "#1f2933"
MUTED = "#6b7785"
GREEN = "#2e8b57"
BLUE = "#2a78d6"
WATER = "#9cc7ef"
BG = "#ffffff"
# publication style (dataviz reference palette: categorical slots 1-3 validate all-pairs in light and dark mode)
SURFACE = "#fcfcfb"
TEXT_1 = "#0b0b0b"
TEXT_2 = "#52514e"
TEXT_3 = "#8a8983"
HAIRLINE = "#e4e3df"
SLOT_1 = "#2a78d6"   # radar VV
SLOT_2 = "#eb6834"   # radar VH
SLOT_3 = "#1baf7a"   # optical greenness (below 3:1 on the surface: always direct-labelled)
PHASE = "#f0efec"    # neutral band for the crop phases
SEASON = ("2026-03-15", "2026-09-24")


def field_curves(field_id: str, track: int = 0) -> dict:
    """Median clear NDVI per 5-day window and median VV / VH per radar pass of one track, over the field's pixels."""
    import warnings

    from .analysis import ndvi_5day as nd
    from .analysis import radar_water as rw
    from .mask_audit import pixels_of

    aoi_id, pids = pixels_of(field_id)
    d = nd.load(aoi_id)
    raw = d["ndvi5d_raw"].reshape(d["ndvi5d_raw"].shape[0], -1)[:, pids]
    windows = pd.DatetimeIndex(d["windows"])
    nd.forget()
    dates, flat = rw.read_series(aoi_id)[track]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ndvi = np.nanmedian(raw, axis=1)
        vv = np.nanmedian(flat["VV"][:, pids], axis=1)
        vh = np.nanmedian(flat["VH"][:, pids], axis=1)
    return {"windows": windows, "ndvi": ndvi, "radar_dates": pd.DatetimeIndex(dates), "vv": vv, "vh": vh}


def radar_story(curves: dict, phases=None, out: str | None = None, title: str = "The radar story of a rice field"):
    """Draw the two-panel season figure. ``phases``: list of (start, end, label, colour) shaded on both panels."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    t0, t1 = pd.Timestamp(SEASON[0]), pd.Timestamp(SEASON[1])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.75), sharex=True, dpi=150,
                                 gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.08})
    fig.patch.set_facecolor(BG)
    for i, (start, end, label, colour) in enumerate(phases or []):
        for a in (a1, a2):
            a.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=colour, alpha=0.18, lw=0)
        # labels of neighbouring (short) phases alternate in height so they never overlap
        a1.text(pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2, 1.03 + 0.1 * (i % 2), label,
                ha="center", va="bottom", fontsize=11, color=INK, transform=a1.get_xaxis_transform())
    w = curves["windows"]
    m = (w >= t0) & (w <= t1) & np.isfinite(curves["ndvi"])
    a1.plot(w[m], curves["ndvi"][m], "o-", color=GREEN, lw=2, ms=6)
    a1.set_ylabel("Greenness (optical)", color=INK, fontsize=11)
    a1.set_ylim(min(-0.1, float(np.nanmin(curves["ndvi"][m])) - 0.05), 1.0)
    a1.text(0.01, 0.06, "dots: the few cloud-free views", transform=a1.transAxes, fontsize=9, color=MUTED)
    last_clear = w[m][-1]
    if last_clear < t1 - pd.Timedelta(days=30):
        a1.text(last_clear + (t1 - last_clear) / 2, 0.45, f"no cloud-free view after {last_clear.day} {last_clear:%B}:\nonly the radar can follow the crop",
                ha="center", va="center", fontsize=11, color=INK, style="italic")
    r = curves["radar_dates"]
    k = (r >= t0) & (r <= t1)
    a2.plot(r[k], curves["vv"][k], "o-", color=BLUE, lw=2, ms=5, label="Radar VV")
    a2.plot(r[k], curves["vh"][k], "o--", color="#7a3ab4", lw=2, ms=5, label="Radar VH")
    a2.set_ylabel("Radar backscatter (dB)", color=INK, fontsize=11)
    a2.legend(frameon=False, fontsize=10, loc="lower left")
    a2.text(0.01, 0.92, "radar: every pass, clouds or not", transform=a2.transAxes, fontsize=9, color=MUTED)
    for a in (a1, a2):
        a.set_facecolor(BG)
        a.grid(True, alpha=0.25)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.tick_params(colors=INK, labelsize=10)
    a2.xaxis.set_major_locator(mdates.MonthLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    a2.set_xlim(t0, t1)
    fig.suptitle(title, x=0.06, ha="left", fontsize=16, color=INK, fontweight="bold", y=0.99)
    fig.text(0.06, 0.925, "Water at transplanting makes the radar signal drop; the growing crop brings it back.",
             fontsize=11, color=MUTED)
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.07, right=0.98)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=BG)
    return fig


def clouds_diagram(out: str | None = None):
    """A drawn sketch: the optical satellite's view stops at the monsoon clouds, the radar passes through to the field."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.75)
    ax.axis("off")
    # ground: paddies with water and seedlings
    ax.add_patch(Rectangle((0, 0), 12, 1.1, color="#d9c7a3"))
    for x in np.arange(0.3, 12, 1.45):
        ax.add_patch(Rectangle((x, 0.35), 1.25, 0.55, color=WATER))
        for s in np.linspace(x + 0.15, x + 1.1, 5):
            ax.plot([s, s], [0.55, 0.85], color=GREEN, lw=2)
    # clouds
    for cx, cy, wdt in ((2.2, 3.7, 3.2), (4.2, 3.9, 3.4), (6.3, 3.6, 3.6), (8.6, 3.9, 3.3), (10.4, 3.7, 3.0)):
        ax.add_patch(Ellipse((cx, cy), wdt, 1.3, color="#c9d1d9", alpha=0.95))
    ax.text(6, 3.2, "monsoon clouds", ha="center", fontsize=12, color=INK)

    def satellite(x, y, colour, name):
        ax.add_patch(Rectangle((x - 0.35, y - 0.18), 0.7, 0.36, color=colour))
        ax.add_patch(Rectangle((x - 1.05, y - 0.08), 0.6, 0.16, color="#9aa5b1"))
        ax.add_patch(Rectangle((x + 0.45, y - 0.08), 0.6, 0.16, color="#9aa5b1"))
        ax.text(x, y + 0.35, name, ha="center", fontsize=12, color=INK, fontweight="bold")

    satellite(2.8, 5.55, GREEN, "Optical satellite")
    satellite(9.0, 5.55, BLUE, "Radar satellite")
    ax.add_patch(FancyArrowPatch((2.8, 5.3), (3.2, 4.45), arrowstyle="-|>", mutation_scale=22, color=GREEN, lw=2.5))
    ax.text(3.35, 4.9, "blocked", fontsize=12, color=GREEN)
    ax.plot([3.05, 3.35], [4.35, 4.65], color="#c0392b", lw=3)
    ax.plot([3.05, 3.35], [4.65, 4.35], color="#c0392b", lw=3)
    ax.add_patch(FancyArrowPatch((9.0, 5.3), (8.3, 1.0), arrowstyle="-|>", mutation_scale=22, color=BLUE, lw=2.5))
    ax.add_patch(FancyArrowPatch((8.0, 1.0), (7.0, 1.9), arrowstyle="-|>", mutation_scale=18, color=BLUE, lw=1.8,
                                 linestyle="--"))
    ax.text(9.2, 2.3, "sees through clouds,\nday and night", fontsize=12, color=BLUE)
    ax.text(6.3, 1.95, "calm water reflects the signal away:\na flooded paddy looks dark", fontsize=10, color=INK,
            ha="right")
    ax.text(0.2, 6.5, "Why radar for monsoon rice", fontsize=16, color=INK, fontweight="bold")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=BG)
    return fig


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=HAIRLINE, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(HAIRLINE)
    ax.tick_params(colors=TEXT_2, labelsize=10, length=0)


def monthly_clear_share(dates_csv: str) -> pd.DataFrame:
    """Share of pixel-dates the cloud mask keeps, per month, pixel-weighted over the AOIs of a ``mask_share`` table
    (dates whose Cloud Score+ is missing are left out)."""
    d = pd.read_csv(dates_csv)
    d = d[~d["cs_missing"]]
    d["month"] = pd.to_datetime(d["date"]).dt.to_period("M")
    rows = [{"month": m, "kept": float(np.average(g["kept"], weights=g["pixels"]))} for m, g in d.groupby("month")]
    return pd.DataFrame(rows)


def season_events(curves: dict, before_passes: int = 5) -> dict:
    """The numbers written on the infographic, read from the curves: the flood = the darkest VH pass of the season, the
    level before it = the highest VH of the ``before_passes`` passes before, the canopy = the highest VH after it, and
    the last cloud-free optical view."""
    t0, t1 = pd.Timestamp(SEASON[0]), pd.Timestamp(SEASON[1])
    r = curves["radar_dates"]
    k = np.flatnonzero((r >= t0) & (r <= t1) & np.isfinite(curves["vh"]))
    vh = curves["vh"]
    f = k[np.argmin(vh[k])]
    before = [i for i in k if i < f][-before_passes:]
    b = before[int(np.argmax(vh[before]))]
    after = [i for i in k if i > f]
    a = after[int(np.argmax(vh[after]))]
    w = curves["windows"]
    seen = np.flatnonzero((w >= t0) & (w <= t1) & np.isfinite(curves["ndvi"]))
    return {"before": (r[b], float(vh[b])), "flood": (r[f], float(vh[f])), "after": (r[a], float(vh[a])),
            "last_clear": w[seen[-1]]}


def season_infographic(curves: dict, monthly: pd.DataFrame, plots: dict, tiles: list, events: dict,
                       out: str | None = None):
    """One page for a general audience: headline numbers, one field's season (optical and radar, separate panels, the
    crop phases and the measured changes written on the chart), the monthly share of cloud-free views, and the
    accuracy on surveyed plots. No place, no id, no track appears.

    ``plots``: {"Region A": 99.3, ...} share of surveyed rice plots called rice; ``tiles``: [(big, small), ...];
    ``events``: dates and values to annotate: {"phases": [(start, end, label)], "flood": (date, dB), "before": (date, dB),
    "after": (date, dB), "last_clear": date}."""
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 12), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[0.62, 2.35, 1.35], hspace=0.33, wspace=0.28,
                  left=0.06, right=0.97, top=0.885, bottom=0.06)
    fig.text(0.06, 0.975, "Mapping monsoon rice through the clouds", fontsize=24, fontweight="bold", color=TEXT_1,
             va="top")
    fig.text(0.06, 0.937, "Radar (Sentinel-1) sees the flooding and the growing crop when optical satellites cannot. "
             "Optical views (Sentinel-2) are used wherever the sky is clear.", fontsize=12.5, color=TEXT_2, va="top")
    # headline tiles
    for i, (big, small) in enumerate(tiles[:4]):
        ax = fig.add_subplot(gs[0, i])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f4f3f0", lw=0))
        ax.text(0.06, 0.62, big, fontsize=26, fontweight="bold", color=TEXT_1, transform=ax.transAxes, va="center")
        ax.text(0.06, 0.24, small, fontsize=10.5, color=TEXT_2, transform=ax.transAxes, va="center", wrap=True)
    t0, t1 = pd.Timestamp(SEASON[0]), pd.Timestamp(SEASON[1])
    inner = gs[1, :].subgridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.12)
    a1 = fig.add_subplot(inner[0])
    a2 = fig.add_subplot(inner[1], sharex=a1)
    for k, (start, end, label) in enumerate(events.get("phases", [])):
        for a in (a1, a2):
            a.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=PHASE if k % 2 == 0 else SURFACE, lw=0, zorder=0)
        a1.text(pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2, 1.06, label, ha="center",
                fontsize=11, color=TEXT_2, transform=a1.get_xaxis_transform(), fontweight="bold")
    w = curves["windows"]
    m = (w >= t0) & (w <= t1) & np.isfinite(curves["ndvi"])
    a1.plot(w[m], curves["ndvi"][m], "-", color=SLOT_3, lw=2, zorder=3)
    a1.plot(w[m], curves["ndvi"][m], "o", color=SLOT_3, ms=8, mec=SURFACE, mew=2, zorder=4)
    _style(a1)
    a1.set_ylim(-0.1, 1.05)
    a1.set_ylabel("Greenness (NDVI)", color=TEXT_2, fontsize=11)
    a1.text(0.005, 0.04, "optical: cloud-free views only", color=TEXT_2, fontsize=10, transform=a1.transAxes)
    lc = pd.Timestamp(events["last_clear"])
    a1.annotate(f"last cloud-free view: {lc.day} {lc:%b}", xy=(lc, float(curves["ndvi"][m][-1])),
                xytext=(lc + pd.Timedelta(days=12), 0.35), fontsize=10.5, color=TEXT_1,
                arrowprops=dict(arrowstyle="-", color=TEXT_3, lw=1))
    a1.text(lc + (t1 - lc) / 2, 0.62, f"no clear optical view for {(t1 - lc).days // 7} weeks", ha="center",
            fontsize=12, color=TEXT_3, style="italic")
    r = curves["radar_dates"]
    k = (r >= t0) & (r <= t1)
    for series, colour, name in (("vv", SLOT_1, "VV"), ("vh", SLOT_2, "VH")):
        a2.plot(r[k], curves[series][k], "-", color=colour, lw=2, zorder=3)
        a2.plot(r[k], curves[series][k], "o", color=colour, ms=7, mec=SURFACE, mew=2, zorder=4)
        last = np.flatnonzero(k)[-1]
        a2.text(r[last] + pd.Timedelta(days=3), curves[series][last], f"Radar {name}", color=TEXT_1, fontsize=10.5,
                va="center")
    _style(a2)
    a2.set_ylabel("Radar backscatter (dB)", color=TEXT_2, fontsize=11)
    a2.text(r[k][0], a2.get_ylim()[1] if False else float(np.nanmax(curves["vv"][k])) + 1.2,
            "radar: every pass, 12-day revisit, clouds or not", color=TEXT_2, fontsize=10)
    (bd, bv), (fd, fv), (ad, av) = events["before"], events["flood"], events["after"]
    a2.annotate(f"flooded for transplanting: {bv - fv:.1f} dB drop", xy=(pd.Timestamp(fd), fv),
                xytext=(pd.Timestamp(fd) - pd.Timedelta(days=70), fv - 0.8), fontsize=10.5, color=TEXT_1,
                arrowprops=dict(arrowstyle="-", color=TEXT_3, lw=1), va="center")
    a2.annotate(f"canopy grows back: +{av - fv:.1f} dB", xy=(pd.Timestamp(ad), av),
                xytext=(pd.Timestamp(ad) - pd.Timedelta(days=8), av - 5.5), fontsize=10.5, color=TEXT_1,
                arrowprops=dict(arrowstyle="-", color=TEXT_3, lw=1), ha="center")
    a2.set_ylim(float(np.nanmin(curves["vh"][k])) - 2.5, float(np.nanmax(curves["vv"][k])) + 3)
    a2.xaxis.set_major_locator(mdates.MonthLocator())
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    a2.set_xlim(t0, t1 + pd.Timedelta(days=16))
    plt.setp(a1.get_xticklabels(), visible=False)
    a1.set_title("One field, one season: dry-season crop, harvest, flooding, new rice crop", loc="left", fontsize=13,
                 color=TEXT_1, pad=26, fontweight="bold")
    # monthly cloud-free share
    b = fig.add_subplot(gs[2, :2])
    months = [p.strftime("%b") for p in monthly["month"]]
    vals = 100 * monthly["kept"].to_numpy()
    monsoon = np.array([p.month in (6, 7, 8) for p in monthly["month"]])
    b.bar(range(len(vals)), vals, color=np.where(monsoon, SLOT_1, "#b7d3f6"), width=0.72, zorder=3)
    for i in np.flatnonzero(monsoon):
        b.text(i, vals[i] + 2, f"{vals[i]:.0f}%", ha="center", fontsize=10.5, color=TEXT_1, fontweight="bold")
    b.set_xticks(range(len(vals)), months)
    b.set_ylim(0, 105)
    _style(b)
    b.set_ylabel("% of observations cloud-free", color=TEXT_2, fontsize=11)
    b.set_title("Cloud-free optical views per month (Oct 2025 - Sep 2026)", loc="left", fontsize=13, color=TEXT_1,
                fontweight="bold")
    b.text(0.99, 0.95, "monsoon (Jun-Aug) in dark blue", transform=b.transAxes, ha="right", fontsize=10, color=TEXT_2)
    # accuracy on surveyed plots
    c = fig.add_subplot(gs[2, 2:])
    names = list(plots)
    v = np.array([plots[n] for n in names])
    c.barh(range(len(v)), v, color=SLOT_1, height=0.55, zorder=3)
    for i, x in enumerate(v):
        c.text(x - 0.6, i, f"{x:.1f}%", va="center", ha="right", fontsize=12, color=SURFACE, fontweight="bold")
    c.set_yticks(range(len(v)), names)
    c.invert_yaxis()
    c.set_xlim(0, 100)
    _style(c)
    c.set_xlabel("% of ground-surveyed rice plots mapped as rice", color=TEXT_2, fontsize=11)
    c.set_title("Checked against 2,950 surveyed rice plots", loc="left", fontsize=13, color=TEXT_1, fontweight="bold")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE)
    return fig


def pipeline_figure(out: str | None = None):
    """The method as one flow: inputs, the five processing steps with their key facts, the output and the checks."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.4, 8.55, "From satellite passes to a field-by-field rice map", fontsize=24, fontweight="bold",
            color=TEXT_1, va="top")
    ax.text(0.4, 7.95, "Every step is a measured, testable rule: each field is judged against its own history and "
            "against the area's confirmed rice, not against fixed cut-offs.", fontsize=12.5, color=TEXT_2, va="top")

    def box(x, y, w, h, n, title, lines, accent):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f4f3f0",
                                    ec=HAIRLINE, lw=1))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.09), w, 0.09, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=accent, ec="none"))
        two = "\n" in title
        if n:
            ax.text(x + 0.18, y + h - 0.35, n, fontsize=15, color=accent, fontweight="bold", va="top")
        ax.text(x + (0.55 if n else 0.2), y + h - 0.35, title, fontsize=12, color=TEXT_1, fontweight="bold",
                va="top", linespacing=1.15)
        top = y + h - (1.35 if two else 1.0)
        for i, line in enumerate(lines):
            ax.text(x + 0.2, top - 0.42 * i, line, fontsize=9.6, color=TEXT_2, va="center")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=TEXT_3, lw=1.4))

    box(0.4, 4.55, 2.85, 2.65, "", "Radar: Sentinel-1", ["C-band, VV + VH, 10 m", "2-3 orbits per area",
                                                          "12-day revisit per orbit", "day and night, any weather"], SLOT_1)
    box(0.4, 1.4, 2.85, 2.65, "", "Optical: Sentinel-2", ["10 m, every 5 days", "~100 scenes per area / year",
                                                           "under 1 in 10 clear in", "the monsoon months"], SLOT_3)
    steps = [
        ("1", "Screen clouds\nand haze", ["opaque-cloud flag", "Cloud Score+ per pixel", "blue-haze test",
                                           "no score = unknown"]),
        ("2", "Build the\nseason curve", ["best view per 5 days", "upper-envelope fit", "gaps stay gaps",
                                           "sowing = last bare spell"]),
        ("3", "Find the\nwater", ["fall below own level", "as dark as the area's", "flooded fields",
                                   "one orbit is enough"]),
        ("4", "Follow the\ncrop", ["rise vs. own noise", "all orbits combined", "young / standing / cut",
                                    "trees, ponds ruled out"]),
        ("5", "Label every\nfield", ["~430,000 fields", "majority class", "confidence note",
                                     "full audit trail"]),
    ]
    xs = [3.6, 6.0, 8.4, 10.8, 13.2]
    for (n, title, lines), x in zip(steps, xs):
        box(x, 2.75, 2.25, 3.2, n, title, lines, SLOT_1 if n in ("3", "4") else TEXT_3)
    arrow(3.25, 5.9, 3.55, 4.9)
    arrow(3.25, 2.7, 3.55, 3.7)
    for a, b_ in zip(xs[:-1], xs[1:]):
        arrow(a + 2.25, 4.35, b_ - 0.02, 4.35)
    box(3.6, 0.4, 11.85, 1.7, "", "Checked before anything is delivered",
        ["2,950 ground-surveyed rice plots  |  tree cover, bare ground, water and harvested fields as negatives  |  "
         "expert review in GIS", "no change is kept if it lowers plot accuracy or raises false alarms"],
        SLOT_2)
    arrow(9.52, 2.7, 9.52, 2.15)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE)
    return fig


def pattern_medians(curves_parquet: str, groups_csv: str, groups) -> dict:
    """Median NDVI curve of the given plot groups (``analysis/plot_clusters`` outputs): {group: Series by date}."""
    both = pd.read_parquet(curves_parquet)
    lab = pd.read_csv(groups_csv)["group"].to_numpy()
    x = both.filter(like="ndvi_")
    days = pd.to_datetime([c[5:] for c in x.columns])
    return {g: pd.Series(np.nanmedian(x.to_numpy(dtype=float)[lab == g], axis=0), index=days) for g in groups}


def patterns_infographic(examples: list, traits: list, tiles: list, out: str | None = None):
    """Results of the pattern-labelling approach: headline tiles, three real pattern curves (one rice, two not) and the
    learned traits that separate them (rice vs not rice, small multiples: one unit per panel).

    ``examples``: [(label, Series by date, colour)]; ``traits``: [(name, unit, rice value, not-rice value)]."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.04, 0.955, "Teaching a model what rice looks like: from patterns, not pixels", fontsize=25,
             fontweight="bold", color=TEXT_1, va="top")
    fig.text(0.04, 0.905, "Field curves are grouped into patterns, an expert labels each pattern once, and a small model "
             "learns the traits that make a curve rice.", fontsize=13, color=TEXT_2, va="top")
    for i, (big, small) in enumerate(tiles):
        x0 = 0.04 + i * 0.235
        ax = fig.add_axes([x0, 0.72, 0.215, 0.14])
        ax.set_facecolor("#f4f3f0")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.text(0.07, 0.68, big, fontsize=30, fontweight="bold", color=TEXT_1, va="center", transform=ax.transAxes)
        ax.text(0.07, 0.27, small, fontsize=11.5, color=TEXT_2, va="center", transform=ax.transAxes, linespacing=1.3)
    ax = fig.add_axes([0.06, 0.1, 0.5, 0.52])
    _style(ax)
    for ex in examples:
        label, series, colour = ex[:3]
        dy = ex[3] if len(ex) > 3 else 0.0
        ax.plot(series.index, series.values, color=colour, lw=2.4)
        ax.text(series.index[-1] + pd.Timedelta(days=3), series.values[-1] + dy, label, color=TEXT_1, fontsize=10.5,
                va="center")
    ax.set_ylim(-0.4, 1.0)
    ax.set_xlim(series.index[0], series.index[-1] + pd.Timedelta(days=75))
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_ylabel("Greenness (NDVI), median of the pattern", color=TEXT_2, fontsize=11)
    ax.set_title("Three of the labelled patterns", loc="left", fontsize=14, color=TEXT_1, fontweight="bold")
    ax.axhline(0, color=HAIRLINE, lw=1)
    ax.text(ax.get_xlim()[0], -0.36, " below 0 = open water", fontsize=9.5, color=TEXT_3)
    for i, (name, unit, rice, other) in enumerate(traits):
        a = fig.add_axes([0.66, 0.46 - i * 0.18, 0.3, 0.13])
        _style(a)
        a.barh([1, 0], [rice, other], color=[SLOT_3, TEXT_3], height=0.55)
        a.set_yticks([1, 0])
        a.set_yticklabels(["rice", "not rice"], fontsize=10.5, color=TEXT_2)
        for y, v in ((1, rice), (0, other)):
            a.text(v, y, f"  {v:g} {unit}", va="center", fontsize=10.5, color=TEXT_1)
        a.set_xlim(0, max(rice, other) * 1.45)
        a.set_xticks([])
        a.grid(False)
        a.set_title(name, loc="left", fontsize=11.5, color=TEXT_1, fontweight="bold")
    fig.text(0.66, 0.655, "What the model learned", fontsize=14, color=TEXT_1, fontweight="bold", va="top")
    fig.text(0.66, 0.628, "median of the labelled curves", fontsize=10.5, color=TEXT_2, va="top")
    fig.text(0.04, 0.035, "Only the curve from sowing to the latest image decides. Satellite data: Sentinel-1 radar "
             "and Sentinel-2 optical, 2026 monsoon season.", fontsize=10, color=TEXT_3)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE)
    return fig


def patterns_flow(out: str | None = None):
    """The fresh-start method as one flow: inputs, six steps with their key facts, output and checks."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.text(0.4, 8.6, "Rebuilding a rice map from simple, observable steps", fontsize=24, fontweight="bold",
            color=TEXT_1, va="top")
    ax.text(0.4, 8.0, "Each step answers one question a person can check on the image; thresholds come from the data "
            "or the ground survey, not from guesses.", fontsize=12.5, color=TEXT_2, va="top")

    def box(x, y, w, h, n, title, lines, accent):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f4f3f0",
                                    ec=HAIRLINE, lw=1))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.09), w, 0.09, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=accent, ec="none"))
        if n:
            ax.text(x + 0.15, y + h - 0.35, n, fontsize=15, color=accent, fontweight="bold", va="top")
        ax.text(x + (0.48 if n else 0.2), y + h - 0.35, title, fontsize=11.5, color=TEXT_1, fontweight="bold",
                va="top", linespacing=1.15)
        top = y + h - (1.3 if "\n" in title else 0.95)
        for i, line in enumerate(lines):
            ax.text(x + 0.15, top - 0.4 * i, line, fontsize=9.3, color=TEXT_2, va="center")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=TEXT_3, lw=1.4))

    box(0.3, 5.2, 2.3, 2.2, "", "Optical (Sentinel-2)", ["10 m, every 5 days", "greenness curve", "cloudy monsoon"],
        SLOT_3)
    box(0.3, 2.75, 2.3, 2.2, "", "Radar (Sentinel-1)", ["sees through cloud", "water and crop growth",
                                                        "12-day revisit"], SLOT_1)
    box(0.3, 0.3, 2.3, 2.2, "", "Ground survey", ["~2,950 plots", "reported as rice", "checked, not trusted"], SLOT_2)
    steps = [("1", "First clear\nview", ["per pixel, from", "1 September", "clear = near its own", "best cloud score"]),
             ("2", "Keep the\nvegetation", ["green vs not green", "split read from", "the image itself"]),
             ("3", "Crop ground\nor trees", ["went bare since May?", "bare level from the", "ground survey (~0.3)"]),
             ("4", "Sowing\ndate", ["last low of the curve", "late lows need the", "radar to agree"]),
             ("5", "Label\npatterns", ["curves grouped", "into 20 patterns", "expert labels each", "by its median"]),
             ("6", "Learn\nthe traits", ["rise, top, speed,", "radar rise after", "sowing: small tree", "with readable rules"])]
    xs = [3.0, 5.15, 7.3, 9.45, 11.6, 13.75]
    for (n, title, lines), x in zip(steps, xs):
        box(x, 2.9, 1.95, 3.6, n, title, lines, SLOT_1 if n in ("5", "6") else TEXT_3)
    for y in (6.3, 3.85):
        arrow(2.6, y, 2.95, 4.7)
    arrow(2.6, 2.3, 12.4, 2.88)            # the survey labels feed the pattern step (routed above the output box)
    for a_, b_ in zip(xs[:-1], xs[1:]):
        arrow(a_ + 1.95, 4.7, b_ - 0.02, 4.7)
    box(3.0, 0.3, 12.7, 1.55, "", "Output and checks",
        ["monsoon rice  |  late rice (sown in August, harvested Nov-Dec)  |  not rice  |  trees  |  no vegetation",
         "each region is judged by a model trained on the others; the expert reviews every pattern and field sample"],
        SLOT_2)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=SURFACE)
    return fig


# ---------------------------------------------------------------------------- the 132-area delivery milestone
# Why (7 Oct 2026): every area delivered is a milestone the user shares; these two figures are built from the
# delivery's own summary files, so every number on them is the delivered number. Nothing that can locate a place.
import glob  # noqa: E402
import json  # noqa: E402

DELIVERY = "processed/_batch/s2_2026/rice_map_2026-10-05"
CHOSEN = "src/sar_pipeline/analysis/aoi_rules_chosen.json"
REVIEW = "processed/_batch/s2_2026/rice_fresh"
OUT = "processed/outreach"

#: The delivery-milestone figures (7 Oct 2026) use the dataviz default palette, kept apart from the older figures' constants.
D_SURFACE, D_INK, D_INK2, D_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
D_BLUE, D_ORANGE, D_GREEN = "#2a78d6", "#eb6834", "#1baf7a"


def _candidate_sets() -> int:
    from .analysis import aoi_batch as ab

    return len(ab.CANDIDATE_SETS)


def delivery_numbers(delivery: str = DELIVERY, chosen: str = CHOSEN, review: str = REVIEW) -> dict:
    """The headline numbers of the delivery, read from its own summary files."""
    ty = [json.loads(Path(p).read_text()) for p in glob.glob(f"{delivery}/aoi*/aoi*_too_young.json")]
    tc = [json.loads(Path(p).read_text()) for p in glob.glob(f"{delivery}/aoi*/aoi*_trees_cut.json")]
    agree = [v["right_pct"] for v in json.loads(Path(chosen).read_text()).values()]
    return {"areas": len(glob.glob(f"{delivery}/aoi*/MANIFEST.json")),
            "rice_fields": sum(j["rice_fields"] for j in tc),
            "rice_acres_start": sum(j["rice_acres"] for j in ty),
            "too_young_acres": sum(j["too_young_acres"] for j in ty),
            "too_young_fields": sum(j["too_young_fields"] for j in ty),
            "rice_acres_before_trees": sum(j["rice_acres_before"] for j in tc),
            "rice_acres_final": sum(j["rice_acres_after"] for j in tc),
            "fields_trimmed": sum(j["fields_cut"] for j in tc),
            "slivers_removed": sum(j["slivers_dropped"] for j in tc),
            "reviewed_fields": len(glob.glob(f"{review}/aoi*/field_review/verdicts/*.json")),
            "candidate_sets": _candidate_sets(),
            "agreement_pct": agree}


def _d_style(ax):
    ax.set_facecolor(D_SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(D_GRID)
    ax.tick_params(colors=D_INK2, labelsize=12, length=0)
    ax.grid(axis="y", color=D_GRID, lw=1)
    ax.set_axisbelow(True)


def delivery_results_figure(n: dict, path) -> Path:
    """Image 1: four headline tiles, the acres from first map to delivered map, and the spread of how well the chosen
    rule agreed with the blind review per area."""
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 12), dpi=100, facecolor=D_SURFACE)
    fig.text(0.05, 0.955, "Rice maps for 132 areas, field by field", fontsize=30, weight="bold", color=D_INK)
    fig.text(0.05, 0.92, "Radar + optical satellite series, AI-delineated field boundaries, blind-reviewed sample fields",
             fontsize=15, color=D_INK2)
    tiles = [(f"{n['areas']}", "areas delivered"),
             (f"{n['rice_fields'] / 1000:.0f}k", "rice fields labelled"),
             (f"{n['rice_acres_final']:,.0f}", "acres of rice delivered"),
             (f"{n['reviewed_fields'] / 1000:.1f}k", "sample fields blind-reviewed")]
    for i, (big, small) in enumerate(tiles):
        x = 0.05 + i * 0.235
        fig.patches.append(plt.Rectangle((x, 0.73), 0.215, 0.15, transform=fig.transFigure, facecolor="white",
                                         edgecolor=D_GRID, lw=1.2))
        fig.text(x + 0.015, 0.80, big, fontsize=34, weight="bold", color=D_INK)
        fig.text(x + 0.015, 0.755, small, fontsize=13, color=D_INK2)

    ax1 = fig.add_axes([0.07, 0.10, 0.40, 0.52])
    _d_style(ax1)
    start, ty, mid, fin = (n["rice_acres_start"], n["too_young_acres"], n["rice_acres_before_trees"],
                           n["rice_acres_final"])
    steps = [("First map", start, 0, D_BLUE), ("Too young\nremoved", ty, mid, D_ORANGE),
             ("Edge trees and\nroofs cut", mid - fin, fin, D_ORANGE), ("Delivered", fin, 0, D_GREEN)]
    for i, (lab, h, base, c) in enumerate(steps):
        ax1.bar(i, h, bottom=base, color=c, width=0.62, edgecolor=D_SURFACE, lw=2)
        txt = f"{h:,.0f} ac" if i in (0, 3) else f"-{h:,.0f} ac"
        ax1.text(i, base + h + 600, txt, ha="center", fontsize=13, color=D_INK, weight="bold")
    ax1.set_xticks(range(4), [s[0] for s in steps], fontsize=12, color=D_INK2)
    ax1.set_ylim(60000, start * 1.04)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    ax1.set_title("Rice acres, first map to delivery (axis starts at 60k)", loc="left", fontsize=15, color=D_INK, pad=14)

    ax2 = fig.add_axes([0.57, 0.10, 0.38, 0.52])
    _d_style(ax2)
    a = np.asarray(n["agreement_pct"])
    bins = np.arange(20, 101, 10)
    cnt, _ = np.histogram(a, bins)
    cols = [D_ORANGE if b < 50 else D_BLUE for b in bins[:-1]]
    ax2.bar(bins[:-1] + 5, cnt, width=8.6, color=cols, edgecolor=D_SURFACE, lw=2)
    for x, c in zip(bins[:-1] + 5, cnt):
        if c:
            ax2.text(x, c + 0.6, str(c), ha="center", fontsize=12, color=D_INK)
    ax2.set_xticks(bins, [f"{b}%" for b in bins], fontsize=11)
    ax2.set_ylim(0, cnt.max() * 1.45)                 # room above the bars for the two notes
    ax2.set_title("Areas by agreement of the chosen rule with the blind review", loc="left", fontsize=15, color=D_INK,
                  pad=14)
    low = int((a < 50).sum())
    ax2.text(0.98, 0.95, f"{low} areas below 50 %: flagged for a closer look", transform=ax2.transAxes, fontsize=12,
             ha="right",
             color=D_ORANGE, weight="bold")
    ax2.text(0.98, 0.89, f"median {np.median(a):.0f} % (seven-class match, the strictest test)", ha="right",
             transform=ax2.transAxes, fontsize=12, color=D_INK2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=D_SURFACE)
    plt.close(fig)
    return path


def delivery_method_figure(n: dict, path) -> Path:
    """Image 2: the method as a flow of numbered steps with their real facts."""
    import matplotlib
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=D_SURFACE)
    fig.text(0.04, 0.92, "From satellite series to clean field polygons", fontsize=28, weight="bold", color=D_INK)
    steps = [("1  Inputs", ["Radar every 6-12 days", "Optical where clear", "AI field boundaries", "0.3 m basemap"]),
             ("2  Rules per pixel", ["Water, growth, harvest", "Own-range signals", "No fixed dB cut-offs", f"{n.get('candidate_sets', 13)} candidate rule sets"]),
             ("3  Blind review", [f"{n['reviewed_fields']:,} sample fields", "Curves + clear views", "No map shown", "Best rule set per area"]),
             ("4  Field labels", [f"{n['rice_fields']:,} rice fields", "Majority per field", "Rice vs trees as one", "Small patches sieved"]),
             ("5  Clean-up", [f"{n['too_young_fields']} too-young fields out", f"{n['fields_trimmed']:,} edge trims", "Straight cuts", f"{n['slivers_removed']:,} slivers removed"])]
    w = 0.172
    for i, (head, facts) in enumerate(steps):
        x = 0.04 + i * (w + 0.018)
        fig.patches.append(plt.Rectangle((x, 0.30), w, 0.52, transform=fig.transFigure, facecolor="white",
                                         edgecolor=D_GRID, lw=1.2))
        fig.patches.append(plt.Rectangle((x, 0.76), w, 0.06, transform=fig.transFigure, facecolor=D_BLUE, lw=0))
        fig.text(x + 0.01, 0.778, head, fontsize=15, weight="bold", color="white")
        for k, f in enumerate(facts):
            fig.text(x + 0.012, 0.69 - k * 0.09, f, fontsize=13, color=D_INK)
        if i < len(steps) - 1:
            fig.text(x + w + 0.002, 0.55, "›", fontsize=26, color=D_INK2)
    fig.patches.append(plt.Rectangle((0.04, 0.08), 0.92, 0.15, transform=fig.transFigure, facecolor="white",
                                     edgecolor=D_GRID, lw=1.2))
    fig.text(0.055, 0.185, "How it is checked", fontsize=15, weight="bold", color=D_GREEN)
    fig.text(0.055, 0.135, "Every rule set scored on blind-reviewed fields  ·  tree cuts tested on a hand QC with half "
             "the area held out", fontsize=12.5, color=D_INK2)
    fig.text(0.055, 0.10, "No slivers or tails in the cleaned layers  ·  delivered files never overwritten, new "
             "files added beside them", fontsize=12.5, color=D_INK2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=D_SURFACE)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    import argparse

    import matplotlib

    matplotlib.use("Agg")
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.outreach_figures", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", choices=["radar-story", "clouds", "infographic", "pipeline", "patterns", "patterns-flow",
                                      "delivery-132"])
    p.add_argument("--clusters", default="processed/_batch/s2_2026/rice_fresh/plot_clusters",
                   help="patterns: the plot_clusters output folder")
    p.add_argument("--dates-csv", default="processed/_batch/s2_2026/report/mask_share_year/dates.csv",
                   help="infographic: a mask_share dates table covering the months to show")
    p.add_argument("field_id", nargs="?", help="radar-story: the field (not printed on the image)")
    p.add_argument("--phases", nargs="*", default=[], metavar="START,END,LABEL",
                   help="radar-story: shaded phases, e.g. 2026-04-01,2026-06-10,'Dry-season crop'")
    p.add_argument("--track", type=int, default=0)
    p.add_argument("--out", help="the image (for delivery-132: a folder for its two images)")
    args = p.parse_args(argv)
    if args.figure == "delivery-132":
        n = delivery_numbers()
        d = Path(args.out or OUT)
        print(delivery_results_figure(n, d / "delivery_132_results.png"))
        print(delivery_method_figure(n, d / "delivery_132_method.png"))
        return 0
    if args.out is None:
        p.error("--out is required")
    if args.figure == "patterns-flow":
        patterns_flow(args.out)
    elif args.figure == "patterns":
        c = Path(args.clusters)
        med = pattern_medians(str(c / "plot_curves.parquet"), str(c / "plot_groups_k20.csv"), (7, 1, 17))
        examples = [("rice: bare in June,\nfull canopy by September", med[7], SLOT_3, 0.07),
                    ("not rice: green all year", med[1], TEXT_3, -0.05),
                    ("not rice: drowned, no crop", med[17], SLOT_1)]
        # medians of the labelled curves (rice_features check, 30 Sep): rise and speed from sowing, radar VH rise
        traits = [("Rise in greenness after sowing", "NDVI", 0.65, 0.29),
                  ("Days to reach 90% of that rise", "days", 70, 30),
                  ("Radar (VH) rise after sowing", "dB", 4.5, 1.9)]
        tiles = [("2,950", "ground-surveyed plots\ngrouped into 20 patterns"),
                 ("5 of 20", "survey patterns turned out\nnot to be rice"),
                 ("160", "patterns labelled by an expert\n(survey + 7 test areas)"),
                 ("23 s \u2192 1.5 s", "to inspect one field\nin the review notebook")]
        patterns_infographic(examples, traits, tiles, args.out)
    elif args.figure == "clouds":
        clouds_diagram(args.out)
    elif args.figure == "pipeline":
        pipeline_figure(args.out)
    elif args.figure == "infographic":
        # headline numbers of the delivered map (docs/17 and the validation of the stage-10 delivery)
        tiles = [("~99%", "of 2,950 ground-surveyed rice\nplots mapped as rice"),
                 ("113,000+", "acres analysed in 132 areas,\nfield by field"),
                 ("6%", "of July observations were\ncloud-free (7 test areas)"),
                 ("12 days", "radar revisit per orbit:\nthe crop is never out of sight")]
        plots = {"Region A": 99.3, "Region B": 99.4, "Region C": 98.4}
        curves = field_curves(args.field_id, args.track)
        events = {"phases": [(a, b, c) for a, b, c in (s.split(",", 2) for s in args.phases)], **season_events(curves)}
        season_infographic(curves, monthly_clear_share(args.dates_csv), plots, tiles, events, args.out)
    else:
        colours = ["#f2c14e", "#e76f51", WATER, "#8bc34a", "#b0bec5"]
        phases = []
        for i, spec in enumerate(args.phases):
            start, end, label = spec.split(",", 2)
            phases.append((start, end, label, colours[i % len(colours)]))
        radar_story(field_curves(args.field_id, args.track), phases, args.out)
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
