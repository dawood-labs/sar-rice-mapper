"""One call to see any pixel's optical season: raw observations, the smoothed curve, and the dates.

Why
---
Every judgement about the optical map comes down to looking at a pixel and asking whether the
detected emergence, peak and harvest sit where the curve actually turns. Rebuilding the cube, the
5-day grid and the Whittaker fit by hand each time is slow and, worse, easy to do slightly
differently from the detector — at which point the picture no longer shows what the detector saw.

So this module does it once, exactly the way ``optical_phenology`` does it, and keeps the AOI's cube
in memory: the first call for an AOI reads the files, every later call for another pixel of the same
AOI is instant. That is what makes it usable for flicking through pixels.

Use it
------
>>> from sar_pipeline.analysis import pixel_curve as pc
>>> pc.show(146, 4043)                        # matplotlib figure
>>> pc.show(146, 4043, backend="plotly")      # hover for exact dates and values
>>> pc.cycles(146, 4043)                      # the numbers behind the lines, as a DataFrame

What is drawn
-------------
* **Raw clear observations** as points — only the dates that passed the clear-score test, so the
  gaps are the cloud.
* **The smoothed curve** the detector reads: the same 5-day grid and the same Whittaker fit. Where
  two clear observations are more than ``max_gap_days`` apart the fit is left empty, so a break in
  this line is a stretch of season nobody observed.
* **The detected dates** of the pixel's largest cycle. If the dashed lines disagree with the points,
  the detector is wrong for that pixel and that is the thing worth reporting.
* **LSWI** underneath when the AOI was exported with SWIR, because that is the only band
  combination here that says anything about water.
* **VH, VV and VH - VV** from :mod:`sar_curve` when ``with_sar`` is on: both Sentinel-1 tracks
  merged, binned onto the same 5-day windows and smoothed the same way, so the radar and the optical
  are read off one date axis instead of two.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import optical_phenology as op

#: AOI cubes already read, keyed by the settings they were read with. Cleared with :func:`forget`.
_CUBES: dict[tuple, tuple] = {}

MARKS = (("start_date", "start", "#8a8985"), ("emergence_date", "emergence", "#2a78d6"),
         ("peak_date", "peak", "#52514e"), ("harvest_date", "harvest", "#eb6834"))


def pick_folder(aoi_id: int, folder: str | None = None, cache_root: str | None = None) -> tuple[str, str]:
    """Which exported band set to use. Prefers the SWIR one when this AOI has it cached locally."""
    if folder is not None:
        return folder, cache_root or f"data/{folder}"
    from .. import config as config_mod

    for name in ("s2_windows5d", "s2_reference_swir", "s2_reference"):
        root = cache_root or f"data/{name}"
        probe = Path(root)
        if not probe.is_absolute():
            probe = config_mod.repo_root() / probe
        if (probe / f"aoi{aoi_id}").exists():
            return name, root
    return "s2_reference", cache_root or "data/s2_reference"


def load(aoi_id: int, folder: str | None = None, cache_root: str | None = None,
         clear_min: float = op.CLEAR_MIN, start="2025-03-01", end="2026-01-31",
         step_days: int = 5):
    """The AOI's cube plus the grid the detector uses, read once and kept in memory."""
    folder, cache_root = pick_folder(aoi_id, folder, cache_root)
    source = op.SOURCES.get(folder, "dates")
    key = (aoi_id, folder, cache_root, clear_min, start, end, step_days)
    if key in _CUBES:
        return _CUBES[key]
    if source == "windows":
        dates, ndvi, ndwi, lswi, ok, loc = op.read_windows(aoi_id, cache_root, folder)
    else:
        dates, ndvi, ndwi, lswi, ok, loc = op.read_cube(aoi_id, clear_min, cache_root, folder)
    keep = (dates >= start) & (dates <= end)
    dates, ndvi, ndwi, lswi, ok = dates[keep], ndvi[keep], ndwi[keep], lswi[keep], ok[keep]
    days = (dates - dates[0]).days.to_numpy().astype("float64")
    if source == "windows":
        grid_dates, grid_days = dates, days
    else:
        grid_dates = pd.date_range(dates[0], dates[-1], freq=f"{step_days}D")
        grid_days = (grid_dates - dates[0]).days.to_numpy().astype("float64")
    flat = {name: a.reshape(len(dates), -1) for name, a in
            (("ndvi", ndvi), ("ndwi", ndwi), ("lswi", lswi), ("ok", ok))}
    _CUBES[key] = (dates, days, grid_dates, grid_days, flat, loc, folder, source)
    return _CUBES[key]


def forget():
    """Drop the cached cubes, e.g. after re-exporting the imagery."""
    _CUBES.clear()


def series(aoi_id: int, pid: int, max_gap_days: float = 35, lmbd: float = op.LMBD,
           with_sar: bool = False, sar_lmbd: float | None = None, **load_kw) -> dict:
    """Raw and smoothed NDVI, NDWI and LSWI for one pixel, plus its cycles.

    ``raw`` holds only the clear observations; ``smooth`` is on the 5-day grid with empty stretches
    where the gap between clear observations was too wide to interpolate across.
    """
    dates, days, grid_dates, grid_days, flat, loc, folder, source = load(aoi_id, **load_kw)
    width = int(loc["grid"]["width"])
    if not 0 <= pid < width * int(loc["grid"]["height"]):
        raise ValueError(f"pid {pid} is outside AOI {aoi_id}'s grid")
    good = flat["ok"][:, pid]
    raw = pd.DataFrame({"date": dates[good]})
    smooth = pd.DataFrame({"date": grid_dates})
    for name in ("ndvi", "ndwi", "lswi"):
        column = flat[name][:, pid]
        raw[name] = column[good]
        seen = good & np.isfinite(column)
        if not seen.any():
            smooth[name] = np.full(len(grid_dates), np.nan)
        elif source == "windows":
            smooth[name] = op.blank_long_gaps(
                op.whittaker(np.where(seen, column, np.nan), lmbd), grid_days, seen, max_gap_days)
        else:
            smooth[name] = op.whittaker(op.regrid(days, column, good, grid_days, max_gap_days), lmbd)
    found = op.pixel_cycles(grid_dates, smooth["ndvi"].to_numpy(), smooth["ndwi"].to_numpy(),
                            None if smooth["lswi"].isna().all() else smooth["lswi"].to_numpy())
    if with_sar:
        from . import sar_curve as sarc

        radar = sarc.pixel(aoi_id, int(pid), window_starts=grid_dates, lmbd=sar_lmbd)
        for name in ("VH", "VV", "VHmVV"):
            raw[name] = np.nan
            smooth[name] = radar[name].to_numpy()
        # The dots are the individual acquisitions, not the 5-day means the curve is fitted to:
        # a window mean is already an average and plotting it as an observation hides the scatter.
        acquisitions = sarc.raw_pixel(loc, int(pid))
        raw = pd.concat([raw, acquisitions], ignore_index=True).sort_values("date")
        sar_lmbd = radar.attrs["lmbd"]
    row, col = divmod(int(pid), width)
    return {"aoi": loc["aoi"], "pid": int(pid), "row": row, "col": col, "folder": folder,
            "sar_lmbd": sar_lmbd,
            "source": source, "lmbd": lmbd,
            "raw": raw, "smooth": smooth, "cycles": found,
            "main": max(found, key=lambda c: c["amplitude"]) if found else None,
            "n_clear": int(good.sum())}


def cycles(aoi_id: int, pid: int, **kw) -> pd.DataFrame:
    """Every cycle measured for a pixel, one row each, newest measurement of the same data."""
    found = series(aoi_id, pid, **kw)["cycles"]
    return pd.DataFrame(found) if found else pd.DataFrame()


def show(aoi_id: int, pid: int, backend: str = "matplotlib", panels=None, out_path=None,
         width=None, height=None, **kw):
    """Draw one pixel's season. ``panels`` defaults to NDVI, plus LSWI when the AOI has SWIR."""
    data = series(aoi_id, pid, **kw)
    if panels is None:
        panels = ("ndvi", "lswi") if data["smooth"]["lswi"].notna().any() else ("ndvi",)
        if kw.get("with_sar") or "VH" in data["smooth"]:
            panels = panels + ("VH", "VV", "VHmVV")
    panels = tuple(panels)
    if backend == "plotly":
        return _plotly(data, panels, out_path, width or 1500, height or 330 * len(panels) + 140)
    return _matplotlib(data, panels, out_path, width or 16, height or 3.4 * len(panels) + 1.2)


LABELS = {"ndvi": "NDVI", "ndwi": "NDWI  (green, NIR)", "lswi": "LSWI  (NIR, SWIR) — water",
          "VH": "VH  (dB)", "VV": "VV  (dB)", "VHmVV": "VH − VV  (dB)"}


def _title(data):
    m = data["main"]
    head = (f"{data['aoi']}  pid {data['pid']}  (row {data['row']}, col {data['col']})   |   "
            f"{data['n_clear']} clear {'windows' if data.get('source') == 'windows' else 'dates'}"
            f"   |   {data['folder']}   |   Whittaker λ={data.get('lmbd', op.LMBD)}"
            + (f" (optical), λ={data['sar_lmbd']:g} (radar)" if data.get("sar_lmbd") else ""))
    if not m:
        return head + "\nno cycle measured"
    return (head + f"\ncycle: {pd.Timestamp(m['start_date']):%d %b} start · "
            f"amplitude {m['amplitude']:.2f} · "
            f"{m['emergence_to_harvest_days']:.0f} d emergence to harvest · "
            f"{'complete' if m['complete'] else 'INCOMPLETE — runs into the edge of the series'}")


def _matplotlib(data, panels, out_path, width, height):
    import matplotlib.pyplot as plt

    from .curves import INK, INK_SECONDARY, SERIES_COLORS, SURFACE

    raw, smooth, main = data["raw"], data["smooth"], data["main"]
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, height), sharex=True, facecolor=SURFACE,
                             squeeze=False)
    axes = axes[:, 0]
    for ax, name in zip(axes, panels):
        colour = SERIES_COLORS[2] if name == "ndvi" else SERIES_COLORS[0]
        ax.set_facecolor(SURFACE)
        ax.plot(raw["date"], raw[name], "o", ms=4, color=INK_SECONDARY, alpha=.55,
                label=f"observed (n={raw[name].notna().sum()})")
        ax.plot(smooth["date"], smooth[name], lw=2.4, color=colour,
                label="smoothed (Whittaker, d=2)")
        if name == "lswi":
            ax.axhline(0, lw=1, color=INK_SECONDARY, alpha=.4)
        if main:
            for key, label, mark in MARKS:
                if main.get(key) is not None and pd.notna(main.get(key)):
                    ax.axvline(pd.Timestamp(main[key]), ls="--", lw=1.8, color=mark,
                               label=f"{label} {pd.Timestamp(main[key]):%d %b}" if name == panels[0] else None)
        ax.set_ylabel(LABELS.get(name, name), fontsize=9)
        ax.grid(True, alpha=.2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=3)
    fig.suptitle(_title(data), x=0.01, ha="left", fontsize=11, color=INK)
    fig.subplots_adjust(left=.05, right=.99, top=1 - 1.0 / height, bottom=.06, hspace=.16)
    if out_path:
        fig.savefig(out_path, dpi=140, facecolor=SURFACE)
    return fig


def _plotly(data, panels, out_path, width, height):
    """Plotly version.

    Three things that had to be handled by hand: ``add_vline`` writes its annotation onto *every*
    subplot it spans, so four dates over two panels produced eight labels stacked on each other;
    adjacent dates can be days apart, so the labels are staggered over two rows; and centred subplot
    titles sat exactly where those labels land, so the panel names live on the y-axes instead.
    """
    from plotly.subplots import make_subplots

    from .curves import SERIES_COLORS

    raw, smooth, main = data["raw"], data["smooth"], data["main"]
    fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=.05)
    for i, name in enumerate(panels, start=1):
        colour = SERIES_COLORS[2] if name == "ndvi" else SERIES_COLORS[0]
        fig.add_scatter(x=raw["date"], y=raw[name], mode="markers", name=f"{name} observed",
                        marker=dict(size=6, color="#52514e", opacity=.55),
                        hovertemplate="%{x|%d %b %Y}<br>raw %{y:.3f}<extra></extra>", row=i, col=1)
        fig.add_scatter(x=smooth["date"], y=smooth[name], mode="lines", name=f"{name} smoothed",
                        line=dict(color=colour, width=2.6),
                        hovertemplate="%{x|%d %b %Y}<br>smoothed %{y:.3f}<extra></extra>", row=i, col=1)
        fig.update_yaxes(title_text=LABELS.get(name, name), title_font=dict(size=11), row=i, col=1)
        if name == "lswi":
            fig.add_hline(y=0, line=dict(color="#8a8985", width=1), row=i, col=1)

    if main:
        marks = [(label, pd.Timestamp(main[key]), colour) for key, label, colour in MARKS
                 if main.get(key) is not None and pd.notna(main.get(key))]
        for level, (label, when, colour) in enumerate(marks):
            fig.add_vline(x=when, line=dict(color=colour, width=2, dash="dash"), row="all", col=1)
            fig.add_annotation(x=when, xref="x", y=1.0 + 0.045 * (level % 2), yref="paper",
                               text=f"{label} {when:%d %b}", showarrow=False, xshift=4,
                               xanchor="left", yanchor="bottom", font=dict(size=11, color=colour))

    fig.update_layout(title=dict(text=_title(data).replace("\n", "<br>"), y=0.985, yanchor="top",
                                 font=dict(size=13)),
                      width=width, height=height, template="plotly_white", hovermode="x unified",
                      margin=dict(t=150, l=70, r=30, b=70),
                      legend=dict(orientation="h", y=-0.12))
    if out_path:
        fig.write_html(str(out_path))
    return fig
