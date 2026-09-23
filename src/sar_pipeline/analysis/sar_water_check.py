"""Does the radar see water where the optical curve says the field was transplanted?

Why
---
The optical water test (LSWI) missed most of the certain-rice plots in one region, and even after
it was read relative to the field's own dry level it still leaves gaps where the transplanting week
fell under cloud. Radar does not care about cloud, turbidity or depth: a flooded field is a mirror
to the radar, the signal bounces away from the sensor, and **VV** (the more water-sensitive
polarisation) drops by 5-15 dB; **VH** drops less but then climbs steeply as the canopy fills in.

So, for every field plot: take the plot's radar series (median over its pixels of the 5x5-mean dB
stacks, per track), line it up on the plot's optical transplant date (the NDVI trough), and read

* the **dry level** before transplanting: the median in ``[-60, -20]`` days;
* the **flood level**: the minimum in ``[-10, +15]`` days around the trough;
* the **dip** = dry level - flood level, in dB, per polarisation.

A dip of several dB in VV at the trough, in a plot where the optical LSWI saw nothing, is the
evidence that the water was there and the optical test was the thing that failed. Tracks are kept
separate (no offset correction) because everything is relative to the same plot and track.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import pixel_report as pr
from . import sar_curve
from .plot_curves import plot_pixels

DRY_WINDOW = (-60, -20)
FLOOD_WINDOW = (-10, 15)
#: A VV drop at least this large below the field's own dry level counts as flooding.
DIP_MIN_DB = 3.0


def plot_series(aoi_id: int, plots, season_key: str = "monsoon2026", window: int = 5) -> pd.DataFrame:
    """Median VH and VV (dB) per plot, date and track, from the season's stacked run."""
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    pixels = plot_pixels(plots, loc["grid"])
    frames = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        vh = cubes["VH"].reshape(len(dates), -1)
        vv = cubes["VV"].reshape(len(dates), -1)
        for pid, pix in pixels.items():
            frames.append(pd.DataFrame({
                "plot_id": pid, "track": track, "date": dates, "n_pixels": len(pix),
                "VH": np.nanmedian(vh[:, pix], axis=1), "VV": np.nanmedian(vv[:, pix], axis=1)}))
    out = pd.concat(frames, ignore_index=True)
    out["VHmVV"] = out["VH"] - out["VV"]
    out.attrs["aoi"] = loc["aoi"]
    return out


def align(series: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Add ``days`` = date minus the plot's optical trough (transplant) date."""
    ref = events[["plot_id", "trough_date"]].copy()
    ref["trough_date"] = pd.to_datetime(ref["trough_date"])
    out = series.merge(ref, on="plot_id", how="inner")
    out["days"] = (pd.to_datetime(out["date"]) - out["trough_date"]).dt.days
    return out


def dips(aligned: pd.DataFrame, dry=DRY_WINDOW, flood=FLOOD_WINDOW) -> pd.DataFrame:
    """Per plot and track: dry level, flood level and dip for VV and VH; NaN where a window is empty."""
    rows = []
    for (pid, track), g in aligned.groupby(["plot_id", "track"]):
        before = g[(g["days"] >= dry[0]) & (g["days"] <= dry[1])]
        around = g[(g["days"] >= flood[0]) & (g["days"] <= flood[1])]
        row = {"plot_id": pid, "track": track, "n_dry_dates": len(before), "n_flood_dates": len(around)}
        for pol in ("VV", "VH"):
            dry_level = before[pol].median() if len(before) else np.nan
            flood_level = around[pol].min() if len(around) else np.nan
            row[f"{pol}_dry"] = dry_level
            row[f"{pol}_flood"] = flood_level
            row[f"{pol}_dip"] = dry_level - flood_level
            row[f"{pol}_flood_day"] = (around.loc[around[pol].idxmin(), "days"] if len(around) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def summary(dip_table: pd.DataFrame, dip_min: float = DIP_MIN_DB) -> pd.DataFrame:
    """Per track: median dips and the share of plots whose VV dropped by at least ``dip_min``."""
    ok = dip_table.dropna(subset=["VV_dip"])
    return (ok.groupby("track").agg(
        plots=("plot_id", "nunique"),
        VV_dry=("VV_dry", "median"), VV_flood=("VV_flood", "median"), VV_dip=("VV_dip", "median"),
        VH_dip=("VH_dip", "median"),
        flooded_pct=("VV_dip", lambda v: round(100 * float((v >= dip_min).mean()))),
        flood_day=("VV_flood_day", "median")).round(1).reset_index())


def figure(aligned: pd.DataFrame, title: str = "", out_path=None, span=(-70, 70)):
    """Median VV and VH against days from the optical transplant date, one line per track."""
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, SERIES_COLORS, SURFACE

    a = aligned[(aligned["days"] >= span[0]) & (aligned["days"] <= span[1])].copy()
    a["bin"] = (a["days"] // 6) * 6
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), facecolor=SURFACE)
    for ax, pol in zip(axes, ("VV", "VH")):
        ax.set_facecolor(SURFACE)
        for (track, g), colour in zip(a.groupby("track"), SERIES_COLORS):
            q = g.groupby("bin")[pol].quantile([.25, .5, .75]).unstack()
            ax.fill_between(q.index, q[.25], q[.75], color=colour, alpha=.18, lw=0)
            ax.plot(q.index, q[.5], "o-", ms=4, color=colour, label=f"{track} (median, n={g['plot_id'].nunique()} plots)")
        ax.axvline(0, ls="--", color=INK_MUTED, lw=1.2)
        ax.text(1, 0.98, "optical transplant\n(NDVI trough)", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK_MUTED, va="top")
        ax.set_xlabel("days from transplant date")
        ax.set_ylabel(f"{pol} (dB), 5x5 mean, median over plot")
        ax.grid(True, alpha=.2)
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    return fig


def run(aoi_id: int, plots, events: pd.DataFrame, out_root="processed/_batch/s2_2026",
        season_key: str = "monsoon2026") -> dict:
    """Series, alignment, dips, summary and figure for one AOI; writes the CSVs and PNG."""
    series = plot_series(aoi_id, plots, season_key)
    aligned = align(series, events[events["aoi"] == series.attrs["aoi"]])
    table = dips(aligned)
    folder = Path(out_root) / series.attrs["aoi"]
    table.to_csv(folder / f"{series.attrs['aoi']}_sar_water_dips.csv", index=False)
    fig = figure(aligned, f"{series.attrs['aoi']}: radar at the optical transplant date, field plots",
                 folder / "figures" / f"{series.attrs['aoi']}_sar_water_check.png")
    return {"aoi": series.attrs["aoi"], "summary": summary(table), "dips": table, "aligned": aligned, "figure": fig}
