"""One NDVI and LSWI curve per field plot, from the pixels inside it.

Why
---
Field plots are the only places where the crop is known from the ground. To learn what that crop
looks like from space, each plot is reduced to one curve: the **median** over the pixels whose
centre lies inside the plot (optionally after shrinking the plot inwards by ``inner_buffer_m``, so
pixels straddling a bund, a path or a ditch are left out). The median rather than the mean, because
one mixed edge pixel should not bend a small plot's curve.

The curves come from the gap-free 5-day series of ``ndvi_5day`` (so ``nd.build(<aoi>)`` must have
run for the AOI), which keeps every plot on the same windows and lets curves be stacked into a
plots x time picture. ``n_pixels`` is kept per plot: a plot of two pixels is weaker evidence than a
plot of forty, and the smallest plots here are under one pixel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd


def plot_pixels(plots, grid: dict, inner_buffer_m: float = 0.0) -> dict:
    """``{plot_id: flat pixel indices}`` for pixels whose centre lies inside each plot.

    ``plots`` is a GeoDataFrame with ``plot_id`` and polygons in any CRS; ``grid`` is the AOI grid
    (``crs``, ``x0``, ``y0``, ``res``, ``width``, ``height``). Plots with no pixel centre inside
    (smaller than a pixel, or shrunk away by the buffer) are left out.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    shape = (int(grid["height"]), int(grid["width"]))
    transform = from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"])
    local = plots.to_crs(grid["crs"])
    if inner_buffer_m:
        local = local.assign(geometry=local.buffer(-inner_buffer_m))
    local = local[~(local.geometry.isna() | local.geometry.is_empty)]
    # burn plot_id + 1 so 0 stays "no plot"; overlapping plots: the later one wins, as drawn
    burnt = rasterize(((g, int(i) + 1) for g, i in zip(local.geometry, local["plot_id"])),
                      out_shape=shape, transform=transform, fill=0, dtype="int32",
                      all_touched=False).ravel()
    idx = np.flatnonzero(burnt)
    ids = burnt[idx] - 1
    order = np.argsort(ids, kind="stable")
    ids, idx = ids[order], idx[order]
    splits = np.flatnonzero(np.diff(ids)) + 1
    return {int(group_ids[0]): pix for group_ids, pix in zip(np.split(ids, splits), np.split(idx, splits))}


def curves(aoi_id: int, plots, inner_buffer_m: float = 0.0) -> pd.DataFrame:
    """Median fitted NDVI and LSWI per plot and 5-day window, long format."""
    d = nd.load(aoi_id)
    pixels = plot_pixels(plots, d["loc"]["grid"], inner_buffer_m)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    gaps = d["gapdays5d"].reshape(d["gapdays5d"].shape[0], -1)
    frames = []
    for pid, pix in pixels.items():
        frames.append(pd.DataFrame({
            "plot_id": pid, "window": d["windows"], "n_pixels": len(pix),
            "ndvi": np.nanmedian(ndvi[:, pix], axis=1), "lswi": np.nanmedian(lswi[:, pix], axis=1),
            "gap_days": np.nanmedian(gaps[:, pix], axis=1)}))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.attrs["aoi"] = d["loc"]["aoi"]
    return out


def figure(c: pd.DataFrame, title: str = "", out_path=None, min_pixels: int = 3):
    """All plot curves with their median, and a plots x time NDVI heatmap sorted by last peak."""
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, SERIES_COLORS, SURFACE

    c = c[c["n_pixels"] >= min_pixels]
    wide = c.pivot(index="plot_id", columns="window", values="ndvi")
    lswi = c.pivot(index="plot_id", columns="window", values="lswi")
    windows = wide.columns
    # sort plots by the window of their highest NDVI in the last 120 days, so the heatmap reads as
    # "who greened up when" in the current season
    recent = windows >= windows[-1] - pd.Timedelta(days=120)
    order = wide.loc[:, recent].to_numpy().argmax(axis=1).argsort(kind="stable")
    wide, lswi = wide.iloc[order], lswi.iloc[order]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(16, 12), facecolor=SURFACE,
                                     gridspec_kw={"height_ratios": [1.2, 0.8, 1.4]})
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
    a1.plot(windows, wide.T.to_numpy(), lw=.5, color=SERIES_COLORS[2], alpha=.12)
    a1.plot(windows, wide.median().to_numpy(), lw=3, color=INK, label="median of plots")
    a1.fill_between(windows, wide.quantile(.25), wide.quantile(.75), color=SERIES_COLORS[2], alpha=.25,
                    lw=0, label="middle half of plots")
    a1.set_ylabel("NDVI (fit)")
    a2.plot(windows, lswi.median().to_numpy(), lw=2.5, color=SERIES_COLORS[0], label="median LSWI")
    a2.fill_between(windows, lswi.quantile(.25), lswi.quantile(.75), color=SERIES_COLORS[0], alpha=.25, lw=0)
    a2.plot(windows, wide.median().to_numpy(), lw=1.5, ls="--", color=SERIES_COLORS[2], label="median NDVI")
    a2.axhline(0, lw=.8, color=INK_MUTED)
    a2.set_ylabel("LSWI (fit)")
    for a in (a1, a2):
        a.grid(True, alpha=.2)
        a.legend(loc="upper left", fontsize=8.5, frameon=False)
        a.set_xlim(windows[0], windows[-1])
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    im = a3.imshow(wide.to_numpy(), aspect="auto", cmap="YlGn", vmin=0, vmax=0.9, interpolation="nearest",
                   extent=(0, len(windows), len(wide), 0))
    ticks = [i for i, w in enumerate(windows) if w.day <= 5]
    a3.set_xticks(ticks)
    a3.set_xticklabels([windows[i].strftime("%b %y") for i in ticks], fontsize=8)
    a3.set_ylabel(f"plots ({len(wide)}), sorted by recent peak")
    fig.colorbar(im, ax=a3, fraction=.02, pad=.01, label="NDVI")
    n_px = c.groupby("plot_id")["n_pixels"].first()
    fig.suptitle(title or f"{c.attrs.get('aoi', '')}: {len(wide)} plots with >= {min_pixels} pixels "
                 f"(median {n_px.median():.0f} px per plot)", x=0.01, ha="left", color=INK)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=100, facecolor=SURFACE)
    return fig


def run_aois(aoi_ids, plots, where: pd.DataFrame, out_root="processed/_batch/s2_2026",
             inner_buffer_m: float = 0.0, build: bool = True, log=print) -> pd.DataFrame:
    """Build the 5-day series (optional), then plot curves and a figure, for every AOI in turn.

    ``where`` is ``field_plots.in_aois`` output: each plot is taken in the AOI that holds most of it.
    Writes ``<aoi>/<aoi>_plot_curves.csv`` and ``<aoi>/figures/<aoi>_plot_curves.png``; returns all
    curves together with an ``aoi`` column. One failing AOI is logged and skipped.
    """
    import time

    import matplotlib.pyplot as plt

    frames = []
    for aoi_id in aoi_ids:
        key = f"aoi{aoi_id}"
        t0 = time.time()
        try:
            if build:
                nd.build(aoi_id, out_root=out_root, log=log)
            mine = plots[plots["plot_id"].isin(where.loc[where["aoi"] == key, "plot_id"])]
            c = curves(aoi_id, mine, inner_buffer_m)
            if c.empty:
                log(f"{key}: no plot has a pixel centre inside it")
                continue
            c.to_csv(Path(out_root) / key / f"{key}_plot_curves.csv", index=False)
            fig = figure(c, out_path=Path(out_root) / key / "figures" / f"{key}_plot_curves.png")
            plt.close(fig)
            frames.append(c.assign(aoi=key))
            log(f"{key}: {c['plot_id'].nunique()} plots with pixels ({time.time() - t0:.0f} s)")
        except Exception as error:  # noqa: BLE001 - one AOI must not stop the rest
            log(f"{key}: FAILED - {type(error).__name__}: {error}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def season_events(c: pd.DataFrame, start="2026-06-01", end="2026-09-24", min_pixels: int = 3) -> pd.DataFrame:
    """Per plot, the events of one season read off its curve: the trough, the water at it, the onset.

    Why: a rule for "rice standing now" needs to know, from plots where rice is known, *when* the
    field was at its lowest, whether it was wet then (LSWI above NDVI), when the canopy started to
    climb (the acceleration onset, as in ``optical_phenology``), and how green it is on the last
    window. Reading those per plot, then summarising per AOI, turns the field survey into numbers a
    rule can be checked against — and gives the sowing-to-onset lead the detector needs.
    """
    from . import optical_phenology as op

    out = []
    for pid, g in c[c["n_pixels"] >= min_pixels].groupby("plot_id"):
        g = g.sort_values("window")
        s = g[(g["window"] >= start) & (g["window"] < end)]
        if len(s) < 6:
            continue
        ndvi, lswi = s["ndvi"].to_numpy(), s["lswi"].to_numpy()
        days = (s["window"] - s["window"].iloc[0]).dt.days.to_numpy().astype(float)
        low = int(np.nanargmin(ndvi))
        peak = low + int(np.nanargmax(ndvi[low:]))
        onset = op.acceleration_onset(ndvi, days, low, peak)
        out.append({
            "plot_id": pid, "aoi": g["aoi"].iloc[0] if "aoi" in g else c.attrs.get("aoi"),
            "n_pixels": int(g["n_pixels"].iloc[0]),
            "trough_date": s["window"].iloc[low], "trough_ndvi": round(float(ndvi[low]), 3),
            "lswi_at_trough": round(float(lswi[low]), 3),
            "wet_at_trough": bool(lswi[low] > ndvi[low]),
            "onset_date": None if onset is None else s["window"].iloc[onset],
            "trough_to_onset_days": np.nan if onset is None else float(days[onset] - days[low]),
            "last_ndvi": round(float(ndvi[-1]), 3), "peak_ndvi_after_trough": round(float(ndvi[peak]), 3),
        })
    return pd.DataFrame(out)


def aoi_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Median dates and values per AOI, with the share of plots wet at the trough."""
    def med_date(s):
        s = pd.to_datetime(s.dropna())
        return None if s.empty else s.median().strftime("%d %b")

    return (events.groupby("aoi").agg(
        plots=("plot_id", "size"), trough=("trough_date", med_date), trough_ndvi=("trough_ndvi", "median"),
        lswi_at_trough=("lswi_at_trough", "median"), wet_pct=("wet_at_trough", lambda v: round(100 * v.mean())),
        onset=("onset_date", med_date), trough_to_onset_days=("trough_to_onset_days", "median"),
        last_ndvi=("last_ndvi", "median"), last_ndvi_p25=("last_ndvi", lambda v: v.quantile(.25)),
        peak_after=("peak_ndvi_after_trough", "median")).round(2).reset_index())


def interior_pixels(pixels: dict, width: int) -> dict:
    """``{plot_id: boolean array}``: True where a plot pixel's four neighbours are all in the same plot.

    Why: a 10 m pixel on a plot's edge straddles the bund, a path or the neighbour's field, and the
    5x5 radar mean straddles even more. When a plot's edge pixels disagree with its interior the
    disagreement is the mixing, not the crop; when a whole plot disagrees, the label or the polygon
    deserves a look.
    """
    out = {}
    for pid, pix in pixels.items():
        inside = set(int(p) for p in pix)
        out[pid] = np.array([all(q in inside for q in (p - 1, p + 1, p - width, p + width)) for p in pix])
    return out
