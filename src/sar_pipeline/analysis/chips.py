"""Look at what the detector decided, in the imagery a person can read.

Why
---
Everything downstream now rests on one claim: that the cycle measured from a pixel's NDVI curve is
really that field's crop, with emergence and harvest on the dates the code says. That claim has to
be checked by eye before it carries any more weight, and the way to check it is the way an analyst
already does it — a strip of **5-3-2** images (red edge, green, blue) across the season. In that
combination a paddy field goes: bare or flooded at transplanting, light green for the first six
weeks or so, dark green through heading, then back to light green, then yellow, then cut. If the
detected emergence and harvest dates do not sit at the ends of that colour sequence, the detector is
wrong and we need to know now.

Two choices that matter
-----------------------
* **One stretch for the whole series, computed on clear pixels only.** Each chip is scaled with
  percentiles taken once over every date, not per image: per-image scaling would renormalise every
  chip to mid-grey and destroy exactly the colour progression the check depends on. Clouds are
  excluded from that calculation even though they stay visible in the chips, because they are by far
  the brightest thing in the series -- including them put the upper limit at 0.97 reflectance instead
  of 0.30 and rendered every field almost black.
* **Context around the pixel.** A single 10 m pixel tells you nothing by eye, so a square of
  neighbourhood is drawn with the pixel marked, and the field it sits in is visible.

Cloudy dates are dropped on the **whole chip**, not only the centre pixel. Judging by the centre
alone let through dates where the pixel scored clear but the surrounding field was under haze, and a
hazy chip is worse than no chip: it looks like a pale bare field and invites the wrong reading.

Cloud Score+ alone does not catch thin haze here -- dates it scores above 0.8 still came out milky.
Blue reflectance does, because blue is the most scattered wavelength, but **only within a short
period**: across the year a dry bare field is legitimately bright in blue, so an absolute blue
threshold would reject the real bare-soil dates that the check depends on, and the blue-to-red-edge
ratio was tried and does not separate either (a hazy May date scored 0.62 against 0.57 for a clean
July one). So the cycle is cut into equal periods and the **least blue** date is taken from each.
Inside one period the surface barely changes, which makes the comparison a haze comparison.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import pixel_report as pr

#: File bands making up the usual combinations, as Sentinel-2 band names.
COMBINATIONS = {"5-3-2": ("B5", "B3", "B2"), "8-3-2": ("B8", "B3", "B2"), "4-3-2": ("B4", "B3", "B2")}


def read_rgb(aoi_id: int, combination: str = "5-3-2", clear_min: float = 60,
             cache_root="data/s2_reference", folder: str = "s2_reference"):
    """The three display bands and the clear score for every date, as (dates, band, row, col)."""
    import rasterio

    from ..optical_export import band_index

    names = COMBINATIONS[combination]
    loc = pr.locate(aoi_id, 0)
    dates, stack, clear = [], [], []
    for path in sorted(pr.sync_s2(loc, cache_root, folder).glob("*.tif")):
        dates.append(pd.to_datetime(path.stem.rsplit("_S2_", 1)[1]))
        with rasterio.open(path) as ds:
            stack.append(np.stack([ds.read(band_index(ds, n)).astype("float32") for n in names]))
            clear.append(ds.read(band_index(ds, "clear")).astype("float32"))
    order = np.argsort(dates)
    return pd.DatetimeIndex(np.array(dates)[order]), np.stack(stack)[order], np.stack(clear)[order], loc


def stretch_limits(stack, clear=None, clear_min: float = 60, low: float = 2.0, high: float = 98.0):
    """Per-band display limits taken once over every date, so colours stay comparable.

    ``clear`` (dates, row, col) excludes cloud from the percentiles. Cloud is the brightest thing in
    the series by a wide margin, so leaving it in drives the upper limit far above any land surface
    and renders the fields black.
    """
    flat = stack.reshape(stack.shape[0], stack.shape[1], -1)
    mask = None if clear is None else (clear.reshape(clear.shape[0], -1) >= clear_min)
    out = []
    for b in range(stack.shape[1]):
        v = flat[:, b]
        v = v[mask] if mask is not None and mask.any() else v.ravel()
        out.append(np.nanpercentile(v, [low, high]))
    return np.stack(out)


def to_display(chip, limits):
    """Scale one (band, row, col) chip to 0..1 with the series-wide limits."""
    lo = limits[:, 0][:, None, None]
    hi = limits[:, 1][:, None, None]
    return np.clip((chip - lo) / np.maximum(hi - lo, 1e-6), 0, 1).transpose(1, 2, 0)


def window(row, col, half, shape):
    """Chip bounds around a pixel, slid inside the grid rather than clipped.

    Clipping made every chip for a pixel near the AOI edge smaller *and* put the marker in a corner
    with no context on one side, which reads as if the detector had picked the wrong place. Sliding
    keeps the chip the same size everywhere; only the marker moves off centre.
    """
    height = min(2 * half + 1, shape[0])
    width = min(2 * half + 1, shape[1])
    r0 = min(max(0, row - half), shape[0] - height)
    c0 = min(max(0, col - half), shape[1] - width)
    return r0, r0 + height, c0, c0 + width


def chip_dates(dates, clear_chip, cycle, clear_min: float = 60, pad_before: int = 30,
               pad_after: int = 45, limit: int = 20, min_clear_frac: float = 0.85, haze=None):
    """Clear dates spanning one cycle, thinned evenly to at most ``limit`` so the sheet stays readable.

    The window runs from ``pad_before`` days before the cycle starts to ``pad_after`` days after
    harvest, so the bare field before the crop and the cut field after it are both on the sheet.

    ``clear_chip`` is the clear score over the drawn square, shaped (dates, rows, cols) or
    (dates,) for a single pixel. A date is kept only when at least ``min_clear_frac`` of the square
    is clear, because a chip that is mostly haze reads as a pale bare field.

    ``haze`` is one number per date, lower being clearer (chip-median blue reflectance). When given,
    the window is cut into ``limit`` equal periods and the clearest date of each period is taken, so
    the strip stays evenly spread without putting a milky image on the sheet where a clean one
    exists a few days away. Without it the candidates are thinned evenly instead.
    """
    start = pd.Timestamp(cycle["start_date"]) - pd.Timedelta(days=pad_before)
    end_ref = cycle.get("harvest_date") or cycle["end_date"]
    end = pd.Timestamp(end_ref) + pd.Timedelta(days=pad_after)
    clear_chip = np.asarray(clear_chip, dtype="float32")
    frac = ((clear_chip >= clear_min).reshape(len(dates), -1)).mean(axis=1)
    keep = np.where((dates >= start) & (dates <= end) & (frac >= min_clear_frac))[0]
    if haze is None:
        if len(keep) <= limit:
            return keep
        return keep[np.linspace(0, len(keep) - 1, limit).round().astype(int)]
    if len(keep) < 2:
        return keep
    haze = np.asarray(haze, dtype="float64")
    day = (dates[keep] - dates[keep][0]).days.to_numpy().astype("float64")
    edges = np.linspace(0, day[-1] + 1e-6, limit + 1)
    picked = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = np.where((day >= lo) & (day < hi))[0]
        if len(inside):
            picked.append(keep[inside[int(np.argmin(haze[keep[inside]]))]])
    return np.array(sorted(set(picked)), dtype=int)


def contact_sheet(aoi_id: int, pid: int, cycle: dict, ndvi_series=None, combination: str = "5-3-2",
                  half: int = 15, out_path=None, clear_min: float = 60,
                  cache_root="data/s2_reference", folder: str = "s2_reference", cached=None):
    """One page for one pixel: its NDVI curve with the detected dates, and the 5-3-2 strip.

    ``cached`` is the tuple returned by :func:`read_rgb`, passed in when several pixels of the same
    AOI are drawn so the images are read once.
    """
    import matplotlib.pyplot as plt

    from .curves import INK, INK_SECONDARY, SERIES_COLORS, SURFACE

    dates, stack, clear, loc = cached if cached is not None else read_rgb(
        aoi_id, combination, clear_min, cache_root, folder)
    row, col = divmod(int(pid), int(loc["grid"]["width"]))
    limits = stretch_limits(stack, clear, clear_min)
    r0, r1, c0, c1 = window(row, col, half, stack.shape[2:])
    blue = np.nanmedian(stack[:, 2, r0:r1, c0:c1].reshape(len(dates), -1), axis=1)
    picks = chip_dates(dates, clear[:, r0:r1, c0:c1], cycle, clear_min, haze=blue)
    if not len(picks):
        raise ValueError(f"pixel {pid} has no clear date inside its cycle")

    cols = min(8, max(1, len(picks)))
    rows = int(np.ceil(len(picks) / cols))
    fig = plt.figure(figsize=(2.1 * cols, 2.5 + 2.35 * rows), facecolor=SURFACE)
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1.5] + [1] * rows, hspace=0.35, wspace=0.06)

    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(SURFACE)
    if ndvi_series is not None:
        ax.plot(ndvi_series["date"], ndvi_series["ndvi"], color=SERIES_COLORS[2], lw=2,
                marker="o", ms=3.5, label="NDVI, clear dates")
    marks = (("greenup_onset", SERIES_COLORS[0]), ("peak_date", INK_SECONDARY),
             ("harvest_date", SERIES_COLORS[1]))
    for key, colour in marks:
        if cycle.get(key) is not None and pd.notna(cycle.get(key)):
            ax.axvline(pd.Timestamp(cycle[key]), color=colour, lw=2, ls="--",
                       label=f"{key.replace('_date', '')} {pd.Timestamp(cycle[key]):%d %b}")
    for d in dates[picks]:
        ax.axvline(d, color=INK_SECONDARY, lw=0.6, alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=4)
    ax.grid(True, alpha=0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"{loc['aoi']}  pid {pid}   |   cycle: "
                 f"{pd.Timestamp(cycle['start_date']):%d %b} start, "
                 f"{cycle.get('greenup_to_harvest_days', float('nan')):.0f} d green-up to harvest, "
                 f"amplitude {cycle['amplitude']:.2f}\n"
                 f"Sentinel-2 {combination} below — does the detected cycle match the colours?",
                 loc="left", fontsize=11, color=INK)

    for i, k in enumerate(picks):
        a = fig.add_subplot(gs[1 + i // cols, i % cols])
        a.imshow(to_display(stack[k, :, r0:r1, c0:c1], limits), interpolation="nearest")
        a.plot(col - c0, row - r0, marker="s", ms=7, mfc="none", mec="#ff2d55", mew=1.6)
        a.set_title(f"{dates[k]:%d %b}", fontsize=8.5, color=INK_SECONDARY, pad=3)
        a.set_xticks([])
        a.set_yticks([])
    # No tight_layout: the chip axes carry images with a fixed aspect, which it cannot lay out,
    # and a suptitle over a left-aligned axes title collided, so the heading lives in the title.
    fig.subplots_adjust(left=0.05, right=0.985, top=0.90, bottom=0.02)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, facecolor=SURFACE)
        plt.close(fig)
        return out_path
    return fig


def stratified_sample(main: pd.DataFrame, per_group: int = 2, seed: int = 0) -> pd.DataFrame:
    """A few pixels from each corner of the measured distribution, not only the typical ones.

    Checking only average pixels would confirm the detector where it is least likely to be wrong.
    The groups deliberately include the short cycles and the weak-amplitude pixels, which are the
    ones a wrong detector would produce.
    """
    d = main.dropna(subset=["greenup_to_harvest_days", "amplitude", "start_date"]).copy()
    early, late = d["start_date"].quantile([0.05, 0.95])
    groups = {
        "typical": d[d["greenup_to_harvest_days"].between(*d["greenup_to_harvest_days"].quantile([0.4, 0.6]))
                     & (d["amplitude"] > d["amplitude"].median())],
        "earliest start": d[d["start_date"] <= early],
        "latest start": d[d["start_date"] >= late],
        "short cycle": d[d["greenup_to_harvest_days"] <= d["greenup_to_harvest_days"].quantile(0.05)],
        "long cycle": d[d["greenup_to_harvest_days"] >= d["greenup_to_harvest_days"].quantile(0.95)],
        "weak amplitude": d[d["amplitude"] <= d["amplitude"].quantile(0.05)],
        "two cycles": d[d.get("n_cycles", pd.Series(1, index=d.index)) >= 2],
    }
    out = []
    for name, frame in groups.items():
        if len(frame):
            picked = frame.sample(min(per_group, len(frame)), random_state=seed).copy()
            picked["group"] = name
            out.append(picked)
    return pd.concat(out, ignore_index=True) if out else d.head(0)
