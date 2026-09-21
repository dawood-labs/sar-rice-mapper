"""Label-free seasonal statistics over a whole AOI: the flooding/canopy swing, and track agreement.

What this is for
----------------
The supervised path in :mod:`sar_pipeline.analysis` starts from labelled ground-truth polygons.
This project has none, so the first question is not "how accurate is the model" but **"which pixels
behave like a flooded-then-canopied crop at all?"** That question can be answered from the stack
alone, with no labels, and it is what this module computes.

Two independent, label-free signals
-----------------------------------
1. **Seasonal swing.** Paddy rice is flooded before planting, so its VH backscatter collapses (water
   reflects the beam away), then rises steeply as the canopy closes, then drops at harvest. The
   season's **minimum**, **maximum**, **variance** and **amplitude** (max − min) capture that.
   Permanent water stays low all season; forest and built-up stay high and flat. Both therefore have
   a small amplitude, and the crop does not.

2. **Between-track agreement.** Two Sentinel-1 tracks observe the same ground from different orbits,
   at different incidence angles, on different days. Where a pixel carries a real seasonal signal,
   both tracks see the same shape and their series correlate strongly. Where the variation is
   speckle, they correlate around zero. This is a *free* confirmation that a swing is real, and it
   needs no labels and no second sensor.

Neither signal identifies rice on its own. They isolate *fields with a flooding cycle*; another
flooded crop looks similar. Naming the class is a human decision.

Two rules that are not optional
-------------------------------
* **Average in linear power, report in dB.** Speckle is multiplicative; averaging dB biases the mean
  low. Every spatial and temporal average here happens in linear power.
* **Restrict the window to one cropping cycle.** Where fields are double-cropped, a window spanning
  two cycles puts the *deeper* of the two flooding minima into ``sigma_min`` — which may belong to
  the other crop. Pass the window that matches the crop you are after; see
  :func:`window_indices`.
"""
from __future__ import annotations

import datetime as dt
import re

import numpy as np

#: Band descriptions written by the stack step, e.g. ``VH_20250502``.
_BAND_DATE = re.compile(r"_(\d{8})$")


def to_linear(db):
    """dB -> linear power."""
    return np.power(10.0, np.asarray(db, dtype="float32") / 10.0)


def to_db(linear):
    """Linear power -> dB, with non-positive values treated as missing rather than -inf."""
    linear = np.asarray(linear, dtype="float32")
    out = np.full(linear.shape, np.nan, dtype="float32")
    good = linear > 0
    out[good] = 10.0 * np.log10(linear[good])
    return out


def read_stack(vrt_path):
    """Read a ``stack_<POL>.vrt`` into ``(dates, cube_db)`` with nodata as NaN.

    ``cube_db`` is ``(n_dates, height, width)`` in dB. Dates come from the band descriptions the
    stack step wrote, so the time axis cannot drift out of step with the pixels.
    """
    import rasterio

    with rasterio.open(vrt_path) as ds:
        cube = ds.read().astype("float32")
        nodata = ds.nodata
        descriptions = list(ds.descriptions)
    if nodata is not None:
        cube[cube == nodata] = np.nan

    dates = []
    for i, name in enumerate(descriptions):
        match = _BAND_DATE.search(name or "")
        if match is None:
            raise ValueError(f"band {i + 1} description {name!r} carries no YYYYMMDD date")
        dates.append(dt.datetime.strptime(match.group(1), "%Y%m%d").date())
    return dates, cube


def window_indices(dates, start=None, end=None) -> np.ndarray:
    """Indices of ``dates`` within ``[start, end]`` inclusive; either bound may be None.

    Use this to cut the cube down to **one** cropping cycle before computing features. Bounds are
    :class:`datetime.date` or ``"YYYY-MM-DD"`` strings.
    """
    def parse(value):
        if value is None or isinstance(value, dt.date):
            return value
        return dt.datetime.strptime(value, "%Y-%m-%d").date()

    lo, hi = parse(start), parse(end)
    keep = [i for i, d in enumerate(dates)
            if (lo is None or d >= lo) and (hi is None or d <= hi)]
    if not keep:
        raise ValueError(f"no dates in window {start}..{end}")
    return np.array(keep, dtype=int)


def spatial_mean(cube_db, size: int = 5):
    """NaN-aware ``size x size`` mean of each date, computed in **linear power**, returned in dB.

    This is the step that makes per-pixel series usable: a single pixel's residual noise is around
    2 dB, and a 5x5 window (0.62 acre at 10 m) cuts it to the point where the flooding date is
    reproducible. Doing it in dB instead would bias every window low.

    Windows are masked where fewer than half the cells are valid, so field edges and the AOI border
    do not silently borrow values from outside.
    """
    from scipy.ndimage import uniform_filter

    if size <= 1:
        return np.asarray(cube_db, dtype="float32")

    linear = to_linear(cube_db)
    valid = np.isfinite(linear)
    linear = np.where(valid, linear, 0.0)

    out = np.empty_like(linear)
    for i in range(linear.shape[0]):
        total = uniform_filter(linear[i], size=size, mode="constant", cval=0.0)
        share = uniform_filter(valid[i].astype("float32"), size=size, mode="constant", cval=0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / share
        out[i] = np.where(share >= 0.5, mean, np.nan)
    return to_db(out)


def seasonal_features(cube_db) -> dict:
    """Per-pixel seasonal statistics of a dB cube, plus the date index of the minimum.

    ``amplitude`` (max − min) is the headline number: it is what separates a cropping cycle from a
    permanently wet or permanently dry surface, and unlike ``var`` it is in dB and reads directly.
    """
    import warnings

    cube = np.asarray(cube_db, dtype="float32")
    # A pixel with no valid date at all is expected (outside the swath, masked); numpy warns on it,
    # and the result is handled explicitly below, so the warning is noise.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        count = np.isfinite(cube).sum(axis=0)
        mn = np.nanmin(cube, axis=0)
        mx = np.nanmax(cube, axis=0)
        var = np.nanvar(cube, axis=0)
    empty = count == 0
    for arr in (mn, mx, var):
        arr[empty] = np.nan
    argmin = np.full(cube.shape[1:], -1, dtype="int16")
    filled = np.where(np.isfinite(cube), cube, np.inf)
    argmin_ok = np.argmin(filled, axis=0).astype("int16")
    argmin[~empty] = argmin_ok[~empty]
    return {"min": mn, "max": mx, "amplitude": mx - mn, "var": var,
            "argmin": argmin, "n_dates": count}


def half_month_bins(dates) -> np.ndarray:
    """Label each date by calendar half-month (day 1-15 -> 0, 16-end -> 1), as ``YYYYMMH``.

    Half-month bins are the pipeline's own temporal unit: one track revisits every ~6-12 days, so a
    15-day bin almost always holds at least one acquisition, which is what makes two tracks
    comparable bin by bin even though they fly on different days.
    """
    return np.array([d.year * 100_0 + d.month * 10 + (0 if d.day <= 15 else 1) for d in dates])


def bin_in_linear(cube_db, dates):
    """Average a dB cube into half-month bins in linear power. Returns ``(bin_labels, cube_db)``."""
    labels = half_month_bins(dates)
    unique = np.unique(labels)
    linear = to_linear(cube_db)
    out = np.empty((unique.size,) + cube_db.shape[1:], dtype="float32")
    for i, label in enumerate(unique):
        sel = linear[labels == label]
        with np.errstate(invalid="ignore"):
            out[i] = np.nanmean(sel, axis=0)
    return unique, to_db(out)


def track_correlation(cube_a_db, dates_a, cube_b_db, dates_b):
    """Per-pixel Pearson correlation between two tracks, on their shared half-month bins.

    Both cubes are binned into half-months (in linear power) and correlated over the bins present in
    both. Bins rather than raw dates, because the two tracks never acquire on the same days.

    A high value means two independent viewing geometries agree that the pixel's seasonal shape is
    real. A value near zero means the variation did not survive a change of geometry, which is what
    speckle does. Pixels with fewer than three shared finite bins return NaN.
    """
    labels_a, binned_a = bin_in_linear(cube_a_db, dates_a)
    labels_b, binned_b = bin_in_linear(cube_b_db, dates_b)
    shared = np.intersect1d(labels_a, labels_b)
    if shared.size < 3:
        raise ValueError(f"tracks share only {shared.size} half-month bins; need at least 3")
    a = binned_a[np.isin(labels_a, shared)]
    b = binned_b[np.isin(labels_b, shared)]

    good = np.isfinite(a) & np.isfinite(b)
    n = good.sum(axis=0)
    a = np.where(good, a, np.nan)
    b = np.where(good, b, np.nan)
    with np.errstate(invalid="ignore"):
        am = np.nanmean(a, axis=0)
        bm = np.nanmean(b, axis=0)
        da, dbv = a - am, b - bm
        cov = np.nansum(da * dbv, axis=0)
        sa = np.sqrt(np.nansum(da * da, axis=0))
        sb = np.sqrt(np.nansum(dbv * dbv, axis=0))
        corr = cov / (sa * sb)
    corr[(n < 3) | (sa == 0) | (sb == 0)] = np.nan
    return corr.astype("float32"), shared.size


def aoi_mask(aoi_path, reference_vrt):
    """Boolean mask of the AOI polygon on the stack grid: True inside, False outside.

    Statistics over a chunked grid otherwise include the buffer and the corners of chunks that reach
    beyond the AOI, which drags in whatever land happens to sit there.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    with rasterio.open(reference_vrt) as ds:
        transform, shape, crs = ds.transform, (ds.height, ds.width), ds.crs
    shapes = gpd.read_file(aoi_path).to_crs(crs).geometry
    return rasterize(((g, 1) for g in shapes), out_shape=shape, transform=transform,
                     fill=0, dtype="uint8").astype(bool)


# --------------------------------------------------------------------------- plotting
#: Sequential blue ramp, light -> dark, for continuous magnitude. One hue, never a rainbow:
#: a rainbow ramp invents category boundaries where the data has none.
_SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
               "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#8a8985"
_SURFACE = "#fcfcfb"


def _sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL)
    cmap.set_bad(_SURFACE)          # outside the AOI reads as surface, not as a low value
    return cmap


def plot_feature_maps(panels, out_path, title=None, subtitle=None, cols=2):
    """Maps of per-pixel seasonal features, one panel per feature.

    ``panels`` is a sequence of ``(label, array, unit, vmin, vmax)``. Pass ``NaN`` outside the AOI;
    those pixels are drawn as the page surface so the AOI's shape stays legible and masked ground is
    never mistaken for a low value.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 4.6 * rows),
                             squeeze=False, facecolor=_SURFACE)
    cmap = _sequential_cmap()

    for ax, (label, array, unit, vmin, vmax) in zip(axes.ravel(), panels):
        ax.set_facecolor(_SURFACE)
        im = ax.imshow(np.asarray(array, dtype="float32"), cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(label, fontsize=10.5, color=_INK_SECONDARY, loc="left", pad=6)
        bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        bar.set_label(unit, fontsize=9, color=_INK_SECONDARY)
        bar.ax.tick_params(colors=_INK_SECONDARY, labelsize=8, length=3)
        bar.outline.set_visible(False)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    height_in = fig.get_size_inches()[1]
    band_in = 0.0
    if title:
        fig.text(0.012, 1 - 0.28 / height_in, title, fontsize=13, color=_INK, ha="left", va="top")
        band_in = 0.46
    if subtitle:
        fig.text(0.012, 1 - (band_in + 0.20) / height_in, subtitle, fontsize=9.5,
                 color=_INK_SECONDARY, ha="left", va="top")
        band_in += 0.36
    fig.tight_layout(rect=(0, 0, 1, 1 - (band_in + 0.14) / height_in if band_in else 1.0))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_distributions(panels, out_path, title=None, subtitle=None, bins=60):
    """Histograms of per-pixel features, one panel each, with the median marked.

    A histogram is here to answer one question the percentile table cannot: is the AOI **two
    populations or one continuum**? A cherry-picked handful of pixels can suggest a clean split that
    the full distribution does not have.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.4), squeeze=False, facecolor=_SURFACE)
    for ax, (label, values, unit) in zip(axes.ravel(), panels):
        values = np.asarray(values, dtype="float32")
        values = values[np.isfinite(values)]
        ax.set_facecolor(_SURFACE)
        ax.hist(values, bins=bins, color=_SEQUENTIAL[7], edgecolor="none")
        median = float(np.median(values))
        ax.axvline(median, color=_INK_SECONDARY, linewidth=1.4, linestyle="--")
        ax.annotate(f"median {median:.2f}", xy=(median, ax.get_ylim()[1] * 0.92),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=_INK_SECONDARY, va="top")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_INK_MUTED); ax.spines[spine].set_linewidth(0.8)
        ax.grid(True, axis="y", color=_INK_MUTED, alpha=0.18, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_title(label, fontsize=10.5, color=_INK_SECONDARY, loc="left", pad=6)
        ax.set_xlabel(unit, fontsize=9, color=_INK_SECONDARY)
        ax.set_ylabel("pixels", fontsize=9, color=_INK_SECONDARY)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=8, length=3)

    height_in = fig.get_size_inches()[1]
    band_in = 0.0
    if title:
        fig.text(0.012, 1 - 0.28 / height_in, title, fontsize=13, color=_INK, ha="left", va="top")
        band_in = 0.46
    if subtitle:
        fig.text(0.012, 1 - (band_in + 0.20) / height_in, subtitle, fontsize=9.5,
                 color=_INK_SECONDARY, ha="left", va="top")
        band_in += 0.36
    fig.tight_layout(rect=(0, 0, 1, 1 - (band_in + 0.14) / height_in if band_in else 1.0))
    fig.savefig(out_path, dpi=150, facecolor=_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
