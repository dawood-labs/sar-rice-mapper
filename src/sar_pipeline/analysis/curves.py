"""Small-multiple plots of backscatter time series, for looking at curves with your own eyes.

Why a plotting module of its own
--------------------------------
Most of the decisions in this project are made by *looking at a curve*: is this field rice, does
smoothing help, do two tracks agree, did the rain rule leave a spike behind. Those plots get
regenerated for a different AOI, track or season every time, so the styling belongs in one tested
function rather than being re-typed into a notebook cell each time (and drifting each time).

`maps.py` draws rasters; this module draws time series.

Series are passed with **their own dates**, because the question that comes up most often - do two
Sentinel-1 tracks tell the same story? - compares series that were acquired on different days and
have different lengths. A single shared x array cannot express that.
"""
from __future__ import annotations

import numpy as np

#: Categorical slots 1-3 of the project palette. These three validate on the all-pairs
#: colour-vision gate, which is the gate that applies to small multiples; a fourth series would
#: put yellow next to orange and fail it, so panels are capped at three series.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
MAX_SERIES = len(SERIES_COLORS)

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
SURFACE = "#fcfcfb"


def plot_panels(panels, out_path, title=None, subtitle=None, ylabel="VH (dB)",
                mark_min=True, cols=3, spans=None, sharey=False):
    """Draw one panel per location, each with up to three dated series.

    Parameters
    ----------
    panels : sequence
        ``(panel_label, series)`` pairs, where ``series`` is a sequence of
        ``(dates, values, name)`` or ``(dates, values, name, recessive)``. ``recessive=True`` draws
        that series thin and semi-transparent - use it for a noisy reference trace that should not
        compete with the series the reader is meant to judge.
    mark_min : bool
        Mark each series' minimum with a dot. On rice curves the minimum *is* the feature under
        discussion (the flooded field), so it is marked by default.
    spans : sequence, optional
        ``(start_date, end_date, label)`` periods to shade in every panel, e.g. the crop-calendar
        phases. Shading the phases lets a reader check a curve against the calendar at a glance
        instead of reading dates off the axis.
    sharey : bool
        One y scale for every panel. Use it whenever panels are meant to be *compared*: with free
        scales a 1 dB wiggle and a 12 dB swing fill their panels equally and look alike.

    Colours follow the series **name**, not its position: two series with the same name (say the
    25th and 75th percentile, both labelled as the spread) share a colour and a legend entry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError("no panels to plot")
    for label, series in panels:
        if len({entry[2] for entry in series}) > MAX_SERIES:
            raise ValueError(f"panel {label!r} has more than {MAX_SERIES} distinct series; at most "
                             f"{MAX_SERIES} keep the palette colour-vision safe")

    n = len(panels)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.3 * rows),
                             squeeze=False, facecolor=SURFACE, sharey=sharey)
    seen: dict[str, object] = {}
    colour_of: dict[str, str] = {}

    for ax, (label, series) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)
            ax.spines[spine].set_linewidth(0.8)
        ax.grid(True, color=INK_MUTED, alpha=0.18, linewidth=0.7)
        ax.set_axisbelow(True)
        for start, end, span_label in (spans or []):
            ax.axvspan(start, end, color=INK_MUTED, alpha=0.10, linewidth=0, zorder=0)
            ax.text(start, 1.0, f" {span_label}", transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color=INK_SECONDARY, va="top", ha="left")

        for entry in series:
            dates, values, name = entry[0], np.asarray(entry[1], dtype=float), entry[2]
            recessive = entry[3] if len(entry) > 3 else False
            if name not in colour_of:
                if len(colour_of) >= MAX_SERIES:
                    raise ValueError(f"more than {MAX_SERIES} distinct series names")
                colour_of[name] = SERIES_COLORS[len(colour_of)]
            color = colour_of[name]
            line, = ax.plot(dates, values, color=color,
                            linewidth=1.0 if recessive else 2.0,
                            alpha=0.55 if recessive else 1.0,
                            solid_capstyle="round")
            seen.setdefault(name, line)
            if mark_min and np.isfinite(values).any():
                j = int(np.nanargmin(values))
                ax.plot([dates[j]], [values[j]], marker="o",
                        markersize=5 if recessive else 8, color=color,
                        markeredgecolor=SURFACE, markeredgewidth=1.6,
                        alpha=0.55 if recessive else 1.0, zorder=5, linestyle="none")

        ax.set_title(label, fontsize=10, color=INK_SECONDARY, loc="left", pad=6)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    # A legend is always present for two or more series: identity must never be colour alone.
    if len(seen) > 1:
        fig.legend(list(seen.values()), list(seen), loc="lower center", ncol=len(seen),
                   frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
                   bbox_to_anchor=(0.5, -0.01))

    # The title band is placed in INCHES, not figure fractions: a one-row figure is under half the
    # height of a two-row one, so a fixed fractional offset collides with the subtitle there.
    height_in = fig.get_size_inches()[1]
    band_in = 0.0
    if title:
        fig.text(0.012, 1 - 0.28 / height_in, title, fontsize=13, color=INK, ha="left", va="top")
        band_in = 0.46
    if subtitle:
        fig.text(0.012, 1 - (band_in + 0.20) / height_in, subtitle, fontsize=9.5,
                 color=INK_SECONDARY, ha="left", va="top")
        band_in += 0.36

    bottom = 0.045 if len(seen) > 1 else 0.0
    top = 1 - (band_in + 0.14) / height_in if band_in else 1.0
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
