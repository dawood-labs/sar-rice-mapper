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


#: Sequential blue ramp (light -> dark) for heatmaps: one hue, never a rainbow.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
              "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def _title_band(fig, title, subtitle):
    height_in = fig.get_size_inches()[1]
    band_in = 0.0
    if title:
        fig.text(0.012, 1 - 0.28 / height_in, title, fontsize=13, color=INK, ha="left", va="top")
        band_in = 0.46
    if subtitle:
        fig.text(0.012, 1 - (band_in + 0.20) / height_in, subtitle, fontsize=9.5,
                 color=INK_SECONDARY, ha="left", va="top")
        band_in += 0.36
    return 1 - (band_in + 0.14) / height_in if band_in else 1.0


def plot_heatmap(table, out_path, title=None, subtitle=None, unit="", vmin=None, vmax=None,
                 fmt="{:.0f}", rotate=0):
    """A labelled heatmap of a (rows x columns) table, every cell annotated with its value.

    Used for "which group floods in which month" style evidence, where the reader needs the exact
    number, not just a colour. Missing cells are left blank rather than coloured as zero.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    values = np.asarray(table.to_numpy(), dtype=float)
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.0 + 0.62 * values.shape[1]),
                                    max(3.2, 1.4 + 0.42 * values.shape[0])), facecolor=SURFACE)
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    lo = np.nanmin(values) if vmin is None else vmin
    hi = np.nanmax(values) if vmax is None else vmax
    for (i, j), v in np.ndenumerate(values):
        if np.isfinite(v):
            dark = (v - lo) / (hi - lo + 1e-9) > 0.55
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(values.shape[1]), [str(c) for c in table.columns], fontsize=8,
                  color=INK_SECONDARY, rotation=rotate, ha="right" if rotate else "center")
    ax.set_yticks(range(values.shape[0]), [str(r) for r in table.index], fontsize=8.5,
                  color=INK_SECONDARY)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label(unit, fontsize=9, color=INK_SECONDARY)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, _title_band(fig, title, subtitle)))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grouped_bars(table, out_path, title=None, subtitle=None, ylabel="", ymax=None,
                      fmt="{:.0f}", rotate=0):
    """Grouped bars: one group per row of ``table``, one bar per column (at most three columns).

    Every bar is labelled with its value, and a legend names the columns, so identity never rests
    on colour alone.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if table.shape[1] > MAX_SERIES:
        raise ValueError(f"at most {MAX_SERIES} bar series keep the palette colour-vision safe")
    n_rows, n_cols = table.shape
    width = 0.8 / n_cols
    top = float(np.nanmax(table.to_numpy(dtype=float))) if ymax is None else float(ymax)
    offset = 0.01 * top           # label gap scales with the data, not a fixed unit
    fig, ax = plt.subplots(figsize=(1.2 + 0.9 * n_rows, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(n_rows)
    for k, col in enumerate(table.columns):
        vals = table[col].to_numpy(dtype=float)
        bars = ax.bar(x + (k - (n_cols - 1) / 2) * width, vals, width * 0.92,
                      color=SERIES_COLORS[k], label=str(col))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + offset, fmt.format(v), ha="center",
                    va="bottom", fontsize=7, color=INK_SECONDARY)
    ax.set_xticks(x, [str(i) for i in table.index], fontsize=8.5, color=INK_SECONDARY,
                  rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel(ylabel, fontsize=9, color=INK_SECONDARY)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.grid(True, axis="y", color=INK_MUTED, alpha=0.18)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)
    if n_cols > 1:   # a single series is named by the title; a legend box would only add clutter
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, ncol=n_cols,
                  loc="upper left", bbox_to_anchor=(0, 1.02))
    fig.tight_layout(rect=(0, 0, 1, _title_band(fig, title, subtitle)))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path
