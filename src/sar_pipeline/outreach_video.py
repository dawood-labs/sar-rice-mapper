"""A short square video for a public post about the delivered rice map: no place, no area id, no map outline.

Why
---
A post with a moving picture reaches people who never read a chart: managers and buyers who want to know what the
work gives them. The video tells the method in five scenes, all from real results:

1. the problem: monsoon clouds hide the fields for weeks;
2. one real field's season: the clear optical views (few) against the radar passes (every one);
3. the same field's curves drawn as time runs: water first, then a crop rising out of it;
4. the delivered numbers, counted up;
5. what it is for.

Nothing on screen can locate a field: only month names, the measured curves and totals. The field's curves come from
``analysis/explainer_curves`` (a reviewed field where the reviewer and the rule agree); the totals are passed in by
the caller from the delivery report. Check the video by eye before posting it.

Use::

    python -m sar_pipeline.outreach_video --out processed/_batch/s2_2026/outreach/rice_delivery_video.mp4

Needs ``imageio-ffmpeg`` (a bundled ffmpeg binary).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .outreach_figures import HAIRLINE, PHASE, SLOT_1, SLOT_2, SLOT_3, SURFACE, TEXT_1, TEXT_2, TEXT_3

CURVES = "processed/_batch/s2_2026/rice_map_2026-10-05_qc/_summary/explainer_curves.json"
#: The delivered totals (client delivery of 16 Oct 2026, ``client_delivery``; areas = union of the AOIs).
TOTALS = (("233", "farming areas"), ("159,979", "acres of farmland scanned"), ("199,655", "rice fields mapped"),
          ("99,347", "acres of rice"))
#: Scene boundaries in seconds.
SCENES = (0.0, 5.0, 12.0, 22.0, 30.0, 37.0)
FPS = 30
T0 = pd.Timestamp("2026-03-01")
T1 = pd.Timestamp("2026-10-05")
#: Phases of the example field (read from its curves; see ``explainer_curves`` page): start, end, label.
PHASES = (("2026-05-12", "2026-07-03", "earlier crop"), ("2026-07-04", "2026-07-24", "flooded"),
          ("2026-07-25", "2026-10-04", "rice growing"))


def ease(x: float) -> float:
    """0 -> 1 with a soft start and end (smoothstep), clipped."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def scene_at(t: float, bounds=SCENES) -> tuple[int, float]:
    """(scene index, seconds into it) at time ``t``."""
    i = int(np.searchsorted(bounds, t, side="right") - 1)
    i = min(max(i, 0), len(bounds) - 2)
    return i, t - bounds[i]


def count_text(final: str, frac: float) -> str:
    """A number string counted up to ``final`` (keeps its thousands commas)."""
    n = int(final.replace(",", ""))
    v = int(round(n * ease(frac)))
    return f"{v:,}" if "," in final else str(v)


def example(path: str = CURVES, cls: str = "rice standing transplanted") -> dict:
    """The example field's clear optical views, radar passes (all tracks) and curves, as day numbers since T0."""
    e = next(x for x in json.loads(Path(path).read_text()) if x["class"] == cls)
    day = lambda s: (pd.Timestamp(s) - T0).days  # noqa: E731
    t = max(e["tracks"], key=lambda k: np.isfinite(np.array(k["VH"], dtype=float)).sum())
    vh = np.array(t["VH"], dtype=float)
    radar = sorted({day(d) for k in e["tracks"] for d in k["dates"]})
    fit = [(day(d), v) for d, v in zip(e["fit"]["dates"], e["fit"]["ndvi"]) if v is not None]
    return {"clear": [day(d) for d, _ in e["clear"]], "radar": radar,
            "vh": [(day(d), v) for d, v in zip(t["dates"], vh) if np.isfinite(v)],
            "fit": fit, "clear_vals": [(day(d), v) for d, v in e["clear"]]}


def _months(ax, y=None):
    ticks = pd.date_range(T0, T1, freq="MS")
    ax.set_xticks([(d - T0).days for d in ticks], [d.strftime("%b") for d in ticks], color=TEXT_2, fontsize=15)
    ax.set_xlim(0, (T1 - T0).days)


def _clean(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def render(out: str, ex: dict, totals=TOTALS, fps: int = FPS, size_px: int = 1080, seconds: float | None = None) -> Path:
    """Writes the video (H.264 MP4, square ``size_px``). ``seconds`` cuts it short (tests)."""
    import imageio_ffmpeg
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi, facecolor=SURFACE)
    total = SCENES[-1] if seconds is None else seconds
    jun = (pd.Timestamp("2026-06-01") - T0).days
    oct1 = (pd.Timestamp("2026-10-01") - T0).days
    n_clear = sum(jun <= d < oct1 for d in ex["clear"])
    n_radar = sum(jun <= d < oct1 for d in ex["radar"])
    rng = np.random.default_rng(3)
    clouds = rng.uniform([0, 0.2, 0.18, 0.06], [1, 1, 0.42, 0.14], size=(14, 4))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    w = FFMpegWriter(fps=fps, codec="libx264", bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])
    with w.saving(fig, str(out), dpi):
        for f in range(int(total * fps)):
            _draw(fig, f / fps, ex, totals, clouds, n_clear, n_radar)
            w.grab_frame(facecolor=SURFACE)
    plt.close(fig)
    return out


def _draw(fig, t, ex, totals, clouds, n_clear, n_radar):
    fig.clf()
    fig.set_facecolor(SURFACE)
    s, u = scene_at(t)
    {0: _hook, 1: _views, 2: _curves, 3: _numbers, 4: _close}[s](fig, u, ex, totals, clouds, n_clear, n_radar)


def stills(ex: dict, times, out_dir: str, totals=TOTALS, size_px: int = 1080) -> list[Path]:
    """PNG frames at the given seconds, to check the layout by eye before rendering the video."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(size_px / 100, size_px / 100), dpi=100)
    jun, oct1 = (pd.Timestamp("2026-06-01") - T0).days, (pd.Timestamp("2026-10-01") - T0).days
    clouds = np.random.default_rng(3).uniform([0, 0.2, 0.18, 0.06], [1, 1, 0.42, 0.14], size=(14, 4))
    paths = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for t in times:
        _draw(fig, t, ex, totals, clouds, sum(jun <= d < oct1 for d in ex["clear"]),
              sum(jun <= d < oct1 for d in ex["radar"]))
        paths.append(Path(out_dir) / f"still_{t:05.1f}s.png")
        fig.savefig(paths[-1], facecolor=SURFACE)
    plt.close(fig)
    return paths


def _txt(fig, x, y, s, size, alpha=1.0, color=TEXT_1, weight="normal", **kw):
    fig.text(x, y, s, fontsize=size, color=color, alpha=alpha, weight=weight, ha=kw.pop("ha", "center"),
             va=kw.pop("va", "center"), **kw)


def _hook(fig, u, ex, totals, clouds, *_):
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    for x, y, rx, ry in clouds:          # soft grey clouds drifting across
        from matplotlib.patches import Ellipse
        ax.add_patch(Ellipse(((x + 0.03 * u) % 1.2 - 0.1, y), rx, ry, color="#d9d8d3", alpha=0.35 * ease(u / 1.5)))
    _txt(fig, 0.5, 0.66, "Monsoon season.", 46, ease(u / 0.8), weight="bold")
    _txt(fig, 0.5, 0.56, "Clouds hide the fields for weeks.", 30, ease((u - 0.9) / 0.8), TEXT_2)
    _txt(fig, 0.5, 0.36, "So how do you count rice\nyou cannot see?", 38, ease((u - 2.2) / 0.8), weight="bold",
         linespacing=1.3)


def _views(fig, u, ex, totals, clouds, n_clear, n_radar):
    _txt(fig, 0.5, 0.88, "One real field, March to October", 34, ease(u / 0.6), weight="bold")
    ax = fig.add_axes([0.08, 0.38, 0.84, 0.36])
    _clean(ax)
    cur = (T1 - T0).days * ease((u - 0.6) / 4.5)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    _months(ax)
    for y in (0, 1):
        ax.axhline(y, color=HAIRLINE, lw=2, zorder=0)
    ax.text(0, 1.32, "Optical camera: clear views only", color=TEXT_1, fontsize=18, weight="bold")
    ax.text(0, 0.32, "Radar satellite: every pass", color=TEXT_1, fontsize=18, weight="bold")
    c = [d for d in ex["clear"] if d <= cur]
    r = [d for d in ex["radar"] if d <= cur]
    ax.scatter(c, [1] * len(c), s=170, color=SLOT_3, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter(r, [0] * len(r), s=110, color=SLOT_1, edgecolor=SURFACE, linewidth=2, zorder=3, marker="D")
    ax.axvline(cur, color=TEXT_3, lw=1.5, alpha=0.6)
    a = ease((u - 5.2) / 0.8)
    _txt(fig, 0.5, 0.25, f"June to September: {n_clear} clear camera views,", 25, a, TEXT_2)
    _txt(fig, 0.5, 0.19, f"{n_radar} radar passes. Radar sees through cloud.", 25, a, TEXT_2)


def _curves(fig, u, ex, totals, clouds, *_):
    _txt(fig, 0.5, 0.92, "Every rice field leaves a signature", 34, ease(u / 0.6), weight="bold")
    cur = (T1 - T0).days * ease((u - 0.5) / 7.0)
    axes = [fig.add_axes([0.08, 0.53, 0.86, 0.28]), fig.add_axes([0.08, 0.16, 0.86, 0.28])]
    for k, ax in enumerate(axes):
        _clean(ax)
        ax.grid(axis="y", color=HAIRLINE, lw=1)
        _months(ax)
        if k == 0:
            ax.set_xticklabels([])
        for a, b, lab in PHASES:
            d0, d1 = (pd.Timestamp(a) - T0).days, (pd.Timestamp(b) - T0).days
            if cur >= d0:
                col = "#dbe9f8" if lab == "flooded" else PHASE
                ax.axvspan(d0, min(d1, cur), color=col, zorder=0)
                if k == 0 and cur >= (d0 + d1) / 2:
                    ax.text((d0 + d1) / 2, 0.98, lab, ha="center", va="top", fontsize=16, color=TEXT_1, weight="bold")
    x, y = zip(*[p for p in ex["fit"] if p[0] <= cur]) if any(p[0] <= cur for p in ex["fit"]) else ((), ())
    axes[0].plot(x, y, color=SLOT_3, lw=3.5)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_yticks([0, 0.5, 1], ["", "", ""])
    axes[0].set_title("Greenness from the camera", loc="left", fontsize=17, color=TEXT_2, pad=6)
    x, y = zip(*[p for p in ex["vh"] if p[0] <= cur]) if any(p[0] <= cur for p in ex["vh"]) else ((), ())
    axes[1].plot(x, y, color=SLOT_1, lw=3.5, marker="o", ms=9, mec=SURFACE, mew=2)
    axes[1].set_ylim(-23, -11)
    axes[1].set_yticks([-21, -17, -13], ["", "", ""])
    axes[1].set_title("Radar echo: drops when the field floods, climbs as the crop grows", loc="left", fontsize=17,
                      color=TEXT_2, pad=6)
    _txt(fig, 0.5, 0.06, "Water first, then a crop rising out of it.", 25, ease((u - 7.8) / 0.8), TEXT_2)


def _numbers(fig, u, ex, totals, *_):
    _txt(fig, 0.5, 0.88, "Delivered: field by field", 38, ease(u / 0.6), weight="bold")
    for i, (num, lab) in enumerate(totals):
        x, y = (0.28, 0.72)[i % 2], (0.62, 0.36)[i // 2]
        a = ease((u - 0.6 - 0.5 * i) / 0.6)
        _txt(fig, x, y, count_text(num, (u - 0.6 - 0.5 * i) / 2.5), 50, a, SLOT_1 if i == 3 else TEXT_1, weight="bold")
        _txt(fig, x, y - 0.07, lab, 22, a, TEXT_2)
    _txt(fig, 0.5, 0.14, "Every area checked by an analyst before delivery.", 23, ease((u - 4.5) / 0.8), TEXT_2)


def _close(fig, u, *_):
    _txt(fig, 0.5, 0.64, "Know what is growing, where,\nbefore the harvest.", 38, ease(u / 0.8), weight="bold",
         linespacing=1.3)
    _txt(fig, 0.5, 0.44, "Radar + optical satellites + expert review", 26, ease((u - 1.2) / 0.8), TEXT_2)
    _txt(fig, 0.5, 0.30, "For agribusiness, lenders, insurers and supply chains.", 24, ease((u - 2.2) / 0.8), TEXT_2)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.outreach_video", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="processed/_batch/s2_2026/outreach/rice_delivery_video.mp4")
    p.add_argument("--curves", default=CURVES)
    p.add_argument("--stills", type=float, nargs="*", help="only PNG frames at these seconds (layout check)")
    a = p.parse_args(argv)
    if a.stills:
        for q in stills(example(a.curves), a.stills, str(Path(a.out).parent / "stills")):
            print(q)
        return 0
    print(render(a.out, example(a.curves)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
