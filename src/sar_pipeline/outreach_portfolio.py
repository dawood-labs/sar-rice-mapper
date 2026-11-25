"""Portfolio visuals for the public site (2 Nov 2026): four JPGs in the site's dark style, from real outputs only.

Why: the portfolio shows the work to people who never open a notebook. The images follow the site's shared style
(dark ``#07111c``, light text, green / blue accents, legend, scale bar or lat/lon note, north arrow on maps) and name no
client, company or exact place: maps say only the country, with a scale bar and no coordinates (user, 2 Nov 2026).

1. :func:`hero_map`: one AOI's QC'd rice fields over its clearest Sentinel-2 view (true colour);
2. :func:`curves`: one reviewed rice field's season, optical greenness and radar backscatter, phases named;
3. :func:`accuracy`: rice vs non-rice agreement with the human QC (confusion matrix in acres + the four metrics);
4. :func:`scale`: every mapped AOI as a dot on the country outline, with the delivered totals.

Inputs (local, gitignored): the QC'd layers, ``_summary/`` (all_aois.gpkg, explainer_curves.json), ``accuracy/``,
the AOI's best clear image (``best_clear_s2_sep_oct_2026`` on GCS, fetched once) and a public country outline
(Natural Earth admin-0, 10 m). Output: ``processed/_batch/s2_2026/outreach/portfolio/*.jpg`` (1600 px, 85 %).

Run::

    python -m sar_pipeline.outreach_portfolio --aoi 160 --country <country name>
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("processed/_batch/s2_2026/outreach/portfolio")
QC = Path("processed/_batch/s2_2026/rice_map_2026-10-05_qc")
COUNTRY_ZIP = Path("../data/basemaps_public/ne_10m_admin_0_countries.zip")
BG, FG, MUTED, GRID = "#07111c", "#e6edf3", "#8b98a5", "#1c2a38"
GREEN, BLUE, ORANGE = "#34d399", "#38bdf8", "#f59e0b"
ACRE = 4046.8564224


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.edgecolor": GRID, "figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG})
    return plt


def save_jpg(fig, path: Path, longest: int = 1600, quality: int = 85, max_kb: int = 400) -> Path:
    """The figure as a JPG ``longest`` px on its longest side, ``quality`` %, lowered until under ``max_kb``."""
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)
    im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    s = longest / max(im.size)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    q = quality
    while True:
        im.save(path, "JPEG", quality=q, optimize=True, progressive=True)
        if path.stat().st_size <= max_kb * 1024 or q <= 60:
            return path
        q -= 5


def scale_bar(ax, length_m: float, label: str, x0: float = 0.05, y0: float = 0.05) -> None:
    """A two-tone scale bar in data units (metres), bottom left."""
    from matplotlib.patches import Rectangle

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x = xmin + x0 * (xmax - xmin)
    y = ymin + y0 * (ymax - ymin)
    h = 0.012 * (ymax - ymin)
    for i, c in enumerate((FG, BG)):
        ax.add_patch(Rectangle((x + i * length_m / 2, y), length_m / 2, h, facecolor=c, edgecolor=FG, lw=1, zorder=5))
    ax.text(x + length_m / 2, y + 2.2 * h, label, ha="center", va="bottom", fontsize=11, color=FG, zorder=5)


def north_arrow(ax, x: float = 0.94, y: float = 0.88) -> None:
    ax.annotate("N", xy=(x, y + 0.07), xytext=(x, y - 0.02), xycoords="axes fraction", ha="center", va="center",
                fontsize=14, color=FG, weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=FG, lw=2, mutation_scale=18))


def best_clear_image(aoi: int, cache: Path = OUT / "cache") -> Path:
    """The AOI's chosen clear Sentinel-2 image (``best_clear_s2_sep_oct_2026``), downloaded once."""
    from google.cloud import storage

    from . import auth
    from .analysis.rice_map_delivery import bucket_name
    from .newest_imagery import season_config

    cfg = season_config(aoi)
    client = storage.Client(credentials=auth.credentials(cfg), project=cfg["auth"]["project"])
    pre = f"{cfg['gcs']['base_folder']}/best_clear_s2_sep_oct_2026/"
    name = next(b.name for b in client.list_blobs(cfg["gcs"]["bucket"], prefix=pre + f"{bucket_name(aoi)}_S2_"))
    out = cache / name.rsplit("/", 1)[-1]
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        client.bucket(cfg["gcs"]["bucket"]).blob(name).download_to_filename(str(out))
    return out


def hero_map(aoi: int, country: str, out: Path = OUT) -> Path:
    """QC'd rice fields over the AOI's clearest Sentinel-2 view, true colour, AOI-relative metres (no coordinates)."""
    import geopandas as gpd
    import rasterio
    from matplotlib.patches import Patch

    from .optical_export import band_index

    plt = _style()
    img = best_clear_image(aoi)
    fields = gpd.read_file(QC / "first_set" / f"aoi{aoi}" / f"aoi{aoi}_fields_qc.gpkg")
    with rasterio.open(img) as ds:
        rgb = np.dstack([ds.read(band_index(ds, b)).astype("float32") for b in ("B4", "B3", "B2")])
        bounds, crs = ds.bounds, ds.crs
    lo, hi = np.nanpercentile(rgb[rgb > 0], (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.85
    fields = fields.to_crs(crs)
    x0, y0 = fields.total_bounds[0], fields.total_bounds[1]
    ext = (bounds.left - x0, bounds.right - x0, bounds.bottom - y0, bounds.top - y0)
    fig, ax = plt.subplots(figsize=(10, 10 * (ext[3] - ext[2]) / (ext[1] - ext[0])))
    ax.imshow(rgb, extent=ext, interpolation="bilinear")
    f = fields.translate(-x0, -y0)
    f.plot(ax=ax, facecolor=GREEN + "66", edgecolor=GREEN, linewidth=0.5)
    ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    acres = fields.to_crs(32646).area.sum() / ACRE
    ax.text(0.03, 0.97, "Rice fields, monsoon 2026", transform=ax.transAxes, fontsize=20, weight="bold", va="top",
            color=FG, bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=6))
    ax.text(0.03, 0.905, f"{len(f):,} fields, {acres:,.0f} acres  ·  Sentinel-1 radar + Sentinel-2  ·  {country}",
            transform=ax.transAxes, fontsize=12, va="top", color=FG,
            bbox=dict(facecolor=BG, alpha=0.75, edgecolor="none", pad=5))
    ax.legend(handles=[Patch(facecolor=GREEN + "66", edgecolor=GREEN, label="rice field (after QC)")],
              loc="lower right", frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11)
    width = ext[1] - ext[0]
    km = max(0.5, round(width / 5000 * 2) / 2)
    scale_bar(ax, km * 1000, f"{km:g} km")
    north_arrow(ax)
    p = save_jpg(fig, out / "rice_fields_hero.jpg")
    plt.close(fig)
    return p


def curves(country: str, out: Path = OUT, cls: str = "rice standing transplanted") -> Path:
    """One reviewed rice field's season: optical greenness and radar VH / VV on two panels, phases shaded."""
    plt = _style()
    e = next(x for x in json.loads((QC / "_summary" / "explainer_curves.json").read_text()) if x["class"] == cls)
    t0 = pd.Timestamp("2026-03-01")
    day = lambda s: (pd.Timestamp(s) - t0).days  # noqa: E731
    phases = (("2026-05-12", "2026-07-03", "earlier crop", "#16324a"), ("2026-07-04", "2026-07-24", "flooded", "#0e3a5a"),
              ("2026-07-25", "2026-10-04", "rice growing", "#123d2f"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True, gridspec_kw={"hspace": 0.12})
    for ax in (a1, a2):
        for a, b, lab, c in phases:
            ax.axvspan(day(a), day(b), color=c, zorder=0)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for a, b, lab, c in phases:
        a1.text((day(a) + day(b)) / 2, 0.98, lab, ha="center", va="top", fontsize=11, color=FG, weight="bold")
    fx = [(day(d), v) for d, v in zip(e["fit"]["dates"], e["fit"]["ndvi"]) if v is not None]
    a1.plot(*zip(*fx), color=GREEN, lw=2.5, label="greenness (NDVI, 5-day fit)")
    cx = [(day(d), v) for d, v in e["clear"]]
    a1.scatter(*zip(*cx), s=28, color=GREEN, edgecolor=BG, zorder=3, label="clear optical view")
    a1.set_ylim(-0.05, 1.05)
    a1.set_ylabel("NDVI")
    t = max(e["tracks"], key=lambda k: np.isfinite(np.array(k["VH"], dtype=float)).sum())
    for pol, c in (("VH", BLUE), ("VV", ORANGE)):
        xy = [(day(d), v) for d, v in zip(t["dates"], t[pol]) if v == v]
        a2.plot(*zip(*xy), color=c, lw=2, marker="o", ms=5, mec=BG, label=f"radar {pol}")
    a2.set_ylabel("backscatter (dB)")
    ticks = pd.date_range(t0, "2026-10-01", freq="MS")
    a2.set_xticks([(d - t0).days for d in ticks], [d.strftime("%b") for d in ticks])
    a2.set_xlim(0, day("2026-10-05"))
    a1.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=FG, ncol=2)
    a2.legend(loc="upper left", frameon=False, fontsize=10, labelcolor=FG, ncol=2)
    fig.suptitle("A rice field's season: water first, then the crop rises out of it", fontsize=16, weight="bold",
                 x=0.06, ha="left", y=0.98)
    fig.text(0.06, 0.925, f"Real field, {country} 2026. Radar sees through monsoon cloud; optical adds greenness on clear days.",
             fontsize=11, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.07)
    p = save_jpg(fig, out / "radar_optical_curves.jpg")
    plt.close(fig)
    return p


def accuracy(out: Path = OUT) -> Path:
    """Confusion matrix (acres) and precision / recall / F1 / accuracy of the map against the human QC."""
    plt = _style()
    m = json.loads(Path("processed/_batch/s2_2026/accuracy/qc_metrics.json").read_text())
    a = m["ac"]
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_axes([0.1, 0.08, 0.38, 0.64])
    tot = sum(a[k] for k in ("tp", "fp", "fn", "tn"))
    cells = [("tp", 0, 1, True), ("fn", 1, 1, False), ("fp", 0, 0, False), ("tn", 1, 0, True)]
    for k, x, y, good in cells:
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor="#123d2f" if good else "#3a2410", edgecolor=BG, lw=4))
        ax.text(x + 0.5, y + 0.58, f"{a[k]:,.0f} ac", ha="center", fontsize=18, weight="bold", color=FG)
        ax.text(x + 0.5, y + 0.36, f"{100 * a[k] / tot:.1f} % of area", ha="center", fontsize=11, color=MUTED)
    ax.set_xlim(0, 2), ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["rice", "non-rice"], fontsize=12)
    ax.set_yticks([1.5, 0.5], ["rice", "non-rice"], fontsize=12)
    ax.xaxis.tick_top(), ax.xaxis.set_label_position("top")
    ax.set_xlabel("map says", color=MUTED, fontsize=12, labelpad=10)
    ax.set_ylabel("human QC says", color=MUTED, fontsize=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    bx = fig.add_axes([0.58, 0.12, 0.39, 0.62])
    names, keys = ["Accuracy", "Precision", "Recall", "F1"], ["accuracy", "precision", "recall", "f1"]
    bars = bx.bar(names, [a[k] for k in keys], color=[BLUE, BLUE, GREEN, GREEN], width=0.6)
    for b, k in zip(bars, keys):
        bx.text(b.get_x() + b.get_width() / 2, a[k] + 1.5, f"{a[k]:.1f} %", ha="center", fontsize=13, weight="bold",
                color=FG)
    bx.set_ylim(0, 110)
    bx.set_yticks([0, 25, 50, 75, 100])
    bx.grid(axis="y", color=GRID)
    bx.set_axisbelow(True)
    for s in ("top", "right", "left"):
        bx.spines[s].set_visible(False)
    fig.text(0.05, 0.93, "Rice vs non-rice: map against a field-by-field human QC", fontsize=17, weight="bold")
    fig.text(0.05, 0.885, f"{m['aois']} areas, by area (acres). Agreement with the QC, not a field survey.",
             fontsize=11, color=MUTED)
    p = save_jpg(fig, out / "qc_agreement.jpg")
    plt.close(fig)
    return p


def scale(country: str, out: Path = OUT) -> Path:
    """Every mapped AOI as a dot on the country outline, with the delivered totals."""
    import geopandas as gpd

    plt = _style()
    world = gpd.read_file(f"zip://{COUNTRY_ZIP.resolve()}")
    outline = world[world["ADMIN"] == country].to_crs(32646)
    aois = gpd.read_file(QC / "_summary" / "all_aois.gpkg")
    aois = aois[aois["status"] == "mapped"].to_crs(32646)
    pts = aois.geometry.centroid
    stats = pd.read_csv(QC / "_summary" / "aoi_stats.csv")
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_axes([0.0, 0.03, 0.55, 0.94])
    outline.plot(ax=ax, facecolor="#0f2233", edgecolor="#3b5873", lw=0.8)
    ax.scatter(pts.x, pts.y, s=np.clip(aois["aoi_ac"].to_numpy() / 40, 8, 120), color=GREEN, alpha=0.8,
               edgecolor=BG, lw=0.4, zorder=3)
    ax.set_axis_off()
    ax.set_aspect("equal")
    scale_bar(ax, 200_000, "200 km", x0=0.06, y0=0.04)
    north_arrow(ax, x=0.9, y=0.88)
    ax.text(0.5, 0.01, country, transform=ax.transAxes, ha="center", fontsize=12, color=MUTED)
    import shapely

    union_ac = shapely.union_all(aois.geometry.to_numpy()).area / ACRE   # AOIs overlap: their union, counted once
    tiles = [(f"{int((stats['status'] == 'mapped').sum()):,}", "farming areas mapped"),
             ("99,347", "acres of rice, field by field"),
             ("199,655", "rice fields delivered"),
             (f"{union_ac:,.0f}", "acres analysed")]
    fig.text(0.58, 0.88, "Monsoon-2026 rice,\nmapped from space", fontsize=19, weight="bold", va="bottom",
             linespacing=1.15)
    fig.text(0.58, 0.835, "Sentinel-1 radar + Sentinel-2 optical,\nper-pixel rules, human QC", fontsize=11,
             color=MUTED, va="top")
    for i, (num, lab) in enumerate(tiles):
        y = 0.68 - i * 0.16
        fig.text(0.58, y, num, fontsize=30, weight="bold", color=GREEN if i == 1 else FG)
        fig.text(0.58, y - 0.045, lab, fontsize=12, color=MUTED)
    fig.text(0.58, 0.05, "dot size = area of the AOI", fontsize=10, color=MUTED)
    p = save_jpg(fig, out / "rice_scale_map.jpg")
    plt.close(fig)
    return p

STAGE = Path("processed/_batch/s2_2026/crop_stage")
STAGE_DAY = "2026-11-15"


def _stage_layer(aoi: int) -> Path:
    """The newest delivered crop-stage field layer of the area (highest version)."""
    def ver(f: Path) -> int:
        tail = f.stem.rsplit("_v", 1)
        return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 1

    return max((STAGE / f"aoi{aoi}" / "delivery").glob("*_crop_stage_*.gpkg"), key=ver)


def stage_map(aoi: int, country: str, out: Path = OUT, day: str = STAGE_DAY, layer: Path | None = None,
              image: Path | None = None) -> Path:
    """Every field outlined and tinted by its status on the reference day (standing / harvested), over that day's clear
    Sentinel-2 image (true colour, dimmed). Why: shows the crop-stage result at a glance. No coordinates are drawn."""
    import geopandas as gpd
    import rasterio
    from matplotlib.patches import Patch

    from .optical_export import band_index

    plt = _style()
    layer = layer or _stage_layer(aoi)
    image = image or next(Path("data/s2_dates_masks").glob(f"aoi{aoi}/*S2_{day}.tif"))
    with rasterio.open(image) as ds:
        rgb = np.dstack([ds.read(band_index(ds, b)).astype("float32") for b in ("B4", "B3", "B2")])
        bounds, crs = ds.bounds, ds.crs
    lo, hi = np.nanpercentile(rgb[rgb > 0], (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.85 * 0.7          # dimmed so the outlines read
    fields = gpd.read_file(layer).to_crs(crs)
    xa, ya, xb, yb = fields.total_bounds
    m = 0.06 * max(xb - xa, yb - ya)
    xa, ya, xb, yb = xa - m, ya - m, xb + m, yb + m
    x0, y0 = xa, ya                                                    # AOI-relative metres, no coordinates
    ext = (bounds.left - x0, bounds.right - x0, bounds.bottom - y0, bounds.top - y0)
    w, h = xb - xa, yb - ya
    fw, mh, hd = 10.0, 10.0 * h / w, 1.1
    fig = plt.figure(figsize=(fw, mh + hd))
    ax = fig.add_axes([0, 0, 1, mh / (mh + hd)])
    ax.imshow(rgb, extent=ext, interpolation="bilinear")
    col = {"standing": GREEN, "harvested": ORANGE}
    for st, c in col.items():
        fields[fields["status"] == st].translate(-x0, -y0).plot(ax=ax, facecolor=c + "55", edgecolor=c, linewidth=0.9)
    ax.set_xlim(0, w), ax.set_ylim(0, h)
    ax.set_axis_off()
    ac = fields.assign(a=fields["area_ac"]).groupby("status")["a"].agg(["size", "sum"])
    hand = [Patch(facecolor=col[k] + "55", edgecolor=col[k],
                  label=f"{k}: {int(ac.loc[k, 'size']):,} fields, {ac.loc[k, 'sum']:,.0f} acres")
            for k in col if k in ac.index]
    ax.legend(handles=hand, loc="lower right", frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=11)
    km = max(0.2, round(w / 5000 * 10) / 10)
    scale_bar(ax, km * 1000, f"{km:g} km")
    north_arrow(ax, y=0.84)
    fig.text(0.03, 1 - 0.38 / (mh + hd), "Standing or harvested, field by field", fontsize=22, weight="bold",
             va="center", color=FG)
    fig.text(0.03, 1 - 0.82 / (mh + hd), f"Pilot area, {country}  ·  reference day mid-November 2026", fontsize=13,
             va="center", color=MUTED)
    p = save_jpg(fig, out / "crop_stage_map.jpg")
    plt.close(fig)
    return p


def stage_calendar(aoi: int, out: Path = OUT, numbers: dict | None = None) -> Path:
    """One field from planting to harvest (radar and greenness on two stacked panels, never two scales on one axis) and
    the fields per planting week split by status. ``numbers`` = :func:`outreach_figures.stage_numbers` output."""
    import matplotlib.dates as mdates

    if numbers is None:
        from . import outreach_figures as of

        numbers = of.stage_numbers(aoi)
    n = numbers
    plt = _style()
    e = n["example"]
    fig = plt.figure(figsize=(16, 8.6))
    fig.text(0.04, 0.94, "From planting to harvest, read from space", fontsize=26, weight="bold", va="center")
    heads = [f"{n['fields']} fields"]
    if "checked" in n:
        heads.append(f"{n['all_right']} of {n['checked']} right in the ground check")
    heads.append(f"{n['age_harvested']:.0f} days median planting to harvest")
    fig.text(0.04, 0.875, "   ·   ".join(heads), fontsize=15, color=GREEN, weight="bold", va="center")
    x0 = pd.Timestamp(e["sowing"]) - pd.Timedelta(days=45)
    x1 = pd.Timestamp(e["green_dates"].max()) + pd.Timedelta(days=4)
    fig.text(0.04, 0.80, "One real field", fontsize=15, color=MUTED)
    ax1 = fig.add_axes([0.06, 0.45, 0.50, 0.30])
    ax2 = fig.add_axes([0.06, 0.09, 0.50, 0.30], sharex=ax1)
    for ax, d, v, c, lab in ((ax1, e["radar_dates"], e["radar"], BLUE, "Radar echo (dB), sees water and plants through cloud"),
                             (ax2, e["green_dates"], e["green"], GREEN, "Greenness (0-1), sees the canopy on clear days")):
        msk = (d >= np.datetime64(x0)) & (d <= np.datetime64(x1))
        ax.plot(d[msk], v[msk], color=c, lw=2.2, marker="o", ms=5, mec=BG)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvline(e["sowing"], color=FG, lw=1.2, ls=(0, (4, 3)))
        ax.axvspan(e["harvest_from"], e["harvest_to"], color=ORANGE, alpha=0.28, lw=0)
        ax.set_title(lab, loc="left", fontsize=12, color=FG, pad=6)
    ax1.set_xlim(x0, x1)
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelsize=11)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    lo, hi = ax1.get_ylim()
    ax1.set_ylim(lo, hi + (hi - lo) * 0.25)
    ax1.text(e["sowing"], ax1.get_ylim()[1], " field flooded:\n planting", va="top", fontsize=11.5)
    lo, hi = ax2.get_ylim()
    ax2.set_ylim(lo, hi + (hi - lo) * 0.25)
    ax2.text(e["harvest_from"], ax2.get_ylim()[1], "harvest ", va="top", ha="right", fontsize=11.5, color=ORANGE,
             weight="bold")
    ax3 = fig.add_axes([0.66, 0.09, 0.31, 0.66])
    labs = [w for w, _s, _h in n["sow_weeks"]]
    st = np.array([s for _w, s, _h in n["sow_weeks"]])
    hv = np.array([h for _w, _s, h in n["sow_weeks"]])
    xs = np.arange(len(labs))
    ax3.bar(xs, st, color=GREEN, width=0.62, label="standing")
    ax3.bar(xs, hv, bottom=st, color=ORANGE, width=0.62, label="harvested")
    for x, a, b in zip(xs, st, hv):
        ax3.text(x, a + b + (st + hv).max() * 0.015, str(a + b), ha="center", fontsize=12, weight="bold")
    ax3.set_xticks(xs, [w.replace(" week ", "\nweek ") for w in labs], fontsize=10)
    ax3.set_ylim(0, (st + hv).max() * 1.2)
    ax3.grid(axis="y", color=GRID, lw=0.8)
    ax3.set_axisbelow(True)
    for s in ("top", "right"):
        ax3.spines[s].set_visible(False)
    ax3.legend(frameon=False, fontsize=12, loc="upper left", labelcolor=FG)
    fig.text(0.66, 0.80, "Fields by planting week", fontsize=15, color=MUTED)
    p = save_jpg(fig, out / "crop_stage_calendar.jpg")
    plt.close(fig)
    return p


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--aoi", type=int, default=160)
    p.add_argument("--only", nargs="+", choices=["stage_map", "stage_calendar"],
                   help="make only these crop-stage figures (default: the four original figures)")
    p.add_argument("--stage-aoi", type=int, default=1246, help="area of the crop-stage figures")
    p.add_argument("--country", required=True, help="the country name shown on the images (and its outline)")
    a = p.parse_args(argv)
    if a.only:
        figs = {"stage_map": lambda: stage_map(a.stage_aoi, a.country), "stage_calendar": lambda: stage_calendar(a.stage_aoi)}
        made = [figs[k]() for k in a.only]
    else:
        made = [hero_map(a.aoi, a.country), curves(a.country), accuracy(), scale(a.country)]
    for f in made:
        print(f, f"{f.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
