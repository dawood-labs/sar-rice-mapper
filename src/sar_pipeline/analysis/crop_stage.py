"""Crop stage of the QC'd rice fields of one AOI up to today (18 Nov 2026): the field curves first, read together.

Why (client request via the user, 18 Nov 2026): for one pilot AOI the client wants, per rice field and as of today,
either (standing) its sowing date and crop age, or (already cut) its harvest date. The earlier automatic sowing dates
were found mostly wrong (16 Oct), so this starts from the evidence: each QC'd field's radar backscatter per pass (every
track, the newest run fetched to today) and its clear optical greenness per date, drawn field by field so the user and
the assistant can read sowing and harvest on real fields before any rule is written.

Inputs: the QC'd field layer (``field_id`` + polygons; pieces of one field share the id), the AOI's grid, a radar run
(``direct_fetch``), and the per-date Sentinel-2 files with the delivered series' mask (``mask_experiment`` variant).
Outputs (``processed/_batch/s2_2026/crop_stage/aoi<N>/``): ``field_curves.parquet`` (long table: field, source,
track, date, value), ``fields.csv`` (field, acres, pixels) and review sheets ``sheet_<k>.png``.

Run::

    python -m sar_pipeline.analysis.crop_stage curves --aoi 1246 --qc ../data/qc/qc_b3_2026-10-30/AOI_246.gpkg \\
        --run v005_20261118
    python -m sar_pipeline.analysis.crop_stage sheet --aoi 1246 --n 9 --seed 0
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("processed/_batch/s2_2026/crop_stage")
ACRE = 4046.8564224
VARIANT = "hyb40m1late_20261001"
#: A date counts for a field when at least this share of its pixels is clear.
MIN_CLEAR_SHARE = 0.5


def out_dir(aoi: int) -> Path:
    return OUT / f"aoi{aoi}"


def load_fields(qc_file):
    """The QC'd field layer with an id on every polygon: a polygon the reviewer drew has none, so it gets
    ``<aoi name>_qc001``, ``_qc002`` ... in file order (as ``qc_post`` names them), the AOI name taken from the ids
    of the other fields. Why (user, 19 Nov 2026): the drawn fields are rice too and belong in the table."""
    import geopandas as gpd

    f = gpd.read_file(qc_file)
    ids = f["field_id"].astype("object")
    missing = ids.isna() | ids.astype(str).isin(["", "None", "nan"])
    stem = str(ids[~missing].iloc[0]).rsplit("_", 1)[0] if (~missing).any() else Path(qc_file).stem
    ids[missing] = [f"{stem}_qc{i + 1:03d}" for i in range(int(missing.sum()))]
    f["field_id"] = ids.astype(str)
    return f


#: Pixel ids of the last :func:`field_pixels` call that are interior to their field (no neighbour outside it).
_INTERIOR: set = set()


def field_pixels(qc_file, gd: dict) -> tuple[dict, pd.DataFrame]:
    """{field_id: flat pixel ids} of pixels whose centre is inside the field (all its pieces), and a table with
    each field's acres (EPSG:32646) and pixel count."""
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    f = load_fields(qc_file).to_crs(gd["crs"])
    ids = sorted(f["field_id"].unique())
    code = {k: i + 1 for i, k in enumerate(ids)}
    w, h = int(gd["width"]), int(gd["height"])
    lab = rasterize(((g, code[k]) for g, k in zip(f.geometry, f["field_id"])), out_shape=(h, w),
                    transform=from_origin(float(gd["x0"]), float(gd["y0"]), float(gd["res"]), float(gd["res"])),
                    fill=0, dtype="int32").ravel()
    pix = {k: np.flatnonzero(lab == code[k]) for k in ids}
    global _INTERIOR
    g2 = lab.reshape(h, w)
    same = np.ones((h, w), dtype=bool)                     # interior = the four neighbours belong to the same field
    same[1:, :] &= g2[1:, :] == g2[:-1, :]
    same[:-1, :] &= g2[:-1, :] == g2[1:, :]
    same[:, 1:] &= g2[:, 1:] == g2[:, :-1]
    same[:, :-1] &= g2[:, :-1] == g2[:, 1:]
    same[[0, -1], :] = False
    same[:, [0, -1]] = False
    _INTERIOR = set(np.flatnonzero(same.ravel() & (lab > 0)).tolist())
    ac = f.to_crs(32646).assign(a=lambda x: x.area / ACRE).groupby("field_id")["a"].sum()
    table = pd.DataFrame({"field_id": ids, "acres": [round(float(ac[k]), 3) for k in ids],
                          "pixels": [len(pix[k]) for k in ids]})
    return pix, table


def curves(aoi: int, qc_file, run: str) -> Path:
    """Field medians: VV / VH per pass of every track of ``run``, NDVI per clear date (>= MIN_CLEAR_SHARE)."""
    from . import ndvi_5day as nd
    from . import radar_water as rw
    from .. import config as cm
    from .mask_experiment import VARIANTS
    from ..newest_imagery import season_config

    cfg = season_config(aoi)
    gd = json.loads((cm.grid_dir(cfg) / "grid_def.json").read_text())
    pix, table = field_pixels(qc_file, gd)
    rows = []
    for (dates, flat), track in zip(rw.read_series(aoi, run_id=run), [t["track_id"] for t in cfg["s1"]["tracks"]]):
        dd = pd.DatetimeIndex(dates)
        for pol in ("VV", "VH"):
            v = flat[pol]
            for k, p in pix.items():
                if not len(p):
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    med = np.nanmedian(v[:, p], axis=1)
                rows += [(k, pol, track, d, float(x)) for d, x in zip(dd, med) if np.isfinite(x)]
    dates, ndvi, _ndwi, _lswi, ok, _scl, _loc = nd.read_dates(aoi, **VARIANTS[VARIANT])
    dd = pd.DatetimeIndex(dates)
    n = ndvi.reshape(len(dd), -1)
    o = ok.reshape(len(dd), -1).astype(bool)
    for k, p in pix.items():
        if not len(p):
            continue
        share = o[:, p].mean(axis=1)
        for i in np.flatnonzero(share >= MIN_CLEAR_SHARE):
            rows.append((k, "NDVI", "S2", dd[i], float(np.median(n[i, p][o[i, p]]))))
    t = pd.DataFrame(rows, columns=["field_id", "source", "track", "date", "value"])
    d = out_dir(aoi)
    d.mkdir(parents=True, exist_ok=True)
    t.to_parquet(d / "field_curves.parquet", index=False)
    table.to_csv(d / "fields.csv", index=False)
    (d / "inputs.json").write_text(json.dumps({"qc": str(Path(qc_file).resolve()), "run": run}, indent=1))
    return d / "field_curves.parquet"


def sheet(aoi: int, field_ids=None, n: int = 9, seed: int = 0, start: str = "2026-05-01", name: str = "sheet_1") -> Path:
    """Review sheet: per field, clear NDVI (top) and radar VH / VV per track (bottom), months on the axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = out_dir(aoi)
    t = pd.read_parquet(d / "field_curves.parquet")
    fields = pd.read_csv(d / "fields.csv")
    if field_ids is None:
        f = fields[fields["pixels"] >= 9].sort_values("acres")
        field_ids = list(f.iloc[np.linspace(0, len(f) - 1, n).round().astype(int)]["field_id"].drop_duplicates()) \
            if seed == 0 else list(f.sample(min(n, len(f)), random_state=seed)["field_id"])
    t0, t1 = pd.Timestamp(start), pd.Timestamp.now().normalize() + pd.Timedelta(days=3)
    cols = 3
    rows = int(np.ceil(len(field_ids) / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(6.2 * cols, 4.4 * rows), squeeze=False, sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.2] * rows, "hspace": 0.12, "wspace": 0.18})
    colours = {("VH", 0): "#2a78d6", ("VV", 0): "#eb6834", ("VH", 1): "#7fb2ef", ("VV", 1): "#f4a77e"}
    tracks = sorted(t.loc[t["source"] != "NDVI", "track"].unique())
    for i, fid in enumerate(field_ids):
        r, c = divmod(i, cols)
        a1, a2 = axes[2 * r, c], axes[2 * r + 1, c]
        g = t[(t["field_id"] == fid) & (t["date"] >= t0)]
        o = g[g["source"] == "NDVI"]
        a1.plot(o["date"], o["value"], "o-", color="#1baf7a", ms=4, lw=1)
        a1.set_ylim(-0.1, 1.0)
        a1.grid(alpha=0.3)
        acres = float(fields.loc[fields["field_id"] == fid, "acres"].iloc[0])
        a1.text(0.01, 0.95, f"{fid}  ({acres:.2f} ac)", transform=a1.transAxes, fontsize=10, va="top",
                weight="bold", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))
        for j, tr in enumerate(tracks):
            for pol in ("VH", "VV"):
                s = g[(g["source"] == pol) & (g["track"] == tr)].sort_values("date")
                a2.plot(s["date"], s["value"], ".-", color=colours[(pol, j % 2)], lw=1.2, ms=5,
                        label=f"{pol} {tr}" if i == 0 else None)
        a2.grid(alpha=0.3)
        for ax in (a1, a2):
            ax.set_xlim(t0, t1)
            ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    for k in range(len(field_ids), rows * cols):
        r, c = divmod(k, cols)
        axes[2 * r, c].set_visible(False)
        axes[2 * r + 1, c].set_visible(False)
    fig.legend(loc="upper right", ncol=4, fontsize=9, frameon=False)
    fig.suptitle(f"aoi{aoi}: QC'd rice fields to {pd.Timestamp.now():%d %b %Y}  (top: clear NDVI; bottom: radar dB)",
                 x=0.01, ha="left", fontsize=12)
    p = d / f"{name}.png"
    fig.savefig(p, dpi=90, bbox_inches="tight")
    plt.close(fig)
    pd.Series(field_ids, name="field_id").to_csv(d / f"{name}_fields.csv", index=False)
    return p


def _rain(aoi: int, track: str) -> pd.Series:
    """Rain (mm in the 6 h before the pass, from the run's pass table) of one track's passes that had any, by date.
    Why (user, 19 Nov 2026): a wet surface lifts the radar (12 Nov: +2.3 dB on every field), so rainy passes are
    marked on the plot instead of being dropped."""
    from .. import config as cm
    from ..newest_imagery import season_config

    run = json.loads((out_dir(aoi) / "inputs.json").read_text())["run"]
    f = Path(cm.analysis_run_dir(season_config(aoi))).parent / run / "stack" / f"track_{track}" / "dates.csv"
    t = pd.read_csv(f)
    t = t[t["rain_6h_mm"] >= 0.5]
    return pd.Series(t["rain_6h_mm"].to_numpy(), index=pd.to_datetime(t["date_utc"]).dt.normalize())


def resolve_field(aoi: int, field) -> str:
    """The full field id from what the user types: ``b3_aoi246_p000111``, ``p000111``, ``000111`` or ``111``."""
    ids = pd.read_csv(out_dir(aoi) / "fields.csv")["field_id"].astype(str)
    f = str(field).strip()
    if f in set(ids):
        return f
    hits = [i for i in ids if i.endswith(f) or i.endswith(f"p{int(f):06d}" if f.isdigit() else f)]
    if len(hits) != 1:
        raise KeyError(f"field {field!r}: {len(hits)} matches in aoi{aoi}")
    return hits[0]


def plot_field(aoi: int, field, start: str = "2026-04-01"):
    """One field, radar first (user, 19 Nov 2026: SAR leads, NDVI supports, as in the harvest work): VH and VV per
    track on their own panels (each pass a dot, dates readable), clear NDVI below. Returns (figure, info dict)."""
    import matplotlib

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fid = resolve_field(aoi, field)
    d = out_dir(aoi)
    t = pd.read_parquet(d / "field_curves.parquet")
    fields = pd.read_csv(d / "fields.csv").set_index("field_id")
    g = t[(t["field_id"] == fid) & (t["date"] >= pd.Timestamp(start))]
    t1 = pd.Timestamp.now().normalize() + pd.Timedelta(days=3)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]})
    colours = ["#2a78d6", "#eb6834"]
    tracks = sorted(t.loc[t["source"] != "NDVI", "track"].unique())
    for ax, pol in zip(axes[:2], ("VH", "VV")):
        for j, tr in enumerate(tracks):
            s = g[(g["source"] == pol) & (g["track"] == tr)].sort_values("date")
            ax.plot(s["date"], s["value"], "o-", color=colours[j % 2], lw=1.6, ms=6, label=tr)
            for x, y in zip(s["date"], s["value"]):
                ax.annotate(x.strftime("%d %b"), (x, y), textcoords="offset points", xytext=(0, 7), ha="center",
                            fontsize=6.5, color=colours[j % 2])
            rain = _rain(aoi, tr)
            wet = s[s["date"].isin(rain.index)]
            if len(wet):                                  # rain in the 6 h before the pass: ring grows with the amount
                mm = rain.reindex(wet["date"]).to_numpy()
                ax.scatter(wet["date"], wet["value"], s=60 + 25 * mm, facecolors="none", edgecolors="#1565c0",
                           linewidths=1.6, zorder=4)
                for x, y, r_ in zip(wet["date"], wet["value"], mm):
                    ax.annotate(f"{r_:.0f} mm", (x, y), textcoords="offset points", xytext=(0, -13), ha="center",
                                fontsize=6.5, color="#1565c0")
        ax.set_ylabel(f"{pol} (dB)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, title="blue ring = rain before the pass (mm in 6 h)", title_fontsize=8)
    o = g[g["source"] == "NDVI"].sort_values("date")
    axes[2].plot(o["date"], o["value"], "o-", color="#1baf7a", lw=1, ms=4)
    axes[2].set_ylim(-0.1, 1.0)
    axes[2].set_ylabel("NDVI (clear)")
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(pd.Timestamp(start), t1)
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes:
        ax.grid(which="minor", axis="x", alpha=0.15)
        ax.tick_params(axis="x", which="both", labelbottom=True)     # dates under every panel (user, 19 Nov)
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d"))
        ax.tick_params(axis="x", which="minor", labelsize=7, colors="#666666")
    ac, px = float(fields.loc[fid, "acres"]), int(fields.loc[fid, "pixels"])
    rule = ""
    if (d / "field_stage.csv").exists():                   # the rule's reading of this field, to compare by eye
        fs = pd.read_csv(d / "field_stage.csv").set_index("field_id")
        if fid in fs.index:
            r = fs.loc[fid]
            rule = (f"\nrule: planting {r['sowing_median']} (pixels {r['sowing_q25']} .. {r['sowing_q75']}), "
                    f"transplanted {r['transplanted_pct']:.0f} %\n{str(r.get('status', '')).upper()} (field curve)"
                    + (f": cut between {r['harvest_from']} and {r['harvest_to']}" if isinstance(r["harvest_from"], str) else "")
                    + f", age {r['age_median']:.0f} d")
            for ax in axes:
                if isinstance(r["sowing_median"], str):
                    ax.axvline(pd.Timestamp(r["sowing_median"]), color="#444444", ls="--", lw=1)
                if isinstance(r["harvest_from"], str):
                    ax.axvspan(pd.Timestamp(r["harvest_from"]), pd.Timestamp(r["harvest_to"]), color="#cc3333", alpha=0.15)
    fig.suptitle(f"{fid}  ·  {ac:.3f} ac  ·  {px} pixels  ·  radar first, NDVI as support{rule}", x=0.01, ha="left",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    info = {"field_id": fid, "acres": ac, "pixels": px,
            "last_pass": {tr: str(g.loc[(g["track"] == tr), "date"].max().date()) for tr in tracks},
            "last_clear_ndvi": str(o["date"].max().date()) if len(o) else None}
    plt.close(fig)
    return fig, info


def field_chips(aoi: int, field, pad_px: int = 12, cols: int = 8, folder: str = "data/s2_dates_masks"):
    """Every Sentinel-2 date on which this field is clear (its NDVI points), as 5-3-2 chips around the field with its
    outline (user, 19 Nov 2026: all clear chips, as notebook 08 showed), each chip stretched on its own clear pixels
    like notebook 08's chips."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import from_bounds

    from .. import config as cm
    from ..newest_imagery import season_config
    from ..optical_export import band_index
    from . import pixel_2026 as p26

    fid = resolve_field(aoi, field)
    d = out_dir(aoi)
    qc = json.loads((d / "inputs.json").read_text())["qc"]
    f = load_fields(qc)
    shape = f[f["field_id"] == fid]
    t = pd.read_parquet(d / "field_curves.parquet")
    o = t[(t["field_id"] == fid) & (t["source"] == "NDVI")].sort_values("date")
    key = season_config(aoi)["aoi"]["key"]
    root = cm.repo_root() / folder / key
    rows = int(np.ceil(len(o) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.5 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    g, rings = None, []
    for ax, (date, ndvi) in zip(axes.ravel(), zip(o["date"], o["value"])):
        ds_path = root / f"{key}_{date:%Y-%m}_S2_{date:%Y-%m-%d}.tif"
        with rasterio.open(ds_path) as ds:
            if g is None:
                g = shape.to_crs(ds.crs)
                rings = [np.asarray(r.coords).T for poly in g.geometry for part in getattr(poly, "geoms", [poly])
                         for r in [part.exterior, *part.interiors]]
            x0, y0, x1, y1 = g.total_bounds
            r = ds.res[0] * pad_px
            win = from_bounds(x0 - r, y0 - r, x1 + r, y1 + r, ds.transform).round_offsets().round_lengths()
            rgb = np.dstack([ds.read(band_index(ds, b), window=win, boundless=True, fill_value=0).astype("float32")
                             for b in ("B5", "B3", "B2")])
            clear = ds.read(band_index(ds, "clear"), window=win, boundless=True, fill_value=0)
            wt = ds.window_transform(win)
        # per chip, per band 2-98 % of its clear pixels + gamma, as notebook 08's chips (user, 19 Nov: the fixed
        # 0-3000 stretch gave flat brown chips)
        lim = p26.stretch_limits(np.moveaxis(rgb, 2, 0)[None], clear[None])
        rgb = np.dstack([np.clip((rgb[..., k] - lim[k, 0]) / max(lim[k, 1] - lim[k, 0], 1), 0, 1) for k in range(3)])
        rgb = rgb ** p26.GAMMA
        ext = (wt.c, wt.c + wt.a * rgb.shape[1], wt.f + wt.e * rgb.shape[0], wt.f)
        ax.imshow(rgb, extent=ext, interpolation="nearest")
        for ring in rings:                            # plain lines: geopandas' plot redraws the whole figure each call
            ax.plot(*ring, color="yellow", linewidth=1.2)
        ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"{date:%d %b}  NDVI {ndvi:.2f}", fontsize=9)
    fig.suptitle(f"{fid}: every clear Sentinel-2 view (5-3-2), field outline in yellow", x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.9 / (2.5 * rows), wspace=0.05, hspace=0.25)
    plt.close(fig)
    return fig


def date_map(aoi: int, date: str, bands=("B5", "B3", "B2"), tiles: int = 2, folder: str = "data/s2_dates_masks") -> list[Path]:
    """One Sentinel-2 date over all fields, cut into ``tiles`` x ``tiles`` pictures, every field outlined in the
    colour of its status (red = harvested, cyan = standing) and numbered. Why (user, 19 Nov 2026: "look at every
    polygon yourself"): the status of each field is checked by eye on the latest clear image, as he does in QGIS."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import from_bounds

    from .. import config as cm
    from ..newest_imagery import season_config
    from ..optical_export import band_index
    from . import pixel_2026 as p26

    d = out_dir(aoi)
    f = load_fields(json.loads((d / "inputs.json").read_text())["qc"])
    st = pd.read_csv(d / "field_stage.csv").set_index("field_id")["status"] if (d / "field_stage.csv").exists() else {}
    key = season_config(aoi)["aoi"]["key"]
    path = cm.repo_root() / folder / key / f"{key}_{date[:7]}_S2_{date}.tif"
    outs = []
    with rasterio.open(path) as ds:
        g = f.to_crs(ds.crs)
        x0, y0, x1, y1 = g.total_bounds
        pad = 3 * ds.res[0]
        win = from_bounds(x0 - pad, y0 - pad, x1 + pad, y1 + pad, ds.transform).round_offsets().round_lengths()
        img = np.dstack([ds.read(band_index(ds, b), window=win, boundless=True, fill_value=0).astype("float32") for b in bands])
        clear = ds.read(band_index(ds, "clear"), window=win, boundless=True, fill_value=0)
        wt = ds.window_transform(win)
    lim = p26.stretch_limits(np.moveaxis(img, 2, 0)[None], clear[None])
    img = np.dstack([np.clip((img[..., k] - lim[k, 0]) / max(lim[k, 1] - lim[k, 0], 1), 0, 1) for k in range(3)]) ** p26.GAMMA
    ext = (wt.c, wt.c + wt.a * img.shape[1], wt.f + wt.e * img.shape[0], wt.f)
    xs, ys = np.linspace(ext[0], ext[1], tiles + 1), np.linspace(ext[3], ext[2], tiles + 1)
    for r in range(tiles):
        for c in range(tiles):
            fig, ax = plt.subplots(figsize=(13, 13 * abs(ys[r + 1] - ys[r]) / (xs[c + 1] - xs[c])))
            ax.imshow(img, extent=ext, interpolation="nearest")
            for row in g.itertuples():
                col = "#ff2020" if st.get(row.field_id) == "harvested" else "#00e5ff"
                for part in getattr(row.geometry, "geoms", [row.geometry]):
                    ax.plot(*np.asarray(part.exterior.coords).T, color=col, linewidth=1.1)
                pt = row.geometry.representative_point()
                if xs[c] <= pt.x <= xs[c + 1] and ys[r + 1] <= pt.y <= ys[r]:
                    ax.text(pt.x, pt.y, row.field_id.split("_", 2)[-1].replace("p000", ""), color="white", fontsize=7,
                            ha="center", va="center", weight="bold",
                            bbox={"facecolor": "black", "alpha": 0.45, "pad": 0.6, "linewidth": 0})
            ax.set_xlim(xs[c], xs[c + 1]), ax.set_ylim(ys[r + 1], ys[r])
            ax.set_axis_off()
            ax.set_title(f"{date}  {'-'.join(bands)}  tile {r + 1},{c + 1}  red = harvested, cyan = standing", fontsize=10)
            out = d / "date_maps" / f"{date}_{''.join(b[1:] for b in bands)}_r{r + 1}c{c + 1}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=110, bbox_inches="tight")
            plt.close(fig)
            outs.append(out)
    return outs


def s2_clear_dates(aoi: int, folder: str = "data/s2_dates_masks") -> pd.DataFrame:
    """Every local Sentinel-2 file of the AOI with the share of the AOI the delivered series' mask keeps (clear), to
    pick dates to open in QGIS (true colour = bands 3,2,1; 5-3-2 = 4,2,1; 8-3-2 = 5,2,1)."""
    from . import ndvi_5day as nd
    from .. import config as cm
    from ..newest_imagery import season_config
    from .mask_experiment import VARIANTS

    dates, _n, _w, _l, ok, _s, loc = nd.read_dates(aoi, **VARIANTS[VARIANT])
    inside = nd.inside_from_loc(loc)
    o = ok.reshape(len(dates), -1)[:, inside].astype(bool)
    key = season_config(aoi)["aoi"]["key"]
    root = cm.repo_root() / folder / key
    out = pd.DataFrame({"date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"), "clear_pct": (100 * o.mean(axis=1)).round(1)})
    out["file"] = [str(root / f"{key}_{x[:7]}_S2_{x}.tif") for x in out["date"]]
    out.to_csv(out_dir(aoi) / "s2_clear_dates.csv", index=False)
    return out


# ------------------------------------------------------------------------------------------------ the stage rule
#: Age is counted up to this day (user, 19 Nov 2026).
AGE_DAY = "2026-11-15"
#: Days a direct-seeded pixel (no clear water dip) is dated before its radar low. 0 since 19 Nov 2026: the user first
#: gave 15, but he reads the low itself as the planting in those fields too, and without the lead the rule is closer
#: to his readings (14 of 21 fields within 10 days against 10; "ghatao nahi").
DIRECT_SEEDED_LEAD_DAYS = 0


def wet_peak(days, vh, vv) -> float:
    """Day of one track's brightest pass over the AOI: the highest AOI-median of (VH + VV) / 2 over all field pixels.

    Why: before planting the monsoon wets the bare fields and the radar reaches its season high (mid-June here);
    then the fields are flooded and it falls. A planting dip is searched only AFTER that high, so the dry-season lows
    (bare dry soil) and single rain spikes (7 April on one track) cannot be taken for planting, with no calendar date.
    The largest jump was tried first and picked that April rain spike."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = (np.nanmedian(vh, axis=1) + np.nanmedian(vv, axis=1)) / 2
    return float(days[int(np.nanargmax(med))])


def _own_position(x, ref):
    """``x`` on the pixel's own scale: 0 at the lowest and 1 at the highest of ``ref`` (its values in the period)."""
    lo, hi = np.nanmin(ref), np.nanmax(ref)
    return (x - lo) / (hi - lo) if hi > lo else np.full_like(x, np.nan, dtype="float64")


def ndvi_harvest(nd_days, nd_vals):
    """``(green_day, harvest_from, harvest_to)`` of one NDVI curve (a pixel's or a field's), harvest = NaN while the
    crop stands; None when the curve has too few views or no rise. The tests are described in :func:`pixel_stage`.
    Per pixel this is kept for information (``harvested_pct``); the field's status comes from :func:`field_status`."""
    nd_days, nd_vals = np.asarray(nd_days, float), np.asarray(nd_vals, float)
    if len(nd_vals) < 6:
        return None
    ipk = int(np.nanargmax(nd_vals))
    if ipk == 0:
        return None
    iem = int(np.nanargmin(nd_vals[:ipk]))
    empty, peak = float(nd_vals[iem]), float(nd_vals[ipk])
    half = empty + (peak - empty) / 2
    # first green view = start of the LAST rise: back from the peak while the views stay above the half-way level (a
    # single lower view inside is haze; two in a row end it). An earlier green spell (weeds / an earlier crop in May-June)
    # is therefore not this crop's green-up.
    # ... but a single view down near the empty level is the bare / flooded field itself, not haze, and ends the spell
    # (19 Nov 2026, fields 505 / 515 / 524: green cover in May-June, one clear view on 25 July at 0.15-0.25, green again
    # in September; taking that view for haze put the green-up in May and left the field without a planting date).
    quarter = empty + (half - empty) / 2
    ig = ipk
    i = ipk - 1
    while i >= 0 and (nd_vals[i] >= half or (nd_vals[i] >= quarter and i >= 1 and nd_vals[i - 1] >= half)):
        if nd_vals[i] >= half:
            ig = i
        i -= 1
    green_day = float(nd_days[ig])
    v = nd_vals[ig:]
    best = None
    for k in range(1, len(v) - 1):                         # step from view k-1 to k, at least two views from k on
        step = v[k - 1] - v[k]
        later = v[k:]
        if step <= 0 or (later >= (v[k - 1] + v[k]) / 2).any():
            continue
        before = float(np.max(v[:k]))
        if step >= (before - float(np.median(later))) / 2 and float(np.max(later)) < float(np.min(v[:k])):
            if best is None or step > best[0]:
                best = (step, k)
    if best is not None:
        k = best[1]
        return green_day, float(nd_days[ig + k - 1]), float(nd_days[ig + k])
    return green_day, np.nan, np.nan


#: A field is harvested when its latest level lies more than this share of its own season rise (empty -> peak) below
#: its green canopy. Set between the user's 17 checked fields on the 15 Nov image (19 Nov 2026): standing ones fall
#: at most 0.10 of their rise (459, ripening), cut ones at least 0.18 (qc001 0.18, 130 0.19, 174 0.22, others 0.27+).
CUT_SHARE = 0.15
#: The latest level = median of the last views, so one hazy view (449 on 7 Nov) cannot cut a field.
LAST_VIEWS = 3


def field_status(nd_days, nd_vals) -> dict:
    """Standing or harvested from a FIELD's NDVI curve (median of its interior pixels on clear dates) as of its last
    view (user, 19 Nov 2026: decide on the field, not by a vote of pixels).

    Why not the step test of :func:`ndvi_harvest`: a field that ripens first (NDVI sliding 0.78 -> 0.63 through
    October) and is cut later fails "one step carries half the loss" (108, 113, 129, 106, 435 stayed standing), and a
    field that peaks on its next-to-last views shows tiny dips that look like a cut (458, 449, 450). The user's reading
    is simpler: is the field, now, clearly below its own green canopy?

    * canopy = median of the views from green-up that are above the half-way level between empty and peak (the
      latest views left out), so haze dips and the rising crop do not set it;
    * now = median of the last :data:`LAST_VIEWS` views;
    * harvested when ``canopy - now`` exceeds :data:`CUT_SHARE` of the field's rise (``peak - empty``); the harvest
      lies on the largest single step between the last view above that cut level and the newest view.

    Returns status, harvest_from / harvest_to (day numbers or NaN) and ``drop_share`` = (canopy - now) / rise."""
    out = {"status": "no data", "harvest_from": np.nan, "harvest_to": np.nan, "drop_share": np.nan}
    days, v = np.asarray(nd_days, float), np.asarray(nd_vals, float)
    g = ndvi_harvest(days, v)
    if g is None or len(v) <= LAST_VIEWS:
        return out
    ipk = int(np.argmax(v))
    empty = float(np.min(v[:ipk])) if ipk else float(np.min(v))
    rise = float(v[ipk]) - empty
    body = (days >= g[0]) & (np.arange(len(v)) < len(v) - LAST_VIEWS) & (v >= empty + rise / 2)
    canopy = float(np.median(v[body])) if body.any() else float(v[ipk])
    # ... but the newest view counts by itself too: a field cut between the last two views (203: 0.65 on 14 Nov,
    # 0.43 on 15 Nov, blue stubble on the 15 Nov image) is cut. The newest clear image is the reference the user
    # checks the status on (19 Nov 2026: "15 Nov par dekho, wo saaf hai").
    # Only an abrupt last step counts (it alone exceeds CUT_SHARE of the rise): a slow slide to the same level is
    # ripening (458_b, the part the user called ripening, slid below the level without any step).
    now = float(np.median(v[-LAST_VIEWS:]))
    if rise > 0 and (v[-2] - v[-1]) / rise > CUT_SHARE:
        now = min(now, float(v[-1]))
    out["drop_share"] = (canopy - now) / rise if rise > 0 else np.nan
    cut = canopy - CUT_SHARE * rise
    if now >= cut:
        out["status"] = "standing"
        return out
    i = len(v) - 1
    while i > 0 and v[i - 1] < cut:
        i -= 1
    # the cut itself = the largest single step from the last view above the cut level to the end, not the view that
    # first slid below it (20 Nov 2026, field 174: ripening took it under the level on 7 November, the real fall,
    # 0.57 -> 0.44, came between 14 and 15 November like its neighbours; the user agreed)
    i0 = max(i - 1, 0)
    steps = v[i0:-1] - v[i0 + 1:]
    k = i0 + int(np.argmax(steps)) + 1 if len(steps) else i
    out.update(status="harvested", harvest_from=float(days[k - 1]), harvest_to=float(days[k]))
    return out


def pixel_stage(nd_days, nd_vals, tracks, wet_from: float = -np.inf) -> dict:
    """Stage of one pixel from its own curves; every test is relative to the pixel's own levels and noise (user, 19 Nov
    2026: no fixed dB or NDVI numbers, only relative change, slopes and dips).

    ``nd_days`` / ``nd_vals``: its clear NDVI views (day numbers, ascending). ``tracks``: per radar track
    ``(days, vh, vv)``.

    Optical decides standing / harvested (radar cannot: stubble and wet soil keep it bright, and VV already falls
    while the crop ripens):

    * the crop = the rise of NDVI from its lowest view (``empty``) to its season peak; the first green view is the
      start of the last unbroken spell above the half-way level between the two that leads to the peak;
    * harvested = one step between two neighbouring views after green-up that (a) carries at least half of the whole
      loss from the level before it to the level after it (a slow fade is ripening, not a cut), (b) is never taken
      back: every later view stays below the middle of the step, on at least two views, and (c) ends below every view
      the pixel showed since it turned green (below its own canopy noise, haze dips included). The harvest lies
      between those two views.

    Radar dates the planting (optical is blind in the cloudy weeks):

    * each track's VH and VV are put on the pixel's own scale of the time before the canopy (so ripening lows do not
      count), both tracks merged in time order, ``w`` = mean of the two polarisations;
    * the search runs from ``wet_from`` (the AOI's own wet-soil radar high before planting, :func:`wet_peak`: the
      dry-season lows before it are dry soil, not planting; no calendar date) to green-up; the water spell = the
      passes there that lie below the half-way level between the lowest ``w`` and the level at green-up; planting =
      the LAST pass of the unbroken low spell that holds the minimum (the fields the user read: flooded through
      July, planted at the end; a single later dip, e.g. one rainy pass in September, is not part of it);
    * it is transplanting when the dip is deeper than twice the pixel's own pass-to-pass change within a track;
      otherwise direct seeded, dated ``DIRECT_SEEDED_LEAD_DAYS`` before that pass."""
    out = {"status": "no data", "method": "", "sowing_day": np.nan, "harvest_from": np.nan, "harvest_to": np.nan}
    if len(nd_vals) < 6:
        return out
    g = ndvi_harvest(nd_days, nd_vals)
    if g is None:
        return out
    green_day, hf, ht = g
    out["status"] = "standing"
    if np.isfinite(hf):
        out.update(status="harvested", harvest_from=hf, harvest_to=ht)
    # ---- planting from the radar
    p = planting(tracks, green_day, wet_from)
    if p is not None:
        out.update(sowing_day=p[0], method=p[1])
    return out


def planting(tracks, green_day: float, wet_from: float = -np.inf, previous: bool = False, latest: float = np.inf,
             only=None):
    """``(planting_day, method)`` from the radar before green-up (see :func:`pixel_stage`), or None. ``previous``:
    take the first dip instead of the last (for a field already harvested)."""
    d, w, noise = [], [], []
    for days, vh, vv in tracks:
        pre = days <= green_day
        if pre.sum() < 4:
            continue
        pos = (_own_position(vh, vh[pre]) + _own_position(vv, vv[pre])) / 2
        d.append(days[pre])
        w.append(pos[pre])
        noise.append(np.median(np.abs(np.diff(pos[pre]))))
    if not d:
        return None
    d, w = np.concatenate(d), np.concatenate(w)
    o = np.argsort(d)
    d, w = d[o], w[o]
    ok = np.isfinite(w)
    d, w = d[ok], w[ok]
    if len(w) < 5:
        return None
    first = int(np.searchsorted(d, wet_from, side="right"))
    end = float(w[-1])
    # ``latest``: passes after this day are too close to green-up to be the planting (water among the young plants)
    if only is not None:
        # only the area's common flooding passes may be the planting: the lowest of them on this field's curve
        cand = np.flatnonzero(np.isin(d, list(only)) & (d > wet_from) & (d <= latest))
        if not len(cand):
            return None
        return float(d[cand[int(np.argmin(w[cand]))]]), "common flooding pass"
    stop = int(np.searchsorted(d, latest, side="right"))
    if stop - first >= 3:
        d, w = d[:stop], np.append(w[:stop], end)
        d = np.append(d, np.nan)
    seg = w[first:]
    if len(seg) < 3:
        return None
    lo = float(np.min(seg))
    # Low spells = runs of passes below the half-way level; ONE higher pass inside does not end a run (the two tracks
    # alternate and see the field at different angles), two in a row do. Planting = the last pass of the LAST spell
    # with at least two low passes (a single low pass is one rainy / odd pass, e.g. 5 September in field 458). Why the
    # last and not the deepest (19 Nov 2026, fields 144 / 145 / 475): their dips are weak, the deepest was 3 July, but
    # the optical still shows the bare field on 25 July; the user reads end of July, the later spell.
    mid = (lo + end) / 2
    lowi = np.flatnonzero(seg[:-1] <= mid)
    if not len(lowi):                                      # the lowest pass is the green-up pass itself
        lowi = np.array([int(np.argmin(seg[:-1]))])     # never the closing level itself
    spells, cur = [], [int(lowi[0])]
    for x in lowi[1:]:
        if x - cur[-1] <= 2:
            cur.append(int(x))
        else:
            spells.append(cur)
            cur = [int(x)]
    spells.append(cur)
    real = [sp for sp in spells if len(sp) >= 2] or spells
    pick = real[-1]
    # its DEEPEST pass, not its end (user, 19 Nov 2026, field 473 and the fields dated 12 August or later: the water
    # is deepest on 31 July; the low passes after it, 8-20 August, are water standing among the young plants)
    k = pick[int(np.argmin(seg[pick]))]
    if previous:
        # a field already cut was planted on the EARLIER dip (user, 19 Nov 2026: in harvested fields the late dip,
        # mostly 31 July, is water over a young crop; their planting is the dip before it, the first week of July):
        # * two or more low spells: the end of the spell before the last (a spell of 25 June - 3 July ends on 3 July;
        #   its first low point, 25 June, was too early for the user);
        # * one long spell (148 / 150: the two tracks alternate, one higher pass at a time, so the 3 and 31 July dips
        #   merge): its first low point = the first pass after which the level rises again.
        if len(spells) >= 2:
            pick = spells[-2]
            k = pick[-1]
        else:
            # (the spell's own first pass is the way down from the wet high, not the bottom of a dip: 106 / 116
            # started their spell on 25 June, the user reads the first week of July)
            pick = spells[0]
            k = next((x for x in pick[1:] if x + 1 < len(seg) and seg[x + 1] > seg[x]), pick[-1])
    last = first + k
    lo_sp = float(np.min(seg[pick]))
    depth = min(float(np.max(seg[:pick[0] + 1])), end) - lo_sp
    if depth > 2 * float(np.nanmean(noise)):
        return float(d[last]), "transplanted (water dip)"
    return float(d[last]) - DIRECT_SEEDED_LEAD_DAYS, "direct seeded (rise)"




def stage(aoi: int, qc_file=None, run: str | None = None, age_day: str = AGE_DAY) -> Path:
    """:func:`pixel_stage` for every pixel of every QC'd field; writes ``pixel_stage.parquet`` (one row per pixel) and
    ``field_stage.csv`` (per field, from its interior pixels: status by majority, share harvested, median sowing date
    and its spread, harvest window)."""
    from . import ndvi_5day as nd
    from . import radar_water as rw
    from .. import config as cm
    from ..newest_imagery import season_config
    from .mask_experiment import VARIANTS

    d = out_dir(aoi)
    inp = json.loads((d / "inputs.json").read_text())
    qc_file, run = qc_file or inp["qc"], run or inp["run"]
    cfg = season_config(aoi)
    gd = json.loads((cm.grid_dir(cfg) / "grid_def.json").read_text())
    pix, _ = field_pixels(qc_file, gd)
    t0 = pd.Timestamp("2026-01-01")
    series = []
    for (dates, flat), tr in zip(rw.read_series(aoi, run_id=run), [t["track_id"] for t in cfg["s1"]["tracks"]]):
        # every pass is kept: rain passes (12 Nov: +2.3 dB on all fields) cannot mislead the rule, because the harvest
        # is read from the optical and the planting only from passes before green-up; dropping "passes with rain" would
        # drop half the monsoon (19 Nov 2026: 13 of 23 passes of one track had rain in the 6 h before)
        dd = pd.DatetimeIndex(dates).normalize()
        series.append(((dd - t0).days.to_numpy().astype(float), flat["VH"], flat["VV"]))
    allp = np.concatenate([p for p in pix.values() if len(p)])
    wet_from = min(wet_peak(days, vh[:, allp], vv[:, allp]) for days, vh, vv in series)
    dates, ndvi, _w, _l, ok, _s, _loc = nd.read_dates(aoi, **VARIANTS[VARIANT])
    nday = (pd.DatetimeIndex(dates) - t0).days.to_numpy().astype(float)
    n = ndvi.reshape(len(nday), -1)
    o = ok.reshape(len(nday), -1).astype(bool)
    age0 = (pd.Timestamp(age_day) - t0).days
    rows = []
    for fid, pp in pix.items():
        for p in pp:
            m = o[:, p] & np.isfinite(n[:, p])
            tracks = []
            for days, vh, vv in series:
                g = np.isfinite(vh[:, p]) & np.isfinite(vv[:, p])
                tracks.append((days[g], vh[g, p], vv[g, p]))
            r = pixel_stage(nday[m], n[m, p], tracks, wet_from)
            rows.append({"field_id": fid, "pixel": int(p), "interior": int(p) in _INTERIOR, **r})
    t = pd.DataFrame(rows)
    day = lambda x: t0 + pd.to_timedelta(x, unit="D")  # noqa: E731
    t["sowing_date"] = day(t["sowing_day"])
    t["harvest_from"] = day(t["harvest_from"])
    t["harvest_to"] = day(t["harvest_to"])
    end = np.where(t["status"] == "harvested", (t["harvest_from"] + (t["harvest_to"] - t["harvest_from"]) / 2 - t0).dt.days,
                   age0)
    t["age_days"] = end - t["sowing_day"]
    t.drop(columns="sowing_day").to_parquet(d / "pixel_stage.parquet", index=False)

    # standing / harvested is decided on the FIELD's own NDVI curve (user, 19 Nov 2026: "pixel par na jao, field level
    # par hi karo"): median of its interior pixels (all pixels when it has none) on the dates where enough of them are
    # clear, the same curve the notebook plots; one harvest test on that curve instead of a vote of noisy pixels
    # (field 107: three of six interior pixels said cut, the tie kept it standing)
    field_cut = {}
    for fid, pp in pix.items():
        if not len(pp):
            continue
        inner = [p for p in pp if int(p) in _INTERIOR] or list(pp)
        share = o[:, inner].mean(axis=1)
        keep = np.flatnonzero(share >= MIN_CLEAR_SHARE)
        vals = np.array([np.nanmedian(np.where(o[i, inner], n[i, inner], np.nan)) for i in keep])
        good = np.isfinite(vals)
        field_cut[fid] = field_status(nday[keep][good], vals[good])
        if field_cut[fid]["status"] in ("harvested", "standing"):
            # planting from the FIELD's own radar curve too (user, 19 Nov 2026: field level, not pixels; field 473:
            # its pixels spread over 8-20 August, the field curve is deepest on 31 July)
            gg = ndvi_harvest(nday[keep][good], vals[good])
            ftr = []
            for days, vh, vv in series:
                mvh, mvv = np.nanmedian(vh[:, inner], axis=1), np.nanmedian(vv[:, inner], axis=1)
                ok_ = np.isfinite(mvh) & np.isfinite(mvv)
                ftr.append((days[ok_], mvh[ok_], mvv[ok_]))
            field_cut[fid]["tracks"], field_cut[fid]["green"] = ftr, gg[0] if gg else None
            field_cut[fid]["planting"] = (planting(ftr, gg[0], wet_from, previous=field_cut[fid]["status"] == "harvested")
                                          if gg else None)
    # A dip too close to green-up is not the planting (user, 19 Nov 2026: 473, 541 and the fields dated 12 August or
    # later belong on their July dip). "Too close" comes from the AOI itself: the lower quartile of its own fields'
    # planting -> green-up gaps; a standing field below it is dated again without the passes inside that gap.
    gaps = np.array([c["green"] - c["planting"][0] for c in field_cut.values()
                     if c.get("planting") and c["status"] == "standing" and c["green"] - c["planting"][0] > 0])
    if len(gaps) >= 8:
        min_gap = float(np.quantile(gaps, 0.25))
        print(f"planting -> green-up gap of standing fields: quartiles {np.quantile(gaps, [0.1, 0.25, 0.5, 0.75]).round(0)}, "
              f"minimum used {min_gap:.0f} d", flush=True)
        for c in field_cut.values():
            if c.get("planting") and c["green"] - c["planting"][0] < min_gap:
                again = planting(c["tracks"], c["green"], wet_from, previous=c["status"] == "harvested",
                                 latest=c["green"] - min_gap)
                c["planting"] = again or c["planting"]
    # A field alone on a rare date is dated on one of the area's COMMON flooding passes (user, 20 Nov 2026: a
    # corrected field stands for its group; 284 was on 12 August with one other field, he reads 31 July, the pass
    # most fields flood on). Common = chosen by at least as many standing fields as the average chosen pass.
    days = pd.Series([c["planting"][0] for c in field_cut.values() if c.get("planting") and c["status"] == "standing"])
    if len(days) >= 8:
        cnt = days.value_counts()
        common = set(cnt[cnt >= cnt.mean()].index)
        print("common planting passes:", sorted((t0 + pd.Timedelta(days=x)).strftime("%d %b") for x in common),
              "| rare:", {(t0 + pd.Timedelta(days=x)).strftime("%d %b"): int(n) for x, n in cnt.items() if x not in common},
              flush=True)
        for c in field_cut.values():
            if c.get("planting") and c["status"] == "standing" and c["planting"][0] not in common:
                again = planting(c["tracks"], c["green"], wet_from, only=common)
                if again and again[0] in common:
                    c["planting"] = again
    # A harvested field cannot have grown much shorter than the other harvested fields (user, 20 Nov 2026: 447 and 452,
    # cut on 5 November with a 31 July dip = 97 days, were sown in the first week of July like the rest, ~125 days).
    # Shorter than the lower decile of the harvested fields' own durations -> dated again on the harvested fields'
    # common passes that leave at least that duration.
    cutf = [c for c in field_cut.values() if c.get("planting") and c["status"] == "harvested"]
    if len(cutf) >= 8:
        dur = np.array([c["harvest_to"] - c["planting"][0] for c in cutf])
        # the lower DECILE: with the quartile the fields cut a week later (14 November, 106 days from their 31 July
        # dip) were pulled to 3 July as well, a 134-day crop nobody read
        need = float(np.quantile(dur, 0.10))
        cnt = pd.Series([c["planting"][0] for c in cutf]).value_counts()
        common_h = set(cnt[cnt >= cnt.mean()].index)
        print(f"harvested fields: duration quartiles {np.quantile(dur, [0.1, 0.25, 0.5, 0.75]).round(0)}, common passes "
              f"{sorted((t0 + pd.Timedelta(days=x)).strftime('%d %b') for x in common_h)}", flush=True)
        for c in cutf:
            if c["harvest_to"] - c["planting"][0] < need:
                ok_days = {x for x in common_h if c["harvest_to"] - x >= need}
                again = planting(c["tracks"], c["green"], wet_from, only=ok_days) if ok_days else None
                if again and again[0] in ok_days:
                    c["planting"] = again

    def one(g):
        # the field's label comes from its interior pixels (user, 19 Nov 2026: edge pixels of these small fields mix
        # with the neighbour, so they are not counted); a field too small to have any uses all its pixels
        fid = g.name
        n_all = len(g)
        g = g[g["interior"]] if g["interior"].any() else g
        s = g["sowing_date"].dropna()
        h = g[g["status"] == "harvested"]
        fc = field_cut.get(fid) or {"status": "no data", "drop_share": np.nan}
        cut = fc["status"] == "harvested"
        hf = (t0 + pd.Timedelta(days=fc["harvest_from"])).date() if cut else None
        ht = (t0 + pd.Timedelta(days=fc["harvest_to"])).date() if cut else None
        sow = s.quantile(0.5, interpolation="nearest") if len(s) else None
        if fc.get("planting"):
            sow = t0 + pd.Timedelta(days=fc["planting"][0])
        end = (pd.Timestamp(hf) + (pd.Timestamp(ht) - pd.Timestamp(hf)) / 2) if cut else pd.Timestamp(age_day)
        return pd.Series({
            "pixels": n_all, "interior_pixels": int(g["interior"].sum()),
            "status": "harvested" if cut else "standing",
            "harvested_pct": round(100 * len(h) / len(g), 0),
            # how far below its canopy the field is now, as a share of its rise; near CUT_SHARE = check by eye
            "drop_share": round(float(fc["drop_share"]), 3),
            "sowing_median": sow.date() if sow is not None else None,
            "sowing_q25": s.quantile(0.25, interpolation="nearest").date() if len(s) else None,
            "sowing_q75": s.quantile(0.75, interpolation="nearest").date() if len(s) else None,
            # no pixel with a planting date = no empty-then-green cycle after the wet peak: green all through (fields
            # 517 / 518 / 521, which the user found not to be rice) -> flagged for review, not guessed
            "note": ("" if len(s) else "no planting signal: check, may not be rice")
            or ("review: close to the cut level" if abs(fc["drop_share"] - CUT_SHARE) <= CUT_SHARE * 0.4 else ""),
            "transplanted_pct": round(100 * g["method"].str.startswith("transplanted").mean(), 0),
            "age_median": float((end - sow).days) if sow is not None else np.nan,
            "harvest_from": hf, "harvest_to": ht})

    f = t.groupby("field_id").apply(one, include_groups=False).reset_index()
    f["source"] = "rule"
    ov = d / "overrides.csv"
    if ov.exists():
        # the user's own decisions for single fields (``field`` = last digits of the id, ``sowing_date``, ``status``,
        # ``why``); they win over the rule and are marked, so nobody takes them for a measurement
        for r in pd.read_csv(ov, dtype=str).fillna("").itertuples():
            m = f["field_id"].str.endswith(r.field)
            if r.sowing_date:
                f.loc[m, ["sowing_median", "sowing_q25", "sowing_q75"]] = pd.Timestamp(r.sowing_date).date()
            if r.status:
                f.loc[m, "status"] = r.status
                if r.status == "standing":
                    f.loc[m, ["harvest_from", "harvest_to"]] = None
            end = age0
            if f.loc[m, "status"].iloc[0] == "harvested" and f.loc[m, "harvest_from"].isna().all():
                # a field the user calls harvested without a date takes the window most rule-harvested fields
                # share (user, 19 Nov 2026: "jo median harvest date hai wo laga do")
                h = f[(f["source"] == "rule") & (f["status"] == "harvested")]
                if len(h):
                    f.loc[m, "harvest_from"] = h["harvest_from"].mode().iloc[0]
                    f.loc[m, "harvest_to"] = h["harvest_to"].mode().iloc[0]
            if f.loc[m, "status"].iloc[0] == "harvested" and f.loc[m, "harvest_from"].notna().all():
                a_, b_ = pd.Timestamp(f.loc[m, "harvest_from"].iloc[0]), pd.Timestamp(f.loc[m, "harvest_to"].iloc[0])
                end = (a_ + (b_ - a_) / 2 - t0).days
            if r.sowing_date:
                f.loc[m, "age_median"] = end - (pd.Timestamp(r.sowing_date) - t0).days
            # a reading of the curves and the image by the assistant is named as such, not as the user's decision
            f.loc[m, "source"] = "assistant reading" if r.why.startswith("assistant") else "user decision"
            f.loc[m, "note"] = r.why
    f.to_csv(d / "field_stage.csv", index=False)
    return d / "field_stage.csv"


def score(aoi: int, readings_csv) -> pd.DataFrame:
    """The rule's field planting dates beside the user's own reading of the curves (``field``, ``read_date``: the
    middle of what he said, e.g. "end of July" = 28 July) with the difference in days. Why: his readings are the only
    reference for planting, so every rule change is checked against all of them at once."""
    r = pd.read_csv(readings_csv, dtype={"field": str})
    f = pd.read_csv(out_dir(aoi) / "field_stage.csv")
    f["field"] = f["field_id"].str[-3:]
    t = r.merge(f[["field", "sowing_median", "transplanted_pct", "interior_pixels", "status"]], on="field", how="left")
    t["diff_days"] = (pd.to_datetime(t["sowing_median"]) - pd.to_datetime(t["read_date"])).dt.days
    return t


STAGE_COLUMNS = ("status", "sowing_date", "sow_week", "crop_age", "harvest_date", "harv_week")


#: The week columns count an event one day before the image that shows it (user, 20 Nov 2026): a field flooded on
#: the pass of 15 July was planted in the days up to then, so it belongs to 8-14 July, not to the week starting on
#: the 15th; a field first seen cut on 15 November was cut by the 14th. The date columns keep the image dates.
WEEK_BACK_DAYS = 1


def month_week(day, back_days: int = 0) -> str | None:
    """``"2026-07-29 to 07-31 (Jul week 5)"``: the week of the month a date falls in, written with its first and last
    day. Week 1 = days 1-7, 2 = 8-14, 3 = 15-21, 4 = 22-28, 5 = 29 to the month's end.

    Why (user, 19-20 Nov 2026): the dates come from passes 5-12 days apart, so a week is the honest precision; the
    label starts with the ISO date so that QGIS, which sorts the classes of a categorised style as text, lists the
    weeks in their true order (a label starting with the month name put August before July)."""
    if day is None or pd.isna(day):
        return None
    d = pd.Timestamp(day) - pd.Timedelta(days=back_days)
    k = (d.day - 1) // 7
    a = d.replace(day=7 * k + 1)
    b = d.replace(day=min(7 * k + 7, d.days_in_month))
    return f"{a:%Y-%m-%d} to {b:%m-%d} ({d:%b} week {k + 1})"


#: Columns of the QC layer the delivery leaves out (user, 20 Nov 2026).
DROP_COLUMNS = ("sub_class", "origin", "review")


def deliver(aoi: int, name: str | None = None, drop=DROP_COLUMNS) -> Path:
    """The QC'd field layer with the stage columns added, everything else exactly as it was (user, 19 Nov 2026).

    * ``status``: ``standing`` or ``harvested``;
    * ``sowing_date``: planting date (every field);
    * ``crop_age``: days from planting to :data:`AGE_DAY` for a standing field, to its harvest date for a harvested
      one (user, 20 Nov 2026);
    * ``harvest_date``: harvested fields only, the first clear image that shows the field cut (it was cut between the
      clear image before it and this one);
    * ``sow_week`` / ``harv_week``: the week of the month the event falls in, counted :data:`WEEK_BACK_DAYS` before
      the image date (:func:`month_week`).

    The QC file is copied and the columns are added inside the GeoPackage (SQLite ``ALTER TABLE`` / ``UPDATE``), so
    every geometry and every existing attribute stays byte for byte the same; rows are matched by their order in the
    file (a drawn polygon has no ``field_id`` of its own)."""
    import shutil
    import sqlite3

    d = out_dir(aoi)
    qc = Path(json.loads((d / "inputs.json").read_text())["qc"])
    fs = pd.read_csv(d / "field_stage.csv").set_index("field_id")
    ids = load_fields(qc)["field_id"].tolist()
    out = d / "delivery" / (name or f"{qc.stem}_crop_stage_{AGE_DAY}.gpkg")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(qc, out)
    con = sqlite3.connect(out)

    def _never(*_):
        raise RuntimeError("a geometry trigger ran: the delivery must not touch geometries")

    # the GeoPackage's spatial-index triggers name these functions; SQLite needs them to exist to compile an UPDATE,
    # but they only run when a geometry changes, which must never happen here (they raise if it does)
    for fn in ("ST_IsEmpty", "ST_MinX", "ST_MaxX", "ST_MinY", "ST_MaxY"):
        con.create_function(fn, 1, _never)
    try:
        table = con.execute("SELECT table_name FROM gpkg_contents WHERE data_type = 'features'").fetchone()[0]
        pk = next(r[1] for r in con.execute(f'PRAGMA table_info("{table}")') if r[5])
        rowids = [r[0] for r in con.execute(f'SELECT "{pk}" FROM "{table}" ORDER BY "{pk}"')]
        for c, typ in zip(STAGE_COLUMNS, ("TEXT", "DATE", "TEXT", "INTEGER", "DATE", "TEXT")):
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{c}" {typ}')
        for rid, fid in zip(rowids, ids):
            r = fs.loc[fid]
            standing = r["status"] == "standing"
            # from the delivered date itself, so date and age always agree (the pixel-age median can differ by a pass)
            hd = None if standing or pd.isna(r["harvest_to"]) else str(r["harvest_to"])
            age = (pd.Timestamp(AGE_DAY if standing else hd) - pd.Timestamp(r["sowing_median"])).days \
                if standing or hd else None
            con.execute(f'UPDATE "{table}" SET status = ?, sowing_date = ?, sow_week = ?, crop_age = ?, harvest_date = ?, '
                        f'harv_week = ? WHERE "{pk}" = ?',
                        (r["status"], str(r["sowing_median"]), month_week(r["sowing_median"], WEEK_BACK_DAYS), age, hd,
                         month_week(hd, WEEK_BACK_DAYS), rid))
        have = {r[1] for r in con.execute(f'PRAGMA table_info("{table}")')}
        for c in drop:
            if c in have:
                con.execute(f'ALTER TABLE "{table}" DROP COLUMN "{c}"')
        con.commit()
    finally:
        con.close()
    shapefile(out)
    return out


def shapefile(gpkg) -> Path:
    """The delivery layer once more as a shapefile in EPSG:4326 (``shp/<same name>.shp`` beside the GeoPackage), for
    readers who want that format (user, 19 Nov 2026). Shapefile column names stop at 10 letters, so ``sowing_date``
    and ``harvest_date`` are written as ``sowing_dt`` and ``harvest_dt``; the GeoPackage stays the exact-geometry
    copy, this one is reprojected."""
    import geopandas as gpd

    gpkg = Path(gpkg)
    g = gpd.read_file(gpkg).to_crs(4326).rename(columns={"sowing_date": "sowing_dt", "harvest_date": "harvest_dt",
                                                         "major_class": "major_cls", "label_share": "label_shr"})
    for c in ("sowing_dt", "harvest_dt"):
        g[c] = pd.to_datetime(g[c]).dt.strftime("%Y-%m-%d")
    g["crop_age"] = g["crop_age"].astype("Int64")
    out = gpkg.parent / "shp" / f"{gpkg.stem}.shp"
    out.parent.mkdir(parents=True, exist_ok=True)
    g.to_file(out, driver="ESRI Shapefile")
    # its parts in one zip beside the GeoPackage, the file people download (user, 19 Nov 2026)
    import zipfile

    with zipfile.ZipFile(gpkg.parent / f"{gpkg.stem}_shp.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for part in sorted(out.parent.glob(f"{out.stem}.*")):
            z.write(part, part.name)
    return out


def prepare_fields(src, out=None, name: str | None = None) -> tuple[Path, dict]:
    """A hand-edited field layer made ready for the stage work: no overlaps, a 0.2 m gap where neighbours meet, and
    one unique id per polygon.

    Why (user, 19 Nov 2026): he split the polygons that held two or more fields, so several polygons share one
    ``field_id``, and the layer has polygons lying on top of each other. The cleaning is the project's QC rule
    (``qc_post.clean``): the larger delineated polygon keeps its shape, the other loses the shared part; a polygon
    with no neighbour and no overlap keeps its geometry object untouched; small cut pieces get ``review``.

    Ids: ``field_id`` becomes unique (the first polygon of an id keeps it, the next ones get ``_b``, ``_c`` ...; a
    polygon without an id becomes ``<name>_qc001`` ...), the id it came from stays in ``source_id``, and ``fid_no`` is
    a plain running number 1..n."""
    import geopandas as gpd

    from .qc_post import ACRE_M2, clean

    src = Path(src)
    g = gpd.read_file(src)
    ids = g["field_id"].astype("object")
    drawn = ids.isna() | ids.astype(str).isin(["", "None", "nan"])
    stem = name or (str(ids[~drawn].iloc[0]).rsplit("_", 1)[0] if (~drawn).any() else src.stem)
    ids[drawn] = [f"{stem}_qc{i + 1:03d}" for i in range(int(drawn.sum()))]
    g["field_id"] = ids.astype(str)
    g["drawn"] = drawn.to_numpy()
    g["origin"] = g["origin"].fillna("drawn in QC")
    # each input polygon gets its unique id BEFORE the cleaning, so a piece cut from it can be traced
    # the largest polygon of an id keeps the id (274: its first row was a sliver left by the splitting)
    g = g.assign(a_=g.area).sort_values("a_", ascending=False, kind="stable")
    k = g.groupby("field_id").cumcount()
    g = g.assign(k_=k).sort_index().drop(columns="a_")
    k = g.pop("k_")
    g["source_id"] = g["field_id"]
    g["field_id"] = [f if n == 0 else f"{f}_{chr(ord('a') + n)}" for f, n in zip(g["field_id"], k)]
    src_of = dict(zip(g["field_id"], g["source_id"]))
    f = clean(g)
    f["source_id"] = f["field_id"].map(src_of)
    f["field_id"] = [i if p == 1 else f"{i}_p{p}" for i, p in zip(f["field_id"], f["part"])]
    f = f.drop(columns=["part", "predicted"]).reset_index(drop=True)
    f.insert(0, "fid_no", range(1, len(f) + 1))
    out = Path(out) if out else src.with_name(f"{src.stem}_clean.gpkg")
    out.parent.mkdir(parents=True, exist_ok=True)
    f.to_file(out, driver="GPKG", layer=out.stem)
    sj = gpd.sjoin(f, f, predicate="intersects")
    sj = sj[sj.index < sj.index_right]
    over = sum(f.geometry[a].intersection(f.geometry[b]).area > 0 for a, b in zip(sj.index, sj.index_right))
    rep = {"polygons_in": len(g), "polygons_out": len(f), "acres_in": round(float(g.area.sum()) / ACRE_M2, 3),
           "acres_in_no_double": round(float(g.union_all().area) / ACRE_M2, 3),
           "acres_out": round(float(f.area.sum()) / ACRE_M2, 3), "unique_ids": int(f["field_id"].nunique()),
           "overlapping_pairs": int(over), "touching_pairs": int(len(sj)) - int(over),
           "invalid": int((~f.is_valid).sum()), "small_clipped": int((f["review"] == "small_clipped").sum())}
    return out, rep


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["curves", "sheet", "stage", "score", "deliver", "prepare"])
    p.add_argument("--src", help="prepare: the hand-edited field layer")
    p.add_argument("--out", help="prepare: the cleaned layer to write")
    p.add_argument("--readings", help="score: csv with field, read_date")
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--qc", help="curves: the QC'd field layer")
    p.add_argument("--run", help="curves: the radar run to read")
    p.add_argument("--fields", nargs="*", help="sheet: field ids (default: n fields spread over sizes)")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="sheet_1")
    a = p.parse_args(argv)
    if a.step == "prepare":
        out, rep = prepare_fields(a.src, a.out)
        print(out)
        print(json.dumps(rep, indent=1))
    elif a.step == "deliver":
        print(deliver(a.aoi))
    elif a.step == "score":
        t = score(a.aoi, a.readings)
        print(t.to_string(index=False))
        d = t["diff_days"].dropna().abs()
        print(f"{len(d)} fields with a date: within 10 days {int((d <= 10).sum())}, within 16 days {int((d <= 16).sum())}, "
              f"median |diff| {d.median():.0f} d; no date {int(t['diff_days'].isna().sum())}")
    elif a.step == "stage":
        print(stage(a.aoi, a.qc, a.run))
    elif a.step == "curves":
        print(curves(a.aoi, a.qc, a.run))
    else:
        print(sheet(a.aoi, a.fields, a.n, a.seed, name=a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
