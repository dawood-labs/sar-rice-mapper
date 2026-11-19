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


def field_pixels(qc_file, gd: dict) -> tuple[dict, pd.DataFrame]:
    """{field_id: flat pixel ids} of pixels whose centre is inside the field (all its pieces), and a table with
    each field's acres (EPSG:32646) and pixel count."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    f = gpd.read_file(qc_file)
    f = f[f["field_id"].notna()].copy() if "field_id" in f else f
    f["field_id"] = f["field_id"].astype(str)
    f = f.to_crs(gd["crs"])
    ids = sorted(f["field_id"].unique())
    code = {k: i + 1 for i, k in enumerate(ids)}
    w, h = int(gd["width"]), int(gd["height"])
    lab = rasterize(((g, code[k]) for g, k in zip(f.geometry, f["field_id"])), out_shape=(h, w),
                    transform=from_origin(float(gd["x0"]), float(gd["y0"]), float(gd["res"]), float(gd["res"])),
                    fill=0, dtype="int32").ravel()
    pix = {k: np.flatnonzero(lab == code[k]) for k in ids}
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
        ax.set_ylabel(f"{pol} (dB)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
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
                    f"transplanted {r['transplanted_pct']:.0f} %, harvested {r['harvested_pct']:.0f} % of pixels"
                    + (f" between {r['harvest_from']} and {r['harvest_to']}" if r["harvested_pct"] > 0 else "")
                    + f", age {r['age_median']:.0f} d")
            for ax in axes:
                if isinstance(r["sowing_median"], str):
                    ax.axvline(pd.Timestamp(r["sowing_median"]), color="#444444", ls="--", lw=1)
                if r["harvested_pct"] > 0 and isinstance(r["harvest_from"], str):
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
    f = gpd.read_file(qc)
    shape = f[f["field_id"].astype(str) == fid]
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
#: A direct-seeded pixel (no water dip before its rise) was sown this many days before the radar rise (user, 19 Nov).
DIRECT_SEEDED_LEAD_DAYS = 15


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
      its LAST pass (the fields the user read: flooded through July, planted at the end);
    * it is transplanting when the dip is deeper than twice the pixel's own pass-to-pass change within a track;
      otherwise direct seeded: sown ``DIRECT_SEEDED_LEAD_DAYS`` before that pass (the start of its rise)."""
    out = {"status": "no data", "method": "", "sowing_day": np.nan, "harvest_from": np.nan, "harvest_to": np.nan}
    if len(nd_vals) < 6:
        return out
    ipk = int(np.nanargmax(nd_vals))
    if ipk == 0:
        return out
    iem = int(np.nanargmin(nd_vals[:ipk]))
    empty, peak = float(nd_vals[iem]), float(nd_vals[ipk])
    half = empty + (peak - empty) / 2
    # first green view = start of the LAST rise: back from the peak while the views stay above the half-way level (a
    # single lower view inside is haze; two in a row end it). An earlier green spell (weeds / an earlier crop in May-June)
    # is therefore not this crop's green-up.
    ig = ipk
    i = ipk - 1
    while i >= 0 and (nd_vals[i] >= half or (i >= 1 and nd_vals[i - 1] >= half)):
        if nd_vals[i] >= half:
            ig = i
        i -= 1
    green_day = float(nd_days[ig])
    out["status"] = "standing"
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
        out.update(status="harvested", harvest_from=float(nd_days[ig + k - 1]), harvest_to=float(nd_days[ig + k]))
    # ---- planting from the radar
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
        return out
    d, w = np.concatenate(d), np.concatenate(w)
    o = np.argsort(d)
    d, w = d[o], w[o]
    ok = np.isfinite(w)
    d, w = d[ok], w[ok]
    if len(w) < 5:
        return out
    first = int(np.searchsorted(d, wet_from, side="right"))
    seg = w[first:]
    if len(seg) < 3:
        return out
    end = float(w[-1])
    lo = float(np.min(seg))
    low = np.flatnonzero(seg <= (lo + end) / 2)
    last = first + int(low[-1])
    depth = min(float(np.max(seg[:int(np.argmin(seg)) + 1])), end) - lo
    if depth > 2 * float(np.nanmean(noise)):
        out.update(method="transplanted (water dip)", sowing_day=float(d[last]))
    else:
        out.update(method="direct seeded (rise)", sowing_day=float(d[last]) - DIRECT_SEEDED_LEAD_DAYS)
    return out


def stage(aoi: int, qc_file=None, run: str | None = None, age_day: str = AGE_DAY) -> Path:
    """:func:`pixel_stage` for every pixel of every QC'd field; writes ``pixel_stage.parquet`` (one row per pixel) and
    ``field_stage.csv`` (per field: pixels, share harvested, median sowing date and its spread, harvest window)."""
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
            rows.append({"field_id": fid, "pixel": int(p), **r})
    t = pd.DataFrame(rows)
    day = lambda x: t0 + pd.to_timedelta(x, unit="D")  # noqa: E731
    t["sowing_date"] = day(t["sowing_day"])
    t["harvest_from"] = day(t["harvest_from"])
    t["harvest_to"] = day(t["harvest_to"])
    end = np.where(t["status"] == "harvested", (t["harvest_from"] + (t["harvest_to"] - t["harvest_from"]) / 2 - t0).dt.days,
                   age0)
    t["age_days"] = end - t["sowing_day"]
    t.drop(columns="sowing_day").to_parquet(d / "pixel_stage.parquet", index=False)

    def one(g):
        s = g["sowing_date"].dropna()
        h = g[g["status"] == "harvested"]
        return pd.Series({
            "pixels": len(g), "harvested_pct": round(100 * len(h) / len(g), 0),
            "sowing_median": s.quantile(0.5, interpolation="nearest").date() if len(s) else None,
            "sowing_q25": s.quantile(0.25, interpolation="nearest").date() if len(s) else None,
            "sowing_q75": s.quantile(0.75, interpolation="nearest").date() if len(s) else None,
            "transplanted_pct": round(100 * g["method"].str.startswith("transplanted").mean(), 0),
            "age_median": g["age_days"].median(),
            "harvest_from": h["harvest_from"].min().date() if len(h) else None,
            "harvest_to": h["harvest_to"].max().date() if len(h) else None})

    f = t.groupby("field_id").apply(one, include_groups=False).reset_index()
    f.to_csv(d / "field_stage.csv", index=False)
    return d / "field_stage.csv"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["curves", "sheet", "stage"])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--qc", help="curves: the QC'd field layer")
    p.add_argument("--run", help="curves: the radar run to read")
    p.add_argument("--fields", nargs="*", help="sheet: field ids (default: n fields spread over sizes)")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="sheet_1")
    a = p.parse_args(argv)
    if a.step == "stage":
        print(stage(a.aoi, a.qc, a.run))
    elif a.step == "curves":
        print(curves(a.aoi, a.qc, a.run))
    else:
        print(sheet(a.aoi, a.fields, a.n, a.seed, name=a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
