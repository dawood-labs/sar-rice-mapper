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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["curves", "sheet"])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--qc", help="curves: the QC'd field layer")
    p.add_argument("--run", help="curves: the radar run to read")
    p.add_argument("--fields", nargs="*", help="sheet: field ids (default: n fields spread over sizes)")
    p.add_argument("--n", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="sheet_1")
    a = p.parse_args(argv)
    if a.step == "curves":
        print(curves(a.aoi, a.qc, a.run))
    else:
        print(sheet(a.aoi, a.fields, a.n, a.seed, name=a.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
