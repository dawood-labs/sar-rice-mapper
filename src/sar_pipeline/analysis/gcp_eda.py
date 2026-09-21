"""EDA of the ground-truth fields: per-field time series, class curves and a review list.

Per field and date: mean linear power over the field's PURE pixels (at least `border_m` inside the edge; the
whole field when it is too small) converted to dB, plus the within-field VH spread.

Review flags, per track and class (classes with at least MIN_FIELDS fields):
- far from class:  robust z-score (median/MAD) of the RMSD to the class median curve > Z_MAX
- different shape: correlation with the class median curve < MIN_CORR
- heterogeneous:   robust z-score of the within-field VH spread > Z_MAX (maybe two fields in one polygon)
Classes listed in analysis.eda.flat_classes skip the shape test (a flat median curve has no shape);
classes in analysis.eda.mixed_classes skip far/shape (a mixed class has no single typical curve) and instead
get a note when the field is dark (water-like) or very bright (built-up) all season.
Flags are suggestions for a human check, never automatic deletions.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from shapely.geometry import box

from .. import manifest, resources, run_meta
from . import gcp_qc
from .pixel_features import OK_STATES, analysis_dir, read_chunk_array, run_context

log = logging.getLogger(__name__)
MIN_FIELDS, Z_MAX, MIN_CORR, MIN_PIXELS = 10, 3.5, 0.3, 20
WATER_VH_DB, WATER_VV_DB, BUILTUP_VV_DB = -24.0, -17.0, -3.0


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / mad if mad > 0 else np.zeros_like(x, dtype=float)


def _task(t):
    """Per field part in this chunk: per-band pixel count and sums of linear power, dB and dB^2."""
    track_id, paths, chunk, gd, fields = t
    arr = read_chunk_array(paths, chunk, gd)
    tf = rasterio.transform.from_origin(chunk["xmin"], chunk["ymax"], gd["res"], gd["res"])
    shape = (chunk["height"], chunk["width"])
    out = []
    for gid, wkb in fields:
        m = rasterize([(shapely.from_wkb(wkb), 1)], out_shape=shape, transform=tf, fill=0, dtype="uint8").astype(bool)
        if not m.any():
            continue
        px = arr[:, m]
        ok = np.isfinite(px)
        db = np.where(ok, px, 0.0)
        out.append((track_id, gid, np.array([ok.sum(1), np.where(ok, 10 ** (db / 10), 0.0).sum(1), db.sum(1), (db**2).sum(1)])))
    return out


def field_timeseries(cfg: dict, run_dir: Path) -> pd.DataFrame:
    t0 = time.time()
    a = cfg.get("analysis") or {}
    _, gd, tracks, *_ = run_context(run_dir)
    border_m = float(a.get("border_m", 10))
    g = gcp_qc.load_gcps(cfg).to_crs(gd["crs"])
    boxes = {c["name"]: box(c["xmin"], c["ymin"], c["xmax"], c["ymax"]) for c in gd["chunks"]}
    covered = shapely.union_all(list(boxes.values()))
    g = g[g.intersection(covered).area / g.area > 0.999].copy()
    inner = g.buffer(-border_m)
    g["pure"] = inner.area >= 3 * gd["res"] ** 2
    use = [i if p else geom for i, p, geom in zip(inner, g.pure, g.geometry)]
    names = list(boxes); tree = shapely.STRtree(list(boxes.values()))
    by_chunk: dict[str, list] = {}
    for gid, geom in zip(g.gcp_id, use):
        for i in tree.query(geom, predicate="intersects"):
            by_chunk.setdefault(names[i], []).append((int(gid), shapely.to_wkb(geom)))
    man = manifest.read_manifest(run_dir)
    chunks = {c["name"]: c for c in gd["chunks"]}
    tasks = []
    for tr in tracks:
        for cname, fl in by_chunk.items():
            row = man[(man.track_id == tr) & (man.chunk_name == cname) & man.state.isin(OK_STATES)]
            if not row.empty:
                tasks.append((tr, [Path(run_dir) / p for p in row.local_paths.iloc[0].split(";") if p], chunks[cname], gd, fl))
    rs = resources.detect_resources(cfg)
    workers = resources.cpu_workers(rs, cfg, mem_per_worker_bytes=gd["chunk_px"] ** 2 * 60 * 4 * 3)
    sums: dict[tuple, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(_task, t) for t in tasks]):
            for tr, gid, s in fut.result():  # a field split over chunks adds up
                sums[(tr, gid)] = s if (tr, gid) not in sums else sums[(tr, gid)] + s
    rows = []
    label = g.set_index("gcp_id").label
    pure = g.set_index("gcp_id").pure
    for tr in tracks:
        layout = run_meta.read_band_layout(run_dir, tr)
        vv = layout[layout.pol == "VV"].set_index("acquisition_id").band_idx - 1
        vh = layout[layout.pol == "VH"].set_index("acquisition_id").band_idx - 1
        dates = pd.read_csv(Path(run_dir) / f"stack/track_{tr}/dates.csv").set_index("acquisition_id")
        for (t, gid), (n, lin, sdb, sdb2) in sums.items():
            if t != tr:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                db = 10 * np.log10(lin / n)
                std = np.sqrt(np.maximum(sdb2 / n - (sdb / n) ** 2, 0))
            for acq in vh.index:
                i_vv, i_vh = vv[acq], vh[acq]
                rows.append(dict(gcp_id=gid, label=label[gid], track=tr, date=dates.date_utc[acq],
                                 rain_24h_mm=dates.rain_24h_mm[acq], n_px=int(n[i_vh]), pure=bool(pure[gid]),
                                 VV_db=db[i_vv], VH_db=db[i_vh], VH_std_db=std[i_vh]))
    ts = pd.DataFrame(rows)
    ts["VH_minus_VV_db"] = ts.VH_db - ts.VV_db
    log.info("time series: %d fields, %d rows (%.0f s)", ts.gcp_id.nunique(), len(ts), time.time() - t0)
    return ts


def outlier_flags(ts: pd.DataFrame, flat_classes=(), mixed_classes=()) -> pd.DataFrame:
    frames = []
    for (tr, lab), sub in ts.groupby(["track", "label"]):
        M = sub.pivot(index="gcp_id", columns="date", values="VH_db")
        if len(M) < MIN_FIELDS:
            continue
        med = M.median(axis=0)
        rmsd = np.sqrt(((M - med) ** 2).mean(axis=1))
        corr = M.apply(lambda r: np.corrcoef(r.values, med.values)[0, 1], axis=1)
        het = sub.groupby("gcp_id").VH_std_db.median().reindex(M.index)
        df = pd.DataFrame({"track": tr, "label": lab, "rmsd_db": rmsd, "z_rmsd": robust_z(rmsd.values),
                           "corr_with_class": corr, "within_std_db": het, "z_within_std": robust_z(het.values),
                           "n_px": sub.groupby("gcp_id").n_px.median().reindex(M.index)})
        mixed, flat = lab in mixed_classes, lab in flat_classes
        df["flag_far_from_class"] = (df.z_rmsd > Z_MAX) & (not mixed)
        df["flag_different_shape"] = (df.corr_with_class < MIN_CORR) & (not mixed) & (not flat)
        df["flag_heterogeneous"] = df.z_within_std > Z_MAX
        df["flag_small"] = df.n_px < MIN_PIXELS
        frames.append(df.reset_index())
    return pd.concat(frames, ignore_index=True).round(3) if frames else pd.DataFrame()


def review_list(ts: pd.DataFrame, flags: pd.DataFrame, mixed_classes=()) -> pd.DataFrame:
    """gcp_id, label, qc_status, qc_reason for every field that needs a look (ok fields are not listed)."""
    rows = []
    means = ts.groupby("gcp_id").agg(VH=("VH_db", "mean"), VV=("VV_db", "mean"), label=("label", "first"))
    by_field = flags.groupby("gcp_id") if len(flags) else {}
    for gid, m in means.iterrows():
        f = by_field.get_group(gid) if len(flags) and gid in by_field.groups else None
        if f is not None and (f.flag_far_from_class | f.flag_different_shape).any():
            rows.append((gid, m.label, "check_label", "time series unlike its class: check the label on optical imagery"))
        elif f is not None and f.flag_heterogeneous.any():
            rows.append((gid, m.label, "mixed_pixels", "mixed pixels inside the polygon (maybe two fields)"))
        elif m.label in mixed_classes and m.VH < WATER_VH_DB and m.VV < WATER_VV_DB:
            rows.append((gid, m.label, "note_water", "very low VV and VH all season: water-like surface (kept)"))
        elif m.label in mixed_classes and m.VV > BUILTUP_VV_DB:
            rows.append((gid, m.label, "note_builtup", "very bright VV all season: built-up or metal roof (kept)"))
    return pd.DataFrame(rows, columns=["gcp_id", "label", "qc_status", "qc_reason"])


def class_curves_png(ts: pd.DataFrame, path: Path, codes: dict[str, int]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .maps import PALETTE

    order = sorted(codes, key=codes.get)
    color = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(order)}
    tracks = sorted(ts.track.unique())
    fig, axs = plt.subplots(3, len(tracks), figsize=(7.5 * len(tracks), 12), sharex="col", squeeze=False)
    for j, tr in enumerate(tracks):
        for i, var in enumerate(["VH_db", "VV_db", "VH_minus_VV_db"]):
            ax = axs[i, j]
            for lab, sub in ts[ts.track == tr].groupby("label"):
                q = sub.groupby("date")[var].quantile([0.25, 0.5, 0.75]).unstack()
                t = pd.to_datetime(q.index)
                ax.plot(t, q[0.5], color=color.get(lab), lw=2, label=f"{lab} (n={sub.gcp_id.nunique()})")
                ax.fill_between(t, q[0.25], q[0.75], color=color.get(lab), alpha=0.15)
            ax.set_title(f"{tr}  {var}", loc="left"); ax.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Class median (line) and 25-75 % range (band), pure pixels")
    fig.savefig(path, dpi=85, bbox_inches="tight"); plt.close(fig)


def run(cfg: dict, run_dir: Path) -> Path:
    eda = (cfg.get("analysis") or {}).get("eda") or {}
    flat, mixed = tuple(eda.get("flat_classes") or ()), tuple(eda.get("mixed_classes") or ())
    out_dir = analysis_dir(run_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ts = field_timeseries(cfg, run_dir)
    run_meta.atomic_write_csv(ts, out_dir / "gcp_timeseries.csv")
    flags = outlier_flags(ts, flat, mixed)
    run_meta.atomic_write_csv(flags, out_dir / "gcp_outlier_flags.csv")
    review = review_list(ts, flags, mixed)
    run_meta.atomic_write_csv(review, out_dir / "gcp_review.csv")
    class_curves_png(ts, out_dir / "class_curves.png", gcp_qc.codes_from_config(cfg))
    log.info("review list: %s -> %s", review.qc_status.value_counts().to_dict(), out_dir / "gcp_review.csv")
    return out_dir / "gcp_review.csv"
