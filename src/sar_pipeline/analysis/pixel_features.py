"""Per-pixel time-series features of labelled fields, read straight from a run's chunk files.

For every pixel of every analysed field and every selected track, each *feature set* gives
VH, VV and VH-VV per time bin:

- bins: calendar half-months (1-15, 16-end) or fixed N-day bins anchored at the set's start date;
- start: first date used (None = season start); the end is the day after the last acquisition;
- window: odd window size in pixels; values are the mean linear power over the window (nodata ignored),
  which damps the speckle that remains after filtering;
- rain rule: inside a bin, acquisitions with >= `rain_mm_24h` in the previous 24 h are dropped when the bin
  also has a dry acquisition (rain raises backscatter without any crop change);
- empty bins (no acquisition) are linearly interpolated in time; their number is reported per set and track.

Many sets are computed in ONE pass: identical bins share one computation, and only the field pixels are kept.
Rows with any non-finite value are dropped explicitly (count logged and saved in the metadata), then at most
`max_pixels_per_field` pixels per field are kept with a fixed seed, so every set is evaluated on identical pixels.

Column names: `<track_id>__<set name>__<VH|VV|VHmVV>__<bin start MMDD>`.
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio.features import rasterize
from shapely.geometry import box

from .. import config as config_mod
from .. import grid as grid_mod
from .. import manifest, resources, run_meta
from . import gcp_qc

log = logging.getLogger(__name__)
SEP = "__"
OK_STATES = ("VERIFIED", "VERIFIED_EMPTY")


@dataclass(frozen=True)
class FeatureSet:
    kind: str                  # "half_month" or "days"
    days: int = 15             # bin length for kind "days"
    window: int = 5            # odd window size in pixels
    start: str | None = None   # "YYYY-MM-DD" or None (season start)

    def __post_init__(self):
        if self.kind not in ("half_month", "days"):
            raise ValueError(f"bins kind must be half_month or days, got {self.kind!r}")
        if self.window < 1 or self.window % 2 == 0:
            raise ValueError(f"window must be a positive odd number of pixels, got {self.window}")
        if self.kind == "half_month" and self.start and pd.Timestamp(self.start).day not in (1, 16):
            raise ValueError("half-month bins need a start on the 1st or the 16th of a month")

    @property
    def name(self) -> str:
        bins = "hm" if self.kind == "half_month" else f"d{self.days}"
        start = pd.Timestamp(self.start).strftime("%m%d") if self.start else "season"
        return f"{bins}_w{self.window}_s{start}"


def set_from_config(cfg: dict) -> FeatureSet:
    a = cfg.get("analysis") or {}
    b = a.get("bins") or {}
    return FeatureSet(kind=b.get("kind", "half_month"), days=int(b.get("days", 15)),
                      window=int(a.get("window_px", 5)), start=b.get("start"))


def analysis_dir(run_dir: Path) -> Path:
    """processed/<aoi>/<season>/analysis/<run_id>/"""
    run_dir = Path(run_dir)
    return run_dir.parents[1] / "analysis" / run_dir.name


# ---------------------------------------------------------------- time bins
def make_bins(fs: FeatureSet, season_start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """[(bin start, bin end exclusive)] from the set's start (default season start) up to `end` (exclusive)."""
    start = pd.Timestamp(fs.start) if fs.start else season_start
    out = []
    if fs.kind == "half_month":
        m = pd.Timestamp(year=start.year, month=start.month, day=1)
        while m < end:
            mid, nxt = m + pd.Timedelta(days=15), m + pd.offsets.MonthBegin(1)
            for s, e in ((m, mid), (mid, nxt)):
                if start <= s < end:
                    out.append((s, min(e, end)))
            m = nxt
        return out
    s = start
    while s < end:
        out.append((s, min(s + pd.Timedelta(days=fs.days), end)))
        s = s + pd.Timedelta(days=fs.days)
    return out


def plan_bins(layout: pd.DataFrame, wet: dict[str, bool], bins, origin: pd.Timestamp) -> list[tuple]:
    """[(bin start, centre day, vv band indexes | None, vh band indexes | None)] for one track (0-based bands)."""
    bl = layout.copy()
    bl["t"] = pd.to_datetime(bl["date_utc"])
    bl["wet"] = bl["acquisition_id"].map(wet).fillna(False).astype(bool)
    out = []
    for s, e in bins:
        sub = bl[(bl.t >= s) & (bl.t < e)]
        centre = ((s + (e - s) / 2) - origin) / pd.Timedelta(days=1)
        if sub.empty:
            out.append((s, centre, None, None))
            continue
        dry = sub[~sub.wet]
        sub = dry if len(dry) else sub
        out.append((s, centre, tuple(sub[sub.pol == "VV"].band_idx - 1), tuple(sub[sub.pol == "VH"].band_idx - 1)))
    return out


def fill_linear(series: list, centres: list[float]) -> list:
    """Replace None entries by linear interpolation in time; before the first / after the last value, repeat it."""
    have = [i for i, a in enumerate(series) if a is not None]
    if not have:
        raise ValueError("every bin is empty")
    out = []
    for i, a in enumerate(series):
        if a is not None:
            out.append(a)
            continue
        prev = max((j for j in have if j < i), default=None)
        nxt = min((j for j in have if j > i), default=None)
        if prev is None or nxt is None:
            out.append(series[nxt if prev is None else prev])
        else:
            w = (centres[i] - centres[prev]) / (centres[nxt] - centres[prev])
            out.append((1 - w) * series[prev] + w * series[nxt])
    return out


def box_mean(img: np.ndarray, w: int) -> np.ndarray:
    """Mean of the finite values in a w x w window (edges use the available pixels)."""
    if w == 1:
        return img
    r = w // 2
    val = np.where(np.isfinite(img), img, 0.0)
    ok = np.isfinite(img).astype("float64")

    def summed(a):
        c = np.pad(a, ((r + 1, r), (r + 1, r))).cumsum(0).cumsum(1)
        return c[w:, w:] - c[:-w, w:] - c[w:, :-w] + c[:-w, :-w]

    with np.errstate(invalid="ignore", divide="ignore"):
        return (summed(val) / summed(ok)).astype("float32")


def wet_flags(run_dir: Path, track_id: str, threshold_mm: float) -> dict[str, bool]:
    d = pd.read_csv(Path(run_dir) / f"stack/track_{track_id}/dates.csv")
    return dict(zip(d.acquisition_id, d.rain_24h_mm.fillna(0) >= threshold_mm))


def group_id(x: float, y: float, cell_m: float) -> str:
    """Spatial CV group: the cell_m x cell_m grid cell of a field centroid."""
    return f"{int(x // cell_m)}_{int(y // cell_m)}"


def drop_non_finite(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, int]:
    vals = df[cols].to_numpy(dtype="float64")
    bad = ~np.isfinite(vals).all(axis=1)
    return df.loc[~bad].copy(), int(bad.sum())


def cap_per_field(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most n pixels per field, chosen reproducibly (sort by pixel id, shuffle with seed, take the first n)."""
    return (df.sort_values("pid").reset_index(drop=True).sample(frac=1.0, random_state=seed)
              .groupby("gcp_id").head(n).reset_index(drop=True))


def columns_for_set(columns, set_name: str) -> list[str]:
    return [c for c in columns if c.count(SEP) == 3 and c.split(SEP)[1] == set_name]


def sets_in(columns) -> list[str]:
    return sorted({c.split(SEP)[1] for c in columns if c.count(SEP) == 3})


def read_chunk_array(paths: list[Path], chunk: dict, gd: dict) -> np.ndarray:
    """(bands, height, width) float32 array of one chunk, nodata as NaN; Earth Engine part files pasted in place."""
    arr = None
    for p in paths:
        with rasterio.open(p) as ds:
            a = ds.read().astype("float32")
            a[a == ds.nodata] = np.nan
            r0 = int(round((gd["y0"] - ds.transform.f) / gd["res"])) - chunk["row_off"]
            c0 = int(round((ds.transform.c - gd["x0"]) / gd["res"])) - chunk["col_off"]
        if arr is None:
            arr = np.full((a.shape[0], chunk["height"], chunk["width"]), np.nan, dtype="float32")
        arr[:, r0:r0 + a.shape[1], c0:c0 + a.shape[2]] = a
    return arr


def track_features(lin: np.ndarray, rr, cc, track_id: str, plans: dict, cache: dict | None = None) -> dict[str, np.ndarray]:
    """{column: vector at (rr, cc)} for every set in `plans` = {set name: (window, planned bins)}."""
    cache = {} if cache is None else cache
    feats = {}
    for set_name, (window, bins) in plans.items():
        vh_s, vv_s, centres = [], [], []
        for _s, centre, vv_i, vh_i in bins:
            centres.append(centre)
            if vv_i is None:
                vh_s.append(None); vv_s.append(None)
                continue
            key = (vv_i, vh_i, window)
            if key not in cache:
                with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # all-nodata pixels (outside the AOI) -> NaN
                    vv = 10 * np.log10(box_mean(np.nanmean(lin[list(vv_i)], axis=0), window))[rr, cc]
                    vh = 10 * np.log10(box_mean(np.nanmean(lin[list(vh_i)], axis=0), window))[rr, cc]
                cache[key] = (vh.astype("float32"), vv.astype("float32"))
            vh_s.append(cache[key][0]); vv_s.append(cache[key][1])
        vh_s, vv_s = fill_linear(vh_s, centres), fill_linear(vv_s, centres)
        for (s, *_), vh, vv in zip(bins, vh_s, vv_s):
            tag = s.strftime("%m%d")
            feats[f"{track_id}{SEP}{set_name}{SEP}VH{SEP}{tag}"] = vh
            feats[f"{track_id}{SEP}{set_name}{SEP}VV{SEP}{tag}"] = vv
            with np.errstate(invalid="ignore"):  # nodata pixels stay NaN
                feats[f"{track_id}{SEP}{set_name}{SEP}VHmVV{SEP}{tag}"] = vh - vv
    return feats


def _chunk_task(task):
    track_id, paths, chunk, gd, fields, plans, interior_only = task
    arr = read_chunk_array(paths, chunk, gd)
    res = gd["res"]
    tf = rasterio.transform.from_origin(chunk["xmin"], chunk["ymax"], res, res)
    shape = (chunk["height"], chunk["width"])
    rr_l, cc_l, gid_l, border_l = [], [], [], []
    for gid, wkb_full, wkb_inner in fields:
        full = rasterize([(shapely.from_wkb(wkb_full), 1)], out_shape=shape, transform=tf, fill=0, dtype="uint8").astype(bool)
        if not full.any():
            continue
        inner = (rasterize([(shapely.from_wkb(wkb_inner), 1)], out_shape=shape, transform=tf, fill=0, dtype="uint8").astype(bool)
                 if wkb_inner else np.zeros(shape, dtype=bool))
        use = inner if (interior_only and inner.any()) else full
        rr, cc = np.nonzero(use)
        rr_l.append(rr); cc_l.append(cc); gid_l.append(np.full(len(rr), gid)); border_l.append(~inner[rr, cc])
    if not rr_l:
        return None
    rr, cc = np.concatenate(rr_l), np.concatenate(cc_l)
    feats = track_features(10 ** (arr / 10), rr, cc, track_id, plans)
    pid = (chunk["row_off"] + rr).astype(np.int64) * gd["width"] + (chunk["col_off"] + cc)
    return track_id, np.concatenate(gid_l), pid, np.concatenate(border_l), feats


def run_context(run_dir: Path) -> tuple[dict, dict, list[str], pd.Timestamp, pd.Timestamp]:
    """(run config, grid, track ids, season start, end = day after the last acquisition)."""
    run_cfg = config_mod.load_run_config(run_dir)
    gd = grid_mod.load_grid(run_cfg)
    tracks = [t["track_id"] for t in run_cfg["s1"]["tracks"]]
    layout = run_meta.read_band_layout(run_dir)
    end = pd.to_datetime(layout["date_utc"]).max() + pd.Timedelta(days=1)
    return run_cfg, gd, tracks, pd.Timestamp(run_cfg["season"]["start"]), end


def plans_for(run_dir: Path, track_id: str, sets: list[FeatureSet], wet: dict, season_start, end) -> tuple[dict, list[dict]]:
    layout = run_meta.read_band_layout(run_dir, track_id)
    plans, report = {}, []
    for fs in sets:
        bins = plan_bins(layout, wet, make_bins(fs, season_start, end), season_start)
        plans[fs.name] = (fs.window, bins)
        report.append({"set": fs.name, "track": track_id, "n_bins": len(bins), "interpolated_bins": sum(b[2] is None for b in bins)})
    return plans, report


def extract(cfg: dict, run_dir: Path, sets: list[FeatureSet], interior_only: bool = False, name: str = "features") -> Path:
    t0 = time.time()
    a = cfg.get("analysis") or {}
    run_cfg, gd, tracks, season_start, end = run_context(run_dir)
    border_m, cell = float(a.get("border_m", 10)), float(a.get("group_cell_m", 5000))
    fields = gcp_qc.load_fields(cfg, gd["crs"])
    inner = fields.buffer(-border_m)
    boxes = {c["name"]: box(c["xmin"], c["ymin"], c["xmax"], c["ymax"]) for c in gd["chunks"]}
    names = list(boxes)
    tree = shapely.STRtree(list(boxes.values()))
    by_chunk: dict[str, list] = {}
    for r, inn in zip(fields.itertuples(), inner):
        wkb_inner = shapely.to_wkb(inn) if (inn is not None and not inn.is_empty) else b""
        for i in tree.query(r.geometry, predicate="intersects"):
            by_chunk.setdefault(names[i], []).append((int(r.gcp_id), shapely.to_wkb(r.geometry), wkb_inner))
    man = manifest.read_manifest(run_dir)
    chunks = {c["name"]: c for c in gd["chunks"]}
    tasks, bins_report, missing = [], [], []
    for tr in tracks:
        plans, rep = plans_for(run_dir, tr, sets, wet_flags(run_dir, tr, float(a.get("rain_mm_24h", 5))), season_start, end)
        bins_report += rep
        for cname, fl in by_chunk.items():
            row = man[(man.track_id == tr) & (man.chunk_name == cname) & man.state.isin(OK_STATES)]
            if row.empty:
                missing.append(f"{tr}:{cname}")
                continue
            paths = [Path(run_dir) / p for p in row.local_paths.iloc[0].split(";") if p]
            tasks.append((tr, paths, chunks[cname], gd, fl, plans, interior_only))
    if missing:
        log.warning("%d chunk files with fields are not verified and were skipped: %s", len(missing), missing[:10])
    rs = resources.detect_resources(cfg)
    workers = resources.cpu_workers(rs, cfg, mem_per_worker_bytes=gd["chunk_px"] ** 2 * 60 * 4 * 5)
    log.info("%d fields, %d chunk files, %d processes", len(fields), len(tasks), workers)
    frames = {tr: [] for tr in tracks}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, fut in enumerate(as_completed([pool.submit(_chunk_task, t) for t in tasks]), 1):
            out = fut.result()
            if out:
                tr, gids, pid, border, feats = out
                d = pd.DataFrame(feats); d["pid"] = pid; d["gcp_id"] = gids; d["border"] = border
                frames[tr].append(d)
            if k % 50 == 0 or k == len(tasks):
                log.info("  %d/%d chunk files (%.0f s)", k, len(tasks), time.time() - t0)
    X = None
    for tr in tracks:
        d = pd.concat(frames[tr]).sort_values(["pid", "gcp_id"]).drop_duplicates("pid")  # overlapping fields: keep one
        X = d if X is None else X.merge(d.drop(columns="border"), on=["pid", "gcp_id"], how="inner")
    label = fields.set_index("gcp_id")["label"]
    cen = fields.set_index("gcp_id").geometry.centroid
    X["label"] = X.gcp_id.map(label).astype(str)
    X["group"] = X.gcp_id.map(pd.Series([group_id(p.x, p.y, cell) for p in cen], index=cen.index)).astype(str)
    feat_cols = [c for c in X.columns if c.count(SEP) == 3]
    X, dropped = drop_non_finite(X, feat_cols)
    if dropped:
        log.warning("dropped %d pixel rows with non-finite feature values", dropped)
    X = cap_per_field(X, int(a.get("max_pixels_per_field", 200)), int(a.get("seed", 0)))
    out_dir = analysis_dir(run_dir); out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    tmp = out_dir / f".{name}.tmp.parquet"
    X.to_parquet(tmp, index=False)
    run_meta.replace_with_retry(tmp, path)
    meta = {"sets": {fs.name: asdict(fs) for fs in sets}, "interior_only": interior_only, "bins": bins_report,
            "dropped_non_finite_rows": dropped, "skipped_chunk_files": missing, "pixels_per_label": X.label.value_counts().to_dict(),
            "fields": int(X.gcp_id.nunique()), "season_start": str(season_start.date()), "end_exclusive": str(end.date())}
    run_meta.atomic_write_json(out_dir / f"{name}.json", meta)
    log.info("features: %d pixels, %d fields, %d columns -> %s (%.0f s)", len(X), meta["fields"], len(feat_cols), path, time.time() - t0)
    return path
