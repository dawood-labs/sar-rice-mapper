"""Is there newer Sentinel-1 or Sentinel-2 imagery than the inputs we already hold? (read-only)

Why this exists
---------------
A map is only as recent as its newest radar pass and its newest clear optical view. Before anyone
starts Earth Engine exports to bring an AOI up to date, the cheap question comes first: *has the
satellite actually delivered anything new over this AOI since our last export?* This module answers
it from metadata only: it lists Sentinel-1 GRD slices (``audit.list_slices``) and Sentinel-2 scenes
(``optical_export.scene_dates``) and compares them with what is already on disk. Nothing is exported,
downloaded or written.

What "new" means
----------------
* **Sentinel-1**: per configured track (``s1.tracks`` of the AOI's config), the passes dated after the
  last pass of the radar run the analysis reads now (``config.analysis_run_dir``). Only configured
  tracks count: a pass of another track would not be exported by the pipeline anyway.
* **Sentinel-2**: acquisition dates after the newest per-date file already exported to the project's
  ``s2_dates_masks`` folder (local mirror and GCS listing; the later of the two). For each new date
  the check also reports whether Cloud Score+ exists yet for every scene of that day (a date exported
  before its score exists gets a ``clear`` band of 0, issue 32) and the share of the AOI that Cloud
  Score+ calls clear (``cs_cdf >= 0.6``, the threshold ``optical_export.best_date`` uses), so a reader
  sees at once how many of the new dates are worth having.

After the new inputs are built, two read-only checks show how much the OLD part of the season moved
(``compare-radar``: a new radar run against the old one, pass by pass; ``compare-series``: a new 5-day series
against the old one, window by window).

Use::

    python -m sar_pipeline.newest_imagery check --aois 13 28 39 [--end 2026-10-03]
    python -m sar_pipeline.newest_imagery compare-radar  --aois 13 --old v001_20260923 --new v002_20261002
    python -m sar_pipeline.newest_imagery inputs --aois 72 [--run v002_20261002] [--series-root <series folder>]
    python -m sar_pipeline.newest_imagery compare-series --aois 13 --old processed/_batch/<old series> \
        --new processed/_batch/<new series>
    python -m sar_pipeline.newest_imagery describe --aois 13 --dates 2026-10-03   # after the download: no-data /
        # clear / series-mask share, median NDVI, quick-look PNG (processed/_batch/s2_2026/qgis_review/inspect/)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as config_mod

#: Share of AOI pixels Cloud Score+ must call clear (``cs_cdf >= CLEAR_CDF``) for a date to count as
#: "mostly clear" in the report. Why 50 %: below that the date mostly adds cloud-masked pixels; the
#: number is for the report only and never selects or drops a date (every date is exported).
MOSTLY_CLEAR_PCT = 50.0
#: Cloud Score+ cut used for the AOI clear share, the same as ``optical_export.best_date``.
CLEAR_CDF = 0.6


def season_config(aoi_id: int, season_key: str = "monsoon2026") -> dict:
    """The AOI's (gitignored) config for the season, resolved against the repository."""
    return config_mod.load_config(config_mod.repo_root() / "config" / f"aoi{aoi_id}_{season_key}.yaml")


def run_last_dates(run_path) -> dict[str, pd.Timestamp]:
    """{track: date of its last pass} in a radar run's stack (``stack/track_*/dates.csv``)."""
    out = {}
    for d in sorted(Path(run_path, "stack").glob("track_*")):
        dates = pd.to_datetime(pd.read_csv(d / "dates.csv")["date_utc"])
        out[d.name.removeprefix("track_")] = dates.max().normalize()
    return out


def passes_after(slices: pd.DataFrame, after: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Acquisitions (track, UTC date) in ``slices`` later than the track's date in ``after``.

    ``slices`` has ``track_id``, ``datetime_utc``, ``platform`` (one row per GRD slice, as
    ``audit.list_slices`` returns). Tracks missing from ``after`` are ignored: only the configured
    tracks are exported. Kept free of Earth Engine so it can be tested on a small table."""
    cols = ["track_id", "date_utc", "platforms", "n_slices"]
    if slices is None or len(slices) == 0:
        return pd.DataFrame(columns=cols)
    s = pd.DataFrame(slices).copy()
    s["date_utc"] = pd.to_datetime(s["datetime_utc"]).dt.tz_localize(None).dt.normalize()
    s = s[s["track_id"].isin(list(after))]
    s = s[s["date_utc"] > s["track_id"].map(after)]
    if s.empty:
        return pd.DataFrame(columns=cols)
    g = s.groupby(["track_id", "date_utc"]).agg(platforms=("platform", lambda x: ";".join(sorted(set(map(str, x))))),
                                               n_slices=("platform", "size")).reset_index()
    return g[cols].sort_values(["track_id", "date_utc"]).reset_index(drop=True)


def s2_newest_local(aoi_key: str, folder: str = "s2_dates_masks") -> pd.Timestamp | None:
    """Newest date among the per-date Sentinel-2 files in the local mirror ``data/<folder>/<aoi>/``."""
    files = sorted((config_mod.repo_root() / "data" / folder / aoi_key).glob("*_S2_*.tif"))
    if not files:
        return None
    return max(pd.Timestamp(p.stem.rsplit("_S2_", 1)[1]) for p in files)


def s2_newest_gcs(cfg: dict, folder: str = "s2_dates_masks") -> pd.Timestamp | None:
    """Newest date among the per-date Sentinel-2 files on GCS (a listing; nothing is downloaded)."""
    from google.cloud import storage

    from . import auth

    prefix = f"{cfg['gcs']['base_folder']}/{folder}/{cfg['aoi']['key']}/"
    client = storage.Client(credentials=auth.credentials(cfg), project=cfg["auth"]["project"])
    dates = [pd.Timestamp(b.name.rsplit("_S2_", 1)[1][:10]) for b in client.list_blobs(cfg["gcs"]["bucket"], prefix=prefix)
             if b.name.endswith(".tif") and "_S2_" in b.name]
    return max(dates) if dates else None


def aoi_clear_pct(polygon_4326, dates, threshold: float = CLEAR_CDF) -> dict[str, float | None]:
    """Per date: percent of the AOI polygon that Cloud Score+ calls clear on that day's mosaic (no data = not clear).
    None when the day has no Cloud Score+ image at all. Metadata-sized request, nothing exported."""
    import ee

    from .optical_export import CLOUD_SCORE_COLLECTION

    region = ee.Geometry(polygon_4326)
    out = {}
    for d in dates:
        day = ee.Date(str(d)[:10])
        cs = ee.ImageCollection(CLOUD_SCORE_COLLECTION).filterBounds(region).filterDate(day, day.advance(1, "day"))
        if cs.size().getInfo() == 0:
            out[str(d)[:10]] = None
            continue
        frac = cs.select("cs_cdf").mosaic().gte(threshold).unmask(0) \
            .reduceRegion(ee.Reducer.mean(), region, 30, maxPixels=1e10).get("cs_cdf").getInfo()
        out[str(d)[:10]] = round(100 * float(frac or 0), 1)
    return out


def check_aoi(aoi_id: int, end: str, season_key: str = "monsoon2026") -> dict:
    """Everything newer than our inputs for one AOI: ``{"s1": DataFrame, "s2": DataFrame, ...}``. Read-only."""
    import geopandas as gpd

    from . import audit
    from . import optical_export as oe

    cfg = season_config(aoi_id, season_key)
    run = config_mod.analysis_run_dir(cfg)
    last = run_last_dates(run)
    start = (min(last.values()) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    s1 = passes_after(audit.list_slices(cfg, start=start, end=end), last)

    newest_local = s2_newest_local(cfg["aoi"]["key"])
    newest_gcs = s2_newest_gcs(cfg)
    newest = max(d for d in (newest_local, newest_gcs) if d is not None)
    polygon = oe.aoi_geojson_2d(gpd.read_file(config_mod.aoi_path(cfg)))
    s2_start = (newest + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    clouds = oe.scene_dates(polygon, s2_start, end)
    s2 = pd.DataFrame({"date": sorted(clouds)})
    if len(s2):
        cover = oe.cloud_score_coverage(polygon, s2_start, end)
        s2["scene_cloud_pct_min"] = [round(clouds[d], 1) for d in s2["date"]]
        s2["s2_images"] = [cover.get(d, {}).get("s2", 0) for d in s2["date"]]
        s2["cloud_score_images"] = [cover.get(d, {}).get("cloud_score", 0) for d in s2["date"]]
        s2["cloud_score_complete"] = s2["s2_images"] == s2["cloud_score_images"]
        clear = aoi_clear_pct(polygon, s2["date"])
        s2["aoi_clear_pct"] = [clear.get(d) for d in s2["date"]]
        s2["mostly_clear"] = s2["aoi_clear_pct"].fillna(-1) >= MOSTLY_CLEAR_PCT
    return {"aoi": cfg["aoi"]["key"], "radar_run": run.name, "s1_last": {k: str(v.date()) for k, v in last.items()},
            "s1_new": s1, "s2_newest_local": str(newest_local.date()) if newest_local is not None else None,
            "s2_newest_gcs": str(newest_gcs.date()) if newest_gcs is not None else None, "s2_new": s2}


# ---------------------------------------------------------------------------------------------
# After the update: how much did the old part of the inputs move?
#
# Why: a new radar run is a full re-export of the season (the multi-temporal speckle filter makes every
# pass depend on its three neighbours on each side, so the newest passes cannot be computed alone), and a
# new 5-day series is a new global smoothing fit. Both should leave the older part of the season (almost)
# where it was; these two checks show with numbers where it moved, before anyone switches an AOI.
# ---------------------------------------------------------------------------------------------

def _stack_dates(ds) -> list[str]:
    return [d.rsplit("_", 1)[1] for d in ds.descriptions]


def diff_stats(a, b, tol: float) -> dict:
    """Over pixels valid in both arrays: the median signed difference (``b - a``), the median and 99th-percentile
    absolute difference, and the percent of those pixels differing by more than ``tol``."""
    import numpy as np

    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    both = np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return {"pixels": 0, "median_diff": None, "median_abs": None, "p99_abs": None, "pct_over_tol": None}
    signed = b[both] - a[both]
    d = np.abs(signed)
    return {"pixels": int(both.sum()), "median_diff": round(float(np.median(signed)), 4),
            "median_abs": round(float(np.median(d)), 4),
            "p99_abs": round(float(np.percentile(d, 99)), 4), "pct_over_tol": round(100 * float((d > tol).mean()), 2)}


def compare_radar_runs(aoi_id: int, old_run: str, new_run: str, season_key: str = "monsoon2026",
                       tol_db: float = 0.01) -> pd.DataFrame:
    """Per track, polarisation and pass date: the new run's values against the old run's (dB, single pixels, the
    stack as exported), plus the passes found in one run only (``in`` = old / new / both)."""
    import numpy as np
    import rasterio

    cfg = season_config(aoi_id, season_key)
    rows = []
    for t in [t["track_id"] for t in cfg["s1"]["tracks"]]:
        for pol in ("VV", "VH"):
            paths = [config_mod.run_dir(cfg, r) / "stack" / f"track_{t}" / f"stack_{pol}.vrt" for r in (old_run, new_run)]
            if not all(p.exists() for p in paths):
                continue
            with rasterio.open(paths[0]) as a, rasterio.open(paths[1]) as b:
                da, db = _stack_dates(a), _stack_dates(b)
                for d in sorted(set(da) | set(db)):
                    row = {"track": t, "pol": pol, "date": d, "in": "both" if d in da and d in db else
                           ("old" if d in da else "new")}
                    if row["in"] == "both":
                        x = a.read(da.index(d) + 1).astype("float32")
                        y = b.read(db.index(d) + 1).astype("float32")
                        for arr, nod in ((x, a.nodata), (y, b.nodata)):
                            if nod is not None:
                                arr[arr == nod] = np.nan
                        row.update(diff_stats(x, y, tol_db))
                    rows.append(row)
    return pd.DataFrame(rows)


def compare_series(aoi_id: int, old_root: str, new_root: str, name: str = "ndvi5d", tol: float = 0.02) -> pd.DataFrame:
    """Per 5-day window: the new series' fitted values against the old one's; for a window only the new series has,
    ``observed_pct`` = share of the grid with a real (clear) observation in it (from ``<aoi>_ndvi5d_raw.tif``)."""
    import numpy as np
    import rasterio

    key = f"aoi{aoi_id}"
    with rasterio.open(Path(old_root) / key / f"{key}_{name}.tif") as a, \
            rasterio.open(Path(new_root) / key / f"{key}_{name}.tif") as b, \
            rasterio.open(Path(old_root) / key / f"{key}_gapdays5d.tif") as old_gap:
        wa, wb, wg = list(a.descriptions), list(b.descriptions), list(old_gap.descriptions)
        rows = []
        for w in wa:
            if w not in wb:
                rows.append({"window": w, "in": "old"})
                continue
            x = a.read(wa.index(w) + 1).astype("float32")
            y = b.read(wb.index(w) + 1).astype("float32")
            x[x == a.nodata] = np.nan
            y[y == b.nodata] = np.nan
            gap = old_gap.read(wg.index(w) + 1) if w in wg else None
            rows.append({"window": w, "in": "both", **diff_stats(x, y, tol),
                         # how far the OLD value was from a real observation: a big change there is a filled guess
                         # replaced by the new date, not a measurement that moved
                         "old_gap_days_median": float(np.median(gap)) if gap is not None else None})
    # a window only the new series has: how much of the grid a clear view actually observed (the rest is filled)
    with rasterio.open(Path(new_root) / key / f"{key}_ndvi5d_raw.tif") as r:
        wr = list(r.descriptions)
        for w in wb:
            if w not in wa:
                raw = r.read(wr.index(w) + 1)
                rows.append({"window": w, "in": "new",
                             "observed_pct": round(100 * float((raw != r.nodata).mean()), 1)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# After a new date is downloaded: is it worth having?
#
# Why: the metadata check above scores a date with Cloud Score+ alone, over the AOI polygon. Once the file is on
# disk, three more things decide whether the date helps the series: how much of the AOI the scene actually covers
# (an AOI on a swath edge can be half empty, which is no data, not cloud), how much the series' own mask
# (``mask_experiment.VARIANTS``, default ``hyb40m1late``) keeps, and whether the kept pixels look like the season
# (median NDVI). One table row per date and a quick-look picture answer that before anyone rebuilds a series.
# ---------------------------------------------------------------------------------------------

#: Mask variant of the series the analysis reads now (``ANALYSIS_SERIES.txt`` points at an ``hyb40m1late`` folder).
SERIES_VARIANT = "hyb40m1late"


def date_quality(b2, b3, b4, b8, qa60, cs, variant: str = SERIES_VARIANT, cs_cut: float = 100 * CLEAR_CDF) -> dict:
    """Shares (percent of the given pixels, normally the AOI polygon) for one date's bands, already cut to the AOI.

    * ``no_data_pct``: pixels the scene does not cover (swath edge / outside the tile), by the series' data test
      (``ndvi_5day.data_mask``); these are missing, not cloudy.
    * ``cs_clear_pct``: pixels with data and Cloud Score+ ``clear`` >= ``cs_cut`` (60, as ``aoi_clear_pct``);
      ``None`` when the file's Cloud Score+ band is 0 everywhere (score did not exist at export, issue 32).
    * ``series_kept_pct``: pixels the series mask ``variant`` keeps (``ndvi_5day.clear_mask`` with its settings).
    * ``median_ndvi_kept``: median NDVI of the kept pixels (``None`` when none is kept).
    Pure numpy, no file access, so it can be tested on small arrays."""
    from .analysis import mask_experiment as me
    from .analysis import ndvi_5day as nd

    v = me.VARIANTS[variant]
    b2, b3, b4, b8, qa60, cs = (np.asarray(a, dtype="float32") for a in (b2, b3, b4, b8, qa60, cs))
    n = max(b4.size, 1)
    data = nd.data_mask(b3, b4, b8, v.get("b8_zero_water", False))
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(data, (b8 - b4) / (b8 + b4), np.nan)
    missing = nd.cloud_score_missing(cs, data)
    cs_min = None if (missing and v.get("cs_missing_unknown")) else v.get("cs_min")
    bits = 1 << 10 if v.get("qa60_mode") == "opaque" else (1 << 10) | (1 << 11)
    kept = nd.clear_mask(data, qa60, cs, b8, ndvi, cs_min, bits, v.get("keep_dark", False),
                         v.get("drop_haze", False), b2, b4)
    return {"no_data_pct": round(100 * float((~data).sum()) / n, 1),
            "cs_clear_pct": None if missing else round(100 * float((data & (cs >= cs_cut)).sum()) / n, 1),
            "series_kept_pct": round(100 * float(kept.sum()) / n, 1),
            "median_ndvi_kept": round(float(np.median(ndvi[kept])), 3) if kept.any() else None}


def describe_dates(aoi_id: int, dates, s2_dir: str = "data/s2_dates_masks",
                   png_dir: str | None = "processed/_batch/s2_2026/qgis_review/inspect",
                   variant: str = SERIES_VARIANT) -> pd.DataFrame:
    """:func:`date_quality` inside the AOI polygon for the given local per-date files, plus (with ``png_dir``) a
    quick-look ``aoi<N>_newdate_<date>.png`` per date: true colour (4-3-2) and 5-3-2 side by side, AOI outline in
    yellow, no-data pixels black. Reads local files only; nothing is exported."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from .optical_export import band_index

    cfg = season_config(aoi_id)
    root = config_mod.repo_root()
    rows = []
    for d in [str(x)[:10] for x in dates]:
        hits = sorted((root / s2_dir / f"aoi{aoi_id}").glob(f"*_S2_{d}.tif"))
        if not hits:
            rows.append({"date": d, "file": None})
            continue
        with rasterio.open(hits[0]) as ds:
            b = {k: ds.read(band_index(ds, k)).astype("float32") for k in ("B2", "B3", "B4", "B5", "B8", "QA60", "clear")}
            shapes = gpd.read_file(config_mod.aoi_path(cfg)).to_crs(ds.crs).geometry
            inside = rasterize(((g, 1) for g in shapes), out_shape=(ds.height, ds.width), transform=ds.transform,
                               fill=0, dtype="uint8").astype(bool)
            extent = (ds.bounds.left, ds.bounds.right, ds.bounds.bottom, ds.bounds.top)
        q = date_quality(*(b[k][inside] for k in ("B2", "B3", "B4", "B8", "QA60", "clear")), variant=variant)
        row = {"date": d, "file": str(hits[0].relative_to(root)), "size_mb": round(hits[0].stat().st_size / 1e6, 2), **q}
        if png_dir:
            row["png"] = str(_quicklook(aoi_id, d, b, shapes, extent, q, root / png_dir).relative_to(root))
        rows.append(row)
    return pd.DataFrame(rows)


def _quicklook(aoi_id, date, b, shapes, extent, q, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .review import _stretch

    empty = b["B4"] <= 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for a, (r, g, bl), title in zip(axes, (("B4", "B3", "B2"), ("B5", "B3", "B2")), ("true colour 4-3-2", "5-3-2")):
        img = np.dstack([_stretch(b[r]), _stretch(b[g]), _stretch(b[bl])])
        img[empty] = 0
        a.imshow(img, extent=extent, interpolation="nearest")
        shapes.boundary.plot(ax=a, color="yellow", linewidth=1)
        a.set_title(title, fontsize=10)
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle(f"aoi{aoi_id} {date}: no data {q['no_data_pct']} %, Cloud Score+ clear {q['cs_clear_pct']} %, "
                 f"series mask keeps {q['series_kept_pct']} %, median NDVI kept {q['median_ndvi_kept']}", fontsize=10)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"aoi{aoi_id}_newdate_{date}.png"
    fig.savefig(p, dpi=90)
    plt.close(fig)
    return p


def analysis_inputs(aoi_id: int, series_root: str, run_id: str | None = None, read_radar: bool = True) -> dict:
    """What the rule would read for an AOI: the radar run (and, with ``read_radar``, the last pass per track as
    ``radar_water.read_series`` actually returns it) and the raw Sentinel-2 dates of ``series_root`` (its date record).
    Why: after newer inputs were added beside the old ones, "the accepted AOIs still read the old inputs" must be
    shown, not assumed; and after a switch the same call shows the AOI now reads the new ones."""
    from .analysis import ndvi_5day as nd
    from .analysis import pixel_report as pr
    from .analysis import radar_water as rw

    loc = pr.locate(aoi_id, 0, season_key="monsoon2026", run_id=run_id)
    out = {"aoi": loc["aoi"], "radar_run": Path(loc["run"]).name}
    if read_radar:
        out["radar_last_pass"] = {t["track_id"]: str(pd.DatetimeIndex(dd).max().date())
                                  for t, (dd, _) in zip(loc["cfg"]["s1"]["tracks"], rw.read_series(aoi_id, run_id=run_id))}
    dates = nd.series_dates(series_root, loc["aoi"])
    out["series_root"] = str(series_root)
    out["series_dates"] = (f"{len(dates)} dates {dates[0]} .. {dates[-1]}" if dates
                           else "no date record: reads every local date")
    return out


def main(argv=None) -> int:
    import argparse

    from . import auth

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.newest_imagery", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["check", "compare-radar", "compare-series", "inputs", "describe"])
    p.add_argument("--dates", nargs="+", help="describe: the local per-date files to describe (YYYY-MM-DD)")
    p.add_argument("--series-root", default="processed/_batch/s2_2026_hyb40m1late",
                   help="inputs: the 5-day series folder to report on")
    p.add_argument("--run", help="inputs: an explicit radar run id (default: the pinned / analysis run)")
    p.add_argument("--aois", nargs="+", type=int, required=True)
    p.add_argument("--old", help="compare-*: the old radar run id / series root")
    p.add_argument("--new", help="compare-*: the new radar run id / series root")
    p.add_argument("--out", help="compare-*: write the table(s) to this CSV (AOI column added)")
    p.add_argument("--end", default=(pd.Timestamp.utcnow().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                   help="exclusive end of the search (default: tomorrow, UTC)")
    p.add_argument("--season", default="monsoon2026")
    p.add_argument("--json", help="also write the result to this JSON file")
    args = p.parse_args(argv)
    if args.step == "describe":
        if not args.dates:
            p.error("describe needs --dates")
        for a in args.aois:
            print(f"== aoi{a}")
            print(describe_dates(a, args.dates).to_string(index=False))
        return 0
    if args.step == "inputs":
        for a in args.aois:
            print(json.dumps(analysis_inputs(a, args.series_root, args.run)))
        return 0
    if args.step in ("compare-radar", "compare-series"):
        if not (args.old and args.new):
            p.error(f"{args.step} needs --old and --new")
        frames = []
        for a in args.aois:
            t = (compare_radar_runs(a, args.old, args.new, args.season) if args.step == "compare-radar"
                 else compare_series(a, args.old, args.new))
            frames.append(t.assign(aoi=f"aoi{a}"))
            print(f"== aoi{a}")
            print(t.to_string(index=False))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            pd.concat(frames, ignore_index=True).to_csv(args.out, index=False)
        return 0
    auth.init_ee(season_config(args.aois[0], args.season))
    report = []
    for a in args.aois:
        r = check_aoi(a, args.end, args.season)
        print(f"== {r['aoi']}  radar run {r['radar_run']}  last passes {r['s1_last']}")
        print(r["s1_new"].to_string(index=False) if len(r["s1_new"]) else "   no new Sentinel-1 pass")
        print(f"   Sentinel-2 newest: local {r['s2_newest_local']}, GCS {r['s2_newest_gcs']}")
        print(r["s2_new"].to_string(index=False) if len(r["s2_new"]) else "   no new Sentinel-2 date")
        report.append({**{k: v for k, v in r.items() if k not in ("s1_new", "s2_new")},
                       "s1_new": json.loads(r["s1_new"].to_json(orient="records", date_format="iso")),
                       "s2_new": json.loads(r["s2_new"].to_json(orient="records", date_format="iso"))})
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
