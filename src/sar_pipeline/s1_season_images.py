"""Per-date Sentinel-1 images (VV, VH, VV-VH) and a season summary (VVmin, VHmax, VH range) for every AOI.

Why this exists (user request, 3 Oct 2026)
-------------------------------------------
The rule reads the radar through the project's own stacks, which only this pipeline can open. The user
asked for the radar of the monsoon season as plain GeoTIFFs that anyone can open in QGIS or any GIS
tool, for EVERY AOI, from 1 May 2026 to the newest pass Earth Engine holds:

1. **One image per pass** (per acquisition date and track): band 1 VV, band 2 VH, band 3 VV minus VH,
   all in dB. VV and VH are the two radar "colours"; their difference grows when a canopy of stems
   and leaves scatters the signal (VH rises more than VV), so it is a simple crop indicator.
2. **One summary image per AOI** with exactly three bands that tell the paddy story in one look:

   * ``VVmin``  - the lowest VV of the season. Standing water mirrors the radar away, so a flooded
     paddy (transplanting) or a freshly sown wet field reaches its lowest VV here.
   * ``VHmax``  - the highest VH **on or after** the VVmin date: the crop peak that follows the water.
     (A VH peak before the water belongs to an earlier crop or to weeds, so it is not taken.)
   * ``VH range`` - VHmax minus the season's base VH (the lowest VH from 1 May up to the VHmax date):
     how much canopy grew. Water and bare soil stay low; trees and houses are bright but flat (small
     range); a rice crop goes from very dark to bright (large range).

   No fixed threshold is applied anywhere: these are the pixel's own extremes; the user reads them.
   The dates of VVmin and VHmax (day of year) go into a small sidecar image so the summary keeps
   exactly three bands.

How (and why this route)
------------------------
* **Same processing as the project's radar** (``s1_ard``): ``COPERNICUS/S1_GRD_FLOAT`` (linear power), IW,
  VV+VH, terrain flattening, refined Lee + multi-temporal (Quegan) speckle filter in linear power, then dB,
  on the AOI's own pipeline grid (``processed/<aoi>/<season>/grid/grid_def.json``, UTM, 10 m), so a
  pixel id of ``pixel_index.tif`` is the same pixel in every file here.
* **One track per AOI, the descending one** (user decision, 3 Oct 2026). The tracks look from opposite
  sides at different incidence angles and read about 1 dB apart; with one viewing geometry every pass of
  the series is comparable, and the summary needs no track combining. The configured descending track is
  used (``s1.tracks``; with several, the one with the most passes since 1 May); an AOI with none
  configured uses a descending track Earth Engine offers that covers it, else it is skipped
  (``choose_descending``; the reason is in ``aoi<N>/track_choice.json``). An ascending track is never
  substituted.
* **One Earth Engine task per AOI and grid chunk** exports all of the track's season passes as one
  multi-band stack. Why not one task per pass: the multi-temporal filter of each pass reads its 3
  neighbours on each side, so per-pass tasks would recompute every pass ~7 times and need ~15 times more
  tasks (thousands; the project queue holds 3,000 shared with every other job). Per-AOI stacks need
  ~150 tasks for all 132 AOIs; splitting a stack into per-pass files and computing the summary is
  seconds of local work per AOI. The few passes BEFORE 1 May that the filter needs as neighbours are part
  of the computation but are not exported, so the May passes get exactly the filter they get in the
  project's runs.
* **The summary is computed locally** from the per-pass files, through the same helpers the rule uses
  (``sar_curve.box_mean``, the artefact-pass list ``sar_curve.BAD_PASSES``), so it is consistent with what
  the rule sees:

  - each pass is averaged over the same 5 x 5 window (in linear power) the rule reads, which cuts speckle
    from ~2 dB to ~1 dB: a minimum or maximum of single speckled pixels would mostly measure speckle;
  - the known artefact passes (e.g. one descending pass in June read VH several dB low) are blanked in
    the polarisation that is broken. Passes newer than the list are not screened (the list is only
    extended by ``final_audit.add_new_passes`` on the project's runs).

* All work is resumable: a local work folder holds the plan, the task book and every file; Earth Engine
  tasks are found again by their unique description, and finished steps are skipped. Nothing existing on
  GCS or on disk is ever overwritten (uploads use ``if_generation_match=0``).

Outputs
-------
Local work folder ``processed/_batch/<folder>/`` and GCS ``<base folder>/<folder>/`` (``<folder>`` =
``s1_may_to_latest_<newest date>``)::

    aoi<N>/dates/aoi<N>_<track>_<YYYYMMDD>.tif     3 bands VV, VH, VV-VH: dB x 100, int16, nodata -32768
    aoi<N>/aoi<N>_s1_summary.tif                   3 bands VVmin, VHmax, VH range: dB x 100, int16, nodata -32768
    aoi<N>/aoi<N>_s1_summary_dates.tif             2 bands VVmin day of year, VHmax day of year: uint16, nodata 0
    manifest.csv                                   one row per file (AOI, track, date, file, size, ...)
    README.txt                                     what the bands are
    _ee_stacks/aoi<N>/<track>/<chunk>.tif          the raw Earth Engine exports (float32 dB, nodata -9999)

Dates are UTC acquisition dates (the project's convention; a descending pass at ~23:30 UTC is the next
morning in local time; ``manifest.csv`` has both).

Run (from the repository root)::

    python -m sar_pipeline.s1_season_images plan     [--end 2026-10-03]       # newest pass -> work folder
    python -m sar_pipeline.s1_season_images run      --aois 39 --yes            # export, download, build, upload
    python -m sar_pipeline.s1_season_images verify   --aoi 39 --pixels 21405 8384 30425
    python -m sar_pipeline.s1_season_images pixel    --aoi 39 --pixel 21405     # the series behind the summary
    python -m sar_pipeline.s1_season_images run      --aois all --yes
    python -m sar_pipeline.s1_season_images finalize --yes                      # manifest.csv + README.txt
    python -m sar_pipeline.s1_season_images status
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as config_mod

log = logging.getLogger(__name__)

#: First day of the season covered (user, 3 Oct 2026: "from 1 May to the newest").
START = "2026-05-01"
#: The AOI configs (tracks, grid, credentials) of the monsoon season are reused as they are.
SEASON_KEY = "monsoon2026"
FOLDER_PREFIX = "s1_may_to_latest"
LOCAL_BATCH = "processed/_batch"
#: dB x 100 as int16: 0.01 dB steps, far finer than the ~1 dB speckle, at half the size of float32.
SCALE = 100
NODATA = -32768
#: What Earth Engine writes where it has no data (float32 export, as the project's runs).
EE_NODATA = -9999.0
DOY_NODATA = 0
#: Spatial window of the summary, the same 5 x 5 box the rule reads (``sar_curve.read_track`` default).
WINDOW = 5
#: How far back to list passes so the multi-temporal filter's neighbours before START are found.
CONTEXT_LOOKBACK_DAYS = 75
#: How far back to look for the newest pass when choosing the end of the season.
NEWEST_LOOKBACK_DAYS = 45
TASK_MAX_ATTEMPTS = 3
POLL_SECONDS = 30
#: The storage client keeps 10 HTTP connections; more threads only open and drop extra connections.
GCS_POOL = 10
BANDS = ("VV", "VH", "VV_minus_VH")
SUMMARY_BANDS = ("VVmin", "VHmax", "VH_range")
SUMMARY_DATE_BANDS = ("VVmin_doy", "VHmax_doy")
TASK_COLUMNS = ["task_key", "aoi", "track", "chunk", "n_bands", "description", "gcs_prefix", "state",
                "task_id", "attempts", "submitted_at", "finished_at", "error"]
AOI_COLUMNS = ["aoi", "state", "n_tasks", "n_dates", "n_files", "bytes", "error", "t_plan_s", "t_export_s",
               "t_download_s", "t_build_s", "t_upload_s"]


# ================================================================ pure helpers (tested without network)
def folder_name(newest) -> str:
    """``s1_may_to_latest_<YYYY-MM-DD>``: the newest pass date names the delivery, so a later run with newer
    passes goes into a NEW folder and never mixes with (or overwrites) this one."""
    return f"{FOLDER_PREFIX}_{pd.Timestamp(newest).strftime('%Y-%m-%d')}"


def season_rows(acq: pd.DataFrame, start: str, end: str, half_window: int) -> pd.DataFrame:
    """The passes to compute, per track: every pass from ``start`` to ``end`` (UTC dates, inclusive) flagged
    ``in_season``, plus the last ``half_window`` passes before ``start`` as filter context (not exported).

    Why the context: the multi-temporal speckle filter of a pass averages its ``half_window`` neighbours on
    each side; without the passes before ``start`` the first May passes would get a weaker filter than in
    the project's runs and would not match them."""
    a = acq.copy()
    d = pd.to_datetime(a["date_utc"])
    a = a[d <= pd.Timestamp(end)].copy()
    a["in_season"] = pd.to_datetime(a["date_utc"]) >= pd.Timestamp(start)
    parts = []
    for _, g in a.sort_values("datetime_utc").groupby("track_id", sort=True):
        before = g[~g["in_season"]].tail(int(half_window))
        parts.append(pd.concat([before, g[g["in_season"]]]))
    if not parts:
        return a.iloc[0:0]
    return pd.concat(parts).sort_values(["track_id", "datetime_utc"]).reset_index(drop=True)


def encode_db(a) -> np.ndarray:
    """dB (float, NaN = missing) -> dB x ``SCALE`` as int16 with ``NODATA``."""
    a = np.asarray(a, dtype="float64")
    out = np.full(a.shape, NODATA, dtype="int16")
    ok = np.isfinite(a)
    out[ok] = np.clip(np.round(a[ok] * SCALE), -32767, 32767).astype("int16")
    return out


def decode_db(a) -> np.ndarray:
    """dB x ``SCALE`` int16 -> float32 dB with NaN for ``NODATA``."""
    a = np.asarray(a)
    out = a.astype("float32") / SCALE
    out[a == NODATA] = np.nan
    return out


def date_bands(vv, vh) -> np.ndarray:
    """The three bands of a per-pass image, encoded: VV, VH, VV - VH (NaN where either is missing)."""
    vv = np.asarray(vv, dtype="float32")
    vh = np.asarray(vh, dtype="float32")
    return np.stack([encode_db(vv), encode_db(vh), encode_db(vv - vh)])


def place(target: np.ndarray, tile: np.ndarray, row_off: int, col_off: int) -> None:
    """Copy ``tile`` (bands, rows, cols) into ``target`` at the given offset, clipped to the target."""
    _, h, w = target.shape
    th, tw = tile.shape[1:]
    r0, c0 = max(row_off, 0), max(col_off, 0)
    r1, c1 = min(row_off + th, h), min(col_off + tw, w)
    if r1 <= r0 or c1 <= c0:
        return
    target[:, r0:r1, c0:c1] = tile[:, r0 - row_off:r1 - row_off, c0 - col_off:c1 - col_off]


def season_summary(dates, vv, vh) -> dict:
    """VVmin, VHmax after it, and the VH range, per pixel, from a (t, h, w) series in dB.

    * ``vvmin``: lowest VV; ``vvmin_day``: its date (the first one on a tie).
    * ``vhmax``: highest VH on a date >= the VVmin date (same-day passes of another track count);
      ``vhmax_day``: its date.
    * ``base``: lowest VH from the first date up to and including the VHmax date; ``vh_range`` = vhmax - base
      (>= 0 by construction).

    Dates are compared as calendar days, missing values (NaN) are ignored. A pixel without any VV, or without
    VH after its VVmin, gets NaN (and NaT dates)."""
    days = pd.DatetimeIndex(dates).values.astype("datetime64[D]")
    t = len(days)
    out_shape = np.shape(vv)[1:]
    vv = np.asarray(vv, dtype="float32").reshape(t, -1)
    vh = np.asarray(vh, dtype="float32").reshape(t, -1)
    n = vv.shape[1]
    has_vv = np.isfinite(vv).any(axis=0)
    i_min = np.argmin(np.where(np.isfinite(vv), vv, np.inf), axis=0)
    vvmin = np.where(has_vv, vv[i_min, np.arange(n)], np.nan)
    vvmin_day = np.where(has_vv, days[i_min], np.datetime64("NaT"))

    after = days[:, None] >= vvmin_day[None, :]               # NaT compares False -> nothing selected
    vh_after = np.where(after & np.isfinite(vh), vh, -np.inf)
    i_max = np.argmax(vh_after, axis=0)
    has_max = np.isfinite(vh_after[i_max, np.arange(n)])
    vhmax = np.where(has_max, vh[i_max, np.arange(n)], np.nan)
    vhmax_day = np.where(has_max, days[i_max], np.datetime64("NaT"))

    upto = days[:, None] <= vhmax_day[None, :]
    vh_upto = np.where(upto & np.isfinite(vh), vh, np.inf)
    base = np.where(has_max, vh_upto.min(axis=0), np.nan)
    return {"vvmin": vvmin.reshape(out_shape), "vvmin_day": vvmin_day.reshape(out_shape),
            "vhmax": vhmax.reshape(out_shape), "vhmax_day": vhmax_day.reshape(out_shape),
            "base": base.reshape(out_shape), "vh_range": (vhmax - base).reshape(out_shape)}


def day_of_year(days) -> np.ndarray:
    """datetime64[D] array -> day of year as uint16, ``DOY_NODATA`` (0) for NaT."""
    days = np.asarray(days, dtype="datetime64[D]")
    ok = ~np.isnat(days)
    years = days.astype("datetime64[Y]")
    doy = (days - years.astype("datetime64[D]")).astype("int64") + 1
    return np.where(ok, doy, DOY_NODATA).astype("uint16")


def token_date(token: str) -> pd.Timestamp:
    """``20260705`` or ``20260705_2`` (second pass of a track on one UTC date) -> Timestamp."""
    return pd.Timestamp(datetime.strptime(token[:8], "%Y%m%d"))


def smooth_db(cube_db: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """``window`` x ``window`` mean of every pass, in linear power, back to dB (``sar_curve.box_mean``: NaN-aware)."""
    from .analysis import seasonal_stats as ss
    from .analysis.sar_curve import box_mean

    if window <= 1:
        return np.asarray(cube_db, dtype="float32")
    power = ss.to_linear(cube_db)
    return ss.to_db(np.stack([box_mean(p, window) for p in power]))


def known_bad(table: pd.DataFrame, aoi_key: str, track: str, pol: str) -> set:
    """UTC dates of the recorded artefact passes for one AOI, track and polarisation.

    The AOI's own rows, plus every pass judged bad ``track_wide`` on this track in any AOI: such a pass is broken
    over the whole swath (``final_audit.screen_passes``), so it is dropped also in an AOI the list never judged
    (an AOI whose descending track was not in its config, hence not in the project's runs)."""
    if table is None or table.empty:
        return set()
    bad = table[table["bad"].astype(bool) & (table["track"] == track) & (table["pol"] == pol)]
    own = bad["aoi"] == aoi_key
    wide = bad["reason"].fillna("") == "track_wide" if "reason" in bad else pd.Series(False, index=bad.index)
    return set(pd.to_datetime(bad.loc[own | wide, "date"]).dt.normalize())


# ================================================================ work folder, plan
def work_root() -> Path:
    return config_mod.repo_root() / LOCAL_BATCH


def current_work(folder: str | None = None) -> Path:
    """The work folder: ``folder`` if given, else the newest ``s1_may_to_latest_*`` that holds a plan."""
    if folder:
        return work_root() / folder
    found = sorted(p for p in work_root().glob(f"{FOLDER_PREFIX}_*") if (p / "plan.json").exists())
    if not found:
        raise FileNotFoundError("no plan yet: run `python -m sar_pipeline.s1_season_images plan` first")
    return found[-1]


def aoi_config(aoi_id: int) -> dict:
    return config_mod.load_config(config_mod.repo_root() / "config" / f"aoi{aoi_id}_{SEASON_KEY}.yaml")


def all_aoi_ids() -> list[int]:
    ids = []
    for p in (config_mod.repo_root() / "config").glob(f"aoi*_{SEASON_KEY}.yaml"):
        ids.append(int(p.name.split("_")[0].removeprefix("aoi")))
    return sorted(ids)


def export_cfg(cfg: dict) -> dict:
    """The AOI config with the export encoding this module needs (float32 dB, nodata ``EE_NODATA``):
    the summary is computed from full-precision values, the int16 encoding happens locally."""
    c = json.loads(json.dumps({k: v for k, v in cfg.items()}, default=str))
    c["export"] = dict(c.get("export") or {}, dtype="float32", nodata_float32=EE_NODATA)
    return c


def _track_orbit(track: str) -> tuple[int, str]:
    num, direction = track.removeprefix("RO").split("_")
    return int(num), {"ASC": "ASCENDING", "DSC": "DESCENDING"}[direction]


def newest_pass(aoi_ids, until: str) -> dict:
    """The newest Sentinel-1 pass (UTC date) of every configured descending track over the AOIs that use it, up to
    ``until``.

    One metadata query per track over the AOI centres: cheap, and it fixes the end of the season for the
    whole batch BEFORE any export, so every AOI ends on the same, recorded date."""
    import ee
    import geopandas as gpd
    import shapely

    from . import auth

    by_track: dict[str, list] = {}
    cfg0 = None
    for a in aoi_ids:
        cfg = aoi_config(a)
        cfg0 = cfg0 or cfg
        pt = shapely.union_all(gpd.read_file(config_mod.aoi_path(cfg)).to_crs("EPSG:4326").geometry.values)
        pt = pt.representative_point()
        for t in cfg["s1"]["tracks"]:
            if t["track_id"].endswith("_DSC"):
                by_track.setdefault(t["track_id"], []).append([pt.x, pt.y])
    auth.init_ee(cfg0)
    end = pd.Timestamp(until) + pd.Timedelta(days=1)
    out = {}
    for track, pts in sorted(by_track.items()):
        orbit, direction = _track_orbit(track)
        col = (ee.ImageCollection(cfg0["s1"]["collection"])
               .filterBounds(ee.Geometry.MultiPoint(pts))
               .filterDate((end - pd.Timedelta(days=NEWEST_LOOKBACK_DAYS)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
               .filter(ee.Filter.eq("instrumentMode", cfg0["s1"].get("instrument_mode", "IW")))
               .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
               .filter(ee.Filter.eq("orbitProperties_pass", direction)))
        for pol in cfg0["s1"]["pols"]:
            col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
        ms = col.aggregate_max("system:time_start").getInfo()
        out[track] = None if ms is None else pd.Timestamp(int(ms), unit="ms").strftime("%Y-%m-%d")
    return out


def plan(end: str | None = None, aoi_ids=None) -> Path:
    """Fix the season end (= the newest pass of any configured track up to ``end``, default today UTC) and
    create the work folder ``processed/_batch/s1_may_to_latest_<newest>/plan.json``. Read-only on Earth Engine.
    Re-running on the same day finds the same folder (an existing plan is never rewritten)."""
    aoi_ids = aoi_ids or all_aoi_ids()
    until = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    newest = newest_pass(aoi_ids, until)
    dates = [d for d in newest.values() if d]
    if not dates:
        raise RuntimeError("no Sentinel-1 pass found for any track")
    top = max(dates)
    work = work_root() / folder_name(top)
    work.mkdir(parents=True, exist_ok=True)
    path = work / "plan.json"
    if path.exists():
        log.info("plan exists: %s", path)
        return work
    cfg0 = aoi_config(aoi_ids[0])
    meta = {"folder": work.name, "start": START, "end": top, "checked_until": until, "newest_by_track": newest,
            "gcs_bucket": cfg0["gcs"]["bucket"], "gcs_root": f"{cfg0['gcs']['base_folder']}/{work.name}",
            "season_key": SEASON_KEY, "window": WINDOW, "created_utc": datetime.now(timezone.utc).isoformat(),
            "bad_passes": str(_bad_passes_path())}
    path.write_text(json.dumps(meta, indent=2))
    return work


def _plan(work: Path) -> dict:
    return json.loads((work / "plan.json").read_text())


def _bad_passes_path() -> Path:
    from .analysis import sar_curve

    return config_mod.repo_root() / sar_curve.BAD_PASSES


def _bad_table() -> pd.DataFrame:
    p = _bad_passes_path()
    if not p.exists():
        warnings.warn(f"{p} is missing: artefact passes will NOT be dropped from the summaries", RuntimeWarning)
        return pd.DataFrame(columns=["aoi", "track", "pol", "date", "bad"])
    return pd.read_csv(p)


# ================================================================ per-AOI acquisitions and tasks
def _aoi_dir(work: Path, aoi_id: int) -> Path:
    return work / f"aoi{aoi_id}"


def choose_descending(acq: pd.DataFrame, configured: list[str], start: str,
                      min_coverage: float = 90.0) -> tuple[str | None, str]:
    """The one descending track an AOI uses, and why.

    User decision (3 Oct 2026): one track per AOI, the descending one, so the images and the summary never mix
    viewing geometries. Order: the configured descending track (``s1.tracks``, vetted by the audit); if several are
    configured, the one with the most passes since ``start``; if none is configured, a descending track Earth Engine
    offers whose median pass covers >= ``min_coverage`` % of the AOI (most passes wins); else None (the AOI is
    skipped; an ascending track is never substituted)."""
    a = acq[(acq["track_id"].str.endswith("_DSC")) & (pd.to_datetime(acq["date_utc"]) >= pd.Timestamp(start))]
    if a.empty:
        return None, "no descending pass since the start"
    n = a.groupby("track_id").size()
    conf = [t for t in configured if t.endswith("_DSC") and t in n.index]
    if conf:
        best = max(conf, key=lambda t: (n[t], t))
        why = "configured" if len(conf) == 1 else f"configured; most passes of {conf} ({dict(n[conf])})"
        return best, why
    cov = a.groupby("track_id")["aoi_coverage_pct"].median()
    ok = [t for t in n.index if cov[t] >= min_coverage]
    if not ok:
        return None, f"no descending track configured; none in Earth Engine covers >= {min_coverage} % ({dict(cov.round(1))})"
    best = max(ok, key=lambda t: (n[t], t))
    return best, f"not configured; offered by Earth Engine ({n[best]} passes, median cover {cov[best]:.1f} %)"


def acquisitions(work: Path, aoi_id: int) -> pd.DataFrame:
    """The AOI's passes of its descending track (season + filter context), listed once and kept in
    ``aoi<N>/acquisitions.csv`` so a resumed run exports exactly the same passes. The choice is recorded in
    ``aoi<N>/track_choice.json``; an AOI without a descending track gets an empty table."""
    from . import audit, grid

    path = _aoi_dir(work, aoi_id) / "acquisitions.csv"
    if path.exists():
        return pd.read_csv(path)
    meta = _plan(work)
    cfg = aoi_config(aoi_id)
    gd = grid.load_grid(cfg)
    lo = (pd.Timestamp(meta["start"]) - pd.Timedelta(days=CONTEXT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    hi = (pd.Timestamp(meta["end"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    slices = audit.list_slices(cfg, lo, hi, gd)
    acq = audit.group_acquisitions(slices, audit._aoi_union_4326(cfg), cfg)
    acq = pd.DataFrame(acq.drop(columns="geometry"))
    acq = acq[pd.to_datetime(acq["date_utc"]) <= pd.Timestamp(meta["end"])]
    configured = [t["track_id"] for t in cfg["s1"]["tracks"]]
    track, why = choose_descending(acq, configured, meta["start"])
    path.parent.mkdir(parents=True, exist_ok=True)
    offered = acq[pd.to_datetime(acq["date_utc"]) >= pd.Timestamp(meta["start"])].groupby("track_id").size()
    (path.parent / "track_choice.json").write_text(json.dumps(
        {"track": track, "why": why, "configured": configured, "offered_since_start": offered.to_dict()}, indent=2))
    acq = acq[acq["track_id"] == track]
    ard = cfg.get("ard") or {}
    half = int(ard.get("temporal_half_window", 0) or 0) if ard.get("multitemporal") else 0
    rows = season_rows(acq, meta["start"], meta["end"], half)
    rows.to_csv(path, index=False)
    return rows


def track_choice(work: Path, aoi_id: int) -> dict:
    return json.loads((_aoi_dir(work, aoi_id) / "track_choice.json").read_text())


def _band_names(rows: pd.DataFrame, track: str) -> tuple[list[str], list[str]]:
    """(all band names of the track incl. context, the season band names to export) - ``s1_ard`` naming."""
    from .s1_ard.pipeline import band_names

    r = rows[rows["track_id"] == track].sort_values("datetime_utc").reset_index(drop=True)
    names = band_names(r, ("VV", "VH"))
    season = [n for i, n in enumerate(names) if bool(r["in_season"].iloc[i // 2])]
    return names, season


def plan_tasks(work: Path, aoi_id: int) -> pd.DataFrame:
    """One task row per (track, grid chunk) of the AOI, appended to ``tasks.csv`` once."""
    from . import grid
    from .export import sanitise_description

    book = read_tasks(work)
    if (book["aoi"] == aoi_id).any():
        return book[book["aoi"] == aoi_id]
    meta = _plan(work)
    cfg = aoi_config(aoi_id)
    gd = grid.load_grid(cfg)
    rows = acquisitions(work, aoi_id)
    new = []
    for track in sorted(set(rows["track_id"])):
        if not (rows[rows["track_id"] == track]["in_season"]).any():
            continue
        _, season = _band_names(rows, track)
        for ch in gd["chunks"]:
            key = f"aoi{aoi_id}__{track}__{ch['name']}"
            new.append(dict(task_key=key, aoi=aoi_id, track=track, chunk=ch["name"], n_bands=len(season),
                            description=sanitise_description(f"{meta['folder']}_{key}"),
                            gcs_prefix=f"{meta['gcs_root']}/_ee_stacks/aoi{aoi_id}/{track}/{ch['name']}",
                            state="PLANNED", task_id="", attempts=0, submitted_at="", finished_at="", error=""))
    book = pd.concat([book, pd.DataFrame(new, columns=TASK_COLUMNS)], ignore_index=True)
    write_tasks(work, book)
    return book[book["aoi"] == aoi_id]


def read_tasks(work: Path) -> pd.DataFrame:
    p = work / "tasks.csv"
    if not p.exists():
        return pd.DataFrame(columns=TASK_COLUMNS)
    return pd.read_csv(p, dtype={"task_id": str, "submitted_at": str, "finished_at": str, "error": str}).fillna(
        {"task_id": "", "submitted_at": "", "finished_at": "", "error": ""})


def write_tasks(work: Path, book: pd.DataFrame) -> None:
    tmp = work / "tasks.csv.tmp"
    book[TASK_COLUMNS].to_csv(tmp, index=False)
    tmp.replace(work / "tasks.csv")


def read_aois(work: Path) -> pd.DataFrame:
    p = work / "aois.csv"
    if not p.exists():
        return pd.DataFrame(columns=AOI_COLUMNS)
    return pd.read_csv(p).fillna({"error": ""})


def _set_aoi(work: Path, aoi_id: int, **kw) -> None:
    t = read_aois(work)
    if not (t["aoi"] == aoi_id).any():
        t = pd.concat([t, pd.DataFrame([{"aoi": aoi_id}])], ignore_index=True)
    for k, v in kw.items():
        if k not in t.columns:
            t[k] = np.nan
        t[k] = t[k].astype(object)
        t.loc[t["aoi"] == aoi_id, k] = v
    tmp = work / "aois.csv.tmp"
    t.sort_values("aoi").to_csv(tmp, index=False)
    tmp.replace(work / "aois.csv")


# ================================================================ Earth Engine: submit and follow
def _build_image(work: Path, row: pd.Series):
    """The export image of one task: the AOI's ARD stack of the track (season + context passes), season bands only."""
    from . import grid
    from .s1_ard import pipeline as ard

    cfg = export_cfg(aoi_config(int(row["aoi"])))
    gd = grid.load_grid(cfg)
    rows = acquisitions(work, int(row["aoi"]))
    acq = rows[rows["track_id"] == row["track"]].sort_values("datetime_utc").reset_index(drop=True)
    chunk = next(c for c in gd["chunks"] if c["name"] == row["chunk"])
    _, season = _band_names(rows, row["track"])
    image = ard.build_chunk_image(cfg, gd, chunk, row["track"], acq).select(season)
    return image, cfg, gd, chunk


def _export_params(row, cfg: dict, gd: dict, chunk: dict, bucket: str) -> dict:
    from . import grid

    return dict(description=row["description"], bucket=bucket, fileNamePrefix=row["gcs_prefix"],
                region=grid.chunk_region_coords(chunk), crs=gd["crs"], crsTransform=list(gd["transform"]),
                maxPixels=1e13, fileFormat="GeoTIFF", formatOptions={"noData": EE_NODATA})


def _blobs(bucket, prefix: str) -> list:
    """The GCS files Earth Engine wrote for one task prefix (one file, or several tiles ``<prefix>-r-c.tif``)."""
    return [b for b in bucket.list_blobs(prefix=prefix)
            if b.name == prefix + ".tif" or (b.name.startswith(prefix + "-") and b.name.endswith(".tif"))]


def refresh_and_submit(work: Path, backend, bucket, aoi_ids, confirmed: bool) -> pd.DataFrame:
    """One round: update every submitted task from the project task listing, adopt tasks or outputs left by an
    interrupted round, retry failures (up to ``TASK_MAX_ATTEMPTS``), then start PLANNED tasks within the queue
    headroom (``export.queue_headroom``: the project's queue is shared with every other job)."""
    from .export import classify_error, queue_headroom

    meta = _plan(work)
    book = read_tasks(work)
    cutoff = pd.Timestamp(meta["created_utc"]).to_pydatetime() - timedelta(hours=1)
    snap = backend.project_snapshot(cutoff=cutoff)
    by_desc: dict[str, list] = {}
    for t in snap["tasks"]:
        by_desc.setdefault(t["description"], []).append(t)
    now = datetime.now(timezone.utc).isoformat()
    mine = book["aoi"].isin(aoi_ids)
    for i in book.index[mine]:
        r = book.loc[i]
        if r["state"] in ("COMPLETED", "FAILED"):
            continue
        tasks = sorted(by_desc.get(r["description"], []), key=lambda t: t["create_time"])
        if r["state"] == "SUBMITTED" and r["task_id"]:
            st = next((t for t in tasks if t["id"] == r["task_id"]), None)
            if st is None:
                continue                                  # not in the listing yet: next round
            if st["state"] == "COMPLETED":
                book.loc[i, ["state", "finished_at"]] = ["COMPLETED", now]
            elif st["state"] in ("FAILED", "CANCELLED"):
                err = st.get("error_message", "") or st["state"]
                retry = int(r["attempts"]) < TASK_MAX_ATTEMPTS and classify_error(err) != "cancelled"
                book.loc[i, ["state", "error", "task_id"]] = ["PLANNED" if retry else "FAILED", err[:500],
                                                              "" if retry else r["task_id"]]
                log.warning("%s %s: %s", r["task_key"], "retry" if retry else "FAILED", err[:200])
        elif r["state"] == "PLANNED" and tasks:
            # interrupted after start(): adopt a live or finished task of this exact description
            live = [t for t in tasks if t["state"] in ("READY", "RUNNING", "COMPLETED")]
            if live and int(r["attempts"]) == 0:
                t = live[-1]
                book.loc[i, ["state", "task_id", "attempts"]] = ["SUBMITTED", t["id"], 1]
                if t["state"] == "COMPLETED":
                    book.loc[i, ["state", "finished_at"]] = ["COMPLETED", now]
    if bucket is not None:
        for i in book.index[mine & (book["state"] == "PLANNED") & (book["attempts"] == 0)]:
            if _blobs(bucket, book.loc[i, "gcs_prefix"]):      # output already there (earlier session)
                book.loc[i, ["state", "finished_at"]] = ["COMPLETED", now]
    write_tasks(work, book)

    todo = book[mine & (book["state"] == "PLANNED")]
    if todo.empty or not confirmed:
        return book
    cfg0 = aoi_config(int(todo["aoi"].iloc[0]))
    own_active = int((book["state"] == "SUBMITTED").sum())
    room = queue_headroom(cfg0, snap["counts"], own_active)
    for i in todo.index[:room]:
        r = book.loc[i]
        try:
            image, cfg, gd, chunk = _build_image(work, r)
            task_id = backend.start_export(image, _export_params(r, cfg, gd, chunk, meta["gcs_bucket"]))
        except Exception as exc:  # noqa: BLE001 - recorded per task; the next round retries
            cls = classify_error(str(exc))
            book.loc[i, "error"] = f"start failed ({cls}): {exc}"[:500]
            book.loc[i, "attempts"] = int(r["attempts"]) + 1
            if int(r["attempts"]) + 1 >= TASK_MAX_ATTEMPTS and cls != "quota":
                book.loc[i, "state"] = "FAILED"
            log.warning("could not start %s: %s", r["task_key"], exc)
            continue
        book.loc[i, ["state", "task_id", "submitted_at"]] = ["SUBMITTED", str(task_id), datetime.now(timezone.utc).isoformat()]
        book.loc[i, "attempts"] = int(r["attempts"]) + 1
        write_tasks(work, book)
    write_tasks(work, book)
    return book


# ================================================================ download, build, upload (local, per AOI)
def download_aoi(work: Path, aoi_id: int, bucket) -> list[Path]:
    """Download the AOI's finished Earth Engine stacks to ``aoi<N>/ee/<track>/`` (skips files already complete)."""
    from concurrent.futures import ThreadPoolExecutor

    from . import resources

    book = read_tasks(work)
    rows = book[book["aoi"] == aoi_id]
    jobs = []
    for _, r in rows.iterrows():
        blobs = _blobs(bucket, r["gcs_prefix"])
        if not blobs:
            raise RuntimeError(f"{r['task_key']}: no output on GCS under {r['gcs_prefix']}")
        for b in blobs:
            dest = _aoi_dir(work, aoi_id) / "ee" / r["track"] / Path(b.name).name
            jobs.append((b, dest))

    def get(job):
        b, dest = job
        if dest.exists() and dest.stat().st_size == b.size:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(".part")
        b.download_to_filename(str(part))
        part.replace(dest)
        return dest

    n = resources.io_threads(resources.detect_resources(aoi_config(aoi_id)))
    with ThreadPoolExecutor(max_workers=max(1, min(n, GCS_POOL, len(jobs)))) as pool:
        return list(pool.map(get, jobs))


def _track_cube(work: Path, aoi_id: int, track: str, gd: dict, n_bands: int) -> np.ndarray:
    """The track's season stack on the AOI grid (bands, rows, cols) in float32 dB, NaN = no data."""
    import rasterio

    cube = np.full((n_bands, int(gd["height"]), int(gd["width"])), np.nan, dtype="float32")
    files = sorted((_aoi_dir(work, aoi_id) / "ee" / track).glob("*.tif"))
    if not files:
        raise RuntimeError(f"aoi{aoi_id} {track}: no downloaded stack")
    for f in files:
        with rasterio.open(f) as ds:
            if ds.count != n_bands:
                raise RuntimeError(f"{f}: {ds.count} bands, expected {n_bands}")
            if abs(ds.transform.a - gd["res"]) > 1e-6 or abs(ds.transform.e + gd["res"]) > 1e-6:
                raise RuntimeError(f"{f}: pixel size {ds.transform.a} differs from the grid")
            col = (ds.transform.c - gd["x0"]) / gd["res"]
            row = (gd["y0"] - ds.transform.f) / gd["res"]
            if abs(col - round(col)) > 1e-3 or abs(row - round(row)) > 1e-3:
                raise RuntimeError(f"{f}: not aligned with the AOI grid")
            a = ds.read().astype("float32")
            a[(a == EE_NODATA) | ~np.isfinite(a)] = np.nan
            place(cube, a, int(round(row)), int(round(col)))
    return cube


def _profile(gd: dict, count: int, dtype: str, nodata) -> dict:
    from rasterio.transform import from_origin

    return dict(driver="GTiff", width=int(gd["width"]), height=int(gd["height"]), count=count, dtype=dtype,
                crs=gd["crs"], nodata=nodata, transform=from_origin(gd["x0"], gd["y0"], gd["res"], gd["res"]),
                compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256)


def _aoi_mask(cfg: dict) -> np.ndarray:
    import rasterio

    with rasterio.open(config_mod.grid_dir(cfg) / "pixel_index.tif") as ds:
        return ds.read(3) == 1


def build_aoi(work: Path, aoi_id: int) -> pd.DataFrame:
    """Per-pass GeoTIFFs and the season summary of one AOI from its downloaded stacks.

    Returns the AOI's manifest rows (also written to ``aoi<N>/files.csv``)."""
    import rasterio

    from . import grid

    meta = _plan(work)
    cfg = aoi_config(aoi_id)
    gd = grid.load_grid(cfg)
    mask = _aoi_mask(cfg)
    rows = acquisitions(work, aoi_id)
    book = read_tasks(work)
    bad = _bad_table()
    out = _aoi_dir(work, aoi_id)
    (out / "dates").mkdir(parents=True, exist_ok=True)
    tracks = sorted(set(book.loc[book["aoi"] == aoi_id, "track"]))     # the one descending track
    if len(tracks) != 1:
        raise RuntimeError(f"aoi{aoi_id}: expected one track, found {tracks}")
    manifest, series = [], []
    for track in tracks:
        r = rows[(rows["track_id"] == track)].sort_values("datetime_utc").reset_index(drop=True)
        r = r[r["in_season"].astype(bool)].reset_index(drop=True)
        _, names = _band_names(rows, track)
        cube = _track_cube(work, aoi_id, track, gd, len(names))
        dates, vv_l, vh_l = [], [], []
        for k in range(len(names) // 2):
            vv_name, vh_name = names[2 * k], names[2 * k + 1]
            token = vv_name.split("_", 1)[1]
            acq = r.iloc[k]
            vv, vh = cube[2 * k], cube[2 * k + 1]
            path = out / "dates" / f"aoi{aoi_id}_{track}_{token}.tif"
            if not path.exists():
                with rasterio.open(path.with_suffix(".tmp.tif"), "w", **_profile(gd, 3, "int16", NODATA)) as ds:
                    ds.write(date_bands(vv, vh))
                    for b, name in enumerate(BANDS, start=1):
                        ds.set_band_description(b, f"{name} (dB x {SCALE})")
                    ds.update_tags(track=track, date_utc=acq["date_utc"], datetime_utc=acq["datetime_utc"],
                                   date_local=acq["date_local"], platforms=str(acq["platforms"]),
                                   scale=f"{1 / SCALE}", units="dB", processing="S1_GRD_FLOAT IW, terrain flattening, "
                                   "refined Lee + multi-temporal speckle filter, linear -> dB")
                path.with_suffix(".tmp.tif").replace(path)
            valid = np.isfinite(vv) & np.isfinite(vh)
            bad_pols = [p for p in ("VV", "VH") if token_date(token) in known_bad(bad, f"aoi{aoi_id}", track, p)]
            manifest.append(dict(aoi=f"aoi{aoi_id}", kind="date", track=track, date_utc=acq["date_utc"],
                                 date_local=acq["date_local"], datetime_utc=acq["datetime_utc"],
                                 platforms=acq["platforms"], valid_pct_aoi=round(100 * valid[mask].mean(), 2),
                                 artefact_pols=";".join(bad_pols), local=str(path)))
            d = token_date(token)
            v_vv, v_vh = vv.copy(), vh.copy()
            if "VV" in bad_pols:
                v_vv[:] = np.nan
            if "VH" in bad_pols:
                v_vh[:] = np.nan
            dates.append(d)
            vv_l.append(v_vv)
            vh_l.append(v_vh)
        series.append((pd.DatetimeIndex(dates), {"VV": smooth_db(np.stack(vv_l)), "VH": smooth_db(np.stack(vh_l))}))

    dates, cubes = series[0]
    s = season_summary(dates, cubes["VV"], cubes["VH"])
    summ = out / f"aoi{aoi_id}_s1_summary.tif"
    sdates = out / f"aoi{aoi_id}_s1_summary_dates.tif"
    tags = dict(start=meta["start"], end=meta["end"], tracks=";".join(tracks), window_px=str(WINDOW),
                n_passes=str(len(dates)), method="one descending track; 5x5 mean in linear power; known artefact "
                "passes dropped in the broken polarisation")
    if not summ.exists():
        with rasterio.open(summ.with_suffix(".tmp.tif"), "w", **_profile(gd, 3, "int16", NODATA)) as ds:
            ds.write(np.stack([encode_db(s["vvmin"]), encode_db(s["vhmax"]), encode_db(s["vh_range"])]))
            for b, name in enumerate(SUMMARY_BANDS, start=1):
                ds.set_band_description(b, f"{name} (dB x {SCALE})")
            ds.update_tags(**tags)
        summ.with_suffix(".tmp.tif").replace(summ)
    if not sdates.exists():
        with rasterio.open(sdates.with_suffix(".tmp.tif"), "w", **_profile(gd, 2, "uint16", DOY_NODATA)) as ds:
            ds.write(np.stack([day_of_year(s["vvmin_day"]), day_of_year(s["vhmax_day"])]))
            for b, name in enumerate(SUMMARY_DATE_BANDS, start=1):
                ds.set_band_description(b, f"{name} (day of year {pd.Timestamp(meta['start']).year}, UTC date)")
            ds.update_tags(**tags)
        sdates.with_suffix(".tmp.tif").replace(sdates)
    for kind, p in (("summary", summ), ("summary_dates", sdates)):
        manifest.append(dict(aoi=f"aoi{aoi_id}", kind=kind, track=";".join(tracks), date_utc=f"{meta['start']}..{meta['end']}",
                             date_local="", datetime_utc="", platforms="", valid_pct_aoi=round(
                                 100 * np.isfinite(s["vh_range"])[mask].mean(), 2), artefact_pols="", local=str(p)))
    m = pd.DataFrame(manifest)
    m["size_bytes"] = [Path(p).stat().st_size for p in m["local"]]
    m["file"] = [f"gs://{meta['gcs_bucket']}/{meta['gcs_root']}/{Path(p).relative_to(work).as_posix()}" for p in m["local"]]
    m.to_csv(out / "files.csv", index=False)
    return m


def upload_aoi(work: Path, aoi_id: int, bucket) -> int:
    """Upload the AOI's per-pass files and summaries. Never overwrites: a file already on GCS with the same size
    counts as uploaded; a different one is an error (it is left alone)."""
    from concurrent.futures import ThreadPoolExecutor

    from google.api_core.exceptions import PreconditionFailed

    from . import resources

    meta = _plan(work)
    files = pd.read_csv(_aoi_dir(work, aoi_id) / "files.csv")

    def put(row):
        name = row["file"].split(f"gs://{meta['gcs_bucket']}/", 1)[1]
        blob = bucket.blob(name)
        try:
            blob.upload_from_filename(row["local"], if_generation_match=0)
        except PreconditionFailed:
            blob.reload()
            if int(blob.size) != int(row["size_bytes"]):
                raise RuntimeError(f"{name} exists on GCS with another size ({blob.size} != {row['size_bytes']}); "
                                   "left as it is") from None
        return int(row["size_bytes"])

    n = resources.io_threads(resources.detect_resources(aoi_config(aoi_id)))
    with ThreadPoolExecutor(max_workers=max(1, min(n, GCS_POOL, len(files)))) as pool:
        return sum(pool.map(put, [r for _, r in files.iterrows()]))


# ================================================================ the loop
def run(aoi_ids, work: Path | None = None, confirmed: bool = False, poll: int = POLL_SECONDS,
        backend=None) -> pd.DataFrame:
    """Export, download, build and upload ``aoi_ids``: Earth Engine tasks of all AOIs run in parallel (within the
    queue headroom); each AOI is processed locally, one at a time, as soon as all its tasks are finished. Safe to stop
    and start again. Without ``confirmed`` nothing is exported (dry run: plans and shows the tasks)."""
    from . import auth
    from .export import EEBackend
    from .locking import run_lock

    work = work or current_work()
    meta = _plan(work)
    aoi_ids = [int(a) for a in aoi_ids]
    cfg0 = aoi_config(aoi_ids[0])
    auth.init_ee(cfg0)
    bucket = auth.gcs_bucket(cfg0)
    backend = backend or EEBackend()
    lock = run_lock(work, "season_images")
    try:
        for a in aoi_ids:
            done = read_aois(work)
            if (done["aoi"] == a).any() and done.loc[done["aoi"] == a, "state"].iloc[0] in ("UPLOADED", "FAILED", "SKIPPED"):
                continue
            t0 = time.monotonic()
            try:
                n = len(plan_tasks(work, a))
                if n == 0:
                    _set_aoi(work, a, state="SKIPPED", n_tasks=0, error=track_choice(work, a)["why"],
                             t_plan_s=round(time.monotonic() - t0, 1))
                    continue
                _set_aoi(work, a, state="PLANNED", t_plan_s=round(time.monotonic() - t0, 1), n_tasks=n)
            except Exception as exc:  # noqa: BLE001 - one AOI must not stop the batch
                log.exception("aoi%d: planning failed", a)
                _set_aoi(work, a, state="FAILED", error=f"plan: {exc}"[:300])
        if not confirmed:
            print(read_tasks(work).groupby("state").size().to_string())
            return read_aois(work)
        started = {a: time.monotonic() for a in aoi_ids}
        while True:
            book = refresh_and_submit(work, backend, bucket, aoi_ids, confirmed)
            aois = read_aois(work)
            pending = [a for a in aoi_ids if (aois["aoi"] == a).any()
                       and aois.loc[aois["aoi"] == a, "state"].iloc[0] == "PLANNED"]
            for a in pending:
                rows = book[book["aoi"] == a]
                if (rows["state"] == "FAILED").any():
                    _set_aoi(work, a, state="FAILED", error="export: " + "; ".join(rows.loc[rows["state"] == "FAILED", "error"].astype(str))[:300])
                    continue
                if not (rows["state"] == "COMPLETED").all():
                    continue
                _process(work, a, bucket, round(time.monotonic() - started[a], 1))
            states = read_aois(work).set_index("aoi").reindex(aoi_ids)["state"]
            counts = book[book["aoi"].isin(aoi_ids)].groupby("state").size().to_dict()
            log.info("tasks %s | AOIs %s", counts, states.value_counts().to_dict())
            if not states.isin(["PLANNED"]).any():
                break
            time.sleep(poll)
    finally:
        lock.release()
    return read_aois(work)


def _process(work: Path, aoi_id: int, bucket, t_export: float) -> None:
    t = {"t_export_s": t_export}
    try:
        t0 = time.monotonic()
        download_aoi(work, aoi_id, bucket)
        t["t_download_s"] = round(time.monotonic() - t0, 1)
        t0 = time.monotonic()
        m = build_aoi(work, aoi_id)
        t["t_build_s"] = round(time.monotonic() - t0, 1)
        t0 = time.monotonic()
        size = upload_aoi(work, aoi_id, bucket)
        t["t_upload_s"] = round(time.monotonic() - t0, 1)
        _set_aoi(work, aoi_id, state="UPLOADED", n_dates=int((m["kind"] == "date").sum()), n_files=len(m),
                 bytes=size, error="", **t)
        log.info("aoi%d done: %d files, %s", aoi_id, len(m), t)
    except Exception as exc:  # noqa: BLE001 - recorded; the AOI can be re-run with `run --retry`
        log.exception("aoi%d failed", aoi_id)
        _set_aoi(work, aoi_id, state="FAILED", error=f"{exc}"[:300], **t)


README_TEXT = """Sentinel-1 per-pass images and season summaries, {start} to {end} (UTC dates).

aoi<N>/dates/aoi<N>_<track>_<YYYYMMDD>.tif  one radar pass. Band 1 VV, band 2 VH, band 3 VV minus VH.
    Values are dB x 100 (int16; divide by 100 for dB); -32768 = no data. Same processing as the project's radar:
    GRD (linear power), terrain flattening, refined Lee + multi-temporal speckle filter, then dB; 10 m, single pixels.
    Pixels are those of the AOI's pipeline grid (pixel ids of pixel_index.tif).
aoi<N>/aoi<N>_s1_summary.tif  three bands, dB x 100 (int16, -32768 = no data):
    1 VVmin    lowest VV of the season (water / sowing low)
    2 VHmax    highest VH on or after the VVmin date (crop peak)
    3 VH_range VHmax minus the lowest VH from {start} up to the VHmax date (canopy growth)
    Computed from the AOI's one descending track on 5 x 5 pixel means (linear power); known artefact passes dropped.
aoi<N>/aoi<N>_s1_summary_dates.tif  band 1 day of year of VVmin, band 2 day of year of VHmax (uint16, 0 = no data).
manifest.csv  one row per file: AOI, kind, track, date, valid share of the AOI, artefact polarisations, file, size.
_ee_stacks/  the raw Earth Engine exports (float32 dB, -9999 = no data), kept for reference.
"""


def finalize(work: Path | None = None, confirmed: bool = False) -> Path:
    """Write ``manifest.csv`` (every uploaded AOI) and ``README.txt`` at the folder root, locally and on GCS.
    Never overwrites: if a manifest already exists on GCS, this one is uploaded as ``manifest_<UTC time>.csv``."""
    from google.api_core.exceptions import PreconditionFailed

    from . import auth

    work = work or current_work()
    meta = _plan(work)
    aois = read_aois(work)
    done = aois.loc[aois["state"] == "UPLOADED", "aoi"].astype(int)
    parts = [pd.read_csv(_aoi_dir(work, a) / "files.csv") for a in sorted(done)]
    m = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    cols = ["aoi", "kind", "track", "date_utc", "date_local", "datetime_utc", "platforms", "valid_pct_aoi",
            "artefact_pols", "file", "size_bytes"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    local = work / f"manifest_{stamp}.csv"
    m[cols].to_csv(local, index=False)
    (work / "README.txt").write_text(README_TEXT.format(start=meta["start"], end=meta["end"]))
    if not confirmed:
        return local
    bucket = auth.gcs_bucket(aoi_config(int(done.iloc[0]) if len(done) else all_aoi_ids()[0]))
    for src, name in ((local, "manifest.csv"), (work / "README.txt", "README.txt")):
        blob = bucket.blob(f"{meta['gcs_root']}/{name}")
        try:
            blob.upload_from_filename(str(src), if_generation_match=0)
        except PreconditionFailed:
            alt = name.replace(".", f"_{stamp}.", 1)
            bucket.blob(f"{meta['gcs_root']}/{alt}").upload_from_filename(str(src), if_generation_match=0)
            log.info("%s exists on GCS; uploaded as %s", name, alt)
    return local


# ================================================================ checks
def read_dates(work: Path, aoi_id: int, track: str) -> tuple[pd.DatetimeIndex, dict]:
    """Per-pass VV / VH of one track from the local per-pass files: ``(dates, {"VV": (t, h, w), "VH": ...})`` dB."""
    import rasterio

    files = sorted((_aoi_dir(work, aoi_id) / "dates").glob(f"aoi{aoi_id}_{track}_*.tif"))
    files = [f for f in files if not f.name.endswith(".tmp.tif")]
    dates, vv, vh = [], [], []
    for f in files:
        with rasterio.open(f) as ds:
            a = ds.read([1, 2])
        dates.append(token_date(f.stem.removeprefix(f"aoi{aoi_id}_{track}_")))
        vv.append(decode_db(a[0]))
        vh.append(decode_db(a[1]))
    return pd.DatetimeIndex(dates), {"VV": np.stack(vv), "VH": np.stack(vh)}


def verify(aoi_id: int, pixels=(), work: Path | None = None) -> dict:
    """Compare the new per-pass images with the project's radar run (the run the analysis is pinned to) on the
    passes both hold, and show the summary at ``pixels``.

    Per track and date: median and 95th percentile of |new - old| over the AOI (single pixels and 5 x 5 means) and
    both values at each pixel. The two must agree to within speckle-filter differences: the passes near the old
    run's end differ a little because the multi-temporal filter now also sees the newer neighbours."""
    import rasterio

    from .analysis import pixel_report as pr
    from .analysis import sar_curve

    work = work or current_work()
    loc = pr.locate(aoi_id, 0, season_key=SEASON_KEY)
    mask = _aoi_mask(loc["cfg"])
    rows, px_rows = [], []
    for track in [track_choice(work, aoi_id)["track"]]:
        if track not in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
            continue                                  # not in the project's run: nothing to compare with
        nd, ncube = read_dates(work, aoi_id, track)
        od, ocube = sar_curve.read_track(loc, track, window=1, drop_bad=False)
        common = sorted(set(nd) & set(od))
        for d in common:
            i, j = list(nd).index(d), list(od).index(d)
            for pol in ("VV", "VH"):
                new, old = ncube[pol][i], ocube[pol][j]
                diff = np.abs(new - old)[mask & np.isfinite(new) & np.isfinite(old)]
                ns, os_ = smooth_db(new[None])[0], smooth_db(old[None])[0]
                dsm = np.abs(ns - os_)[mask & np.isfinite(ns) & np.isfinite(os_)]
                rows.append(dict(track=track, date=d.date(), pol=pol, n=len(diff),
                                 median_abs_db=round(float(np.median(diff)), 3) if len(diff) else None,
                                 p95_abs_db=round(float(np.percentile(diff, 95)), 3) if len(diff) else None,
                                 median_abs_db_5x5=round(float(np.median(dsm)), 3) if len(dsm) else None,
                                 old_run=Path(loc["run"]).name))
                for p in pixels:
                    r, c = divmod(int(p), int(loc["grid"]["width"]))
                    px_rows.append(dict(pixel=p, track=track, date=d.date(), pol=pol, new_db=round(float(new[r, c]), 2),
                                        old_db=round(float(old[r, c]), 2), new_5x5=round(float(ns[r, c]), 2),
                                        old_5x5=round(float(os_[r, c]), 2)))
    dates_cmp = pd.DataFrame(rows)
    px_cmp = pd.DataFrame(px_rows)
    summ = []
    with rasterio.open(_aoi_dir(work, aoi_id) / f"aoi{aoi_id}_s1_summary.tif") as ds:
        s = ds.read()
    with rasterio.open(_aoi_dir(work, aoi_id) / f"aoi{aoi_id}_s1_summary_dates.tif") as ds:
        sd = ds.read()
    year = pd.Timestamp(_plan(work)["start"]).year
    for p in pixels:
        r, c = divmod(int(p), int(loc["grid"]["width"]))
        dd = [pd.Timestamp(year, 1, 1) + pd.Timedelta(days=int(v) - 1) if v else None for v in sd[:, r, c]]
        summ.append(dict(pixel=p, VVmin_db=s[0, r, c] / SCALE, VVmin_date=dd[0].date() if dd[0] else None,
                         VHmax_db=s[1, r, c] / SCALE, VHmax_date=dd[1].date() if dd[1] else None,
                         VH_range_db=s[2, r, c] / SCALE))
    rep = work / "report"
    rep.mkdir(exist_ok=True)
    dates_cmp.to_csv(rep / f"aoi{aoi_id}_verify_dates.csv", index=False)
    px_cmp.to_csv(rep / f"aoi{aoi_id}_verify_pixels.csv", index=False)
    pd.DataFrame(summ).to_csv(rep / f"aoi{aoi_id}_summary_pixels.csv", index=False)
    return {"dates": dates_cmp, "pixels": px_cmp, "summary": pd.DataFrame(summ)}


def pixel_series(aoi_id: int, pid: int, work: Path | None = None) -> pd.DataFrame:
    """The 5 x 5 series the summary was computed from, at one pixel (artefact passes already blanked)."""
    work = work or current_work()
    cfg = aoi_config(aoi_id)
    track = track_choice(work, aoi_id)["track"]
    dates, cubes = _summary_input(work, aoi_id, track)
    r, c = divmod(int(pid), int(json.loads((config_mod.grid_dir(cfg) / "grid_def.json").read_text())["width"]))
    return pd.DataFrame({"date": dates, "track": track, "VV": cubes["VV"][:, r, c].round(2),
                         "VH": cubes["VH"][:, r, c].round(2), "VV_minus_VH": (cubes["VV"] - cubes["VH"])[:, r, c].round(2)})


def _summary_input(work: Path, aoi_id: int, track: str) -> tuple:
    """The track's 5 x 5 series with artefact passes blanked, read back from the per-pass files (what
    ``build_aoi`` computes the summary from)."""
    bad = _bad_table()
    d, cube = read_dates(work, aoi_id, track)
    for pol in ("VV", "VH"):
        b = known_bad(bad, f"aoi{aoi_id}", track, pol)
        for i, x in enumerate(d):
            if x in b:
                cube[pol][i] = np.nan
        cube[pol] = smooth_db(cube[pol])
    return d, cube


def status(work: Path | None = None) -> str:
    work = work or current_work()
    meta = _plan(work)
    book, aois = read_tasks(work), read_aois(work)
    lines = [f"{work.name}: {meta['start']} .. {meta['end']} -> gs://{meta['gcs_bucket']}/{meta['gcs_root']}",
             "tasks: " + json.dumps(book.groupby("state").size().to_dict()),
             "AOIs:  " + json.dumps(aois.groupby("state").size().to_dict())]
    for _, r in aois[aois["state"].isin(["FAILED", "SKIPPED"])].iterrows():
        lines.append(f"  aoi{int(r['aoi'])} {r['state']}: {r['error']}")
    return "\n".join(lines)


def retry(work: Path, aoi_ids) -> None:
    """Put FAILED AOIs (and their FAILED tasks) back in line; their finished tasks and files are kept."""
    book = read_tasks(work)
    sel = book["aoi"].isin(aoi_ids) & (book["state"] == "FAILED")
    book.loc[sel, ["state", "attempts", "task_id"]] = ["PLANNED", 0, ""]
    write_tasks(work, book)
    for a in aoi_ids:
        if (read_aois(work)["aoi"] == a).any():
            _set_aoi(work, a, state="PLANNED", error="")


# ================================================================ CLI
def _ids(values) -> list[int]:
    if values == ["all"]:
        return all_aoi_ids()
    return [int(v) for v in values]


def main(argv=None) -> int:
    import argparse
    import os

    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("googleapiclient", "google", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.s1_season_images", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", help="work folder name (default: the newest planned one)")
    sub = p.add_subparsers(dest="step", required=True)
    a = sub.add_parser("plan", help="find the newest pass and create the work folder (read-only on Earth Engine)")
    a.add_argument("--end", help="look for passes up to this UTC date (default today)")
    a = sub.add_parser("run", help="export, download, build and upload AOIs")
    a.add_argument("--aois", nargs="+", required=True, help="AOI numbers, or 'all'")
    a.add_argument("--yes", action="store_true", help="really start Earth Engine exports and uploads")
    a.add_argument("--retry", action="store_true", help="put FAILED AOIs of the list back in line first")
    a = sub.add_parser("verify", help="compare with the project's radar run and show the summary at pixels")
    a.add_argument("--aoi", type=int, required=True)
    a.add_argument("--pixels", nargs="*", type=int, default=[])
    a = sub.add_parser("pixel", help="the 5x5 series behind the summary at one pixel")
    a.add_argument("--aoi", type=int, required=True)
    a.add_argument("--pixel", type=int, required=True)
    a = sub.add_parser("finalize", help="write manifest.csv and README.txt (uploads with --yes)")
    a.add_argument("--yes", action="store_true")
    sub.add_parser("status", help="tasks and AOIs by state")
    args = p.parse_args(argv)
    os.chdir(config_mod.repo_root())
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    if args.step == "plan":
        print(plan(args.end))
        print(status(current_work(None)))
        return 0
    work = current_work(args.folder)
    if args.step == "run":
        ids = _ids(args.aois)
        if args.retry:
            retry(work, ids)
        print(run(ids, work, confirmed=args.yes).to_string(index=False))
        print(status(work))
    elif args.step == "verify":
        out = verify(args.aoi, args.pixels, work)
        for k, v in out.items():
            print(f"--- {k}\n{v.to_string(index=False)}")
    elif args.step == "pixel":
        print(pixel_series(args.aoi, args.pixel, work).to_string(index=False))
    elif args.step == "finalize":
        print(finalize(work, args.yes))
    else:
        print(status(work))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
