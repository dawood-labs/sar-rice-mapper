"""Run the standing-rice map for many AOIs: Sentinel-2 per-date exports, 5-day series, the rule.

Why a driver
------------
The rule (``analysis/monsoon_rule``) needs three inputs per AOI: the per-date Sentinel-2 export
with its mask bands, the gap-free 5-day series built from it, and a Sentinel-1 season run for the
water check. For one AOI the steps are typed by hand (docs/09). For a hundred they must be the same
every time and safe to re-run, so this module does them in order and remembers nothing that is not
already on disk: every step skips what is done (a file on GCS, a series already built).

The one thing that needs care is the Earth Engine queue. A project may hold about 3,000 queued or
running tasks, and one AOI has 100-170 Sentinel-2 dates, so a batch of 25 AOIs alone can fill it.
Before each AOI's dates are submitted, :func:`wait_for_headroom` reads the project's active task
count and waits until that AOI fits under ``limit``. The Sentinel-1 runs keep using the pipeline's
own export/monitor commands, which throttle themselves the same way.

Command line (``--yes`` is required for anything that starts Earth Engine exports)::

    python -m sar_pipeline.monsoon_batch s2-export --ids 119 116 --yes
    python -m sar_pipeline.monsoon_batch rule --ids 119 116     # needs the exports downloaded
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

S2_START = "2025-09-01"
S2_END = "2026-09-24"
S2_FOLDER = "s2_dates_masks"
OUT_ROOT = "processed/_batch/s2_2026"
QUEUE_LIMIT = 2500          # below the ~3,000 project limit, leaving room for the Sentinel-1 runs
POLL_SECONDS = 60


def config_path(aoi_id: int, season_key: str = "monsoon2026") -> Path:
    return Path("config") / f"aoi{aoi_id}_{season_key}.yaml"


def fits(active: int, needed: int, limit: int = QUEUE_LIMIT) -> bool:
    """True when ``needed`` more tasks keep the project's active count at or below ``limit``.

    A job larger than the whole limit is allowed once the queue is empty, so it can never wait forever.
    """
    return active + needed <= limit or active == 0


def active_tasks(backend=None) -> int:
    """Queued + running Earth Engine tasks of the project (bounded listing, see ``export.scan_operations``)."""
    from .export import EEBackend

    backend = backend or EEBackend()
    cutoff = datetime.now(timezone.utc) - timedelta(days=2)
    counts = backend.project_snapshot(cutoff=cutoff)["counts"]
    return int(counts.get("READY", 0)) + int(counts.get("RUNNING", 0))


def wait_for_headroom(needed: int, limit: int = QUEUE_LIMIT, count=active_tasks, sleep=time.sleep,
                      poll_seconds: float = POLL_SECONDS, log=print) -> int:
    """Block until ``needed`` tasks fit under ``limit``; returns the active count seen last."""
    while True:
        active = count()
        if fits(active, needed, limit):
            return active
        log(f"queue: {active} active, {needed} to add, limit {limit}; waiting {poll_seconds:.0f}s")
        sleep(poll_seconds)


def s2_export(ids, start=S2_START, end=S2_END, limit=QUEUE_LIMIT, log=print) -> pd.DataFrame:
    """Submit every missing Sentinel-2 date of each AOI, one AOI at a time, never overfilling the queue.

    Returns the manifest rows (also written to ``<OUT_ROOT>/<aoi>_export_manifest.csv``, appended).
    """
    import geopandas as gpd

    from . import auth, config as config_mod, optical_export as ox

    rows = []
    for aoi_id in ids:
        path = config_path(aoi_id)
        cfg = config_mod.load_config(path)
        auth.init_ee(cfg)
        polygon = ox.aoi_geojson_2d(gpd.read_file(config_mod.aoi_path(cfg)))
        needed = len(ox.scene_dates(polygon, start, end))
        wait_for_headroom(needed, limit, log=log)
        done = ox.export_all_dates([str(path)], start, end, bucket=cfg["gcs"]["bucket"],
                                   prefix=cfg["gcs"]["base_folder"] + "/" + S2_FOLDER, log=log,
                                   bands=ox.BANDS_WITH_MASKS)
        if done:
            out = Path(OUT_ROOT) / f"aoi{aoi_id}_export_manifest.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(done)
            frame.to_csv(out, mode="a", header=not out.exists(), index=False)
        rows.extend(done)
    return pd.DataFrame(rows)


def wait_for_s2(aoi_id: int, log=print, status_of=None, sleep=None) -> list[str]:
    """Wait until every Sentinel-2 task in the AOI's manifest has finished; return the dates that failed.

    Why: the 5-day series is built from whatever files are on GCS. Building it while exports are
    still running would silently leave dates out, and a missing date inside the transplanting
    window is exactly what the rule cannot afford.
    """
    from . import auth, config as config_mod, optical_export as ox

    path = Path(OUT_ROOT) / f"aoi{aoi_id}_export_manifest.csv"
    if not path.exists():
        return []
    manifest = pd.read_csv(path)
    ids = [t for t in manifest.get("task_id", pd.Series(dtype=str)).dropna().astype(str) if t]
    if not ids:
        return []
    if status_of is None:
        auth.init_ee(config_mod.load_config(config_path(aoi_id)))
    states = ox.wait_for_tasks(ids, poll_seconds=POLL_SECONDS, status_of=status_of, sleep=sleep, log=log)
    bad = {t for t, s in states.items() if s in ("FAILED", "CANCELLED")}
    return manifest.loc[manifest["task_id"].astype(str).isin(bad), "date"].astype(str).tolist()


def rule(ids, out_csv=None, log=print, water: str | None = None, suffix: str = "") -> pd.DataFrame:
    """Build the 5-day series (if missing) and run the rule for each AOI; acres per class in one table.

    A failing AOI is recorded with its error and the others carry on. ``water`` picks the water test
    version (default ``monsoon_rule.WATER_DEFAULT``); ``suffix`` writes ``<aoi>_monsoon2026<suffix>.tif``
    so another version can sit next to the delivered map for comparison.
    """
    from .analysis import monsoon_rule as mr
    from .analysis import ndvi_5day as nd

    rows = []
    for aoi_id in ids:
        try:
            if not (Path(OUT_ROOT) / f"aoi{aoi_id}" / f"aoi{aoi_id}_ndvi5d.tif").exists():
                if not (Path(OUT_ROOT) / f"aoi{aoi_id}_export_manifest.csv").exists():
                    raise FileNotFoundError("no Sentinel-2 export manifest yet: run s2-export first")
                failed = wait_for_s2(aoi_id, log=log)
                if failed:
                    log(f"aoi{aoi_id}: {len(failed)} Sentinel-2 dates failed; run s2-export again to retry them")
                nd.build(aoi_id, log=log)
            rows.append(mr.run_aoi(aoi_id, water=water or mr.WATER_DEFAULT, suffix=suffix))
        except Exception as exc:  # keep the batch going; the table says which AOI failed
            log(f"aoi{aoi_id}: {type(exc).__name__}: {exc}")
            rows.append({"aoi": f"aoi{aoi_id}", "error": f"{type(exc).__name__}: {exc}"})
        log(f"aoi{aoi_id}: done")
    table = pd.DataFrame(rows)
    if out_csv:
        merge_rows(table, out_csv)
    return table


def merge_rows(table: pd.DataFrame, path) -> pd.DataFrame:
    """Write ``table`` into the CSV at ``path``, replacing rows of the same AOI and keeping the rest."""
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[~old["aoi"].isin(table["aoi"])], table], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.monsoon_batch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("s2-export", help="submit the per-date Sentinel-2 exports (starts Earth Engine tasks)")
    s.add_argument("--ids", nargs="+", type=int, required=True)
    s.add_argument("--limit", type=int, default=QUEUE_LIMIT, help=f"max active project tasks (default {QUEUE_LIMIT})")
    s.add_argument("--yes", action="store_true", help="really start the exports")
    r = sub.add_parser("rule", help="5-day series + standing-rice rule, acres per class")
    r.add_argument("--ids", nargs="+", type=int, required=True)
    r.add_argument("--out", default=f"{OUT_ROOT}/monsoon2026_rule_acres_final.csv")
    r.add_argument("--water", choices=("v1", "v2"), default=None, help="water test version (default: v2)")
    r.add_argument("--suffix", default="", help="map file suffix, e.g. _v1 for a comparison map")
    args = p.parse_args(argv)
    if args.step == "s2-export":
        if not args.yes:
            print(f"would export every Sentinel-2 date {S2_START}..{S2_END} for AOIs {args.ids}; add --yes")
            return 0
        rows = s2_export(args.ids, limit=args.limit)
        print(f"submitted {len(rows)} dates for {len(args.ids)} AOIs")
    else:
        print(rule(args.ids, out_csv=args.out, water=args.water, suffix=args.suffix).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
