"""`python -m sar_pipeline.prep <step> --config <yaml>` — AOI and availability checks. No exports.

Steps:
  aoi-qc           duplicate detection, area table in acres, and (with --split-dir) a cross-check
                   of a deduplicated per-AOI folder against the original file
  s1-availability  Sentinel-1 tracks, platforms and acquisition dates per month over the AOI
                   bounding box, read from Earth Engine metadata only

Both steps are read-only: nothing is written to a run folder, no Earth Engine task is created and
nothing is uploaded. Run them before `grid`, while you are still choosing the season window.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd

from .. import config as config_mod

log = logging.getLogger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sar_pipeline.prep", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)

    qc = sub.add_parser("aoi-qc", help="AOI duplicate detection, areas in acres, optional cross-check")
    qc.add_argument("--config", required=True, help="your config (config/<aoi>_<season>.yaml)")
    qc.add_argument("--split-dir", default=None,
                    help="folder of one-feature-per-AOI files to cross-check against aoi.path")
    qc.add_argument("--pattern", default="*/*.gpkg",
                    help="glob for the split files inside --split-dir (default: */*.gpkg)")
    qc.add_argument("--id-field", default="id", help="id column in the AOI files (default: id)")
    qc.add_argument("--area-column", default="area",
                    help="supplied area column to verify against geometry; '' to skip")
    qc.add_argument("--top", type=int, default=10, help="how many largest AOIs to list (default: 10)")
    qc.add_argument("--json", dest="as_json", action="store_true", help="print JSON instead of text")

    av = sub.add_parser("s1-availability", help="Sentinel-1 tracks and dates over the AOI (metadata only)")
    av.add_argument("--config", required=True)
    av.add_argument("--start", default=None, help="override season.start (YYYY-MM-DD)")
    av.add_argument("--end", default=None, help="override season.end (YYYY-MM-DD, exclusive)")
    av.add_argument("--min-dates", type=int, default=4,
                    help="flag months with fewer distinct dates than this (default: 4)")
    av.add_argument("--json", dest="as_json", action="store_true", help="print JSON instead of text")
    return parser


def _run_aoi_qc(cfg: dict, args) -> int:
    from . import aoi_qc

    original = gpd.read_file(config_mod.aoi_path(cfg))
    crs = cfg["grid"]["crs"]
    payload: dict = {"n_features": len(original), "crs_of_file": str(original.crs), "grid_crs": crs}

    duplicates = aoi_qc.find_duplicates(original)
    payload["duplicates"] = {
        "n_unique": duplicates.n_unique,
        "n_duplicate_rows": duplicates.n_duplicate_rows,
        "group_sizes": duplicates.group_sizes,
        "counts_add_up": duplicates.is_consistent(),
    }

    # Deduplicate before summarising, so repeated polygons do not double-count the total area.
    unique = original.assign(_wkb=original.geometry.normalize().to_wkb()).drop_duplicates("_wkb")
    areas = aoi_qc.area_table(unique, crs, id_field=args.id_field)
    payload["area_acres"] = {
        "total": round(float(areas["acres"].sum()), 1),
        "median": round(float(areas["acres"].median()), 1),
        "min": round(float(areas["acres"].min()), 2),
        "max": round(float(areas["acres"].max()), 1),
        "largest": [{"id": int(r.id), "acres": round(float(r.acres), 1)}
                    for r in areas.head(args.top).itertuples()],
    }

    if args.area_column and args.area_column in original.columns:
        payload["area_column_check"] = aoi_qc.compare_area_column(unique, crs, args.area_column)

    if args.split_dir:
        split = aoi_qc.read_split_aois(Path(args.split_dir).glob(args.pattern), id_field=args.id_field)
        report = aoi_qc.crosscheck(original, split, id_field=args.id_field)
        payload["crosscheck"] = {
            "n_split_files": report.n_split_files,
            "missing_from_split": report.missing_from_split,
            "unexpected_in_split": report.unexpected_in_split,
            "dropped_to_kept": report.dropped_to_kept,
            "orphaned": report.orphaned,
            "id_mismatches": report.id_mismatches,
            "ok": report.ok,
        }

    if args.as_json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"AOI file            : {config_mod.aoi_path(cfg)}")
        print(f"features            : {payload['n_features']}  (file CRS {payload['crs_of_file']})")
        print(f"distinct shapes     : {duplicates.n_unique}  "
              f"(duplicate rows {duplicates.n_duplicate_rows}, groups {duplicates.group_sizes})")
        if not duplicates.is_consistent():
            print("WARNING: group sizes do not add up to the feature count")
        acres = payload["area_acres"]
        print(f"area (acres)        : total {acres['total']:,} | median {acres['median']:,} | "
              f"min {acres['min']:,} | max {acres['max']:,}")
        print(f"largest {args.top} AOIs:")
        for row in acres["largest"]:
            print(f"  id {row['id']:>4}  {row['acres']:>10,.1f} acres")
        if "area_column_check" in payload:
            check = payload["area_column_check"]
            print(f"supplied '{check['column']}' column looks like {check['likely_unit']} "
                  f"(ratio {check['ratio_to_computed_acres']}, "
                  f"max disagreement {check['max_disagreement_pct']}%)")
        if "crosscheck" in payload:
            print()
            print(report.summary())
    return 0 if (not args.split_dir or payload["crosscheck"]["ok"]) else 1


def _run_s1_availability(cfg: dict, args) -> int:
    from .. import auth
    from . import s1_availability

    start = args.start or cfg["season"]["start"]
    end = args.end or cfg["season"]["end"]
    aoi = gpd.read_file(config_mod.aoi_path(cfg)).to_crs(4326)
    bounds = tuple(float(v) for v in aoi.total_bounds)

    auth.init_ee(cfg)
    collection = (cfg.get("s1") or {}).get("collection", "COPERNICUS/S1_GRD")
    # The float collection holds the same scenes as the byte one; metadata queries work on either,
    # and S1_GRD is the safer default for a pure availability count.
    if collection.endswith("_FLOAT"):
        collection = collection[: -len("_FLOAT")]
    mode = (cfg.get("s1") or {}).get("instrument_mode", "IW")

    report = s1_availability.availability(bounds, start, end, collection=collection,
                                          instrument_mode=mode)
    thin = report.months_below(args.min_dates)

    if args.as_json:
        print(json.dumps({
            "start": report.start, "end": report.end, "n_scenes": report.n_scenes,
            "tracks": [{"key": t.key, "relative_orbit": t.relative_orbit,
                        "pass": t.orbit_pass, "n_scenes": t.n_scenes} for t in report.tracks],
            "platforms": report.platforms,
            "dates_per_month": report.dates_per_month,
            "months_below_min": thin,
        }, indent=2))
    else:
        print(report.summary())
        if thin:
            print(f"\nmonths with fewer than {args.min_dates} distinct dates: {', '.join(thin)}")
        print("\nNote: counts are over the AOI bounding box. Use `audit` to decide s1.tracks.")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parser().parse_args(argv)
    cfg = config_mod.load_config(args.config)
    if args.step == "aoi-qc":
        return _run_aoi_qc(cfg, args)
    if args.step == "s1-availability":
        return _run_s1_availability(cfg, args)
    raise SystemExit(f"unknown step {args.step}")


if __name__ == "__main__":
    sys.exit(main())
