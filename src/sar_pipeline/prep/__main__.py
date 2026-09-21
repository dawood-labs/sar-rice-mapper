"""`python -m sar_pipeline.prep <step> --config <yaml>` — AOI and availability checks. No exports.

Steps:
  aoi-qc           duplicate detection, area table in acres, and (with --split-dir) a cross-check
                   of a deduplicated per-AOI folder against the original file
  s1-availability  Sentinel-1 tracks, platforms and acquisition dates per month over the AOI
                   bounding box, read from Earth Engine metadata only
  batch-configs    one config (and AOI file) per AOI from a template, tracks left empty
  batch-tracks     after each AOI's audit, choose its tracks by rule and write the reasons

None of these steps creates an Earth Engine task or uploads anything. `aoi-qc` and
`s1-availability` only read; the two batch steps write config files (gitignored) and, for
`batch-tracks`, a CSV of the reasons behind every track choice under processed/_batch/.
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

    bc = sub.add_parser("batch-configs", help="write one config per AOI from a template")
    bc.add_argument("--template", required=True, help="a working config to copy settings from")
    bc.add_argument("--split-dir", required=True, help="folder with one sub-folder per AOI")
    bc.add_argument("--season-key", required=True)
    bc.add_argument("--start", required=True, help="season start YYYY-MM-DD (inclusive)")
    bc.add_argument("--end", required=True, help="season end YYYY-MM-DD (exclusive)")
    bc.add_argument("--ids", nargs="*", type=int, default=None, help="only these AOI ids")
    bc.add_argument("--force", action="store_true", help="overwrite configs that already have tracks")

    bt = sub.add_parser("batch-tracks", help="fill s1.tracks from each AOI's latest audit")
    bt.add_argument("--season-key", required=True)
    bt.add_argument("--flood-start", required=True, help="flooding window start YYYY-MM-DD")
    bt.add_argument("--flood-end", required=True, help="flooding window end YYYY-MM-DD")
    bt.add_argument("--max-gap-days", type=int, default=20)
    bt.add_argument("--fallback-secondary", action="store_true",
                    help="if no second track is eligible, allow one excluded only for a flooding gap "
                         "as secondary (confirmation only, never primary)")
    return parser


def _run_batch_configs(args) -> int:
    import shutil
    import yaml
    from . import batch

    template = yaml.safe_load(Path(args.template).read_text())
    ids = args.ids or batch.aoi_ids_in(args.split_dir)
    written = skipped = 0
    for aoi_id in ids:
        folder = next(Path(args.split_dir).glob(f"*_{aoi_id:03d}"), None)
        source = next(folder.glob(f"*_{aoi_id:03d}.gpkg"), None) if folder else None
        if source is None:
            print(f"AOI {aoi_id}: no split file found, skipped")
            skipped += 1
            continue
        aoi_path = Path("data/aoi") / f"aoi_{aoi_id:03d}.gpkg"   # padded: see README "Running many AOIs"
        if not aoi_path.exists():
            aoi_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, aoi_path)
        out = Path("config") / f"aoi{aoi_id}_{args.season_key}.yaml"
        if out.exists() and not args.force:
            existing = yaml.safe_load(out.read_text()) or {}
            if (existing.get("s1") or {}).get("tracks"):
                skipped += 1
                continue
        out.write_text(batch.render_config(template, aoi_id, str(aoi_path), args.season_key,
                                           args.start, args.end))
        written += 1
    print(f"configs written: {written}, skipped: {skipped}, AOIs: {len(ids)}")
    return 0


def _run_batch_tracks(args) -> int:
    import pandas as pd
    import yaml
    from . import batch

    rows = []
    configs = sorted(Path("config").glob(f"aoi*_{args.season_key}.yaml"))
    for path in configs:
        cfg = config_mod.load_config(path)
        try:
            audit = config_mod.latest_audit_dir(cfg)
        except Exception as exc:  # no audit yet for this AOI
            rows.append({"config": path.name, "primary": None, "secondary": None,
                         "reasons": f"no audit: {exc}"})
            continue
        choice = batch.choose_tracks_from_audit(audit, (args.flood_start, args.flood_end),
                                                max_gap_days=args.max_gap_days,
                                                fallback_secondary=args.fallback_secondary)
        raw = yaml.safe_load(path.read_text())
        raw["s1"]["tracks"] = choice.tracks
        header = "".join(line + "\n" for line in path.read_text().splitlines() if line.startswith("#"))
        path.write_text(header + yaml.safe_dump(raw, sort_keys=False, default_flow_style=None))
        rows.append({"config": path.name, "primary": choice.primary, "secondary": choice.secondary,
                     "reasons": " | ".join(f"{k}: {v}" for k, v in choice.reasons.items())})
    table = pd.DataFrame(rows)
    out = Path("processed") / "_batch" / f"{args.season_key}_track_choice.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(table[["config", "primary", "secondary"]].to_string(index=False))
    print(f"\nno primary track: {int(table['primary'].isna().sum())} of {len(table)}")
    print(f"reasons for every choice: {out}")
    return 0


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
    if args.step == "batch-configs":
        return _run_batch_configs(args)
    if args.step == "batch-tracks":
        return _run_batch_tracks(args)
    cfg = config_mod.load_config(args.config)
    if args.step == "aoi-qc":
        return _run_aoi_qc(cfg, args)
    if args.step == "s1-availability":
        return _run_s1_availability(cfg, args)
    raise SystemExit(f"unknown step {args.step}")


if __name__ == "__main__":
    sys.exit(main())
