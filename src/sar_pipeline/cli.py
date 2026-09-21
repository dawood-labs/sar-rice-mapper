"""Command-line interface: one sub-command per pipeline stage.

Usage (see docs/04_runbook.md for the full, ordered procedure):

    python -m sar_pipeline --config config/<aoi>_<season>.yaml grid
    python -m sar_pipeline --config ... audit
    python -m sar_pipeline --config ... new-run
    python -m sar_pipeline --config ... export  [--run RUN] (--pilot CHUNK | --all | --retry-failed) [--yes]
    python -m sar_pipeline --config ... monitor [--run RUN] [--yes]
    python -m sar_pipeline --config ... download [--run RUN] [--yes] [--retry-failed] [--deep]
    python -m sar_pipeline --config ... stack   [--run RUN] [--track TRACK]
    python -m sar_pipeline --config ... pixel   [--run RUN] --track TRACK (--pid PID | --lon LON --lat LAT) [--plot out.png]
    python -m sar_pipeline --config ... resources

Why `--yes`?
    Earth Engine exports cost quota and bulk downloads cost time and disk. Without `--yes`, the
    `export`, `monitor` and `download` commands only print what WOULD happen and exit with code 2.
    Run them again with `--yes` once you have checked the summary. `export` without `--yes` records
    the plan as PLANNED rows; only `export --yes` confirms that scope (PENDING). `monitor` never
    submits PLANNED rows, so a dry run of `--all` can never be exported by a later pilot confirmation.

Which config is used?
    `--config` is YOUR editable config. It is used by grid, audit, new-run and resources, and to find
    the project/season folders. `new-run` freezes a copy into runs/<run_id>/run_config.yaml.
    Run-scoped commands (export, monitor, download, stack, pixel) use that FROZEN copy for everything,
    so a run is always interpreted with the settings that produced it (dtype, nodata, filters, dates).
    If your config differs from the frozen one, a warning lists the differing keys: such changes need
    a new run. Exceptions taken from YOUR config: `qa.acknowledged_issues` (acknowledging QA issues
    is a decision made after the run was created), the `resources` section and `auth.key_file`
    (a run may be exported on a laptop and stacked on a cloud notebook server). `auth.project`
    stays the run's project: its tasks and queue live there.

Crash-safe re-runs:
    Every stage resumes by default: finished, verified work is detected and skipped. If a command
    crashed (or the kernel died), simply run the same command again. `--force` (audit, export plan,
    download, stack) recomputes derived files inside the current audit/run folder only; it never
    touches grid/ (immutable) and never deletes other runs.

Exit codes: 0 = success, 1 = pipeline error (message printed), 2 = confirmation needed / bad usage.

Pipeline modules are imported lazily inside each handler (via `_mod`) so that `--help` works
even when optional heavy dependencies are missing, and so tests can replace modules.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFIRM = 2


def _mod(name: str):
    """Import `sar_pipeline.<name>` at call time (returns the entry in sys.modules if already loaded)."""
    return importlib.import_module(f"sar_pipeline.{name}")


def _print(msg: str = "") -> None:
    print(msg, flush=True)


def _confirm_hint(command: str) -> None:
    _print(f"\nNothing was started. Re-run `{command}` with --yes to proceed.")


# Config sections the frozen run config decides. Differences between the user's config and the frozen
# run config in these sections are reported (not applied). `audit` and `rain` are included: the stack QA
# of a run uses the frozen gap/event settings, so an edited value would otherwise be ignored silently.
RUN_SECTIONS = ("aoi", "season", "grid", "gcs", "s1", "ard", "export", "qa", "audit", "rain")
# Taken from the current (user) config: they describe the machine running the command, not the run.
MACHINE_SECTIONS = ("resources",)
MACHINE_KEYS = (("auth", "key_file"),)
# Keys that are user decisions made AFTER a run exists: taken from the user's config.
USER_DECISION_KEYS = frozenset({"qa.acknowledged_issues"})
_MISSING = object()


def _flatten(value, prefix: str) -> dict:
    if isinstance(value, dict) and value:
        out = {}
        for k, v in value.items():
            out.update(_flatten(v, f"{prefix}.{k}"))
        return out
    return {prefix: value}


def config_differences(user_cfg: dict, run_cfg: dict) -> list[str]:
    """Dotted keys in RUN_SECTIONS whose values differ (USER_DECISION_KEYS excluded)."""
    diffs = []
    for section in RUN_SECTIONS:
        a = _flatten(user_cfg.get(section, _MISSING), section)
        b = _flatten(run_cfg.get(section, _MISSING), section)
        for key in sorted(set(a) | set(b)):
            if key in USER_DECISION_KEYS:
                continue
            if a.get(key, _MISSING) != b.get(key, _MISSING):
                diffs.append(key)
    return diffs


def load_run_context(user_cfg: dict, run_id: str | None = None, warn=None) -> tuple[dict, Path]:
    """Return (frozen run config, run folder) for `run_id` (default: the newest run).

    The frozen config wins for everything, except
    - `qa.acknowledged_issues` (a user decision made after the run), and
    - machine-specific settings: the `resources` section and `auth.key_file` (a run exported on a laptop
      may be downloaded/stacked on a cloud notebook server with its own key location and hardware).
    `auth.project` belongs to the RUN: its tasks, queue and duplicate detection live in that Earth Engine
    project. A different project in the user's config is reported and ignored.
    Differences in run-defining sections are reported via `warn` (default: stderr); they do not stop the command.
    """
    config_mod = _mod("config")
    run_path = config_mod.run_dir(user_cfg, run_id)
    run_cfg = config_mod.load_run_config(run_path)
    say = warn or (lambda m: print(m, file=sys.stderr, flush=True))

    diffs = config_differences(user_cfg, run_cfg)
    if diffs:
        say(
            f"WARNING: your config differs from the frozen config of run {run_path.name} in: "
            f"{', '.join(diffs)}.\n"
            f"         This run keeps using its frozen settings (runs/{run_path.name}/run_config.yaml). "
            "To apply these changes, create a new run (new-run)."
        )
    user_project = (user_cfg.get("auth") or {}).get("project")
    run_project = (run_cfg.get("auth") or {}).get("project")
    if user_project and run_project and user_project != run_project:
        say(
            f"WARNING: your config uses Earth Engine project {user_project!r}, but run {run_path.name} belongs to "
            f"project {run_project!r}. The run's project is used (its tasks and queue live there)."
        )

    for section in MACHINE_SECTIONS:
        if section in user_cfg:
            run_cfg[section] = user_cfg[section]
    for section, key in MACHINE_KEYS:
        value = (user_cfg.get(section) or {}).get(key)
        if value is not None:
            run_cfg.setdefault(section, {})
            if not isinstance(run_cfg[section], dict):
                run_cfg[section] = {}
            run_cfg[section][key] = value

    user_qa = user_cfg.get("qa") or {}
    if "acknowledged_issues" in user_qa:
        if not isinstance(run_cfg.get("qa"), dict):
            run_cfg["qa"] = {}
        run_cfg["qa"]["acknowledged_issues"] = list(user_qa.get("acknowledged_issues") or [])
    return run_cfg, run_path


def _state_counts(manifest) -> str:
    if manifest is None or len(manifest) == 0:
        return "(manifest is empty)"
    counts = manifest["state"].value_counts().sort_index()
    return ", ".join(f"{state}={n}" for state, n in counts.items())


def require_tracks(cfg: dict) -> list[str]:
    """Selected track ids from s1.tracks; raises AuditRequired when none are selected."""
    tracks = (cfg.get("s1") or {}).get("tracks") or []
    if not tracks:
        errors = _mod("errors")
        raise errors.AuditRequired(
            "s1.tracks is empty in the config. Review the audit (track_summary.csv, "
            "track_recommendation.csv, gaps_report.md) and fill s1.tracks first."
        )
    return [t["track_id"] for t in tracks]


# ---------------------------------------------------------------- handlers
def cmd_grid(cfg: dict, args) -> int:
    grid, index = _mod("grid"), _mod("index")
    gd = grid.build_grid(cfg)
    path = index.build_pixel_index(cfg, gd)
    _print(f"Grid: {gd['width']} x {gd['height']} px at {gd['res']} m in {gd['crs']}, "
           f"{len(gd['chunks'])} chunks (chunk_px={gd['chunk_px']}), pid dtype {gd['pid_dtype']}")
    _print(f"Grid folder : {_mod('config').grid_dir(cfg)}")
    _print(f"Pixel index : {path}")
    return EXIT_OK


def cmd_audit(cfg: dict, args) -> int:
    _mod("auth").init_ee(cfg)
    kwargs = {"force": args.force}
    if args.audit_date:
        kwargs["audit_date"] = args.audit_date
    folder = _mod("audit").run_audit(cfg, **kwargs)
    _print(f"Audit written to: {folder}")
    _print("Next: review track_summary.csv, track_recommendation.csv and gaps_report.md, "
           "then fill s1.tracks in your config.")
    return EXIT_OK


def cmd_new_run(cfg: dict, args) -> int:
    track_ids = require_tracks(cfg)  # a run without selected tracks cannot be exported
    path = _mod("config").new_run(cfg)
    _print(f"New run created: {path}")
    _print(f"Tracks        : {', '.join(track_ids)}")
    _print("Your config is now frozen in run_config.yaml; later edits to run settings need another new-run.")
    return EXIT_OK


def _count(manifest, states) -> int:
    return int(manifest["state"].isin(list(states)).sum()) if len(manifest) else 0


def _submission_lock(run_path: Path):
    """The run's "monitor" lock: while a monitor runs for this run, `export --yes` must not submit too.

    Taken before anything is confirmed, so a refused command changes nothing (the monitor would otherwise
    start the newly confirmed rows while this command also tries to).
    """
    return _mod("locking").run_lock(run_path, "monitor")


def cmd_export(cfg: dict, args) -> int:
    export = _mod("export")
    cfg, run_path = load_run_context(cfg, args.run)
    _mod("auth").init_ee(cfg)
    _print(f"Run          : {run_path.name}")
    if args.retry_failed:
        manifest = _mod("manifest").read_manifest(run_path)
        n_failed = _count(manifest, ["FAILED"])
        _print(f"FAILED rows  : {n_failed} would be put back to PENDING and exported again")
        if not args.yes:
            _confirm_hint("export --retry-failed")
            return EXIT_CONFIRM
        lock = _submission_lock(run_path)
        try:
            export.retry_failed(run_path, cfg=cfg)
            manifest = export.submit_pending(cfg, run_path, confirmed=True)
        finally:
            lock.release()
        _print(f"Submitted. Manifest: {_state_counts(manifest)}")
        _print("Next: `monitor --yes` to track tasks and apply the retry policy.")
        return EXIT_OK

    chunk_names = [args.pilot] if args.pilot else None
    manifest = export.plan_exports(cfg, run_path, chunk_names=chunk_names, force=args.force)
    scope_rows = manifest if chunk_names is None else manifest[manifest["chunk_name"].isin(chunk_names)]
    planned = scope_rows[scope_rows["state"] == "PLANNED"] if len(scope_rows) else scope_rows
    scope = f"pilot chunk {args.pilot}" if args.pilot else "all chunks"
    _print(f"Scope        : {scope}")
    _print(f"To export    : {len(planned)} tasks  (tracks: "
           f"{', '.join(sorted(set(planned['track_id']))) if len(planned) else '-'})")
    _print(f"Not covered  : {_count(scope_rows, ['NOT_COVERED'])} (track never covers the chunk; never exported)")
    _print(f"Manifest     : {_state_counts(manifest)}")
    if not args.yes:
        _print("Planned rows stay PLANNED: nothing is exported until this command runs with --yes.")
        _confirm_hint("export")
        return EXIT_CONFIRM
    lock = _submission_lock(run_path)
    try:
        export.confirm_exports(cfg, run_path, chunk_names=chunk_names)
        manifest = export.submit_pending(cfg, run_path, confirmed=True)
    finally:
        lock.release()
    _print(f"Submitted. Manifest: {_state_counts(manifest)}")
    _print("Next: `monitor --yes` to track tasks and apply the retry policy.")
    return EXIT_OK


def cmd_monitor(cfg: dict, args) -> int:
    export, manifest_mod = _mod("export"), _mod("manifest")
    cfg, run_path = load_run_context(cfg, args.run)
    manifest = manifest_mod.read_manifest(run_path)
    _print(f"Run      : {run_path.name}")
    _print(f"Manifest : {_state_counts(manifest)}")
    _print(f"Pending  : {_count(manifest, ['PENDING'])} confirmed rows waiting to be submitted")
    _print(f"Planned  : {_count(manifest, ['PLANNED'])} rows NOT confirmed (monitor never submits them; "
           "confirm with `export --pilot/--all --yes`)")
    if not args.yes:
        _print("monitor polls Earth Engine and submits confirmed rows, including retries and split chunks.")
        _confirm_hint("monitor")
        return EXIT_CONFIRM
    _mod("auth").init_ee(cfg)
    manifest = export.monitor(cfg, run_path, confirmed=True, until_done=True)
    report = export.write_failed_report(run_path)
    _print(f"Finished. Manifest: {_state_counts(manifest)}")
    _print(f"Failed-chunk report: {report}")
    return EXIT_OK


def cmd_download(cfg: dict, args) -> int:
    download = _mod("download")
    cfg, run_path = load_run_context(cfg, args.run)
    est = download.estimate_download(cfg, run_path)
    gb = 1024**3
    _print(f"Run        : {run_path.name}")
    _print(f"Files      : {est['n_files']}")
    _print(f"Size       : {est['bytes'] / gb:.2f} GB")
    _print(f"Disk free  : {est['disk_free_bytes'] / gb:.2f} GB")
    _print(f"Fits       : {est['fits']}")
    if not args.yes:
        _confirm_hint("download")
        return EXIT_CONFIRM
    manifest = download.download_completed(cfg, run_path, confirmed=True, force=args.force,
                                           retry_failed=args.retry_failed, deep=args.deep)
    _print(f"Download finished. Manifest: {_state_counts(manifest)}")
    return EXIT_OK


def cmd_stack(cfg: dict, args) -> int:
    stack = _mod("stack")
    cfg, run_path = load_run_context(cfg, args.run)
    track_ids = [args.track] if args.track else require_tracks(cfg)
    # All tracks are built first; QA issues of all tracks are then reported together
    # (one decisions_required.md, one DecisionRequired), so one track never blocks another.
    outputs = stack.build_stacks(cfg, run_path, track_ids, force=args.force)
    for track_id, out in outputs.items():
        _print(f"Stack for {track_id}: {out}")
    return EXIT_OK


def cmd_pixel(cfg: dict, args) -> int:
    pq = _mod("pixel_query")
    cfg, run_path = load_run_context(cfg, args.run)
    if args.pid is not None:
        df = pq.pixel_timeseries(cfg, run_path, args.track, args.pid)
        title = f"{args.track} pid {args.pid}"
    else:
        df = pq.pixel_timeseries_lonlat(cfg, run_path, args.track, args.lon, args.lat)
        title = f"{args.track} lon {args.lon:.5f} lat {args.lat:.5f}"
    _print(df.to_string(index=False))
    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        ax = pq.plot_timeseries(df, cfg=cfg, title=title)
        ax.figure.savefig(args.plot, dpi=150, bbox_inches="tight")
        _print(f"Plot saved: {args.plot}")
    return EXIT_OK


def cmd_resources(cfg: dict, args) -> int:
    r = _mod("resources")
    res = r.detect_resources(cfg)
    gb = 1024**3
    season = _mod("config").season_dir(cfg)
    _print(f"Machine          : {res.describe()}")
    _print(f"Memory budget    : {r.memory_budget_bytes(res, cfg) / gb:.2f} GB (resources.memory_fraction)")
    _print(f"Download threads : {r.io_threads(res, cfg)}")
    _print(f"CPU workers      : {r.cpu_workers(res, cfg)}")
    _print(f"EE submit threads: {r.submit_threads(res, cfg)}")
    _print(f"GDAL cache       : {r.gdal_cache_bytes(res) / 1024**2:.0f} MB")
    _print(f"Disk free        : {r.disk_free_bytes(season) / gb:.1f} GB (at {season})")
    return EXIT_OK


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sar_pipeline",
        description="Sentinel-1 SAR preprocessing + time-series feature pipeline (Google Earth Engine).",
    )
    p.add_argument("--config", required=True, help="Path to the pipeline YAML config.")
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    s = sub.add_parser("grid", help="Build the master grid, chunks and pixel index (local).")
    s.set_defaults(func=cmd_grid)

    s = sub.add_parser("audit", help="Sentinel-1 availability audit (Earth Engine metadata only, no exports).")
    s.add_argument("--audit-date", metavar="YYYYMMDD", help="Resume/refresh this existing audit folder instead of today's.")
    s.add_argument("--force", action="store_true", help="Recompute every audit file in the audit folder.")
    s.set_defaults(func=cmd_audit)

    s = sub.add_parser("new-run", help="Create a new immutable run folder runs/v<NNN>_<date>.")
    s.set_defaults(func=cmd_new_run)

    s = sub.add_parser("export", help="Plan and (with --yes) confirm and start chunked Earth Engine exports.")
    s.add_argument("--run", help="Run id (default: the newest run).")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--pilot", metavar="CHUNK", help="Export only this chunk (e.g. chunk_r03c01).")
    g.add_argument("--all", action="store_true", help="Export every chunk.")
    g.add_argument("--retry-failed", action="store_true",
                   help="Put FAILED export rows back to PENDING and submit them again.")
    s.add_argument("--yes", action="store_true", help="Confirm the scope and actually start the exports.")
    s.add_argument("--force", action="store_true",
                   help="Rebuild the export PLAN (band layout, pending rows). Never re-submits finished tasks.")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("monitor", help="Poll export tasks, apply retry/split policy, write failed_chunks.csv.")
    s.add_argument("--run")
    s.add_argument("--yes", action="store_true", help="Allow polling and re-submission.")
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser("download", help="Download completed chunks from GCS and verify them.")
    s.add_argument("--run")
    s.add_argument("--yes", action="store_true", help="Actually download.")
    s.add_argument("--force", action="store_true", help="Re-download and re-verify files already marked VERIFIED.")
    s.add_argument("--retry-failed", action="store_true", help="Try FAILED_DOWNLOAD rows again.")
    s.add_argument("--deep", action="store_true", help="Re-hash every local file instead of trusting size + mtime.")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("stack", help="Build per-date and multi-date VRT stacks + QA.")
    s.add_argument("--run")
    s.add_argument("--track", help="Track id (default: every track in s1.tracks).")
    s.add_argument("--force", action="store_true", help="Rebuild VRTs, dates.csv and QA for this run.")
    s.set_defaults(func=cmd_stack)

    s = sub.add_parser("pixel", help="Print (and optionally plot) the time series of one pixel.")
    s.add_argument("--run")
    s.add_argument("--track", required=True)
    s.add_argument("--pid", type=int)
    s.add_argument("--lon", type=float)
    s.add_argument("--lat", type=float)
    s.add_argument("--plot", metavar="PNG", help="Save a plot to this file.")
    s.set_defaults(func=cmd_pixel)

    s = sub.add_parser("resources", help="Show detected machine resources and derived worker counts.")
    s.set_defaults(func=cmd_resources)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "export" and not (args.pilot or args.all or args.retry_failed):
        parser.error("export: give --pilot CHUNK, --all or --retry-failed")

    if args.command == "pixel":
        has_pid = args.pid is not None
        has_lonlat = args.lon is not None and args.lat is not None
        if has_pid == has_lonlat or (not has_pid and (args.lon is None) != (args.lat is None)):
            parser.error("pixel: give either --pid, or both --lon and --lat")

    errors = _mod("errors")
    try:
        cfg = _mod("config").load_config(args.config)
        return int(args.func(cfg, args))
    except errors.PipelineError as exc:
        print(f"ERROR ({type(exc).__name__}): {exc}", file=sys.stderr)
        return EXIT_ERROR
    except FileNotFoundError as exc:
        print(f"ERROR (FileNotFoundError): {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
