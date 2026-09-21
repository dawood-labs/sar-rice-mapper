"""Stage 3 — Chunked Earth Engine exports with queue throttling, retries and crash-safe resume.

What happens here
-----------------
1. ``plan_exports`` freezes *what* this run exports: the band layout (which acquisition/polarisation
   is in which band) and one PLANNED manifest row per (track, chunk). Chunks a track never covers
   (audit ``chunk_track_coverage.csv``) become NOT_COVERED and are never exported.
2. ``confirm_exports`` turns the PLANNED rows of a chosen scope (a pilot chunk, or everything) into
   PENDING. Only PENDING rows are ever submitted, so a dry run can never be exported by accident.
3. ``submit_pending`` runs one *submission round*:
   a. one project-wide task listing gives the queue size, task states AND existing tasks by description;
   b. interrupted submissions (SUBMITTING rows) are adopted if their task or output exists;
   c. at most ``headroom`` PENDING rows are started — only with ``confirmed=True``.
4. ``refresh_status`` updates submitted rows from the same listing and applies the retry policy.
5. ``monitor`` repeats 4 + 3 until nothing is left to do. All state lives in the manifest, so it can be
   stopped and restarted at any time ("resume, never redo"). Only one monitor per run can run at once.

Grid alignment (the most important detail)
------------------------------------------
Every task uses the MASTER grid ``crsTransform`` and a region slightly inset inside the chunk.
Earth Engine then writes exactly the chunk's pixels on the master grid, so all chunk files line up
without resampling and pixel ids are identical across chunks and runs.

Queue throttling
----------------
An Earth Engine project accepts at most ``export.max_queued_tasks`` queued tasks (READY + RUNNING),
shared with every other job on that project. Before EVERY round we count them project-wide and submit
at most ``floor(max_queued_tasks * queue_safety_margin) - queued`` new tasks (and, if
``export.max_active_tasks`` is an integer, no more than that many rows of this run active at once).
How many tasks Earth Engine actually RUNS in parallel depends on the account tier; it is measured
each round and used for the ETA, never assumed.

Crash safety and duplicate prevention
-------------------------------------
- Only one process submits for a run at a time: ``submit_pending`` holds the run's "monitor" lock, and a
  row is *claimed* PENDING -> SUBMITTING with a compare-and-set, so it can never be started twice.
- The claim records ``submitted_at`` *before* ``task.start()``. If ``start()`` raises, the request may still
  have reached Earth Engine (network timeout, 5xx). The row stays SUBMITTING. The next round looks for the
  task by its unique description (created after ``submitted_at``) and adopts it; only if no task and no
  output appear within ``SUBMITTING_GRACE_SECONDS`` of the last activity is the row submitted again.
- Rows with ``adopted = -1`` must be a FRESH export (their earlier output failed verification): they never
  adopt tasks or GCS files older than their own ``submitted_at``.
- The run is validated first (Earth Engine project, grid): the wrong project's listing would make running
  tasks look lost and export them twice.
- Descriptions and GCS prefixes contain the run's unique id (``run_uid.txt``), so a run on another
  machine can never adopt this run's tasks or files.

Retry policy (docs/developer/interfaces.md §6.4)
------------------------------------------------
- submission rejected with ``quota`` ("Too many tasks", rate limits) -> backoff + jitter, up to
  ``submit_max_attempts``; then PENDING for a later round (does not consume ``task_max_attempts``)
- task FAILED ``memory``/``timeout``      -> split the chunk into 4 sub-chunks (once); sub-chunk fails -> FAILED
- task FAILED ``quota``/``transient``     -> back to PENDING while attempts < ``task_max_attempts``
- ``cancelled``/``other``                 -> FAILED (``retry_failed`` puts FAILED rows back on request)
- a task that disappears (``lost``)        -> SUBMITTING lookup, then resubmitted, bounded by ``task_max_attempts``
"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from . import manifest as mf
from . import resources, run_meta
from .config import gcs_prefix, latest_audit_dir, root
from .errors import AuditRequired, ConfirmationRequired, PipelineError
from .locking import run_lock
# Re-exported for callers that used to import these names from export:
from .run_meta import BAND_LAYOUT, BAND_LAYOUT_COLUMNS, EXPORT_META, config_hash, read_band_layout  # noqa: F401

log = logging.getLogger(__name__)

FAILED_REPORT = "failed_chunks.csv"
FAILED_COLUMNS = ["task_key", "track_id", "chunk_name", "state", "attempts", "error_class", "last_error", "gcs_prefix"]
_STATUS_LOOKUP_MAX = 50        # getTaskStatus fallback sends one HTTP request per id: bound it per round
_DESC_MAX = 100
_MIN_COMPLETED_FOR_ETA = 3
_ADOPTABLE = {"COMPLETED": 3, "RUNNING": 2, "READY": 1}   # preference when several tasks share a description
SUBMITTING_GRACE_SECONDS = 600  # an interrupted submission is re-sent only after this long without a task
UNKNOWN_POLLS_TO_LOOKUP = 3     # a submitted task missing this many polls in a row is looked up by description
CREATE_TIME_SKEW_SECONDS = 120  # clock skew allowed when matching tasks/GCS files to a row's submitted_at;
                                # must equal download.STALE_BLOB_TOLERANCE_S (a test enforces it) so both agree
BLOB_INDEX_TTL_SECONDS = 900    # re-list a track's GCS folder at most this often per session
_OPEN_STATES = ["PENDING", *mf.OWN_ACTIVE_STATES]

# Summary of the most recent submission round (for notebooks / CLI display).
LAST_ROUND: dict = {}


def _utcnow() -> datetime:
    """Current UTC time (a function so tests can move the clock)."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------- project task listing
_ACTIVE_OP_STATES = {"PENDING": "READY", "RUNNING": "RUNNING", "CANCELLING": "CANCEL_REQUESTED"}
_TERMINAL_OP_STATES = {"SUCCEEDED": "COMPLETED", "FAILED": "FAILED", "CANCELLED": "CANCELLED"}
LISTING_PAGE_SIZE = 500


def _parse_utc(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except (ValueError, TypeError):
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def scan_operations(fetch_page: Callable[[str | None], dict], cutoff: datetime | None = None) -> dict:
    """Read the project's operation list page by page and stop as early as it is safe.

    Why not ``ee.data.listOperations()``? It follows EVERY page. A busy project keeps tens of thousands
    of finished operations (observed: 71,273 -> 88 s per call), and we need the list on every poll.

    What we need from the listing:
    - the number of ACTIVE operations (queued + running), for queue throttling;
    - the state and error of THIS run's tasks (so status polling needs no per-task requests);
    - the descriptions of THIS run's tasks, for crash recovery (adoption).

    The API returns active operations first, then finished ones newest-first by ``createTime``. So once
    a page contains a finished operation created before ``cutoff`` (older than anything this run could
    have submitted), every active operation and every task of this run has been seen: stop.

    We do not trust that ordering blindly: if an active operation appears after a finished one, a warning
    is logged and paging continues to the end (full listing) for this call.

    ``fetch_page(page_token) -> {"operations": [...], "nextPageToken": "..."}``.
    Returns ``{"counts": {"READY", "RUNNING"}, "tasks": [{id, description, state, create_time, error_message}],
    "stats": {"pages", "operations", "stopped_early", "full_fallback", "seconds"}}``.
    """
    t0 = time.monotonic()
    counts = {"READY": 0, "RUNNING": 0}
    tasks: list[dict] = []
    pages = n_ops = 0
    seen_terminal = order_violation = stopped_early = False
    token: str | None = None
    while True:
        resp = fetch_page(token) or {}
        pages += 1
        older_than_cutoff = False
        for op in resp.get("operations", []) or []:
            n_ops += 1
            md = op.get("metadata") or {}
            raw = md.get("state", "")
            created = _parse_utc(md.get("createTime"))
            if raw in _ACTIVE_OP_STATES:
                state = _ACTIVE_OP_STATES[raw]
                if seen_terminal and not order_violation:
                    order_violation = True
                    log.warning("task listing: active operation found after finished ones (API ordering changed?); "
                                "reading the full listing this round")
                counts["READY" if state == "READY" else "RUNNING"] += 1
            else:
                state = _TERMINAL_OP_STATES.get(raw, raw)
                seen_terminal = True
                if cutoff is not None and created is not None and created < cutoff:
                    older_than_cutoff = True
            tasks.append(dict(id=str(op.get("name", "")).rsplit("/", 1)[-1], description=md.get("description", ""),
                              state=state, create_time=md.get("createTime", ""),
                              error_message=((op.get("error") or {}).get("message", "") or "")))
        token = resp.get("nextPageToken")
        if not token:
            break
        if older_than_cutoff and not order_violation:
            stopped_early = True
            break
    stats = dict(pages=pages, operations=n_ops, stopped_early=stopped_early, full_fallback=order_violation,
                 seconds=round(time.monotonic() - t0, 2))
    return dict(counts=counts, tasks=tasks, stats=stats)


# ---------------------------------------------------------------- backend
class EEBackend:
    """Thin wrapper around the Earth Engine task API (tests substitute a fake with the same methods).

    - Task states come from the project listing (``project_snapshot``), read with bounded pagination
      and cached for ``max_age`` seconds. Tasks started through this backend are added to the cached
      snapshot, so a cached count never under-counts our own submissions.
    - ``task_statuses`` (``ee.data.getTaskStatus``: one HTTP request per id) is only a bounded fallback
      for tasks missing from the listing.
    """

    def __init__(self, page_size: int = LISTING_PAGE_SIZE):
        self.page_size = int(page_size)
        self._lock = threading.Lock()
        self._cache: dict | None = None      # {"at": monotonic, "cutoff": datetime|None, "snapshot": {...}}

    def start_export(self, image, params: dict) -> str:
        import ee

        p = dict(params)
        coords = p.pop("region")
        p["region"] = ee.Geometry.Rectangle(coords, proj=p["crs"], geodesic=False)
        task = ee.batch.Export.image.toCloudStorage(image=image, **p)
        task.start()
        with self._lock:
            if self._cache is not None:
                snap = self._cache["snapshot"]
                snap["counts"]["READY"] += 1
                snap["tasks"].append(dict(id=task.id, description=params.get("description", ""), state="READY",
                                          create_time=mf.now_utc(), error_message=""))
        return task.id

    def task_statuses(self, task_ids: list[str]) -> dict[str, dict]:
        import ee

        out: dict[str, dict] = {}
        for i in range(0, len(task_ids), _STATUS_LOOKUP_MAX):
            for st in ee.data.getTaskStatus(task_ids[i:i + _STATUS_LOOKUP_MAX]):
                out[st.get("id")] = {"state": st.get("state", "UNKNOWN"), "error_message": st.get("error_message", "")}
        return out

    def _fetch_page(self, token: str | None) -> dict:
        import ee

        kwargs = dict(name=ee.data._get_projects_path(), pageSize=self.page_size)
        if token:
            kwargs["pageToken"] = token
        return ee.data._execute_cloud_call(ee.data._get_cloud_projects().operations().list(**kwargs))

    def project_snapshot(self, cutoff: datetime | None = None, max_age: float = 0.0) -> dict:
        """Project-wide {"counts": {"READY", "RUNNING"}, "tasks": [...], "stats": {...}}.

        ``cutoff``: stop paging after finished operations older than this (None = read everything).
        ``max_age``: re-use a snapshot younger than this many seconds, if it was read with a cutoff
        at least as old as the requested one.
        """
        with self._lock:
            c = self._cache
            if (c is not None and max_age > 0 and time.monotonic() - c["at"] < max_age
                    and (c["cutoff"] is None or (cutoff is not None and c["cutoff"] <= cutoff))):
                return c["snapshot"]
        snap = scan_operations(self._fetch_page, cutoff)
        st = snap["stats"]
        log.info("project task listing: %d operations in %d pages, %.1fs (stopped_early=%s, full_fallback=%s)",
                 st["operations"], st["pages"], st["seconds"], st["stopped_early"], st["full_fallback"])
        with self._lock:
            self._cache = dict(at=time.monotonic(), cutoff=cutoff, snapshot=snap)
        return snap


# ---------------------------------------------------------------- monitor session (per-process caches)
@dataclass
class MonitorSession:
    """Things loaded once per monitor session instead of once per round (grid, audit, GCS listings)."""
    gd: dict | None = None
    selected: dict | None = None
    blob_index: dict = field(default_factory=dict)      # track -> {"at": monotonic, "names": {base: [names]}}
    unknown_polls: dict = field(default_factory=dict)   # task_key -> consecutive polls without a status


_SESSIONS: dict[str, MonitorSession] = {}
_SESSIONS_GUARD = threading.Lock()


def _session(run_dir: Path, session: MonitorSession | None) -> MonitorSession:
    if session is not None:
        return session
    with _SESSIONS_GUARD:
        return _SESSIONS.setdefault(str(Path(run_dir).resolve()), MonitorSession())


# ---------------------------------------------------------------- helpers
def task_key(track_id: str, chunk_name: str) -> str:
    return f"{track_id}__{chunk_name}"


def split_task_key(key: str) -> tuple[str, str]:
    track_id, _, chunk_name = key.partition("__")
    return track_id, chunk_name


def nodata_value(cfg: dict) -> float | int:
    exp = cfg["export"]
    dtype = exp.get("dtype", "float32")
    if dtype == "float32":
        return exp.get("nodata_float32", -9999)
    if dtype == "int16":
        return exp.get("nodata_int16", -32768)
    raise PipelineError(f"export.dtype must be float32 or int16, got {dtype!r}")


def sanitise_description(text: str) -> str:
    """EE task descriptions allow letters, digits, space and . , : ; _ - and at most 100 characters.

    Long names are shortened but keep a hash of the full text, so descriptions stay unique.
    """
    clean = re.sub(r"[^A-Za-z0-9 .,:;_-]", "_", text)
    if len(clean) <= _DESC_MAX:
        return clean
    digest = hashlib.sha1(text.encode()).hexdigest()[:10]
    return clean[: _DESC_MAX - 11] + "_" + digest


def task_description(cfg: dict, run_uid: str, key: str) -> str:
    """Unique task description: aoi, season, the run's unique id and the task key."""
    return sanitise_description(f"{cfg['aoi']['key']}_{cfg['season']['key']}_{run_uid}_{key}")


_TRANSIENT_PATTERNS = ("internal error", "service unavailable", "backend error", "unavailable", "deadline",
                       "connection reset", "connection aborted", "broken pipe", "temporarily", "try again",
                       "read timed out", "read timeout", "connect timeout", "connection timed out",
                       " 500", " 502", " 503", " 504", "http 5", "error 5")


def classify_error(message: str) -> str:
    """Map an Earth Engine error message to: memory | timeout | quota | transient | cancelled | other.

    ``timeout`` means the COMPUTATION was too big ("Computation timed out"): splitting the chunk helps.
    ``transient`` means the service or the network had a problem (5xx, deadline exceeded, read timed out):
    simply retrying helps. Network time-outs are checked before computation time-outs for that reason.
    """
    m = (message or "").lower()
    if "cancel" in m:
        return "cancelled"
    if "memory" in m:
        return "memory"
    if any(s in m for s in ("quota", "too many", "rate limit", "429", "resource exhausted",
                            "resource_exhausted", "queue")):
        return "quota"
    if any(s in m for s in _TRANSIENT_PATTERNS):
        return "transient"
    if "timed out" in m or "timeout" in m:
        return "timeout"
    return "other"


def _backoff_seconds(attempt: int) -> float:
    return min(60.0, 2.0 ** attempt) + random.uniform(0, 1)


def _chunk_for_row(gd: dict, chunk_name: str) -> dict:
    from . import grid

    try:
        return grid.chunk_by_name(gd, chunk_name)
    except KeyError as exc:
        raise PipelineError(f"chunk {chunk_name!r} is not in the grid") from exc


def check_run(cfg: dict, run_dir: str | Path) -> None:
    """Validate the run against the config BEFORE its manifest is touched (keys, grid, Earth Engine project).

    Tasks, queue counts and duplicate detection all live in the run's Earth Engine project. Working on a
    run with another project's task listing would mark its running tasks as lost and export them again.
    """
    if (Path(run_dir) / EXPORT_META).exists():
        run_meta.read_export_meta(cfg, run_dir)


def _band_layout_rows(cfg: dict, selected: dict[str, pd.DataFrame]) -> list[dict]:
    pols = list(cfg["s1"].get("pols", ["VV", "VH"]))
    rows = []
    for track_id in sorted(selected):
        acq = selected[track_id].sort_values("datetime_utc")
        idx = 1
        for _, a in acq.iterrows():
            for pol in pols:
                rows.append(dict(
                    track_id=track_id, band_idx=idx, acquisition_id=a["acquisition_id"],
                    date_utc=str(a["date_utc"]), datetime_utc=str(a["datetime_utc"]),
                    date_local=str(a.get("date_local", "")), pol=pol,
                    platforms=str(a.get("platforms", "")), n_slices=int(a.get("n_slices", 0) or 0),
                ))
                idx += 1
    return rows


_SUB_SUFFIX = re.compile(r"_s\d+$")


def _coverage_checker(audit_path: Path) -> Callable[[str, str], bool]:
    """Returns not_covered(track_id, chunk_name) from the audit's chunk_track_coverage.csv.

    The audit lists (chunk, track) pairs with at least one acquisition. A pair missing for a track that
    appears in the file means the track never covers that chunk. Without the file nothing is excluded.
    """
    path = Path(audit_path) / "chunk_track_coverage.csv"
    if not path.exists():
        log.warning("%s not found: every (track, chunk) is planned; uncovered chunks will export as nodata", path)
        return lambda track_id, chunk_name: False
    cov = pd.read_csv(path, dtype={"chunk_name": str, "track_id": str}, keep_default_na=False)
    covered = {(t, c) for t, c, n in zip(cov["track_id"], cov["chunk_name"], cov["n_acq"]) if int(n) > 0}
    known = set(cov["track_id"])
    return lambda track_id, chunk_name: track_id in known and (track_id, _SUB_SUFFIX.sub("", chunk_name)) not in covered


def _same_meta(old: dict, new: dict) -> bool:
    for k, v in new.items():
        if k == "created_utc":
            continue
        if k not in old and k in run_meta.OPTIONAL_META_KEYS:
            continue  # planned before this key existed
        if old.get(k) != v:
            return False
    return True


# ---------------------------------------------------------------- planning
def plan_exports(cfg: dict, run_dir: str | Path, chunk_names: list[str] | None = None,
                 force: bool = False) -> pd.DataFrame:
    """Freeze the band layout and add PLANNED rows for every (selected track × chunk).

    Nothing is exported by planning: PLANNED rows must be confirmed (``confirm_exports``) first.
    (track, chunk) pairs the track never covers are added as NOT_COVERED.

    Idempotent: existing rows are untouched; a second call only adds missing rows (e.g. the remaining
    chunks after a pilot) and rewrites nothing when the layout is unchanged.

    If the selected acquisitions / grid changed since the first call, the stored layout no longer
    describes the planned tasks: raises PipelineError. ``force=True`` rewrites the layout, but only while
    no row has been submitted (tasks exported with the old layout would disagree with it).
    """
    from . import audit, grid

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_meta.clean_stale_tmp(run_dir)
    gd = grid.load_grid(cfg)
    meta_path, layout_path = run_dir / EXPORT_META, run_dir / BAND_LAYOUT
    old = json.loads(meta_path.read_text()) if meta_path.exists() else None
    # Re-use the audit this run was planned with, so a newer audit never silently changes the layout.
    if old is not None:
        audit_path = root(cfg) / old["audit_dir"]
    else:
        try:
            audit_path = latest_audit_dir(cfg)
        except FileNotFoundError as exc:
            raise AuditRequired(str(exc)) from exc
    selected = audit.selected_acquisitions(cfg, audit_path)
    layout = pd.DataFrame(_band_layout_rows(cfg, selected), columns=BAND_LAYOUT_COLUMNS)
    if layout.empty:
        raise PipelineError("no acquisitions selected — nothing to export")
    layout_csv = layout.to_csv(index=False)

    uid = run_meta.run_uid(run_dir)
    meta = dict(
        run_id=run_dir.name,
        dtype=cfg["export"].get("dtype", "float32"),
        nodata=nodata_value(cfg),
        int16_scale=cfg["export"].get("int16_scale", 100),
        grid_crs=gd["crs"],
        grid_transform=list(gd["transform"]),
        grid_fingerprint=run_meta.grid_fingerprint(cfg),
        audit_dir=run_meta.relative_to_root(cfg, audit_path),
        tracks=sorted(selected),
        config_hash=config_hash(cfg),
        created_utc=mf.now_utc(),
        ee_project=(cfg.get("auth") or {}).get("project", ""),
        run_uid=uid,
    )
    with mf.lock(run_dir):
        manifest = mf.read_manifest(run_dir)
        if old is not None:
            same_meta = _same_meta(old, meta)
            layout_ok = layout_path.exists() and layout_path.read_text() == layout_csv
            if same_meta and not layout_path.exists():
                run_meta.atomic_write_text(layout_path, layout_csv)   # deleted layout rebuilt from identical inputs
            elif not (same_meta and layout_ok):
                progressed = manifest[~manifest["state"].isin(["PLANNED", "PENDING", "NOT_COVERED"])]
                if not force:
                    raise PipelineError(
                        f"{meta_path} / {BAND_LAYOUT} were planned with a different band layout, grid, audit or "
                        "export settings. A run is immutable; create a new run (config.new_run), or use "
                        "force=True if nothing was exported yet."
                    )
                if not progressed.empty:
                    raise PipelineError(
                        f"force=True refused: {len(progressed)} rows of this run were already submitted with the "
                        "old layout. Create a new run instead."
                    )
                log.warning("plan_exports(force=True): rewriting export_meta.json and band_layout.csv")
                run_meta.atomic_write_text(layout_path, layout_csv)
                run_meta.atomic_write_json(meta_path, meta)
        else:
            run_meta.atomic_write_text(layout_path, layout_csv)       # layout first, meta last (commit marker)
            run_meta.atomic_write_json(meta_path, meta)

        chunks = list(grid.iter_chunks(gd)) if chunk_names is None else [_chunk_for_row(gd, n) for n in chunk_names]
        not_covered = _coverage_checker(audit_path)
        existing = set(manifest["task_key"])
        n_bands_by_track = layout.groupby("track_id").size().to_dict()
        new_rows = []
        for track_id in sorted(selected):
            for ch in chunks:
                if task_key(track_id, ch["name"]) in existing:
                    continue
                state = "NOT_COVERED" if not_covered(track_id, ch["name"]) else "PLANNED"
                new_rows.append(_planned_row(cfg, run_dir, track_id, ch, int(n_bands_by_track[track_id]), "", state))
        if new_rows:
            mf.update_manifest(run_dir, new_rows, return_df=False)
    n_nc = sum(r["state"] == "NOT_COVERED" for r in new_rows)
    log.info("plan_exports: %d new rows (%d not covered by their track), %d already planned",
             len(new_rows), n_nc, len(chunks) * len(selected) - len(new_rows))
    mf.compact(run_dir)
    return mf.read_manifest(run_dir)


def _planned_row(cfg: dict, run_dir: Path, track_id: str, ch: dict, n_bands: int, parent: str,
                 state: str = "PLANNED") -> dict:
    key = task_key(track_id, ch["name"])
    uid = run_meta.run_uid(run_dir)
    return dict(
        task_key=key, track_id=track_id, chunk_name=ch["name"], parent_chunk=parent,
        row_off=int(ch["row_off"]), col_off=int(ch["col_off"]), width=int(ch["width"]), height=int(ch["height"]),
        n_bands=n_bands,
        gcs_prefix=gcs_prefix(cfg, "runs", uid, "raw_chunks", f"track_{track_id}", ch["name"]),
        description=task_description(cfg, uid, key),
        task_id="", state=state, attempts=0, last_error="", error_class="",
        submitted_at="", local_paths="", adopted=0,
    )


def confirm_exports(cfg: dict, run_dir: str | Path, chunk_names: list[str] | None = None) -> pd.DataFrame:
    """The user's confirmation of a scope: PLANNED rows of these chunks (None = all) become PENDING.

    Only PENDING rows are submitted. Rows outside the scope stay PLANNED, so confirming a pilot never
    releases the rest of a full plan.
    """
    run_dir = Path(run_dir)
    with mf.lock(run_dir):
        df = mf.read_manifest(run_dir)
        sel = df[df["state"] == "PLANNED"]
        if chunk_names is not None:
            sel = sel[sel["chunk_name"].isin(list(chunk_names))]
        if not sel.empty:
            mf.update_manifest(run_dir, [dict(task_key=k, state="PENDING") for k in sel["task_key"]], return_df=False)
    log.info("confirm_exports: %d planned rows confirmed for export (scope: %s)",
             len(sel), "all" if chunk_names is None else ", ".join(chunk_names))
    mf.compact(run_dir)
    return mf.read_manifest(run_dir)


def retry_failed(run_dir: str | Path, classes: list[str] | None = None, cfg: dict | None = None) -> pd.DataFrame:
    """Put FAILED rows (optionally only these error classes) back to PENDING for another export attempt.

    Pass ``cfg`` to validate the run (Earth Engine project, grid) before the manifest is changed.
    """
    run_dir = Path(run_dir)
    if cfg is not None:
        check_run(cfg, run_dir)
    with mf.lock(run_dir):
        df = mf.read_manifest(run_dir)
        sel = df[df["state"] == "FAILED"]
        if classes is not None:
            sel = sel[sel["error_class"].isin(list(classes))]
        updates = [dict(task_key=r.task_key, state="PENDING", task_id="", error_class="",
                        last_error=f"retry requested (previous: {r.error_class}: {r.last_error})"[:1000])
                   for r in sel.itertuples()]
        if updates:
            mf.update_manifest(run_dir, updates, return_df=False)
    log.info("retry_failed: %d FAILED rows back to PENDING", len(updates))
    mf.compact(run_dir)
    return mf.read_manifest(run_dir)


def export_params(cfg: dict, gd: dict, chunk: dict, task_key: str, run_uid: str) -> dict:
    """Serialisable export parameters. ``region`` is [xmin, ymin, xmax, ymax] in the grid CRS
    (EEBackend turns it into an ee.Geometry.Rectangle with proj=crs, geodesic=False)."""
    from . import grid

    track_id, chunk_name = split_task_key(task_key)
    return dict(
        description=task_description(cfg, run_uid, task_key),
        bucket=cfg["gcs"]["bucket"],
        fileNamePrefix=gcs_prefix(cfg, "runs", run_uid, "raw_chunks", f"track_{track_id}", chunk_name),
        region=grid.chunk_region_coords(chunk),
        crs=gd["crs"],
        crsTransform=list(gd["transform"]),  # MASTER grid transform, never a per-chunk one
        maxPixels=1e13,
        fileFormat="GeoTIFF",
        formatOptions={"noData": nodata_value(cfg)},
    )


# ---------------------------------------------------------------- throttling + ETA
def queue_headroom(cfg: dict, project_counts: dict, own_active: int) -> int:
    """How many tasks may be queued now: project queue headroom, capped by an optional own limit."""
    exp = cfg["export"]
    limit = math.floor(float(exp.get("max_queued_tasks", 3000)) * float(exp.get("queue_safety_margin", 0.9)))
    queued = int(project_counts.get("READY", 0)) + int(project_counts.get("RUNNING", 0))
    headroom = limit - queued
    own_cap = exp.get("max_active_tasks", "auto")
    if isinstance(own_cap, int) and not isinstance(own_cap, bool):
        headroom = min(headroom, own_cap - own_active)
    return max(0, int(headroom))


def median_task_seconds(df: pd.DataFrame) -> float | None:
    """Median submitted_at -> completion time of this run's COMPLETED rows (None if fewer than 3)."""
    done = df[(df["state"] == "COMPLETED") & (df["submitted_at"] != "") & (df["updated_at"] != "")]
    if len(done) < _MIN_COMPLETED_FOR_ETA:
        return None
    start = pd.to_datetime(done["submitted_at"], utc=True, errors="coerce")
    end = pd.to_datetime(done["updated_at"], utc=True, errors="coerce")
    secs = (end - start).dt.total_seconds().dropna()
    secs = secs[secs >= 0]
    if len(secs) < _MIN_COMPLETED_FOR_ETA:
        return None
    return float(secs.median())


def estimate_eta_seconds(df: pd.DataFrame, running: int) -> float | None:
    """ETA = unfinished rows / observed concurrently running tasks * median task duration."""
    med = median_task_seconds(df)
    if med is None:
        return None
    unfinished = int(df["state"].isin(_OPEN_STATES).sum())
    return unfinished / max(1, int(running)) * med


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    h, rem = divmod(int(round(seconds)), 3600)
    return f"{h}h{rem // 60:02d}m"


# ---------------------------------------------------------------- resume: adopt existing work
_BLOB_BASE_RE = re.compile(rf"^(?P<base>.+?){run_meta.EE_PART_SUFFIX}$")


def _gcs_has_output(cfg: dict, run_dir: Path, session: MonitorSession, gcs, track_id: str, prefix: str,
                    newer_than: datetime | None = None) -> bool:
    """Whether GCS holds output files for exactly this chunk (sub-chunk files never count for a parent).

    ``newer_than``: only files created at/after this moment count (files without a creation time do not).
    Each track folder is listed once and parsed into a dict, then re-listed at most every
    BLOB_INDEX_TTL_SECONDS: lookups are O(1) no matter how many chunks the run has.
    """
    entry = session.blob_index.get(track_id)
    if entry is None or time.monotonic() - entry["at"] > BLOB_INDEX_TTL_SECONDS:
        folder = gcs_prefix(cfg, "runs", run_meta.run_uid(run_dir), "raw_chunks", f"track_{track_id}") + "/"
        names: dict[str, list[dict]] = {}
        for blob in gcs.list_blobs(folder):
            m = _BLOB_BASE_RE.match(blob["name"])
            if m:
                names.setdefault(m.group("base"), []).append(
                    dict(name=blob["name"], time_created=blob.get("time_created", "") or ""))
        entry = session.blob_index[track_id] = dict(at=time.monotonic(), names=names)
    files = entry["names"].get(prefix, [])
    if newer_than is None:
        return bool(files)
    for f in files:
        created = _parse_utc(f["time_created"])
        if created is not None and created >= newer_than:
            return True
    return False


def _adopt_existing(cfg: dict, run_dir: Path, df: pd.DataFrame, snapshot: dict, gcs,
                    session: MonitorSession) -> dict:
    """Resolve interrupted submissions (SUBMITTING rows) without ever exporting a chunk twice.

    Two timestamps matter:
    - ``submitted_at`` is written with the claim, BEFORE ``start()``. Earth Engine cannot have created this
      submission's task earlier, so older tasks and files belong to an earlier attempt. Failure updates never
      overwrite it, so a ``start()`` that raises minutes later ("Deadline exceeded") still finds its task.
    - ``updated_at`` is the last activity; the grace period before resubmitting counts from it.

    For each SUBMITTING row, in this order:
    1. its recorded task id is in the listing (READY/RUNNING/COMPLETED) -> take that state;
    2. a task with its unique description, created at/after submitted_at (minus clock skew) -> adopt it;
    3. its output files are in GCS -> COMPLETED (the download stage verifies them);
    4. still nothing SUBMITTING_GRACE_SECONDS after the last activity -> PENDING again (or FAILED when the
       attempts are used up).
    Adopted rows get ``adopted=1``. Rows with ``adopted=-1`` must be a FRESH export: they only adopt tasks
    created and files written after their submitted_at, and keep -1 until download verifies the new files.
    """
    cands = df[df["state"] == "SUBMITTING"]
    stats = dict(from_ee=0, from_gcs=0, reset=0, waiting=0, failed=0)
    if cands.empty:
        return stats
    tasks = snapshot.get("tasks", []) or []
    by_id = {str(t.get("id")): t for t in tasks}
    by_desc: dict[str, list[dict]] = {}
    for t in tasks:
        if t.get("state") in _ADOPTABLE:
            by_desc.setdefault(t.get("description", ""), []).append(t)
    task_max = int(cfg["export"].get("task_max_attempts", 3))
    now = _utcnow()
    skew = timedelta(seconds=CREATE_TIME_SKEW_SECONDS)

    updates = []
    for row in cands.itertuples():
        fresh_only = int(row.adopted) == -1
        submitted, touched = _parse_utc(row.submitted_at), _parse_utc(row.updated_at)
        since = submitted or touched
        threshold = since - skew if since is not None else None
        last_activity = max((t for t in (submitted, touched) if t is not None), default=None)

        def created_after_submission(t: dict) -> bool:
            created = _parse_utc(t.get("create_time"))
            if created is None:
                return not fresh_only          # unknown age: acceptable only for a normal row
            return threshold is None or created >= threshold

        task = None
        if row.task_id and by_id.get(row.task_id, {}).get("state") in _ADOPTABLE:
            task = by_id[row.task_id]           # the claim clears task_id, so a recorded id is this attempt's
        else:
            fresh = [t for t in by_desc.get(row.description, []) if created_after_submission(t)]
            if fresh:
                task = max(fresh, key=lambda t: _ADOPTABLE[t["state"]])
        flag = -1 if fresh_only else 1
        if task is not None:
            updates.append(dict(task_key=row.task_key, task_id=str(task["id"]), state=task["state"],
                                attempts=max(1, int(row.attempts)), submitted_at=row.submitted_at or mf.now_utc(),
                                last_error="", error_class="", adopted=flag))
            stats["from_ee"] += 1
            continue
        if gcs is not None and not (fresh_only and threshold is None) and _gcs_has_output(
                cfg, run_dir, session, gcs, row.track_id, row.gcs_prefix,
                newer_than=threshold if fresh_only else None):
            updates.append(dict(task_key=row.task_key, state="COMPLETED", last_error="", error_class="",
                                adopted=flag))
            stats["from_gcs"] += 1
            continue
        if last_activity is not None and (now - last_activity).total_seconds() < SUBMITTING_GRACE_SECONDS:
            stats["waiting"] += 1
            continue
        if int(row.attempts) >= task_max:
            updates.append(dict(task_key=row.task_key, state="FAILED", error_class=row.error_class or "lost",
                                last_error=(f"no Earth Engine task found after {row.attempts} attempt(s); "
                                            f"last error: {row.last_error}")[:1000]))
            stats["failed"] += 1
        else:
            updates.append(dict(task_key=row.task_key, state="PENDING", task_id="",
                                last_error=(f"interrupted submission, no task or output found after "
                                            f"{SUBMITTING_GRACE_SECONDS}s; resubmitting ({row.last_error})")[:1000]))
            stats["reset"] += 1
    if updates:
        mf.update_manifest(run_dir, updates, return_df=False)
    if any(v for k, v in stats.items() if k != "waiting"):
        log.info("resume: %d tasks adopted from Earth Engine, %d chunks already in GCS, %d interrupted submissions "
                 "reset, %d failed, %d still within the grace period",
                 stats["from_ee"], stats["from_gcs"], stats["reset"], stats["failed"], stats["waiting"])
    return stats


def _requeue_rejected_adoptions(cfg: dict, run_dir: Path, df: pd.DataFrame) -> int:
    """Rows whose earlier output failed verification become FRESH exports (``adopted=-1``).

    The download stage does this itself (PENDING, adopted=-1). This migrates rows written by older versions:
    FAILED_DOWNLOAD + verify + adopted=1 (-> PENDING) and PENDING + verify + adopted=0.
    """
    failed = df[(df["state"] == "FAILED_DOWNLOAD") & (df["error_class"] == "verify") & (df["adopted"] == 1)]
    pending = df[(df["state"] == "PENDING") & (df["error_class"] == "verify") & (df["adopted"] == 0)]
    updates = [dict(task_key=r.task_key, state="PENDING", task_id="", adopted=-1, error_class="verify",
                    last_error=f"adopted output failed verification; exporting again ({r.last_error})"[:1000])
               for r in failed.itertuples()]
    updates += [dict(task_key=r.task_key, adopted=-1) for r in pending.itertuples()]
    if not updates:
        return 0
    mf.update_manifest(run_dir, updates, return_df=False)
    log.warning("%d chunks whose output failed verification are exported again (fresh exports)", len(updates))
    return len(updates)


# ---------------------------------------------------------------- submission
def listing_cutoff(df: pd.DataFrame, run_dir: str | Path) -> datetime:
    """Oldest moment a task of this run can have been created, minus 1 day of safety.

    = min(oldest submitted_at, oldest updated_at of SUBMITTING rows, run creation time) - 1 day.
    Run creation time = the date in the run id (v001_YYYYMMDD, midnight UTC) or run_config.yaml's
    modification time, whichever is older. Finished operations created before this cannot belong
    to the run, so the project listing can stop there.
    """
    run_dir = Path(run_dir)
    candidates: list[datetime] = []
    for text in list(df.loc[df["submitted_at"] != "", "submitted_at"]) + \
            list(df.loc[df["state"] == "SUBMITTING", "updated_at"]):
        t = _parse_utc(text)
        if t is not None:
            candidates.append(t)
    m = re.search(r"_(\d{8})$", run_dir.name)
    if m:
        candidates.append(datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc))
    cfg_file = run_dir / "run_config.yaml"
    if cfg_file.exists():
        candidates.append(datetime.fromtimestamp(cfg_file.stat().st_mtime, tz=timezone.utc))
    if not candidates:
        candidates.append(_utcnow())
    return min(candidates) - timedelta(days=1)


def _project_snapshot(backend, cutoff: datetime, max_age: float) -> dict:
    """Call backend.project_snapshot with cutoff/max_age when it supports them (simple fakes do not)."""
    params = inspect.signature(backend.project_snapshot).parameters
    kwargs = {}
    if "cutoff" in params:
        kwargs["cutoff"] = cutoff
    if "max_age" in params:
        kwargs["max_age"] = max_age
    return backend.project_snapshot(**kwargs)


def _default_gcs(cfg: dict, backend, gcs):
    if gcs is None and isinstance(backend, EEBackend):
        from .download import GCSBackend

        return GCSBackend(cfg)
    return gcs


def submit_pending(
    cfg: dict, run_dir: str | Path, backend=None, confirmed: bool = False,
    sleep: Callable[[float], None] = time.sleep, gcs=None, session: MonitorSession | None = None,
    compact_after: bool = True,
) -> pd.DataFrame:
    """One submission round: list the project's tasks, resolve interrupted submissions, start at most
    `headroom` PENDING rows. PLANNED rows are never touched (see ``confirm_exports``).

    ``gcs`` is a GCSBackend-like object used to detect outputs of interrupted submissions. It defaults to
    the real GCSBackend when the real EEBackend is used; with a fake EE backend and no ``gcs`` the GCS
    check is skipped.

    The run is validated first (Earth Engine project, grid), then the run's "monitor" lock is held for the
    round: another process or thread submitting for the same run fails immediately instead of starting the
    same chunks twice. Inside ``monitor`` (same thread) the lock is simply re-entered.
    """
    if not confirmed:
        raise ConfirmationRequired("Earth Engine exports need confirmed=True (CLI: --yes).")

    run_dir = Path(run_dir)
    check_run(cfg, run_dir)
    lock = run_lock(run_dir, "monitor")
    try:
        return _submission_round(cfg, run_dir, backend, sleep, gcs, session, compact_after)
    finally:
        lock.release()


def _submission_round(cfg: dict, run_dir: Path, backend, sleep, gcs, session: MonitorSession | None,
                      compact_after: bool) -> pd.DataFrame:
    backend = backend or EEBackend()
    gcs = _default_gcs(cfg, backend, gcs)
    session = _session(run_dir, session)
    run_meta.clean_stale_tmp(run_dir)
    df = mf.read_manifest(run_dir)
    if _requeue_rejected_adoptions(cfg, run_dir, df):
        df = mf.read_manifest(run_dir)
    if not df["state"].isin(["PENDING", "SUBMITTING"]).any():
        LAST_ROUND.clear()
        LAST_ROUND.update(queued=0, running=0, submitted=0, headroom=0, pending_left=0, eta_seconds=None)
        return df

    snapshot = _project_snapshot(backend, listing_cutoff(df, run_dir), float(cfg["export"].get("poll_seconds", 30)))
    counts = snapshot["counts"]
    _adopt_existing(cfg, run_dir, df, snapshot, gcs, session)

    df = mf.read_manifest(run_dir)
    pending = df[df["state"] == "PENDING"]
    own_active = int(df["state"].isin(mf.OWN_ACTIVE_STATES).sum())
    headroom = queue_headroom(cfg, counts, own_active)
    todo = pending.head(headroom)
    submitted = 0
    if not todo.empty:
        submitted = _submit_rows(cfg, run_dir, backend, todo, sleep, session)
    elif not pending.empty:
        log.info("queue full (queued=%d, running=%d): waiting for the next poll before submitting",
                 counts.get("READY", 0), counts.get("RUNNING", 0))

    df = mf.read_manifest(run_dir)
    pending_left = int((df["state"] == "PENDING").sum())
    eta = estimate_eta_seconds(df, counts.get("RUNNING", 0))
    LAST_ROUND.clear()
    LAST_ROUND.update(queued=int(counts.get("READY", 0)), running=int(counts.get("RUNNING", 0)),
                      submitted=submitted, headroom=headroom, pending_left=pending_left, eta_seconds=eta)
    log.info("submission round: queued=%d running=%d submitted=%d headroom=%d pending_left=%d eta=%s",
             LAST_ROUND["queued"], LAST_ROUND["running"], submitted, headroom, pending_left, _fmt_eta(eta))
    if compact_after:
        mf.compact(run_dir)
        df = mf.read_manifest(run_dir)
    return df


def _load_run_inputs(cfg: dict, run_dir: Path, session: MonitorSession) -> tuple[dict, dict]:
    """Grid and selected acquisitions, loaded once per session and checked against the frozen band layout."""
    from . import audit, grid

    if session.gd is None:
        session.gd = grid.load_grid(cfg)
    if session.selected is None:
        meta = run_meta.read_export_meta(cfg, run_dir)
        layout = read_band_layout(run_dir)
        selected = audit.selected_acquisitions(cfg, root(cfg) / meta["audit_dir"])
        for track_id in layout["track_id"].unique():
            want = list(layout[layout["track_id"] == track_id].drop_duplicates("acquisition_id")["acquisition_id"])
            have = list(selected[track_id].sort_values("datetime_utc")["acquisition_id"]) if track_id in selected else []
            if want != have:
                raise PipelineError(
                    f"acquisitions for {track_id} differ from this run's band_layout.csv; the audit/track "
                    "selection changed after planning. Create a new run."
                )
        session.selected = selected
    return session.gd, session.selected


def _submit_rows(cfg: dict, run_dir: Path, backend, todo: pd.DataFrame, sleep, session: MonitorSession) -> int:
    gd, selected = _load_run_inputs(cfg, run_dir, session)
    max_attempts = int(cfg["export"].get("submit_max_attempts", 5))
    uid = run_meta.run_uid(run_dir)

    def _submit(row: pd.Series) -> bool:
        from .s1_ard import pipeline as ard

        key = row["task_key"]
        try:
            chunk = _chunk_for_row(gd, row["chunk_name"])
            acq = selected[row["track_id"]].sort_values("datetime_utc").reset_index(drop=True)
            params = export_params(cfg, gd, chunk, key, uid)
            image = ard.build_chunk_image(cfg, gd, chunk, row["track_id"], acq)
        except Exception as exc:  # noqa: BLE001 — nothing reached Earth Engine: safe to fail the row
            mf.claim(run_dir, key, ["PENDING"], dict(state="FAILED", error_class=classify_error(str(exc)),
                                                     last_error=f"could not build the export: {exc}"[:1000]))
            return False

        for attempt in range(max_attempts):
            # Claim the row (compare-and-set PENDING -> SUBMITTING) and record submitted_at BEFORE start():
            # nobody else can start this row, and a crash or a late error after start is recovered by
            # looking the task up by its description (created after submitted_at).
            if not mf.claim(run_dir, key, ["PENDING"], dict(state="SUBMITTING", task_id="",
                                                             description=params["description"],
                                                             submitted_at=mf.now_utc())):
                log.info("%s is no longer PENDING (claimed elsewhere); skipped", key)
                return False
            try:
                task_id = backend.start_export(image, params)
            except Exception as exc:  # noqa: BLE001 — every failure is classified and recorded
                cls = classify_error(str(exc))
                if cls == "quota":
                    # an explicit rejection ("Too many tasks", rate limit): no task was created
                    mf.update_manifest(run_dir, [dict(task_key=key, state="PENDING")], return_df=False)
                    if attempt + 1 < max_attempts:
                        wait = _backoff_seconds(attempt)
                        log.warning("quota error submitting %s (attempt %d), retrying in %.1fs", key, attempt + 1, wait)
                        sleep(wait)
                        continue
                    mf.update_manifest(run_dir, [dict(task_key=key, state="PENDING", error_class="quota",
                                                       last_error=f"submission deferred: {exc}"[:1000])],
                                       return_df=False)
                    return False
                # Ambiguous: the request may have reached Earth Engine. Keep SUBMITTING (and submitted_at) so the
                # next round adopts the task if it exists, or resubmits after the grace period if it does not.
                mf.update_manifest(run_dir, [dict(
                    task_key=key, state="SUBMITTING", error_class=cls, attempts=int(row["attempts"]) + 1,
                    last_error=f"start() raised ({cls}); the task may or may not exist: {exc}"[:1000],
                )], return_df=False)
                log.warning("submitting %s raised (%s); will look for the task next round", key, cls)
                return False
            mf.update_manifest(run_dir, [dict(
                task_key=key, task_id=str(task_id), state="SUBMITTED", attempts=int(row["attempts"]) + 1,
                last_error="", error_class="",
            )], return_df=False)
            return True
        return False

    workers = min(resources.submit_threads(resources.detect_resources(cfg), cfg), len(todo))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return sum(pool.map(_submit, [r for _, r in todo.iterrows()]))


# ---------------------------------------------------------------- status + retry policy
def refresh_status(cfg: dict, run_dir: str | Path, backend=None, session: MonitorSession | None = None) -> pd.DataFrame:
    """Update submitted rows and apply the retry policy.

    States and error messages come from the project listing (one bounded paged read per round, shared
    with the submission round). Only tasks missing from it are asked for individually, at most
    ``_STATUS_LOOKUP_MAX`` per round. A task without any status for UNKNOWN_POLLS_TO_LOOKUP polls goes
    back to SUBMITTING, where it is adopted if found or resubmitted (bounded by task_max_attempts).
    """
    from . import grid

    run_dir = Path(run_dir)
    check_run(cfg, run_dir)
    backend = backend or EEBackend()
    session = _session(run_dir, session)
    df = mf.read_manifest(run_dir)
    active = df[df["state"].isin(mf.ACTIVE_STATES) & (df["task_id"] != "")]
    if active.empty:
        return df

    snapshot = _project_snapshot(backend, listing_cutoff(df, run_dir), float(cfg["export"].get("poll_seconds", 30)))
    by_id = {str(t.get("id")): t for t in snapshot.get("tasks", []) or []}
    statuses: dict[str, dict] = {}
    missing: list[str] = []
    for tid in active["task_id"]:
        t = by_id.get(tid)
        if t is not None:
            statuses[tid] = {"state": t.get("state", ""), "error_message": t.get("error_message", "")}
        else:
            missing.append(tid)
    if missing:
        lookup = missing[:_STATUS_LOOKUP_MAX]
        try:
            for tid, st in (backend.task_statuses(lookup) or {}).items():
                if str(st.get("state", "")).upper() not in ("", "UNKNOWN"):
                    statuses[tid] = st
        except Exception as exc:  # noqa: BLE001 — a failed lookup is retried next poll
            log.warning("status lookup for %d tasks failed (%s); retrying next poll", len(lookup), exc)
        if len(missing) > len(lookup):
            log.info("%d tasks missing from the listing; %d looked up this round", len(missing), len(lookup))

    exp = cfg["export"]
    task_max = int(exp.get("task_max_attempts", 3))
    split_ok = bool(exp.get("split_on_resource_error", True))
    existing = set(df["task_key"])
    updates: list[dict] = []

    for row in active.itertuples():
        key = row.task_key
        st = statuses.get(row.task_id)
        if st is None:
            n = session.unknown_polls.get(key, 0) + 1
            session.unknown_polls[key] = n
            if n >= UNKNOWN_POLLS_TO_LOOKUP:
                session.unknown_polls.pop(key, None)
                updates.append(dict(task_key=key, state="SUBMITTING", error_class="lost",
                                    last_error=f"task {row.task_id} not found in Earth Engine for {n} polls; "
                                               "looking it up by description"))
            continue
        session.unknown_polls.pop(key, None)
        state = str(st.get("state", "")).upper()
        msg = st.get("error_message", "") or ""
        if state in ("READY", "RUNNING"):
            if state != row.state:
                updates.append(dict(task_key=key, state=state))
        elif state == "UNSUBMITTED":
            continue
        elif state == "COMPLETED":
            updates.append(dict(task_key=key, state="COMPLETED", last_error="", error_class=""))
        elif state in ("CANCELLED", "CANCEL_REQUESTED"):
            updates.append(dict(task_key=key, state="FAILED", error_class="cancelled",
                                last_error=msg or f"task {state.lower()}"))
        elif state == "FAILED":
            cls = classify_error(msg)
            if cls in ("memory", "timeout"):
                if split_ok and not row.parent_chunk:
                    if session.gd is None:
                        session.gd = grid.load_grid(cfg)
                    gd = session.gd
                    parent = _chunk_for_row(gd, row.chunk_name)
                    updates.append(dict(task_key=key, state="SPLIT", error_class=cls, last_error=msg))
                    for part in grid.split_chunk(parent, gd["res"]):
                        sub = _planned_row(cfg, run_dir, row.track_id, part, int(row.n_bands),
                                           parent=row.chunk_name, state="PENDING")  # parent was confirmed
                        if sub["task_key"] not in existing:  # crash-safe: never re-add sub-chunks
                            updates.append(sub)
                else:
                    updates.append(dict(task_key=key, state="FAILED", error_class=cls, last_error=msg))
            elif cls in ("quota", "transient") and int(row.attempts) < task_max:
                updates.append(dict(task_key=key, state="PENDING", error_class=cls, last_error=msg, task_id=""))
            else:
                updates.append(dict(task_key=key, state="FAILED", error_class=cls, last_error=msg))
        else:
            log.warning("unexpected EE task state %r for %s", state, key)
    if updates:
        mf.update_manifest(run_dir, updates, return_df=False)
    return mf.read_manifest(run_dir)


_RECOVERABLE_LOOP_ERRORS = ("quota", "transient", "timeout")


def monitor(
    cfg: dict, run_dir: str | Path, backend=None, confirmed: bool = False, until_done: bool = True,
    sleep: Callable[[float], None] = time.sleep, max_cycles: int | None = None, gcs=None,
) -> pd.DataFrame:
    """Refresh + submit in a loop until no row is PENDING or active. Safe to restart after a crash.

    Only one monitor per run at a time (a second one fails immediately naming the first). Earth Engine
    quota/service errors during a round are waited out with backoff instead of stopping the monitor.
    PLANNED rows are never submitted: confirm a scope first.
    """
    if not confirmed:
        raise ConfirmationRequired("monitor submits Earth Engine exports; pass confirmed=True (CLI: --yes).")
    run_dir = Path(run_dir)
    check_run(cfg, run_dir)
    backend = backend or EEBackend()
    gcs = _default_gcs(cfg, backend, gcs)
    poll = float(cfg["export"].get("poll_seconds", 30))
    session = MonitorSession()
    with _SESSIONS_GUARD:
        _SESSIONS[str(run_dir.resolve())] = session
    lock = run_lock(run_dir, "monitor")
    try:
        cycles = errors_in_row = 0
        while True:
            lock.touch()
            cycles += 1
            try:
                refresh_status(cfg, run_dir, backend, session=session)
                df = submit_pending(cfg, run_dir, backend, confirmed=True, sleep=sleep, gcs=gcs, session=session,
                                    compact_after=False)
                errors_in_row = 0
            except PipelineError:
                raise
            except Exception as exc:  # noqa: BLE001
                if classify_error(str(exc)) not in _RECOVERABLE_LOOP_ERRORS:
                    raise
                errors_in_row += 1
                wait = min(300.0, _backoff_seconds(errors_in_row))
                log.warning("monitor round failed with a recoverable Earth Engine error (%s); retrying in %.0fs",
                            exc, wait)
                if max_cycles is not None and cycles >= max_cycles:
                    break
                sleep(wait)
                continue
            open_rows = df["state"].isin(_OPEN_STATES)
            if not until_done or not open_rows.any() or (max_cycles is not None and cycles >= max_cycles):
                break
            sleep(poll)
        write_failed_report(run_dir)
        mf.compact(run_dir)
        return mf.read_manifest(run_dir)
    finally:
        lock.release()


def write_failed_report(run_dir: str | Path) -> Path:
    """failed_chunks.csv: every FAILED export and FAILED_DOWNLOAD row (header only when none)."""
    run_dir = Path(run_dir)
    df = mf.read_manifest(run_dir)
    failed = df[df["state"].isin(["FAILED", "FAILED_DOWNLOAD"])]
    path = run_dir / FAILED_REPORT
    run_meta.atomic_write_text(path, failed[FAILED_COLUMNS].to_csv(index=False))
    return path
