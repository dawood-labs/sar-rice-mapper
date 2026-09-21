"""Export manifest: the single record of every export task in a run.

Why a state file?
-----------------
Exporting hundreds or thousands of chunks takes hours or days. The notebook may crash, the laptop may
sleep, Earth Engine may reject tasks. If the state lived only in memory, we would lose track of which
chunks were already submitted and could submit them twice (wasting quota) or forget some. So every
state change is written immediately, and any stage can be restarted and continues from the file.

Life of one row::

    PLANNED --(user confirms the scope)--> PENDING -> SUBMITTING -> SUBMITTED -> READY -> RUNNING -> COMPLETED
       |                                                  |                                     |
       |                                                  +--------------> FAILED <-------------+  (or SPLIT)
       +--> NOT_COVERED   (the track never covers this chunk: never exported)
    COMPLETED -> DOWNLOADED -> VERIFIED | VERIFIED_EMPTY (valid file, all nodata) | FAILED_DOWNLOAD

PLANNED rows are never submitted. Only an explicit confirmation of a scope (CLI `export --yes`,
`export.confirm_exports`) turns them into PENDING, so a dry run can never be exported by accident later.

SUBMITTING is written *before* an export is started. If the process dies between starting the task and
recording its id, the next round finds the task by its unique description instead of starting a duplicate.

Storage (safe for several processes, fast for large runs)
---------------------------------------------------------
- ``export_manifest.csv``            snapshot (human-readable table)
- ``export_manifest.journal.jsonl``  append-only log of changes made since the snapshot

An update appends one JSON line per changed row (cheap even with 30,000 rows) instead of rewriting the
whole table. Reading = snapshot + replay of the journal. When the journal gets long, or at the end of a
stage, ``compact`` writes a new snapshot atomically and empties the journal.

All reads and writes happen under a cross-process file lock (``.manifest.lock``), so a monitor in a
terminal and a download in a notebook can work on the same run without losing each other's updates.
Every journal entry sets absolute values (never increments), so replaying an entry twice is harmless.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import run_meta
from .locking import FileLock

log = logging.getLogger(__name__)

STATES = [
    "PLANNED", "PENDING", "SUBMITTING", "SUBMITTED", "READY", "RUNNING", "COMPLETED", "FAILED", "SPLIT",
    "NOT_COVERED", "DOWNLOADED", "VERIFIED", "VERIFIED_EMPTY", "FAILED_DOWNLOAD",
]
ERROR_CLASSES = ["", "memory", "timeout", "quota", "transient", "cancelled", "lost", "verify", "io", "other"]
COLUMNS = [
    "task_key", "track_id", "chunk_name", "parent_chunk", "row_off", "col_off", "width", "height",
    "n_bands", "gcs_prefix", "description", "task_id", "state", "attempts", "last_error",
    "error_class", "submitted_at", "updated_at", "local_paths", "adopted",
]
INT_COLUMNS = ["row_off", "col_off", "width", "height", "n_bands", "attempts", "adopted"]
ACTIVE_STATES = ["SUBMITTED", "READY", "RUNNING"]          # rows with an EE task id to poll
OWN_ACTIVE_STATES = ["SUBMITTING", *ACTIVE_STATES]         # rows occupying (or about to occupy) a queue slot
FILENAME = "export_manifest.csv"
JOURNAL = "export_manifest.journal.jsonl"
LOCKNAME = ".manifest.lock"
COMPACT_MIN_LINES = 2000

_CACHE: dict[str, dict] = {}          # per process: parsed state, re-read incrementally
_CACHE_GUARD = threading.Lock()


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / FILENAME


def journal_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / JOURNAL


def lock(run_dir: str | Path) -> FileLock:
    """Cross-process, re-entrant lock for read-modify-write sequences on this run's manifest."""
    return FileLock(Path(run_dir) / LOCKNAME, timeout=600.0, stale_after=300.0)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- parsing
def _default_record(key: str) -> dict:
    rec = {c: "" for c in COLUMNS}
    rec.update(task_key=key, state="PENDING", attempts=0, adopted=0)
    return rec


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Stable column order and types: integers as int, everything else as str ('' = empty, never NaN)."""
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[COLUMNS].copy()
    for col in COLUMNS:
        if col in INT_COLUMNS:
            df[col] = pd.to_numeric(df[col].replace("", 0), errors="coerce").fillna(0).astype("int64")
        else:
            df[col] = df[col].fillna("").astype(str)
    return df.reset_index(drop=True)


def _file_sig(path: Path) -> tuple | None:
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


def _load_snapshot(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for rec in df.to_dict("records"):
        full = _default_record(rec.get("task_key", ""))
        full.update({k: v for k, v in rec.items() if k in COLUMNS})
        rows[full["task_key"]] = full
    return rows


def _apply_journal(rows: dict[str, dict], path: Path, offset: int) -> tuple[int, int]:
    """Replay complete journal lines from `offset`; returns (new offset, number of lines read)."""
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except FileNotFoundError:
        return 0, 0
    consumed = n_lines = 0
    for raw in data.split(b"\n")[:-1]:          # the part after the last newline is incomplete (or empty)
        consumed += len(raw) + 1
        n_lines += 1
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
            key, values = entry["k"], entry["v"]
        except (ValueError, KeyError, TypeError):
            log.warning("manifest journal %s: ignoring a damaged line (interrupted write)", path)
            continue
        rec = rows.get(key)
        if rec is None:
            rec = rows[key] = _default_record(key)
        rec.update(values)
    tail = len(data) - consumed
    if tail:
        log.debug("manifest journal %s: %d bytes of an incomplete last line ignored", path, tail)
    return offset + consumed, n_lines


def _state(run_dir: Path) -> dict:
    """Up-to-date parsed state for this run (call with the lock held). Re-reads only what changed."""
    key = str(Path(run_dir).resolve())
    snap, jour = manifest_path(run_dir), journal_path(run_dir)
    snap_sig = _file_sig(snap)
    jsig = _file_sig(jour)
    jsize = jsig[1] if jsig else 0
    with _CACHE_GUARD:
        st = _CACHE.get(key)
    if st is None or st["snap_sig"] != snap_sig or jsize < st["offset"]:
        st = dict(snap_sig=snap_sig, rows=_load_snapshot(snap), offset=0, lines=0)
    if jsize > st["offset"]:
        st["offset"], n = _apply_journal(st["rows"], jour, st["offset"])
        st["lines"] += n
    with _CACHE_GUARD:
        _CACHE[key] = st
    return st


def _to_frame(rows: dict[str, dict]) -> pd.DataFrame:
    return _normalise(pd.DataFrame.from_records(list(rows.values()), columns=COLUMNS))


# ---------------------------------------------------------------- public API
def read_manifest(run_dir: str | Path) -> pd.DataFrame:
    """Current manifest (empty frame with all columns if it does not exist yet)."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return _normalise(pd.DataFrame(columns=COLUMNS))
    with lock(run_dir):
        st = _state(run_dir)
        return _to_frame(st["rows"])


def _validate(row: dict) -> str:
    key = row.get("task_key")
    if not key:
        raise ValueError(f"manifest row without task_key: {row}")
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise ValueError(f"unknown manifest columns: {sorted(unknown)}")
    state = row.get("state")
    if state is not None and state not in STATES:
        raise ValueError(f"invalid state {state!r} for {key}")
    return key


def update_manifest(run_dir: str | Path, rows: list[dict], return_df: bool = True) -> pd.DataFrame | None:
    """Insert or update rows by ``task_key`` (only the given fields change).

    ``updated_at`` is set automatically. New rows get defaults (state PENDING, attempts 0, adopted 0).
    The batch is appended to the journal with ONE lock acquisition and ONE fsync. Use
    ``return_df=False`` in hot loops: building the full DataFrame of a large manifest costs more than
    the update itself.
    """
    run_dir = Path(run_dir)
    if not rows:
        return read_manifest(run_dir) if return_df else None
    stamp = now_utc()
    lines = []
    for row in rows:
        key = _validate(row)
        values = {c: ("" if v is None else (int(v) if c in INT_COLUMNS and str(v) != "" else v))
                  for c, v in row.items() if c != "task_key"}
        values.setdefault("updated_at", stamp)
        lines.append(json.dumps({"k": key, "v": values}, default=str, separators=(",", ":")))
    payload = ("\n".join(lines) + "\n").encode("utf-8")

    run_dir.mkdir(parents=True, exist_ok=True)
    jour = journal_path(run_dir)
    with lock(run_dir):
        with open(jour, "ab") as f:
            if f.tell() > 0:
                # an interrupted earlier append may have left an unterminated line: terminate it first
                with open(jour, "rb") as r:
                    r.seek(-1, os.SEEK_END)
                    if r.read(1) != b"\n":
                        f.write(b"\n")
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        st = _state(run_dir)
        if st["lines"] > max(COMPACT_MIN_LINES, len(st["rows"]) // 10):
            _compact_locked(run_dir, st)
            st = _state(run_dir)
        return _to_frame(st["rows"]) if return_df else None


def claim(run_dir: str | Path, task_key: str, expected_states, values: dict) -> bool:
    """Compare-and-set: apply ``values`` to ``task_key`` only if its current state is in ``expected_states``.

    Why: before an Earth Engine export is started, the row is claimed PENDING -> SUBMITTING. Two processes
    (or threads) that both read the row as PENDING cannot both win the claim, so the same chunk is never
    started twice. Returns True when the update was applied.
    """
    run_dir = Path(run_dir)
    expected = {expected_states} if isinstance(expected_states, str) else set(expected_states)
    with lock(run_dir):
        rec = _state(run_dir)["rows"].get(task_key)
        if rec is None or rec.get("state") not in expected:
            return False
        update_manifest(run_dir, [dict(values, task_key=task_key)], return_df=False)
        return True


def _compact_locked(run_dir: Path, st: dict) -> None:
    df = _to_frame(st["rows"])
    run_meta.atomic_write_csv(df, manifest_path(run_dir))
    jour = journal_path(run_dir)
    # Truncate the journal: every entry is now in the snapshot. A reader that loaded the old snapshot
    # sees the snapshot signature change and reloads; replaying entries twice is harmless anyway.
    with open(jour, "wb") as f:
        f.flush()
        os.fsync(f.fileno())
    key = str(Path(run_dir).resolve())
    with _CACHE_GUARD:
        _CACHE.pop(key, None)


def compact(run_dir: str | Path) -> None:
    """Write the full table to export_manifest.csv and empty the journal (no-op when nothing changed)."""
    run_dir = Path(run_dir)
    if not journal_path(run_dir).exists():
        return
    with lock(run_dir):
        st = _state(run_dir)
        if st["lines"] == 0 and manifest_path(run_dir).exists():
            return
        _compact_locked(run_dir, st)
