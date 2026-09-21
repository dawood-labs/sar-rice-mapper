"""Stage 4a — Download finished chunk files from Google Cloud Storage and verify them.

Why verify?
-----------
A file that downloaded without a network error can still be wrong: truncated, a stale file from
an older export, or (if an export parameter was wrong) on a shifted pixel grid. Such a file would
silently corrupt the stack. Every file is therefore checked before it is accepted:

- size and CRC32C checksum equal to what GCS stores (catches truncated/corrupt bytes);
- CRS and pixel size equal to the master grid;
- the files of one chunk cover exactly the chunk's extent, snapped to the master grid;
- the expected number of bands, the run's data type and nodata value.

A correct file whose pixels are ALL nodata (e.g. a chunk at the edge of a swath) is not an error:
it becomes VERIFIED_EMPTY and shows up in the stack QA, instead of failing again and again.

Failures and retries
--------------------
- Network / checksum errors ("io") are retried up to ``IO_ATTEMPTS`` times.
- A file that downloads fine but fails verification ("verify") is downloaded ONE more time (it
  might have been replaced in GCS meanwhile); if it fails again, retrying cannot help:
  FAILED_DOWNLOAD.
- Output we cannot trust is sent back to be exported again instead of failing: if the row was
  *adopted* by the export stage (output found in GCS without us starting the task), or its files
  are older than the row's own submission (leftovers of an earlier export), the row goes back to
  PENDING with ``adopted = -1`` ("must be a fresh export").
- For a row with ``adopted = -1`` files created before its submission are ignored when the chunk's
  files are grouped, so old parts of the previous bad export are never mixed with the new ones.
  Once such a row verifies, ``adopted`` returns to 0. If the fresh export fails verification again,
  the row becomes FAILED_DOWNLOAD (no endless export/download loop).
- A file locked by another program (Excel, QGIS, antivirus) never counts as a failed attempt: the
  rename is retried; if it stays locked the chunk is left untouched for the next run.

Scaling and resume
------------------
- GCS is listed ONCE per track folder per call; the estimate and the download share that listing.
- ``download_index.jsonl`` (append-only) records size, mtime and CRC32C of every accepted file.
  A re-run skips chunks whose files still have the recorded size and mtime without any network
  call or re-hashing (``deep=True`` forces re-hashing). A changed mtime triggers one re-hash.
- If the index was lost AND the GCS objects are gone (e.g. a bucket lifecycle rule deleted them),
  already VERIFIED chunks are re-inspected from their local files and kept when those are still
  correct: local data is never thrown away just because the cloud copy disappeared.
- Manifest and index updates are written in batches, not per file. After a crash, work that was
  not yet recorded is simply re-verified from the local files (no re-download when they match GCS).
- Only one download process may work on a run at a time (self-refreshing run lock).

Earth Engine may split a large export into several files named
``<prefix>-<yoffset>-<xoffset>.tif``; those are treated as one chunk (``run_meta.EE_PART_SUFFIX``,
the same rule the stack uses).
"""
from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import google_crc32c
import numpy as np
import pandas as pd
import rasterio

from . import manifest as mf
from . import resources, run_meta
from .errors import ConfirmationRequired, PipelineError
from .locking import run_lock

log = logging.getLogger(__name__)

_TOL = 1e-6
IO_ATTEMPTS = 3
_SPACE_MARGIN = 1.05  # keep 5 % headroom on disk
INDEX = "download_index.jsonl"
LEGACY_INDEX = "download_index.csv"  # written by earlier versions; read, then folded into INDEX
_READ_BLOCK = 8 * 1024 * 1024
FLUSH_EVERY_N = 50        # manifest/index batch size
FLUSH_EVERY_S = 10.0      # ... or at least this often
_SELECT_STATES = ["COMPLETED", "DOWNLOADED", "VERIFIED", "VERIFIED_EMPTY"]
_DONE_STATES = {"VERIFIED", "VERIFIED_EMPTY"}
# "<prefix>.tif" or "<prefix>-<10 digits>-<10 digits>.tif", from the one shared pattern in run_meta
_BLOB_RE = re.compile(rf"^(?P<prefix>.+?){run_meta.EE_PART_SUFFIX}$")
# A GCS object counts as "older than the submission" only beyond this margin: it absorbs clock
# differences between this machine (submitted_at) and GCS (time_created). Real leftovers are
# hours or days older; fresh export output is always created after the task started.
STALE_BLOB_TOLERANCE_S = 120.0
FRESH_EXPORT = -1  # manifest `adopted` value: the row must be re-exported, old GCS output is not trusted


# ---------------------------------------------------------------- time helpers
def _utc(value) -> datetime | None:
    """Parse an ISO-8601 time (or datetime) to an aware UTC datetime; None when empty/unparseable."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return _utc(dt).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _older_than(blob: dict, since: datetime | None) -> bool:
    created = _utc(blob.get("time_created"))
    return since is not None and created is not None and (since - created).total_seconds() > STALE_BLOB_TOLERANCE_S


# ---------------------------------------------------------------- backend
class GCSBackend:
    """Google Cloud Storage access (tests substitute a fake with the same methods)."""

    def __init__(self, cfg: dict | None = None, bucket=None):
        if bucket is None:
            from .auth import gcs_bucket

            bucket = gcs_bucket(cfg)
        self.bucket = bucket

    def list_blobs(self, prefix: str) -> list[dict]:
        """{name, size, crc32c, time_created (ISO-8601 UTC, "" if unknown)} for every object under prefix."""
        blobs = self.bucket.client.list_blobs(self.bucket, prefix=prefix)
        return [{"name": b.name, "size": int(b.size or 0), "crc32c": b.crc32c or "",
                 "time_created": _iso_utc(b.time_created) if getattr(b, "time_created", None) else ""}
                for b in blobs]

    def download(self, name: str, dest: Path) -> None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        # unique temp name: two processes (or a crashed earlier one) never share a temp file
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}{run_meta.TMP_SUFFIX}")
        try:
            # checksum="crc32c" makes the client compare against the stored CRC32C and raise on mismatch
            self.bucket.blob(name).download_to_filename(str(tmp), checksum="crc32c")
            run_meta.replace_with_retry(tmp, dest)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


class _TransferError(Exception):
    """Bytes on disk differ from GCS after a download."""


def _is_locked(exc: BaseException) -> bool:
    return isinstance(exc, PermissionError) or isinstance(getattr(exc, "__cause__", None), PermissionError)


# ---------------------------------------------------------------- checksums
def bytes_crc32c_b64(data: bytes) -> str:
    """CRC32C in the base64 big-endian form GCS reports in ``blob.crc32c``."""
    return base64.b64encode(google_crc32c.Checksum(data).digest()).decode()


def file_crc32c_b64(path: str | Path) -> str:
    c = google_crc32c.Checksum()
    with open(path, "rb") as f:
        while block := f.read(_READ_BLOCK):
            c.update(block)
    return base64.b64encode(c.digest()).decode()


def local_matches(path: Path, size: int, crc32c: str | None) -> bool:
    """True if the local file exists with this size and (when known) this CRC32C. Size is checked first (cheap)."""
    path = Path(path)
    try:
        if path.stat().st_size != int(size):
            return False
    except OSError:
        return False
    return not crc32c or file_crc32c_b64(path) == crc32c


# ---------------------------------------------------------------- index (append-only)
def _load_index(run_dir: Path) -> tuple[dict[str, dict], int, bool]:
    """(live entries by task_key, number of journal lines, legacy CSV present)."""
    out: dict[str, dict] = {}
    legacy = run_dir / LEGACY_INDEX
    has_legacy = legacy.exists()
    if has_legacy:
        df = pd.read_csv(legacy, dtype=str, keep_default_na=False)
        for key, g in df.groupby("task_key", sort=False):
            out[key] = dict(task_key=key, state="VERIFIED", verified_at=str(g["verified_at"].iloc[-1]), files=[
                dict(local_path=r["local_path"], size=int(r["size"]), mtime_ns=None, crc32c=r["crc32c"])
                for _, r in g.iterrows()
            ])
    n_lines = 0
    path = run_dir / INDEX
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue  # torn last line after a crash: that chunk is simply re-verified
                if entry.get("files"):
                    out[entry["task_key"]] = entry
                else:
                    out.pop(entry.get("task_key"), None)  # tombstone
    return out, n_lines, has_legacy


def read_index(run_dir: str | Path) -> dict[str, dict]:
    """Accepted files by task_key: {"task_key", "state", "files": [{local_path, size, mtime_ns, crc32c}], "verified_at"}."""
    return _load_index(Path(run_dir))[0]


def _append_index(run_dir: Path, entries: list[dict]) -> None:
    if not entries:
        return
    path = run_dir / INDEX
    lead = ""
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:  # a crash may have left a torn last line without newline
            f.seek(-1, os.SEEK_END)
            lead = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(lead + "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries))
        f.flush()
        os.fsync(f.fileno())


def _compact_index(run_dir: Path) -> None:
    entries, n_lines, has_legacy = _load_index(run_dir)
    if not has_legacy and n_lines <= 2 * len(entries) + 100:
        return
    run_meta.atomic_write_text(run_dir / INDEX, "".join(json.dumps(e, sort_keys=True) + "\n" for e in entries.values()))
    if has_legacy:
        try:
            (run_dir / LEGACY_INDEX).unlink()
        except OSError as exc:
            log.warning("could not remove legacy %s (%s); it is re-read and ignored where superseded", LEGACY_INDEX, exc)


def _entry_still_valid(run_dir: Path, entry: dict, deep: bool) -> tuple[bool, bool]:
    """(all files unchanged, any file re-hashed). Unchanged size + mtime is trusted unless deep=True."""
    rehashed = False
    for f in entry["files"]:
        p = run_dir / f["local_path"]
        try:
            st = p.stat()
        except OSError:
            return False, rehashed
        if st.st_size != int(f["size"]):
            return False, rehashed
        if not deep and f.get("mtime_ns") is not None and int(f["mtime_ns"]) == st.st_mtime_ns:
            continue
        rehashed = True
        if f.get("crc32c") and file_crc32c_b64(p) != f["crc32c"]:
            return False, rehashed
        f["mtime_ns"] = st.st_mtime_ns
    return True, rehashed


# ---------------------------------------------------------------- blob grouping
def group_blobs_by_prefix(blobs: list[dict]) -> dict[str, list[dict]]:
    """Map chunk prefix -> its files. ``<prefix>.tif`` and ``<prefix>-<10 digits>-<10 digits>.tif`` belong to ``<prefix>``.

    Parsing each name once makes grouping O(n) for a whole track folder (thousands of chunks).
    """
    out: dict[str, list[dict]] = defaultdict(list)
    for b in blobs:
        m = _BLOB_RE.match(b["name"])
        if m:
            out[m.group("prefix")].append(b)
    for files in out.values():
        files.sort(key=lambda b: b["name"])
    return dict(out)


def chunk_blob_names(blobs: list[dict], prefix: str) -> list[dict]:
    """Blobs that belong to exactly this chunk prefix.

    A plain ``startswith`` is wrong: prefix ``.../chunk_r00c00`` also matches the sub-chunk file
    ``.../chunk_r00c00_s0.tif``. Only ``<prefix>.tif`` and ``<prefix>-<digits>-<digits>.tif`` count.
    """
    return group_blobs_by_prefix(blobs).get(prefix, [])


# ---------------------------------------------------------------- verification
def _chunk_from_row(gd: dict, row) -> dict:
    res = float(gd["res"])
    xmin = gd["x0"] + int(row["col_off"]) * res
    ymax = gd["y0"] - int(row["row_off"]) * res
    return dict(
        name=row["chunk_name"], row_off=int(row["row_off"]), col_off=int(row["col_off"]),
        width=int(row["width"]), height=int(row["height"]),
        xmin=xmin, ymax=ymax, xmax=xmin + int(row["width"]) * res, ymin=ymax - int(row["height"]) * res,
    )


def _same_nodata(a, b) -> bool:
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= _TOL


def inspect_chunk_files(paths: list[Path], gd: dict, chunk: dict, n_bands: int, nodata,
                        dtype: str | None = None) -> tuple[str, str]:
    """Check the files of one chunk against the master grid and the run's encoding.

    Returns (status, reason) with status "ok", "empty" (correct but entirely nodata) or "bad".
    The CRS test is ``run_meta.same_crs``, the same one the stack uses, so a file accepted here is
    never rejected when the stack is built.
    """
    if not paths:
        return "bad", "no files"
    res = float(gd["res"])
    lefts, tops, rights, bottoms = [], [], [], []
    n_pixels = 0
    any_valid = False
    for p in paths:
        name = Path(p).name
        try:
            src = rasterio.open(p)
        except Exception as exc:  # noqa: BLE001
            return "bad", f"{name}: cannot open ({exc})"
        with src:
            if src.crs is None or not run_meta.same_crs(src.crs, gd["crs"]):
                return "bad", f"{name}: CRS {src.crs} != {gd['crs']}"
            t = src.transform
            if abs(t.a - res) > _TOL or abs(t.e + res) > _TOL or abs(t.b) > _TOL or abs(t.d) > _TOL:
                return "bad", f"{name}: pixel size/rotation {tuple(t)[:6]} != res {res}"
            for coord, origin in ((t.c, gd["x0"]), (t.f, gd["y0"])):
                steps = (coord - origin) / res
                if abs(coord - (origin + round(steps) * res)) > _TOL:
                    return "bad", f"{name}: origin {coord} not snapped to the master grid"
            if src.count != n_bands:
                return "bad", f"{name}: {src.count} bands, expected {n_bands}"
            if dtype is not None and any(d != dtype for d in src.dtypes):
                return "bad", f"{name}: dtype {src.dtypes[0]} != run dtype {dtype}"
            if nodata is not None and not _same_nodata(src.nodata, nodata):
                return "bad", f"{name}: nodata tag {src.nodata} != run nodata {nodata}"
            b = src.bounds
            lefts.append(b.left); tops.append(b.top); rights.append(b.right); bottoms.append(b.bottom)
            n_pixels += src.width * src.height
            if not any_valid:
                try:
                    for _, win in src.block_windows(1):
                        arr = src.read(window=win)
                        valid = ~np.isnan(arr) & (arr != nodata) if arr.dtype.kind == "f" else arr != nodata
                        if valid.any():
                            any_valid = True
                            break
                except Exception as exc:  # noqa: BLE001 — truncated data blocks
                    return "bad", f"{name}: cannot read pixels ({exc})"
    extent = (min(lefts), min(bottoms), max(rights), max(tops))
    expected = (chunk["xmin"], chunk["ymin"], chunk["xmax"], chunk["ymax"])
    if any(abs(a - e) > _TOL for a, e in zip(extent, expected)):
        return "bad", f"extent {extent} != chunk extent {expected}"
    if n_pixels != int(chunk["width"]) * int(chunk["height"]):
        return "bad", f"files hold {n_pixels} pixels, chunk has {int(chunk['width']) * int(chunk['height'])} (gap/overlap)"
    if not any_valid:
        return "empty", "all pixels are nodata"
    return "ok", "ok"


def verify_chunk_files(paths: list[Path], gd: dict, chunk: dict, n_bands: int, nodata,
                       dtype: str | None = None) -> tuple[bool, str]:
    """(ok, reason). An entirely-nodata chunk is reported as not ok; see inspect_chunk_files for the 3-way status."""
    status, reason = inspect_chunk_files(paths, gd, chunk, n_bands, nodata, dtype)
    return status == "ok", reason


# ---------------------------------------------------------------- planning (one listing per track folder)
def _dest(run_dir: Path, row, blob: dict) -> Path:
    return run_dir / "raw_chunks" / f"track_{row['track_id']}" / blob["name"].rsplit("/", 1)[-1]


def _folder(prefix: str) -> str:
    return prefix.rsplit("/", 1)[0] + "/" if "/" in prefix else prefix


def _adopted(row) -> int:
    try:
        return int(row.get("adopted") or 0)
    except (TypeError, ValueError):
        return 0


def _restore_from_local(run_dir: Path, row: dict, gd: dict, meta: dict) -> dict | None:
    """Re-accept an already verified chunk from its local files (index lost, GCS objects gone).

    Returns {"index": entry, "manifest": update or None} when every local file is present and passes
    verification, else None (the row then takes the normal download path).
    """
    rels = [p for p in str(row.get("local_paths") or "").split(";") if p]
    if not rels:
        return None
    paths = [run_dir / rel for rel in rels]
    if not all(p.is_file() for p in paths):
        return None
    status, reason = inspect_chunk_files(paths, gd, _chunk_from_row(gd, row), int(row["n_bands"]),
                                         meta["nodata"], meta["dtype"])
    if status not in ("ok", "empty"):
        log.warning("%s: GCS files are gone and the local files fail verification (%s)", row["task_key"], reason)
        return None
    state = "VERIFIED" if status == "ok" else "VERIFIED_EMPTY"
    files = []
    for rel, p in zip(rels, paths):
        st = p.stat()
        files.append(dict(local_path=rel, size=int(st.st_size), mtime_ns=st.st_mtime_ns, crc32c=file_crc32c_b64(p)))
    log.info("%s: no files in GCS, but the local files verify -> kept as %s and re-indexed", row["task_key"], state)
    update = None if state == row["state"] else dict(task_key=row["task_key"], state=state)
    return dict(index=dict(task_key=row["task_key"], state=state, files=files, verified_at=mf.now_utc()),
                manifest=update)


def _plan(cfg: dict, run_dir: Path, backend, retry_failed: bool, force: bool, deep: bool) -> dict:
    df = mf.read_manifest(run_dir)
    states = _SELECT_STATES + (["FAILED_DOWNLOAD"] if retry_failed else [])
    rows = df[df["state"].isin(states)].to_dict("records")
    index, _, _ = _load_index(run_dir)

    todo, refreshed, skipped = [], [], 0
    for row in rows:
        entry = index.get(row["task_key"])
        if row["state"] in _DONE_STATES and not force and entry and entry.get("state") == row["state"]:
            ok, rehashed = _entry_still_valid(run_dir, entry, deep)
            if ok:
                skipped += 1
                if rehashed:
                    refreshed.append(entry)
                continue
        todo.append(row)

    by_prefix: dict[str, list[dict]] = {}
    for folder in sorted({_folder(r["gcs_prefix"]) for r in todo}):
        by_prefix.update(group_blobs_by_prefix(backend.list_blobs(folder)))

    row_blobs: dict[str, list[dict]] = {}
    restored: list[dict] = []
    gd = meta = None
    remaining = []
    for row in todo:
        blobs = by_prefix.get(row["gcs_prefix"], [])
        if _adopted(row) == FRESH_EXPORT and blobs:
            since = _utc(row.get("submitted_at"))
            kept = [b for b in blobs if not _older_than(b, since)]
            if len(kept) != len(blobs):
                log.info("%s: ignoring %d GCS file(s) created before this fresh export was submitted",
                         row["task_key"], len(blobs) - len(kept))
            blobs = kept
        if not blobs and row["state"] in _DONE_STATES:
            if gd is None:
                from . import grid

                gd = grid.load_grid(cfg)
                meta = run_meta.read_export_meta(cfg, run_dir)
            res = _restore_from_local(run_dir, row, gd, meta)
            if res is not None:
                restored.append(res)
                skipped += 1
                continue
        row_blobs[row["task_key"]] = blobs
        remaining.append(row)

    n_files = total = 0
    missing = []
    for row in remaining:
        blobs = row_blobs[row["task_key"]]
        if not blobs:
            missing.append(row["task_key"])
        for b in blobs:
            dest = _dest(run_dir, row, b)
            try:
                present = dest.stat().st_size == int(b["size"])
            except OSError:
                present = False
            if force or not present:  # size-only here; checksums are checked during the download
                n_files += 1
                total += int(b["size"])
    free = resources.disk_free_bytes(run_dir)
    estimate = dict(n_tasks=len(rows), n_skipped_verified=skipped, n_files=n_files, bytes=int(total),
                    disk_free_bytes=int(free), fits=bool(total * _SPACE_MARGIN <= free),
                    tasks_without_files=missing)
    return dict(rows=remaining, blobs=row_blobs, refreshed=refreshed, restored=restored, estimate=estimate)


# ---------------------------------------------------------------- one chunk
def _remove_files(paths: list[Path]) -> None:
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("could not remove %s (%s)", p, exc)


def _process_row(run_dir: Path, row: dict, blobs: list[dict], backend, gd: dict, meta: dict, force: bool) -> dict:
    """Download + verify one chunk. Never writes the manifest: returns the updates for batching."""
    key = row["task_key"]
    adopted = _adopted(row)
    submitted = _utc(row.get("submitted_at"))
    # leftovers of an earlier export under the same prefix (only judged for rows not already fresh exports)
    stale_files = adopted != FRESH_EXPORT and any(_older_than(b, submitted) for b in blobs)
    chunk = _chunk_from_row(gd, row)
    dests = [_dest(run_dir, row, b) for b in blobs]
    fresh = force
    io_failures = 0
    verify_retried = False
    reason, cls = "", ""
    fetched = 0
    while True:
        if not blobs:
            reason, cls = "no files found in GCS for this prefix", "verify"
            break
        try:
            for b, dest in zip(blobs, dests):
                if fresh or not local_matches(dest, b["size"], b["crc32c"]):
                    backend.download(b["name"], dest)
                    fetched += 1
                    if not local_matches(dest, b["size"], b["crc32c"]):
                        raise _TransferError(f"{dest.name}: size/CRC32C differs from GCS after download")
        except Exception as exc:  # noqa: BLE001 — checksum mismatch, network error, locked file ...
            if _is_locked(exc):
                log.warning("%s: a local file is locked by another program (%s); left for the next run", key, exc)
                return dict(outcome="locked")
            io_failures += 1
            reason, cls = f"download error: {exc}", "io"
            log.warning("download failed for %s (attempt %d/%d): %s", key, io_failures, IO_ATTEMPTS, exc)
            if io_failures >= IO_ATTEMPTS:
                break
            fresh = True
            continue

        status, reason = inspect_chunk_files(dests, gd, chunk, int(row["n_bands"]), meta["nodata"], meta["dtype"])
        if status in ("ok", "empty"):
            state = "VERIFIED" if status == "ok" else "VERIFIED_EMPTY"
            rels = [d.relative_to(run_dir).as_posix() for d in dests]
            files = []
            for b, d, rel in zip(blobs, dests, rels):
                files.append(dict(local_path=rel, size=int(b["size"]), mtime_ns=d.stat().st_mtime_ns, crc32c=b["crc32c"]))
            if status == "empty":
                log.info("%s: files are correct but contain only nodata -> VERIFIED_EMPTY", key)
            update = dict(task_key=key, state=state, local_paths=";".join(rels), last_error="", error_class="")
            if adopted == FRESH_EXPORT:
                update["adopted"] = 0  # the fresh export verified: back to a normal row
            return dict(
                outcome="empty" if status == "empty" else ("downloaded" if fetched else "reused"),
                manifest=update,
                index=dict(task_key=key, state=state, files=files, verified_at=mf.now_utc()),
            )
        cls = "verify"
        if not verify_retried:
            log.warning("%s failed verification (%s); downloading once more", key, reason)
            _remove_files(dests)
            verify_retried, fresh = True, True
            continue
        break

    _remove_files(dests)
    tombstone = dict(task_key=key, files=[])
    if cls == "verify" and (adopted == 1 or stale_files):
        why = "was adopted from existing GCS output" if adopted == 1 else "has GCS files older than its submission"
        log.warning("%s %s that fails verification (%s): re-queued for a fresh export", key, why, reason)
        return dict(outcome="requeued", index=tombstone, manifest=dict(
            task_key=key, state="PENDING", adopted=FRESH_EXPORT, error_class="verify", last_error=reason[:1000],
            attempts=int(row.get("attempts") or 0) + 1, local_paths=""))
    return dict(outcome="failed", index=tombstone, manifest=dict(
        task_key=key, state="FAILED_DOWNLOAD", error_class=cls, last_error=reason[:1000], local_paths=""))


# ---------------------------------------------------------------- public API
def estimate_download(cfg: dict, run_dir: str | Path, backend=None, retry_failed: bool = False,
                      force: bool = False, deep: bool = False) -> dict:
    """Bytes still to download (files already present with the right size are not counted) vs free disk space."""
    run_dir = Path(run_dir)
    return _plan(cfg, run_dir, backend or GCSBackend(cfg), retry_failed, force, deep)["estimate"]


def _flush(run_dir: Path, manifest_rows: list[dict], index_entries: list[dict]) -> None:
    # index first: if we crash in between, the manifest still says COMPLETED and the chunk is re-verified
    _append_index(run_dir, index_entries)
    if manifest_rows:
        mf.update_manifest(run_dir, manifest_rows, return_df=False)


def download_completed(
    cfg: dict, run_dir: str | Path, backend=None, confirmed: bool = False, retry_failed: bool = False,
    force: bool = False, deep: bool = False,
) -> pd.DataFrame:
    """Download + verify every COMPLETED chunk and re-check accepted ones. Needs confirmed=True.

    ``force=True`` re-downloads every file; ``retry_failed=True`` also retries FAILED_DOWNLOAD rows;
    ``deep=True`` re-hashes already accepted local files even when size and mtime are unchanged.
    """
    if not confirmed:
        raise ConfirmationRequired("Bulk downloads need confirmed=True (CLI: --yes).")
    from . import export, grid

    run_dir = Path(run_dir)
    backend = backend or GCSBackend(cfg)
    lock = run_lock(run_dir, "download")  # refreshed by its own heartbeat thread while we work
    try:
        run_meta.clean_stale_tmp(run_dir / "raw_chunks", recursive=True)
        plan = _plan(cfg, run_dir, backend, retry_failed, force, deep)
        est = plan["estimate"]
        if not est["fits"]:
            gb = 1024**3
            raise PipelineError(
                f"not enough disk space: need ~{est['bytes'] * _SPACE_MARGIN / gb:.2f} GB, "
                f"free {est['disk_free_bytes'] / gb:.2f} GB at {run_dir}"
            )
        meta = run_meta.read_export_meta(cfg, run_dir)
        todo = plan["rows"]
        outcome = Counter(skipped=est["n_skipped_verified"] - len(plan["restored"]), restored=len(plan["restored"]))
        pending_rows: list[dict] = [r["manifest"] for r in plan["restored"] if r["manifest"]]
        pending_index: list[dict] = list(plan["refreshed"]) + [r["index"] for r in plan["restored"]]
        try:
            if todo:
                gd = grid.load_grid(cfg)
                workers = max(1, min(resources.io_threads(resources.detect_resources(cfg), cfg), len(todo)))
                last_flush = time.monotonic()
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(_process_row, run_dir, row, plan["blobs"].get(row["task_key"], []),
                                           backend, gd, meta, force) for row in todo]
                    for fut in as_completed(futures):
                        res = fut.result()
                        outcome[res["outcome"]] += 1
                        if res.get("manifest"):
                            pending_rows.append(res["manifest"])
                        if res.get("index"):
                            pending_index.append(res["index"])
                        if len(pending_rows) >= FLUSH_EVERY_N or time.monotonic() - last_flush >= FLUSH_EVERY_S:
                            _flush(run_dir, pending_rows, pending_index)
                            pending_rows, pending_index = [], []
                            last_flush = time.monotonic()
        finally:
            _flush(run_dir, pending_rows, pending_index)
        _compact_index(run_dir)
        n = est["n_tasks"]
        log.info("download: skipped %d/%d chunks (already verified), restored %d from local files, reused %d with "
                 "matching local files, downloaded %d, empty %d, re-queued for export %d, locked %d, failed %d",
                 outcome["skipped"], n, outcome["restored"], outcome["reused"], outcome["downloaded"],
                 outcome["empty"], outcome["requeued"], outcome["locked"], outcome["failed"])
        export.write_failed_report(run_dir)
        return mf.read_manifest(run_dir)
    finally:
        lock.release()
