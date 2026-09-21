"""Shared run metadata readers and crash-safe file helpers.

Every module that writes files or reads a run's frozen metadata uses this module, so there is ONE
implementation of each rule instead of several copies that slowly drift apart.

Crash-safe writes
-----------------
`atomic_write_*` write to a uniquely named temporary file in the same folder, flush and fsync it,
then rename it over the target. A crash (or power loss) leaves either the old file or the new file,
never a half-written one. Unique temporary names mean two processes never write the same temp file.
On Windows/WSL a rename can fail while another program (Excel, QGIS, antivirus) holds the file;
the rename is retried with backoff and a clear message is raised if it keeps failing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

from .config import grid_dir, root
from .errors import GridMismatch, PipelineError

log = logging.getLogger(__name__)

EXPORT_META = "export_meta.json"
BAND_LAYOUT = "band_layout.csv"
RUN_UID_FILE = "run_uid.txt"

# Required keys of export_meta.json. `ee_project` and `run_uid` were added later: validated when present.
EXPORT_META_KEYS = ["run_id", "dtype", "nodata", "int16_scale", "grid_crs", "grid_transform", "grid_fingerprint",
                    "audit_dir", "tracks", "config_hash", "created_utc"]
OPTIONAL_META_KEYS = ["ee_project", "run_uid"]
BAND_LAYOUT_COLUMNS = ["track_id", "band_idx", "acquisition_id", "date_utc", "datetime_utc", "date_local",
                       "pol", "platforms", "n_slices"]

TMP_SUFFIX = ".tmp"

# Earth Engine names a file "<prefix>.tif", or "<prefix>-<yoff>-<xoff>.tif" (10-digit offsets) when it
# splits a large export. Every module that matches exported files must use this one pattern.
EE_PART_SUFFIX = r"(?:-\d{10}-\d{10})?\.tif"


def ee_part_regex(name: str):
    """Compiled regex matching the file names Earth Engine writes for export prefix basename `name`."""
    import re

    return re.compile(rf"^{re.escape(name)}{EE_PART_SUFFIX}$")


def same_crs(a, b) -> bool:
    """True when two CRS definitions (EPSG string, WKT, rasterio/pyproj CRS) describe the same CRS."""
    from pyproj import CRS

    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:  # noqa: BLE001 - unparseable input is simply "not the same"
        return str(a) == str(b)


STALE_TMP_SECONDS = 6 * 3600

# Config sections that change the exported pixels. Operational settings (threads, polling, queue limits)
# are deliberately excluded: changing them must not invalidate a run.
_OUTPUT_SECTIONS = ("aoi", "season", "grid", "s1", "ard")
_OUTPUT_EXPORT_KEYS = ("dtype", "int16_scale", "nodata_float32", "nodata_int16")


# ---------------------------------------------------------------- crash-safe writes
def replace_with_retry(src: str | Path, dst: str | Path, attempts: int = 8) -> None:
    delay = 0.1
    for i in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            if i == attempts - 1:
                raise PipelineError(
                    f"Could not write {dst}: the file is locked by another program (Excel, QGIS, antivirus?). "
                    f"Close it and re-run the stage. ({exc})"
                ) from exc
            time.sleep(delay)
            delay = min(2.0, delay * 2)


def _fsync_dir(folder: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass  # some file systems (e.g. WSL drvfs) do not support fsync on directories
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=TMP_SUFFIX, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        replace_with_retry(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    atomic_write_text(path, df.to_csv(index=False))


def atomic_write_json(path: str | Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, default=str))


def clean_stale_tmp(folder: str | Path, recursive: bool = False, min_age_seconds: float = STALE_TMP_SECONDS) -> int:
    """Delete temporary files left by crashed processes. Returns how many were removed.

    Only files older than `min_age_seconds` are removed: a younger temp file may belong to another
    process that is writing right now.
    """
    folder = Path(folder)
    if not folder.exists():
        return 0
    now = time.time()
    n = 0
    for p in (folder.rglob(f"*{TMP_SUFFIX}") if recursive else folder.glob(f"*{TMP_SUFFIX}")):
        try:
            if p.is_file() and now - p.stat().st_mtime >= min_age_seconds:
                p.unlink()
                n += 1
        except OSError:
            continue
    if n:
        log.info("removed %d stale temporary file(s) in %s", n, folder)
    return n


# ---------------------------------------------------------------- hashes and ids
def sha256_file(path: str | Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(block), b""):
            h.update(piece)
    return h.hexdigest()


def grid_fingerprint(cfg: dict) -> str:
    """sha256 of the RAW BYTES of grid/grid_def.json."""
    path = grid_dir(cfg) / "grid_def.json"
    if not path.exists():
        raise PipelineError(f"{path} not found - build the grid first.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_hash(cfg: dict) -> str:
    """sha256 of the config settings that change the exported pixels."""
    relevant = {k: cfg.get(k) for k in _OUTPUT_SECTIONS}
    relevant["export"] = {k: (cfg.get("export") or {}).get(k) for k in _OUTPUT_EXPORT_KEYS}
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()


def run_uid(run_dir: str | Path) -> str:
    """Globally unique id of a run, used in GCS prefixes and EE task descriptions.

    Run folder names (v001_<date>) are only unique on one machine: a fresh clone elsewhere would create
    the same name and could adopt another machine's tasks. `config.new_run` writes a random suffix to
    run_uid.txt. Runs created before this existed fall back to the folder name.
    """
    run_dir = Path(run_dir)
    f = run_dir / RUN_UID_FILE
    if f.exists():
        return f.read_text().strip()
    return run_dir.name


def relative_to_root(cfg: dict, path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(root(cfg).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------- readers
def read_export_meta(cfg: dict, run_dir: str | Path, check_grid: bool = True) -> dict:
    """The run's frozen export settings (export_meta.json), validated.

    - every required key must exist (nodata / int16_scale fall back to cfg.export with a warning);
    - the grid fingerprint must equal the current grid_def.json (pixel ids would otherwise be wrong);
    - if the run recorded its Earth Engine project, the config must use the same project (tasks,
      queue counts and adoption all live in that project).
    """
    run_dir = Path(run_dir)
    path = run_dir / EXPORT_META
    if not path.exists():
        raise PipelineError(f"{path} not found - run the export planning stage first.")
    meta = json.loads(path.read_text())
    required = [k for k in EXPORT_META_KEYS if k not in ("nodata", "int16_scale", "audit_dir")]
    missing = [k for k in required if k not in meta]
    if missing:
        raise PipelineError(f"{path} is missing keys {missing}")

    dtype = str(meta["dtype"]).lower()
    if dtype not in ("float32", "int16"):
        raise PipelineError(f"Unsupported export dtype {dtype!r} in {path}")
    ecfg = cfg.get("export", {}) or {}
    if meta.get("nodata") is None:
        fallback = ecfg.get(f"nodata_{dtype}", -9999 if dtype == "float32" else -32768)
        log.warning("%s has no 'nodata'; using config export.nodata_%s = %s", path, dtype, fallback)
        meta["nodata"] = fallback
    if meta.get("int16_scale") is None:
        fallback = ecfg.get("int16_scale", 100)
        log.warning("%s has no 'int16_scale'; using config export.int16_scale = %s", path, fallback)
        meta["int16_scale"] = fallback

    if check_grid:
        current = grid_fingerprint(cfg)
        if meta["grid_fingerprint"] != current:
            raise GridMismatch(
                f"{path}: grid_fingerprint {meta['grid_fingerprint'][:12]}... does not match the current "
                f"grid_def.json ({current[:12]}...). The chunks of this run were exported on a different grid; "
                "use the grid the run was made with, or start a new season/aoi key."
            )

    project = (cfg.get("auth") or {}).get("project")
    if meta.get("ee_project") and project and meta["ee_project"] != project:
        raise PipelineError(
            f"Run {run_dir.name} was exported in Earth Engine project {meta['ee_project']!r}, but the config "
            f"uses {project!r}. Tasks, queue counts and duplicate detection live in the run's project: set "
            "auth.project back to the run's project (the key file may differ per machine)."
        )
    return {**meta, "dtype": dtype, "nodata": float(meta["nodata"]), "int16_scale": float(meta["int16_scale"])}


def read_band_layout(run_dir: str | Path, track_id: str | None = None) -> pd.DataFrame:
    """band_layout.csv (all BAND_LAYOUT_COLUMNS required), optionally one track, sorted by time then band."""
    path = Path(run_dir) / BAND_LAYOUT
    if not path.exists():
        raise PipelineError(f"{path} not found - run the export planning stage first.")
    bl = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in BAND_LAYOUT_COLUMNS if c not in bl.columns]
    if missing:
        raise PipelineError(f"{path} is missing columns {missing}")
    bl["band_idx"] = bl["band_idx"].astype(int)
    bl["n_slices"] = pd.to_numeric(bl["n_slices"], errors="coerce").fillna(0).astype(int)
    if track_id is not None:
        bl = bl[bl["track_id"] == track_id]
        if bl.empty:
            raise PipelineError(f"No bands for track {track_id} in {path}")
    return bl.sort_values(["track_id", "datetime_utc", "band_idx"], kind="stable").reset_index(drop=True)
