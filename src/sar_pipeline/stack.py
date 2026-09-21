"""Stage 4b — Build the multi-date raster stack (VRTs) from verified chunk files, and run QA.

What this stage produces (per track, inside ``runs/<run>/stack/track_<track_id>/``)::

    VV/VV_<YYYYMMDD>.vrt   one single-band virtual mosaic per acquisition and polarisation
    VH/VH_<YYYYMMDD>.vrt
    *.vrt.sources.sha256   fingerprint of the inputs each VRT was built from (resume check)
    stack_VV.vrt           all VV dates as bands, in date order (band 1 = first date)
    stack_VH.vrt
    dates.csv              what each stack band is (date, platforms, rain, valid %, flags)
    _valid_cache/          cached valid-pixel counts per (chunk file, band), one part file per batch
    _aoi_cache.json        cached number of AOI pixels of the planned scope and of the grid

Polarisations come from the run's band layout (usually VV and VH), never from a fixed list.

Why VRTs instead of a big mosaic GeoTIFF?
------------------------------------------
A VRT is a small XML file that *points* at the chunk GeoTIFFs; no pixels are copied. A full-AOI
mosaic would be hundreds of GB and would need a lot of RAM to write. With VRTs, GDAL reads only
the chunk files that intersect the window you ask for. Paths inside the VRTs are relative, so the
whole run folder can be copied to another machine and still works.

The VRT XML is written by this module directly (no GDAL Python bindings): the position of every
chunk file on the master grid is known exactly, and each file's header (CRS, pixel size, origin,
size, data type, band count) is checked with rasterio before it is referenced. A file that does
not line up with the grid raises an error instead of being dropped.

Valid percentage and planned scope
----------------------------------
``valid_pct`` = valid pixels / AOI pixels inside the run's PLANNED scope, i.e. the union of the
chunk windows the manifest planned for that track. Rows that are not part of the scope:
``SPLIT`` parents (their sub-chunk rows stand in for them), ``NOT_COVERED`` (the track's swath never
covers that chunk) and ``PLANNED`` (planned in a dry run but never confirmed for export).
A planned chunk that failed or is not verified yet still counts as invalid. ``dates.csv`` column
``aoi_planned_pct`` says how much of the whole AOI the planned scope covers (e.g. 3 % for a pilot).
``VERIFIED_EMPTY`` chunks (verified files that hold only nodata) are part of the stack and of the
scope; they are reported as an informational ``EMPTY_CHUNK`` issue.

Why QA here?
------------
This is the first moment we can see real pixels for every date. Dates with too few valid pixels,
acquisitions the audit expected but the export does not contain, and chunks that failed or were
never verified are written to ``decisions_required.md`` and the pipeline stops with
``DecisionRequired``. Nothing is silently skipped: the user either fixes the problem or adds the
issue id to ``qa.acknowledged_issues`` in the config and re-runs this stage.

Resume, never redo
------------------
Re-running after a crash, or after more chunks became VERIFIED, only does the missing work:
- a VRT is rebuilt only when the fingerprint of its inputs changed (file list with sizes and
  modification times, band, grid, data type, nodata); the fingerprint sits in a small sidecar file,
  so checking a VRT never parses its XML or opens any chunk file;
- valid-pixel counts are cached per chunk file + band, keyed by file size, modification time and
  the file's planned window. Each finished batch is appended as a new part file (no rewrite of the
  whole cache), and the parts are compacted into one file at the end.
Every file is written atomically. ``force=True`` recomputes everything in this run's stack folder.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window

from . import resources
from .config import grid_dir, root
from .errors import AuditRequired, DecisionRequired, GridMismatch, PipelineError
from .grid import load_grid
from .run_meta import same_crs
from .run_meta import (  # noqa: F401  (re-exported for callers of the stack module)
    BAND_LAYOUT_COLUMNS,
    atomic_write_csv,
    atomic_write_text,
    clean_stale_tmp,
    grid_fingerprint,
    read_band_layout,
    read_export_meta,
)

log = logging.getLogger(__name__)

ISSUE_TYPES = ("MISSING_ACQUISITION", "LOW_VALID", "FAILED_CHUNK", "UNVERIFIED_CHUNK", "EMPTY_CHUNK")
RAIN_FLAG_MM_24H = 5.0
VALID_CACHE = "_valid_cache"            # folder of part-XXXXXX.csv files
LEGACY_VALID_CACHE = "_valid_cache.csv"  # single-file cache of earlier versions (removed on compaction)
AOI_CACHE = "_aoi_cache.json"
SOURCES_SUFFIX = ".sources.sha256"
_CACHE_COLUMNS = ["path", "size", "mtime_ns", "n_bands", "band_idx", "key", "valid_pixels"]
_FAILED_STATES = {"FAILED", "FAILED_DOWNLOAD"}
_FILE_STATES = {"VERIFIED", "VERIFIED_EMPTY"}              # rows whose local files belong in the stack
_OUT_OF_SCOPE_STATES = {"SPLIT", "NOT_COVERED", "PLANNED"}  # rows that are not part of the planned scope
_GDAL_TYPES = {"float32": "Float32", "int16": "Int16"}
_MB = 1024**2

_ISSUE_HELP = {
    "MISSING_ACQUISITION": (
        "The audit found this acquisition over the AOI, but it is not part of the exported band layout.",
        "Check whether it was excluded on purpose (e.g. low coverage). If it should be included, start a "
        "new run after updating the track/date selection; otherwise acknowledge it.",
    ),
    "LOW_VALID": (
        "Too few valid (non-nodata) pixels inside the AOI for this date. Causes: partial swath coverage, "
        "layover/shadow masking, border-noise masking, or chunks that failed/are unverified.",
        "Look at the date VRT in QGIS. If the gap is real (the satellite did not cover the AOI), acknowledge "
        "it and treat the date as partial in analysis. If chunks are missing, fix those first.",
    ),
    "FAILED_CHUNK": (
        "This chunk's export or download failed after all retries, so its pixels are nodata in every date.",
        "See failed_chunks.csv for the error. Re-run export/monitor/download for the run, or acknowledge "
        "it if the chunk lies outside the area you need.",
    ),
    "UNVERIFIED_CHUNK": (
        "This chunk has not reached VERIFIED yet (still exporting, downloading, or not checked), so it is "
        "not in the stack.",
        "Run monitor and download until the chunk is VERIFIED, then rebuild the stack.",
    ),
    "EMPTY_CHUNK": (
        "This chunk was exported and verified, but its file holds only nodata (for example the area lies "
        "outside the swath on every date, or every pixel was masked).",
        "Usually informational. Check the chunk location; if the empty result is expected, acknowledge it.",
    ),
}


# ---------------------------------------------------------------- manifest / scope helpers
def _read_manifest(run_dir: Path) -> pd.DataFrame:
    from .manifest import read_manifest

    return read_manifest(run_dir)


def _acq_token(acquisition_id: str, track_id: str) -> str:
    """Date part of an acquisition id (``RO123_ASC_20260405`` -> ``20260405``; keeps suffixes like ``_2``)."""
    prefix = f"{track_id}_"
    return acquisition_id[len(prefix):] if acquisition_id.startswith(prefix) else acquisition_id


def _band_name(pol: str, acquisition_id: str, track_id: str) -> str:
    return f"{pol}_{_acq_token(acquisition_id, track_id)}"


def _acquisition_order(bl: pd.DataFrame, track_id: str) -> list[str]:
    """Acquisition ids in time order; their date tokens must be unique (they become file names)."""
    acq = list(dict.fromkeys(bl["acquisition_id"]))
    tokens = [_acq_token(a, track_id) for a in acq]
    if len(set(tokens)) != len(tokens):
        dup = sorted({t for t in tokens if tokens.count(t) > 1})
        raise PipelineError(f"Acquisition ids of {track_id} are not unique per date ({dup}); VRT file names would collide.")
    return acq


def _pols(bl: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(bl["pol"]))


def _window_id(row) -> str:
    return f"{int(row['row_off'])}:{int(row['col_off'])}:{int(row['height'])}:{int(row['width'])}"


def planned_scope(run_dir: str | Path, track_id: str) -> pd.DataFrame:
    """Manifest rows that define what this run planned for a track.

    Excluded: SPLIT parents, NOT_COVERED and PLANNED (unconfirmed) rows. Raises PipelineError when
    nothing is planned or two planned windows overlap (the valid-pixel denominator would count
    pixels twice).
    """
    man = _read_manifest(Path(run_dir))
    scope = man[(man["track_id"] == track_id) & (~man["state"].isin(_OUT_OF_SCOPE_STATES))].copy()
    if scope.empty:
        raise PipelineError(f"No planned chunks for track {track_id} in the manifest of {run_dir}.")
    scope = scope.sort_values(["row_off", "col_off"]).reset_index(drop=True)
    # Only windows in the same original chunk cell can overlap (sub-chunks); check those pairwise.
    cell_px = int(max(scope["height"].max(), scope["width"].max()))
    cells: dict[tuple[int, int], list] = {}
    for r in scope.itertuples():
        for key in {(r.row_off // cell_px, r.col_off // cell_px),
                    ((r.row_off + r.height - 1) // cell_px, (r.col_off + r.width - 1) // cell_px)}:
            for o in cells.get(key, []):
                if (r.row_off < o.row_off + o.height and o.row_off < r.row_off + r.height
                        and r.col_off < o.col_off + o.width and o.col_off < r.col_off + r.width):
                    raise PipelineError(f"Planned windows overlap for {track_id}: {o.chunk_name} and {r.chunk_name}")
            cells.setdefault(key, []).append(r)
    return scope


def scope_fingerprint(scope: pd.DataFrame) -> str:
    ids = sorted(_window_id(r) for _, r in scope.iterrows())
    return hashlib.sha256("|".join(ids).encode()).hexdigest()


def _verified_file_windows(run_dir: Path, track_id: str) -> dict[Path, str]:
    """{absolute file path: planned window id of its manifest row} for VERIFIED / VERIFIED_EMPTY rows."""
    man = _read_manifest(run_dir)
    rows = man[(man["track_id"] == track_id) & (man["state"].isin(_FILE_STATES))]
    out: dict[Path, str] = {}
    for _, row in rows.iterrows():
        for part in str(row["local_paths"]).split(";"):
            part = part.strip()
            if not part:
                continue
            p = Path(part)
            p = p if p.is_absolute() else run_dir / p
            if not p.exists():
                raise PipelineError(f"Manifest says {row['state']} but the file is missing: {p}")
            out[p.resolve()] = _window_id(row)
    return out


def verified_chunk_files(run_dir: str | Path, track_id: str) -> list[Path]:
    """Absolute paths of all local files of VERIFIED / VERIFIED_EMPTY manifest rows of this track."""
    return sorted(_verified_file_windows(Path(run_dir), track_id))


def _audit_dir(cfg: dict, meta: dict) -> Path:
    """The audit this run was planned from (``export_meta.json`` ``audit_dir``, relative to the project root).

    A missing audit is an error, not a skip: without it we cannot tell which acquisitions are missing.
    """
    value = meta.get("audit_dir")
    if not value:
        raise AuditRequired("export_meta.json has no 'audit_dir': cannot check the stack against the audit.")
    p = Path(value)
    p = p if p.is_absolute() else root(cfg) / p
    if not p.is_dir():
        raise AuditRequired(f"Audit folder of this run not found: {p}. Restore it (it is needed for QA).")
    if not (p / "s1_acquisitions.csv").exists():
        raise AuditRequired(f"{p / 's1_acquisitions.csv'} not found: the audit folder is incomplete.")
    return p


def _audit_acquisitions(audit_dir: Path) -> pd.DataFrame:
    try:
        from .audit import load_acquisitions
    except ImportError:
        load_acquisitions = None
    if load_acquisitions is not None:
        return load_acquisitions(audit_dir)
    return pd.read_csv(audit_dir / "s1_acquisitions.csv", dtype=str)


def stack_dir(run_dir: str | Path, track_id: str) -> Path:
    return Path(run_dir) / "stack" / f"track_{track_id}"


# ---------------------------------------------------------------- VRT writing (no GDAL bindings)
@dataclasses.dataclass(frozen=True)
class _Source:
    path: Path
    row_off: int
    col_off: int
    width: int
    height: int
    count: int
    dtype: str
    block_x: int
    block_y: int


def _read_header(path: Path, gd: dict, grid_crs: CRS, dtype: str, min_bands: int) -> _Source:
    """Header of one chunk file, checked against the master grid. Raises on any mismatch."""
    res = float(gd["res"])
    with rasterio.open(path) as src:
        t = src.transform
        if src.crs is None or not same_crs(src.crs, grid_crs):
            raise PipelineError(f"{path}: CRS {src.crs} differs from grid CRS {gd['crs']}")
        if abs(t.a - res) > 1e-6 or abs(t.e + res) > 1e-6 or abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
            raise PipelineError(f"{path}: pixel size/rotation {tuple(t)[:6]} does not match the {res} m grid")
        col_f = (t.c - gd["x0"]) / res
        row_f = (gd["y0"] - t.f) / res
        col, row = int(round(col_f)), int(round(row_f))
        if abs(col_f - col) > 1e-6 or abs(row_f - row) > 1e-6:
            raise PipelineError(f"{path}: origin ({t.c}, {t.f}) is not snapped to the master grid")
        if col < 0 or row < 0 or col + src.width > gd["width"] or row + src.height > gd["height"]:
            raise PipelineError(f"{path}: window row {row} col {col} size {src.height}x{src.width} lies outside the grid")
        if src.dtypes[0] != dtype:
            raise PipelineError(f"{path}: data type {src.dtypes[0]} differs from the run's export dtype {dtype}")
        if src.count < min_bands:
            raise PipelineError(f"{path}: {src.count} bands, but the band layout needs {min_bands}")
        block_y, block_x = src.block_shapes[0]
        return _Source(path, row, col, src.width, src.height, src.count, src.dtypes[0], block_x, block_y)


def _nodata_text(nodata: float) -> str:
    v = float(nodata)
    if math.isnan(v):
        return "nan"
    return str(int(v)) if v.is_integer() else repr(v)


def _relpath(path: Path, vrt_dir: Path) -> tuple[str, int]:
    try:
        return Path(os.path.relpath(path, vrt_dir)).as_posix(), 1
    except ValueError:  # different drive on Windows: keep absolute
        return Path(path).as_posix(), 0


def _vrt_header(gd: dict, wkt: str) -> list[str]:
    res = float(gd["res"])
    return [
        f'<VRTDataset rasterXSize="{int(gd["width"])}" rasterYSize="{int(gd["height"])}">',
        f"  <SRS>{escape(wkt)}</SRS>",
        f"  <GeoTransform>{float(gd['x0'])!r}, {res!r}, 0.0, {float(gd['y0'])!r}, 0.0, {-res!r}</GeoTransform>",
    ]


def _complex_source(rel: str, relative: int, band: int, width: int, height: int, gdal_type: str,
                    block_x: int, block_y: int, col_off: int, row_off: int, nodata: str) -> str:
    return (
        "    <ComplexSource>\n"
        f'      <SourceFilename relativeToVRT="{relative}">{escape(rel)}</SourceFilename>\n'
        f"      <SourceBand>{band}</SourceBand>\n"
        f'      <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="{gdal_type}" '
        f'BlockXSize="{block_x}" BlockYSize="{block_y}" />\n'
        f'      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />\n'
        f'      <DstRect xOff="{col_off}" yOff="{row_off}" xSize="{width}" ySize="{height}" />\n'
        f"      <NODATA>{nodata}</NODATA>\n"
        "    </ComplexSource>"
    )


def _date_vrt_xml(gd: dict, wkt: str, gdal_type: str, nodata: str, name: str, band_idx: int,
                  sources: list[_Source], vrt_dir: Path) -> str:
    lines = _vrt_header(gd, wkt)
    lines += [f'  <VRTRasterBand dataType="{gdal_type}" band="1">',
              f"    <Description>{escape(name)}</Description>",
              f"    <NoDataValue>{nodata}</NoDataValue>"]
    for s in sources:
        rel, relative = _relpath(s.path, vrt_dir)
        lines.append(_complex_source(rel, relative, band_idx, s.width, s.height, gdal_type,
                                     s.block_x, s.block_y, s.col_off, s.row_off, nodata))
    lines += ["  </VRTRasterBand>", "</VRTDataset>", ""]
    return "\n".join(lines)


def _stack_vrt_xml(gd: dict, wkt: str, gdal_type: str, nodata: str, names: list[str], date_vrts: list[Path],
                   vrt_dir: Path) -> str:
    lines = _vrt_header(gd, wkt)
    w, h = int(gd["width"]), int(gd["height"])
    for i, (name, p) in enumerate(zip(names, date_vrts), start=1):
        rel, relative = _relpath(p, vrt_dir)
        lines += [f'  <VRTRasterBand dataType="{gdal_type}" band="{i}">',
                  f"    <Description>{escape(name)}</Description>",
                  f"    <NoDataValue>{nodata}</NoDataValue>",
                  _complex_source(rel, relative, 1, w, h, gdal_type, 128, 128, 0, 0, nodata),
                  "  </VRTRasterBand>"]
    lines += ["</VRTDataset>", ""]
    return "\n".join(lines)


def _sidecar(vrt: Path) -> Path:
    return vrt.with_name(vrt.name + SOURCES_SUFFIX)


def _is_current(vrt: Path, digest: str) -> bool:
    side = _sidecar(vrt)
    try:
        return vrt.stat().st_size > 0 and side.read_text().strip() == digest
    except OSError:
        return False


def _write_vrt(vrt: Path, xml: str, digest: str) -> None:
    # VRT first, fingerprint last: a crash in between leaves an old/missing fingerprint -> rebuilt next time.
    atomic_write_text(vrt, xml)
    atomic_write_text(_sidecar(vrt), digest + "\n")


def _base_digest(gd: dict, meta: dict, files: list[Path], run_dir: Path) -> str:
    h = hashlib.sha256()
    h.update(json.dumps([gd["crs"], gd["res"], gd["x0"], gd["y0"], gd["width"], gd["height"],
                         meta["dtype"], _nodata_text(meta["nodata"])]).encode())
    base = run_dir.resolve()
    for f in sorted(files, key=str):
        st = f.stat()
        try:
            rel = f.relative_to(base).as_posix()
        except ValueError:
            rel = f.as_posix()
        h.update(f"\n{rel}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def build_date_vrts(cfg: dict, run_dir: str | Path, track_id: str, force: bool = False) -> list[Path]:
    """One single-band VRT per (acquisition, polarisation), covering the full grid extent.

    Chunk file headers are read (and checked against the grid) only when at least one VRT needs to
    be (re)built. A VRT whose input fingerprint is unchanged is skipped without reading anything.
    """
    run_dir = Path(run_dir)
    gd = load_grid(cfg)
    meta = read_export_meta(cfg, run_dir)
    bl = read_band_layout(run_dir, track_id)
    order = _acquisition_order(bl, track_id)
    files = verified_chunk_files(run_dir, track_id)
    if not files:
        raise PipelineError(f"No VERIFIED chunk files for track {track_id} in {run_dir} - download first.")

    out_dir = stack_dir(run_dir, track_id)
    clean_stale_tmp(out_dir, recursive=True)
    gdal_type = _GDAL_TYPES[meta["dtype"]]
    nodata = _nodata_text(meta["nodata"])
    base = _base_digest(gd, meta, files, run_dir)
    headers: list[_Source] | None = None
    wkt: str | None = None
    outputs: list[Path] = []
    rebuilt = 0
    for pol in _pols(bl):
        pol_rows = bl[bl["pol"] == pol].drop_duplicates("acquisition_id").set_index("acquisition_id")
        (out_dir / pol).mkdir(parents=True, exist_ok=True)
        for acq_id in order:
            if acq_id not in pol_rows.index:
                continue
            band_idx = int(pol_rows.loc[acq_id, "band_idx"])
            name = _band_name(pol, acq_id, track_id)
            out = out_dir / pol / f"{name}.vrt"
            outputs.append(out)
            digest = hashlib.sha256(f"{base}|{band_idx}|{name}".encode()).hexdigest()
            if not force and _is_current(out, digest):
                continue
            if headers is None:
                grid_crs = CRS.from_user_input(gd["crs"])
                min_bands = int(bl["band_idx"].max())
                headers = [_read_header(f, gd, grid_crs, meta["dtype"], min_bands) for f in files]
                wkt = grid_crs.to_wkt()
            _write_vrt(out, _date_vrt_xml(gd, wkt, gdal_type, nodata, name, band_idx, headers, out.parent), digest)
            rebuilt += 1
    log.info("date VRTs for %s: rebuilt %d/%d, skipped %d/%d (inputs unchanged)",
             track_id, rebuilt, len(outputs), len(outputs) - rebuilt, len(outputs))
    return outputs


def build_stack_vrts(cfg: dict, run_dir: str | Path, track_id: str, force: bool = False) -> dict[str, Path]:
    """stack_<POL>.vrt = the per-date VRTs as bands, in acquisition-time order."""
    run_dir = Path(run_dir)
    gd = load_grid(cfg)
    bl = read_band_layout(run_dir, track_id)
    order = _acquisition_order(bl, track_id)
    meta = read_export_meta(cfg, run_dir)
    gdal_type = _GDAL_TYPES[meta["dtype"]]
    nodata = _nodata_text(meta["nodata"])
    out_dir = stack_dir(run_dir, track_id)
    result = {}
    for pol in _pols(bl):
        present = set(bl.loc[bl["pol"] == pol, "acquisition_id"])
        names = [_band_name(pol, a, track_id) for a in order if a in present]
        date_vrts = [out_dir / pol / f"{n}.vrt" for n in names]
        h = hashlib.sha256(json.dumps([gd["width"], gd["height"], gd["x0"], gd["y0"], gd["res"], gd["crs"],
                                       meta["dtype"], nodata]).encode())
        for n, p in zip(names, date_vrts):
            try:
                side = _sidecar(p).read_text().strip()
            except OSError:
                side = ""
            if not p.exists() or not side:
                raise PipelineError(f"Missing date VRT {p} - run build_date_vrts first.")
            h.update(f"\n{n}|{side}".encode())
        digest = h.hexdigest()
        out = out_dir / f"stack_{pol}.vrt"
        result[pol] = out
        if not force and _is_current(out, digest):
            log.info("stack VRT %s is current - skipped", out.name)
            continue
        wkt = CRS.from_user_input(gd["crs"]).to_wkt()
        _write_vrt(out, _stack_vrt_xml(gd, wkt, gdal_type, nodata, names, date_vrts, out.parent), digest)
    return result


# ---------------------------------------------------------------- valid pixel counting
def _init_worker(cache_mb: int) -> None:
    """Worker process setup: its own GDAL block cache, sized inside the pipeline's memory budget."""
    os.environ["GDAL_CACHEMAX"] = str(int(cache_mb))
    env = rasterio.Env(GDAL_CACHEMAX=int(cache_mb))
    env.__enter__()  # active for the whole life of the worker process


def _count_valid_file(path: str, gd_x0: float, gd_y0: float, res: float, index_path: str,
                      nodata: float, is_float: bool, rows_per_block: int) -> tuple[np.ndarray, int]:
    """Valid-pixel counts inside the AOI for every band of one chunk file, reading block by block.

    Module-level so it can run in a worker process.
    """
    with rasterio.open(path) as src, rasterio.open(index_path) as idx:
        col_off = int(round((src.transform.c - gd_x0) / res))
        row_off = int(round((gd_y0 - src.transform.f) / res))
        counts = np.zeros(src.count, dtype=np.int64)
        blocks = 0
        for r in range(0, src.height, rows_per_block):
            h = min(rows_per_block, src.height - r)
            data = src.read(window=Window(0, r, src.width, h))
            aoi = idx.read(3, window=Window(col_off, row_off + r, src.width, h)) == 1
            valid = data != nodata
            if is_float:
                valid &= np.isfinite(data)
            valid &= aoi[None, :, :]
            counts += valid.sum(axis=(1, 2))
            blocks += 1
    return counts, blocks


def _count_aoi_window(index_path: str, row_off: int, col_off: int, height: int, width: int,
                      rows_per_block: int) -> tuple[int, int]:
    total, blocks = 0, 0
    with rasterio.open(index_path) as idx:
        for r in range(0, height, rows_per_block):
            h = min(rows_per_block, height - r)
            total += int((idx.read(3, window=Window(col_off, row_off + r, width, h)) == 1).sum())
            blocks += 1
    return total, blocks


def _cache_parts(sdir: Path) -> list[Path]:
    d = sdir / VALID_CACHE
    if not d.exists():
        return []
    parts = []
    for p in d.glob("part-*.csv"):
        try:
            parts.append((int(p.stem.split("-", 1)[1]), p))
        except ValueError:
            continue
    return [p for _, p in sorted(parts)]


def _load_valid_cache(sdir: Path) -> pd.DataFrame:
    """All cached counts; for duplicate (path, band) entries the newest part file wins."""
    frames = []
    for i, p in enumerate(_cache_parts(sdir)):
        try:
            df = pd.read_csv(p, dtype={"path": str, "key": str})
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, OSError):
            log.warning("valid-pixel cache part %s unreadable - ignored", p)
            continue
        if set(_CACHE_COLUMNS) - set(df.columns):
            continue
        df = df[_CACHE_COLUMNS].copy()
        df["_part"] = i
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=_CACHE_COLUMNS)
    allc = pd.concat(frames, ignore_index=True).sort_values("_part", kind="stable")
    return allc.drop_duplicates(["path", "band_idx"], keep="last").drop(columns="_part").reset_index(drop=True)


def _next_part_path(sdir: Path) -> Path:
    parts = _cache_parts(sdir)
    n = int(parts[-1].stem.split("-", 1)[1]) + 1 if parts else 1
    return sdir / VALID_CACHE / f"part-{n:06d}.csv"


def _append_cache_part(sdir: Path, rows: list[pd.DataFrame]) -> None:
    if rows:
        atomic_write_csv(pd.concat(rows, ignore_index=True)[_CACHE_COLUMNS], _next_part_path(sdir))


def _compact_cache(sdir: Path, current: pd.DataFrame, n_loaded: int) -> None:
    """Merge all parts into one (dropping entries of files that are no longer verified)."""
    parts = _cache_parts(sdir)
    legacy = sdir / LEGACY_VALID_CACHE
    if len(parts) <= 1 and n_loaded == len(current) and not legacy.exists():
        return
    if not current.empty or parts:
        atomic_write_csv(current[_CACHE_COLUMNS] if len(current) else pd.DataFrame(columns=_CACHE_COLUMNS),
                         _next_part_path(sdir))
    for p in parts:  # older parts; the newest part (just written) already holds everything
        p.unlink(missing_ok=True)
    legacy.unlink(missing_ok=True)


def compute_valid_pct(cfg: dict, run_dir: str | Path, track_id: str, force: bool = False) -> pd.DataFrame:
    """Percentage of planned-scope AOI pixels that hold data, per acquisition and polarisation.

    Denominator = AOI pixels (pixel_index.tif band 3) inside the planned scope of this track (see
    module docstring). Pixels of planned chunks that failed or are unverified count as missing.
    Work is split per chunk file; workers, block size and each worker's GDAL cache come from the
    machine's resources (the caches are counted inside the memory budget).

    Caches (resume, never redo):
    - ``_valid_cache/part-*.csv``: counts per (chunk file, band), keyed by relative path, size, mtime,
      the pixel-index file, dtype/nodata and the file's planned window. One new part per finished
      batch (a crash loses at most one batch); compacted into a single part at the end. The key holds
      the file's OWN planned window rather than the whole scope, so planning more chunks later
      (pilot -> full run) does not force a re-read of files that were already counted.
    - ``_aoi_cache.json``: AOI pixel totals, keyed by the same index key and the scope fingerprint.
    """
    run_dir = Path(run_dir)
    gd = load_grid(cfg)
    meta = read_export_meta(cfg, run_dir)
    bl = read_band_layout(run_dir, track_id)
    scope = planned_scope(run_dir, track_id)
    scope_fp = scope_fingerprint(scope)
    file_windows = _verified_file_windows(run_dir, track_id)
    index_path = grid_dir(cfg) / "pixel_index.tif"
    if not index_path.exists():
        raise PipelineError(f"{index_path} not found - build the grid and pixel index first.")
    sdir = stack_dir(run_dir, track_id)
    sdir.mkdir(parents=True, exist_ok=True)
    clean_stale_tmp(sdir, recursive=True)

    is_float = meta["dtype"] == "float32"
    bytes_per_value = 4 if is_float else 2
    n_bands = int(bl["band_idx"].max())
    res = resources.detect_resources(cfg)
    workers = resources.cpu_workers(res, cfg)
    budget = resources.memory_budget_bytes(res, cfg)
    cache_bytes = max(16 * _MB, min(resources.gdal_cache_bytes(res), budget // (4 * max(1, workers))))
    frac = float(((cfg.get("resources") or {}).get("memory_fraction", 0.6)) or 0.6)
    # Worker GDAL caches live inside the same RAM: shrink what block_rows may plan with.
    reduced = dataclasses.replace(
        res, memory_available_bytes=max(0, res.memory_available_bytes - int(cache_bytes * workers / frac)))
    max_w = max([gd["chunk_px"]] + [int(ch["width"]) for ch in gd["chunks"]])
    rows = resources.block_rows(reduced, max_w, n_bands, bytes_per_value, workers=workers, cfg=cfg)
    cache_mb = max(16, cache_bytes // _MB)
    idx_stat = index_path.stat()
    key = f"{idx_stat.st_size}:{idx_stat.st_mtime_ns}:{meta['dtype']}:{meta['nodata']:g}"

    # ---- cache lookup
    def rel(p: Path) -> str:
        try:
            return p.relative_to(run_dir.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    file_info = {}
    for f, window in file_windows.items():
        st = f.stat()
        file_info[rel(f)] = (f, int(st.st_size), int(st.st_mtime_ns), f"{key}|{window}")
    cache = pd.DataFrame(columns=_CACHE_COLUMNS) if force else _load_valid_cache(sdir)
    n_loaded = len(cache)
    kept_rows, results = [], {}
    for rp, grp in cache.groupby("path", sort=False):
        if rp not in file_info:
            continue  # file no longer verified: dropped on compaction
        _, size, mtime, fkey = file_info[rp]
        ok = grp[(grp["size"] == size) & (grp["mtime_ns"] == mtime) & (grp["key"] == fkey)]
        if len(ok) and len(ok) == int(ok["n_bands"].iloc[0]) and set(ok["band_idx"]) == set(range(1, len(ok) + 1)):
            counts = np.zeros(len(ok), dtype=np.int64)
            counts[ok["band_idx"].to_numpy(int) - 1] = ok["valid_pixels"].to_numpy(np.int64)
            results[rp] = counts
            kept_rows.append(ok)
    reused_bands = int(sum(len(c) for c in results.values()))
    todo = [rp for rp in file_info if rp not in results]

    # ---- AOI denominators (cached)
    blocks = 0
    aoi_cache_path = sdir / AOI_CACHE
    planned_px = total_px = None
    if not force and aoi_cache_path.exists():
        try:
            c = json.loads(aoi_cache_path.read_text())
            if c.get("key") == key and c.get("scope_fingerprint") == scope_fp and c.get("n_chunks") == len(gd["chunks"]):
                planned_px, total_px = int(c["aoi_pixels_planned"]), int(c["aoi_pixels_total"])
        except (ValueError, KeyError, TypeError):
            planned_px = total_px = None

    planned_args = [(str(index_path), int(r.row_off), int(r.col_off), int(r.height), int(r.width), rows)
                    for r in scope.itertuples()]
    total_args = [(str(index_path), ch["row_off"], ch["col_off"], ch["height"], ch["width"], rows) for ch in gd["chunks"]]
    n_aoi_jobs = 0 if planned_px is not None else len(planned_args) + len(total_args)
    use_pool = workers > 1 and (len(todo) + n_aoi_jobs) > 1
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                initializer=_init_worker, initargs=(cache_mb,)) if use_pool else None)
    new_rows: list[pd.DataFrame] = []
    try:
        with rasterio.Env(GDAL_CACHEMAX=int(cache_mb)):
            if planned_px is None:
                args = planned_args + total_args
                aoi_results = (list(pool.map(_count_aoi_window, *zip(*args))) if pool
                               else [_count_aoi_window(*a) for a in args])
                planned_px = sum(t for t, _ in aoi_results[: len(planned_args)])
                total_px = sum(t for t, _ in aoi_results[len(planned_args):])
                blocks += sum(b for _, b in aoi_results)
                atomic_write_text(aoi_cache_path, json.dumps({
                    "key": key, "scope_fingerprint": scope_fp, "n_chunks": len(gd["chunks"]),
                    "aoi_pixels_planned": planned_px, "aoi_pixels_total": total_px,
                }))

            # ---- chunk files, in batches; each finished batch becomes one cache part file
            batch = max(1, workers * 2)
            for i in range(0, len(todo), batch):
                part = todo[i:i + batch]
                args = [(str(file_info[rp][0]), gd["x0"], gd["y0"], gd["res"], str(index_path), meta["nodata"],
                         is_float, rows) for rp in part]
                outs = list(pool.map(_count_valid_file, *zip(*args))) if pool else [_count_valid_file(*a) for a in args]
                batch_rows = []
                for rp, (counts, b) in zip(part, outs):
                    results[rp] = counts
                    blocks += b
                    _, size, mtime, fkey = file_info[rp]
                    batch_rows.append(pd.DataFrame({
                        "path": rp, "size": size, "mtime_ns": mtime, "n_bands": len(counts),
                        "band_idx": np.arange(1, len(counts) + 1), "key": fkey, "valid_pixels": counts,
                    }))
                _append_cache_part(sdir, batch_rows)
                new_rows.extend(batch_rows)
    finally:
        if pool:
            pool.shutdown()

    current = (pd.concat(kept_rows + new_rows, ignore_index=True)[_CACHE_COLUMNS] if (kept_rows or new_rows)
               else pd.DataFrame(columns=_CACHE_COLUMNS))
    _compact_cache(sdir, current, n_loaded + int(sum(len(r) for r in new_rows)))

    total_bands = int(sum(len(c) for c in results.values()))
    log.info("valid pixels for %s: reused %d/%d cached chunk-band results, read %d chunk files",
             track_id, reused_bands, total_bands, len(todo))

    counts = np.zeros(n_bands, dtype=np.int64)
    for c in results.values():
        counts[: min(len(c), n_bands)] += c[:n_bands]

    out = bl[["acquisition_id", "date_utc", "pol", "band_idx"]].copy()
    out["valid_pixels"] = [int(counts[i - 1]) for i in out["band_idx"]]
    out["aoi_pixels"] = planned_px
    out["aoi_pixels_total"] = total_px
    out["aoi_planned_pct"] = planned_px / total_px * 100.0 if total_px else np.nan
    out["valid_pct"] = out["valid_pixels"] / planned_px * 100.0 if planned_px else np.nan
    out = out.reset_index(drop=True)
    out.attrs.update(blocks_read=blocks, rows_per_block=rows, workers=workers, reused=reused_bands,
                     total=total_bands, computed_files=len(todo), scope_fingerprint=scope_fp,
                     gdal_cache_mb=int(cache_mb))
    return out


# ---------------------------------------------------------------- dates table
def _rain_table(audit_dir: Path) -> pd.DataFrame | None:
    """Rain per acquisition; None when rain flags are disabled/not produced (columns then stay NaN)."""
    if not (audit_dir / "rain_flags.csv").exists():
        return None
    rain = pd.read_csv(audit_dir / "rain_flags.csv")
    if rain.empty:
        return None
    # level "chunk" has one row per chunk: summarise to the acquisition mean
    return rain.groupby("acquisition_id")[["rain_6h_mm", "rain_24h_mm"]].mean().reset_index()


def temporal_neighbor_counts(n: int, cfg: dict) -> list[int]:
    """Acquisitions inside the multi-temporal speckle window of each date (same track).

    The window is ±``ard.temporal_half_window`` acquisitions, clipped at the ends of the series, so
    the first and last dates average fewer images and keep more speckle. 1 when the multi-temporal
    filter is off.
    """
    ard = cfg.get("ard") or {}
    if not ard.get("multitemporal", False):
        return [1] * n
    half = int(ard.get("temporal_half_window", 0) or 0)
    return [min(n - 1, k + half) - max(0, k - half) + 1 for k in range(n)]


def dates_table(cfg: dict, run_dir: str | Path, track_id: str, valid: pd.DataFrame) -> pd.DataFrame:
    run_dir = Path(run_dir)
    meta = read_export_meta(cfg, run_dir)
    bl = read_band_layout(run_dir, track_id)
    order = _acquisition_order(bl, track_id)
    pols = _pols(bl)
    acq = bl.drop_duplicates("acquisition_id").set_index("acquisition_id").loc[order].reset_index()
    audit_dir = _audit_dir(cfg, meta)

    rain = _rain_table(audit_dir)
    if rain is not None:
        acq = acq.merge(rain, on="acquisition_id", how="left")
    else:
        acq["rain_6h_mm"] = np.nan
        acq["rain_24h_mm"] = np.nan

    for pol in pols:
        v = valid[valid["pol"] == pol].set_index("acquisition_id")["valid_pct"]
        acq[f"valid_pct_{pol}"] = acq["acquisition_id"].map(v)

    min_valid = float((cfg.get("qa", {}) or {}).get("min_valid_pct", 80))
    max_gap = float((cfg.get("audit", {}) or {}).get("max_gap_days", 12))
    dates = pd.to_datetime(acq["date_utc"].str[:10])
    gaps = dates.diff().dt.days
    flags = []
    for i, r in acq.iterrows():
        f = [f"LOW_VALID_{p}" for p in pols if pd.notna(r[f"valid_pct_{p}"]) and r[f"valid_pct_{p}"] < min_valid]
        if pd.notna(r["rain_24h_mm"]) and r["rain_24h_mm"] >= RAIN_FLAG_MM_24H:
            f.append("RAIN_24H")
        if pd.notna(gaps.iloc[i]) and gaps.iloc[i] > max_gap:
            f.append("GAP_BEFORE")
        flags.append(";".join(f))
    acq["flags"] = flags
    acq["stack_band"] = range(1, len(acq) + 1)
    acq["aoi_planned_pct"] = float(valid["aoi_planned_pct"].iloc[0]) if len(valid) else np.nan
    acq["n_temporal_neighbors"] = temporal_neighbor_counts(len(acq), cfg)
    cols = (["stack_band", "acquisition_id", "date_utc", "datetime_utc", "date_local", "platforms", "n_slices",
             "rain_6h_mm", "rain_24h_mm"] + [f"valid_pct_{p}" for p in pols]
            + ["aoi_planned_pct", "n_temporal_neighbors", "flags"])
    acq["date_utc"] = acq["date_utc"].str[:10]
    return acq[cols]


def write_dates_table(cfg: dict, run_dir: str | Path, track_id: str, valid: pd.DataFrame) -> Path:
    out = stack_dir(run_dir, track_id) / "dates.csv"
    atomic_write_csv(dates_table(cfg, run_dir, track_id, valid), out)
    return out


# ---------------------------------------------------------------- QA issues
def _issue(type_: str, track_id: str, subject: str, detail: str, acknowledged: set[str]) -> dict:
    issue_id = f"{type_}:{track_id}:{subject}"
    return dict(issue_id=issue_id, type=type_, track_id=track_id, subject=subject, detail=detail,
                acknowledged=issue_id in acknowledged)


def qa_issues(cfg: dict, run_dir: str | Path, track_id: str, valid: pd.DataFrame) -> list[dict]:
    """Every problem the user must see before trusting the stack of this track."""
    run_dir = Path(run_dir)
    ack = set((cfg.get("qa", {}) or {}).get("acknowledged_issues", []) or [])
    min_valid = float((cfg.get("qa", {}) or {}).get("min_valid_pct", 80))
    bl = read_band_layout(run_dir, track_id)
    issues: list[dict] = []

    # 1. acquisitions the audit expected but the export does not contain
    audit_acq = _audit_acquisitions(_audit_dir(cfg, read_export_meta(cfg, run_dir)))
    expected = audit_acq[audit_acq["track_id"] == track_id]
    have = set(bl["acquisition_id"])
    for _, r in expected.sort_values("acquisition_id").iterrows():
        if r["acquisition_id"] not in have:
            subj = _acq_token(str(r["acquisition_id"]), track_id)
            cov = r.get("aoi_coverage_pct", "")
            issues.append(_issue("MISSING_ACQUISITION", track_id, subj,
                                 f"{r['acquisition_id']} (AOI coverage {cov}%) is in the audit but not exported.", ack))

    # 2. low valid percentage (one issue per date, listing the polarisations)
    for acq_id, grp in valid.groupby("acquisition_id", sort=False):
        low = grp[grp["valid_pct"] < min_valid]
        if not low.empty:
            parts = ", ".join(f"{r.pol} {r.valid_pct:.1f}%" for r in low.itertuples())
            scope_pct = float(grp["aoi_planned_pct"].iloc[0]) if "aoi_planned_pct" in grp else float("nan")
            issues.append(_issue("LOW_VALID", track_id, _acq_token(str(acq_id), track_id),
                                 f"{acq_id}: valid pixels inside the planned AOI scope {parts} (minimum {min_valid:g}%; "
                                 f"planned scope = {scope_pct:.1f}% of the AOI).", ack))

    # 3/4/5. failed, unverified and empty chunks (rows outside the planned scope are not issues)
    man = _read_manifest(run_dir)
    man = man[man["track_id"] == track_id]
    for _, r in man.sort_values("chunk_name").iterrows():
        state = r["state"]
        if state in _OUT_OF_SCOPE_STATES:
            continue
        if state in _FAILED_STATES:
            issues.append(_issue("FAILED_CHUNK", track_id, r["chunk_name"],
                                 f"state {state}, attempts {r['attempts']}, {r['error_class']}: {r['last_error']}", ack))
        elif state == "VERIFIED_EMPTY":
            issues.append(_issue("EMPTY_CHUNK", track_id, r["chunk_name"],
                                 "verified file(s) contain only nodata", ack))
        elif state != "VERIFIED":
            issues.append(_issue("UNVERIFIED_CHUNK", track_id, r["chunk_name"], f"state {state}", ack))
    return issues


_DECISION_COLUMNS = ["issue_id", "type", "track_id", "subject", "detail", "acknowledged"]


def write_decisions_required(run_dir: str | Path, issues: list[dict], track_id: str | list[str] | None = None,
                             acknowledged: list[str] | None = None) -> Path | None:
    """Write decisions_required.md/.csv and raise DecisionRequired for unacknowledged issues.

    ``track_id`` is the track, or list of tracks, whose issues are being (re)written: their old
    entries are replaced, issues of other tracks already in the file are kept. When None, the
    tracks present in ``issues`` are used. Returns None when, after merging, there are no issues.
    """
    run_dir = Path(run_dir)
    ack = set(acknowledged or [])
    new = pd.DataFrame([{**i, "acknowledged": bool(i.get("acknowledged")) or i["issue_id"] in ack} for i in issues],
                       columns=_DECISION_COLUMNS)
    if track_id is None:
        tracks = set(new["track_id"])
    else:
        tracks = {track_id} if isinstance(track_id, str) else set(track_id)
    csv_path, md_path = run_dir / "decisions_required.csv", run_dir / "decisions_required.md"
    if csv_path.exists():
        old = pd.read_csv(csv_path, dtype=str).fillna("")
        old = old[~old["track_id"].isin(tracks)]
        old["acknowledged"] = old["acknowledged"].astype(str).str.lower() == "true"
        allissues = pd.concat([old, new], ignore_index=True) if not new.empty else old
    else:
        allissues = new
    unack = new[~new["acknowledged"].astype(bool)]

    if allissues.empty:
        if csv_path.exists() or md_path.exists():
            atomic_write_csv(pd.DataFrame(columns=_DECISION_COLUMNS), csv_path)
            atomic_write_text(md_path, "# Decisions required\n\nNo open issues.\n")
        return None

    atomic_write_csv(allissues, csv_path)
    lines = [
        "# Decisions required",
        "",
        "The stack QA found the issues below. Nothing was skipped silently: each issue either needs a fix,",
        "or you accept it by adding its `issue_id` to `qa.acknowledged_issues` in your config and re-running",
        "the stack stage.",
        "",
    ]
    for type_ in ISSUE_TYPES:
        sub = allissues[allissues["type"] == type_]
        if sub.empty:
            continue
        what, todo = _ISSUE_HELP[type_]
        lines += [f"## {type_} ({len(sub)})", "", f"**What it means:** {what}", "", f"**What you can do:** {todo}", "",
                  "| issue_id | acknowledged | detail |", "|---|---|---|"]
        for r in sub.itertuples():
            detail = str(r.detail).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{r.issue_id}` | {'yes' if str(r.acknowledged).lower() == 'true' else 'no'} | {detail} |")
        lines.append("")
    atomic_write_text(md_path, "\n".join(lines))

    if not unack.empty:
        raise DecisionRequired(
            f"{len(unack)} unacknowledged QA issue(s) for track(s) {sorted(tracks)}; see {md_path}"
        )
    return md_path


# ---------------------------------------------------------------- orchestration
def build_stacks(cfg: dict, run_dir: str | Path, track_ids: list[str], force: bool = False) -> dict[str, Path]:
    """Build the stack of EVERY track first, then report QA issues of all tracks together.

    For each track: date VRTs -> stack VRTs -> valid % -> dates.csv -> QA issues. Only after all
    tracks are processed is ``decisions_required.md/.csv`` written once and a single
    ``DecisionRequired`` raised, so one track's issues never prevent the next track from being built.

    Errors that concern the whole run (``GridMismatch``, ``AuditRequired``) stop immediately. Any
    other ``PipelineError`` of one track is collected; the remaining tracks are still built and the
    collected errors are raised together as a ``PipelineError`` at the end (open decisions of the
    built tracks are mentioned in the same message).
    Safe to re-run at any time: only changed/new work is done (see module docstring).
    """
    run_dir = Path(run_dir)
    track_ids = list(dict.fromkeys(track_ids))
    if not track_ids:
        raise AuditRequired("No tracks to stack: select tracks in s1.tracks after reviewing the audit.")
    res = resources.detect_resources(cfg)
    outputs: dict[str, Path] = {}
    issues: list[dict] = []
    errors: dict[str, str] = {}
    for track_id in track_ids:
        try:
            with rasterio.Env(GDAL_CACHEMAX=max(64, resources.gdal_cache_bytes(res) // _MB)):
                build_date_vrts(cfg, run_dir, track_id, force=force)
                build_stack_vrts(cfg, run_dir, track_id, force=force)
                valid = compute_valid_pct(cfg, run_dir, track_id, force=force)
            write_dates_table(cfg, run_dir, track_id, valid)
            track_issues = qa_issues(cfg, run_dir, track_id, valid)
        except (GridMismatch, AuditRequired):
            raise
        except PipelineError as exc:
            errors[track_id] = str(exc)
            log.error("stack for %s failed: %s", track_id, exc)
            continue
        issues.extend(track_issues)
        outputs[track_id] = stack_dir(run_dir, track_id)
        log.info("stack for %s built (%d QA issues)", track_id, len(track_issues))

    decision: DecisionRequired | None = None
    if outputs:
        try:
            write_decisions_required(run_dir, issues, track_id=list(outputs))
        except DecisionRequired as exc:
            decision = exc
    if errors:
        msg = "; ".join(f"{t}: {e}" for t, e in errors.items())
        if decision:
            msg += f". Also open: {decision}"
        raise PipelineError(f"Stack failed for {len(errors)} of {len(track_ids)} track(s): {msg}")
    if decision:
        raise decision
    return outputs


def build_stack(cfg: dict, run_dir: str | Path, track_id: str, force: bool = False) -> Path:
    """Single-track convenience wrapper around ``build_stacks``."""
    return build_stacks(cfg, run_dir, [track_id], force=force)[track_id]
