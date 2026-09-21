"""Stage 0 — Master grid and chunks.

Why a master grid?
------------------
Earth Engine places every export on whatever pixel grid you ask for. If two chunks were exported
on grids shifted by even a few metres, the mosaic (VRT) would show seams or double pixels, and the
same ground location would get a different pixel id in different chunks.

So we define ONE master grid for the whole AOI:
- one projected CRS (``grid.crs``, e.g. UTM 51N) and one pixel size (``grid.res``, e.g. 10 m);
- the top-left corner snapped to a multiple of the pixel size;
- chunks are pixel-exact windows of that grid (``row_off``/``col_off``/``width``/``height``).

Every export uses the master grid transform, so all chunk files line up exactly.

``grid_def.json`` is written once and never overwritten with different content: all runs of a
season share the same grid, so pixel ids stay stable over time.
"""
from __future__ import annotations

import json
import logging
import math
import re
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box

from . import run_meta
from .config import aoi_path, grid_dir
from .errors import GridMismatch

_FLOAT_TOL = 1e-6
log = logging.getLogger(__name__)


# ---------------------------------------------------------------- helpers
def _snap_down(v: float, res: float) -> float:
    return round(math.floor(v / res) * res, 6)


def _snap_up(v: float, res: float) -> float:
    return round(math.ceil(v / res) * res, 6)


def _name_digits(n_rows: int, n_cols: int) -> int:
    """2 digits normally; 3 when the grid has >=100 chunk rows/cols; 4 (or more) when >=1000."""
    m = max(n_rows, n_cols)
    if m < 100:
        return 2
    if m < 1000:
        return 3
    return max(4, len(str(m - 1)))


def chunk_name(grid_row: int, grid_col: int, digits: int) -> str:
    return f"chunk_r{grid_row:0{digits}d}c{grid_col:0{digits}d}"


def read_aoi(cfg: dict, crs: str) -> gpd.GeoDataFrame:
    """AOI polygons in the grid CRS, exploded to single parts (better spatial-index selectivity)."""
    aoi = gpd.read_file(aoi_path(cfg)).to_crs(crs)
    aoi = aoi[~aoi.geometry.is_empty & aoi.geometry.notna()]
    return aoi[["geometry"]].explode(index_parts=False).reset_index(drop=True)


# ---------------------------------------------------------------- grid definition
def compute_grid_def(cfg: dict) -> dict:
    g = cfg["grid"]
    res, crs = float(g["res"]), g["crs"]
    buf, chunk_px = float(g["buffer_m"]), int(g["chunk_px"])

    aoi = read_aoi(cfg, crs)
    if aoi.empty:
        raise ValueError(f"AOI has no geometries: {aoi_path(cfg)}")
    minx, miny, maxx, maxy = aoi.total_bounds
    x0 = _snap_down(minx - buf, res)
    y0 = _snap_up(maxy + buf, res)
    x1 = _snap_up(maxx + buf, res)
    y1 = _snap_down(miny - buf, res)
    width = int(round((x1 - x0) / res))
    height = int(round((y0 - y1) / res))

    n_rows = math.ceil(height / chunk_px)
    n_cols = math.ceil(width / chunk_px)
    digits = _name_digits(n_rows, n_cols)

    # All candidate chunk boxes as vectorised arrays (≈1e5 boxes for a country-sized AOI).
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    row_off = rr * chunk_px
    col_off = cc * chunk_px
    hh = np.minimum(chunk_px, height - row_off)
    ww = np.minimum(chunk_px, width - col_off)
    bx0 = x0 + col_off * res
    by1 = y0 - row_off * res
    bx1 = bx0 + ww * res
    by0 = by1 - hh * res
    boxes = shapely.box(bx0, by0, bx1, by1)

    # Spatial index over AOI parts: only (box, aoi-part) pairs whose envelopes overlap are tested.
    tree = shapely.STRtree(aoi.geometry.values)
    box_idx, part_idx = tree.query(boxes, predicate="intersects")
    order = np.argsort(box_idx, kind="stable")
    box_idx, part_idx = box_idx[order], part_idx[order]

    parts = aoi.geometry.values
    chunks = []
    uniq, starts = np.unique(box_idx, return_index=True)
    ends = list(starts[1:]) + [len(box_idx)]
    for b, s, e in zip(uniq, starts, ends):
        cell = boxes[b]
        pieces = shapely.intersection(parts[part_idx[s:e]], cell)
        # AOI features may overlap each other; union the few pieces inside this cell.
        inter_area = shapely.union_all(pieces).area if e - s > 1 else pieces[0].area
        if inter_area <= 0:  # touches only along an edge/corner
            continue
        chunks.append(
            dict(
                chunk_id=len(chunks) + 1,
                name=chunk_name(int(rr[b]), int(cc[b]), digits),
                grid_row=int(rr[b]), grid_col=int(cc[b]),
                row_off=int(row_off[b]), col_off=int(col_off[b]),
                width=int(ww[b]), height=int(hh[b]),
                xmin=round(float(bx0[b]), 6), ymin=round(float(by0[b]), 6),
                xmax=round(float(bx1[b]), 6), ymax=round(float(by1[b]), 6),
                aoi_frac=round(min(1.0, inter_area / cell.area), 9),
            )
        )

    return dict(
        crs=crs, res=res, x0=x0, y0=y0, width=width, height=height,
        chunk_px=chunk_px, n_chunk_rows=n_rows, n_chunk_cols=n_cols,
        transform=[res, 0.0, x0, 0.0, -res, y0],  # GDAL/rasterio (a, b, c, d, e, f)
        pid_dtype="int32" if width * height < 2**31 - 1 else "int64",
        aoi_file=str(cfg["aoi"]["path"]),
        chunks=chunks,
    )


def _same(a, b, path: str = "") -> str | None:
    """Return a description of the first difference, or None if equal (floats within tolerance)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return f"{path or '<root>'}: keys differ ({sorted(set(a) ^ set(b))})"
        for k in a:
            d = _same(a[k], b[k], f"{path}.{k}" if path else k)
            if d:
                return d
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _same(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a == b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return None if math.isclose(a, b, rel_tol=0, abs_tol=_FLOAT_TOL) else f"{path}: {a!r} != {b!r}"
    return None if a == b else f"{path}: {a!r} != {b!r}"


def _write_chunks_gpkg(gd: dict, path: Path) -> None:
    """Write chunks.gpkg crash-safely.

    The GPKG driver needs a real file with a .gpkg extension, so the layer is written into a private
    temporary folder first and its bytes are then committed with run_meta.atomic_write_bytes
    (unique temp name + fsync + rename): a crash leaves either the old file or the new one.
    """
    gdf = gpd.GeoDataFrame(
        [dict(ch) for ch in gd["chunks"]],
        geometry=[box(ch["xmin"], ch["ymin"], ch["xmax"], ch["ymax"]) for ch in gd["chunks"]],
        crs=gd["crs"],
    )
    with tempfile.TemporaryDirectory(prefix="chunks_gpkg_") as tmp_dir:
        tmp = Path(tmp_dir) / "chunks.gpkg"
        gdf.to_file(tmp, layer="chunks", driver="GPKG")
        run_meta.atomic_write_bytes(path, tmp.read_bytes())


def _chunks_gpkg_ok(gd: dict, path) -> bool:
    if not path.exists():
        return False
    try:
        gdf = gpd.read_file(path, layer="chunks")
    except Exception:  # corrupt / partially written file
        return False
    return len(gdf) == len(gd["chunks"]) and list(gdf["name"]) == [ch["name"] for ch in gd["chunks"]]


# Keys that define pixel positions and chunk windows. Only these may block a rebuild: the AOI file path
# and aoi_frac (floating-point area shares that can differ slightly between PROJ/GEOS versions) do not
# move any pixel, so they must not raise GridMismatch.
GEOMETRY_KEYS = ("crs", "res", "x0", "y0", "width", "height", "chunk_px", "pid_dtype")
CHUNK_GEOMETRY_KEYS = ("name", "row_off", "col_off", "width", "height")
AOI_FRAC_WARN = 1e-3


def geometry_diff(old: dict, new: dict) -> str | None:
    """First difference in the geometry-defining part of two grid definitions, or None."""
    for key in GEOMETRY_KEYS:
        d = _same(old.get(key), new.get(key), key)
        if d:
            return d
    old_chunks, new_chunks = old.get("chunks", []), new.get("chunks", [])
    if len(old_chunks) != len(new_chunks):
        return f"chunks: {len(old_chunks)} stored vs {len(new_chunks)} recomputed"
    for i, (a, b) in enumerate(zip(old_chunks, new_chunks)):
        for key in CHUNK_GEOMETRY_KEYS:
            if a.get(key) != b.get(key):
                return f"chunks[{i}].{key}: {a.get(key)!r} != {b.get(key)!r}"
    return None


def _max_aoi_frac_drift(old: dict, new: dict) -> float:
    drift = 0.0
    for a, b in zip(old.get("chunks", []), new.get("chunks", [])):
        if a.get("aoi_frac") is not None and b.get("aoi_frac") is not None:
            drift = max(drift, abs(float(a["aoi_frac"]) - float(b["aoi_frac"])))
    return drift


def build_grid(cfg: dict) -> dict:
    """Write grid_def.json + chunks.gpkg. If a grid already exists its geometry must be identical.

    Idempotent and crash-safe: both files are written atomically; when grid_def.json exists but
    chunks.gpkg is missing or corrupt, only chunks.gpkg is regenerated from the stored grid.

    Raises GridMismatch when the stored grid's geometry (CRS, pixel size, origin, size, chunk size,
    pid dtype, chunk names and windows) differs from the one recomputed from the current config/AOI:
    changing it would silently change every pixel id. A moved AOI file or tiny aoi_frac differences
    only log a warning; the stored grid is kept unchanged.
    """
    out = grid_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    gd = compute_grid_def(cfg)
    gd_path = out / "grid_def.json"
    gpkg_path = out / "chunks.gpkg"

    if gd_path.exists():
        old = json.loads(gd_path.read_text())
        diff = geometry_diff(old, gd)
        if diff:
            raise GridMismatch(
                f"{gd_path} already exists and differs from the recomputed grid ({diff}). "
                "The grid must stay fixed; use a new aoi/season key to create a different grid."
            )
        if old.get("aoi_file") != gd.get("aoi_file"):
            log.warning("AOI file path changed (%r -> %r); grid geometry is identical, keeping the stored grid",
                        old.get("aoi_file"), gd.get("aoi_file"))
        drift = _max_aoi_frac_drift(old, gd)
        if drift > AOI_FRAC_WARN:
            log.warning("aoi_frac differs from the stored grid by up to %.4f (AOI geometry or GEOS/PROJ changed); "
                        "pixel geometry is identical, keeping the stored grid", drift)
        if not _chunks_gpkg_ok(old, gpkg_path):
            log.info("chunks.gpkg missing or corrupt; regenerating it from %s", gd_path)
            _write_chunks_gpkg(old, gpkg_path)
        return old

    run_meta.atomic_write_text(gd_path, json.dumps(gd, indent=2))
    _write_chunks_gpkg(gd, gpkg_path)
    return gd


def load_grid(cfg: dict) -> dict:
    return json.loads((grid_dir(cfg) / "grid_def.json").read_text())


def iter_chunks(gd: dict) -> Iterator[dict]:
    yield from gd["chunks"]


# ---------------------------------------------------------------- lookups (small bounded cache)
# Building the name/position dictionaries is O(n_chunks). They are cached so repeated lookups on the same
# grid dict are O(1). The cache is an LRU of a few entries: a long-running monitor calls load_grid every
# round (a new dict each time), and an unbounded cache would keep every one of those grids alive.
_LOOKUP_CACHE_SIZE = 4
_LOOKUP_CACHE: "OrderedDict[int, tuple[dict, int, dict, dict]]" = OrderedDict()
_LOOKUP_GUARD = threading.Lock()


def _lookups(gd: dict) -> tuple[dict, dict]:
    """(name -> chunk, (grid_row, grid_col) -> chunk) for this grid dict."""
    key = id(gd)
    with _LOOKUP_GUARD:
        cached = _LOOKUP_CACHE.get(key)
        # The cache holds a reference to gd, so its id cannot be reused by another dict while cached.
        if cached is not None and cached[0] is gd and cached[1] == len(gd["chunks"]):
            _LOOKUP_CACHE.move_to_end(key)
            return cached[2], cached[3]
    by_name = {ch["name"]: ch for ch in gd["chunks"]}
    by_rc = {(ch["grid_row"], ch["grid_col"]): ch for ch in gd["chunks"]}
    with _LOOKUP_GUARD:
        _LOOKUP_CACHE[key] = (gd, len(gd["chunks"]), by_name, by_rc)
        _LOOKUP_CACHE.move_to_end(key)
        while len(_LOOKUP_CACHE) > _LOOKUP_CACHE_SIZE:
            _LOOKUP_CACHE.popitem(last=False)
    return by_name, by_rc


def chunk_at(gd: dict, grid_row: int, grid_col: int) -> dict | None:
    """Chunk at a chunk-grid position, or None if that position does not touch the AOI."""
    return _lookups(gd)[1].get((grid_row, grid_col))


_SUB_RE = re.compile(r"^(?P<base>chunk_r\d+c\d+)(?:_s(?P<i>[0-3]))?$")


def chunk_by_name(gd: dict, name: str) -> dict:
    """Find a chunk by name; also resolves sub-chunk names '<chunk>_s<i>' created by split_chunk."""
    m = _SUB_RE.match(name)
    if not m:
        raise KeyError(name)
    base = _lookups(gd)[0].get(m.group("base"))
    if base is None:
        raise KeyError(name)
    if m.group("i") is None:
        return base
    for part in split_chunk(base, gd["res"]):
        if part["name"] == name:
            return part
    raise KeyError(name)


def split_chunk(ch: dict, res: float) -> list[dict]:
    """Split a chunk into up to 4 pixel-aligned parts (used when an EE task runs out of memory/time).

    Parts are numbered s0 (top-left), s1 (top-right), s2 (bottom-left), s3 (bottom-right); empty
    parts (1-pixel-wide chunks) are dropped but numbering is kept stable. Sub-chunks keep the parent's
    chunk_id/grid_row/grid_col and inherit its aoi_frac (approximate).
    """
    res = float(res)
    hw, hh = math.ceil(ch["width"] / 2), math.ceil(ch["height"] / 2)
    layout = [
        (0, 0, hh, hw),
        (0, hw, hh, ch["width"] - hw),
        (hh, 0, ch["height"] - hh, hw),
        (hh, hw, ch["height"] - hh, ch["width"] - hw),
    ]
    parts = []
    for i, (dr, dc, h, w) in enumerate(layout):
        if h <= 0 or w <= 0:
            continue
        x = ch["xmin"] + dc * res
        y = ch["ymax"] - dr * res
        parts.append(
            dict(
                chunk_id=ch["chunk_id"], name=f"{ch['name']}_s{i}",
                grid_row=ch["grid_row"], grid_col=ch["grid_col"],
                row_off=ch["row_off"] + dr, col_off=ch["col_off"] + dc, width=w, height=h,
                xmin=round(x, 6), ymin=round(y - h * res, 6),
                xmax=round(x + w * res, 6), ymax=round(y, 6),
                aoi_frac=ch.get("aoi_frac"),
            )
        )
    return parts


def chunk_region_coords(chunk: dict, inset_m: float = 0.25) -> list[float]:
    """[xmin, ymin, xmax, ymax] in the grid CRS, shrunk by `inset_m` on every side.

    Earth Engine exports every pixel that *touches* the region. A region lying exactly on pixel
    edges can pick up an extra row/column from floating-point noise; a small inset (much less than
    one pixel) keeps exactly the chunk's pixels.
    """
    return [chunk["xmin"] + inset_m, chunk["ymin"] + inset_m, chunk["xmax"] - inset_m, chunk["ymax"] - inset_m]
