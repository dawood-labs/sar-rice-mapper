"""Stage 0/5 — Pixel index raster and pixel-id conversions.

Pixel id (pid) = row * width + col on the master grid.
- From a pid you get row/col with divmod, and from row/col the map coordinate.
- So reading one pixel's time series never needs the whole raster: pid -> row/col -> 1x1 window.

pixel_index.tif bands (one dtype for all bands, as GeoTIFF requires):
  1 = pid, 2 = chunk_id, 3 = aoi_mask (1 inside AOI, 0 outside)
Only pixels of chunks that intersect the AOI get values. The file is sparse: tiles without any such
chunk are never written, read back as nodata (-1) and take no disk space.

The file records which AOI geometry its mask was made from (GeoTIFF tag ``AOI_FINGERPRINT``). If the
AOI polygon changes while the grid geometry stays the same (for example a hole is cut out of the AOI),
the stored grid is still valid but the mask is not, so the next build rewrites the index.
"""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import rasterio.crs
import shapely
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from . import resources, run_meta
from .config import grid_dir
from .grid import chunk_at, read_aoi

NODATA = -1
TILE = 512  # GeoTIFF tile size of pixel_index.tif
SESSION_TILES = 256  # tiles written between flush + progress checkpoints
FINAL_SAMPLES_MAX = 200  # final verification samples min(FINAL_SAMPLES_MAX, 2 * n_tiles) positions
AOI_TAG = "AOI_FINGERPRINT"
log = logging.getLogger(__name__)
_TRANSFORMERS: dict[tuple[str, str], Transformer] = {}


def affine(gd: dict) -> Affine:
    a, b, c, d, e, f = gd["transform"]
    return Affine(a, b, c, d, e, f)


def _transformer(src: str, dst: str) -> Transformer:
    key = (src, dst)
    if key not in _TRANSFORMERS:
        _TRANSFORMERS[key] = Transformer.from_crs(src, dst, always_xy=True)
    return _TRANSFORMERS[key]


# ---------------------------------------------------------------- conversions
def rowcol_to_pid(gd: dict, row: int, col: int) -> int:
    if not (0 <= row < gd["height"] and 0 <= col < gd["width"]):
        raise ValueError(f"row/col outside the grid: {row},{col}")
    return int(row) * int(gd["width"]) + int(col)


def pid_to_rowcol(gd: dict, pid: int) -> tuple[int, int]:
    if not (0 <= pid < gd["width"] * gd["height"]):
        raise ValueError(f"pid outside the grid: {pid}")
    row, col = divmod(int(pid), int(gd["width"]))
    return row, col


def rowcol_to_xy(gd: dict, row: int, col: int) -> tuple[float, float]:
    """Map coordinate (grid CRS) of the pixel CENTRE."""
    return gd["x0"] + (col + 0.5) * gd["res"], gd["y0"] - (row + 0.5) * gd["res"]


def xy_to_rowcol(gd: dict, x: float, y: float) -> tuple[int, int]:
    """Pixel containing (x, y). Raises ValueError outside the grid."""
    row = int(np.floor((gd["y0"] - y) / gd["res"]))
    col = int(np.floor((x - gd["x0"]) / gd["res"]))
    if not (0 <= row < gd["height"] and 0 <= col < gd["width"]):
        raise ValueError(f"point outside the grid: x={x}, y={y}")
    return row, col


def pid_to_lonlat(gd: dict, pid: int) -> tuple[float, float]:
    x, y = rowcol_to_xy(gd, *pid_to_rowcol(gd, pid))
    return _transformer(gd["crs"], "EPSG:4326").transform(x, y)


def lonlat_to_pid(gd: dict, lon: float, lat: float) -> int:
    x, y = _transformer("EPSG:4326", gd["crs"]).transform(lon, lat)
    return rowcol_to_pid(gd, *xy_to_rowcol(gd, x, y))


def chunk_of_pid(gd: dict, pid: int) -> dict | None:
    """Chunk containing a pid in O(1): chunk position = row // chunk_px, col // chunk_px."""
    row, col = pid_to_rowcol(gd, pid)
    return chunk_at(gd, row // gd["chunk_px"], col // gd["chunk_px"])


# ---------------------------------------------------------------- AOI fingerprint
def aoi_fingerprint(cfg: dict, gd: dict) -> str:
    """sha256 of the AOI geometry as used for the mask: union of all parts in the grid CRS, normalised.

    Based on the geometry, not on the file: moving or renaming the AOI file, or storing its features in
    a different order, gives the same fingerprint. Coordinates are rounded to res/100 (10 cm on a 10 m
    grid), far below what changes the rasterised mask, so tiny reprojection differences between
    machines do not trigger a rebuild.
    """
    geoms = read_aoi(cfg, gd["crs"]).geometry.values
    union = shapely.union_all(geoms) if len(geoms) else shapely.GeometryCollection()
    norm = shapely.normalize(shapely.set_precision(union, float(gd["res"]) / 100.0))
    return hashlib.sha256(shapely.to_wkb(norm, output_dimension=2)).hexdigest()


def _file_fingerprint(path: Path) -> str | None:
    try:
        with rasterio.open(path) as src:
            return src.tags().get(AOI_TAG)
    except Exception:
        return None


# ---------------------------------------------------------------- raster
def _touched_tiles(gd: dict) -> list[tuple[int, int]]:
    """(tile_row, tile_col) of every TILE x TILE block that overlaps at least one AOI chunk."""
    tiles = set()
    for ch in gd["chunks"]:
        for tr_ in range(ch["row_off"] // TILE, (ch["row_off"] + ch["height"] - 1) // TILE + 1):
            for tc in range(ch["col_off"] // TILE, (ch["col_off"] + ch["width"] - 1) // TILE + 1):
                tiles.add((tr_, tc))
    return sorted(tiles)


def _tile_window(gd: dict, tile: tuple[int, int]) -> Window:
    ro, co = tile[0] * TILE, tile[1] * TILE
    return Window(co, ro, min(TILE, gd["width"] - co), min(TILE, gd["height"] - ro))


def _chunk_overlaps(gd: dict, win: Window):
    """(chunk, r0, r1, c0, c1) in grid pixel coordinates for every chunk overlapping a window."""
    ro, co, h, w = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
    cp = gd["chunk_px"]
    for gr in range(ro // cp, (ro + h - 1) // cp + 1):
        for gc in range(co // cp, (co + w - 1) // cp + 1):
            ch = chunk_at(gd, gr, gc)
            if ch is None:
                continue
            r0, r1 = max(ro, ch["row_off"]), min(ro + h, ch["row_off"] + ch["height"])
            c0, c1 = max(co, ch["col_off"]), min(co + w, ch["col_off"] + ch["width"])
            if r0 < r1 and c0 < c1:
                yield ch, r0, r1, c0, c1


def _expected_ids(gd: dict, tile: tuple[int, int]) -> tuple[Window, np.ndarray, np.ndarray, np.ndarray]:
    """(window, pid band, chunk_id band, covered mask) that one tile must contain.

    Pure arithmetic on the grid definition: used both to build a tile and to verify a tile read back
    from disk, so the check never trusts the file it is checking.
    """
    dtype = gd["pid_dtype"]
    win = _tile_window(gd, tile)
    ro, co, h, w = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
    rows = np.arange(ro, ro + h, dtype=np.int64)[:, None]
    cols = np.arange(co, co + w, dtype=np.int64)[None, :]
    chunk_id = np.full((h, w), NODATA, dtype=dtype)
    covered = np.zeros((h, w), dtype=bool)
    for ch, r0, r1, c0, c1 in _chunk_overlaps(gd, win):
        chunk_id[r0 - ro:r1 - ro, c0 - co:c1 - co] = ch["chunk_id"]
        covered[r0 - ro:r1 - ro, c0 - co:c1 - co] = True
    pid = np.where(covered, rows * int(gd["width"]) + cols, NODATA).astype(dtype)
    return win, pid, chunk_id, covered


def _tile_array(gd: dict, tile: tuple[int, int], aoi_geoms: np.ndarray, tree: shapely.STRtree, tr: Affine):
    """All 3 bands of one tile, fully assembled in memory so the tile is written in one call.

    Writing a compressed GeoTIFF tile piecewise (e.g. one band or one chunk at a time) makes GDAL
    re-read partially flushed tiles and can fill other bands with 0 instead of nodata.
    """
    dtype = gd["pid_dtype"]
    win, pid, chunk_id, covered = _expected_ids(gd, tile)
    h, w = int(win.height), int(win.width)
    arr = np.empty((3, h, w), dtype=dtype)
    arr[0], arr[1] = pid, chunk_id

    xmin, ymax = tr * (int(win.col_off), int(win.row_off))
    hits = tree.query(shapely.box(xmin, ymax - h * gd["res"], xmin + w * gd["res"], ymax), predicate="intersects")
    if len(hits):
        mask = rasterize(
            [(geom, 1) for geom in aoi_geoms[hits]], out_shape=(h, w),
            transform=window_transform(win, tr), fill=0, dtype="uint8",
        )
        arr[2] = np.where(covered, mask.astype(dtype), NODATA)
    else:
        arr[2] = np.where(covered, 0, NODATA)
    return win, arr


def _tile_matches(src, gd: dict, tile: tuple[int, int]) -> bool:
    """Read one tile back and compare it with what it must contain.

    pid and chunk_id must match exactly; the AOI mask band must be -1 outside chunks and 0/1 inside
    (its exact values depend on polygon rasterisation, which is not recomputed for a check; which AOI
    it was made from is guarded by the AOI fingerprint instead).
    """
    win, pid, chunk_id, covered = _expected_ids(gd, tile)
    try:
        got = src.read(window=win)
    except Exception:  # undecodable block after a hard kill
        return False
    if not (np.array_equal(got[0], pid) and np.array_equal(got[1], chunk_id)):
        return False
    mask = got[2]
    return bool(np.all(mask[~covered] == NODATA) and np.all((mask[covered] == 0) | (mask[covered] == 1)))


def _verify_done_tiles(path: Path, gd: dict, claimed: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Subset of the tiles recorded as done whose data really reads back correctly.

    The progress file is written only after a session is flushed, but a hard kill (power loss, OOM
    kill, `kill -9`) can still leave recorded tiles unreadable or stale. Every recorded tile is
    therefore re-read and compared before it is skipped.
    """
    good: set[tuple[int, int]] = set()
    try:
        with rasterio.open(path) as src:
            for tile in sorted(claimed):
                if _tile_matches(src, gd, tile):
                    good.add(tile)
    except Exception:
        return set()
    return good


def _profile(gd: dict, tr: Affine) -> dict:
    return dict(
        driver="GTiff", width=gd["width"], height=gd["height"], count=3, dtype=gd["pid_dtype"],
        crs=gd["crs"], transform=tr, nodata=NODATA, tiled=True, blockxsize=TILE, blockysize=TILE,
        compress="deflate", predictor=2, BIGTIFF="IF_SAFER", SPARSE_OK="TRUE",
    )


def _profile_matches(path: Path, gd: dict) -> bool:
    """True when an existing file has the grid's shape, dtype, CRS, transform, band count and tiling."""
    try:
        with rasterio.open(path) as src:
            return (
                (src.width, src.height, src.count) == (gd["width"], gd["height"], 3)
                and all(dt == gd["pid_dtype"] for dt in src.dtypes)
                and src.crs is not None and src.crs == rasterio.crs.CRS.from_user_input(gd["crs"])
                and src.transform.almost_equals(affine(gd), precision=1e-6)
                and src.block_shapes[0] == (TILE, TILE)
                and src.nodata == NODATA
            )
    except Exception:  # unreadable / corrupt file
        return False


def _expected_pixel(gd: dict, row: int, col: int) -> tuple[int, int]:
    ch = chunk_at(gd, row // gd["chunk_px"], col // gd["chunk_px"])
    if ch is None:
        return NODATA, NODATA
    return rowcol_to_pid(gd, row, col), int(ch["chunk_id"])


def _verify_final(path: Path, gd: dict, n_samples: int | None = None) -> bool:
    """Integrity check of a finished index without reading the whole file.

    Checks min(FINAL_SAMPLES_MAX, 2 * n_tiles) positions spread evenly over all written tiles. At each
    position one random pixel of the tile and the centre of the first chunk overlapping the tile are
    compared with the pid/chunk_id they must hold; the AOI mask must be -1 outside chunks, 0/1 inside.
    """
    if not _profile_matches(path, gd):
        return False
    tiles = _touched_tiles(gd)
    if not tiles:
        return True
    n = n_samples if n_samples is not None else min(FINAL_SAMPLES_MAX, 2 * len(tiles))
    positions = np.linspace(0, len(tiles) - 1, num=max(1, n)).round().astype(int)
    rng = np.random.default_rng(0)
    try:
        with rasterio.open(path) as src:
            for k in positions:
                win = _tile_window(gd, tiles[int(k)])
                ro, co, h, w = int(win.row_off), int(win.col_off), int(win.height), int(win.width)
                pixels = [(ro + int(rng.integers(h)), co + int(rng.integers(w)))]
                first = next(_chunk_overlaps(gd, win), None)
                if first is not None:
                    _, r0, r1, c0, c1 = first
                    pixels.append(((r0 + r1) // 2, (c0 + c1) // 2))
                for r, c in pixels:
                    vals = src.read(window=Window(c, r, 1, 1))[:, 0, 0]
                    exp_pid, exp_cid = _expected_pixel(gd, r, c)
                    if int(vals[0]) != exp_pid or int(vals[1]) != exp_cid:
                        return False
                    if exp_pid == NODATA and int(vals[2]) != NODATA:
                        return False
                    if exp_pid != NODATA and int(vals[2]) not in (0, 1):
                        return False
    except Exception:
        return False
    return True


def _read_progress_meta(path: Path) -> dict | None:
    try:
        meta = json.loads(path.read_text())
        meta["done_tiles"] = {tuple(t) for t in meta["done_tiles"]}
        return meta
    except Exception:
        return None


def _read_progress(path: Path) -> set[tuple[int, int]]:
    meta = _read_progress_meta(path)
    return meta["done_tiles"] if meta else set()


def _write_progress(path: Path, done: set, n_total: int, fingerprint: str) -> None:
    run_meta.atomic_write_text(path, json.dumps(
        {"n_total_tiles": n_total, "aoi_fingerprint": fingerprint, "done_tiles": sorted(done)}))


def _remove(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            p.unlink()


def build_pixel_index(cfg: dict, gd: dict, overwrite: bool = False, force: bool = False) -> str:
    """Write grid/pixel_index.tif tile by tile (never the full grid in memory). Resumable.

    Resume, never redo: work goes into ``pixel_index.tmp.tif``; after every session of tiles the file is
    closed (flushed) and the finished tiles are recorded in ``pixel_index.progress.json`` (atomic write).
    After a crash, re-running re-reads every recorded tile and skips only those whose data is intact;
    damaged ones are written again. The finished file is verified before it is renamed to
    ``pixel_index.tif``, then the progress file is deleted.

    - existing final file: verified (profile + sampled pixels over all tiles) and its AOI fingerprint
      compared with the current AOI geometry; returned if both match;
    - AOI geometry changed (grid geometry identical): the index is rewritten into the temporary file with
      the new mask; pid and chunk_id come out identical because they are pure grid arithmetic. The old
      file stays in place until the new one is verified, and the rename gives it a new mtime, so caches
      keyed on the index file's size/mtime (stack AOI counts) refresh automatically;
    - ``overwrite=True``: rebuild the final file even if valid (a matching temporary file is still resumed);
    - ``force=True``: discard final, temporary and progress files and start from scratch;
    - a temporary file with a different profile, a different AOI fingerprint or without a progress file
      is discarded.

    Unit of work = one GeoTIFF tile (TILE x TILE px). With the default chunk_px == TILE a tile is exactly
    one chunk. Tile arrays are computed in a thread pool sized from the machine's resources; writes
    happen in the main thread (a GDAL dataset must not be written from several threads).
    """
    out = grid_dir(cfg) / "pixel_index.tif"
    tmp = out.with_name("pixel_index.tmp.tif")
    progress = out.with_name("pixel_index.progress.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = aoi_fingerprint(cfg, gd)

    if force:
        _remove(out, tmp, progress)
    elif out.exists() and not overwrite:
        if _verify_final(out, gd):
            recorded = _file_fingerprint(out)
            if recorded == fingerprint:
                return str(out)
            log.warning(
                "AOI geometry changed since %s was built (recorded AOI fingerprint %s, current %s): rewriting "
                "the index with the new aoi_mask; pid and chunk_id stay identical and the current file is kept "
                "until the new one is verified", out, (recorded or "none")[:12], fingerprint[:12])
        else:
            log.warning("%s failed verification; rebuilding it", out)

    tr = affine(gd)
    tiles = _touched_tiles(gd)
    done: set[tuple[int, int]] = set()
    if tmp.exists():
        meta = _read_progress_meta(progress) if progress.exists() else None
        if (meta is not None and meta.get("aoi_fingerprint") == fingerprint and _profile_matches(tmp, gd)
                and _file_fingerprint(tmp) == fingerprint):
            claimed = meta["done_tiles"] & set(tiles)
            done = _verify_done_tiles(tmp, gd, claimed)
            if len(done) < len(claimed):
                log.warning("pixel index: %d of %d recorded tiles failed read-back verification; rewriting them",
                            len(claimed) - len(done), len(claimed))
                _write_progress(progress, done, len(tiles), fingerprint)
        else:
            log.warning("discarding stale %s (profile mismatch, no progress file or made for a different AOI)", tmp)
            _remove(tmp, progress)
    else:
        _remove(progress)
    if not tmp.exists():
        with rasterio.open(tmp, "w", **_profile(gd, tr)) as dst:
            dst.set_band_description(1, "pid")
            dst.set_band_description(2, "chunk_id")
            dst.set_band_description(3, "aoi_mask")
            dst.update_tags(**{AOI_TAG: fingerprint})
        _write_progress(progress, done, len(tiles), fingerprint)

    todo = [t for t in tiles if t not in done]
    log.info("pixel index: skipped %d/%d tiles already written and verified", len(done), len(tiles))

    aoi = read_aoi(cfg, gd["crs"])
    aoi_geoms = aoi.geometry.values
    tree = shapely.STRtree(aoi_geoms)
    res = resources.detect_resources(cfg)
    itemsize = np.dtype(gd["pid_dtype"]).itemsize
    per_tile = TILE * TILE * (3 * itemsize * 2 + 8 * 2 + 2)  # bands + temporaries, rows*cols, mask/covered
    workers = resources.cpu_workers(res, cfg, mem_per_worker_bytes=per_tile)
    batch = max(1, workers * 2)                              # at most `batch` tile arrays alive at once
    session = -(-max(batch, SESSION_TILES) // batch) * batch  # tiles per open/flush/progress cycle
    cache_mb = max(64, resources.gdal_cache_bytes(res) // 1024**2)

    with rasterio.Env(GDAL_CACHEMAX=cache_mb), ThreadPoolExecutor(max_workers=workers) as pool:
        for s in range(0, len(todo), session):
            part = todo[s:s + session]
            with rasterio.open(tmp, "r+") as dst:
                for i in range(0, len(part), batch):
                    futures = [pool.submit(_tile_array, gd, t, aoi_geoms, tree, tr) for t in part[i:i + batch]]
                    for fut in futures:
                        win, arr = fut.result()
                        dst.write(arr, window=win)
            done.update(part)
            _write_progress(progress, done, len(tiles), fingerprint)

    if not _verify_final(tmp, gd) or _file_fingerprint(tmp) != fingerprint:
        raise RuntimeError(f"{tmp} failed verification after writing; re-run the stage to rewrite damaged tiles")
    run_meta.replace_with_retry(tmp, out)
    _remove(progress)
    return str(out)
