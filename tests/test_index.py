"""Unit tests for sar_pipeline.index (no network)."""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

from sar_pipeline import grid, index
from sar_pipeline.config import aoi_path, grid_dir


@pytest.fixture
def gd_small(make_project):
    cfg = make_project({"grid": {"chunk_px": 64}})
    return cfg, grid.build_grid(cfg)


def test_pid_rowcol_roundtrip_and_corners(gd_small):
    _, gd = gd_small
    w, h = gd["width"], gd["height"]
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    assert index.rowcol_to_pid(gd, 0, 0) == 0
    assert index.rowcol_to_pid(gd, h - 1, w - 1) == w * h - 1
    for r, c in corners:
        pid = index.rowcol_to_pid(gd, r, c)
        assert index.pid_to_rowcol(gd, pid) == (r, c)
    rng = np.random.default_rng(0)
    for pid in rng.integers(0, w * h, 200):
        assert index.rowcol_to_pid(gd, *index.pid_to_rowcol(gd, int(pid))) == pid
    for bad in [(-1, 0), (0, w), (h, 0)]:
        with pytest.raises(ValueError):
            index.rowcol_to_pid(gd, *bad)
    for bad in [-1, w * h]:
        with pytest.raises(ValueError):
            index.pid_to_rowcol(gd, bad)


def test_xy_and_lonlat_roundtrip(gd_small):
    _, gd = gd_small
    w, h = gd["width"], gd["height"]
    for r, c in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (h // 2, w // 3)]:
        x, y = index.rowcol_to_xy(gd, r, c)
        assert x == pytest.approx(gd["x0"] + (c + 0.5) * gd["res"])
        assert y == pytest.approx(gd["y0"] - (r + 0.5) * gd["res"])
        assert index.xy_to_rowcol(gd, x, y) == (r, c)
        # any point inside the pixel maps to it
        assert index.xy_to_rowcol(gd, x - 4.9, y + 4.9) == (r, c)
        pid = index.rowcol_to_pid(gd, r, c)
        lon, lat = index.pid_to_lonlat(gd, pid)
        assert -30.3 < lon < -29.6 and 39.7 < lat < 40.4
        assert index.lonlat_to_pid(gd, lon, lat) == pid
    with pytest.raises(ValueError):
        index.xy_to_rowcol(gd, gd["x0"] - 1, gd["y0"] - 1)
    with pytest.raises(ValueError):
        index.xy_to_rowcol(gd, gd["x0"] + 1, gd["y0"] + 1)


def test_chunk_of_pid_matches_brute_force(gd_small):
    _, gd = gd_small
    rng = np.random.default_rng(1)
    found_none = found_chunk = False
    for pid in rng.integers(0, gd["width"] * gd["height"], 2000):
        r, c = index.pid_to_rowcol(gd, int(pid))
        expected = None
        for ch in gd["chunks"]:
            if ch["row_off"] <= r < ch["row_off"] + ch["height"] and ch["col_off"] <= c < ch["col_off"] + ch["width"]:
                expected = ch
                break
        got = index.chunk_of_pid(gd, int(pid))
        assert got is expected
        found_none |= got is None
        found_chunk |= got is not None
    assert found_chunk


def test_pixel_index_raster(make_project):
    # small chunks so that some chunk positions fall outside the AOI (sparse area)
    cfg = make_project({"grid": {"chunk_px": 16}})
    gd = grid.build_grid(cfg)
    path = index.build_pixel_index(cfg, gd)
    assert index.build_pixel_index(cfg, gd) == path  # no overwrite by default

    aoi = gpd.read_file(aoi_path(cfg)).to_crs(gd["crs"]).union_all()
    with rasterio.open(path) as src:
        assert (src.width, src.height, src.count) == (gd["width"], gd["height"], 3)
        assert src.dtypes == ("int32", "int32", "int32")
        assert src.nodata == -1
        assert src.crs.to_epsg() == 6933
        assert tuple(src.transform)[:6] == pytest.approx(tuple(index.affine(gd))[:6])
        assert src.descriptions == ("pid", "chunk_id", "aoi_mask")

        rng = np.random.default_rng(2)
        n_inside = n_outside = 0
        for pid in rng.integers(0, gd["width"] * gd["height"], 1500):
            r, c = index.pid_to_rowcol(gd, int(pid))
            vals = src.read(window=Window(c, r, 1, 1))[:, 0, 0]
            ch = index.chunk_of_pid(gd, int(pid))
            if ch is None:
                assert list(vals) == [-1, -1, -1]
                continue
            assert vals[0] == pid and vals[1] == ch["chunk_id"]
            pt = Point(*index.rowcol_to_xy(gd, r, c))
            if aoi.boundary.distance(pt) <= gd["res"]:
                continue  # rasterisation at the boundary is ambiguous by design
            inside = aoi.contains(pt)
            assert vals[2] == int(inside)
            n_inside += inside
            n_outside += not inside
        assert n_inside > 0 and n_outside > 0

        # at least one chunk position is not listed -> fully nodata (sparse)
        listed = {(ch["grid_row"], ch["grid_col"]) for ch in gd["chunks"]}
        missing = [(r, c) for r in range(gd["n_chunk_rows"]) for c in range(gd["n_chunk_cols"]) if (r, c) not in listed]
        assert missing
        r, c = missing[0]
        cp = gd["chunk_px"]
        block = src.read(window=Window(c * cp, r * cp, min(cp, gd["width"] - c * cp), min(cp, gd["height"] - r * cp)))
        assert (block == -1).all()


def test_pixel_index_int64_path(make_project):
    """Full-AOI grids use int64 pids; make sure writing/reading int64 (deflate + predictor 2) works."""
    cfg = make_project({"grid": {"chunk_px": 512}})
    gd = grid.build_grid(cfg)
    gd["pid_dtype"] = "int64"
    path = index.build_pixel_index(cfg, gd)
    with rasterio.open(path) as src:
        assert src.dtypes == ("int64", "int64", "int64")
        r, c = gd["height"] // 2, gd["width"] // 2
        assert src.read(1, window=Window(c, r, 1, 1))[0, 0] == index.rowcol_to_pid(gd, r, c)


def _read_all(path):
    with rasterio.open(path) as src:
        return src.read()


def test_pixel_index_overwrite_and_force(make_project):
    cfg = make_project({"grid": {"chunk_px": 128}})
    gd = grid.build_grid(cfg)
    path = index.build_pixel_index(cfg, gd)
    first = _read_all(path)
    assert (np.array_equal(_read_all(index.build_pixel_index(cfg, gd, overwrite=True)), first))
    assert (np.array_equal(_read_all(index.build_pixel_index(cfg, gd, force=True)), first))
    assert not (grid_dir(cfg) / "pixel_index.tmp.tif").exists()
    assert not (grid_dir(cfg) / "pixel_index.progress.json").exists()


def test_pixel_index_resumes_after_crash(make_project, monkeypatch):
    cfg = make_project({"grid": {"chunk_px": 64}})
    gd = grid.build_grid(cfg)
    monkeypatch.setattr(index, "TILE", 64)          # many small tiles -> many units of work
    monkeypatch.setattr(index, "SESSION_TILES", 8)  # checkpoint every 8 tiles
    monkeypatch.setattr(index.resources, "cpu_workers", lambda *a, **k: 2)  # batch = 4, deterministic
    n_tiles = len(index._touched_tiles(gd))
    assert n_tiles > 20

    real = index._tile_array
    calls = {"n": 0, "crash_at": 19}
    lock = threading.Lock()

    def counting(*args, **kwargs):
        with lock:
            calls["n"] += 1
            if calls["crash_at"] and calls["n"] == calls["crash_at"]:
                raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(index, "_tile_array", counting)
    with pytest.raises(RuntimeError, match="simulated crash"):
        index.build_pixel_index(cfg, gd)
    gdir = grid_dir(cfg)
    assert not (gdir / "pixel_index.tif").exists()
    assert (gdir / "pixel_index.tmp.tif").exists()
    done = index._read_progress(gdir / "pixel_index.progress.json")
    assert 0 < len(done) < n_tiles

    calls.update(n=0, crash_at=0)
    path = index.build_pixel_index(cfg, gd)
    assert calls["n"] == n_tiles - len(done)  # only the remaining tiles were computed
    assert not (gdir / "pixel_index.progress.json").exists()
    resumed = _read_all(path)

    calls.update(n=0)
    clean = _read_all(index.build_pixel_index(cfg, gd, force=True))
    assert calls["n"] == n_tiles
    assert np.array_equal(resumed, clean)


def test_pixel_index_corrupt_final_is_rebuilt(make_project):
    cfg = make_project({"grid": {"chunk_px": 128}})
    gd = grid.build_grid(cfg)
    path = index.build_pixel_index(cfg, gd)
    good = _read_all(path)
    Path(path).write_bytes(b"not a tiff at all")
    assert np.array_equal(_read_all(index.build_pixel_index(cfg, gd)), good)

    # valid GeoTIFF but wrong content (pid band zeroed) -> verification fails -> rebuilt
    with rasterio.open(path, "r+") as dst:
        dst.write(np.zeros((gd["height"], gd["width"]), dtype="int32"), 1)
    assert np.array_equal(_read_all(index.build_pixel_index(cfg, gd)), good)


def test_stale_tmp_with_other_profile_is_discarded(make_project):
    cfg = make_project({"grid": {"chunk_px": 128}})
    gd = grid.build_grid(cfg)
    tmp = grid_dir(cfg) / "pixel_index.tmp.tif"
    with rasterio.open(tmp, "w", driver="GTiff", width=10, height=10, count=1, dtype="uint8") as dst:
        dst.write(np.ones((1, 10, 10), dtype="uint8"))
    (grid_dir(cfg) / "pixel_index.progress.json").write_text('{"done_tiles": [[0, 0]]}')
    path = index.build_pixel_index(cfg, gd)
    assert index._verify_final(Path(path), gd)


def _small_tiles(monkeypatch):
    monkeypatch.setattr(index, "TILE", 64)          # many small tiles -> many units of work
    monkeypatch.setattr(index, "SESSION_TILES", 8)  # checkpoint every 8 tiles
    monkeypatch.setattr(index.resources, "cpu_workers", lambda *a, **k: 2)  # batch = 4, deterministic


def test_resume_rewrites_corrupted_done_tile(make_project, monkeypatch):
    cfg = make_project({"grid": {"chunk_px": 64}})
    gd = grid.build_grid(cfg)
    _small_tiles(monkeypatch)
    n_tiles = len(index._touched_tiles(gd))
    real = index._tile_array
    calls = {"n": 0, "crash_at": 19}
    lock = threading.Lock()

    def counting(*args, **kwargs):
        with lock:
            calls["n"] += 1
            if calls["crash_at"] and calls["n"] == calls["crash_at"]:
                raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(index, "_tile_array", counting)
    with pytest.raises(RuntimeError):
        index.build_pixel_index(cfg, gd)
    gdir = grid_dir(cfg)
    tmp = gdir / "pixel_index.tmp.tif"
    done = index._read_progress(gdir / "pixel_index.progress.json")
    assert len(done) >= 2

    # Damage one tile that the progress file claims is finished.
    victim = sorted(done)[len(done) // 2]
    win = index._tile_window(gd, victim)
    with rasterio.open(tmp, "r+") as dst:
        dst.write(np.full((3, int(win.height), int(win.width)), 7, dtype=gd["pid_dtype"]), window=win)
    assert index._verify_done_tiles(tmp, gd, done) == done - {victim}

    calls.update(n=0, crash_at=0)
    path = index.build_pixel_index(cfg, gd)
    assert calls["n"] == n_tiles - len(done) + 1  # remaining tiles + the damaged one
    resumed = _read_all(path)
    clean = _read_all(index.build_pixel_index(cfg, gd, force=True))
    assert np.array_equal(resumed, clean)


_KILL_SCRIPT = """
import os, sys, threading
from sar_pipeline import grid, index
from sar_pipeline.config import load_config

cfg = load_config(sys.argv[1])
gd = grid.load_grid(cfg)
index.TILE = 64
index.SESSION_TILES = 8
index.resources.cpu_workers = lambda *a, **k: 2
real, lock, n, crash_at = index._tile_array, threading.Lock(), [0], int(sys.argv[2])

def killer(*args, **kwargs):
    with lock:
        n[0] += 1
        if n[0] == crash_at:
            os._exit(3)  # hard kill: no cleanup, dataset left open in update mode
    return real(*args, **kwargs)

index._tile_array = killer
index.build_pixel_index(cfg, gd)
"""


def test_hard_kill_mid_write_resumes_to_identical_raster(make_project, monkeypatch, tmp_path):
    cfg = make_project({"grid": {"chunk_px": 64}})
    gd = grid.build_grid(cfg)
    script = tmp_path / "kill_writer.py"
    script.write_text(_KILL_SCRIPT)
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run([sys.executable, str(script), cfg["_config_path"], "19"], env=env,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 3, proc.stderr

    _small_tiles(monkeypatch)
    n_tiles = len(index._touched_tiles(gd))
    gdir = grid_dir(cfg)
    assert not (gdir / "pixel_index.tif").exists()
    assert (gdir / "pixel_index.tmp.tif").exists()
    assert len(index._read_progress(gdir / "pixel_index.progress.json")) < n_tiles

    path = index.build_pixel_index(cfg, gd)
    assert not (gdir / "pixel_index.progress.json").exists()
    resumed = _read_all(path)
    clean = _read_all(index.build_pixel_index(cfg, gd, force=True))
    assert np.array_equal(resumed, clean)


# ---------------------------------------------------------------- AOI changes (same grid geometry)
def _write_aoi_with_hole(cfg, radius_deg: float = 0.002):
    """Cut a small hole (~200 m) into the AOI: the outline, and so the chunks it touches, stay the same."""
    aoi = gpd.read_file(aoi_path(cfg))
    poly = aoi.geometry.iloc[0]
    holed = poly.difference(poly.centroid.buffer(radius_deg))
    gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[holed], crs=aoi.crs).to_file(aoi_path(cfg), driver="GPKG")


def _tag(path):
    with rasterio.open(path) as src:
        return src.tags().get(index.AOI_TAG)


def test_index_records_aoi_fingerprint(make_project):
    cfg = make_project({"grid": {"chunk_px": 128}})
    gd = grid.build_grid(cfg)
    path = index.build_pixel_index(cfg, gd)
    assert _tag(path) == index.aoi_fingerprint(cfg, gd) and len(_tag(path)) == 64


def test_aoi_geometry_change_rebuilds_mask_only(make_project, caplog):
    cfg = make_project({"grid": {"chunk_px": 64}})
    gd = grid.build_grid(cfg)
    path = Path(index.build_pixel_index(cfg, gd))
    before = _read_all(path)
    fp_before, mtime_before = _tag(path), path.stat().st_mtime_ns

    _write_aoi_with_hole(cfg)
    assert grid.geometry_diff(gd, grid.compute_grid_def(cfg)) is None  # the grid itself is unchanged
    with caplog.at_level(logging.WARNING, logger="sar_pipeline.index"):
        assert index.build_pixel_index(cfg, gd) == str(path)
    after = _read_all(path)

    assert "AOI geometry changed" in caplog.text
    assert np.array_equal(after[0], before[0]) and np.array_equal(after[1], before[1])  # pid, chunk_id untouched
    assert ((before[2] == 1) & (after[2] == 0)).sum() > 0   # pixels inside the hole left the AOI
    assert not ((before[2] == 0) & (after[2] == 1)).any()   # nothing joined the AOI
    assert np.array_equal(after[2] == -1, before[2] == -1)  # nodata layout unchanged
    assert _tag(path) == index.aoi_fingerprint(cfg, gd) != fp_before
    assert path.stat().st_mtime_ns != mtime_before           # stack AOI caches key on size/mtime
    assert not (grid_dir(cfg) / "pixel_index.tmp.tif").exists()
    assert not (grid_dir(cfg) / "pixel_index.progress.json").exists()

    clean = _read_all(index.build_pixel_index(cfg, gd, force=True))
    assert np.array_equal(after, clean)


def test_moved_aoi_file_with_same_geometry_does_not_rebuild(make_project, monkeypatch):
    cfg = make_project({"grid": {"chunk_px": 128}})
    gd = grid.build_grid(cfg)
    path = Path(index.build_pixel_index(cfg, gd))
    mtime = path.stat().st_mtime_ns

    moved = Path(cfg["_root"]) / "data" / "moved" / "renamed_aoi.gpkg"
    moved.parent.mkdir(parents=True)
    gpd.read_file(aoi_path(cfg)).to_file(moved, driver="GPKG")
    cfg["aoi"]["path"] = "data/moved/renamed_aoi.gpkg"

    def must_not_rewrite(*args, **kwargs):
        raise AssertionError("tiles must not be rewritten when only the AOI file path changed")

    monkeypatch.setattr(index, "_tile_array", must_not_rewrite)
    assert index.build_pixel_index(cfg, gd) == str(path)
    assert path.stat().st_mtime_ns == mtime


def test_index_without_fingerprint_is_rewritten(make_project, caplog):
    """Indexes built before fingerprints existed carry no tag: rewrite them once, with identical ids."""
    cfg = make_project({"grid": {"chunk_px": 128}})
    gd = grid.build_grid(cfg)
    path = index.build_pixel_index(cfg, gd)
    before = _read_all(path)
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(**{index.AOI_TAG: ""})
    with caplog.at_level(logging.WARNING, logger="sar_pipeline.index"):
        index.build_pixel_index(cfg, gd)
    assert "AOI geometry changed" in caplog.text
    assert _tag(path) == index.aoi_fingerprint(cfg, gd)
    assert np.array_equal(_read_all(path), before)


def test_partial_build_for_another_aoi_is_discarded(make_project, monkeypatch):
    cfg = make_project({"grid": {"chunk_px": 64}})
    gd = grid.build_grid(cfg)
    _small_tiles(monkeypatch)
    n_tiles = len(index._touched_tiles(gd))
    real = index._tile_array
    calls = {"n": 0, "crash_at": 19}
    lock = threading.Lock()

    def counting(*args, **kwargs):
        with lock:
            calls["n"] += 1
            if calls["crash_at"] and calls["n"] == calls["crash_at"]:
                raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(index, "_tile_array", counting)
    with pytest.raises(RuntimeError):
        index.build_pixel_index(cfg, gd)
    assert index._read_progress(grid_dir(cfg) / "pixel_index.progress.json")

    _write_aoi_with_hole(cfg)  # AOI changes while a build is half done
    calls.update(n=0, crash_at=0)
    path = index.build_pixel_index(cfg, gd)
    assert calls["n"] == n_tiles  # nothing reused from the build made for the old AOI
    assert _tag(path) == index.aoi_fingerprint(cfg, gd)
    resumed = _read_all(path)
    assert np.array_equal(resumed, _read_all(index.build_pixel_index(cfg, gd, force=True)))
