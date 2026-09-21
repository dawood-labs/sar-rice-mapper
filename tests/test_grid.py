"""Unit tests for sar_pipeline.grid (no network)."""
from __future__ import annotations

import gc
import json
import logging
import tracemalloc

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from sar_pipeline import grid
from sar_pipeline.config import aoi_path, grid_dir
from sar_pipeline.errors import GridMismatch


def _windows_intersecting_aoi(cfg, gd):
    """Brute force: every chunk-grid position whose box overlaps the AOI with positive area."""
    aoi = gpd.read_file(aoi_path(cfg)).to_crs(gd["crs"]).union_all()
    res, cp = gd["res"], gd["chunk_px"]
    out = set()
    for r in range(gd["n_chunk_rows"]):
        for c in range(gd["n_chunk_cols"]):
            ro, co = r * cp, c * cp
            h, w = min(cp, gd["height"] - ro), min(cp, gd["width"] - co)
            x0 = gd["x0"] + co * res
            y1 = gd["y0"] - ro * res
            if box(x0, y1 - h * res, x0 + w * res, y1).intersection(aoi).area > 0:
                out.add((r, c))
    return out


def test_snapping_and_extent(make_project):
    cfg = make_project()
    gd = grid.compute_grid_def(cfg)
    res = gd["res"]
    assert gd["x0"] % res == 0 and gd["y0"] % res == 0
    assert gd["transform"] == [res, 0.0, gd["x0"], 0.0, -res, gd["y0"]]
    minx, miny, maxx, maxy = gpd.read_file(aoi_path(cfg)).to_crs(gd["crs"]).total_bounds
    buf = cfg["grid"]["buffer_m"]
    assert gd["x0"] <= minx - buf and gd["y0"] >= maxy + buf
    assert gd["x0"] + gd["width"] * res >= maxx + buf
    assert gd["y0"] - gd["height"] * res <= miny - buf
    # snapped extent is no larger than one pixel beyond the buffered bounds on each side
    assert gd["x0"] > minx - buf - res and gd["y0"] < maxy + buf + res
    assert gd["pid_dtype"] == "int32"
    assert gd["aoi_file"] == cfg["aoi"]["path"]


@pytest.mark.parametrize("chunk_px", [64, 256, 100])
def test_chunks_tile_grid_exactly(make_project, chunk_px):
    cfg = make_project({"grid": {"chunk_px": chunk_px}})
    gd = grid.compute_grid_def(cfg)
    assert gd["n_chunk_rows"] == -(-gd["height"] // chunk_px)
    assert gd["n_chunk_cols"] == -(-gd["width"] // chunk_px)

    covered = np.zeros((gd["height"], gd["width"]), dtype=np.int32)
    for ch in grid.iter_chunks(gd):
        assert ch["row_off"] == ch["grid_row"] * chunk_px and ch["col_off"] == ch["grid_col"] * chunk_px
        assert ch["width"] == min(chunk_px, gd["width"] - ch["col_off"])
        assert ch["height"] == min(chunk_px, gd["height"] - ch["row_off"])
        # geographic bounds consistent with pixel window
        assert ch["xmin"] == pytest.approx(gd["x0"] + ch["col_off"] * gd["res"])
        assert ch["ymax"] == pytest.approx(gd["y0"] - ch["row_off"] * gd["res"])
        assert ch["xmax"] - ch["xmin"] == pytest.approx(ch["width"] * gd["res"])
        assert ch["ymax"] - ch["ymin"] == pytest.approx(ch["height"] * gd["res"])
        assert 0 < ch["aoi_frac"] <= 1
        covered[ch["row_off"]:ch["row_off"] + ch["height"], ch["col_off"]:ch["col_off"] + ch["width"]] += 1
    assert covered.max() == 1  # no overlaps

    listed = {(ch["grid_row"], ch["grid_col"]) for ch in gd["chunks"]}
    assert listed == _windows_intersecting_aoi(cfg, gd)
    ids = [ch["chunk_id"] for ch in gd["chunks"]]
    assert ids == list(range(1, len(ids) + 1))
    names = [ch["name"] for ch in gd["chunks"]]
    assert len(set(names)) == len(names)


def test_edge_chunks_are_partial(make_project):
    cfg = make_project({"grid": {"chunk_px": 256}})
    gd = grid.compute_grid_def(cfg)
    assert gd["width"] % 256 != 0 or gd["height"] % 256 != 0
    last_col = [ch for ch in gd["chunks"] if ch["grid_col"] == gd["n_chunk_cols"] - 1]
    last_row = [ch for ch in gd["chunks"] if ch["grid_row"] == gd["n_chunk_rows"] - 1]
    assert last_col and all(ch["width"] == gd["width"] - (gd["n_chunk_cols"] - 1) * 256 for ch in last_col)
    assert last_row and all(ch["height"] == gd["height"] - (gd["n_chunk_rows"] - 1) * 256 for ch in last_row)


def test_aoi_frac_values(make_project):
    cfg = make_project({"grid": {"chunk_px": 32}})
    gd = grid.compute_grid_def(cfg)
    aoi = gpd.read_file(aoi_path(cfg)).to_crs(gd["crs"]).union_all()
    fracs = [ch["aoi_frac"] for ch in gd["chunks"]]
    assert max(fracs) == pytest.approx(1.0)  # interior chunks fully inside
    assert min(fracs) < 1.0                  # boundary chunks partly inside
    for ch in gd["chunks"][:: max(1, len(gd["chunks"]) // 25)]:
        b = box(ch["xmin"], ch["ymin"], ch["xmax"], ch["ymax"])
        assert ch["aoi_frac"] == pytest.approx(b.intersection(aoi).area / b.area, abs=1e-6)


def test_name_digits():
    assert grid._name_digits(5, 99) == 2
    assert grid._name_digits(100, 3) == 3
    assert grid._name_digits(999, 999) == 3
    assert grid._name_digits(1000, 2) == 4
    assert grid._name_digits(12000, 2) == 5
    assert grid.chunk_name(3, 1, 2) == "chunk_r03c01"
    assert grid.chunk_name(3, 1, 4) == "chunk_r0003c0001"


def test_int64_pid_dtype_for_huge_grid(make_project):
    # ~530 x 560 km at 10 m in EPSG:6933 -> > 2**31 pixels. Compute only (no raster). Open ocean.
    big = [(-35.0, 35.0), (-29.5, 35.0), (-29.5, 40.5), (-35.0, 40.5), (-35.0, 35.0)]
    cfg = make_project({"grid": {"chunk_px": 65536}}, aoi_lonlat=big)
    gd = grid.compute_grid_def(cfg)
    assert gd["width"] * gd["height"] >= 2**31 - 1
    assert gd["pid_dtype"] == "int64"


def _pixels(ch):
    return {(r, c) for r in range(ch["row_off"], ch["row_off"] + ch["height"])
            for c in range(ch["col_off"], ch["col_off"] + ch["width"])}


@pytest.mark.parametrize("w,h", [(256, 256), (255, 129), (1, 7), (3, 1), (1, 1)])
def test_split_chunk_tiles_parent(w, h):
    res = 10.0
    parent = dict(chunk_id=4, name="chunk_r01c02", grid_row=1, grid_col=2, row_off=256, col_off=512,
                  width=w, height=h, xmin=1000.0 + 512 * res, ymax=5000.0 - 256 * res,
                  xmax=1000.0 + (512 + w) * res, ymin=5000.0 - (256 + h) * res, aoi_frac=0.5)
    parts = grid.split_chunk(parent, res)
    seen = set()
    for p in parts:
        px = _pixels(p)
        assert not (px & seen)
        seen |= px
        assert p["xmin"] == pytest.approx(1000.0 + p["col_off"] * res)
        assert p["ymax"] == pytest.approx(5000.0 - p["row_off"] * res)
        assert p["xmax"] - p["xmin"] == pytest.approx(p["width"] * res)
        assert p["ymax"] - p["ymin"] == pytest.approx(p["height"] * res)
        assert p["chunk_id"] == 4 and p["name"].startswith("chunk_r01c02_s")
    assert seen == _pixels(parent)
    assert all(p["width"] > 0 and p["height"] > 0 for p in parts)


def test_chunk_by_name_and_region(make_project):
    cfg = make_project({"grid": {"chunk_px": 64}})
    gd = grid.compute_grid_def(cfg)
    ch = gd["chunks"][len(gd["chunks"]) // 2]
    assert grid.chunk_by_name(gd, ch["name"]) is ch
    sub = grid.chunk_by_name(gd, ch["name"] + "_s3")
    assert sub == grid.split_chunk(ch, gd["res"])[3]
    for bad in ["chunk_r99c99", ch["name"] + "_s7", "nonsense", ch["name"] + "_s0_s1"]:
        with pytest.raises(KeyError):
            grid.chunk_by_name(gd, bad)
    assert grid.chunk_at(gd, ch["grid_row"], ch["grid_col"]) is ch
    xmin, ymin, xmax, ymax = grid.chunk_region_coords(ch)
    assert (xmin, ymin, xmax, ymax) == pytest.approx((ch["xmin"] + 0.25, ch["ymin"] + 0.25, ch["xmax"] - 0.25, ch["ymax"] - 0.25))
    assert grid.chunk_region_coords(ch, inset_m=0) == [ch["xmin"], ch["ymin"], ch["xmax"], ch["ymax"]]


def test_build_grid_writes_and_is_immutable(make_project):
    cfg = make_project()
    gd = grid.build_grid(cfg)
    out = grid_dir(cfg)
    assert (out / "grid_def.json").exists() and (out / "chunks.gpkg").exists()
    assert grid.load_grid(cfg) == json.loads(json.dumps(gd))
    gpkg = gpd.read_file(out / "chunks.gpkg", layer="chunks")
    assert len(gpkg) == len(gd["chunks"]) and str(gpkg.crs).endswith("6933")

    # Rebuilding with the same inputs returns the stored grid.
    assert grid.build_grid(cfg) == grid.load_grid(cfg)

    # Tiny float noise in the stored file is tolerated.
    stored = json.loads((out / "grid_def.json").read_text())
    stored["chunks"][0]["aoi_frac"] += 1e-9
    (out / "grid_def.json").write_text(json.dumps(stored))
    grid.build_grid(cfg)

    # Changing the AOI must not silently change the grid.
    moved = [(-30.000, 40.000), (-29.920, 40.002), (-29.942, 40.060), (-29.998, 40.038), (-30.000, 40.000)]
    gpd.GeoDataFrame({"name": ["aoi"]}, geometry=[Polygon(moved)], crs="EPSG:4326").to_file(aoi_path(cfg), driver="GPKG")
    with pytest.raises(GridMismatch):
        grid.build_grid(cfg)


def test_build_grid_regenerates_missing_or_corrupt_gpkg(make_project):
    cfg = make_project()
    gd = grid.build_grid(cfg)
    gpkg = grid_dir(cfg) / "chunks.gpkg"
    gpkg.unlink()
    grid.build_grid(cfg)
    assert len(gpd.read_file(gpkg, layer="chunks")) == len(gd["chunks"])
    gpkg.write_bytes(b"garbage")
    grid.build_grid(cfg)
    assert len(gpd.read_file(gpkg, layer="chunks")) == len(gd["chunks"])
    assert not list(grid_dir(cfg).glob("*.tmp*"))


def test_same_helper():
    assert grid._same({"a": 1.0, "b": [1, 2.0]}, {"a": 1.0 + 1e-9, "b": [1, 2.0]}) is None
    assert grid._same({"a": 1.0}, {"a": 1.1})
    assert grid._same({"a": 1}, {"b": 1})
    assert grid._same([1, 2], [1, 2, 3])
    assert grid._same({"crs": "EPSG:6933"}, {"crs": "EPSG:3857"})


def test_non_geometry_changes_do_not_raise(make_project, caplog):
    cfg = make_project()
    grid.build_grid(cfg)
    gd_path = grid_dir(cfg) / "grid_def.json"
    stored = json.loads(gd_path.read_text())
    stored["aoi_file"] = "somewhere/else/aoi.gpkg"       # AOI file moved
    stored["chunks"][0]["aoi_frac"] = max(0.0, stored["chunks"][0]["aoi_frac"] - 0.01)  # GEOS/PROJ drift
    gd_path.write_text(json.dumps(stored))
    before = gd_path.read_bytes()
    with caplog.at_level(logging.WARNING, logger="sar_pipeline.grid"):
        kept = grid.build_grid(cfg)
    assert kept == stored                                 # stored grid returned unchanged
    assert gd_path.read_bytes() == before                 # and never rewritten
    text = caplog.text
    assert "AOI file path changed" in text and "aoi_frac differs" in text


@pytest.mark.parametrize("mutate", [
    lambda g: g["chunks"][0].__setitem__("row_off", g["chunks"][0]["row_off"] + 1),
    lambda g: g["chunks"][-1].__setitem__("name", "chunk_r99c99"),
    lambda g: g.__setitem__("res", 20.0),
    lambda g: g.__setitem__("x0", g["x0"] + 10.0),
    lambda g: g.__setitem__("pid_dtype", "int64"),
    lambda g: g.__setitem__("chunks", g["chunks"][:-1]),
])
def test_geometry_changes_raise(make_project, mutate):
    cfg = make_project()
    grid.build_grid(cfg)
    gd_path = grid_dir(cfg) / "grid_def.json"
    stored = json.loads(gd_path.read_text())
    mutate(stored)
    gd_path.write_text(json.dumps(stored))
    with pytest.raises(GridMismatch):
        grid.build_grid(cfg)


def test_geometry_diff_helper(make_project):
    gd = grid.compute_grid_def(make_project())
    other = json.loads(json.dumps(gd))
    other["aoi_file"] = "x"
    for ch in other["chunks"]:
        ch["aoi_frac"] = 0.5
    assert grid.geometry_diff(gd, other) is None
    other["chunks"][0]["width"] -= 1
    assert "chunks[0].width" in grid.geometry_diff(gd, other)


def _synthetic_big_grid(n_side: int = 120, n_chunks: int = 14341) -> dict:
    cp, res = 512, 10.0
    chunks = []
    for r in range(n_side):
        for c in range(n_side):
            if len(chunks) == n_chunks:
                break
            chunks.append(dict(chunk_id=len(chunks) + 1, name=grid.chunk_name(r, c, 3), grid_row=r, grid_col=c,
                               row_off=r * cp, col_off=c * cp, width=cp, height=cp,
                               xmin=c * cp * res, ymax=-r * cp * res,
                               xmax=(c + 1) * cp * res, ymin=-(r + 1) * cp * res, aoi_frac=1.0))
    return dict(crs="EPSG:6933", res=res, x0=0.0, y0=0.0, width=n_side * cp, height=n_side * cp, chunk_px=cp,
                n_chunk_rows=n_side, n_chunk_cols=n_side, transform=[res, 0.0, 0.0, 0.0, -res, 0.0],
                pid_dtype="int64", aoi_file="data/aoi/test_aoi.gpkg", chunks=chunks)


def test_lookup_cache_memory_is_bounded(make_project):
    """A long monitor loads the grid every round; lookups must not keep every loaded grid alive."""
    cfg = make_project()
    out = grid_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid_def.json").write_text(json.dumps(_synthetic_big_grid()))

    def cycle():
        gd = grid.load_grid(cfg)
        assert grid.chunk_at(gd, 5, 5)["name"] == "chunk_r005c005"
        assert grid.chunk_by_name(gd, "chunk_r010c003_s1")["row_off"] == 10 * 512

    grid._LOOKUP_CACHE.clear()
    tracemalloc.start()
    try:
        for _ in range(10):
            cycle()
        gc.collect()
        after_warmup = tracemalloc.get_traced_memory()[0]
        for _ in range(40):
            cycle()
        gc.collect()
        after_50 = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    assert len(grid._LOOKUP_CACHE) <= grid._LOOKUP_CACHE_SIZE
    assert after_50 - after_warmup < 5 * 1024**2, f"grew by {(after_50 - after_warmup) / 1024**2:.1f} MB"
