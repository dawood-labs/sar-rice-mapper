"""Tests for sar_pipeline.stack (no network). The synthetic run comes from tests/_stack_fixtures.py."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _stack_fixtures import (  # noqa: E402
    ACQS, CPX, H, MANIFEST_COLUMNS, RES, TRACK, TRACK2, W, X0, Y0, add_track, aoi_mask, build_synthetic_run,
    expected_valid_pct, manifest_row, set_manifest, write_tif, write_window,
)

from sar_pipeline import resources, stack  # noqa: E402
from sar_pipeline.config import grid_dir, season_dir  # noqa: E402
from sar_pipeline.errors import AuditRequired, DecisionRequired, GridMismatch, PipelineError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(make_project):
    return make_project({"resources": {"cpu_workers": 1}, "qa": {"min_valid_pct": 80, "acknowledged_issues": []}})


# ---------------------------------------------------------------- VRT content
@pytest.mark.parametrize("dtype", ["float32", "int16"])
def test_vrt_values_match_sources_across_chunk_borders(cfg, dtype):
    syn = build_synthetic_run(cfg, dtype)
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    stack.build_stack_vrts(cfg, syn["run"], TRACK)
    sdir = stack.stack_dir(syn["run"], TRACK)
    samples = [(3, 0), (4, 0), (7, 7), (7, 8), (8, 7), (8, 8), (11, 3), (12, 4), (14, 15), (0, 16), (10, 19), (2, 3), (0, 0)]
    for pol, bands in (("VV", [0, 2, 4]), ("VH", [1, 3, 5])):
        with rasterio.open(sdir / f"stack_{pol}.vrt") as src:
            assert (src.width, src.height, src.count) == (W, H, 3)
            assert src.transform == Affine(RES, 0, X0, 0, -RES, Y0)
            assert src.dtypes[0] == dtype
            assert src.nodata == syn["nodata"]
            assert src.crs.to_epsg() == 6933
            full = src.read()
        for r, c in samples:
            np.testing.assert_array_equal(full[:, r, c], syn["mosaic"][bands, r, c])
        np.testing.assert_array_equal(full, syn["mosaic"][bands])
    # single-date VRT, including the four sub-chunks of the SPLIT chunk (rows 8-14, cols 0-7)
    with rasterio.open(sdir / "VH" / "VH_20260417.vrt") as src:
        band = src.read(1)
        np.testing.assert_array_equal(band, syn["mosaic"][3])
        np.testing.assert_array_equal(band[8:15, 0:8], syn["data"][3, 8:15, 0:8])
        assert src.descriptions[0] == "VH_20260417"


def test_stack_modules_do_not_import_gdal_bindings():
    code = "import sys, sar_pipeline.stack, sar_pipeline.pixel_query; print('osgeo' in sys.modules)"
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True)
    assert out.stdout.strip() == "False"


def test_stack_band_order_matches_dates_csv(cfg):
    syn = build_synthetic_run(cfg)
    with pytest.raises(DecisionRequired):
        stack.build_stack(cfg, syn["run"], TRACK)
    sdir = stack.stack_dir(syn["run"], TRACK)
    dates = pd.read_csv(sdir / "dates.csv", dtype={"date_utc": str})
    assert list(dates["stack_band"]) == [1, 2, 3]
    assert list(dates["acquisition_id"]) == [a for a, _, _ in ACQS]
    for pol in ("VV", "VH"):
        with rasterio.open(sdir / f"stack_{pol}.vrt") as src:
            assert list(src.descriptions) == [f"{pol}_{d.replace('-', '')}" for d in dates["date_utc"]]


def test_valid_pct_exact(cfg):
    syn = build_synthetic_run(cfg)
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    exp = expected_valid_pct(syn)
    got = valid.sort_values("band_idx")["valid_pct"].to_numpy()
    np.testing.assert_allclose(got, exp)
    assert (valid["aoi_pixels"] == aoi_mask().sum()).all()


def test_valid_pct_int16(cfg):
    syn = build_synthetic_run(cfg, "int16")
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    np.testing.assert_allclose(valid.sort_values("band_idx")["valid_pct"].to_numpy(), expected_valid_pct(syn))


def test_tiny_memory_budget_reads_in_small_blocks(cfg, monkeypatch):
    syn = build_synthetic_run(cfg)
    ref = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    monkeypatch.setattr(resources, "memory_budget_bytes", lambda *a, **k: 1)
    tiny = stack.compute_valid_pct(cfg, syn["run"], TRACK, force=True)
    assert tiny.attrs["rows_per_block"] == 1
    assert tiny.attrs["blocks_read"] > ref.attrs["blocks_read"]
    pd.testing.assert_frame_equal(ref, tiny)


def test_gdal_cache_counts_inside_memory_budget(cfg, monkeypatch):
    syn = build_synthetic_run(cfg)
    gb = 1024**3
    fake = resources.Resources(cpus=4, memory_total_bytes=8 * gb, memory_available_bytes=8 * gb, notes="fake")
    monkeypatch.setattr(resources, "detect_resources", lambda *a, **k: fake)
    seen = {}
    orig = resources.block_rows

    def spy(res, *a, **k):
        seen["available"] = res.memory_available_bytes
        return orig(res, *a, **k)

    monkeypatch.setattr(resources, "block_rows", spy)
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    cache_bytes = valid.attrs["gdal_cache_mb"] * 1024**2
    assert cache_bytes >= 16 * 1024**2
    assert seen["available"] < fake.memory_available_bytes
    assert fake.memory_available_bytes - seen["available"] >= cache_bytes  # 1 worker's cache reserved (/fraction)


def test_multiprocess_matches_serial(make_project):
    cfg1 = make_project({"resources": {"cpu_workers": 1}})
    syn = build_synthetic_run(cfg1)
    serial = stack.compute_valid_pct(cfg1, syn["run"], TRACK)
    cfg2 = dict(cfg1, resources={"cpu_workers": 3})
    parallel = stack.compute_valid_pct(cfg2, syn["run"], TRACK, force=True)
    assert parallel.attrs["workers"] == 3
    assert parallel.attrs["computed_files"] == 8
    cols = [c for c in serial.columns]
    pd.testing.assert_frame_equal(serial[cols], parallel[cols])


def test_dates_table_flags_and_rain(cfg):
    syn = build_synthetic_run(cfg)
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    d = stack.dates_table(cfg, syn["run"], TRACK, valid).set_index("acquisition_id")
    assert d.loc[ACQS[1][0], "rain_24h_mm"] == pytest.approx(7.2)
    assert "RAIN_24H" in d.loc[ACQS[1][0], "flags"]
    assert np.isnan(d.loc[ACQS[2][0], "rain_24h_mm"])
    assert set(d.loc[ACQS[2][0], "flags"].split(";")) == {"LOW_VALID_VV", "LOW_VALID_VH", "GAP_BEFORE"}
    assert d.loc[ACQS[0][0], "platforms"] == "A"
    assert list(d.columns[-3:]) == ["aoi_planned_pct", "n_temporal_neighbors", "flags"]


@pytest.mark.parametrize("ard,expected", [
    ({"multitemporal": True, "temporal_half_window": 1}, [2, 3, 2]),
    ({"multitemporal": True, "temporal_half_window": 3}, [3, 3, 3]),
    ({"multitemporal": False, "temporal_half_window": 3}, [1, 1, 1]),
])
def test_n_temporal_neighbors(make_project, ard, expected):
    cfg = make_project({"resources": {"cpu_workers": 1}, "ard": ard})
    syn = build_synthetic_run(cfg)
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    d = stack.dates_table(cfg, syn["run"], TRACK, valid)
    assert list(d["n_temporal_neighbors"]) == expected


def test_temporal_neighbor_counts_long_series():
    cfg = {"ard": {"multitemporal": True, "temporal_half_window": 3}}
    assert stack.temporal_neighbor_counts(10, cfg) == [4, 5, 6, 7, 7, 7, 7, 6, 5, 4]


def test_issues_and_acknowledged(make_project):
    cfg = make_project({"resources": {"cpu_workers": 1}})
    syn = build_synthetic_run(cfg)
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    issues = stack.qa_issues(cfg, syn["run"], TRACK, valid)
    ids = {i["issue_id"] for i in issues}
    assert ids == {
        f"MISSING_ACQUISITION:{TRACK}:20260423",
        f"LOW_VALID:{TRACK}:20260511",
        f"FAILED_CHUNK:{TRACK}:chunk_r01c02",
        f"UNVERIFIED_CHUNK:{TRACK}:chunk_r00c02",
    }
    assert all(not i["acknowledged"] for i in issues)
    with pytest.raises(DecisionRequired):
        stack.write_decisions_required(syn["run"], issues, track_id=TRACK)
    md = (syn["run"] / "decisions_required.md").read_text()
    for i in ids:
        assert i in md
    assert "What you can do" in md
    csv = pd.read_csv(syn["run"] / "decisions_required.csv")
    assert set(csv["issue_id"]) == ids

    cfg_ack = dict(cfg, qa={"min_valid_pct": 80, "acknowledged_issues": sorted(ids)})
    issues_ack = stack.qa_issues(cfg_ack, syn["run"], TRACK, valid)
    assert all(i["acknowledged"] for i in issues_ack)
    assert stack.write_decisions_required(syn["run"], issues_ack, track_id=TRACK) == syn["run"] / "decisions_required.md"

    with pytest.raises(DecisionRequired):
        stack.write_decisions_required(syn["run"], issues, track_id=TRACK, acknowledged=sorted(ids)[:2])

    assert stack.write_decisions_required(syn["run"], [], track_id=TRACK) is None
    assert pd.read_csv(syn["run"] / "decisions_required.csv").empty


def test_decisions_keep_other_tracks(cfg):
    syn = build_synthetic_run(cfg)
    other = [dict(issue_id="FAILED_CHUNK:RO010_DSC:chunk_r00c00", type="FAILED_CHUNK", track_id="RO010_DSC",
                  subject="chunk_r00c00", detail="x", acknowledged=False)]
    with pytest.raises(DecisionRequired):
        stack.write_decisions_required(syn["run"], other, track_id="RO010_DSC")
    stack.write_decisions_required(syn["run"], [], track_id=TRACK)  # other track still open -> file kept
    csv = pd.read_csv(syn["run"] / "decisions_required.csv")
    assert list(csv["track_id"]) == ["RO010_DSC"]


def test_vrt_paths_are_relative_and_run_is_portable(cfg, tmp_path):
    syn = build_synthetic_run(cfg)
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    stack.build_stack_vrts(cfg, syn["run"], TRACK)
    sdir = stack.stack_dir(syn["run"], TRACK)
    for vrt in sdir.rglob("*.vrt"):
        for el in ET.parse(vrt).getroot().iter("SourceFilename"):
            assert el.get("relativeToVRT") == "1", vrt
            assert not Path(el.text).is_absolute()
    moved = tmp_path / "elsewhere" / "v001_20260915"
    shutil.copytree(syn["run"], moved)
    shutil.rmtree(syn["run"] / "raw_chunks")  # prove the copy does not read the original files
    with rasterio.open(moved / "stack" / f"track_{TRACK}" / "stack_VV.vrt") as src:
        np.testing.assert_array_equal(src.read(), syn["mosaic"][[0, 2, 4]])


def test_missing_verified_file_raises(cfg):
    syn = build_synthetic_run(cfg)
    (syn["run"] / "raw_chunks" / f"track_{TRACK}" / "chunk_r01c01.tif").unlink()
    with pytest.raises(PipelineError, match="missing"):
        stack.build_date_vrts(cfg, syn["run"], TRACK)


def test_other_crs_source_is_not_silently_dropped(cfg):
    syn = build_synthetic_run(cfg)
    bad = syn["run"] / "raw_chunks" / f"track_{TRACK}" / "chunk_r01c01.tif"
    with rasterio.open(bad) as src:
        arr, prof = src.read(), src.profile
    prof.update(crs="EPSG:32650")
    with rasterio.open(bad, "w", **prof) as dst:
        dst.write(arr)
    with pytest.raises(PipelineError, match="differs from grid CRS"):
        stack.build_date_vrts(cfg, syn["run"], TRACK)


def test_unsnapped_origin_and_wrong_dtype_raise(cfg):
    syn = build_synthetic_run(cfg)
    bad = syn["run"] / "raw_chunks" / f"track_{TRACK}" / "chunk_r01c01.tif"
    with rasterio.open(bad) as src:
        arr, prof = src.read(), src.profile
    prof.update(transform=Affine(RES, 0, X0 + 8 * RES + 3.0, 0, -RES, Y0 - 8 * RES))
    with rasterio.open(bad, "w", **prof) as dst:
        dst.write(arr)
    with pytest.raises(PipelineError, match="not snapped"):
        stack.build_date_vrts(cfg, syn["run"], TRACK)
    prof.update(transform=Affine(RES, 0, X0 + 8 * RES, 0, -RES, Y0 - 8 * RES), dtype="float64")
    with rasterio.open(bad, "w", **prof) as dst:
        dst.write(arr.astype("float64"))
    with pytest.raises(PipelineError, match="data type"):
        stack.build_date_vrts(cfg, syn["run"], TRACK, force=True)


# ---------------------------------------------------------------- manifest states
def test_verified_empty_chunk_is_in_stack_and_reported(cfg):
    syn = build_synthetic_run(cfg)
    empty = np.full((6, 7, 8), syn["nodata"], dtype=syn["dtype"])
    write_tif(syn["run"] / "raw_chunks" / f"track_{TRACK}" / "chunk_r01c01.tif", empty, 8, 8, syn["dtype"], syn["nodata"])
    man_path = syn["run"] / "export_manifest.csv"
    man = pd.read_csv(man_path, dtype=str, keep_default_na=False)
    man.loc[man["chunk_name"] == "chunk_r01c01", "state"] = "VERIFIED_EMPTY"
    man.to_csv(man_path, index=False)

    stack.build_date_vrts(cfg, syn["run"], TRACK)
    with rasterio.open(stack.stack_dir(syn["run"], TRACK) / "VV" / "VV_20260405.vrt") as src:
        assert (src.read(1)[8:15, 8:16] == syn["nodata"]).all()
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    issues = {i["issue_id"]: i for i in stack.qa_issues(cfg, syn["run"], TRACK, valid)}
    assert f"EMPTY_CHUNK:{TRACK}:chunk_r01c01" in issues
    assert f"UNVERIFIED_CHUNK:{TRACK}:chunk_r01c01" not in issues
    assert f"FAILED_CHUNK:{TRACK}:chunk_r01c01" not in issues
    with pytest.raises(DecisionRequired):
        stack.write_decisions_required(syn["run"], list(issues.values()), track_id=TRACK)
    assert "EMPTY_CHUNK" in (syn["run"] / "decisions_required.md").read_text()


def test_not_covered_and_planned_rows_are_outside_scope(cfg):
    syn = build_synthetic_run(cfg)
    set_manifest(syn, [
        manifest_row(TRACK, "chunk_a", "VERIFIED", 12, 0, 3, 4, [write_window(syn, TRACK, "a.tif", 12, 0, 3, 4)]),
        manifest_row(TRACK, "chunk_b", "NOT_COVERED", 12, 4, 3, 4, []),
        manifest_row(TRACK, "chunk_c", "PLANNED", 8, 0, 4, 4, []),
    ])
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert (valid["aoi_pixels"] == 12).all()
    assert (valid["valid_pct"] == 100.0).all()
    np.testing.assert_allclose(valid["aoi_planned_pct"], 12 / aoi_mask().sum() * 100)
    ids = {i["issue_id"] for i in stack.qa_issues(cfg, syn["run"], TRACK, valid)}
    assert not any("chunk_b" in i or "chunk_c" in i for i in ids)


def test_manifest_with_adopted_column_is_accepted(cfg):
    syn = build_synthetic_run(cfg)
    rows = [dict(manifest_row(TRACK, "chunk_a", "VERIFIED", 12, 0, 3, 4,
                              [write_window(syn, TRACK, "a.tif", 12, 0, 3, 4)]), adopted=0)]
    set_manifest(syn, rows, columns=MANIFEST_COLUMNS + ["adopted"])
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert (valid["valid_pct"] == 100.0).all()


def test_polarisations_come_from_band_layout(cfg):
    syn = build_synthetic_run(cfg)
    path = syn["run"] / "band_layout.csv"
    bl = pd.read_csv(path)
    bl[bl["pol"] == "VH"].to_csv(path, index=False)
    outs = stack.build_date_vrts(cfg, syn["run"], TRACK)
    assert {p.parent.name for p in outs} == {"VH"}
    assert set(stack.build_stack_vrts(cfg, syn["run"], TRACK)) == {"VH"}
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    d = stack.dates_table(cfg, syn["run"], TRACK, valid)
    assert "valid_pct_VH" in d.columns and "valid_pct_VV" not in d.columns
    assert not (stack.stack_dir(syn["run"], TRACK) / "stack_VV.vrt").exists()


# ---------------------------------------------------------------- resume, never redo
def _run_until_decision(cfg, run):
    with pytest.raises(DecisionRequired):
        stack.build_stack(cfg, run, TRACK)


def test_second_build_reads_zero_blocks_and_skips_vrts(cfg, monkeypatch):
    syn = build_synthetic_run(cfg)
    _run_until_decision(cfg, syn["run"])
    sdir = stack.stack_dir(syn["run"], TRACK)
    mtimes = {p: p.stat().st_mtime_ns for p in sdir.rglob("*") if p.is_file() and "_valid_cache" not in p.parts}
    parts = [p.name for p in stack._cache_parts(sdir)]
    calls = {"file": 0, "aoi": 0}
    orig_f, orig_a = stack._count_valid_file, stack._count_aoi_window
    monkeypatch.setattr(stack, "_count_valid_file", lambda *a: (calls.__setitem__("file", calls["file"] + 1), orig_f(*a))[1])
    monkeypatch.setattr(stack, "_count_aoi_window", lambda *a: (calls.__setitem__("aoi", calls["aoi"] + 1), orig_a(*a))[1])
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert valid.attrs["blocks_read"] == 0 and calls == {"file": 0, "aoi": 0}
    assert valid.attrs["reused"] == valid.attrs["total"] == 8 * 6
    assert [p.name for p in stack._cache_parts(sdir)] == parts  # nothing rewritten
    _run_until_decision(cfg, syn["run"])
    now = {p: p.stat().st_mtime_ns for p in sdir.rglob("*.vrt*")}
    assert all(now[p] == mtimes[p] for p in now)
    np.testing.assert_allclose(valid.sort_values("band_idx")["valid_pct"], expected_valid_pct(syn))


def test_unchanged_inputs_skip_without_parsing_xml_or_opening_chunks(cfg, monkeypatch):
    syn = build_synthetic_run(cfg)
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    stack.build_stack_vrts(cfg, syn["run"], TRACK)
    calls = {"header": 0, "parse": 0}
    orig_h, orig_p = stack._read_header, ET.parse
    monkeypatch.setattr(stack, "_read_header", lambda *a: (calls.__setitem__("header", calls["header"] + 1), orig_h(*a))[1])
    monkeypatch.setattr(ET, "parse", lambda *a, **k: (calls.__setitem__("parse", calls["parse"] + 1), orig_p(*a, **k))[1])
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    stack.build_stack_vrts(cfg, syn["run"], TRACK)
    assert calls == {"header": 0, "parse": 0}
    for vrt in stack.stack_dir(syn["run"], TRACK).rglob("*.vrt"):
        assert (vrt.parent / (vrt.name + stack.SOURCES_SUFFIX)).exists()


def test_crash_mid_valid_pct_then_resume_with_cache_parts(cfg, monkeypatch):
    syn = build_synthetic_run(cfg)
    sdir = stack.stack_dir(syn["run"], TRACK)
    orig = stack._count_valid_file
    n = {"calls": 0}

    def crashing(*a):
        n["calls"] += 1
        if n["calls"] == 4:
            raise RuntimeError("simulated crash")
        return orig(*a)

    monkeypatch.setattr(stack, "_count_valid_file", crashing)
    with pytest.raises(RuntimeError):
        stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert len(stack._cache_parts(sdir)) >= 1
    persisted = stack._load_valid_cache(sdir)["path"].nunique()
    assert 0 < persisted < 8

    counted = {"calls": 0}
    monkeypatch.setattr(stack, "_count_valid_file", lambda *a: (counted.__setitem__("calls", counted["calls"] + 1), orig(*a))[1])
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert counted["calls"] == 8 - persisted == valid.attrs["computed_files"]
    np.testing.assert_allclose(valid.sort_values("band_idx")["valid_pct"], expected_valid_pct(syn))
    parts = stack._cache_parts(sdir)
    assert len(parts) == 1  # compacted
    assert len(pd.read_csv(parts[0])) == 8 * 6


def test_batches_append_parts_instead_of_rewriting(cfg, monkeypatch):
    syn = build_synthetic_run(cfg)
    sdir = stack.stack_dir(syn["run"], TRACK)
    written = []
    orig = stack.atomic_write_csv

    def spy(df, path):
        written.append((Path(path).name, len(df)))
        return orig(df, path)

    monkeypatch.setattr(stack, "atomic_write_csv", spy)
    stack.compute_valid_pct(cfg, syn["run"], TRACK)
    part_writes = [w for w in written if w[0].startswith("part-")]
    # 8 files, batch of 2 (1 worker) -> 4 batch parts of 2 files each, then one compaction
    assert [n for _, n in part_writes[:4]] == [12, 12, 12, 12]
    assert part_writes[-1][1] == 48
    assert len(stack._cache_parts(sdir)) == 1


def test_touching_one_chunk_recomputes_only_that_chunk(cfg):
    syn = build_synthetic_run(cfg)
    stack.compute_valid_pct(cfg, syn["run"], TRACK)
    f = syn["run"] / "raw_chunks" / f"track_{TRACK}" / "chunk_r01c01.tif"
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert valid.attrs["computed_files"] == 1
    assert valid.attrs["reused"] == 7 * 6


def test_old_stale_tmp_removed_and_force_recomputes(cfg):
    syn = build_synthetic_run(cfg)
    _run_until_decision(cfg, syn["run"])
    sdir = stack.stack_dir(syn["run"], TRACK)
    stale = sdir / "VV" / "VV_20260405.vrt.tmp"
    stale.write_text("half written")
    old = time.time() - 7 * 3600
    os.utime(stale, (old, old))
    fresh = sdir / "VV" / "fresh.vrt.tmp"  # may belong to a live writer: kept
    fresh.write_text("in progress")
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK, force=True)
    assert not stale.exists() and fresh.exists()
    assert valid.attrs["computed_files"] == 8 and valid.attrs["reused"] == 0


def test_vrts_rebuilt_only_when_chunk_set_changes(cfg):
    syn = build_synthetic_run(cfg)
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    stack.build_stack_vrts(cfg, syn["run"], TRACK)
    sdir = stack.stack_dir(syn["run"], TRACK)
    date_vrt = sdir / "VV" / "VV_20260417.vrt"
    before = date_vrt.stat().st_mtime_ns
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    assert date_vrt.stat().st_mtime_ns == before
    assert stack.compute_valid_pct(cfg, syn["run"], TRACK).attrs["computed_files"] == 8

    man_path = syn["run"] / "export_manifest.csv"
    man = pd.read_csv(man_path, dtype=str).fillna("")
    man.loc[man["chunk_name"] == "chunk_r00c02", "state"] = "VERIFIED"
    man.to_csv(man_path, index=False)
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    srcs = [el.text for el in ET.parse(date_vrt).getroot().iter("SourceFilename")]
    assert any("chunk_r00c02" in s for s in srcs)
    stack.build_stack_vrts(cfg, syn["run"], TRACK)
    with rasterio.open(sdir / "stack_VV.vrt") as src:
        got = src.read(2)
    np.testing.assert_array_equal(got[0:8, 16:20], syn["data"][2, 0:8, 16:20])
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert valid.attrs["computed_files"] == 1


def test_crash_between_vrt_and_fingerprint_rebuilds(cfg):
    syn = build_synthetic_run(cfg)
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    vrt = stack.stack_dir(syn["run"], TRACK) / "VV" / "VV_20260405.vrt"
    (vrt.parent / (vrt.name + stack.SOURCES_SUFFIX)).unlink()
    vrt.write_text("<half")
    stack.build_date_vrts(cfg, syn["run"], TRACK)
    with rasterio.open(vrt) as src:
        np.testing.assert_array_equal(src.read(1), syn["mosaic"][0])


# ---------------------------------------------------------------- pinned schemas / audit / grid checks
def _edit_meta(run, **changes):
    path = run / "export_meta.json"
    meta = json.loads(path.read_text())
    for k, v in changes.items():
        if v is None:
            meta.pop(k, None)
        else:
            meta[k] = v
    path.write_text(json.dumps(meta))


def test_missing_audit_dir_key_raises(cfg):
    syn = build_synthetic_run(cfg)
    _edit_meta(syn["run"], audit_dir=None)
    with pytest.raises(AuditRequired, match="audit_dir"):
        stack.build_stack(cfg, syn["run"], TRACK)


def test_missing_audit_folder_raises(cfg):
    syn = build_synthetic_run(cfg)
    shutil.rmtree(season_dir(cfg) / "audit" / "20260915")
    with pytest.raises(AuditRequired, match="not found"):
        stack.build_stack(cfg, syn["run"], TRACK)


def test_missing_acquisitions_csv_raises(cfg):
    syn = build_synthetic_run(cfg)
    (season_dir(cfg) / "audit" / "20260915" / "s1_acquisitions.csv").unlink()
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    with pytest.raises(AuditRequired, match="s1_acquisitions.csv"):
        stack.qa_issues(cfg, syn["run"], TRACK, valid)


def test_grid_fingerprint_mismatch_raises(cfg):
    syn = build_synthetic_run(cfg)
    gpath = grid_dir(cfg) / "grid_def.json"
    gpath.write_text(gpath.read_text() + "\n")
    with pytest.raises(GridMismatch):
        stack.build_date_vrts(cfg, syn["run"], TRACK)
    with pytest.raises(GridMismatch):
        stack.compute_valid_pct(cfg, syn["run"], TRACK)


def test_missing_required_meta_key_raises(cfg):
    syn = build_synthetic_run(cfg)
    _edit_meta(syn["run"], grid_fingerprint=None)
    with pytest.raises(PipelineError, match="grid_fingerprint"):
        stack.read_export_meta(cfg, syn["run"])


def test_nodata_and_scale_fallback_warns(make_project, caplog):
    cfg = make_project({"export": {"nodata_float32": -9999, "int16_scale": 100}})
    syn = build_synthetic_run(cfg)
    _edit_meta(syn["run"], nodata=None, int16_scale=None)
    with caplog.at_level(logging.WARNING, logger="sar_pipeline.run_meta"):
        meta = stack.read_export_meta(cfg, syn["run"])
    assert meta["nodata"] == -9999.0 and meta["int16_scale"] == 100.0
    assert "no 'nodata'" in caplog.text and "no 'int16_scale'" in caplog.text


def test_band_layout_missing_column_raises(cfg):
    syn = build_synthetic_run(cfg)
    path = syn["run"] / "band_layout.csv"
    pd.read_csv(path).drop(columns=["date_local"]).to_csv(path, index=False)
    with pytest.raises(PipelineError, match="date_local"):
        stack.read_band_layout(syn["run"], TRACK)


# ---------------------------------------------------------------- planned scope (pilot runs)
def test_pilot_scope_is_fully_valid(cfg):
    syn = build_synthetic_run(cfg)
    set_manifest(syn, [
        manifest_row(TRACK, "chunk_r01c00", "SPLIT", 8, 0, 7, 8, []),           # parent: ignored
        manifest_row(TRACK, "chunk_r01c00_s2", "VERIFIED", 12, 0, 3, 4,
                     [write_window(syn, TRACK, "chunk_r01c00_s2.tif", 12, 0, 3, 4)], parent="chunk_r01c00"),
    ])
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert (valid["valid_pct"] == 100.0).all()
    assert (valid["aoi_pixels"] == 12).all()
    expected_scope = 12 / aoi_mask().sum() * 100
    np.testing.assert_allclose(valid["aoi_planned_pct"], expected_scope)
    issues = stack.qa_issues(cfg, syn["run"], TRACK, valid)
    assert not [i for i in issues if i["type"] == "LOW_VALID"]
    dates = stack.dates_table(cfg, syn["run"], TRACK, valid)
    np.testing.assert_allclose(dates["aoi_planned_pct"], expected_scope)
    assert not dates["flags"].str.contains("LOW_VALID").any()


def test_planned_chunks_missing_lower_valid_and_raise_issues(cfg):
    syn = build_synthetic_run(cfg)
    set_manifest(syn, [
        manifest_row(TRACK, "chunk_a", "VERIFIED", 12, 0, 3, 4, [write_window(syn, TRACK, "a.tif", 12, 0, 3, 4)]),
        manifest_row(TRACK, "chunk_b", "FAILED", 12, 4, 3, 4, [], err="User memory limit exceeded"),
        manifest_row(TRACK, "chunk_c", "DOWNLOADED", 8, 0, 4, 4, [write_window(syn, TRACK, "c.tif", 8, 0, 4, 4)]),
    ])
    valid = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert (valid["aoi_pixels"] == 40).all()
    np.testing.assert_allclose(valid["valid_pct"], 30.0)
    types = {i["issue_id"] for i in stack.qa_issues(cfg, syn["run"], TRACK, valid)}
    assert f"FAILED_CHUNK:{TRACK}:chunk_b" in types
    assert f"UNVERIFIED_CHUNK:{TRACK}:chunk_c" in types
    assert {f"LOW_VALID:{TRACK}:{a[-8:]}" for a, _, _ in ACQS} <= types


def test_scope_change_updates_denominator_but_reuses_file_counts(cfg):
    syn = build_synthetic_run(cfg)
    rows = [manifest_row(TRACK, "chunk_a", "VERIFIED", 12, 0, 3, 4, [write_window(syn, TRACK, "a.tif", 12, 0, 3, 4)])]
    set_manifest(syn, rows)
    first = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert (first["valid_pct"] == 100.0).all()
    rows.append(manifest_row(TRACK, "chunk_b", "PENDING", 12, 4, 3, 4, []))
    set_manifest(syn, rows)
    second = stack.compute_valid_pct(cfg, syn["run"], TRACK)
    assert second.attrs["scope_fingerprint"] != first.attrs["scope_fingerprint"]
    assert second.attrs["computed_files"] == 0 and second.attrs["reused"] == 6
    np.testing.assert_allclose(second["valid_pct"], 50.0)


def test_overlapping_planned_windows_raise(cfg):
    syn = build_synthetic_run(cfg)
    set_manifest(syn, [
        manifest_row(TRACK, "chunk_a", "VERIFIED", 8, 0, 4, 4, [write_window(syn, TRACK, "a.tif", 8, 0, 4, 4)]),
        manifest_row(TRACK, "chunk_b", "PENDING", 10, 2, 4, 4, []),
    ])
    with pytest.raises(PipelineError, match="overlap"):
        stack.compute_valid_pct(cfg, syn["run"], TRACK)


# ---------------------------------------------------------------- several tracks
def test_two_tracks_built_before_one_aggregated_decision(cfg):
    syn = build_synthetic_run(cfg)
    add_track(cfg, syn)
    with pytest.raises(DecisionRequired) as exc:
        stack.build_stacks(cfg, syn["run"], [TRACK, TRACK2])
    assert TRACK in str(exc.value) and TRACK2 in str(exc.value)
    for t in (TRACK, TRACK2):
        sdir = stack.stack_dir(syn["run"], t)
        assert (sdir / "dates.csv").exists() and (sdir / "stack_VV.vrt").exists() and (sdir / "stack_VH.vrt").exists()
        with rasterio.open(sdir / "stack_VH.vrt") as src:
            np.testing.assert_array_equal(src.read(), syn["mosaic"][[1, 3, 5]])
    csv = pd.read_csv(syn["run"] / "decisions_required.csv")
    assert set(csv["track_id"]) == {TRACK, TRACK2}
    assert f"MISSING_ACQUISITION:{TRACK2}:20260423" in set(csv["issue_id"])
    md = (syn["run"] / "decisions_required.md").read_text()
    assert TRACK in md and TRACK2 in md


def test_one_track_error_does_not_block_other_track(cfg):
    syn = build_synthetic_run(cfg)
    add_track(cfg, syn, with_manifest=False)   # TRACK2 has bands but nothing planned
    with pytest.raises(PipelineError) as exc:
        stack.build_stacks(cfg, syn["run"], [TRACK2, TRACK])
    assert not isinstance(exc.value, DecisionRequired)
    assert TRACK2 in str(exc.value) and "Also open" in str(exc.value)
    assert (stack.stack_dir(syn["run"], TRACK) / "dates.csv").exists()
    assert set(pd.read_csv(syn["run"] / "decisions_required.csv")["track_id"]) == {TRACK}


def test_chunk_px_constant_matches_fixture():
    assert CPX == 8  # fixture sanity (sub-chunk windows above assume 8 px chunks)


def test_adopted_minus_one_is_tolerated(make_project):
    """Newer manifests carry an 'adopted' column; -1 (never adopt old output) must behave like 0 for the stack."""
    from _stack_fixtures import MANIFEST_COLUMNS, build_synthetic_run as _build

    cfg = make_project({"resources": {"cpu_workers": 1}})
    syn = _build(cfg, "float32")
    with pytest.raises(DecisionRequired):
        stack.build_stack(cfg, syn["run"], TRACK)
    dates = syn["run"] / "stack" / f"track_{TRACK}" / "dates.csv"
    baseline = pd.read_csv(dates)
    man_path = syn["run"] / "export_manifest.csv"
    man = pd.read_csv(man_path, dtype=str, keep_default_na=False)
    man["adopted"] = "-1"
    man.to_csv(man_path, index=False, columns=MANIFEST_COLUMNS + ["adopted"])
    with pytest.raises(DecisionRequired):
        stack.build_stack(cfg, syn["run"], TRACK, force=True)
    after = pd.read_csv(dates)
    valid_cols = [c for c in baseline.columns if c.startswith("valid_pct")]
    pd.testing.assert_frame_equal(baseline[valid_cols], after[valid_cols])
