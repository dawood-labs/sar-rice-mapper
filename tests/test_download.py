"""Unit tests for GCS download + verification, with a fake bucket and synthetic GeoTIFFs (no network)."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from sar_pipeline import config, download, export, grid, resources, run_meta
from sar_pipeline import manifest as mf
from sar_pipeline.errors import ConfirmationRequired, PipelineError

RES = 10.0
X0, Y0 = 300000.0, 1900000.0
NODATA = -9999.0
TRACK = "RO123_ASC"
REPO = Path(__file__).resolve().parents[1]


def make_gd():
    chunks = [
        dict(chunk_id=1, name="chunk_r00c00", grid_row=0, grid_col=0, row_off=0, col_off=0, width=64, height=64,
             xmin=X0, ymin=Y0 - 640, xmax=X0 + 640, ymax=Y0, aoi_frac=1.0),
        dict(chunk_id=2, name="chunk_r00c01", grid_row=0, grid_col=1, row_off=0, col_off=64, width=40, height=64,
             xmin=X0 + 640, ymin=Y0 - 640, xmax=X0 + 1040, ymax=Y0, aoi_frac=1.0),
    ]
    return dict(crs="EPSG:6933", res=RES, x0=X0, y0=Y0, width=104, height=64, chunk_px=64, n_chunk_rows=1,
                n_chunk_cols=2, transform=[RES, 0.0, X0, 0.0, -RES, Y0], pid_dtype="int32", aoi_file="x",
                chunks=chunks)


def write_tif(path: Path, left, top, width, height, bands=4, fill=-12.5, crs="EPSG:6933", res=RES,
              nodata=NODATA, dtype="float32"):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((bands, height, width), fill, dtype=dtype)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=bands, dtype=dtype,
                       crs=crs, transform=from_origin(left, top, res, res), nodata=nodata) as dst:
        dst.write(data)
    return path


# ---------------------------------------------------------------- verification
@pytest.fixture
def gd():
    return make_gd()


def test_verify_ok(tmp_path, gd):
    ch = gd["chunks"][0]
    p = write_tif(tmp_path / "a.tif", ch["xmin"], ch["ymax"], 64, 64)
    assert download.verify_chunk_files([p], gd, ch, 4, NODATA) == (True, "ok")
    assert download.inspect_chunk_files([p], gd, ch, 4, NODATA, "float32") == ("ok", "ok")


def test_verify_multipart_union(tmp_path, gd):
    ch = gd["chunks"][0]
    top = write_tif(tmp_path / "c-0000000000-0000000000.tif", ch["xmin"], ch["ymax"], 64, 30)
    bottom = write_tif(tmp_path / "c-0000000030-0000000000.tif", ch["xmin"], ch["ymax"] - 300, 64, 34, fill=NODATA)
    assert download.verify_chunk_files([top, bottom], gd, ch, 4, NODATA)[0]
    ok, reason = download.verify_chunk_files([top], gd, ch, 4, NODATA)
    assert not ok and "extent" in reason


@pytest.mark.parametrize("kwargs,expect", [
    (dict(left_shift=3.0), "snapped"),
    (dict(bands=3), "bands"),
    (dict(crs="EPSG:3857"), "CRS"),
    (dict(res=20.0, width=32, height=32), "pixel size"),
    (dict(width=63), "extent"),
    (dict(dtype="int16", fill=-1250, nodata=-32768), "dtype"),
    (dict(nodata=0.0), "nodata tag"),
])
def test_verify_rejects(tmp_path, gd, kwargs, expect):
    ch = gd["chunks"][0]
    shift = kwargs.pop("left_shift", 0.0)
    params = dict(width=64, height=64)
    params.update(kwargs)
    p = write_tif(tmp_path / "bad.tif", ch["xmin"] + shift, ch["ymax"], **params)
    status, reason = download.inspect_chunk_files([p], gd, ch, 4, NODATA, "float32")
    assert status == "bad" and expect in reason, reason


@pytest.mark.parametrize("fill", [NODATA, np.nan])
def test_all_nodata_is_empty_not_bad(tmp_path, gd, fill):
    ch = gd["chunks"][0]
    p = write_tif(tmp_path / "empty.tif", ch["xmin"], ch["ymax"], 64, 64, fill=fill)
    assert download.inspect_chunk_files([p], gd, ch, 4, NODATA, "float32")[0] == "empty"
    assert download.verify_chunk_files([p], gd, ch, 4, NODATA) == (False, "all pixels are nodata")


def test_chunk_blob_names_excludes_subchunks():
    prefix = "base/runs/v001/raw_chunks/track_T/chunk_r00c00"
    blobs = [{"name": n, "size": 1, "crc32c": ""} for n in (
        prefix + ".tif", prefix + "-0000000000-0000000256.tif", prefix + "_s0.tif", prefix + "_s0-0000000000-0000000000.tif",
        prefix + ".tif.aux", prefix + "0.tif",
    )]
    names = [b["name"] for b in download.chunk_blob_names(blobs, prefix)]
    assert names == [prefix + "-0000000000-0000000256.tif", prefix + ".tif"]
    groups = download.group_blobs_by_prefix(blobs)
    assert len(groups[prefix + "_s0"]) == 2 and len(groups[prefix + "0"]) == 1


# ---------------------------------------------------------------- environment
class FakeGCS:
    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.fail_times: dict[str, int] = {}      # name -> remaining failures ("crc mismatch")
        self.download_calls: dict[str, int] = {}
        self.list_calls: list[str] = []
        self.created: dict[str, str] = {}         # name -> ISO-8601 UTC creation time

    @property
    def n_downloads(self):
        return sum(self.download_calls.values())

    def put(self, name, local: Path, created: str | None = None):
        self.objects[name] = local.read_bytes()
        self.created[name] = created or mf.now_utc()

    def list_blobs(self, prefix):
        self.list_calls.append(prefix)
        return [{"name": n, "size": len(b), "crc32c": download.bytes_crc32c_b64(b),
                 "time_created": self.created.get(n, "")}
                for n, b in self.objects.items() if n.startswith(prefix)]

    def download(self, name, dest: Path):
        self.download_calls[name] = self.download_calls.get(name, 0) + 1
        if self.fail_times.get(name, 0) > 0:
            self.fail_times[name] -= 1
            raise RuntimeError("DataCorruption: Checksum mismatch while downloading")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part.tmp")
        tmp.write_bytes(self.objects[name])
        os.replace(tmp, dest)


@pytest.fixture(autouse=True)
def manifest_contract(monkeypatch):
    """The shared manifest contract (new states + `adopted` column) even if manifest.py predates it."""
    for state in ("PLANNED", "NOT_COVERED", "VERIFIED_EMPTY"):
        if state not in mf.STATES:
            monkeypatch.setattr(mf, "STATES", [*mf.STATES, state])
    if "adopted" not in mf.COLUMNS:
        monkeypatch.setattr(mf, "COLUMNS", [*mf.COLUMNS, "adopted"])
        monkeypatch.setattr(mf, "INT_COLUMNS", [*mf.INT_COLUMNS, "adopted"])


def _row(run, ch, track=TRACK, **extra):
    key = f"{track}__{ch['name']}"
    return dict(task_key=key, track_id=track, chunk_name=ch["name"], row_off=ch["row_off"],
                col_off=ch["col_off"], width=ch["width"], height=ch["height"], n_bands=4,
                gcs_prefix=f"fake_base/runs/{run.name}_uid/raw_chunks/track_{track}/{ch['name']}",
                task_id="T" + key, state="COMPLETED", attempts=1, **extra)


def _setup(make_project, monkeypatch, tmp_path, tracks=(TRACK,), dtype="float32"):
    cfg = make_project()
    gd = make_gd()
    monkeypatch.setattr(grid, "load_grid", lambda c: gd)
    gdir = config.grid_dir(cfg)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "grid_def.json").write_text(json.dumps(gd))
    run = config.new_run(cfg)
    meta = dict(run_id=run.name, dtype=dtype, nodata=NODATA, int16_scale=100, grid_crs=gd["crs"],
                grid_transform=gd["transform"], grid_fingerprint=run_meta.grid_fingerprint(cfg), audit_dir="x",
                tracks=list(tracks), config_hash="h", created_utc="2026-09-15T00:00:00Z")
    (run / run_meta.EXPORT_META).write_text(json.dumps(meta))
    rows = [_row(run, ch, track) for track in tracks for ch in gd["chunks"]]
    mf.update_manifest(run, rows)
    src = tmp_path / "src"
    gcs = FakeGCS()
    for row in rows:
        ch = next(c for c in gd["chunks"] if c["name"] == row["chunk_name"])
        f = write_tif(src / f"{row['task_key']}.tif", ch["xmin"], ch["ymax"], ch["width"], ch["height"])
        gcs.put(row["gcs_prefix"] + ".tif", f)
    return type("E", (), dict(cfg=cfg, gd=gd, run=run, gcs=gcs, rows=rows, src=src))


@pytest.fixture
def denv(make_project, monkeypatch, tmp_path):
    return _setup(make_project, monkeypatch, tmp_path)


def _states(run):
    return mf.read_manifest(run).set_index("task_key")


def _run(env, **kw):
    return download.download_completed(env.cfg, env.run, env.gcs, confirmed=True, **kw)


# ---------------------------------------------------------------- basic flow
def test_download_requires_confirmation(denv):
    with pytest.raises(ConfirmationRequired):
        download.download_completed(denv.cfg, denv.run, denv.gcs)


def test_estimate_download(denv):
    est = download.estimate_download(denv.cfg, denv.run, denv.gcs)
    assert est["n_tasks"] == 2 and est["n_files"] == 2
    assert est["bytes"] == sum(len(b) for b in denv.gcs.objects.values())
    assert est["fits"] is True and est["disk_free_bytes"] > 0


def test_download_and_verify(denv):
    df = _run(denv)
    assert set(df["state"]) == {"VERIFIED"}
    for _, row in df.iterrows():
        paths = row["local_paths"].split(";")
        assert paths == [f"raw_chunks/track_{TRACK}/{row['chunk_name']}.tif"]
        assert (denv.run / paths[0]).exists()
    assert not list(denv.run.rglob("*.tmp"))
    calls = denv.gcs.n_downloads
    _run(denv)
    assert denv.gcs.n_downloads == calls


def test_single_listing_per_track_folder(make_project, monkeypatch, tmp_path):
    env = _setup(make_project, monkeypatch, tmp_path, tracks=("RO123_ASC", "RO045_DSC"))
    df = _run(env)
    assert set(df["state"]) == {"VERIFIED"} and len(df) == 4
    assert len(env.gcs.list_calls) == 2                       # estimate + download share one listing per track
    assert all(p.endswith("/") for p in env.gcs.list_calls)  # folder listings, not per chunk
    assert len(set(env.gcs.list_calls)) == 2


def test_transient_crc_failure_retried(denv):
    name = denv.rows[0]["gcs_prefix"] + ".tif"
    denv.gcs.fail_times[name] = 2
    assert set(_run(denv)["state"]) == {"VERIFIED"}


def test_persistent_crc_failure_marks_failed_download(denv):
    name = denv.rows[0]["gcs_prefix"] + ".tif"
    denv.gcs.fail_times[name] = 99
    df = _run(denv).set_index("task_key")
    key = denv.rows[0]["task_key"]
    assert df.loc[key, "state"] == "FAILED_DOWNLOAD" and "Checksum" in df.loc[key, "last_error"]
    assert df.loc[key, "error_class"] == "io"
    assert denv.gcs.fail_times[name] == 99 - download.IO_ATTEMPTS
    report = pd.read_csv(denv.run / export.FAILED_REPORT, keep_default_na=False)
    assert list(report["task_key"]) == [key]


def test_misaligned_file_one_redownload_then_failed_verify(denv):
    ch = denv.gd["chunks"][1]
    bad = write_tif(denv.src / "bad.tif", ch["xmin"] + 5, ch["ymax"], ch["width"], ch["height"])
    name = denv.rows[1]["gcs_prefix"] + ".tif"
    denv.gcs.put(name, bad)
    df = _run(denv).set_index("task_key")
    row = df.loc[denv.rows[1]["task_key"]]
    assert row["state"] == "FAILED_DOWNLOAD" and row["error_class"] == "verify" and "snapped" in row["last_error"]
    assert denv.gcs.download_calls[name] == 2
    assert not (denv.run / f"raw_chunks/track_{TRACK}/chunk_r00c01.tif").exists()
    assert df.loc[denv.rows[0]["task_key"], "state"] == "VERIFIED"


def test_adopted_row_failing_verification_is_requeued(denv):
    ch = denv.gd["chunks"][1]
    bad = write_tif(denv.src / "bad.tif", ch["xmin"], ch["ymax"], ch["width"], ch["height"], bands=3)
    denv.gcs.put(denv.rows[1]["gcs_prefix"] + ".tif", bad)
    mf.update_manifest(denv.run, [dict(task_key=denv.rows[1]["task_key"], adopted=1)])
    row = _run(denv).set_index("task_key").loc[denv.rows[1]["task_key"]]
    assert row["state"] == "PENDING" and int(row["adopted"]) == download.FRESH_EXPORT
    assert row["error_class"] == "verify" and int(row["attempts"]) == 2
    assert not (denv.run / "failed_chunks.csv").exists() or denv.rows[1]["task_key"] not in (
        denv.run / "failed_chunks.csv").read_text()


def test_all_nodata_file_becomes_verified_empty(denv):
    ch = denv.gd["chunks"][0]
    empty = write_tif(denv.src / "empty.tif", ch["xmin"], ch["ymax"], 64, 64, fill=NODATA)
    denv.gcs.put(denv.rows[0]["gcs_prefix"] + ".tif", empty)
    df = _run(denv).set_index("task_key")
    assert df.loc[denv.rows[0]["task_key"], "state"] == "VERIFIED_EMPTY"
    assert denv.gcs.download_calls[denv.rows[0]["gcs_prefix"] + ".tif"] == 1
    calls = denv.gcs.n_downloads
    _run(denv)                                    # accepted: skipped on re-run
    assert denv.gcs.n_downloads == calls and len(denv.gcs.list_calls) == 1


def test_dtype_mismatch_fails_verify(denv):
    ch = denv.gd["chunks"][0]
    bad = write_tif(denv.src / "i16.tif", ch["xmin"], ch["ymax"], 64, 64, dtype="int16", fill=-1250, nodata=NODATA)
    denv.gcs.put(denv.rows[0]["gcs_prefix"] + ".tif", bad)
    row = _run(denv).set_index("task_key").loc[denv.rows[0]["task_key"]]
    assert row["state"] == "FAILED_DOWNLOAD" and "dtype" in row["last_error"]


def test_missing_files_fail(denv):
    denv.gcs.objects.pop(denv.rows[0]["gcs_prefix"] + ".tif")
    df = _run(denv).set_index("task_key")
    assert df.loc[denv.rows[0]["task_key"], "state"] == "FAILED_DOWNLOAD"
    assert "no files" in df.loc[denv.rows[0]["task_key"], "last_error"]


def test_multipart_download(denv):
    ch = denv.gd["chunks"][0]
    prefix = denv.rows[0]["gcs_prefix"]
    denv.gcs.objects.pop(prefix + ".tif")
    a = write_tif(denv.src / "p1.tif", ch["xmin"], ch["ymax"], 64, 32)
    b = write_tif(denv.src / "p2.tif", ch["xmin"], ch["ymax"] - 320, 64, 32)
    denv.gcs.put(prefix + "-0000000000-0000000000.tif", a)
    denv.gcs.put(prefix + "-0000000032-0000000000.tif", b)
    row = _run(denv).set_index("task_key").loc[denv.rows[0]["task_key"]]
    assert row["state"] == "VERIFIED" and len(row["local_paths"].split(";")) == 2


def test_refuses_when_disk_too_small(denv, monkeypatch):
    monkeypatch.setattr(resources, "disk_free_bytes", lambda p: 10)
    with pytest.raises(PipelineError, match="disk space"):
        _run(denv)
    assert set(mf.read_manifest(denv.run)["state"]) == {"COMPLETED"}


def test_retry_failed(denv):
    name = denv.rows[0]["gcs_prefix"] + ".tif"
    denv.gcs.fail_times[name] = 99
    _run(denv)
    denv.gcs.fail_times[name] = 0
    assert _states(denv.run).loc[denv.rows[0]["task_key"], "state"] == "FAILED_DOWNLOAD"
    _run(denv)                                          # without retry_failed: untouched
    assert _states(denv.run).loc[denv.rows[0]["task_key"], "state"] == "FAILED_DOWNLOAD"
    _run(denv, retry_failed=True)
    assert _states(denv.run).loc[denv.rows[0]["task_key"], "state"] == "VERIFIED"


def test_manifest_updates_are_batched(denv, monkeypatch):
    calls = []
    real = mf.update_manifest
    monkeypatch.setattr(mf, "update_manifest",
                        lambda run_dir, rows, **kw: calls.append((len(rows), kw.get("return_df"))) or real(run_dir, rows, **kw))
    _run(denv)
    assert calls == [(2, False)]                         # one batch for both chunks, no DataFrame rebuilt


# ---------------------------------------------------------------- locked files (Windows / WSL)
class _FakeBlob:
    def __init__(self, data):
        self.data = data

    def download_to_filename(self, path, checksum=None):
        Path(path).write_bytes(self.data)


class _FakeBucket:
    def __init__(self, objects):
        self.objects = objects

    def blob(self, name):
        return _FakeBlob(self.objects[name])


def test_locked_rename_is_retried_without_consuming_attempts(denv, monkeypatch):
    backend = download.GCSBackend(bucket=_FakeBucket(denv.gcs.objects))
    backend.list_blobs = denv.gcs.list_blobs
    real_replace = os.replace
    failures = {"n": 2}

    def flaky(src, dst):
        if str(dst).endswith(".tif") and failures["n"] > 0:
            failures["n"] -= 1
            raise PermissionError("[WinError 32] The process cannot access the file")
        return real_replace(src, dst)

    monkeypatch.setattr(run_meta.os, "replace", flaky)
    monkeypatch.setattr(run_meta.time, "sleep", lambda s: None)
    df = download.download_completed(denv.cfg, denv.run, backend, confirmed=True)
    assert set(df["state"]) == {"VERIFIED"} and failures["n"] == 0
    assert set(df["error_class"]) == {""}
    assert not list(denv.run.rglob("*.tmp"))


def test_permanently_locked_file_leaves_row_for_next_run(denv, monkeypatch):
    backend = download.GCSBackend(bucket=_FakeBucket(denv.gcs.objects))
    backend.list_blobs = denv.gcs.list_blobs
    real_replace = os.replace
    locked = denv.rows[0]["chunk_name"] + ".tif"

    def replace(src, dst):
        if str(dst).endswith(locked):
            raise PermissionError("locked")
        return real_replace(src, dst)

    monkeypatch.setattr(run_meta.os, "replace", replace)
    monkeypatch.setattr(run_meta.time, "sleep", lambda s: None)
    df = download.download_completed(denv.cfg, denv.run, backend, confirmed=True).set_index("task_key")
    assert df.loc[denv.rows[0]["task_key"], "state"] == "COMPLETED"   # untouched, no attempt consumed
    assert df.loc[denv.rows[0]["task_key"], "attempts"] == 1
    assert df.loc[denv.rows[1]["task_key"], "state"] == "VERIFIED"    # the pool kept working


# ---------------------------------------------------------------- resume, never redo
def test_crc32c_matches_gcs_format(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    assert download.file_crc32c_b64(p) == download.bytes_crc32c_b64(b"hello world") == "yZRlqg=="


def test_rerun_downloads_nothing_already_verified(denv, caplog):
    _run(denv)
    index = download.read_index(denv.run)
    assert len(index) == 2 and all(f["crc32c"] and f["mtime_ns"] for e in index.values() for f in e["files"])
    calls, lists = denv.gcs.n_downloads, len(denv.gcs.list_calls)
    with caplog.at_level(logging.INFO, logger="sar_pipeline.download"):
        df = _run(denv)
    assert set(df["state"]) == {"VERIFIED"}
    assert denv.gcs.n_downloads == calls and len(denv.gcs.list_calls) == lists   # no network at all
    assert "skipped 2/2" in caplog.text
    assert download.estimate_download(denv.cfg, denv.run, denv.gcs)["bytes"] == 0


def test_rerun_does_not_rehash_unchanged_files(denv, monkeypatch):
    _run(denv)
    hashed = []
    real = download.file_crc32c_b64
    monkeypatch.setattr(download, "file_crc32c_b64", lambda p: hashed.append(Path(p).name) or real(p))
    _run(denv)
    assert hashed == []                                            # size + mtime unchanged: trusted
    local = denv.run / f"raw_chunks/track_{TRACK}/chunk_r00c00.tif"
    st = local.stat()
    os.utime(local, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))  # touched, same bytes
    _run(denv)
    assert hashed == ["chunk_r00c00.tif"]                          # one re-hash, still accepted
    assert denv.gcs.n_downloads == 2
    hashed.clear()
    _run(denv)
    assert hashed == []                                            # new mtime recorded
    _run(denv, deep=True)
    assert sorted(hashed) == ["chunk_r00c00.tif", "chunk_r00c01.tif"]


def test_truncated_local_file_is_redownloaded(denv):
    _run(denv)
    local = denv.run / f"raw_chunks/track_{TRACK}/chunk_r00c00.tif"
    data = local.read_bytes()
    local.write_bytes(data[: len(data) // 2])
    calls = denv.gcs.n_downloads
    df = _run(denv).set_index("task_key")
    assert denv.gcs.n_downloads == calls + 1
    assert df.loc[denv.rows[0]["task_key"], "state"] == "VERIFIED"
    assert local.read_bytes() == data


def test_files_downloaded_before_crash_are_reused(denv):
    for row in denv.rows:
        name = row["gcs_prefix"] + ".tif"
        dest = denv.run / f"raw_chunks/track_{TRACK}" / name.rsplit("/", 1)[-1]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(denv.gcs.objects[name])            # downloaded, but state never updated
    df = _run(denv)
    assert set(df["state"]) == {"VERIFIED"} and denv.gcs.n_downloads == 0


def test_stale_temp_files_removed_but_young_ones_kept(denv):
    folder = denv.run / f"raw_chunks/track_{TRACK}"
    folder.mkdir(parents=True, exist_ok=True)
    stale, young = folder / ".chunk_r00c00.tif.old.tmp", folder / ".chunk_r00c01.tif.live.tmp"
    stale.write_bytes(b"partial")
    young.write_bytes(b"another process is writing")
    old = time.time() - run_meta.STALE_TMP_SECONDS - 60
    os.utime(stale, (old, old))
    _run(denv)
    assert not stale.exists() and young.exists()


def test_force_redownloads(denv):
    _run(denv)
    calls = denv.gcs.n_downloads
    _run(denv, force=True)
    assert denv.gcs.n_downloads == calls + 2


def test_legacy_csv_index_is_read_and_compacted(denv):
    _run(denv)
    index = download.read_index(denv.run)
    legacy = pd.DataFrame([dict(task_key=k, local_path=f["local_path"], size=f["size"], crc32c=f["crc32c"],
                                verified_at=e["verified_at"]) for k, e in index.items() for f in e["files"]])
    (denv.run / download.INDEX).unlink()
    legacy.to_csv(denv.run / download.LEGACY_INDEX, index=False)
    calls = denv.gcs.n_downloads
    _run(denv)
    assert denv.gcs.n_downloads == calls                        # legacy entries honoured (one re-hash)
    assert not (denv.run / download.LEGACY_INDEX).exists()
    assert set(download.read_index(denv.run)) == set(index)
    assert all(f["mtime_ns"] for e in download.read_index(denv.run).values() for f in e["files"])


def test_torn_index_line_is_ignored(denv):
    _run(denv)
    with open(denv.run / download.INDEX, "a") as f:
        f.write('{"task_key": "RO123_ASC__chunk_r00c00", "files": [{"local_pa')   # crash mid-append
    assert len(download.read_index(denv.run)) == 2
    assert set(_run(denv)["state"]) == {"VERIFIED"}


def test_index_lookup_scales(tmp_path, monkeypatch):
    run = tmp_path / "run"
    folder = run / "raw_chunks" / "track_T"
    folder.mkdir(parents=True)
    entries, rows = [], []
    for i in range(10_000):
        p = folder / f"c{i}.tif"
        p.write_bytes(b"x")
        key = f"T__c{i}"
        entries.append(dict(task_key=key, state="VERIFIED", verified_at="t", files=[
            dict(local_path=f"raw_chunks/track_T/c{i}.tif", size=1, mtime_ns=p.stat().st_mtime_ns, crc32c="abc")]))
        rows.append(dict(task_key=key, track_id="T", chunk_name=f"c{i}", state="VERIFIED", gcs_prefix=f"g/{key}"))
    download._append_index(run, entries)
    manifest = pd.DataFrame(rows)
    monkeypatch.setattr(mf, "read_manifest", lambda r: manifest)
    hashed = []
    monkeypatch.setattr(download, "file_crc32c_b64", lambda p: hashed.append(p) or "abc")

    class NoList:
        def list_blobs(self, prefix):
            raise AssertionError("no listing expected")

    t0 = time.perf_counter()
    est = download.estimate_download({}, run, NoList())
    assert time.perf_counter() - t0 < 15
    assert est["n_skipped_verified"] == 10_000 and est["n_files"] == 0 and hashed == []


# ---------------------------------------------------------------- one download process per run
def test_second_download_process_is_blocked(denv):
    code = (
        "import sys, time; sys.path.insert(0, sys.argv[1]);"
        "from sar_pipeline.locking import run_lock;"
        "lock = run_lock(sys.argv[2], 'download'); print('locked', flush=True); time.sleep(30)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(REPO / "src"), str(denv.run)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(PipelineError, match="Another 'download' process"):
            _run(denv)
        assert set(mf.read_manifest(denv.run)["state"]) == {"COMPLETED"}
    finally:
        proc.kill()
        proc.wait()


# ---------------------------------------------------------------- shared file-name and CRS rules
def test_part_names_follow_the_shared_run_meta_rule():
    prefix = "base/track_T/chunk_r00c00"
    names = [prefix + ".tif", prefix + "-0000000000-0000000256.tif", prefix + "-12-34.tif",
             prefix + "_s1.tif", prefix + ".tif.tmp"]
    blobs = [{"name": n, "size": 1, "crc32c": ""} for n in names]
    got = {b["name"] for b in download.chunk_blob_names(blobs, prefix)}
    rule = run_meta.ee_part_regex(prefix)
    assert got == {n for n in names if rule.match(n)} == {prefix + ".tif", prefix + "-0000000000-0000000256.tif"}


def test_equivalent_crs_definition_is_accepted(tmp_path, gd):
    from rasterio.crs import CRS

    ch = gd["chunks"][0]
    p = write_tif(tmp_path / "wkt.tif", ch["xmin"], ch["ymax"], 64, 64, crs=CRS.from_epsg(6933).to_wkt())
    assert download.inspect_chunk_files([p], gd, ch, 4, NODATA, "float32") == ("ok", "ok")
    assert run_meta.same_crs(CRS.from_epsg(6933), gd["crs"])


# ---------------------------------------------------------------- M3: keep verified local data
def test_verified_rows_restored_from_local_files_when_gcs_and_index_are_gone(denv):
    _run(denv)
    (denv.run / download.INDEX).unlink()                  # index lost (e.g. not copied with the run)
    denv.gcs.objects.clear()                              # objects deleted by a lifecycle rule
    est = download.estimate_download(denv.cfg, denv.run, denv.gcs)
    assert est["n_skipped_verified"] == 2 and est["tasks_without_files"] == []
    calls = denv.gcs.n_downloads
    df = _run(denv)
    assert set(df["state"]) == {"VERIFIED"} and denv.gcs.n_downloads == calls
    index = download.read_index(denv.run)
    assert set(index) == {r["task_key"] for r in denv.rows}
    assert all(f["crc32c"] and f["mtime_ns"] for e in index.values() for f in e["files"])
    hashed_again = download.estimate_download(denv.cfg, denv.run, denv.gcs)
    assert hashed_again["n_skipped_verified"] == 2        # rebuilt index is trusted on the next call


def test_restore_refuses_bad_local_files(denv):
    _run(denv)
    (denv.run / download.INDEX).unlink()
    denv.gcs.objects.clear()
    local = denv.run / f"raw_chunks/track_{TRACK}/chunk_r00c00.tif"
    local.write_bytes(local.read_bytes()[:200])           # corrupt
    df = _run(denv).set_index("task_key")
    assert df.loc[denv.rows[0]["task_key"], "state"] == "FAILED_DOWNLOAD"
    assert df.loc[denv.rows[1]["task_key"], "state"] == "VERIFIED"


# ---------------------------------------------------------------- M4: fresh exports and old GCS output
def _old(days=2):
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def test_fresh_export_row_ignores_parts_of_the_previous_export(denv, caplog):
    ch = denv.gd["chunks"][0]
    prefix = denv.rows[0]["gcs_prefix"]
    key = denv.rows[0]["task_key"]
    # leftovers of an earlier (bad) export: two part files that would break the pixel count
    a = write_tif(denv.src / "old1.tif", ch["xmin"], ch["ymax"], 64, 32)
    b = write_tif(denv.src / "old2.tif", ch["xmin"], ch["ymax"] - 320, 64, 32, bands=3)
    denv.gcs.put(prefix + "-0000000000-0000000000.tif", a, created=_old())
    denv.gcs.put(prefix + "-0000000032-0000000000.tif", b, created=_old())
    mf.update_manifest(denv.run, [dict(task_key=key, adopted=download.FRESH_EXPORT, submitted_at=_old(1))])
    with caplog.at_level(logging.INFO, logger="sar_pipeline.download"):
        row = _run(denv).set_index("task_key").loc[key]
    assert row["state"] == "VERIFIED" and int(row["adopted"]) == 0
    assert row["local_paths"] == f"raw_chunks/track_{TRACK}/chunk_r00c00.tif"
    assert "ignoring 2 GCS file(s)" in caplog.text


def test_row_with_files_older_than_its_submission_is_requeued_as_fresh_export(denv):
    ch = denv.gd["chunks"][1]
    key = denv.rows[1]["task_key"]
    bad = write_tif(denv.src / "stale.tif", ch["xmin"], ch["ymax"], ch["width"], ch["height"], bands=3)
    denv.gcs.put(denv.rows[1]["gcs_prefix"] + ".tif", bad, created=_old(3))
    mf.update_manifest(denv.run, [dict(task_key=key, submitted_at=_old(1))])
    row = _run(denv).set_index("task_key").loc[key]
    assert row["state"] == "PENDING" and int(row["adopted"]) == download.FRESH_EXPORT
    assert row["error_class"] == "verify" and int(row["attempts"]) == 2


def test_fresh_export_failing_again_does_not_loop(denv):
    ch = denv.gd["chunks"][1]
    key = denv.rows[1]["task_key"]
    bad = write_tif(denv.src / "new_bad.tif", ch["xmin"], ch["ymax"], ch["width"], ch["height"], bands=3)
    denv.gcs.put(denv.rows[1]["gcs_prefix"] + ".tif", bad)             # created now, after submission
    mf.update_manifest(denv.run, [dict(task_key=key, adopted=download.FRESH_EXPORT, submitted_at=_old(1))])
    row = _run(denv).set_index("task_key").loc[key]
    assert row["state"] == "FAILED_DOWNLOAD" and row["error_class"] == "verify"


def test_clock_skew_within_tolerance_does_not_hide_fresh_output(denv):
    from datetime import datetime, timedelta, timezone

    key = denv.rows[0]["task_key"]
    ahead = (datetime.now(timezone.utc) + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    mf.update_manifest(denv.run, [dict(task_key=key, adopted=download.FRESH_EXPORT, submitted_at=ahead)])
    row = _run(denv).set_index("task_key").loc[key]
    assert row["state"] == "VERIFIED"


# ---------------------------------------------------------------- lock held longer than stale_after
def test_long_running_download_lock_is_not_taken_over(denv):
    code = (
        "import sys, time; sys.path.insert(0, sys.argv[1]);"
        "from sar_pipeline.locking import run_lock;"
        "lock = run_lock(sys.argv[2], 'download'); print('locked', flush=True); time.sleep(30)"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(REPO / "src"), str(denv.run)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        lock_file = denv.run / ".download.lock"
        ancient = time.time() - 24 * 3600                  # far beyond stale_after (15 min)
        os.utime(lock_file, (ancient, ancient))
        with pytest.raises(PipelineError, match="Another 'download' process"):
            _run(denv)
        assert set(mf.read_manifest(denv.run)["state"]) == {"COMPLETED"}
    finally:
        proc.kill()
        proc.wait()
