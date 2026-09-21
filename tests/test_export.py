"""Unit tests for export planning, confirmation, submission throttling, the retry state machine and reports.

Earth Engine is replaced by FakeBackend; audit and s1_ard are replaced by tiny fake modules, so these
tests need no network and do not depend on those modules being finished.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import textwrap
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import sar_pipeline
from sar_pipeline import config, export, grid, run_meta
from sar_pipeline import manifest as mf
from sar_pipeline.errors import ConfirmationRequired, PipelineError
from sar_pipeline.locking import run_lock

RES = 10.0
X0, Y0 = 300000.0, 1900000.0
SRC = str(Path(__file__).resolve().parents[1] / "src")


def make_gd(width=600, height=300, chunk_px=256):
    chunks = []
    for r in range((height + chunk_px - 1) // chunk_px):
        for c in range((width + chunk_px - 1) // chunk_px):
            ro, co = r * chunk_px, c * chunk_px
            h, w = min(chunk_px, height - ro), min(chunk_px, width - co)
            xmin, ymax = X0 + co * RES, Y0 - ro * RES
            chunks.append(dict(chunk_id=len(chunks) + 1, name=f"chunk_r{r:02d}c{c:02d}", grid_row=r, grid_col=c,
                               row_off=ro, col_off=co, width=w, height=h, xmin=xmin, ymin=ymax - h * RES,
                               xmax=xmin + w * RES, ymax=ymax, aoi_frac=1.0))
    return dict(crs="EPSG:6933", res=RES, x0=X0, y0=Y0, width=width, height=height, chunk_px=chunk_px,
                n_chunk_rows=2, n_chunk_cols=3, transform=[RES, 0.0, X0, 0.0, -RES, Y0], pid_dtype="int32",
                aoi_file="x", chunks=chunks)


def make_acq(track, n, start="2026-04-03"):
    t0 = datetime.fromisoformat(start).replace(hour=10, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        t = t0 + timedelta(days=12 * i)
        rows.append(dict(acquisition_id=f"{track}_{t:%Y%m%d}", track_id=track,
                         datetime_utc=t.strftime("%Y-%m-%dT%H:%M:%SZ"), date_utc=t.strftime("%Y-%m-%d"),
                         date_local=t.strftime("%Y-%m-%d"), platforms="C", n_slices=1, system_indexes=f"S{i}"))
    return pd.DataFrame(rows)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


class FakeBackend:
    def __init__(self, ready=0, running=0):
        self.counts = {"READY": ready, "RUNNING": running}
        self.count_calls = 0
        self.status_calls = 0
        self.started: list[dict] = []
        self.start_errors: list[str] = []   # raised in order by start_export
        self.statuses: dict[str, dict] = {}  # answered by the getTaskStatus fallback only
        self.tasks: list[dict] = []          # project-wide tasks visible in the listing
        self.snapshot_errors: list[str] = []
        self._n = 0

    def start_export(self, image, params):
        if self.start_errors:
            raise RuntimeError(self.start_errors.pop(0))
        self._n += 1
        tid = f"TASK{self._n:04d}"
        self.started.append(dict(task_id=tid, image=image, params=params))
        return tid

    def task_statuses(self, ids):
        self.status_calls += 1
        return {i: self.statuses[i] for i in ids if i in self.statuses}

    def project_snapshot(self):
        if self.snapshot_errors:
            raise RuntimeError(self.snapshot_errors.pop(0))
        self.count_calls += 1
        return {"counts": dict(self.counts), "tasks": list(self.tasks)}


@pytest.fixture
def env(make_project, monkeypatch):
    """Synthetic project + run folder with grid/audit/s1_ard faked."""
    cfg = make_project({"export": {"max_queued_tasks": 3000, "queue_safety_margin": 0.9,
                                   "max_active_tasks": "auto", "poll_seconds": 0}})
    gd = make_gd()
    state = {"selected": {"RO123_ASC": make_acq("RO123_ASC", 3), "RO201_DSC": make_acq("RO201_DSC", 2)},
             "build_error": None}

    monkeypatch.setattr(grid, "load_grid", lambda c: gd)
    fake_audit = types.ModuleType("sar_pipeline.audit")
    fake_audit.selected_acquisitions = lambda c, audit_path=None: state["selected"]
    monkeypatch.setitem(sys.modules, "sar_pipeline.audit", fake_audit)
    monkeypatch.setattr(sar_pipeline, "audit", fake_audit, raising=False)

    def build(c, g, ch, tr, acq):
        if state["build_error"]:
            raise ValueError(state["build_error"])
        return ("IMAGE", ch["name"], tr, len(acq))

    fake_pipe = types.ModuleType("sar_pipeline.s1_ard.pipeline")
    fake_pipe.build_chunk_image = build
    fake_pkg = types.ModuleType("sar_pipeline.s1_ard")
    fake_pkg.pipeline = fake_pipe
    monkeypatch.setitem(sys.modules, "sar_pipeline.s1_ard", fake_pkg)
    monkeypatch.setitem(sys.modules, "sar_pipeline.s1_ard.pipeline", fake_pipe)
    monkeypatch.setattr(sar_pipeline, "s1_ard", fake_pkg, raising=False)

    config.grid_dir(cfg).mkdir(parents=True, exist_ok=True)
    (config.grid_dir(cfg) / "grid_def.json").write_text(json.dumps(gd))
    config.audit_dir(cfg, "20260915").mkdir(parents=True)
    run = config.new_run(cfg)
    return types.SimpleNamespace(cfg=cfg, gd=gd, run=run, state=state, uid=run_meta.run_uid(run))


def states(run):
    df = mf.read_manifest(run)
    return dict(zip(df["task_key"], df["state"]))


def plan_confirm(env, chunk_names=None):
    export.plan_exports(env.cfg, env.run, chunk_names=chunk_names)
    return export.confirm_exports(env.cfg, env.run, chunk_names=chunk_names)


def move_clock(monkeypatch, minutes):
    monkeypatch.setattr(export, "_utcnow", lambda: datetime.now(timezone.utc) + timedelta(minutes=minutes))


# ---------------------------------------------------------------- planning
def test_plan_exports_layout_meta_and_rows(env):
    df = export.plan_exports(env.cfg, env.run)
    n_chunks = len(env.gd["chunks"])
    assert len(df) == 2 * n_chunks and set(df["state"]) == {"PLANNED"}
    layout = run_meta.read_band_layout(env.run)
    asc = layout[layout["track_id"] == "RO123_ASC"]
    assert list(asc["band_idx"]) == [1, 2, 3, 4, 5, 6]
    assert list(asc["pol"]) == ["VV", "VH"] * 3
    assert int(df[df["track_id"] == "RO123_ASC"]["n_bands"].iloc[0]) == 6
    assert int(df[df["track_id"] == "RO201_DSC"]["n_bands"].iloc[0]) == 4
    raw = json.loads((env.run / export.EXPORT_META).read_text())
    assert list(raw) == run_meta.EXPORT_META_KEYS + run_meta.OPTIONAL_META_KEYS
    meta = run_meta.read_export_meta(env.cfg, env.run)
    assert meta["grid_transform"] == env.gd["transform"] and meta["nodata"] == -9999 and meta["dtype"] == "float32"
    assert meta["grid_crs"] == "EPSG:6933" and meta["int16_scale"] == 100 and meta["run_id"] == env.run.name
    assert meta["grid_fingerprint"] == hashlib.sha256((config.grid_dir(env.cfg) / "grid_def.json").read_bytes()).hexdigest()
    assert meta["audit_dir"] == "processed/test_aoi/test_season/audit/20260915"
    assert meta["tracks"] == ["RO123_ASC", "RO201_DSC"]
    assert meta["ee_project"] == "fake-project" and meta["run_uid"] == env.uid != env.run.name
    assert list(layout.columns) == export.BAND_LAYOUT_COLUMNS == run_meta.BAND_LAYOUT_COLUMNS
    row = df[df["task_key"] == "RO123_ASC__chunk_r00c01"].iloc[0]
    assert row["gcs_prefix"] == f"fake_base/test_aoi/test_season/runs/{env.uid}/raw_chunks/track_RO123_ASC/chunk_r00c01"
    assert row["description"] == f"test_aoi_test_season_{env.uid}_RO123_ASC__chunk_r00c01"


def test_run_uid_makes_prefix_and_description_unique_per_run(env):
    export.plan_exports(env.cfg, env.run, chunk_names=["chunk_r00c00"])
    run2 = config.new_run(env.cfg)
    export.plan_exports(env.cfg, run2, chunk_names=["chunk_r00c00"])
    a = mf.read_manifest(env.run).set_index("task_key").loc["RO123_ASC__chunk_r00c00"]
    b = mf.read_manifest(run2).set_index("task_key").loc["RO123_ASC__chunk_r00c00"]
    assert a["gcs_prefix"] != b["gcs_prefix"] and a["description"] != b["description"]
    assert run_meta.run_uid(run2) in b["gcs_prefix"] and run_meta.run_uid(run2) in b["description"]


def test_plan_exports_idempotent_and_pilot_then_all(env):
    df = export.plan_exports(env.cfg, env.run, chunk_names=["chunk_r00c00"])
    assert len(df) == 2
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="SUBMITTED", task_id="X")])
    df = export.plan_exports(env.cfg, env.run)
    assert len(df) == 2 * len(env.gd["chunks"]) and df["task_key"].is_unique
    assert states(env.run)["RO123_ASC__chunk_r00c00"] == "SUBMITTED"   # not reset
    assert len(export.plan_exports(env.cfg, env.run)) == len(df)


def test_plan_exports_rejects_changed_acquisitions(env):
    export.plan_exports(env.cfg, env.run)
    env.state["selected"]["RO123_ASC"] = make_acq("RO123_ASC", 4)
    with pytest.raises(PipelineError):
        export.plan_exports(env.cfg, env.run)


def test_plan_exports_unknown_chunk(env):
    with pytest.raises(PipelineError):
        export.plan_exports(env.cfg, env.run, chunk_names=["chunk_r09c09"])


def test_not_covered_chunks_are_planned_but_never_exported(env):
    audit = config.audit_dir(env.cfg, "20260915")
    rows = [dict(chunk_name=ch["name"], track_id=t, n_acq=3, n_acq_full_cover=3, mean_cover_pct=100.0)
            for ch in env.gd["chunks"] for t in ("RO123_ASC", "RO201_DSC")
            if not (t == "RO123_ASC" and ch["name"] == "chunk_r01c02")]
    pd.DataFrame(rows).to_csv(audit / "chunk_track_coverage.csv", index=False)
    export.plan_exports(env.cfg, env.run)
    assert states(env.run)["RO123_ASC__chunk_r01c02"] == "NOT_COVERED"
    export.confirm_exports(env.cfg, env.run)
    be = AutoBackend()
    export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None)
    s = states(env.run)
    assert s["RO123_ASC__chunk_r01c02"] == "NOT_COVERED"
    assert list(s.values()).count("COMPLETED") == 2 * len(env.gd["chunks"]) - 1
    assert not any(t["params"]["fileNamePrefix"].endswith("track_RO123_ASC/chunk_r01c02") for t in be.started)
    assert len(be.started) == 2 * len(env.gd["chunks"]) - 1


# ---------------------------------------------------------------- confirmation of scope (C2)
def test_dry_run_plan_is_never_submitted_by_monitor(env):
    export.plan_exports(env.cfg, env.run)                   # e.g. `export --all` without --yes
    be = AutoBackend()
    export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None)
    assert be.started == [] and set(states(env.run).values()) == {"PLANNED"}


def test_pilot_confirmation_does_not_release_a_later_full_plan(env):
    plan_confirm(env, ["chunk_r00c00"])
    export.plan_exports(env.cfg, env.run)                   # full plan, not confirmed
    be = AutoBackend()
    export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None)
    assert sorted(t["params"]["fileNamePrefix"].rsplit("/", 2)[-1] for t in be.started) == ["chunk_r00c00"] * 2
    s = states(env.run)
    assert s["RO123_ASC__chunk_r00c00"] == "COMPLETED" and s["RO123_ASC__chunk_r00c01"] == "PLANNED"


def test_confirm_all_promotes_only_planned(env):
    export.plan_exports(env.cfg, env.run)
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="VERIFIED")])
    export.confirm_exports(env.cfg, env.run)
    s = states(env.run)
    assert s.pop("RO123_ASC__chunk_r00c00") == "VERIFIED" and set(s.values()) == {"PENDING"}


# ---------------------------------------------------------------- export params
def test_export_params_alignment(env):
    for ch in env.gd["chunks"]:
        p = export.export_params(env.cfg, env.gd, ch, f"RO123_ASC__{ch['name']}", env.uid)
        assert p["crsTransform"] == env.gd["transform"]          # master transform, not per chunk
        assert p["crs"] == "EPSG:6933" and p["formatOptions"] == {"noData": -9999}
        xmin, ymin, xmax, ymax = p["region"]
        assert ch["xmin"] < xmin < ch["xmin"] + RES / 2 and ch["xmax"] - RES / 2 < xmax < ch["xmax"]
        assert ch["ymin"] < ymin < ch["ymin"] + RES / 2 and ch["ymax"] - RES / 2 < ymax < ch["ymax"]
        assert p["fileNamePrefix"].endswith(f"{env.uid}/raw_chunks/track_RO123_ASC/{ch['name']}")
        assert p["maxPixels"] == 1e13 and p["fileFormat"] == "GeoTIFF"


def test_export_params_int16_nodata(env):
    env.cfg["export"]["dtype"] = "int16"
    ch = env.gd["chunks"][0]
    assert export.export_params(env.cfg, env.gd, ch, "RO123_ASC__chunk_r00c00", "v001")["formatOptions"] == {"noData": -32768}


def test_description_sanitised_and_unique():
    a = export.sanitise_description("v001_" + "RO123_ASC__chunk_r00c00_s1/" + "x" * 200)
    b = export.sanitise_description("v001_" + "RO123_ASC__chunk_r00c00_s2/" + "x" * 200)
    assert len(a) <= 100 and len(b) <= 100 and a != b
    assert "/" not in export.sanitise_description("a/b")


@pytest.mark.parametrize("msg,cls", [
    ("User memory limit exceeded.", "memory"),
    ("Computation timed out.", "timeout"),
    ("Deadline exceeded", "transient"),
    ("Internal error.", "transient"),
    ("Service unavailable: backend error", "transient"),
    ("HttpError 503 when requesting ...", "transient"),
    ("Read timed out. (read timeout=60)", "transient"),
    ("Too many tasks already in the queue (3000).", "quota"),
    ("Quota exceeded for quota metric", "quota"),
    ("HTTP 429 rate limit", "quota"),
    ("Cancelled by user.", "cancelled"),
    ("Image.select: band not found", "other"),
    ("", "other"),
])
def test_classify_error(msg, cls):
    assert export.classify_error(msg) == cls


# ---------------------------------------------------------------- confirmation gates
def test_confirmation_required(env):
    plan_confirm(env)
    with pytest.raises(ConfirmationRequired):
        export.submit_pending(env.cfg, env.run, FakeBackend())
    with pytest.raises(ConfirmationRequired):
        export.monitor(env.cfg, env.run, FakeBackend())


# ---------------------------------------------------------------- throttling
def _submit(env, backend, **kw):
    return export.submit_pending(env.cfg, env.run, backend, confirmed=True, sleep=lambda s: None, **kw)


def test_no_submission_when_project_queue_full(env, caplog):
    plan_confirm(env)
    be = FakeBackend(ready=2900, running=50)            # 2950 > floor(3000*0.9)=2700
    with caplog.at_level(logging.INFO, logger="sar_pipeline.export"):
        _submit(env, be)
    assert be.started == [] and set(states(env.run).values()) == {"PENDING"}
    assert export.LAST_ROUND["headroom"] == 0
    assert "queue full" in caplog.text


def test_partial_headroom_fills_exactly(env):
    plan_confirm(env)                                   # 2 tracks x 6 chunks = 12 pending
    be = FakeBackend(ready=2680, running=15)            # headroom 2700-2695 = 5
    _submit(env, be)
    assert len(be.started) == 5
    assert list(states(env.run).values()).count("SUBMITTED") == 5
    assert export.LAST_ROUND == dict(queued=2680, running=15, submitted=5, headroom=5, pending_left=7, eta_seconds=None)


def test_auto_cap_submits_all_when_headroom_large(env):
    plan_confirm(env)
    be = FakeBackend(ready=0, running=12)
    _submit(env, be)
    assert len(be.started) == 12


def test_int_own_cap_respected(env):
    env.cfg["export"]["max_active_tasks"] = 3
    plan_confirm(env)
    be = FakeBackend()
    _submit(env, be)
    assert len(be.started) == 3
    _submit(env, be)                                     # own rows already occupy the cap
    assert len(be.started) == 3
    first = mf.read_manifest(env.run).query("state == 'SUBMITTED'").iloc[0]["task_key"]
    mf.update_manifest(env.run, [dict(task_key=first, state="COMPLETED")])
    _submit(env, be)
    assert len(be.started) == 4


def test_project_count_requeried_each_round(env):
    plan_confirm(env)
    be = FakeBackend(ready=2698, running=0)             # headroom 2
    _submit(env, be)
    assert be.count_calls == 1 and len(be.started) == 2
    be.counts["READY"] = 2697                            # EE state changed between rounds
    _submit(env, be)
    assert be.count_calls == 2 and len(be.started) == 5


# ---------------------------------------------------------------- submission errors
def test_quota_submission_backoff_then_success(env):
    plan_confirm(env, ["chunk_r00c00"])
    be = FakeBackend()
    be.start_errors = ["Too many tasks in the queue", "Quota exceeded"]
    sleeps = []
    env.cfg["export"]["max_workers"] = 1                  # deterministic order: ASC row first
    export.submit_pending(env.cfg, env.run, be, confirmed=True, sleep=sleeps.append)
    df = mf.read_manifest(env.run).set_index("task_key")
    assert set(df["state"]) == {"SUBMITTED"}
    assert df.loc["RO123_ASC__chunk_r00c00", "attempts"] == 1
    assert len(sleeps) == 2 and len(be.started) == 2


def test_quota_submission_exhausted_stays_pending(env):
    env.cfg["export"].update(submit_max_attempts=3, max_workers=1)
    plan_confirm(env, ["chunk_r00c00"])
    be = FakeBackend()
    be.start_errors = ["Too many tasks"] * 100
    _submit(env, be)
    df = mf.read_manifest(env.run)
    assert set(df["state"]) == {"PENDING"} and set(df["error_class"]) == {"quota"}
    assert set(df["attempts"]) == {0}                    # deferred, not counted as a task attempt


def test_error_while_building_image_fails_row(env):
    env.cfg["export"]["max_workers"] = 1
    plan_confirm(env, ["chunk_r00c00"])
    env.state["build_error"] = "Image.select: Pattern 'VV' did not match any bands."
    be = FakeBackend()
    _submit(env, be)
    df = mf.read_manifest(env.run)
    assert set(df["state"]) == {"FAILED"} and set(df["error_class"]) == {"other"} and be.started == []


def test_ambiguous_start_error_keeps_submitting_and_adopts_without_duplicate(env):
    """start() raised after the request reached Earth Engine: the task exists, nothing is exported twice."""
    env.cfg["export"]["max_workers"] = 1
    plan_confirm(env, ["chunk_r00c00"])
    be = FakeBackend()
    be.start_errors = ["HttpError 503 backend error", "Read timed out"]
    _submit(env, be)
    df = mf.read_manifest(env.run).set_index("task_key")
    assert set(df["state"]) == {"SUBMITTING"} and be.started == []
    assert set(df["attempts"]) == {1} and set(df["error_class"]) == {"transient"}
    # the listing now shows both tasks (created after the rows went SUBMITTING)
    be.tasks = [dict(id=f"EE_{k}", description=df.loc[k, "description"], state="RUNNING",
                     create_time=iso(datetime.now(timezone.utc))) for k in df.index]
    _submit(env, be)
    df = mf.read_manifest(env.run).set_index("task_key")
    assert set(df["state"]) == {"RUNNING"} and set(df["adopted"]) == {1}
    assert be.started == []                               # no duplicate export


def test_submitting_row_reset_only_after_grace_period(env, monkeypatch):
    plan_confirm(env, ["chunk_r00c00"])
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="SUBMITTING")])
    mf.update_manifest(env.run, [dict(task_key="RO201_DSC__chunk_r00c00", state="SUBMITTED", task_id="T0")])
    be = FakeBackend()
    _submit(env, be)
    assert states(env.run)["RO123_ASC__chunk_r00c00"] == "SUBMITTING" and be.started == []
    move_clock(monkeypatch, 11)
    _submit(env, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc["RO123_ASC__chunk_r00c00"]
    assert row["state"] == "SUBMITTED" and len(be.started) == 1


def test_interrupted_submission_without_task_is_resubmitted(env, monkeypatch):
    plan_confirm(env, ["chunk_r00c00"])
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="SUBMITTING")])
    be = FakeBackend()
    be.tasks = [dict(id="OLD", description=export.task_description(env.cfg, env.uid, "RO123_ASC__chunk_r00c00"),
                     state="FAILED")]                      # failed tasks are never adopted
    move_clock(monkeypatch, 11)
    _submit(env, be)
    assert set(states(env.run).values()) == {"SUBMITTED"} and len(be.started) == 2


def test_submitting_row_with_attempts_used_up_fails(env, monkeypatch):
    env.cfg["export"]["task_max_attempts"] = 2
    plan_confirm(env, ["chunk_r00c00"])
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="SUBMITTING", attempts=2,
                                      error_class="transient", last_error="HttpError 503")])
    move_clock(monkeypatch, 11)
    _submit(env, FakeBackend())
    row = mf.read_manifest(env.run).set_index("task_key").loc["RO123_ASC__chunk_r00c00"]
    assert row["state"] == "FAILED" and row["error_class"] == "transient"


def test_task_created_before_submitting_is_not_adopted(env):
    """An old task with the same description (e.g. rejected earlier) must not be taken for a new submission."""
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    mf.update_manifest(env.run, [dict(task_key=key, state="SUBMITTING")])
    be = FakeBackend()
    old = datetime.now(timezone.utc) - timedelta(hours=3)
    be.tasks = [dict(id="OLD", description=export.task_description(env.cfg, env.uid, key), state="COMPLETED",
                     create_time=iso(old))]
    _submit(env, be)
    assert states(env.run)[key] == "SUBMITTING"


# ---------------------------------------------------------------- re-export after failed verification (M6)
def test_fresh_export_row_ignores_old_task_and_old_gcs_files(env, monkeypatch):
    """A row requeued by download (adopted = -1) must be exported again, never re-adopt its old output."""
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    row0 = mf.read_manifest(env.run).set_index("task_key").loc[key]
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    # what download does when adopted output fails verification
    mf.update_manifest(env.run, [dict(task_key=key, state="PENDING", error_class="verify", adopted=-1, attempts=1,
                                      task_id="OLD"),
                                 dict(task_key="RO201_DSC__chunk_r00c00", state="VERIFIED")])
    be = FakeBackend()
    be.tasks = [dict(id="OLD", description=row0["description"], state="COMPLETED", create_time=iso(old))]
    gcs = ListOnlyGCS([(row0["gcs_prefix"] + ".tif", iso(old))])
    env.cfg["export"]["max_workers"] = 1
    be.start_errors = ["connection reset by peer"]       # crash-like ambiguity right after SUBMITTING
    _submit(env, be, gcs=gcs)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "SUBMITTING" and row["adopted"] == -1 and row["task_id"] == "" and row["submitted_at"]
    move_clock(monkeypatch, 1)
    _submit(env, be, gcs=gcs)                             # old task and old files must not be adopted
    assert states(env.run)[key] == "SUBMITTING"
    move_clock(monkeypatch, 11)
    _submit(env, be, gcs=gcs)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "SUBMITTED" and row["task_id"] != "OLD" and len(be.started) == 1
    assert row["adopted"] == -1                           # marker kept until download verifies the new files


def test_fresh_export_row_adopts_only_files_written_after_its_submission(env):
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    prefix = mf.read_manifest(env.run).set_index("task_key").loc[key, "gcs_prefix"]
    t_sub = datetime.now(timezone.utc) - timedelta(minutes=1)
    mf.update_manifest(env.run, [dict(task_key=key, state="SUBMITTING", adopted=-1, submitted_at=iso(t_sub),
                                      error_class="transient"),
                                 dict(task_key="RO201_DSC__chunk_r00c00", state="VERIFIED")])
    gcs_old = ListOnlyGCS([(prefix + ".tif", iso(t_sub - timedelta(hours=3))),
                           (prefix + "-0000000000-0000000000.tif", "")])      # old and undated parts
    _submit(env, FakeBackend(), gcs=gcs_old)
    assert states(env.run)[key] == "SUBMITTING"
    export._SESSIONS.clear()                              # new session: re-list GCS
    gcs_new = ListOnlyGCS([(prefix + ".tif", iso(t_sub + timedelta(seconds=30)))])
    _submit(env, FakeBackend(), gcs=gcs_new)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "COMPLETED" and row["adopted"] == -1


def test_normal_submitting_row_adopts_existing_gcs_output(env):
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    prefix = mf.read_manifest(env.run).set_index("task_key").loc[key, "gcs_prefix"]
    mf.update_manifest(env.run, [dict(task_key=key, state="SUBMITTING", submitted_at=mf.now_utc()),
                                 dict(task_key="RO201_DSC__chunk_r00c00", state="VERIFIED")])
    _submit(env, FakeBackend(), gcs=ListOnlyGCS([prefix + ".tif"]))
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "COMPLETED" and row["adopted"] == 1


def test_legacy_failed_download_adopted_rows_requeued(env):
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    mf.update_manifest(env.run, [dict(task_key=key, state="FAILED_DOWNLOAD", error_class="verify", adopted=1),
                                 dict(task_key="RO201_DSC__chunk_r00c00", state="FAILED_DOWNLOAD",
                                      error_class="verify", adopted=0)])
    be = FakeBackend()
    _submit(env, be)
    df = mf.read_manifest(env.run).set_index("task_key")
    assert df.loc[key, "state"] == "SUBMITTED" and df.loc[key, "adopted"] == -1
    assert df.loc["RO201_DSC__chunk_r00c00", "state"] == "FAILED_DOWNLOAD"


def test_legacy_pending_verify_row_becomes_fresh_export(env):
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    mf.update_manifest(env.run, [dict(task_key=key, state="PENDING", error_class="verify", adopted=0)])
    _submit(env, FakeBackend())
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "SUBMITTED" and row["adopted"] == -1


def test_lost_fresh_export_keeps_marker(env):
    be = FakeBackend()
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    mf.update_manifest(env.run, [dict(task_key=key, adopted=-1)])
    _submit(env, be)
    for _ in range(export.UNKNOWN_POLLS_TO_LOOKUP):
        export.refresh_status(env.cfg, env.run, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "SUBMITTING" and row["error_class"] == "lost" and row["adopted"] == -1


# ---------------------------------------------------------------- slow ambiguous start (H2)
def test_ambiguous_error_keeps_submitted_at_from_the_claim(env):
    env.cfg["export"]["max_workers"] = 1
    plan_confirm(env, ["chunk_r00c00"])
    be = FakeBackend()
    be.start_errors = ["Deadline exceeded"]
    _submit(env, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc["RO123_ASC__chunk_r00c00"]
    assert row["state"] == "SUBMITTING" and row["submitted_at"] != "" and row["submitted_at"] <= row["updated_at"]


def test_slow_ambiguous_start_error_still_adopts_its_task(env, monkeypatch):
    """EE created the task at T0; the client kept retrying and raised "Deadline exceeded" 5 minutes later."""
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    desc = mf.read_manifest(env.run).set_index("task_key").loc[key, "description"]
    t0 = datetime.now(timezone.utc) - timedelta(minutes=5)
    # the claim at T0 (submitted_at), then the failure update written 5 minutes later (updated_at = now)
    mf.update_manifest(env.run, [dict(task_key=key, state="SUBMITTING", submitted_at=iso(t0), updated_at=iso(t0))])
    mf.update_manifest(env.run, [dict(task_key=key, state="SUBMITTING", error_class="transient", attempts=1,
                                      last_error="Deadline exceeded")])
    be = FakeBackend()
    be.tasks = [dict(id="EE1", description=desc, state="RUNNING", create_time=iso(t0 + timedelta(seconds=2)))]
    move_clock(monkeypatch, 11)                           # even after the grace period: adopted, never restarted
    _submit(env, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "RUNNING" and row["task_id"] == "EE1" and row["adopted"] == 1
    assert all(t["params"]["description"] != desc for t in be.started)


# ---------------------------------------------------------------- one submitter per run (H1)
class SlowAutoBackend(FakeBackend):
    """Tasks complete on the poll after submission; start() is slow, to widen race windows."""

    def __init__(self):
        super().__init__()
        self.lock = threading.Lock()
        self.key_by_id = {}

    def start_export(self, image, params):
        time.sleep(0.02)
        with self.lock:
            tid = super().start_export(image, params)
        self.key_by_id[tid] = params["fileNamePrefix"]
        return tid

    def task_statuses(self, ids):
        return {i: {"state": "COMPLETED"} for i in ids}


def test_submit_pending_refused_while_another_thread_holds_the_monitor_lock(env):
    plan_confirm(env)
    held, release = threading.Event(), threading.Event()

    def hold():
        lock = run_lock(env.run, "monitor")
        held.set()
        release.wait(30)
        lock.release()

    t = threading.Thread(target=hold)
    t.start()
    assert held.wait(30)
    be = FakeBackend()
    try:
        with pytest.raises(PipelineError, match="monitor"):
            _submit(env, be)
    finally:
        release.set()
        t.join(30)
    assert be.started == [] and set(states(env.run).values()) == {"PENDING"}
    _submit(env, be)                                      # lock free again
    assert len(be.started) == 12


def test_submit_pending_while_monitor_runs_never_duplicates(env):
    plan_confirm(env)
    be = SlowAutoBackend()
    done = threading.Event()

    def run_monitor():
        try:
            export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: time.sleep(0.01))
        finally:
            done.set()

    t = threading.Thread(target=run_monitor)
    t.start()
    deadline = time.time() + 30
    while not (env.run / ".monitor.lock").exists() and not done.is_set():   # the monitor starts first
        assert time.time() < deadline
        time.sleep(0.001)
    refused = 0
    while not done.is_set():
        try:
            export.submit_pending(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None)
        except PipelineError:
            refused += 1
        time.sleep(0.005)
    t.join(60)
    prefixes = [x["params"]["fileNamePrefix"] for x in be.started]
    assert len(prefixes) == len(set(prefixes)) == 12      # every chunk started exactly once
    assert set(states(env.run).values()) == {"COMPLETED"}


def test_claim_skips_rows_no_longer_pending(env):
    env.cfg["export"]["max_workers"] = 1
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    be = FakeBackend()
    real_claim = mf.claim

    def claim_after_someone_else(run_dir, task_key, expected, values):
        if task_key == key:   # another process claimed and submitted this row in between
            mf.update_manifest(run_dir, [dict(task_key=key, state="SUBMITTED", task_id="ELSEWHERE")])
        return real_claim(run_dir, task_key, expected, values)

    orig = export.mf.claim
    export.mf.claim = claim_after_someone_else
    try:
        _submit(env, be)
    finally:
        export.mf.claim = orig
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["task_id"] == "ELSEWHERE"
    assert all(not t["params"]["fileNamePrefix"].endswith("track_RO123_ASC/chunk_r00c00") for t in be.started)


# ---------------------------------------------------------------- project validation before changes (H4)
def _manifest_bytes(run):
    j = mf.journal_path(run)
    return mf.manifest_path(run).read_bytes(), (j.read_bytes() if j.exists() else b"")


def test_project_mismatch_detected_before_any_manifest_change(env):
    plan_confirm(env, ["chunk_r00c00"])
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="SUBMITTED", task_id="T1",
                                      submitted_at=mf.now_utc()),
                                 dict(task_key="RO201_DSC__chunk_r00c00", state="FAILED", error_class="other")])
    mf.compact(env.run)
    before = _manifest_bytes(env.run)
    env.cfg["auth"]["project"] = "another-project"
    be = FakeBackend()
    calls = [
        lambda: _submit(env, be),
        lambda: export.refresh_status(env.cfg, env.run, be),
        lambda: export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None),
        lambda: export.retry_failed(env.run, cfg=env.cfg),
    ]
    for call in calls:
        with pytest.raises(PipelineError, match="another-project"):
            call()
        assert _manifest_bytes(env.run) == before
    assert be.started == [] and be.count_calls == 0 and not (env.run / ".monitor.lock").exists()


# ---------------------------------------------------------------- removed duplicates (L1)
def test_duplicate_helpers_removed():
    assert not hasattr(export, "_BLOB_RE") and not hasattr(export, "read_export_meta")
    assert not hasattr(export.EEBackend, "project_task_counts")
    assert not hasattr(export.EEBackend, "project_active_task_count")
    assert not hasattr(mf, "clean_stale_tmp")
    name = "fake_base/a/b/runs/u/raw_chunks/track_T/chunk_r00c00"
    assert export._BLOB_BASE_RE.match(name + "-0000000000-0000000256.tif").group("base") == name
    assert export._BLOB_BASE_RE.match(name + "_s1.tif").group("base") == name + "_s1"


# ---------------------------------------------------------------- retry state machine
def _submit_all(env, be):
    plan_confirm(env, ["chunk_r00c00", "chunk_r01c02"])
    _submit(env, be)
    df = mf.read_manifest(env.run)
    return dict(zip(df["task_key"], df["task_id"]))


def test_refresh_running_completed_cancelled_unknown(env):
    be = FakeBackend()
    ids = _submit_all(env, be)
    be.statuses = {
        ids["RO123_ASC__chunk_r00c00"]: {"state": "RUNNING"},
        ids["RO123_ASC__chunk_r01c02"]: {"state": "COMPLETED"},
        ids["RO201_DSC__chunk_r00c00"]: {"state": "CANCELLED", "error_message": "Cancelled by user"},
    }
    export.refresh_status(env.cfg, env.run, be)
    s = states(env.run)
    assert s["RO123_ASC__chunk_r00c00"] == "RUNNING"
    assert s["RO123_ASC__chunk_r01c02"] == "COMPLETED"
    assert s["RO201_DSC__chunk_r00c00"] == "FAILED"
    assert s["RO201_DSC__chunk_r01c02"] == "SUBMITTED"   # unknown to EE for one poll -> unchanged


def test_states_come_from_listing_without_per_task_requests(env):
    be = FakeBackend()
    ids = _submit_all(env, be)
    msgs = {"RO123_ASC__chunk_r00c00": ("RUNNING", ""), "RO123_ASC__chunk_r01c02": ("COMPLETED", ""),
            "RO201_DSC__chunk_r00c00": ("FAILED", "Image.select: band not found"),
            "RO201_DSC__chunk_r01c02": ("READY", "")}
    be.tasks = [dict(id=ids[k], description="d", state=st, error_message=m) for k, (st, m) in msgs.items()]
    export.refresh_status(env.cfg, env.run, be)
    df = mf.read_manifest(env.run).set_index("task_key")
    assert be.status_calls == 0
    assert df.loc["RO201_DSC__chunk_r00c00", "state"] == "FAILED"
    assert df.loc["RO201_DSC__chunk_r00c00", "last_error"] == "Image.select: band not found"
    assert df.loc["RO201_DSC__chunk_r01c02", "state"] == "READY"


def test_status_fallback_lookups_are_bounded(env, monkeypatch):
    monkeypatch.setattr(export, "_STATUS_LOOKUP_MAX", 2)
    be = FakeBackend()
    _submit_all(env, be)
    asked = []
    be.task_statuses = lambda ids: asked.append(list(ids)) or {}
    export.refresh_status(env.cfg, env.run, be)
    assert len(asked) == 1 and len(asked[0]) == 2


def test_lost_task_goes_to_lookup_then_fails_when_attempts_used(env, monkeypatch):
    env.cfg["export"]["task_max_attempts"] = 1
    be = FakeBackend()
    plan_confirm(env, ["chunk_r00c00"])
    _submit(env, be)
    key = "RO123_ASC__chunk_r00c00"
    for _ in range(export.UNKNOWN_POLLS_TO_LOOKUP - 1):
        export.refresh_status(env.cfg, env.run, be)
        assert states(env.run)[key] == "SUBMITTED"
    export.refresh_status(env.cfg, env.run, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "SUBMITTING" and row["error_class"] == "lost" and row["task_id"] != ""
    move_clock(monkeypatch, 11)
    _submit(env, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "FAILED" and row["error_class"] == "lost"
    assert len(be.started) == 2                          # the original two submissions only


def test_lost_task_found_again_by_id_is_adopted(env):
    be = FakeBackend()
    plan_confirm(env, ["chunk_r00c00"])
    _submit(env, be)
    key = "RO123_ASC__chunk_r00c00"
    for _ in range(export.UNKNOWN_POLLS_TO_LOOKUP):
        export.refresh_status(env.cfg, env.run, be)
    tid = mf.read_manifest(env.run).set_index("task_key").loc[key, "task_id"]
    be.tasks = [dict(id=tid, description="whatever", state="COMPLETED", create_time="")]
    _submit(env, be)
    assert states(env.run)[key] == "COMPLETED"


def test_memory_failure_splits_once_then_fails(env):
    be = FakeBackend()
    ids = _submit_all(env, be)
    key = "RO123_ASC__chunk_r00c00"
    be.statuses = {ids[key]: {"state": "FAILED", "error_message": "User memory limit exceeded."}}
    export.refresh_status(env.cfg, env.run, be)
    df = mf.read_manifest(env.run).set_index("task_key")
    assert df.loc[key, "state"] == "SPLIT"
    subs = df[df["parent_chunk"] == "chunk_r00c00"]
    assert sorted(subs.index) == [f"RO123_ASC__chunk_r00c00_s{i}" for i in range(4)]
    assert set(subs["state"]) == {"PENDING"}
    assert subs["width"].sum() == 2 * 256 and subs["height"].sum() == 2 * 256
    export.refresh_status(env.cfg, env.run, be)
    assert mf.read_manifest(env.run)["task_key"].is_unique

    _submit(env, be)
    df = mf.read_manifest(env.run)
    sub_ids = dict(zip(df["task_key"], df["task_id"]))
    sub_key = "RO123_ASC__chunk_r00c00_s2"
    be.statuses = {sub_ids[sub_key]: {"state": "FAILED", "error_message": "Computation timed out."}}
    export.refresh_status(env.cfg, env.run, be)
    assert states(env.run)[sub_key] == "FAILED"
    assert not any(k.startswith(sub_key + "_") for k in states(env.run))   # no second-level split


def test_split_disabled_marks_failed(env):
    env.cfg["export"]["split_on_resource_error"] = False
    be = FakeBackend()
    ids = _submit_all(env, be)
    be.statuses = {ids["RO123_ASC__chunk_r00c00"]: {"state": "FAILED", "error_message": "User memory limit exceeded."}}
    export.refresh_status(env.cfg, env.run, be)
    assert states(env.run)["RO123_ASC__chunk_r00c00"] == "FAILED"


@pytest.mark.parametrize("message,cls", [("Too many tasks", "quota"), ("Internal error.", "transient")])
def test_retryable_task_failure_retries_until_max_attempts(env, message, cls):
    env.cfg["export"]["task_max_attempts"] = 2
    be = FakeBackend()
    key = "RO123_ASC__chunk_r00c00"
    ids = _submit_all(env, be)
    be.statuses = {ids[key]: {"state": "FAILED", "error_message": message}}
    export.refresh_status(env.cfg, env.run, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "PENDING" and row["attempts"] == 1 and row["task_id"] == "" and row["error_class"] == cls
    _submit(env, be)
    new_id = mf.read_manifest(env.run).set_index("task_key").loc[key, "task_id"]
    be.statuses = {new_id: {"state": "FAILED", "error_message": message}}
    export.refresh_status(env.cfg, env.run, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "FAILED" and row["attempts"] == 2 and row["error_class"] == cls


def test_retry_failed_puts_rows_back(env):
    plan_confirm(env, ["chunk_r00c00"])
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="FAILED", error_class="other",
                                      task_id="X", last_error="boom"),
                                 dict(task_key="RO201_DSC__chunk_r00c00", state="FAILED", error_class="cancelled")])
    export.retry_failed(env.run, classes=["other"])
    s = states(env.run)
    assert s["RO123_ASC__chunk_r00c00"] == "PENDING" and s["RO201_DSC__chunk_r00c00"] == "FAILED"
    row = mf.read_manifest(env.run).set_index("task_key").loc["RO123_ASC__chunk_r00c00"]
    assert row["task_id"] == "" and "boom" in row["last_error"]
    export.retry_failed(env.run)
    assert set(states(env.run).values()) == {"PENDING"}


# ---------------------------------------------------------------- monitor, resume, report, ETA
class AutoBackend(FakeBackend):
    """Tasks complete on the poll after submission; chosen chunks fail with a given message."""

    def __init__(self, fail: dict[str, str] | None = None):
        super().__init__()
        self.fail = fail or {}
        self.key_by_id = {}

    def start_export(self, image, params):
        tid = super().start_export(image, params)
        self.key_by_id[tid] = params["fileNamePrefix"]
        return tid

    def task_statuses(self, ids):
        out = {}
        for i in ids:
            prefix = self.key_by_id[i]
            msg = next((m for name, m in self.fail.items() if prefix.endswith(name)), None)
            out[i] = {"state": "FAILED", "error_message": msg} if msg else {"state": "COMPLETED"}
        return out


def test_monitor_end_to_end_and_failed_report(env):
    plan_confirm(env)
    be = AutoBackend(fail={"track_RO201_DSC/chunk_r01c00": "Image.select: band not found"})
    df = export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None)
    s = dict(zip(df["task_key"], df["state"]))
    assert s.pop("RO201_DSC__chunk_r01c00") == "FAILED"
    assert set(s.values()) == {"COMPLETED"}
    assert len(be.started) == len(df)                     # every row submitted exactly once
    report = pd.read_csv(env.run / "failed_chunks.csv", keep_default_na=False)
    assert list(report.columns) == export.FAILED_COLUMNS
    assert list(report["task_key"]) == ["RO201_DSC__chunk_r01c00"]
    assert report.iloc[0]["error_class"] == "other"
    assert mf.journal_path(env.run).stat().st_size == 0   # compacted at the end: the CSV is up to date


def test_monitor_resume_after_crash_does_not_resubmit(env):
    plan_confirm(env)
    be1 = AutoBackend()
    env.cfg["export"]["max_active_tasks"] = 3
    export.monitor(env.cfg, env.run, be1, confirmed=True, sleep=lambda s: None, max_cycles=1)  # "crash"
    submitted_first = len(be1.started)
    assert submitted_first == 3
    be2 = AutoBackend()
    be2.key_by_id = be1.key_by_id                         # same EE project, new process
    be2._n = be1._n
    df = export.monitor(env.cfg, env.run, be2, confirmed=True, sleep=lambda s: None)
    assert set(df["state"]) == {"COMPLETED"}
    assert submitted_first + len(be2.started) == len(df)


def test_monitor_survives_transient_listing_errors(env):
    plan_confirm(env, ["chunk_r00c00"])
    be = AutoBackend()
    be.snapshot_errors = ["HttpError 503 Service unavailable", "Deadline exceeded"]
    waits = []
    df = export.monitor(env.cfg, env.run, be, confirmed=True, sleep=waits.append)
    assert set(df["state"]) == {"COMPLETED"} and len([w for w in waits if w > 0]) >= 2


def test_monitor_stops_on_non_recoverable_error(env):
    plan_confirm(env, ["chunk_r00c00"])
    be = AutoBackend()
    be.snapshot_errors = ["Permission denied for project"]
    with pytest.raises(RuntimeError, match="Permission denied"):
        export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None)


_LOCK_HOLDER = textwrap.dedent("""
    import sys, time, pathlib
    sys.path.insert(0, {src!r})
    from sar_pipeline.locking import run_lock
    run = pathlib.Path(sys.argv[1])
    lock = run_lock(run, "monitor")
    (run / "holder_ready").write_text("1")
    while not (run / "holder_release").exists():
        time.sleep(0.05)
    lock.release()
""")


def test_second_monitor_process_is_blocked_by_run_lock(env, tmp_path):
    plan_confirm(env, ["chunk_r00c00"])
    script = tmp_path / "holder.py"
    script.write_text(_LOCK_HOLDER.format(src=SRC))
    proc = subprocess.Popen([sys.executable, str(script), str(env.run)])
    try:
        deadline = time.time() + 60
        while not (env.run / "holder_ready").exists():
            assert time.time() < deadline and proc.poll() is None
            time.sleep(0.05)
        be = AutoBackend()
        with pytest.raises(PipelineError, match="monitor"):
            export.monitor(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None)
        assert be.started == []
    finally:
        (env.run / "holder_release").write_text("1")
        proc.wait(timeout=60)
    df = export.monitor(env.cfg, env.run, AutoBackend(), confirmed=True, sleep=lambda s: None)   # free again
    assert set(df["state"]) == {"COMPLETED"}


def test_empty_failed_report_has_header(env):
    export.plan_exports(env.cfg, env.run)
    path = export.write_failed_report(env.run)
    assert path.read_text().strip() == ",".join(export.FAILED_COLUMNS)


def _completed_rows(durations):
    base = datetime(2026, 9, 1, tzinfo=timezone.utc)
    rows = []
    for i, d in enumerate(durations):
        rows.append(dict(task_key=f"c{i}", state="COMPLETED", submitted_at=base.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         updated_at=(base + timedelta(seconds=d)).strftime("%Y-%m-%dT%H:%M:%SZ")))
    return rows


def test_eta_from_completed_durations(tmp_path):
    rows = _completed_rows([60, 120, 180])
    rows += [dict(task_key=f"p{i}", state="PENDING") for i in range(8)]
    rows += [dict(task_key=f"r{i}", state="RUNNING") for i in range(2)]
    df = mf.update_manifest(tmp_path, rows)
    assert export.median_task_seconds(df) == 120
    assert export.estimate_eta_seconds(df, running=5) == pytest.approx(10 / 5 * 120)
    assert export.estimate_eta_seconds(df, running=0) == pytest.approx(10 * 120)   # guards divide-by-zero


def test_eta_needs_three_completed(tmp_path):
    df = mf.update_manifest(tmp_path, _completed_rows([60, 120]) + [dict(task_key="p", state="PENDING")])
    assert export.estimate_eta_seconds(df, running=4) is None


def test_round_log_line(env, caplog):
    plan_confirm(env)
    mf.update_manifest(env.run, _completed_rows([100, 200, 300]))
    be = FakeBackend(ready=2676, running=14)              # headroom 2700-2690 = 10 of 12 pending
    with caplog.at_level(logging.INFO, logger="sar_pipeline.export"):
        _submit(env, be)
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("submission round")][-1]
    for part in ("queued=2676", "running=14", "submitted=10", "headroom=10", "pending_left=2"):
        assert part in line
    # (2 pending + 10 submitted) / 14 running * median 200 s = 171 s -> "0h02m"
    assert "eta=0h02m" in line


# ---------------------------------------------------------------- resume, never redo
class Crash(BaseException):
    """Simulates the process dying (not an Exception, so nothing in the pipeline catches it)."""


class CrashAfterStart(FakeBackend):
    def start_export(self, image, params):
        super().start_export(image, params)
        raise Crash()


def test_crash_between_start_and_manifest_write_adopts_task(env):
    env.cfg["export"]["max_workers"] = 1
    plan_confirm(env, ["chunk_r00c00"])
    be1 = CrashAfterStart()
    with pytest.raises(Crash):
        _submit(env, be1)
    df = mf.read_manifest(env.run)
    crashed = df[df["state"] == "SUBMITTING"]
    assert len(crashed) >= 1 and (crashed["task_id"] == "").all()

    be2 = FakeBackend()
    be2.tasks = [dict(id=t["task_id"], description=t["params"]["description"], state="RUNNING") for t in be1.started]
    be2.tasks.append(dict(id="OTHER", description="someone_elses_task", state="READY"))
    _submit(env, be2)
    df = mf.read_manifest(env.run).set_index("task_key")
    for t in be1.started:
        row = df[df["description"] == t["params"]["description"]].iloc[0]
        assert row["state"] == "RUNNING" and row["task_id"] == t["task_id"] and row["adopted"] == 1
    assert len(be2.started) == len(df) - len(be1.started)   # only never-started rows were submitted
    assert "SUBMITTING" not in set(df["state"])


def test_completed_ee_task_adopted_for_submitting_row(env):
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO201_DSC__chunk_r00c00"
    mf.update_manifest(env.run, [dict(task_key=key, state="SUBMITTING")])
    be = FakeBackend()
    desc = export.task_description(env.cfg, env.uid, key)
    be.tasks = [dict(id="A1", description=desc, state="READY"), dict(id="A2", description=desc, state="COMPLETED")]
    _submit(env, be)
    row = mf.read_manifest(env.run).set_index("task_key").loc[key]
    assert row["state"] == "COMPLETED" and row["task_id"] == "A2" and row["adopted"] == 1
    assert [t["params"]["description"] for t in be.started] == [export.task_description(env.cfg, env.uid, "RO123_ASC__chunk_r00c00")]


def test_pending_rows_are_never_adopted(env):
    plan_confirm(env, ["chunk_r00c00"])
    key = "RO123_ASC__chunk_r00c00"
    be = FakeBackend()
    be.tasks = [dict(id="A", description=export.task_description(env.cfg, env.uid, key), state="COMPLETED")]
    _submit(env, be)
    assert states(env.run)[key] == "SUBMITTED" and len(be.started) == 2


class ListOnlyGCS:
    """GCS listing fake. Entries are names, or (name, time_created ISO string) tuples."""

    def __init__(self, names):
        self.entries = [(n, "") if isinstance(n, str) else tuple(n) for n in names]
        self.list_calls = 0

    def list_blobs(self, prefix):
        self.list_calls += 1
        return [{"name": n, "size": 1, "crc32c": "", "time_created": tc} for n, tc in self.entries
                if n.startswith(prefix)]


def test_gcs_output_present_marks_submitting_row_completed_without_export(env):
    plan_confirm(env, ["chunk_r00c00", "chunk_r00c01"])
    df = mf.read_manifest(env.run).set_index("task_key")
    done = df.loc["RO123_ASC__chunk_r00c00", "gcs_prefix"]
    other = df.loc["RO123_ASC__chunk_r00c01", "gcs_prefix"]
    mf.update_manifest(env.run, [dict(task_key=k, state="SUBMITTING") for k in
                                 ("RO123_ASC__chunk_r00c00", "RO123_ASC__chunk_r00c01", "RO201_DSC__chunk_r00c00")])
    gcs = ListOnlyGCS([done + "-0000000000-0000000000.tif", other + "_s0.tif"])   # sub-chunk file must not count
    be = FakeBackend()
    export.submit_pending(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None, gcs=gcs)
    s = states(env.run)
    assert s["RO123_ASC__chunk_r00c00"] == "COMPLETED"
    assert s["RO123_ASC__chunk_r00c01"] == "SUBMITTING"      # still within the grace period
    assert s["RO201_DSC__chunk_r00c01"] == "SUBMITTED"       # a PENDING row: exported, never GCS-checked
    assert len(be.started) == 1
    assert gcs.list_calls == 2                              # one listing per track
    export.submit_pending(env.cfg, env.run, be, confirmed=True, sleep=lambda s: None, gcs=gcs)
    assert gcs.list_calls == 2                              # cached for the session


def test_description_contains_aoi_and_season(env):
    d = export.task_description(env.cfg, "v001_20260915_abc123", "RO123_ASC__chunk_r00c00")
    assert d == "test_aoi_test_season_v001_20260915_abc123_RO123_ASC__chunk_r00c00"


def test_plan_rerun_is_noop(env):
    export.plan_exports(env.cfg, env.run)
    meta, layout = env.run / export.EXPORT_META, env.run / export.BAND_LAYOUT
    before = (meta.stat().st_mtime_ns, layout.stat().st_mtime_ns, meta.read_text(), layout.read_text(),
              mf.manifest_path(env.run).read_text())
    export.plan_exports(env.cfg, env.run)
    after = (meta.stat().st_mtime_ns, layout.stat().st_mtime_ns, meta.read_text(), layout.read_text(),
             mf.manifest_path(env.run).read_text())
    assert before == after


def test_legacy_meta_without_new_keys_is_accepted(env):
    export.plan_exports(env.cfg, env.run)
    path = env.run / export.EXPORT_META
    meta = json.loads(path.read_text())
    for k in run_meta.OPTIONAL_META_KEYS:
        meta.pop(k)
    path.write_text(json.dumps(meta))
    export.plan_exports(env.cfg, env.run)                  # no "planned with a different layout" error


def test_changed_layout_force_rules(env):
    export.plan_exports(env.cfg, env.run)
    env.state["selected"]["RO123_ASC"] = make_acq("RO123_ASC", 4)
    with pytest.raises(PipelineError, match="immutable"):
        export.plan_exports(env.cfg, env.run)
    export.plan_exports(env.cfg, env.run, force=True)      # nothing exported yet -> allowed
    layout = run_meta.read_band_layout(env.run)
    assert len(layout[layout["track_id"] == "RO123_ASC"]) == 8
    mf.update_manifest(env.run, [dict(task_key="RO123_ASC__chunk_r00c00", state="SUBMITTED", task_id="X")])
    env.state["selected"]["RO123_ASC"] = make_acq("RO123_ASC", 5)
    with pytest.raises(PipelineError, match="force=True refused"):
        export.plan_exports(env.cfg, env.run, force=True)


def test_old_stale_tmp_cleaned_on_submit(env):
    plan_confirm(env)
    stale = env.run / "export_manifest.csv.tmp"
    stale.write_text("half written")
    past = time.time() - 7 * 3600
    import os
    os.utime(stale, (past, past))
    fresh = env.run / ".other_process_writing.tmp"
    fresh.write_text("in progress")
    _submit(env, FakeBackend())
    assert not stale.exists() and fresh.exists()


def test_plan_requires_an_audit(make_project, env):
    import shutil
    shutil.rmtree(config.season_dir(env.cfg) / "audit")
    run2 = config.new_run(env.cfg)
    from sar_pipeline.errors import AuditRequired
    with pytest.raises(AuditRequired):
        export.plan_exports(env.cfg, run2)


def test_plan_keeps_its_audit_when_a_newer_audit_appears(env):
    seen = []
    import sar_pipeline as sp
    orig = sp.audit.selected_acquisitions
    sp.audit.selected_acquisitions = lambda c, audit_path=None: (seen.append(str(audit_path)), orig(c, audit_path))[1]
    export.plan_exports(env.cfg, env.run)
    config.audit_dir(env.cfg, "20261001").mkdir(parents=True)
    export.plan_exports(env.cfg, env.run)
    assert all(p.endswith("20260915") for p in seen)


def test_config_hash_ignores_operational_settings(env):
    h = export.config_hash(env.cfg)
    env.cfg["export"]["max_workers"] = 99
    env.cfg["export"]["poll_seconds"] = 5
    assert export.config_hash(env.cfg) == h
    env.cfg["ard"]["speckle_kernel"] = 7
    assert export.config_hash(env.cfg) != h


def test_shared_reader_accepts_export_meta_from_plan_exports(env):
    export.plan_exports(env.cfg, env.run)
    meta = run_meta.read_export_meta(env.cfg, env.run)
    assert meta["dtype"] == "float32" and meta["nodata"] == -9999.0 and meta["int16_scale"] == 100.0
    assert meta["grid_fingerprint"] == run_meta.grid_fingerprint(env.cfg)
    assert meta["audit_dir"] == "processed/test_aoi/test_season/audit/20260915"


def test_project_mismatch_is_refused(env):
    plan_confirm(env, ["chunk_r00c00"])
    env.cfg["auth"]["project"] = "another-project"
    with pytest.raises(PipelineError, match="another-project"):
        run_meta.read_export_meta(env.cfg, env.run)
    be = FakeBackend()
    with pytest.raises(PipelineError, match="project"):
        _submit(env, be)
    assert be.started == []


# ---------------------------------------------------------------- bounded project task listing
NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)


def _iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FakePagedOps:
    """Mimics projects().operations().list: active ops first, then finished ops newest-first (1 per minute)."""

    def __init__(self, n_terminal=71_271, active=(("RUNNING", "our_a"), ("RUNNING", "our_b")), page_size=500,
                 inject=None):
        self.page_size = page_size
        self.pages_fetched = 0
        head = [dict(name=f"projects/p/operations/A{i}", done=False,
                     metadata=dict(state=st, description=d, createTime=_iso(NOW - timedelta(minutes=i))))
                for i, (st, d) in enumerate(active)]
        self.ops = head + [
            dict(name=f"projects/p/operations/T{i}", done=True,
                 metadata=dict(state="SUCCEEDED" if i % 100 else "CANCELLED", description=f"old_{i}",
                               createTime=_iso(NOW - timedelta(minutes=i + 1))))
            for i in range(n_terminal)
        ]
        for pos, op in (inject or {}).items():
            self.ops.insert(pos, op)

    def __call__(self, token):
        self.pages_fetched += 1
        start = int(token or 0)
        end = start + self.page_size
        resp = {"operations": self.ops[start:end]}
        if end < len(self.ops):
            resp["nextPageToken"] = str(end)
        return resp


def test_scan_stops_early_after_cutoff():
    fake = FakePagedOps()
    assert len(fake.ops) == 71_273
    snap = export.scan_operations(fake, cutoff=NOW - timedelta(days=1))   # 1440 finished ops per day
    assert snap["counts"] == {"READY": 0, "RUNNING": 2}
    st = snap["stats"]
    assert st["stopped_early"] and not st["full_fallback"]
    assert st["pages"] == fake.pages_fetched == 3 and st["operations"] == 1500
    descs = {t["description"] for t in snap["tasks"]}
    assert {"our_a", "our_b", "old_0", "old_1400"} <= descs
    assert snap["tasks"][2]["state"] == "CANCELLED" and snap["tasks"][3]["state"] == "COMPLETED"


def test_scan_without_cutoff_reads_everything():
    fake = FakePagedOps(n_terminal=2_998)
    snap = export.scan_operations(fake, cutoff=None)
    assert snap["stats"]["pages"] == 6 and snap["stats"]["operations"] == 3000
    assert not snap["stats"]["stopped_early"]


def test_scan_active_after_terminal_falls_back_to_full(caplog):
    late_active = dict(name="projects/p/operations/LATE", done=False,
                       metadata=dict(state="PENDING", description="late", createTime=_iso(NOW - timedelta(days=3))))
    fake = FakePagedOps(n_terminal=9_998, inject={700: late_active})
    with caplog.at_level(logging.WARNING, logger="sar_pipeline.export"):
        snap = export.scan_operations(fake, cutoff=NOW - timedelta(days=1))
    assert snap["counts"] == {"READY": 1, "RUNNING": 2}
    assert snap["stats"]["full_fallback"] and not snap["stats"]["stopped_early"]
    assert snap["stats"]["pages"] == 21 and snap["stats"]["operations"] == 10_001   # 2 active + 9,998 + 1 late
    assert "full listing" in caplog.text


def test_scan_cutoff_on_last_page_and_empty_project():
    assert export.scan_operations(lambda tok: {}, cutoff=NOW)["counts"] == {"READY": 0, "RUNNING": 0}
    fake = FakePagedOps(n_terminal=10)
    snap = export.scan_operations(fake, cutoff=NOW - timedelta(days=30))
    assert snap["stats"]["pages"] == 1 and not snap["stats"]["stopped_early"]


def test_listing_cutoff(tmp_path):
    run = tmp_path / "v002_20260910"
    run.mkdir()
    df = mf.read_manifest(run)
    assert export.listing_cutoff(df, run) == datetime(2026, 9, 9, tzinfo=timezone.utc)
    mf.update_manifest(run, [
        dict(task_key="a", state="SUBMITTED", submitted_at="2026-09-12T08:00:00Z"),
        dict(task_key="b", state="SUBMITTING", updated_at="2026-09-01T06:30:00Z"),
    ])
    assert export.listing_cutoff(mf.read_manifest(run), run) == datetime(2026, 8, 31, 6, 30, tzinfo=timezone.utc)


def test_eebackend_snapshot_cache_and_own_submissions(monkeypatch):
    import ee

    be = export.EEBackend()
    fake = FakePagedOps(n_terminal=5_000)
    monkeypatch.setattr(be, "_fetch_page", fake)
    cutoff = NOW - timedelta(days=1)
    s1 = be.project_snapshot(cutoff=cutoff, max_age=30)
    pages = fake.pages_fetched
    assert pages == 3
    assert be.project_snapshot(cutoff=cutoff, max_age=30) is s1 and fake.pages_fetched == pages   # cached

    class Task:
        id = "NEW1"

        def start(self):
            pass

    monkeypatch.setattr(ee.Geometry, "Rectangle", staticmethod(lambda *a, **k: "RECT"))
    monkeypatch.setattr(ee.batch.Export.image, "toCloudStorage", staticmethod(lambda **k: Task()))
    be.start_export("IMG", dict(description="mine", region=[0, 0, 1, 1], crs="EPSG:6933"))
    s2 = be.project_snapshot(cutoff=cutoff, max_age=30)
    assert s2["counts"] == {"READY": 1, "RUNNING": 2}                       # own submission added
    assert any(t["description"] == "mine" and t["id"] == "NEW1" for t in s2["tasks"])

    be.project_snapshot(cutoff=cutoff - timedelta(days=5), max_age=30)       # older cutoff -> cache not enough
    assert fake.pages_fetched > pages
    n = fake.pages_fetched
    be.project_snapshot(cutoff=cutoff, max_age=0)                            # max_age 0 -> always refresh
    assert fake.pages_fetched > n


def test_task_statuses_use_get_task_status_in_batches(monkeypatch):
    import ee

    calls = []
    monkeypatch.setattr(ee.data, "getTaskStatus",
                        lambda ids: calls.append(len(ids)) or [dict(id=i, state="RUNNING") for i in ids])
    monkeypatch.setattr(ee.data, "listOperations", lambda *a, **k: pytest.fail("listing must not be used"))
    out = export.EEBackend().task_statuses([f"T{i}" for i in range(120)])
    assert calls == [50, 50, 20] and len(out) == 120 and out["T7"]["state"] == "RUNNING"


class CutoffAwareBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.snapshot_args = []

    def project_snapshot(self, cutoff=None, max_age=0.0):
        self.snapshot_args.append((cutoff, max_age))
        return super().project_snapshot()


def test_submit_pending_passes_cutoff_and_poll_age(env):
    env.cfg["export"]["poll_seconds"] = 45
    plan_confirm(env, ["chunk_r00c00"])
    be = CutoffAwareBackend()
    _submit(env, be)
    cutoff, max_age = be.snapshot_args[0]
    run_date = datetime.strptime(env.run.name.split("_")[1], "%Y%m%d").replace(tzinfo=timezone.utc)
    assert cutoff <= run_date - timedelta(days=1) and max_age == 45


def test_clock_skew_tolerance_matches_download():
    """export (adoption window, fresh-export GCS filter) and download (stale blobs) must use the same margin:
    a blob counts as older than a row's submitted_at only if it predates it by more than this many seconds."""
    from sar_pipeline import download

    assert float(export.CREATE_TIME_SKEW_SECONDS) == float(download.STALE_BLOB_TOLERANCE_S)
