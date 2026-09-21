"""Unit tests for the export manifest (state file shared by export, download and stack)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

from sar_pipeline import manifest as mf

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_empty_manifest_has_all_columns(tmp_path):
    df = mf.read_manifest(tmp_path)
    assert list(df.columns) == mf.COLUMNS
    assert df.empty
    assert mf.COLUMNS[-1] == "adopted"


def test_insert_update_and_round_trip_types(tmp_path):
    mf.update_manifest(tmp_path, [dict(task_key="T__a", track_id="T", chunk_name="a", task_id="0123", width=256)])
    df = mf.update_manifest(tmp_path, [dict(task_key="T__a", state="SUBMITTED", attempts=2)])
    assert len(df) == 1
    mf.compact(tmp_path)
    mf._CACHE.clear()                                   # force a read from disk
    df = mf.read_manifest(tmp_path)
    row = df.iloc[0]
    assert row["task_id"] == "0123"          # leading zero kept: strings stay strings
    assert row["state"] == "SUBMITTED"
    assert row["attempts"] == 2 and df["attempts"].dtype == "int64"
    assert row["width"] == 256 and row["height"] == 0 and row["adopted"] == 0
    assert row["last_error"] == "" and row["parent_chunk"] == ""   # empty, not NaN
    assert row["updated_at"].endswith("Z")
    assert list(pd.read_csv(mf.manifest_path(tmp_path)).columns) == mf.COLUMNS


def test_new_rows_default_to_pending(tmp_path):
    df = mf.update_manifest(tmp_path, [dict(task_key="k")])
    assert df.iloc[0]["state"] == "PENDING" and df.iloc[0]["attempts"] == 0


@pytest.mark.parametrize("row", [dict(state="PENDING"), dict(task_key="k", state="DONE"), dict(task_key="k", foo=1)])
def test_invalid_rows_rejected(tmp_path, row):
    with pytest.raises(ValueError):
        mf.update_manifest(tmp_path, [row])


@pytest.mark.parametrize("state", ["PLANNED", "SUBMITTING", "NOT_COVERED", "VERIFIED_EMPTY"])
def test_new_states_are_valid(tmp_path, state):
    assert mf.update_manifest(tmp_path, [dict(task_key="k", state=state)]).iloc[0]["state"] == state


def test_concurrent_upserts_from_threads(tmp_path):
    n_threads, per_thread = 16, 25

    def work(t):
        for i in range(per_thread):
            mf.update_manifest(tmp_path, [dict(task_key=f"k{t}_{i}", attempts=i)], return_df=False)
            mf.update_manifest(tmp_path, [dict(task_key="shared", attempts=t)], return_df=False)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    df = mf.read_manifest(tmp_path)
    assert len(df) == n_threads * per_thread + 1
    assert df["task_key"].is_unique
    assert not list(tmp_path.glob("*.tmp"))


_WORKER = textwrap.dedent("""
    import sys
    sys.path.insert(0, {src!r})
    from sar_pipeline import manifest as mf
    run, tag, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    for i in range(n):
        mf.update_manifest(run, [dict(task_key=f"{{tag}}_{{i}}", track_id=tag, attempts=i)], return_df=False)
        mf.update_manifest(run, [dict(task_key="shared_" + tag, attempts=i)], return_df=False)
""")


def test_concurrent_updates_from_two_processes_lose_nothing(tmp_path, monkeypatch):
    """Two OS processes (e.g. monitor + download) updating the same run: every update survives.

    Compaction is forced to happen often so that snapshot rewrites race with appends too.
    """
    script = tmp_path / "worker.py"
    script.write_text(_WORKER.format(src=SRC).replace("from sar_pipeline import manifest as mf",
                                                      "from sar_pipeline import manifest as mf\nmf.COMPACT_MIN_LINES = 25"))
    run = tmp_path / "run"
    run.mkdir()
    n = 120
    procs = [subprocess.Popen([sys.executable, str(script), str(run), tag, str(n)]) for tag in ("A", "B")]
    for p in procs:
        assert p.wait(timeout=240) == 0
    mf._CACHE.clear()
    df = mf.read_manifest(run).set_index("task_key")
    for tag in ("A", "B"):
        assert {f"{tag}_{i}" for i in range(n)} <= set(df.index)
        assert df.loc[f"shared_{tag}", "attempts"] == n - 1
    assert len(df) == 2 * n + 2


def test_torn_last_journal_line_is_tolerated(tmp_path, caplog):
    mf.update_manifest(tmp_path, [dict(task_key="a", state="PLANNED")])
    with open(mf.journal_path(tmp_path), "ab") as f:
        f.write(b'{"k":"a","v":{"state":"SUBMI')              # process killed mid-write
    mf._CACHE.clear()
    assert mf.read_manifest(tmp_path).iloc[0]["state"] == "PLANNED"
    mf.update_manifest(tmp_path, [dict(task_key="b", state="PENDING")])   # next append terminates the torn line
    mf._CACHE.clear()
    df = mf.read_manifest(tmp_path).set_index("task_key")
    assert df.loc["a", "state"] == "PLANNED" and df.loc["b", "state"] == "PENDING"
    assert "damaged line" in caplog.text


def test_compaction_is_equivalent_and_empties_journal(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, "COMPACT_MIN_LINES", 10)
    for i in range(40):
        mf.update_manifest(tmp_path, [dict(task_key=f"k{i % 7}", attempts=i, state="PENDING")], return_df=False)
    before = mf.read_manifest(tmp_path)
    mf.compact(tmp_path)
    assert mf.journal_path(tmp_path).stat().st_size == 0
    mf._CACHE.clear()
    after = mf.read_manifest(tmp_path)
    pd.testing.assert_frame_equal(before, after)
    assert len(after) == 7


def test_compact_without_changes_does_not_rewrite(tmp_path):
    mf.update_manifest(tmp_path, [dict(task_key="k")])
    mf.compact(tmp_path)
    stamp = mf.manifest_path(tmp_path).stat().st_mtime_ns
    time.sleep(0.01)
    mf.compact(tmp_path)
    assert mf.manifest_path(tmp_path).stat().st_mtime_ns == stamp


def test_update_on_large_manifest_is_fast(tmp_path):
    mf.update_manifest(tmp_path, [dict(task_key=f"k{i}", track_id="T", state="PLANNED", width=512)
                                  for i in range(30_000)], return_df=False)
    mf.compact(tmp_path)
    mf.update_manifest(tmp_path, [dict(task_key="k1", state="PENDING")], return_df=False)   # warm the cache
    t0 = time.perf_counter()
    n = 20
    for i in range(n):
        mf.update_manifest(tmp_path, [dict(task_key="k5", state="PENDING", attempts=i)], return_df=False)
    per_update = (time.perf_counter() - t0) / n
    assert per_update < 0.05, f"{per_update * 1000:.1f} ms per update"
    assert mf.read_manifest(tmp_path).set_index("task_key").loc["k5", "attempts"] == n - 1


def test_external_snapshot_change_is_detected(tmp_path):
    mf.update_manifest(tmp_path, [dict(task_key="k", state="PLANNED")])
    mf.compact(tmp_path)
    mf.read_manifest(tmp_path)                                 # cache warm
    df = pd.read_csv(mf.manifest_path(tmp_path), dtype=str, keep_default_na=False)
    df.loc[0, "state"] = "FAILED"
    time.sleep(0.01)
    df.to_csv(mf.manifest_path(tmp_path), index=False)          # another process rewrote the snapshot
    assert mf.read_manifest(tmp_path).iloc[0]["state"] == "FAILED"


def test_clean_stale_tmp_respects_age(tmp_path):
    from sar_pipeline import run_meta

    fresh = tmp_path / "export_manifest.csv.tmp"
    fresh.write_text("x")
    sub = tmp_path / "raw_chunks" / "t"
    sub.mkdir(parents=True)
    old = sub / "a.tif.tmp"
    old.write_text("x")
    (sub / "a.tif").write_text("keep")
    past = time.time() - 7 * 3600
    os.utime(old, (past, past))
    assert run_meta.clean_stale_tmp(tmp_path, recursive=True) == 1   # only the old one: the fresh one may be in use
    assert fresh.exists() and not old.exists() and (sub / "a.tif").exists()
    assert run_meta.clean_stale_tmp(tmp_path, min_age_seconds=0) == 1


def test_claim_is_compare_and_set_across_threads(tmp_path):
    import threading

    mf.update_manifest(tmp_path, [dict(task_key="k", state="PENDING")])
    wins = []

    def go(i):
        if mf.claim(tmp_path, "k", ["PENDING"], dict(state="SUBMITTING", task_id=f"t{i}")):
            wins.append(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(wins) == 1
    row = mf.read_manifest(tmp_path).iloc[0]
    assert row["state"] == "SUBMITTING" and row["task_id"] == f"t{wins[0]}"
    assert not mf.claim(tmp_path, "missing", ["PENDING"], dict(state="SUBMITTING"))
    assert not mf.claim(tmp_path, "k", "PENDING", dict(state="FAILED"))


_CLAIMER = """
import sys, time, pathlib
sys.path.insert(0, {src!r})
from sar_pipeline import manifest as mf
run = pathlib.Path(sys.argv[1])
while not (run / "go").exists():
    time.sleep(0.005)
won = mf.claim(run, "k", ["PENDING"], dict(state="SUBMITTING", task_id=sys.argv[2]))
print("WIN" if won else "LOSE")
"""


def test_claim_is_compare_and_set_across_processes(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    src = str(Path(__file__).resolve().parents[1] / "src")
    run = tmp_path / "run"
    mf.update_manifest(run, [dict(task_key="k", state="PENDING")])
    script = tmp_path / "claimer.py"
    script.write_text(_CLAIMER.format(src=src))
    procs = [subprocess.Popen([sys.executable, str(script), str(run), f"p{i}"], stdout=subprocess.PIPE, text=True)
             for i in range(4)]
    time.sleep(1.0)                                            # let every process import and wait
    (run / "go").write_text("1")
    outs = [p.communicate(timeout=120)[0].strip() for p in procs]
    assert sorted(outs) == ["LOSE", "LOSE", "LOSE", "WIN"]
    assert mf.read_manifest(run).iloc[0]["state"] == "SUBMITTING"
