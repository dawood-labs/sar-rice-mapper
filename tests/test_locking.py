"""locking.FileLock: mutual exclusion, stale detection, heartbeat, break race, re-entrancy."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

import pytest

from sar_pipeline import locking
from sar_pipeline.errors import PipelineError


def _hold(path: str, seconds: float, ready) -> None:
    lock = locking.FileLock(path, timeout=5, stale_after=1.0)
    with lock:
        ready.set()
        time.sleep(seconds)


def test_live_holder_is_never_broken_by_age(tmp_path):
    path = str(tmp_path / ".x.lock")
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_hold, args=(path, 4.0, ready))
    p.start()
    try:
        assert ready.wait(30)
        old = time.time() - 3600
        os.utime(path, (old, old))  # looks ancient, but the holder is alive
        with pytest.raises(locking.LockTimeout):
            locking.FileLock(path, timeout=1.5, stale_after=1.0).acquire()
    finally:
        p.join(30)


def test_dead_holder_same_space_is_broken_immediately(tmp_path):
    path = tmp_path / ".x.lock"
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=time.sleep, args=(0,))
    p.start()
    p.join()
    path.write_text(json.dumps({"pid": p.pid, "pid_start": "", "space": locking._space_id(), "token": "dead"}))
    with locking.FileLock(path, timeout=2, stale_after=10_000):
        assert locking._read_holder(path)["pid"] == os.getpid()


def test_foreign_holder_expires_only_by_age(tmp_path):
    path = tmp_path / ".x.lock"
    path.write_text(json.dumps({"pid": 1, "space": "other-host|boot|ns", "token": "foreign"}))
    with pytest.raises(locking.LockTimeout):
        locking.FileLock(path, timeout=0.3, stale_after=60).acquire()
    old = time.time() - 120
    os.utime(path, (old, old))
    with locking.FileLock(path, timeout=2, stale_after=60):
        assert locking._read_holder(path)["token"] != "foreign"


def test_break_race_cannot_remove_a_fresh_lock(tmp_path):
    """B observed the old stale token; A already replaced the lock. B's break must not remove A's lock."""
    path = tmp_path / ".x.lock"
    a = locking.FileLock(path, timeout=2, stale_after=60)
    a.acquire()
    try:
        b = locking.FileLock(tmp_path / "other" / ".x.lock")  # separate state; call internals on target path
        b.path = path
        b._break_stale("old-stale-token")  # stale token no longer matches the live holder
        assert path.exists() and locking._read_holder(path)["token"] == a._state[2]
    finally:
        a.release()


def test_reentrant_and_threads_serialised(tmp_path):
    path = tmp_path / ".x.lock"
    counter = {"n": 0}

    def work():
        for _ in range(200):
            with locking.FileLock(path, timeout=30):
                with locking.FileLock(path, timeout=30):  # re-entrant
                    v = counter["n"]
                    counter["n"] = v + 1

    threads = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert counter["n"] == 800
    assert not path.exists()


def test_run_lock_wait_true_does_not_crash(tmp_path):
    lock = locking.run_lock(tmp_path, "download", wait=True)
    try:
        assert (tmp_path / ".download.lock").exists()
    finally:
        lock.release()


def test_run_lock_heartbeat_refreshes_file(tmp_path, monkeypatch):
    lock = locking.FileLock(tmp_path / ".m.lock", timeout=1, stale_after=4.0, heartbeat=True)  # beats every 1 s
    lock.acquire()
    try:
        f = tmp_path / ".m.lock"
        old = time.time() - 1000
        os.utime(f, (old, old))
        time.sleep(2.5)
        assert time.time() - f.stat().st_mtime < 5
    finally:
        lock.release()


def test_second_run_lock_fails_fast_with_message(tmp_path):
    lock = locking.run_lock(tmp_path, "monitor")
    try:
        # same process -> re-entrant; simulate another process by a foreign fresh holder file instead
        pass
    finally:
        lock.release()
    (tmp_path / ".monitor.lock").write_text(json.dumps({"pid": 1, "space": "elsewhere", "token": "t"}))
    with pytest.raises(PipelineError, match="already working"):
        locking.run_lock(tmp_path, "monitor")
