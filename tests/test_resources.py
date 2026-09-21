"""resources.py against fake /proc and /sys trees (cgroup v1, v2, nesting, page cache, fallbacks)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sar_pipeline import resources as r

GB = 1024**3


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel.lstrip("/")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def fake_sys(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "_SYS_ROOT", tmp_path)
    monkeypatch.setattr(r.os, "sched_getaffinity", lambda pid: set(range(16)), raising=False)
    _write(tmp_path, "/proc/meminfo", f"MemTotal: {64 * GB // 1024} kB\nMemAvailable: {60 * GB // 1024} kB\n")
    return tmp_path


def test_no_cgroup_uses_host(fake_sys):
    res = r.detect_resources()
    assert res.cpus == 16
    assert res.memory_total_bytes == 64 * GB
    assert res.memory_available_bytes == 60 * GB


def test_cgroup_v2_nested_takes_tightest_ancestor(fake_sys):
    _write(fake_sys, "/proc/self/cgroup", "0::/kubepods/pod1/c1\n")
    # own cgroup: 8 GB / 4 CPUs; ancestor pod: 4 GB / 2 CPUs (tighter, must win)
    _write(fake_sys, "/sys/fs/cgroup/kubepods/pod1/c1/memory.max", str(8 * GB))
    _write(fake_sys, "/sys/fs/cgroup/kubepods/pod1/c1/memory.current", str(6 * GB))
    _write(fake_sys, "/sys/fs/cgroup/kubepods/pod1/c1/cpu.max", "400000 100000")
    _write(fake_sys, "/sys/fs/cgroup/kubepods/pod1/memory.max", str(4 * GB))
    _write(fake_sys, "/sys/fs/cgroup/kubepods/pod1/memory.current", str(3 * GB))
    _write(fake_sys, "/sys/fs/cgroup/kubepods/pod1/memory.stat", f"anon 100\ninactive_file {1 * GB}\n")
    _write(fake_sys, "/sys/fs/cgroup/kubepods/pod1/cpu.max", "200000 100000")
    res = r.detect_resources()
    assert res.cpus == 2
    assert res.memory_total_bytes == 4 * GB
    assert res.memory_available_bytes == 4 * GB - (3 * GB - 1 * GB)
    assert "cgroup" in res.notes


def test_cgroup_v2_unlimited_falls_back_to_host(fake_sys):
    _write(fake_sys, "/proc/self/cgroup", "0::/\n")
    _write(fake_sys, "/sys/fs/cgroup/memory.max", "max")
    _write(fake_sys, "/sys/fs/cgroup/cpu.max", "max 100000")
    res = r.detect_resources()
    assert (res.cpus, res.memory_total_bytes) == (16, 64 * GB)


def test_page_cache_is_not_counted_as_used(fake_sys):
    _write(fake_sys, "/proc/self/cgroup", "0::/\n")
    _write(fake_sys, "/sys/fs/cgroup/memory.max", str(16 * GB))
    _write(fake_sys, "/sys/fs/cgroup/memory.current", str(int(15.9 * GB)))  # mostly cache after a download
    _write(fake_sys, "/sys/fs/cgroup/memory.stat", f"inactive_file {12 * GB}\n")
    res = r.detect_resources()
    assert res.memory_available_bytes == pytest.approx(16 * GB - (15.9 * GB - 12 * GB), rel=1e-6)
    assert r.memory_budget_bytes(res) > 1 * GB


def test_cgroup_v1(fake_sys):
    _write(fake_sys, "/proc/self/cgroup", "4:memory:/docker/abc\n3:cpu,cpuacct:/docker/abc\n")
    _write(fake_sys, "/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes", str(2 * GB))
    _write(fake_sys, "/sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes", str(int(1.5 * GB)))
    _write(fake_sys, "/sys/fs/cgroup/memory/docker/abc/memory.stat", f"total_inactive_file {int(0.5 * GB)}\n")
    _write(fake_sys, "/sys/fs/cgroup/cpu,cpuacct/docker/abc/cpu.cfs_quota_us", "150000")
    _write(fake_sys, "/sys/fs/cgroup/cpu,cpuacct/docker/abc/cpu.cfs_period_us", "100000")
    res = r.detect_resources()
    assert res.cpus == 1
    assert res.memory_total_bytes == 2 * GB
    assert res.memory_available_bytes == 1 * GB


def test_cgroup_v1_unlimited_value_ignored(fake_sys):
    _write(fake_sys, "/proc/self/cgroup", "4:memory:/\n")
    _write(fake_sys, "/sys/fs/cgroup/memory/memory.limit_in_bytes", str(1 << 62))
    assert r.detect_resources().memory_total_bytes == 64 * GB


def test_memory_unknown_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "_SYS_ROOT", tmp_path)  # no meminfo, no cgroup
    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil -> ImportError
    res = r.detect_resources()
    assert res.memory_total_bytes == 4 * GB and "assumed" in res.notes


def test_config_caps_and_bool_guard(fake_sys):
    assert r.detect_resources({"resources": {"cpus": 3}}).cpus == 3
    assert r.detect_resources({"resources": {"cpus": True}}).cpus == 16  # bool is not an integer setting
    res = r.detect_resources()
    assert r.io_threads(res, {"resources": {"io_threads": 5}}) == 5
    assert r.io_threads(res, {"resources": {"io_threads": "auto"}}) == 32
    assert r.cpu_workers(res) == 15
    assert r.cpu_workers(res, {"resources": {"cpu_workers": 4}}) == 4
    # RAM budget limits workers: 60 GB * 0.6 = 36 GB -> at most 3 workers of 12 GB
    assert r.cpu_workers(res, None, mem_per_worker_bytes=12 * GB) == 3
    assert r.submit_threads(res, {"export": {"max_workers": "auto"}}) == 8
    assert r.submit_threads(res, {"export": {"max_workers": True}}) == 8


def test_block_rows_and_cache_bounds(fake_sys):
    res = r.detect_resources()
    assert r.block_rows(res, width=10**9, n_bands=80, bytes_per_value=8, workers=64) == 1
    rows = r.block_rows(res, width=2000, n_bands=80, bytes_per_value=4, workers=4)
    assert rows * 2000 * 80 * 4 * 3 <= r.memory_budget_bytes(res) // 4
    assert r.gdal_cache_bytes(res) >= 64 * 1024**2


def test_ancestor_headroom_limits_available_memory(fake_sys):
    """Own cgroup 16 GB with 1 GB used, but the shared parent slice has 20 GB with 19 GB used -> 1 GB free."""
    _write(fake_sys, "/proc/self/cgroup", "0::/users.slice/user-a\n")
    _write(fake_sys, "/sys/fs/cgroup/users.slice/user-a/memory.max", str(16 * GB))
    _write(fake_sys, "/sys/fs/cgroup/users.slice/user-a/memory.current", str(1 * GB))
    _write(fake_sys, "/sys/fs/cgroup/users.slice/memory.max", str(20 * GB))
    _write(fake_sys, "/sys/fs/cgroup/users.slice/memory.current", str(19 * GB))
    res = r.detect_resources()
    assert res.memory_total_bytes == 16 * GB
    assert res.memory_available_bytes == 1 * GB
