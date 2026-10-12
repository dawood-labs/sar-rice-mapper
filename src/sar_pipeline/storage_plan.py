"""Which local files the inspection notebook needs, and how much of the rest could move to cloud storage.

Why (user, 12 Oct 2026): the pod's disk is 86 % full and most of it is this project. The user wants to keep locally
only what notebook 08 (``qgis_review.inspect_field`` / ``inspect_pixel``: one field's or one pixel's data) needs, back
the rest up to the project's cloud bucket, and clear it from the pod.

Guessing the dependencies is how files go missing, so they are measured: :func:`trace_reads` runs the two inspect
functions and records every file they open (Python ``open`` through an audit hook; rasterio, pyogrio / geopandas and
parquet readers wrapped while they run).

The move itself (user, 12 Oct 2026: "70 GB wala", "use maximum threads"):

1. ``list``: the files to move (:func:`backup_list`, rules in :data:`MOVE_DIRS` and below), with sizes;
2. ``backup``: upload them to ``gs://<bucket>/<base>/<BACKUP>/<path from the project folder>``, many processes with
   threads each (``handover._upload_part``), and the manifest beside them;
3. ``verify``: every file's size AND md5 compared with the bucket; the result written next to the manifest;
4. ``clear``: deletes the local copy of exactly the files ``verify`` passed, then the empty folders;
5. ``restore``: downloads files back (all, or those under a path).

Run::

    python -m sar_pipeline.storage_plan trace --field aoi66_p000013 --aoi 33 --pixel 9412
    python -m sar_pipeline.storage_plan list
    python -m sar_pipeline.storage_plan backup
    python -m sar_pipeline.storage_plan verify
    python -m sar_pipeline.storage_plan clear
    python -m sar_pipeline.storage_plan restore --under sar-rice-mapper/processed/_batch/s2_2026/sar_only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


#: The sets of the traces running now. An audit hook can never be removed, so one hook is installed once and feeds them.
_ACTIVE: list[set[str]] = []


def _hook_once() -> None:
    if getattr(_hook_once, "done", False):
        return

    def hook(event, a):
        if _ACTIVE and event == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)):
            for s in _ACTIVE:
                s.add(os.fsdecode(a[0]))

    sys.addaudithook(hook)
    _hook_once.done = True


def trace_reads(fn, *args, **kw) -> set[str]:
    """Paths (relative to the working directory when inside it) of every file ``fn(*args, **kw)`` opens.

    Python ``open`` (numpy, csv, json) is seen by an audit hook; rasters, GeoPackages and parquet files are opened in
    C, so ``rasterio.open``, the pyogrio readers and the parquet readers are wrapped for the duration."""
    import importlib

    seen: set[str] = set()
    _hook_once()
    targets = [("rasterio", "open"), ("pyogrio", "read_dataframe"), ("pyogrio", "read_info"),
               ("pyogrio", "read_arrow"), ("pyogrio.raw", "read"), ("pyogrio.raw", "read_arrow"),
               ("pyarrow.parquet", "read_table"), ("pyarrow.parquet", "ParquetFile")]
    saved = []

    def wrap(f):
        def w(path, *a, **k):
            if isinstance(path, (str, os.PathLike)):
                seen.add(os.fspath(path))
            return f(path, *a, **k)
        return w

    _ACTIVE.append(seen)
    try:
        for mod, name in targets:
            m = importlib.import_module(mod)
            saved.append((m, name, getattr(m, name)))
            setattr(m, name, wrap(getattr(m, name)))
        fn(*args, **kw)
    finally:
        _ACTIVE.remove(seen)
        for m, name, f in saved:
            setattr(m, name, f)
    cwd = Path.cwd().resolve()
    out = set()
    for p in seen:
        try:
            r = Path(p).resolve()
        except (OSError, ValueError):
            continue
        if r.is_file() and "site-packages" not in str(r) and not str(r).startswith(
                (sys.base_prefix, "/usr", "/opt", "/proc", "/etc")):
            out.add(str(r.relative_to(cwd)) if r.is_relative_to(cwd) else str(r))
    return out


def with_vrt_sources(paths: set[str]) -> set[str]:
    """Adds the files a ``.vrt`` reads (its ``SourceFilename`` entries, resolved like GDAL does): the VRT is only an
    index, the pixels live in those files, and GDAL opens them in C where :func:`trace_reads` cannot see."""
    import re

    out = set(paths)
    cwd = Path.cwd().resolve()
    todo = [p for p in paths if p.endswith(".vrt")]
    while todo:
        v = Path(todo.pop())
        try:
            text = v.read_text()
        except OSError:
            continue
        for rel, name in re.findall(r'<SourceFilename relativeToVRT="(\d)"[^>]*>([^<]+)</SourceFilename>', text):
            src = (v.parent / name if rel == "1" else Path(name)).resolve()
            s = str(src.relative_to(cwd)) if src.is_relative_to(cwd) else str(src)
            if s not in out:
                out.add(s)
                if s.endswith(".vrt"):
                    todo.append(s)
    return out


def patterns(paths) -> dict[str, int]:
    """Paths collapsed to patterns (``aoi<N>``, ``<date>``, ``<id>``) with how many files each stands for."""
    import re
    from collections import Counter

    c = Counter()
    for p in paths:
        q = re.sub(r"aoi\d+", "aoi<N>", p)
        q = re.sub(r"20\d{6}|20\d\d-\d\d-\d\d|20\d\d-\d\d", "<date>", q)
        q = re.sub(r"_[0-9a-f]{12}\.npy$", "_<hash>.npy", q)
        q = re.sub(r"p?\d{6}", "<id>", q)
        c[q] += 1
    return dict(sorted(c.items()))


PROJECT = Path(__file__).resolve().parents[3]          # the project folder that holds the repository
SR = "sar-rice-mapper/processed"
BACKUP = "_pod_backup_2026-10-12"
#: Whole folders not read by the inspection notebook (``trace``) nor by the delivery: old experiments, review images,
#: superseded deliveries, past seasons (12 Oct 2026).
MOVE_DIRS = (f"{SR}/_batch/s2_2026/sar_only", f"{SR}/_batch/s2_2026/delivery_v3_baseline",
             f"{SR}/_batch/s2_2026/review", f"{SR}/_batch/s2_2026/review_v2", f"{SR}/_batch/s2_2026/baseline_v4",
             f"{SR}/_batch/s2_2026/qgis_review", f"{SR}/_batch/s2_2026/rice_fresh_v2",
             f"{SR}/_batch/s2_2026/rice_fresh_v3", f"{SR}/_batch/s1_may_to_latest_2026-09-29",
             f"{SR}/_batch/optical_v1", f"{SR}/_batch/optical_v2", f"{SR}/_batch/optical_v3",
             f"{SR}/_batch/model_v1", f"{SR}/_batch/model_v2", f"{SR}/_batch/sar_v1",
             "sar-rice-mapper/data/s2_windows5d", "reference")
#: The Sentinel-2 series the maps are pinned to stays; every other mask experiment moves.
KEEP_SERIES = "s2_2026_hyb40m1late_20261001"
#: Old per-AOI map files: only these three versions are read (``qgis_review.VERSIONS``).
KEEP_OLD_MAPS = ("_monsoon2026.tif", "_monsoon2026_final.tif", "_monsoon2026_v1.tif")


def _files(d: Path):
    for dirpath, _, names in os.walk(d, followlinks=False):
        for n in names:
            p = Path(dirpath) / n
            if p.is_file() and not p.is_symlink():
                yield p


def backup_list(root: Path = PROJECT) -> list[tuple[str, int]]:
    """(path from ``root``, size) of every file to move."""
    root = Path(root)
    picked: list[Path] = []
    for d in MOVE_DIRS:
        if (root / d).exists():
            picked += _files(root / d)
    batch = root / SR / "_batch"
    for d in batch.glob("s2_2026_*"):
        if d.is_dir() and d.name != KEEP_SERIES:
            picked += _files(d)
    fresh = batch / "s2_2026" / "rice_fresh"
    for d in fresh.glob("aoi*/field_review/sheets"):
        picked += _files(d)
    rep = batch / "s2_2026" / "report"
    for p in _files(rep):
        if not p.relative_to(rep).parts[0] == "final_audit":     # bad_passes_all.csv is read
            picked.append(p)
    for d in (batch / "s2_2026").glob("aoi*"):
        if d.is_dir():
            picked += [p for p in _files(d) if not p.name.endswith(KEEP_OLD_MAPS)]
    for a in (root / SR).glob("aoi*"):
        picked += _files(a / "monsoon2025")
        y = a / "year2025"
        picked += [p for p in _files(y) if p.relative_to(y).parts[0] not in ("grid", "LATEST.txt")]
        m = a / "monsoon2026"
        keep = {(m / n).read_text().strip().rsplit("/", 1)[-1] for n in ("ANALYSIS_RUN.txt", "LATEST.txt")
                if (m / n).exists()}
        for r in (m / "runs").glob("*"):
            if r.is_dir() and r.name not in keep:
                picked += _files(r)
    seen, out = set(), []
    for p in picked:
        rel = str(p.relative_to(root))
        if rel not in seen:
            seen.add(rel)
            out.append((rel, p.stat().st_size))
    return sorted(out)


def _bucket_and_base():
    import yaml

    from .handover import _client

    cfg = yaml.safe_load((PROJECT / "sar-rice-mapper" / "config" / "aoi19_monsoon2026.yaml").read_text())
    key = str(PROJECT / "sar-rice-mapper" / cfg["auth"]["key_file"])
    return _client(key, pool=64).bucket(cfg["gcs"]["bucket"]), cfg["gcs"]["base_folder"], key


def _manifest_path(root: Path = PROJECT) -> Path:
    return Path(root) / "sar-rice-mapper" / "logs" / f"{BACKUP}_manifest.csv"


def backup(processes: int = 0, threads: int = 8, root: Path = PROJECT) -> dict:
    """Uploads :func:`backup_list` (files already in the bucket with the same size are skipped), then the manifest."""
    import csv
    import time
    from concurrent.futures import ProcessPoolExecutor

    from . import handover as ho
    from . import resources

    files = backup_list(root)
    mp_ = _manifest_path(root)
    mp_.parent.mkdir(parents=True, exist_ok=True)
    with open(mp_, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "size"])
        w.writerows(files)
    b, base, key = _bucket_and_base()
    dest = f"{base}/{BACKUP}"
    have = {x.name: x.size for x in b.list_blobs(prefix=dest + "/")}
    todo = [(r, s) for r, s in files if have.get(f"{dest}/{r}") != s]
    n = processes or resources.detect_resources().cpus
    # big files first, dealt round-robin, so every process gets a similar share of bytes
    todo.sort(key=lambda x: -x[1])
    parts = [todo[i::n] for i in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(n) as ex:
        res = list(ex.map(ho._upload_part, [(str(root), b.name, dest, key, threads, p, i) for i, p in enumerate(parts)]))
    b.blob(f"{dest}/_manifest.csv").upload_from_filename(str(mp_))
    sent = sum(r[0] for r in res)
    failed = [f for r in res for f in r[1]]
    return {"files": len(files), "bytes": sum(s for _, s in files), "sent_files": len(todo) - len(failed),
            "sent_gb": round(sent / 1e9, 2), "failed": failed[:20], "n_failed": len(failed),
            "minutes": round((time.time() - t0) / 60, 1), "streams": n * threads}


def _md5(path: str) -> str:
    import base64
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def verify(processes: int = 0, root: Path = PROJECT) -> dict:
    """Size and md5 of every manifest file against the bucket; writes ``<manifest>_verified.csv`` (path, ok)."""
    from concurrent.futures import ProcessPoolExecutor

    import pandas as pd

    from . import resources

    m = pd.read_csv(_manifest_path(root))
    b, base, _ = _bucket_and_base()
    dest = f"{base}/{BACKUP}"
    remote = {x.name[len(dest) + 1:]: (x.size, x.md5_hash) for x in b.list_blobs(prefix=dest + "/")}
    exists = [Path(root) / p for p in m["path"]]
    with ProcessPoolExecutor(processes or resources.detect_resources().cpus) as ex:
        md5 = list(ex.map(_md5, [str(p) for p in exists], chunksize=64))
    ok = [remote.get(p, (None, None)) == (s, h) for p, s, h in zip(m["path"], m["size"], md5)]
    m["ok"] = ok
    m.to_csv(str(_manifest_path(root)).replace(".csv", "_verified.csv"), index=False)
    return {"files": len(m), "ok": int(sum(ok)), "bad": int(len(m) - sum(ok)),
            "ok_gb": round(float(m.loc[m["ok"], "size"].sum()) / 1e9, 2)}


def clear(root: Path = PROJECT) -> dict:
    """Deletes the local copy of every file ``verify`` passed (and only those), then folders left empty."""
    import pandas as pd

    v = pd.read_csv(str(_manifest_path(root)).replace(".csv", "_verified.csv"))
    gone, freed, dirs = 0, 0, set()
    for p, s in v.loc[v["ok"], ["path", "size"]].itertuples(index=False):
        f = Path(root) / p
        if f.is_file() and f.stat().st_size == s:
            f.unlink()
            gone += 1
            freed += s
            dirs.add(f.parent)
    removed = 0
    for d in sorted(dirs, key=lambda x: -len(x.parts)):
        while d != Path(root) and d.exists() and not any(d.iterdir()):
            d.rmdir()
            removed += 1
            d = d.parent
    return {"deleted_files": gone, "freed_gb": round(freed / 1e9, 2), "empty_folders_removed": removed,
            "kept_unverified": int((~v["ok"]).sum())}


def restore(under: str = "", root: Path = PROJECT, threads: int = 32) -> dict:
    """Downloads backed-up files (those whose path starts with ``under``) back to their place; existing files with
    the same size are skipped."""
    from concurrent.futures import ThreadPoolExecutor

    b, base, _ = _bucket_and_base()
    dest = f"{base}/{BACKUP}"
    blobs = [x for x in b.list_blobs(prefix=f"{dest}/{under}".rstrip("/")) if not x.name.endswith("_manifest.csv")]

    def one(x):
        p = Path(root) / x.name[len(dest) + 1:]
        if p.exists() and p.stat().st_size == x.size:
            return 0
        p.parent.mkdir(parents=True, exist_ok=True)
        x.download_to_filename(str(p))
        return x.size

    with ThreadPoolExecutor(threads) as ex:
        got = sum(ex.map(one, blobs))
    return {"files": len(blobs), "downloaded_gb": round(got / 1e9, 2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["trace", "list", "backup", "verify", "clear", "restore"])
    p.add_argument("--under", default="", help="restore: only files under this path (from the project folder)")
    p.add_argument("--threads", type=int, default=8, help="backup: threads per process")
    p.add_argument("--field", action="append", default=[], help="field ids for inspect_field")
    p.add_argument("--aoi", type=int, action="append", default=[], help="AOIs for inspect_pixel (with --pixel)")
    p.add_argument("--pixel", type=int, action="append", default=[])
    a = p.parse_args(argv)
    if a.step == "list":
        f = backup_list()
        from collections import Counter

        top = Counter()
        for r, sz in f:
            parts = r.split("/")
            top["/".join(parts[:5] if parts[0] == "sar-rice-mapper" else parts[:2])] += sz
        for k, v in top.most_common(25):
            print(f"{v / 1e9:7.2f} GB  {k}")
        print(f"{len(f):,} files, {sum(s for _, s in f) / 1e9:.1f} GB to move")
        return 0
    if a.step in ("backup", "verify", "clear", "restore"):
        fn = {"backup": lambda: backup(threads=a.threads), "verify": verify, "clear": clear,
              "restore": lambda: restore(a.under)}[a.step]
        print(json.dumps(fn(), indent=1, default=str))
        return 0
    import matplotlib

    matplotlib.use("Agg")
    from . import qgis_review as q

    reads: set[str] = set()
    for f in a.field:
        reads |= trace_reads(q.inspect_field, f, keep_cache=False)
    for aoi, px in zip(a.aoi, a.pixel):
        reads |= trace_reads(q.inspect_pixel, aoi, px, block=3, keep_cache=False)
    reads = {r for r in with_vrt_sources(reads) if not r.endswith(".pyc")}
    print(json.dumps(patterns(reads), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
