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
    python -m sar_pipeline.storage_plan backup --round 2      # round 2 (13 Oct): every data file; then verify / clear
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


#: Round 2 (user, 13 Oct 2026: "zruri files ko gcs pr rakho ... free up the space, jitni easily ho skti"): every data
#: file of the project, code / venv / configs / docs / logs / secrets stay. Files already in the bucket elsewhere
#: (``s2_dates_masks``: the Earth Engine exports themselves) are not uploaded again; their copy there is verified.
ALL_DATA_DIRS = ("sar-rice-mapper/processed", "sar-rice-mapper/data", "data", "outputs")
#: Local folder -> the bucket folder (under the base folder) that already holds the same files.
ALREADY_IN_BUCKET = {"sar-rice-mapper/data/s2_dates_masks/": "s2_dates_masks/"}
ROUNDS = {"1": "_pod_backup_2026-10-12", "2": "_pod_backup_2026-10-13"}


def backup_list_all(root: Path = PROJECT) -> list[tuple[str, int]]:
    """(path from ``root``, size) of every file under :data:`ALL_DATA_DIRS`."""
    out = []
    for d in ALL_DATA_DIRS:
        if (Path(root) / d).exists():
            out += [(str(p.relative_to(root)), p.stat().st_size) for p in _files(Path(root) / d)]
    return sorted(out)


def gcs_name(path: str, base: str, dest: str) -> str:
    """Where a backed-up file lives in the bucket: its existing copy for :data:`ALREADY_IN_BUCKET`, else
    ``<dest>/<path>``."""
    for local, remote in ALREADY_IN_BUCKET.items():
        if path.startswith(local):
            return f"{base}/{remote}{path[len(local):]}"
    return f"{dest}/{path}"


def _manifest_path(root: Path = PROJECT, rnd: str = "1") -> Path:
    return Path(root) / "sar-rice-mapper" / "logs" / f"{ROUNDS[rnd]}_manifest.csv"


def _manifest(rnd: str, root: Path = PROJECT):
    """The round's manifest with a ``gcs`` column (round 1 was written without it: every file under its prefix)."""
    import pandas as pd

    m = pd.read_csv(_manifest_path(root, rnd))
    if "gcs" not in m:
        _, base, _ = _bucket_and_base()
        m["gcs"] = [f"{base}/{ROUNDS[rnd]}/{q}" for q in m["path"]]
    return m


def backup(processes: int = 0, threads: int = 8, root: Path = PROJECT, rnd: str = "1") -> dict:
    """Uploads the round's files (``backup_list`` / :func:`backup_list_all`); a file already in the bucket at its
    :func:`gcs_name` with the same size is skipped. The manifest (path, size, gcs) goes next to the files."""
    import csv
    import time
    from concurrent.futures import ProcessPoolExecutor

    from . import handover as ho
    from . import resources

    files = backup_list(root) if rnd == "1" else backup_list_all(root)
    b, base, key = _bucket_and_base()
    dest = f"{base}/{ROUNDS[rnd]}"
    have = {x.name: x.size for x in b.list_blobs(prefix=dest + "/")}
    for remote in set(ALREADY_IN_BUCKET.values()):
        have.update({x.name: x.size for x in b.list_blobs(prefix=f"{base}/{remote}")})
    rows = []
    for r, sz in files:
        g = gcs_name(r, base, dest)
        if not g.startswith(dest + "/") and have.get(g) != sz:      # not (or not the same) where it should be
            g = f"{dest}/{r}"
        rows.append((r, sz, g))
    mp_ = _manifest_path(root, rnd)
    mp_.parent.mkdir(parents=True, exist_ok=True)
    with open(mp_, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "size", "gcs"])
        w.writerows(rows)
    todo = [(r, sz) for r, sz, g in rows if g.startswith(dest + "/") and have.get(g) != sz]
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
    return {"files": len(files), "bytes": sum(sz for _, sz in files),
            "already_in_bucket": len(rows) - sum(g.startswith(dest + "/") for *_, g in rows),
            "sent_files": len(todo) - len(failed), "sent_gb": round(sent / 1e9, 2), "failed": failed[:20],
            "n_failed": len(failed), "minutes": round((time.time() - t0) / 60, 1), "streams": n * threads}


def _md5(path: str) -> str:
    import base64
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def verify(processes: int = 0, root: Path = PROJECT, rnd: str = "1") -> dict:
    """Size and md5 of every manifest file against its object in the bucket; writes ``<manifest>_verified.csv``."""
    from concurrent.futures import ProcessPoolExecutor

    from . import resources

    m = _manifest(rnd, root)
    b, base, _ = _bucket_and_base()
    remote = {}
    for pre in sorted({g.rsplit("/", 1)[0].split("/")[0] + "/" + g.split("/")[1] for g in m["gcs"]}):
        remote.update({x.name: (x.size, x.md5_hash) for x in b.list_blobs(prefix=pre + "/")})
    with ProcessPoolExecutor(processes or resources.detect_resources().cpus) as ex:
        md5 = list(ex.map(_md5, [str(Path(root) / p) for p in m["path"]], chunksize=64))
    m["ok"] = [remote.get(g, (None, None)) == (sz, h) for g, sz, h in zip(m["gcs"], m["size"], md5)]
    m.to_csv(str(_manifest_path(root, rnd)).replace(".csv", "_verified.csv"), index=False)
    return {"files": len(m), "ok": int(m["ok"].sum()), "bad": int((~m["ok"]).sum()),
            "ok_gb": round(float(m.loc[m["ok"], "size"].sum()) / 1e9, 2)}


def clear(root: Path = PROJECT, rnd: str = "1") -> dict:
    """Deletes the local copy of every file ``verify`` passed (and only those), then folders left empty."""
    import pandas as pd

    v = pd.read_csv(str(_manifest_path(root, rnd)).replace(".csv", "_verified.csv"))
    gone, freed, dirs = 0, 0, set()
    for p, sz in v.loc[v["ok"], ["path", "size"]].itertuples(index=False):
        f = Path(root) / p
        if f.is_file() and f.stat().st_size == sz:
            f.unlink()
            gone += 1
            freed += sz
            dirs.add(f.parent)
    removed = 0
    for d in sorted(dirs, key=lambda x: -len(x.parts)):
        while d != Path(root) and d.exists() and not any(d.iterdir()):
            d.rmdir()
            removed += 1
            d = d.parent
    return {"deleted_files": gone, "freed_gb": round(freed / 1e9, 2), "empty_folders_removed": removed,
            "kept_unverified": int((~v["ok"]).sum())}


def restore(under: str = "", root: Path = PROJECT, threads: int = 32, rnd: str = "") -> dict:
    """Downloads backed-up files whose path starts with ``under`` back to their place, from both rounds (or one),
    using each round's manifest (``gcs`` column: a file may live in its original bucket folder); files already there
    with the same size are skipped."""
    from concurrent.futures import ThreadPoolExecutor

    b, _, _ = _bucket_and_base()
    rows = []
    for r in ([rnd] if rnd else list(ROUNDS)):
        if _manifest_path(root, r).exists():
            m = _manifest(r, root)
            rows += [(p, sz, g) for p, sz, g in m[["path", "size", "gcs"]].itertuples(index=False)
                     if p.startswith(under)]

    def one(row):
        p, sz, g = row
        f = Path(root) / p
        if f.exists() and f.stat().st_size == sz:
            return 0
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f".{f.name}.part")         # an interrupted download never leaves a half file in place
        b.blob(g).download_to_filename(str(tmp))
        tmp.replace(f)
        return sz

    with ThreadPoolExecutor(threads) as ex:
        got = sum(ex.map(one, rows))
    return {"files": len(rows), "downloaded_gb": round(got / 1e9, 2)}


def aoi_prefixes(aoi: int) -> list[str]:
    """Paths (from the project folder) that the inspection notebook reads for one AOI (measured with ``trace``)."""
    r, b = "sar-rice-mapper/", "sar-rice-mapper/processed/_batch/s2_2026"
    out = [f"{r}data/s2_dates_masks/aoi{aoi}/", f"{r}data/aoi/", f"{r}processed/aoi{aoi}/", f"{b}/aoi{aoi}/",
           f"{b}/baseline_v3/aoi{aoi}_", f"{b}/report/final_audit/", f"{b}/rice_fresh/aoi{aoi}/",
           f"{b}/rice_fresh/labels_v2/", f"{b}/rice_fresh/locked/aoi{aoi}/", f"{b}/fields/aoi{aoi}_",
           f"{r}processed/_batch/{KEEP_SERIES}/aoi{aoi}/"]
    if aoi >= 1000:                                            # the second AOI set's own AOI file
        out.append(f"data/aoi_batch2_2026-09-29/split/b2_{aoi}/")
    return out


_ROUND2 = None


def ensure_aoi(aoi: int, root: Path = PROJECT, threads: int = 32, log=print) -> dict:
    """Downloads, from the round-2 backup, the files the inspection notebook needs for ``aoi`` that are not on the
    pod (user, 14 Oct 2026: "jis aoi k lye data chaiye hoga wo download ho ajye run time pr"). A file already there
    with the same size is not fetched again, so the second call for an AOI costs one look at the disk."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    global _ROUND2
    if _ROUND2 is None:
        mp_ = _manifest_path(root, "2")
        if not mp_.exists():
            return {"aoi": aoi, "files": 0, "downloaded_gb": 0.0}
        _ROUND2 = _manifest("2", root)
    pre = tuple(aoi_prefixes(aoi))
    m = _ROUND2[_ROUND2["path"].str.startswith(pre)]
    todo = [(p, sz, g) for p, sz, g in m[["path", "size", "gcs"]].itertuples(index=False)
            if not ((Path(root) / p).is_file() and (Path(root) / p).stat().st_size == sz)]
    if not todo:
        return {"aoi": aoi, "files": 0, "downloaded_gb": 0.0}
    t0 = time.time()
    log(f"aoi{aoi}: fetching {len(todo)} files ({sum(t[1] for t in todo) / 1e9:.2f} GB) from the backup ...")
    b, _, _ = _bucket_and_base()

    def one(row):
        p, sz, g = row
        f = Path(root) / p
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f".{f.name}.part")
        b.blob(g).download_to_filename(str(tmp))
        tmp.replace(f)
        return sz

    with ThreadPoolExecutor(threads) as ex:
        got = sum(ex.map(one, todo))
    log(f"aoi{aoi}: done in {time.time() - t0:.0f} s")
    return {"aoi": aoi, "files": len(todo), "downloaded_gb": round(got / 1e9, 2)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["trace", "list", "backup", "verify", "clear", "restore"])
    p.add_argument("--under", default="", help="restore: only files under this path (from the project folder)")
    p.add_argument("--restore-round", default="", choices=["", *ROUNDS], help="restore: one round only (default both)")
    p.add_argument("--threads", type=int, default=8, help="backup: threads per process")
    p.add_argument("--round", dest="rnd", default="1", choices=list(ROUNDS),
                   help="list / backup / verify / clear: 1 = files not needed (12 Oct), 2 = all data files (13 Oct)")
    p.add_argument("--field", action="append", default=[], help="field ids for inspect_field")
    p.add_argument("--aoi", type=int, action="append", default=[], help="AOIs for inspect_pixel (with --pixel)")
    p.add_argument("--pixel", type=int, action="append", default=[])
    a = p.parse_args(argv)
    if a.step == "list":
        f = backup_list() if a.rnd == "1" else backup_list_all()
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
        fn = {"backup": lambda: backup(threads=a.threads, rnd=a.rnd), "verify": lambda: verify(rnd=a.rnd),
              "clear": lambda: clear(rnd=a.rnd),
              "restore": lambda: restore(a.under, rnd=a.restore_round)}[a.step]
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
