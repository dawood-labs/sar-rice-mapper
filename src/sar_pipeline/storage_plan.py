"""Which local files the inspection notebook needs, and how much of the rest could move to cloud storage.

Why (user, 12 Oct 2026): the pod's disk is 86 % full and most of it is this project. The user wants to keep locally
only what notebook 08 (``qgis_review.inspect_field`` / ``inspect_pixel``: one field's or one pixel's data) needs, back
the rest up to the project's cloud bucket, and clear it from the pod.

Guessing the dependencies is how files go missing, so they are measured: :func:`trace_reads` runs the two inspect
functions and records every file they open (Python ``open`` through an audit hook; rasterio, pyogrio / geopandas and
parquet readers wrapped while they run).

Read only: nothing is moved or deleted here.

Run::

    python -m sar_pipeline.storage_plan trace --field aoi66_p000013 --aoi 33 --pixel 9412
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def trace_reads(fn, *args, **kw) -> set[str]:
    """Paths (relative to the working directory when inside it) of every file ``fn(*args, **kw)`` opens.

    Python ``open`` (numpy, csv, json) is seen by an audit hook; rasters, GeoPackages and parquet files are opened in
    C, so ``rasterio.open``, the pyogrio readers and the parquet readers are wrapped for the duration."""
    import importlib

    seen: set[str] = set()

    def hook(event, a):
        if event == "open" and a and isinstance(a[0], (str, bytes, os.PathLike)):
            seen.add(os.fsdecode(a[0]))

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

    sys.addaudithook(hook)
    try:
        for mod, name in targets:
            m = importlib.import_module(mod)
            saved.append((m, name, getattr(m, name)))
            setattr(m, name, wrap(getattr(m, name)))
        fn(*args, **kw)
    finally:
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
                (sys.base_prefix, "/usr", "/opt", "/proc", "/etc", "/tmp")):
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["trace"])
    p.add_argument("--field", action="append", default=[], help="field ids for inspect_field")
    p.add_argument("--aoi", type=int, action="append", default=[], help="AOIs for inspect_pixel (with --pixel)")
    p.add_argument("--pixel", type=int, action="append", default=[])
    a = p.parse_args(argv)
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
