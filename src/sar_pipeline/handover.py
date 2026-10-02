"""Hand a whole project folder over through Google Cloud Storage, exactly as it is on disk.

Why
---
The project is handed over to a colleague who continues on another machine (user, 2 Oct 2026: "upload the whole base
folder as it is"). ``gsutil`` / ``gcloud`` are not installed on the pod, a GCS object store has no symbolic links, and
the folder holds ~160k files. This module:

* ``upload``: every regular file under ``root`` goes to ``gs://<bucket>/<prefix>/<root name>/<relative path>`` with
  many threads; files already uploaded with the same size are skipped, so an interrupted upload simply resumes.
  Symbolic links are NOT followed (a link to a 2.3 GB file would be uploaded twice, a venv's ``python`` link would
  point at the wrong machine); they are recorded in ``_handover/symlinks.json`` together with a manifest of every
  file and its size (``_handover/manifest.csv``), both uploaded too.
* ``verify``: compares the bucket listing with the manifest (count and size of every file).
* ``download``: the reverse on the new machine, then recreates the symbolic links (``restore_links``).

Use (from ``sar-rice-mapper/``; the service account key in ``secrets/`` must exist)::

    python -m sar_pipeline.handover upload   --root <base folder> --bucket <bucket> --prefix <folder in bucket>
    python -m sar_pipeline.handover verify   --root ... --bucket ... --prefix ...
    python -m sar_pipeline.handover download --root <base folder> --bucket ... --prefix ... --key <key.json>

The private handover note names the real bucket, folder and key.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

META = "_handover"


def _client(key: str | None, pool: int = 10):
    """A storage client whose HTTP pool holds ``pool`` connections: the default 10 made 32 upload threads queue for a
    connection (2 Oct: 16-23 MB/s on small files)."""
    import requests
    from google.cloud import storage

    client = storage.Client.from_service_account_json(key) if key else storage.Client()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=3)
    client._http.mount("https://", adapter)
    return client


def _upload_part(args):
    """One process's share of the upload (its own client and thread pool); returns (sent bytes, failed list)."""
    root, bucket, dest, key, workers, items, part = args
    root_p = Path(root)
    b = _client(key, pool=workers).bucket(bucket)
    sent, failed, t0 = 0, [], time.time()

    def one(item):
        rel, size = item
        blob = b.blob(f"{dest}/{rel}")
        if size > 100 * 2 ** 20:
            blob.chunk_size = 64 * 2 ** 20
        err = None
        for attempt in range(4):
            try:
                blob.upload_from_filename(str(root_p / rel), timeout=600)
                return item, None
            except Exception as e:
                err = e
                time.sleep(2 ** attempt)
        return item, err

    with ThreadPoolExecutor(workers) as pool:
        for n, f in enumerate(as_completed([pool.submit(one, it) for it in items]), 1):
            (rel, size), err = f.result()
            if err:
                failed.append((rel, repr(err)))
            else:
                sent += size
            if n % 2000 == 0 or n == len(items):
                print(f"[part {part}] {n}/{len(items)} files, {sent / 1e9:.1f} GB, "
                      f"{sent / max(time.time() - t0, 1) / 1e6:.0f} MB/s, {len(failed)} failed", flush=True)
    return sent, failed


def scan(root: Path) -> tuple[list[tuple[str, int]], list[dict]]:
    """(files as (relative path, size), symlinks as {path, target}) under ``root``; links are not followed."""
    files, links = [], []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        for name in list(dirnames):
            p = d / name
            if p.is_symlink():
                links.append({"path": str(p.relative_to(root)), "target": os.readlink(p)})
                dirnames.remove(name)
        for name in filenames:
            p = d / name
            rel = str(p.relative_to(root))
            if p.is_symlink():
                links.append({"path": rel, "target": os.readlink(p)})
            elif p.is_file():
                files.append((rel, p.stat().st_size))
    return sorted(files), links


def _existing(bucket, prefix: str) -> dict:
    return {b.name: b.size for b in bucket.list_blobs(prefix=prefix + "/")}


def upload(root: str, bucket: str, prefix: str, key: str | None = None, workers: int = 32,
           processes: int = 8) -> dict:
    """Upload ``root`` (see the module docstring); returns counts and bytes."""
    root_p = Path(root).resolve()
    dest = f"{prefix.strip('/')}/{root_p.name}"
    client = _client(key)
    b = client.bucket(bucket)
    files, links = scan(root_p)
    meta_dir = root_p / META
    meta_dir.mkdir(exist_ok=True)
    (meta_dir / "symlinks.json").write_text(json.dumps(links, indent=1))
    with open(meta_dir / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "size"])
        w.writerows(files)
    files, links = scan(root_p)                     # now including the two metadata files
    have = _existing(b, dest)
    todo = [(r, s) for r, s in files if have.get(f"{dest}/{r}") != s]
    total = sum(s for _, s in todo)
    print(f"{len(files)} files ({sum(s for _, s in files) / 1e9:.1f} GB), {len(links)} links; "
          f"{len(files) - len(todo)} already there; uploading {len(todo)} ({total / 1e9:.1f} GB)", flush=True)
    # spread over processes (each with its own threads and connections); big files dealt round-robin first so every
    # process gets a fair share of bytes, small files after
    todo.sort(key=lambda it: -it[1])
    parts = [todo[i::processes] for i in range(processes)]
    from concurrent.futures import ProcessPoolExecutor

    failed, sent = [], 0
    with ProcessPoolExecutor(processes) as pp:
        for s_, f_ in pp.map(_upload_part, [(str(root_p), bucket, dest, key, workers, part, i)
                                           for i, part in enumerate(parts)]):
            sent += s_
            failed += f_
    print(f"done: {sent / 1e9:.1f} GB sent, {len(failed)} failed", flush=True)
    if failed:
        (meta_dir / "failed.json").write_text(json.dumps(failed, indent=1))
    return {"files": len(files), "uploaded": len(todo) - len(failed), "failed": len(failed), "links": len(links)}


def verify(root: str, bucket: str, prefix: str, key: str | None = None) -> dict:
    """Every file of the manifest is in the bucket with the same size."""
    root_p = Path(root).resolve()
    dest = f"{prefix.strip('/')}/{root_p.name}"
    have = _existing(_client(key).bucket(bucket), dest)
    with open(root_p / META / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    missing = [r["path"] for r in rows if have.get(f"{dest}/{r['path']}") != int(r["size"])]
    return {"manifest_files": len(rows), "in_bucket": len(have), "missing_or_wrong_size": len(missing),
            "examples": missing[:10]}


def restore_links(root: str) -> int:
    """Recreate the symbolic links recorded at upload time; returns how many were made."""
    root_p = Path(root)
    n = 0
    for link in json.loads((root_p / META / "symlinks.json").read_text()):
        p = root_p / link["path"]
        if p.is_symlink() or p.exists():
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(link["target"], p)
        n += 1
    return n


def download(root: str, bucket: str, prefix: str, key: str | None = None, workers: int = 32) -> dict:
    """Download the uploaded folder into ``root`` (its parent is created), then recreate the links."""
    root_p = Path(root)
    dest = f"{prefix.strip('/')}/{root_p.name}"
    b = _client(key, pool=workers).bucket(bucket)
    blobs = list(b.list_blobs(prefix=dest + "/"))

    def one(blob):
        target = root_p / blob.name[len(dest) + 1:]
        if target.exists() and target.stat().st_size == blob.size:
            return 0
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        return 1

    with ThreadPoolExecutor(workers) as pool:
        got = sum(pool.map(one, blobs))
    return {"objects": len(blobs), "downloaded": got, "links": restore_links(root)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["upload", "verify", "download", "restore-links"])
    p.add_argument("--root", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True, help="folder inside the bucket; the root's own name is appended")
    p.add_argument("--key", help="service-account key file (default: application default credentials)")
    p.add_argument("--workers", type=int, default=32, help="upload threads per process")
    p.add_argument("--processes", type=int, default=8)
    a = p.parse_args(argv)
    key = a.key if a.key and Path(a.key).exists() else None
    if a.step == "upload":
        print(upload(a.root, a.bucket, a.prefix, key, a.workers, a.processes))
    elif a.step == "verify":
        print(verify(a.root, a.bucket, a.prefix, key))
    elif a.step == "download":
        print(download(a.root, a.bucket, a.prefix, key, a.workers))
    else:
        print({"links": restore_links(a.root)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
