"""Earth Engine straight to the pod ("route B"): the radar run's chunks and the Sentinel-2 dates, no Cloud Storage.

Why (user, 18 Nov 2026): from now on data comes from Earth Engine directly to the pod (``ee.data.computePixels``, as
the harvest-monitoring and standing-cane projects do) and is kept on S3; Cloud Storage is no longer the landing place.
The images themselves do not change:

* radar: the same image the batch export builds (``s1_ard.pipeline.build_chunk_image``: border noise, terrain
  flattening, Refined Lee and the multi-temporal filter, dB, nodata filled), so a run fetched here equals an exported
  run and the rules read it unchanged (user: keep our radar processing). Each planned chunk of a run (``export`` without
  ``--yes`` plans them) is fetched in tiles that stay under the request limit, written to the path the download step
  uses, checked (``download.inspect_chunk_files``) and recorded VERIFIED in the manifest and the download index, so
  ``stack`` and everything after it work as before;
* Sentinel-2: the same-day mosaic with the same bands as ``optical_export.export_image`` (+ ``clear`` = Cloud Score+
  x 100, uint16), on the AOI's grid, written as ``data/s2_dates_masks/<aoi>/<aoi>_<YYYY-MM>_S2_<date>.tif`` with the
  band names as descriptions (``optical_export.band_index`` reads them).

Requests go to the high-volume endpoint with a workload tag (cost attribution), several at a time, each retried with
back-off. Run::

    python -m sar_pipeline --config config/aoi1246_monsoon2026.yaml new-run
    python -m sar_pipeline --config config/aoi1246_monsoon2026.yaml export --run <run> --all     # plans, no --yes
    python -m sar_pipeline.direct_fetch s1 --config config/aoi1246_monsoon2026.yaml --run <run>
    python -m sar_pipeline --config config/aoi1246_monsoon2026.yaml stack --run <run>
    python -m sar_pipeline.direct_fetch s2 --config config/aoi1246_monsoon2026.yaml --start 2026-03-01 --end <tomorrow>
    source secrets/aws_env.sh && python -m sar_pipeline.direct_fetch upload --to s3://<bucket>/<prefix> \
        --pair processed/aoi1246/monsoon2026/runs/<run> s1/<run> --pair data/s2_dates_masks/aoi1246 s2
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

WORKLOAD_TAG = "sar-rice-direct"
#: Bytes one computePixels request may return (the service limit is 48 MB; kept well below).
MAX_REQUEST_BYTES = 32 * 1024 * 1024
THREADS = 16


def _init(cfg: dict) -> None:
    import ee

    from . import auth

    auth.init_ee(cfg, high_volume=True)
    ee.data.setDefaultWorkloadTag(WORKLOAD_TAG)


def _retry(fn, attempts: int = 6):
    for i in range(attempts):
        try:
            return fn()
        except Exception:                             # noqa: BLE001 - transient service errors are retried
            if i == attempts - 1:
                raise
            time.sleep(min(60, 2 ** i))


def tiles(width: int, height: int, n_bands: int, bytes_per_px: int, limit: int = MAX_REQUEST_BYTES):
    """(col_off, row_off, w, h) tiles covering width x height so that one tile of ``n_bands`` stays under ``limit``."""
    side = int(max(16, min(1024, (limit / max(1, n_bands * bytes_per_px)) ** 0.5)))
    return [(c, r, min(side, width - c), min(side, height - r))
            for r in range(0, height, side) for c in range(0, width, side)]


def fetch_array(image, crs: str, x0: float, y0: float, res: float, width: int, height: int, dtype,
                threads: int = THREADS) -> np.ndarray:
    """(bands, height, width) array of ``image`` on the grid (x0, y0 = top-left corner), fetched tile by tile."""
    import ee

    names = image.bandNames().getInfo()
    out = np.empty((len(names), height, width), dtype=dtype)

    def one(t):
        c, r, w, h = t
        req = {"expression": image, "fileFormat": "NUMPY_NDARRAY", "bandIds": names,
               "grid": {"dimensions": {"width": w, "height": h}, "crsCode": crs,
                        "affineTransform": {"scaleX": res, "shearX": 0, "translateX": x0 + c * res,
                                            "shearY": 0, "scaleY": -res, "translateY": y0 - r * res}}}
        a = _retry(lambda: ee.data.computePixels(req))
        out[:, r:r + h, c:c + w] = np.stack([a[n] for n in names]).astype(dtype)

    ts = tiles(width, height, len(names), np.dtype(dtype).itemsize)
    with ThreadPoolExecutor(min(threads, len(ts))) as ex:
        list(ex.map(one, ts))
    return out


def _write(path: Path, arr: np.ndarray, crs: str, x0: float, y0: float, res: float, nodata, names=None) -> None:
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with rasterio.open(tmp, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                       dtype=arr.dtype, crs=crs, transform=from_origin(x0, y0, res, res), nodata=nodata,
                       compress="deflate", tiled=True) as ds:
        ds.write(arr)
        if names:
            ds.descriptions = tuple(names)
    tmp.replace(path)


def s1_run(config: str, run: str, threads: int = THREADS) -> dict:
    """Every planned chunk of a radar run fetched to ``raw_chunks/track_<t>/<chunk>.tif`` and recorded VERIFIED."""
    from . import config as cm
    from . import download as dl
    from . import export as ex
    from . import manifest as mf
    from .s1_ard import pipeline as ard

    cfg = cm.load_config(config)
    run_dir = Path(cm.analysis_run_dir(cfg)).parent / run
    _init(cfg)
    man = mf.read_manifest(run_dir)
    todo = man[man["state"].isin(["PLANNED", "PENDING"])]
    gd, selected = ex._load_run_inputs(cfg, run_dir, ex._session(run_dir, None))
    nodata = ex.nodata_value(cfg)
    dtype = np.int16 if cfg["export"]["dtype"] == "int16" else np.float32
    done = []
    for _, row in todo.iterrows():
        chunk = ex._chunk_for_row(gd, row["chunk_name"])
        acq = selected[row["track_id"]].sort_values("datetime_utc").reset_index(drop=True)
        image = ard.build_chunk_image(cfg, gd, chunk, row["track_id"], acq)
        arr = fetch_array(image, gd["crs"], float(chunk["xmin"]), float(chunk["ymax"]), float(gd["res"]),
                          int(chunk["width"]), int(chunk["height"]), dtype, threads)
        rel = Path("raw_chunks") / f"track_{row['track_id']}" / f"{row['chunk_name']}.tif"
        _write(run_dir / rel, arr, gd["crs"], float(chunk["xmin"]), float(chunk["ymax"]), float(gd["res"]), nodata)
        status, reason = dl.inspect_chunk_files([run_dir / rel], gd, chunk, arr.shape[0], nodata, cfg["export"]["dtype"])
        if status == "bad":
            raise RuntimeError(f"{row['task_key']}: fetched file failed the check: {reason}")
        state = "VERIFIED_EMPTY" if status == "empty" else "VERIFIED"
        p = run_dir / rel
        dl._append_index(run_dir, [{"files": [{"crc32c": dl.file_crc32c_b64(p), "local_path": str(rel),
                                               "mtime_ns": p.stat().st_mtime_ns, "size": p.stat().st_size}],
                                    "state": state, "task_key": row["task_key"], "verified_at": mf.now_utc()}])
        mf.update_manifest(run_dir, [dict(task_key=row["task_key"], state=state, local_paths=str(rel),
                                          last_error="", error_class="")], return_df=False)
        done.append({"task_key": row["task_key"], "bands": int(arr.shape[0]), "state": state})
    return {"run": run, "chunks": done}


def s2_dates(config: str, start: str, end: str, folder: str = "data/s2_dates_masks", threads: int = THREADS,
             bands=None) -> list[dict]:
    """Every Sentinel-2 date over the AOI in [start, end) fetched on its grid; a date already on the pod is kept."""
    import ee
    import geopandas as gpd

    from . import config as cm
    from . import optical_export as oe

    cfg = cm.load_config(config)
    _init(cfg)
    key = cfg["aoi"]["key"]
    gd = json.loads((cm.grid_dir(cfg) / "grid_def.json").read_text())
    poly = oe.aoi_geojson_2d(gpd.read_file(cm.aoi_path(cfg)))
    dates = sorted(oe.scene_dates(poly, start, end))
    bands = list(bands or oe.BANDS_WITH_MASKS)
    out_dir = cm.repo_root() / folder / key
    rows = []
    for d in dates:
        path = out_dir / f"{key}_{d[:7]}_S2_{d}.tif"
        if path.exists():
            rows.append({"date": d, "state": "kept"})
            continue
        day = ee.Date(d)
        region = ee.Geometry(poly)
        s2 = (ee.ImageCollection(oe.S2_COLLECTION).filterBounds(region).filterDate(day, day.advance(1, "day"))
              .linkCollection(ee.ImageCollection(oe.CLOUD_SCORE_COLLECTION), ["cs_cdf"]))
        mosaic = s2.mosaic()
        image = mosaic.select(bands).addBands(mosaic.select("cs_cdf").multiply(100).rename("clear")).toUint16()
        arr = fetch_array(image, gd["crs"], float(gd["x0"]), float(gd["y0"]), float(gd["res"]), int(gd["width"]),
                          int(gd["height"]), np.uint16, threads)
        _write(path, arr, gd["crs"], float(gd["x0"]), float(gd["y0"]), float(gd["res"]), 0, [*bands, "clear"])
        rows.append({"date": d, "state": "fetched", "file": str(path.relative_to(cm.repo_root()))})
    return rows


def s3_sync(pairs, prefix: str) -> dict:
    """``aws s3 sync`` of each (local folder, sub-folder) to ``<prefix>/<sub-folder>/``, then every local file's size
    compared with its object (credentials from the environment). Why: the data is kept on S3 (user, 18 Nov 2026)."""
    import subprocess

    prefix = prefix.rstrip("/")
    files, ok = 0, 0
    for local, sub in pairs:
        local = Path(local)
        dest = f"{prefix}/{sub.strip('/')}"
        subprocess.run(["aws", "s3", "sync", "--only-show-errors", str(local), dest + "/"], check=True)
        ls = subprocess.run(["aws", "s3", "ls", "--recursive", dest + "/"], check=True, capture_output=True,
                            text=True).stdout
        key_root = dest.split("/", 3)[3] + "/"
        remote = {ln.split(maxsplit=3)[3][len(key_root):]: int(ln.split()[2]) for ln in ls.splitlines()
                  if len(ln.split(maxsplit=3)) == 4}
        for f in local.rglob("*"):
            if f.is_file():
                files += 1
                ok += remote.get(str(f.relative_to(local))) == f.stat().st_size
    return {"files": files, "size_ok": ok, "s3": prefix + "/"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("what", choices=["s1", "s2", "upload"])
    p.add_argument("--to", help="upload: S3 prefix (s3://bucket/...)")
    p.add_argument("--pair", nargs=2, action="append", metavar=("LOCAL", "SUB"), help="upload: local folder and its sub-folder")
    p.add_argument("--config", help="s1 / s2: the AOI config")
    p.add_argument("--run", help="s1: the planned run id")
    p.add_argument("--start", help="s2: first date")
    p.add_argument("--end", help="s2: end date (exclusive)")
    p.add_argument("--threads", type=int, default=THREADS)
    a = p.parse_args(argv)
    if a.what == "upload":
        print(json.dumps(s3_sync(a.pair, a.to), indent=1))
        return 0
    if a.what == "s1":
        print(json.dumps(s1_run(a.config, a.run, a.threads), indent=1))
    else:
        r = s2_dates(a.config, a.start, a.end, threads=a.threads)
        print(pd.DataFrame(r).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
