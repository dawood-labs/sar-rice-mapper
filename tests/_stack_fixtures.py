"""Synthetic run builder shared by test_stack.py and test_pixel_query.py.

A synthetic run is built by hand: grid_def.json, pixel_index.tif, band layout, manifest, audit files
and chunk GeoTIFFs with known values, including
- a chunk split by Earth Engine into two files,
- a SPLIT chunk replaced by four VERIFIED sub-chunks,
- an unverified (DOWNLOADED) chunk and a FAILED chunk,
- a date with a large nodata area, and a NaN pixel.

Not a test module (leading underscore); the test files import it after adding this folder to
sys.path, which works with every pytest import mode.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from sar_pipeline.config import grid_dir, root, season_dir
from sar_pipeline.run_meta import BAND_LAYOUT_COLUMNS

TRACK = "RO123_ASC"
TRACK2 = "RO045_DSC"
RES = 10.0
X0, Y0 = 300000.0, 1900000.0
W, H, CPX = 20, 15, 8
ACQS = [  # acquisition_id, datetime_utc, platforms
    (f"{TRACK}_20260405", "2026-04-05T10:05:12Z", "A"),
    (f"{TRACK}_20260417", "2026-04-17T10:05:10Z", "C;D"),
    (f"{TRACK}_20260511", "2026-05-11T10:05:09Z", "C"),
]
AUDIT_EXTRA = (f"{TRACK}_20260423", "2026-04-23T10:05:11Z", "D")
MANIFEST_COLUMNS = [  # deliberately WITHOUT the newer "adopted" column: stack must tolerate old manifests
    "task_key", "track_id", "chunk_name", "parent_chunk", "row_off", "col_off", "width", "height",
    "n_bands", "gcs_prefix", "description", "task_id", "state", "attempts", "last_error",
    "error_class", "submitted_at", "updated_at", "local_paths",
]


def grid_def() -> dict:
    chunks = []
    for r in range(2):
        for c in range(3):
            ro, co = r * CPX, c * CPX
            h, w = min(CPX, H - ro), min(CPX, W - co)
            chunks.append(dict(
                chunk_id=len(chunks) + 1, name=f"chunk_r{r:02d}c{c:02d}", grid_row=r, grid_col=c,
                row_off=ro, col_off=co, width=w, height=h,
                xmin=X0 + co * RES, ymin=Y0 - (ro + h) * RES, xmax=X0 + (co + w) * RES, ymax=Y0 - ro * RES, aoi_frac=1.0,
            ))
    return dict(crs="EPSG:6933", res=RES, x0=X0, y0=Y0, width=W, height=H, chunk_px=CPX, n_chunk_rows=2,
                n_chunk_cols=3, transform=[RES, 0.0, X0, 0.0, -RES, Y0], pid_dtype="int32",
                aoi_file="data/aoi/test_aoi.gpkg", chunks=chunks)


def aoi_mask() -> np.ndarray:
    rows, cols = np.mgrid[0:H, 0:W]
    m = cols < 17
    m[0, 0] = False
    return m


def truth(dtype: str) -> np.ndarray:
    """Full-grid values of the 6 bands (VV/VH × 3 dates) as they are stored in chunk files."""
    rows, cols = np.mgrid[0:H, 0:W]
    pid = rows * W + cols
    arr = np.stack([-10.0 - b - pid / 1000.0 for b in range(1, 7)]).astype("float64")
    nod = -9999.0 if dtype == "float32" else -32768
    if dtype == "int16":
        arr = np.round(arr * 100)
    arr[4:6, 0:10, :] = nod                      # date 3: large nodata area
    if dtype == "float32":
        arr[1, 2, 3] = np.nan                    # a NaN pixel in VH of date 1
    else:
        arr[1, 2, 3] = nod
    return arr.astype(dtype)


def write_tif(path: Path, arr: np.ndarray, row_off: int, col_off: int, dtype: str, nodata: float,
              crs: str = "EPSG:6933") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tr = Affine(RES, 0, X0 + col_off * RES, 0, -RES, Y0 - row_off * RES)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1], count=arr.shape[0],
                       dtype=dtype, crs=crs, transform=tr, nodata=nodata) as dst:
        dst.write(arr)
    return str(path)


def build_synthetic_run(cfg: dict, dtype: str = "float32") -> dict:
    """Create the whole synthetic run; returns paths and expectations."""
    gd = grid_def()
    gdir = grid_dir(cfg)
    gdir.mkdir(parents=True, exist_ok=True)
    (gdir / "grid_def.json").write_text(json.dumps(gd, indent=2))
    rows, cols = np.mgrid[0:H, 0:W]
    idx = np.stack([rows * W + cols, np.zeros((H, W)), aoi_mask()]).astype("int32")
    for ch in gd["chunks"]:
        idx[1, ch["row_off"]:ch["row_off"] + ch["height"], ch["col_off"]:ch["col_off"] + ch["width"]] = ch["chunk_id"]
    with rasterio.open(gdir / "pixel_index.tif", "w", driver="GTiff", width=W, height=H, count=3, dtype="int32",
                       crs="EPSG:6933", transform=Affine(RES, 0, X0, 0, -RES, Y0), nodata=-1) as dst:
        dst.write(idx)

    run = season_dir(cfg) / "runs" / "v001_20260915"
    run.mkdir(parents=True)
    nodata = -9999.0 if dtype == "float32" else -32768.0
    adir = season_dir(cfg) / "audit" / "20260915"
    (run / "export_meta.json").write_text(json.dumps(dict(
        run_id=run.name, dtype=dtype, nodata=nodata, int16_scale=100, grid_crs=gd["crs"],
        grid_transform=gd["transform"],
        grid_fingerprint=hashlib.sha256((gdir / "grid_def.json").read_bytes()).hexdigest(),
        audit_dir=adir.relative_to(root(cfg)).as_posix(), tracks=[TRACK], config_hash="0" * 64,
        created_utc="2026-09-15T00:00:00Z",
    ), indent=2))

    bl = []
    for i, (acq, dt, plat) in enumerate(ACQS):
        for j, pol in enumerate(("VV", "VH")):
            bl.append(dict(track_id=TRACK, band_idx=2 * i + j + 1, acquisition_id=acq, date_utc=dt[:10],
                           datetime_utc=dt, date_local=dt[:10], pol=pol, platforms=plat, n_slices=2))
    pd.DataFrame(bl, columns=BAND_LAYOUT_COLUMNS).to_csv(run / "band_layout.csv", index=False)

    adir.mkdir(parents=True)
    acq_rows = [dict(acquisition_id=a, track_id=TRACK, datetime_utc=d, date_utc=d[:10], date_local=d[:10],
                     platforms=p, n_slices=2, aoi_coverage_pct=100.0) for a, d, p in ACQS + [AUDIT_EXTRA]]
    pd.DataFrame(acq_rows).to_csv(adir / "s1_acquisitions.csv", index=False)
    pd.DataFrame([
        dict(acquisition_id=ACQS[0][0], track_id=TRACK, datetime_utc=ACQS[0][1], rain_6h_mm=0.0, rain_24h_mm=0.4, source="x", level="aoi", chunk_name=""),
        dict(acquisition_id=ACQS[1][0], track_id=TRACK, datetime_utc=ACQS[1][1], rain_6h_mm=3.0, rain_24h_mm=7.2, source="x", level="aoi", chunk_name=""),
    ]).to_csv(adir / "rain_flags.csv", index=False)

    data = truth(dtype)
    raw = Path("raw_chunks") / f"track_{TRACK}"
    man = []

    def add(name, state, row_off, col_off, h, w, paths, parent="", err=""):
        man.append(manifest_row(TRACK, name, state, row_off, col_off, h, w, paths, parent=parent, err=err))

    def write_part(fname, r0, c0, h, w):
        write_tif(run / raw / fname, data[:, r0:r0 + h, c0:c0 + w], r0, c0, dtype, nodata)
        return str(raw / fname)

    # r00c00: split by Earth Engine into two files
    add("chunk_r00c00", "VERIFIED", 0, 0, 8, 8, [
        write_part("chunk_r00c00-0000000000-0000000000.tif", 0, 0, 4, 8),
        write_part("chunk_r00c00-0000000004-0000000000.tif", 4, 0, 4, 8),
    ])
    add("chunk_r00c01", "VERIFIED", 0, 8, 8, 8, [write_part("chunk_r00c01.tif", 0, 8, 8, 8)])
    add("chunk_r00c02", "DOWNLOADED", 0, 16, 8, 4, [write_part("chunk_r00c02.tif", 0, 16, 8, 4)])
    add("chunk_r01c00", "SPLIT", 8, 0, 7, 8, [])
    for i, (dr, dc, h, w) in enumerate([(0, 0, 4, 4), (0, 4, 4, 4), (4, 0, 3, 4), (4, 4, 3, 4)]):
        add(f"chunk_r01c00_s{i}", "VERIFIED", 8 + dr, dc, h, w,
            [write_part(f"chunk_r01c00_s{i}.tif", 8 + dr, dc, h, w)], parent="chunk_r01c00")
    add("chunk_r01c01", "VERIFIED", 8, 8, 7, 8, [write_part("chunk_r01c01.tif", 8, 8, 7, 8)])
    add("chunk_r01c02", "FAILED", 8, 16, 7, 4, [], err="User memory limit exceeded")
    pd.DataFrame(man, columns=MANIFEST_COLUMNS).to_csv(run / "export_manifest.csv", index=False)

    mosaic = data.copy()
    mosaic[:, :, 16:20] = nodata  # r00c02 unverified + r01c02 failed
    return dict(run=run, gd=gd, data=data, mosaic=mosaic, nodata=nodata, dtype=dtype)


def expected_valid_pct(syn: dict) -> np.ndarray:
    m = syn["mosaic"]
    valid = m != syn["nodata"]
    if syn["dtype"] == "float32":
        valid &= np.isfinite(m)
    mask = aoi_mask()
    return (valid & mask[None]).sum(axis=(1, 2)) / mask.sum() * 100.0


def manifest_row(track, name, state, r0, c0, h, w, paths, parent="", err=""):
    return dict(task_key=f"{track}__{name}", track_id=track, chunk_name=name, parent_chunk=parent,
                row_off=r0, col_off=c0, width=w, height=h, n_bands=6, gcs_prefix="p", description="d",
                task_id="t", state=state, attempts=1, last_error=err, error_class="other" if err else "",
                submitted_at="", updated_at="", local_paths=";".join(paths))


def write_window(syn, track, fname, r0, c0, h, w, values=None):
    rel = Path("raw_chunks") / f"track_{track}" / fname
    arr = syn["data"][:, r0:r0 + h, c0:c0 + w] if values is None else values
    write_tif(syn["run"] / rel, arr, r0, c0, syn["dtype"], syn["nodata"])
    return str(rel)


def set_manifest(syn, rows, columns=None):
    pd.DataFrame(rows, columns=columns or MANIFEST_COLUMNS).to_csv(syn["run"] / "export_manifest.csv", index=False)


def add_track(cfg, syn, track2=TRACK2, with_manifest=True):
    run = syn["run"]
    shutil.copytree(run / "raw_chunks" / f"track_{TRACK}", run / "raw_chunks" / f"track_{track2}")
    man = pd.read_csv(run / "export_manifest.csv", dtype=str, keep_default_na=False)
    if with_manifest:
        man2 = man[man["track_id"] == TRACK].copy()
        for col in ("track_id", "task_key", "local_paths"):
            man2[col] = man2[col].str.replace(TRACK, track2)
        pd.concat([man, man2]).to_csv(run / "export_manifest.csv", index=False)
    bl = pd.read_csv(run / "band_layout.csv", dtype=str, keep_default_na=False)
    bl2 = bl[bl["track_id"] == TRACK].copy()
    for col in ("track_id", "acquisition_id"):
        bl2[col] = bl2[col].str.replace(TRACK, track2)
    pd.concat([bl, bl2]).to_csv(run / "band_layout.csv", index=False)
    apath = season_dir(cfg) / "audit" / "20260915" / "s1_acquisitions.csv"
    acq = pd.read_csv(apath, dtype=str, keep_default_na=False)
    acq2 = acq[acq["track_id"] == TRACK].copy()
    for col in ("track_id", "acquisition_id"):
        acq2[col] = acq2[col].str.replace(TRACK, track2)
    pd.concat([acq, acq2]).to_csv(apath, index=False)
    meta = json.loads((run / "export_meta.json").read_text())
    meta["tracks"] = sorted({TRACK, track2})
    (run / "export_meta.json").write_text(json.dumps(meta))
