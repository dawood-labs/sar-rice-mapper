"""Per-AOI rasters for looking at any pixel's time series directly in QGIS.

Why
---
Checking a map by eye means clicking a pixel and seeing its season. QGIS cannot read the pipeline's
per-date VRTs across thousands of acquisition dates conveniently, but its *Temporal/Spectral
Profile Tool* plugin plots every band of a multi-band raster for the clicked pixel. So each AOI gets
three small multi-band GeoTIFFs in which **band N is the same calendar half-month for every AOI**:

* ``<aoi>_VH_halfmonth.tif``, ``<aoi>_VV_halfmonth.tif``, ``<aoi>_VHmVV_halfmonth.tif``: 5x5
  linear-power means in dB, one band per half-month, exactly the values the classifier sees;
* the class map, the rice and monsoon-rice probabilities, and the pixel index, on the same grid.

Band descriptions carry the half-month start date, and a small text file lists band number to date,
because the plugin's x-axis shows band numbers.
"""
from __future__ import annotations

import datetime as dt
import shutil
from pathlib import Path

import numpy as np


def bin_label_to_date(label: int) -> dt.date:
    """``YYYYMMH`` half-month label -> the date that half-month starts on."""
    label = int(label)
    return dt.date(label // 1000, (label // 10) % 100, 1 if label % 10 == 0 else 16)


def write_stack(path, shape, rows, cols, values, grid_def, descriptions, nodata=-9999.0):
    """Write ``values`` (n_pixels, n_bands) onto the grid as a float32 multi-band GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin

    n_bands = values.shape[1]
    cube = np.full((n_bands,) + tuple(shape), nodata, dtype="float32")
    cube[:, rows, cols] = values.T
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], count=n_bands, dtype="float32",
                   crs=grid_def["crs"], transform=from_origin(grid_def["x0"], grid_def["y0"],
                                                             grid_def["res"], grid_def["res"]),
                   nodata=nodata, compress="deflate", predictor=3, tiled=True,
                   blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(cube)
        for i, text in enumerate(descriptions, start=1):
            ds.set_band_description(i, text)
    return path


def build(npz_path, maps_dir, pixel_index_path, grid_def, out_dir):
    """Build one AOI's QGIS folder from its feature file and class maps. Returns the folder."""
    z = np.load(npz_path)
    aoi = Path(npz_path).stem
    out = Path(out_dir) / aoi
    out.mkdir(parents=True, exist_ok=True)
    bins = z["bins"]
    n = len(bins)
    dates = [bin_label_to_date(b) for b in bins]
    shape = tuple(int(v) for v in z["shape"])
    X = z["X"].astype("float32")
    for k, name in enumerate(("VH", "VV", "VHmVV")):
        desc = [f"{name} {d:%Y-%m-%d}" for d in dates]
        write_stack(out / f"{aoi}_{name}_halfmonth.tif", shape, z["rows"], z["cols"],
                    X[:, k * n:(k + 1) * n], grid_def, desc)
    for suffix in ("class", "rice_prob", "monsoon_rice_prob"):
        src = Path(maps_dir) / f"{aoi}_{suffix}.tif"
        if src.exists():
            shutil.copy2(src, out / src.name)
    shutil.copy2(pixel_index_path, out / f"{aoi}_pixel_index.tif")
    lines = ["band  half-month starting", "----  -------------------"]
    lines += [f"{i:>4}  {d:%d %b %Y}" for i, d in enumerate(dates, start=1)]
    (out / "BANDS.txt").write_text("\n".join(lines) + "\n")
    return out
