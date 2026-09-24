"""Minimum mapping unit: remove patches smaller than a field could be, and choose that size from the plots.

Why
---
A per-pixel map carries specks: single pixels of "rice-like, water unconfirmed" inside a confirmed
rice field (a bund, a weak radar pixel at its edge), single rice pixels in a dry-land field, small
holes. Farmers manage whole fields, so a patch smaller than the smallest real field is noise and is
better given the class that surrounds it. ``rasterio.features.sieve`` does exactly that: every
connected patch (8-connected) smaller than ``size`` pixels takes the value of its largest neighbour.

The size must not be a guess. The field plots give the smallest real fields (5 % of them are under
0.07 acre, about 3 pixels), but a plot usually sits inside a larger rice area, so the choice is made
by scoring several sizes on the plots and on the negative sets (``evaluate``) and keeping the largest
size that does not cost plot recall. No-data (255) is never filled and never spreads.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr

SRC = "processed/_batch/s2_2026"
NODATA = 255


def sieve_classes(classes, size: int, connectivity: int = 8) -> np.ndarray:
    """Patches smaller than ``size`` pixels take their largest neighbour's class; no-data untouched."""
    from rasterio.features import sieve

    c = np.asarray(classes).astype("uint8")
    if size <= 1:
        return c.copy()
    valid = c != NODATA
    out = sieve(c, size=size, connectivity=connectivity, mask=valid)
    return np.where(valid, out, NODATA).astype("uint8")


def evaluate(aoi_ids, plots, sizes=(1, 2, 3, 4, 6, 10, 21, 40), src_root=SRC) -> pd.DataFrame:
    """Class shares of every reference set (plots interior/edge, negatives) after each sieve size."""
    import rasterio

    from . import ndvi_5day as nd
    from . import validation as va

    rows = []
    for aoi_id in aoi_ids:
        refs = va.reference_sets(aoi_id, plots[plots["aoi"] == f"aoi{aoi_id}"] if "aoi" in plots else plots)
        with rasterio.open(Path(src_root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
            c = ds.read(1)
        for size in sizes:
            s = sieve_classes(c, size).ravel()
            v = s[refs["pixel"].to_numpy()]
            frame = refs.assign(cls=v)
            for (name, region), g in frame.groupby(["set", "region"]):
                row = {"aoi": f"aoi{aoi_id}", "size": size, "set": name, "region": region, "pixels": len(g)}
                for code, label in mr.CLASSES.items():
                    if code != NODATA:
                        row[f"{label}_pct"] = round(100 * float((g["cls"] == code).mean()), 2)
                rows.append(row)
            counts = np.bincount(s[s != NODATA], minlength=6)
            rows.append({"aoi": f"aoi{aoi_id}", "size": size, "set": "_acres", "region": "",
                         "pixels": int((s != NODATA).sum()),
                         **{f"{mr.CLASSES[k]}_acres": round(mr.acres(int(counts[k])), 1) for k in range(6)}})
        nd.forget()
    return pd.DataFrame(rows)


def apply(aoi_id: int, size: int, src_root=SRC, suffix_in: str = "", suffix_out: str = "_sieved") -> dict:
    """Write ``<aoi>_monsoon2026<suffix_out>.tif`` sieved at ``size`` pixels; return acres per class."""
    import rasterio

    src = Path(src_root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{suffix_in}.tif"
    with rasterio.open(src) as ds:
        c = ds.read(1)
        profile = ds.profile.copy()
    s = sieve_classes(c, size)
    out = src.with_name(f"aoi{aoi_id}_monsoon2026{suffix_out}.tif")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(s, 1)
    counts = np.bincount(s[s != NODATA], minlength=6)
    return {"aoi": f"aoi{aoi_id}", **{f"{mr.CLASSES[k]}_acres": round(mr.acres(int(counts[k])), 1) for k in range(6)}}
