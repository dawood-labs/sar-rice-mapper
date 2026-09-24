"""Does a "rice-like, water unconfirmed" pixel look like the confirmed rice of its own AOI?

Why
---
The water test is a threshold, and a threshold splits a continuum: in one AOI the class-3 fields
had exactly the curves of the class-1 fields, only a slightly shallower June flood. In other AOIs
the class-3 fields were a different crop altogether (sown on rain-wet soil, bright radar all
monsoon, never flooded). Instead of loosening the threshold everywhere, this module asks each AOI
the direct question: is the class-3 pixel, over its whole season (fitted NDVI every 5 days and VV/VH
of every track in 12-day bins), as close to the AOI's confirmed rice as confirmed rice pixels are to
each other?

* ``features``: the season profile of each pixel, standardised per AOI;
* ``similarity``: distance to the k-th nearest confirmed-rice pixel, compared with the same distance
  among confirmed-rice pixels themselves (leave-one-out). "Same as rice" = within the 90th percentile
  of that within-rice spread;
* the **null**: the same test on the AOI's other crops (class 0 pixels that carried a canopy, and
  class 4, harvested) gives the rate at which non-rice passes; the test is only trusted in an AOI
  where that rate is low.

Nothing here changes the map; the result is a proposal for the user.

**Result on first use (docs/12): not discriminative.** Class 3 passed the same optical rule as
class 1, so its NDVI profile is rice-like by construction, and the few radar weeks that differ are
diluted among ~90 dimensions: 63-85 % of class 3 but also 22-41 % of the AOIs' other crops came out
"like rice". Kept as a documented negative result; do not use it to relabel pixels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from . import ndvi_5day as nd
from . import pixel_report as pr
from . import sar_curve

SRC = "processed/_batch/s2_2026"
START = "2026-04-01"
BIN_DAYS = 12


def features(aoi_id: int, pixels, d=None) -> np.ndarray:
    """Season profile per pixel: fitted NDVI (5-day windows from ``START``) + VV/VH per track in 12-day bins."""
    d = d or nd.load(aoi_id)
    pix = np.asarray(pixels).astype(int)
    win = pd.DatetimeIndex(d["windows"])
    keep = win >= pd.Timestamp(START)
    parts = [d["ndvi5d"].reshape(len(win), -1)[keep][:, pix].T * 10.0]   # NDVI x10 ~ dB scale
    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    edges = pd.date_range(START, win[-1] + pd.Timedelta(days=5), freq=f"{BIN_DAYS}D")
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track)
        day = pd.DatetimeIndex(dates)
        for pol in ("VV", "VH"):
            cube = cubes[pol].reshape(len(dates), -1)[:, pix]
            cols = []
            for a, b in zip(edges[:-1], edges[1:]):
                m = (day >= a) & (day < b)
                cols.append(np.nanmean(cube[m], axis=0) if m.any() else np.full(len(pix), np.nan))
            parts.append(np.stack(cols, axis=1))
    X = np.concatenate(parts, axis=1)
    # fill a missing bin with the pixel's neighbouring bins (a track can miss one date)
    X = pd.DataFrame(X).T.interpolate(limit_direction="both").T.to_numpy()
    return X


def knn_distance(ref, query, k: int = 10, exclude_self: bool = False) -> np.ndarray:
    """Distance from each query row to its k-th nearest row of ``ref`` (Euclidean)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(ref)
    kk = k + 1 if exclude_self else k
    dist, _ = tree.query(query, k=kk)
    return dist[:, -1]


def similarity(aoi_id: int, n_ref: int = 3000, n_query: int = 3000, k: int = 10, pct: float = 90,
               seed: int = 0) -> dict:
    """Share of class-3 pixels that look like the AOI's confirmed rice, and the same share for non-rice crops."""
    import rasterio

    rng = np.random.default_rng(seed)
    d, ev, _ = mr.aoi_events(aoi_id, radar_season=None)
    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
        c = ds.read(1).ravel()
    peak = ev["peak_after"].to_numpy()
    pools = {
        "rice": np.flatnonzero(c == 1),
        "class3": np.flatnonzero(c == 3),
        "other_crop": np.flatnonzero(((c == 0) & (peak >= mr.CANOPY_MIN)) | (c == 4)),
    }
    pick = {k_: (rng.choice(v, size=min(n_ref if k_ == "rice" else n_query, len(v)), replace=False)
                 if len(v) else v) for k_, v in pools.items()}
    if len(pick["rice"]) < 50 or not len(pick["class3"]):
        nd.forget()
        return {"aoi": f"aoi{aoi_id}", "rice_ref": int(len(pick["rice"]))}
    allpix = np.concatenate(list(pick.values()))
    X = features(aoi_id, allpix, d)
    ok = np.isfinite(X).all(axis=1)
    mu, sd = np.nanmean(X[ok], axis=0), np.nanstd(X[ok], axis=0) + 1e-6
    Z = (X - mu) / sd
    sizes = np.cumsum([0] + [len(v) for v in pick.values()])
    blocks = {k_: Z[sizes[i]:sizes[i + 1]] for i, k_ in enumerate(pick)}
    oks = {k_: ok[sizes[i]:sizes[i + 1]] for i, k_ in enumerate(pick)}
    ref = blocks["rice"][oks["rice"]]
    within = knn_distance(ref, ref, k, exclude_self=True)
    cut = np.percentile(within, pct)
    out = {"aoi": f"aoi{aoi_id}", "rice_ref": int(len(ref)), "cut": round(float(cut), 2)}
    for name in ("class3", "other_crop"):
        q = blocks[name][oks[name]]
        if len(q):
            dq = knn_distance(ref, q, k)
            out[f"{name}_n"] = int(len(q))
            out[f"{name}_like_rice_pct"] = round(100 * float((dq <= cut).mean()), 1)
            if name == "class3":
                folder = Path(SRC) / "report" / "similarity"
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"pixel": pick[name][oks[name]], "dist": dq, "like_rice": dq <= cut}).to_parquet(
                    folder / f"aoi{aoi_id}_class3.parquet")
    nd.forget()
    return out


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.rice_similarity")
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--out", default=f"{SRC}/report/similarity")
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    rows = []
    for a in args.ids:
        r = similarity(a)
        rows.append(r)
        print(r)
        pd.DataFrame([r]).to_csv(Path(args.out) / f"aoi{a}_summary.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
