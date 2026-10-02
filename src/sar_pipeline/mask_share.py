"""How many clear-looking Sentinel-2 observations does the delivered mask throw away, and which test does it?

Why
---
Clear optical dates are rare in this area, so every observation the mask removes wrongly leaves a hole in
the NDVI curve (and the upper-envelope fit then bridges the hole). Before changing any test we measure
first: per AOI and date, inside the AOI polygon, how many pixels are removed by each test of the delivered
mask (``hyb40``: no data, opaque-cloud bit, Cloud Score+ below the cut unless dark, blue-band haze). The
key number is ``haze_only``: pixels that data, the opaque bit and Cloud Score+ all call clear and that ONLY
the haze test removes. If that share is large, the haze test is the place where clear views are lost.
Nothing is changed and nothing is exported; the module only reads the local per-date files.

A date whose Cloud Score+ band is 0 over the whole AOI has no score yet (issue 32: the score arrives days
after the image). It is reported as ``cs_missing`` and left out of the summary, never counted as cloud.


Use::

    python -m sar_pipeline.mask_share --ids 13 28 39 63 72 116 160
    # writes report/mask_share/dates.csv (one row per AOI and date) and summary.csv (one row per AOI)
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import ndvi_5day as nd

S2_DIR = "data/s2_dates_masks"
OUT = "processed/_batch/s2_2026/report/mask_share"
SINCE = "2026-05-01"
MAP_DATE = "2026-09-21"
OPAQUE_BIT = 1 << 10   # the only QA60 bit the delivered mask uses
CS_MIN = 40            # Cloud Score+ cut of the delivered mask (hyb40)


def test_flags(b2, b4, b8, qa, cs, cs_min: float = CS_MIN) -> dict:
    """Per-pixel result of every test of the delivered mask (True = the test removes the pixel)."""
    data = (b4 > 0) & (b8 > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = np.where(data, (b8 - b4) / (b8 + b4), np.nan)
        dark = (b8 < nd.DARK_NIR_MAX) & (ndvi < nd.DARK_NDVI_MAX)
        haze = (b2 > nd.HAZE_B2_MIN) & ((ndvi >= nd.HAZE_NDVI_MIN) | (b2 >= nd.HAZE_BLUE_TO_RED_MIN * b4))
    opaque = (qa.astype("int64") & OPAQUE_BIT) > 0
    cs_fail = ~((cs >= cs_min) | dark)
    return {"no_data": ~data, "opaque": data & opaque, "cs_fail": data & ~opaque & cs_fail,
            "haze_only": data & ~opaque & ~cs_fail & haze,
            "kept": data & ~opaque & ~cs_fail & ~haze}


def date_rows(aoi_id: int, s2_dir: str = S2_DIR, since: str = SINCE, map_date: str = MAP_DATE) -> pd.DataFrame:
    """One row per Sentinel-2 date from ``since``: the share of the AOI's pixels lost to each test."""
    import rasterio
    from rasterio.features import rasterize

    import geopandas as gpd

    from . import config as config_mod
    from .optical_export import band_index

    loc = nd.load(aoi_id)["loc"]
    nd.forget()
    rows = []
    inside = None
    for path in sorted(glob.glob(f"{s2_dir}/aoi{aoi_id}/*_S2_*.tif")):
        date = pd.Timestamp(re.search(r"_S2_(\d{4}-\d{2}-\d{2})", path).group(1))
        if date < pd.Timestamp(since):
            continue
        with rasterio.open(path) as ds:
            if inside is None:
                shapes = gpd.read_file(config_mod.aoi_path(loc["cfg"])).to_crs(ds.crs).geometry
                inside = rasterize(((g, 1) for g in shapes), out_shape=(ds.height, ds.width),
                                   transform=ds.transform, fill=0, dtype="uint8").astype(bool)
            band = lambda n: ds.read(band_index(ds, n)).astype("float32")[inside]  # noqa: E731
            b2, b4, b8, qa, cs = (band(n) for n in ("B2", "B4", "B8", "QA60", "clear"))
        f = test_flags(b2, b4, b8, qa, cs)
        row = {"aoi": f"aoi{aoi_id}", "date": date.date(), "pixels": int(inside.sum()),
               "after_map_date": bool(date > pd.Timestamp(map_date)),
               "cs_missing": bool(not cs.any())}
        row.update({k: round(float(v.mean()), 4) for k, v in f.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def summarise(dates: pd.DataFrame) -> pd.DataFrame:
    """Per AOI, over usable dates (Cloud Score+ present, on or before the map date): pixel-date shares."""
    use = dates[~dates["cs_missing"] & ~dates["after_map_date"]]
    out = []
    for aoi, g in use.groupby("aoi"):
        w = g["pixels"]
        share = lambda c: float(np.average(g[c], weights=w))  # noqa: E731
        lost_to_haze_only = share("haze_only")
        out.append({"aoi": aoi, "dates": len(g), "kept": round(share("kept"), 4), "haze_only": round(lost_to_haze_only, 4),
                    "cs_fail": round(share("cs_fail"), 4), "opaque": round(share("opaque"), 4),
                    "no_data": round(share("no_data"), 4),
                    "kept_gain_if_haze_ignored": round(lost_to_haze_only / max(share("kept"), 1e-9), 3),
                    "dates_haze_only_ge_20pct": int((g["haze_only"] >= 0.20).sum())})
    return pd.DataFrame(out)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.mask_share", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--since", default=SINCE)
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)
    dates = pd.concat([date_rows(a, since=args.since) for a in args.ids], ignore_index=True)
    summary = summarise(dates)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    dates.to_csv(f"{args.out}/dates.csv", index=False)
    summary.to_csv(f"{args.out}/summary.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
