"""Is an AOI's class 3 (rice-like cycle, water not confirmed) the same crop as its confirmed rice?

Why
---
The rule confirms rice by the transplanting flood in the radar. Where rice is sown dry
(direct-seeded, rain-fed) there is no flood to find, and the rice lands in class 3 next to the
dry-land crops that class 3 also holds. The two cannot be told apart by water, because neither has
any; they can be told apart by the crop itself: sowing calendar, canopy peak, leaf water (LSWI at
the peak) and the fall by the map date. So, per AOI, the descriptors of class 3 are compared with
the descriptors of the AOI's own confirmed rice (class 1 + 6). Where class 3 matches, it is the
same crop grown without a visible flood (user review of aoi160, September 2026: class 3 green-up
8 Jul vs 18 Jul, peak 0.80 vs 0.80, LSWI 0.35 vs 0.37); where it differs, it is another crop.

Descriptors per pixel, on the fitted 5-day series from 1 May:

* ``greenup_doy``: first window after the monsoon minimum with NDVI >= 0.50;
* ``peak_ndvi`` and ``peak_doy``;
* ``lswi_at_peak``: leaf / soil water at the canopy peak;
* ``fall_from_peak``: peak minus the last value.

A class matches the AOI's rice when the medians agree within ``MATCH`` (days, NDVI, LSWI).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from . import ndvi_5day as nd

SRC = "processed/_batch/s2_2026"
OUT = f"{SRC}/report/class3_phenology.csv"
RICE = (1, 6)
MATCH = {"greenup_doy": 15.0, "peak_ndvi": 0.06, "lswi_at_peak": 0.06, "fall_from_peak": 0.08}
MIN_PIXELS = 200
SAMPLE = 3000


def descriptors(aoi_id: int, pixels: np.ndarray, season_from: str = "2026-05-01") -> pd.DataFrame:
    d = nd.load(aoi_id)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)[:, pixels]
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)[:, pixels]
    win = pd.DatetimeIndex(d["windows"])
    mon = np.asarray(win >= season_from)
    nd_m, ls_m = ndvi[mon], lswi[mon]
    doy = win[mon].dayofyear.to_numpy().astype(float)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        i_min = np.nanargmin(np.where(np.isfinite(nd_m), nd_m, np.inf), axis=0)
        step = np.arange(len(doy))[:, None]
        crossed = (step > i_min[None, :]) & (nd_m >= mr.CANOPY_MIN)
        first = np.where(crossed.any(axis=0), crossed.argmax(axis=0), -1)
        greenup = np.where(first >= 0, doy[np.clip(first, 0, len(doy) - 1)], np.nan)
        peak = np.nanmax(nd_m, axis=0)
        i_pk = np.nanargmax(np.where(np.isfinite(nd_m), nd_m, -np.inf), axis=0)
    return pd.DataFrame({"greenup_doy": greenup, "peak_ndvi": peak, "peak_doy": doy[i_pk],
                         "lswi_at_peak": ls_m[i_pk, np.arange(len(pixels))], "fall_from_peak": peak - ndvi[-1]})


def compare(aoi_id: int, seed: int = 0) -> dict:
    """Medians of the descriptors for the AOI's rice and its class 3, and whether they match."""
    import rasterio

    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_{'monsoon2026'}_final.tif") as ds:
        classes = ds.read(1).ravel()
    inside = nd.inside_aoi(aoi_id)
    rng = np.random.default_rng(seed)
    out = {"aoi": f"aoi{aoi_id}"}
    med = {}
    for name, codes in (("rice", RICE), ("class3", (3,))):
        pix = np.flatnonzero(inside & np.isin(classes, codes))
        out[f"{name}_acres"] = round(mr.acres(len(pix)), 1)
        if len(pix) < MIN_PIXELS:
            continue
        pix = rng.choice(pix, min(SAMPLE, len(pix)), replace=False)
        m = descriptors(aoi_id, pix).median(numeric_only=True)
        med[name] = m
        for k, v in m.items():
            out[f"{name}_{k}"] = round(float(v), 2)
    if "rice" in med and "class3" in med:
        diffs = {k: abs(float(med["class3"][k] - med["rice"][k])) for k in MATCH}
        out["match"] = all(diffs[k] <= MATCH[k] for k in MATCH)
        out["greenup_diff_days"] = round(diffs["greenup_doy"], 1)
    else:
        out["match"] = None
    return out


def run(aoi_ids, out_path=OUT) -> pd.DataFrame:
    """One AOI at a time, the series cache dropped after each (the loader keeps every AOI it read
    in memory: 132 AOIs are more than a 20 GB pod holds), results appended to the CSV as they come
    so an interrupted run resumes where it stopped."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if Path(out_path).exists():
        prev = pd.read_csv(out_path)
        done = set(prev["aoi"])
    for a in aoi_ids:
        if f"aoi{a}" in done:
            continue
        try:
            row = compare(a)
        except Exception as exc:  # keep the batch going
            row = {"aoi": f"aoi{a}", "error": f"{type(exc).__name__}: {exc}"[:120]}
        finally:
            nd.forget()
        pd.DataFrame([row]).to_csv(out_path, mode="a", header=not Path(out_path).exists(), index=False)
    return pd.read_csv(out_path)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.class3_phenology", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", type=int, default=None, help="default: every AOI with a final map")
    args = p.parse_args(argv)
    ids = args.ids or sorted(int(q.parent.name[3:]) for q in Path(SRC).glob("aoi*/aoi*_monsoon2026_final.tif"))
    t = run(ids)
    pd.set_option("display.width", 250)
    both = t.dropna(subset=["match"]) if "match" in t else t
    print(both[["aoi", "rice_acres", "class3_acres", "rice_greenup_doy", "class3_greenup_doy", "rice_peak_ndvi", "class3_peak_ndvi",
                "rice_lswi_at_peak", "class3_lswi_at_peak", "rice_fall_from_peak", "class3_fall_from_peak", "match"]].to_string(index=False))
    if len(both):
        m = both[both["match"] == True]
        print(f"\nAOIs compared: {len(both)}; class 3 matches the AOI's rice in {len(m)} AOIs "
              f"holding {m['class3_acres'].sum():,.0f} of {both['class3_acres'].sum():,.0f} class-3 acres "
              f"(all AOIs: {t['class3_acres'].sum():,.0f}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
