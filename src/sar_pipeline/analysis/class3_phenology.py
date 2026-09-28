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
REFERENCE_SETS = ("rice_plot_interior", "rice_plot_edge", "evergreen", "water", "bare_or_built", "cut_before_map_date")
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


def matching_aois(csv_path=OUT, plot_like: bool = True) -> set:
    """AOIs whose class 3 is the same crop as their confirmed rice (from :func:`run`'s table); with
    ``plot_like`` both classes must also look like the surveyed rice (peak >= 0.70, LSWI >= 0.26)."""
    t = pd.read_csv(csv_path).dropna(subset=["match"])
    ok = t["match"] == True
    if plot_like:
        ok &= ((t["rice_peak_ndvi"] >= 0.70) & (t["rice_lswi_at_peak"] >= 0.26)
               & (t["class3_peak_ndvi"] >= 0.70) & (t["class3_lswi_at_peak"] >= 0.26))
    return set(t.loc[ok, "aoi"])


def experiment(aoi_ids, plots, lookback: int = 140, csv_path=OUT) -> pd.DataFrame:
    """Score three variants on the given AOIs without touching any file: the current rule
    (110-day lookback), the longer lookback alone (A: sowing from early May counts), and the
    longer lookback plus class 3 relabelled to rice where the AOI's class 3 matches its rice (B).
    Reports class acres inside the AOI and the share of every reference set delivered as rice
    (classes 1 + 6), rule map before the sieve."""
    import functools
    import gc

    from . import validation as va

    match = matching_aois(csv_path)
    orig = mr.pixel_events
    out_csv = Path(SRC) / "report" / f"class3_experiment_lookback{lookback}.csv"
    done = set(pd.read_csv(out_csv)["aoi"]) if out_csv.exists() else set()
    rows = []
    for a in aoi_ids:
        if f"aoi{a}" in done:
            continue
        out = {}
        for name, lb in (("baseline_110", mr.LOOKBACK_DAYS), (f"A_lookback_{lookback}", lookback)):
            mr.pixel_events = functools.partial(orig, lookback_days=lb)
            try:
                d, ev, _ = mr.aoi_events(a)
            finally:
                mr.pixel_events = orig
            cls = mr.classify(ev, radar_wet=ev["radar_wet"], never_bare=ev["never_bare"], radar_trough=True,
                              map_date=d["windows"][-1])
            out[name] = cls
            del ev, d                       # the evidence tables of a large AOI are several GB; the pod has 20
            gc.collect()
        # the longer lookback as a FALLBACK only: where the 110-day rule found no cycle at all
        base, longer = out["baseline_110"], out[f"A_lookback_{lookback}"]
        out["A2_fallback"] = np.where(base == 0, longer, base).astype(base.dtype)
        for src, name in ((f"A_lookback_{lookback}", "AB_relabel_class3"), ("A2_fallback", "A2B_fallback_relabel"),
                          ("baseline_110", "B_only_relabel")):
            relabel = out[src].copy()
            if f"aoi{a}" in match:
                relabel[relabel == 3] = 1
            out[name] = relabel
        inside = nd.inside_aoi(a)
        here = plots[plots["aoi"] == f"aoi{a}"] if plots is not None and "aoi" in plots else None
        refs = va.reference_sets(a, here if here is not None and len(here) else None)
        aoi_rows = []
        for name, cls in out.items():
            row = {"aoi": f"aoi{a}", "variant": name, "class3_matches_rice": f"aoi{a}" in match}
            for k in (0, 1, 2, 3, 6, 7):
                row[f"c{k}_ac"] = round(mr.acres(int(((cls == k) & inside).sum())), 0)
            row["delivered_ac"] = row["c1_ac"] + row["c6_ac"]
            for s_name, g in refs.groupby("set"):
                pix = g["pixel"].to_numpy()
                row[f"{s_name}_pct"] = round(100 * float(np.isin(cls[pix], (1, 6)).mean()), 1)
                row[f"{s_name}_n"] = len(pix)
            aoi_rows.append(row)
        rows += aoi_rows
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # a fixed column set: an AOI without plots has fewer reference sets, and rows appended to a
        # CSV must line up with its header
        cols = (["aoi", "variant", "class3_matches_rice"] + [f"c{k}_ac" for k in (0, 1, 2, 3, 6, 7)] + ["delivered_ac"]
                + [f"{s}_{x}" for s in REFERENCE_SETS for x in ("pct", "n")])
        pd.DataFrame(aoi_rows).reindex(columns=cols).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        nd.forget()
        gc.collect()
    return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(rows)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.class3_phenology", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", type=int, default=None, help="default: every AOI with a final map")
    p.add_argument("--experiment", action="store_true", help="score baseline / longer lookback / + class-3 relabel on the ids")
    p.add_argument("--lookback", type=int, default=140)
    args = p.parse_args(argv)
    ids = args.ids or sorted(int(q.parent.name[3:]) for q in Path(SRC).glob("aoi*/aoi*_monsoon2026_final.tif"))
    if args.experiment:
        from .compare_runs import load_plots

        t = experiment(ids, load_plots(), args.lookback)
        pd.set_option("display.width", 300)
        pd.set_option("display.max_columns", 40)
        print(t.to_string(index=False))
        return 0
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
