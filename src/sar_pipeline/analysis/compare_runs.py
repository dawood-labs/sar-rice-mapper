"""Measure a new map against the frozen baseline before anything is adopted (fix plan, stage 0).

Why
---
Every fix in the plan changes the map somewhere. Whether it made the map better cannot be judged
from a few fields: the change must be counted in acres per class, checked on the surveyed plots
and negatives (recall must not fall, false alarms must not rise), and checked on the ~430 fields
the reviewers judged by eye (wrong fields fixed, right fields untouched). This module prints that
one table for a run, always against the same baseline copy (``baseline_v3/``, the delivered map of
24 Sep 2026), so stages can be compared with each other too.

Three views:

1. ``transitions`` — acres that moved from each baseline class to each new class, per AOI and in
   total (the class map is compared pixel by pixel, so the polygon-area inflation of issue 13
   plays no part).
2. ``plot_scores`` — the reference-set shares of ``field_rice.evaluate`` (plot interiors, edges,
   evergreen / water / bare / cut negatives) reduced to: delivered-rice share of the plot interiors
   per region, and delivered-rice share of each negative set.
3. ``verdict_scores`` — the reviewers' fields (``review_verdicts``) against the new field labels.

Use::

    python -m sar_pipeline.analysis.compare_runs --stage stage1 [--ids ...]
    # writes report/compare/<stage>_transitions.csv, _plots.csv, _verdicts.csv and prints a summary
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr

SRC = "processed/_batch/s2_2026"
BASELINE = f"{SRC}/baseline_v3"
OUT = f"{SRC}/report/compare"
PLOT_AOIS = (20, 24, 25, 26, 27, 28, 29, 64, 65, 66, 67, 68)
DELIVERED = (1, 6)
NODATA = 255


def load_plots(folder="../ground_data"):
    """The surveyed plots with the AOI each one lies in (column ``aoi``), as the scorers expect."""
    from .. import season_screen as ss
    from ..prep import field_plots as fp

    plots = fp.repair(fp.load(folder))
    where = fp.in_aois(plots, ss.aoi_polygons())
    return plots.merge(where[["plot_id", "aoi"]], on="plot_id").dropna(subset=["aoi"])


def transitions(aoi_id: int, new_path=None, baseline_dir=BASELINE) -> pd.DataFrame:
    """Acres per (baseline class, new class) for one AOI."""
    import rasterio

    base = Path(baseline_dir) / f"aoi{aoi_id}_monsoon2026_final.tif"
    new = Path(new_path) if new_path else Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026_final.tif"
    with rasterio.open(base) as ds:
        a = ds.read(1).ravel()
    with rasterio.open(new) as ds:
        b = ds.read(1).ravel()
    if a.shape != b.shape:
        raise ValueError(f"aoi{aoi_id}: baseline and new map differ in shape")
    keep = (a != NODATA) & (b != NODATA)
    pair = a[keep].astype(int) * 256 + b[keep].astype(int)
    counts = np.bincount(pair, minlength=256 * 256)
    rows = []
    for code, n in enumerate(counts):
        if n:
            rows.append({"aoi": f"aoi{aoi_id}", "from": code // 256, "to": code % 256,
                         "acres": round(mr.acres(int(n)), 1)})
    return pd.DataFrame(rows)


def transition_matrix(t: pd.DataFrame) -> pd.DataFrame:
    """Total acres from -> to as a matrix, with the net change per class as the last column."""
    m = t.pivot_table(index="from", columns="to", values="acres", aggfunc="sum", fill_value=0.0)
    m.index = [mr.CLASSES.get(i, i) for i in m.index]
    m.columns = [mr.CLASSES.get(i, i) for i in m.columns]
    before = m.sum(axis=1)
    after = m.sum(axis=0).reindex(m.index, fill_value=0.0)
    m["baseline"] = before
    m["new"] = after
    m["net"] = after - before
    return m.round(1)


def plot_scores(aoi_ids=PLOT_AOIS, plots=None, map_suffix: str = "_final") -> pd.DataFrame:
    """Delivered-rice share (%) of plot interiors per region and of each negative set."""
    from . import field_rice

    plots = load_plots() if plots is None else plots
    ids = [a for a in aoi_ids if (Path(SRC) / f"aoi{a}" / f"aoi{a}_monsoon2026{map_suffix}.tif").exists()]
    ev = field_rice.evaluate(ids, plots, map_suffix)
    ev["delivered_pct"] = ev["rice_pct"] + ev["young_rice_pct"]
    rows = []
    for (m, s, region), g in ev.groupby(["map", "set", "region"]):
        w = g["pixels"].to_numpy()
        rows.append({"map": m, "set": s, "region": region, "pixels": int(w.sum()),
                     "delivered_pct": round(float(np.average(g["delivered_pct"], weights=w)), 2),
                     "harvested_pct": round(float(np.average(g["harvested_pct"], weights=w)), 2),
                     "young_pct": round(float(np.average(g["young_pct"], weights=w)), 2)})
    return pd.DataFrame(rows)


def verdict_scores(fields_dir=f"{SRC}/fields") -> tuple[pd.DataFrame, pd.DataFrame]:
    """The reviewers' fields against the labels in ``fields_dir``; returns (summary, per field)."""
    import pyogrio

    from . import review_verdicts as rv

    v = pd.read_csv(rv.OUT)
    frames = []
    for aoi in sorted(v["aoi"].unique(), key=lambda s: int(s[3:])):
        path = Path(fields_dir) / f"{aoi}_fields_monsoon2026.gpkg"
        if path.exists():
            g = pyogrio.read_dataframe(path, read_geometry=False, columns=["field_id", "label"])
            frames.append(g[g["field_id"].isin(v["field_id"])])
    labels = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["field_id", "label"])
    per_field = v.merge(labels.rename(columns={"label": "new_label"}), on="field_id", how="left")
    return rv.score(labels, v), per_field


def summary_text(matrix: pd.DataFrame, plots: pd.DataFrame, verdicts: pd.DataFrame) -> str:
    lines = ["Class acres, baseline -> new (pixel map inside the AOIs):", matrix.to_string(), ""]
    for m in sorted(plots["map"].unique()):
        p = plots[plots["map"] == m]
        lines += [f"Delivered rice share (%) of the reference sets, {m} map (pixels in brackets):",
                  p.assign(cell=p["delivered_pct"].round(1).astype(str) + " (" + p["pixels"].astype(str) + ")")
                  .pivot_table(index="set", columns="region", values="cell", aggfunc="first").to_string(), ""]
    lines += ["Reviewed fields (right fields must stay, wrong fields must flip):", verdicts.to_string(index=False)]
    return "\n".join(lines)


def run(stage: str, aoi_ids=None, plots=None, out_dir=OUT) -> dict:
    ids = aoi_ids or sorted(int(p.name[3:].split("_")[0]) for p in Path(BASELINE).glob("aoi*_monsoon2026_final.tif"))
    t = pd.concat([transitions(a) for a in ids], ignore_index=True)
    matrix = transition_matrix(t)
    ps = plot_scores(plots=plots)
    vs, per_field = verdict_scores()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / f"{stage}_transitions.csv", index=False)
    matrix.to_csv(out / f"{stage}_matrix.csv")
    ps.to_csv(out / f"{stage}_plots.csv", index=False)
    vs.to_csv(out / f"{stage}_verdicts.csv", index=False)
    per_field.to_csv(out / f"{stage}_verdict_fields.csv", index=False)
    return {"transitions": t, "matrix": matrix, "plots": ps, "verdicts": vs, "per_field": per_field}


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.compare_runs", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, help="name for the output files, e.g. stage1")
    p.add_argument("--ids", nargs="*", type=int, default=None, help="default: every AOI in the baseline")
    args = p.parse_args(argv)
    r = run(args.stage, args.ids)
    print(summary_text(r["matrix"], r["plots"], r["verdicts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
