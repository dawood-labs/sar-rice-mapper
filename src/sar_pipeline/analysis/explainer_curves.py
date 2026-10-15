"""Real field curves for the method explainer page (internal demo, 16 Oct 2026).

Why: the founders asked how the radar (Sentinel-1) and the optical (Sentinel-2) are used, shown on curves rather than
slides. This module picks, for each class, a blind-reviewed field where the reviewer and the rule agree, and exports
the curves the notebook draws (field medians): the 5-day fitted NDVI, the clear optical views, and VV / VH per radar
pass for each track, plus the rule's class, shares, sowing date and the few rule features the page explains.

Output: ``<OUT>/explainer_curves.json``. Read only otherwise.

Run::

    python -m sar_pipeline.analysis.explainer_curves --aois 1142 1173 1166 1199 1218
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("processed/_batch/s2_2026/rice_map_2026-10-05_qc/_summary")
CLASSES = ("rice standing transplanted", "rice standing direct seeded", "young rice", "rice harvested",
           "flooded / bare", "other vegetation", "tree/orchard")
FEATURES = ("water_spell", "water_depth", "crop_age", "ndvi_low_peak", "fit_low_peak", "vh_pos_end", "ndvi_left",
            "days_at_top", "radar_rise_days")


def field_curves(aoi: int, field_id: str, start: str = "2026-03-01") -> dict:
    """The field-median curves and the rule's verdict of one reviewed field."""
    from .. import qgis_review as q
    from . import ndvi_5day as nd
    from . import radar_water as rw

    _, f, pids = q.field_by_id(field_id)
    d = nd.load(aoi, out_root=q.series_root(aoi))
    w = pd.DatetimeIndex(d["windows"])
    fit = np.nanmedian(d["ndvi5d"].reshape(len(w), -1)[:, pids], axis=1)
    dt = pd.DatetimeIndex(d["dates"])
    nv, ok = (d[k].reshape(len(dt), -1)[:, pids] for k in ("ndvi", "ok"))
    clear = [(x.strftime("%Y-%m-%d"), round(float(np.median(nv[i][ok[i].astype(bool)])), 3))
             for i, x in enumerate(dt) if x >= pd.Timestamp(start) and ok[i].astype(bool).mean() >= 0.5]
    tracks = []
    for dd, flat in rw.read_series(aoi):
        dd = pd.DatetimeIndex(dd)
        sel = dd >= pd.Timestamp(start)
        tracks.append({"dates": [x.strftime("%Y-%m-%d") for x in dd[sel]],
                       "VV": [round(float(v), 2) for v in np.nanmedian(flat["VV"][:, pids], axis=1)[sel]],
                       "VH": [round(float(v), 2) for v in np.nanmedian(flat["VH"][:, pids], axis=1)[sel]]})
    rule = q.rule_at(aoi, pids)
    keep = w >= pd.Timestamp(start)
    return {"aoi": aoi, "field_id": field_id, "pixels": int(len(pids)),
            "acres": round(float(f.to_crs(32646).area.iloc[0]) / 4046.8564224, 2),
            "fit": {"dates": [x.strftime("%Y-%m-%d") for x in w[keep]],
                    "ndvi": [None if not np.isfinite(v) else round(float(v), 3) for v in fit[keep]]},
            "clear": clear, "tracks": tracks, "rule_class": rule["rule_class"], "rule_shares": rule["rule_shares"],
            "rule_sowing": str(rule.get("rule_sowing", ""))[:10],
            "features": {k: rule["rule_features"].get(k) for k in FEATURES if k in rule["rule_features"]}}


def pick(aois, verdicts: pd.DataFrame, per_class: int = 40) -> list[dict]:
    """For each class, the first reviewed field (high confidence before medium) whose rule class agrees."""
    out = []
    for cls in CLASSES:
        c = verdicts[(verdicts["class"] == cls) & verdicts["aoi"].isin(aois)]
        c = pd.concat([c[c["confidence"] == k] for k in ("high", "medium", "low")]).head(per_class)
        for r in c.itertuples():
            try:
                cur = field_curves(int(r.aoi), r.field_id)
            except Exception as e:                       # a field without pixels or data: try the next one
                print(f"{r.field_id}: {type(e).__name__}: {e}", flush=True)
                continue
            if cur["rule_class"] == cls:
                out.append({**cur, "class": cls, "review_confidence": r.confidence, "review_sowing": r.sowing})
                print(f"{cls}: {r.field_id} ({r.confidence})", flush=True)
                break
    return out


def main(argv=None) -> int:
    import glob

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--aois", type=int, nargs="+", required=True)
    a = p.parse_args(argv)
    rows = []
    for x in a.aois:
        for f in glob.glob(f"processed/_batch/s2_2026/rice_fresh/aoi{x}/field_review/verdicts/*.json"):
            v = json.loads(Path(f).read_text())
            rows.append({"aoi": x, **{k: v.get(k) for k in ("field_id", "class", "confidence", "sowing")}})
    ex = pick(a.aois, pd.DataFrame(rows))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "explainer_curves.json").write_text(json.dumps(ex, default=str))
    print(f"{len(ex)} classes with an example -> {OUT / 'explainer_curves.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
