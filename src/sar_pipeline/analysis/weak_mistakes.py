"""Where the weak AOIs' rules still go wrong, and which pixel signal could tell the wrong fields from the right ones.

Why (user, 8 Oct 2026, plan step 2 for the weak AOIs): after each weak AOI got the best of the existing rule sets
(``weak_rules``), three mistakes still cost most of the delivery (``field_review.delivered_agreement``):

* **A** standing rice called "rice harvested" (rice missed);
* **B** flooded / bare fields called "young rice" (rice added);
* **C** other vegetation or trees called rice (rice added).

A new switch for one of them must be built on evidence, not guessed. This module puts, for every reviewed field of
the weak AOIs, the field's median of each pixel feature (``curve_rules.own_range_features``: all on the pixel's own
range, no fixed dB or NDVI numbers) next to the reviewer's class and the map's class under the AOI's rules (the
adopted v2 rules where there are some). For each mistake it then ranks the features by how well they separate the
fields the map gets WRONG from those it gets RIGHT within the same map class (area under the ROC curve: 0.5 = no
help, 0 or 1 = a clean split).

Output: ``<FRESH>/weak_rules/field_features.parquet`` (one row per reviewed field) and
``weak_rules/mistake_features.csv`` (one row per mistake and feature). Read only: no map or delivery changes.

Run::

    python -m sar_pipeline.analysis.weak_mistakes build
    python -m sar_pipeline.analysis.weak_mistakes rank
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .weak_rules import FRESH, OUT, RICE_NAMES

#: The three mistakes: (map class(es), reviewer classes that make it WRONG, reviewer classes that make it RIGHT).
MISTAKES = {
    "A_standing_called_harvested": (("rice harvested",),
                                    ("rice standing transplanted", "rice standing direct seeded"),
                                    ("rice harvested", "flooded / bare", "other vegetation")),
    "B_bare_called_young": (("young rice",), ("flooded / bare",), ("young rice", "rice standing transplanted",
                                                                    "rice standing direct seeded")),
    "C_veg_called_rice": (RICE_NAMES, ("other vegetation", "tree/orchard"), RICE_NAMES),
}


def rules_of(aoi: int) -> dict:
    """The AOI's rules as delivered last: the adopted v2 rules (``weak_rules``) when there are some, else its own."""
    from . import curve_rules as cr

    p = OUT / f"aoi{aoi}.json"
    if p.exists():
        rec = json.loads(p.read_text())
        if rec.get("adopt"):
            return dict(rec["rules"])
    return {k: v for k, v in cr.AOI_OVERRIDES.get(aoi, {}).items() if k != "SIEVE_ACRES"}


def field_features(aoi: int) -> pd.DataFrame:
    """One row per reviewed field of ``aoi``: reviewer class, map class (majority under :func:`rules_of`) and the
    median of every numeric pixel feature over the field's pixels."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from .. import qgis_review as q
    from . import curve_rules as cr
    from . import field_review as fr

    t = fr.truth(aoi, FRESH)
    with rasterio.open(Path(FRESH) / f"aoi{aoi}" / f"aoi{aoi}_rel_class.tif") as ds:
        crs, transform, shape = ds.crs, ds.transform, ds.shape
    f = gpd.read_file(Path(q.SRC) / "fields" / f"aoi{aoi}_fields_monsoon2026.gpkg").to_crs(crs)
    f = f[f["field_id"].isin(t.index)].reset_index(drop=True)
    ids = rasterize(((g, i + 1) for i, g in enumerate(f.geometry)), out_shape=shape, transform=transform, fill=0,
                    dtype="int32").ravel()
    px = np.flatnonzero(ids)
    key = -995
    cr.AOI_OVERRIDES[key] = rules_of(aoi)
    try:
        with cr.rules_for(key):
            feat = cr.own_range_features(aoi, px)
            cls = cr.classify_relative(feat).to_numpy()
    finally:
        cr.AOI_OVERRIDES.pop(key, None)
    feat = feat.apply(pd.to_numeric, errors="coerce").select_dtypes("number")
    feat["_field"] = ids[px]
    feat["_cls"] = cls
    rows = []
    for i, fid in enumerate(f["field_id"]):
        g = feat[feat["_field"] == i + 1]
        if g.empty:
            continue
        vc = g["_cls"].value_counts()
        rows.append({"aoi": aoi, "field_id": fid, "truth": t.get(fid), "map": vc.index[0],
                     "map_share": round(float(vc.iloc[0] / len(g)), 2), "pixels": int(len(g)),
                     **g.drop(columns=["_field", "_cls"]).median().to_dict()})
    return pd.DataFrame(rows)


def build(aois) -> pd.DataFrame:
    """:func:`field_features` for every AOI, one after another (each uses all cores inside), saved as parquet."""
    frames = []
    for a in aois:
        frames.append(field_features(a))
        print(f"aoi{a}: {len(frames[-1])} fields", flush=True)
    t = pd.concat(frames, ignore_index=True)
    t.to_parquet(OUT / "field_features.parquet", index=False)
    return t


def auc(wrong: np.ndarray, right: np.ndarray) -> float:
    """Probability that a WRONG field has a higher value than a RIGHT one (Mann-Whitney; ties count half)."""
    wrong, right = wrong[np.isfinite(wrong)], right[np.isfinite(right)]
    if not len(wrong) or not len(right):
        return np.nan
    allv = np.concatenate([wrong, right])
    ranks = pd.Series(allv).rank().to_numpy()
    rw = ranks[:len(wrong)].sum() - len(wrong) * (len(wrong) + 1) / 2
    return float(rw / (len(wrong) * len(right)))


def rank(t: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per mistake and feature: fields wrong / right, their medians and the AUC; sorted by distance from 0.5."""
    t = pd.read_parquet(OUT / "field_features.parquet") if t is None else t
    skip = {"aoi", "field_id", "truth", "map", "map_share", "pixels"}
    feats = [c for c in t.columns if c not in skip and pd.api.types.is_numeric_dtype(t[c])]
    rows = []
    for name, (mapped, bad, good) in MISTAKES.items():
        m = t[t["map"].isin(mapped)]
        w, r = m[m["truth"].isin(bad)], m[m["truth"].isin(good) & ~m["truth"].isin(bad)]
        for c in feats:
            a = auc(w[c].to_numpy(float), r[c].to_numpy(float))
            rows.append({"mistake": name, "feature": c, "wrong_fields": len(w), "right_fields": len(r),
                         "wrong_median": round(float(w[c].median()), 3) if len(w) else np.nan,
                         "right_median": round(float(r[c].median()), 3) if len(r) else np.nan,
                         "auc": round(a, 3) if np.isfinite(a) else np.nan,
                         "aois_with_wrong": int(w["aoi"].nunique())})
    out = pd.DataFrame(rows)
    out["strength"] = (out["auc"] - 0.5).abs()
    out = out.sort_values(["mistake", "strength"], ascending=[True, False])
    out.to_csv(OUT / "mistake_features.csv", index=False)
    return out


def best_cut(mistake: str, feature: str, t: pd.DataFrame | None = None) -> dict:
    """The cut on ``feature`` that best separates the mistake's wrong fields from its right ones, chosen on the PICK
    half of the fields (``weak_rules.half``) and scored on both halves: share of wrong fields caught and of right fields
    wrongly caught. The side (above / below) follows the AUC. Field medians only: a guide for a switch, which is then
    tested pixel by pixel in ``weak_rules.search``."""
    from .weak_rules import half

    t = pd.read_parquet(OUT / "field_features.parquet") if t is None else t
    mapped, bad, good = MISTAKES[mistake]
    m = t[t["map"].isin(mapped) & (t["truth"].isin(bad) | t["truth"].isin(good))].copy()
    m["wrong"] = m["truth"].isin(bad)
    m["half"] = m["field_id"].map(half)
    x = m[feature].to_numpy(float)
    above = auc(x[m["wrong"].to_numpy()], x[~m["wrong"].to_numpy()]) >= 0.5
    pick = m[m["half"] == "pick"]
    best = None
    for c in np.unique(np.round(pick[feature].dropna(), 3)):
        hit = pick[feature] >= c if above else pick[feature] <= c
        gain = int((hit & pick["wrong"]).sum()) - int((hit & ~pick["wrong"]).sum())
        if best is None or gain > best[1]:
            best = (float(c), gain)
    c = best[0]
    out = {"mistake": mistake, "feature": feature, "side": ">=" if above else "<=", "cut": c}
    for h in ("pick", "check"):
        g = m[m["half"] == h]
        hit = g[feature] >= c if above else g[feature] <= c
        out[f"{h}_wrong_caught"] = int((hit & g["wrong"]).sum())
        out[f"{h}_wrong"] = int(g["wrong"].sum())
        out[f"{h}_right_caught"] = int((hit & ~g["wrong"]).sum())
        out[f"{h}_right"] = int((~g["wrong"]).sum())
    return out


def weak_aois() -> list[int]:
    """Every AOI the rule search ran on (``weak_rules``)."""
    return sorted(int(p.stem[3:]) for p in OUT.glob("aoi*.json"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["build", "rank"])
    p.add_argument("--aois", type=int, nargs="*", default=[])
    p.add_argument("--top", type=int, default=8)
    a = p.parse_args(argv)
    if a.step == "build":
        build(a.aois or weak_aois())
    r = rank()
    for name, g in r.groupby("mistake", sort=False):
        print(f"\n{name}: {int(g['wrong_fields'].iloc[0])} wrong vs {int(g['right_fields'].iloc[0])} right fields")
        print(g.head(a.top).drop(columns=["mistake", "strength"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
