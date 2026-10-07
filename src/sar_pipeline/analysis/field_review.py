"""Field review by eye: pick fields, draw BLIND sheets, collect verdicts, score the rule map against them.

Why
---
User (5 Oct 2026): "instead of only scripts, let the assistant judge fields from the 5-3-2 chips and the radar + NDVI
curves; fields give fewer but cleaner data points than pixels". Judging every field by eye does not scale (thousands
of fields per AOI) and is not reproducible, so the eye is used where it helps most: a sample of fields, weighted to the
ones where the rule sets disagree, judged WITHOUT seeing the rule's answer. The verdicts (checked in part by the user)
measure how good the eye is, show where the rule fails, and become labelled training data.

Steps (one AOI)::

    python -m sar_pipeline.analysis.field_review pick   --aoi 83 --n 40          # fields.csv
    python -m sar_pipeline.analysis.field_review render --aoi 83                 # one blind sheet per field
    # reviewers write verdicts/<field_id>.json (see VERDICT_KEYS); then
    python -m sar_pipeline.analysis.field_review score  --aoi 83                 # verdict vs rule map

Outputs in ``processed/_batch/s2_2026/rice_fresh/aoi<N>/field_review/``: ``fields.csv`` (the sample, with each rule
set's majority class, NOT shown to reviewers), ``sheets/<field_id>_curve.png`` / ``_chips.png`` / ``.txt`` (the numbers:
the field's median clear-view NDVI / LSWI and radar per pass), ``verdicts/<field_id>.json``, ``score.csv``.
"""
from __future__ import annotations

import multiprocessing as mp

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FRESH = "processed/_batch/s2_2026/rice_fresh"
#: What a reviewer writes per field (JSON): the map class name (``curve_rules.MAP_CLASSES``), sowing / transplanting
#: date (YYYY-MM-DD or ""), confidence ("high" / "medium" / "low") and the reason in one or two sentences.
VERDICT_KEYS = ("field_id", "class", "sowing", "confidence", "reason")


CLASSES = ("rice standing transplanted", "rice standing direct seeded", "rice harvested", "young rice",
           "flooded / bare", "other vegetation", "tree/orchard")


def clean_class(c) -> str:
    """A reviewer's class string -> one of ``CLASSES`` (a reviewer sometimes adds a note in brackets: "other vegetation
    (not a field: strip along a canal)"); unknown strings are returned as they are."""
    c = str(c).strip()
    return next((k for k in CLASSES if c.lower().startswith(k)), c)


def review_dir(aoi: int, fresh: str = FRESH) -> Path:
    return Path(fresh) / f"aoi{aoi}" / "field_review"


def _majority(classes: np.ndarray, names: dict) -> tuple[str, float]:
    v = classes[classes != 255]
    if not len(v):
        return "no data", 0.0
    k, n = np.unique(v, return_counts=True)
    return names.get(int(k[n.argmax()]), str(k[n.argmax()])), float(n.max() / len(v))


def field_classes(aoi: int, fresh: str = FRESH, min_acres: float = 0.5) -> pd.DataFrame:
    """Every delineated field of the AOI (>= ``min_acres``) with its majority class in the AOI's own raw map and in each
    ``rule_trials`` map, the share of that class, and how many different answers the rule sets give."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from .. import qgis_review as q
    from . import curve_rules as cr

    names = {k: v[0] for k, v in cr.MAP_CLASSES.items()}
    maps = {"map": Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}_rel_class.tif"}
    for d in sorted((Path(fresh) / f"aoi{aoi}" / "rule_trials").glob("rules_aoi*")):
        maps[d.name.replace("rules_", "")] = d / f"aoi{aoi}" / f"aoi{aoi}_rel_class.tif"
    with rasterio.open(maps["map"]) as ds:
        crs, transform, shape = ds.crs, ds.transform, ds.shape
    rasters = {k: rasterio.open(p).read(1).ravel() for k, p in maps.items() if p.exists()}
    f = gpd.read_file(Path(q.SRC) / "fields" / f"aoi{aoi}_fields_monsoon2026.gpkg").to_crs(crs)
    f = f[f["area_acres"] >= min_acres].reset_index(drop=True)
    ids = rasterize(((g, i + 1) for i, g in enumerate(f.geometry)), out_shape=shape, transform=transform, fill=0,
                    dtype="int32").ravel()
    order = np.argsort(ids, kind="stable")
    bounds = np.searchsorted(ids[order], np.arange(len(f) + 2))
    rows = []
    for i, fid in enumerate(f["field_id"]):
        px = order[bounds[i + 1]:bounds[i + 2]]
        if not len(px):
            continue
        r = {"field_id": fid, "acres": round(float(f.at[i, "area_acres"]), 2), "pixels": int(len(px))}
        for k, v in rasters.items():
            r[k], r[f"{k}_share"] = _majority(v[px], names)
        rows.append(r)
    out = pd.DataFrame(rows)
    sets = [k for k in rasters if k != "map"]
    out["answers"] = out[sets].nunique(axis=1) if sets else 1
    return out


def pick(aoi: int, n: int = 40, seed: int = 0, fresh: str = FRESH, min_acres: float = 0.5) -> pd.DataFrame:
    """The sample (a further call adds a new round of fields not picked before): half among fields where the rule sets
    disagree (largest disagreement first, then random), half spread
    over the AOI map's classes (at least two per class present). Written to ``fields.csv``."""
    fc = field_classes(aoi, fresh, min_acres=min_acres)
    d = review_dir(aoi, fresh)
    before = pd.read_csv(d / "fields.csv") if (d / "fields.csv").exists() else None
    if before is not None:                          # a further round: never the same field twice, appended
        fc = fc[~fc["field_id"].isin(before["field_id"])]
    rng = np.random.default_rng(seed)
    hard = fc[fc["answers"] >= 2].sample(frac=1, random_state=seed).sort_values("answers", ascending=False,
                                                                                 kind="stable")
    take = list(hard["field_id"].head(n // 2))
    rest = fc[~fc["field_id"].isin(take)]
    classes = rest["map"].unique()
    per = max(2, (n - len(take)) // max(len(classes), 1))
    for c in classes:
        pool = rest[rest["map"] == c]["field_id"].to_numpy()
        take += list(rng.choice(pool, size=min(per, len(pool)), replace=False))
    left = fc[~fc["field_id"].isin(take)]["field_id"].to_numpy()
    if len(take) < n and len(left):
        take += list(rng.choice(left, size=min(n - len(take), len(left)), replace=False))
    out = fc[fc["field_id"].isin(take[:n])].reset_index(drop=True).assign(round=1 if before is None else
                                                                          int(before.get("round", pd.Series([1])).max()) + 1)
    d.mkdir(parents=True, exist_ok=True)
    pd.concat([before, out], ignore_index=True).to_csv(d / "fields.csv", index=False) if before is not None else \
        out.to_csv(d / "fields.csv", index=False)
    return out


def add_fields(aoi: int, field_ids, fresh: str = FRESH) -> pd.DataFrame:
    """Add these fields to the review as a new round (e.g. the fields holding the user's pixel labels, the direct check
    of the reviewers); fields already in the review are skipped."""
    d = review_dir(aoi, fresh)
    before = pd.read_csv(d / "fields.csv") if (d / "fields.csv").exists() else pd.DataFrame(columns=["field_id"])
    fc = field_classes(aoi, fresh, min_acres=0.0)        # a labelled field may be small
    new = fc[fc["field_id"].isin(set(field_ids) - set(before["field_id"]))].assign(
        round=int(before["round"].max()) + 1 if "round" in before and len(before) else 1)
    d.mkdir(parents=True, exist_ok=True)
    pd.concat([before, new], ignore_index=True).to_csv(d / "fields.csv", index=False)
    return new


def _numbers(aoi: int, pids, start: str = "2026-04-01", series=None) -> str:
    """The field's median clear-view NDVI / LSWI per date and median VV / VH per radar pass, as text (blind: no rule
    output)."""
    from .. import qgis_review as q
    from . import ndvi_5day as nd
    from . import radar_water as rw

    d = nd.load(aoi, out_root=q.series_root(aoi))
    dt = pd.DatetimeIndex(d["dates"])
    nv, lv, ok = (d[k].reshape(len(dt), -1)[:, pids] for k in ("ndvi", "lswi", "ok"))
    lines = ["clear views (date: median NDVI / median LSWI / share of the field clear):"]
    for i, x in enumerate(dt):
        sel = ok[i].astype(bool)
        if x >= pd.Timestamp(start) and sel.mean() >= 0.5:
            lines.append(f"  {x:%Y-%m-%d}: {np.median(nv[i][sel]):.2f} / {np.median(lv[i][sel]):.2f} / "
                         f"{sel.mean():.0%}")
    for j, (dd, flat) in enumerate(series if series is not None else rw.read_series(aoi)):
        dd = pd.DatetimeIndex(dd)
        lines.append(f"radar track {j + 1} (date: median VV / VH dB):")
        for i, x in enumerate(dd):
            if x >= pd.Timestamp(start):
                vv, vh = np.nanmedian(flat["VV"][i, pids]), np.nanmedian(flat["VH"][i, pids])
                lines.append(f"  {x:%Y-%m-%d}: {vv:.1f} / {vh:.1f}")
    return "\n".join(lines)


def render(aoi: int, fresh: str = FRESH, part: int = 0, parts: int = 1) -> list[str]:
    """One BLIND sheet per sampled field: the curve figure and the 5-3-2 chips with the field outline
    (``water_investigation.group_sheet``, no rule class, no sowing line) and the numbers as text."""
    import matplotlib
    import matplotlib.pyplot as plt

    from .. import qgis_review as q
    from . import ndvi_5day as nd
    from . import pixel_2026 as p26
    from . import water_investigation as wi

    from . import pixel_report as pr
    from . import sar_curve

    matplotlib.use("Agg")
    d0 = review_dir(aoi, fresh)
    # the AOI's radar once for all its sheets (the numbers and the curve figure both slice it)
    series = [(dd, {p: c.reshape(len(dd), -1) for p, c in cubes.items()})
              for dd, cubes in sar_curve.cache_tracks(pr.locate(aoi, 0, season_key="monsoon2026"))]
    out = d0 / "sheets"
    out.mkdir(parents=True, exist_ok=True)
    done = []
    ids = pd.read_csv(d0 / "fields.csv")["field_id"].tolist()
    for fid in ids[part::parts]:                       # part / parts: several processes share the drawing
        if (out / f"{fid}.txt").exists():               # drawn in an earlier round
            continue
        aoi_id, f, fpids = q.field_by_id(fid)
        d = nd.load(aoi_id, out_root=q.series_root(aoi_id))
        g = d["loc"]["grid"]
        rr, cc = np.divmod(fpids, int(g["width"]))
        centre = int(np.round(np.median(rr)) * int(g["width"]) + np.round(np.median(cc)))
        r0, c0 = divmod(centre, int(g["width"]))
        row = f.iloc[0]
        geom = row.geometry if row.geometry.geom_type == "Polygon" else max(row.geometry.geoms, key=lambda x: x.area)
        xs, ys = geom.exterior.xy
        ox = (np.asarray(xs) - g["x0"]) / g["res"] - 0.5 - (c0 - p26.HALF)
        oy = (g["y0"] - np.asarray(ys)) / g["res"] - 0.5 - (r0 - p26.HALF)
        fig, sheet, _ = wi.group_sheet(aoi_id, centre, out_dir=out, file_stem=fid, label=f"field {fid}",
                                       pids=fpids, outline=(ox, oy), series_root=q.series_root(aoi_id),
                                       sowing=np.nan, sowing_label="")
        plt.close("all")
        (out / f"{fid}.txt").write_text(f"field {fid}: {len(fpids)} pixels, {row['area_acres']:.2f} acres\n"
                                        + _numbers(aoi_id, fpids, series=series))
        done.append(fid)
    return done


def check(aoi: int, field_id: str, label: str, note: str = "", fresh: str = FRESH) -> pd.DataFrame:
    """Store the USER's verdict on a reviewed field (``user_checks.csv``, latest row per field wins; label "" = the user
    could not tell). The user's word overrides the reviewer's in :func:`truth`."""
    path = review_dir(aoi, fresh) / "user_checks.csv"
    old = pd.read_csv(path, keep_default_na=False) if path.exists() else pd.DataFrame(columns=["field_id", "label", "note"])
    new = pd.concat([old[old["field_id"] != field_id],
                     pd.DataFrame([{"field_id": field_id, "label": label, "note": note}])], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def truth(aoi: int, fresh: str = FRESH) -> pd.Series:
    """Field id -> class: the user's check where given (an empty label drops the field), else the reviewer's verdict."""
    d0 = review_dir(aoi, fresh)
    t = pd.Series({json.loads(p.read_text())["field_id"]: clean_class(json.loads(p.read_text())["class"])
                   for p in sorted((d0 / "verdicts").glob("*.json"))}, dtype=object)
    path = d0 / "user_checks.csv"
    if path.exists():
        for r in pd.read_csv(path, keep_default_na=False).itertuples():
            if r.label:
                t[r.field_id] = r.label
            else:
                t = t.drop(r.field_id, errors="ignore")
    return t


def score(aoi: int, fresh: str = FRESH) -> pd.DataFrame:
    """Each verdict next to the AOI map's and every rule set's majority class; agreement per rule set."""
    d0 = review_dir(aoi, fresh)
    fc = pd.read_csv(d0 / "fields.csv")
    vs = [json.loads(p.read_text()) for p in sorted((d0 / "verdicts").glob("*.json"))]
    v = pd.DataFrame(vs)[list(VERDICT_KEYS)].rename(columns={"class": "verdict"})
    v["verdict"] = v["verdict"].map(clean_class)
    out = fc.merge(v, on="field_id", how="inner")
    out.to_csv(d0 / "score.csv", index=False)
    return out


def _classify(args) -> np.ndarray:
    """Map class codes of these pixels under a full rule set, in a worker process (:func:`try_variants`)."""
    from . import curve_rules as cr

    aoi, rules, px = args
    names = {v[0]: k for k, v in cr.MAP_CLASSES.items()}
    key = -998
    cr.AOI_OVERRIDES[key] = dict(rules)
    try:
        with cr.rules_for(key):
            return cr.classify_relative(cr.own_range_features(aoi, px)).map(names).to_numpy()
    finally:
        cr.AOI_OVERRIDES.pop(key, None)


def try_variants(aoi: int, variants: dict, fresh: str = FRESH, sets: dict | None = None,
                 jobs: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Field majority class under the AOI's rules plus each variant's extra switches ({name: {constant: value}}), and
    under each full rule set in ``sets`` ({name: switches}, replacing the AOI's own), on the reviewed fields, against
    :func:`truth`. Read-only (no map written). One process per variant (5 Oct audit: 18 variants one after another took
    4.5 min). Returns (per field, % right per variant)."""
    from concurrent.futures import ProcessPoolExecutor

    from .. import resources
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from .. import qgis_review as q
    from . import curve_rules as cr

    t = truth(aoi, fresh)
    with rasterio.open(Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}_rel_class.tif") as ds:
        crs, transform, shape = ds.crs, ds.transform, ds.shape
    f = gpd.read_file(Path(q.SRC) / "fields" / f"aoi{aoi}_fields_monsoon2026.gpkg").to_crs(crs)
    f = f[f["field_id"].isin(t.index)].reset_index(drop=True)
    ids = rasterize(((g, i + 1) for i, g in enumerate(f.geometry)), out_shape=shape, transform=transform, fill=0,
                    dtype="int32").ravel()
    px = np.flatnonzero(ids)
    names = {v[0]: k for k, v in cr.MAP_CLASSES.items()}
    out = pd.DataFrame({"field_id": f["field_id"], "truth": t.reindex(f["field_id"]).to_numpy()})
    base = cr.AOI_OVERRIDES.get(aoi, {})
    combos = {"current": dict(base), **{n: {**base, **e} for n, e in variants.items()}, **(sets or {})}
    r = resources.detect_resources()
    n = jobs or max(1, min(len(combos), r.cpus, int(r.memory_available_bytes / 3e9)))
    work = [(aoi, rules, px) for rules in combos.values()]
    if n == 1:
        codes = [_classify(w) for w in work]
    else:
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=resources.limit_worker_threads,
                                 initargs=(max(1, r.cpus // n),)) as ex:     # spawn: see curve_rules._trial
            codes = list(ex.map(_classify, work, timeout=1800))
    rev = {v: k_ for k_, v in names.items()}
    for name, k in zip(combos, codes):
        out[name] = [_majority(k[ids[px] == i + 1], rev)[0] for i in range(len(f))]
    right = pd.Series({c: round(100 * float((out[c] == out["truth"]).mean()), 1) for c in out.columns[2:]})
    return out, right


def pixel_label_fields(aoi: int, fresh: str = FRESH) -> pd.DataFrame:
    """The user's own pixel labels (``curve_labels``) next to the field each one falls in: the field id and, where that
    field was reviewed, the reviewer's verdict. Why: the most direct check of the reviewers is against the user's
    pixel verdicts, which were made before any reviewer existed."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from .. import qgis_review as q
    from .curve_labels import load
    from .rule_generalization import truth as label_class

    lab = load()
    lab = lab[lab["aoi"] == aoi]
    if lab.empty:
        return pd.DataFrame()
    with rasterio.open(Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}_rel_class.tif") as ds:
        crs, transform, shape = ds.crs, ds.transform, ds.shape
    f = gpd.read_file(Path(q.SRC) / "fields" / f"aoi{aoi}_fields_monsoon2026.gpkg").to_crs(crs)
    ids = rasterize(((g, i + 1) for i, g in enumerate(f.geometry)), out_shape=shape, transform=transform, fill=0,
                    dtype="int32").ravel()
    k = ids[lab["pixel"].to_numpy().astype(int)]
    out = pd.DataFrame({"pixel": lab["pixel"].to_numpy(),
                        "user": [label_class(r.label, r.state, r.establishment) for r in lab.itertuples()],
                        "field_id": np.where(k > 0, f["field_id"].to_numpy()[np.clip(k - 1, 0, None)], "")})
    d0 = review_dir(aoi, fresh) / "verdicts"
    out["verdict"] = [json.loads((d0 / f"{fid}.json").read_text())["class"] if fid and (d0 / f"{fid}.json").exists()
                      else "" for fid in out["field_id"]]
    return out


def agreement(scored: pd.DataFrame) -> pd.Series:
    sets = [c for c in scored.columns if c == "map" or c.startswith("aoi") and not c.endswith("_share")]
    return pd.Series({s: round(100 * float((scored[s] == scored["verdict"]).mean()), 1) for s in sets})


RICE_NAMES = ("rice standing direct seeded", "rice standing transplanted", "young rice")


def compare_verdicts(aoi: int, a: str = "verdicts", b: str = "verdicts_sonnet", fresh: str = FRESH) -> dict:
    """Agreement of two reviewers' verdict folders on the fields both judged: the exact class, standing / young taken
    as one rice class, and rice vs non-rice. Why (user, 7 Oct 2026): a cheaper review model is used only if it agrees
    with the reviewed verdicts on at least 90 % of the fields."""
    d0 = review_dir(aoi, fresh)
    load = lambda f: {p.stem: clean_class(json.loads(p.read_text()).get("class")) for p in (d0 / f).glob("*.json")}
    va, vb = load(a), load(b)
    ids = sorted(set(va) & set(vb))
    t = pd.DataFrame({"a": [va[i] for i in ids], "b": [vb[i] for i in ids]}, index=ids)
    rice = lambda x: x.isin(RICE_NAMES)
    merged = lambda x: x.where(~rice(x), "rice")
    pct = lambda m: round(100 * float(m.mean()), 1) if len(m) else float("nan")
    return {"fields": len(t), "exact_pct": pct(t["a"] == t["b"]), "rice_merged_pct": pct(merged(t["a"]) == merged(t["b"])),
            "rice_vs_non_rice_pct": pct(rice(t["a"]) == rice(t["b"])),
            "disagree": {f"{x} -> {y}": int(n) for (x, y), n in t[t["a"] != t["b"]].value_counts().head(10).items()}}


def delivered_agreement(aoi: int, rules: dict | None = None, fresh: str = FRESH) -> dict:
    """How the AOI's rule set does on its reviewed fields, two ways: the 7 classes, and only what the client receives
    (rice = standing direct seeded, standing transplanted or young rice; anything else is not delivered).

    Why (7 Oct 2026): an AOI was called weak when fewer than half its fields had the exact class right, but standing
    vs young, or tree vs other vegetation, does not change the delivery. Before writing new rules for a weak AOI this
    shows whether its delivery is actually wrong, and where: rice missed (reviewer rice, map not) or rice added
    (map rice, reviewer not). ``rules`` defaults to the AOI's own switches."""
    from . import curve_rules as cr

    rice = {cr.MAP_CLASSES[c][0] for c in (1, 7, 3)}
    sets = {"rules": dict(rules if rules is not None else cr.AOI_OVERRIDES.get(aoi, {}))}
    o, r = try_variants(aoi, {}, sets=sets, fresh=fresh)
    t, m = o["truth"].isin(rice), o["rules"].isin(rice)
    conf = o[o["truth"] != o["rules"]].groupby(["truth", "rules"]).size().sort_values(ascending=False)
    return {"aoi": aoi, "fields": int(len(o)), "class_right_pct": float(r["rules"]),
            "delivered_right_pct": round(100 * float((t == m).mean()), 1),
            "reviewer_rice": int(t.sum()), "rice_missed": int((t & ~m).sum()), "rice_added": int((~t & m).sum()),
            "top_mixups": "; ".join(f"{a} -> {b}: {n}" for (a, b), n in conf.head(4).items())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["pick", "render", "score", "compare", "delivered"])
    p.add_argument("--a", default="verdicts", help="compare: the first verdict folder")
    p.add_argument("--b", default="verdicts_sonnet", help="compare: the second verdict folder")
    p.add_argument("--aoi", type=int, default=None)
    p.add_argument("--aois", type=int, nargs="*", default=[], help="delivered: AOIs to check")
    p.add_argument("--out", default=None, help="delivered: CSV to write")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--parts", type=int, default=0, help="render: processes drawing in parallel (0 = from the CPUs)")
    a = p.parse_args(argv)
    if a.step == "delivered":
        rows = []
        for x in a.aois or [a.aoi]:
            rows.append(delivered_agreement(x))
            print(json.dumps(rows[-1]), flush=True)
        if a.out:
            pd.DataFrame(rows).to_csv(a.out, index=False)
        return 0
    if a.step == "compare":
        print(json.dumps(compare_verdicts(a.aoi, a.a, a.b), indent=1, default=str))
        return 0
    if a.step == "pick":
        print(pick(a.aoi, a.n, a.seed).to_string(index=False))
    elif a.step == "render":
        from concurrent.futures import ProcessPoolExecutor

        from .. import resources
        r = resources.detect_resources()
        # each process holds the AOI's radar (a few GB for a large AOI): as many processes as CPUs and RAM allow
        n = a.parts or max(1, min(r.cpus, int(r.memory_available_bytes / 6e9)))
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=resources.limit_worker_threads,
                                 initargs=(max(1, r.cpus // n),)) as ex:
            done = sum(len(x) for x in ex.map(render, [a.aoi] * n, [FRESH] * n, range(n), [n] * n))
        print(f"{done} sheets in {review_dir(a.aoi) / 'sheets'}")
    else:
        s = score(a.aoi)
        pd.set_option("display.width", 250)
        print(s[["field_id", "acres", "map", "verdict", "confidence", "sowing"]].to_string(index=False))
        print("\nagreement with the verdicts (%):\n", agreement(s).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

