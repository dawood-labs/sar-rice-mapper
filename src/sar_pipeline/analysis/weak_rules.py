"""Own rules for the weak AOIs: the AOIs whose chosen rule set got fewer than half of the reviewed fields right.

Why (user, 7 Oct 2026: "inko khud kro, in k liye apne rules bnao"): 30 reviewed AOIs (27 of the first set, 3 of the
second) and 20 more of the second set (reviewed first, ``aoi_batch2.WEAK_REVIEW``) were mapped with a rule set chosen on
the exact class of each field. What the client receives is only rice or not rice, and on that the same AOIs were 68 %
right on average (``field_review.delivered_agreement``). This module searches each weak AOI's own rule set on that
measure.

How, per AOI (user decisions, 7 Oct 2026):

1. The reviewed fields are split in two halves by a hash of the field id (fixed, so a rerun gives the same halves):
   the PICK half chooses, the CHECK half only judges.
2. Every candidate rule set (``aoi_batch.CANDIDATE_SETS``) and the AOI's current rules are scored on the PICK half
   by delivered agreement (rice = standing direct seeded, standing transplanted, young rice).
3. On top of the best one, switches (``aoi_batch.CANDIDATE_SWITCHES`` and :data:`EXTRA_SWITCHES`) are added one at a
   time, each round keeping the one that gains most on the PICK half, until none gains.
4. The result is adopted only when it is also better than the current rules on the CHECK half (a rule that only fits
   the fields it was chosen on is not adopted).

Output: ``<FRESH>/weak_rules/aoi<N>.json`` per AOI and ``weak_rules.csv`` (one row per AOI). Nothing is delivered
here; ``rice_map_delivery`` writes the v2 files from the adopted rules.

Run::

    python -m sar_pipeline.analysis.weak_rules search --aois 146 140 1143
    python -m sar_pipeline.analysis.weak_rules table
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

FRESH = "processed/_batch/s2_2026/rice_fresh"
OUT = Path(FRESH) / "weak_rules"
#: Classes the client receives as rice (``rice_map_delivery.RICE_CODES`` 1, 7, 3).
RICE_NAMES = ("rice standing direct seeded", "rice standing transplanted", "young rice")
#: Switches made for the weak AOIs' main mistakes, tried with the others (filled as they are written and validated).
EXTRA_SWITCHES: dict[str, dict] = {}


def half(field_id: str) -> str:
    """'pick' or 'check', fixed by the field id."""
    return "pick" if int(hashlib.md5(field_id.encode()).hexdigest(), 16) % 2 == 0 else "check"


def _sets() -> dict[str, dict]:
    from . import aoi_batch as ab
    from . import curve_rules as cr

    ab._fill_universal()
    return {(s if isinstance(s, str) else f"aoi{s}"): dict(cr.NAMED_SETS[s] if isinstance(s, str)
                                                          else cr.REVIEWED_SETS.get(s, {}))
            for s in ab.CANDIDATE_SETS}


def _switches() -> dict[str, dict]:
    from . import aoi_batch as ab

    ab._fill_universal()
    return {**{k: dict(v) for k, v in ab.CANDIDATE_SWITCHES.items()}, **EXTRA_SWITCHES}


def _scores(o: pd.DataFrame, cols) -> pd.DataFrame:
    """Delivered agreement (%) per column on each half and on all fields."""
    t = o["truth"].isin(RICE_NAMES)
    h = o["field_id"].map(half)
    rows = {}
    for c in cols:
        ok = o[c].isin(RICE_NAMES) == t
        rows[c] = {"pick": 100 * ok[h == "pick"].mean(), "check": 100 * ok[h == "check"].mean(), "all": 100 * ok.mean()}
    return pd.DataFrame(rows).T.round(1)


def search(aoi: int) -> dict:
    """Steps 2-4 of the module docstring for one AOI; writes and returns its record."""
    from . import curve_rules as cr
    from . import field_review as fr

    current = {k: v for k, v in cr.AOI_OVERRIDES.get(aoi, {}).items() if k != "SIEVE_ACRES"}
    sets = {"current": current, **_sets()}
    o, _ = fr.try_variants(aoi, {}, sets=sets)
    sc = _scores(o, sets)
    best = max(sets, key=lambda c: (sc.at[c, "pick"], sc.at[c, "all"]))
    rules, name, trail = dict(sets[best]), best, [(best, float(sc.at[best, "pick"]))]
    left = _switches()
    while left:
        trial = {f"{name}+{k}": {**rules, **v} for k, v in left.items()}
        o2, _ = fr.try_variants(aoi, {}, sets=trial)
        s2 = _scores(o2, trial)
        top = max(trial, key=lambda c: (s2.at[c, "pick"], s2.at[c, "all"]))
        if s2.at[top, "pick"] <= trail[-1][1]:
            break
        k = top.split("+")[-1]
        rules, name = trial[top], top
        trail.append((name, float(s2.at[top, "pick"])))
        left.pop(k)
    o3, _ = fr.try_variants(aoi, {}, sets={"current": current, "found": rules})
    s3 = _scores(o3, ["current", "found"])
    adopt = bool(s3.at["found", "check"] > s3.at["current", "check"])
    rec = {"aoi": aoi, "fields": int(len(o3)), "found": name, "rules": rules, "trail": trail,
           "current_pick": s3.at["current", "pick"], "current_check": s3.at["current", "check"],
           "current_all": s3.at["current", "all"], "found_pick": s3.at["found", "pick"],
           "found_check": s3.at["found", "check"], "found_all": s3.at["found", "all"], "adopt": adopt,
           "at": str(pd.Timestamp.now().floor("s"))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"aoi{aoi}.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


V2 = "processed/_batch/s2_2026/rice_fresh_v2"
V2_FILES = ("aoi{a}_fields_v2.gpkg", "aoi{a}_fields_v2.qml", "aoi{a}_fields_v2_too_young.gpkg",
            "aoi{a}_fields_v2_too_young.qml", "aoi{a}_v2_too_young.json", "aoi{a}_fields_v2_trees_cut.gpkg",
            "aoi{a}_fields_v2_trees_cut.qml", "aoi{a}_v2_trees_cut.json", "aoi{a}_v2.json")


def build_v2(aoi: int) -> dict:
    """The new map and field layers of a weak AOI whose found rules were adopted (``search``), as NEW files next to
    the delivered ones (user, 7 Oct 2026: ``aoi<N>_fields_v2.gpkg`` and its too-young and trees-cut layers; the
    delivered files are never touched).

    The map, sieve and fields are rebuilt in :data:`V2` (``rice_fresh_v2/aoi<N>/``), never in ``rice_fresh``, so the
    locked map and the delivered layers stay as they were. ``aoi<N>_v2.json`` records the rules and the scores."""
    import shutil

    from . import curve_rules as cr
    from . import ndvi_5day as nd
    from . import rice_map_delivery as rd

    rec = json.loads((OUT / f"aoi{aoi}.json").read_text())
    if not rec["adopt"]:
        raise ValueError(f"aoi{aoi}: the found rules did not win on the check half; nothing to build")
    rules = {**rec["rules"], "SIEVE_ACRES": rd.SLIVER_ACRES}
    d = rd.out_dir(aoi)
    gp = d / f"aoi{aoi}_fields_v2.gpkg"
    w = Path(V2) / f"aoi{aoi}"
    tag = f"{int(round(rd.SLIVER_ACRES * 100)):03d}"
    if gp.exists():                     # built already (a rerun only adds what is missing, e.g. the trees-cut layer)
        return _finish_v2(aoi, rec, rules, gp)
    w.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(FRESH) / f"aoi{aoi}" / f"aoi{aoi}_step1_cover.tif", w)   # the map's grid and profile
    old = cr.AOI_OVERRIDES.get(aoi)
    cr.AOI_OVERRIDES[aoi] = rules                          # sieve and fields read the AOI's rules too
    try:
        with cr.rules_for(aoi):
            cr._run(aoi, nd.analysis_series_root(aoi), V2)
        cr.sieve(aoi, fresh=V2, force=True)
        cr.fields(aoi, fresh=V2, force=True)
    finally:
        if old is None:
            cr.AOI_OVERRIDES.pop(aoi, None)
        else:
            cr.AOI_OVERRIDES[aoi] = old
    t = rd.field_table(aoi, path=w / f"aoi{aoi}_rel_fields_sliver{tag}.gpkg")
    tmp = d / f"aoi{aoi}_fields_v2.tmp.gpkg"
    t.to_file(tmp, driver="GPKG")
    tmp.replace(gp)
    rd.fields_qml(gp)
    return _finish_v2(aoi, rec, rules, gp)


def _finish_v2(aoi: int, rec: dict, rules: dict, gp: Path) -> dict:
    """Too-young and trees-cut layers of the v2 field file (each kept once written) and ``aoi<N>_v2.json``."""
    import geopandas as gpd

    from . import rice_map_delivery as rd
    from . import rule_records as rr

    d = gp.parent
    t = gpd.read_file(gp)
    young = rd.step_too_young(aoi, tag="_v2")
    trees = rd.step_trees_cut(aoi, tag="_v2")
    old_t = rd.field_table(aoi)
    res = {"aoi": aoi, "rules_from": rec["found"], "rules": rules,
           "reviewed_fields": rec["fields"], "delivered_right_pct_before": rec["current_all"],
           "delivered_right_pct_after": rec["found_all"], "check_half_before": rec["current_check"],
           "check_half_after": rec["found_check"],
           "rice_acres_v1": round(float(old_t.loc[old_t["major_class"] == "rice", "acres"].sum()), 1),
           "rice_acres_v2": round(float(t.loc[t["major_class"] == "rice", "acres"].sum()), 1),
           "too_young": young, "trees_cut": trees, "file": gp.name, "sha256": rd._sha(gp)}
    first = not (d / f"aoi{aoi}_v2.json").exists()
    (d / f"aoi{aoi}_v2.json").write_text(json.dumps(res, indent=1, default=str))
    if first:
            rr.record(aoi, "v2 rules (weak AOI)", f"`{rec['found']}` -> `{gp.name}` (delivered files unchanged)",
                  reason=f"rice vs not right on reviewed fields {rec['current_all']} -> {rec['found_all']} % "
                         f"(held-out half {rec['current_check']} -> {rec['found_check']} %); user, 7 Oct 2026")
    return res


def upload_v2(aoi: int) -> dict:
    """Uploads the AOI's v2 files only (``rice_map_delivery._upload_new_only``: never replaces a different file, and
    proves the files already in the bucket kept their md5). Waits for the trees-cut layer: a v2 without it would be
    one file short of the delivery."""
    from . import rice_map_delivery as rd

    names = [n.format(a=aoi) for n in V2_FILES]
    missing = [n for n in names if not (rd.out_dir(aoi) / n).exists()]
    if missing:
        raise FileNotFoundError(f"aoi{aoi}: v2 files missing {missing} (trees cut needs the basemap credentials)")
    return rd._upload_new_only(aoi, names)


def table() -> pd.DataFrame:
    """All AOI records in one table, written to ``weak_rules.csv``."""
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("aoi*.json"))]
    t = pd.DataFrame(rows).drop(columns=["rules", "trail"], errors="ignore")
    t.to_csv(OUT / "weak_rules.csv", index=False)
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["search", "table", "build-v2", "upload-v2"])
    p.add_argument("--aois", type=int, nargs="*", default=[])
    a = p.parse_args(argv)
    if a.step == "search":
        for x in a.aois:
            r = search(x)
            print(json.dumps({k: v for k, v in r.items() if k != "rules"}, default=str), flush=True)
    if a.step in ("build-v2", "upload-v2"):
        fn = build_v2 if a.step == "build-v2" else upload_v2
        for x in a.aois:
            try:
                r = fn(x)
                print(json.dumps({k: v for k, v in r.items() if k != "rules"}, default=str), flush=True)
            except Exception as e:                     # one AOI's failure (e.g. credentials) does not stop the rest
                print(json.dumps({"aoi": x, "error": f"{type(e).__name__}: {e}"}), flush=True)
        return 0
    t = table()
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
