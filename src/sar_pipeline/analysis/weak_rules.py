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
#: Cuts from ``weak_mistakes.best_cut`` (field medians, chosen on the pick half; 8 Oct 2026).
EXTRA_SWITCHES: dict[str, dict] = {
    "harvest_not_if_radar_top": {"HARVEST_NOT_IF_RADAR_TOP": 0.94},
    "young_not_if_deep_low": {"YOUNG_NOT_IF_DEEP_LOW": -0.43},
    "rice_needs_empty_field": {"RICE_NEEDS_EMPTY_FIELD": 0.435},
}


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


def search(aoi: int, out: Path = OUT) -> dict:
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
    out.mkdir(parents=True, exist_ok=True)
    (out / f"aoi{aoi}.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


V2 = "processed/_batch/s2_2026/rice_fresh_v2"
#: Each new version of a weak AOI's layer: the search records it is built from and its own map folder.
#: v2 (7 Oct 2026): the best existing rule set and switches; v3 (8 Oct 2026, user: "nayi v3 files"): with the new
#: switches of ``weak_mistakes`` (search round2), only where round 2 beat what the AOI has now on its held-out half.
VERSIONS = {"v2": (OUT, V2), "v3": (OUT / "round2", "processed/_batch/s2_2026/rice_fresh_v3")}
FILES = ("aoi{a}_fields_{v}.gpkg", "aoi{a}_fields_{v}.qml", "aoi{a}_fields_{v}_too_young.gpkg",
         "aoi{a}_fields_{v}_too_young.qml", "aoi{a}_{v}_too_young.json", "aoi{a}_fields_{v}_trees_cut.gpkg",
         "aoi{a}_fields_{v}_trees_cut.qml", "aoi{a}_{v}_trees_cut.json", "aoi{a}_{v}.json")
V2_FILES = tuple(n.replace("{v}", "v2") for n in FILES)


def now_score(aoi: int) -> dict:
    """What the AOI is delivered with now: its v2 when that was built (adopted round 1), else its own rules; with the
    scores on all reviewed fields and on the held-out half."""
    r1 = json.loads((OUT / f"aoi{aoi}.json").read_text())
    if r1["adopt"]:
        return {"version": "v2", "rules": r1["found"], "all": r1["found_all"], "check": r1["found_check"]}
    return {"version": "v1", "rules": "own", "all": r1["current_all"], "check": r1["current_check"]}


def round2_better(aoi: int) -> bool:
    """Round 2 picked one of the new switches and beat what the AOI has now on its held-out half."""
    p = VERSIONS["v3"][0] / f"aoi{aoi}.json"
    if not p.exists():
        return False
    r2 = json.loads(p.read_text())
    return any(k in r2["found"] for k in EXTRA_SWITCHES) and r2["found_check"] > now_score(aoi)["check"]


def build_version(aoi: int, ver: str = "v2") -> dict:
    """The new map and field layers of a weak AOI as NEW files next to the delivered ones (``aoi<N>_fields_<ver>.gpkg``
    and its too-young and trees-cut layers; files already delivered are never touched; user, 7 and 8 Oct 2026).

    The map, sieve and fields are rebuilt in the version's own folder (``rice_fresh_<ver>/aoi<N>/``), never in
    ``rice_fresh``, so the locked map and the delivered layers stay as they were. ``aoi<N>_<ver>.json`` records the
    rules and the scores."""
    import shutil

    from . import curve_rules as cr
    from . import ndvi_5day as nd
    from . import rice_map_delivery as rd

    rec_dir, fresh_v = VERSIONS[ver]
    rec = json.loads((rec_dir / f"aoi{aoi}.json").read_text())
    ok = rec["adopt"] if ver == "v2" else round2_better(aoi)
    if not ok:
        raise ValueError(f"aoi{aoi}: {ver} rules did not win on the held-out half; nothing to build")
    rules = {**rec["rules"], "SIEVE_ACRES": rd.SLIVER_ACRES}
    d = rd.out_dir(aoi)
    gp = d / f"aoi{aoi}_fields_{ver}.gpkg"
    w = Path(fresh_v) / f"aoi{aoi}"
    tag = f"{int(round(rd.SLIVER_ACRES * 100)):03d}"
    if gp.exists():                     # built already (a rerun only adds what is missing, e.g. the trees-cut layer)
        return _finish_version(aoi, ver, rec, rules, gp)
    w.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(FRESH) / f"aoi{aoi}" / f"aoi{aoi}_step1_cover.tif", w)   # the map's grid and profile
    old = cr.AOI_OVERRIDES.get(aoi)
    cr.AOI_OVERRIDES[aoi] = rules                          # sieve and fields read the AOI's rules too
    try:
        with cr.rules_for(aoi):
            cr._run(aoi, nd.analysis_series_root(aoi), fresh_v)
        cr.sieve(aoi, fresh=fresh_v, force=True)
        cr.fields(aoi, fresh=fresh_v, force=True)
    finally:
        if old is None:
            cr.AOI_OVERRIDES.pop(aoi, None)
        else:
            cr.AOI_OVERRIDES[aoi] = old
    t = rd.field_table(aoi, path=w / f"aoi{aoi}_rel_fields_sliver{tag}.gpkg")
    tmp = d / f"aoi{aoi}_fields_{ver}.tmp.gpkg"
    t.to_file(tmp, driver="GPKG")
    tmp.replace(gp)
    rd.fields_qml(gp)
    return _finish_version(aoi, ver, rec, rules, gp)


def build_v2(aoi: int) -> dict:
    return build_version(aoi, "v2")


def _finish_version(aoi: int, ver: str, rec: dict, rules: dict, gp: Path) -> dict:
    """Too-young and trees-cut layers of the version's field file (each kept once written) and ``aoi<N>_<ver>.json``."""
    import geopandas as gpd

    from . import rice_map_delivery as rd
    from . import rule_records as rr

    d = gp.parent
    t = gpd.read_file(gp)
    young = rd.step_too_young(aoi, tag=f"_{ver}")
    # the trees-cut step returns its file name and sha256 only when read back from its own json; leave them out so
    # this record has the same bytes whether the layer was cut in this run or an earlier one
    trees = {k: v for k, v in rd.step_trees_cut(aoi, tag=f"_{ver}").items() if k not in ("file", "sha256")}
    rice = lambda g: round(float(g.loc[g["major_class"] == "rice", "acres"].sum()), 1)  # noqa: E731
    if ver == "v2":                     # the keys of the v2 records already delivered
        before = {"all": rec["current_all"], "check": rec["current_check"], "version": "v1"}
        prev = rd.field_table(aoi)
    else:
        before = now_score(aoi)
        prev = gpd.read_file(d / f"aoi{aoi}_fields_v2.gpkg") if before["version"] == "v2" else rd.field_table(aoi)
    res = {"aoi": aoi, "rules_from": rec["found"], "rules": rules, "reviewed_fields": rec["fields"],
           "compared_with": before["version"],
           "delivered_right_pct_before": before["all"], "delivered_right_pct_after": rec["found_all"],
           "check_half_before": before["check"], "check_half_after": rec["found_check"],
           (f"rice_acres_v1" if ver == "v2" else "rice_acres_before"): rice(prev), f"rice_acres_{ver}": rice(t),
           "too_young": young, "trees_cut": trees, "file": gp.name, "sha256": rd._sha(gp)}
    if ver == "v2":
        res.pop("compared_with")
    js = d / f"aoi{aoi}_{ver}.json"
    first = not js.exists()
    js.write_text(json.dumps(res, indent=1, default=str))
    if first:
        rr.record(aoi, f"{ver} rules (weak AOI)", f"`{rec['found']}` -> `{gp.name}` (delivered files unchanged)",
                  reason=f"rice vs not right on reviewed fields {before['all']} -> {rec['found_all']} % "
                         f"(held-out half {before['check']} -> {rec['found_check']} %, vs {before['version']})")
    return res


def upload_version(aoi: int, ver: str = "v2") -> dict:
    """Uploads the AOI's files of this version only (``rice_map_delivery._upload_new_only``: never replaces a different
    file, and proves the files already in the bucket kept their md5). Waits for the trees-cut layer: a version without
    it would be one file short."""
    from . import rice_map_delivery as rd

    names = [n.format(a=aoi, v=ver) for n in FILES]
    missing = [n for n in names if not (rd.out_dir(aoi) / n).exists()]
    if missing:
        raise FileNotFoundError(f"aoi{aoi}: {ver} files missing {missing} (trees cut needs the basemap credentials)")
    return rd._upload_new_only(aoi, names)


def upload_v2(aoi: int) -> dict:
    return upload_version(aoi, "v2")


def v2_summary() -> pd.DataFrame:
    """One row per AOI with v2 files: rules, delivered agreement before / after, rice acres of the delivered layer and of
    the v2 layer after the too-young and trees-cut steps. Written to ``<delivery>/weak_v2_summary.csv`` for the user."""
    import glob

    from . import aoi_batch2 as b2
    from . import rice_map_delivery as rd

    rows = []
    for f in sorted(glob.glob(str(Path(rd.OUT_ROOT) / rd.DELIVERY / "aoi*" / "aoi*_v2.json"))):
        r = json.loads(Path(f).read_text())
        tc = r.get("trees_cut", {})
        rows.append({"aoi": b2.delivery_name(r["aoi"]), "rules_v2": r["rules_from"],
                     "right_pct_before": r["delivered_right_pct_before"],
                     "right_pct_after": r["delivered_right_pct_after"],
                     "check_half_before": r["check_half_before"], "check_half_after": r["check_half_after"],
                     "rice_acres_v1": r["rice_acres_v1"], "rice_acres_v2": r["rice_acres_v2"],
                     "too_young_acres_v2": r["too_young"].get("too_young_acres"),
                     "rice_acres_v2_trees_cut": tc.get("rice_acres_after")})
    t = pd.DataFrame(rows)
    t.to_csv(Path(rd.OUT_ROOT) / rd.DELIVERY / "weak_v2_summary.csv", index=False)
    return t


def table(out: Path = OUT) -> pd.DataFrame:
    """All AOI records in one table, written to ``weak_rules.csv``."""
    rows = [json.loads(p.read_text()) for p in sorted(out.glob("aoi*.json"))]
    t = pd.DataFrame(rows).drop(columns=["rules", "trail"], errors="ignore")
    t.to_csv(out / "weak_rules.csv", index=False)
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["search", "table", "build-v2", "upload-v2", "v2-summary", "build", "upload",
                                      "round2-better"])
    p.add_argument("--aois", type=int, nargs="*", default=[])
    p.add_argument("--version", default="v3", help="build / upload: v2 or v3")
    p.add_argument("--round", default="", help="search / table: sub-folder of weak_rules/ for a later search round "
                                               "(round2: with EXTRA_SWITCHES), so the adopted records stay")
    a = p.parse_args(argv)
    p_ver = a.version
    out = OUT / a.round if a.round else OUT
    if a.step == "search":
        for x in a.aois:
            r = search(x, out)
            print(json.dumps({k: v for k, v in r.items() if k != "rules"}, default=str), flush=True)
    if a.step == "v2-summary":
        print(v2_summary().to_string(index=False))
        return 0
    if a.step == "round2-better":
        print(*[x for x in sorted(int(q.stem[3:]) for q in OUT.glob("aoi*.json")) if round2_better(x)])
        return 0
    if a.step in ("build-v2", "upload-v2", "build", "upload"):
        ver = "v2" if a.step.endswith("-v2") else p_ver
        fn = (lambda x: build_version(x, ver)) if a.step.startswith("build") else (lambda x: upload_version(x, ver))
        for x in a.aois:
            try:
                r = fn(x)
                print(json.dumps({k: v for k, v in r.items() if k != "rules"}, default=str), flush=True)
            except Exception as e:                     # one AOI's failure (e.g. credentials) does not stop the rest
                print(json.dumps({"aoi": x, "error": f"{type(e).__name__}: {e}"}), flush=True)
        return 0
    t = table(out)
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
