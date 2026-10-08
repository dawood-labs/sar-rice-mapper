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


def table() -> pd.DataFrame:
    """All AOI records in one table, written to ``weak_rules.csv``."""
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("aoi*.json"))]
    t = pd.DataFrame(rows).drop(columns=["rules", "trail"], errors="ignore")
    t.to_csv(OUT / "weak_rules.csv", index=False)
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["search", "table"])
    p.add_argument("--aois", type=int, nargs="*", default=[])
    a = p.parse_args(argv)
    if a.step == "search":
        for x in a.aois:
            r = search(x)
            print(json.dumps({k: v for k, v in r.items() if k != "rules"}, default=str), flush=True)
    t = table()
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
