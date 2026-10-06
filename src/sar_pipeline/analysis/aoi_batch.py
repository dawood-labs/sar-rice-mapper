"""Batch runner: many AOIs from new imagery to a delivered rice map, resumable, neighbours together.

Why
---
User (5-6 Oct 2026): process every AOI the way aoi125 was done - reviewers judge a sample of fields, the rule set that
matches them best makes the map - as fast as the machine allows, never repeating a finished step. One AOI took about
75 minutes step by step; most of it was waiting (Earth Engine) or one-process work. This runner

* submits the radar exports of a whole batch at once and waits for all of them together (``imagery``),
* prepares several AOIs at once, each step one process per rule set where that helps (``prepare``): pass screening,
  pins, the 5-day series, steps 1-2, every finished AOI's rule set side by side, the field sample and its blind sheets,
* scores every rule set and switch against the reviewers' verdicts and records the winner (``choose``) in
  ``aoi_rules_chosen.json`` (tracked, read by ``curve_rules`` at import, so the map can always be rebuilt),
* finishes the AOI (``finish``): raw map, sieve, fields, lock, delivery folder and upload, rule record.

The reviewers themselves (``docs/field_review_agent_brief.md``) are run by the operator between ``prepare`` and
``choose``; ``prepare`` writes ``field_review/groups.txt`` (20 fields a line) for them.

Each AOI has ``processed/_batch/s2_2026/aoi_batch/aoi<N>/status.json``; a finished step is skipped on a rerun.

Use::

    python -m sar_pipeline.analysis.aoi_batch order --start 72 116 39            # neighbour order of all AOIs
    python -m sar_pipeline.analysis.aoi_batch imagery --aois 72 116 39
    python -m sar_pipeline.analysis.aoi_batch prepare --aois 72 116 39
    python -m sar_pipeline.analysis.aoi_batch choose  --aois 72
    python -m sar_pipeline.analysis.aoi_batch finish  --aois 72
    python -m sar_pipeline.analysis.aoi_batch status  --aois 72 116 39
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FRESH = "processed/_batch/s2_2026/rice_fresh"
STATE = Path("processed/_batch/s2_2026/aoi_batch")
#: The inputs of this delivery round (the same for every AOI, so all maps share one map date).
SEASON_END = "2026-10-05"
S2_NEW = ("2026-10-01", "2026-10-02")
SERIES_VARIANT = "hyb40m1late_20261001"
#: Rule sets compared on every new AOI: every AOI with its own reviewed set, plus the defaults (160).
CANDIDATE_SETS = (160, 28, 20, 33, 72, 116, 39, 118, 83, 13, 125)
#: Switches tried on top of the best set (each alone).
CANDIDATE_SWITCHES = {
    "any_water_transplanted": {"ANY_WATER_TRANSPLANTED": True},
    "young_while_radar_low": {"YOUNG_WHILE_RADAR_LOW": True},
    "long_flood": {"LONG_WATER_DAYS": 30, "LONG_WATER_LEVEL": 0.4, "LONG_WATER_WET_VIEW": True,
                   "LONG_WATER_GREEN_LEAD": 0},
    "sowing_from_radar": {"SOWING_FROM_RADAR": True},
}
#: Field sample: 5 % of the AOI's fields, at least 200, at most 400 (user, 5 Oct).
SAMPLE_SHARE, SAMPLE_MIN, SAMPLE_MAX = 0.05, 200, 400
CHOSEN = Path(__file__).with_name("aoi_rules_chosen.json")
PY = [sys.executable, "-W", "ignore", "-m"]


def _status_path(aoi: int) -> Path:
    return STATE / f"aoi{aoi}" / "status.json"


def status(aoi: int) -> dict:
    p = _status_path(aoi)
    return json.loads(p.read_text()) if p.exists() else {}


def _mark(aoi: int, step: str, result) -> None:
    st = status(aoi)
    st[step] = {"done": True, "result": result, "at": str(pd.Timestamp.now().floor("s"))}
    p = _status_path(aoi)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1, default=str))
    tmp.replace(p)


def _done(aoi: int, step: str) -> bool:
    return bool(status(aoi).get(step, {}).get("done"))


def _sh(args, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, args))} failed (exit {r.returncode}), see {log}")


# ------------------------------------------------------------------------------------------------ neighbour order
def neighbour_order(start, aois=None, aoi_dir: str = "data/aoi") -> list[int]:
    """All AOIs in a nearest-neighbour chain from ``start`` (centroid distances), so neighbours run together and each
    AOI's chosen set is tried first on the next one."""
    import geopandas as gpd

    pts = {}
    for p in sorted(Path(aoi_dir).glob("aoi_[0-9]*.gpkg")):
        a = int(p.stem.split("_")[1])
        if aois is None or a in aois:
            g = gpd.read_file(p).to_crs("EPSG:32646")
            pts[a] = np.array(g.geometry.union_all().centroid.coords[0])
    order = [a for a in start if a in pts]
    left = set(pts) - set(order)
    while left:
        last = pts[order[-1]]
        nxt = min(left, key=lambda a: float(np.hypot(*(pts[a] - last))))
        order.append(nxt)
        left.remove(nxt)
    return order


# ------------------------------------------------------------------------------------------------------- imagery
def _set_season_end(aoi: int) -> None:
    p = Path(f"config/aoi{aoi}_monsoon2026.yaml")
    import re

    s = p.read_text()
    s2 = re.sub(r"(season: \{[^}]*end: ')[0-9-]+(')", rf"\g<1>{SEASON_END}\g<2>", s, count=1)
    if s2 != s:
        p.write_text(s2)


def _new_run(aoi: int, logs: Path) -> str:
    cfg = f"config/aoi{aoi}_monsoon2026.yaml"
    out = logs / f"aoi{aoi}_newrun.log"
    _sh(PY + ["sar_pipeline", "--config", cfg, "audit"], logs / f"aoi{aoi}_audit.log")
    _sh(PY + ["sar_pipeline", "--config", cfg, "new-run"], out)
    import re

    runs = re.findall(r"/runs/(\S+)", out.read_text())
    if not runs:
        raise RuntimeError(f"aoi{aoi}: no run id in {out}")
    return runs[-1].rstrip("/").split("/")[0]


def imagery(aois, logs: Path) -> None:
    """New radar run + new S2 dates for every AOI of the batch: exports submitted for all first, then all monitored
    together (Earth Engine runs them in parallel), then downloaded and stacked."""
    from concurrent.futures import ThreadPoolExecutor

    todo = [a for a in aois if not _done(a, "imagery")]
    if not todo:
        return
    for a in todo:
        _set_season_end(a)
    _sh(PY + ["sar_pipeline.optical_export", "dates", "--configs",
              *[f"config/aoi{a}_monsoon2026.yaml" for a in todo], "--start", S2_NEW[0], "--end", S2_NEW[1]],
        logs / "s2_dates.log")
    runs = {}
    for a in todo:
        runs[a] = status(a).get("run") or _new_run(a, logs)
        st = status(a)
        st["run"] = runs[a]
        _status_path(a).parent.mkdir(parents=True, exist_ok=True)
        _status_path(a).write_text(json.dumps(st, indent=1, default=str))
        _sh(PY + ["sar_pipeline", "--config", f"config/aoi{a}_monsoon2026.yaml", "export", "--run", runs[a], "--all",
                  "--yes"], logs / f"aoi{a}_export.log")

    def finish(a):
        cfg = f"config/aoi{a}_monsoon2026.yaml"
        for stage in ("monitor", "download"):
            _sh(PY + ["sar_pipeline", "--config", cfg, stage, "--run", runs[a], "--yes"], logs / f"aoi{a}_{stage}.log")
        _sh(PY + ["sar_pipeline", "--config", cfg, "stack", "--run", runs[a]], logs / f"aoi{a}_stack.log")
        _mark(a, "imagery", {"run": runs[a]})
        return a

    with ThreadPoolExecutor(len(todo)) as ex:
        for a in ex.map(finish, todo):
            print(f"aoi{a}: imagery done (run {runs[a]})", flush=True)


# ------------------------------------------------------------------------------------------------------- prepare
def _prepare_one(aoi: int, logs: str) -> str:
    """Screening, pins, series, steps 1-2, rule-set trials, field sample and blind sheets for one AOI."""
    from . import curve_rules as cr
    from . import field_review as fr
    from . import ndvi_5day as nd

    logs = Path(logs)
    run = status(aoi)["imagery"]["result"]["run"]
    cfg = f"config/aoi{aoi}_monsoon2026.yaml"
    sr = f"processed/_batch/s2_2026_{SERIES_VARIANT}"
    if not _done(aoi, "inputs"):
        _sh(PY + ["sar_pipeline.analysis.final_audit", "new-passes", "--ids", str(aoi), "--run", run],
            logs / f"aoi{aoi}_newpasses.log")
        _sh(PY + ["sar_pipeline", "--config", cfg, "pin-run", "--run", run, "--replace"], logs / f"aoi{aoi}_pin.log")
        _sh(PY + ["sar_pipeline.analysis.mask_experiment", "build", "--ids", str(aoi), "--variants", SERIES_VARIANT,
                  "--jobs", "1"], logs / f"aoi{aoi}_series.log")
        nd.pin_analysis_series(aoi, sr, replace=True)
        for args in (["sar_pipeline.analysis.first_clear", "--aoi", str(aoi), "--start", "2026-09-01",
                      "--end", "2026-09-30"],
                     ["sar_pipeline.analysis.first_clear", "--aoi", str(aoi), "--vegetation"],
                     ["sar_pipeline.analysis.vegetation_types", "--aoi", str(aoi), "--series-root", sr],
                     ["sar_pipeline.analysis.sowing_fresh", "--aoi", str(aoi), "--series-root", sr]):
            _sh(PY + args, logs / f"aoi{aoi}_steps.log")
        _mark(aoi, "inputs", {"run": run, "series": sr})
    if not _done(aoi, "trials"):
        t = cr.try_rules(aoi, sources=CANDIDATE_SETS)
        t.to_csv(logs / f"aoi{aoi}_trials.csv", index=False)
        # the provisional map the field sample is stratified on: the defaults' trial (no choice made yet)
        import shutil

        src = Path(FRESH) / f"aoi{aoi}" / "rule_trials" / "rules_aoi160" / f"aoi{aoi}"
        for n in (f"aoi{aoi}_rel_class.tif", f"aoi{aoi}_rel_class.qml"):
            shutil.copy2(src / n, Path(FRESH) / f"aoi{aoi}" / n)
        _mark(aoi, "trials", {"sets": list(CANDIDATE_SETS)})
    if not _done(aoi, "sheets"):
        fc = fr.field_classes(aoi)
        n = int(min(SAMPLE_MAX, max(SAMPLE_MIN, round(SAMPLE_SHARE * len(fc)))))
        fields = fr.pick(aoi, n=n)
        _sh(PY + ["sar_pipeline.analysis.field_review", "render", "--aoi", str(aoi)], logs / f"aoi{aoi}_render.log")
        ids = fields["field_id"].tolist()
        (fr.review_dir(aoi) / "groups.txt").write_text("\n".join(" ".join(ids[i:i + 20])
                                                                  for i in range(0, len(ids), 20)) + "\n")
        _mark(aoi, "sheets", {"fields": len(ids), "groups": (len(ids) + 19) // 20})
    return f"aoi{aoi}: prepared"


def prepare(aois, logs: Path, jobs: int = 0) -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .. import resources

    todo = [a for a in aois if _done(a, "imagery") and not _done(a, "sheets")]
    r = resources.detect_resources()
    n = jobs or max(1, min(len(todo), 3, int(r.memory_available_bytes / 15e9)))
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_prepare_one, a, str(logs)): a for a in todo}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:                    # one AOI's failure must not stop the batch
                print(f"aoi{futs[f]} FAILED in prepare: {type(e).__name__}: {e}", flush=True)


# -------------------------------------------------------------------------------------------------------- choose
def choose(aoi: int) -> dict:
    """Scores every candidate set, then each switch on top of the best set, against the reviewers' verdicts; the
    winner (most fields right; ties: most right with standing / young merged) goes to ``aoi_rules_chosen.json``."""
    from . import curve_rules as cr
    from . import field_review as fr
    from . import rule_records as rr

    co = (lambda x: "rice" if str(x).startswith("rice standing") or x == "young rice" else x)
    sets = {f"aoi{s}": dict(cr.REVIEWED_SETS.get(s, {})) for s in CANDIDATE_SETS}
    o, r = fr.try_variants(aoi, {}, sets=sets)
    merged = {c: round(100 * float((o[c].map(co) == o["truth"].map(co)).mean()), 1) for c in sets}
    best = max(sets, key=lambda c: (r[c], merged[c]))
    v = {n: {**sets[best], **sw} for n, sw in CANDIDATE_SWITCHES.items()}
    o2, r2 = fr.try_variants(aoi, {}, sets={best: sets[best], **{f"{best}+{n}": x for n, x in v.items()}})
    m2 = {c: round(100 * float((o2[c].map(co) == o2["truth"].map(co)).mean()), 1) for c in o2.columns[3:]}
    win = max(m2, key=lambda c: (r2[c], m2[c]))
    rules = sets[best] if win == best else v[win.split("+", 1)[1]]
    rules = {**rules, "SIEVE_ACRES": 0.15}
    table = pd.DataFrame([{"rule set": c, "fields right %": r[c], "standing/young merged %": merged[c]} for c in sets]
                         + [{"rule set": c, "fields right %": r2[c], "standing/young merged %": m2[c]}
                            for c in m2 if c != best]).sort_values("fields right %", ascending=False)
    chosen = json.loads(CHOSEN.read_text()) if CHOSEN.exists() else {}
    chosen[str(aoi)] = {"rules": rules, "from": win, "fields": int(len(o)), "right_pct": float(r2[win]),
                        "merged_pct": float(m2[win]), "at": str(pd.Timestamp.now().floor("s"))}
    CHOSEN.write_text(json.dumps(chosen, indent=1, sort_keys=True, default=str))
    cr.AOI_OVERRIDES[aoi] = rules
    rr.record(aoi, "rule set chosen", f"`{win}` (switches below)",
              reason=f"best of {len(table)} sets / switches on {len(o)} fields judged by reviewers", scores=table)
    _mark(aoi, "choose", chosen[str(aoi)])
    return chosen[str(aoi)]


# -------------------------------------------------------------------------------------------------------- finish
def finish(aoi: int) -> dict:
    """Raw map, sieve and fields with the chosen rules, then lock, delivery and upload (``rice_map_delivery``). An AOI
    locked in an earlier round keeps its old lock as ``locked/aoi<N>_v1`` (user, 5 Oct: redo the earlier AOIs)."""
    import shutil

    from . import curve_rules as cr
    from . import rice_map_delivery as rmd
    from . import rule_records as rr

    lk = Path(FRESH) / "locked" / f"aoi{aoi}"
    if lk.exists() and not _done(aoi, "finish"):
        old = lk.with_name(f"aoi{aoi}_v1")
        if not old.exists():
            shutil.copytree(lk, old)
        shutil.rmtree(lk)
    if not _done(aoi, "map"):
        cr.run(aoi, force=True)
        cr.sieve(aoi, force=True)
        cr.fields(aoi, force=True)
        _mark(aoi, "map", {"sieve_acres": 0.15})
    # delivered in an earlier round: lock, package and upload again with the new map
    st = rmd.run_one(aoi, redo=("lock", "package", "upload") if rmd.load_status(aoi) else ())
    rr.record(aoi, "locked and delivered", f"{st['upload']['result']['gcs']} ({len(st['package']['result']['s2_dates'])}"
                                           f" S2 dates >= 80 % clear)")
    _mark(aoi, "finish", st["upload"]["result"])
    return st["upload"]["result"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["order", "imagery", "prepare", "choose", "finish", "status"])
    p.add_argument("--aois", type=int, nargs="*", default=[])
    p.add_argument("--start", type=int, nargs="*", default=[])
    p.add_argument("--logs", default="logs/aoi_batch")
    p.add_argument("--jobs", type=int, default=0)
    a = p.parse_args(argv)
    logs = Path(a.logs)
    if a.step == "order":
        print(" ".join(map(str, neighbour_order(a.start or [72]))))
    elif a.step == "imagery":
        imagery(a.aois, logs)
    elif a.step == "prepare":
        prepare(a.aois, logs, a.jobs)
    elif a.step == "choose":
        for aoi in a.aois:
            print(aoi, choose(aoi), flush=True)
    elif a.step == "finish":
        for aoi in a.aois:
            print(aoi, finish(aoi), flush=True)
    else:
        steps = ("imagery", "inputs", "trials", "sheets", "choose", "map", "finish")
        for aoi in a.aois:
            st = status(aoi)
            print(f"aoi{aoi}: " + " ".join(f"{s}={'ok' if st.get(s, {}).get('done') else '-'}" for s in steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
