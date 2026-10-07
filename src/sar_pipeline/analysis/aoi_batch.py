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
    python -m sar_pipeline.analysis.aoi_batch flow --start 72 116 39 --skip 160 28 ...   # all AOIs, rolling window
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
#: Rule sets compared on every new AOI: every AOI with its own reviewed set, the defaults (160), the junior's aoi63 set
#: and the junior's universal new-AOI switches (``curve_rules.NAMED_SETS``; user, 6 Oct).
CANDIDATE_SETS = (160, 28, 20, 33, 72, 116, 39, 118, 83, 13, 125, 63, "universal")
#: Switches tried on top of the best set (each alone).
CANDIDATE_SWITCHES = {
    "any_water_transplanted": {"ANY_WATER_TRANSPLANTED": True},
    "young_while_radar_low": {"YOUNG_WHILE_RADAR_LOW": True},
    "long_flood": {"LONG_WATER_DAYS": 30, "LONG_WATER_LEVEL": 0.4, "LONG_WATER_WET_VIEW": True,
                   "LONG_WATER_GREEN_LEAD": 0},
    "sowing_from_radar": {"SOWING_FROM_RADAR": True},
    "universal": None,                     # the junior's universal switches on top of the best set (filled below)
}
#: Field sample: 5 % of the AOI's fields, at least 200, at most 400 (user, 5 Oct).
SAMPLE_SHARE, SAMPLE_MIN, SAMPLE_MAX = 0.05, 200, 400
#: Fields of at least this size can be sampled (4 pixels). Why (6 Oct): with 0.5 ac, 49 AOIs of small fields had fewer
#: than 200 candidates (aoi66: 5, aoi67: 23) and their rule set was chosen on far too few verdicts.
SAMPLE_MIN_ACRES = 0.1
CHOSEN = Path(__file__).with_name("aoi_rules_chosen.json")


def _fill_universal() -> None:
    from . import curve_rules as cr

    CANDIDATE_SWITCHES["universal"] = dict(cr.NEW_AOI_SWITCHES)
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


# ---------------------------------------------------------------------------------------------------------- flow
#: A radar date with fewer valid pixels than the QA minimum (80 %) is accepted as a partial date when at least this
#: share is valid; the missing pixels stay empty and the rules skip them. Why (7 Oct 2026): some AOIs of the second
#: set sit on a swath edge (72-77 % valid on 5 dates of one track); the first set never had this. Below it, or for
#: any other QA issue, the AOI stops for the user.
PARTIAL_MIN_PCT = 50.0


def acknowledge_partial_dates(aoi: int, run: str, min_pct: float = PARTIAL_MIN_PCT) -> list[str]:
    """Adds the run's LOW_VALID issues with at least ``min_pct`` valid pixels to the config's
    ``qa.acknowledged_issues``; returns them, or [] when any open issue is of another kind or lower."""
    import re

    import yaml

    md = Path(f"processed/aoi{aoi}/monsoon2026/runs/{run}/decisions_required.md")
    if not md.exists():
        return []
    rows = re.findall(r"\| `([A-Z_]+:[^`]+)` \| no \| (.+?) \|", md.read_text())
    ok = []
    for issue, detail in rows:
        pct = [float(x) for x in re.findall(r"V[VH] ([0-9.]+)%", detail)]
        if not issue.startswith("LOW_VALID:") or not pct or min(pct) < min_pct:
            return []
        ok.append(issue)
    path = Path(f"config/aoi{aoi}_monsoon2026.yaml")
    head = "".join(line + "\n" for line in path.read_text().splitlines() if line.startswith("#"))
    raw = yaml.safe_load(path.read_text())
    have = list(raw.setdefault("qa", {}).get("acknowledged_issues") or [])
    raw["qa"]["acknowledged_issues"] = have + [i for i in ok if i not in have]
    path.write_text(head + yaml.safe_dump(raw, sort_keys=False, default_flow_style=None))
    return ok


def _download_stack(aoi: int, run: str, logs: Path) -> int:
    cfg = f"config/aoi{aoi}_monsoon2026.yaml"
    _sh(PY + ["sar_pipeline", "--config", cfg, "download", "--run", run, "--yes"], logs / f"aoi{aoi}_download.log")
    try:
        _sh(PY + ["sar_pipeline", "--config", cfg, "stack", "--run", run], logs / f"aoi{aoi}_stack.log")
    except RuntimeError:
        acked = acknowledge_partial_dates(aoi, run)
        if not acked:
            raise
        from . import rule_records as rr
        rr.record(aoi, "partial radar dates accepted", f"{len(acked)} dates with {PARTIAL_MIN_PCT:.0f}-80 % valid "
                                                       f"pixels: {', '.join(acked)}")
        _sh(PY + ["sar_pipeline", "--config", cfg, "stack", "--run", run], logs / f"aoi{aoi}_stack.log")
    return aoi


def _submit(aoi: int, logs: Path) -> tuple[int, str]:
    run = status(aoi).get("run") or _new_run(aoi, logs)
    st = status(aoi)
    st["run"] = run
    _status_path(aoi).parent.mkdir(parents=True, exist_ok=True)
    _status_path(aoi).write_text(json.dumps(st, indent=1, default=str))
    _sh(PY + ["sar_pipeline", "--config", f"config/aoi{aoi}_monsoon2026.yaml", "export", "--run", run, "--all", "--yes"],
        logs / f"aoi{aoi}_export.log")
    return aoi, run


def flow(order, logs: Path, window: int = 30, prep_jobs: int = 0, poll: float = 30.0, sleep=None) -> None:
    """The whole imagery-to-sheets stage as one flow (user, 6 Oct: keep the Earth Engine queue full, check every 30 s,
    go on with each AOI the moment its data is there):

    * ``window`` AOIs are always in Earth Engine; when one AOI's exports are all finished the next one is submitted,
    * every ``poll`` seconds ONE project task listing serves all AOIs (one shared Earth Engine backend: its listing is
      re-used for ``poll`` seconds), then each run's monitor round (refresh + resubmit failures) runs on it,
    * a finished AOI is downloaded and stacked at once, then prepared (inputs, trials, sample, sheets) in a process
      pool; ``READY aoi<N>`` is printed when its sheets are done (the reviewers' turn).
    Resumable: AOIs already past a step (``status.json``) skip it."""
    import time as _time
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    from .. import auth, cli, export
    from .. import config as cm
    from .. import resources

    sleep = sleep or _time.sleep
    logs.mkdir(parents=True, exist_ok=True)
    queue = [a for a in order if not _done(a, "imagery")]
    r = resources.detect_resources()
    prep = ProcessPoolExecutor(prep_jobs or max(1, min(3, int(r.memory_available_bytes / 15e9))),
                               mp_context=mp.get_context("spawn"))
    io = ThreadPoolExecutor(8)
    backend, inflight, downloads, preps = None, {}, {}, {}
    for a in order:                                   # imagery already done earlier: straight to preparation
        if _done(a, "imagery") and not _done(a, "sheets"):
            preps[a] = prep.submit(_prepare_one, a, str(logs))
    say = (lambda m: print(f"{pd.Timestamp.now():%H:%M:%S} {m}", flush=True))
    while queue or inflight or downloads or preps:
        free = window - len(inflight) - len(downloads)
        if free > 0 and queue:
            take, queue[:] = queue[:free], queue[free:]
            for a in take:
                _set_season_end(a)
            try:
                _sh(PY + ["sar_pipeline.optical_export", "dates", "--configs",
                          *[f"config/aoi{a}_monsoon2026.yaml" for a in take], "--start", S2_NEW[0], "--end", S2_NEW[1]],
                    logs / "s2_dates.log")
            except RuntimeError as e:
                say(f"S2 dates for {take} FAILED ({e}); radar goes on")
            for f in [io.submit(_submit, a, logs) for a in take]:
                try:
                    a, run = f.result()
                except Exception as e:
                    say(f"submit FAILED: {e}")
                    continue
                cfg, rp = cli.load_run_context(cm.load_config(f"config/aoi{a}_monsoon2026.yaml"), run,
                                               warn=lambda m: None)
                if backend is None:
                    auth.init_ee(cfg)
                    backend = export.EEBackend()
                inflight[a] = (cfg, rp, run)
                say(f"aoi{a}: submitted (run {run}); in Earth Engine: {len(inflight)}")
        for a, (cfg, rp, run) in list(inflight.items()):
            try:
                df = export.monitor(cfg, rp, backend=backend, confirmed=True, max_cycles=1)
            except Exception as e:                    # one run's trouble must not stop the others
                say(f"aoi{a}: monitor round failed ({type(e).__name__}: {e}); next round")
                continue
            if not df["state"].isin(export._OPEN_STATES).any():
                n_failed = int(df["state"].isin(["FAILED"]).sum())
                say(f"aoi{a}: exports finished ({len(df)} tasks, {n_failed} failed); downloading")
                del inflight[a]
                downloads[a] = (io.submit(_download_stack, a, run, logs), run)
        for a, (f, run) in list(downloads.items()):
            if f.done():
                del downloads[a]
                try:
                    f.result()
                    _mark(a, "imagery", {"run": run})
                    preps[a] = prep.submit(_prepare_one, a, str(logs))
                except Exception as e:
                    say(f"aoi{a}: download / stack FAILED: {e}")
        for a, f in list(preps.items()):
            if f.done():
                del preps[a]
                try:
                    msg = f.result()
                    if isinstance(msg, str) and ("neighbour" in msg or "waits" in msg):
                        say(f"INPUTS {msg}")
                    else:
                        say(f"READY aoi{a}: sheets done, groups in {Path(FRESH) / f'aoi{a}' / 'field_review' / 'groups.txt'}")
                except Exception as e:
                    say(f"aoi{a}: prepare FAILED: {type(e).__name__}: {e}")
        if queue or inflight or downloads or preps:
            sleep(poll)
    prep.shutdown()
    io.shutdown()


# ------------------------------------------------------------------------------------------------------- prepare
def _prepare_one(aoi: int, logs: str) -> str:
    """Screening, pins, series, steps 1-2, rule-set trials, field sample and blind sheets for one AOI."""
    from . import curve_rules as cr
    from . import field_review as fr
    from . import ndvi_5day as nd

    from . import aoi_batch2 as b2

    if b2.is_batch2(aoi) and not b2.s2_ready(aoi):    # the full S2 export first (aoi_batch2.s2_export)
        return f"aoi{aoi}: waits for its full Sentinel-2 export"
    logs = Path(logs)
    run = status(aoi)["imagery"]["result"]["run"]
    cfg = f"config/aoi{aoi}_monsoon2026.yaml"
    sr = f"processed/_batch/s2_2026_{SERIES_VARIANT}"
    if not _done(aoi, "inputs"):
        # the series first: the artefact-pass check needs its evergreen pixels, and an AOI of the second set
        # (aoi_batch2) has no older series to borrow them from (7 Oct 2026)
        _sh(PY + ["sar_pipeline", "--config", cfg, "pin-run", "--run", run, "--replace"], logs / f"aoi{aoi}_pin.log")
        _sh(PY + ["sar_pipeline.analysis.mask_experiment", "build", "--ids", str(aoi), "--variants", SERIES_VARIANT,
                  "--jobs", "1"], logs / f"aoi{aoi}_series.log")
        nd.pin_analysis_series(aoi, sr, replace=True)
        _sh(PY + ["sar_pipeline.analysis.final_audit", "new-passes", "--ids", str(aoi), "--run", run],
            logs / f"aoi{aoi}_newpasses.log")
        for args in (["sar_pipeline.analysis.first_clear", "--aoi", str(aoi), "--start", "2026-09-01",
                      "--end", "2026-09-30"],
                     ["sar_pipeline.analysis.first_clear", "--aoi", str(aoi), "--vegetation"],
                     ["sar_pipeline.analysis.vegetation_types", "--aoi", str(aoi), "--series-root", sr],
                     ["sar_pipeline.analysis.sowing_fresh", "--aoi", str(aoi), "--series-root", sr]):
            _sh(PY + args, logs / f"aoi{aoi}_steps.log")
        _mark(aoi, "inputs", {"run": run, "series": sr})
    if b2.is_batch2(aoi) and aoi not in b2.REVIEW:    # rule set from a neighbour, no review (aoi_batch2.finish_loop)
        return f"aoi{aoi}: inputs done; rule set by neighbour"
    if b2.is_batch2(aoi):                              # the review's field file (the first set had one already)
        b2.review_fields(aoi)
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
        fc = fr.field_classes(aoi, min_acres=SAMPLE_MIN_ACRES)
        n = int(min(SAMPLE_MAX, max(SAMPLE_MIN, round(SAMPLE_SHARE * len(fc)))))
        prev = fr.review_dir(aoi) / "fields.csv"
        have = len(pd.read_csv(prev)) if prev.exists() else 0      # an earlier, smaller sample is kept and topped up
        if n > have:
            fr.pick(aoi, n=n - have, min_acres=SAMPLE_MIN_ACRES)
        _sh(PY + ["sar_pipeline.analysis.field_review", "render", "--aoi", str(aoi)], logs / f"aoi{aoi}_render.log")
        vd = fr.review_dir(aoi) / "verdicts"
        all_ids = pd.read_csv(prev)["field_id"].tolist()
        ids = [i for i in all_ids if not (vd / f"{i}.json").exists()]   # only fields still without a verdict
        (fr.review_dir(aoi) / "groups.txt").write_text("\n".join(" ".join(ids[i:i + 20])
                                                                  for i in range(0, len(ids), 20)) + "\n")
        _mark(aoi, "sheets", {"fields": len(all_ids), "groups": (len(ids) + 19) // 20})
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

    _fill_universal()
    co = (lambda x: "rice" if str(x).startswith("rice standing") or x == "young rice" else x)
    sets = {(s if isinstance(s, str) else f"aoi{s}"): dict(cr.NAMED_SETS[s] if isinstance(s, str)
                                                          else cr.REVIEWED_SETS.get(s, {})) for s in CANDIDATE_SETS}
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
    # an earlier round's lock (present before this runner rebuilt the map) is kept as aoi<N>_v1; a lock this runner
    # started (map already rebuilt here, e.g. a run killed half-way through locking) is simply replaced
    if lk.exists() and not _done(aoi, "finish"):
        old = lk.with_name(f"aoi{aoi}_v1")
        if not _done(aoi, "map") and (lk / "MANIFEST.json").exists() and not old.exists():
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


def target_sample(aoi: int) -> int:
    from . import field_review as fr

    fc = fr.field_classes(aoi, min_acres=SAMPLE_MIN_ACRES)
    return int(min(len(fc), SAMPLE_MAX, max(SAMPLE_MIN, round(SAMPLE_SHARE * len(fc)))))


def resample(order) -> list[int]:
    """AOIs whose sample is below the target (``SAMPLE_MIN_ACRES`` rule) get their sheets step reopened (topped up, not
    redrawn) and, if already chosen / finished, those steps too; a lock made from the too-small sample is removed (it
    was never accepted). Returns the AOIs reopened."""
    import shutil

    from . import field_review as fr

    out = []
    for a in order:
        st = status(a)
        if not st.get("sheets", {}).get("done"):
            continue
        prev = fr.review_dir(a) / "fields.csv"
        have = len(pd.read_csv(prev)) if prev.exists() else 0
        want = target_sample(a)
        if have >= want:
            continue
        finished_here = bool(st.get("finish", {}).get("done"))
        for k in ("sheets", "choose", "map", "finish"):
            st.pop(k, None)
        _status_path(a).write_text(json.dumps(st, indent=1, default=str))
        lk = Path(FRESH) / "locked" / f"aoi{a}"
        # only a lock THIS runner made (the AOI was finished here) is removed; an earlier round's lock is never touched
        # (finish keeps it as locked/aoi<N>_v1 before replacing it)
        if finished_here and lk.exists():
            shutil.rmtree(lk)
        out.append(a)
        print(f"aoi{a}: sample {have} < {want}: reopened", flush=True)
    return out


def _choose_finish(aoi: int) -> dict:
    if not _done(aoi, "choose"):
        choose(aoi)
    return finish(aoi)


def autofinish(order, poll: float = 60.0, jobs: int = 3, sleep=None) -> None:
    """Watches the AOIs whose sheets are out: when every sampled field has a reviewer verdict, the AOI's rule set is
    chosen and the AOI finished (map, lock, delivery, upload) in a process pool. Prints ``DELIVERED aoi<N> <gcs>``."""
    import time as _time
    from concurrent.futures import ProcessPoolExecutor

    from . import field_review as fr

    sleep = sleep or _time.sleep
    say = (lambda m: print(f"{pd.Timestamp.now():%H:%M:%S} {m}", flush=True))
    pool = ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn"))
    running = {}
    while True:
        for a in order:
            if a in running or _done(a, "finish") or not _done(a, "sheets"):
                continue
            want = int(status(a)["sheets"]["result"]["fields"])
            if want < target_sample(a):                   # a sample below the target is never used (6 Oct)
                continue
            have = len(list((fr.review_dir(a) / "verdicts").glob("*.json"))) if (fr.review_dir(a) / "verdicts").exists() \
                else 0
            if want and have >= want:
                running[a] = pool.submit(_choose_finish, a)
                say(f"aoi{a}: {have} verdicts in, choosing and finishing")
        for a, f in list(running.items()):
            if f.done():
                del running[a]
                try:
                    say(f"DELIVERED aoi{a} {f.result().get('gcs')}  rule: {status(a).get('choose', {}).get('result', {}).get('from')}"
                        f" ({status(a).get('choose', {}).get('result', {}).get('right_pct')} % of fields)")
                except Exception as e:
                    say(f"aoi{a}: choose / finish FAILED: {type(e).__name__}: {e}")
        if all(_done(a, "finish") for a in order) and not running:
            break
        sleep(poll)
    pool.shutdown()


def pending_review(order) -> pd.DataFrame:
    """AOIs whose sheets are out but whose sampled fields do not all have a verdict yet, with the groups.txt lines that
    still hold fields without one. Why (6 Oct): READY notices can be missed (a watcher expired); this lists the work
    from the files themselves."""
    from . import field_review as fr

    rows = []
    for a in order:
        if not _done(a, "sheets") or _done(a, "finish"):
            continue
        d = fr.review_dir(a)
        lines = [l.split() for l in (d / "groups.txt").read_text().splitlines() if l.strip()]
        todo = [i + 1 for i, ids in enumerate(lines) if any(not (d / "verdicts" / f"{x}.json").exists() for x in ids)]
        if todo:
            rows.append({"aoi": a, "lines": todo})
    return pd.DataFrame(rows, columns=["aoi", "lines"])


STAGES = ("delivered (first round, locked)", "delivered", "verdicts in: choosing / finishing", "under review",
          "sheets ready, waiting for review", "preparing (inputs, rule trials, sheets)",
          "imagery (Earth Engine / download)")


def stage_of(aoi: int, first_round=()) -> str:
    """The one stage an AOI is in now, read from its status file and its verdicts folder. Why (6 Oct): the user asks
    how many AOIs are done, in process and pending; this answers from the files, not from log lines."""
    from . import field_review as fr

    st = status(aoi)
    if _done(aoi, "finish"):
        return STAGES[1]
    if not st and aoi in set(first_round):
        return STAGES[0]
    if _done(aoi, "sheets"):
        want = int((st["sheets"].get("result") or {}).get("fields", 0)) if isinstance(st["sheets"].get("result"),
                                                                                       dict) else 0
        v = fr.review_dir(aoi) / "verdicts"
        have = len(list(v.glob("*.json"))) if v.exists() else 0
        return STAGES[2] if want and have >= want else STAGES[3] if have else STAGES[4]
    return STAGES[5] if _done(aoi, "imagery") else STAGES[6]


def stage_summary(aois, first_round=()) -> pd.DataFrame:
    """One row per stage: how many AOIs and which ones."""
    s = pd.Series({a: stage_of(a, first_round) for a in aois})
    return pd.DataFrame([{"stage": g, "aois": int((s == g).sum()), "ids": " ".join(str(a) for a in sorted(s.index[s == g]))}
                         for g in STAGES])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["order", "imagery", "flow", "autofinish", "resample", "pending", "prepare", "choose",
                                    "finish", "status"])
    p.add_argument("--aois", type=int, nargs="*", default=[])
    p.add_argument("--start", type=int, nargs="*", default=[])
    p.add_argument("--logs", default="logs/aoi_batch")
    p.add_argument("--jobs", type=int, default=0)
    p.add_argument("--window", type=int, default=30, help="flow: AOIs kept in Earth Engine at once")
    p.add_argument("--skip", type=int, nargs="*", default=[], help="flow: AOIs left out (finished earlier)")
    a = p.parse_args(argv)
    logs = Path(a.logs)
    if a.step == "order":
        print(" ".join(map(str, neighbour_order(a.start or [72]))))
    elif a.step == "imagery":
        imagery(a.aois, logs)
    elif a.step == "flow":
        order = a.aois or [x for x in neighbour_order(a.start or [72]) if x not in set(a.skip)]
        flow(order, logs, window=a.window, prep_jobs=a.jobs)
    elif a.step == "pending":
        order = a.aois or [x for x in neighbour_order(a.start or [72]) if x not in set(a.skip)]
        print(pending_review(order).to_json(orient="records"))
    elif a.step == "resample":
        order = a.aois or [x for x in neighbour_order(a.start or [72]) if x not in set(a.skip)]
        print(resample(order))
    elif a.step == "autofinish":
        order = a.aois or [x for x in neighbour_order(a.start or [72]) if x not in set(a.skip)]
        autofinish(order, jobs=a.jobs or 3)
    elif a.step == "prepare":
        prepare(a.aois, logs, a.jobs)
    elif a.step == "choose":
        for aoi in a.aois:
            print(aoi, choose(aoi), flush=True)
    elif a.step == "finish":
        for aoi in a.aois:
            print(aoi, finish(aoi), flush=True)
    elif not a.aois:                                  # no AOIs given: the stage of every AOI, counted
        allaois = neighbour_order([72])
        print(stage_summary(allaois, first_round=a.skip).to_string(index=False))
        print(f"total: {len(allaois)} AOIs")
    else:
        steps = ("imagery", "inputs", "trials", "sheets", "choose", "map", "finish")
        for aoi in a.aois:
            st = status(aoi)
            print(f"aoi{aoi}: " + " ".join(f"{s}={'ok' if st.get(s, {}).get('done') else '-'}" for s in steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
