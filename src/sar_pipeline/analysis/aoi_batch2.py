"""A second set of AOIs, brought into the same pipeline as the first one.

Why (7 Oct 2026): the client shared more AOIs, each with its own field delineation and basemap. They are numbered
from 133 up, which collides with the first set's 133-160, so inside the pipeline they get ``OFFSET`` + their number
(new 133 -> 1133) and every file, config and status follows the first set's layout unchanged; the delivery names
them ``b2_aoi133`` (:func:`delivery_name`).

User decisions (7 Oct 2026):

* exact duplicates of another new AOI, and AOIs without area, are skipped (:data:`SKIP`);
* where a new AOI overlaps an AOI of the first set, that part was delivered already and is cut out of the new AOI
  (and its fields), so no field is mapped twice;
* the same dates as the first set (radar to 4 Oct, optical to 1 Oct, map 5 Oct, delivery ``rice_map_2026-10-05``);
* rules: each new AOI takes the rule set chosen for its nearest first-set AOI (:func:`nearest_rules`); a few new
  AOIs are blind-reviewed to check that this holds.

Inputs (local, never in the repository): the AOI polygons and the ``Final_Output`` delineations downloaded under
:data:`SRC`. Outputs: ``<SRC>/split/b2_<id>/b2_<id>.gpkg`` (the clipped AOIs, the layout ``prep batch-configs``
reads), ``<SRC>/delineation_merged.gpkg`` (all fields with ``source_aoi``, as ``field_rice.load_fields`` reads) and
``<SRC>/aoi_index.csv``.

Run::

    python -m sar_pipeline.analysis.aoi_batch2 build
    python -m sar_pipeline.prep batch-configs --template config/aoi19_monsoon2026.yaml \\
        --split-dir ../data/aoi_batch2_2026-09-29/split --season-key monsoon2026 --start 2026-03-15 --end 2026-10-05
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

OFFSET = 1000
SRC = "../data/aoi_batch2_2026-09-29"
#: 165 = 164, 169 = 168, 185 = 184 (identical polygons); 220 and 222 have no area (220 has no delineation).
SKIP = (165, 169, 185, 220, 222)
ACRE_M2 = 4046.8564224
METRIC = "EPSG:32646"
#: Pieces of a clipped field smaller than this are dropped (m2).
MIN_FIELD_M2 = 10.0


def internal_id(number: int) -> int:
    return OFFSET + int(number)


def is_batch2(aoi: int) -> bool:
    return int(aoi) >= OFFSET


def delivery_name(aoi: int) -> str:
    """The name the client sees: ``aoi19`` for the first set, ``b2_aoi133`` for new AOI 133."""
    return f"b2_aoi{int(aoi) - OFFSET}" if is_batch2(aoi) else f"aoi{int(aoi)}"


def _number(path) -> int:
    return int(re.search(r"(\d+)(?:_Final_Output)?\.gpkg$", str(path)).group(1))


def new_aois(src: str = SRC):
    """The new AOI polygons (EPSG:4326) with ``number`` and ``aoi`` (internal id), skipped ones left out."""
    import geopandas as gpd

    frames = []
    for p in sorted(glob.glob(f"{src}/aoi/*/A*_*.gpkg")):
        n = _number(p)
        if n in SKIP:
            continue
        g = gpd.read_file(p)
        g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
        frames.append(gpd.GeoDataFrame({"number": [n], "aoi": [internal_id(n)]},
                                       geometry=[g.geometry.union_all()], crs=4326))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)


def first_set_union(folder: str = "data/aoi"):
    import geopandas as gpd
    import shapely

    parts = [gpd.read_file(p).to_crs(METRIC).geometry.union_all() for p in sorted(Path(folder).glob("aoi_[0-9]*.gpkg"))
             if int(Path(p).stem.split("_")[1]) < OFFSET]
    return shapely.union_all(parts)


def clip_to_new_ground(aois, old_union):
    """Each new AOI minus the first set's AOIs (user, 7 Oct: the overlap was delivered already)."""
    import shapely

    m = aois.to_crs(METRIC)
    before = m.area / ACRE_M2
    m["geometry"] = [shapely.make_valid(g.difference(old_union)) for g in m.geometry]
    m["acres"] = (m.area / ACRE_M2).round(2)
    m["acres_cut_as_delivered"] = (before - m["acres"]).round(2)
    return m.to_crs(4326)


def write_split(clipped, src: str = SRC) -> list[Path]:
    """One folder per AOI in the layout ``prep batch-configs`` reads (``*_<id>/*_<id>.gpkg``)."""
    out = []
    for r in clipped.itertuples():
        d = Path(src) / "split" / f"b2_{r.aoi}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"b2_{r.aoi}.gpkg"
        clipped.loc[[r.Index], ["aoi", "number", "acres", "geometry"]].assign(
            aoi_name=delivery_name(r.aoi)).to_file(f, driver="GPKG")
        out.append(f)
    return out


def merged_delineation(clipped, src: str = SRC) -> Path:
    """All new fields in one file with ``source_aoi`` = ``b2_<id>_delineation`` (so ``field_rice.load_fields`` finds an
    AOI by its id), 2D, cut to the clipped AOI; pieces below :data:`MIN_FIELD_M2` dropped."""
    import geopandas as gpd
    import shapely

    out = Path(src) / "delineation_merged.gpkg"
    frames = []
    shapes = clipped.to_crs(METRIC).set_index("number").geometry
    for p in sorted(glob.glob(f"{src}/delineation/*/*/*/Final_Output/*_Final_Output.gpkg")):
        n = _number(p)
        if n in SKIP or n not in shapes.index:
            continue
        f = gpd.read_file(p)
        f["geometry"] = shapely.force_2d(f.geometry.values)
        f = f.to_crs(METRIC)
        f["geometry"] = [shapely.make_valid(g.intersection(shapes[n])) for g in f.geometry]
        f = f[~f.is_empty]
        f = f.explode(index_parts=False)
        f = f[f.geom_type == "Polygon"]
        f = f[f.area >= MIN_FIELD_M2]
        f.insert(0, "source_aoi", f"b2_{internal_id(n)}_delineation")
        frames.append(f.drop(columns=["src_uids"], errors="ignore").to_crs(4326))
    allf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    tmp = out.with_suffix(".tmp.gpkg")
    allf.to_file(tmp, driver="GPKG")
    tmp.replace(out)
    return out


def build(src: str = SRC, old_folder: str = "data/aoi") -> pd.DataFrame:
    """Split AOIs, merged delineation and the index table (one row per new AOI)."""
    import pyogrio

    clipped = clip_to_new_ground(new_aois(src), first_set_union(old_folder))
    write_split(clipped, src)
    m = merged_delineation(clipped, src)
    counts = pyogrio.read_dataframe(m, columns=["source_aoi"], read_geometry=False)["source_aoi"].value_counts()
    t = pd.DataFrame({"number": clipped["number"], "aoi": clipped["aoi"],
                      "delivery_name": [delivery_name(a) for a in clipped["aoi"]],
                      "acres": clipped["acres"], "acres_cut_as_delivered": clipped["acres_cut_as_delivered"],
                      "fields": [int(counts.get(f"b2_{a}_delineation", 0)) for a in clipped["aoi"]],
                      "lon": clipped.geometry.representative_point().x.round(4)})
    t.to_csv(Path(src) / "aoi_index.csv", index=False)
    return t


def nearest_rules(new_index: pd.DataFrame, chosen: dict, folder: str = "data/aoi") -> pd.DataFrame:
    """For each new AOI the first-set AOI with a chosen rule set whose centroid is nearest, and its distance (km)."""
    import geopandas as gpd

    pts = {}
    for p in sorted(Path(folder).glob("aoi_[0-9]*.gpkg")):
        a = int(p.stem.split("_")[1])
        if a < OFFSET and str(a) in chosen:
            pts[a] = gpd.read_file(p).to_crs(METRIC).geometry.union_all().centroid
    old = np.array([[g.x, g.y] for g in pts.values()])
    ids = np.array(list(pts))
    rows = []
    for r in new_index.itertuples():
        g = gpd.read_file(next(Path(SRC, "split", f"b2_{r.aoi}").glob("*.gpkg"))).to_crs(METRIC).geometry.union_all()
        c = g.centroid
        d = np.hypot(old[:, 0] - c.x, old[:, 1] - c.y)
        k = int(np.argmin(d))
        rows.append({"aoi": r.aoi, "nearest": int(ids[k]), "km": round(float(d[k]) / 1000, 1),
                     "rule": chosen[str(ids[k])]["from"]})
    return pd.DataFrame(rows)


#: Rules (user, 7 Oct 2026): an AOI within NEAR_KM of a first-set AOI takes that AOI's chosen rule set; the farther
#: AOIs are checked by a blind review of these 8 (spread over the far ground, at least 200 ac each, picked by
#: farthest-point sampling) and take the rule set chosen for the nearest of them once it is chosen.
NEAR_KM = 20.0
REVIEW = (1142, 1143, 1166, 1173, 1174, 1194, 1199, 1218)


def _centroid(aoi: int):
    import geopandas as gpd

    f = next(Path(SRC, "split", f"b2_{aoi}").glob("*.gpkg"))
    return gpd.read_file(f).to_crs(METRIC).geometry.union_all().centroid


def rule_source(aoi: int, chosen: dict) -> tuple[int | None, str]:
    """(AOI whose chosen rule set this AOI takes, why). ``None`` while a far AOI's reviewed neighbour has no choice
    yet. A reviewed AOI chooses its own (``aoi_batch.choose``), so it gets ``(aoi, 'reviewed')``."""
    if aoi in REVIEW:
        return aoi, "blind-reviewed here"
    t = pd.read_csv(Path(SRC) / "nearest_rules.csv").set_index("aoi")
    if float(t.loc[aoi, "km"]) <= NEAR_KM:
        n = int(t.loc[aoi, "nearest"])
        return n, f"nearest first-set AOI aoi{n}, {t.loc[aoi, 'km']} km"
    c = _centroid(aoi)
    d = {r: c.distance(_centroid(r)) / 1000 for r in REVIEW}
    r = min(d, key=d.get)
    if str(r) not in chosen:
        return None, f"waits for the review of {delivery_name(r)} ({d[r]:.1f} km)"
    return r, f"nearest reviewed new AOI {delivery_name(r)}, {d[r]:.1f} km"


def choose_by_neighbour(aoi: int) -> dict | None:
    """Writes the AOI's rule set (its source's chosen set) to ``aoi_rules_chosen.json`` and the reason to its rule
    record; returns None while it has to wait."""
    import json

    from . import aoi_batch as ab
    from . import curve_rules as cr
    from . import rule_records as rr

    chosen = json.loads(ab.CHOSEN.read_text()) if ab.CHOSEN.exists() else {}
    src, why = rule_source(aoi, chosen)
    if src is None or src == aoi:
        return None
    rules = dict(chosen[str(src)]["rules"])
    chosen[str(aoi)] = {"rules": rules, "from": f"{delivery_name(src)}: {chosen[str(src)]['from']}", "fields": 0,
                        "right_pct": None, "merged_pct": None, "at": str(pd.Timestamp.now().floor("s")),
                        "by": "neighbour (no review)"}
    ab.CHOSEN.write_text(json.dumps(chosen, indent=1, sort_keys=True, default=str))
    cr.AOI_OVERRIDES[aoi] = {k: v for k, v in rules.items() if k != "SIEVE_ACRES"}
    rr.record(aoi, "rule set by neighbour", f"{why}: {chosen[str(src)]['from']} (not reviewed here; user, 7 Oct 2026)")
    ab._mark(aoi, "choose", chosen[str(aoi)])
    return chosen[str(aoi)]


def finish_ready(aois, jobs: int = 3) -> list[int]:
    """Every second-set AOI with its inputs done, not reviewed, not finished, and whose rule source is known: rule set
    chosen by neighbour, then map, fields, lock, delivery and upload (``aoi_batch.finish``)."""
    import json
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    from . import aoi_batch as ab

    chosen = json.loads(ab.CHOSEN.read_text()) if ab.CHOSEN.exists() else {}
    todo = [a for a in aois if a not in REVIEW and ab._done(a, "inputs") and not ab._done(a, "finish")
            and rule_source(a, chosen)[0] is not None]
    if not todo:
        return []
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
        list(ex.map(_finish_one, todo))
    return todo


def _finish_one(aoi: int) -> str:
    from . import aoi_batch as ab

    say = lambda m: print(f"{pd.Timestamp.now():%H:%M:%S} {m}", flush=True)
    try:
        if not ab._done(aoi, "choose"):
            choose_by_neighbour(aoi)
        r = ab.finish(aoi)
        say(f"DELIVERED aoi{aoi} ({delivery_name(aoi)}) {r.get('gcs')}  rule: {ab.status(aoi)['choose']['result']['from']}")
    except Exception as e:
        say(f"aoi{aoi}: neighbour finish FAILED: {type(e).__name__}: {e}")
    return str(aoi)


def finish_loop(aois, poll: float = 60.0, jobs: int = 3) -> None:
    """:func:`finish_ready` every ``poll`` seconds until every non-reviewed AOI is finished."""
    import time

    from . import aoi_batch as ab

    rest = [a for a in aois if a not in REVIEW]
    while any(not ab._done(a, "finish") for a in aois):
        # an AOI whose radar came in while its Sentinel-2 export was still running was skipped by the flow's
        # preparation; it is prepared here once both are in (reviewed AOIs go on to trials and sheets)
        waiting = [a for a in aois if ab._done(a, "imagery") and s2_ready(a) and not ab._done(a, "inputs")
                   or a in REVIEW and ab._done(a, "imagery") and s2_ready(a) and not ab._done(a, "sheets")]
        if waiting:
            ab.prepare(waiting, Path("logs/batch2"), jobs=jobs)
        finish_ready(rest, jobs)
        time.sleep(poll)


#: The Sentinel-2 per-date export of the second set: from the start of the dry season the rules read
#: (``curve_rules.DRY_SEASON_FROM``, 1 March 2026; the user asked for April, March is the rules' own floor) to 1 Oct 2026
#: (end exclusive). The 5-day series calendar still starts in September 2025; its windows before March stay empty and
#: no rule reads them.
S2_START, S2_END = "2026-03-01", "2026-10-02"
S2_CHUNK = 15          # AOIs per export round (about 150 dates each)
S2_QUEUE_MAX = 500     # the next round waits until fewer Earth Engine tasks than this are queued or running


def s2_marker(aoi: int, src: str = SRC) -> Path:
    return Path(src) / "s2_ready" / f"aoi{aoi}"


def s2_ready(aoi: int, src: str = SRC) -> bool:
    """True once the AOI's full Sentinel-2 export is on GCS. Why: a series built from only the newest date (what the
    first set's flow exports) was nonsense, and the artefact-pass check built on it was wrong too (aoi1166, 7 Oct)."""
    return s2_marker(aoi, src).exists()


def _ee_open_tasks() -> int:
    import ee

    return sum(1 for t in ee.data.getTaskList() if t.get("state") in ("READY", "RUNNING"))


def s2_export(aois, logs: Path = Path("logs/batch2"), chunk: int = S2_CHUNK, queue_max: int = S2_QUEUE_MAX,
              poll: float = 60.0) -> None:
    """Exports every Sentinel-2 date S2_START..S2_END for ``aois`` in rounds of ``chunk`` AOIs, each round only when
    Earth Engine holds fewer than ``queue_max`` open tasks (the project's queue takes about 3,000, and the radar
    exports share it). When everything is done a second pass resubmits whatever failed; then each AOI is marked ready
    (:func:`s2_marker`). Files already on GCS are skipped (``optical_export.export_all_dates``)."""
    import subprocess
    import time

    from .. import auth
    from .. import config as cm

    py = [".venv/bin/python", "-W", "ignore", "-m", "sar_pipeline.optical_export", "dates", "--no-wait"]
    say = lambda m: print(f"{pd.Timestamp.now():%H:%M:%S} {m}", flush=True)
    auth.init_ee(cm.load_config(f"config/aoi{aois[0]}_monsoon2026.yaml"))
    logs.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        todo = [a for a in aois if not s2_ready(a)]
        for i in range(0, len(todo), chunk):
            while _ee_open_tasks() >= queue_max:
                time.sleep(poll)
            part = todo[i:i + chunk]
            cfgs = [f"config/aoi{a}_monsoon2026.yaml" for a in part]
            with open(logs / "s2_full_export.log", "a") as log:
                r = subprocess.run(py + ["--configs", *cfgs, "--start", S2_START, "--end", S2_END], stdout=log,
                                   stderr=subprocess.STDOUT)
            say(f"S2 round {attempt}: submitted {part} (exit {r.returncode}); open tasks {_ee_open_tasks()}")
        while _ee_open_tasks() > 0:
            time.sleep(poll)
        say(f"S2 round {attempt}: Earth Engine queue empty")
    for a in aois:                                    # a third listing: nothing left to export means ready
        with open(logs / "s2_full_export.log", "a") as log:
            r = subprocess.run(py + ["--configs", f"config/aoi{a}_monsoon2026.yaml", "--start", S2_START, "--end",
                                     S2_END], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log.write(r.stdout)
        if r.returncode == 0 and ", 0 to export" in r.stdout:
            s2_marker(a).parent.mkdir(parents=True, exist_ok=True)
            s2_marker(a).write_text(str(pd.Timestamp.now()))
            say(f"S2 READY aoi{a}")
        else:
            say(f"aoi{a}: S2 still incomplete after two rounds; see {logs / 's2_full_export.log'}")


def all_ids(src: str = SRC) -> list[int]:
    return [int(a) for a in pd.read_csv(Path(src) / "aoi_index.csv")["aoi"]]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["build", "finish-loop", "s2-export"])
    a = p.parse_args(argv)
    if a.step == "s2-export":
        s2_export(all_ids())
        return 0
    if a.step == "finish-loop":
        finish_loop(all_ids())
        return 0
    if a.step == "build":
        t = build()
        print(t.to_string(index=False))
        print(f"{len(t)} AOIs, {t['acres'].sum():,.1f} ac to map ({t['acres_cut_as_delivered'].sum():,.1f} ac cut as "
              f"delivered already), {t['fields'].sum():,} fields")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
