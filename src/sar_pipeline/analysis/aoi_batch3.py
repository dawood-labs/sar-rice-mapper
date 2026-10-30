"""Third AOI set (27 Oct 2026): the client's late AOIs, one file per polygon, ready for the field delineation run.

Why: after the delivery the client sent 7 more AOI polygons (one zip) and a pilot site (another zip). The delineation
team runs on one GeoPackage per AOI in its own folder (``AOI_<N>/AOI_<N>.gpkg``), as for the second set, so these
continue the second set's numbering (:data:`FIRST_NUMBER` onward), the client's main zip first (files by name, polygons
in file order), the pilot site last. Each file keeps the client's columns as given and adds ``layer`` (source file name), ``path`` (zip
folder / file) and ``AOI_NUMBER``, like the second set's files. Delivered names will be ``b3_aoi<N>``.

Inputs: the zips unpacked under :data:`CLIENT_DIR` (``<zip name>/<folder>/<file>.shp``; a zip found again inside
another zip is read once) and the gitignored :data:`CONFIG` (``{"zips": [...in numbering order], "s3_prefix": ...}``: both
name the client, so they stay out of the repository). Outputs: ``<OUT_DIR>/AOI_<N>/AOI_<N>.gpkg`` + ``index.csv``;
``upload`` copies them to the S3 prefix with the AWS CLI (credentials from the environment) and checks each object's size.

Processing (user, 27 Oct 2026: start while the delineation runs), the second set's recipe
(``scripts/monsoon_batch_chain.sh``, ``aoi_batch2``) with this set's dates: internal ids ``OFFSET`` + number (1239...),
season 15 March to today, Sentinel-2 per-date export from 1 March (the rules' dry-season floor) to today, the newest
images as the map date (user, 29 Sep: next runs use the latest date).

* ``split``: ``<SRC>/split/b3_<id>/b3_<id>.gpkg`` (the layout ``prep batch-configs`` reads);
* ``setup``: configs from a second-set template, grid + audit per AOI, tracks by rule (``prep batch-tracks``);
* ``radar``: new run + export per AOI, then monitor, download and stack (``aoi_batch._download_stack``);
* ``s2``: the full per-date Sentinel-2 export (``aoi_batch2.s2_export`` with this set's dates);
* ``inputs``: pin the run, build the series and the per-AOI steps (``aoi_batch._prepare_one``, up to its inputs);
* ``map``: each AOI's rule set from its neighbour (:data:`RULE_SOURCE`), then the pixel map and sieve;
* ``delineation``: the delineation team's fields from S3, merged for ``field_rice.load_fields``;
* ``radar-to-map-date``: a new radar run ending at the delivered sets' map date (then ``inputs`` and ``map`` again);
* ``fields``: the delineated fields labelled from the sieved map;
* ``deliver``: the QC package per AOI, uploaded like the delivered sets'.

Run::

    python -m sar_pipeline.analysis.aoi_batch3 prepare
    source secrets/aws_env.sh && python -m sar_pipeline.analysis.aoi_batch3 upload
    python -m sar_pipeline.analysis.aoi_batch3 split
    python -m sar_pipeline.analysis.aoi_batch3 setup
    python -m sar_pipeline.analysis.aoi_batch3 radar      # and, at the same time:  ... s2
    python -m sar_pipeline.analysis.aoi_batch3 inputs
    python -m sar_pipeline.analysis.aoi_batch3 map
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

CLIENT_DIR = Path("../data/client_aoi_files_2026-10-26/unzipped")
OUT_DIR = Path("../data/aoi_batch3_2026-10-27/split")
FIRST_NUMBER = 239
CONFIG = Path("config/aoi_batch3.json")                        # gitignored: zip names and bucket path name the client


def _config() -> dict:
    return json.loads(CONFIG.read_text())


def shapefiles(root: Path = CLIENT_DIR, zips=None) -> list[Path]:
    """The client shapefiles in numbering order (``zips`` order, files by name), each (folder, file) once."""
    seen, out = set(), []
    for z in zips or _config()["zips"]:
        for p in sorted((root / z).glob("*/*.shp"), key=lambda p: p.name.lower()):
            if (p.parent.name, p.name) not in seen:
                seen.add((p.parent.name, p.name))
                out.append(p)
    return out


def prepare(root: Path = CLIENT_DIR, out: Path = OUT_DIR, first: int = FIRST_NUMBER, zips=None) -> pd.DataFrame:
    """One GeoPackage per polygon (``AOI_<N>/AOI_<N>.gpkg``, layer ``AOI_<N>``) and ``index.csv``."""
    import geopandas as gpd

    rows, n = [], first
    for p in shapefiles(root, zips):
        g = gpd.read_file(p)
        for i in range(len(g)):
            one = g.iloc[[i]].copy()
            one["layer"] = p.stem
            one["path"] = f"{p.parent.name}/{p.name}"
            one["AOI_NUMBER"] = n
            d = out / f"AOI_{n}"
            d.mkdir(parents=True, exist_ok=True)
            f = d / f"AOI_{n}.gpkg"
            one.to_file(f, driver="GPKG", layer=f"AOI_{n}")
            rows.append({"AOI_NUMBER": n, "delivered_name": f"b3_aoi{n}", "source_file": one["path"].iloc[0],
                         "row_in_file": i, "company": one.get("Company", pd.Series([None])).iloc[0],
                         "acres": round(float(one.to_crs(32646).area.iloc[0]) / 4046.8564224, 3)})
            n += 1
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "index.csv", index=False)
    return idx


def boundaries(src: Path = OUT_DIR) -> Path:
    """All the set's AOI polygons in one GeoPackage (``<SRC>/b3_aois_boundaries.gpkg``): ``aoi_id`` (``b3_aoi<N>``),
    ``area_ac`` (acres, 3 decimals, EPSG:32646) and the client's columns as given, EPSG:4326. Why (user, 30 Oct 2026):
    one file to download and open in QGIS."""
    import geopandas as gpd

    parts = []
    for n in pd.read_csv(src / "index.csv")["AOI_NUMBER"]:
        g = gpd.read_file(src / f"AOI_{n}" / f"AOI_{n}.gpkg").to_crs(4326)
        g.insert(0, "aoi_id", f"b3_aoi{n}")
        g.insert(1, "area_ac", (g.to_crs(32646).area / 4046.8564224).round(3))
        parts.append(g)
    out = SRC / "b3_aois_boundaries.gpkg"
    gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326).to_file(out, driver="GPKG", layer="b3_aois")
    return out


def upload(out: Path = OUT_DIR, prefix: str | None = None) -> dict:
    """``aws s3 cp`` of every prepared file to the prefix, then a size check of each object."""
    prefix = (prefix or _config()["s3_prefix"]).rstrip("/")
    files = sorted(p for p in out.rglob("*") if p.is_file())
    subprocess.run(["aws", "s3", "cp", "--recursive", "--only-show-errors", str(out), prefix + "/"], check=True)
    ls = subprocess.run(["aws", "s3", "ls", "--recursive", prefix + "/"], check=True, capture_output=True, text=True)
    remote = {}
    for line in ls.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4:
            remote[parts[3].split(prefix.split("/", 3)[-1] + "/", 1)[-1]] = int(parts[2])
    ok = [remote.get(str(p.relative_to(out))) == p.stat().st_size for p in files]
    return {"files": len(files), "size_ok": int(sum(ok)), "s3": prefix + "/"}


SRC = Path("../data/aoi_batch3_2026-10-27")
TEMPLATE = "config/aoi1133_monsoon2026.yaml"
SEASON = "monsoon2026"
SEASON_START = "2026-03-15"
S2_START = "2026-03-01"
LOGS = Path("logs/batch3")
PY = [".venv/bin/python", "-W", "ignore", "-m"]


def season_end() -> str:
    """Tomorrow (the season end is exclusive): every image up to today."""
    return (pd.Timestamp.now().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def ids(src: Path = OUT_DIR) -> list[int]:
    from .aoi_batch2 import internal_id

    return [internal_id(n) for n in pd.read_csv(src / "index.csv")["AOI_NUMBER"]]


def split(src: Path = OUT_DIR, dest: Path = SRC) -> list[Path]:
    """One folder per AOI (``split/b3_<id>/b3_<id>.gpkg``: ``aoi``, ``number``, ``acres``, ``aoi_name``)."""
    import geopandas as gpd

    from .aoi_batch2 import delivery_name, internal_id

    out = []
    for n in pd.read_csv(src / "index.csv")["AOI_NUMBER"]:
        g = gpd.read_file(src / f"AOI_{n}" / f"AOI_{n}.gpkg").to_crs(4326)
        a = internal_id(int(n))
        one = gpd.GeoDataFrame({"aoi": [a], "number": [int(n)],
                                "acres": [round(float(g.to_crs(32646).area.sum()) / 4046.8564224, 3)],
                                "aoi_name": [delivery_name(a)]}, geometry=[g.geometry.union_all()], crs=4326)
        d = dest / "split" / f"b3_{a}"
        d.mkdir(parents=True, exist_ok=True)
        one.to_file(d / f"b3_{a}.gpkg", driver="GPKG")
        out.append(d / f"b3_{a}.gpkg")
    return out


def _sh(args: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as fh:
        r = subprocess.run(args, stdout=fh, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{' '.join(args[3:6])} failed (exit {r.returncode}); see {log}")


def setup(aois=None) -> None:
    """Configs, grid, audit and track choice for every AOI (the second set's recipe)."""
    from concurrent.futures import ThreadPoolExecutor

    aois = aois or ids()
    _sh(PY + ["sar_pipeline.prep", "batch-configs", "--template", TEMPLATE, "--split-dir", str(SRC / "split"),
              "--season-key", SEASON, "--start", SEASON_START, "--end", season_end(), "--ids", *map(str, aois)],
        LOGS / "configs.log")

    def one(a):
        cfg = f"config/aoi{a}_{SEASON}.yaml"
        _sh(PY + ["sar_pipeline", "--config", cfg, "grid"], LOGS / f"aoi{a}_prep.log")
        _sh(PY + ["sar_pipeline", "--config", cfg, "audit"], LOGS / f"aoi{a}_prep.log")
        return a

    with ThreadPoolExecutor(len(aois)) as ex:
        list(ex.map(one, aois))
    _sh(PY + ["sar_pipeline.prep", "batch-tracks", "--season-key", SEASON, "--flood-start", "2026-05-01",
              "--flood-end", "2026-08-31", "--ids", *map(str, aois)], LOGS / "tracks.log")


def radar(aois=None) -> dict:
    """New run + export, then monitor until done, download and stack, every AOI in its own thread."""
    import re
    from concurrent.futures import ThreadPoolExecutor

    from . import aoi_batch as ab

    aois = aois or ids()

    def one(a):
        cfg = f"config/aoi{a}_{SEASON}.yaml"
        st = ab.status(a)
        run = st.get("run")
        if not run:
            log = LOGS / f"aoi{a}_newrun.log"
            _sh(PY + ["sar_pipeline", "--config", cfg, "new-run"], log)
            run = re.findall(r"/runs/(\S+)", log.read_text())[-1].rstrip("/").split("/")[0]
            st["run"] = run
            ab._status_path(a).parent.mkdir(parents=True, exist_ok=True)
            ab._status_path(a).write_text(json.dumps(st, indent=1, default=str))
        if not ab._done(a, "imagery"):
            _sh(PY + ["sar_pipeline", "--config", cfg, "export", "--run", run, "--all", "--yes"],
                LOGS / f"aoi{a}_export.log")
            _sh(PY + ["sar_pipeline", "--config", cfg, "monitor", "--run", run, "--yes"], LOGS / f"aoi{a}_monitor.log")
            ab._download_stack(a, run, LOGS)
            ab._mark(a, "imagery", {"run": run})
        return a, run

    with ThreadPoolExecutor(len(aois)) as ex:
        return dict(ex.map(one, aois))


#: Map date of the delivered sets (user, 27 Oct 2026: map the third set like the others, ~5 Oct): radar to 4 Oct
#: (season end 5 Oct, exclusive); the optical series (``aoi_batch.SERIES_VARIANT``) already ends on the 1 Oct images.
MAP_SEASON_END = "2026-10-05"


def radar_to_map_date(aois=None, end: str = MAP_SEASON_END) -> dict:
    """A new radar run per AOI ending at the map date, then the AOI's inputs again on it (pin, pass audit, steps).
    Why a new run and not a cut of the old one: the multi-temporal speckle filter mixes each pass with its
    neighbours, so only a run that ends there equals the other sets' runs."""
    import re as _re

    from . import aoi_batch as ab

    aois = list(aois or ids())
    for a in aois:
        p = Path(f"config/aoi{a}_{SEASON}.yaml")
        p.write_text(_re.sub(r"(season: \{[^}]*end: ')[0-9-]+(')", rf"\g<1>{end}\g<2>", p.read_text(), count=1))
        # the export takes its passes from the latest audit, so the audit is redone for the new season end first
        # (27 Oct: without it the new run still held the passes to 19 Oct)
        _sh(PY + ["sar_pipeline", "--config", str(p), "audit"], LOGS / f"aoi{a}_prep.log")
        st = ab.status(a)
        old = (st.get("imagery") or {}).get("result", {}).get("run")
        if old and (st.get("run_to_map_date") or {}).get("audited") != end:
            for k in ("run", "imagery", "inputs", "choose"):
                st.pop(k, None)
            st["run_to_map_date"] = {"from": old, "end": end, "audited": end}
            ab._status_path(a).write_text(json.dumps(st, indent=1, default=str))
    return radar(aois)


def s2(aois=None) -> None:
    from . import aoi_batch2 as b2

    b2.s2_export(aois or ids(), logs=LOGS, start=S2_START, end=season_end())


def inputs(aois=None) -> list[str]:
    """Pin, series and the per-AOI steps for every AOI whose radar and Sentinel-2 are both in."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    from . import aoi_batch as ab
    from . import aoi_batch2 as b2

    todo = [a for a in (aois or ids()) if ab._done(a, "imagery") and b2.s2_ready(a)]
    with ProcessPoolExecutor(min(len(todo), 4) or 1, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(ab._prepare_one, todo, [str(LOGS)] * len(todo)))


MIN_FIELD_M2 = 10.0


def delineation(aois=None) -> Path:
    """Downloads each AOI's ``Final_Output`` delineation from the S3 prefix (newest run) and writes all fields to
    ``<SRC>/delineation_merged.gpkg`` with ``source_aoi`` = ``b3_<id>_delineation`` (how ``field_rice.load_fields``
    finds an AOI), 2D, cut to the AOI, pieces under :data:`MIN_FIELD_M2` dropped (the second set's recipe)."""
    import geopandas as gpd
    import shapely

    from .aoi_batch2 import internal_id

    prefix = _config()["s3_prefix"].rstrip("/")
    ls = subprocess.run(["aws", "s3", "ls", "--recursive", prefix + "/"], check=True, capture_output=True, text=True)
    keys = sorted(line.split(maxsplit=3)[3] for line in ls.stdout.splitlines()
                  if line.endswith("_Final_Output.gpkg"))
    bucket = prefix.split("/")[2]
    frames = []
    for n in pd.read_csv(OUT_DIR / "index.csv")["AOI_NUMBER"]:
        a = internal_id(int(n))
        mine = [k for k in keys if f"/AOI_{n}/" in k]
        if not mine or (aois and a not in aois):
            continue
        local = SRC / "delineation" / f"AOI_{n}_Final_Output.gpkg"
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "cp", "--only-show-errors", f"s3://{bucket}/{mine[-1]}", str(local)], check=True)
        shape = gpd.read_file(SRC / "split" / f"b3_{a}" / f"b3_{a}.gpkg").to_crs(32646).geometry.union_all()
        f = gpd.read_file(local)
        f["geometry"] = shapely.force_2d(f.geometry.values)
        f = f.to_crs(32646)
        f["geometry"] = [shapely.make_valid(g.intersection(shape)) for g in f.geometry]
        f = f[~f.is_empty].explode(index_parts=False)
        f = f[(f.geom_type == "Polygon") & (f.area >= MIN_FIELD_M2)]
        f.insert(0, "source_aoi", f"b3_{a}_delineation")
        frames.append(f.drop(columns=["src_uids"], errors="ignore").to_crs(4326))
    out = SRC / "delineation_merged.gpkg"
    allf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)
    tmp = out.with_suffix(".tmp.gpkg")
    allf.to_file(tmp, driver="GPKG")
    tmp.replace(out)
    return out


#: Whose accepted rule set each AOI takes (user, 27 Oct 2026, the second set's rule: the nearest AOI within 20 km):
#: the two near Shwe Se fields -> aoi61; Farmer x3 and Wakhema -> aoi145 (their nearest second-set AOIs had no rice to
#: learn from); the pilot site -> aoi28 (its locked own rules). AOI 241 is 31 km from any mapped AOI: blind review
#: first, so it only gets a provisional map with the default rules (:data:`REVIEW`).
RULE_SOURCE = {1239: 145, 1240: 145, 1242: 145, 1243: 61, 1244: 61, 1245: 145, 1246: 28}
REVIEW = (1241,)


def choose(aoi: int) -> dict | None:
    """Writes the AOI's rule set (its source's) to ``aoi_rules_chosen.json`` and the rule record."""
    from . import aoi_batch as ab
    from . import curve_rules as cr
    from . import rule_records as rr
    from .aoi_batch2 import delivery_name

    src = RULE_SOURCE.get(aoi)
    if src is None:
        return None
    chosen = json.loads(ab.CHOSEN.read_text()) if ab.CHOSEN.exists() else {}
    if str(src) in chosen:
        rules, origin = dict(chosen[str(src)]["rules"]), chosen[str(src)]["from"]
    else:                                              # a locked AOI with its own overrides (aoi28)
        rules, origin = dict(cr.AOI_OVERRIDES[src]), f"aoi{src} own (locked)"
    entry = {"rules": rules, "from": f"{delivery_name(src)}: {origin}", "fields": 0, "right_pct": None,
             "merged_pct": None, "at": str(pd.Timestamp.now().floor("s")), "by": "neighbour (no review)"}
    ab.set_chosen(aoi, entry)
    cr.AOI_OVERRIDES[aoi] = {k: v for k, v in rules.items() if k != "SIEVE_ACRES"}
    rr.record(aoi, "rule set by neighbour", f"{entry['from']} (nearest mapped AOI; user, 27 Oct 2026)")
    ab._mark(aoi, "choose", entry)
    return entry


def _map_one(aoi: int) -> dict:
    from . import curve_rules as cr

    e = choose(aoi)
    t = cr.run(aoi, force=True)
    cr.sieve(aoi, force=True)
    return {"aoi": aoi, "rules_from": e["from"] if e else "defaults (provisional, awaits review)",
            "classes": t.to_dict("records") if hasattr(t, "to_dict") else str(t)}


def _fields_one(aoi: int) -> dict:
    from . import curve_rules as cr

    t = cr.fields(aoi, force=True)
    return {"aoi": aoi, "acres": t.to_dict("records") if hasattr(t, "to_dict") else str(t)}


def fields(aois=None) -> list[dict]:
    """Field labels from the sieved map on the delineation (``curve_rules.fields``), every AOI in parallel."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    todo = list(aois or ids())
    with ProcessPoolExecutor(min(len(todo), 8), mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_fields_one, todo))


def _deliver_one(aoi: int, redo=()) -> tuple[int, dict | str]:
    from . import rice_map_delivery as rd

    st0 = rd.load_status(aoi)
    if not st0.get("trees_cut", {}).get("done") and st0.get("upload", {}).get("done"):
        redo = tuple(redo) + ("trees_cut", "upload")   # trees cut later (e.g. after S3 creds came back): upload again
    try:
        st = rd.run_one(aoi, redo=redo)
        return aoi, {k: (v.get("done"), str(v.get("result"))[:200]) for k, v in st.items()}
    except Exception as e:                            # one AOI's failure does not stop the others
        return aoi, f"FAILED {type(e).__name__}: {e}"


def deliver(aois=None) -> dict:
    """The QC package of every AOI, as for the delivered sets (``rice_map_delivery.run_one``: lock, package, too-young,
    trees cut, upload to ``<base>/rice_map_2026-10-05/b3_aoi<N>/`` with its clear S2 dates in ``s2/``). Why (user,
    28 Oct 2026): the QC assistant takes the files from the same place as before."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    todo = list(aois or ids())
    with ProcessPoolExecutor(min(len(todo), 8), mp_context=mp.get_context("spawn")) as ex:
        return dict(ex.map(_deliver_one, todo))


def pixel_map(aois=None) -> list[dict]:
    """Rule set by neighbour, then the pixel map and its sieve (fields wait for the delineation)."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    todo = list(aois or ids())
    with ProcessPoolExecutor(min(len(todo), 8), mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(_map_one, todo))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["prepare", "upload", "split", "setup", "radar", "s2", "inputs", "map",
                                      "delineation", "radar-to-map-date", "fields", "deliver", "boundaries"])
    a = p.parse_args(argv)
    if a.step == "prepare":
        print(prepare().to_string(index=False))
    elif a.step == "upload":
        print(json.dumps(upload(), indent=1))
    elif a.step == "split":
        print("\n".join(map(str, split())))
    elif a.step == "setup":
        setup()
    elif a.step == "radar":
        print(json.dumps(radar(), indent=1))
    elif a.step == "s2":
        s2()
    elif a.step == "radar-to-map-date":
        print(json.dumps(radar_to_map_date(), indent=1))
    elif a.step == "boundaries":
        print(boundaries())
    elif a.step == "deliver":
        print(json.dumps(deliver(), indent=1, default=str))
    elif a.step == "fields":
        for r in fields():
            print(json.dumps(r, default=str))
    elif a.step == "delineation":
        print(delineation())
    elif a.step == "map":
        for r in pixel_map():
            print(json.dumps(r, default=str))
    else:
        print("\n".join(inputs()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
