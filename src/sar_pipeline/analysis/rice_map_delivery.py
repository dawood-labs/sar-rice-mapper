"""Rice map delivery, one folder per AOI: index raster, class rasters, fields with rice / non-rice, clear S2 dates.

Why
---
User (5 Oct 2026): every finished AOI goes to the client as one folder - the pixel-index raster (to look a pixel up
in notebook 08), the raw and sieved class rasters, the field layer with a field id (to open a field in notebook 08),
two major classes (rice / non-rice) with the detailed class as an attribute, and the Sentinel-2 images of the dates
that are clear over the AOI - and the folder is mirrored to the project bucket, AOI by AOI. The work must resume
where it stopped and never repeat a finished step.

Steps per AOI (``status.json`` in the AOI's delivery folder records each one; a finished step is skipped):

1. ``fields``   - an AOI that is not locked yet gets its sieve and field layer if they are missing.
2. ``lock``     - the AOI is frozen with ``curve_rules.lock`` (sliver ``SLIVER_ACRES``) unless already locked.
3. ``package``  - the delivery folder is written from the LOCKED copy (so it is exactly what was accepted).
4. ``upload``   - the folder goes to ``<bucket>/<base folder>/<delivery name>/aoi<N>/``; the S2 images are copied
                  inside the bucket from ``s2_dates_masks/`` (no re-upload); a blob of the same size is skipped.

Use::

    python -m sar_pipeline.analysis.rice_map_delivery run --aois 160 28 118     # all steps, several AOIs at once
    python -m sar_pipeline.analysis.rice_map_delivery status --aois 160 28

The bucket and base folder come from the AOI's config (``config/aoi<N>_monsoon2026.yaml``, not tracked).
"""
from __future__ import annotations

import multiprocessing as mp

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

FRESH = "processed/_batch/s2_2026/rice_fresh"
#: Delivery name (folder locally and in the bucket): the map date of this round.
DELIVERY = "rice_map_2026-10-05"
OUT_ROOT = "processed/_batch/s2_2026"
SLIVER_ACRES = 0.15
#: Map classes counted as rice (user, 5 Oct: agreed; harvested is non-rice): standing direct seeded,
#: standing transplanted and young rice are rice; harvested, flooded / bare, other vegetation, tree / orchard are not).
RICE_CODES = (1, 7, 3)
#: A Sentinel-2 date is delivered when at least this share of the AOI's pixels is clear that day (the series' clear
#: flag, the same Cloud Score+ test the notebook-08 chips use).
CLEAR_SHARE_MIN = 0.8      # user, 5 Oct: "kam az kam 80 percent saaf"
#: ...and only dates of the season (the same start as the notebook-08 chips and the configs' season.start).
SEASON_FROM = "2026-03-15"
STEPS = ("fields", "lock", "package", "upload")


def out_dir(aoi: int, root: str = OUT_ROOT, name: str = DELIVERY) -> Path:
    return Path(root) / name / f"aoi{aoi}"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_status(aoi: int, **kw) -> dict:
    p = out_dir(aoi, **kw) / "status.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_status(aoi: int, st: dict, **kw) -> None:
    d = out_dir(aoi, **kw)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "status.json.tmp"
    tmp.write_text(json.dumps(st, indent=1, default=str))
    tmp.replace(d / "status.json")                 # atomic: a crash never leaves half a status file


def _locked(aoi: int, fresh: str = FRESH) -> Path:
    return Path(fresh) / "locked" / f"aoi{aoi}"


def step_fields(aoi: int, fresh: str = FRESH) -> str:
    """Sieve and fields for an AOI that has only its raw map (skipped when locked or already there)."""
    from . import curve_rules as cr

    if (_locked(aoi, fresh) / "MANIFEST.json").exists():
        return "locked already"
    src = Path(fresh) / f"aoi{aoi}"
    tag = f"{int(round(SLIVER_ACRES * 100)):03d}"
    did = []
    if not (src / f"aoi{aoi}_rel_class_sieved.tif").exists():
        with cr.rules_for(aoi):
            cr.sieve(aoi, fresh=fresh)
        did.append("sieve")
    if not (src / f"aoi{aoi}_rel_fields_sliver{tag}.gpkg").exists():
        cr.fields(aoi, fresh=fresh)
        did.append("fields")
    return "+".join(did) or "present"


def step_lock(aoi: int, fresh: str = FRESH) -> str:
    from . import curve_rules as cr

    if (_locked(aoi, fresh) / "MANIFEST.json").exists():
        return "locked already"
    cr.lock(aoi, SLIVER_ACRES, fresh=fresh)
    return "locked now (add the AOI to curve_rules.LOCKED_AOIS)"


def field_table(aoi: int, fresh: str = FRESH):
    """The locked field layer with the delivery attributes: ``field_id`` (``aoi<N>_<polygon id>``, opens in notebook
    08), ``major_class`` (rice / non-rice), ``sub_class`` (the map class), ``acres``, ``origin``."""
    import geopandas as gpd

    from . import curve_rules as cr

    tag = f"{int(round(SLIVER_ACRES * 100)):03d}"
    f = gpd.read_file(_locked(aoi, fresh) / f"aoi{aoi}_rel_fields_sliver{tag}.gpkg")
    rice = {cr.MAP_CLASSES[k][0] for k in RICE_CODES}
    out = gpd.GeoDataFrame({
        "field_id": [f"aoi{aoi}_{p}" for p in f["polygon_id"]],
        "major_class": np.where(f["class_name"].isin(rice), "rice", "non-rice"),
        "sub_class": f["class_name"].to_numpy(),
        "acres": f["acres"].round(3).to_numpy(),
        "label_share": f["label_share"].round(2).to_numpy(),
        "origin": f["origin"].to_numpy()}, geometry=f.geometry, crs=f.crs)
    return out


def fields_qml(path) -> None:
    """QGIS style: filled by ``major_class`` (rice green, non-rice grey), outlines visible."""
    cats = [("rice", "35,139,69"), ("non-rice", "150,150,150")]
    c = "".join(f'<category value="{v}" symbol="{i}" label="{v}" render="true"/>' for i, (v, _) in enumerate(cats))
    s = "".join(f'''<symbol type="fill" name="{i}" alpha="1"><layer class="SimpleFill"><Option type="Map">
<Option type="QString" name="color" value="{rgb},110"/><Option type="QString" name="outline_color" value="{rgb},255"/>
<Option type="QString" name="outline_width" value="0.4"/></Option></layer></symbol>''' for i, (_, rgb) in enumerate(cats))
    Path(path).with_suffix(".qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28"><renderer-v2 type="categorizedSymbol" attr="major_class"><categories>{c}</categories>
<symbols>{s}</symbols></renderer-v2></qgis>
""")


def clear_dates(aoi: int) -> pd.DataFrame:
    """The AOI's Sentinel-2 dates with the share of its pixels clear that day (the pinned series' clear flag);
    ``deliver`` = at least ``CLEAR_SHARE_MIN``."""
    from . import ndvi_5day as nd

    d = nd.load(aoi, out_root=nd.analysis_series_root(aoi))
    dt = pd.DatetimeIndex(d["dates"])
    inside = nd.inside_from_loc(d["loc"])
    ok = d["ok"].reshape(len(dt), -1)[:, inside].astype(bool)
    nd.forget()
    out = pd.DataFrame({"date": dt.strftime("%Y-%m-%d"), "clear_share": ok.mean(axis=1).round(3)})
    out["deliver"] = (out["clear_share"] >= CLEAR_SHARE_MIN) & (dt >= pd.Timestamp(SEASON_FROM))
    return out


def s2_name(aoi: int, date: str) -> str:
    return f"aoi{aoi}_{date[:7]}_S2_{date}.tif"


def step_package(aoi: int, fresh: str = FRESH, **kw) -> dict:
    """Writes the delivery folder from the locked copy; returns the file list with sha256 (MANIFEST.json)."""
    from . import pixel_report as pr

    lk = _locked(aoi, fresh)
    man_lock = json.loads((lk / "MANIFEST.json").read_text())
    d = out_dir(aoi, **kw)
    d.mkdir(parents=True, exist_ok=True)
    g = Path(pr.locate(aoi, 0, season_key="monsoon2026")["run"]).parents[1] / "grid" / "pixel_index.tif"
    copies = {g: f"aoi{aoi}_pixel_index.tif",
              lk / f"aoi{aoi}_rel_class.tif": f"aoi{aoi}_class_raw.tif",
              lk / f"aoi{aoi}_rel_class.qml": f"aoi{aoi}_class_raw.qml",
              lk / f"aoi{aoi}_rel_class_sieved.tif": f"aoi{aoi}_class_sieved.tif",
              lk / f"aoi{aoi}_rel_class_sieved.qml": f"aoi{aoi}_class_sieved.qml"}
    for src, name in copies.items():
        if src.exists() and (not (d / name).exists() or (d / name).stat().st_size != src.stat().st_size):
            shutil.copy2(src, d / name)
    gp = d / f"aoi{aoi}_fields.gpkg"
    if not gp.exists():
        tmp = d / f"aoi{aoi}_fields.tmp.gpkg"
        field_table(aoi, fresh).to_file(tmp, driver="GPKG")
        tmp.replace(gp)
    fields_qml(gp)
    cd = clear_dates(aoi)
    cd.to_csv(d / f"aoi{aoi}_s2_dates.csv", index=False)
    s2 = [s2_name(aoi, x) for x in cd.loc[cd["deliver"], "date"]]
    files = {p.name: _sha(p) for p in sorted(d.iterdir()) if p.is_file() and p.suffix in (".tif", ".qml", ".gpkg", ".csv")}
    man = {"aoi": aoi, "delivery": DELIVERY, "locked": man_lock.get("locked"), "inputs": man_lock.get("inputs"),
           "rules": man_lock.get("rules"), "sliver_acres": SLIVER_ACRES,
           "rice_classes": [x for x in RICE_CODES], "clear_share_min": CLEAR_SHARE_MIN, "s2_dates": s2, "files": files}
    (d / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
    return man


def _bucket(aoi: int, key: str | None):
    import yaml

    from ..handover import _client

    cfg = yaml.safe_load(Path(f"config/aoi{aoi}_monsoon2026.yaml").read_text())
    keys = sorted(Path("secrets").glob("*.json"))      # the service-account key kept in secrets/ (never tracked)
    key = key or (str(keys[0]) if keys else None)
    client = _client(key, pool=32)
    return client.bucket(cfg["gcs"]["bucket"]), cfg["gcs"]["base_folder"]


def step_upload(aoi: int, key: str | None = None, workers: int = 16, **kw) -> dict:
    """Uploads the AOI's delivery folder and copies its clear S2 dates inside the bucket; skips blobs of equal size."""
    from concurrent.futures import ThreadPoolExecutor

    d = out_dir(aoi, **kw)
    man = json.loads((d / "MANIFEST.json").read_text())
    bucket, base = _bucket(aoi, key)
    dest = f"{base}/{DELIVERY}/aoi{aoi}"
    have = {b.name: b.size for b in bucket.list_blobs(prefix=dest + "/")}
    local = [p for p in sorted(d.iterdir()) if p.is_file() and p.name != "status.json" and not p.name.endswith(".tmp")]

    def up(p):
        name = f"{dest}/{p.name}"
        if have.get(name) == p.stat().st_size:
            return 0
        bucket.blob(name).upload_from_filename(str(p), timeout=600)
        return 1

    def cp(n):
        name = f"{dest}/s2/{n}"
        if name in have:
            return 0
        src = bucket.blob(f"{base}/s2_dates_masks/aoi{aoi}/{n}")
        bucket.copy_blob(src, bucket, name)
        return 1

    with ThreadPoolExecutor(workers) as ex:
        n_up = sum(ex.map(up, local))
        n_cp = sum(ex.map(cp, man["s2_dates"]))
    # S2 copies no longer in the manifest (the clear-share rule changed, user 5 Oct: 50 -> 80 %) are removed, only
    # inside this delivery's own s2/ folder
    keep = {f"{dest}/s2/{n}" for n in man["s2_dates"]}
    stale = [b for b in have if b.startswith(f"{dest}/s2/") and b not in keep]
    for b in stale:
        bucket.blob(b).delete()
    return {"gcs": f"gs://{bucket.name}/{dest}/", "uploaded": n_up, "s2_copied": n_cp, "s2_removed": len(stale),
            "s2_dates": len(man["s2_dates"])}


def run_one(aoi: int, fresh: str = FRESH, key: str | None = None, redo=()) -> dict:
    """All steps for one AOI, each skipped when ``status.json`` says it is done (``redo``: steps to run again, e.g.
    after a delivery setting changed)."""
    st = load_status(aoi)
    for s_ in redo:
        st.pop(s_, None)
    if "package" in redo:
        (out_dir(aoi) / f"aoi{aoi}_fields.gpkg").unlink(missing_ok=True)
    for step in STEPS:
        if st.get(step, {}).get("done"):
            continue
        if step == "fields":
            r = step_fields(aoi, fresh)
        elif step == "lock":
            r = step_lock(aoi, fresh)
        elif step == "package":
            r = {k: v for k, v in step_package(aoi, fresh).items() if k in ("s2_dates", "locked")}
        else:
            r = step_upload(aoi, key)
        st[step] = {"done": True, "result": r, "at": str(pd.Timestamp.now().floor("s"))}
        _save_status(aoi, st)
    return st


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["run", "status"])
    p.add_argument("--aois", type=int, nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=0, help="AOIs at once (0 = from the CPUs and RAM)")
    p.add_argument("--key", default=None)
    p.add_argument("--redo", nargs="*", default=[], choices=list(STEPS), help="run these steps again")
    a = p.parse_args(argv)
    if a.step == "status":
        for aoi in a.aois:
            st = load_status(aoi)
            print(f"aoi{aoi}: " + ", ".join(f"{s}={'done' if st.get(s, {}).get('done') else '-'}" for s in STEPS),
                  st.get("upload", {}).get("result", {}).get("gcs", ""))
        return 0
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from .. import resources
    r = resources.detect_resources()
    jobs = a.jobs or max(1, min(len(a.aois), r.cpus // 4, int(r.memory_available_bytes / 8e9)))
    with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_one, aoi, FRESH, a.key, tuple(a.redo)): aoi for aoi in a.aois}
        for f in as_completed(futs):
            aoi = futs[f]
            try:
                st = f.result()
                print(f"aoi{aoi} done: {st['upload']['result']['gcs']}  ({st['package']['result']['s2_dates'].__len__()} "
                      f"S2 dates)", flush=True)
            except Exception as e:                      # one AOI's failure must not stop the others
                print(f"aoi{aoi} FAILED: {type(e).__name__}: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
