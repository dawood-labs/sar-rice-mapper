"""Post-processing of the assistant's QC files (second QC round, 15 Oct 2026): from a QC'd field layer to a clean,
rice-only layer ready for QGIS "dissolve".

Why: the assistant QC'd every delivered AOI in QGIS (``qc_compare.qc2_files``). In the files: rice drawn as new
polygons without class, some non-rice fields given the major class rice, fields deleted or reshaped. The user's rules
(15 Oct 2026):

1. a drawn polygon (no ``field_id``) is rice: ``sub_class`` "rice (added in QC)", ``origin`` "drawn in QC", a new id;
2. a polygon the QC calls rice whose ``sub_class`` is not a rice class (e.g. "rice harvested", "tree/orchard": the
   assistant gave a non-rice field the major class rice) is rice with ``sub_class`` "rice (QC)". Read from the QC
   file alone, the assistant's final word (user, 15 Oct 2026: "why not only wo jo assistant ne qc kr k di h");
3. a reshaped field that now has several pieces keeps its id; ``part`` numbers the pieces (1, 2, ...);
4. polygons of another AOI inside a QC file (one QC file holds a neighbouring AOI's fields) are kept as they are;
5. the clean step of the first QC round (``qc_compare.clean_qc``, user: "bilkul sahi"): rice only, overlaps cut with
   the delineated (SAMGeo) outline winning over a drawn one, a 0.2 m gap only where neighbours meet, and the checks
   (overlap, touching, invalid) all 0;
6. the delivery rules for polygon outputs (manager, 12 Oct 2026): ``area_ac`` in acres with 3 decimals, measured in a
   metric CRS; a polygon that no step changed keeps its exact original geometry (no repair, no reprojection); a small
   piece made by cutting gets ``review`` = "small_clipped" (below :data:`SMALL_CLIPPED_AC`); ``predicted`` = 7, the
   official Crop Scan code of rice, styled #1B5E20.

Outputs (local, then uploaded as new files; the delivery is never touched)::

    processed/_batch/s2_2026/rice_map_2026-10-05_qc/first_set/aoi<N>/aoi<N>_fields_qc.gpkg (+ .qml, .json)
    processed/_batch/s2_2026/rice_map_2026-10-05_qc/second_set/aoi<1000+N>/aoi<1000+N>_fields_qc.gpkg

Run::

    python -m sar_pipeline.analysis.qc_post run --aois 19 1210     # or no --aois: every QC file
    python -m sar_pipeline.analysis.qc_post upload
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .qc_compare import CRUMB_M2, DRAWN_SUB_CLASS, SHRINK_M, check_layer, qc2_files, qc_base

OUT = "processed/_batch/s2_2026/rice_map_2026-10-05_qc"
GCS_FOLDER = "rice_map_2026-10-05_qc"
FLIPPED_SUB_CLASS = "rice (QC)"
#: Official Crop Scan code and colour of rice (the company's "Class - Labels 2.0").
RICE_CODE, RICE_HEX = 7, "#1B5E20"
#: A piece made by cutting (overlap or gap) smaller than this is flagged for review (4 pixels of 10 m).
SMALL_CLIPPED_AC = 0.1
#: A piece the 0.2 m gap cuts off (a narrow neck pinched through) smaller than this is dropped: cutting waste, not a
#: field, and it would sit millimetres from its own field (one AOI, 15 Oct 2026: 20 m2, 5 mm away).
GAP_CRUMB_M2 = 50.0
ACRE_M2 = 4046.8564224
COLUMNS = ["field_id", "part", "major_class", "sub_class", "predicted", "area_ac", "origin", "review", "geometry"]


def set_folder(aoi: int) -> str:
    return "second_set" if aoi >= 1000 else "first_set"


def out_path(aoi: int, root: str = OUT) -> Path:
    return Path(root) / set_folder(aoi) / f"aoi{aoi}" / f"aoi{aoi}_fields_qc.gpkg"


#: Sub classes that are rice in the delivery (``rice_map_delivery.RICE_CODES``): a rice polygon with any other sub class
#: was turned into rice by the QC.
RICE_SUBS = ("rice standing direct seeded", "rice standing transplanted", "young rice")


def label(aoi: int, qc) -> pd.DataFrame:
    """Rules 1-2 on the QC file alone: drawn polygons and fields turned into rice labelled; rice only (rule 5)."""
    rice = qc[(qc["major_class"] == "rice") | qc["field_id"].isna()].copy()
    drawn = rice["field_id"].isna()
    rice.loc[drawn, "field_id"] = [f"aoi{aoi}_qc{i + 1:04d}" for i in range(int(drawn.sum()))]
    rice.loc[drawn, "sub_class"] = DRAWN_SUB_CLASS
    rice.loc[drawn, "origin"] = "drawn in QC"
    flipped = ~drawn & ~rice["sub_class"].isin(RICE_SUBS)
    rice.loc[flipped, "sub_class"] = FLIPPED_SUB_CLASS
    rice["major_class"] = "rice"
    rice["drawn"] = drawn.to_numpy()
    rice["flipped"] = flipped.to_numpy()
    return rice


def clean(rice, shrink_m: float = SHRINK_M, crumb_m2: float = CRUMB_M2, small_ac: float = SMALL_CLIPPED_AC):
    """Rules 3, 5 and 6 on the labelled rice layer. A polygon is cut only where it overlaps an earlier one
    (delineated before drawn, largest first) or has a neighbour within ``2 * shrink_m``; any other polygon keeps the
    QC's geometry object as it is."""
    import geopandas as gpd
    import shapely
    from shapely.strtree import STRtree

    rice = rice.assign(sz_=rice.area, ord_=range(len(rice))).sort_values(["drawn", "sz_"], ascending=[True, False])
    taken, rows = [], []
    for r in rice.itertuples():
        g = r.geometry
        hit = [taken[i] for i in STRtree(taken).query(g, predicate="intersects")] if taken else []
        hit = [h for h in hit if h.intersection(g).area > 0]
        if hit:
            pieces = [p for p in _polygons(shapely.make_valid(g.difference(shapely.union_all(hit)))) if p.area >= crumb_m2]
            cut = True
        else:
            pieces, cut = [g] if g.is_valid else [p for p in _polygons(shapely.make_valid(g)) if p.area >= crumb_m2], \
                not g.is_valid
        for p in pieces:
            taken.append(p)
            rows.append({**r._asdict(), "geometry": p, "cut_": cut})
    if not rows:                                    # the QC kept no rice in this AOI: an empty layer, same columns
        return gpd.GeoDataFrame({c: [] for c in COLUMNS[:-1]}, geometry=[], crs=rice.crs)
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=rice.crs)
    geoms = list(out.geometry)
    tree = STRtree(geoms)
    final = []
    for i, r in enumerate(out.itertuples()):
        near = [geoms[j] for j in tree.query(r.geometry, predicate="dwithin", distance=2 * shrink_m) if j != i]
        if near:
            g = shapely.make_valid(r.geometry.difference(shapely.union_all(near).buffer(shrink_m, join_style="mitre")))
            pieces, cut = [p for p in _polygons(g) if p.area >= max(crumb_m2, GAP_CRUMB_M2)], True
        else:
            pieces, cut = [r.geometry], r.cut_
        for p in pieces:
            final.append({"field_id": r.field_id, "sub_class": r.sub_class, "origin": r.origin, "ord_": r.ord_,
                          "cut_": cut, "geometry": p})
    f = gpd.GeoDataFrame(final, geometry="geometry", crs=rice.crs).sort_values(["ord_", "field_id"])
    f["part"] = f.groupby("field_id").cumcount() + 1
    f["major_class"] = "rice"
    f["predicted"] = RICE_CODE
    f["area_ac"] = (f.area / ACRE_M2).round(3)
    f["review"] = ["small_clipped" if c and a < small_ac else "" for c, a in zip(f["cut_"], f["area_ac"])]
    return f[COLUMNS].reset_index(drop=True)


def _polygons(geom):
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]


def check(f) -> dict:
    """``qc_compare.check_layer`` with (field_id, part) as the id (rule 3 keeps the id on every piece)."""
    if len(f) == 0:
        return {"polygons_out": 0, "invalid": 0, "not_polygon": 0, "duplicate_ids": 0, "overlapping_pairs": 0,
                "touching_pairs": 0}
    c = check_layer(f.assign(field_id=f["field_id"].astype(str) + "#" + f["part"].astype(str)))
    return {**c, "invalid": int((~f.is_valid).sum())}


def qml(path) -> None:
    """QGIS style: rice filled in the official colour."""
    Path(path).with_suffix(".qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28"><renderer-v2 type="singleSymbol"><symbols><symbol type="fill" name="0" alpha="1">
<layer class="SimpleFill"><Option type="Map"><Option type="QString" name="color" value="27,94,32,160"/>
<Option type="QString" name="outline_color" value="27,94,32,255"/><Option type="QString" name="outline_width" value="0.4"/>
</Option></layer></symbol></symbols></renderer-v2></qgis>
""")


def run_one(aoi: int, qc_file, root: str = OUT) -> dict:
    """Rules 1-6 for one AOI; writes the layer, its style and a json report."""
    import geopandas as gpd

    from .qc_compare import DELIVERED_DIR
    from .rice_map_delivery import bucket_name

    # the delivered layer the QC was made on: for the record and for the sowing step's rule version only
    b = qc_base(aoi, qc_file) if (Path(DELIVERED_DIR) / bucket_name(aoi)).exists() else None
    lay = b.iloc[0]["layer"] if b is not None and len(b) else ""
    qc = gpd.read_file(qc_file)
    rice = label(aoi, qc)
    f = clean(rice)
    out = out_path(aoi, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}")
    f.to_file(tmp, driver="GPKG", layer=out.stem)
    tmp.replace(out)
    qml(out)
    rep = {"aoi": bucket_name(aoi), "qc_file": Path(qc_file).name, "base_layer": lay,
           "base_same_geometry_pct": float(b.iloc[0]["same_geometry_pct"]) if lay else None, "qc_polygons": len(qc),
           "rice_in": len(rice), "drawn": int(rice["drawn"].sum()), "flipped": int(rice["flipped"].sum()),
           "polygons_out": len(f), "fields_out": int(f["field_id"].nunique()),
           "rice_acres_in": round(float(rice.area.sum()) / ACRE_M2, 3), "rice_acres_out": round(float(f["area_ac"].sum()), 3),
           "small_clipped": int((f["review"] == "small_clipped").sum()), **check(f)}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def _one(args):
    aoi, qc_file = args
    try:
        return run_one(aoi, qc_file)
    except Exception as e:                                   # one AOI's failure does not stop the others
        return {"aoi": aoi, "error": f"{type(e).__name__}: {e}"}


def run(aois=None, jobs: int = 0) -> pd.DataFrame:
    """Every QC file (or ``aois``), one process per core; the summary goes to ``<OUT>/_summary/qc_summary.csv``."""
    from concurrent.futures import ProcessPoolExecutor

    from .. import resources

    files = qc2_files()
    todo = [(a, files[a]) for a in (aois or sorted(files))]
    with ProcessPoolExecutor(jobs or resources.detect_resources().cpus) as ex:
        rows = list(ex.map(_one, todo))
    t = pd.DataFrame(rows)
    s = Path(OUT) / "_summary"
    s.mkdir(parents=True, exist_ok=True)
    t.to_csv(s / "qc_summary.csv", index=False)
    return t


SOWING_GROUP = "with_sowing_no_clear_sep"     # user, 15 Oct 2026: the 43 AOIs with sowing in their own folder


def upload(threads: int = 32) -> dict:
    """Uploads every AOI's QC layer, style and report, plus the summaries, to ``<base>/<GCS_FOLDER>/`` (a new folder;
    nothing delivered is touched). The AOIs of ``qc_sowing.NO_CLEAR_SEPTEMBER`` go under :data:`SOWING_GROUP`, so
    they are easy to review. Second-set files carry the client's name inside and out
    (``rice_map_delivery.client_file``). Each upload is checked by md5 afterwards."""
    import base64
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from . import qc_sowing as qs
    from . import rice_map_delivery as rd

    b, base = rd._bucket(19, None)
    dest = f"{base}/{GCS_FOLDER}"
    items = []
    for d in sorted(Path(OUT).glob("*_set/aoi*")):
        aoi = int(d.name[3:])
        group = f"{SOWING_GROUP}/" if aoi in qs.NO_CLEAR_SEPTEMBER else ""
        for f in sorted(d.glob(f"aoi{aoi}_fields_qc.*")):
            c = rd.client_file(aoi, f)
            items.append((c, f"{dest}/{group}{set_folder(aoi)}/{rd.bucket_name(aoi)}/{c.name}"))
    for name, src in (("qc_summary.csv", Path(OUT) / "_summary" / "qc_summary.csv"),
                      ("sowing_review_score.csv", qs.SOWING_DIR / "review_score.csv")):
        if src.exists():
            items.append((src, f"{dest}/_summary/{name}"))
    md5 = lambda p: base64.b64encode(hashlib.md5(Path(p).read_bytes()).digest()).decode()  # noqa: E731

    def one(it):
        src, name = it
        b.blob(name).upload_from_filename(str(src), timeout=600)
        return b.get_blob(name).md5_hash == md5(src)

    with ThreadPoolExecutor(threads) as ex:
        ok = list(ex.map(one, items))
    return {"files": len(items), "verified": int(sum(ok)), "gcs": f"gs://{b.name}/{dest}/"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["run", "upload"])
    p.add_argument("--aois", type=int, nargs="*", default=[])
    a = p.parse_args(argv)
    if a.step == "upload":
        print(json.dumps(upload(), indent=1))
        return 0
    t = run(a.aois or None)
    print(t.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
