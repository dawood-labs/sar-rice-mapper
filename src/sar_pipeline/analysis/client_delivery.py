"""The three layers the client receives: their original AOIs, the AOIs we mapped, and every rice field (16 Oct 2026).

What the manager asked for (via the user, 16 Oct 2026):

1. **Original AOIs** (``1_original_aois/original_aois.shp``): every AOI polygon the client sent, from all their
   shapefiles, with their attributes exactly as given (a column a file did not have stays empty; Z values kept), plus
   ``source`` (which of their shapefiles the polygon came from; user: "daal do").
2. **Final AOIs** (``2_final_aois/final_aois.shp``): the AOIs we worked on, i.e. without the duplicates and the empty
   / invalid polygons: the 132 of the first set and the 101 mapped of the second. Each is the client's original
   polygon (user: "client k original", also where we cut out the part the first set already covered) with
   ``aoi_id`` (``aoi<N>`` / ``b2_aoi<N>``, as delivered), ``Company``, ``District``, ``VT`` (village tract name in
   local script) and ``VTE`` (the same in English). Their files name these columns differently (``Company`` /
   ``JCompany`` / ``J_Company`` ...) and put the local-script name sometimes in ``VT``, sometimes in ``VT_2`` or ``J_VT``,
   so each value is taken from the first column of its kind that holds it, local-script values by their script. A value the
   client left empty stays empty (user: "jese client ne dia wese hi"); only surrounding blanks / line breaks are cut.
3. **Rice** (``3_rice/rice_fields.shp``): every rice polygon of the final QC'd layers (``qc_post``) of all AOIs in one
   layer, dissolved and split into single parts, with only ``predicted`` (7 = Rice, our standard class code) and
   ``area_ac`` (acres, 3 decimals, measured in EPSG:32646). Why dissolve: AOIs overlap, and a few hold the fields of
   another AOI (aoi117 holds aoi83's, aoi118 holds aoi72's), so about 1,060 acres would be counted twice. Only the
   polygons that overlap or touch another are merged; every other polygon keeps its geometry (the 0.2 m gap the QC
   step leaves keeps neighbouring fields apart, tested on aoi160, aoi28 and b2_aoi198: nothing merged). Delivered
   in EPSG:4326 (user), the area measured in EPSG:32646. Slivers under 1 m2 are left out.

All three are written as ESRI shapefiles in WGS84 (EPSG:4326) under :data:`OUT_DIR`, and ``upload`` copies them to
``<base>/client_delivery_2026-10-16/`` on GCS (md5-checked).

Inputs: the client's files unpacked to :data:`CLIENT_DIR` (``<zip name>/<folder>/<file>.shp``; a zip delivered twice,
e.g. ``x.zip`` and ``x (1).zip``, is read once), our AOI tables (``aoi_stats.FIRST_SET`` / ``SECOND_SET_ALL``) and
the QC layers.

Run::

    python -m sar_pipeline.analysis.client_delivery originals
    python -m sar_pipeline.analysis.client_delivery final-aois
    python -m sar_pipeline.analysis.client_delivery rice
    python -m sar_pipeline.analysis.client_delivery upload
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .aoi_stats import ACRE_M2, FIRST_SET, METRIC, SECOND_SET_ALL

CLIENT_DIR = Path("../data/client_aoi_files_2026-10-16/unzipped")
OUT_DIR = Path("processed/_batch/s2_2026/client_delivery_2026-10-16")
GCS_FOLDER = "client_delivery_2026-10-16"
DELIVERY_CRS = "EPSG:4326"
RICE_CODE = 7
#: Rice pieces smaller than this (m2) are slivers, not fields, and are left out.
MIN_PIECE_M2 = 1.0
#: Columns that hold the same thing in the client's different files, in the order they are tried.
COMPANY = ("Company", "JCompany", "J_Company")
DISTRICT = ("District", "JDistrict", "J_District")
VILLAGE_TRACT = ("VTE", "VT", "VT_2", "VTE_2", "JVT", "JVTE", "J_VT", "J_VTE")
LOCAL_SCRIPT = re.compile("[က-႟]")


def client_shapefiles(root: Path = CLIENT_DIR) -> list[Path]:
    """Every client shapefile once: ``<zip>/<folder>/<file>.shp``, the same folder/file from a second copy of a zip
    (``x (1).zip``) skipped."""
    seen, out = set(), []
    for p in sorted(root.glob("*/*/*.shp"), key=lambda p: (p.parts[-3].endswith("(1)"), str(p))):
        key = (p.parent.name, p.name)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return sorted(out, key=lambda p: (p.parent.name, p.name))


def originals(root: Path = CLIENT_DIR) -> "gpd.GeoDataFrame":
    """All client polygons in one table, their columns as given plus ``source``."""
    import geopandas as gpd

    parts = []
    for p in client_shapefiles(root):
        g = gpd.read_file(p)
        g["source"] = f"{p.parent.name}/{p.name}"
        parts.append(g.to_crs(DELIVERY_CRS) if g.crs and g.crs != DELIVERY_CRS else g)
    out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=DELIVERY_CRS)
    # shapefile column names are case-insensitive: one file has ``AREA`` beside the others' ``area``, so it is kept
    # under ``AREA_1`` (the name GDAL itself would give it)
    out = out.rename(columns={c: f"{c}_1" for c in out.columns if c != c.lower() and c.lower() in out.columns})
    for c in out.columns:
        # a column numeric in one file and text in another: written as text (a shapefile column has one type)
        if c != "geometry" and out[c].dtype == object:
            out[c] = out[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    return out


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def pick(row: pd.Series, cols, local: bool | None = None):
    """The first non-empty value among ``cols``; ``local`` True / False keeps only a value in / not in the local
    script."""
    for c in cols:
        v = _clean(row.get(c))
        if v is not None and (local is None or bool(LOCAL_SCRIPT.search(v)) == local):
            return v
    return None


def final_aois() -> "gpd.GeoDataFrame":
    """The 233 AOIs we mapped, with the client's original polygon and the five delivered columns."""
    import geopandas as gpd

    from .aoi_batch2 import OFFSET
    from .rice_map_delivery import bucket_name

    f = gpd.read_file(FIRST_SET).to_crs(DELIVERY_CRS)
    f["aoi"] = f["id"].astype(int)
    s = gpd.read_file(SECOND_SET_ALL).to_crs(DELIVERY_CRS)
    s = s[s["status"] == "mapped"].copy()
    s["aoi"] = OFFSET + s["number"].astype(int)
    rows = []
    for t in (f, s):
        for _, r in t.iterrows():
            rows.append({"aoi_id": bucket_name(int(r["aoi"])), "Company": pick(r, COMPANY),
                         "District": pick(r, DISTRICT), "VT": pick(r, VILLAGE_TRACT, local=True),
                         "VTE": pick(r, VILLAGE_TRACT, local=False), "geometry": r.geometry})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=DELIVERY_CRS)


def dissolve_overlaps(geoms: np.ndarray) -> np.ndarray:
    """Merges the polygons that overlap or touch (connected groups) and returns single parts; a polygon that touches
    nothing is returned as it is."""
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    tree = shapely.STRtree(geoms)
    i, j = tree.query(geoms, predicate="intersects")
    n = len(geoms)
    _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)
    size = np.bincount(lab)
    out = list(geoms[size[lab] == 1])
    for k in np.flatnonzero(size > 1):
        u = shapely.union_all(geoms[lab == k])
        out += list(shapely.get_parts(u)) if u.geom_type != "Polygon" else [u]
    return np.array([g for g in out if g.geom_type == "Polygon" and not g.is_empty], dtype=object)


def drop_empty_holes(poly, min_area: float = 1e-12):
    """The polygon without holes of (almost) no area (``min_area`` in squared degrees, ~0.01 m2). Why: a collapsed hole
    has no ring direction, and a shapefile reader then takes it for a second outer ring ("nested shells", one polygon
    on 16 Oct)."""
    import shapely

    keep = [h for h in poly.interiors if shapely.Polygon(h).area >= min_area]
    return poly if len(keep) == len(poly.interiors) else shapely.Polygon(poly.exterior, keep)


def rice() -> tuple["gpd.GeoDataFrame", dict]:
    """Every rice polygon of every QC layer, dissolved where they overlap, single parts, ``predicted`` + ``area_ac``."""
    import geopandas as gpd
    import shapely

    from .qc_post import OUT

    layers = sorted(Path(OUT).glob("*_set/aoi*/aoi*_fields_qc.gpkg"))
    parts = [gpd.read_file(p).to_crs(METRIC).geometry for p in layers]
    g = np.concatenate([p.to_numpy() for p in parts if len(p)])
    single = gpd.GeoSeries(dissolve_overlaps(g), crs=METRIC).to_crs(DELIVERY_CRS)
    # The reprojection to degrees turns a few narrow polygons self-intersecting (373 of ~200,000 on 16 Oct; all valid
    # in the metric QC layers): repaired, and only the polygon parts kept, as single parts.
    bad = ~single.is_valid
    fixed = shapely.make_valid(single[bad].to_numpy())
    parts = np.concatenate([single[~bad].to_numpy()] + [shapely.get_parts(shapely.get_parts(f)) for f in fixed])
    parts = np.array([drop_empty_holes(x) for x in parts if x.geom_type == "Polygon" and not x.is_empty and x.is_valid],
                     dtype=object)
    out = gpd.GeoDataFrame({"predicted": np.full(len(parts), RICE_CODE, dtype="int32")}, geometry=parts,
                           crs=DELIVERY_CRS)
    m2 = out.to_crs(METRIC).area                                          # measured in metres, delivered in degrees
    # slivers under 1 m2 (78 on 16 Oct, 0.001 ac in all: pinch-point pieces of the repair and crumbs of the QC clip)
    # carry no field and would show as 0.000 ac
    crumbs = m2 < MIN_PIECE_M2
    out, m2 = out[~crumbs].reset_index(drop=True), m2[~crumbs].reset_index(drop=True)
    out["area_ac"] = (m2 / ACRE_M2).round(3)
    rep = {"qc_layers": len(layers), "polygons_in": int(len(g)), "polygons_out": int(len(out)),
           "repaired_after_reprojection": int(bad.sum()), "slivers_dropped": int(crumbs.sum()),
           "acres_in": round(float(sum(x.area for x in g)) / ACRE_M2, 3), "acres_out": round(float(out["area_ac"].sum()), 3)}
    return out, rep


def write(gdf, name: str, folder: str) -> Path:
    d = OUT_DIR / folder
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.shp"
    gdf.to_file(p, driver="ESRI Shapefile", encoding="UTF-8")
    return p


def upload(threads: int = 32) -> dict:
    """Copies every delivered file to ``<base>/<GCS_FOLDER>/`` and checks each by md5."""
    import base64
    import hashlib
    from concurrent.futures import ThreadPoolExecutor

    from . import rice_map_delivery as rd

    b, base = rd._bucket(19, None)
    files = sorted(p for p in OUT_DIR.rglob("*") if p.is_file())
    md5 = lambda p: base64.b64encode(hashlib.md5(p.read_bytes()).digest()).decode()  # noqa: E731

    def one(f):
        name = f"{base}/{GCS_FOLDER}/{f.relative_to(OUT_DIR)}"
        b.blob(name).upload_from_filename(str(f), timeout=600)
        return b.get_blob(name).md5_hash == md5(f)

    with ThreadPoolExecutor(threads) as ex:
        ok = list(ex.map(one, files))
    return {"files": len(files), "verified": int(sum(ok)), "gcs": f"gs://{b.name}/{base}/{GCS_FOLDER}/"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["originals", "final-aois", "rice", "upload"])
    a = p.parse_args(argv)
    if a.step == "originals":
        g = originals()
        print(write(g, "original_aois", "1_original_aois"), len(g), "polygons from", g["source"].nunique(), "files")
    elif a.step == "final-aois":
        g = final_aois()
        print(write(g, "final_aois", "2_final_aois"), len(g), "AOIs;", g[["Company", "District", "VT", "VTE"]]
              .notna().sum().to_dict())
    elif a.step == "rice":
        g, rep = rice()
        print(write(g, "rice_fields", "3_rice"), json.dumps(rep))
    else:
        print(json.dumps(upload(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
