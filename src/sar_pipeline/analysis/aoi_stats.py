"""One layer and one table of every AOI with its area, rice and non-rice (manager via the user, 16 Oct 2026).

What the manager asked for:

1. one merged shapefile of all AOIs, each with its unique id (``aoi<N>`` for the first set, ``b2_aoi<N>`` for the
   second, as delivered);
2. in it, each AOI's total area and its rice and non-rice area;
3. a CSV with the same per AOI plus further stats, and the totals over all AOIs.

User decisions (16 Oct 2026):

* non-rice = AOI area minus rice area (roads, villages, water and other crops included), so rice + non-rice = AOI;
* a second-set AOI is its polygon with the part already delivered in the first set cut out (``aoi_batch2``), so no
  area counts twice;
* the five second-set AOIs that were not mapped (three duplicates, two empty) are in the layer with status
  "dropped" and their reason, and are left out of the totals;
* rice is the final QC'd layer (``qc_post``); its area is counted inside the AOI only (a delineated field can reach a
  few metres past the AOI edge), the fields' full area is kept beside it.

Further stats per AOI (to help the manager): rice fields and their mean size, polygons drawn and classes changed in
the QC, small clipped pieces, whether the AOI had a clear September view (and a newer clear view after it), and for
the AOIs with sowing dates (``qc_sowing``) the median crop age on 15 September, the share of rice fields 50 days old
or younger, and how many fields were sown in each month.

Outputs (``processed/_batch/s2_2026/rice_map_2026-10-05_qc/_summary/``): ``all_aois.shp`` (+ .dbf .shx .prj .cpg),
``all_aois.gpkg`` (the same with full column names), ``aoi_stats.csv`` (one row per AOI, then a TOTAL row). Areas in
acres with 3 decimals, measured in EPSG:32646.

Run::

    python -m sar_pipeline.analysis.aoi_stats build
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .qc_post import OUT, out_path

ACRE_M2 = 4046.8564224
METRIC = "EPSG:32646"
FIRST_SET = "../data/aoi/first_set_132_merged.gpkg"
SECOND_SET_SPLIT = "../data/aoi_batch2_2026-09-29/split"
SECOND_SET_ALL = "../data/aoi_batch2_2026-09-29/aois_new_106_merged.gpkg"
SECOND_SET_INDEX = "../data/aoi_batch2_2026-09-29/aoi_index.csv"
NEW_DATES_REPORT = "processed/_batch/s2_2026/report/new_s2_check_2026-10-12.json"
SUMMARY = Path(OUT) / "_summary"
#: Shapefile field names (at most 10 characters) for the GeoPackage / CSV names.
SHP_NAMES = {"aoi_id": "aoi_id", "set": "set", "status": "status", "reason": "reason", "aoi_ac": "aoi_ac",
             "rice_ac": "rice_ac", "nonrice_ac": "nonrice_ac", "rice_pct": "rice_pct", "rice_fields": "rice_flds",
             "overlap_cut_ac": "ovl_cut_ac"}


def aoi_polygons() -> "gpd.GeoDataFrame":
    """Every AOI: ``aoi`` (internal id), ``aoi_id`` (delivered name), ``set``, ``status``, ``reason``,
    ``overlap_cut_ac``, geometry in :data:`METRIC`."""
    import geopandas as gpd

    from .aoi_batch2 import OFFSET, SKIP_REASONS
    from .rice_map_delivery import bucket_name

    f = gpd.read_file(FIRST_SET).to_crs(METRIC)
    first = gpd.GeoDataFrame({"aoi": f["id"].astype(int), "set": "first", "status": "mapped", "reason": "",
                              "overlap_cut_ac": 0.0}, geometry=f.geometry.to_numpy(), crs=METRIC)
    parts = [gpd.read_file(p).to_crs(METRIC) for p in sorted(Path(SECOND_SET_SPLIT).glob("b2_*/b2_*.gpkg"))]
    s = pd.concat(parts, ignore_index=True)
    cut = pd.read_csv(SECOND_SET_INDEX).set_index("aoi")["acres_cut_as_delivered"]
    second = gpd.GeoDataFrame({"aoi": s["aoi"].astype(int), "set": "second", "status": "mapped", "reason": "",
                               "overlap_cut_ac": s["aoi"].map(cut).fillna(0.0).round(3)},
                              geometry=s.geometry.to_numpy(), crs=METRIC)
    a = gpd.read_file(SECOND_SET_ALL)
    d = a[a["status"] == "dropped"].to_crs(METRIC)
    dropped = gpd.GeoDataFrame({"aoi": OFFSET + d["number"].astype(int), "set": "second", "status": "dropped",
                                "reason": d["number"].astype(int).map(SKIP_REASONS), "overlap_cut_ac": 0.0},
                               geometry=d.geometry.to_numpy(), crs=METRIC)
    out = gpd.GeoDataFrame(pd.concat([first, second, dropped], ignore_index=True), crs=METRIC)
    out.insert(0, "aoi_id", out["aoi"].map(bucket_name))
    return out


def _rice_stats(args) -> dict:
    """Rice and QC stats of one mapped AOI (a worker process)."""
    import geopandas as gpd
    import shapely

    aoi, wkb = args
    geom = shapely.from_wkb(wkb)
    p = out_path(aoi)
    r = {"aoi": aoi}
    if not p.exists():
        return {**r, "qc_layer": "missing"}
    lay = gpd.read_file(p)
    rep = json.loads(p.with_suffix(".json").read_text())
    inside = shapely.union_all(lay.geometry.to_numpy()).intersection(geom) if len(lay) else None
    r.update({"rice_ac": round(inside.area / ACRE_M2, 3) if inside is not None else 0.0,
              "rice_fields_full_ac": round(float(lay["area_ac"].sum()), 3) if len(lay) else 0.0,
              "rice_fields": int(lay["field_id"].nunique()) if len(lay) else 0,
              "rice_polygons": len(lay),
              "mean_field_ac": round(float(lay.groupby("field_id")["area_ac"].sum().mean()), 3) if len(lay) else None,
              "qc_drawn_fields": int((lay["origin"] == "drawn in QC").groupby(lay["field_id"]).any().sum()) if len(lay) else 0,
              "qc_drawn_ac": round(float(lay.loc[lay["origin"] == "drawn in QC", "area_ac"].sum()), 3) if len(lay) else 0.0,
              "qc_changed_to_rice_fields": int((lay["sub_class"] == "rice (QC)").groupby(lay["field_id"]).any().sum())
              if len(lay) else 0,
              "small_clipped": int((lay["review"] == "small_clipped").sum()) if len(lay) else 0,
              "qc_base_layer": rep.get("base_layer", "")})
    if "crop_age_15sep" in lay.columns and len(lay):
        f = lay.drop_duplicates("field_id")
        age = pd.to_numeric(f["crop_age_15sep"], errors="coerce").dropna()
        month = pd.to_datetime(f["sowing_date"], errors="coerce").dt.month
        r.update({"median_crop_age_15sep": float(age.median()) if len(age) else None,
                  "rice_fields_age_le50_pct": round(100 * float((age <= 50).mean()), 1) if len(age) else None,
                  **{f"sown_{m}": int((month == i).sum()) for i, m in
                     ((5, "may"), (6, "jun"), (7, "jul"), (8, "aug"), (9, "sep"))},
                  "sown_after_15sep": int((f["sowing_note"] == "sown after 15 Sep").sum())})
    return r


def build(jobs: int = 0) -> tuple[Path, Path, Path]:
    """The layer (shapefile + GeoPackage) and the CSV of :mod:`this module <aoi_stats>`."""
    from concurrent.futures import ProcessPoolExecutor

    import geopandas as gpd
    import shapely

    from .. import resources
    from . import qc_sowing as qs

    a = aoi_polygons()
    a["aoi_ac"] = (a.area / ACRE_M2).round(3)
    mapped = a[a["status"] == "mapped"]
    with ProcessPoolExecutor(jobs or resources.detect_resources().cpus) as ex:
        stats = pd.DataFrame(list(ex.map(_rice_stats, [(int(x), shapely.to_wkb(g)) for x, g in
                                                       zip(mapped["aoi"], mapped.geometry)])))
    a = a.merge(stats, on="aoi", how="left")
    a["nonrice_ac"] = np.where(a["status"] == "mapped", (a["aoi_ac"] - a["rice_ac"]).round(3), np.nan)
    a["rice_pct"] = np.where(a["status"] == "mapped", (100 * a["rice_ac"] / a["aoi_ac"]).round(2), np.nan)
    a["clear_sep_view"] = np.where(a["status"] == "mapped", ~a["aoi"].isin(qs.NO_CLEAR_SEPTEMBER), pd.NA)
    newer = set()
    if Path(NEW_DATES_REPORT).exists():
        from ..newest_imagery import clear_dates_to_fetch

        newer = {int(k[3:]) for k in clear_dates_to_fetch(json.loads(Path(NEW_DATES_REPORT).read_text()))}
    a["clear_oct_view_added"] = np.where(a["aoi"].isin(qs.NO_CLEAR_SEPTEMBER), a["aoi"].isin(newer), pd.NA)
    a = a.sort_values(["set", "aoi"]).reset_index(drop=True)
    SUMMARY.mkdir(parents=True, exist_ok=True)
    cols = [c for c in a.columns if c not in ("geometry", "aoi")]
    gp = SUMMARY / "all_aois.gpkg"
    a[cols + ["geometry"]].to_file(gp, driver="GPKG", layer="all_aois")
    shp = SUMMARY / "all_aois.shp"
    a[list(SHP_NAMES) + ["geometry"]].rename(columns=SHP_NAMES).to_file(shp, driver="ESRI Shapefile")
    t = a[cols].copy()
    m = t["status"] == "mapped"
    union_ac = shapely.union_all(a.loc[m, "geometry"].to_numpy()).area / ACRE_M2
    total = {"aoi_id": "TOTAL (mapped AOIs)", "set": "all", "status": f"{int(m.sum())} mapped, "
             f"{int((~m).sum())} dropped (not counted)",
             "reason": f"AOIs overlap each other by {t.loc[m, 'aoi_ac'].sum() - union_ac:.3f} ac; "
                       f"area of their union {union_ac:.3f} ac"}
    for c in ("aoi_ac", "rice_ac", "nonrice_ac", "rice_fields_full_ac", "rice_fields", "rice_polygons",
              "qc_drawn_fields", "qc_drawn_ac", "qc_changed_to_rice_fields", "small_clipped", "overlap_cut_ac",
              "sown_may", "sown_jun", "sown_jul", "sown_aug", "sown_sep", "sown_after_15sep"):
        if c in t:
            total[c] = round(float(pd.to_numeric(t.loc[m, c], errors="coerce").sum()), 3)
    total["rice_pct"] = round(100 * total["rice_ac"] / total["aoi_ac"], 2)
    t = pd.concat([t, pd.DataFrame([total])], ignore_index=True)
    csv = SUMMARY / "aoi_stats.csv"
    t.to_csv(csv, index=False)
    return shp, gp, csv


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["build"])
    p.parse_args(argv)
    for f in build():
        print(f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
