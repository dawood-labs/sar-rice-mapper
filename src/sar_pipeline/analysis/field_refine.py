"""Refine the field delineation before and after labelling (fix plan, stage 3).

Why
---
The delineation (a segmentation model's output) is used as delivered, and the review found four
kinds of artefact in it: "monster" outlines drawn around a dozen fields that also have their own
polygons (so their acres are counted twice and the monster is labelled from the bunds between
them), roads / canals / tree lines traced as long thin "fields", ponds traced as fields, and tiny
polygons (1-3 pixels) inside a field of the same class. The polygon areas were also computed in Web
Mercator and run ~9 % high. This module fixes those on the delineation itself, so the labelling
(``field_rice``) and the delivery inherit clean geometry.

Two phases, because some rules need the class labels:

* :func:`geometry_stage` (before labelling): overlaps resolved (the smaller polygon wins; a larger
  one is cut to what the smaller ones leave), monsters that lose most of their area dropped when the
  remainder is small or a strip, long thin polygons flagged as ``strip`` (roads, canals, tree
  lines), outlines simplified by 0.5 m (the traced pixel staircase), areas recomputed in the local
  UTM zone. Every polygon keeps its original ``field_id`` (the id used in the review and the
  delivery), so a field can be followed across versions.
* :func:`label_stage` (after labelling): polygons smaller than 4 pixels whose touching neighbours
  all share their label are merged into the largest such neighbour; polygons that are water all
  season are flagged ``pond``. Flagged polygons stay in the file with ``is_field = False`` and a
  ``refine_flag`` saying why, so nothing is silently deleted.

Use::

    python -m sar_pipeline.analysis.field_refine geometry --ids 40 116     # -> fields_refined/aoi<N>_delineation_refined.gpkg
    (then field_rice labels those files; label_stage runs inside field_rice.run when asked)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SRC = "processed/_batch/s2_2026"
OUT = f"{SRC}/fields_refined"
SQM_PER_ACRE = 4046.8564224
PIXEL_M2 = 100.0
#: A polygon covered by smaller polygons over at least this share is a "monster" outline.
MONSTER_COVER = 0.5
#: What is left of a cut monster must be at least this many pixels, and not a strip, to stay.
MIN_REMAINDER_PX = 4
#: A strip: mean width (2 x area / perimeter) under this, and longer than ``STRIP_MIN_LENGTH_M``.
#: Why 15 m: the fields here are 20-100 m wide; roads, canals and tree lines are 5-15 m. Real
#: narrow terraces exist (one AOI in the south-west); their length is usually under 150 m.
STRIP_MAX_WIDTH_M = 15.0
STRIP_MIN_LENGTH_M = 150.0
#: Simplification tolerance for the traced outlines (far below the 10 m pixel).
SIMPLIFY_M = 0.5
#: A polygon under this many pixels inside same-label neighbours is merged into the largest one.
TINY_PX = 4
#: A polygon is a pond when this share of its pixels was water all season.
POND_SHARE = 0.6


def _utm(g):
    return g.estimate_utm_crs()


def cut_overlaps(g):
    """Resolve overlaps: every polygon loses the parts covered by SMALLER polygons.

    ``g`` in a metric CRS with valid geometries. Returns (geometry series, share of area lost).
    Why smaller wins: the segmentation traces small fields precisely and draws the large outline
    around groups of them; the large outline's own content is the bunds and paths between fields.
    """
    import geopandas as gpd
    import shapely

    a = g.geometry.area.to_numpy()
    idx = gpd.GeoDataFrame({"a": a}, geometry=g.geometry.to_numpy(), crs=g.crs)
    pairs = gpd.sjoin(idx[["geometry", "a"]], idx[["geometry", "a"]], predicate="intersects", how="inner")
    pairs = pairs[(pairs.index != pairs["index_right"]) & (pairs["a_right"] < pairs["a_left"])]
    geoms = g.geometry.to_numpy().copy()
    lost = np.zeros(len(g))
    for left, grp in pairs.groupby(level=0):
        smaller = shapely.union_all(geoms[grp["index_right"].to_numpy()])
        new = shapely.difference(geoms[left], smaller)
        lost[left] = 1 - (shapely.area(new) / a[left] if a[left] > 0 else 0)
        geoms[left] = new
    return gpd.GeoSeries(geoms, index=g.index, crs=g.crs), lost


def strip_like(geom_series) -> np.ndarray:
    """True for long thin polygons (mean width under ``STRIP_MAX_WIDTH_M``, length over ``STRIP_MIN_LENGTH_M``)."""
    area = geom_series.area.to_numpy()
    per = geom_series.length.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        width = np.where(per > 0, 2 * area / per, 0)
        length = np.where(width > 0, area / width, 0)
    return (width < STRIP_MAX_WIDTH_M) & (length > STRIP_MIN_LENGTH_M) & (area > 0)


def geometry_stage(fields, aoi_id: int | None = None):
    """Phase A on the fields of one AOI (lon/lat GeoDataFrame as ``field_rice.load_fields`` returns).

    Returns a GeoDataFrame in lon/lat with the original columns plus ``field_id`` (positional,
    as ``field_rice.label_aoi`` assigns it), ``refine_flag``, ``is_field``, ``area_acres`` (UTM,
    non-overlapping) and ``area_acres_delivered`` (the delineation's own value, for the record).
    """
    import geopandas as gpd

    g = fields.reset_index(drop=True).copy()
    if "field_id" not in g:
        g.insert(0, "field_id", [f"aoi{aoi_id}_{i:06d}" for i in range(len(g))])
    g["area_acres_delivered"] = g["area_acres"] if "area_acres" in g else np.nan
    utm = _utm(g)
    m = g.to_crs(utm)
    m["geometry"] = m.geometry.make_valid().simplify(SIMPLIFY_M, preserve_topology=True)
    a0 = m.geometry.area.to_numpy()
    geoms, lost = cut_overlaps(m)
    m["geometry"] = geoms
    a1 = m.geometry.area.to_numpy()
    flag = np.full(len(m), "", dtype=object)
    monster = lost >= MONSTER_COVER
    remainder_strip = strip_like(m.geometry)
    drop = monster & ((a1 < MIN_REMAINDER_PX * PIXEL_M2) | remainder_strip)
    flag[monster & ~drop] = "monster_cut"
    flag[drop] = "monster_dropped"
    strip = strip_like(m.geometry) & ~monster
    flag[strip] = "strip"
    m["refine_flag"] = flag
    m["overlap_lost_share"] = np.round(lost, 3)
    m["is_field"] = ~(drop | strip)
    m["area_acres"] = np.round(a1 / SQM_PER_ACRE, 4)
    m["area_acres_original"] = np.round(a0 / SQM_PER_ACRE, 4)
    out = m.to_crs(4326)
    # dropped monsters are kept as records (empty geometry would break readers): keep the geometry,
    # is_field False and the flag say what happened; they get no label acres in the delivery
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326)


def merge_tiny(g, label_col: str = "label"):
    """Phase B, part 1: tiny polygons inside same-label neighbours merge into the largest neighbour.

    ``g`` in a metric CRS. Returns (GeoDataFrame, merged_into Series: field_id of the absorber or "").
    """
    import geopandas as gpd
    import shapely

    g = g.copy()
    a = g.geometry.area.to_numpy()
    tiny = (a < TINY_PX * PIXEL_M2) & g["is_field"].to_numpy()
    merged_into = pd.Series("", index=g.index, dtype=object)
    if not tiny.any():
        return g, merged_into
    idx = gpd.GeoDataFrame({"lab": g[label_col].to_numpy(), "a": a, "fid": g["field_id"].to_numpy(),
                            "is_field": g["is_field"].to_numpy()}, geometry=g.geometry.to_numpy(), crs=g.crs)
    # sjoin suffixes only the columns both sides share: lab -> lab_left / lab_right, the rest keep their names
    touch = gpd.sjoin(idx[tiny][["geometry", "lab"]], idx[["geometry", "lab", "a", "fid", "is_field"]],
                      predicate="dwithin", distance=1.0, how="inner")
    touch = touch[(touch.index != touch["index_right"]) & touch["is_field"]]
    geoms = g.geometry.to_numpy().copy()
    for left, grp in touch.groupby(level=0):
        if not (grp["lab_right"] == grp["lab_left"]).all():
            continue
        best = grp.sort_values("a", ascending=False).iloc[0]
        right = int(best["index_right"])
        if merged_into.iloc[right] != "":
            continue                                   # the absorber was itself merged away
        geoms[right] = shapely.union(geoms[right], geoms[left])
        merged_into.iloc[left] = best["fid"]
    g["geometry"] = gpd.GeoSeries(geoms, index=g.index, crs=g.crs)
    gone = merged_into != ""
    g.loc[gone, "is_field"] = False
    g.loc[gone, "refine_flag"] = "merged"
    g["merged_into"] = merged_into
    return g, merged_into


def flag_ponds(g, water_share: np.ndarray):
    """Phase B, part 2: ``water_share`` per polygon (share of its pixels that were water all season)."""
    g = g.copy()
    pond = (np.asarray(water_share) >= POND_SHARE) & g["is_field"].to_numpy()
    g.loc[pond, "is_field"] = False
    g.loc[pond, "refine_flag"] = "pond"
    return g


def label_stage(fields_labelled, water_share=None):
    """Phase B on labelled fields (lon/lat, with ``label``, ``is_field``, ``refine_flag``)."""
    m = fields_labelled.to_crs(_utm(fields_labelled))
    m, _ = merge_tiny(m)
    if water_share is not None:
        m = flag_ponds(m, water_share)
    m["area_acres"] = np.round(m.geometry.area / SQM_PER_ACRE, 4)
    return m.to_crs(4326)


def summary(g) -> dict:
    a = g["area_acres"]
    out = {"polygons": len(g), "fields": int(g["is_field"].sum()),
           "acres_delivered_attribute": round(float(g["area_acres_delivered"].sum()), 1) if "area_acres_delivered" in g else None,
           "acres_utm_original": round(float(g["area_acres_original"].sum()), 1) if "area_acres_original" in g else None,
           "acres_refined_fields": round(float(a[g["is_field"]].sum()), 1)}
    for k, n in g["refine_flag"].value_counts().items():
        if k:
            out[f"{k}_n"] = int(n)
            out[f"{k}_acres"] = round(float(a[g["refine_flag"] == k].sum()), 1)
    return out


def run_geometry(aoi_ids, out_dir=OUT) -> pd.DataFrame:
    from . import field_rice

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for aoi_id in aoi_ids:
        fields = field_rice.load_fields(aoi_id)
        if fields.empty:
            rows.append({"aoi": f"aoi{aoi_id}", "polygons": 0})
            continue
        g = geometry_stage(fields, aoi_id)
        path = Path(out_dir) / f"aoi{aoi_id}_delineation_refined.gpkg"
        path.unlink(missing_ok=True)
        g.to_file(path, layer="fields", driver="GPKG")
        rows.append({"aoi": f"aoi{aoi_id}", **summary(g)})
    t = pd.DataFrame(rows)
    t.to_csv(Path(out_dir) / "geometry_stage_summary.csv", mode="a",
             header=not (Path(out_dir) / "geometry_stage_summary.csv").exists(), index=False)
    return t


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.field_refine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["geometry"])
    p.add_argument("--ids", nargs="+", type=int, required=True)
    args = p.parse_args(argv)
    print(run_geometry(args.ids).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
