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
* :func:`label_stage` (after labelling): crumbs under 4 pixels merge into the neighbour they share
  most outline with, small same-label pieces (under 0.5 ac, mostly enclosed by one field at least
  twice their size) merge into that field, and polygons that are water all season are flagged
  ``pond``. Flagged polygons stay in the file with ``is_field = False`` and a ``refine_flag`` saying
  why, so nothing is silently deleted (the delivery copies fields, strips and ponds only).

Geometry hygiene throughout (user review of aoi116): only valid polygons (never lines or
geometry collections), thin or tiny holes left by cutting a neighbour out are filled, the
tail-removing opening is clipped to the original outline (its mitred corners spiked into
neighbours), a polygon cut into pieces keeps one outline per field (other pieces become fields of
their own or are dropped as crumbs), duplicate outlines are cut to nothing. :func:`audit` measures
all of this on a delivered file.

Use::

    python -m sar_pipeline.analysis.field_refine geometry --ids 40 116     # -> fields_refined/aoi<N>_delineation_refined.gpkg
    python -m sar_pipeline.analysis.field_refine audit --ids 116           # hygiene numbers of the delivered fields file
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
#: What is left of a cut monster must be at least this many pixels, not a strip, and in at most
#: ``MAX_REMAINDER_PARTS`` pieces to stay: a remainder scattered in many small patches between the
#: fields is the bund network and the unsegmented gaps, not a field (a 49-ac outline around dozens
#: of fields left 6 ac in such patches).
MIN_REMAINDER_PX = 4
MAX_REMAINDER_PARTS = 3
#: A strip: nothing of the polygon survives an erosion of half ``STRIP_MAX_WIDTH_M`` (no part is
#: wider than 15 m) and its longest extent is over ``STRIP_MIN_LENGTH_M``. Why erosion and not
#: 2 x area / perimeter: the traced outlines are so irregular that a 95 x 98 m field had a
#: perimeter of 560 m and a "mean width" of 17 m. Why 15 m: fields here are 20-100 m wide; roads,
#: canals and tree lines are 5-15 m. Real narrow terraces exist (one AOI in the south-west);
#: their length is usually under 150 m.
STRIP_MAX_WIDTH_M = 15.0
STRIP_MIN_LENGTH_M = 150.0
#: Tails: parts of a polygon narrower than twice this are removed by a morphological opening
#: (erode then dilate). The delineation runs field outlines out along roads and canals; 3 m
#: removes those 6 m tails and the bund networks left when a monster outline is cut, and leaves
#: every real field (>= 20 m wide) as it was.
TAIL_RADIUS_M = 3.0
#: Simplification tolerance for the traced outlines (far below the 10 m pixel).
SIMPLIFY_M = 0.5
#: A polygon under this many pixels is a crumb of the segmentation, not a field: it merges into the
#: neighbour it shares most boundary with (whatever that neighbour's label: 1-3 pixels cannot carry
#: a label of their own), or is dropped when nothing touches it.
TINY_PX = 4
#: A small piece of a field (under ``SMALL_PIECE_M2``) that shares at least ``PIECE_SHARED_MIN`` of its
#: outline with ONE same-label neighbour at least ``PIECE_RATIO_MIN`` times its size is part of that
#: field: the segmentation cut a corner or a strip off it (user review, aoi116).
SMALL_PIECE_M2 = 0.5 * 4046.8564224
PIECE_SHARED_MIN = 0.5
PIECE_RATIO_MIN = 2.0
#: Holes narrower than twice this (a line left where an overlapping neighbour was cut out) or
#: smaller than ``TINY_PX`` pixels are filled; wider holes (a pond, a house) stay.
HOLE_THIN_M = 3.0
#: Merging closes gaps narrower than twice this between a piece and its absorber.
MERGE_GAP_M = 0.6
#: A polygon is a pond when this share of its pixels was water all season.
POND_SHARE = 0.6


def _utm(g):
    return g.estimate_utm_crs()


def polygonal(geoms):
    """Valid polygons only: set operations on traced outlines leave line and point crumbs and
    geometry collections behind, which no reader treats as a field. Empty where nothing is left."""
    import shapely

    geoms = shapely.make_valid(np.asarray(geoms, dtype=object))
    out = np.empty(len(geoms), dtype=object)
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            out[i] = shapely.Polygon()
            continue
        parts = [q for q in shapely.get_parts(g) if q.geom_type == "Polygon" and q.area > 0]
        out[i] = shapely.Polygon() if not parts else (parts[0] if len(parts) == 1 else shapely.MultiPolygon(parts))
    return out


def robust(op, a, b):
    """``op(a, b)`` (a shapely set operation) that survives the broken topology of traced outlines.

    GEOS raises "side location conflict" / "non-noded intersection" / "ring edge missing" on
    outlines that ``make_valid`` accepts. First retry: both operands rebuilt by a zero buffer.
    Second: both snapped to a 1 cm grid (noding on a grid is what GEOS recommends for such
    inputs). Third: to a 10 cm grid. Why not always snap: a snapped operand can move an edge by
    up to half the grid, so it is done only where needed.
    """
    import shapely

    try:
        return op(a, b)
    except shapely.errors.GEOSException:
        pass
    try:
        return op(shapely.buffer(a, 0), shapely.buffer(b, 0))
    except shapely.errors.GEOSException:
        pass
    for grid in (0.01, 0.1):
        try:
            return op(shapely.make_valid(shapely.set_precision(a, grid)), shapely.make_valid(shapely.set_precision(b, grid)))
        except shapely.errors.GEOSException:
            continue
    raise


def clean(geoms, grid_m: float = 0.01):
    """Snap coordinates to a 1 cm grid and keep valid polygons: removes the near-duplicate vertices
    and hairline self-touches that reprojection and set operations leave (an "invalid" outline in
    QGIS) without moving any edge visibly."""
    import shapely

    return polygonal(shapely.set_precision(shapely.make_valid(np.asarray(geoms, dtype=object)), grid_m))


def fill_small_holes(geoms, thin_m: float = HOLE_THIN_M, min_m2: float = TINY_PX * PIXEL_M2):
    """Fill interior rings that are thin (nothing survives an erosion of ``thin_m``) or under
    ``min_m2``: the lines and specks that cutting an overlapping neighbour out of a field leaves.
    Returns (geometries, number of holes filled per polygon)."""
    import shapely

    out = np.asarray(geoms, dtype=object).copy()
    filled = np.zeros(len(out), dtype=int)
    for i, g in enumerate(out):
        if g is None or g.is_empty or shapely.get_num_interior_rings(g) == 0 and g.geom_type == "Polygon":
            continue
        polys = []
        for q in shapely.get_parts(g):
            if q.geom_type != "Polygon":
                continue
            keep = []
            for ring in q.interiors:
                hole = shapely.Polygon(ring)
                if hole.area < min_m2 or shapely.buffer(hole, -thin_m).is_empty:
                    filled[i] += 1
                else:
                    keep.append(ring)
            polys.append(shapely.Polygon(q.exterior, keep))
        out[i] = polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)
    return out, filled


def split_parts(m, min_m2: float = TINY_PX * PIXEL_M2):
    """One outline per field: a polygon cut into pieces keeps its largest piece; other pieces of
    at least ``min_m2`` become fields of their own (``field_id`` + a letter, flag ``split_part``),
    smaller ones are crumbs and are dropped (their share is recorded in ``crumbs_dropped_share``).
    ``m`` in a metric CRS. Returns the frame with the new rows appended."""
    import geopandas as gpd
    import shapely

    m = m.copy()
    geoms = m.geometry.to_numpy().copy()
    dropped = np.zeros(len(m))
    new_rows = []
    is_field = m["is_field"].to_numpy() if "is_field" in m else np.ones(len(m), dtype=bool)
    for i, g in enumerate(geoms):
        parts = [q for q in shapely.get_parts(g) if q.geom_type == "Polygon" and q.area > 0]
        if len(parts) <= 1 or not is_field[i]:         # flagged records keep their outline as it is
            continue
        parts.sort(key=lambda q: q.area, reverse=True)
        total = sum(q.area for q in parts)
        geoms[i] = parts[0]
        letters = "bcdefghijklmnopqrstuvwxyz"
        n_new = 0
        for q in parts[1:]:
            if q.area >= min_m2 and n_new < len(letters):
                row = m.iloc[i].copy()
                row["geometry"] = q
                row["field_id"] = f"{m['field_id'].iloc[i]}{letters[n_new]}"
                new_rows.append(row)
                n_new += 1
            else:
                dropped[i] += q.area / total
    m["geometry"] = gpd.GeoSeries(geoms, index=m.index, crs=m.crs)
    m["crumbs_dropped_share"] = np.round(dropped, 3)
    if new_rows:
        extra = gpd.GeoDataFrame(new_rows, crs=m.crs)
        extra["refine_flag"] = "split_part"
        extra["crumbs_dropped_share"] = 0.0
        m = gpd.GeoDataFrame(pd.concat([m, extra], ignore_index=True), geometry="geometry", crs=m.crs)
    return m


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
    # the smaller polygon wins; of two identical outlines (the delineation holds duplicates) the
    # later one wins, so the earlier one is cut to nothing and dropped as a crumb
    pairs = pairs[(pairs.index != pairs["index_right"])
                  & ((pairs["a_right"] < pairs["a_left"])
                     | ((pairs["a_right"] == pairs["a_left"]) & (pairs["index_right"] > pairs.index)))]
    geoms = g.geometry.to_numpy().copy()
    lost = np.zeros(len(g))
    for left, grp in pairs.groupby(level=0):
        try:
            smaller = shapely.union_all(geoms[grp["index_right"].to_numpy()])
        except shapely.errors.GEOSException:
            smaller = shapely.union_all(shapely.make_valid(shapely.set_precision(geoms[grp["index_right"].to_numpy()], 0.01)))
        new = robust(shapely.difference, geoms[left], smaller)
        lost[left] = 1 - (shapely.area(new) / a[left] if a[left] > 0 else 0)
        geoms[left] = new
    return gpd.GeoSeries(polygonal(geoms), index=g.index, crs=g.crs), lost


def strip_like(geom_series) -> np.ndarray:
    """True for long thin polygons: no part wider than ``STRIP_MAX_WIDTH_M`` (erosion by half of it
    leaves nothing) and a longest extent over ``STRIP_MIN_LENGTH_M``."""
    import shapely

    geoms = geom_series.to_numpy()
    area = shapely.area(geoms)
    core = shapely.buffer(geoms, -STRIP_MAX_WIDTH_M / 2, quad_segs=2, join_style="mitre")
    thin = shapely.is_empty(core) | (shapely.area(core) <= 0)
    mrr = shapely.minimum_rotated_rectangle(geoms)
    # the longest side of the minimum rotated rectangle, from its corner coordinates
    length = np.zeros(len(geoms))
    for i, r in enumerate(mrr):
        if r.is_empty or r.geom_type != "Polygon":
            continue
        c = np.asarray(r.exterior.coords)[:3]
        length[i] = max(np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1])))
    return thin & (length > STRIP_MIN_LENGTH_M) & (area > 0)


def remove_tails(geom_series, radius: float = TAIL_RADIUS_M):
    """Morphological opening: parts narrower than ``2 * radius`` disappear. Returns (geometries,
    share of area removed); an empty result means the polygon was nothing but tails."""
    import shapely

    geoms = geom_series.to_numpy()
    opened = shapely.buffer(shapely.buffer(geoms, -radius, quad_segs=2, join_style="mitre"), radius,
                            quad_segs=2, join_style="mitre")
    # the dilation's mitred corners can spike past the original outline (into a neighbour); an
    # opening is contained in its input by definition, so clip it back
    opened = polygonal([robust(shapely.intersection, o, g) for o, g in zip(opened, geoms)])
    a0 = shapely.area(geoms)
    with np.errstate(divide="ignore", invalid="ignore"):
        removed = np.where(a0 > 0, 1 - shapely.area(opened) / a0, 0.0)
    return opened, np.clip(removed, 0, 1)


def geometry_stage(fields, aoi_id: int | None = None):
    """Phase A on the fields of one AOI (lon/lat GeoDataFrame as ``field_rice.load_fields`` returns).

    Returns a GeoDataFrame in lon/lat with the original columns plus ``field_id`` (positional,
    as ``field_rice.label_aoi`` assigns it), ``refine_flag``, ``is_field``, ``area_acres`` (UTM,
    non-overlapping) and ``area_acres_delivered`` (the delineation's own value, for the record).
    """
    import geopandas as gpd
    import shapely

    g = fields.reset_index(drop=True).copy()
    if "field_id" not in g:
        g.insert(0, "field_id", [f"aoi{aoi_id}_{i:06d}" for i in range(len(g))])
    g["area_acres_delivered"] = g["area_acres"] if "area_acres" in g else np.nan
    utm = _utm(g)
    m = g.to_crs(utm)
    m["geometry"] = gpd.GeoSeries(polygonal(m.geometry.make_valid().simplify(SIMPLIFY_M, preserve_topology=True)),
                                  index=m.index, crs=m.crs)
    a0 = m.geometry.area.to_numpy()
    geoms, lost = cut_overlaps(m)
    filled_geoms, holes_filled = fill_small_holes(geoms.to_numpy())
    geoms = gpd.GeoSeries(filled_geoms, index=m.index, crs=m.crs)
    strip_before = strip_like(geoms)                 # judged before the opening: a 6 m canal is a strip, not "nothing"
    opened, tail_share = remove_tails(geoms)
    # polygons under 4 pixels are crumbs for the label stage (merge_pieces), not fields with tails:
    # the opening would erase them and lose the merge
    tiny = geoms.area.to_numpy() < TINY_PX * PIXEL_M2
    opened[tiny], tail_share[tiny] = geoms.to_numpy()[tiny], 0.0
    m["geometry"] = gpd.GeoSeries(opened, index=m.index, crs=m.crs)
    m["holes_filled"] = holes_filled
    # a monster outline cut around many fields is left in scattered patches: judge the number of
    # pieces before they are split into fields of their own
    parts = shapely.get_num_geometries(m.geometry.to_numpy())
    a1 = m.geometry.area.to_numpy()
    flag = np.full(len(m), "", dtype=object)
    monster = lost >= MONSTER_COVER
    thin = strip_before | (a1 <= 0)
    drop = monster & ((a1 < MIN_REMAINDER_PX * PIXEL_M2) | thin | (parts > MAX_REMAINDER_PARTS))
    flag[monster & ~drop] = "monster_cut"
    flag[drop] = "monster_dropped"
    strip = thin & ~monster
    flag[strip] = "strip"
    m["refine_flag"] = flag
    m["overlap_lost_share"] = np.round(lost, 3)
    m["tail_removed_share"] = np.round(tail_share, 3)
    m["is_field"] = ~(drop | strip)
    m["area_acres_original"] = np.round(a0 / SQM_PER_ACRE, 4)
    # a polygon that is nothing but tails (or a duplicate cut to nothing) has no geometry left:
    # keep the raw outline for the record, flagged, not a field
    empty = m.geometry.is_empty.to_numpy()
    m.loc[empty, "geometry"] = geoms[empty]
    still_empty = m.geometry.is_empty.to_numpy()
    m.loc[still_empty, "geometry"] = fields.to_crs(utm).geometry.to_numpy()[still_empty]
    m.loc[empty & (flag == ""), "refine_flag"] = "crumb"
    m.loc[empty, "is_field"] = False
    # one outline per field: pieces of a cut polygon become fields of their own or crumbs (after
    # the clean-up, which can itself split a self-touching outline into pieces)
    m["geometry"] = gpd.GeoSeries(clean(m.geometry.to_numpy()), index=m.index, crs=m.crs)
    m = split_parts(m)
    m["area_acres"] = np.round(m.geometry.area.to_numpy() / SQM_PER_ACRE, 4)
    out = m.to_crs(4326)
    # dropped monsters are kept as records (empty geometry would break readers): keep the geometry,
    # is_field False and the flag say what happened; they get no label acres in the delivery
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326)


def shared_outline(geom, other, tol: float = 1.0) -> float:
    """Length of ``geom``'s outline that runs within ``tol`` metres of ``other``."""
    import shapely

    return float(shapely.length(robust(shapely.intersection, geom.boundary, shapely.buffer(other, tol))))


def merge_pieces(g, label_col: str = "label", passes: int = 3):
    """Phase B, part 1: crumbs and small same-label pieces merge into the field they belong to.

    ``g`` in a metric CRS. Returns (GeoDataFrame, merged_into Series: field_id of the absorber or "").
    * a polygon under ``TINY_PX`` pixels joins the neighbour it shares most outline with, whatever
      that neighbour's label (a crumb's own 1-3 pixels are no label); with no neighbour it is a
      ``crumb`` and is not delivered;
    * a polygon under ``SMALL_PIECE_M2`` joins a same-label neighbour that holds at least
      ``PIECE_SHARED_MIN`` of its outline and is at least ``PIECE_RATIO_MIN`` times larger.
    Merging closes the hairline gap between the two outlines and never enters a third polygon.
    Repeated up to ``passes`` times, as one merge can make the next possible.
    """
    import geopandas as gpd
    import shapely

    g = g.copy()
    if "merged_into" not in g:
        g["merged_into"] = ""
    merged_into = pd.Series(g["merged_into"].to_numpy().astype(object), index=g.index)
    geoms = g.geometry.to_numpy().copy()
    labels = g[label_col].to_numpy()
    fids = g["field_id"].to_numpy()
    flags = g["refine_flag"].to_numpy().astype(object)
    active = g["is_field"].to_numpy().copy()
    for _ in range(passes):
        area = shapely.area(geoms)
        tree = shapely.STRtree(geoms)
        order = np.argsort(area)                       # smallest first: a crumb joins before its absorber moves
        changed = False
        for i in order:
            if not active[i]:
                continue
            tiny = area[i] < TINY_PX * PIXEL_M2
            small = area[i] < SMALL_PIECE_M2
            if not (tiny or small):
                break
            near = [j for j in tree.query(shapely.buffer(geoms[i], 1.0), predicate="intersects") if j != i and active[j]]
            if not near:
                if tiny:
                    active[i] = False
                    flags[i] = "crumb"
                    changed = True
                continue
            shared = {j: shared_outline(geoms[i], geoms[j]) for j in near}
            perimeter = max(shapely.length(geoms[i]), 1e-9)
            j = max(shared, key=shared.get)
            if tiny:
                if shared[j] <= 0:
                    continue
            else:
                same = [k for k in near if labels[k] == labels[i] and area[k] >= PIECE_RATIO_MIN * area[i]
                        and shared[k] >= PIECE_SHARED_MIN * perimeter]
                if not same:
                    continue
                j = max(same, key=shared.get)
            others = [k for k in near if k != j]
            joined = robust(shapely.union, geoms[j], geoms[i])
            joined = shapely.buffer(shapely.buffer(joined, MERGE_GAP_M, join_style="mitre"), -MERGE_GAP_M, join_style="mitre")
            if others:
                block = robust(lambda x, y: shapely.union_all([x, y]), shapely.union_all(shapely.buffer([geoms[k] for k in others], 0)),
                               shapely.Polygon())
                joined = robust(shapely.difference, joined, block)
            joined, _ = fill_small_holes(polygonal([joined]))
            joined = joined[0]
            if joined.is_empty or joined.geom_type != "Polygon":
                if tiny:                                   # not attached after all: a crumb on its own
                    active[i] = False
                    flags[i] = "crumb"
                    changed = True
                continue
            geoms[j] = joined
            active[i] = False
            flags[i] = "merged"
            merged_into.iloc[i] = fids[j]
            changed = True
        if not changed:
            break
    g["geometry"] = gpd.GeoSeries(geoms, index=g.index, crs=g.crs)
    g["is_field"] = active
    g["refine_flag"] = flags
    g["merged_into"] = merged_into
    return g, merged_into


merge_tiny = merge_pieces        # the name the first version used


def flag_ponds(g, water_share: np.ndarray):
    """Phase B, part 2: ``water_share`` per polygon (share of its pixels that were water all season)."""
    g = g.copy()
    pond = (np.asarray(water_share) >= POND_SHARE) & g["is_field"].to_numpy()
    g.loc[pond, "is_field"] = False
    g.loc[pond, "refine_flag"] = "pond"
    return g


def label_stage(fields_labelled, water_share=None):
    """Phase B on labelled fields (lon/lat, with ``label``, ``is_field``, ``refine_flag``)."""
    import geopandas as gpd

    m = fields_labelled.to_crs(_utm(fields_labelled))
    m["geometry"] = gpd.GeoSeries(polygonal(m.geometry.to_numpy()), index=m.index, crs=m.crs)
    m, _ = merge_pieces(m)
    if water_share is not None:
        m = flag_ponds(m, water_share)
    m["geometry"] = gpd.GeoSeries(clean(m.geometry.to_numpy()), index=m.index, crs=m.crs)
    m = split_parts(m)
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


def examples(aoi_id: int, n: int = 2, half_m: float = 150, out_dir=OUT) -> list:
    """Before / after crops on the latest clear true-colour image for the largest polygon of each
    refine flag: raw outlines (yellow, the flagged one magenta) beside the refined outlines (cyan).
    Why: the refinement must be checked by eye before it is trusted, like every other step."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.windows import from_bounds
    from shapely.geometry import box

    from ..optical_export import band_index
    from ..review import _stretch, latest_clear
    from . import field_rice

    raw = field_rice.load_fields(aoi_id, refined_dir=None)
    ref = gpd.read_file(field_rice.refined_path(aoi_id, out_dir))
    path, date, _ = latest_clear(aoi_id)
    made = []
    with rasterio.open(path) as ds:
        raw_m, ref_m = raw.to_crs(ds.crs), ref.to_crs(ds.crs)
        raw_m["field_id"] = ref_m["field_id"].to_numpy()
        for flag in [f for f in ref_m["refine_flag"].unique() if f]:
            pick = ref_m[ref_m["refine_flag"] == flag].sort_values("area_acres_original", ascending=False).head(n)
            for _, row in pick.iterrows():
                geom = raw_m.loc[raw_m["field_id"] == row["field_id"]].geometry.iloc[0]
                c = geom.centroid
                minx, miny, maxx, maxy = geom.bounds
                r = max(half_m, (maxx - minx) / 2 + 30, (maxy - miny) / 2 + 30)
                bx = (c.x - r, c.y - r, c.x + r, c.y + r)
                win = from_bounds(*bx, transform=ds.transform)
                img = np.dstack([_stretch(ds.read(band_index(ds, b), window=win, boundless=True, fill_value=0)
                                          .astype("float32")) for b in ("B4", "B3", "B2")])
                fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
                for ax, frame, title in ((axes[0], raw_m, "raw delineation"), (axes[1], ref_m[ref_m["is_field"]], "refined (fields only)")):
                    ax.imshow(img, extent=(bx[0], bx[2], bx[1], bx[3]))
                    near = frame[frame.intersects(box(*bx))]
                    near.boundary.plot(ax=ax, color="yellow" if title.startswith("raw") else "cyan", linewidth=0.7)
                    if title.startswith("raw"):
                        gpd.GeoSeries([geom], crs=ds.crs).boundary.plot(ax=ax, color="magenta", linewidth=1.8)
                    ax.set_xlim(bx[0], bx[2]), ax.set_ylim(bx[1], bx[3]), ax.set_xticks([]), ax.set_yticks([])
                    ax.set_title(title, fontsize=9)
                fig.suptitle(f"{flag}: {row['field_id']} ({row['area_acres_original']:.2f} ac raw -> "
                             f"{row['area_acres']:.2f} ac, is_field={row['is_field']}); S2 {date.date()}", fontsize=9)
                out = Path(out_dir) / "examples"
                out.mkdir(parents=True, exist_ok=True)
                f = out / f"aoi{aoi_id}_{flag}_{row['field_id']}.png"
                fig.savefig(f, dpi=90, bbox_inches="tight")
                plt.close(fig)
                made.append(f)
    return made


def audit(path, layer=None) -> dict:
    """Geometry hygiene of one fields file: what a reviewer sees in QGIS, as numbers.

    Overlaps are measured in the local UTM zone on the polygons as written (lon/lat rounding
    makes hairline overlaps of a few m2 unavoidable; pairs over ``PIXEL_M2`` are real).
    """
    import geopandas as gpd
    import shapely

    g = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    m = g.to_crs(_utm(g))
    geoms = m.geometry.to_numpy()
    valid = shapely.is_valid(geoms)
    types = pd.Series(shapely.get_type_id(geoms)).value_counts().to_dict()
    area = shapely.area(shapely.make_valid(geoms))
    holes_thin = holes_tiny = holes_wide = 0
    for gm in shapely.make_valid(geoms):
        for q in shapely.get_parts(gm):
            if q.geom_type != "Polygon":
                continue
            for ring in q.interiors:
                hole = shapely.Polygon(ring)
                if shapely.buffer(hole, -HOLE_THIN_M).is_empty:
                    holes_thin += 1
                elif hole.area < TINY_PX * PIXEL_M2:
                    holes_tiny += 1
                else:
                    holes_wide += 1
    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    inter = shapely.area(shapely.intersection(shapely.make_valid(geoms[left]), shapely.make_valid(geoms[right])))
    labels = m["label"].to_numpy() if "label" in m else np.zeros(len(m))
    # small same-label pieces mostly enclosed by one larger same-label neighbour (what merge_pieces removes)
    enclosed = 0
    for i in np.flatnonzero(area < SMALL_PIECE_M2):
        near = [j for j in tree.query(shapely.buffer(geoms[i], 1.0), predicate="intersects") if j != i]
        per = max(shapely.length(geoms[i]), 1e-9)
        if any(labels[j] == labels[i] and area[j] >= PIECE_RATIO_MIN * area[i]
               and shared_outline(geoms[i], geoms[j]) >= PIECE_SHARED_MIN * per for j in near):
            enclosed += 1
    return {"polygons": len(m), "invalid": int((~valid).sum()),
            "non_polygon_features": int(sum(n for t, n in types.items() if t not in (3, 6))),
            "multipart": int((shapely.get_num_geometries(geoms) > 1).sum()),
            "under_4px": int((area < TINY_PX * PIXEL_M2).sum()),
            "holes_thin": holes_thin, "holes_tiny": holes_tiny, "holes_wide": holes_wide,
            "overlap_pairs_over_1m2": int((inter > 1).sum()), "overlap_pairs_over_1px": int((inter > PIXEL_M2).sum()),
            "overlap_m2_total": round(float(inter[inter > 1].sum()), 1),
            "small_same_label_pieces_enclosed": enclosed}


def review_crops(aoi_id: int, new_path, prev_path=None, field_ids=(), n_random: int = 6, half_m: float = 120,
                 seed: int = 0, out_dir=f"{OUT}/review") -> list:
    """Crops of the latest clear true-colour image with the delivered outlines drawn on it: the
    new file (cyan, fields filled by class colour at 35 %) beside the previous one (yellow), around
    the named fields and ``n_random`` random fields. Why: the numbers of :func:`audit` say the
    geometry is clean; only a look says the fields look like fields."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.patches import Polygon as MplPolygon
    from rasterio.windows import from_bounds
    from shapely.geometry import box

    from ..optical_export import band_index
    from ..review import _stretch, latest_clear

    colours = {0: "#a070c0", 1: "#60b0e0", 2: "#40d0c0", 3: "#e0b060", 4: "#8060c0", 5: "#a0c060", 6: "#e070c0",
               7: "#3c78c8", 8: "#c8aa78"}
    new = gpd.read_file(new_path)
    prev = gpd.read_file(prev_path) if prev_path else None
    path, date, _ = latest_clear(aoi_id)
    rng = np.random.default_rng(seed)
    made = []
    with rasterio.open(path) as ds:
        new_m = new.to_crs(ds.crs)
        prev_m = prev.to_crs(ds.crs) if prev is not None else None
        known = set(new_m["field_id"]) | (set(prev_m["field_id"]) if prev_m is not None else set())
        picks = [f for f in field_ids if f in known]
        pool = new_m[new_m["is_field"]] if "is_field" in new_m else new_m
        picks += list(pool["field_id"].sample(n_random, random_state=int(rng.integers(1 << 30))))
        for fid in picks:
            src = new_m if fid in set(new_m["field_id"]) else prev_m
            c = src.loc[src["field_id"] == fid].geometry.iloc[0].centroid
            bx = (c.x - half_m, c.y - half_m, c.x + half_m, c.y + half_m)
            win = from_bounds(*bx, transform=ds.transform)
            img = np.dstack([_stretch(ds.read(band_index(ds, b), window=win, boundless=True, fill_value=0)
                                      .astype("float32")) for b in ("B4", "B3", "B2")])
            panels = [("new", new_m, "cyan")] + ([("previous", prev_m, "yellow")] if prev_m is not None else [])
            fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6), squeeze=False)
            for ax, (title, frame, colour) in zip(axes[0], panels):
                ax.imshow(img, extent=(bx[0], bx[2], bx[1], bx[3]))
                near = frame[frame.intersects(box(*bx))]
                for geom, lab in zip(near.geometry, near["label"] if "label" in near else [1] * len(near)):
                    for part in getattr(geom, "geoms", [geom]):
                        if part.geom_type != "Polygon":        # the previous files hold line crumbs
                            continue
                        ax.add_patch(MplPolygon(np.asarray(part.exterior.coords), closed=True, facecolor=colours.get(int(lab), "grey"),
                                                edgecolor=colour, linewidth=0.8, alpha=0.35))
                        for ring in part.interiors:
                            ax.add_patch(MplPolygon(np.asarray(ring.coords), closed=True, facecolor="white", edgecolor="red",
                                                    linewidth=0.8, alpha=0.6))
                ax.set_xlim(bx[0], bx[2]), ax.set_ylim(bx[1], bx[3]), ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(f"{title}: {len(near)} polygons", fontsize=9)
            fig.suptitle(f"{fid}; S2 {date.date()}; holes drawn white/red", fontsize=9)
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            f = out / f"aoi{aoi_id}_{fid}.png"
            fig.savefig(f, dpi=80, bbox_inches="tight")
            plt.close(fig)
            made.append(f)
    return made


def run_geometry(aoi_ids, out_dir=OUT) -> pd.DataFrame:
    from . import field_rice

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for aoi_id in aoi_ids:
        path = Path(out_dir) / f"aoi{aoi_id}_delineation_refined.gpkg"
        if path.exists():
            continue                                    # re-runnable after an interruption
        fields = field_rice.load_fields(aoi_id, refined_dir=None)      # always the raw delineation
        if fields.empty:
            rows.append({"aoi": f"aoi{aoi_id}", "polygons": 0})
            continue
        try:
            g = geometry_stage(fields, aoi_id)
        except Exception as exc:  # keep the batch going; the table says which AOI failed
            rows.append({"aoi": f"aoi{aoi_id}", "error": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"aoi{aoi_id}: FAILED {type(exc).__name__}: {exc}"[:200])
            continue
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
    p.add_argument("step", choices=["geometry", "examples", "audit"])
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--file", default=f"{SRC}/delivery/fields/aoi{{}}_fields_standing_rice_2026-09-21.gpkg",
                   help="audit: file pattern with {} for the AOI number")
    args = p.parse_args(argv)
    if args.step == "audit":
        rows = [{"aoi": f"aoi{a}", **audit(args.file.format(a))} for a in args.ids]
        print(pd.DataFrame(rows).to_string(index=False))
        return 0
    if args.step == "examples":
        for a in args.ids:
            print("\n".join(str(f) for f in examples(a)))
        return 0
    print(run_geometry(args.ids).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
