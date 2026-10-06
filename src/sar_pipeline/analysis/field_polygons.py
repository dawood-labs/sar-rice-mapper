"""Field polygons from the relative-rule class map (curve_rules): the delineation owns the geometry, the map the label.

Not to be confused with ``field_labels`` (the stage-10 / SAR-only field labelling with the field-level model): this is
the fresh-start version (2 Oct 2026) with multi-class cuts, in-field sliver absorption and the final tidy.

Why
---
The client receives fields, not pixels. User (2 Oct 2026): fields must carry the class of the map; where no delineated
field exists a polygon must be made; a delineated polygon holding two real fields with different labels must be cut
(aoi160_004059: 8.7 ac, 67 % direct-seeded rice and 29 % flooded); same-label polygons that touch, overlap or nest
must then be merged so boundary noise disappears. The method is ported from the user's earlier sugarcane project
(``cane_scanner``, ``scripts/label_field_polygons.py`` and ``FIELD_LABELLING.md``), generalised from two classes (cane
or not) to the seven classes of ``curve_rules.MAP_CLASSES``:

1. **Overlaps** between delineated polygons go to the SMALLER polygon (it keeps the finest tracing; the larger is cut
   around it), in two array operations (``_resolve_overlaps``).
2. **Label** each polygon from the pixels whose centre falls in it:
   * clean: one class holds at least ``CLEAN_SHARE`` -> that class;
   * fewer than ``MIN_PIXELS`` pixels: majority label, flagged "too small to judge";
   * mixed: try ONE straight cut along the polygon's own long axis or across it (field subdivisions run parallel to
     the field's edges), placed where the two sides separate best (purity = share of pixels in each side's own
     majority class). The cut is made only if it raises purity by ``MIN_CUT_GAIN`` AND both sides are still
     field-shaped (survive an erosion by half of ``MIN_SPLIT_WIDTH_M``: no needles shaved off an edge); otherwise the
     polygon keeps its majority label, flagged. Pieces are tried once more (``CUT_DEPTH``: three fields in one
     polygon).
3. **No polygon here**: pixels of a class that no delineated polygon covers are traced into polygons, opened by half of
   ``MIN_FIELD_WIDTH_M`` (strips the tentacles, keeps the core), kept from ``MIN_ORPHAN_ACRES`` up when field-shaped
   (rotated-rectangle fill and compactness gates) and simplified at half a pixel; traced polygons never cut a
   delineated one (they are clipped to the ground the delineation does not cover).
4. **Slivers**: pieces under a threshold (0.05 / 0.10 / 0.15 ac compared) join the neighbour they share the longest
   boundary with, a piece of the same field first; same-label pieces of one field become one. Different delineated
   fields are never dissolved together (user, 2 Oct: "no monster polygons, preserve the field boundaries"), and no two
   output polygons overlap (``overlap_acres`` = 0).

Every output polygon carries ``origin`` (delineation / split / derived from map), ``decision``, ``label_share`` and
``pixels``, plus after the merge how many delineated fields, split pieces and derived blocks went into it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

SQM_PER_ACRE = 4046.8564224
#: One class holding this share of a polygon's pixels: the polygon is that class, no cut tried (cane_scanner 0.85).
CLEAN_SHARE = 0.85
#: Fewer pixels than this and a share is not a measurement (cane_scanner).
MIN_PIXELS = 10
#: A cut must raise purity by this much, or it invents a boundary the ground does not have (cane_scanner).
MIN_CUT_GAIN = 0.08
#: Each side of a cut must survive an erosion by half this width (metres): no needle shaved off an edge.
MIN_SPLIT_WIDTH_M = 20.0
#: Cut a piece again at most this many times (two cuts = up to three fields in one polygon).
CUT_DEPTH = 2
#: Derived (no-polygon) blocks: opening radius = half this width, smallest kept area, shape gates, simplify tolerance.
MIN_FIELD_WIDTH_M = 14.0
#: A RICE polygon narrower than this (short side of its rotated bounding box) is a road / path strip, not a field.
#: Why (aoi19 QC, 6 Oct 2026): the reviewer deleted narrow strips on pathways and roads; below 15 m (1.5 pixels) this
#: caught 20 of them and 27 kept polygons (1.8 ac). None = off.
STRIP_MAX_WIDTH_M = None
STRIP_LABEL = 9
STRIP_FROM_LABELS = (1, 7, 3)
#: Judge a polygon as RICE (all rice classes together) against each other class. Why (user, aoi19 trial, 6 Oct): a
#: small tree inside a rice field turned the whole polygon into tree, because 35 % tree beat each rice class alone
#: (direct seeded / transplanted / young). Tree (or anything else) now wins only with more pixels than all rice
#: together; the piece keeps its most common rice class. Off = False.
RICE_GROUP_MAJORITY = False
#: A tree block on a rice field's edge is cut off with smaller limits than other cuts (MIN_SPLIT_WIDTH_M,
#: MIN_PIXELS): the reviewer still reshaped rice fields by hand to cut edge trees (aoi19 QC). None = off.
TREE_CUT_MIN_WIDTH_M = None
TREE_CUT_MIN_PIXELS = 4
TREE_LABEL = 6
MIN_ORPHAN_ACRES = 0.15
MIN_RECTANGULARITY = 0.45
MIN_COMPACTNESS = 0.18
SIMPLIFY_M = 5.0
#: Pieces smaller than this (m^2) after overlap arithmetic are topological slivers, not fields.
SLIVER_SQM = 10.0
#: Final tidy of every output polygon (user, 2 Oct, QGIS: slits and hooks left where absorbed slivers met their new
#: field): slits narrower than 2 x CLOSE_M are closed, holes under the sliver threshold filled, and spurs narrower than
#: 2 x DESPIKE_M cut off while the true edges and corners are kept (cane_scanner's despike: open, dilate a little past
#: the radius, intersect back with the original).
CLOSE_M = 1.0
DESPIKE_M = 4.0
#: Same-label polygons closer than this (metres) count as touching when merging.
MERGE_TOUCH_M = 0.5


# ---------------------------------------------------------------------------------------------------- geometry help
def _polygons_only(geoms):
    """Valid, single-part polygons from anything ``make_valid`` hands back (collections nest multipolygons)."""
    import shapely

    g = shapely.make_valid(np.asarray(geoms, dtype=object))
    for _ in range(5):
        g = shapely.get_parts(g)
        kinds = shapely.get_type_id(g)
        if not np.isin(kinds, (4, 5, 6, 7)).any():           # multi* / collection
            break
    g = g[shapely.get_type_id(g) == 3]                        # polygons
    return g[~shapely.is_empty(g)]


def _resolve_overlaps(geoms, priority):
    """Each overlapped patch goes to the polygon with the LOWER ``priority`` (cane_scanner): every intersecting pair
    from the index at once, each polygon cut once against the union of all better claimants touching it."""
    import shapely

    geoms = np.asarray(geoms, dtype=object)
    priority = np.asarray(priority, dtype=float)
    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    wins = (priority[right] < priority[left]) | ((priority[right] == priority[left]) & (right < left))
    left, right = left[wins], right[wins]
    if not left.size:
        return geoms
    order = np.argsort(left, kind="stable")
    left, right = left[order], right[order]
    targets, starts = np.unique(left, return_index=True)
    ends = np.append(starts[1:], left.size)
    cutters = np.array([shapely.union_all(geoms[right[a:b]]) for a, b in zip(starts, ends)], dtype=object)
    out = geoms.copy()
    out[targets] = shapely.difference(geoms[targets], cutters)
    return out


def field_like(geom, min_width: float) -> bool:
    """Still has a core once eroded by half ``min_width``: needles, ribbons and edge shavings vanish, fields keep one."""
    if geom.is_empty:
        return False
    core = geom.buffer(-min_width / 2.0)
    return (not core.is_empty) and core.area > 0.0


def _orientation(geom) -> float:
    """Angle of the polygon's long axis (its minimum rotated rectangle's longest edge)."""
    rect = geom.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return 0.0
    c = np.array(rect.exterior.coords[:4])
    e = np.diff(np.vstack([c, c[:1]]), axis=0)
    longest = e[int(np.argmax(np.hypot(e[:, 0], e[:, 1])))]
    return float(np.arctan2(longest[1], longest[0]))


def best_cut(xs, ys, cls, angles, n_classes: int, min_pixels: int = MIN_PIXELS):
    """(angle, offset, purity) of the single straight cut that best separates the classes, multi-class: purity = the
    share of pixels that fall in their own side's majority class (each side takes its own majority)."""
    best = (0.0, 0.0, 0.0)
    onehot = np.eye(n_classes, dtype=np.int32)[cls]
    n = len(cls)
    if n < 2 * min_pixels:
        return best
    for theta in angles:
        proj = xs * np.cos(theta) + ys * np.sin(theta)
        order = np.argsort(proj, kind="stable")
        before = np.cumsum(onehot[order], axis=0)            # (n, C): counts on the low side
        total = before[-1]
        after = total[None, :] - before
        purity = (before.max(axis=1) + after.max(axis=1)) / n
        purity[:min_pixels - 1] = 0.0
        purity[n - min_pixels:] = 0.0
        k = int(np.argmax(purity))
        if purity[k] > best[2]:
            sp = proj[order]
            best = (float(theta), float((sp[k] + sp[min(k + 1, n - 1)]) / 2), float(purity[k]))
    return best


def _halves(geom, theta: float, offset: float):
    """The polygon on either side of the line perpendicular to ``theta`` at ``offset`` (anchored on the polygon)."""
    from shapely.geometry import Polygon

    minx, miny, maxx, maxy = geom.bounds
    span = float(np.hypot(maxx - minx, maxy - miny)) + 10.0
    ux, uy = np.cos(theta), np.sin(theta)
    px, py = -uy, ux
    c = geom.centroid
    along = c.x * px + c.y * py
    cx, cy = offset * ux + along * px, offset * uy + along * py

    def side(sign):
        a = (cx + px * span, cy + py * span)
        b = (cx - px * span, cy - py * span)
        return Polygon([a, b, (b[0] + sign * ux * span, b[1] + sign * uy * span),
                        (a[0] + sign * ux * span, a[1] + sign * uy * span)])

    return geom.intersection(side(+1.0)), geom.intersection(side(-1.0))


# ---------------------------------------------------------------------------------------------------------- labelling
def _label_one(geom, xs, ys, cls, n_classes: int, depth: int, origin: str):  # noqa: C901
    """Rows (dict, geometry) for one polygon: clean / too small / cut (recursively) / majority."""
    n = len(cls)
    if n == 0:
        return [({"label": -1, "label_share": np.nan, "pixels": 0, "origin": origin,
                  "decision": "no pixel centre inside"}, geom)]
    grouped = RICE_GROUP_MAJORITY or TREE_CUT_MIN_WIDTH_M is not None
    rice_px = np.isin(cls, STRIP_FROM_LABELS)
    judge = np.where(rice_px, STRIP_FROM_LABELS[0], cls) if grouped else cls    # all rice as one class
    counts = np.bincount(judge, minlength=n_classes)
    top = int(counts.argmax())
    share = float(counts[top] / n)
    if grouped and top == STRIP_FROM_LABELS[0]:
        top = int(np.bincount(cls[rice_px], minlength=n_classes).argmax())         # the piece's own rice class
    row = {"label": top, "label_share": round(share, 3), "pixels": int(n), "origin": origin}
    if n < MIN_PIXELS:
        return [(dict(row, decision="too small to judge, majority label"), geom)]
    if share >= CLEAN_SHARE:
        return [(dict(row, decision="clean"), geom)]
    if depth >= CUT_DEPTH:
        return [(dict(row, decision="mixed, majority label (cut depth reached)"), geom)]
    th = _orientation(geom)
    tree_cut = TREE_CUT_MIN_WIDTH_M is not None and (judge == TREE_LABEL).sum() >= TREE_CUT_MIN_PIXELS
    min_px = min(MIN_PIXELS, TREE_CUT_MIN_PIXELS) if tree_cut else MIN_PIXELS
    angle, offset, purity = best_cut(xs, ys, judge, (th, th + np.pi / 2), n_classes, min_pixels=min_px)
    if purity - share < MIN_CUT_GAIN:
        return [(dict(row, decision="mixed, majority label (a cut does not help)"), geom)]
    a, b = _halves(geom, angle, offset)
    hi = xs * np.cos(angle) + ys * np.sin(angle) > offset
    width = MIN_SPLIT_WIDTH_M
    if tree_cut:                                  # one side tree, the other rice: the smaller edge-tree limits
        maj = [np.bincount(judge[sel], minlength=n_classes).argmax() if sel.any() else -1 for sel in (hi, ~hi)]
        if sorted(maj) == sorted([TREE_LABEL, STRIP_FROM_LABELS[0]]):
            width = TREE_CUT_MIN_WIDTH_M
        elif min((hi).sum(), (~hi).sum()) < MIN_PIXELS:
            return [(dict(row, decision="mixed, majority label (a cut does not help)"), geom)]
    if not (field_like(a, width) and field_like(b, width)):
        return [(dict(row, decision="mixed, cut would leave a sliver"), geom)]
    out = []
    for part, sel in ((a, hi), (b, ~hi)):
        for piece in getattr(part, "geoms", [part]):
            if piece.geom_type != "Polygon" or piece.area < SLIVER_SQM:
                continue
            out += _label_one(piece, xs[sel], ys[sel], cls[sel], n_classes, depth + 1, "split")
    return out or [(dict(row, decision="mixed, cut failed"), geom)]


def label_fields(fields, classes, transform, n_classes: int, valid=None):
    """Label (and cut) the delineated polygons; ``fields`` in the raster's (metric) CRS. Returns a GeoDataFrame of
    labelled pieces and the raster of polygon ids (0 = no polygon) used to find unclaimed ground."""
    import geopandas as gpd
    import shapely
    from rasterio.features import rasterize

    geoms = _polygons_only(fields.geometry.to_numpy())
    geoms = _resolve_overlaps(geoms, shapely.area(geoms))
    geoms = _polygons_only(geoms)
    geoms = geoms[shapely.area(geoms) > SLIVER_SQM]
    ids = rasterize(((g, i + 1) for i, g in enumerate(geoms)), out_shape=classes.shape, transform=transform,
                    fill=0, dtype="int32")
    ok = (ids > 0) & (valid if valid is not None else True)
    rr, cc = np.nonzero(ok)
    xs, ys = transform * (cc + 0.5, rr + 0.5)
    xs, ys = np.asarray(xs), np.asarray(ys)
    fid = ids[rr, cc]
    cls = classes[rr, cc].astype(int)
    order = np.argsort(fid, kind="stable")
    fid_s = fid[order]
    starts = np.searchsorted(fid_s, np.arange(1, len(geoms) + 1), side="left")
    ends = np.searchsorted(fid_s, np.arange(1, len(geoms) + 1), side="right")
    rows, out_geoms = [], []
    for i, g in enumerate(geoms):
        m = order[starts[i]:ends[i]]
        for row, geom in _label_one(g, xs[m], ys[m], cls[m], n_classes, 0, "delineation"):
            rows.append(dict(row, parent=i))
            out_geoms.append(geom)
    return gpd.GeoDataFrame(rows, geometry=out_geoms, crs=fields.crs), ids


def derived_polygons(classes, free, transform, crs, codes):
    """Polygons for map classes where no delineated polygon lies (``free``), shaped like fields (cane_scanner):
    opening by half ``MIN_FIELD_WIDTH_M``, area floor, rectangle-fill and compactness gates, half-pixel simplify."""
    import geopandas as gpd
    import shapely
    from rasterio.features import shapes
    from shapely.geometry import shape

    rows, geoms = [], []
    r = MIN_FIELD_WIDTH_M / 2.0
    for code in codes:
        m = free & (classes == code)
        if not m.any():
            continue
        g = np.array([shape(s) for s, _ in shapes(m.astype("uint8"), mask=m, transform=transform)], dtype=object)
        g = shapely.buffer(shapely.buffer(g, -r, join_style="mitre"), r, join_style="mitre")
        g = _polygons_only(g)
        g = g[shapely.area(g) >= MIN_ORPHAN_ACRES * SQM_PER_ACRE]
        if not len(g):
            continue
        rect = shapely.area(shapely.oriented_envelope(g))
        with np.errstate(invalid="ignore", divide="ignore"):
            keep = (shapely.area(g) / rect >= MIN_RECTANGULARITY) & \
                (4 * np.pi * shapely.area(g) / shapely.length(g) ** 2 >= MIN_COMPACTNESS)
        g = shapely.simplify(g[keep], SIMPLIFY_M, preserve_topology=True)
        g = g[shapely.is_valid(g) & ~shapely.is_empty(g)]
        geoms += list(g)
        rows += [{"label": int(code), "label_share": 1.0, "pixels": int(round(shapely.area(x) / 100.0)),
                  "origin": "derived from map", "decision": "no delineation polygon here"} for x in g]
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=crs) if rows else \
        gpd.GeoDataFrame({"label": [], "label_share": [], "pixels": [], "origin": [], "decision": []},
                         geometry=[], crs=crs)


def _absorb(f, lim: float, touch_m: float, passes: int):
    """Sliver absorption passes (see ``absorb_slivers``) until nothing moves."""
    import shapely

    for _ in range(passes):
        geoms = f.geometry.to_numpy()
        area = shapely.area(geoms)
        small = np.flatnonzero(area < lim)
        if not small.size:
            break
        tree = shapely.STRtree(geoms)
        si, ti = tree.query(shapely.buffer(geoms[small], touch_m), predicate="intersects")
        si = small[si]
        keep = si != ti
        si, ti = si[keep], ti[keep]
        if not si.size:
            break
        shared = shapely.length(shapely.intersection(shapely.boundary(geoms[si]), shapely.buffer(geoms[ti], touch_m)))
        same_field = f["parent"].to_numpy()[si] == f["parent"].to_numpy()[ti]
        bigger = area[ti] >= area[si]
        score = same_field * 1e12 + bigger * 1e9 + shared
        best = pd.DataFrame({"s": si, "t": ti, "score": score}).sort_values("score").groupby("s").tail(1)
        moving = set(best["s"])
        best = best[~best["t"].isin(moving)]
        if best.empty:
            # every sliver points at another sliver: let the largest of each such pair stay put this pass
            best = pd.DataFrame({"s": si, "t": ti, "score": score}).sort_values("score").groupby("s").tail(1)
            best = best[area[best["t"].to_numpy()] > area[best["s"].to_numpy()]]
            best = best[~best["t"].isin(set(best["s"]))]
            if best.empty:
                break
        groups = best.groupby("t")["s"].apply(list)
        new_geoms = geoms.copy()
        for t, ss in groups.items():
            new_geoms[t] = shapely.union_all(np.r_[[geoms[t]], geoms[ss]])
        f = f.copy()
        f["geometry"] = new_geoms
        f.loc[groups.index, "absorbed"] += groups.apply(len).to_numpy()
        f.loc[groups.index, "pixels"] += [int(f.loc[ss, "pixels"].sum()) for ss in groups]
        f = f.drop(index=best["s"].to_numpy()).reset_index(drop=True)
    return f


def absorb_slivers(frame, min_acres: float, touch_m: float = MERGE_TOUCH_M, passes: int = 30):
    """Pieces smaller than ``min_acres`` join the neighbour they share the LONGEST boundary with, preferring a piece of
    the same delineated field (``parent``), and take its label (user, 2 Oct: "remove slivers by merging them to their
    bigger neighbour within the context of a field"; "no monster polygons": different fields are never dissolved
    together, only a sliver moves into one neighbour). Pieces of one field with the same label that touch become one.
    A small piece with no neighbour (a real small field alone) is kept, flagged. Runs a few passes so a sliver whose
    only neighbours were slivers still finds a home. Geometry stays disjoint: a sliver is united with exactly one
    neighbour."""
    import shapely

    f = frame.reset_index(drop=True).copy()
    f["absorbed"] = 0
    lim = min_acres * SQM_PER_ACRE
    f = _absorb(f, lim, touch_m, passes)
    # pieces of one field with the same label that touch: one polygon (inside the field's own outline)
    out = []
    for (par, lab), grp in f.groupby(["parent", "label"]):
        if len(grp) == 1:
            out.append(grp)
            continue
        # pieces cut from one field share the cut line exactly: a plain union joins them with no overhang (a buffered
        # union overhung the neighbours and trimming it back left new crumbs, 2 Oct)
        parts = _polygons_only(shapely.get_parts(shapely.union_all(grp.geometry.to_numpy())))
        rows = [{"parent": par, "label": lab, "origin": "/".join(sorted(set(grp["origin"]))),
                 "decision": "pieces of one field, same label", "pixels": int(grp["pixels"].sum()),
                 "label_share": float(grp["label_share"].mean()), "absorbed": int(grp["absorbed"].sum())}
                for _ in parts]
        import geopandas as gpd

        out.append(gpd.GeoDataFrame(rows, geometry=list(parts), crs=f.crs))
    import geopandas as gpd

    res = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=f.crs)
    res = res.explode(index_parts=False)
    res = res[res.geom_type == "Polygon"].reset_index(drop=True)
    res["absorbed"] = res["absorbed"].fillna(0).astype(int)
    res = _absorb(res, lim, touch_m, passes)
    # tidy every polygon (slits, small holes, spurs), then give any re-created overlap back to the smaller polygon
    res["geometry"] = tidy(res.geometry.to_numpy(), hole_sqm=lim)
    res = res[~res.geometry.is_empty & (res.area > SLIVER_SQM)].reset_index(drop=True)
    res["geometry"] = _resolve_overlaps(res.geometry.to_numpy(), shapely.area(res.geometry.to_numpy()))
    res = res[~res.geometry.is_empty].explode(index_parts=False)
    res = res[(res.geom_type == "Polygon") & (res.area > SLIVER_SQM)].reset_index(drop=True)
    res = _absorb(res, lim, touch_m, passes)
    # last tidy after the last absorption (a sliver that only nearly touched made a two-part polygon); a polygon keeps
    # its largest part, the detached crumbs are dropped and counted
    before = float(res.area.sum())
    res["geometry"] = tidy(res.geometry.to_numpy(), hole_sqm=lim)
    res["geometry"] = _resolve_overlaps(res.geometry.to_numpy(), shapely.area(res.geometry.to_numpy()))
    res["geometry"] = [max(getattr(x, "geoms", [x]), key=lambda q: q.area) if not x.is_empty else x
                       for x in shapely.make_valid(res.geometry.to_numpy())]
    res = res[~res.geometry.is_empty & (res.geom_type == "Polygon") & (res.area > SLIVER_SQM)].reset_index(drop=True)
    res.attrs["tidy_lost_acres"] = round((before - float(res.area.sum())) / SQM_PER_ACRE, 2)
    # safety: anything still overlapping (should be none) goes to the smaller polygon, then crumbs are absorbed again
    for _ in range(3):
        if overlap_acres(res) == 0:
            break
        res["geometry"] = _resolve_overlaps(res.geometry.to_numpy(), shapely.area(res.geometry.to_numpy()))
        res = res[~res.geometry.is_empty].explode(index_parts=False)
        res = res[(res.geom_type == "Polygon") & (res.area > SLIVER_SQM)].reset_index(drop=True)
        res = _absorb(res, lim, touch_m, passes)
    res["small_alone"] = shapely.area(res.geometry.to_numpy()) < lim
    res["acres"] = (res.area / SQM_PER_ACRE).round(3)
    return res


def tidy(geoms, hole_sqm: float, close_m: float = CLOSE_M, despike_m: float = DESPIKE_M):
    """Close slits, fill small holes and cut thin spurs off each polygon, keeping its real edges (see ``CLOSE_M``).
    A polygon that would vanish under the despike (a small or narrow real field) keeps its closed outline."""
    import shapely
    from shapely.geometry import Polygon

    g = np.asarray(geoms, dtype=object)
    closed = shapely.buffer(shapely.buffer(g, close_m, quad_segs=2, join_style="mitre"), -close_m, quad_segs=2,
                            join_style="mitre")
    closed = shapely.make_valid(closed)

    def fill(geom):
        parts = []
        for p in getattr(geom, "geoms", [geom]):
            if p.geom_type != "Polygon":
                continue
            holes = [h for h in p.interiors if Polygon(h).area >= hole_sqm]
            parts.append(Polygon(p.exterior, holes))
        return shapely.union_all(parts) if parts else geom

    filled = np.array([fill(x) for x in closed], dtype=object)
    body = shapely.buffer(shapely.buffer(filled, -despike_m, quad_segs=2), despike_m + 0.5, quad_segs=2)
    cut = shapely.intersection(filled, body)
    keep = ~shapely.is_empty(cut) & (shapely.area(cut) > 0.5 * shapely.area(filled))
    out = np.where(keep, cut, filled)
    # largest part only: a despiked or closed shape may leave a detached crumb
    return np.array([max(getattr(x, "geoms", [x]), key=lambda q: q.area) if not x.is_empty else x for x in
                     shapely.make_valid(out)], dtype=object)


def overlap_acres(frame) -> float:
    """Ground claimed by more than one polygon, in acres (should be 0)."""
    import shapely

    g = frame.geometry.to_numpy()
    left, right = shapely.STRtree(g).query(g, predicate="intersects")
    keep = left < right
    if not keep.any():
        return 0.0
    return float(shapely.area(shapely.intersection(g[left[keep]], g[right[keep]])).sum() / SQM_PER_ACRE)


def short_side_m(gdf) -> np.ndarray:
    """Short side of each polygon's minimum rotated rectangle (metres): the width of a strip."""
    out = []
    for g in gdf.geometry.minimum_rotated_rectangle():
        x, y = g.exterior.coords.xy
        a, b = np.hypot(x[1] - x[0], y[1] - y[0]), np.hypot(x[2] - x[1], y[2] - y[1])
        out.append(min(a, b))
    return np.asarray(out)


def run(fields, classes, transform, crs, class_names: dict, nodata: int = 255, sliver_acres=(0.10,)):
    """The field layer for one AOI: (labelled pieces before sliver absorption, {threshold: final layer})."""
    import geopandas as gpd
    import shapely

    f = fields.to_crs(crs)
    valid = (classes != nodata) & (classes != 0)
    n_classes = int(max(class_names)) + 1
    lab, ids = label_fields(f, classes, transform, n_classes, valid)
    lab = lab[lab["label"] > 0]
    free = valid & (ids == 0)
    der = derived_polygons(classes, free, transform, crs, [c for c in class_names if c])
    if len(der):
        # derived polygons never cut a traced one
        covered = shapely.union_all(lab.geometry.to_numpy())
        der["geometry"] = shapely.difference(der.geometry.to_numpy(), covered)
        der = der[~der.geometry.is_empty & (der.area > SLIVER_SQM)].explode(index_parts=False)
        der = der[der.geom_type == "Polygon"].reset_index(drop=True)
        der["parent"] = -1 - np.arange(len(der))                 # each derived block is its own "field"
    pieces = gpd.GeoDataFrame(pd.concat([lab, der], ignore_index=True), crs=crs)
    pieces["class_name"] = pieces["label"].map(class_names)
    pieces["acres"] = (pieces.area / SQM_PER_ACRE).round(3)
    finals = {}
    for thr in sliver_acres:
        fin = absorb_slivers(pieces, thr)
        if STRIP_MAX_WIDTH_M is not None and STRIP_LABEL in class_names:
            fin = fin.copy()
            fin.loc[fin["label"].isin(STRIP_FROM_LABELS) & (short_side_m(fin) < STRIP_MAX_WIDTH_M), "label"] = STRIP_LABEL
        fin["class_name"] = fin["label"].map(class_names)
        fin.insert(0, "polygon_id", [f"p{i:06d}" for i in range(len(fin))])
        finals[thr] = fin
    return pieces, finals
