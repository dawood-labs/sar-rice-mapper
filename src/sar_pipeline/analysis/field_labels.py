"""Label field polygons from a class map, keeping the delineation's geometry.

A delineation traced on high-resolution imagery has the edges a client wants; a 10 m class map knows
what grows where but not where anything ends. One rule decides every question below:

    the delineation owns the geometry, the class map owns only the label.

A raster edge never becomes an output edge, except where the delineation has no polygon at all.

Order of the pipeline (each step's counts go into the summary):

  1. repair    make_valid, explode until nothing is nested, resolve overlaps in favour of the SMALLER
               polygon (the finer tracing wins), drop topological slivers.
  2. despike   take tails narrower than 2 x DESPIKE_M off, then drop what is left of a line rather
               than a field (compactness and area floors).
  3. label     class counts per polygon from the raster: majority class, purity, pixels. Below
               MIN_PIXELS a share is not a measurement: the polygon is labelled and flagged.
  4. cut       polygons under PURE hold more than one field: one straight cut along the polygon's own
               orientation, kept only if purity gains MIN_CUT_GAIN and neither side is a sliver.
  5. derive    classified pixels no polygon claims get a polygon of their own, above the area floor
               and only if the shape could be a field; tagged so a reviewer sees it is not traced.
  6. model     a second, independent label per polygon from the field-level model (the mean of the
               field's pixel features), so two methods can be compared per field.
  7. tidy      final validity, no overlapping area, traced geometry ahead of derived.

Everything is measured in acres.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from rasterio import features as rfeatures
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import Polygon
from shapely.geometry import shape as to_shape
from shapely.validation import make_valid

from .. import config as config_mod
from .. import manifest, resources, run_meta
from . import gcp_qc
from .pixel_features import (OK_STATES, analysis_dir, columns_for_set, plans_for, read_chunk_array, run_context,
                             set_from_config, track_features, wet_flags)

log = logging.getLogger(__name__)

SQM_PER_ACRE = 4046.8564224

#: Fewer classified pixels than this and a class share is not a measurement, only noise.
MIN_PIXELS = 10

#: At or above this share of one class a polygon holds one thing; below it, more than one.
PURE = 0.85

#: A cut has to earn its place. On a polygon that is already 80 % one class a cut can add almost
#: nothing, and a cut that adds nothing invents a boundary the ground does not have.
MIN_CUT_GAIN = 0.08

#: Classified ground with no polygon over it gets one, down to this size. Below it the blobs are
#: 10 m edge effects rather than fields.
MIN_DERIVED_ACRES = 0.15

#: Below this a piece is topological debris from the overlap arithmetic, not a field. It has to stay
#: well under one pixel: a one-pixel floor throws away small but real polygons.
SLIVER_SQM = 10.0

#: How wide a spur has to be to be believed: anything narrower than twice this is a tail from tracing.
DESPIKE_M = 4.0

#: Perimeter relative to that of a square of the same area. A square is 1.0 and a circle 0.89, so
#: anything at or below this cannot carry a tail and is skipped rather than buffered.
DESPIKE_SLENDERNESS = 1.15

#: Smallest a polygon may be once its tails are off. Below this there was never a field, only the tail.
MIN_BODY_ACRES = 0.05

#: Polsby-Popper compactness below which a shape is a line rather than a field, applied to everything
#: once the tails are off.
MIN_COMPACTNESS_ANY = 0.10

#: A field is not a ribbon: anything that vanishes when eroded by half this width was the strip of
#: disagreement between a traced boundary and a 10 m raster, never a field of its own.
MIN_FIELD_WIDTH_M = 14.0

#: The same test on a piece a cut produced, where it has to be stricter: a narrow piece means the cut
#: ran along an edge that was already there and shaved a needle off it.
MIN_SPLIT_WIDTH_M = 20.0

#: How much of its own minimum rotated rectangle a shape fills. The floor sits below 0.5 on purpose:
#: a triangle fills exactly half its rectangle, and triangular fields are real.
MIN_RECTANGULARITY = 0.45

#: Polsby-Popper compactness for derived shapes. A circle is 1 and a square 0.79, so this rejects
#: only shapes carrying far more edge than a field has.
MIN_COMPACTNESS_DERIVED = 0.18

#: Output coordinate grid in degrees (about a centimetre): coarse enough that reprojection rounding
#: cannot split a shared edge into overlapping slivers.
GRID = 1e-7

#: Windows used for the zonal passes. Large enough to be efficient, small enough to bound memory.
WINDOW_PX = 1024

#: Outlines are simplified by this many metres before anything else. A delineation vectorised from a
#: sub-metre mask carries a vertex every few decimetres (a pixel staircase, not a boundary anyone drew),
#: and every set operation downstream pays for each vertex. Measured on a traced layer: 0.5 m keeps 11 %
#: of the vertices, changes area by 0.13 % and moves the median boundary by 0.5 m - far below the 10 m
#: pixel that decides the label - while buffering runs about 100 times faster. 0 turns it off.
SIMPLIFY_M = 0.5


def fields_dir(run_dir: Path, name: str) -> Path:
    """processed/<aoi>/<season>/fields/<name>/"""
    run_dir = Path(run_dir)
    return run_dir.parents[1] / "fields" / name


def acres(geoms) -> np.ndarray:
    return np.asarray(shapely.area(np.asarray(geoms, dtype=object))) / SQM_PER_ACRE


def compactness(frame: gpd.GeoDataFrame) -> pd.Series:
    """Polsby-Popper: 4 pi A / P^2. A circle is 1, a square 0.79, a ribbon near 0."""
    return 4 * np.pi * frame.area / frame.length.pow(2).replace(0, np.nan)


def field_like(geom, min_width: float = MIN_FIELD_WIDTH_M) -> bool:
    """Does the shape still have a core when eroded by half a field's width?

    The one test that catches every shape a field is not - a needle, a ribbon, the shaving off the
    edge of a bigger polygon - while a field, however irregular, keeps a core.
    """
    if geom is None or geom.is_empty:
        return False
    # The test buffers a 1 m simplified copy: whether a core survives a several-metre erosion does not
    # depend on sub-metre vertices, and traced outlines carry hundreds of them, which is the whole cost.
    core = shapely.simplify(geom, 1.0).buffer(-min_width / 2.0)
    return (not core.is_empty) and core.area > 0.0


# ------------------------------------------------------------------ geometry cleanup

def simplify_outlines(frame: gpd.GeoDataFrame, tolerance_m: float = SIMPLIFY_M) -> tuple[gpd.GeoDataFrame, dict]:
    """Drop redundant vertices (pixel staircases) within `tolerance_m`, in a metric CRS.

    Each polygon is simplified on its own, so neighbours can gain or lose a few square decimetres where
    they meet; `repair` resolves any overlap that creates. Returns the frame and before/after counts.
    """
    geoms = frame.geometry.to_numpy()
    stats = {"tolerance_m": tolerance_m, "vertices_in": int(shapely.get_num_coordinates(geoms).sum()),
             "acres_in": round(float(shapely.area(geoms).sum() / SQM_PER_ACRE), 1)}
    if tolerance_m <= 0:
        return frame, {**stats, "vertices_out": stats["vertices_in"], "acres_out": stats["acres_in"]}
    simple = shapely.simplify(geoms, tolerance_m, preserve_topology=True)
    stats["vertices_out"] = int(shapely.get_num_coordinates(simple).sum())
    stats["acres_out"] = round(float(shapely.area(simple).sum() / SQM_PER_ACRE), 1)
    log.info("simplify %.2f m: %.1f -> %.1f million vertices (%.0f %%), %.0f -> %.0f acres", tolerance_m,
             stats["vertices_in"] / 1e6, stats["vertices_out"] / 1e6,
             100 * stats["vertices_out"] / max(stats["vertices_in"], 1), stats["acres_in"], stats["acres_out"])
    return frame.assign(geometry=simple), stats


def resolve_overlaps(frame: gpd.GeoDataFrame, priority=None) -> tuple[np.ndarray, int]:
    """Give every twice-claimed patch of ground to the polygon with the better claim (lower `priority`).

    Default priority is area, so the smaller polygon keeps its shape and the larger is cut around it:
    letting the larger win would erase exactly the detail the tracing was done for.

    Two array operations, not a loop: every intersecting pair comes back from the index at once and
    each polygon is cut once against the union of everything with a better claim. Cutting against the
    ORIGINAL better-claiming geometry also removes the order dependence a sequential pass has.
    """
    if priority is None:
        priority = frame.area.to_numpy()
    priority = np.asarray(priority, dtype=float)
    geoms = frame.geometry.to_numpy()
    # "overlaps" and containment, never "intersects": neighbours that merely share an edge have no
    # ground in common, and in a traced layer they are the overwhelming majority of intersecting pairs.
    # Asking for them returns millions of pairs and the difference over them exhausts memory for nothing.
    pairs = [frame.sindex.query(frame.geometry, predicate=p) for p in ("overlaps", "contains", "within")]
    left = np.concatenate([p[0] for p in pairs])
    right = np.concatenate([p[1] for p in pairs])
    keep = left != right
    left, right = left[keep], right[keep]
    if left.size:
        unique = np.unique(left.astype("int64") * len(frame) + right.astype("int64"))
        left, right = (unique // len(frame)).astype("int64"), (unique % len(frame)).astype("int64")
    wins = (priority[right] < priority[left]) | ((priority[right] == priority[left]) & (right < left))
    left, right = left[wins], right[wins]
    if left.size == 0:
        return geoms, 0
    order = np.argsort(left, kind="stable")
    left, right = left[order], right[order]
    targets, starts = np.unique(left, return_index=True)
    ends = np.append(starts[1:], left.size)
    cutters = np.array([shapely.union_all(geoms[right[lo:hi]]) for lo, hi in zip(starts, ends)], dtype=object)
    out = geoms.copy()
    out[targets] = safe_binary(shapely.difference, geoms[targets], cutters)
    return out, int(targets.size)


def explode_fully(frame: gpd.GeoDataFrame, rounds: int = 5) -> gpd.GeoDataFrame:
    """Single-part valid polygons, however nested the input is.

    `make_valid` hands back GeometryCollections, and a collection can hold a MultiPolygon inside it:
    one explode leaves those nested Multis intact and a `geom_type == Polygon` filter then silently
    drops their area. Explode until nothing is nested.
    """
    for _ in range(rounds):
        bad = ~frame.geometry.is_valid
        if bad.any():
            frame = frame.copy()
            frame.loc[bad, "geometry"] = frame.loc[bad, "geometry"].apply(make_valid)
        frame = frame.explode(index_parts=False)
        if not frame.geom_type.isin(["MultiPolygon", "GeometryCollection", "MultiLineString", "MultiPoint"]).any():
            break
    frame = frame[frame.geom_type == "Polygon"]
    return frame[~frame.geometry.is_empty & frame.geometry.is_valid].reset_index(drop=True)


def repair(fields: gpd.GeoDataFrame, report_coverage: bool = False) -> tuple[gpd.GeoDataFrame, dict]:
    """Valid, single-part, non-overlapping polygons in the input's (metric) CRS.

    `report_coverage` dissolves the whole layer to say how much ground it covers with overlaps counted
    once. That is one log line, and on a layer of tens of thousands of traced polygons it costs more
    memory than everything else here together, so it is off by default.
    """
    before_n, before_acres = len(fields), float(fields.area.sum() / SQM_PER_ACRE)
    covered = float(shapely.union_all(fields.geometry.to_numpy()).area / SQM_PER_ACRE) if report_coverage else float("nan")
    frame = explode_fully(fields)
    exploded_acres = float(frame.area.sum() / SQM_PER_ACRE)
    log.info("repair: %d polygons -> %d parts (%.0f -> %.0f acres%s)", before_n, len(frame), before_acres,
             exploded_acres, "" if np.isnan(covered) else f", {covered:.0f} acres of ground")

    geoms, trimmed = resolve_overlaps(frame)
    frame = frame.assign(geometry=geoms)
    frame = explode_fully(frame)
    slivers = frame.area <= SLIVER_SQM
    log.info("repair: overlaps trimmed %d polygons; dropped %d slivers holding %.1f acres",
             trimmed, int(slivers.sum()), float(frame.area[slivers].sum() / SQM_PER_ACRE))
    frame = frame[~slivers].reset_index(drop=True)
    stats = {"polygons_in": before_n, "acres_in": round(before_acres, 1),
             "acres_of_ground": None if np.isnan(covered) else round(covered, 1),
             "parts_after_explode": len(frame) + int(slivers.sum()), "acres_after_explode": round(exploded_acres, 1),
             "overlaps_trimmed": trimmed, "slivers_dropped": int(slivers.sum()),
             "polygons_out": len(frame), "acres_out": round(float(frame.area.sum() / SQM_PER_ACRE), 1)}
    return frame, stats


def safe_binary(op, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """A vectorised shapely set operation that survives the odd GEOS failure on a traced layer.

    One self-touching ring among tens of thousands raises "non-noded intersection" and takes the whole
    vectorised call down with it. The documented remedy is to redo the operation on a fine grid; if even
    that fails, the offending polygon is kept whole, because a tail left on one field (or a sliver of
    overlap) beats a hole in the layer.
    """
    try:
        return op(left, right)
    except shapely.errors.GEOSException:
        log.info("%s: retrying on a 1 mm grid after a GEOS topology error", op.__name__)
    try:
        return op(left, right, grid_size=0.001)
    except shapely.errors.GEOSException:
        out, failed = np.empty(len(left), dtype=object), 0
        for i, (a, b) in enumerate(zip(left, right)):
            try:
                out[i] = op(a, b)
            except shapely.errors.GEOSException:
                out[i], failed = a, failed + 1
        log.warning("%s: kept %d polygons unchanged after GEOS refused them", op.__name__, failed)
        return out


def despike(geoms: gpd.GeoSeries) -> gpd.GeoSeries:
    """Take tails narrower than 2 x DESPIKE_M off without moving the boundaries that matter.

    Eroding and dilating alone would round every corner, a worse change than the one being fixed. So
    the opened body is dilated slightly past its own radius and intersected back with the ORIGINAL,
    which restores true edges and corners exactly while leaving the tails outside. A polygon that
    vanishes under the erosion was a line, not a field, and comes back empty for the caller to drop.

    There is deliberately no "leave it alone if it would lose more than X %" clause: the polygons that
    lose most are precisely the ones that are mostly tail.
    """
    slender = geoms.length / (4.0 * np.sqrt(np.maximum(geoms.area, 1e-9)))
    worth_it = (slender > DESPIKE_SLENDERNESS).to_numpy()
    subset = geoms[worth_it]
    result = geoms.copy()
    if len(subset) == 0:
        log.info("despike: nothing slender enough to carry a tail")
        return result
    # The buffering runs on a simplified copy and the result is intersected back with the original, so
    # simplification only decides which tails are found and never touches an output edge. Mitred joins
    # keep convex corners sharp through the erosion and the dilation, so a square comes back a square;
    # rounded joins chamfer every corner and then need a wider dilation to hide it.
    work = shapely.simplify(subset.to_numpy(), 1.0)
    core = shapely.buffer(work, -DESPIKE_M, join_style="mitre")
    body = shapely.make_valid(shapely.buffer(core, DESPIKE_M, join_style="mitre"))
    cleaned = safe_binary(shapely.intersection, subset.to_numpy(), body)
    usable = shapely.is_valid(cleaned) & ~shapely.is_empty(cleaned)
    gone = shapely.is_empty(core)
    result.loc[subset.index[usable]] = cleaned[usable]
    result.loc[subset.index[gone]] = None
    log.info("despike: %d of %d polygons trimmed (%d already compact), %d had no body at all",
             int(usable.sum()), len(geoms), int((~worth_it).sum()), int(gone.sum()))
    return result


def drop_lines(frame: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """Once every polygon has lost its tails, whatever is still line-shaped always was one."""
    comp = compactness(frame)
    body = frame.area >= MIN_BODY_ACRES * SQM_PER_ACRE
    shaped = comp >= MIN_COMPACTNESS_ANY
    stats = {"dropped_as_lines": int((~shaped).sum()),
             "dropped_as_too_small": int((shaped & ~body).sum()),
             "acres_dropped": round(float(frame.area[~(shaped & body)].sum() / SQM_PER_ACRE), 1)}
    log.info("shape gates: dropped %d as lines and %d as too small a body (%.1f acres)",
             stats["dropped_as_lines"], stats["dropped_as_too_small"], stats["acres_dropped"])
    return frame[shaped & body].reset_index(drop=True), stats


# ------------------------------------------------------------------ zonal reading

def windows_of(width: int, height: int, size: int = WINDOW_PX):
    for row in range(0, height, size):
        for col in range(0, width, size):
            yield Window(col, row, min(size, width - col), min(size, height - row))


def zonal_counts(frame: gpd.GeoDataFrame, class_path: Path, prob_path: Path | None, codes: list[int],
                 claimed: np.ndarray | None = None) -> dict:
    """Class counts, target-probability sums and pixel totals per polygon, read window by window.

    `frame` carries `fid` (1..N) and lies in the raster's CRS. `claimed`, when given, is set to 1
    wherever any polygon covers a pixel, so the caller can find ground no polygon claims.
    """
    n = len(frame)
    counts = np.zeros((n + 1, len(codes) + 1), dtype="int64")   # last column: classified 0 / nodata
    prob_sum = np.zeros(n + 1)
    prob_n = np.zeros(n + 1)
    tree = shapely.STRtree(frame.geometry.values)
    fids = frame["fid"].to_numpy()
    lookup = np.zeros(256, dtype="int64") + len(codes)
    for i, c in enumerate(codes):
        lookup[c] = i
    prob_ds = rasterio.open(prob_path) if prob_path else None
    with rasterio.open(class_path) as cds:
        transform = cds.transform
        for win in windows_of(cds.width, cds.height):
            bounds = rasterio.windows.bounds(win, transform)
            idx = tree.query(shapely.box(*bounds))
            if idx.size == 0:
                continue
            fid_img = rfeatures.rasterize(zip(frame.geometry.values[idx], fids[idx]),
                                          out_shape=(int(win.height), int(win.width)),
                                          transform=window_transform(win, transform), fill=0, dtype="int32")
            inside = fid_img > 0
            if claimed is not None:
                claimed[int(win.row_off):int(win.row_off + win.height),
                        int(win.col_off):int(win.col_off + win.width)][inside] = 1
            if not inside.any():
                continue
            here = fid_img[inside]
            cls = cds.read(1, window=win)[inside]
            np.add.at(counts, (here, lookup[cls]), 1)
            if prob_ds is not None:
                prob = prob_ds.read(1, window=win)[inside]
                ok = prob != prob_ds.nodata
                np.add.at(prob_sum, here[ok], prob[ok].astype(float))
                np.add.at(prob_n, here[ok], 1)
    if prob_ds is not None:
        prob_ds.close()
    return {"counts": counts, "prob_sum": prob_sum, "prob_n": prob_n}


def collect_pixels(frame: gpd.GeoDataFrame, class_path: Path, wanted: np.ndarray, codes: list[int]) -> dict:
    """{fid: (x, y, class index)} for the pixels of the polygons in `wanted`, in the raster's CRS."""
    if wanted.size == 0:
        return {}
    sel = frame[frame["fid"].isin(wanted)]
    tree = shapely.STRtree(sel.geometry.values)
    fids = sel["fid"].to_numpy()
    lookup = np.zeros(256, dtype="int64") + len(codes)
    for i, c in enumerate(codes):
        lookup[c] = i
    parts: dict[int, list] = {}
    with rasterio.open(class_path) as cds:
        transform = cds.transform
        for win in windows_of(cds.width, cds.height):
            idx = tree.query(shapely.box(*rasterio.windows.bounds(win, transform)))
            if idx.size == 0:
                continue
            wt = window_transform(win, transform)
            fid_img = rfeatures.rasterize(zip(sel.geometry.values[idx], fids[idx]),
                                          out_shape=(int(win.height), int(win.width)),
                                          transform=wt, fill=0, dtype="int32")
            inside = fid_img > 0
            if not inside.any():
                continue
            rows, cols = np.nonzero(inside)
            xs, ys = wt * (cols + 0.5, rows + 0.5)
            cls = lookup[cds.read(1, window=win)[inside]]
            here = fid_img[inside]
            # Sort once and slice, instead of one full-window comparison per polygon: with thousands of
            # mixed polygons in a window the per-polygon mask is O(polygons x pixels).
            order = np.argsort(here, kind="stable")
            here, xs, ys, cls = here[order], np.asarray(xs)[order], np.asarray(ys)[order], cls[order]
            fids_here, starts = np.unique(here, return_index=True)
            for fid, lo, hi in zip(fids_here, starts, np.append(starts[1:], here.size)):
                parts.setdefault(int(fid), []).append((xs[lo:hi], ys[lo:hi], cls[lo:hi]))
    return {fid: tuple(np.concatenate(a) for a in zip(*chunks)) for fid, chunks in parts.items()}


# ------------------------------------------------------------------ cutting

def orientation(geom) -> float:
    """Angle of the polygon's long axis: field subdivisions run parallel to it."""
    rect = geom.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return 0.0
    coords = np.array(rect.exterior.coords[:4])
    edges = np.diff(np.vstack([coords, coords[:1]]), axis=0)
    longest = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
    return float(np.arctan2(longest[1], longest[0]))


def best_cut(xs, ys, class_idx, n_classes: int, angles) -> tuple[float, float, float]:
    """Where along which axis one straight cut best separates the classes.

    Returns (angle, offset, purity), where purity is the share of pixels on the correct side once each
    side takes its own majority class.
    """
    best = (0.0, 0.0, 0.0)
    n = class_idx.size
    if n < 2 * MIN_PIXELS:
        return best
    for theta in angles:
        projection = xs * np.cos(theta) + ys * np.sin(theta)
        order = np.argsort(projection, kind="stable")
        onehot = np.zeros((n, n_classes), dtype="int32")
        onehot[np.arange(n), class_idx[order]] = 1
        left = np.cumsum(onehot, axis=0)                      # counts in the first k pixels
        right = left[-1] - left
        purity = (left.max(axis=1) + right.max(axis=1)) / n
        purity[:MIN_PIXELS] = 0.0
        purity[-MIN_PIXELS:] = 0.0
        position = int(np.argmax(purity))
        if purity[position] > best[2]:
            sorted_projection = projection[order]
            offset = float((sorted_projection[position] + sorted_projection[min(position + 1, n - 1)]) / 2)
            best = (theta, offset, float(purity[position]))
    return best


def halves(geom, theta: float, offset: float):
    """The polygon either side of the line perpendicular to `theta` at `offset`."""
    minx, miny, maxx, maxy = geom.bounds
    span = float(np.hypot(maxx - minx, maxy - miny)) + 10.0
    ux, uy = np.cos(theta), np.sin(theta)
    px, py = -uy, ux
    # Anchor the cut on the polygon: the foot of the perpendicular from the coordinate origin lies
    # hundreds of kilometres away in a projected CRS and the half-plane would miss the field.
    centre = geom.centroid
    along = centre.x * px + centre.y * py
    cx, cy = offset * ux + along * px, offset * uy + along * py

    def side(sign: float):
        a = (cx + px * span, cy + py * span)
        b = (cx - px * span, cy - py * span)
        c = (b[0] + sign * ux * span, b[1] + sign * uy * span)
        d = (a[0] + sign * ux * span, a[1] + sign * uy * span)
        return Polygon([a, b, c, d])

    return geom.intersection(side(+1.0)), geom.intersection(side(-1.0))


# ------------------------------------------------------------------ labelling

def label_polygons(frame: gpd.GeoDataFrame, zonal: dict, class_path: Path, names: list[str],
                   codes: list[int], target: str) -> tuple[gpd.GeoDataFrame, dict]:
    """Majority label per polygon, cutting the ones a straight line can genuinely improve."""
    t0 = time.time()
    counts = zonal["counts"][frame["fid"].to_numpy()]
    n_classes = len(codes)
    classified = counts[:, :n_classes].sum(axis=1)
    total = counts.sum(axis=1)
    majority_i = counts[:, :n_classes].argmax(axis=1)
    purity = np.where(classified > 0, counts[:, :n_classes].max(axis=1) / np.maximum(classified, 1), np.nan)

    mixed = (classified >= MIN_PIXELS) & (purity < PURE)
    pixels = collect_pixels(frame, class_path, frame["fid"].to_numpy()[mixed], codes)
    log.info("labelling: %d polygons, %d mixed (below %.2f purity), %d with fewer than %d pixels",
             len(frame), int(mixed.sum()), PURE, int((classified < MIN_PIXELS).sum()), MIN_PIXELS)

    target_i = names.index(target)
    rows, geoms = [], []
    tally = {"clean": 0, "cut": 0, "majority": 0, "too_few_pixels": 0, "cut_no_gain": 0, "cut_sliver": 0}
    for i, geom in enumerate(frame.geometry.values):
        fid = int(frame["fid"].iat[i])
        base = {"source_fid": fid, "n_pixels": int(total[i]), "n_classified": int(classified[i]),
                "classified_share": round(float(total[i] and classified[i] / total[i]), 3),
                "target_pixel_share": round(float(classified[i] and counts[i, target_i] / max(classified[i], 1)), 3),
                "origin": "delineation"}
        if classified[i] < MIN_PIXELS:
            tally["too_few_pixels"] += 1
            rows.append({**base, "majority_class": names[majority_i[i]] if classified[i] else None,
                         "majority_share": round(float(purity[i]), 3) if classified[i] else None,
                         "decision": "too few pixels to judge"})
            geoms.append(geom)
            continue
        if purity[i] >= PURE:
            tally["clean"] += 1
            rows.append({**base, "majority_class": names[majority_i[i]],
                         "majority_share": round(float(purity[i]), 3), "decision": "clean"})
            geoms.append(geom)
            continue

        xs, ys, cls = pixels.get(fid, (np.array([]), np.array([]), np.array([], dtype=int)))
        # Unclassified pixels carry index n_classes; they take no part in choosing or scoring a cut.
        classified_px = cls < n_classes
        xs, ys, cls = xs[classified_px], ys[classified_px], cls[classified_px]
        theta = orientation(geom)
        angle, offset, cut_purity = best_cut(xs, ys, cls, n_classes, (theta, theta + np.pi / 2)) \
            if cls.size >= 2 * MIN_PIXELS else (0.0, 0.0, 0.0)
        if cut_purity - purity[i] < MIN_CUT_GAIN:
            tally["majority"] += 1
            tally["cut_no_gain"] += 1
            rows.append({**base, "majority_class": names[majority_i[i]], "majority_share": round(float(purity[i]), 3),
                         "decision": "mixed, a cut would not help"})
            geoms.append(geom)
            continue

        first, second = halves(geom, angle, offset)
        if not all(field_like(part, MIN_SPLIT_WIDTH_M) for part in (first, second)):
            tally["majority"] += 1
            tally["cut_sliver"] += 1
            rows.append({**base, "majority_class": names[majority_i[i]], "majority_share": round(float(purity[i]), 3),
                         "decision": "mixed, the cut would leave a sliver"})
            geoms.append(geom)
            continue

        keep = (xs * np.cos(angle) + ys * np.sin(angle)) > offset
        made = False
        for part, side in ((first, cls[keep]), (second, cls[~keep])):
            if part.is_empty or side.size < MIN_PIXELS:
                continue
            share = np.bincount(side, minlength=n_classes)
            for piece in (part.geoms if part.geom_type == "MultiPolygon" else [part]):
                if piece.is_empty or piece.area < SLIVER_SQM:
                    continue
                rows.append({**base, "origin": "split", "n_classified": int(side.size),
                             "n_pixels": int(side.size),
                             "target_pixel_share": round(float(share[target_i] / side.size), 3),
                             "majority_class": names[int(share.argmax())],
                             "majority_share": round(float(share.max() / side.size), 3),
                             "decision": f"cut, purity {purity[i]:.2f} -> {cut_purity:.2f}"})
                geoms.append(piece)
                made = True
        if made:
            tally["cut"] += 1
        else:
            tally["majority"] += 1
            rows.append({**base, "majority_class": names[majority_i[i]], "majority_share": round(float(purity[i]), 3),
                         "decision": "mixed, the cut produced nothing"})
            geoms.append(geom)

    out = gpd.GeoDataFrame(rows, geometry=geoms, crs=frame.crs)
    log.info("labelling: %s (%.0f s)", tally, time.time() - t0)
    return out, tally


def derived_polygons(claimed: np.ndarray, class_path: Path, names: list[str], codes: list[int],
                     crs, target: str) -> tuple[gpd.GeoDataFrame, dict]:
    """Polygons for classified ground no polygon claims, regularised so they read as fields.

    An opening first, which is a repair rather than a rejection: the tentacles come off and whatever
    solid core is left survives as the field. Then the area floor on what is left, then two shape
    gates. Most of what vanishes is the strip of disagreement between a traced boundary and a 10 m
    raster, which is itself evidence that the delineation is the better geometry.
    """
    t0 = time.time()
    with rasterio.open(class_path) as ds:
        cls = ds.read(1)
        transform, raster_crs = ds.transform, ds.crs
    free = (cls > 0) & (claimed == 0)
    stats = {"unclaimed_acres": round(float(free.sum() * abs(transform.a * transform.e) / SQM_PER_ACRE), 1)}
    if not free.any():
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs), stats

    records = [(to_shape(geom), int(value)) for geom, value in
               rfeatures.shapes(np.where(free, cls, 0), mask=free, transform=transform)]
    frame = gpd.GeoDataFrame({"class_code": [v for _, v in records]},
                             geometry=[g for g, _ in records], crs=raster_crs).to_crs(crs)
    stats["blocks"] = len(frame)

    radius = MIN_FIELD_WIDTH_M / 2.0
    frame = frame.assign(geometry=frame.geometry.buffer(-radius).buffer(radius))
    frame = frame[~frame.geometry.is_empty & frame.geometry.is_valid].explode(index_parts=False)
    frame = frame[frame.geom_type == "Polygon"].reset_index(drop=True)
    stats["after_opening"] = len(frame)
    frame = frame[frame.area >= MIN_DERIVED_ACRES * SQM_PER_ACRE].reset_index(drop=True)
    stats["above_area_floor"] = len(frame)
    if frame.empty:
        log.info("derived: %d blocks, none survived the opening and the area floor", stats["blocks"])
        return frame.assign(**{c: [] for c in ("origin", "decision")}), stats

    rect = pd.Series(shapely.area(shapely.minimum_rotated_rectangle(frame.geometry.to_numpy())),
                     index=frame.index).replace(0, np.nan)
    keep = (frame.area / rect >= MIN_RECTANGULARITY) & (compactness(frame) >= MIN_COMPACTNESS_DERIVED)
    stats["field_shaped"] = int(keep.sum())
    frame = frame[keep].reset_index(drop=True)
    # A rasterised edge is a staircase; simplifying at half a pixel takes the steps off without moving
    # the boundary anywhere a reviewer would notice.
    pixel_m = abs(transform.a)
    frame["geometry"] = frame.geometry.simplify(pixel_m / 2.0, preserve_topology=True)
    frame = frame[~frame.geometry.is_empty & frame.geometry.is_valid].reset_index(drop=True)
    code_to_name = dict(zip(codes, names))
    frame["majority_class"] = frame["class_code"].map(code_to_name)
    frame["majority_share"] = 1.0
    frame["target_pixel_share"] = (frame["majority_class"] == target).astype(float)
    frame["n_pixels"] = np.round(frame.area / (pixel_m ** 2)).astype("int64")
    frame["n_classified"] = frame["n_pixels"]
    frame["classified_share"] = 1.0
    frame["origin"] = "derived"
    frame["decision"] = "no delineation polygon here"
    frame["source_fid"] = -1
    stats["acres"] = round(float(frame.area.sum() / SQM_PER_ACRE), 1)
    log.info("derived: %d blocks (%.0f acres unclaimed) -> %d after opening, %d above %.2f acres, "
             "%d field-shaped holding %.0f acres (%.0f s)", stats["blocks"], stats["unclaimed_acres"],
             stats["after_opening"], stats["above_area_floor"], MIN_DERIVED_ACRES, stats["field_shaped"],
             stats["acres"], time.time() - t0)
    return frame.drop(columns="class_code"), stats


def tidy(frame: gpd.GeoDataFrame, out_crs=4326) -> gpd.GeoDataFrame:
    """Final pass: tails off, no lines, valid geometry, no overlapping area, traced ahead of derived.

    Cleaning in the metric CRS and reprojecting afterwards looks equivalent and is not: converting
    metres to degrees rounds coordinates and the rounding reopens what was just closed. So the
    overlap pass runs in the CRS the file is written in.
    """
    metric_crs = frame.crs
    frame = frame.copy()
    frame["geometry"] = despike(frame.geometry)
    frame = frame[frame.geometry.notna()]
    frame = explode_fully(frame)
    frame, _ = drop_lines(frame)

    frame = explode_fully(frame.to_crs(out_crs))
    frame["geometry"] = shapely.set_precision(frame.geometry.to_numpy(), GRID)
    frame = explode_fully(frame)

    metric_area = frame.to_crs(metric_crs).area.to_numpy()
    span = float(metric_area.max()) + 1.0 if metric_area.size else 1.0
    # Traced geometry first, smaller before larger; derived polygons last, so a boundary the class map
    # invented can never cut one that was drawn on imagery.
    rank = metric_area + (frame["origin"].to_numpy() == "derived") * span
    geoms, _ = resolve_overlaps(frame, priority=rank)
    frame = explode_fully(frame.assign(geometry=geoms))
    frame["geometry"] = shapely.set_precision(frame.geometry.to_numpy(), GRID)
    frame = explode_fully(frame)

    metric_area = frame.to_crs(metric_crs).area.to_numpy()
    small_derived = (frame["origin"].to_numpy() == "derived") & (metric_area < MIN_DERIVED_ACRES * SQM_PER_ACRE)
    frame = frame[~small_derived & (metric_area > SLIVER_SQM)].reset_index(drop=True)
    log.info("tidy: %d polygons, %.0f acres, %d invalid",
             len(frame), frame.to_crs(metric_crs).area.sum() / SQM_PER_ACRE, int((~frame.geometry.is_valid).sum()))
    return frame


# ------------------------------------------------------------------ the field-level model

def field_means(frame: gpd.GeoDataFrame, cfg: dict, train_run: Path, map_run: Path,
                cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(mean feature per polygon, pixels used), read chunk by chunk from the map run's files.

    Rain flags come from the TRAINING run: rain is averaged over each run's AOI, so a different AOI
    can flag different dates as wet, which would make bins pick different acquisitions than the model
    was trained on.
    """
    t0 = time.time()
    fs = set_from_config(cfg)
    _, gd, tracks, season_start, end = run_context(map_run)
    rain_mm = float((cfg.get("analysis") or {}).get("rain_mm_24h", 5))
    plans = {tr: plans_for(map_run, tr, [fs], wet_flags(train_run, tr, rain_mm), season_start, end)[0] for tr in tracks}
    man = manifest.read_manifest(map_run)
    n = len(frame)
    total = np.zeros((n + 1, len(cols)))
    used = np.zeros(n + 1)
    tree = shapely.STRtree(frame.geometry.values)
    fids = frame["fid"].to_numpy()
    grid_transform = Affine(*gd["transform"])
    for k, ch in enumerate(gd["chunks"], 1):
        h, w = ch["height"], ch["width"]
        win = Window(ch["col_off"], ch["row_off"], w, h)
        idx = tree.query(shapely.box(*rasterio.windows.bounds(win, grid_transform)))
        if idx.size == 0:
            continue
        fid_img = rfeatures.rasterize(zip(frame.geometry.values[idx], fids[idx]), out_shape=(h, w),
                                      transform=window_transform(win, grid_transform), fill=0, dtype="int32")
        inside = fid_img.reshape(-1) > 0
        if not inside.any():
            continue
        rr, cc = np.indices((h, w)).reshape(2, -1)
        feats = {}
        for tr in tracks:
            row = man[(man.track_id == tr) & (man.chunk_name == ch["name"]) & man.state.isin(OK_STATES)]
            if row.empty:
                continue
            paths = [map_run / p for p in row.local_paths.iloc[0].split(";") if p]
            feats.update(track_features(10 ** (read_chunk_array(paths, ch, gd) / 10), rr, cc, tr, plans[tr]))
        if not all(c in feats for c in cols):
            continue
        values = np.column_stack([feats[c] for c in cols])[inside]
        here = fid_img.reshape(-1)[inside]
        good = np.isfinite(values).all(axis=1)
        if good.any():
            np.add.at(total, here[good], values[good])
            np.add.at(used, here[good], 1)
        if k % 10 == 0 or k == len(gd["chunks"]):
            log.info("  field features: chunk %d/%d (%.0f s)", k, len(gd["chunks"]), time.time() - t0)
    return total[fids], used[fids]


def apply_field_model(frame: gpd.GeoDataFrame, cfg: dict, train_run: Path, map_run: Path,
                      model_path: Path, target: str) -> gpd.GeoDataFrame:
    """Second, independent label per polygon: the field-level model on the mean of its pixel features."""
    import joblib

    bundle = joblib.load(model_path)
    cols, labels, threshold = bundle["columns"], bundle["labels"], bundle.get("threshold")
    if bundle.get("stat", "mean") != "mean":
        log.warning("the field model was fitted on the %s of each field; this step supplies the mean",
                    bundle.get("stat"))
    total, used = field_means(frame, cfg, train_run, map_run, cols)
    have = used > 0
    pred = np.full(len(frame), None, dtype=object)
    prob = np.full(len(frame), np.nan)
    if have.any():
        X = (total[have] / used[have, None]).astype("float32")
        p = bundle["model"].predict_proba(X)
        ti = labels.index(target)
        other = p.copy()
        other[:, ti] = -np.inf
        pred[have] = np.array(labels)[np.where(p[:, ti] >= threshold, ti, other.argmax(axis=1))]
        prob[have] = np.round(100 * p[:, ti], 1)
    frame = frame.copy()
    frame["field_model_class"] = pred
    frame["field_model_target_prob"] = prob
    frame["field_model_pixels"] = used.astype("int64")
    frame["agree"] = frame["majority_class"] == frame["field_model_class"]
    log.info("field model: %d of %d polygons had features; the two labels agree on %.1f %%",
             int(have.sum()), len(frame), 100 * float(frame["agree"].mean()))
    return frame


# ------------------------------------------------------------------ orchestration

def class_acres(frame: gpd.GeoDataFrame, column: str, metric_crs) -> dict:
    metric = frame.to_crs(metric_crs)
    by = metric.assign(acres=metric.area / SQM_PER_ACRE).groupby(column, dropna=False).acres.sum()
    return {("none" if pd.isna(k) else str(k)): round(float(v), 1) for k, v in by.items()}


def run(cfg: dict, map_run: Path, delineation: Path, class_raster: Path, prob_raster: Path | None = None,
        model_path: Path | None = None, train_cfg: dict | None = None, train_run: Path | None = None,
        name: str = "fields", simplify_m: float = SIMPLIFY_M) -> Path:
    """Whole pipeline; writes `fields_labelled.gpkg/.parquet` and `summary.json` and returns the folder."""
    t0 = time.time()
    train_cfg = train_cfg or cfg
    train_run = Path(train_run or map_run)
    target = (cfg.get("analysis") or {}).get("target_class")
    code_map = gcp_qc.codes_from_config(train_cfg)
    names = sorted(code_map)
    codes = [code_map[n] for n in names]
    out_dir = fields_dir(map_run, name)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(class_raster) as ds:
        raster_crs, shape = ds.crs, (ds.height, ds.width)
    fields = gpd.read_file(delineation, columns=[]).to_crs(raster_crs)   # geometry only: the attributes are not used
    log.info("delineation: %s, %d polygons, %.0f acres", Path(delineation).name, len(fields),
             fields.area.sum() / SQM_PER_ACRE)
    summary: dict = {"delineation": Path(delineation).name, "class_raster": Path(class_raster).name,
                     "target_class": target, "classes": names}

    t = time.time()
    fields, summary["simplify"] = simplify_outlines(fields[["geometry"]], simplify_m)
    summary["simplify"]["seconds"] = round(time.time() - t)

    t = time.time()
    frame, summary["repair"] = repair(fields[["geometry"]])
    summary["repair"]["seconds"] = round(time.time() - t)
    log.info("repair: done (%.0f s)", time.time() - t)
    del fields
    t = time.time()
    frame["geometry"] = despike(frame.geometry)
    frame = frame[frame.geometry.notna()]
    frame = explode_fully(frame)
    frame, line_stats = drop_lines(frame)
    summary["despike"] = {**line_stats, "polygons_out": len(frame),
                          "acres_out": round(float(frame.area.sum() / SQM_PER_ACRE), 1),
                          "seconds": round(time.time() - t)}
    frame = frame.reset_index(drop=True)
    frame["fid"] = np.arange(1, len(frame) + 1, dtype="int32")

    t = time.time()
    claimed = np.zeros(shape, dtype="uint8")
    zonal = zonal_counts(frame, class_raster, prob_raster, codes, claimed=claimed)
    labelled, tally = label_polygons(frame, zonal, class_raster, names, codes, target)
    prob_mean = np.where(zonal["prob_n"] > 0, zonal["prob_sum"] / np.maximum(zonal["prob_n"], 1), np.nan)
    labelled["target_prob_mean"] = np.round(prob_mean[labelled["source_fid"].to_numpy()], 1)
    summary["label"] = {**tally, "polygons_out": len(labelled), "seconds": round(time.time() - t)}

    derived, derived_stats = derived_polygons(claimed, class_raster, names, codes, frame.crs, target)
    summary["derived"] = derived_stats
    if len(derived):
        derived["target_prob_mean"] = np.nan
        labelled = gpd.GeoDataFrame(pd.concat([labelled, derived], ignore_index=True),
                                    geometry="geometry", crs=frame.crs)

    t = time.time()
    labelled = tidy(labelled, out_crs=4326)
    summary["tidy"] = {"polygons": len(labelled), "seconds": round(time.time() - t)}

    labelled = labelled.reset_index(drop=True)
    labelled["fid"] = np.arange(1, len(labelled) + 1, dtype="int32")
    metric = labelled.to_crs(raster_crs)
    labelled["area_acres"] = np.round(metric.area / SQM_PER_ACRE, 4)
    labelled["compactness"] = np.round(compactness(metric), 3)

    if model_path and Path(model_path).exists():
        t = time.time()
        labelled = apply_field_model(labelled.to_crs(raster_crs), cfg, train_run, Path(map_run),
                                     Path(model_path), target).to_crs(4326)
        summary["field_model"] = {"model": Path(model_path).name,
                                  "labelled": int(labelled["field_model_class"].notna().sum()),
                                  "agreement_pct": round(100 * float(labelled["agree"].mean()), 1),
                                  "seconds": round(time.time() - t)}
    else:
        log.info("no field-level model given: only the majority label is written")

    order = ["fid", "source_fid", "origin", "decision", "majority_class", "majority_share",
             "target_pixel_share", "target_prob_mean", "field_model_class", "field_model_target_prob",
             "field_model_pixels", "agree", "n_pixels", "n_classified", "classified_share",
             "area_acres", "compactness", "geometry"]
    labelled = labelled[[c for c in order if c in labelled.columns]]
    summary["output"] = {"polygons": len(labelled), "acres": round(float(labelled["area_acres"].sum()), 1),
                         "by_origin": {str(k): int(v) for k, v in labelled.groupby("origin").size().items()},
                         "acres_by_majority_class": class_acres(labelled, "majority_class", raster_crs)}
    if "field_model_class" in labelled.columns:
        summary["output"]["acres_by_field_model_class"] = class_acres(labelled, "field_model_class", raster_crs)
    summary["seconds"] = round(time.time() - t0)

    gpkg = out_dir / "fields_labelled.gpkg"
    labelled.to_file(gpkg, layer="fields", driver="GPKG")
    labelled.to_parquet(out_dir / "fields_labelled.parquet")
    run_meta.atomic_write_json(out_dir / "summary.json", summary)
    log.info("field labels: %d polygons, %.0f acres -> %s (%.0f s)",
             len(labelled), labelled["area_acres"].sum(), out_dir, time.time() - t0)
    return out_dir
