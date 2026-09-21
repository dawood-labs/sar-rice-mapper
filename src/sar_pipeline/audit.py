"""Stage 1 — Sentinel-1 availability audit (metadata only, no exports).

What the audit answers, before anything expensive is started
------------------------------------------------------------
1. Which Sentinel-1 images exist over the AOI in the season window, from which satellite (A/C/D),
   on which track (relative orbit + pass direction)?
2. How often does each track see the AOI, how complete is each acquisition, and where are the gaps?
3. Which track(s) should the time series use? (a recommendation only — the user decides)
4. Was it raining just before each acquisition? (rain wets soil and canopy and raises backscatter,
   which can be mistaken for crop growth)
5. How steep is the terrain? (tells how much radiometric terrain flattening will matter)

Design rules
------------
- Metadata is pulled in bulk (paged requests), never one request per image.
- All geometry maths (coverage, chunk overlap) happens locally with shapely/geopandas, so the audit
  scales to country-sized AOIs with tens of thousands of chunks.
- Areas are computed in an equal-area CRS (EPSG:6933), never in degrees.
- **No Earth Engine request grows with the AOI's size or complexity.** The full AOI geometry is never
  sent. Requests use coarse region blocks (groups of grid chunks, AOI part simplified in the grid CRS),
  every geometry sent is capped at MAX_GEOM_VERTICES, and requests are batched by feature count AND
  estimated payload size. Results per request stay below Earth Engine's 5,000-element limit.

See docs/developer/interfaces.md §6.2 for the file/column contract.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon, mapping, shape

from .config import aoi_path, audit_dir, grid_dir, latest_audit_dir
from .errors import AuditRequired
from .run_meta import (STALE_TMP_SECONDS, atomic_write_csv, atomic_write_text, clean_stale_tmp,
                       replace_with_retry)

log = logging.getLogger(__name__)

EQUAL_AREA_CRS = "EPSG:6933"
ACQ_GAP_MINUTES = 15              # slices of one acquisition start within this many minutes of each other
FULL_COVER_PCT = 99.0             # "chunk fully covered" threshold used in chunk_track_coverage

# Earth Engine request bounds (limits of the EE API, not of the local machine).
EE_BATCH = 200                    # max features (acquisitions / region blocks) per server request
MAX_REQUEST_BYTES = 2_000_000     # max estimated GeoJSON payload of the geometries in one request
MAX_GEOM_VERTICES = 2_000         # max vertices of any single geometry sent to Earth Engine
MAX_FEATURES_PER_REQUEST = 4_000  # max result features per request (EE refuses > 5,000 elements)
REGION_BLOCK_M = 50_000           # side of the coarse region blocks (groups of grid chunks)
SLOPE_BIN_DEG = 0.1               # slope histogram bin width; percentiles are interpolated inside a bin

SLICE_COLUMNS = ["system_index", "track_id", "relative_orbit", "pass", "platform", "datetime_utc",
                 "slice_number", "polarisations", "resolution_meters"]
ACQ_COLUMNS = ["acquisition_id", "track_id", "relative_orbit", "pass", "datetime_utc", "date_utc", "date_local",
               "platforms", "n_slices", "system_indexes", "aoi_coverage_pct", "mean_incidence_angle",
               "coverage_basis"]
SUMMARY_COLUMNS = ["track_id", "relative_orbit", "pass", "n_acq", "n_acq_ok_coverage", "first_date", "last_date",
                   "median_gap_days", "max_gap_days", "platforms", "mean_aoi_coverage_pct", "mean_incidence_angle"]
COVERAGE_COLUMNS = ["chunk_name", "track_id", "n_acq", "n_acq_full_cover", "mean_cover_pct"]
REC_COLUMNS = ["track_id", "role", "reason"]
GAP_COLUMNS = ["track_id", "gap_start", "gap_end", "gap_days", "events_in_gap"]
RAIN_COLUMNS = ["acquisition_id", "track_id", "datetime_utc", "rain_6h_mm", "rain_24h_mm", "source", "level",
                "chunk_name", "n_images_expected", "n_images_found", "rain_gap_images"]

# Hours represented by one image of each rain dataset (the band is a rate in mm/h).
RAIN_DATASETS = {
    "NASA/GPM_L3/IMERG_V07": {"band": "precipitation", "hours_per_image": 0.5},
    "JAXA/GPM_L3/GSMaP/v8/operational": {"band": "hourlyPrecipRate", "hours_per_image": 1.0},
}
COVERAGE_BASIS_FOOTPRINT = "footprint"
COVERAGE_BASIS_NO_BORDER_MASK = "footprint, border-noise mask not applied"


# ---------------------------------------------------------------- small helpers
def track_id(relative_orbit: int, pass_: str) -> str:
    """RO123_ASC / RO045_DSC (see interfaces §4)."""
    short = {"ASCENDING": "ASC", "DESCENDING": "DSC"}.get(str(pass_).upper(), str(pass_).upper()[:3])
    return f"RO{int(relative_orbit):03d}_{short}"


def _iso_utc(ts: pd.Timestamp) -> str:
    return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True)


def _tz(cfg: dict) -> ZoneInfo:
    return ZoneInfo(cfg.get("season", {}).get("timezone") or "UTC")


def _aoi_union_4326(cfg: dict):
    """Full AOI union for LOCAL geometry maths only (never sent to Earth Engine as is)."""
    aoi = gpd.read_file(aoi_path(cfg)).to_crs("EPSG:4326")
    return shapely.union_all(aoi.geometry.values)


def _as_geometry(aoi):
    if isinstance(aoi, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return shapely.union_all(aoi.to_crs("EPSG:4326").geometry.values if isinstance(aoi, gpd.GeoDataFrame)
                                 else aoi.to_crs("EPSG:4326").values)
    return aoi


def _grid_for(cfg: dict, gd: dict | None) -> dict:
    if gd is not None:
        return gd
    from . import grid as grid_mod

    return grid_mod.load_grid(cfg) if (grid_dir(cfg) / "grid_def.json").exists() else grid_mod.compute_grid_def(cfg)


def n_vertices(geom) -> int:
    return 0 if geom is None else int(shapely.get_num_coordinates(geom))


def geojson_bytes(geom) -> int:
    """Estimated request payload of one geometry (its GeoJSON text length)."""
    if geom is None or geom.is_empty:
        return 0
    return len(json.dumps(mapping(geom)))


def _polygonal(geom):
    """Valid polygonal part of `geom`: repairs self-intersections (make_valid) and drops points/lines.

    Returns the input object unchanged when it is already a valid (Multi)Polygon.
    """
    if geom is None or geom.is_empty:
        return geom
    if geom.is_valid and geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    fixed = shapely.make_valid(geom) if not geom.is_valid else geom
    polys, stack = [], [fixed]
    while stack:
        g = stack.pop()
        if g.is_empty:
            continue
        if g.geom_type == "Polygon":
            polys.append(g)
        elif g.geom_type in ("MultiPolygon", "GeometryCollection"):
            stack.extend(shapely.get_parts(g))
    out = shapely.union_all(polys) if polys else shapely.Polygon()
    if not out.is_valid:
        out = out.buffer(0)
    return out


def _accept(candidate, original, max_vertices: int, min_cover: float = 0.98):
    """Candidate if valid, within the vertex cap and covering (almost) all of the original area, else None."""
    candidate = _polygonal(candidate)
    if candidate is None or candidate.is_empty or n_vertices(candidate) > max_vertices:
        return None
    if original.area > 0 and shapely.intersection(candidate, original).area < min_cover * original.area:
        return None
    return candidate


def _log_inflation(candidate, original, how: str):
    if original.area > 0:
        factor = candidate.area / original.area
        if factor > 1.5:
            log.warning("geometry generalised by %s for an Earth Engine request: area inflated %.1fx", how, factor)
    return candidate


def bounded_geometry(geom, max_vertices: int | None = None):
    """A valid EPSG:4326 geometry with at most `max_vertices` vertices, as close to `geom` as needed.

    The input is first repaired (self-intersecting rings are made valid, non-polygon parts dropped).
    Then, in order, the first candidate that fits the vertex cap and still covers >= 98 % of the input:
    1. the geometry itself;
    2. simplified with growing tolerance (~50 m ... 5 km);
    3. morphological closing (buffer out, buffer back in) with growing distance, then simplified:
       nearby fragments merge, but clusters far apart stay separate, so the area grows only by the
       gaps between neighbouring parts;
    4. union of per-0.5°-cell convex hulls (can inflate a lot for scattered parts);
    5. the envelope.
    Any result more than 1.5x larger than the input is logged with its inflation factor. Used only for
    server-side reductions whose result is a regional mean or histogram.
    """
    max_vertices = MAX_GEOM_VERTICES if max_vertices is None else max_vertices
    geom = _polygonal(geom)
    if geom is None or geom.is_empty or n_vertices(geom) <= max_vertices:
        return geom

    tol = 0.0005
    while tol <= 0.05:
        found = _accept(shapely.simplify(geom, tol, preserve_topology=True), geom, max_vertices)
        if found is not None:
            return found
        tol *= 2

    for d in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
        grown = shapely.buffer(geom, d, quad_segs=2, join_style="mitre")
        closed = shapely.buffer(grown, -d, quad_segs=2, join_style="mitre")
        found = _accept(shapely.simplify(closed, d / 2, preserve_topology=True), geom, max_vertices)
        if found is not None:
            return _log_inflation(found, geom, f"closing at {d:g} deg")

    minx, miny, maxx, maxy = geom.bounds
    cell = 0.5
    pieces = []
    for i in range(int(math.floor(minx / cell)), int(math.ceil(maxx / cell))):
        for j in range(int(math.floor(miny / cell)), int(math.ceil(maxy / cell))):
            part = shapely.intersection(geom, shapely.box(i * cell, j * cell, (i + 1) * cell, (j + 1) * cell))
            if not part.is_empty and part.area > 0:
                pieces.append(part.convex_hull)
    if pieces:
        hulls = shapely.union_all(pieces)
        for candidate in (hulls, shapely.simplify(hulls, 0.01, preserve_topology=True)):
            found = _accept(candidate, geom, max_vertices, min_cover=0.95)
            if found is not None:
                return _log_inflation(found, geom, "per-cell convex hulls")
    envelope = shapely.box(*geom.bounds)
    log.warning("geometry with %d vertices generalised to its envelope for an Earth Engine request", n_vertices(geom))
    return _log_inflation(envelope, geom, "envelope")


def _ee_geometry(geom):
    """shapely (EPSG:4326) -> ee.Geometry, capped at MAX_GEOM_VERTICES."""
    import ee

    return ee.Geometry(mapping(bounded_geometry(geom)), proj="EPSG:4326", geodesic=False)


def plan_batches(records: list[dict], weight=None, max_weight: int | None = None,
                 max_bytes: int | None = None, size=None) -> list[list[dict]]:
    """Split records into deterministic request batches bounded by total weight AND payload bytes.

    weight(r): result features the record produces (default 1); size(r): payload bytes (default: the
    GeoJSON size of r["geometry"] + 256). A single record larger than a bound gets a batch of its own.
    Deterministic for the same input order, which the resume logic relies on.
    """
    max_weight = EE_BATCH if max_weight is None else max_weight
    max_bytes = MAX_REQUEST_BYTES if max_bytes is None else max_bytes
    weight = weight or (lambda r: 1)
    size = size or (lambda r: geojson_bytes(r.get("geometry")) + 256)
    batches, cur, cur_w, cur_b = [], [], 0, 0
    for r in records:
        w, b = int(weight(r)), int(size(r))
        if cur and (cur_w + w > max_weight or cur_b + b > max_bytes):
            batches.append(cur)
            cur, cur_w, cur_b = [], 0, 0
        cur.append(r)
        cur_w += w
        cur_b += b
    if cur:
        batches.append(cur)
    return batches


def _compute_features(fc, page_size: int = 1000) -> list[dict]:
    """All features of a server-side FeatureCollection, fetched in pages (never one huge getInfo)."""
    import ee

    features, token = [], None
    while True:
        params = {"expression": fc, "pageSize": page_size}
        if token:
            params["pageToken"] = token
        page = ee.data.computeFeatures(params)
        features.extend(page.get("features", []))
        token = page.get("nextPageToken") or page.get("next_page_token")
        if not token:
            return features


# ---------------------------------------------------------------- region blocks (bounded requests)
def region_blocks(cfg: dict, gd: dict) -> gpd.GeoDataFrame:
    """Coarse blocks of grid chunks (about REGION_BLOCK_M wide) with the AOI part inside each block.

    Columns: block_id, n_chunks, geometry = (AOI ∩ block) simplified in the grid CRS with a tolerance of
    5 × grid resolution, returned in EPSG:4326 and capped at MAX_GEOM_VERTICES. Server-side requests use
    these instead of the full AOI, so their size depends on the block, not on the AOI.
    """
    from .grid import iter_chunks, read_aoi

    chunks = list(iter_chunks(gd))
    if not chunks:
        return gpd.GeoDataFrame({"block_id": [], "n_chunks": []}, geometry=[], crs="EPSG:4326")
    k = max(1, int(round(REGION_BLOCK_M / (int(gd["chunk_px"]) * float(gd["res"])))))
    groups: dict[tuple[int, int], list[dict]] = {}
    for ch in chunks:
        groups.setdefault((ch["grid_row"] // k, ch["grid_col"] // k), []).append(ch)
    parts = read_aoi(cfg, gd["crs"]).geometry.values
    tree = shapely.STRtree(parts)
    tol = 5 * float(gd["res"])
    rows, geoms = [], []
    for (br, bc), chs in sorted(groups.items()):
        rect = shapely.box(min(c["xmin"] for c in chs), min(c["ymin"] for c in chs),
                           max(c["xmax"] for c in chs), max(c["ymax"] for c in chs))
        hits = tree.query(rect, predicate="intersects")
        if not len(hits):
            continue
        piece = shapely.union_all(shapely.intersection(parts[hits], rect))
        simple = shapely.simplify(piece, tol, preserve_topology=True)
        piece = simple if not simple.is_empty else piece
        if piece.is_empty or piece.area <= 0:
            continue
        rows.append({"block_id": f"b{br:04d}_{bc:04d}", "n_chunks": len(chs)})
        geoms.append(piece)
    out = gpd.GeoDataFrame(rows, geometry=geoms, crs=gd["crs"]).to_crs("EPSG:4326")
    out["geometry"] = [bounded_geometry(g) for g in out.geometry.values]
    return out


def region_filter_boxes(blocks: gpd.GeoDataFrame, pad_deg: float = 0.01) -> list:
    """Padded envelopes (EPSG:4326) of the region blocks: tiny, fixed-size geometries for filterBounds."""
    return [shapely.box(*shapely.buffer(shapely.box(*g.bounds), pad_deg).bounds) for g in blocks.geometry.values]


# ---------------------------------------------------------------- 1. slices
def _s1_collection(cfg: dict, region_ee, start: str | None = None, end: str | None = None):
    import ee

    s1 = cfg["s1"]
    col = (
        ee.ImageCollection(s1["collection"])
        .filterBounds(region_ee)
        .filterDate(start or cfg["season"]["start"], end or cfg["season"]["end"])
        .filter(ee.Filter.eq("instrumentMode", s1.get("instrument_mode", "IW")))
        .filter(ee.Filter.eq("resolution_meters", 10))
    )
    for pol in s1.get("pols", ["VV", "VH"]):
        col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
    return col


def _slices_from_features(features: list[dict]) -> gpd.GeoDataFrame:
    """Parse GeoJSON features (as returned by Earth Engine) into the s1_slices table + footprints.

    Kept free of Earth Engine calls so it can be unit-tested with synthetic payloads.
    """
    rows, geoms = [], []
    for f in features:
        p = f.get("properties", {}) or {}
        ts = pd.Timestamp(int(p["time_start"]), unit="ms", tz="UTC")
        pols = p.get("polarisations")
        rows.append(
            dict(
                system_index=p["system_index"],
                track_id=track_id(p["relative_orbit"], p["pass"]),
                relative_orbit=int(p["relative_orbit"]),
                pass_=p["pass"],
                platform=p.get("platform"),
                datetime_utc=_iso_utc(ts),
                slice_number=p.get("slice_number"),
                polarisations=";".join(pols) if isinstance(pols, (list, tuple)) else pols,
                resolution_meters=p.get("resolution_meters"),
            )
        )
        g = f.get("geometry")
        geom = shape(g) if g else None
        if geom is not None and geom.geom_type in ("LinearRing", "LineString"):
            geom = Polygon(geom.coords)
        geoms.append(geom)
    df = pd.DataFrame(rows, columns=[c if c != "pass" else "pass_" for c in SLICE_COLUMNS]).rename(columns={"pass_": "pass"})
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    return gdf.sort_values(["track_id", "datetime_utc", "system_index"]).reset_index(drop=True)


def list_slices(cfg: dict, start: str | None = None, end: str | None = None, gd: dict | None = None) -> gpd.GeoDataFrame:
    """Every Sentinel-1 GRD slice over the AOI in the season window (one row per Earth Engine image).

    Returns a GeoDataFrame (a pandas DataFrame with footprint geometry, EPSG:4326) with the
    `s1_slices.csv` columns. The server-side filter uses padded envelopes of the region blocks (fixed
    size, batched), so the request never carries the AOI geometry; the exact intersection with the AOI
    happens locally and slices that do not touch the AOI are dropped.
    """
    import ee

    blocks = region_blocks(cfg, _grid_for(cfg, gd))
    if blocks.empty:
        return _slices_from_features([])
    aoi = _aoi_union_4326(cfg)

    def to_feature(img):
        img = ee.Image(img)
        return ee.Feature(img.geometry(), {
            "system_index": img.get("system:index"),
            "time_start": img.get("system:time_start"),
            "relative_orbit": img.get("relativeOrbitNumber_start"),
            "pass": img.get("orbitProperties_pass"),
            "platform": img.get("platform_number"),
            "slice_number": img.get("sliceNumber"),
            "polarisations": img.get("transmitterReceiverPolarisation"),
            "resolution_meters": img.get("resolution_meters"),
        })

    records = [{"geometry": b} for b in region_filter_boxes(blocks)]
    features: dict[str, dict] = {}
    for batch in plan_batches(records):
        region = ee.FeatureCollection([ee.Feature(ee.Geometry(mapping(r["geometry"]), proj="EPSG:4326", geodesic=False))
                                       for r in batch])
        fc = ee.FeatureCollection(_s1_collection(cfg, region, start, end).map(to_feature))
        for f in _compute_features(fc):
            features.setdefault(f["properties"]["system_index"], f)  # blocks overlap: de-duplicate

    gdf = _slices_from_features(list(features.values()))
    if gdf.empty:
        return gdf
    keep = gdf.geometry.notna() & gdf.geometry.intersects(aoi)
    return gdf[keep].reset_index(drop=True)


# ---------------------------------------------------------------- 2. acquisitions
def assign_acquisition_ids(slices: pd.DataFrame, timezone_name: str = "UTC") -> pd.Series:
    """Acquisition id for every slice: same track + start within ACQ_GAP_MINUTES of the previous slice.

    Grouping by time proximity (not by date string) keeps a pass together even if it crosses
    midnight UTC, and keeps S1C/S1D passes on the same track one day apart as separate acquisitions.
    """
    if slices.empty:
        return pd.Series([], dtype=object, index=slices.index)
    t = _to_utc(slices["datetime_utc"].values)
    t.index = slices.index
    order = slices.assign(_t=t).sort_values(["track_id", "_t"]).index
    ids = pd.Series(index=slices.index, dtype=object)
    seen: dict[str, int] = {}
    prev_track, prev_t, current = None, None, None
    for i in order:
        tr, ti = slices.at[i, "track_id"], t.at[i]
        new = tr != prev_track or (ti - prev_t) > pd.Timedelta(minutes=ACQ_GAP_MINUTES)
        if new:
            base = f"{tr}_{ti.strftime('%Y%m%d')}"
            n = seen.get(base, 0) + 1
            seen[base] = n
            current = base if n == 1 else f"{base}_{n}"
            if n > 1:
                warnings.warn(f"Two acquisitions of {tr} on the same UTC date; second one named {current}")
        ids.at[i] = current
        prev_track, prev_t = tr, ti
    return ids


def coverage_basis(cfg: dict) -> str:
    """What `aoi_coverage_pct` is based on.

    Coverage is computed from the image footprints. When the extra border-noise mask is enabled
    (ard.border_noise_correction), the masked far/near-range strips are NOT subtracted, so the
    coverage then slightly overstates the usable area; the column says so.
    """
    return COVERAGE_BASIS_NO_BORDER_MASK if (cfg.get("ard") or {}).get("border_noise_correction") else COVERAGE_BASIS_FOOTPRINT


def group_acquisitions(slices: gpd.GeoDataFrame, aoi_4326, cfg: dict) -> gpd.GeoDataFrame:
    """One row per acquisition (all slices of one pass over the AOI).

    `aoi_coverage_pct` = area(union of slice footprints ∩ AOI) / area(AOI) * 100, in an equal-area CRS.
    The returned GeoDataFrame's geometry is (union footprint ∩ AOI) in EPSG:4326; it is used for
    server-side reductions (incidence angle, rain; capped by bounded_geometry) and dropped when
    written to CSV.
    """
    if "geometry" not in slices or slices.empty:
        if slices.empty:
            return gpd.GeoDataFrame(columns=ACQ_COLUMNS, geometry=[], crs="EPSG:4326")
        raise ValueError("group_acquisitions needs slice footprints (a GeoDataFrame from list_slices)")
    aoi = _as_geometry(aoi_4326)
    tz = _tz(cfg)
    basis = coverage_basis(cfg)
    ids = assign_acquisition_ids(slices)
    t = _to_utc(slices["datetime_utc"].values)
    t.index = slices.index
    s = slices.assign(acquisition_id=ids, _t=t)

    aoi_ea = gpd.GeoSeries([aoi], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).iloc[0]
    aoi_area = aoi_ea.area
    rows, geoms = [], []
    for acq_id, g in s.groupby("acquisition_id", sort=False):
        g = g.sort_values("_t")
        first = pd.Timestamp(g["_t"].iloc[0])
        if first.tzinfo is None:
            first = first.tz_localize("UTC")
        foot = shapely.union_all(g.geometry.values)
        inter = shapely.intersection(foot, aoi)
        inter_ea = gpd.GeoSeries([inter], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).iloc[0]
        cov = 100.0 * inter_ea.area / aoi_area if aoi_area > 0 else float("nan")
        rows.append(
            dict(
                acquisition_id=acq_id,
                track_id=g["track_id"].iloc[0],
                relative_orbit=int(g["relative_orbit"].iloc[0]),
                pass_=g["pass"].iloc[0],
                datetime_utc=_iso_utc(first),
                date_utc=first.strftime("%Y-%m-%d"),
                date_local=first.tz_convert(tz).strftime("%Y-%m-%d"),
                platforms=";".join(sorted({str(p) for p in g["platform"] if pd.notna(p)})),
                n_slices=len(g),
                system_indexes=";".join(g["system_index"]),
                aoi_coverage_pct=round(min(100.0, cov), 3),
                mean_incidence_angle=float("nan"),
                coverage_basis=basis,
            )
        )
        geoms.append(inter)
    df = pd.DataFrame(rows).rename(columns={"pass_": "pass"})[ACQ_COLUMNS]
    out = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    return out.sort_values(["datetime_utc", "track_id"]).reset_index(drop=True)


def _images_by_index(cfg: dict, system_indexes: list[str], times: list[str]):
    """The S1 images with the given system:index values (date-filtered first, which EE indexes)."""
    import ee

    t = pd.to_datetime(pd.Series(times), utc=True)
    start = (t.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    end = (t.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    return (ee.ImageCollection(cfg["s1"]["collection"]).filterDate(start, end)
            .filter(ee.Filter.inList("system:index", sorted(set(system_indexes)))))


def _angle_batch(cfg: dict, records: list[dict]) -> dict[str, float | None]:
    """One Earth Engine request: mean `angle` for each record {acquisition_id, system_indexes, datetime_utc, geometry}."""
    import ee

    base = _images_by_index(cfg, [s for r in records for s in r["system_indexes"]], [r["datetime_utc"] for r in records])
    feats = [ee.Feature(_ee_geometry(r["geometry"]), {"acquisition_id": r["acquisition_id"],
                                                      "system_indexes": r["system_indexes"]}) for r in records]

    def reduce(f):
        f = ee.Feature(f)
        img = base.filter(ee.Filter.inList("system:index", f.get("system_indexes"))).select("angle").mosaic()
        val = img.reduceRegion(ee.Reducer.mean(), f.geometry(), scale=1000, bestEffort=True, maxPixels=1e9)
        return ee.Feature(None, {"acquisition_id": f.get("acquisition_id"), "angle": val.get("angle")})

    feats_out = _compute_features(ee.FeatureCollection(feats).map(reduce))
    return {f["properties"]["acquisition_id"]: f["properties"].get("angle") for f in feats_out}


def incidence_angles(cfg: dict, acquisitions: pd.DataFrame, partial_dir: Path | None = None) -> pd.Series:
    """Mean incidence angle (degrees) of each acquisition over (footprint ∩ AOI).

    Batched by count and payload, reduced server-side at 1 km (the angle band is smooth, so a coarse
    scale is exact enough and cheap). With `partial_dir`, each finished batch is saved there, so a
    crash resumes from the next batch instead of starting over.
    """
    if acquisitions.empty:
        return pd.Series(dtype=float, name="mean_incidence_angle")
    has_geom = isinstance(acquisitions, gpd.GeoDataFrame) and "geometry" in acquisitions
    aoi = None
    records = []
    for _, r in acquisitions.iterrows():
        if has_geom and r.geometry is not None and not r.geometry.is_empty:
            geom = r.geometry
        else:
            aoi = aoi if aoi is not None else bounded_geometry(_aoi_union_4326(cfg))
            geom = aoi
        records.append({"acquisition_id": r["acquisition_id"], "geometry": bounded_geometry(geom),
                        "datetime_utc": r["datetime_utc"], "system_indexes": str(r["system_indexes"]).split(";")})

    def run(batch):
        vals = _angle_batch(cfg, batch)
        return pd.DataFrame({"acquisition_id": list(vals), "angle": [v for v in vals.values()]})

    df = _batched(records, run, partial_dir, "incidence_angles", ["acquisition_id", "angle"])
    return pd.Series(pd.to_numeric(df["angle"], errors="coerce").to_numpy(), index=df["acquisition_id"],
                     name="mean_incidence_angle", dtype=float)


def _batched(records: list, run, partial_dir: Path | None, step: str, columns: list[str],
             weight=None, max_weight: int | None = None, size=None) -> pd.DataFrame:
    """Run `run(batch) -> DataFrame` over bounded batches (plan_batches), persisting each batch result.

    Batches are deterministic (same records, same order, same bounds), so a resumed run loads finished
    batch files and only requests the missing ones. Files are written atomically; a batch file that is
    unreadable or has other columns is recomputed.
    """
    batches = plan_batches(records, weight=weight, max_weight=max_weight, size=size)
    parts, cached = [], 0
    for b, batch in enumerate(batches):
        path = Path(partial_dir) / f"{step}_{b:05d}.csv" if partial_dir else None
        if path is not None and path.exists():
            try:
                df = pd.read_csv(path)
                if list(df.columns) == columns:
                    parts.append(df)
                    cached += 1
                    continue
            except Exception:  # noqa: BLE001 - unreadable partial -> recompute this batch
                pass
        t0 = time.monotonic()
        df = run(batch).reindex(columns=columns)
        log.debug("%s batch %d/%d: %d records in %.1f s", step, b + 1, len(batches), len(batch), time.monotonic() - t0)
        if path is not None:
            atomic_write_csv(df, path)
        parts.append(df)
    if cached:
        log.info("%s: skipped %d/%d batches: already fetched", step, cached, len(batches))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------- 3. summaries
def _gap_days(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates)).sort_values().dt.normalize()
    return d.diff().dt.days.dropna().to_numpy()


def track_summary(acquisitions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-track counts, date range, revisit gaps (UTC calendar days), platforms, coverage."""
    min_cov = float(cfg["audit"].get("min_aoi_coverage_pct", 90))
    rows = []
    for tid, g in acquisitions.groupby("track_id"):
        gaps = _gap_days(g["date_utc"])
        angles = pd.to_numeric(g.get("mean_incidence_angle"), errors="coerce")
        platforms = sorted({p for s in g["platforms"].astype(str) for p in s.split(";") if p and p != "nan"})
        rows.append(dict(
            track_id=tid,
            relative_orbit=int(g["relative_orbit"].iloc[0]),
            pass_=g["pass"].iloc[0],
            n_acq=len(g),
            n_acq_ok_coverage=int((g["aoi_coverage_pct"] >= min_cov).sum()),
            first_date=g["date_utc"].min(),
            last_date=g["date_utc"].max(),
            median_gap_days=float(np.median(gaps)) if len(gaps) else float("nan"),
            max_gap_days=float(gaps.max()) if len(gaps) else float("nan"),
            platforms=";".join(platforms),
            mean_aoi_coverage_pct=round(float(g["aoi_coverage_pct"].mean()), 3),
            mean_incidence_angle=round(float(angles.mean()), 3) if angles is not None and angles.notna().any() else float("nan"),
        ))
    if not rows:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    return pd.DataFrame(rows).rename(columns={"pass_": "pass"})[SUMMARY_COLUMNS].sort_values(
        ["n_acq_ok_coverage", "n_acq"], ascending=False).reset_index(drop=True)


def _events(cfg: dict) -> list[tuple[pd.Timestamp, str]]:
    evs = cfg.get("audit", {}).get("constellation_events") or []
    return sorted((pd.Timestamp(e["date"]), str(e.get("note", ""))) for e in evs)


def detect_gaps(acquisitions: pd.DataFrame, cfg: dict, now: datetime | None = None) -> pd.DataFrame:
    """Gaps longer than audit.max_gap_days, per track, in UTC calendar days.

    Includes the leading gap (season start -> first acquisition) and the trailing gap
    (last acquisition -> min(season end, today)); the trailing gap simply means "no newer data yet"
    when the season is still running. Constellation events falling inside a gap are listed.
    """
    max_gap = int(cfg["audit"].get("max_gap_days", 12))
    season_start = pd.Timestamp(cfg["season"]["start"])
    season_end_incl = pd.Timestamp(cfg["season"]["end"]) - pd.Timedelta(days=1)
    today = pd.Timestamp((now or datetime.now(timezone.utc)).date())
    horizon = min(season_end_incl, today)
    events = _events(cfg)

    rows = []
    for tid, g in acquisitions.groupby("track_id"):
        dates = sorted(pd.to_datetime(g["date_utc"]).dt.normalize().unique())
        bounds = [(season_start, dates[0], True)] if dates else []
        bounds += [(a, b, False) for a, b in zip(dates[:-1], dates[1:])]
        if dates and horizon > dates[-1]:
            bounds.append((dates[-1], horizon, True))
        for a, b, edge in bounds:
            days = int((pd.Timestamp(b) - pd.Timestamp(a)).days)
            if days > max_gap:
                inside = [f"{d.strftime('%Y-%m-%d')} {n}".strip() for d, n in events if a <= d <= b]
                rows.append(dict(track_id=tid, gap_start=pd.Timestamp(a).strftime("%Y-%m-%d"),
                                 gap_end=pd.Timestamp(b).strftime("%Y-%m-%d"), gap_days=days,
                                 events_in_gap="; ".join(inside)))
    return pd.DataFrame(rows, columns=GAP_COLUMNS)


def chunk_track_coverage(cfg: dict, gd: dict, footprints: gpd.GeoDataFrame) -> pd.DataFrame:
    """For every chunk and track: how many acquisitions touch the chunk's AOI part, and how many cover it fully.

    Coverage of a chunk = area(acquisition footprint ∩ chunk ∩ AOI) / area(chunk ∩ AOI).
    Uses a spatial join (STRtree) + vectorised intersections in the grid CRS, so it scales to
    10k+ chunks × thousands of acquisitions.
    """
    from .grid import iter_chunks, read_aoi

    if footprints is None or footprints.empty or not gd.get("chunks"):
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    crs = gd["crs"]
    fp = footprints.copy()
    if "acquisition_id" not in fp:
        fp["acquisition_id"] = assign_acquisition_ids(fp)
    acq = fp.dissolve(by="acquisition_id", aggfunc={"track_id": "first"}).reset_index().to_crs(crs)

    aoi_parts = read_aoi(cfg, crs).geometry.values
    aoi_tree = shapely.STRtree(aoi_parts)
    names, targets = [], []
    for ch in iter_chunks(gd):
        cell = shapely.box(ch["xmin"], ch["ymin"], ch["xmax"], ch["ymax"])
        hits = aoi_tree.query(cell, predicate="intersects")
        part = shapely.union_all(shapely.intersection(aoi_parts[hits], cell)) if len(hits) else shapely.Polygon()
        if part.area > 0:
            names.append(ch["name"])
            targets.append(part)
    if not targets:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    targets = np.array(targets, dtype=object)
    areas = shapely.area(targets)

    acq_geoms = acq.geometry.values
    tree = shapely.STRtree(acq_geoms)
    t_idx, a_idx = tree.query(targets, predicate="intersects")
    if len(t_idx) == 0:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)
    covered = shapely.covers(acq_geoms[a_idx], targets[t_idx])
    cover_pct = np.full(len(t_idx), 100.0)
    partial = ~covered
    if partial.any():
        inter = shapely.intersection(targets[t_idx[partial]], acq_geoms[a_idx[partial]])
        cover_pct[partial] = 100.0 * shapely.area(inter) / areas[t_idx[partial]]
    pairs = pd.DataFrame({
        "chunk_name": np.array(names)[t_idx],
        "track_id": acq["track_id"].to_numpy()[a_idx],
        "cover": cover_pct,
    })
    pairs = pairs[pairs["cover"] > 0]
    out = pairs.groupby(["chunk_name", "track_id"]).agg(
        n_acq=("cover", "size"),
        n_acq_full_cover=("cover", lambda c: int((c >= FULL_COVER_PCT).sum())),
        mean_cover_pct=("cover", "mean"),
    ).reset_index()
    out["mean_cover_pct"] = out["mean_cover_pct"].round(3)
    return out[COVERAGE_COLUMNS]


def recommend_tracks(summary: pd.DataFrame, coverage: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    """Suggest track roles (the user confirms in config s1.tracks; nothing is auto-applied).

    primary   = most acquisitions with AOI coverage >= audit.min_aoi_coverage_pct (tie: smaller max gap)
    secondary = any other track with >= 50 % of the primary's count
    unused    = the rest
    """
    if summary.empty:
        return pd.DataFrame(columns=REC_COLUMNS)
    min_cov = cfg["audit"].get("min_aoi_coverage_pct", 90)
    s = summary.assign(_gap=summary["max_gap_days"].fillna(np.inf)).sort_values(
        ["n_acq_ok_coverage", "_gap", "track_id"], ascending=[False, True, True]).reset_index(drop=True)
    primary = s.iloc[0]
    n_primary = int(primary["n_acq_ok_coverage"])
    n_chunks = coverage["chunk_name"].nunique() if coverage is not None and not coverage.empty else 0

    rows = []
    for i, r in s.iterrows():
        n_ok = int(r["n_acq_ok_coverage"])
        gap = "n/a" if not np.isfinite(r["_gap"]) else f"{r['_gap']:.0f} d"
        base = f"{n_ok} of {int(r['n_acq'])} acquisitions cover >= {min_cov}% of the AOI; max gap {gap}"
        if i == 0 and n_ok > 0:
            role, reason = "primary", "most well-covering acquisitions; " + base
        elif n_primary > 0 and n_ok >= 0.5 * n_primary:
            role, reason = "secondary", f">= 50% of the primary's {n_primary}; " + base
        else:
            role, reason = "unused", base
        if n_chunks:
            c = coverage[coverage["track_id"] == r["track_id"]]
            never_full = n_chunks - int((c["n_acq_full_cover"] > 0).sum())
            if never_full > 0:
                reason += f"; {never_full} of {n_chunks} chunks never fully covered by this track"
        rows.append(dict(track_id=r["track_id"], role=role, reason=reason))
    return pd.DataFrame(rows, columns=REC_COLUMNS)


# ---------------------------------------------------------------- 4. rain
def rain_source_for(t: pd.Timestamp, coverage_end: dict[str, pd.Timestamp | None], order: list[str]) -> str | None:
    """First dataset (in `order`) whose data reaches timestamp t, or None."""
    for ds in order:
        end = coverage_end.get(ds)
        if end is not None and end >= t:
            return ds
    return None


def expected_rain_images(t: pd.Timestamp, hours: float, step_hours: float) -> int:
    """Number of dataset images overlapping the window [t - hours, t).

    Rain datasets are regular grids of images (IMERG every 30 min, GSMaP every hour) starting at
    multiples of the step since the Unix epoch. An image k covers [k·step, (k+1)·step); it overlaps
    the window when k·step < t and (k+1)·step > t - hours.
    """
    step = float(step_hours) * 3600.0
    end = pd.Timestamp(t).timestamp()
    start = end - float(hours) * 3600.0
    return int(math.ceil(end / step) - math.floor(start / step))


def rain_window_weights(t: pd.Timestamp, hours: float, images: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[float]:
    """Share of each image [start, end) that falls inside [t - hours, t) (0 … 1).

    Pure-python mirror of the server-side weighting in `_rain_image`: images fully inside count fully,
    the partly overlapping first/last images count by their overlap fraction, images outside count 0.
    Rain (mm) = Σ rate (mm/h) × image duration (h) × weight.
    """
    t_end = pd.Timestamp(t).timestamp()
    t_start = t_end - float(hours) * 3600.0
    out = []
    for s, e in images:
        s_, e_ = pd.Timestamp(s).timestamp(), pd.Timestamp(e).timestamp()
        overlap = max(0.0, min(e_, t_end) - max(s_, t_start))
        out.append(overlap / (e_ - s_) if e_ > s_ else 0.0)
    return out


def _dataset_end(dataset: str, start: str, end: str) -> pd.Timestamp | None:
    """End time of the newest image of a rain dataset inside [start, end) (None if no image)."""
    import ee

    col = ee.ImageCollection(dataset).filterDate(start, end)
    info = ee.Dictionary({"n": col.size(), "t": col.aggregate_max("system:time_start")}).getInfo()
    if not info.get("n"):
        return None
    hours = RAIN_DATASETS.get(dataset, {}).get("hours_per_image", 1.0)
    return pd.Timestamp(int(info["t"]), unit="ms", tz="UTC") + pd.Timedelta(hours=hours)


def _rain_image(ds: str, t: pd.Timestamp, hours: int):
    """(rain image in mm for [t-hours, t), number of dataset images found) — server-side.

    Selects every image overlapping the window (starts from the step boundary at or before t-hours,
    ends before t) and weights each by its overlap fraction (see rain_window_weights). An empty
    selection gives a fully masked image (reduces to null -> NaN) instead of a 0-band image that
    would crash the whole request.
    """
    import ee

    meta = RAIN_DATASETS[ds]
    step_ms = int(meta["hours_per_image"] * 3600 * 1000)
    t_ms = int(pd.Timestamp(t).timestamp() * 1000)
    w0 = t_ms - int(hours * 3600 * 1000)
    first = (w0 // step_ms) * step_ms
    band = meta["band"]
    col = ee.ImageCollection(ds).filterDate(ee.Date(first), ee.Date(t_ms)).select(band)

    def weigh(img):
        img = ee.Image(img)
        s = ee.Number(img.get("system:time_start"))
        e = ee.Number(ee.Algorithms.If(img.get("system:time_end"), img.get("system:time_end"), s.add(step_ms)))
        overlap = e.min(t_ms).subtract(s.max(w0)).max(0)
        duration_ms = e.subtract(s).max(1)
        mm_factor = overlap.divide(duration_ms).multiply(duration_ms.divide(3600 * 1000))  # weight × duration (h)
        return img.multiply(mm_factor).rename(band)

    empty = ee.Image.constant(0).rename(band).updateMask(ee.Image.constant(0))
    n = col.size()
    total = ee.Image(ee.Algorithms.If(n.gt(0), col.map(weigh).sum().rename(band), empty))
    return total, n


def _rain_batch(cfg: dict, records: list[dict], level: str, gd: dict | None) -> list[dict]:
    """One Earth Engine request for a batch of rain records.

    level "aoi":   records {acquisition_id, datetime_utc, source, geometry} -> one feature each.
    level "chunk": records {acquisition_id, datetime_utc, source, chunks: [chunk dicts]} -> one feature
                   per chunk (callers bound the chunks per request, see rain_flags).
    Returns rows with rain_6h_mm, rain_24h_mm (NaN when no data), n_images_found (24 h window).
    """
    import ee

    parts = []
    for r in records:
        t = pd.Timestamp(r["datetime_utc"])
        img6, _ = _rain_image(r["source"], t, 6)
        img24, n24 = _rain_image(r["source"], t, 24)
        both = img6.rename("rain_6h_mm").addBands(img24.rename("rain_24h_mm"))
        props = {"acquisition_id": r["acquisition_id"], "source": r["source"], "n_images_found": n24}
        if level == "chunk":
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Rectangle([c["xmin"], c["ymin"], c["xmax"], c["ymax"]], proj=gd["crs"], geodesic=False),
                           {"chunk_name": c["name"]})
                for c in r["chunks"]
            ])
            red = both.reduceRegions(fc, ee.Reducer.mean(), scale=1000, tileScale=2)
            parts.append(red.map(lambda f, p=props: ee.Feature(f).set(p)))
        else:
            val = both.reduceRegion(ee.Reducer.mean(), _ee_geometry(r["geometry"]), scale=1000, bestEffort=True, maxPixels=1e9)
            parts.append(ee.FeatureCollection([ee.Feature(None, val).set(props)]))
    return parse_rain_features(_compute_features(ee.FeatureCollection(parts).flatten()), level)


def parse_rain_features(features: list[dict], level: str) -> list[dict]:
    """Rain result features -> rows. No dataset image in the window -> NaN rain and source 'none'."""
    out = []
    for f in features:
        p = f.get("properties", {}) or {}
        found = p.get("n_images_found")
        found = int(found) if found is not None else None
        no_data = found == 0
        out.append(dict(
            acquisition_id=p["acquisition_id"],
            source="none" if no_data else p.get("source"),
            rain_6h_mm=float("nan") if no_data else _round_or_nan(p.get("rain_6h_mm")),
            rain_24h_mm=float("nan") if no_data else _round_or_nan(p.get("rain_24h_mm")),
            chunk_name=p.get("chunk_name", "") if level == "chunk" else "",
            n_images_found=found if found is not None else float("nan"),
        ))
    return out


def _chunks_touched(gd: dict, geom_4326) -> list[dict]:
    """Grid chunks whose rectangle intersects an EPSG:4326 geometry (local, STRtree)."""
    from .grid import iter_chunks

    chunks = list(iter_chunks(gd))
    if not chunks or geom_4326 is None or geom_4326.is_empty:
        return []
    rects = np.array([shapely.box(c["xmin"], c["ymin"], c["xmax"], c["ymax"]) for c in chunks], dtype=object)
    g = gpd.GeoSeries([geom_4326], crs="EPSG:4326").to_crs(gd["crs"]).iloc[0]
    idx = shapely.STRtree(rects).query(g, predicate="intersects")
    return [chunks[i] for i in sorted(idx)]


def rain_flags(cfg: dict, acquisitions: pd.DataFrame, gd: dict | None = None,
               partial_dir: Path | None = None) -> pd.DataFrame:
    """Rain (mm) in the 6 h and 24 h before each acquisition.

    level "aoi":   mean over (footprint ∩ AOI) — one value per acquisition.
    level "chunk": mean over each chunk the acquisition covers (needs gd) — for large AOIs where one
                   mean would mix places hundreds of km apart. Chunks are split into requests of at
                   most MAX_FEATURES_PER_REQUEST results.
    The primary dataset is used when its data reaches the acquisition time, else the fallback;
    `source` records which ("none" + NaN when neither covers it or the window has no images).
    `n_images_expected` / `n_images_found` count the dataset images of the 24 h window;
    `rain_gap_images` > 0 means images are missing inside the window (the sum under-counts).
    With `partial_dir`, finished batches are saved so a crash resumes from the next batch.
    """
    rcfg = cfg.get("rain", {}) or {}
    if not rcfg.get("enabled", True) or acquisitions.empty:
        return pd.DataFrame(columns=RAIN_COLUMNS)
    level = rcfg.get("level", "aoi")
    if level == "chunk" and gd is None:
        raise ValueError("rain.level == 'chunk' needs the grid definition (gd)")
    order = [d for d in [rcfg.get("primary"), rcfg.get("fallback")] if d]
    unknown = [d for d in order if d not in RAIN_DATASETS]
    if unknown:
        raise ValueError(f"Unsupported rain dataset(s) {unknown}; supported: {list(RAIN_DATASETS)}")

    times = _to_utc(acquisitions["datetime_utc"].values)
    q_start = (times.min() - pd.Timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    q_end = (times.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    coverage_end = {ds: _dataset_end(ds, q_start, q_end) for ds in order}

    has_geom = isinstance(acquisitions, gpd.GeoDataFrame) and "geometry" in acquisitions
    info = {r["acquisition_id"]: r for _, r in acquisitions.iterrows()}
    aoi = None
    records, pending, expected = [], [], {}
    for (_, r), t in zip(acquisitions.iterrows(), times):
        ds = rain_source_for(t, coverage_end, order)
        if ds is None:
            records.append(dict(acquisition_id=r["acquisition_id"], source="none", rain_6h_mm=float("nan"),
                                rain_24h_mm=float("nan"), chunk_name="", n_images_found=float("nan")))
            continue
        expected[r["acquisition_id"]] = expected_rain_images(t, 24, RAIN_DATASETS[ds]["hours_per_image"])
        if has_geom and r.geometry is not None and not r.geometry.is_empty:
            geom = r.geometry
        else:
            aoi = aoi if aoi is not None else _aoi_union_4326(cfg)
            geom = aoi
        base = {"acquisition_id": r["acquisition_id"], "datetime_utc": r["datetime_utc"], "source": ds}
        if level == "chunk":
            touched = _chunks_touched(gd, geom)
            step = MAX_FEATURES_PER_REQUEST
            for i in range(0, len(touched), step):
                pending.append({**base, "chunks": touched[i:i + step]})
        else:
            pending.append({**base, "geometry": bounded_geometry(geom)})

    cols = ["acquisition_id", "source", "rain_6h_mm", "rain_24h_mm", "chunk_name", "n_images_found"]
    if level == "chunk":
        batch_kw = dict(weight=lambda rec: len(rec["chunks"]), max_weight=MAX_FEATURES_PER_REQUEST,
                        size=lambda rec: 256 * (len(rec["chunks"]) + 1))
    else:
        batch_kw = {}
    fetched = _batched(pending, lambda b: pd.DataFrame(_rain_batch(cfg, b, level, gd), columns=cols),
                       partial_dir, f"rain_{level}", cols, **batch_kw)
    df = pd.concat([pd.DataFrame(records, columns=cols), fetched], ignore_index=True)
    df["chunk_name"] = df["chunk_name"].fillna("")
    df["track_id"] = df["acquisition_id"].map(lambda a: info[a]["track_id"])
    df["datetime_utc"] = df["acquisition_id"].map(lambda a: info[a]["datetime_utc"])
    df["level"] = level
    df["n_images_expected"] = df["acquisition_id"].map(expected)
    found = pd.to_numeric(df["n_images_found"], errors="coerce")
    no_images = found == 0
    df.loc[no_images, ["rain_6h_mm", "rain_24h_mm"]] = float("nan")
    df.loc[no_images, "source"] = "none"
    df["rain_gap_images"] = (df["n_images_expected"] - found).clip(lower=0)
    return df[RAIN_COLUMNS].sort_values(["datetime_utc", "track_id", "chunk_name"]).reset_index(drop=True)


def _round_or_nan(v) -> float:
    return float("nan") if v is None else round(float(v), 3)


# ---------------------------------------------------------------- 5. slope
def _dem_image(dem_id: str):
    """DEM as an ee.Image with a real default projection.

    Collections such as COPERNICUS/DEM/GLO30_2024_1 must be mosaicked; a mosaic has a 1-degree
    default projection, so ee.Terrain.slope would be wrong unless we set the tiles' native projection.
    """
    import ee

    asset_type = str(ee.data.getAsset(dem_id).get("type", "")).upper()
    if "COLLECTION" in asset_type:
        col = ee.ImageCollection(dem_id)
        first = col.first()
        band = ee.String(ee.Algorithms.If(first.bandNames().contains("DEM"), "DEM", first.bandNames().get(0)))
        return col.select([band]).mosaic().setDefaultProjection(first.select([band]).projection())
    img = ee.Image(dem_id)
    return img.select([0])


def percentiles_from_histogram(lower_edges, counts, bin_width: float,
                               qs=(50, 75, 90, 95, 99)) -> dict[str, float]:
    """Percentiles from a (merged) fixed-width histogram, interpolated linearly inside the bin."""
    lower = np.asarray(lower_edges, dtype=float)
    cnt = np.asarray(counts, dtype=float)
    order = np.argsort(lower)
    lower, cnt = lower[order], cnt[order]
    total = cnt.sum()
    out = {}
    if total <= 0:
        return {f"p{q}": float("nan") for q in qs}
    cum = np.cumsum(cnt)
    for q in qs:
        target = q / 100.0 * total
        i = int(np.searchsorted(cum, target, side="left"))
        i = min(i, len(cnt) - 1)
        before = cum[i - 1] if i > 0 else 0.0
        frac = (target - before) / cnt[i] if cnt[i] > 0 else 0.0
        out[f"p{q}"] = float(lower[i] + min(max(frac, 0.0), 1.0) * bin_width)
    return out


def aggregate_slope_histograms(histograms: list, bin_width: float = SLOPE_BIN_DEG) -> dict:
    """Merge per-region fixedHistogram outputs ([[lower_edge, count], ...]) into slope statistics."""
    merged: dict[float, float] = {}
    for h in histograms:
        for lower, count in (h or []):
            key = round(float(lower), 6)
            merged[key] = merged.get(key, 0.0) + float(count or 0.0)
    lower = np.array(sorted(merged), dtype=float)
    counts = np.array([merged[k] for k in sorted(merged)], dtype=float)
    total = counts.sum()

    def share_above(threshold: float) -> float:
        return float("nan") if total <= 0 else round(100.0 * counts[lower >= threshold - 1e-9].sum() / total, 3)

    pcts = percentiles_from_histogram(lower, counts, bin_width) if len(lower) else {f"p{q}": float("nan") for q in (50, 75, 90, 95, 99)}
    return {
        "percentiles_deg": {k: _round_or_nan(v) if np.isfinite(v) else float("nan") for k, v in pcts.items()},
        "pct_area_gt_5deg": share_above(5.0),
        "pct_area_gt_10deg": share_above(10.0),
        "pct_area_gt_15deg": share_above(15.0),
    }


def _slope_hist_batch(cfg: dict, features: list[dict], scale: int) -> list[dict]:
    """One Earth Engine request: slope histogram (0–90°, SLOPE_BIN_DEG bins) for each region block."""
    import ee

    slope = ee.Terrain.slope(_dem_image(cfg["ard"]["dem"])).rename("slope")
    fc = ee.FeatureCollection([ee.Feature(_ee_geometry(f["geometry"]), {"block_id": f["block_id"]}) for f in features])
    reducer = ee.Reducer.fixedHistogram(0, 90, int(round(90 / SLOPE_BIN_DEG)))
    red = slope.reduceRegions(fc, reducer, scale=scale, crs=cfg["grid"]["crs"], tileScale=4)
    return [{"block_id": f["properties"]["block_id"], "histogram": f["properties"].get("histogram")}
            for f in _compute_features(red)]


def slope_stats(cfg: dict, gd: dict | None = None) -> dict:
    """Terrain slope distribution over the AOI (degrees), computed in the grid CRS.

    The AOI is reduced per region block (bounded geometry, batched requests); per-block histograms
    are merged locally, so no request depends on the AOI's total size or complexity.
    """
    dem_id = cfg["ard"]["dem"]
    gd = _grid_for(cfg, gd)
    aoi = _aoi_union_4326(cfg)
    area_km2 = gpd.GeoSeries([aoi], crs="EPSG:4326").to_crs(EQUAL_AREA_CRS).area.iloc[0] / 1e6
    # DEM native is ~30 m; for very large AOIs a coarser scale keeps the reduction tractable.
    scale = 30 if area_km2 <= 50_000 else 90
    blocks = region_blocks(cfg, gd)
    records = [{"block_id": b, "geometry": g} for b, g in zip(blocks["block_id"], blocks.geometry.values)]
    histograms, batches = [], plan_batches(records)
    for batch in batches:
        histograms += [h["histogram"] for h in _slope_hist_batch(cfg, batch, scale) if h.get("histogram")]
    log.info("slope_stats: %d region blocks in %d request(s) at %d m", len(records), len(batches), scale)
    return {"dem": dem_id, "scale_m": scale, **aggregate_slope_histograms(histograms, SLOPE_BIN_DEG)}


# ---------------------------------------------------------------- 6. report + orchestration
def _md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return "_none_\n"
    head = df.head(max_rows)
    d = head.astype(object).where(head.notna(), "").map(str)  # pandas>=3 keeps NaN through astype(str)
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(v.replace("|", "/") for v in row) + " |" for row in d.to_numpy()]
    more = f"\n_{len(df) - max_rows} more rows in the CSV_\n" if len(df) > max_rows else ""
    return "\n".join(lines) + "\n" + more


def write_gaps_report(path: Path, cfg: dict, slices, acquisitions, summary, gaps, recommendation,
                      rain: pd.DataFrame | None, slope: dict | None, coverage: pd.DataFrame | None) -> Path:
    min_cov = cfg["audit"].get("min_aoi_coverage_pct", 90)
    max_gap = cfg["audit"].get("max_gap_days", 12)
    low = acquisitions[acquisitions["aoi_coverage_pct"] < min_cov] if not acquisitions.empty else acquisitions
    basis = coverage_basis(cfg)
    L = [
        f"# Sentinel-1 audit — {cfg['aoi']['key']} / {cfg['season']['key']}",
        "",
        f"Window: {cfg['season']['start']} to {cfg['season']['end']} (end exclusive). "
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        f"- Earth Engine images (slices) found: **{len(slices)}**",
        f"- Acquisitions (one satellite pass = one or more slices): **{len(acquisitions)}**",
        f"- Tracks: **{acquisitions['track_id'].nunique() if not acquisitions.empty else 0}**",
        "",
        "## How to read this report",
        "",
        "- **Track** = relative orbit + pass direction (ASC evening pass looking east, DSC morning pass looking west).",
        "  Images of the same track share the same viewing geometry, so they are directly comparable over time.",
        f"- **AOI coverage** = share of the AOI inside the acquisition footprint. Below {min_cov}% the date has holes.",
        f"  Coverage basis: **{basis}**." + (
            " The optional border-noise mask (ard.border_noise_correction) removes thin strips at the swath"
            " edges that the footprint still includes, so coverage can be a few percent too high near the edges."
            if basis == COVERAGE_BASIS_NO_BORDER_MASK else ""),
        f"- **Gap** = more than {max_gap} days without an acquisition on a track. Crop stages can be missed inside a gap.",
        "- **Constellation events** (satellite start/end/manoeuvre) inside a gap usually explain it; a platform change",
        "  (e.g. A -> D) can also cause a small radiometric step in the time series.",
        "",
        "## Tracks",
        "",
        _md_table(summary),
        "## Recommendation (not applied automatically)",
        "",
        _md_table(recommendation),
        "Decide and write your choice into the config, e.g.:",
        "",
        "```yaml",
        "s1:",
        "  tracks:",
    ]
    for _, r in recommendation[recommendation["role"] != "unused"].iterrows() if not recommendation.empty else []:
        L.append(f"    - {{track_id: {r['track_id']}, role: {r['role']}}}")
    L += ["```", "", f"## Gaps longer than {max_gap} days", "", _md_table(gaps)]
    L += [f"## Acquisitions with AOI coverage below {min_cov}%", "",
          _md_table(low[["acquisition_id", "platforms", "n_slices", "aoi_coverage_pct"]] if not low.empty else low)]
    events = _events(cfg)
    L += ["## Constellation events", ""]
    L += [f"- {d.strftime('%Y-%m-%d')}: {n}" for d, n in events] or ["_none configured_"]
    L += [""]
    if coverage is not None and not coverage.empty:
        n_chunks = coverage["chunk_name"].nunique()
        L += ["## Chunk coverage by track", "",
              f"{n_chunks} chunks. Full table: `chunk_track_coverage.csv`.", ""]
        agg = coverage.groupby("track_id").agg(chunks_touched=("chunk_name", "nunique"),
                                                chunks_fully_covered_at_least_once=("n_acq_full_cover", lambda s: int((s > 0).sum())),
                                                min_acq_per_chunk=("n_acq", "min"), max_acq_per_chunk=("n_acq", "max")).reset_index()
        L += [_md_table(agg)]
    L += ["## Rain before acquisitions", ""]
    if rain is None or rain.empty:
        L += ["_not computed_", ""]
    else:
        src = rain["source"].value_counts().to_dict()
        wet = rain[pd.to_numeric(rain["rain_24h_mm"], errors="coerce") >= 5]
        L += [f"Data sources used: {src}. Acquisitions with >= 5 mm in the previous 24 h: **{wet['acquisition_id'].nunique()}**.",
              "Rain wets soil and leaves and raises backscatter (especially VV early in the season); treat spikes on",
              "these dates with care.", "", _md_table(wet)]
        if (rain["source"] == "none").any():
            L += ["", "**Warning:** some acquisitions have no rain data (source = none)."]
        if "rain_gap_images" in rain:
            gappy = rain[pd.to_numeric(rain["rain_gap_images"], errors="coerce") > 0]
            if not gappy.empty:
                L += ["", f"**Warning:** {gappy['acquisition_id'].nunique()} acquisition(s) have dataset images missing "
                          "inside the 24 h window (`rain_gap_images` > 0); their rain totals are under-counted."]
        L += [""]
    L += ["## Terrain slope", ""]
    if slope:
        pcts = slope["percentiles_deg"]
        L += [f"DEM `{slope['dem']}` at {slope['scale_m']} m. Median slope {pcts.get('p50')}°, 90th percentile {pcts.get('p90')}°.",
              f"Area steeper than 5°: {slope['pct_area_gt_5deg']}%, 10°: {slope['pct_area_gt_10deg']}%, 15°: {slope['pct_area_gt_15deg']}%.",
              "On flat land terrain flattening changes little; on slopes facing or away from the radar it removes",
              "brightness that comes from geometry rather than from the crop."]
    else:
        L += ["_not computed_"]
    atomic_write_text(path, "\n".join(L) + "\n")
    return path


def _drop_geometry(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(df.drop(columns="geometry")) if "geometry" in df else df


# ---- checkpoint keys (interfaces §1 rule 9: resume, never redo)
def _atomic_gpkg(gdf: gpd.GeoDataFrame, path: Path, layer: str) -> None:
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)  # GDAL needs the .gpkg extension
    if tmp.exists():
        tmp.unlink()
    gdf.to_file(tmp, layer=layer, driver="GPKG")
    replace_with_retry(tmp, path)


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _file_sha(path: Path) -> str | None:
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _aoi_sha(cfg: dict) -> str:
    """Hash of the AOI file and its sidecars (.dbf/.shx/.prj/... for shapefiles)."""
    p = aoi_path(cfg)
    files = sorted(q for q in p.parent.glob(p.stem + ".*") if not q.name.endswith(".tmp")) or [p]
    return _sha({q.name: _file_sha(q) for q in files})


def _valid_csv(path: Path, columns: list[str]) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001 - missing, empty or unparsable -> not valid
        return None
    return df if list(df.columns) == columns else None


def _remove_stale_tmp(folder: Path) -> None:
    """Temp files older than STALE_TMP_SECONDS (younger ones may belong to a live process)."""
    clean_stale_tmp(folder, recursive=True)
    now = time.time()
    for p in folder.rglob("*.tmp.gpkg"):
        try:
            if now - p.stat().st_mtime >= STALE_TMP_SECONDS:
                p.unlink()
        except OSError:
            pass


class _Checkpoints:
    """step -> input key, stored in <audit>/_cache_keys.json (written atomically after every step)."""

    def __init__(self, folder: Path, force: bool):
        self.path = folder / "_cache_keys.json"
        self.force = force
        try:
            self.keys = json.loads(self.path.read_text())
        except Exception:  # noqa: BLE001
            self.keys = {}

    def fresh(self, step: str, key: str) -> bool:
        return not self.force and self.keys.get(step) == key

    def done(self, step: str, key: str) -> None:
        self.keys[step] = key
        atomic_write_text(self.path, json.dumps(self.keys, indent=2, sort_keys=True))


def run_audit(cfg: dict, now: datetime | None = None, audit_date: str | None = None, force: bool = False) -> Path:
    """Run every audit step and write all audit files into processed/<aoi>/<season>/audit/<YYYYMMDD>/.

    Resumable: every step's output is a checkpoint keyed by the inputs that produced it (config
    sections + AOI file hash + hashes of upstream output files), stored in `_cache_keys.json`.
    A step is skipped when its output verifies (loads, expected columns) and its key matches;
    otherwise it is recomputed, and downstream steps follow because their keys include upstream
    file hashes. Long Earth Engine batch loops (incidence angles, rain) persist each batch in
    `_partial/` and resume from the next batch after a crash. `force=True` recomputes everything
    in this audit folder. Re-running on the same (UTC) day resumes the same folder; pass `audit_date`
    (YYYYMMDD) to resume an older one.
    """
    from . import grid as grid_mod

    out = audit_dir(cfg, audit_date)
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "_partial"
    _remove_stale_tmp(out)
    ck = _Checkpoints(out, force)
    if force and partial.exists():
        for p in partial.rglob("*.csv"):
            p.unlink()

    gd = grid_mod.load_grid(cfg) if (grid_dir(cfg) / "grid_def.json").exists() else grid_mod.compute_grid_def(cfg)
    gd_sha = _sha({k: v for k, v in gd.items()})
    aoi_sha = _aoi_sha(cfg)
    aoi = _aoi_union_4326(cfg)
    today = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    f = {name: out / name for name in [
        "s1_slices.csv", "s1_footprints.gpkg", "s1_acquisitions.csv", "track_summary.csv", "gaps.csv",
        "chunk_track_coverage.csv", "track_recommendation.csv", "rain_flags.csv", "slope_stats.json", "gaps_report.md"]}

    def skip(step):
        log.info("skipped %s: cached", step)

    def computed(step):
        log.info("computed %s", step)

    # 1. slices (+ footprints) ------------------------------------------------------------------
    key = _sha({"aoi": aoi_sha, "season": [cfg["season"]["start"], cfg["season"]["end"]],
                "s1": {k: cfg["s1"].get(k) for k in ("collection", "instrument_mode", "pols")}})
    slices = None
    if ck.fresh("slices", key):
        slices = _load_cached_slices(f["s1_slices.csv"], f["s1_footprints.gpkg"])
    if slices is not None:
        skip("slices")
    else:
        slices = list_slices(cfg, gd=gd)
        table = pd.DataFrame(_drop_geometry(slices)).reindex(columns=SLICE_COLUMNS)
        if not slices.empty:
            _atomic_gpkg(slices[["system_index", "track_id", "datetime_utc", "geometry"]], f["s1_footprints.gpkg"], "footprints")
        elif f["s1_footprints.gpkg"].exists():
            f["s1_footprints.gpkg"].unlink()
        atomic_write_csv(table, f["s1_slices.csv"])
        ck.done("slices", key)
        computed("slices")
    slices = slices.copy()
    slices["acquisition_id"] = assign_acquisition_ids(slices) if not slices.empty else pd.Series(dtype=object)
    slices_sha = _sha([_file_sha(f["s1_slices.csv"]), _file_sha(f["s1_footprints.gpkg"])])

    # 2. acquisitions (+ incidence angles, batched) -------------------------------------------
    acq = group_acquisitions(slices, aoi, cfg)  # local and cheap; gives geometries for later EE steps
    use_angle = bool(cfg["audit"].get("compute_incidence_angle", True))
    key = _sha({"slices": slices_sha, "aoi": aoi_sha, "tz": cfg["season"].get("timezone"), "angle": use_angle,
                "basis": coverage_basis(cfg)})
    cached = _valid_csv(f["s1_acquisitions.csv"], ACQ_COLUMNS) if ck.fresh("acquisitions", key) else None
    if cached is not None and set(cached["acquisition_id"]) == set(acq["acquisition_id"]):
        acq["mean_incidence_angle"] = acq["acquisition_id"].map(
            pd.to_numeric(cached.set_index("acquisition_id")["mean_incidence_angle"], errors="coerce"))
        skip("acquisitions")
    else:
        if use_angle and not acq.empty:
            angles = incidence_angles(cfg, acq, partial_dir=partial / key[:16])
            acq["mean_incidence_angle"] = acq["acquisition_id"].map(angles).round(3)
        atomic_write_csv(_drop_geometry(acq)[ACQ_COLUMNS], f["s1_acquisitions.csv"])
        ck.done("acquisitions", key)
        _clean_partial(partial, key[:16])
        computed("acquisitions")
    acq_sha = _file_sha(f["s1_acquisitions.csv"])
    acq_table = _drop_geometry(acq)[ACQ_COLUMNS]

    # 3-6. local derived tables --------------------------------------------------------------
    def local_step(step, fname, columns, key_obj, compute):
        k = _sha(key_obj)
        df = _valid_csv(f[fname], columns) if ck.fresh(step, k) else None
        if df is not None:
            skip(step)
            return pd.read_csv(f[fname])
        df = compute()
        atomic_write_csv(df.reindex(columns=columns), f[fname])
        ck.done(step, k)
        computed(step)
        return df

    audit_cfg = cfg["audit"]
    summary = local_step("track_summary", "track_summary.csv", SUMMARY_COLUMNS,
                         {"acq": acq_sha, "min_cov": audit_cfg.get("min_aoi_coverage_pct")},
                         lambda: track_summary(acq_table, cfg))
    gaps = local_step("gaps", "gaps.csv", GAP_COLUMNS,
                      {"acq": acq_sha, "max_gap": audit_cfg.get("max_gap_days"), "events": audit_cfg.get("constellation_events"),
                       "season": [cfg["season"]["start"], cfg["season"]["end"]], "today": today},
                      lambda: detect_gaps(acq_table, cfg, now=now))
    coverage = local_step("chunk_coverage", "chunk_track_coverage.csv", COVERAGE_COLUMNS,
                          {"slices": slices_sha, "grid": gd_sha, "aoi": aoi_sha},
                          lambda: chunk_track_coverage(cfg, gd, slices))
    rec = local_step("recommendation", "track_recommendation.csv", REC_COLUMNS,
                     {"summary": _file_sha(f["track_summary.csv"]), "coverage": _file_sha(f["chunk_track_coverage.csv"]),
                      "min_cov": audit_cfg.get("min_aoi_coverage_pct")},
                     lambda: recommend_tracks(summary, coverage, cfg))

    # 7. rain (batched) -----------------------------------------------------------------------
    rcfg = cfg.get("rain", {}) or {}
    key = _sha({"acq": acq_sha, "rain": rcfg, "grid": gd_sha if rcfg.get("level") == "chunk" else None, "aoi": aoi_sha,
                "columns": RAIN_COLUMNS})
    rain = _valid_csv(f["rain_flags.csv"], RAIN_COLUMNS) if ck.fresh("rain", key) else None
    if rain is not None:
        rain = pd.read_csv(f["rain_flags.csv"], keep_default_na=True)
        skip("rain")
    else:
        rain = rain_flags(cfg, acq, gd, partial_dir=partial / key[:16])
        atomic_write_csv(rain.reindex(columns=RAIN_COLUMNS), f["rain_flags.csv"])
        ck.done("rain", key)
        _clean_partial(partial, key[:16])
        computed("rain")

    # 8. slope ---------------------------------------------------------------------------------
    key = _sha({"aoi": aoi_sha, "dem": cfg["ard"]["dem"], "crs": cfg["grid"]["crs"], "grid": gd_sha,
                "method": "block-histograms"})
    slope = _load_json(f["slope_stats.json"]) if ck.fresh("slope", key) else None
    if slope is not None and {"dem", "scale_m", "percentiles_deg"} <= set(slope):
        skip("slope")
    else:
        slope = slope_stats(cfg, gd=gd)
        atomic_write_text(f["slope_stats.json"], json.dumps(slope, indent=2))
        ck.done("slope", key)
        computed("slope")

    # 9. report --------------------------------------------------------------------------------
    key = _sha({"files": {n: _file_sha(p) for n, p in f.items() if n != "gaps_report.md"},
                "audit": audit_cfg, "names": [cfg["aoi"]["key"], cfg["season"]["key"]], "basis": coverage_basis(cfg)})
    report_ok = f["gaps_report.md"].exists() and f["gaps_report.md"].stat().st_size > 0
    if ck.fresh("report", key) and report_ok:
        skip("report")
    else:
        write_gaps_report(f["gaps_report.md"], cfg, slices, acq_table, summary, gaps, rec, rain, slope, coverage)
        ck.done("report", key)
        computed("report")
    return out


def _load_cached_slices(csv_path: Path, gpkg_path: Path) -> gpd.GeoDataFrame | None:
    """Slices table + footprints from a previous run, or None if anything is missing/inconsistent."""
    table = _valid_csv(csv_path, SLICE_COLUMNS)
    if table is None:
        return None
    table = pd.read_csv(csv_path, dtype={"system_index": str, "track_id": str, "platform": str,
                                         "datetime_utc": str, "polarisations": str, "pass": str})
    if table.empty:
        return gpd.GeoDataFrame(table, geometry=[], crs="EPSG:4326")
    try:
        fp = gpd.read_file(gpkg_path, layer="footprints")
    except Exception:  # noqa: BLE001
        return None
    if len(fp) != len(table) or set(fp["system_index"]) != set(table["system_index"]):
        return None
    geom = fp.set_index("system_index").geometry.to_crs("EPSG:4326")
    return gpd.GeoDataFrame(table, geometry=table["system_index"].map(geom).values, crs="EPSG:4326")


def _load_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001
        return None


def _clean_partial(partial: Path, key_prefix: str) -> None:
    folder = partial / key_prefix
    if folder.exists():
        for p in folder.glob("*"):
            p.unlink()
        folder.rmdir()
    if partial.exists() and not any(partial.iterdir()):
        partial.rmdir()


# ---------------------------------------------------------------- 7. readers used by later stages
def load_acquisitions(audit_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(audit_path) / "s1_acquisitions.csv",
                     dtype={"acquisition_id": str, "track_id": str, "platforms": str, "system_indexes": str,
                            "date_utc": str, "date_local": str, "datetime_utc": str, "pass": str,
                            "coverage_basis": str})
    df["platforms"] = df["platforms"].fillna("")
    return df


def selected_acquisitions(cfg: dict, audit_path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """{track_id: acquisitions sorted by time} for the tracks chosen in config s1.tracks.

    Raises AuditRequired when no audit exists, when s1.tracks is empty (the user has not reviewed
    the audit yet) or when a configured track does not appear in the audit.
    """
    tracks = (cfg.get("s1", {}) or {}).get("tracks") or []
    if not tracks:
        raise AuditRequired("config s1.tracks is empty: review the audit (gaps_report.md) and choose the track(s) first.")
    if audit_path is None:
        try:
            audit_path = latest_audit_dir(cfg)
        except FileNotFoundError as e:
            raise AuditRequired(f"No audit found; run the audit first ({e}).") from e
    path = Path(audit_path)
    if not (path / "s1_acquisitions.csv").exists():
        raise AuditRequired(f"{path / 's1_acquisitions.csv'} missing; run the audit first.")
    acq = load_acquisitions(path)
    out: dict[str, pd.DataFrame] = {}
    for t in tracks:
        tid = t["track_id"] if isinstance(t, dict) else str(t)
        rows = acq[acq["track_id"] == tid]
        if rows.empty:
            raise AuditRequired(f"Track {tid} from config s1.tracks is not in the audit {path}.")
        out[tid] = rows.sort_values("datetime_utc").reset_index(drop=True)
    return out
