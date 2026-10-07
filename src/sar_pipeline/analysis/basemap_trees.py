"""Trees on the high-resolution basemap, and cutting them off the edges of rice fields.

Why (manager via the user, 6 Oct 2026): the reviewer still cuts trees off the corners and edges of rice fields by hand
in QGIS, and that takes most of the QC time. In aoi19 the QC trimmed 7 ac from 54 fields in 63 pieces (median 300 m2,
16 m wide): too small for 10 m Sentinel-2 pixels (tree vs rice AUC about 0.8 there), but plain on the 0.3 m basemap
the field delineation was made from (dark-green, textured crowns against bare or green fields).

Steps:

1. :func:`mosaic` - the delineation's cached basemap tiles (read from storage, never changed) joined and warped to the
   AOI's UTM grid at 1 m (``data/basemap/aoi<N>/aoi<N>_basemap_1m_utm.tif``). The tiles are Web Mercator, whose areas
   are about 9 % too large here, so area work happens in UTM.
2. :func:`features` - per 1 m pixel: the three bands, greenness (excess green), brightness and local texture (the
   brightness' standard deviation in 3 x 3 and 7 x 7 windows: a tree crown is rough, a field smooth).
3. :func:`train` - a random forest learnt from a QC: the parts the reviewer cut off rice fields are trees, the inside
   of the QC's rice polygons (3 m in from the edge) is not. :func:`evaluate` checks it on the half of the AOI it did not
   learn from.
4. :func:`cut_edge_trees` - in each rice polygon, tree blobs of at least ``MIN_TREE_M2`` that touch the polygon's edge
   are cut off (as tree/orchard pieces); a tree inside the field that does not reach the edge stays (user, 6 Oct:
   a small tree inside a rice field must not change the field).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

#: Downloaded tiles and mosaics live in the project's data folder, next to the repository (never tracked).
BASEMAP = str(Path(__file__).resolve().parents[3].parent / "data" / "basemap")
#: Smallest tree blob that is cut (m2): about half a Sentinel-2 pixel; smaller crowns stay.
MIN_TREE_M2 = 50.0
#: No slivers (user, 6 Oct, aoi19 trial): a piece left by a cut that is smaller than SLIVER_M2 or narrower than
#: SLIVER_WIDTH_M (it vanishes when shrunk by half that width) joins the other piece of the same field it shares the
#: most outline with (a rice remnant inside a crown becomes tree; a tree crumb on a rice edge becomes rice).
SLIVER_M2 = 150.0
SLIVER_WIDTH_M = 5.0
#: A blob "touches the edge" when it comes within this many metres of the polygon outline.
EDGE_M = 2.0
#: Probability above which a pixel is tree.
TREE_P = 0.5
ACRE_M2 = 4046.856


def fetch(aoi: int, s3_prefix: str, root: str = BASEMAP) -> list[Path]:
    """Copies the AOI's cached delineation tiles (``*/mosaic.tif`` under ``s3_prefix``) to ``<root>/aoi<N>/``. Read only on
    the storage side (``aws s3 cp``); the credentials come from the environment. ``s3_prefix`` is given at run time
    (the tiles folder of the AOI's delineation), so no storage path lives in this public repository."""
    d = Path(root) / f"aoi{aoi}"
    d.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", "--exclude", "*", "--include", "*mosaic.tif",
                    s3_prefix.rstrip("/") + "/", str(d) + "/"], check=True, env=_env())
    return sorted(d.rglob("mosaic.tif"))


#: Buildings (user, 7 Oct 2026: "ghar bhi nikalne hn ... base map hi use kr lo"). Google Open Buildings has no
#: polygons and no 2.5D values here, so roofs are read from the basemap itself, only where they are plain: blue roofs
#: (blue clearly above red and green) and white / grey roofs (bright with almost no colour). Brown / red roofs look like
#: bare soil in the dry-season basemap and are left out. Blobs of ROOF_MIN_M2..ROOF_MAX_M2, widened by
#: BUILDING_BUFFER_M, are cut like trees.
ROOF_BLUE_MARGIN = 25.0
ROOF_BLUE_BRIGHT = 120.0
ROOF_BRIGHT = 195.0
ROOF_GREY_SPREAD = 18.0
#: A roof is compact: blobs whose rotated box is more than this many times longer than wide (roads, bunds) are dropped.
ROOF_MAX_ELONGATION = 4.0
ROOF_MIN_M2 = 12.0
ROOF_MAX_M2 = 1500.0
BUILDING_BUFFER_M = 2.0
#: A whole rice field becomes non-rice when trees and buildings (any blob, edge or inside) cover more than this share.
FIELD_NON_RICE_SHARE = 0.5


def roof_mask(img: np.ndarray) -> np.ndarray:
    """Plain roofs on the 1 m basemap (see :data:`ROOF_BLUE_MARGIN`): blue or white / grey blobs of a house's size."""
    from scipy import ndimage

    r, g, b = img
    blue = (b > r + ROOF_BLUE_MARGIN) & (b > g + ROOF_BLUE_MARGIN / 2) & ((r + g + b) / 3 > ROOF_BLUE_BRIGHT)
    hi, lo = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    grey = ((r + g + b) / 3 > ROOF_BRIGHT) & (hi - lo < ROOF_GREY_SPREAD)
    m = ndimage.binary_opening(blue | grey, iterations=1)
    lab, n = ndimage.label(m)
    if not n:
        return m
    size = ndimage.sum(m, lab, range(1, n + 1))
    ok = (size >= ROOF_MIN_M2) & (size <= ROOF_MAX_M2)
    for k, sl in enumerate(ndimage.find_objects(lab)):       # compact blobs only (a road is long and thin)
        if ok[k]:
            h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            fill = size[k] / max(h * w, 1)
            if max(h, w) / max(min(h, w), 1) > ROOF_MAX_ELONGATION or fill < 0.35:
                ok[k] = False
    return np.isin(lab, 1 + np.flatnonzero(ok))


def mosaic(aoi: int, root: str = BASEMAP, crs: str = "EPSG:32646", res: float = 1.0) -> Path:
    """Joins the AOI's downloaded tiles (``<root>/aoi<N>/<tile>/<zoom>/mosaic.tif``) and warps them to ``crs`` at
    ``res`` metres (block average). All cores (GDAL_NUM_THREADS)."""
    d = Path(root) / f"aoi{aoi}"
    tiles = sorted(str(p) for p in d.glob("*/*/mosaic.tif"))
    if not tiles:
        raise FileNotFoundError(f"no tiles under {d}")
    vrt, out = d / "mosaic.vrt", d / f"aoi{aoi}_basemap_1m_utm.tif"
    # tile DECODING on one thread: the LZW tiles of some AOIs fail to decode with GDAL_NUM_THREADS=ALL_CPUS ("LZWDecode:
    # Strip 0 not terminated with EOI code", 7 Oct 2026, 12 AOIs); the warp itself still uses every core (-wo NUM_THREADS)
    env = {"GDAL_NUM_THREADS": "1"}
    _gdal(["gdalbuildvrt", "-q", "-overwrite", str(vrt), *tiles], {**_env(), **env})
    _gdal(["gdalwarp", "-q", "-overwrite", "-t_srs", crs, "-tr", str(res), str(res), "-r", "average", "-multi",
           "-wo", "NUM_THREADS=ALL_CPUS", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", str(vrt), str(out)],
          {**_env(), **env})
    # the same extent at a third of the pixel (3 x 3 sub-pixels per 1 m cell, aligned): the fine texture of a crown
    import rasterio

    with rasterio.open(out) as ds:
        b, (h, w) = ds.bounds, ds.shape
    fine = d / f"aoi{aoi}_basemap_fine_utm.tif"
    _gdal(["gdalwarp", "-q", "-overwrite", "-t_srs", crs, "-te", str(b.left), str(b.bottom), str(b.right),
           str(b.top), "-ts", str(3 * w), str(3 * h), "-r", "bilinear", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
           "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", str(vrt), str(fine)], {**_env(), **env})
    return out


def _gdal(cmd: list[str], env: dict, tries: int = 2) -> None:
    """Runs a GDAL command; on failure tries once more and then raises with GDAL's own message. Why (7 Oct 2026): 12 of
    76 AOIs failed in gdalwarp while 4 AOIs ran at once, and the same command ran fine alone; ``-q`` hid the reason."""
    for k in range(tries):
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if r.returncode == 0:
            return
    raise RuntimeError(f"{cmd[0]} failed: {(r.stderr or r.stdout).strip()[-600:]}")


def _env() -> dict:
    import os

    return dict(os.environ)


def read(aoi: int, root: str = BASEMAP):
    """(bands as float32 (3, rows, cols), transform, crs, fine texture (2, rows, cols) or None) of the 1 m UTM mosaic.
    The fine texture is the standard deviation of brightness and of green over the 3 x 3 sub-pixels of each 1 m cell
    (shadows and gaps inside a crown; a field is smooth)."""
    import rasterio

    d = Path(root) / f"aoi{aoi}"
    with rasterio.open(d / f"aoi{aoi}_basemap_1m_utm.tif") as ds:
        img, transform, crs = ds.read().astype("float32"), ds.transform, ds.crs
    fine = None
    if (d / f"aoi{aoi}_basemap_fine_utm.tif").exists():
        with rasterio.open(d / f"aoi{aoi}_basemap_fine_utm.tif") as ds:
            f = ds.read().astype("float32")
        h, w = img.shape[1:]
        f = f[:, :3 * h, :3 * w]
        br = f.mean(axis=0).reshape(h, 3, w, 3)
        g = f[1].reshape(h, 3, w, 3)
        fine = np.stack([br.std(axis=(1, 3)), g.std(axis=(1, 3))])
    return img, transform, crs, fine


def features(img: np.ndarray, fine: np.ndarray | None = None) -> np.ndarray:
    """(n_features, rows, cols): R, G, B, excess green 2G-R-B, brightness, green share, brightness std in 3 x 3 and
    7 x 7, excess green mean in 5 x 5. Zero (no-data) pixels stay zero."""
    from scipy import ndimage

    r, g, b = img
    bright = (r + g + b) / 3
    exg = 2 * g - r - b
    share = g / np.maximum(r + g + b, 1)

    def std(x, k):
        m = ndimage.uniform_filter(x, k)
        return np.sqrt(np.maximum(ndimage.uniform_filter(x * x, k) - m * m, 0))

    base = [r, g, b, exg, bright, share, std(bright, 3), std(bright, 7), std(bright, 15), ndimage.uniform_filter(exg, 5)]
    if fine is not None:
        base += [fine[0], fine[1], ndimage.uniform_filter(fine[0], 5), ndimage.uniform_filter(fine[1], 5)]
    return np.stack(base).astype("float32")


FEATURE_NAMES = ["R", "G", "B", "exg", "bright", "green_share", "tex3", "tex7", "tex15", "exg5",
                 "fine_bright_std", "fine_green_std", "fine_bright_std5", "fine_green_std5"]


def labels_from_qc(aoi: int, qc, shape, transform, inside_m: float = 3.0):
    """(tree, rice) pixel masks from a QC: tree = the parts the reviewer cut off delivered rice fields; rice = the QC's
    rice polygons shrunk by ``inside_m`` (their edges are where the trees were)."""
    from rasterio import features as rf

    from . import qc_compare as q

    o = q.delivered_fields(aoi).to_crs(qc.crs)
    keep = qc[qc["major_class"] == "rice"]
    kept = keep.dissolve("field_id")
    cut = []
    for fid, g in zip(o["field_id"], o.geometry):
        if fid in kept.index and o.loc[o["field_id"] == fid, "major_class"].iloc[0] == "rice":
            c = g.difference(kept.loc[fid, "geometry"])
            if c.area > 1:
                cut.append(c)
    burn = lambda geoms: rf.rasterize(((g, 1) for g in geoms if not g.is_empty), out_shape=shape, transform=transform,
                                      fill=0, dtype="uint8").astype(bool)
    tree = burn(cut)
    rice = burn(keep.geometry.buffer(-inside_m))
    return tree & ~rice, rice & ~tree


def train(X: np.ndarray, tree: np.ndarray, rice: np.ndarray, where=None, n_per_class: int = 60000, seed: int = 0):
    """Random forest on a balanced sample of tree / rice pixels (``where``: limit to part of the AOI)."""
    from sklearn.ensemble import RandomForestClassifier

    from .. import resources

    rng = np.random.default_rng(seed)
    ok = X.reshape(len(X), -1).T
    t, r = tree.ravel(), rice.ravel()
    if where is not None:
        t, r = t & where.ravel(), r & where.ravel()
    ti, ri = np.flatnonzero(t), np.flatnonzero(r)
    ti = rng.choice(ti, min(len(ti), n_per_class), replace=False)
    ri = rng.choice(ri, min(len(ri), n_per_class), replace=False)
    idx = np.r_[ti, ri]
    y = np.r_[np.ones(len(ti)), np.zeros(len(ri))]
    m = RandomForestClassifier(200, min_samples_leaf=5, n_jobs=resources.detect_resources().cpus, random_state=seed)
    m.fit(ok[idx], y)
    return m


def predict(m, X: np.ndarray, mask=None) -> np.ndarray:
    """Tree probability per pixel (``mask``: only these pixels; others 0)."""
    flat = X.reshape(len(X), -1).T
    p = np.zeros(flat.shape[0], dtype="float32")
    idx = np.flatnonzero(mask.ravel()) if mask is not None else np.arange(flat.shape[0])
    for s in range(0, len(idx), 2_000_000):
        part = idx[s:s + 2_000_000]
        p[part] = m.predict_proba(flat[part])[:, 1]
    return p.reshape(X.shape[1:])


def evaluate(aoi: int, qc) -> dict:
    """Learn on the north half of the AOI, test on the south half, and the other way round: tree recall (share of the
    QC's cut-off area found) and false tree (share of the QC's rice interior called tree), in area."""
    img, transform, _, fine = read(aoi)
    X = features(img, fine)
    tree, rice = labels_from_qc(aoi, qc, img.shape[1:], transform)
    rows = np.arange(img.shape[1])[:, None] * np.ones((1, img.shape[2]), bool)
    north = rows < img.shape[1] // 2
    out = {}
    for name, learn in (("learn north, test south", north), ("learn south, test north", ~north)):
        m = train(X, tree, rice, where=learn)
        test = ~learn
        p = predict(m, X, mask=(tree | rice) & test) > TREE_P
        out[name] = {"tree_m2": int((tree & test).sum()), "tree_found_pct": round(100 * float(p[tree & test].mean()), 1),
                     "rice_m2": int((rice & test).sum()), "rice_called_tree_pct": round(100 * float(p[rice & test].mean()), 2)}
    return out


def tree_prob(aoi: int, qc=None, model=None):
    """(tree probability on the 1 m grid, transform, crs, model): from ``model``, or one learnt from ``qc`` on the
    whole AOI."""
    img, transform, crs, fine = read(aoi)
    X = features(img, fine)
    if model is None:
        tree, rice = labels_from_qc(aoi, qc, img.shape[1:], transform)
        model = train(X, tree, rice)
    return predict(model, X, mask=img.sum(axis=0) > 0), transform, crs, model


#: Mask tidying (1 m pixels): close gaps inside a crown, drop lone pixels, fill holes smaller than this many m2.
CLOSE_PX = 2
OPEN_PX = 1
HOLE_M2 = 30


def tidy_mask(m: np.ndarray) -> np.ndarray:
    """Tree crowns as solid blobs. Why (aoi19 trial): raw pixels left speckle holes inside crowns and pixel steps on
    the cut lines."""
    from scipy import ndimage

    st = ndimage.generate_binary_structure(2, 1)
    m = ndimage.binary_closing(m, st, iterations=CLOSE_PX)
    m = ndimage.binary_opening(m, st, iterations=OPEN_PX)
    holes = ndimage.binary_fill_holes(m) & ~m
    lab, n = ndimage.label(holes)
    if n:
        size = ndimage.sum(holes, lab, range(1, n + 1))
        m |= np.isin(lab, 1 + np.flatnonzero(size < HOLE_M2))
    return m


def tree_mask(aoi: int, qc=None, model=None, p_min: float = TREE_P):
    p, transform, crs, model = tree_prob(aoi, qc, model)
    return tidy_mask(p > p_min), transform, crs, model


def sweep(aoi: int, qc, ps=(0.5, 0.7, 0.85), min_m2s=(50.0, 100.0, 200.0)) -> pd.DataFrame:
    """Edge-tree cuts for each probability / smallest-blob pair, scored against the QC (:func:`score_against_qc`).
    The model is learnt once (on the whole AOI, so the scores are a little optimistic; see :func:`evaluate`)."""
    from . import qc_compare as q

    p, transform, crs, _ = tree_prob(aoi, qc)
    fields = q.delivered_fields(aoi).to_crs(crs)
    rows = []
    for pm in ps:
        for mm in min_m2s:
            layer, stats = cut_edge_trees(fields, tidy_mask(p > pm), transform, min_tree_m2=mm)
            rows.append({"p_min": pm, "min_m2": mm, "fields_cut": int((stats["cut_m2"] > 0).sum()),
                         **score_against_qc(aoi, qc, layer)})
    return pd.DataFrame(rows)


#: Straight cuts (user, 7 Oct 2026: "kaatne k baad shape seedhi honi chahiye, fuzzy nhi"): the tree / roof blob is
#: closed and opened by STRAIGHT_SMOOTH_M (fills the pixel teeth) and simplified to STRAIGHT_TOLERANCE_M, so the cut
#: line runs in straight segments instead of 1 m pixel steps. The field's own outline is not touched.
STRAIGHT_SMOOTH_M = 2.0
STRAIGHT_TOLERANCE_M = 2.5
#: Field-like pieces (user, 7 Oct 2026: "field boundary ek field ki tarah dikhni chahiye"): after a cut, both the rice
#: piece and the tree / roof piece lose every finger, hook or spike narrower than 2 x FINGER_M (an opening by FINGER_M,
#: mitred corners); the tree piece is what the cleaned rice piece leaves of the field. Untouched fields keep their
#: delineated outline.
FINGER_M = 2.0


def straighten(g, smooth_m: float = STRAIGHT_SMOOTH_M, tol_m: float = STRAIGHT_TOLERANCE_M):
    """A pixel-stepped blob as a polygon with straight edges."""
    if g.is_empty:
        return g
    s = g.buffer(smooth_m, join_style="mitre").buffer(-smooth_m, join_style="mitre")      # close the teeth
    s = s.buffer(-smooth_m / 2, join_style="mitre").buffer(smooth_m / 2, join_style="mitre")  # open the spurs
    s = s.simplify(tol_m, preserve_topology=True).buffer(0)
    return s if not s.is_empty else g


def open_shape(g, r: float = FINGER_M):
    """``g`` without parts narrower than ``2 r`` (fingers, hooks, spikes), corners kept (mitre); pieces below
    :data:`SLIVER_M2` dropped."""
    import shapely

    if g.is_empty:
        return g
    o = shapely.make_valid(g.buffer(-r, join_style="mitre").buffer(r, join_style="mitre").intersection(g))
    parts = [p for p in getattr(o, "geoms", [o]) if p.geom_type == "Polygon" and p.area >= SLIVER_M2]
    return shapely.union_all(parts) if parts else shapely.geometry.Polygon()


def field_pieces(field, rice, r: float = FINGER_M):
    """(rice, non-rice) of a cut field, both field-like: the rice piece opened, the rest of the field as the non-rice
    piece, opened too (what the openings drop is a strip narrower than ``2 r`` between the two, left as a gap)."""
    rice_c = open_shape(rice, r)
    other = open_shape(field.difference(rice_c), r) if not rice_c.is_empty else open_shape(field, r)
    return rice_c, other


def _sliver(g, min_m2: float = SLIVER_M2, width_m: float = SLIVER_WIDTH_M) -> bool:
    return g.area < min_m2 or g.buffer(-width_m / 2).is_empty


def merge_slivers(rice, tree, min_m2: float = SLIVER_M2, width_m: float = SLIVER_WIDTH_M):
    """(rice, tree) of one field with every sliver piece moved to the other side, repeated until none is left; the
    smallest pieces go first. A piece that touches nothing of the other side stays where it is unless it is the only
    one left (then the whole field takes the other class)."""
    import shapely

    def parts(g):
        return [p for p in getattr(g, "geoms", [g]) if not p.is_empty and p.geom_type == "Polygon"]

    side = {"rice": parts(rice), "tree": parts(tree)}
    for _ in range(50):
        small = sorted(((p.area, k, i) for k in side for i, p in enumerate(side[k]) if _sliver(p, min_m2, width_m)))
        moved = False
        for _, k, i in small:
            other = "tree" if k == "rice" else "rice"
            p = side[k][i]
            if not side[other]:
                continue
            touch = [j for j, q in enumerate(side[other]) if p.buffer(0.5).intersects(q)]
            if not touch:
                continue
            j = max(touch, key=lambda j: p.buffer(0.5).intersection(side[other][j]).area)
            side[other][j] = shapely.union_all([side[other][j], p]).buffer(0)
            side[k].pop(i)
            moved = True
            break                                     # indices changed: start again from the smallest
        if not moved:
            break
    if not side["rice"] or all(_sliver(p, min_m2, width_m) for p in side["rice"]):
        side["tree"], side["rice"] = side["tree"] + side["rice"], []     # nothing worth a rice field is left
    to = lambda ps: shapely.union_all(ps).buffer(0) if ps else shapely.geometry.Polygon()
    return to(side["rice"]), to(side["tree"])


def _cut_one(field, cut):
    """(rice, non-rice) of one field from its tree / roof blobs: straight cut, slivers moved, field-like pieces."""
    import shapely

    trees = straighten(shapely.union_all(cut)).intersection(field)
    rest, trees = merge_slivers(field.difference(trees), trees)
    rest, trees = field_pieces(field, rest)
    if rest.is_empty:                                 # nothing field-like left of the rice: the whole field goes
        trees = field
    return rest, trees


def cut_edge_trees(fields, tree, transform, min_tree_m2: float = MIN_TREE_M2, edge_m: float = EDGE_M,
                   built=None, field_share: float = FIELD_NON_RICE_SHARE):
    """Rice polygons with the tree blobs that touch their edge cut off. Returns (layer, per-field table). A cut piece
    becomes ``tree/orchard`` (non-rice) with the id ``<field_id>_t``; blobs inside the field that do not reach the edge,
    and blobs smaller than ``min_tree_m2``, are left."""
    import geopandas as gpd
    import shapely
    from rasterio import features as rf
    from shapely.geometry import shape

    rows, stats = [], []
    for r in fields.itertuples():
        if r.major_class != "rice":
            rows.append(r._asdict())
            continue
        win = rf.geometry_window_from_bounds(transform, r.geometry.bounds) if hasattr(rf, "geometry_window_from_bounds") \
            else None
        minx, miny, maxx, maxy = r.geometry.bounds
        c0, r0 = ~transform * (minx, maxy)
        c1, r1 = ~transform * (maxx, miny)
        c0, r0, c1, r1 = int(max(c0, 0)), int(max(r0, 0)), int(min(c1 + 1, tree.shape[1])), int(min(r1 + 1, tree.shape[0]))
        sub = tree[r0:r1, c0:c1]
        if sub.size == 0 or not sub.any():
            rows.append(r._asdict())
            stats.append({"field_id": r.field_id, "cut_m2": 0.0})
            continue
        t = transform * transform.translation(c0, r0)
        blobs = [shape(g) for g, v in rf.shapes(sub.astype("uint8"), mask=sub, transform=t) if v]
        edge = r.geometry.boundary.buffer(edge_m)
        allb = [b.intersection(r.geometry) for b in blobs]
        allb = [c for c in allb if c.area >= min_tree_m2]
        if allb and sum(c.area for c in allb) > field_share * r.geometry.area:
            cut = [r.geometry]                   # mostly trees / buildings: the whole field goes
        else:
            cut = [c for c in allb if c.intersects(edge)]
        if not cut:
            rows.append(r._asdict())
            stats.append({"field_id": r.field_id, "cut_m2": 0.0})
            continue
        try:
            rest, trees = _cut_one(r.geometry, cut)
        except shapely.errors.GEOSException:
            # coordinates snapped to 1 cm and made valid: removes the near-duplicate vertices behind a TopologyException
            snap = lambda g: shapely.make_valid(shapely.set_precision(g, 0.01))
            try:
                rest, trees = _cut_one(snap(r.geometry), [snap(c) for c in cut])
            except shapely.errors.GEOSException:
                rows.append(r._asdict())              # this field is left uncut and counted; the AOI goes on
                stats.append({"field_id": r.field_id, "cut_m2": 0.0, "geometry_error": True})
                continue
        if trees.is_empty:
            rows.append(r._asdict())
            stats.append({"field_id": r.field_id, "cut_m2": 0.0})
            continue
        if not rest.is_empty:
            rows.append({**r._asdict(), "geometry": rest})
        kind = "building" if built is not None and trees.intersection(built).area > 0.5 * trees.area else "tree/orchard"
        rows.append({**r._asdict(), "field_id": f"{r.field_id}_t", "major_class": "non-rice", "sub_class": kind,
                     "geometry": trees})
        stats.append({"field_id": r.field_id, "cut_m2": float(trees.area)})
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=fields.crs).drop(columns=["Index"], errors="ignore")
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.area >= 10].reset_index(drop=True)
    k = out.groupby("field_id").cumcount()
    out["field_id"] = [f if j == 0 else f"{f}_{j + 1}" for f, j in zip(out["field_id"], k)]
    out["acres"] = (out.area / ACRE_M2).round(3)
    return out, pd.DataFrame(stats, columns=["field_id", "cut_m2", "geometry_error"])


#: Spikes / tails (user, 6 Oct, QGIS: zero-width lines sticking out of a cut polygon): an opening by this many metres
#: (mitred corners, so field corners stay sharp) removes them.
DESPIKE_M = 0.25


def despike(g, d: float = DESPIKE_M):
    """``g`` without zero-width tails and needles; corners keep their shape (mitre joins). Falls back to ``g`` when the
    opening would lose more than 1 % of the area or 5 m2, whichever is larger (a polygon that is itself very thin)."""
    import shapely

    polys = lambda x: shapely.union_all([p for p in getattr(x, "geoms", [x]) if p.geom_type in ("Polygon", "MultiPolygon")])
    gv = polys(shapely.make_valid(g))                   # an invalid input (self-touching tail) made valid first
    o = g.buffer(-d, join_style="mitre").buffer(d, join_style="mitre")
    try:
        o = shapely.make_valid(o.intersection(gv))
    except shapely.errors.GEOSException:          # near-duplicate vertices (aoi74, 90, 113, 139, 148): snap to 1 mm
        try:
            o = shapely.make_valid(shapely.intersection(o, gv, grid_size=0.001))
        except shapely.errors.GEOSException:
            return gv                             # left as it is rather than stopping the AOI
    o = shapely.union_all([p for p in getattr(o, "geoms", [o]) if p.geom_type == "Polygon" and p.area > 1.0]) \
        if not o.is_empty else o
    return o if (not o.is_empty and gv.area - o.area <= max(0.01 * gv.area, 5.0)) else gv


def absorb_layer_slivers(layer, min_m2: float = SLIVER_M2, width_m: float = SLIVER_WIDTH_M, touch_m: float = 0.5):
    """The whole field layer without slivers (user, 6 Oct: "koi slivers nahi hone chahiye kisi bhi qism ke"): every
    polygon below ``min_m2`` or narrower than ``width_m`` joins the neighbour (within ``touch_m``) it shares the most
    outline with and takes that neighbour's id and class; smallest first, repeated. A sliver with no neighbour is
    dropped. Returns (layer, report)."""
    import shapely
    from shapely.strtree import STRtree

    g = layer.reset_index(drop=True).copy()
    g["geometry"] = [despike(x) for x in g.geometry]          # tails off first; pieces it frees are slivers below
    g = g.explode(index_parts=False).reset_index(drop=True)
    geoms = list(g.geometry)
    alive = [True] * len(geoms)
    absorbed = dropped = 0
    dropped_m2 = 0.0
    while True:
        tree = STRtree(geoms)
        cand = sorted((geoms[i].area, i) for i in range(len(geoms)) if alive[i] and _sliver(geoms[i], min_m2, width_m))
        if not cand:
            break
        changed = False
        for _, i in cand:
            if not alive[i]:
                continue
            gi = geoms[i].buffer(touch_m)
            nb = [j for j in tree.query(gi, predicate="intersects") if j != i and alive[j]]
            if not nb:
                alive[i] = False
                dropped += 1
                dropped_m2 += geoms[i].area
                changed = True
                continue
            j = max(nb, key=lambda j: gi.intersection(geoms[j]).area)
            geoms[j] = shapely.union_all([geoms[j], geoms[i].buffer(touch_m).intersection(gi.union(geoms[j]))]).buffer(0)
            alive[i] = False
            absorbed += 1
            changed = True
        if not changed:
            break
    # a join can leave a new tail: despike once more, but only where the polygon stays one piece (no new slivers)
    one = lambda x, d: d if d.geom_type == "Polygon" else x
    g["geometry"] = [one(x, despike(x)) for x in geoms]
    g = g[alive].reset_index(drop=True)
    g = g.explode(index_parts=False).reset_index(drop=True)
    k = g.groupby("field_id").cumcount()
    g["field_id"] = [f if j == 0 else f"{f}_{j + 1}" for f, j in zip(g["field_id"], k)]
    g["acres"] = (g.area / ACRE_M2).round(3)
    return g, {"slivers_absorbed": absorbed, "slivers_dropped": dropped, "dropped_m2": round(dropped_m2, 1),
               "slivers_left": int(sum(_sliver(x, min_m2, width_m) for x in g.geometry))}


def score_against_qc(aoi: int, qc, layer) -> dict:
    """Area the tool cut as tree that the QC also cut (found), that the QC kept as rice (false cut), and the QC's cut
    area the tool missed."""
    from . import qc_compare as q

    o = q.delivered_fields(aoi)
    keep = qc[qc["major_class"] == "rice"].dissolve("field_id")
    qc_cut = []
    for fid, g, mc in zip(o["field_id"], o.geometry, o["major_class"]):
        if mc == "rice" and fid in keep.index:
            c = g.difference(keep.loc[fid, "geometry"])
            if c.area > 1:
                qc_cut.append(c)
    import shapely

    qc_cut = shapely.union_all(qc_cut)
    tool = shapely.union_all(list(layer.loc[layer["field_id"].str.contains("_t"), "geometry"]))
    qr = shapely.union_all(list(keep.geometry))
    a = lambda g: round(g.area / ACRE_M2, 2)
    return {"qc_cut_ac": a(qc_cut), "tool_cut_ac": a(tool), "found_ac": a(tool.intersection(qc_cut)),
            "missed_ac": a(qc_cut.difference(tool)), "false_cut_in_qc_rice_ac": a(tool.intersection(qr))}


#: The tree model used on every AOI: learnt once from the aoi19 QC (the only hand QC so far) and kept on disk.
MODEL_AOI = 19
MODEL_QC = str(Path(BASEMAP).parent / "qc" / "aoi19_qc_2026-10-06.gpkg")
P_MIN = 0.85
CUT_MIN_M2 = 100.0


def model_path(root: str = BASEMAP) -> Path:
    return Path(root) / "models" / f"trees_rf_aoi{MODEL_AOI}.joblib"


def load_model(root: str = BASEMAP):
    """The saved tree model, learnt from the aoi19 QC on first use. Why: one model for all AOIs, so every delivery is
    cut by the same rule, and it is not relearnt (20 s) per AOI."""
    import geopandas as gpd
    import joblib

    p = model_path(root)
    if p.exists():
        return joblib.load(p)
    img, tr, _, fine = read(MODEL_AOI, root)
    X = features(img, fine)
    tree, rice = labels_from_qc(MODEL_AOI, gpd.read_file(MODEL_QC), img.shape[1:], tr)
    m = train(X, tree, rice)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, p)
    return m


def tiles_prefix(aoi: int) -> str:
    """The AOI's tiles folder from the local, untracked ``config/basemap_local.yaml`` (``s3_tiles_template``)."""
    import yaml

    from .. import config as config_mod

    cfg = yaml.safe_load((config_mod.repo_root() / "config" / "basemap_local.yaml").read_text())
    return cfg["s3_tiles_template"].format(aoi=aoi)


def cut_aoi(aoi: int, fields_path, out_path, root: str = BASEMAP, keep_tiles: bool = False,
            cleanup: bool = True) -> dict:
    """Everything for one AOI: tiles fetched (read only) and mosaicked if needed (the raw tiles then deleted; they can
    be fetched again), trees from the saved model plus plain roofs, cut off ``fields_path``'s rice polygons, cleaned (no
    slivers, tails or fingers, straight cuts), written to ``out_path`` (+ .qml). Returns the counts. ``cleanup``: the
    AOI's whole basemap folder (tiles and mosaics) is deleted afterwards, so the disk does not fill up (user, 7 Oct
    2026); the saved tree model is kept."""
    import shutil

    import geopandas as gpd
    import shapely
    from rasterio import features as rf
    from scipy import ndimage
    from shapely.geometry import shape

    d = Path(root) / f"aoi{aoi}"
    if not (d / f"aoi{aoi}_basemap_fine_utm.tif").exists():
        fetch(aoi, tiles_prefix(aoi), root)
        mosaic(aoi, root)
    if not keep_tiles:
        for t in [p for p in d.iterdir() if p.is_dir()]:
            shutil.rmtree(t)                          # local copies of the stored tiles only
        (d / "mosaic.vrt").unlink(missing_ok=True)
    tree, transform, crs, _ = tree_mask(aoi, model=load_model(root), p_min=P_MIN)
    roofs = ndimage.binary_dilation(roof_mask(read(aoi, root)[0]), iterations=int(BUILDING_BUFFER_M))
    geoms = [shape(g) for g, v in rf.shapes(roofs.astype("uint8"), mask=roofs, transform=transform) if v]
    built = shapely.union_all(geoms) if geoms else None
    tree |= roofs
    fields = gpd.read_file(fields_path).to_crs(crs)
    layer, stats = cut_edge_trees(fields, tree, transform, min_tree_m2=CUT_MIN_M2, built=built)
    layer, r1 = absorb_layer_slivers(layer)
    layer, r2 = absorb_layer_slivers(layer)
    layer = layer.to_crs(fields.crs) if layer.crs != fields.crs else layer
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.stem + ".tmp.gpkg")
    layer.to_file(tmp, driver="GPKG")
    tmp.replace(out_path)
    from . import rice_map_delivery as rd
    rd.fields_qml(out_path)
    if cleanup:
        load_model(root)                              # the model is saved before its source AOI's mosaic can go
        shutil.rmtree(d, ignore_errors=True)
    rice = lambda g: float(g.loc[g["major_class"] == "rice"].area.sum()) / ACRE_M2
    return {"fields_cut": int((stats["cut_m2"] > 0).sum()), "rice_fields": int(len(stats)),
            "rice_acres_before": round(rice(fields), 2), "rice_acres_after": round(rice(layer), 2),
            "tree_acres": round(float(layer.loc[layer["sub_class"] == "tree/orchard"].area.sum()) / ACRE_M2
                                - float(fields.loc[fields["sub_class"] == "tree/orchard"].area.sum()) / ACRE_M2, 2),
            "slivers_dropped": r1["slivers_dropped"] + r2["slivers_dropped"], "slivers_left": r2["slivers_left"],
            "fields_left_uncut_geometry_error": int(stats["geometry_error"].fillna(False).astype(bool).sum()),
            "roof_blobs": len(geoms), "model": model_path(root).name, "p_min": P_MIN, "min_cut_m2": CUT_MIN_M2}


def main(argv=None) -> int:
    import geopandas as gpd

    from . import qc_compare as q

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["fetch", "mosaic", "evaluate", "sweep", "cut"])
    p.add_argument("--s3-prefix", help="fetch: the tiles folder of the AOI's delineation basemap cache")
    p.add_argument("--train-aoi", type=int, help="cut: learn the trees from this AOI's QC (default: --aoi itself)")
    p.add_argument("--fields", help="cut: the field layer to cut (default: the AOI's delivered fields)")
    p.add_argument("--buildings", action="store_true", help="cut: also cut plain roofs seen on the basemap (+2 m)")
    p.add_argument("--p-min", type=float, default=TREE_P)
    p.add_argument("--min-m2", type=float, default=MIN_TREE_M2)
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--qc", help="a QC'd field file to learn trees from")
    a = p.parse_args(argv)
    if a.step == "fetch":
        for t in fetch(a.aoi, a.s3_prefix):
            print(t)
        return 0
    if a.step == "mosaic":
        print(mosaic(a.aoi))
        return 0
    qc = gpd.read_file(a.qc)
    if a.step == "evaluate":
        for k, v in evaluate(a.aoi, qc).items():
            print(k, v)
        return 0
    if a.step == "sweep":
        print(sweep(a.aoi, qc).to_string(index=False))
        return 0
    model = None
    if a.train_aoi is not None and a.train_aoi != a.aoi:         # learn on one AOI's QC, use on another AOI
        img, tr_, _, fine = read(a.train_aoi)
        X = features(img, fine)
        tree_, rice_ = labels_from_qc(a.train_aoi, qc, img.shape[1:], tr_)
        model = train(X, tree_, rice_)
        del X
    tree, transform, crs, _ = tree_mask(a.aoi, qc, model=model, p_min=a.p_min)
    built = None
    if a.buildings:
        import shapely
        from rasterio import features as rf
        from scipy import ndimage
        from shapely.geometry import shape

        roofs = roof_mask(read(a.aoi)[0])
        roofs = ndimage.binary_dilation(roofs, iterations=int(BUILDING_BUFFER_M))     # the 2 m margin (1 m pixels)
        geoms = [shape(g) for g, v in rf.shapes(roofs.astype("uint8"), mask=roofs, transform=transform) if v]
        built = shapely.union_all(geoms) if geoms else None
        tree |= roofs
        gpd.GeoDataFrame(geometry=geoms, crs=crs).to_file(Path(BASEMAP) / f"aoi{a.aoi}" / f"aoi{a.aoi}_roofs.gpkg",
                                                          driver="GPKG")
        print(f"roof blobs (with margin): {len(geoms)}, {sum(g.area for g in geoms) / ACRE_M2:.2f} ac")
    fields = gpd.read_file(a.fields) if a.fields else q.delivered_fields(a.aoi)
    layer, stats = cut_edge_trees(fields.to_crs(crs), tree, transform, min_tree_m2=a.min_m2, built=built)
    layer, rep = absorb_layer_slivers(layer)
    layer, rep2 = absorb_layer_slivers(layer)        # a second pass: tails split off by the first become slivers here
    print(rep, rep2)
    if a.train_aoi is not None and a.train_aoi != a.aoi:
        stem = Path(a.fields).stem if a.fields else f"aoi{a.aoi}_fields"
        out = Path(q.OUT) / f"aoi{a.aoi}" / f"{stem}_edge_{'trees_buildings' if a.buildings else 'trees'}_cut.gpkg"
        out.parent.mkdir(parents=True, exist_ok=True)
        layer.to_file(out, driver="GPKG")
        from . import rice_map_delivery as rd
        rd.fields_qml(out)
        print(f"fields cut: {(stats['cut_m2'] > 0).sum()} of {len(stats)} rice fields; layer: {out.resolve()}")
        return 0
    out = Path(q.OUT) / f"aoi{a.aoi}" / f"aoi{a.aoi}_fields_edge_trees_cut.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    layer.to_file(out, driver="GPKG")
    from . import rice_map_delivery as rd
    rd.fields_qml(out)
    print(score_against_qc(a.aoi, qc, layer))
    print(f"fields cut: {(stats['cut_m2'] > 0).sum()} of {len(stats)} rice fields; layer: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
