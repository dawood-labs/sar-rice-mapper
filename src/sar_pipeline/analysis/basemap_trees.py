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


def mosaic(aoi: int, root: str = BASEMAP, crs: str = "EPSG:32646", res: float = 1.0) -> Path:
    """Joins the AOI's downloaded tiles (``<root>/aoi<N>/<tile>/<zoom>/mosaic.tif``) and warps them to ``crs`` at
    ``res`` metres (block average). All cores (GDAL_NUM_THREADS)."""
    d = Path(root) / f"aoi{aoi}"
    tiles = sorted(str(p) for p in d.glob("*/*/mosaic.tif"))
    if not tiles:
        raise FileNotFoundError(f"no tiles under {d}")
    vrt, out = d / "mosaic.vrt", d / f"aoi{aoi}_basemap_1m_utm.tif"
    env = {"GDAL_NUM_THREADS": "ALL_CPUS"}
    subprocess.run(["gdalbuildvrt", "-q", "-overwrite", str(vrt), *tiles], check=True, env={**_env(), **env})
    subprocess.run(["gdalwarp", "-q", "-overwrite", "-t_srs", crs, "-tr", str(res), str(res), "-r", "average", "-multi",
                    "-wo", "NUM_THREADS=ALL_CPUS", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", str(vrt), str(out)],
                   check=True, env={**_env(), **env})
    # the same extent at a third of the pixel (3 x 3 sub-pixels per 1 m cell, aligned): the fine texture of a crown
    import rasterio

    with rasterio.open(out) as ds:
        b, (h, w) = ds.bounds, ds.shape
    fine = d / f"aoi{aoi}_basemap_fine_utm.tif"
    subprocess.run(["gdalwarp", "-q", "-overwrite", "-t_srs", crs, "-te", str(b.left), str(b.bottom), str(b.right),
                    str(b.top), "-ts", str(3 * w), str(3 * h), "-r", "bilinear", "-multi", "-wo", "NUM_THREADS=ALL_CPUS",
                    "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES", str(vrt), str(fine)], check=True,
                   env={**_env(), **env})
    return out


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


def cut_edge_trees(fields, tree, transform, min_tree_m2: float = MIN_TREE_M2, edge_m: float = EDGE_M):
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
        cut = [b.intersection(r.geometry) for b in blobs]
        cut = [c for c in cut if c.area >= min_tree_m2 and c.intersects(edge)]
        if not cut:
            rows.append(r._asdict())
            stats.append({"field_id": r.field_id, "cut_m2": 0.0})
            continue
        trees = shapely.union_all(cut).simplify(0.7).buffer(0).intersection(r.geometry)   # smooth the pixel steps
        rest, trees = merge_slivers(r.geometry.difference(trees), trees)
        if trees.is_empty:
            rows.append(r._asdict())
            stats.append({"field_id": r.field_id, "cut_m2": 0.0})
            continue
        if not rest.is_empty:
            rows.append({**r._asdict(), "geometry": rest})
        rows.append({**r._asdict(), "field_id": f"{r.field_id}_t", "major_class": "non-rice", "sub_class": "tree/orchard",
                     "geometry": trees})
        stats.append({"field_id": r.field_id, "cut_m2": float(trees.area)})
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=fields.crs).drop(columns=["Index"], errors="ignore")
    out = out.explode(index_parts=False).reset_index(drop=True)
    out = out[out.area >= 10].reset_index(drop=True)
    k = out.groupby("field_id").cumcount()
    out["field_id"] = [f if j == 0 else f"{f}_{j + 1}" for f, j in zip(out["field_id"], k)]
    out["acres"] = (out.area / ACRE_M2).round(3)
    return out, pd.DataFrame(stats)


#: Spikes / tails (user, 6 Oct, QGIS: zero-width lines sticking out of a cut polygon): an opening by this many metres
#: (mitred corners, so field corners stay sharp) removes them.
DESPIKE_M = 0.25


def despike(g, d: float = DESPIKE_M):
    """``g`` without zero-width tails and needles; corners keep their shape (mitre joins). Falls back to ``g`` when the
    opening would lose more than 1 % of the area (a polygon that is itself very thin)."""
    import shapely

    polys = lambda x: shapely.union_all([p for p in getattr(x, "geoms", [x]) if p.geom_type in ("Polygon", "MultiPolygon")])
    gv = polys(shapely.make_valid(g))                   # an invalid input (self-touching tail) made valid first
    o = g.buffer(-d, join_style="mitre").buffer(d, join_style="mitre")
    o = shapely.make_valid(o.intersection(gv))
    o = shapely.union_all([p for p in getattr(o, "geoms", [o]) if p.geom_type == "Polygon" and p.area > 1.0]) \
        if not o.is_empty else o
    return o if (not o.is_empty and o.area >= 0.99 * gv.area) else gv


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


def main(argv=None) -> int:
    import geopandas as gpd

    from . import qc_compare as q

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["mosaic", "evaluate", "sweep", "cut"])
    p.add_argument("--p-min", type=float, default=TREE_P)
    p.add_argument("--min-m2", type=float, default=MIN_TREE_M2)
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--qc", help="a QC'd field file to learn trees from")
    a = p.parse_args(argv)
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
    tree, transform, crs, _ = tree_mask(a.aoi, qc, p_min=a.p_min)
    layer, stats = cut_edge_trees(q.delivered_fields(a.aoi).to_crs(crs), tree, transform, min_tree_m2=a.min_m2)
    layer, rep = absorb_layer_slivers(layer)
    print(rep)
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
