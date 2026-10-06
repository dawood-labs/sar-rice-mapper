"""Compare a hand-QC'd field file with the delivered one, and look for the pattern behind the QC's deletions.

Why (6 Oct 2026): a reviewer QC'd a delivered AOI in QGIS (deleted some rice fields, cut tree parts out of others,
drew missed rice). Before any rule changes we need two things from the files themselves:

1. :func:`diff` - what changed: fields deleted, fields split, outlines trimmed or grown, new polygons drawn, classes
   edited. It is matched on ``field_id``, because the QC keeps the delivered ids.
2. :func:`removed_vs_kept` - per rice field, the rule's own features (``curve_rules.own_range_features``, the same
   numbers the map was made from) and the bands of one clear Sentinel-2 date, as medians over the field's pixels,
   with a flag for whether the QC deleted the field. :func:`separation` then ranks every feature by how well it
   splits deleted from kept fields (AUC). The reviewer said deletions were "very young" or "clearly flooded on the
   last clear image", so a pattern is expected in crop age and in water on that date.

Run::

    python -m sar_pipeline.analysis.qc_compare diff --aoi 19 --qc ../data/qc/aoi19_qc_2026-10-06.gpkg
    python -m sar_pipeline.analysis.qc_compare pattern --aoi 19 --qc ../data/qc/aoi19_qc_2026-10-06.gpkg --s2-date 2026-09-13
    python -m sar_pipeline.analysis.qc_compare trial --aoi 19 --qc ../data/qc/aoi19_qc_2026-10-06.gpkg   # the new switches
    python -m sar_pipeline.analysis.qc_compare clean --aoi 19 --qc ../data/qc/aoi19_qc_2026-10-06.gpkg   # ready to dissolve
    python -m sar_pipeline.analysis.qc_compare too-young --aoi 39      # delivered fields minus too-young rice

Outputs go to ``processed/_batch/s2_2026/qc/aoi<N>/`` (``diff.csv``, ``fields.csv``, ``separation.csv``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ACRE_M2 = 4046.856
DELIVERY = "processed/_batch/s2_2026/rice_map_2026-10-05"
OUT = "processed/_batch/s2_2026/qc"
S2_DIR = "data/s2_dates_masks"


def delivered_fields(aoi: int, root: str = DELIVERY, name: str | None = None):
    import geopandas as gpd

    return gpd.read_file(Path(root) / f"aoi{aoi}" / (name or f"aoi{aoi}_fields.gpkg"))


def diff(original, qc, min_change_acres: float = 0.01) -> pd.DataFrame:
    """One row per delivered field: ``kept`` / ``deleted`` / ``split`` / ``reshaped``, plus the QC's new polygons
    (no ``field_id``) as ``new``. ``area_change_ac`` is the QC area minus the delivered area."""
    import geopandas as gpd

    qid = qc[qc["field_id"].notna()]
    split = set(qid.loc[qid["field_id"].duplicated(keep=False), "field_id"])
    qd = qid.dissolve("field_id")
    o = original.set_index("field_id")
    rows = []
    for fid, r in o.iterrows():
        if fid not in qd.index:
            rows.append({"field_id": fid, "change": "deleted", "sub_class": r["sub_class"], "acres": r["acres"],
                         "area_change_ac": -r.geometry.area / ACRE_M2})
            continue
        g = qd.loc[fid, "geometry"]
        moved = r.geometry.symmetric_difference(g).area / ACRE_M2
        change = "split" if fid in split else "reshaped" if moved > min_change_acres else "kept"
        rows.append({"field_id": fid, "change": change, "sub_class": r["sub_class"], "acres": r["acres"],
                     "area_change_ac": (g.area - r.geometry.area) / ACRE_M2,
                     "edited_class": str(qd.loc[fid, "sub_class"]) != str(r["sub_class"])})
    new = qc[qc["field_id"].isna()]
    for i, g in enumerate(new.geometry):
        rows.append({"field_id": f"new_{i}", "change": "new", "sub_class": None, "acres": g.area / ACRE_M2,
                     "area_change_ac": g.area / ACRE_M2})
    return pd.DataFrame(rows)


def _field_pixels(aoi: int, fields, root: str = DELIVERY) -> pd.DataFrame:
    """Flat pixel id -> field_id for every pixel whose centre lies in a field (the delivered grid)."""
    import rasterio
    from rasterio import features

    with rasterio.open(Path(root) / f"aoi{aoi}" / f"aoi{aoi}_class_raw.tif") as ds:
        shape, transform = ds.shape, ds.transform
    ids = np.arange(1, len(fields) + 1)
    lab = features.rasterize(zip(fields.geometry, ids), out_shape=shape, transform=transform, fill=0, dtype="int32")
    flat = lab.ravel()
    px = np.flatnonzero(flat)
    return pd.DataFrame({"pid": px, "field_id": fields["field_id"].to_numpy()[flat[px] - 1]})


def s2_bands(aoi: int, date: str, pids, s2_dir: str = S2_DIR) -> pd.DataFrame:
    """The bands of one Sentinel-2 date at the given pixels, plus NDVI, NDWI (green vs NIR: open water > 0) and MNDWI
    (green vs SWIR: water and very wet soil > 0). Only the date's clear pixels (``clear`` band >= 60) are kept."""
    import rasterio

    from ..optical_export import band_index

    path = next(Path(s2_dir, f"aoi{aoi}").glob(f"aoi{aoi}_*_S2_{date}.tif"))
    out = {}
    with rasterio.open(path) as ds:
        for b in ("B2", "B3", "B4", "B5", "B8", "B11", "clear"):
            out[b] = ds.read(band_index(ds, b)).astype("float32").ravel()[pids]
    t = pd.DataFrame(out)
    t = t[t["clear"] >= 60].drop(columns="clear")
    t["ndvi"] = (t.B8 - t.B4) / (t.B8 + t.B4)
    t["ndwi"] = (t.B3 - t.B8) / (t.B3 + t.B8)
    t["mndwi"] = (t.B3 - t.B11) / (t.B3 + t.B11)
    return t.add_prefix("s2_")


def rice_field_features(aoi: int, s2_date: str | None = None, root: str = DELIVERY,
                        name: str | None = None) -> pd.DataFrame:
    """Per delivered RICE field: the median of every rule feature and of one S2 date's bands over its pixels. The
    features are computed with the AOI's own rule set."""
    from . import curve_rules as cr

    o = delivered_fields(aoi, root, name)
    rice = o[o["major_class"] == "rice"].reset_index(drop=True)
    px = _field_pixels(aoi, rice, root)
    with cr.rules_for(aoi):
        f = cr.own_range_features(aoi, px["pid"].to_numpy())
    f = f.select_dtypes("number").reset_index(drop=True)
    f["field_id"] = px["field_id"].to_numpy()
    if s2_date:
        s = s2_bands(aoi, s2_date, px["pid"].to_numpy())
        f = f.join(s)
    t = f.groupby("field_id").median(numeric_only=True)
    t["pixels"] = f.groupby("field_id").size()
    if "s2_ndwi" in f:                            # share of the field's clear pixels that show open water on that date
        t["s2_water_share"] = f.assign(w=f["s2_ndwi"] > 0).dropna(subset=["s2_ndwi"]).groupby("field_id")["w"].mean()
    return rice.set_index("field_id")[["sub_class", "acres"]].join(t)


def removed_vs_kept(aoi: int, qc, s2_date: str | None = None, root: str = DELIVERY) -> pd.DataFrame:
    """:func:`rice_field_features` plus ``deleted`` (True when the QC dropped the field)."""
    t = rice_field_features(aoi, s2_date, root)
    t["deleted"] = ~t.index.isin(set(qc["field_id"].dropna()))
    return t


#: The manager's minimum age (via the user, 6 Oct 2026): only rice older than this many days, counted from water that
#: was seen, goes to the client.
TOO_YOUNG_DAYS = 40


def latest_clear_date(aoi: int, root: str = DELIVERY) -> str:
    """The newest S2 date the delivery counted as clear (at least ``rice_map_delivery.CLEAR_SHARE_MIN`` of the AOI)."""
    d = pd.read_csv(Path(root) / f"aoi{aoi}" / f"aoi{aoi}_s2_dates.csv")
    return str(d[d.iloc[:, 2].astype(str) == "True"].iloc[-1, 0])


def too_young_fields(aoi: int, days: int | None = TOO_YOUNG_DAYS, root: str = DELIVERY,
                     radar_max: float | None = None, max_dry_share: float | None = None,
                     name: str | None = None) -> pd.DataFrame:
    """Per delivered rice field: ``too_young`` when most of its pixels show OPEN water (NDVI below 0) on their newest
    clear view, or showed it within ``days`` of the series end. The same rule as ``curve_rules.TOO_YOUNG_OPEN_WATER_DAYS``,
    judged per field (median of its pixels), as the reviewer judged it. Why (user, 6 Oct 2026): fields flooded in
    September (aoi19, aoi39) are too young for the client; on the aoi19 QC this caught all 42 such deletions.
    The newest clear S2 date's NDVI / NDWI are added, so the call can be checked in a 5-3-2 view.

    Lighter forms (user, 6 Oct, aoi39: "too aggressive"): ``days=None`` judges only the AOI's newest CLEAR date (open
    water there: field NDWI > 0), not partly cloudy later views; ``radar_max`` also needs the radar to be still low now
    (VH position in its own range below it), the manager's "radar backscatter not high". ``max_dry_share`` (with
    ``days=None``): the field goes only when fewer than this share of its clear pixels are NOT water on that date
    (user, 6 Oct: a field with even some dry / green pixels stays; 0.10 = at least 90 % of it under open water)."""
    date = latest_clear_date(aoi, root)
    t = rice_field_features(aoi, date, root, name)
    if days is None and max_dry_share is not None:
        young = (1 - t["s2_water_share"]) < max_dry_share
    elif days is None:
        young = t["s2_ndwi"] > 0
    else:
        young = (t["last_view_open_water"] >= 0.5) | (t["days_since_open_water_view"] <= days)
    if radar_max is not None:
        young &= t["vh_pos_end"] < radar_max
    t["too_young"] = young
    t.attrs["s2_date"] = date
    return t


def remove_too_young(aoi: int, days: int | None = TOO_YOUNG_DAYS, root: str = DELIVERY, out_root: str = OUT,
                     radar_max: float | None = None, max_dry_share: float | None = None):
    """The delivered field layer without its too-young rice fields (:func:`too_young_fields`); the delivery itself is
    not touched. Writes ``<out_root>/aoi<N>/aoi<N>_fields_no_too_young.gpkg`` and ``aoi<N>_too_young_removed.gpkg``
    (the removed fields with their numbers, for checking in QGIS) and returns (kept layer, removed table)."""
    from . import rice_map_delivery as rd

    t = too_young_fields(aoi, days, root, radar_max, max_dry_share)
    o = delivered_fields(aoi, root)
    gone = set(t.index[t["too_young"]])
    tag = ("" if days == TOO_YOUNG_DAYS else "_clear_date" if days is None else f"_{days}d") + \
        ("" if radar_max is None else f"_radar{radar_max:g}") + \
        ("" if max_dry_share is None else f"_dry{int(round(max_dry_share * 100))}")
    out = Path(out_root) / f"aoi{aoi}"
    out.mkdir(parents=True, exist_ok=True)
    kept = o[~o["field_id"].isin(gone)]
    kept.to_file(out / f"aoi{aoi}_fields_no_too_young{tag}.gpkg", driver="GPKG")
    rd.fields_qml(out / f"aoi{aoi}_fields_no_too_young{tag}.gpkg")
    t.attrs["tag"] = tag
    cols = ["field_id", "sub_class", "acres", "last_view_open_water", "days_since_open_water_view", "s2_ndvi", "s2_ndwi",
            "s2_water_share", "vh_pos_end", "vh_since_view", "radar_rise_days", "geometry"]
    removed = o[o["field_id"].isin(gone)].merge(t.reset_index().drop(columns=["sub_class", "acres"]), on="field_id")
    removed[[c for c in cols if c in removed]].to_file(out / f"aoi{aoi}_too_young_removed{tag}.gpkg", driver="GPKG")
    t.to_csv(out / f"aoi{aoi}_rice_field_features.csv")
    return kept, t


def separation(t: pd.DataFrame, min_fields: int = 5) -> pd.DataFrame:
    """Every numeric feature ranked by AUC of deleted vs kept (0.5 = no help; 1 or 0 = perfect split), with both
    groups' medians, so the direction can be read: AUC > 0.5 means deleted fields have the higher values."""
    from scipy.stats import mannwhitneyu

    d, k = t[t["deleted"]], t[~t["deleted"]]
    rows = []
    for c in t.select_dtypes("number").columns:
        a, b = d[c].dropna(), k[c].dropna()
        if len(a) < min_fields or len(b) < min_fields or a.nunique() + b.nunique() < 3:
            continue
        u = mannwhitneyu(a, b).statistic
        rows.append({"feature": c, "auc": u / (len(a) * len(b)), "deleted_median": a.median(),
                     "kept_median": b.median(), "deleted_n": len(a), "kept_n": len(b)})
    s = pd.DataFrame(rows)
    s["strength"] = (s["auc"] - 0.5).abs()
    return s.sort_values("strength", ascending=False).drop(columns="strength")


#: The rule switches tried on aoi19 after its QC (user + manager, 6 Oct 2026); see curve_rules / field_polygons.
QC_SWITCHES = {"TOO_YOUNG_OPEN_WATER_DAYS": 40, "TREE_IF_NEVER_EMPTIED": (0.35, 3.5),
               "field_polygons.STRIP_MAX_WIDTH_M": 15.0, "field_polygons.RICE_GROUP_MAJORITY": True,
               "field_polygons.TREE_CUT_MIN_WIDTH_M": 10.0}


def trial(aoi: int, overrides: dict | None = None, fresh: str = "processed/_batch/s2_2026/rice_fresh",
          out_root: str = OUT, base: str | None = None) -> Path:
    """Map, sieve and fields of one AOI with its own rules PLUS ``overrides`` (default :data:`QC_SWITCHES`), written
    only to ``<out_root>/aoi<N>/trial/``; the real map, the lock and the delivery are not touched. Writes
    ``aoi<N>_fields_trial.gpkg`` in the delivery's attribute format, so it can be laid over the delivered layer in
    QGIS. Why (user, 6 Oct): see the new rules on one AOI before they go anywhere else."""
    import shutil

    import geopandas as gpd

    from . import curve_rules as cr
    from . import rice_map_delivery as rd

    overrides = QC_SWITCHES if overrides is None else overrides
    root = Path(out_root) / f"aoi{aoi}" / ("trial" if base is None else f"trial_{base}")
    (root / f"aoi{aoi}").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}_step1_cover.tif", root / f"aoi{aoi}")
    own = cr.AOI_OVERRIDES.get(aoi)
    if base is not None:          # another rule set in place of the AOI's own (e.g. "aoi160", the QC's best; see choose)
        cr.AOI_OVERRIDES[aoi] = dict(cr.NAMED_SETS[base] if base in cr.NAMED_SETS
                                     else cr.REVIEWED_SETS.get(int(base.removeprefix("aoi")), {}))
    try:
        with cr.switches(overrides, "qc trial"):
            cr.run(aoi, fresh=str(root), force=True)                     # force: only the trial folder is written
            with cr.rules_for(aoi):
                cr.sieve(aoi, fresh=str(root), force=True)
            cr.fields(aoi, fresh=str(root), force=True)
    finally:
        if own is None:
            cr.AOI_OVERRIDES.pop(aoi, None)
        else:
            cr.AOI_OVERRIDES[aoi] = own
    tag = f"{int(round(rd.SLIVER_ACRES * 100)):03d}"
    f = gpd.read_file(root / f"aoi{aoi}" / f"aoi{aoi}_rel_fields_sliver{tag}.gpkg")
    rice = {cr.MAP_CLASSES[k][0] for k in rd.RICE_CODES}
    out = gpd.GeoDataFrame({
        "field_id": [f"aoi{aoi}_{p}" for p in f["polygon_id"]],
        "major_class": np.where(f["class_name"].isin(rice), "rice", "non-rice"),
        "sub_class": f["class_name"].to_numpy(), "acres": f["acres"].round(3).to_numpy(),
        "label_share": f["label_share"].round(2).to_numpy(), "origin": f["origin"].to_numpy()},
        geometry=f.geometry, crs=f.crs)
    gp = root / f"aoi{aoi}_fields_trial.gpkg"
    out.to_file(gp, driver="GPKG")
    rd.fields_qml(gp)
    return gp


def score_trial(aoi: int, qc, trial_gpkg) -> pd.DataFrame:
    """Agreement of a trial field layer with the QC, by area: of the QC's rice, how much the trial calls rice, and of
    the QC's non-rice / empty ground inside the delivered rice, how much the trial left out. Acres, both ways."""
    import geopandas as gpd

    t = gpd.read_file(trial_gpkg)
    o = delivered_fields(aoi)
    qr = qc[(qc["major_class"] == "rice") | qc["field_id"].isna()].union_all()      # the QC's rice (new polygons are rice)
    tr = t[t["major_class"] == "rice"].union_all()
    dr = o[o["major_class"] == "rice"].union_all()
    a = lambda g: round(g.area / ACRE_M2, 1)
    return pd.DataFrame([
        {"what": "QC rice", "acres": a(qr)},
        {"what": "delivered rice", "acres": a(dr)}, {"what": "trial rice", "acres": a(tr)},
        {"what": "QC rice the trial also calls rice", "acres": a(qr.intersection(tr))},
        {"what": "QC rice the trial misses", "acres": a(qr.difference(tr))},
        {"what": "trial rice the QC does not have", "acres": a(tr.difference(qr))},
        {"what": "delivered rice the QC removed", "acres": a(dr.difference(qr))},
        {"what": "  ... of it, the trial removed too", "acres": a(dr.difference(qr).difference(tr))}])


def qc_truth_pixels(aoi: int, qc, root: str = DELIVERY):
    """Pixels the QC judged, with ``True`` for rice: every pixel inside a delivered field or a QC polygon; rice where a QC
    rice polygon or a newly drawn one (no ``field_id``; the user: all new polygons are rice) covers it."""
    import rasterio
    from rasterio import features

    with rasterio.open(Path(root) / f"aoi{aoi}" / f"aoi{aoi}_class_raw.tif") as ds:
        shape, transform = ds.shape, ds.transform
    burn = lambda geoms: features.rasterize(((g, 1) for g in geoms), out_shape=shape, transform=transform, fill=0,
                                            dtype="uint8").ravel().astype(bool)
    judged = burn(delivered_fields(aoi, root).geometry) | burn(qc.geometry)
    rice = burn(qc[(qc["major_class"] == "rice") | qc["field_id"].isna()].geometry)
    px = np.flatnonzero(judged)
    return px, rice[px]


def score_sets(aoi: int, qc, sets: dict, extra: dict | None = None, jobs: int = 0) -> pd.DataFrame:
    """Each full rule set in ``sets`` ({name: switches}) plus ``extra``'s pixel switches, scored pixel by pixel against
    the QC (:func:`qc_truth_pixels`): % agreement on rice / non-rice, rice acres, QC rice missed, rice the QC does not
    have, and how much of the QC's newly drawn rice polygons comes out as rice. One process per set."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    import rasterio
    from rasterio import features

    from .. import resources
    from . import curve_rules as cr
    from .field_review import _classify

    extra = QC_SWITCHES if extra is None else extra
    px, truth = qc_truth_pixels(aoi, qc)
    with rasterio.open(Path(DELIVERY) / f"aoi{aoi}" / f"aoi{aoi}_class_raw.tif") as ds:
        shape, transform = ds.shape, ds.transform
    new = qc[qc["field_id"].isna()]
    drawn = (features.rasterize(((g, 1) for g in new.geometry), out_shape=shape, transform=transform, fill=0,
                                dtype="uint8").ravel()[px] == 1) if len(new) else np.zeros(len(px), bool)
    pix = {k: v for k, v in extra.items() if not k.startswith("field_polygons.")}   # strips etc. are a field step
    rice_codes = [k for k, v in cr.MAP_CLASSES.items() if v[0] in cr.RICE_NAMES]
    acre = 100 / ACRE_M2                                     # one 10 m pixel in acres
    r = resources.detect_resources()
    n = jobs or max(1, min(len(sets), r.cpus, int(r.memory_available_bytes / 3e9)))
    work = [(aoi, {**rules, **pix}, px) for rules in sets.values()]
    with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=resources.limit_worker_threads,
                             initargs=(max(1, r.cpus // n),)) as ex:
        codes = list(ex.map(_classify, work, timeout=3600))
    rows = []
    for name, c in zip(sets, codes):
        rice = np.isin(c, rice_codes)
        rows.append({"rule set": name, "agree_pct": round(100 * float((rice == truth).mean()), 2),
                     "rice_ac": round(rice.sum() * acre, 1), "qc_rice_missed_ac": round((truth & ~rice).sum() * acre, 1),
                     "extra_rice_ac": round((rice & ~truth).sum() * acre, 1),
                     "drawn_rice_as_rice_ac": round((rice & drawn).sum() * acre, 1)})
    out = pd.DataFrame(rows)
    out.attrs["qc_rice_ac"] = round(truth.sum() * acre, 1)
    out.attrs["drawn_ac"] = round(drawn.sum() * acre, 1)
    return out


def own_without(aoi: int, names) -> dict:
    """The AOI's own set, and the same with each named switch turned off (``False`` / ``None``), one set per name."""
    from . import curve_rules as cr

    own = dict(cr.AOI_OVERRIDES.get(aoi, {}))
    off = lambda v: False if isinstance(v, bool) else None
    sets = {"own": own}
    for k in names:
        sets[f"own without {k}"] = {**own, k: off(own.get(k, getattr(cr, k)))}
    if len(names) > 1:
        sets["own without all of them"] = {**own, **{k: off(own.get(k, getattr(cr, k))) for k in names}}
    return sets


def choose_by_qc(aoi: int, qc, extra: dict | None = None, jobs: int = 0) -> pd.DataFrame:
    """Every candidate rule set of the batch (``aoi_batch.CANDIDATE_SETS``), and each batch switch on top of the best,
    with ``extra`` (default :data:`QC_SWITCHES`) laid over each, scored pixel by pixel against the QC's rice / non-rice
    in acres. Why (6 Oct 2026, aoi19): the set the batch chose from agent verdicts turned green rice that the QC drew
    back in into "flooded / bare" (RADAR_DECIDES_WITHOUT_OPTICAL, YOUNG_WHILE_RADAR_LOW); a manager's QC is a better
    truth than the agents' 20 fields per group."""
    from concurrent.futures import ProcessPoolExecutor

    from .. import resources
    from . import aoi_batch as ab
    from . import curve_rules as cr
    from .field_review import _classify

    extra = QC_SWITCHES if extra is None else extra
    px, truth = qc_truth_pixels(aoi, qc)
    pix = {k: v for k, v in extra.items() if not k.startswith("field_polygons.")}   # strips are a field step
    ab._fill_universal()
    sets = {(s if isinstance(s, str) else f"aoi{s}"): dict(cr.NAMED_SETS[s] if isinstance(s, str)
                                                          else cr.REVIEWED_SETS.get(s, {})) for s in ab.CANDIDATE_SETS}
    sets["batch choice"] = dict(cr.AOI_OVERRIDES.get(aoi, {}))
    rice_codes = [k for k, v in cr.MAP_CLASSES.items() if v[0] in cr.RICE_NAMES]
    acre = 100 / ACRE_M2                                     # one 10 m pixel in acres

    first = score_sets(aoi, qc, sets, extra, jobs).sort_values("agree_pct", ascending=False)
    best = first.iloc[0]["rule set"]
    on_top = {f"{best}+{n}": {**sets[best], **sw} for n, sw in ab.CANDIDATE_SWITCHES.items()}
    second = score_sets(aoi, qc, on_top, extra, jobs)
    out = pd.concat([first, second], ignore_index=True).sort_values("agree_pct", ascending=False)
    out.attrs["qc_rice_ac"] = round(truth.sum() * acre, 1)
    return out


#: Gap kept between neighbouring polygons by shrinking each one inwards by this much (metres). Why (user, 6 Oct 2026):
#: the manager runs QGIS "dissolve" on the QC'd layer; touching or overlapping polygons would merge into one and the
#: field boundaries from the delineation would be lost. 0.1 m each side = a 0.2 m gap, nothing against a 10 m pixel.
SHRINK_M = 0.1
#: Pieces smaller than this (m2) left by cutting overlaps are dropped.
CRUMB_M2 = 10.0
#: sub_class given to the polygons the reviewer drew (user: every drawn polygon is rice).
DRAWN_SUB_CLASS = "rice (added in QC)"


def _polygons(geom):
    return [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon" and not g.is_empty]


def clean_qc(aoi: int, qc, shrink_m: float = SHRINK_M, crumb_m2: float = CRUMB_M2):
    """The QC'd field layer made ready for QGIS "dissolve": rice only, drawn polygons as rice, no overlaps, no two
    polygons touching. Returns (clean GeoDataFrame, report dict).

    1. Non-rice polygons are removed; polygons without ``field_id`` (drawn by the reviewer) become rice
       (:data:`DRAWN_SUB_CLASS`, ids ``aoi<N>_qc001``...); a split field's second part gets ``_b`` (``_c``...).
    2. Overlaps: delineated polygons keep their shape (largest first); a drawn polygon loses what a delineated one
       already covers (user, 6 Oct: the SAMGeo outline wins).
    3. Where a neighbour is within ``2 * shrink_m``, each polygon pulls back by ``shrink_m`` from it, so neighbours never
       touch (a gap of ``2 * shrink_m``); edges with no neighbour keep their area.
    Crumbs below ``crumb_m2`` are dropped, every piece is a valid single polygon, and acres are measured again."""
    import geopandas as gpd
    import shapely
    from shapely.strtree import STRtree

    rice = qc[(qc["major_class"] == "rice") | qc["field_id"].isna()].copy()
    drawn = rice["field_id"].isna()
    rice.loc[drawn, "field_id"] = [f"aoi{aoi}_qc{i + 1:03d}" for i in range(int(drawn.sum()))]
    rice.loc[drawn, "major_class"] = "rice"
    rice.loc[drawn, "sub_class"] = DRAWN_SUB_CLASS
    rice.loc[drawn, "origin"] = "drawn in QC"
    rice["drawn"] = drawn.to_numpy()
    n = rice.groupby("field_id").cumcount()
    rice["field_id"] = [f if k == 0 else f"{f}_{chr(ord('a') + k)}" for f, k in zip(rice["field_id"], n)]
    rice["geometry"] = shapely.make_valid(rice.geometry.to_numpy())
    area_in = float(rice.area.sum())
    # delineated first (largest first), then drawn; each takes only what is still free
    rice = rice.assign(_a=rice.area).sort_values(["drawn", "_a"], ascending=[True, False]).reset_index(drop=True)
    taken, rows = [], []
    for r in rice.itertuples():
        g = r.geometry
        if taken:
            tree = STRtree(taken)
            hit = [taken[i] for i in tree.query(g, predicate="intersects")]
            if hit:
                g = g.difference(shapely.union_all(hit))
        for k, part in enumerate(p for p in _polygons(shapely.make_valid(g)) if p.area >= crumb_m2):
            taken.append(part)
            rows.append({"field_id": r.field_id if k == 0 else f"{r.field_id}_p{k + 1}", "source_id": r.field_id,
                         "major_class": "rice",
                         "sub_class": r.sub_class, "origin": r.origin, "drawn": r.drawn, "geometry": part})
    cut = gpd.GeoDataFrame(rows, geometry="geometry", crs=qc.crs)
    area_cut = float(cut.area.sum())
    # pull back only where a neighbour is near (shared edges, touching vertices): free edges keep their area
    geoms = list(cut.geometry)
    tree = STRtree(geoms)
    shrunk, dropped = [], []
    for i, r in enumerate(cut.itertuples()):
        near = [geoms[j] for j in tree.query(r.geometry, predicate="dwithin", distance=2 * shrink_m) if j != i]
        g = r.geometry.difference(shapely.union_all(near).buffer(shrink_m, join_style="mitre")) if near else r.geometry
        g = shapely.make_valid(g)
        if not any(p.area >= crumb_m2 for p in _polygons(g)):
            dropped.append(r.source_id)
        for k, part in enumerate(p for p in _polygons(g) if p.area >= crumb_m2):
            shrunk.append({**r._asdict(), "field_id": r.field_id if k == 0 else f"{r.field_id}_s{k + 1}",
                           "geometry": part})
    out = gpd.GeoDataFrame(shrunk, geometry="geometry", crs=qc.crs).drop(columns=["Index"], errors="ignore")
    out["acres"] = (out.area / ACRE_M2).round(3)
    lost = sorted(set(rice["field_id"]) - set(out["source_id"]))
    out = out[["field_id", "major_class", "sub_class", "acres", "origin", "drawn", "geometry"]]
    return out, {**check_layer(out), "polygons_in": len(qc), "rice_polygons_in": len(rice), "rice_ids_lost": lost,
                 "rice_ids_lost_acres": round(float(rice[rice["field_id"].isin(lost)].area.sum()) / ACRE_M2, 3),
                 "acres_in_rice": round(area_in / ACRE_M2, 2), "acres_after_overlap_cut": round(area_cut / ACRE_M2, 2),
                 "acres_out": round(float(out.area.sum()) / ACRE_M2, 2),
                 "acres_lost_to_overlap": round((area_in - area_cut) / ACRE_M2, 2),
                 "acres_lost_to_gap": round((area_cut - float(out.area.sum())) / ACRE_M2, 2)}


def check_layer(gdf, touch_m: float = 0.01) -> dict:
    """Counts that must all be 0 before dissolve: invalid shapes, non-polygons, duplicate ids, pairs that overlap, and
    pairs closer than ``touch_m`` (touching edges or vertices)."""
    from shapely.strtree import STRtree

    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    near = tree.query(geoms, predicate="dwithin", distance=touch_m)
    pairs = {(int(a), int(b)) for a, b in zip(*near) if a < b}
    overlap = sum(1 for a, b in pairs if geoms[a].intersection(geoms[b]).area > 0)
    return {"polygons_out": len(gdf), "invalid": int((~gdf.is_valid).sum()),
            "not_polygon": int((gdf.geom_type != "Polygon").sum()), "duplicate_ids": int(gdf["field_id"].duplicated().sum()),
            "overlapping_pairs": overlap, "touching_pairs": len(pairs)}


#: Three-way too-young rule on the newest clear S2 date (user, 6 Oct 2026, aoi19 / aoi39): a rice field with fewer than
#: REMOVE_BELOW of its clear pixels NOT open water is too young as a whole; with at least CUT_FROM not water, its water
#: part is cut off as too young and the rest stays rice; in between it stays as it is. Water pieces smaller than
#: MIN_CUT_PIXELS (10 m pixels) are not cut, so a few wet pixels do not punch holes into a field.
REMOVE_BELOW = 0.05
CUT_FROM = 0.25
MIN_CUT_PIXELS = 3


def cut_too_young(aoi: int, remove_below: float = REMOVE_BELOW, cut_from: float = CUT_FROM,
                  min_cut_pixels: int = MIN_CUT_PIXELS, root: str = DELIVERY, name: str | None = None):
    """The delivered field layer with the three-way rule applied (see :data:`REMOVE_BELOW`). Returns (layer, per-field
    table). Too-young parts become ``major_class`` non-rice, ``sub_class`` "too young (not delivered)"; a cut field's
    water part gets the id ``<field_id>_w``. Water = NDWI above 0 on a clear pixel (``clear`` >= 60) of the date."""
    import geopandas as gpd
    import rasterio
    import shapely
    from rasterio import features
    from shapely.geometry import shape

    from ..optical_export import band_index

    date = latest_clear_date(aoi, root)
    o = delivered_fields(aoi, root, name).reset_index(drop=True)
    with rasterio.open(Path(root) / f"aoi{aoi}" / f"aoi{aoi}_class_raw.tif") as ds:
        grid, transform = ds.shape, ds.transform
    path = next(Path(S2_DIR, f"aoi{aoi}").glob(f"aoi{aoi}_*_S2_{date}.tif"))
    with rasterio.open(path) as ds:
        g3, b8, clear = (ds.read(band_index(ds, b)).astype("float32") for b in ("B3", "B8", "clear"))
    ok = clear >= 60
    water = ok & ((g3 - b8) / (g3 + b8) > 0)
    lab = features.rasterize(((g, i + 1) for i, g in enumerate(o.geometry)), out_shape=grid, transform=transform,
                             fill=0, dtype="int32")
    n_ok = np.bincount(lab[ok], minlength=len(o) + 1)[1:]
    n_wet = np.bincount(lab[water], minlength=len(o) + 1)[1:]
    dry = np.where(n_ok > 0, 1 - n_wet / np.maximum(n_ok, 1), np.nan)
    rice = (o["major_class"] == "rice").to_numpy()
    action = np.where(~rice | np.isnan(dry), "kept", np.where(dry < remove_below, "too young",
                      np.where((dry >= cut_from) & (n_wet >= min_cut_pixels), "cut", "kept")))
    rows = []
    for i, r in o.iterrows():
        if action[i] == "too young":
            rows.append({**r.to_dict(), "major_class": "non-rice", "sub_class": TOO_YOUNG_SUB})
            continue
        if action[i] != "cut":
            rows.append(r.to_dict())
            continue
        mask = (lab == i + 1) & water
        wet = [shape(geom) for geom, v in features.shapes(mask.astype("uint8"), mask=mask, transform=transform) if v]
        wet = [w for w in wet if w.area >= min_cut_pixels * 100.0]
        wet = shapely.union_all(wet).intersection(r.geometry) if wet else None
        if wet is None or wet.is_empty:
            action[i] = "kept"
            rows.append(r.to_dict())
            continue
        rest = r.geometry.difference(wet)
        rows.append({**r.to_dict(), "geometry": rest})
        rows.append({**r.to_dict(), "field_id": f"{r.field_id}_w", "major_class": "non-rice", "sub_class": TOO_YOUNG_SUB,
                     "geometry": wet})
    out = gpd.GeoDataFrame(rows, geometry="geometry", crs=o.crs).explode(index_parts=False).reset_index(drop=True)
    out = out[out.area >= CRUMB_M2].reset_index(drop=True)          # every piece its own polygon, no crumbs
    k = out.groupby("field_id").cumcount()
    out["field_id"] = [f if j == 0 else f"{f}_{j + 1}" for f, j in zip(out["field_id"], k)]
    out["acres"] = (out.area / ACRE_M2).round(3)
    t = pd.DataFrame({"field_id": o["field_id"], "major_class": o["major_class"], "sub_class": o["sub_class"],
                      "acres": o["acres"], "clear_pixels": n_ok,
                      "water_pixels": n_wet, "dry_share": dry, "action": action})
    t.attrs["s2_date"] = date
    return out, t


TOO_YOUNG_SUB = "too young (not delivered)"


def o_ac(t) -> float:
    return float(t.loc[t["major_class"] == "rice", "acres"].sum())


def main(argv=None) -> int:
    import geopandas as gpd

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["diff", "pattern", "trial", "choose", "without", "clean", "too-young", "cut-too-young"])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--qc", help="the QC'd field file (GeoPackage with the delivered field_id); not for too-young")
    p.add_argument("--s2-date", help="pattern: one clear Sentinel-2 date, YYYY-MM-DD")
    p.add_argument("--switches", nargs="*", default=[], help="without: switches of the AOI's own set to turn off")
    p.add_argument("--days", type=int, default=TOO_YOUNG_DAYS, help="too-young: open water seen within this many days")
    p.add_argument("--clear-date-only", action="store_true", help="too-young: judge only the newest clear S2 date")
    p.add_argument("--max-dry-share", type=float,
                   help="too-young, with --clear-date-only: remove only fields with fewer than this share of non-water pixels")
    p.add_argument("--radar-max", type=float, help="too-young: also need VH still below this position of its own range")
    p.add_argument("--base", help="trial: a rule set in place of the AOI's own, e.g. aoi160 (the best of 'choose')")
    a = p.parse_args(argv)
    qc = gpd.read_file(a.qc) if a.qc else None
    out = Path(OUT) / f"aoi{a.aoi}"
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "cut-too-young":
        lay, t = cut_too_young(a.aoi)
        gp = out / f"aoi{a.aoi}_fields_too_young_cut.gpkg"
        lay.to_file(gp, driver="GPKG")
        from . import rice_map_delivery as rd
        rd.fields_qml(gp)
        t.to_csv(out / f"aoi{a.aoi}_too_young_cut_fields.csv", index=False)
        r = t[t["action"] != "kept"]
        print(f"newest clear S2 date: {t.attrs['s2_date']}")
        print(t[t["major_class"] == "rice"].groupby("action")["acres"].agg(["count", "sum"]).round(1).to_string())
        yac = lay.loc[lay["sub_class"] == TOO_YOUNG_SUB, "acres"].sum()
        print(f"rice before {o_ac(t):.1f} ac -> after {lay.loc[lay['major_class'] == 'rice', 'acres'].sum():.1f} ac; "
              f"too young {yac:.1f} ac")
        print(f"layer: {gp.resolve()}")
        return 0
    if a.step == "too-young":
        kept, t = remove_too_young(a.aoi, None if a.clear_date_only else a.days, radar_max=a.radar_max,
                                   max_dry_share=a.max_dry_share)
        y = t[t["too_young"]]
        print(f"newest clear S2 date: {t.attrs['s2_date']}")
        print(f"rice fields {len(t)} ({t['acres'].sum():.1f} ac); too young {len(y)} ({y['acres'].sum():.1f} ac)")
        print(y.groupby("sub_class")["acres"].agg(["count", "sum"]).round(1).to_string())
        tag = t.attrs["tag"]
        print(f"kept layer: {(out / f'aoi{a.aoi}_fields_no_too_young{tag}.gpkg').resolve()}")
        return 0
    if a.step == "diff":
        d = diff(delivered_fields(a.aoi), qc)
        d.to_csv(out / "diff.csv", index=False)
        print(d.groupby(["change", "sub_class"], dropna=False)["acres"].agg(["count", "sum"]).round(2).to_string())
    elif a.step == "choose":
        s = choose_by_qc(a.aoi, qc)
        s.to_csv(out / "choose_by_qc.csv", index=False)
        print(f"QC rice: {s.attrs['qc_rice_ac']} ac")
        print(s.to_string(index=False))
    elif a.step == "without":
        s = score_sets(a.aoi, qc, own_without(a.aoi, a.switches))
        s.to_csv(out / "without.csv", index=False)
        print(f"QC rice: {s.attrs['qc_rice_ac']} ac; drawn (new) rice polygons: {s.attrs['drawn_ac']} ac")
        print(s.to_string(index=False))
    elif a.step == "clean":
        import json

        c, rep = clean_qc(a.aoi, qc)
        dst = Path(a.qc).with_name(Path(a.qc).stem + "_clean.gpkg")
        c.to_file(dst, layer=f"aoi{a.aoi}_qc_clean", driver="GPKG")
        dst.with_suffix(".json").write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        print(f"clean layer: {dst.resolve()}")
    elif a.step == "trial":
        gp = trial(a.aoi, base=a.base)
        s = score_trial(a.aoi, qc, gp)
        s.to_csv(gp.parent / "trial_score.csv", index=False)
        print(s.to_string(index=False))
        print(f"trial layer: {gp.resolve()}")
    else:
        t = removed_vs_kept(a.aoi, qc, a.s2_date)
        t.to_csv(out / "fields.csv")
        s = separation(t)
        s.to_csv(out / "separation.csv", index=False)
        print(s.head(25).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
