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


def delivered_fields(aoi: int, root: str = DELIVERY):
    import geopandas as gpd

    return gpd.read_file(Path(root) / f"aoi{aoi}" / f"aoi{aoi}_fields.gpkg")


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


def removed_vs_kept(aoi: int, qc, s2_date: str | None = None, root: str = DELIVERY) -> pd.DataFrame:
    """Per delivered RICE field: the median of every rule feature and of the S2 date's bands over its pixels, and
    ``deleted`` (True when the QC dropped the field). The features are computed with the AOI's own rule set."""
    from . import curve_rules as cr

    o = delivered_fields(aoi, root)
    rice = o[o["major_class"] == "rice"].reset_index(drop=True)
    gone = set(rice["field_id"]) - set(qc["field_id"].dropna())
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
    t = rice.set_index("field_id")[["sub_class", "acres"]].join(t)
    t["deleted"] = t.index.isin(gone)
    return t


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
QC_SWITCHES = {"TOO_YOUNG_OPEN_WATER_DAYS": 40, "TREE_IF_NEVER_EMPTIED": (0.30, 3.5),
               "field_polygons.STRIP_MAX_WIDTH_M": 15.0}


def trial(aoi: int, overrides: dict | None = None, fresh: str = "processed/_batch/s2_2026/rice_fresh",
          out_root: str = OUT) -> Path:
    """Map, sieve and fields of one AOI with its own rules PLUS ``overrides`` (default :data:`QC_SWITCHES`), written
    only to ``<out_root>/aoi<N>/trial/``; the real map, the lock and the delivery are not touched. Writes
    ``aoi<N>_fields_trial.gpkg`` in the delivery's attribute format, so it can be laid over the delivered layer in
    QGIS. Why (user, 6 Oct): see the new rules on one AOI before they go anywhere else."""
    import shutil

    import geopandas as gpd

    from . import curve_rules as cr
    from . import rice_map_delivery as rd

    overrides = QC_SWITCHES if overrides is None else overrides
    root = Path(out_root) / f"aoi{aoi}" / "trial"
    (root / f"aoi{aoi}").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}_step1_cover.tif", root / f"aoi{aoi}")
    with cr.switches(overrides, "qc trial"):
        cr.run(aoi, fresh=str(root), force=True)                     # force: only the trial folder is written
        with cr.rules_for(aoi):
            cr.sieve(aoi, fresh=str(root), force=True)
        cr.fields(aoi, fresh=str(root), force=True)
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


def main(argv=None) -> int:
    import geopandas as gpd

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["diff", "pattern", "trial"])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--qc", required=True, help="the QC'd field file (GeoPackage with the delivered field_id)")
    p.add_argument("--s2-date", help="pattern: one clear Sentinel-2 date, YYYY-MM-DD")
    a = p.parse_args(argv)
    qc = gpd.read_file(a.qc)
    out = Path(OUT) / f"aoi{a.aoi}"
    out.mkdir(parents=True, exist_ok=True)
    if a.step == "diff":
        d = diff(delivered_fields(a.aoi), qc)
        d.to_csv(out / "diff.csv", index=False)
        print(d.groupby(["change", "sub_class"], dropna=False)["acres"].agg(["count", "sum"]).round(2).to_string())
    elif a.step == "trial":
        gp = trial(a.aoi)
        s = score_trial(a.aoi, qc, gp)
        s.to_csv(out / "trial_score.csv", index=False)
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
