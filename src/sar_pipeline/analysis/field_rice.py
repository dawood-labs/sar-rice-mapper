"""Label every delineated field from the final class map: the field is the unit farmers manage.

Why
---
The class map decides per 10 m pixel. A field's pixels share one crop, one transplanting and one
flood, so pixel-level disagreements inside a field (a weak radar pixel on its edge, a bund, a
speck the sieve could not remove) are noise, and one label per field removes them. The field
polygons (SAMGeo delineation traced on high-resolution imagery, phase 7) own the geometry; the
class map owns only the label. The same idea as ``field_labels`` (built for a previous project),
kept light here: no model, no cutting, one label per traced field.

Steps per AOI:

1. **burn**: fields are rasterised onto the AOI's 10 m grid, largest first, so where two tracings
   overlap the finer (smaller) one keeps the pixel;
2. **count**: class pixels per field (only pixels inside the AOI carry a class);
3. **small fields**: a field that holds no pixel centre (a quarter of the fields are under one
   pixel) takes the class of the pixel under its representative point and is flagged
   ``pixels = 0``;
4. **label**: the class with the most pixels (ties: the lower class code); ``rice_share`` and
   ``unconfirmed_share`` are kept so any other threshold can be applied later.

Everything is in acres. Output: one GeoPackage of fields with their label, per AOI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr

SRC = "processed/_batch/s2_2026"
DELINEATION = "../data/delineation/fields.gpkg"   # the delineation run's merged output, linked locally under this name
NODATA = 255


#: Refined delineation per AOI (``analysis/field_refine``, fix plan stage 3): used when it exists.
REFINED = f"{SRC}/fields_refined"


def refined_path(aoi_id: int, refined_dir=REFINED) -> Path:
    return Path(refined_dir) / f"aoi{aoi_id}_delineation_refined.gpkg"


def load_fields(aoi_id: int, path=DELINEATION, refined_dir=REFINED):
    """The delineated fields of one AOI (``source_aoi`` names end in ``_<NNN>_delineation``).

    When ``field_refine`` has written a refined file for the AOI it is read instead of the raw
    delineation; its ``field_id`` column then keeps the ids of the raw file (same order), so
    reviews and deliveries can be followed across versions.
    """
    import pyogrio

    r = refined_path(aoi_id, refined_dir) if refined_dir else None
    if r is not None and r.exists():
        return pyogrio.read_dataframe(r)
    return pyogrio.read_dataframe(path, where=f"source_aoi LIKE '%\\_{aoi_id:03d}\\_delineation'")


def counts_per_field(field_index, classes, n_fields: int, n_classes: int = mr.N_CLASSES) -> np.ndarray:
    """(n_fields, n_classes) pixel counts from a raster of field indices (-1 = no field) and classes."""
    f = np.asarray(field_index).ravel()
    c = np.asarray(classes).ravel()
    ok = (f >= 0) & (c != NODATA)
    return np.bincount(f[ok] * n_classes + c[ok], minlength=n_fields * n_classes).reshape(n_fields, n_classes)


def label_from_counts(counts: np.ndarray, fallback) -> pd.DataFrame:
    """Plurality label per field; fields without pixels take ``fallback`` (class at their representative point)."""
    n = counts.sum(axis=1)
    label = np.where(n > 0, counts.argmax(axis=1), np.asarray(fallback))
    with np.errstate(invalid="ignore", divide="ignore"):
        share = counts / n[:, None]
    return pd.DataFrame({
        "pixels": n, "label": label.astype(int),
        "rice_share": np.round(np.where(n > 0, share[:, 1], np.nan), 3),
        "unconfirmed_share": np.round(np.where(n > 0, share[:, 3], np.nan), 3),
        "label_share": np.round(np.where(n > 0, share[np.arange(len(n)), counts.argmax(axis=1)], np.nan), 3),
    })


def label_aoi(aoi_id: int, map_suffix: str = "_final", src_root=SRC, path=DELINEATION):
    """Fields of one AOI with their class label; also the grid of field indices (for scoring)."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    fields = load_fields(aoi_id, path)
    with rasterio.open(Path(src_root) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{map_suffix}.tif") as ds:
        classes = ds.read(1)
        transform, crs = ds.transform, ds.crs
    if fields.empty:
        return fields, np.full(classes.shape, -1, dtype="int32"), classes
    g = fields.to_crs(crs)
    g["geometry"] = g.geometry.make_valid()
    order = np.argsort(-g.geometry.area.to_numpy())          # largest first, finer tracing burns last
    idx = rasterize(((geom, int(i)) for i, geom in zip(order, g.geometry.to_numpy()[order])),
                    out_shape=classes.shape, transform=transform, fill=-1, dtype="int32")
    counts = counts_per_field(idx, classes, len(g))
    pts = g.geometry.representative_point()
    rows, cols = rasterio.transform.rowcol(transform, pts.x.to_numpy(), pts.y.to_numpy())
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < classes.shape[0]) & (cols >= 0) & (cols < classes.shape[1])
    fallback = np.full(len(g), NODATA)
    fallback[inside] = classes[rows[inside], cols[inside]]
    lab = label_from_counts(counts, fallback)
    keep = [c for c in ("uid", "area_acres", "Confidence", "is_field", "refine_flag", "area_acres_delivered",
                        "overlap_lost_share") if c in fields]
    out = gpd.GeoDataFrame(pd.concat([fields[keep].reset_index(drop=True), lab], axis=1),
                           geometry=fields.geometry.reset_index(drop=True), crs=fields.crs)
    out["class_name"] = out["label"].map({k: v for k, v in mr.CLASSES.items()})
    out["aoi"] = f"aoi{aoi_id}"
    # the delineation's uid repeats across tiles and AOIs: a unique id per field of this delivery
    # (a refined file already carries the ids of the raw file)
    ids = fields["field_id"].reset_index(drop=True) if "field_id" in fields else \
        pd.Series([f"aoi{aoi_id}_{i:06d}" for i in range(len(out))])
    out.insert(0, "field_id", ids)
    if "is_field" not in out:
        out["is_field"], out["refine_flag"] = True, ""
    return out, idx, classes


def water_share_per_field(aoi_id: int, idx, n_fields: int, out_root=SRC) -> np.ndarray:
    """Share of each field's pixels that were water all season (fitted NDVI < 0.10 in >= 80 % of
    windows, the ``validation.reference_sets`` definition): ponds and channels traced as fields."""
    from . import ndvi_5day as nd

    d = nd.load(aoi_id, out_root=out_root)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    with np.errstate(invalid="ignore"):
        water = np.nanmean(ndvi < 0.10, axis=0) >= 0.80
    f = np.asarray(idx).ravel()
    ok = f >= 0
    n = np.bincount(f[ok], minlength=n_fields)
    w = np.bincount(f[ok], weights=water[ok].astype(float), minlength=n_fields)
    nd.forget()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, w / np.maximum(n, 1), 0.0)


def acres_by_label(fields) -> dict:
    """Acres per class of the delineated fields (their own ``area_acres``; flagged non-fields excluded)."""
    f = fields[fields["is_field"]] if "is_field" in fields else fields
    return {f"{mr.CLASSES.get(k, k)}_acres": round(float(v), 1)
            for k, v in f.groupby("label")["area_acres"].sum().items()}


def field_label_raster(field_index, labels, classes) -> np.ndarray:
    """Every pixel takes its field's label; pixels in no field, or in a polygon whose label is -1
    (a flagged non-field: strip, pond, dropped outline), keep their own class; no-data stays."""
    idx = np.asarray(field_index)
    lab = np.asarray(labels)
    out = np.asarray(classes).copy()
    inf = idx >= 0
    inf[inf] = lab[idx[inf]] >= 0
    out[inf] = lab[idx[inf]]
    out[np.asarray(classes) == NODATA] = NODATA
    return out.astype("uint8")


def evaluate(aoi_ids, plots, map_suffix: str = "_final") -> pd.DataFrame:
    """Reference-set class shares for the pixel map and for the field-labelled map (docs/14)."""
    from . import ndvi_5day as nd
    from . import validation as va

    rows = []
    for aoi_id in aoi_ids:
        refs = va.reference_sets(aoi_id, plots[plots["aoi"] == f"aoi{aoi_id}"])
        fields, idx, classes = label_aoi(aoi_id, map_suffix)
        maps = {"pixels": classes}
        if len(fields):
            maps["fields"] = field_label_raster(idx, fields["label"].to_numpy(), classes)
        pix = refs["pixel"].to_numpy()
        covered = np.asarray(idx).ravel()[pix] >= 0
        for name, m in maps.items():
            frame = refs.assign(cls=m.ravel()[pix], in_field=covered)
            for (s, region), g in frame.groupby(["set", "region"]):
                row = {"aoi": f"aoi{aoi_id}", "map": name, "set": s, "region": region, "pixels": len(g),
                       "in_field_pct": round(100 * float(g["in_field"].mean()), 1)}
                for code, label in mr.CLASSES.items():
                    if code != NODATA:
                        row[f"{label}_pct"] = round(100 * float((g["cls"] == code).mean()), 2)
                rows.append(row)
        nd.forget()
    return pd.DataFrame(rows)


def run(aoi_ids=None, out_dir=f"{SRC}/fields", map_suffix: str = "_final", refine: bool = True) -> pd.DataFrame:
    """Label the fields of every AOI; one GeoPackage per AOI and ``field_acres_by_class.csv``.

    With ``refine`` (and a refined delineation for the AOI) the label-aware refinement runs after
    labelling: slivers merge into same-label neighbours and all-season water is flagged as pond.
    """
    from . import field_refine

    ids = aoi_ids or sorted(int(p.parent.name[3:]) for p in Path(SRC).glob(f"aoi*/aoi*_monsoon2026{map_suffix}.tif"))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = []
    for aoi_id in ids:
        fields, idx, classes = label_aoi(aoi_id, map_suffix)
        if fields.empty:
            rows.append({"aoi": f"aoi{aoi_id}", "fields": 0})
            continue
        if refine and refined_path(aoi_id).exists():
            share = water_share_per_field(aoi_id, idx, len(fields))
            fields = field_refine.label_stage(fields, share)
        # second opinion (analysis/field_level): the rule re-run on the field's mean curves
        fl_path = Path(SRC) / "report" / "field_level" / f"aoi{aoi_id}.parquet"
        if fl_path.exists():
            # positional: field_level kept this AOI's fields with >= MIN_PX pixels inside the AOI, in
            # label_aoi's order (a uid is only unique within a tile, so no join on it)
            from .field_level import MIN_PX

            fl = pd.read_parquet(fl_path)
            owner = np.where(classes.ravel() != NODATA, idx.ravel(), -1)
            keep = np.bincount(owner[owner >= 0], minlength=len(fields)) >= MIN_PX
            rule = np.full(len(fields), -1)
            if int(keep.sum()) == len(fl) and (fl["label"].to_numpy() == fields["label"].to_numpy()[keep]).all():
                rule[keep] = fl["field_rule_label"].to_numpy()
            fields["field_rule_label"] = rule
            fields["label_confidence"] = np.where(fields["field_rule_label"] < 0, "not checked (small field)",
                                            np.where(fields["field_rule_label"] == fields["label"], "high", "mixed"))
        out_file = Path(out_dir) / f"aoi{aoi_id}_fields_monsoon2026.gpkg"
        out_file.unlink(missing_ok=True)         # a fresh file: new columns cannot be added to an old layer
        fields.to_file(out_file, layer="fields", driver="GPKG")
        # the same pixels as the pixel map, each with its field's label: acres inside the AOI,
        # directly comparable with the pixel map (field polygons reach beyond the AOI edge)
        fmap = field_label_raster(idx, np.where(fields["is_field"], fields["label"], -1).astype("int64"), classes)
        inside = {}
        for tag, arr in (("pixelmap", classes), ("fieldmap", fmap)):
            counts = np.bincount(arr[arr != NODATA], minlength=mr.N_CLASSES)
            for k in range(mr.N_CLASSES):
                inside[f"{mr.CLASSES[k]}_acres_{tag}"] = round(mr.acres(int(counts[k])), 1)
        rows.append({"aoi": f"aoi{aoi_id}", "fields": len(fields),
                     "fields_without_pixel_centre": int((fields["pixels"] == 0).sum()),
                     "aoi_pixels_in_a_field_pct": round(100 * float(((idx >= 0) & (classes != NODATA)).sum())
                                                        / max(int((classes != NODATA).sum()), 1), 1),
                     "field_acres": round(float(fields["area_acres"].sum()), 1), **acres_by_label(fields), **inside})
    t = pd.DataFrame(rows).fillna(0)
    t.to_csv(Path(out_dir) / "field_acres_by_class.csv", index=False)
    return t


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.field_rice", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="*", type=int, default=None)
    args = p.parse_args(argv)
    t = run(args.ids)
    print(t.drop(columns=["aoi"]).sum(numeric_only=True).round(0).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def purity_by_shift(aoi_id: int, shifts_m=(0, 10, 20, 30), map_suffix: str = "_final", min_px: int = 9,
                    directions=((1, 0), (0, 1))) -> pd.DataFrame:
    """Registration check: mean label purity of the fields at their traced position and shifted.

    If the tracing sits on the fields of the class map, moving it by 10-30 m must lower the share of
    each field's pixels that agree with its majority; a maximum away from zero means an offset
    between the delineation and the 10 m grid. Shifts east and north, averaged.
    """
    import rasterio
    from rasterio.features import rasterize

    fields = load_fields(aoi_id)
    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{map_suffix}.tif") as ds:
        classes = ds.read(1)
        transform, crs = ds.transform, ds.crs
    g = fields.to_crs(crs)
    g["geometry"] = g.geometry.make_valid()
    rows = []
    for s in shifts_m:
        vals = []
        for dx, dy in ([(s * a, s * b) for a, b in directions] if s else [(0, 0)]):
            geoms = g.geometry.translate(dx, dy)
            order = np.argsort(-geoms.area.to_numpy())
            idx = rasterize(((geom, int(i)) for i, geom in zip(order, geoms.to_numpy()[order])),
                            out_shape=classes.shape, transform=transform, fill=-1, dtype="int32")
            counts = counts_per_field(idx, classes, len(g))
            n = counts.sum(axis=1)
            ok = n >= min_px
            vals.append(float((counts[ok].max(axis=1) / n[ok]).mean()))
        rows.append({"aoi": f"aoi{aoi_id}", "shift_m": s, "mean_purity": round(float(np.mean(vals)), 4)})
    return pd.DataFrame(rows)


def label_change_by_shift(aoi_id: int, dx: float, dy: float, map_suffix: str = "_final") -> dict:
    """Share of field acres whose plurality label changes when the tracing is moved by (dx, dy) m."""
    import rasterio
    from rasterio.features import rasterize

    fields = load_fields(aoi_id)
    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{map_suffix}.tif") as ds:
        classes = ds.read(1)
        transform, crs = ds.transform, ds.crs
    g = fields.to_crs(crs)
    g["geometry"] = g.geometry.make_valid()
    labels = []
    for geoms in (g.geometry, g.geometry.translate(dx, dy)):
        order = np.argsort(-geoms.area.to_numpy())
        idx = rasterize(((geom, int(i)) for i, geom in zip(order, geoms.to_numpy()[order])),
                        out_shape=classes.shape, transform=transform, fill=-1, dtype="int32")
        counts = counts_per_field(idx, classes, len(g))
        labels.append(np.where(counts.sum(axis=1) > 0, counts.argmax(axis=1), -1))
    ok = (labels[0] >= 0) & (labels[1] >= 0)
    a = fields["area_acres"].to_numpy()
    changed = ok & (labels[0] != labels[1])
    rice_before = float(a[ok & (labels[0] == 1)].sum())
    rice_after = float(a[ok & (labels[1] == 1)].sum())
    return {"aoi": f"aoi{aoi_id}", "dx": dx, "dy": dy, "changed_acres_pct": round(100 * float(a[changed].sum() / a[ok].sum()), 2),
            "rice_acres_change_pct": round(100 * (rice_after - rice_before) / max(rice_before, 1e-9), 2)}
