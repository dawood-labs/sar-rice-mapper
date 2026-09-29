"""One folder on GCS with everything a reviewer needs for a set of AOIs.

Why
---
The delivery is spread over several folders (rasters, field files, tables, notes) and the
Sentinel-2 dates live in a per-date archive of ~100 files per AOI, most of them cloudy. A manager
who wants to look at thirty AOIs in QGIS should download one folder: the maps, the field polygons,
the acre tables, the two notes, and only the Sentinel-2 dates that are actually clear over the
AOI. Nothing here is recomputed; files are copied from the delivery, and the Sentinel-2 files are
copied inside GCS (no download, no re-export).

Layout of ``gs://<bucket>/<base_folder>/<name>/``::

    README.md
    rasters/aoi<N>_standing_rice_<map date>.tif        class map, colour table embedded
    fields/aoi<N>_fields_standing_rice_<map date>.gpkg  field polygons with label, confidence, note
    tables/acres_by_class.csv, field_acres_by_class.csv, legend.csv, class3_phenology.csv
    docs/CLASS_GUIDE.md, METHODS.md
    sentinel2/index.csv                                   aoi, date, clear share, file
    sentinel2/aoi<N>/aoi<N>_S2_<date>.tif                 dates with at least MIN_CLEAR_PCT % of the AOI clear (25 % by default: the user wants every usable view)

"Clear" is the same mask the map was built with (Cloud Score+ on bright pixels, dark pixels kept,
haze test; ``analysis/ndvi_5day``), measured inside the AOI polygon.

Use::

    python -m sar_pipeline.manager_package --name manager_package_2026-09-29 --ids 116 160 ...
    python -m sar_pipeline.manager_package --name ... --ids ... --no-upload     # stage locally only
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

SRC = "processed/_batch/s2_2026"
DELIVERY = f"{SRC}/delivery"
S2_FOLDER = "s2_dates_masks"
MIN_CLEAR_PCT = 25.0
SINCE = "2026-05-01"
BANDS_NOTE = ("Sentinel-2 files carry the bands B2 B3 B4 B5 B8 B11 B12 (surface reflectance x 10000), QA60, SCL and "
              "`clear` (Cloud Score+ x 100), named in the band descriptions: look bands up by name. True colour = "
              "B4-B3-B2, false colour = B8-B4-B3, crop contrast = B5-B3-B2.")


def clear_dates(aoi_id: int, since: str = SINCE, min_clear_pct: float = MIN_CLEAR_PCT) -> pd.DataFrame:
    """Dates from ``since`` whose clear share inside the AOI is at least ``min_clear_pct``."""
    from .analysis import ndvi_5day as nd

    d = nd.load(aoi_id)
    try:
        inside = nd.inside_aoi(aoi_id)
        ok = np.asarray(d["ok"]).reshape(len(d["dates"]), -1)[:, inside]
        share = 100 * ok.mean(axis=1)
        t = pd.DataFrame({"date": pd.to_datetime(list(d["dates"])), "clear_pct": np.round(share, 1)})
    finally:
        nd.forget()
    t = t[(t["date"] >= pd.Timestamp(since)) & (t["clear_pct"] >= min_clear_pct)].reset_index(drop=True)
    t.insert(0, "aoi", f"aoi{aoi_id}")
    return t


def s2_object(base_folder: str, aoi_id: int, date: pd.Timestamp) -> str:
    """The archive's object name for one date (``optical_export.object_name``)."""
    return f"{base_folder}/{S2_FOLDER}/aoi{aoi_id}/aoi{aoi_id}_{date:%Y-%m}_S2_{date:%Y-%m-%d}.tif"


def readme(name: str, aoi_ids, map_date: str, n_images: int, since: str, min_clear_pct: float) -> str:
    ids = ", ".join(str(a) for a in aoi_ids)
    return f"""# {name}

Standing monsoon rice map, map date {map_date}, for {len(aoi_ids)} areas: {ids}.

* `rasters/` - one GeoTIFF per area (10 m, UTM), class codes with the colour table embedded;
  `tables/legend.csv` lists the codes.
* `fields/` - one GeoPackage per area: the field polygons with `label`, `class_name`,
  `label_confidence` (high / mixed / low) and `confidence_note`; `is_field` false marks strips and
  ponds kept for context.
* `tables/` - acres per class per area from the pixel map (`acres_by_class.csv`) and from the
  field labels (`field_acres_by_class.csv`); `class3_phenology.csv` is the per-area comparison
  behind the "same crop as the area's confirmed rice" relabel.
* `docs/CLASS_GUIDE.md` - what each class means, one page. `docs/METHODS.md` - how the map is made.
* `sentinel2/` - {n_images} Sentinel-2 dates from {since} with at least {min_clear_pct:.0f} % of the
  area clear (`sentinel2/index.csv` lists each date's clear share). {BANDS_NOTE}

Delivered as rice: classes 1 (rice, standing) and 6 (young rice). Rice with `label_confidence =
low` and the note "water not seen in the radar; same crop cycle as the area's confirmed rice" is
rice sown dry or in shallow water; it can be filtered out if only radar-confirmed rice is wanted.
"""


def build(aoi_ids, name: str, since: str = SINCE, min_clear_pct: float = MIN_CLEAR_PCT, upload: bool = True,
          config_path: str = "config/aoi116_monsoon2026.yaml", local_root=SRC) -> dict:
    """Stage the package locally under ``<local_root>/<name>/`` and, with ``upload``, put it on GCS."""
    from . import auth
    from . import config as config_mod

    aoi_ids = [int(a) for a in aoi_ids]
    local = Path(local_root) / name
    for sub in ("rasters", "fields", "tables", "docs", "sentinel2"):
        (local / sub).mkdir(parents=True, exist_ok=True)
    delivery = Path(DELIVERY)
    rasters = sorted(p for p in delivery.glob("aoi*_standing_rice_*.tif") if int(p.name.split("_")[0][3:]) in aoi_ids)
    if not rasters:
        raise FileNotFoundError(f"no delivered rasters for {aoi_ids} in {delivery}")
    map_date = rasters[0].stem.split("_")[-1]
    for p in rasters:
        shutil.copy2(p, local / "rasters" / p.name)
    for a in aoi_ids:
        for p in (delivery / "fields").glob(f"aoi{a}_fields_*.gpkg"):
            shutil.copy2(p, local / "fields" / p.name)
    keys = {f"aoi{a}" for a in aoi_ids}
    for csv in ("acres_by_class.csv", "field_acres_by_class.csv"):
        t = pd.read_csv(delivery / csv)
        t = t[t["aoi"].isin(keys | {"TOTAL"})] if csv == "acres_by_class.csv" else t[t["aoi"].isin(keys)]
        t.to_csv(local / "tables" / csv, index=False)
    shutil.copy2(delivery / "legend.csv", local / "tables" / "legend.csv")
    ph = Path(SRC) / "report" / "class3_phenology.csv"
    if ph.exists():
        t = pd.read_csv(ph)
        t[t["aoi"].isin(keys)].to_csv(local / "tables" / "class3_phenology.csv", index=False)
    for doc in ("CLASS_GUIDE.md", "METHODS.md"):
        if (delivery / doc).exists():
            shutil.copy2(delivery / doc, local / "docs" / doc)
    # the clear Sentinel-2 dates, one index for all areas
    parts = []
    for a in aoi_ids:
        try:
            parts.append(clear_dates(a, since, min_clear_pct))
        except Exception as exc:  # an AOI without a series must not stop the package
            print(f"aoi{a}: Sentinel-2 dates skipped ({type(exc).__name__}: {exc})"[:160])
    index = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["aoi", "date", "clear_pct"])
    index["file"] = [f"sentinel2/{r.aoi}/{r.aoi}_S2_{r.date:%Y-%m-%d}.tif" for r in index.itertuples()]
    index["date"] = index["date"].dt.strftime("%Y-%m-%d")
    index.to_csv(local / "sentinel2" / "index.csv", index=False)
    (local / "README.md").write_text(readme(name, aoi_ids, map_date, len(index), since, min_clear_pct))
    summary = {"name": name, "local": str(local), "aois": len(aoi_ids), "rasters": len(rasters),
               "fields": len(list((local / "fields").glob("*.gpkg"))), "s2_images": len(index)}
    if not upload:
        return summary
    cfg = config_mod.load_config(config_path)
    bucket = auth.gcs_bucket(cfg)
    base = cfg["gcs"]["base_folder"]
    prefix = f"{base}/{name}"
    uploaded = 0
    for p in sorted(local.rglob("*")):
        if p.is_file() and "sentinel2/aoi" not in str(p):
            bucket.blob(f"{prefix}/{p.relative_to(local).as_posix()}").upload_from_filename(str(p))
            uploaded += 1
    copied = missing = 0
    for r in index.itertuples():
        src = bucket.blob(s2_object(base, int(r.aoi[3:]), pd.Timestamp(r.date)))
        if not src.exists():
            missing += 1
            continue
        bucket.copy_blob(src, bucket, f"{prefix}/{r.file}")
        copied += 1
    summary.update({"gcs": f"gs://{bucket.name}/{prefix}/", "uploaded_files": uploaded, "s2_copied": copied,
                    "s2_missing_in_archive": missing})
    return summary


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.manager_package", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="folder name under the project's GCS base folder")
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--since", default=SINCE)
    p.add_argument("--min-clear", type=float, default=MIN_CLEAR_PCT, help="minimum clear share inside the AOI (%%)")
    p.add_argument("--no-upload", action="store_true", help="stage locally only (no cloud writes)")
    args = p.parse_args(argv)
    out = build(args.ids, args.name, args.since, args.min_clear, upload=not args.no_upload)
    for k, v in out.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
