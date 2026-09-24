"""Package the standing-rice maps for delivery: one GeoTIFF per AOI, one acres table, one legend.

Why
---
The rule writes its maps where the analysis needs them, with class codes only a reader of the code
understands. The map packaged is ``<aoi>_monsoon2026_final.tif`` (``analysis/finalize``: user relabels and the
minimum mapping unit applied) when it exists. The person receiving the maps needs each file to explain itself: a colour table so it
opens readably in any GIS, the class names in the file's metadata, the map date, and one table with
the acres of every class per AOI plus the totals. Nothing is re-computed here; the files are copied
from the rule's output and described.

Use::

    python -m sar_pipeline.delivery --out processed/_batch/s2_2026/delivery
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SRC = "processed/_batch/s2_2026"
#: code -> (name, description, RGBA)
LEGEND = {
    0: ("not rice", "no rice-like crop cycle this monsoon", (230, 230, 230, 255)),
    1: ("rice, standing, water confirmed", "rice-like cycle standing on the map date; transplanting water confirmed by Sentinel-1", (0, 140, 60, 255)),
    2: ("young", "a crop has started but has no full canopy yet on the map date", (170, 220, 120, 255)),
    3: ("rice-like, water not confirmed", "rice-like cycle standing on the map date, but no transplanting water in the radar", (245, 160, 40, 255)),
    4: ("harvested", "a rice-like cycle that was already cut before the map date", (150, 100, 60, 255)),
    6: ("rice, standing, young", "a young rice crop on the map date: transplanting water confirmed by Sentinel-1 and canopy already visible (NDVI >= 0.30)", (110, 200, 60, 255)),
    5: ("rice-like curve, never bare", "the optical curve looks like rice but the radar shows trees or buildings all season (cloud/haze artefact); not rice", (140, 90, 190, 255)),
    7: ("flooded, not yet green", "under water on the map date: transplanting water confirmed by Sentinel-1 within 45 days, crop not yet visible (the next map's rice)", (60, 120, 200, 255)),
    8: ("cut crop, water not confirmed", "a rice-like cycle cut before the map date with no transplanting water in the radar: mostly rain-fed dry-land crops, not paddy", (200, 170, 120, 255)),
}
NODATA = 255


def legend_table() -> pd.DataFrame:
    return pd.DataFrame([{"code": k, "class": v[0], "meaning": v[1]} for k, v in LEGEND.items()])


def package_aoi(aoi: str, out_dir, map_date: str, src_root=SRC) -> dict:
    """Copy one AOI's map with a colour table and class names; return its acres per class."""
    import numpy as np
    import rasterio

    src = Path(src_root) / aoi / f"{aoi}_monsoon2026_final.tif"     # rule + user relabels + sieve
    if not src.exists():
        src = Path(src_root) / aoi / f"{aoi}_monsoon2026.tif"
    with rasterio.open(src) as ds:
        data = ds.read(1)
        profile = ds.profile.copy()
        res = abs(ds.transform.a)
    out = Path(out_dir) / f"{aoi}_standing_rice_{map_date}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.update(nodata=NODATA, compress="deflate")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(data, 1)
        dst.write_colormap(1, {k: v[2] for k, v in LEGEND.items()} | {NODATA: (0, 0, 0, 0)})
        dst.update_tags(map_date=map_date, **{f"class_{k}": v[0] for k, v in LEGEND.items()})
        dst.set_band_description(1, "standing rice classes, see class_* tags")
    acre = res * res / 4046.8564224
    counts = np.bincount(data.ravel(), minlength=256)
    row = {"aoi": aoi}
    for k, v in LEGEND.items():
        row[f"{v[0]} (acres)"] = round(float(counts[k]) * acre, 1)
    row["mapped (acres)"] = round(float(counts[list(LEGEND)].sum()) * acre, 1)
    row["file"] = out.name
    return row


def package(out_dir, map_date: str | None = None, src_root=SRC) -> pd.DataFrame:
    """Every AOI with a map; writes the GeoTIFFs, ``acres_by_class.csv`` and ``legend.csv``."""
    if map_date is None:
        map_date = latest_window(src_root)
    aois = sorted({p.parent.name for p in Path(src_root).glob("aoi*/aoi*_monsoon2026.tif")},
                  key=lambda s: int(s[3:]))
    rows = [package_aoi(a, out_dir, map_date, src_root) for a in aois]
    table = pd.DataFrame(rows)
    total = table.drop(columns=["aoi", "file"]).sum().round(1)
    table = pd.concat([table, pd.DataFrame([{"aoi": "TOTAL", **total.to_dict(), "file": ""}])], ignore_index=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    table.to_csv(Path(out_dir) / "acres_by_class.csv", index=False)
    legend_table().to_csv(Path(out_dir) / "legend.csv", index=False)
    fields_dir = Path(src_root) / "fields"                  # analysis/field_rice (phase 7)
    if fields_dir.exists():
        import shutil

        out_fields = Path(out_dir) / "fields"
        out_fields.mkdir(parents=True, exist_ok=True)
        for f in sorted(fields_dir.glob("aoi*_fields_monsoon2026.gpkg")):
            shutil.copy2(f, out_fields / f.name.replace("monsoon2026", f"standing_rice_{map_date}"))
        if (fields_dir / "field_acres_by_class.csv").exists():
            shutil.copy2(fields_dir / "field_acres_by_class.csv", Path(out_dir) / "field_acres_by_class.csv")
    note = Path(__file__).resolve().parents[2] / "docs" / "delivery_methods_note.md"
    if note.exists():
        (Path(out_dir) / "METHODS.md").write_text(note.read_text())
    return table


def latest_window(src_root=SRC) -> str:
    """The map date: the last 5-day window of the series (read from any AOI's NDVI stack)."""
    import rasterio

    any_stack = next(Path(src_root).glob("aoi*/aoi*_ndvi5d.tif"))
    with rasterio.open(any_stack) as ds:
        return str(pd.to_datetime(ds.descriptions[-1]).date())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.delivery", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=f"{SRC}/delivery")
    p.add_argument("--map-date", default=None, help="YYYY-MM-DD (default: the last 5-day window)")
    args = p.parse_args(argv)
    t = package(args.out, args.map_date)
    print(t.tail(1).T.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
