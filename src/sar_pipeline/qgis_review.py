"""Everything needed to check the maps by eye in QGIS, as few files as possible.

Why
---
The maps exist per AOI and per version (the first water test, the current rule, the sieved finals),
which makes downloading and comparing them in QGIS tedious. This module builds:

* ``maps_all_versions_mosaic.tif`` — one GeoTIFF over all AOIs, **one band per version**, same 10 m
  grid (all AOIs share EPSG:32646), 255 = no data. Bands (also in the band descriptions):
  1 ``rule_v1`` (first water test, per pixel, unsieved), 2 ``rule_current`` (water test v2 with
  artefact passes dropped and class 6 young rice, per pixel, unsieved), 3 ``final_previous``
  (v2 + the user's relabel + 4-pixel sieve, before class 6), 4 ``final_current`` (current rule +
  relabel + sieve; the map the field labels come from);
* ``classes.qml`` — a QGIS style with the class colours (load it on any band);
* ``fields_all_aois.gpkg`` — every delineated field with its label, in one file;
* ``review_aois.csv`` — the AOIs picked for review (mostly rice, least rice, mixed; from all
  directions), with their class shares and the cloud-storage folder of their Sentinel-2 images.

The GDAL command-line tools do the mosaicking (``gdalbuildvrt``, ``gdal_translate``, ``ogr2ogr``).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

SRC = "processed/_batch/s2_2026"
OUT = f"{SRC}/qgis_review"
VERSIONS = (("rule_v1", "_v1"), ("rule_current", ""), ("final_previous", "_final_prev"), ("final_current", "_final"))
COLOURS = {0: ("not rice", "#e6e6e6"), 1: ("rice, standing", "#008c3c"), 2: ("young", "#aadc78"),
           3: ("rice-like, water not confirmed", "#f5a028"), 4: ("harvested", "#96643c"),
           5: ("never bare (trees/houses)", "#8c5abe"), 6: ("young rice, standing", "#6ec83c")}


def version_files(suffix: str, src_root=SRC) -> list[str]:
    """Each AOI's map of one version: exactly ``<aoi>/<aoi>_monsoon2026<suffix>.tif``."""
    return sorted(str(d / f"{d.name}_monsoon2026{suffix}.tif") for d in Path(src_root).glob("aoi*")
                  if (d / f"{d.name}_monsoon2026{suffix}.tif").exists())


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def mosaic(out_dir=OUT, src_root=SRC) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vrts = []
    for name, suffix in VERSIONS:
        files = version_files(suffix, src_root)
        lst = out / f"{name}_files.txt"
        lst.write_text("\n".join(files))
        vrt = out / f"{name}.vrt"
        run(["gdalbuildvrt", "-q", "-srcnodata", "255", "-vrtnodata", "255", "-input_file_list", str(lst), str(vrt)])
        vrts.append(str(vrt))
    stack = out / "maps_all_versions.vrt"
    run(["gdalbuildvrt", "-q", "-separate", "-srcnodata", "255", "-vrtnodata", "255", str(stack), *vrts])
    tif = out / "maps_all_versions_mosaic.tif"
    run(["gdal_translate", "-q", "-of", "GTiff", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
         "-co", "BIGTIFF=IF_SAFER", str(stack), str(tif)])
    import rasterio

    with rasterio.open(tif, "r+") as ds:
        for i, (name, _) in enumerate(VERSIONS, start=1):
            ds.set_band_description(i, name)
    run(["gdaladdo", "-q", "-r", "nearest", str(tif), "2", "4", "8", "16"])
    return tif


def qml(out_dir=OUT) -> Path:
    items = "\n".join(f'        <paletteEntry value="{k}" label="{k} {n}" color="{c}" alpha="255"/>'
                      for k, (n, c) in COLOURS.items())
    text = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{items}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    p = Path(out_dir) / "classes.qml"
    p.write_text(text)
    return p


def merge_fields(out_dir=OUT, src_root=SRC) -> Path:
    out = Path(out_dir) / "fields_all_aois.gpkg"
    out.unlink(missing_ok=True)
    for i, f in enumerate(sorted(Path(src_root, "fields").glob("aoi*_fields_monsoon2026.gpkg"))):
        cmd = ["ogr2ogr", "-f", "GPKG", str(out), str(f), "-nln", "fields"]
        if i:
            cmd.insert(1, "-append")
        run(cmd)
    return out


def review_aois(ids, out_dir=OUT, bucket_folder: str | None = None, shares_csv=f"{SRC}/report/aoi_final_shares.csv") -> pd.DataFrame:
    """The AOIs picked for review with their shares and where their Sentinel-2 images are."""
    t = pd.read_csv(shares_csv)
    t = t[t["aoi"].isin([f"aoi{i}" for i in ids])].copy()
    if bucket_folder:
        t["s2_images_gcs"] = t["aoi"].map(lambda a: f"gs://{bucket_folder}/s2_dates_masks/{a}/")
    t["s2_images_local"] = t["aoi"].map(lambda a: f"data/s2_dates_masks/{a}/")
    keep = [c for c in ("aoi", "region", "lat", "lon", "acres", "rice_pct", "unconf_pct", "notrice_pct",
                        "s2_images_gcs", "s2_images_local") if c in t]
    t = t[keep].round(2)
    t.to_csv(Path(out_dir) / "review_aois.csv", index=False)
    return t


# --- looking at one place: point -> AOI, pixel, field; curves and chips --------------------------------
def locate(lon: float, lat: float, src_root="processed", aoi_dir="data/aoi"):
    """(aoi_id, pixel id, row, col) of the 10 m pixel under a lon/lat point inside an AOI polygon, or None.

    Copy the coordinates from QGIS (right-click > Copy Coordinate, in EPSG:4326)."""
    import json

    import geopandas as gpd
    import pyproj
    from shapely.geometry import Point

    x, y = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32646", always_xy=True).transform(lon, lat)
    for gpath in sorted(Path(src_root).glob("aoi*/monsoon2026/grid/grid_def.json")):
        g = json.loads(gpath.read_text())
        col = int((x - g["x0"]) // g["res"])
        row = int((g["y0"] - y) // g["res"])
        if not (0 <= col < int(g["width"]) and 0 <= row < int(g["height"])):
            continue
        aoi_id = int(gpath.parts[-4][3:])
        poly = gpd.read_file(Path(aoi_dir) / f"aoi_{aoi_id:03d}.gpkg").to_crs(4326).geometry.union_all()
        if poly.contains(Point(lon, lat)):
            return aoi_id, row * int(g["width"]) + col, row, col
    return None


def field_at(aoi_id: int, lon: float, lat: float):
    """The delineated field under the point (GeoDataFrame row in EPSG:32646) and its pixel ids."""
    import geopandas as gpd
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import Point

    from .analysis import field_rice as fr
    from .analysis import pixel_report as pr

    path = Path(SRC) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg"
    fields = gpd.read_file(path) if path.exists() else fr.load_fields(aoi_id)
    pt = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(fields.crs).iloc[0]
    hit = fields[fields.contains(pt)]
    if hit.empty:
        return None, None
    f = hit.iloc[[int(np.argmin(hit.to_crs("EPSG:32646").geometry.area.to_numpy()))]].to_crs("EPSG:32646")  # finest tracing
    g = pr.locate(aoi_id, 0, season_key="monsoon2026")["grid"]
    mask = rasterize([(f.geometry.iloc[0], 1)], out_shape=(int(g["height"]), int(g["width"])),
                     transform=from_origin(g["x0"], g["y0"], g["res"], g["res"]), fill=0, dtype="uint8",
                     all_touched=False)
    pids = np.flatnonzero(mask.ravel())
    return f, pids


def classes_at(aoi_id: int, pid: int) -> dict:
    """The class of the pixel in every map version (``VERSIONS``)."""
    import rasterio

    out = {}
    for name, suffix in VERSIONS:
        p = Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026{suffix}.tif"
        if p.exists():
            with rasterio.open(p) as ds:
                v = int(ds.read(1).ravel()[pid])
            out[name] = f"{v} {COLOURS.get(v, ('no data', ''))[0]}"
    return out


def inspect(lon: float, lat: float, block: int = 3, out_dir=f"{OUT}/inspect") -> dict:
    """Curves (NDVI with clear/hazy observations, VV/VH per track) and every clear 5-3-2 chip for the
    pixel block at the point and, if the point is in a delineated field, for the whole field (outline
    drawn on the chips). PNGs and a text summary go to ``out_dir``; the summary is also returned."""
    import numpy as np

    from .analysis import ndvi_5day as nd
    from .analysis import pixel_2026 as p26
    from .analysis import water_investigation as wi

    hit = locate(lon, lat)
    if hit is None:
        return {"error": "the point is not inside any AOI"}
    aoi_id, pid, row, col = hit
    stem = f"aoi{aoi_id}_{lat:.5f}_{lon:.5f}"
    info = {"aoi": f"aoi{aoi_id}", "pixel_id": pid, "row": row, "col": col, "lon": lon, "lat": lat,
            "classes_at_pixel": classes_at(aoi_id, pid)}
    figs = {}
    figs["pixels_curve"], figs["pixels_chips"], _ = wi.group_sheet(
        aoi_id, pid, block=block, out_dir=out_dir, file_stem=f"{stem}_pixels{block}x{block}",
        label=f"({block}x{block} pixels at the point)")
    f, fpids = field_at(aoi_id, lon, lat)
    if f is not None and len(fpids):
        d = nd.load(aoi_id)
        g = d["loc"]["grid"]
        rr, cc = np.divmod(fpids, int(g["width"]))
        centre = int(np.round(np.median(rr)) * int(g["width"]) + np.round(np.median(cc)))
        r0, c0 = divmod(centre, int(g["width"]))
        xs, ys = f.geometry.iloc[0].exterior.xy if f.geometry.iloc[0].geom_type == "Polygon" else \
            max(f.geometry.iloc[0].geoms, key=lambda q: q.area).exterior.xy
        ox = (np.asarray(xs) - g["x0"]) / g["res"] - 0.5 - (c0 - p26.HALF)
        oy = (g["y0"] - np.asarray(ys)) / g["res"] - 0.5 - (r0 - p26.HALF)
        row_ = f.iloc[0]
        info["field"] = {k: (row_[k].item() if hasattr(row_[k], "item") else row_[k])
                         for k in ("uid", "area_acres", "pixels", "label", "class_name", "rice_share",
                                   "unconfirmed_share", "label_confidence") if k in f.columns}
        figs["field_curve"], figs["field_chips"], _ = wi.group_sheet(
            aoi_id, centre, block=block, out_dir=out_dir, file_stem=f"{stem}_field",
            label=f"(whole field, {len(fpids)} px)", pids=fpids, outline=(ox, oy))
    else:
        info["field"] = None
    nd.forget()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}_info.txt").write_text("\n".join(f"{k}: {v}" for k, v in info.items()))
    info["figures"] = figs        # matplotlib figures, for display in a notebook
    return info


def field_by_id(field_id: str):
    """(aoi_id, field row in EPSG:32646, pixel ids) for a ``field_id`` such as ``aoi116_000123``."""
    import geopandas as gpd
    import numpy as np
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    from .analysis import pixel_report as pr

    aoi_id = int(field_id.split("_")[0][3:])
    path = Path(SRC) / "fields" / f"aoi{aoi_id}_fields_monsoon2026.gpkg"
    f = gpd.read_file(path, where=f"field_id = '{field_id}'")
    if f.empty:
        raise ValueError(f"{field_id} not found in {path}")
    f = f.to_crs("EPSG:32646")
    g = pr.locate(aoi_id, 0, season_key="monsoon2026")["grid"]
    mask = rasterize([(f.geometry.iloc[0], 1)], out_shape=(int(g["height"]), int(g["width"])),
                     transform=from_origin(g["x0"], g["y0"], g["res"], g["res"]), fill=0, dtype="uint8")
    return aoi_id, f, np.flatnonzero(mask.ravel())


def inspect_pixel(aoi_id: int, pid: int, block: int = 3, out_dir=f"{OUT}/inspect") -> dict:
    """Curves and chips for a pixel block, by AOI and pixel id (the pixel id is the value of the AOI's
    ``grid/pixel_index.tif`` in QGIS)."""
    from .analysis import ndvi_5day as nd
    from .analysis import water_investigation as wi

    info = {"aoi": f"aoi{aoi_id}", "pixel_id": pid, "classes_at_pixel": classes_at(aoi_id, pid)}
    figs = {}
    figs["curve"], figs["chips"], ev = wi.group_sheet(aoi_id, int(pid), block=block, out_dir=out_dir,
                                                      file_stem=f"aoi{aoi_id}_pid{pid}_{block}x{block}",
                                                      label=f"({block}x{block} pixels)")
    info["events"] = {k: str(ev[k]) for k in ("trough_date", "climb_date", "trough_ndvi", "peak_after", "last_ndvi")}
    nd.forget()
    info["figures"] = figs
    return info


def inspect_field(field_id: str, out_dir=f"{OUT}/inspect") -> dict:
    """Curves (averaged over all the field's pixels) and chips with the field outline, by ``field_id``."""
    import numpy as np

    from .analysis import ndvi_5day as nd
    from .analysis import pixel_2026 as p26
    from .analysis import water_investigation as wi

    aoi_id, f, fpids = field_by_id(field_id)
    row = f.iloc[0]
    info = {"field_id": field_id, "aoi": f"aoi{aoi_id}", "pixels": int(len(fpids)),
            **{k: (row[k].item() if hasattr(row[k], "item") else row[k])
               for k in ("area_acres", "label", "class_name", "rice_share", "unconfirmed_share",
                         "label_confidence", "field_rule_label") if k in f.columns}}
    if not len(fpids):
        info["note"] = "the field is smaller than one pixel: no curve of its own"
        return info
    d = nd.load(aoi_id)
    g = d["loc"]["grid"]
    rr, cc = np.divmod(fpids, int(g["width"]))
    centre = int(np.round(np.median(rr)) * int(g["width"]) + np.round(np.median(cc)))
    r0, c0 = divmod(centre, int(g["width"]))
    geom = row.geometry if row.geometry.geom_type == "Polygon" else max(row.geometry.geoms, key=lambda q: q.area)
    xs, ys = geom.exterior.xy
    ox = (np.asarray(xs) - g["x0"]) / g["res"] - 0.5 - (c0 - p26.HALF)
    oy = (g["y0"] - np.asarray(ys)) / g["res"] - 0.5 - (r0 - p26.HALF)
    info["classes_at_centre_pixel"] = classes_at(aoi_id, centre)
    figs = {}
    figs["curve"], figs["chips"], ev = wi.group_sheet(aoi_id, centre, out_dir=out_dir, file_stem=field_id,
                                                      label=f"field {field_id} ({len(fpids)} px)",
                                                      pids=fpids, outline=(ox, oy))
    info["events"] = {k: str(ev[k]) for k in ("trough_date", "climb_date", "trough_ndvi", "peak_after", "last_ndvi")}
    nd.forget()
    info["figures"] = figs
    return info


def main(argv=None) -> int:
    import argparse
    import os

    os.environ["PATH"] = "/opt/gis/bin:" + os.environ.get("PATH", "")      # GDAL tools on this machine
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.qgis_review", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build", help="mosaic of all versions, style, merged fields, review AOI list")
    b.add_argument("--review-ids", nargs="*", type=int, default=[])
    f = sub.add_parser("field", help="curves and chips of one field, by field_id")
    f.add_argument("field_id")
    x = sub.add_parser("pixel", help="curves and chips of a pixel block, by AOI and pixel id")
    x.add_argument("--aoi", type=int, required=True)
    x.add_argument("--pid", type=int, required=True)
    x.add_argument("--block", type=int, default=3)
    i = sub.add_parser("inspect", help="curves and chips at a lon/lat point (pixel block and its field)")
    i.add_argument("--lon", type=float, required=True)
    i.add_argument("--lat", type=float, required=True)
    i.add_argument("--block", type=int, default=3, help="pixel block size (1, 3 or 5)")
    args = p.parse_args(argv)
    if args.step == "build":
        import yaml

        mosaic(), qml(), merge_fields()
        cfg = yaml.safe_load(next(Path("config").glob("aoi*_monsoon2026.yaml")).read_text())
        print(review_aois(args.review_ids, bucket_folder=f"{cfg['gcs']['bucket']}/{cfg['gcs']['base_folder']}"))
    elif args.step in ("field", "pixel"):
        r = inspect_field(args.field_id) if args.step == "field" else inspect_pixel(args.aoi, args.pid, args.block)
        for k, v in r.items():
            if k != "figures":
                print(f"{k}: {v}")
        print(f"pictures in {OUT}/inspect/")
    else:
        for k, v in inspect(args.lon, args.lat, args.block).items():
            if k != "figures":
                print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
