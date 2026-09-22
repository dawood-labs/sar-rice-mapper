"""Export one clearest Sentinel-2 image per month per AOI, aligned to the AOI's SAR grid.

Why
---
A SAR-derived rice map has to be checked by eye against something a person can read. Sentinel-2 in
true colour (bands 4-3-2), false colour (8-3-2) or 5-3-2 shows fields, water, bare soil and canopy
directly. One good image per month, laid over the class map in QGIS, turns "the model says rice"
into something anyone can verify.

Three choices that matter
-------------------------
* **Clearest over the AOI, not over the tile.** A scene's cloud percentage describes the whole
  ~110 km tile. The AOI can sit under a cloud on a "5% cloud" day, or be clear on a "60%" day. So
  each candidate *date* is scored by the share of the AOI's own pixels that Cloud Score+ calls clear
  (``cs_cdf >= threshold``), and the best date wins. Tiles from the same day are mosaicked first,
  because an AOI on a tile edge would otherwise be scored on half its area.
* **Exactly the SAR grid.** Exports use the AOI's grid CRS, origin and 10 m pixel size, so every
  optical pixel sits exactly on a class-map pixel. The 20 m red-edge band is resampled onto that grid.
* **A clear-score band is included** so a reader sees, per pixel, how much to trust what they are
  looking at. In the monsoon months even the best available date may be partly cloudy.

Output bands, in file order: ``B2`` (blue), ``B3`` (green), ``B4`` (red), ``B5`` (red edge),
``B8`` (NIR), ``clear`` (Cloud Score+ x 100). Reflectances are the collection's integer values
(reflectance x 10000). Note that the file band number differs from the Sentinel-2 band number;
the README gives the QGIS band mapping for each colour combination.
"""
from __future__ import annotations

import json

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
BANDS = ("B2", "B3", "B4", "B5", "B8")
OUTPUT_BANDS = BANDS + ("clear",)

#: QGIS "Multiband color" settings for the usual combinations, as *file* band numbers.
QGIS_COMBINATIONS = {
    "true colour (4-3-2)": {"red": 3, "green": 2, "blue": 1},
    "false colour (8-3-2)": {"red": 5, "green": 2, "blue": 1},
    "red edge (5-3-2)": {"red": 4, "green": 2, "blue": 1},
}


def months(start: str, end: str) -> list[str]:
    """``YYYY-MM`` for every calendar month from ``start`` up to but excluding ``end`` (both YYYY-MM)."""
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) < (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_bounds(month: str) -> tuple[str, str]:
    """Start (inclusive) and end (exclusive) dates of a ``YYYY-MM`` month."""
    y, m = (int(x) for x in month.split("-"))
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y:04d}-{m:02d}-01", f"{ny:04d}-{nm:02d}-01"


def grid_export_params(grid_def: dict) -> dict:
    """``crs``, ``crsTransform`` and ``dimensions`` that reproduce the SAR grid exactly."""
    res = float(grid_def["res"])
    return {"crs": grid_def["crs"],
            "crsTransform": [res, 0, float(grid_def["x0"]), 0, -res, float(grid_def["y0"])],
            "dimensions": f"{int(grid_def['width'])}x{int(grid_def['height'])}"}


def grid_bounds_4326(grid_def: dict):
    """The grid's extent as a lon/lat rectangle, for filtering the collection."""
    import pyproj

    res = float(grid_def["res"])
    x0, y0 = float(grid_def["x0"]), float(grid_def["y0"])
    x1, y1 = x0 + res * int(grid_def["width"]), y0 - res * int(grid_def["height"])
    tf = pyproj.Transformer.from_crs(grid_def["crs"], "EPSG:4326", always_xy=True)
    lons, lats = tf.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return min(lons), min(lats), max(lons), max(lats)


def object_name(prefix: str, aoi_key: str, month: str, date: str) -> str:
    """GCS object name (without ``.tif``): one folder per AOI, month and exact date in the name."""
    return f"{prefix.strip('/')}/{aoi_key}/{aoi_key}_{month}_S2_{date}"


def aoi_geojson_2d(frame):
    """Union of an AOI's polygons as 2D GeoJSON in EPSG:4326.

    AOI files can carry a third (Z) coordinate, and Earth Engine rejects 3D GeoJSON with an
    unhelpful "Invalid GeoJSON geometry" (the same trap the audit hit), so Z is dropped here.
    """
    import shapely
    from shapely.geometry import mapping

    return mapping(shapely.force_2d(frame.to_crs(4326).geometry.union_all()))


def best_date(aoi_polygon_4326, month: str, threshold: float = 0.6):
    """The date in ``month`` whose same-day mosaic has the largest clear share over the AOI polygon.

    Returns ``(date or None, clear_pct or None, number_of_candidate_dates)``. Areas with no data at
    all that day count as not clear, so a date that only half-covers the AOI cannot win on its half.
    """
    import ee

    start, end = month_bounds(month)
    region = ee.Geometry(aoi_polygon_4326)
    s2 = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end)
          .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), ["cs_cdf"]))
    dates = s2.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct()

    def score(d):
        day = ee.Date.parse("YYYY-MM-dd", d)
        mosaic = s2.filterDate(day, day.advance(1, "day")).mosaic()
        clear = mosaic.select("cs_cdf").gte(threshold).unmask(0)
        frac = clear.reduceRegion(ee.Reducer.mean(), region, 30, maxPixels=1e10).get("cs_cdf")
        return ee.Feature(None, {"date": d, "clear": frac})

    table = ee.FeatureCollection(dates.map(score)).sort("clear", False).getInfo()["features"]
    if not table:
        return None, None, 0
    top = table[0]["properties"]
    return top["date"], round(100 * float(top["clear"] or 0), 1), len(table)


def export_image(date: str, grid_def: dict, bucket: str, name: str, region_4326):
    """Start one export of the same-day mosaic on ``date``; returns the started task."""
    import ee

    day = ee.Date(date)
    region = ee.Geometry(region_4326)
    s2 = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(day, day.advance(1, "day"))
          .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), ["cs_cdf"]))
    mosaic = s2.mosaic()
    image = (mosaic.select(list(BANDS))
             .addBands(mosaic.select("cs_cdf").multiply(100).rename("clear"))
             .toUint16())
    description = name.rsplit("/", 1)[-1].replace("-", "")[:95]
    task = ee.batch.Export.image.toCloudStorage(
        image=image, description=description, bucket=bucket, fileNamePrefix=name,
        fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True}, maxPixels=1e10,
        **grid_export_params(grid_def))
    task.start()
    return task


def plan_and_export(configs, start_month: str, end_month: str, bucket: str, prefix: str,
                    submit: bool = False, log=print):
    """For every config and month: pick the clearest date, and (with ``submit``) export it.

    Returns a list of dict rows for a manifest. Without ``submit`` nothing is exported; the plan
    alone reads Earth Engine metadata and writes nothing anywhere.
    """
    import geopandas as gpd

    from . import config as config_mod

    rows = []
    for path in configs:
        cfg = config_mod.load_config(path)
        aoi = cfg["aoi"]["key"]
        grid_def = json.loads((config_mod.grid_dir(cfg) / "grid_def.json").read_text())
        polygon = aoi_geojson_2d(gpd.read_file(config_mod.aoi_path(cfg)))
        w, s, e, n = grid_bounds_4326(grid_def)
        rect = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        for month in months(start_month, end_month):
            date, clear, n_dates = best_date(polygon, month)
            row = {"aoi": aoi, "month": month, "date": date, "clear_pct_over_aoi": clear,
                   "candidate_dates": n_dates, "gcs_uri": None, "task_id": None}
            if date is not None:
                name = object_name(prefix, aoi, month, date)
                row["gcs_uri"] = f"gs://{bucket}/{name}.tif"
                if submit:
                    row["task_id"] = export_image(date, grid_def, bucket, name, rect).id
            rows.append(row)
            log(f"{aoi} {month}: {date} clear {clear}% of AOI ({n_dates} dates)"
                + (" -> submitted" if row["task_id"] else ""))
    return rows
