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
the README gives the QGIS band mapping for each colour combination. Band *names* are written into
the file, so readers should look a band up by name (:func:`band_index`) rather than by position.

Why a SWIR band set exists
--------------------------
Deciding whether a field held water at sowing needs short-wave infrared. Water absorbs SWIR almost
completely while dry bare soil is bright there, which is the contrast that
LSWI = (B8 - B11) / (B8 + B11) turns into a flooding test. Without SWIR the only available water
proxy is NDWI (green, NIR), and over these fields NDWI is a near-mirror of NDVI (measured r = -0.969
over 904,458 clear pixel-dates), so it reports bare ground rather than water. :data:`BANDS_WITH_SWIR`
adds B11 and B12; they are 20 m bands and are resampled onto the AOI's 10 m grid like B5 already is.

Why a band set with the mask bands exists
-----------------------------------------
The 5-day composites (``s2_windows``) mask with s2cloudless, its projected shadow and a 50 m buffer
*inside* Earth Engine. Looking at the NDVI curves showed that recipe throwing away a large share of
clear observations, and because the mask is baked into the file, changing it means exporting again.

:data:`BANDS_WITH_MASKS` exports every acquisition date **unmasked**, with the two mask bands that
come with the product, so the mask is chosen on this machine and can be changed without a new export:

* ``QA60`` — the product's own cloud bits: bit 10 opaque cloud, bit 11 cirrus. This is the mask
  applied (:func:`qa60_cloud`). It is the lightest mask available, and it has no shadow class, which
  is deliberate: a flooded paddy at transplanting is dark, and shadow masks tend to take it for a
  shadow and delete the very observation that shows the water.
* ``SCL`` — the scene classification (3 shadow, 8/9 cloud, 10 cirrus). Kept for comparison only.

A light mask lets some haze through, and haze always *lowers* NDVI. The smoother that reads these
files is expected to follow the upper envelope of the curve for that reason.

In Earth Engine's copy of the collection QA60 was empty between January 2022 and February 2024;
:func:`mask_summary` checks that it is populated for the dates in hand before anything relies on it.
"""
from __future__ import annotations

import datetime as dt
import json

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
BANDS = ("B2", "B3", "B4", "B5", "B8")
#: Band set that can answer the water-at-sowing question; see the module docstring.
BANDS_WITH_SWIR = BANDS + ("B11", "B12")
OUTPUT_BANDS = BANDS + ("clear",)
#: Every acquisition, unmasked, with the product's own mask bands; see the module docstring.
BANDS_WITH_MASKS = BANDS_WITH_SWIR + ("QA60", "SCL")
#: QA60 bits: 10 = opaque cloud, 11 = cirrus.
QA60_CLOUD_BITS = (1 << 10) | (1 << 11)
#: SCL classes treated as cloud when comparing masks: 3 shadow, 8 and 9 cloud, 10 cirrus.
SCL_CLOUD_CLASSES = (3, 8, 9, 10)


def qa60_cloud(qa60):
    """True where QA60 flags opaque cloud or cirrus. Works on any integer numpy array."""
    import numpy as np

    return (np.asarray(qa60).astype("int64") & QA60_CLOUD_BITS) != 0


def start_with_retries(make_task, attempts: int = 5, pause: float = 2.0, sleep=None):
    """Build and start an export task, retrying transient Earth Engine errors.

    Why: one rejected ``task.start()`` in a long run used to end the run, and everything after it was
    never submitted. A new task is built on every attempt, because Earth Engine ties the request id to
    the task object and retrying the same object collides with itself.
    """
    import time

    import ee

    sleep = sleep or time.sleep
    last = None
    for attempt in range(attempts):
        try:
            task = make_task()
            task.start()
            return task
        except ee.ee_exception.EEException as error:
            last = error
            sleep(pause * (2 ** attempt))
    raise RuntimeError(f"could not start the export after {attempts} attempts") from last


def submit_each(items, submit_one, log=print) -> list[dict]:
    """Call ``submit_one(item)`` for every item, recording a failure and carrying on.

    ``submit_one`` returns a row dict; a failure becomes a row with ``error`` set, so one bad date
    never stops the dates after it. Re-running the export picks the failed ones up, because only
    files that exist on GCS are skipped.
    """
    rows = []
    for item in items:
        try:
            row = submit_one(item)
            row.setdefault("error", None)
        except Exception as error:  # noqa: BLE001 - one date must not abandon the rest
            row = {"item": item, "error": f"{type(error).__name__}: {error}"}
            log(f"  {item}: {row['error']}")
        rows.append(row)
    return rows


def band_index(dataset, name: str) -> int:
    """1-based band number of ``name`` in an exported file, by its written band description.

    Positions differ between band sets (``clear`` is band 6 without SWIR and band 8 with it), so
    nothing should hard-code a band number.
    """
    try:
        return dataset.descriptions.index(name) + 1
    except ValueError:
        raise KeyError(f"band {name!r} not in this file: {dataset.descriptions}") from None

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


def export_image(date: str, grid_def: dict, bucket: str, name: str, region_4326, bands=BANDS):
    """Start one export of the same-day mosaic on ``date``; returns the started task."""
    import ee

    day = ee.Date(date)
    region = ee.Geometry(region_4326)
    s2 = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(day, day.advance(1, "day"))
          .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), ["cs_cdf"]))
    mosaic = s2.mosaic()
    image = (mosaic.select(list(bands))
             .addBands(mosaic.select("cs_cdf").multiply(100).rename("clear"))
             .toUint16())
    description = name.rsplit("/", 1)[-1].replace("-", "")[:95]
    return start_with_retries(lambda: ee.batch.Export.image.toCloudStorage(
        image=image, description=description, bucket=bucket, fileNamePrefix=name,
        fileFormat="GeoTIFF", formatOptions={"cloudOptimized": True}, maxPixels=1e10,
        **grid_export_params(grid_def)))


def plan_and_export(configs, start_month: str, end_month: str, bucket: str, prefix: str,
                    submit: bool = False, log=print, bands=BANDS):
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
                    row["task_id"] = export_image(date, grid_def, bucket, name, rect, bands).id
            rows.append(row)
            log(f"{aoi} {month}: {date} clear {clear}% of AOI ({n_dates} dates)"
                + (" -> submitted" if row["task_id"] else ""))
    return rows


def dates_to_export(available, month_of, existing_names, prefix, aoi_key):
    """Dates whose file does not exist yet. ``available`` are ``YYYY-MM-DD`` strings.

    A date counts as done when its object name (built exactly as :func:`object_name` builds it)
    is already in ``existing_names``, so rerunning never exports the same image twice.
    """
    todo = []
    for date in sorted(set(available)):
        name = object_name(prefix, aoi_key, month_of(date), date) + ".tif"
        if name not in existing_names:
            todo.append(date)
    return todo


def scene_dates(region_4326, start: str, end: str):
    """Distinct acquisition dates over a region, with the lowest scene cloud % seen that day.

    One metadata request; no pixels are read. The scene cloud % describes whole tiles, so it is
    recorded for reference only and not used to choose anything.
    """
    import ee

    region = ee.Geometry(region_4326)
    col = ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end)
    pairs = col.reduceColumns(ee.Reducer.toList(2),
                              ["system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]).getInfo()["list"]
    best = {}
    for epoch_ms, cloud in pairs:
        day = dt.datetime.fromtimestamp(epoch_ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")
        best[day] = min(best.get(day, 101.0), float(cloud))
    return best


def export_all_dates(configs, start: str, end: str, bucket: str, prefix: str, log=print, bands=BANDS):
    """Export every available date per AOI in [start, end), skipping files already on GCS.

    Returns manifest rows. Exports start immediately; clouds are not masked. ``prefix`` selects the
    GCS folder, so a different band set must be given its own prefix rather than overwriting an
    existing one: a file already there is skipped by name and would otherwise keep the old bands.
    """
    import geopandas as gpd
    from google.cloud import storage

    from . import auth
    from . import config as config_mod

    rows = []
    for path in configs:
        cfg = config_mod.load_config(path)
        aoi = cfg["aoi"]["key"]
        grid_def = json.loads((config_mod.grid_dir(cfg) / "grid_def.json").read_text())
        polygon = aoi_geojson_2d(gpd.read_file(config_mod.aoi_path(cfg)))
        w, s, e, n = grid_bounds_4326(grid_def)
        rect = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        client = storage.Client(credentials=auth.credentials(cfg), project=cfg["auth"]["project"])
        existing = {b.name for b in client.list_blobs(bucket, prefix=f"{prefix.strip('/')}/{aoi}/")}
        clouds = scene_dates(polygon, start, end)
        todo = dates_to_export(clouds, lambda d: d[:7], existing, prefix, aoi)
        log(f"{aoi}: {len(clouds)} dates available, {len(clouds) - len(todo)} already on GCS, "
            f"{len(todo)} to export")
        def submit_one(date):
            name = object_name(prefix, aoi, date[:7], date)
            task = export_image(date, grid_def, bucket, name, rect, bands)
            return {"aoi": aoi, "month": date[:7], "date": date,
                    "scene_cloud_pct_min": round(clouds[date], 1),
                    "gcs_uri": f"gs://{bucket}/{name}.tif", "task_id": task.id,
                    "selection": "all_dates"}

        done = submit_each(todo, submit_one, log)
        failed = sum(1 for r in done if r["error"])
        if failed:
            log(f"{aoi}: {failed} of {len(todo)} dates could not be submitted; run again to retry them")
        rows.extend(done)
    return rows


def mask_summary(region_4326, start: str, end: str, scale: int = 20):
    """Per acquisition date over a region: data coverage, and cloud share by QA60 and by SCL.

    Read-only (one ``getInfo``); nothing is exported. Answers two questions before a mask is
    trusted: is QA60 populated at all for these dates (``qa60_present_pct``), and how far do the
    light QA60 mask and the SCL mask disagree (``qa60_cloud_pct`` vs ``scl_cloud_pct``). Cloud
    shares are percentages of the pixels that have data that day.
    """
    import ee
    import pandas as pd

    region = ee.Geometry(region_4326)
    col = ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end)
    dates = col.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct()

    def one(d):
        day = ee.Date.parse("YYYY-MM-dd", d)
        mosaic = col.filterDate(day, day.advance(1, "day")).mosaic()
        data = mosaic.select("B4").mask().gt(0)
        qa = mosaic.select("QA60")
        scl = mosaic.select("SCL")
        stack = ee.Image.cat([
            data.rename("coverage"),
            qa.mask().gt(0).updateMask(data).rename("qa60_present"),
            qa.bitwiseAnd(QA60_CLOUD_BITS).neq(0).updateMask(data).rename("qa60_cloud"),
            scl.remap(list(SCL_CLOUD_CLASSES), [1] * len(SCL_CLOUD_CLASSES), 0)
               .updateMask(data).rename("scl_cloud"),
        ])  # left masked: reduceRegion skips masked pixels, so shares are of the pixels with data
        means = stack.reduceRegion(ee.Reducer.mean(), region, scale, maxPixels=1e10)
        return ee.Feature(None, means.set("date", d))

    table = ee.FeatureCollection(dates.map(one)).getInfo()["features"]
    frame = pd.DataFrame([f["properties"] for f in table])
    if frame.empty:
        return frame
    for col_name in ("coverage", "qa60_present", "qa60_cloud", "scl_cloud"):
        frame[f"{col_name}_pct"] = (100 * frame.pop(col_name).astype(float)).round(1)
    return frame.sort_values("date").reset_index(drop=True)


def wait_for_tasks(task_ids, poll_seconds: float = 30, status_of=None, sleep=None, log=print) -> dict:
    """Block until every task has finished; return ``{task_id: final state}``.

    Why: the Sentinel-2 exports are started and forgotten, so without this a download could begin
    while half the files are still being written, and a failed task would only show up later as a
    missing date. States come from Earth Engine in batches of up to 100 ids per request.
    """
    import time

    sleep = sleep or time.sleep
    if status_of is None:
        import ee

        def status_of(ids):
            return {s["id"]: s["state"] for s in ee.data.getTaskStatus(list(ids))}

    active = {"UNSUBMITTED", "READY", "RUNNING", "CANCEL_REQUESTED"}
    ids = [t for t in task_ids if t]
    states = {}
    while True:
        for i in range(0, len(ids), 100):
            states.update(status_of(ids[i:i + 100]))
        running = [t for t in ids if states.get(t) in active]
        done = {s: sum(1 for t in ids if states.get(t) == s) for s in ("COMPLETED", "FAILED", "CANCELLED")}
        log(f"tasks: {len(ids) - len(running)} of {len(ids)} finished {done}")
        if not running:
            return states
        sleep(poll_seconds)
