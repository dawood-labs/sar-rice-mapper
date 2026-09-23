"""Cloud-masked Sentinel-2 composites on fixed calendar windows, one image per window.

Why this exists alongside ``optical_export``
--------------------------------------------
``optical_export`` writes one file per acquisition date and carries a Cloud Score+ band, leaving the
masking to the reader. That left two weaknesses the pixel checks exposed: a date whose *pixel* scored
clear could still sit under haze that Cloud Score+ does not flag, and the irregular dates had to be
interpolated onto a regular grid before any derivative could be taken, which is a second place for
the answer to change.

This module removes both. Each **fixed 5-day window** produces one image: every acquisition in the
window is masked with **s2cloudless**, and the median of what survives is taken per band. Masking
before compositing means a cloud in one acquisition is replaced by clear ground from another in the
same window rather than averaged with it, and the output already sits on a regular 5-day grid, so
the smoother reads the data directly with nothing interpolated first.

The mask
--------
The standard s2cloudless recipe, with the thresholds carried over from the sowing pipeline written
for an earlier rice project:

* **cloud** — s2cloudless probability above ``prob`` (70%);
* **shadow** — pixels that are dark in NIR (B8 below ``drk``) *and* not flagged as water by SCL, that
  also fall inside the cloud's shadow. The shadow is found by projecting each cloud along the solar
  azimuth for up to ``dist`` km with a directional distance transform, rather than guessing at it;
* **buffer** — the union is dilated by ``buf`` metres, because a cloud's soft edge contaminates
  pixels beyond where it is detected. ``focal_min(2)`` first removes specks so the dilation does not
  grow them.

A ``n_obs`` band counts how many acquisitions survived the mask in that window. **A pixel is valid
only where ``n_obs`` is at least 1**; the reflectance bands are unmasked to 0 for writing, so the
count is what tells clear ground from no data.
"""
from __future__ import annotations

import datetime as dt
import time

S2_SR = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_PROB = "COPERNICUS/S2_CLOUD_PROBABILITY"

#: Mask settings. ``prob`` is a percentage, ``drk`` a reflectance fraction, ``dist`` kilometres and
#: ``buf`` metres.
MSK = {"prob": 70, "drk": 0.15, "dist": 1, "buf": 50}

#: Bands written per window, in file order, followed by ``n_obs``.
BANDS = ("B2", "B3", "B4", "B5", "B8", "B11", "B12")
COUNT_BAND = "n_obs"
STEP_DAYS = 5


def windows(start: str, end: str, step: int = STEP_DAYS) -> list[tuple[str, str]]:
    """Fixed ``step``-day windows covering ``[start, end)``; the last one is clipped, never extended.

    Windows are anchored to ``start`` and never move, so the same calendar grid is produced for every
    AOI and every year — two AOIs can be compared window for window without resampling either.
    """
    first = dt.date.fromisoformat(start)
    last = dt.date.fromisoformat(end)
    out = []
    cursor = first
    while cursor < last:
        nxt = min(cursor + dt.timedelta(days=step), last)
        out.append((cursor.isoformat(), nxt.isoformat()))
        cursor = nxt
    return out


def object_name(prefix: str, aoi_key: str, window_start: str) -> str:
    """GCS object name (without ``.tif``): one folder per AOI, the window's first day in the name."""
    return f"{prefix.strip('/')}/{aoi_key}/{aoi_key}_W{STEP_DAYS}_{window_start}"


def mask_image(img):
    """Mask cloud and its projected shadow on one Sentinel-2 image joined to its s2cloudless band."""
    import ee

    cloud_prob = ee.Image(img.get("s2cloudless")).select("probability")
    is_cloud = cloud_prob.gt(MSK["prob"])

    not_water = img.select("SCL").neq(6)
    dark = img.select("B8").lt(MSK["drk"] * 1e4).multiply(not_water)

    azimuth = ee.Number(90).subtract(ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE")))
    projected = (is_cloud.directionalDistanceTransform(azimuth, MSK["dist"] * 10)
                 .reproject(crs=img.select(0).projection(), scale=100)
                 .select("distance").mask())

    bad = is_cloud.add(projected.multiply(dark)).gt(0)
    keep = (bad.focal_min(2)
            .focal_max(MSK["buf"] * 2 / 20.0)
            .reproject(crs=img.select(0).projection(), scale=20)
            .Not())
    return img.updateMask(keep)


def masked_collection(region, start: str, end: str):
    """Every acquisition over ``region`` in the window, cloud-masked, joined to s2cloudless."""
    import ee

    s2 = ee.ImageCollection(S2_SR).filterBounds(region).filterDate(start, end)
    clouds = ee.ImageCollection(S2_CLOUD_PROB).filterBounds(region).filterDate(start, end)
    joined = ee.Join.saveFirst("s2cloudless").apply(
        primary=s2, secondary=clouds,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"))
    return ee.ImageCollection(joined).map(mask_image)


def window_composite(region, start: str, end: str, bands=BANDS):
    """Median of the masked acquisitions in one window, plus how many survived.

    A fully masked dummy image is merged in so the result keeps its bands even when the window holds
    no usable acquisition at all; without it Earth Engine returns a band-less image and the export
    fails for the whole AOI because of one cloudy fortnight.
    """
    import ee

    bands = list(bands)
    collection = masked_collection(region, start, end).select(bands)
    dummy = ee.Image.constant([0] * len(bands)).rename(bands).toUint16().updateMask(
        ee.Image.constant(0))
    safe = collection.merge(ee.ImageCollection([dummy]))
    median = safe.median().unmask(0).toUint16()
    n_obs = safe.select(bands[0]).count().unmask(0).toUint16().rename(COUNT_BAND)
    return median.addBands(n_obs)


def export_window(region_4326, grid_def: dict, bucket: str, name: str, start: str, end: str,
                  bands=BANDS, attempts: int = 5, pause: float = 2.0):
    """Start one export of a window composite on the AOI's own grid; returns the started task.

    Retries on the transient errors a long submission run hits. Earth Engine rejected one submission
    in a 544-task run with "a different Operation was already started with the given request_id" —
    the client's generated id had collided — and because a single ``task.start()`` was all that stood
    between the run and the floor, five hundred already-queued exports were followed by a traceback
    and two whole AOIs never got submitted. A fresh ``Task`` is built on each attempt, because the id
    lives on the task object and retrying the same one would collide again.
    """
    import ee

    from .optical_export import grid_export_params

    region = ee.Geometry(region_4326)
    description = name.rsplit("/", 1)[-1].replace("-", "")[:95]
    last = None
    for attempt in range(attempts):
        try:
            task = ee.batch.Export.image.toCloudStorage(
                image=window_composite(region, start, end, bands), description=description,
                bucket=bucket, fileNamePrefix=name, fileFormat="GeoTIFF",
                formatOptions={"cloudOptimized": True}, maxPixels=1e10,
                **grid_export_params(grid_def))
            task.start()
            return task
        except ee.ee_exception.EEException as error:
            last = error
            time.sleep(pause * (2 ** attempt))
    raise RuntimeError(f"could not start the export for {name} after {attempts} attempts") from last


def export_all_windows(configs, start: str, end: str, bucket: str, prefix: str, step: int = STEP_DAYS,
                       bands=BANDS, log=print):
    """Export every window for every config, skipping windows already on GCS. Returns manifest rows."""
    import json

    import geopandas as gpd
    from google.cloud import storage

    from . import auth
    from . import config as config_mod
    from .optical_export import aoi_geojson_2d, grid_bounds_4326

    rows = []
    for path in configs:
        cfg = config_mod.load_config(path)
        aoi = cfg["aoi"]["key"]
        grid_def = json.loads((config_mod.grid_dir(cfg) / "grid_def.json").read_text())
        w, s, e, n = grid_bounds_4326(grid_def)
        rect = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
        client = storage.Client(credentials=auth.credentials(cfg), project=cfg["auth"]["project"])
        existing = {b.name for b in client.list_blobs(bucket, prefix=f"{prefix.strip('/')}/{aoi}/")}
        todo = [(a, b) for a, b in windows(start, end, step)
                if object_name(prefix, aoi, a) + ".tif" not in existing]
        log(f"{aoi}: {len(windows(start, end, step))} windows, {len(todo)} to export")
        for first, last in todo:
            name = object_name(prefix, aoi, first)
            row = {"aoi": aoi, "window_start": first, "window_end": last,
                   "gcs_uri": f"gs://{bucket}/{name}.tif", "task_id": None, "error": None}
            try:
                row["task_id"] = export_window(rect, grid_def, bucket, name, first, last, bands).id
            except Exception as error:  # noqa: BLE001 - one window must not abandon the rest
                row["error"] = f"{type(error).__name__}: {error}"
                log(f"  {name}: {row['error']}")
            rows.append(row)
        failed = sum(1 for r in rows if r["aoi"] == aoi and r["error"])
        if failed:
            log(f"{aoi}: {failed} of {len(todo)} windows could not be submitted")
    return rows
