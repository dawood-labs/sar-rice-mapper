"""Sentinel-2 context for a SAR result: where and when optical imagery can corroborate it.

Why optical matters even on a SAR project
-----------------------------------------
Without ground truth, a SAR-derived class is a hypothesis: a pixel with a flooding dip and a canopy
rise *behaves like* a flooded crop. Optical imagery is the cheapest **independent** evidence
available — a green canopy in the visible/near-infrared is a different physical measurement, not a
different threshold on the same one. Anything that separates classes on both sensors is much harder
to dismiss as an artefact.

Why it is only *partial* evidence here
--------------------------------------
The project uses SAR precisely because the growing season is the cloudy season. So the corroboration
is available for some phases and not others, and :func:`cloud_availability` is the first thing to
run: it says, month by month, whether a usable scene exists at all. Concluding "the optical check
failed" when the truth is "there was no cloud-free scene that month" would be a serious mistake, so
availability is reported separately from any comparison.

Nothing here exports or uploads. Cloud statistics come from scene metadata; composites are reduced
server-side and only the sampled numbers come back.
"""
from __future__ import annotations

import collections
import datetime as dt

import numpy as np

#: Surface-reflectance collection, and its scene-level cloud property.
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_PROPERTY = "CLOUDY_PIXEL_PERCENTAGE"

#: Cloud Score+ is the current recommended masking band set for S2_SR_HARMONIZED.
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"


def cloud_availability(bounds, start: str, end: str, thresholds=(20, 40)) -> list[dict]:
    """Per-month Sentinel-2 scene counts and cloud cover over ``bounds`` (min_lon, min_lat, max_lon, max_lat).

    Returns one row per calendar month with the scene count, the **minimum** scene cloud percentage
    (the best case available that month) and how many scenes fall under each threshold. Run this
    before any optical comparison: a month with zero usable scenes is a gap in the evidence, not a
    negative result.
    """
    import ee

    region = ee.Geometry.Rectangle([float(v) for v in bounds])
    collection = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end))
    pairs = collection.reduceColumns(
        ee.Reducer.toList(2), ["system:time_start", CLOUD_PROPERTY]).getInfo()["list"]

    months: dict[str, list[float]] = collections.defaultdict(list)
    for epoch_ms, cloud in pairs:
        stamp = dt.datetime.fromtimestamp(epoch_ms / 1000, tz=dt.timezone.utc)
        months[stamp.strftime("%Y-%m")].append(float(cloud))

    rows = []
    for month in sorted(months):
        values = months[month]
        row = {"month": month, "scenes": len(values), "min_cloud_pct": round(min(values), 1)}
        for threshold in thresholds:
            row[f"under_{threshold}pct"] = sum(1 for v in values if v < threshold)
        rows.append(row)
    return rows


def usable_months(rows, min_scenes: int = 1, max_cloud: int = 20) -> list[str]:
    """Months from :func:`cloud_availability` with at least ``min_scenes`` scenes under ``max_cloud``."""
    key = f"under_{max_cloud}pct"
    return [r["month"] for r in rows if r.get(key, 0) >= min_scenes]


def _ndvi_composite(region, start: str, end: str, max_cloud_score: float = 0.6):
    """Median NDVI over a date range, masked with Cloud Score+.

    A median over a month is used rather than a single scene: it survives partial cloud, and a
    single date would tie the result to whatever the weather did that day.
    """
    import ee

    s2 = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end)
          .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), [CLOUD_SCORE_BAND]))

    def clean(image):
        clear = image.select(CLOUD_SCORE_BAND).gte(max_cloud_score)
        return image.updateMask(clear).normalizedDifference(["B8", "B4"]).rename("ndvi")

    return s2.map(clean).median()


def monthly_ndvi(points, months, max_cloud_score: float = 0.6, scale: int = 10) -> dict:
    """Sample median NDVI at ``points`` for each month in ``months`` (``"YYYY-MM"`` strings).

    ``points`` is a sequence of ``(lon, lat)``. Returns ``{month: array_of_ndvi}`` aligned with the
    input order, with NaN where the pixel was cloudy all month. Sampling happens server-side, one
    request per month.
    """
    import ee

    features = [ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"i": i})
                for i, (lon, lat) in enumerate(points)]
    collection = ee.FeatureCollection(features)
    region = collection.geometry().bounds()

    out = {}
    for month in months:
        year, mon = (int(x) for x in month.split("-"))
        start = f"{year:04d}-{mon:02d}-01"
        end = f"{year + (mon == 12):04d}-{(mon % 12) + 1:02d}-01"
        composite = _ndvi_composite(region, start, end, max_cloud_score)
        sampled = composite.sampleRegions(collection=collection, scale=scale,
                                          geometries=False, tileScale=4).getInfo()["features"]
        values = np.full(len(points), np.nan, dtype="float32")
        for feature in sampled:
            props = feature["properties"]
            if props.get("ndvi") is not None:
                values[int(props["i"])] = float(props["ndvi"])
        out[month] = values
    return out


def compare_groups(values_by_month: dict, labels, group_names=("group A", "group B")) -> list[dict]:
    """Median NDVI per month for two groups of points, with the gap between them.

    ``labels`` is a boolean array: True picks the first group. The **gap** is the number that
    matters — a class separation that appears on SAR and is reproduced in NDVI is evidence; a gap
    near zero says the optical sensor does not see the distinction the radar drew.
    """
    labels = np.asarray(labels, dtype=bool)
    rows = []
    for month in sorted(values_by_month):
        values = values_by_month[month]
        a = values[labels & np.isfinite(values)]
        b = values[(~labels) & np.isfinite(values)]
        rows.append({
            "month": month,
            f"{group_names[0]}_n": int(a.size),
            f"{group_names[0]}_median": round(float(np.median(a)), 3) if a.size else float("nan"),
            f"{group_names[1]}_n": int(b.size),
            f"{group_names[1]}_median": round(float(np.median(b)), 3) if b.size else float("nan"),
            "gap": (round(float(np.median(a) - np.median(b)), 3)
                    if a.size and b.size else float("nan")),
        })
    return rows


#: Spectral indices sampled by :func:`monthly_indices`.
#: NDVI  = (NIR - red) / (NIR + red): green canopy.
#: LSWI  = (NIR - SWIR1) / (NIR + SWIR1): land surface water; rises with leaf water *and* standing water.
#: MNDWI = (green - SWIR1) / (green + SWIR1): open water; positive over water, negative over vegetation.
#: The classic rice flooding rule (Xiao et al.) flags a pixel as flooded when LSWI + 0.05 >= NDVI or
#: EVI: at transplanting the field is mostly water with small seedlings, so the water signal
#: rivals or beats the vegetation signal. Rice is the crop that does this; a pulse crop sown on
#: residual moisture after rice does not.
INDEX_BANDS = {"ndvi": ("B8", "B4"), "lswi": ("B8", "B11"), "mndwi": ("B3", "B11")}

#: Google Dynamic World class-probability bands.
DYNAMIC_WORLD = "GOOGLE/DYNAMICWORLD/V1"
DW_CLASSES = ("water", "trees", "grass", "flooded_vegetation", "crops", "shrub_and_scrub",
              "built", "bare")


def _month_range(month: str):
    year, mon = (int(x) for x in month.split("-"))
    start = f"{year:04d}-{mon:02d}-01"
    end = f"{year + (mon == 12):04d}-{(mon % 12) + 1:02d}-01"
    return start, end


def _sample(image, points, scale, names):
    import ee

    features = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {"i": i})
                                     for i, (lon, lat) in enumerate(points)])
    got = image.sampleRegions(collection=features, scale=scale, geometries=False,
                              tileScale=4).getInfo()["features"]
    out = {n: np.full(len(points), np.nan, dtype="float32") for n in names}
    for f in got:
        p = f["properties"]
        for n in names:
            if p.get(n) is not None:
                out[n][int(p["i"])] = float(p[n])
    return out


def monthly_indices(points, months, max_cloud_score: float = 0.6, scale: int = 10) -> dict:
    """Median NDVI, LSWI and MNDWI per point per month (``{month: {index: array}}``).

    Cloud Score+ masks clouds pixel by pixel; a point cloudy all month comes back NaN. Use
    :func:`cloud_availability` first to know which months can carry evidence at all.
    """
    import ee

    region = ee.Geometry.MultiPoint([[float(a), float(b)] for a, b in points]).bounds()
    out = {}
    for month in months:
        start, end = _month_range(month)
        s2 = (ee.ImageCollection(S2_COLLECTION).filterBounds(region).filterDate(start, end)
              .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), [CLOUD_SCORE_BAND]))

        def indices(image):
            clear = image.select(CLOUD_SCORE_BAND).gte(max_cloud_score)
            bands = [image.normalizedDifference(list(b)).rename(n) for n, b in INDEX_BANDS.items()]
            return ee.Image.cat(bands).updateMask(clear)

        out[month] = _sample(s2.map(indices).median(), points, scale, list(INDEX_BANDS))
    return out


def monthly_dynamic_world(points, months, scale: int = 10) -> dict:
    """Mean Dynamic World class probabilities per point per month (``{month: {class: array}}``).

    Dynamic World is a Sentinel-2 land-cover model with no knowledge of our SAR results, which is
    what makes it useful as an independent opinion. It has a ``crops`` class and a
    ``flooded_vegetation`` class but no rice class; a flooded rice field during transplanting
    typically reads as water or flooded vegetation, and as crops once the canopy closes.
    """
    import ee

    region = ee.Geometry.MultiPoint([[float(a), float(b)] for a, b in points]).bounds()
    out = {}
    for month in months:
        start, end = _month_range(month)
        dw = (ee.ImageCollection(DYNAMIC_WORLD).filterBounds(region).filterDate(start, end)
              .select(list(DW_CLASSES)).mean())
        out[month] = _sample(dw, points, scale, list(DW_CLASSES))
    return out
