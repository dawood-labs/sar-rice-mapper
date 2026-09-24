"""Read-only Earth Engine checks from sources other than the two the map is built on.

Why
---
When the radar sees no water on a rice-like crop, three outside questions help decide whether the
crop is rice: did it rain enough for rainfed paddies to hold water this year (``rain_by_month``),
is there an L-band radar (ALOS-2 PALSAR-2 ScanSAR), which sees water under a crop canopy that
C-band cannot, and how many Landsat 8/9 dates could fill the Sentinel-2 cloud gaps
(``availability``). Everything here is a small server-side reduction returned with ``getInfo``:
no export, nothing written to cloud storage.
"""
from __future__ import annotations

import pandas as pd

IMERG = "NASA/GPM_L3/IMERG_V07"
ALOS2_SCANSAR = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
LANDSAT = ("LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC09/C02/T1_L2")


def rain_by_month(polygons: dict, start: str = "2026-04-01", end: str = "2026-09-24", scale: int = 11000) -> pd.DataFrame:
    """Monthly rainfall (mm) over each AOI polygon from GPM IMERG (half-hourly mm/h, summed x 0.5 h)."""
    import ee

    fc = ee.FeatureCollection([ee.Feature(ee.Geometry(g), {"aoi": k}) for k, g in polygons.items()])
    months = pd.date_range(start, end, freq="MS")
    rows = []
    for m in months:
        m_end = min(m + pd.offsets.MonthBegin(1), pd.Timestamp(end))
        img = (ee.ImageCollection(IMERG).filterDate(str(m.date()), str(m_end.date()))
               .select("precipitation").sum().multiply(0.5).rename("mm"))
        red = img.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale).getInfo()
        for f in red["features"]:
            rows.append({"aoi": f["properties"]["aoi"], "month": m.strftime("%Y-%m"),
                         "rain_mm": round(float(f["properties"].get("mean") or 0), 1)})
    return pd.DataFrame(rows).pivot(index="aoi", columns="month", values="rain_mm").reset_index()


def availability(polygon, collection: str, start: str, end: str) -> pd.DataFrame:
    """Dates (and count) of a collection's images touching the polygon."""
    import ee

    col = ee.ImageCollection(collection).filterBounds(ee.Geometry(polygon)).filterDate(start, end)
    times = col.aggregate_array("system:time_start").getInfo()
    return pd.DataFrame({"date": pd.to_datetime(sorted(times), unit="ms").date, "collection": collection})


def pixels_to_lonlat(grid: dict, pixels):
    """Flat grid pixel ids -> (lon, lat) of the pixel centres."""
    import numpy as np
    import pyproj

    r, c = np.divmod(np.asarray(pixels).astype(int), int(grid["width"]))
    x = grid["x0"] + (c + 0.5) * grid["res"]
    y = grid["y0"] - (r + 0.5) * grid["res"]
    return pyproj.Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True).transform(x, y)


def alos2_sample(lon, lat, start: str = "2026-04-01", end: str = "2026-09-24", scale: int = 25) -> pd.DataFrame:
    """ALOS-2 PALSAR-2 ScanSAR HH and HV (dB) at the given points on every date, read-only.

    L-band (24 cm) passes through a young rice canopy: a flooded paddy is very dark before the crop
    closes and very bright in HH once stems stand in water (the stem-water double bounce), which a
    dry-land crop does not do. dB = 10 log10(DN^2) - 83 (JAXA calibration factor).
    """
    import ee
    import numpy as np

    geom = ee.Geometry.MultiPoint([[float(a), float(b)] for a, b in zip(lon, lat)])
    col = (ee.ImageCollection(ALOS2_SCANSAR).filterBounds(geom).filterDate(start, end).select(["HH", "HV"]))
    rows = col.getRegion(geom, scale).getInfo()
    head, body = rows[0], rows[1:]
    t = pd.DataFrame(body, columns=head).dropna(subset=["HH", "HV"])
    for b in ("HH", "HV"):
        t[b] = 10 * np.log10(t[b].astype(float) ** 2) - 83.0
    t["date"] = pd.to_datetime(t["time"], unit="ms").dt.date
    return t[["longitude", "latitude", "date", "HH", "HV"]]
