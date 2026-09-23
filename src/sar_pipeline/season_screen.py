"""Which AOIs carry a crop in a given season? One read-only Earth Engine pass over every AOI.

Why
---
Exporting and processing an AOI takes a while, and the first AOI tried for the current monsoon turned
out to be under water all season with no crop at all. Before choosing where to work, every AOI is
screened in one request, nothing exported:

* **bare before** — the median NDVI over the pre-season window (default April-May, after the dry-
  season harvest and before the monsoon planting);
* **canopy in season** — the maximum NDVI over the season window (default August to today);
* a pixel counts as **seasonal canopy** when the season maximum is above ``canopy_ndvi`` *and* at
  least ``rise`` above the pre-season median. Trees, orchards and settlements' gardens are green in
  both windows, so the rise removes them; they are reported separately as ``evergreen``.

Only pixels that Cloud Score+ calls clear (``cs_cdf >= cs_min``) enter either window, and every AOI
also reports how much of it was **seen** in both windows at all — so "no crop" and "not observed
under the monsoon cloud" are never confused.

This is a screen for choosing AOIs, not a classification: a seasonal canopy can be any crop.
Areas are in acres.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .optical_export import CLOUD_SCORE_COLLECTION, S2_COLLECTION

SQM_PER_ACRE = 4046.8564224
CANOPY_NDVI = 0.5
RISE = 0.3
CS_MIN = 0.6


def aoi_polygons(folder="data/aoi") -> dict:
    """``{"aoi143": geojson, ...}`` from the per-AOI files ``aoi_NNN.gpkg``, as 2D lon/lat."""
    import geopandas as gpd

    from .optical_export import aoi_geojson_2d

    out = {}
    for path in sorted(Path(folder).glob("aoi_[0-9]*.gpkg")):  # skips aoi_all.gpkg
        out[f"aoi{int(path.stem.split('_')[1])}"] = aoi_geojson_2d(gpd.read_file(path))
    return out


def screen(polygons: dict, bare=("2026-04-01", "2026-06-01"), season=("2026-08-01", "2026-09-24"),
           canopy_ndvi: float = CANOPY_NDVI, rise: float = RISE, cs_min: float = CS_MIN,
           scale: int = 20) -> pd.DataFrame:
    """Seasonal-canopy, evergreen and seen areas per AOI, in acres; see the module docstring."""
    import ee

    region = ee.FeatureCollection([ee.Feature(ee.Geometry(g), {"aoi": k}) for k, g in polygons.items()])
    s2 = (ee.ImageCollection(S2_COLLECTION).filterBounds(region.geometry())
          .linkCollection(ee.ImageCollection(CLOUD_SCORE_COLLECTION), ["cs_cdf"]))

    def ndvi(img):
        return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(img.select("cs_cdf").gte(cs_min))

    before = s2.filterDate(*bare).map(ndvi).median()
    during = s2.filterDate(*season).map(ndvi).max()
    seen = before.mask().And(during.mask())
    green = during.gt(canopy_ndvi)
    seasonal = green.And(during.subtract(before).gt(rise))
    evergreen = green.And(before.gt(canopy_ndvi))
    area = ee.Image.pixelArea()
    stack = ee.Image.cat([
        area.rename("total"),
        area.updateMask(seen).rename("seen"),
        area.updateMask(seen.And(seasonal)).rename("seasonal"),
        area.updateMask(seen.And(evergreen)).rename("evergreen"),
    ])
    sums = stack.reduceRegions(collection=region, reducer=ee.Reducer.sum(), scale=scale, tileScale=4)
    rows = [f["properties"] for f in sums.getInfo()["features"]]
    return summarise(pd.DataFrame(rows))


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert m² sums to acres and add the shares used to rank AOIs."""
    out = pd.DataFrame({"aoi": frame["aoi"]})
    for col in ("total", "seen", "seasonal", "evergreen"):
        out[f"{col}_acres"] = (frame[col].fillna(0) / SQM_PER_ACRE).round(1)
    seen = out["seen_acres"].where(out["seen_acres"] > 0)
    out["seen_pct"] = (100 * out["seen_acres"] / out["total_acres"]).round(1)
    out["seasonal_pct_of_seen"] = (100 * out["seasonal_acres"] / seen).round(1)
    out["evergreen_pct_of_seen"] = (100 * out["evergreen_acres"] / seen).round(1)
    return out.sort_values("seasonal_acres", ascending=False).reset_index(drop=True)
