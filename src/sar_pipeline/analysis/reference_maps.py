"""Compare our pixels against published rice maps, as independent evidence.

Why
---
Without ground truth, one of the strongest checks available is to ask: **do maps that other teams
built, with other data, other methods and other years, call these same pixels rice?** Agreement
across independent products is hard to explain by a shared mistake. Disagreement is informative too,
because each product has known blind spots (for example, optical flood rules under-detect rainfed
and direct-seeded rice).

A published map is not ground truth either. It is another opinion, from a different year, with its
own error rate. Treat agreement as evidence, never as a label.

Implementation note
-------------------
Continental rasters are tens of thousands of pixels across. Sampling millions of points one at a
time is slow, so points are grouped (for example by AOI) and each group reads one small window
around its own bounding box.
"""
from __future__ import annotations

import numpy as np


def sample_raster(path, lons, lats, groups=None, pad_deg: float = 0.01):
    """Value of a single-band EPSG:4326 raster at each ``(lon, lat)``; NaN outside the raster.

    ``groups`` (same length as the points) lets points that are close together share one windowed
    read; without it every point is read in one window over all of them.
    """
    import rasterio
    from rasterio.windows import from_bounds

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out = np.full(lons.size, np.nan, dtype="float32")
    groups = np.zeros(lons.size, dtype=int) if groups is None else np.asarray(groups)

    with rasterio.open(path) as ds:
        for g in np.unique(groups):
            sel = np.where(groups == g)[0]
            window = from_bounds(lons[sel].min() - pad_deg, lats[sel].min() - pad_deg,
                                 lons[sel].max() + pad_deg, lats[sel].max() + pad_deg, ds.transform)
            window = window.round_offsets().round_lengths()
            block = ds.read(1, window=window, boundless=True, fill_value=0)
            transform = ds.window_transform(window)
            cols, rows = ~transform * (lons[sel], lats[sel])
            rows = np.floor(rows).astype(int)
            cols = np.floor(cols).astype(int)
            inside = (rows >= 0) & (rows < block.shape[0]) & (cols >= 0) & (cols < block.shape[1])
            out[sel[inside]] = block[rows[inside], cols[inside]]
    return out


def sample_tiles(tiles, lons, lats, groups=None, pad_deg: float = 0.01):
    """Like :func:`sample_raster` over several non-overlapping tiles: each point takes the first tile
    that contains it."""
    import rasterio

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    out = np.full(lons.size, np.nan, dtype="float32")
    for tile in tiles:
        with rasterio.open(tile) as ds:
            b = ds.bounds
        inside = ((lons >= b.left) & (lons < b.right) & (lats > b.bottom) & (lats <= b.top)
                  & np.isnan(out))
        if inside.any():
            idx = np.where(inside)[0]
            vals = sample_raster(tile, lons[idx], lats[idx],
                                 None if groups is None else np.asarray(groups)[idx], pad_deg)
            out[idx] = vals
    return out
