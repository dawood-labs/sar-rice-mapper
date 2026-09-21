"""Windowed multi-temporal speckle filter (Quegan & Yu 2001).

Idea in plain words: speckle is random from date to date, but the *texture* of the field changes
slowly. For date k we take its spatially filtered value F(I_k) and correct it with the average
"speckle ratio" I_i / F(I_i) of nearby dates of the same track:

    J_k = F(I_k) · mean_i( I_i / F(I_i) ),   i in temporal_neighbors(n, k, half_window)

This removes more speckle than a spatial filter alone while keeping spatial detail (small fields,
field edges), because information comes from time rather than from a bigger window.

Differences from gee_s1_ard's MultiTemporal_Filter (MIT, (c) 2021 Adugna Mullissa), deliberate:
- Neighbours are a fixed, symmetric window of ±half_window acquisitions from the SAME track list
  passed in (clipped at the ends of the series), instead of "N images before the date" searched
  in the whole archive. When new dates are appended later, only the last `half_window` outputs change.
- The mean is taken per pixel over the neighbours that are valid at that pixel.
- Where only one date is valid (count == 1) the Quegan formula collapses to the unfiltered pixel
  (F·I/F = I); we return the mono-temporal filtered value F(I_k) there instead.
"""
from __future__ import annotations

import ee


def temporal_neighbors(n: int, k: int, half_window: int) -> list[int]:
    """Indices of the acquisitions used to filter acquisition k: [k-h, k+h] clipped to [0, n-1]."""
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(k, int) or not 0 <= k < n:
        raise ValueError(f"k must be in [0, {n - 1}], got {k!r}")
    if not isinstance(half_window, int) or isinstance(half_window, bool) or half_window < 0:
        raise ValueError(f"half_window must be a non-negative integer, got {half_window!r}")
    return list(range(max(0, k - half_window), min(n - 1, k + half_window) + 1))


def quegan_series(images: list[ee.Image], filtered: list[ee.Image], bands: list[str], half_window: int) -> list[ee.Image]:
    """Multi-temporal filter over a time-ordered series.

    `images[i]` is the linear image of acquisition i and `filtered[i]` its mono-temporal filtered
    version (same bands). Returns one filtered linear image per acquisition.
    """
    if len(images) != len(filtered):
        raise ValueError("images and filtered must have the same length")
    n = len(images)
    ratios = [images[i].select(bands).divide(filtered[i].select(bands)).rename(bands) for i in range(n)]
    out = []
    for k in range(n):
        window = ee.ImageCollection([ratios[i] for i in temporal_neighbors(n, k, half_window)])
        mean_ratio = window.reduce(ee.Reducer.mean()).rename(bands)
        count = window.reduce(ee.Reducer.count()).rename(bands)
        fk = filtered[k].select(bands)
        jk = fk.multiply(mean_ratio).where(count.lte(1), fk)
        out.append(jk.updateMask(images[k].select(bands).mask()).rename(bands))
    return out
