"""Find the recurring seasonal patterns across many AOIs, without deciding in advance what they are.

Why unsupervised first
----------------------
With no ground truth, asking "where is the rice?" presupposes the answer's shape. Looking at a few
curves already showed more than one crop calendar (a single monsoon crop, a dry-season crop, fields
that carry both) plus surfaces with no cycle at all. Before any rule or label is written, it is
worth letting the data say **how many distinct seasonal behaviours there are and what they look
like**, and only then asking a person to name them.

The method, in order:

1. From each AOI, sample pixels at random (fixed seed, capped per AOI so a large AOI cannot dominate).
2. Take each pixel's 5x5 spatial mean (in linear power), and average its series into calendar
   half-months — the same bins on every AOI, which is what makes pixels from different AOIs and
   different acquisition days comparable.
3. Cluster the binned series with k-means.
4. For each cluster report the median curve and its spread, the share of every AOI it covers, and
   **which tracks its pixels came from** — so a cluster that exists only because one viewing
   geometry reads 1-2 dB brighter than another is visible rather than mistaken for a crop type.

The clusters are descriptions, not classes. Naming them is Dawood's decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import seasonal_stats as ss


def half_month_edges(start, end):
    """Calendar half-month labels (``YYYYMMH`` as in :func:`seasonal_stats.half_month_bins`) from start to end."""
    import datetime as dt

    start, end = (dt.date.fromisoformat(str(x)) for x in (start, end))
    labels, d = [], start
    while d < end:
        labels.append(d.year * 1000 + d.month * 10 + (0 if d.day <= 15 else 1))
        d = (d.replace(day=16) if d.day <= 15
             else (d.replace(year=d.year + 1, month=1, day=1) if d.month == 12
                   else d.replace(month=d.month + 1, day=1)))
    return np.array(labels)


def sample_binned_series(vrt_path, aoi_path, start, end, n_samples=2000, window=5, seed=0,
                         min_valid_bins=None):
    """Sample pixels inside the AOI and return their half-month VH series on a fixed bin grid.

    Returns ``(bins, series, rows, cols)`` where ``series`` is ``(n, n_bins)`` in dB; bins with no
    acquisition for a pixel are NaN and left for the caller to decide about. Pixels are kept only
    when at least ``min_valid_bins`` bins are finite (default: all but two).
    """
    dates, cube = ss.read_stack(vrt_path)
    keep = ss.window_indices(dates, start, end)
    dates = [dates[i] for i in keep]
    smoothed = ss.spatial_mean(cube[keep], size=window)
    labels, binned = ss.bin_in_linear(smoothed, dates)

    grid = half_month_edges(start, end)
    full = np.full((grid.size,) + binned.shape[1:], np.nan, dtype="float32")
    for i, label in enumerate(grid):
        hit = np.where(labels == label)[0]
        if hit.size:
            full[i] = binned[hit[0]]

    mask = ss.aoi_mask(aoi_path, vrt_path)
    need = grid.size - 2 if min_valid_bins is None else min_valid_bins
    ok = mask & (np.isfinite(full).sum(axis=0) >= need)
    rows, cols = np.nonzero(ok)
    if rows.size == 0:
        return grid, np.empty((0, grid.size), dtype="float32"), rows, cols
    rng = np.random.default_rng(seed)
    pick = rng.choice(rows.size, size=min(n_samples, rows.size), replace=False)
    rows, cols = rows[pick], cols[pick]
    return grid, full[:, rows, cols].T.copy(), rows, cols


def fill_gaps(series):
    """Linearly interpolate NaN bins along each row (ends held constant). Clustering needs full rows."""
    out = np.array(series, dtype="float32", copy=True)
    x = np.arange(out.shape[1])
    for row in out:
        good = np.isfinite(row)
        if good.all() or not good.any():
            continue
        row[~good] = np.interp(x[~good], x[good], row[good])
    return out


def cluster(series, k, seed=0):
    """k-means on the binned series. Returns ``(labels, centres)``, clusters renumbered by size."""
    from sklearn.cluster import KMeans

    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(series)
    order = np.argsort(-np.bincount(model.labels_, minlength=k))
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[model.labels_], model.cluster_centers_[order]


def choose_k(series, ks, seed=0, sample=20000):
    """Inertia and silhouette for several k, to see where adding clusters stops helping.

    The silhouette is computed on a subsample because it is quadratic in the number of points.
    Neither number picks k by itself; they show where the curve bends, and a person looks at the
    cluster curves to decide whether two clusters are really one behaviour.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    sub = series[rng.choice(len(series), size=min(sample, len(series)), replace=False)]
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(series)
        rows.append({"k": k, "inertia": float(model.inertia_),
                     "silhouette": float(silhouette_score(sub, model.predict(sub)))})
    return pd.DataFrame(rows)


def cluster_table(labels, groups, name="aoi"):
    """Share of each group's samples in each cluster, in percent (rows: groups, columns: clusters)."""
    table = pd.crosstab(pd.Series(groups, name=name), pd.Series(labels, name="cluster"),
                        normalize="index") * 100
    return table.round(1)


def primary_track(cfg: dict) -> str:
    """The config's primary track id (the one features are computed from)."""
    for track in cfg["s1"]["tracks"]:
        if track.get("role") == "primary":
            return track["track_id"]
    raise ValueError(f"{cfg['aoi']['key']}: no primary track in s1.tracks")


def sample_many(config_paths, start, end, n_samples=2000, seed=0, log=print):
    """Run :func:`sample_binned_series` on every config's newest run, primary track only.

    Returns ``(bins, table)``: ``table`` has one row per sampled pixel with its AOI, track, grid
    row/col and the binned series in columns ``b00``...``bNN``. Each AOI gets the same cap, so a
    large AOI cannot outvote many small ones when clusters are formed; area shares are recovered
    afterwards from per-AOI proportions.
    """
    from .. import config as config_mod

    frames, bins = [], None
    for i, path in enumerate(config_paths):
        cfg = config_mod.load_config(path)
        track = primary_track(cfg)
        run = config_mod.run_dir(cfg)
        vrt = run / "stack" / f"track_{track}" / "stack_VH.vrt"
        bins, series, rows, cols = sample_binned_series(
            vrt, config_mod.aoi_path(cfg), start, end, n_samples=n_samples, seed=seed)
        frame = pd.DataFrame(series, columns=[f"b{j:02d}" for j in range(series.shape[1])])
        frame.insert(0, "col", cols)
        frame.insert(0, "row", rows)
        frame.insert(0, "track", track)
        frame.insert(0, "aoi", cfg["aoi"]["key"])
        frames.append(frame)
        log(f"[{i + 1}/{len(config_paths)}] {cfg['aoi']['key']}: {len(frame)} px, {track}")
    return bins, pd.concat(frames, ignore_index=True)
