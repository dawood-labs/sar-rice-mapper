"""Stage 5 — Time series of a single pixel, without loading any full raster.

How it works
------------
Every pixel of the master grid has a pixel id: ``pid = row * width + col``. From a pid we know the
row and column, so we ask GDAL for a 1 x 1 window of ``stack_<POL>.vrt``. One read returns that
pixel's value for every date (one band per date). GDAL opens only the chunk file that contains the
pixel, so this is fast even for a country-sized stack.

Values are decoded with ``export_meta.json`` (float32 dB, or int16 = dB * scale) and nodata becomes
NaN. ``dates.csv`` tells which date each band is. Polarisations come from the run's band layout.

Repeated queries (e.g. clicking around a map in a notebook) reuse, inside this Python process, the
parsed grid definition, the validated export metadata, dates.csv, the band layout and the opened
VRT datasets. Every cache entry is keyed by the file's size and modification time, so a rebuilt
stack is picked up automatically. ``clear_cache()`` closes everything explicitly.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .config import grid_dir
from .errors import PipelineError
from .index import lonlat_to_pid, pid_to_rowcol
from .run_meta import read_band_layout, read_export_meta
from .stack import stack_dir

RAIN_MARK_MM_24H = 5.0
COLUMNS = ["date_utc", "datetime_utc", "acquisition_id", "platforms", "VV_db", "VH_db", "VH_minus_VV_db",
           "rain_6h_mm", "rain_24h_mm", "valid"]
_BASE_COLUMNS = ["date_utc", "datetime_utc", "acquisition_id", "platforms"]
_RAIN_COLUMNS = ["rain_6h_mm", "rain_24h_mm"]
_MAX_READERS = 32
_READERS: "OrderedDict[str, _Reader]" = OrderedDict()
_READERS_LOCK = threading.Lock()


# ---------------------------------------------------------------- in-process caches
def _stat_key(path: Path) -> tuple:
    try:
        st = Path(path).stat()
        return (str(path), int(st.st_size), int(st.st_mtime_ns))
    except FileNotFoundError:
        return (str(path), -1, -1)


@lru_cache(maxsize=16)
def _grid_cached(stat_key: tuple) -> dict:
    return json.loads(Path(stat_key[0]).read_text())


@lru_cache(maxsize=16)
def _meta_cached(cfg_json: str, run_dir: str, meta_key: tuple, grid_key: tuple) -> dict:
    # meta_key / grid_key are part of the cache key only: a changed file means a new, re-validated entry
    return read_export_meta(json.loads(cfg_json), run_dir)


@lru_cache(maxsize=32)
def _dates_cached(stat_key: tuple) -> pd.DataFrame:
    return pd.read_csv(stat_key[0], dtype={"acquisition_id": str, "date_utc": str}).sort_values("stack_band")


@lru_cache(maxsize=32)
def _pols_cached(run_dir: str, layout_key: tuple, track_id: str) -> tuple[str, ...]:
    bl = read_band_layout(run_dir, track_id)
    return tuple(dict.fromkeys(bl["pol"]))


def _cfg_json(cfg: dict) -> str:
    """The parts of the config that read_export_meta needs, as a hashable string."""
    small = {
        "_root": cfg.get("_root"),
        "aoi": {"key": (cfg.get("aoi") or {}).get("key")},
        "season": {"key": (cfg.get("season") or {}).get("key")},
        "export": cfg.get("export") or {},
        "auth": {"project": (cfg.get("auth") or {}).get("project")},
    }
    return json.dumps(small, sort_keys=True, default=str)


class _Reader:
    """One cached VRT dataset shared by threads.

    GDAL datasets are not safe for concurrent reads from several threads, so each dataset has its own
    lock. `refs` counts threads currently using it: an entry evicted from the cache (LRU, or because the
    file changed) is only closed once no thread is still reading from it.
    """

    __slots__ = ("key", "ds", "lock", "refs", "evicted")

    def __init__(self, key: tuple, ds):
        self.key, self.ds, self.lock, self.refs, self.evicted = key, ds, threading.Lock(), 0, False


def _retire(entry: _Reader) -> None:
    """Close now if unused, else mark it so the last reader closes it. Call with _READERS_LOCK held."""
    entry.evicted = True
    if entry.refs == 0:
        entry.ds.close()


@contextmanager
def _open_reader(vrt: Path):
    key = _stat_key(vrt)
    name = str(vrt)
    with _READERS_LOCK:
        entry = _READERS.get(name)
        if entry is not None and entry.key != key:
            del _READERS[name]
            _retire(entry)
            entry = None
        if entry is None:
            entry = _Reader(key, rasterio.open(vrt))
            _READERS[name] = entry
        else:
            _READERS.move_to_end(name)
        entry.refs += 1
        while len(_READERS) > _MAX_READERS:
            _, old = _READERS.popitem(last=False)
            if old is not entry:
                _retire(old)
            else:  # never evict the entry we are about to use
                _READERS[name] = entry
                break
    try:
        with entry.lock:
            yield entry.ds
    finally:
        with _READERS_LOCK:
            entry.refs -= 1
            if entry.evicted and entry.refs == 0:
                entry.ds.close()


def clear_cache() -> None:
    """Close cached VRT datasets and forget cached metadata (e.g. before deleting a run folder)."""
    with _READERS_LOCK:
        for entry in _READERS.values():
            _retire(entry)
        _READERS.clear()
    for fn in (_grid_cached, _meta_cached, _dates_cached, _pols_cached):
        fn.cache_clear()


# ---------------------------------------------------------------- queries
def _read_pixel(vrt: Path, row: int, col: int, n_expected: int, meta: dict) -> np.ndarray:
    with _open_reader(vrt) as src:
        if src.count != n_expected:
            raise PipelineError(f"{vrt} has {src.count} bands but dates.csv lists {n_expected} dates.")
        raw = src.read(window=Window(col, row, 1, 1))[:, 0, 0]
    vals = raw.astype("float64")
    vals[raw == meta["nodata"]] = np.nan
    if meta["dtype"] == "int16":
        vals = vals / meta["int16_scale"]
    return vals


def pixel_timeseries(cfg: dict, run_dir: str | Path, track_id: str, pid: int) -> pd.DataFrame:
    run_dir = Path(run_dir)
    gpath = grid_dir(cfg) / "grid_def.json"
    if not gpath.exists():
        raise PipelineError(f"{gpath} not found - build the grid first.")
    gd = _grid_cached(_stat_key(gpath))
    row, col = pid_to_rowcol(gd, pid)
    sdir = stack_dir(run_dir, track_id)
    dates_path = sdir / "dates.csv"
    if not dates_path.exists():
        raise PipelineError(f"{dates_path} not found - build the stack first.")
    dates = _dates_cached(_stat_key(dates_path))
    meta = _meta_cached(_cfg_json(cfg), str(run_dir), _stat_key(run_dir / "export_meta.json"), _stat_key(gpath))
    pols = list(_pols_cached(str(run_dir), _stat_key(run_dir / "band_layout.csv"), track_id))

    df = pd.DataFrame({c: (dates[c].to_numpy() if c in dates.columns else np.nan)
                       for c in _BASE_COLUMNS + _RAIN_COLUMNS})
    pol_cols = []
    for pol in pols:
        vrt = sdir / f"stack_{pol}.vrt"
        df[f"{pol}_db"] = _read_pixel(vrt, row, col, len(dates), meta) if vrt.exists() else np.nan
        pol_cols.append(f"{pol}_db")
    extra = []
    if "VV" in pols and "VH" in pols:
        df["VH_minus_VV_db"] = df["VH_db"] - df["VV_db"]
        extra = ["VH_minus_VV_db"]
    df["valid"] = df[pol_cols].notna().all(axis=1) if pol_cols else False
    df = df[_BASE_COLUMNS + pol_cols + extra + _RAIN_COLUMNS + ["valid"]].reset_index(drop=True)
    df.attrs.update(pid=int(pid), row=int(row), col=int(col), track_id=track_id)
    return df


def pixel_timeseries_lonlat(cfg: dict, run_dir: str | Path, track_id: str, lon: float, lat: float) -> pd.DataFrame:
    gpath = grid_dir(cfg) / "grid_def.json"
    if not gpath.exists():
        raise PipelineError(f"{gpath} not found - build the grid first.")
    gd = _grid_cached(_stat_key(gpath))
    df = pixel_timeseries(cfg, run_dir, track_id, lonlat_to_pid(gd, lon, lat))
    df.attrs.update(lon=float(lon), lat=float(lat))
    return df


def plot_timeseries(df: pd.DataFrame, cfg: dict | None = None, ax=None, title: str | None = None):
    """Backscatter per polarisation (left axis) and VH−VV (right axis, when both exist) against date.

    Also drawn: constellation events (dashed lines), gaps longer than ``audit.max_gap_days``
    (grey shading) and acquisitions with ≥ 5 mm rain in the previous 24 h (blue triangles), because
    rain wets soil and leaves and raises backscatter without any crop change.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    audit = (cfg or {}).get("audit", {}) or {}
    max_gap = float(audit.get("max_gap_days", 12))
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))
    dates = pd.to_datetime(df["date_utc"].astype(str).str[:10])

    styles = {"VV_db": ("o-", "#1f77b4"), "VH_db": ("s-", "#d62728")}
    pol_cols = [c for c in df.columns if c.endswith("_db") and c != "VH_minus_VV_db"]
    for i, c in enumerate(pol_cols):
        marker, color = styles.get(c, ("d-", f"C{i + 2}"))
        ax.plot(dates, df[c], marker, color=color, label=f"{c[:-3]} (dB)", ms=4)
    ax.set_ylabel("Backscatter (dB)")
    ax2 = None
    if "VH_minus_VV_db" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(dates, df["VH_minus_VV_db"], "^--", color="#2ca02c", label="VH−VV (dB)", ms=3, alpha=0.8)
        ax2.set_ylabel("VH − VV (dB)")

    for prev, nxt in zip(dates[:-1], dates[1:]):
        if (nxt - prev).days > max_gap:
            ax.axvspan(prev, nxt, color="0.85", alpha=0.6, zorder=0)
    for ev in audit.get("constellation_events", []) or []:
        d = pd.to_datetime(ev["date"])
        ax.axvline(d, color="0.3", ls=":", lw=1)
        ax.annotate(ev.get("note", ""), (d, 1.0), xycoords=("data", "axes fraction"), rotation=90,
                    va="top", ha="right", fontsize=7, color="0.3")
    rain = pd.to_numeric(df["rain_24h_mm"], errors="coerce")
    wet = rain >= RAIN_MARK_MM_24H
    if wet.any():
        vals = df[pol_cols].to_numpy(dtype="float64") if pol_cols else np.array([[0.0]])
        ymin = np.nanmin(vals) if np.isfinite(vals).any() else 0
        ax.scatter(dates[wet], [ymin - 1] * int(wet.sum()), marker="v", color="#17becf", label="rain ≥ 5 mm / 24 h", zorder=5)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels() if ax2 is not None else ([], [])
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.figure.autofmt_xdate()
    if title is None and df.attrs:
        title = f"track {df.attrs.get('track_id', '')}  pid {df.attrs.get('pid', '')}"
    if title:
        ax.set_title(title)
    return ax
