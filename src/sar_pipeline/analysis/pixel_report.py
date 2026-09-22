"""Everything about one pixel on one page: SAR curves, Sentinel-2 NDVI, and what the model said.

Why
---
When a map disagrees with what someone knows about a field, the fastest way to find out who is right
is to look at that one pixel's season from every sensor at once. Give an AOI number and a pixel id
(``pid`` from ``pixel_index.tif``) and :func:`investigate` returns the data; :func:`plot` draws it
with matplotlib or plotly.

What is shown, and why
----------------------
* **VH, VV and VH - VV** from the AOI's primary track, both for the single pixel and for its 5x5
  linear-power mean (the value the classifier uses). The single pixel carries ~2 dB of speckle;
  the 5x5 mean shows the real shape.
* **Rain**: acquisitions with more than ``rain_mm`` in the previous 24 hours are marked, because
  rain raises backscatter for a day or two with no crop change behind it.
* **NDVI and NDWI** from the exported Sentinel-2 images (one per date, on the same grid). A date is
  **dropped** when this pixel's clear score is below ``clear_min`` (default 60 = Cloud Score+ 0.6),
  so only clean observations are plotted; the number dropped is reported. NDWI (green - NIR) /
  (green + NIR) rises above 0 over open water, so dates with NDWI > 0 are marked as flooded. (The
  LSWI flooding test would need the SWIR band B11, which the reference images do not carry.)
* **The model's call** for the pixel: class, rice probability and monsoon-rice probability.

Sentinel-2 files are read from the project's GCS folder and cached locally under
``data/s2_reference/`` (gitignored) the first time an AOI is investigated.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as config_mod
from . import seasonal_stats as ss

PHASES = (("2025-06-01", "2025-07-31", "Jun-Jul"), ("2025-10-01", "2025-11-30", "Oct-Nov"),
          ("2026-02-01", "2026-03-15", "Feb"))
CLASS_NAMES = {0: "outside AOI", 1: "rice: monsoon only", 2: "rice: monsoon + dry season",
               3: "rice: dry season only", 4: "not rice", 255: "uncertain"}


def locate(aoi_id: int, pid: int, season_key: str = "year2025", config_dir="config") -> dict:
    """Config, run, grid and row/col/lon/lat for a pixel id in an AOI."""
    import pyproj

    cfg = config_mod.load_config(Path(config_dir) / f"aoi{aoi_id}_{season_key}.yaml")
    grid = json.loads((config_mod.grid_dir(cfg) / "grid_def.json").read_text())
    width, height = int(grid["width"]), int(grid["height"])
    if not 0 <= pid < width * height:
        raise ValueError(f"pid {pid} is outside AOI {aoi_id}'s grid (0..{width * height - 1})")
    row, col = divmod(int(pid), width)
    x = grid["x0"] + (col + 0.5) * grid["res"]
    y = grid["y0"] - (row + 0.5) * grid["res"]
    lon, lat = pyproj.Transformer.from_crs(grid["crs"], "EPSG:4326", always_xy=True).transform(x, y)
    tracks = {t["role"]: t["track_id"] for t in cfg["s1"]["tracks"]}
    return {"cfg": cfg, "aoi": cfg["aoi"]["key"], "pid": int(pid), "row": row, "col": col,
            "lon": lon, "lat": lat, "grid": grid, "run": config_mod.run_dir(cfg),
            "primary": tracks.get("primary"), "secondary": tracks.get("secondary")}


def _window(row, col, size, shape):
    import rasterio.windows as rw

    h = size // 2
    r0, c0 = max(0, row - h), max(0, col - h)
    r1, c1 = min(shape[0], row + h + 1), min(shape[1], col + h + 1)
    return rw.Window(c0, r0, c1 - c0, r1 - r0)


def sar_series(loc: dict, track: str | None = None, window: int = 5) -> pd.DataFrame:
    """Per-acquisition VH, VV, VH-VV (dB) for the pixel and its ``window`` x ``window`` mean, plus rain."""
    import rasterio

    track = track or loc["primary"]
    stack = Path(loc["run"]) / "stack" / f"track_{track}"
    out = {}
    for pol in ("VH", "VV"):
        with rasterio.open(stack / f"stack_{pol}.vrt") as ds:
            win = _window(loc["row"], loc["col"], window, (ds.height, ds.width))
            block = ds.read(window=win).astype("float32")
            nodata = ds.nodata
            dates = [dt.datetime.strptime(d.rsplit("_", 1)[1], "%Y%m%d").date() for d in ds.descriptions]
        if nodata is not None:
            block[block == nodata] = np.nan
        centre = block[:, loc["row"] - int(win.row_off), loc["col"] - int(win.col_off)]
        with np.errstate(invalid="ignore"):
            mean = ss.to_db(np.nanmean(ss.to_linear(block.reshape(block.shape[0], -1)), axis=1))
        out[f"{pol}_px"], out[f"{pol}_5x5"] = centre, mean
    frame = pd.DataFrame({"date": pd.to_datetime(dates), **out})
    frame["VHmVV_px"] = frame["VH_px"] - frame["VV_px"]
    frame["VHmVV_5x5"] = frame["VH_5x5"] - frame["VV_5x5"]
    info = pd.read_csv(stack / "dates.csv")
    frame["rain_24h_mm"] = pd.to_numeric(info["rain_24h_mm"], errors="coerce").to_numpy()[:len(frame)]
    frame["track"] = track
    return frame


def sync_s2(loc: dict, cache_root="data/s2_reference") -> Path:
    """Download this AOI's exported Sentinel-2 files once; later calls reuse the local copies."""
    from google.cloud import storage

    from .. import auth

    cfg = loc["cfg"]
    prefix = f"{cfg['gcs']['base_folder']}/s2_reference/{loc['aoi']}/"
    local = Path(cache_root) / loc["aoi"]
    local.mkdir(parents=True, exist_ok=True)
    client = storage.Client(credentials=auth.credentials(cfg), project=cfg["auth"]["project"])
    for blob in client.list_blobs(cfg["gcs"]["bucket"], prefix=prefix):
        target = local / blob.name.rsplit("/", 1)[-1]
        if blob.name.endswith(".tif") and not target.exists():
            blob.download_to_filename(str(target))
    return local


def s2_series(loc: dict, cache_root="data/s2_reference", clear_min: float = 60) -> pd.DataFrame:
    """NDVI, NDWI and clear score at the pixel for every exported Sentinel-2 date.

    ``kept`` is False where the pixel's clear score is below ``clear_min`` or it has no data;
    those rows are excluded from plots. ``flooded`` marks NDWI > 0 (open water at the pixel).
    """
    import rasterio

    rows = []
    for path in sorted(sync_s2(loc, cache_root).glob("*.tif")):
        date = pd.to_datetime(path.stem.rsplit("_S2_", 1)[1])
        with rasterio.open(path) as ds:
            v = ds.read(window=((loc["row"], loc["row"] + 1), (loc["col"], loc["col"] + 1)))[:, 0, 0]
        b2, b3, b4, b5, b8, clear = (float(x) for x in v)
        valid = (b8 + b4 > 0) and (b3 + b8 > 0)
        ndvi = (b8 - b4) / (b8 + b4) if valid else np.nan
        ndwi = (b3 - b8) / (b3 + b8) if valid else np.nan
        rows.append({"date": date, "ndvi": ndvi, "ndwi": ndwi, "clear": clear,
                     "flooded": bool(valid and ndwi > 0), "kept": bool(valid and clear >= clear_min)})
    frame = pd.DataFrame(rows).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return frame


def model_call(loc: dict, maps_dir="processed/_batch/model_v2/maps") -> dict:
    """The draft model's class and probabilities at the pixel."""
    import rasterio

    out = {}
    for name in ("class", "rice_prob", "monsoon_rice_prob"):
        path = Path(maps_dir) / f"{loc['aoi']}_{name}.tif"
        if path.exists():
            with rasterio.open(path) as ds:
                out[name] = int(ds.read(1, window=((loc["row"], loc["row"] + 1),
                                                   (loc["col"], loc["col"] + 1)))[0, 0])
    if "class" in out:
        out["class_name"] = CLASS_NAMES.get(out["class"], str(out["class"]))
    return out


def investigate(aoi_id: int, pid: int, clear_min: float = 60, rain_mm: float = 5) -> dict:
    """Gather everything for one pixel. Returns a dict of location, frames and a summary."""
    loc = locate(aoi_id, pid)
    sar = sar_series(loc)
    sar["rainy"] = sar["rain_24h_mm"] > rain_mm
    s2 = s2_series(loc, clear_min=clear_min)
    call = model_call(loc)
    summary = {
        "AOI": loc["aoi"], "pid": pid, "row / col": f"{loc['row']} / {loc['col']}",
        "lon / lat": f"{loc['lon']:.5f} / {loc['lat']:.5f}",
        "SAR track (primary)": loc["primary"], "SAR dates": len(sar),
        "rainy SAR dates (>%g mm/24 h)" % rain_mm: int(sar["rainy"].sum()),
        "S2 dates": len(s2), "S2 dates kept (clear)": int(s2["kept"].sum()),
        "S2 dates dropped (cloud/nodata)": int((~s2["kept"]).sum()),
        "S2 clear dates with open water (NDWI > 0)": int((s2["kept"] & s2["flooded"]).sum()),
        "model class": call.get("class_name", "n/a"),
        "rice probability %": call.get("rice_prob", "n/a"),
        "monsoon-rice probability %": call.get("monsoon_rice_prob", "n/a"),
        "VH 5x5 min (date)": f"{sar['VH_5x5'].min():.1f} dB ({sar.loc[sar['VH_5x5'].idxmin(), 'date']:%d %b %Y})",
        "VH 5x5 max (date)": f"{sar['VH_5x5'].max():.1f} dB ({sar.loc[sar['VH_5x5'].idxmax(), 'date']:%d %b %Y})",
        "VH 5x5 seasonal swing": f"{sar['VH_5x5'].max() - sar['VH_5x5'].min():.1f} dB",
    }
    kept = s2[s2["kept"]]
    if len(kept):
        top = kept.loc[kept["ndvi"].idxmax()]
        summary["NDVI max (date)"] = f"{top['ndvi']:.2f} ({top['date']:%d %b %Y})"
    return {"loc": loc, "sar": sar, "s2": s2, "model": call, "summary": summary}


def _phases():
    return [(pd.Timestamp(a), pd.Timestamp(b), label) for a, b, label in PHASES]


#: Default figure size. matplotlib uses inches, plotly uses pixels.
DEFAULT_SIZE = {"matplotlib": (18, 20), "plotly": (1600, 1650)}

#: Panels, top to bottom. Every y-axis scales to its own data; nothing is fixed.
PANELS = ("VH (dB)", "VV (dB)", "VH - VV (dB)", "NDVI, clear dates only",
          "NDWI, clear dates only (> 0 = open water)")


def plot(result: dict, backend: str = "matplotlib", out_path=None, width=None, height=None):
    """Five stacked panels on one date axis: VH, VV, VH - VV, NDVI, NDWI (clean dates only).

    Each panel's y-axis follows its own data range. ``width`` / ``height`` set the figure size:
    **inches** for matplotlib (default 18 x 20), **pixels** for plotly (default 1600 x 1650).
    """
    sar, s2 = result["sar"], result["s2"][result["s2"]["kept"]]
    title = (f"{result['loc']['aoi']}  pid {result['loc']['pid']}  |  model: "
             f"{result['model'].get('class_name', 'n/a')}  (rice {result['model'].get('rice_prob', 'n/a')}%)")
    w, h = DEFAULT_SIZE["plotly" if backend == "plotly" else "matplotlib"]
    w, h = width or w, height or h
    if backend == "plotly":
        return _plot_plotly(sar, s2, title, out_path, w, h)
    return _plot_matplotlib(sar, s2, title, out_path, w, h)


def _plot_matplotlib(sar, s2, title, out_path, width, height):
    import matplotlib.pyplot as plt

    from .curves import INK_SECONDARY, SERIES_COLORS, SURFACE

    fig, axes = plt.subplots(5, 1, figsize=(width, height), sharex=True, facecolor=SURFACE)
    blue, orange, green = SERIES_COLORS
    rainy = sar[sar["rainy"]]
    for ax, pol, colour in ((axes[0], "VH", blue), (axes[1], "VV", orange)):
        ax.plot(sar["date"], sar[f"{pol}_5x5"], color=colour, lw=2, label=f"{pol} (5x5 mean)")
        ax.plot(sar["date"], sar[f"{pol}_px"], color=colour, lw=0.8, alpha=0.45, label=f"{pol} (single pixel)")
        ax.scatter(rainy["date"], rainy[f"{pol}_5x5"], marker="v", s=45, color=INK_SECONDARY, zorder=5,
                   label="rain > 5 mm in previous 24 h")
    axes[2].plot(sar["date"], sar["VHmVV_5x5"], color=blue, lw=2, label="VH - VV (5x5 mean)")
    axes[2].plot(sar["date"], sar["VHmVV_px"], color=blue, lw=0.8, alpha=0.45, label="VH - VV (single pixel)")
    axes[3].plot(s2["date"], s2["ndvi"], color=green, lw=2, marker="o", ms=5, label="NDVI")
    axes[4].plot(s2["date"], s2["ndwi"], color=blue, lw=2, marker="o", ms=5, label="NDWI")
    wet = s2[s2["flooded"]]
    if len(wet):
        axes[4].scatter(wet["date"], wet["ndwi"], s=110, facecolors="none", edgecolors=orange, lw=2,
                        zorder=5, label="open water at pixel (NDWI > 0)")
    for a, label in zip(axes, PANELS):
        a.set_ylabel(label, fontsize=9)
        for start, end, _ in _phases():
            a.axvspan(start, end, color="#8a8985", alpha=0.10, lw=0)
        a.grid(True, alpha=0.2)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    for start, _, label in _phases():
        axes[0].text(start, 1.0, f" {label}", transform=axes[0].get_xaxis_transform(), fontsize=9,
                     va="bottom", color=INK_SECONDARY)
    fig.suptitle(title, x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    return fig


def _plot_plotly(sar, s2, title, out_path, width, height):
    from plotly.subplots import make_subplots

    from .curves import SERIES_COLORS

    blue, orange, green = SERIES_COLORS
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035, subplot_titles=PANELS)
    rainy = sar[sar["rainy"]]
    for row, pol, colour in ((1, "VH", blue), (2, "VV", orange)):
        fig.add_scatter(x=sar["date"], y=sar[f"{pol}_5x5"], name=f"{pol} (5x5 mean)",
                        line=dict(color=colour, width=2.5), row=row, col=1)
        fig.add_scatter(x=sar["date"], y=sar[f"{pol}_px"], name=f"{pol} (single pixel)",
                        line=dict(color=colour, width=1), opacity=0.45, row=row, col=1)
        fig.add_scatter(x=rainy["date"], y=rainy[f"{pol}_5x5"], mode="markers",
                        name="rain > 5 mm / 24 h", showlegend=(row == 1), legendgroup="rain",
                        marker=dict(symbol="triangle-down", size=10, color="#52514e"),
                        customdata=rainy["rain_24h_mm"],
                        hovertemplate="%{x|%d %b %Y}<br>rain %{customdata:.1f} mm", row=row, col=1)
    fig.add_scatter(x=sar["date"], y=sar["VHmVV_5x5"], name="VH - VV (5x5 mean)", line=dict(color=blue, width=2.5), row=3, col=1)
    fig.add_scatter(x=sar["date"], y=sar["VHmVV_px"], name="VH - VV (single pixel)", line=dict(color=blue, width=1), opacity=0.45, row=3, col=1)
    fig.add_scatter(x=s2["date"], y=s2["ndvi"], name="NDVI", mode="lines+markers",
                    line=dict(color=green, width=2.5), customdata=s2["clear"],
                    hovertemplate="%{x|%d %b %Y}<br>NDVI %{y:.2f}<br>clear %{customdata:.0f}", row=4, col=1)
    fig.add_scatter(x=s2["date"], y=s2["ndwi"], name="NDWI", mode="lines+markers",
                    line=dict(color=blue, width=2.5), customdata=s2["clear"],
                    hovertemplate="%{x|%d %b %Y}<br>NDWI %{y:.2f}<br>clear %{customdata:.0f}", row=5, col=1)
    wet = s2[s2["flooded"]]
    if len(wet):
        fig.add_scatter(x=wet["date"], y=wet["ndwi"], mode="markers", name="open water at pixel",
                        marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=orange, width=2)), row=5, col=1)
    for start, end, label in _phases():
        fig.add_vrect(x0=start, x1=end, fillcolor="#8a8985", opacity=0.10, line_width=0,
                      annotation_text=label, annotation_position="top left")
    fig.update_yaxes(autorange=True)
    fig.update_layout(title=title, width=width, height=height, hovermode="x unified",
                      template="plotly_white", legend=dict(orientation="h", y=-0.05))
    if out_path:
        fig.write_html(str(out_path))
    return fig
