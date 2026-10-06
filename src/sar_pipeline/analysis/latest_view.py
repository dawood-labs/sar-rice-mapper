"""Each pixel's latest CLEAR Sentinel-2 view: what the field looks like now, read by its tone.

Why (user, 5 Oct 2026, universal rule for every new AOI): "before applying rules to an AOI, look at its latest clear image
and say what each pixel is (water, empty land, flooded field, green crop, trees); only the latest clear view counts; where
a pixel is cloudy on the latest image, go back to the previous image, and further back until it is clear". A class is
then accepted only when it fits both this tone and the curves.

* ``latest_clear``: per pixel, the newest date the series calls clear (the series' own mask, ``ndvi_5day.load``), walking
  back past cloudy dates; its date and its bands (B2 B3 B4 B5 B8 B11).
* ``tone``: the view's tone, relative to the pixel itself (no fixed reflectance cut): ``water`` where the shortwave
  moisture index is above the NDVI (or the NDVI is below 0: open water / a flooded field), ``green`` where the view's NDVI
  stands in the upper half of the pixel's own clear-view range since ``DRY_SEASON_FROM``, ``empty`` otherwise.
* ``class_tones`` / ``gallery``: per class of an accepted (locked) map, how its pixels look on their latest clear view, as
  numbers and as chips, to learn what each class looks like (user, 5 Oct: "go through the locked AOIs and remember how
  every class looks").

    python -m sar_pipeline.analysis.latest_view tones --aoi 63            # table per class of the locked map
    python -m sar_pipeline.analysis.latest_view gallery --aoi 63 --png out.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("B2", "B3", "B4", "B5", "B8", "B11")
#: The pixel's own clear-view range starts here (the dry season), as ``curve_rules.DRY_SEASON_FROM``.
DRY_SEASON_FROM = "2026-03-01"
TONES = ("water", "empty", "green")


def newest_clear_index(ok) -> np.ndarray:
    """Per pixel, the index of its newest clear date in ``ok`` ((dates, pixels), dates ascending); -1 if never clear."""
    ok = np.asarray(ok, dtype=bool)
    idx = ok.shape[0] - 1 - np.argmax(ok[::-1], axis=0)
    return np.where(ok.any(axis=0), idx, -1)


def tone(ndvi_now, lswi_now, ndvi_lo, ndvi_hi) -> np.ndarray:
    """``water`` / ``green`` / ``empty`` per pixel (see the module notes); ``""`` where the pixel has no clear view."""
    ndvi_now, lswi_now = np.asarray(ndvi_now, float), np.asarray(lswi_now, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = (ndvi_now - ndvi_lo) / (np.asarray(ndvi_hi, float) - np.asarray(ndvi_lo, float))
    out = np.where(pos >= 0.5, "green", "empty").astype(object)
    out = np.where((lswi_now > ndvi_now) | (ndvi_now < 0), "water", out)
    return np.where(np.isfinite(ndvi_now), out, "").astype(str)


def latest_clear(aoi: int, series_root: str | None = None) -> dict:
    """Per pixel: ``date`` (datetime64, NaT if never clear), ``ndvi``, ``lswi``, ``tone`` and the bands of its latest clear
    view (``bands``: (len(BANDS), pixels) reflectance x 10000), plus the grid ``shape``."""
    import rasterio

    from ..optical_export import band_index
    from . import ndvi_5day as nd
    from . import pixel_report as pr

    series_root = series_root or nd.analysis_series_root(aoi)
    d = nd.load(aoi, out_root=series_root)
    dt = pd.DatetimeIndex(d["dates"])
    n = len(dt)
    ok = d["ok"].reshape(n, -1).astype(bool)
    nv = d["ndvi"].reshape(n, -1).astype("float32")
    lv = d["lswi"].reshape(n, -1).astype("float32")
    shape = d["ndvi"].shape[1:]
    loc = d["loc"]
    nd.forget()
    i = newest_clear_index(ok)
    px = np.arange(ok.shape[1])
    has = i >= 0
    ndvi_now = np.where(has, nv[np.clip(i, 0, None), px], np.nan)
    lswi_now = np.where(has, lv[np.clip(i, 0, None), px], np.nan)
    year = ok & (dt >= pd.Timestamp(DRY_SEASON_FROM))[:, None]
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanmin(np.where(year, nv, np.nan), axis=0)
            hi = np.nanmax(np.where(year, nv, np.nan), axis=0)
    bands = np.full((len(BANDS), ok.shape[1]), np.nan, dtype="float32")
    folder = pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER)
    by_date = {pd.Timestamp(p.stem.rsplit("_S2_", 1)[1]): p for p in Path(folder).glob("*.tif")}
    for j in np.unique(i[has]):
        with rasterio.open(by_date[dt[j]]) as ds:
            arr = ds.read([band_index(ds, b) for b in BANDS]).reshape(len(BANDS), -1).astype("float32")
        sel = i == j
        bands[:, sel] = arr[:, sel]
    date = np.where(has, dt.values[np.clip(i, 0, None)], np.datetime64("NaT"))
    return {"date": date, "ndvi": ndvi_now, "lswi": lswi_now, "tone": tone(ndvi_now, lswi_now, lo, hi),
            "bands": bands, "shape": shape}


def _class_map(aoi: int, path: str | None = None) -> np.ndarray:
    import rasterio

    path = path or f"processed/_batch/s2_2026/rice_fresh/locked/aoi{aoi}/aoi{aoi}_rel_class.tif"
    with rasterio.open(path) as ds:
        return ds.read(1).ravel()


def class_tones(aoi: int, class_path: str | None = None, series_root: str | None = None) -> pd.DataFrame:
    """Per class of the (locked) map: pixels, median date of the latest clear view, the tone shares, and the median band
    values, NDVI and LSWI of that view."""
    from .curve_rules import MAP_CLASSES

    v = latest_clear(aoi, series_root)
    c = _class_map(aoi, class_path)
    rows = []
    for k, (name, _) in MAP_CLASSES.items():
        sel = (c == k) & (v["tone"] != "")
        if not sel.any():
            continue
        row = {"aoi": aoi, "class": name, "pixels": int(sel.sum()),
               "view_date": str(pd.Series(v["date"][sel]).median().date())}
        for t in TONES:
            row[f"{t} %"] = round(100 * float((v["tone"][sel] == t).mean()), 1)
        for b, x in zip(BANDS, v["bands"]):
            row[b] = int(np.nanmedian(x[sel]))
        row["NDVI"] = round(float(np.nanmedian(v["ndvi"][sel])), 2)
        row["LSWI"] = round(float(np.nanmedian(v["lswi"][sel])), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def core_pixels(classes: np.ndarray, shape, k: int, n: int, seed: int = 0) -> np.ndarray:
    """Up to ``n`` pixels of class ``k`` whose whole 5x5 block has that class (field interiors, never edges)."""
    from scipy.ndimage import minimum_filter

    m = (classes.reshape(shape) == k)
    core = minimum_filter(m.astype(np.uint8), size=5, mode="constant") == 1
    ids = np.flatnonzero(core.ravel())
    if len(ids) > n:
        ids = np.sort(np.random.default_rng(seed).choice(ids, n, replace=False))
    return ids


def gallery(aoi: int, out_png: str, per_class: int = 4, half: int = 10, class_path: str | None = None,
            series_root: str | None = None) -> pd.DataFrame:
    """Chips of each class on its pixels' latest clear view (true colour 4-3-2 and 5-3-2), ``per_class`` field-interior
    pixels per class, pixel in the middle; returns the pixels shown."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio

    from ..optical_export import band_index
    from . import ndvi_5day as nd
    from . import pixel_report as pr
    from .curve_rules import MAP_CLASSES

    v = latest_clear(aoi, series_root)
    shape = v["shape"]
    c = _class_map(aoi, class_path)
    loc = pr.locate(aoi, 0)
    folder = pr.sync_s2(loc, f"data/{nd.FOLDER}", nd.FOLDER)
    by_date = {pd.Timestamp(p.stem.rsplit("_S2_", 1)[1]): p for p in Path(folder).glob("*.tif")}
    picks = [(k, name, p) for k, (name, _) in MAP_CLASSES.items() for p in core_pixels(c, shape, k, per_class)
             if v["tone"][p] != ""]
    if not picks:
        raise ValueError(f"aoi{aoi}: no field-interior pixels to show")
    chips = []
    for k, name, p in picks:
        r, col = divmod(int(p), shape[1])
        date = pd.Timestamp(v["date"][p])
        with rasterio.open(by_date[date]) as ds:
            win = rasterio.windows.Window(col - half, r - half, 2 * half + 1, 2 * half + 1)
            chips.append(ds.read([band_index(ds, b) for b in ("B4", "B3", "B2", "B5")], window=win, boundless=True,
                                 fill_value=0).astype("float32"))
    # ONE stretch for every chip of the sheet (per band, 2-98 % over all chips): with a stretch per chip, open water
    # (NDVI -0.7) came out bright yellow-green and a hazy field dark navy, so tones could not be compared
    allc = np.stack(chips)
    lim = {b: np.percentile(allc[:, b][allc[:, b] > 0], (2, 98)) for b in range(4)}
    fig, axes = plt.subplots(len(picks), 2, figsize=(5.2, 2.6 * len(picks)), squeeze=False)
    rows = []
    for ax, (k, name, p), arr in zip(axes, picks, chips):
        date = pd.Timestamp(v["date"][p])
        for a, idx, title in ((ax[0], (0, 1, 2), "4-3-2"), (ax[1], (3, 1, 2), "5-3-2")):
            img = np.stack([np.clip((arr[b] - lim[b][0]) / max(lim[b][1] - lim[b][0], 1), 0, 1) for b in idx], -1)
            a.imshow(img ** 0.7)
            a.plot(half, half, "s", mfc="none", mec="magenta", ms=6)
            a.set_xticks([]), a.set_yticks([])
            a.set_title(f"{name[:26]} | px {p} | {date.date()} {title}\nNDVI {v['ndvi'][p]:.2f} LSWI {v['lswi'][p]:.2f} "
                        f"tone {v['tone'][p]}", fontsize=6.5)
        rows.append({"class": name, "pixel": int(p), "date": date.date(), "ndvi": round(float(v["ndvi"][p]), 2),
                     "lswi": round(float(v["lswi"][p]), 2), "tone": v["tone"][p]})
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("step", choices=["tones", "gallery"])
    ap.add_argument("--aoi", type=int, required=True)
    ap.add_argument("--png", help="gallery: output PNG")
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args(argv)
    if args.step == "tones":
        print(class_tones(args.aoi).to_string(index=False))
    else:
        png = args.png or f"processed/_batch/s2_2026/qgis_review/class_tones/aoi{args.aoi}_class_gallery.png"
        print(gallery(args.aoi, png, per_class=args.per_class).to_string(index=False))
        print(png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
