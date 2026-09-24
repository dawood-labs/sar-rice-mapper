"""Radar pictures for looking at rice by eye in QGIS: multi-temporal VH/VV composites over all AOIs.

Why
---
Optical images are few and hazy in the monsoon; the radar sees every 6-12 days through cloud. Paddy
rice has a radar story no other land cover tells in the same order: a **dry** field before the
season (VH moderate), **standing water** at transplanting (VH very dark: water mirrors the signal
away), then a **growing canopy** (VH bright again). Putting three moments of that story into the
red, green and blue channels turns it into a colour:

* ``sar_rice_rgb_mosaic.tif`` — R = VH mean of August-September (canopy), G = the darkest VH of
  June-July (the flood; for a paddy this is very low), B = VH mean of April (dry season).

  ==========================  ===========================================  ===============
  ground                      R (canopy)  G (flood)  B (dry)               looks
  ==========================  ===========================================  ===============
  paddy rice                  high        very low   low                   red / magenta
  dry-land crop (rain-sown)   high        high       low                   yellow / orange
  trees, villages             high        high       high                  white / light
  permanent water             low         low        low                   black
  bare / fallow               low-mid     mid        low                   dark blue-grey
  ==========================  ===========================================  ===============

* ``sar_monthly_mosaic.tif`` — 12 bands: VH April ... September, then VV April ... September
  (monthly means in linear power over every pass of every track, in dB), so any three months can be
  put in R, G, B in QGIS.

Values are stored as dB x 100 (int16; -32768 = no data). The QGIS styles set the display ranges.
Artefact passes (docs/15) are excluded because the radar is read through ``sar_curve.read_track``.

Run (from the repository root; docs/16 explains how to look at the results in QGIS)::

    python -m sar_pipeline.sar_composites aoi --ids 11 116     # per-AOI files
    python -m sar_pipeline.sar_composites mosaic               # mosaics + QGIS .qml styles
    python -m sar_pipeline.sar_composites preview --ids 116    # PNG: RGB next to the final map
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = "processed/_batch/s2_2026/qgis_review/sar"
MONTHS = ["2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
NODATA = -32768


def _power(db):
    """dB -> linear power. Why: radar values must be averaged as power, never as dB (see below)."""
    return 10 ** (np.asarray(db, dtype="float64") / 10.0)


def _db(power):
    """Linear power -> dB (0 or NaN power gives -inf / NaN without a warning)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(power)


def _scaled(a):
    """dB (float) -> dB x 100 as int16, NaN -> ``NODATA``.

    Why: int16 keeps each 12-band AOI file about four times smaller than float64 while holding
    0.01 dB steps, far finer than the ~1 dB speckle.
    """
    return np.where(np.isfinite(a), np.round(a * 100), NODATA).astype("int16")


def composite_arrays(passes: dict, shape: tuple) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """The composites from a list of radar passes (pure numpy, no files; this is what the tests check).

    ``passes`` maps "VV" and "VH" to lists of ``(pd.Timestamp, 2-D dB array)``, every pass of every
    track. Returns ``(monthly, canopy, flood, dry)``:

    * ``monthly[(pol, "YYYY-MM")]``: mean of that month's passes, taken in **linear power** and
      turned back to dB. Why power: dB is a log scale; averaging -10 and -30 dB in dB gives -20, but
      the real mean power is -13 dB. A NaN-filled array of ``shape`` when a month has no pass.
    * ``canopy``: VH power mean over August-September passes (dB).
    * ``flood``: the **lowest** VH of June-July (dB). Why the minimum and not the mean: the flood in
      a paddy lasts a few weeks and falls on different dates field to field; a mean would wash it out.
    * ``dry``: VH of April (the monthly mean above).

    NaN pixels (no data, artefact passes) are ignored in every mean and minimum.
    """
    import warnings

    h, w = shape
    monthly = {}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for pol in ("VH", "VV"):
            for m in MONTHS:
                sel = [a for d, a in passes[pol] if d.strftime("%Y-%m") == m]
                monthly[(pol, m)] = _db(np.nanmean(_power(np.stack(sel)), axis=0)) if sel else np.full((h, w), np.nan)
        vh = passes["VH"]
        canopy = _db(np.nanmean(_power(np.stack([a for d, a in vh if d.month in (8, 9)])), axis=0))
        flood = np.nanmin(np.stack([a for d, a in vh if d.month in (6, 7)]), axis=0)
        dry = monthly[("VH", "2026-04")]
    return monthly, canopy, flood, dry


def aoi_composites(aoi_id: int, window: int = 1, out_dir=OUT) -> dict:
    """Write ``<aoi>_sar_monthly.tif`` (12 bands) and ``<aoi>_sar_rice_rgb.tif`` (3 bands) on the AOI grid.

    Reads every pass of every track of the AOI's ``monsoon2026`` run (``window`` = 1: single pixels,
    no spatial smoothing, so field edges stay sharp; the monthly averaging over several passes is
    what lowers the speckle), then ``composite_arrays`` does the maths.
    """
    import rasterio
    from rasterio.transform import from_origin

    from .analysis import pixel_report as pr
    from .analysis import sar_curve

    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    g = loc["grid"]
    h, w = int(g["height"]), int(g["width"])
    passes = {"VV": [], "VH": []}
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        for pol in ("VV", "VH"):
            for i, d in enumerate(dates):
                passes[pol].append((pd.Timestamp(d), cubes[pol][i]))
    monthly, canopy, flood, dry = composite_arrays(passes, (h, w))
    profile = dict(driver="GTiff", width=w, height=h, dtype="int16", crs=g["crs"], nodata=NODATA,
                   transform=from_origin(g["x0"], g["y0"], g["res"], g["res"]), compress="deflate", tiled=True)
    out = Path(out_dir) / "per_aoi"
    out.mkdir(parents=True, exist_ok=True)

    names = [f"VH_{m}" for m in MONTHS] + [f"VV_{m}" for m in MONTHS]
    with rasterio.open(out / f"aoi{aoi_id}_sar_monthly.tif", "w", count=12, **profile) as ds:
        for i, name in enumerate(names, start=1):
            pol, m = name.split("_")
            ds.write(_scaled(monthly[(pol, m)]), i)
            ds.set_band_description(i, f"{name} (dB x 100)")
    rgb = (("R_VH_Aug-Sep_mean_canopy", canopy), ("G_VH_Jun-Jul_min_flood", flood), ("B_VH_Apr_mean_dry", dry))
    with rasterio.open(out / f"aoi{aoi_id}_sar_rice_rgb.tif", "w", count=3, **profile) as ds:
        for i, (name, a) in enumerate(rgb, start=1):
            ds.write(_scaled(a), i)
            ds.set_band_description(i, f"{name} (dB x 100)")
    return {"aoi": f"aoi{aoi_id}", "passes_VH": len(passes["VH"])}


def mosaic(out_dir=OUT) -> list[Path]:
    """One file per product over all AOIs (``gdalbuildvrt`` + ``gdal_translate``)."""
    import subprocess

    out = Path(out_dir)
    made = []
    for product, n in (("sar_rice_rgb", 3), ("sar_monthly", 12)):
        files = sorted(str(p) for p in (out / "per_aoi").glob(f"aoi*_{product}.tif"))
        lst = out / f"{product}_files.txt"
        lst.write_text("\n".join(files))
        vrt = out / f"{product}.vrt"
        subprocess.run(["gdalbuildvrt", "-q", "-srcnodata", str(NODATA), "-vrtnodata", str(NODATA),
                        "-input_file_list", str(lst), str(vrt)], check=True)
        tif = out / f"{product}_mosaic.tif"
        subprocess.run(["gdal_translate", "-q", "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
                        "-co", "PREDICTOR=2", "-co", "BIGTIFF=IF_SAFER", str(vrt), str(tif)], check=True)
        subprocess.run(["gdaladdo", "-q", "-r", "average", str(tif), "2", "4", "8", "16"], check=True)
        made.append(tif)
    return made


def qml_rgb(red: int, green: int, blue: int, lo: float, hi: float, path) -> Path:
    """A QGIS multiband-colour style: three bands, one fixed dB range (values are dB x 100).

    Why a fixed range: QGIS stretches each band on its own by default, which makes every AOI and
    every band look different; one fixed range for all three bands keeps the colours comparable
    across AOIs, so "red" means the same radar story everywhere. ``lo``/``hi`` are in dB.
    """
    band = lambda tag, b: (f'      <{tag}ContrastEnhancement><minValue>{lo * 100:.0f}</minValue>'  # noqa: E731
                           f'<maxValue>{hi * 100:.0f}</maxValue><algorithm>StretchToMinimumMaximum</algorithm>'
                           f'</{tag}ContrastEnhancement>')
    text = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="multibandcolor" redBand="{red}" greenBand="{green}" blueBand="{blue}" opacity="1">
{band('red', red)}
{band('green', green)}
{band('blue', blue)}
    </rasterrenderer>
  </pipe>
</qgis>
"""
    Path(path).write_text(text)
    return Path(path)


def preview(aoi_id: int, lo: float = -26, hi: float = -12, out_dir=OUT) -> Path:
    """PNG: the rice RGB composite next to the final class map of one AOI (a quick check of the colours)."""
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap

    from .qgis_review import COLOURS

    with rasterio.open(Path(out_dir) / "per_aoi" / f"aoi{aoi_id}_sar_rice_rgb.tif") as ds:
        rgb = ds.read().astype("float32")
        rgb[rgb == NODATA] = np.nan
    img = np.clip((rgb / 100 - lo) / (hi - lo), 0, 1).transpose(1, 2, 0)
    img = np.nan_to_num(img)
    with rasterio.open(f"processed/_batch/s2_2026/aoi{aoi_id}/aoi{aoi_id}_monsoon2026_final.tif") as ds:
        c = ds.read(1).astype("float32")
    c[c == 255] = np.nan
    cmap = ListedColormap([COLOURS[k][1] for k in range(len(COLOURS))])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 9))
    a1.imshow(img, interpolation="nearest")
    a1.set_title(f"aoi{aoi_id}: R = VH Aug-Sep, G = VH min Jun-Jul, B = VH Apr ({lo}..{hi} dB)", fontsize=10)
    a2.imshow(c, cmap=cmap, vmin=-0.5, vmax=len(COLOURS) - 0.5, interpolation="nearest")
    a2.set_title("final class map (green rice, light green young rice, orange class 3, grey not rice)", fontsize=10)
    for a in (a1, a2):
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    path = Path(out_dir) / f"preview_aoi{aoi_id}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def single_pass(aoi_id: int, month: str = "2026-09", track: str | None = None, window: int = 1,
                out_dir=OUT) -> Path:
    """The latest pass of a month for one AOI: bands VV, VH and VV-VH in dB x 100, with a QGIS style.

    Why: one recent date shows the fields as they stand on the map date. Displayed as R = VH, G = VV,
    B = VH (each band with its own range), a standing rice canopy shows magenta: its vertical stems
    weaken VV more than VH; bare or wet soil shows green (VV high, VH low); trees and houses white;
    water black. ``track`` picks one track; by default the latest pass of any track.
    """
    import rasterio
    from rasterio.transform import from_origin

    from .analysis import pixel_report as pr
    from .analysis import sar_curve

    loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
    g = loc["grid"]
    best = None
    for tr in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        if track and tr != track:
            continue
        dates, cubes = sar_curve.read_track(loc, tr, window)
        for i, d in enumerate(dates):
            if pd.Timestamp(d).strftime("%Y-%m") == month and (best is None or d > best[1]):
                best = (tr, d, cubes["VV"][i], cubes["VH"][i])
    if best is None:
        raise ValueError(f"no pass in {month} for aoi{aoi_id}")
    tr, d, vv, vh = best
    profile = dict(driver="GTiff", width=int(g["width"]), height=int(g["height"]), dtype="int16", count=3,
                   crs=g["crs"], nodata=NODATA, transform=from_origin(g["x0"], g["y0"], g["res"], g["res"]),
                   compress="deflate", tiled=True)
    out = Path(out_dir) / "single_pass"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"aoi{aoi_id}_{tr}_{pd.Timestamp(d).date()}_VV_VH.tif"
    with rasterio.open(path, "w", **profile) as ds:
        for i, (name, a) in enumerate((("VV", vv), ("VH", vh), ("VV_minus_VH", vv - vh)), start=1):
            ds.write(_scaled(a), i)
            ds.set_band_description(i, f"{name} (dB x 100) {tr} {pd.Timestamp(d).date()}")
    style = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="multibandcolor" redBand="2" greenBand="1" blueBand="2" opacity="1">
      <redContrastEnhancement><minValue>-2500</minValue><maxValue>-1000</maxValue><algorithm>StretchToMinimumMaximum</algorithm></redContrastEnhancement>
      <greenContrastEnhancement><minValue>-2000</minValue><maxValue>-400</maxValue><algorithm>StretchToMinimumMaximum</algorithm></greenContrastEnhancement>
      <blueContrastEnhancement><minValue>-2500</minValue><maxValue>-1000</maxValue><algorithm>StretchToMinimumMaximum</algorithm></blueContrastEnhancement>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    path.with_suffix(".qml").write_text(style)
    return path


def preview_single(path, aoi_id: int, ranges=((-25, -10), (-20, -4), (-25, -10))) -> Path:
    """PNG of a single-pass file as R = VH, G = VV, B = VH next to the final class map."""
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap

    from .qgis_review import COLOURS

    with rasterio.open(path) as ds:
        vv, vh = (ds.read(b).astype("float32") / 100 for b in (1, 2))
    chans = [vh, vv, vh]
    img = np.dstack([np.clip((c - lo) / (hi - lo), 0, 1) for c, (lo, hi) in zip(chans, ranges)])
    img = np.nan_to_num(img)
    with rasterio.open(f"processed/_batch/s2_2026/aoi{aoi_id}/aoi{aoi_id}_monsoon2026_final.tif") as ds:
        c = ds.read(1).astype("float32")
    c[c == 255] = np.nan
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 9))
    a1.imshow(img, interpolation="nearest")
    a1.set_title(f"{Path(path).stem}: R = VH, G = VV, B = VH", fontsize=10)
    a2.imshow(c, cmap=ListedColormap([COLOURS[k][1] for k in range(len(COLOURS))]), vmin=-0.5, vmax=len(COLOURS) - 0.5, interpolation="nearest")
    a2.set_title("final class map", fontsize=10)
    for a in (a1, a2):
        a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    out = Path(path).with_suffix(".png")
    fig.savefig(out, dpi=80)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    import argparse
    import os

    os.environ["PATH"] = "/opt/gis/bin:" + os.environ.get("PATH", "")
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.sar_composites", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    a = sub.add_parser("aoi", help="per-AOI composites")
    a.add_argument("--ids", nargs="+", type=int, required=True)
    sub.add_parser("mosaic", help="mosaic the per-AOI files and write the QGIS styles")
    v = sub.add_parser("preview", help="PNG of the rice RGB next to the final class map (per AOI)")
    v.add_argument("--ids", nargs="+", type=int, required=True)
    args = p.parse_args(argv)
    if args.step == "aoi":
        for i in args.ids:
            print(aoi_composites(i))
    elif args.step == "preview":
        for i in args.ids:
            print(preview(i))
    else:
        for t in mosaic():
            print(t)
        qml_rgb(1, 2, 3, -26, -12, Path(OUT) / "sar_rice_rgb.qml")
        qml_rgb(6, 3, 1, -26, -12, Path(OUT) / "sar_monthly_VH_Sep-Jun-Apr.qml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
