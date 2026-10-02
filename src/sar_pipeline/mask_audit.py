"""Why does a field have no NDVI value on a date whose chip looks clear?

Why
---
The NDVI series only keeps a date for a pixel when every mask test passes (data present, QA60 clear,
Cloud Score+ clear, no blue-band haze). When a reviewer sees a clear chip but no curve point, the cause is one
of these tests, or the date is after the map date. The columns follow the delivered mask: opaque-cloud bit only (the
cirrus bit is shown but NOT used), Cloud Score+ below the cut-off unless the pixel is dark, blue-band haze; ``kept`` is
the share of the field's pixels that survive all of them. This module prints, per Sentinel-2 date, how each
test scores on the field's own pixels, so the cause can be read off instead of guessed. It only reads; it
changes nothing.

With ``--events`` it prints instead the rule's own per-pixel evidence (``monsoon_rule.aoi_events``: trough,
climb, flood date and depth, radar canopy rise, last clear view ...) summarised over the field, so a reviewer
sees which condition decided the label. ``field_ndvi_on`` checks a claim over a whole class ("98 % of the
young-rice fields are green on 28 Sep") instead of a handful of examples. (This module replaces the earlier
``field_dates`` and ``review_optical`` helpers.)

Use::

    python -m sar_pipeline.mask_audit aoi63_000184 --since 2026-08-15
    python -m sar_pipeline.mask_audit aoi63_000184 --events      # the rule's evidence (about 1-2 GB of memory)
    python -m sar_pipeline.mask_audit aoi13_000169 --events --curves --root processed/_batch/s2_2026_water3 --water v3
    python -m sar_pipeline.mask_audit aoi160:45142 --events --curves ...   # one pixel (id from grid/pixel_index.tif)
"""
from __future__ import annotations

import glob
import re

import numpy as np
import pandas as pd

from .analysis import ndvi_5day as nd


def nd_root() -> str:
    """The standard series folder (the delivered map)."""
    return "processed/_batch/s2_2026"


S2_DIR = "data/s2_dates_masks"
CS_MIN = 40   # Cloud Score+ cut-off of the delivered mask (hyb40), see analysis/mask_experiment.py
MAP_DATE = "2026-09-21"   # the map ends on this date; later files are not used by the current map


def date_table(field_id: str, since: str = "2026-08-15", s2_dir: str = S2_DIR, map_date: str = MAP_DATE) -> pd.DataFrame:
    """One row per Sentinel-2 date from ``since``: the field's median band values and the share failing each test."""
    import rasterio

    from .optical_export import band_index
    from .qgis_review import field_by_id

    aoi_id, _, pids = field_by_id(field_id)
    rows = []
    for path in sorted(glob.glob(f"{s2_dir}/aoi{aoi_id}/*_S2_*.tif")):
        date = pd.Timestamp(re.search(r"_S2_(\d{4}-\d{2}-\d{2})", path).group(1))
        if date < pd.Timestamp(since):
            continue
        with rasterio.open(path) as ds:
            band = lambda n: ds.read(band_index(ds, n)).astype("float32").reshape(-1)[pids]
            b2, b4, b8, qa, scl, cs = (band(n) for n in ("B2", "B4", "B8", "QA60", "SCL", "clear"))
        ndvi = (b8 - b4) / np.maximum(b8 + b4, 1)
        opaque, cirrus = (qa.astype(int) >> 10) & 1, (qa.astype(int) >> 11) & 1
        with np.errstate(invalid="ignore"):
            haze = (b2 > nd.HAZE_B2_MIN) & ((ndvi >= nd.HAZE_NDVI_MIN) | (b2 >= nd.HAZE_BLUE_TO_RED_MIN * b4))
            dark = (b8 < nd.DARK_NIR_MAX) & (ndvi < nd.DARK_NDVI_MAX)
        cs_ok = (cs >= CS_MIN) | dark
        kept = (b4 > 0) & (b8 > 0) & (opaque == 0) & cs_ok & ~haze
        rows.append({"date": date.date(), "after_map_date": date > pd.Timestamp(map_date),
                     "no_data_b8_or_b4_zero": round(float(np.mean((b4 <= 0) | (b8 <= 0))), 2),
                     "qa60_opaque": round(float(opaque.mean()), 2),
                     "qa60_cirrus": round(float(cirrus.mean()), 2), "cs_fails": round(float(1 - cs_ok.mean()), 2),
                     "haze": round(float(haze.mean()), 2), "kept": round(float(kept.mean()), 2), "cs_median": float(np.median(cs)),
                     "scl_mode": int(np.bincount(scl.astype(int)).argmax()), "ndvi_median": round(float(np.median(ndvi)), 2),
                     "b2_median": float(np.median(b2))})
    return pd.DataFrame(rows)


def field_ndvi_on(aoi_id: int, date: str, labels=(6,), s2_dir: str = S2_DIR) -> pd.DataFrame:
    """Raw median NDVI on one date for every delivered field of the given labels (no cloud mask applied).

    Why: to test a claim about a whole class ("98 % of young-rice fields are green on 28 Sep") on all fields
    instead of on a handful of examples. Fields whose raw view is cloudy are kept; read ``scl_mode`` and
    ``cs_median`` beside the NDVI.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from .optical_export import band_index
    from .qgis_review import SRC

    f = gpd.read_file(f"{SRC}/fields/aoi{aoi_id}_fields_monsoon2026.gpkg")
    f = f[f["label"].isin(labels)].reset_index(drop=True)
    path = glob.glob(f"{s2_dir}/aoi{aoi_id}/*_S2_{date}.tif")[0]
    with rasterio.open(path) as ds:
        band = lambda n: ds.read(band_index(ds, n)).astype("float32")
        b4, b8, scl, cs = band("B4"), band("B8"), band("SCL"), band("clear")
        ids = rasterize([(g, i + 1) for i, g in enumerate(f.to_crs(ds.crs).geometry)], out_shape=b4.shape,
                        transform=ds.transform, fill=0, dtype="int32")
    ndvi = (b8 - b4) / np.maximum(b8 + b4, 1)
    rows = []
    for i, fid in enumerate(f["field_id"], 1):
        m = ids == i
        if m.any():
            rows.append({"field_id": fid, "label": int(f.loc[i - 1, "label"]), "pixels": int(m.sum()),
                         "ndvi_median": float(np.median(ndvi[m])), "scl_mode": int(np.bincount(scl[m].astype(int)).argmax()),
                         "cs_median": float(np.median(cs[m]))})
    return pd.DataFrame(rows)


def summarise_events(events: pd.DataFrame) -> pd.Series:
    """One value per evidence column over a field's pixels: median for numbers, share for booleans, mode for dates."""
    out = {}
    for c in events.columns:
        col = events[c]
        if col.dtype == bool:
            out[c] = f"{col.mean():.2f} share"
        elif np.issubdtype(col.dtype, np.number):
            out[c] = round(float(np.nanmedian(col)), 3) if col.notna().any() else np.nan
        else:
            out[c] = col.mode().iloc[0] if col.notna().any() else None
    return pd.Series(out)


def pixels_of(target: str):
    """(aoi_id, pixel ids) for a field id (``aoi13_000598``) or a single pixel (``aoi160:45142``, the value of
    ``grid/pixel_index.tif`` in QGIS)."""
    if ":" in target:
        aoi, pid = target.split(":")
        return int(aoi.replace("aoi", "")), np.array([int(pid)])
    from .qgis_review import field_by_id

    aoi_id, _, pids = field_by_id(target)
    return aoi_id, pids


def field_events(field_id: str, out_root: str = nd_root(), water: str | None = None) -> pd.Series:
    """The rule's per-pixel evidence (``monsoon_rule.aoi_events``) summarised over the field's pixels.

    ``out_root``: the series folder (a sandbox such as ``processed/_batch/s2_2026_water3``); ``water``: the water test
    (``v2`` / ``v3``; default: the rule's default)."""
    from .analysis import monsoon_rule as mr

    aoi_id, pids = pixels_of(field_id)
    _, events, _ = mr.aoi_events(aoi_id, out_root=out_root, **({"water": water} if water else {}))
    return summarise_events(events.iloc[pids])


def field_classes(field_id: str, out_root: str = nd_root()) -> dict:
    """Pixel count per class of the field on the final map in ``out_root``."""
    import rasterio

    aoi_id, pids = pixels_of(field_id)
    with rasterio.open(f"{out_root}/aoi{aoi_id}/aoi{aoi_id}_monsoon2026_final.tif") as ds:
        c = ds.read(1).ravel()[pids]
    return {int(k): int(v) for k, v in zip(*np.unique(c, return_counts=True))}


def field_curves(field_id: str, since: str = "2026-04-01", out_root: str = nd_root()) -> str:
    """The field's median clear NDVI per 5-day window and median VV / VH per radar pass of every track, as text.

    Why: a reviewer reads the curves in the inspection plot; this prints the same numbers so a claim ("the radar dips in
    July and rises in September") can be checked on the values."""
    from .analysis import radar_water as rw

    aoi_id, pids = pixels_of(field_id)
    d = nd.load(aoi_id, out_root=out_root)
    raw = d["ndvi5d_raw"].reshape(d["ndvi5d_raw"].shape[0], -1)[:, pids]
    nd.forget()
    w = pd.DatetimeIndex(d["windows"])
    lines = []
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(raw, axis=1)
            lines.append("NDVI (clear, field median): " + " ".join(
                f"{a:%d%b}:{b:.2f}" for a, b in zip(w, med) if a >= pd.Timestamp(since) and np.isfinite(b)))
            for i, (dates, flat) in enumerate(rw.read_series(aoi_id)):
                day = pd.DatetimeIndex(dates)
                keep = day >= pd.Timestamp(since)
                vv = np.nanmedian(flat["VV"][:, pids], axis=1)[keep]
                vh = np.nanmedian(flat["VH"][:, pids], axis=1)[keep]
                lines.append(f"track {i} VV/VH: " + " ".join(f"{a:%d%b}:{x:.1f}/{y:.1f}" for a, x, y in zip(day[keep], vv, vh)))
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.mask_audit", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("field_id")
    p.add_argument("--since", default="2026-08-15")
    p.add_argument("--map-date", default=MAP_DATE)
    p.add_argument("--events", action="store_true", help="print the rule's evidence for the field instead of the table")
    p.add_argument("--curves", action="store_true", help="also print the field's NDVI and VV / VH values (with --events)")
    p.add_argument("--root", default=nd_root(), help="series / map folder, e.g. a sandbox (with --events)")
    p.add_argument("--water", choices=["v2", "v3"], help="water test for --events (default: the rule's default)")
    args = p.parse_args(argv)
    if args.events:
        print("classes on the final map (pixels):", field_classes(args.field_id, args.root))
        print(field_events(args.field_id, args.root, args.water).to_string())
        if args.curves:
            print(field_curves(args.field_id, out_root=args.root))
        return 0
    print(date_table(args.field_id, args.since, map_date=args.map_date).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
