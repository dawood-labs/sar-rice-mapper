"""Fresh start, step 1: split the September vegetation into ground that went bare since 1 May and ground that never did.

Why
---
Fresh start (user, 30 Sep 2026): the September first-clear image, reduced to its vegetated pixels
(``first_clear --vegetation``), holds crops, orchards and trees alike. A crop field is bare (or under water) at least
once between seasons; trees, orchards and other permanent cover are not. So, per vegetated pixel, since 1 May (user):

* ``sustained_low``: the lowest level the pixel's fitted NDVI HELD for at least two windows in a row (the lower of each
  pair's higher value). One low window alone is usually a hazy view on a canopy (the lesson of the 40-day rule); the
  upper-envelope fit already down-weights such views.
* The split between "went bare" and "never bare" is read from this AOI's vegetated pixels themselves: Otsu's split of
  their ``sustained_low`` (``first_clear.otsu``), no fixed NDVI.
* The radar is reported beside it, not used to decide: ``vh_low``, the pixel's second-darkest VH pass since 1 May over
  all tracks (dB; a canopy or a building never goes dark, a bare or flooded field does). One pass alone can be speckle.

Outputs in ``<out_root>/rice_fresh/aoi<N>/``: ``aoi<N>_step1_cover.tif`` (0 not vegetation in September, 1 went bare
since 1 May = crop ground, 2 never bare = trees / orchards / permanent cover, 255 outside) + ``.qml``,
``aoi<N>_step1_vh_low.tif``, ``aoi<N>_step1.csv`` (acres) and ``aoi<N>_step1.png`` (map + histogram).

Use::

    python -m sar_pipeline.analysis.vegetation_types --aoi 160 --series-root processed/_batch/s2_2026_hyb40m1late
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from .first_clear import otsu
from .optical_phenology import acres

START = "2026-05-01"
CLASSES = {0: "not vegetation in September", 1: "went bare since 1 May (crop ground)",
           2: "never bare since 1 May (trees / orchards / permanent cover)", 255: "outside"}
COLOURS = {0: "#d9d9d9", 1: "#e3a21a", 2: "#1b5e20"}


def sustained_low(fit, windows, start: str = START) -> np.ndarray:
    """Per pixel (columns of ``fit`` (windows, pixels)): the lowest value held for two windows in a row from ``start``."""
    fit = np.asarray(fit, dtype="float32")
    use = pd.DatetimeIndex(windows) >= pd.Timestamp(start)
    f = fit[use]
    pair = np.fmax(f[:-1], f[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(pair, axis=0)


def second_darkest(series, start: str = START, pol: str = "VH") -> np.ndarray:
    """Per pixel: the second-darkest ``pol`` pass since ``start`` over all tracks (dB); NaN without two passes."""
    parts = []
    for dates, flat in series:
        day = pd.DatetimeIndex(dates)
        parts.append(np.asarray(flat[pol], dtype="float32")[day >= pd.Timestamp(start)])
    x = np.concatenate(parts) if parts else np.empty((0, 0))
    x = np.where(np.isfinite(x), x, np.inf)
    if x.shape[0] < 2:
        return np.full(x.shape[1] if x.ndim == 2 else 0, np.nan)
    two = np.partition(x, 1, axis=0)[1]
    return np.where(np.isfinite(two), two, np.nan)


def swing(series, start: str = START, pol: str = "VH") -> np.ndarray:
    """Per pixel: how far the signal swung since ``start`` on its most-swinging track, in dB: the second-highest pass
    minus the second-lowest (one speckled pass at either end ignored). A crop goes from bare or flooded ground to a
    canopy (a big swing); a tree or an orchard stays near one level all season."""
    best = None
    for dates, flat in series:
        x = np.asarray(flat[pol], dtype="float32")[pd.DatetimeIndex(dates) >= pd.Timestamp(start)]
        if x.shape[0] < 4:
            continue
        lo = np.partition(np.where(np.isfinite(x), x, np.inf), 1, axis=0)[1]
        hi = -np.partition(np.where(np.isfinite(x), -x, np.inf), 1, axis=0)[1]
        sw = np.where(np.isfinite(lo) & np.isfinite(hi), hi - lo, np.nan)
        best = sw if best is None else np.fmax(best, sw)
    return best


def ratio_low(series, start: str = START) -> np.ndarray:
    """Per pixel: the second-lowest VH - VV (dB) since ``start`` over all tracks. Bare soil scatters mostly in VV (a low
    ratio, even when dry and not dark); a canopy (trees, a grown crop) scatters in VH too (a high ratio)."""
    parts = []
    for dates, flat in series:
        use = pd.DatetimeIndex(dates) >= pd.Timestamp(start)
        parts.append((np.asarray(flat["VH"], dtype="float32") - np.asarray(flat["VV"], dtype="float32"))[use])
    x = np.concatenate(parts)
    two = np.partition(np.where(np.isfinite(x), x, np.inf), 1, axis=0)[1]
    return np.where(np.isfinite(two), two, np.nan)


def ground_bare_level(k: float = 2.0) -> float:
    """The highest NDVI a field at sowing still has, from the ground data: the user-confirmed rice plots' NDVI at their
    sowing, median + ``k`` x spread (about 0.30; the user: "a trough is usually around 0.2"). NaN without the files."""
    try:
        from .rice_features import plot_table

        t = plot_table()
    except (FileNotFoundError, OSError, KeyError):
        return float("nan")
    r = t.loc[t["is_rice"].astype(bool), "trough"].dropna().to_numpy()
    if len(r) == 0:
        return float("nan")
    med = np.median(r)
    return float(med + k * 1.4826 * np.median(np.abs(r - med)))


def cover_classes(low, vegetation, inside, floor: float | None = None) -> tuple[np.ndarray, float]:
    """(classes, split): 1 went bare / 2 never bare among the vegetated pixels, split = Otsu of their ``low``, never
    below ``floor`` (the ground data's bare level, :func:`ground_bare_level`): where an AOI has hardly any trees Otsu
    cuts its one group of fields in two (aoi116: split 0.09, 1,584 ac of bare-soil fields at 0.125 called trees)."""
    low = np.asarray(low, dtype=float)
    veg = np.asarray(vegetation, dtype=bool) & np.asarray(inside, dtype=bool)
    # open water (NDVI below 0, a physical boundary) is "empty or wet" like bare soil: counted as 0, so the split falls
    # between ground that went low and ground that stayed green (aoi116: water -0.1, soil 0.1, trees 0.6; without
    # this the split fell between water and soil and 2,623 ac of fields became "trees")
    split = otsu(np.clip(low[veg & np.isfinite(low)], 0, None))
    if floor is not None and np.isfinite(floor):
        split = max(split, floor)
    out = np.full(low.shape, 255, dtype="uint8")
    out[np.asarray(inside, dtype=bool)] = 0
    out[veg & ~(low > split)] = 1
    out[veg & (low > split)] = 2
    return out, split


def radar_went_bare(cover, vh_low, vegetation) -> tuple[np.ndarray, float]:
    """(cover, split_db): a "never bare" pixel (2) becomes crop ground (1) when its radar went dark: the second-darkest
    VH pass since 1 May (two passes, so one speckle cannot do it) at or below this AOI's own Otsu split of that value
    among its September vegetation. Why (user, 1 Oct, aoi39 pixel 24703): no clear optical view from 19 May to 22 Aug,
    the filled NDVI never fell, so the optical said "tree"; the radar showed VH at -21/-22 dB from late June to August
    (a flooded paddy) and -13 in September. The radar fills the optical's gaps; a tree's VH stays near one level."""
    cover = np.array(cover, copy=True)
    vh_low = np.asarray(vh_low, dtype=float)
    veg = np.asarray(vegetation, dtype=bool) & np.isfinite(vh_low)
    if veg.sum() < 2:
        return cover, np.nan
    split = otsu(vh_low[veg])
    with np.errstate(invalid="ignore"):
        cover[(cover == 2) & (vh_low <= split)] = 1
    return cover, split


def run(aoi_id: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
        vegetation_mask: str | None = None, out_root: str = "processed/_batch/s2_2026") -> pd.DataFrame:
    """Step 1 for one AOI; returns the acre table."""
    import rasterio

    from . import radar_water as rw

    vegetation_mask = vegetation_mask or (f"{out_root}/first_clear/aoi{aoi_id}/"
                                          f"aoi{aoi_id}_first_clear_2026-09-01_2026-09-30_vegetation_mask.tif")
    with rasterio.open(vegetation_mask) as ds:
        vm = ds.read(1)
        profile = ds.profile
    shape = vm.shape
    inside = (vm != 255).ravel()
    veg = (vm == 1).ravel()
    d = nd.load(aoi_id, out_root=series_root)
    fit = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    low = sustained_low(fit, d["windows"])
    del fit
    nd.forget()
    vh_low = second_darkest(rw.read_series(aoi_id))
    cover, split = cover_classes(low, veg, inside, floor=ground_bare_level())
    optical_trees = int((cover == 2).sum())
    cover, radar_split = radar_went_bare(cover, vh_low, veg)

    out = Path(out_root) / "rice_fresh" / f"aoi{aoi_id}"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"aoi{aoi_id}_step1"
    with rasterio.open(f"{stem}_cover.tif", "w", **dict(profile, dtype="uint8", nodata=255, count=1)) as ds:
        ds.write(cover.reshape(shape)[None])
    with rasterio.open(f"{stem}_vh_low.tif", "w", **dict(profile, dtype="float32", nodata=np.nan, count=1)) as ds:
        ds.write(np.where(inside, vh_low, np.nan).reshape(shape).astype("float32")[None])
    entries = "\n".join(f'        <paletteEntry value="{k}" color="{c}" alpha="255" label="{k} {CLASSES[k]}"/>'
                        for k, c in COLOURS.items())
    Path(f"{stem}_cover.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" nodataColor="">
      <colorPalette>
{entries}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
""")
    rows = []
    for k in (0, 1, 2):
        m = cover == k
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows.append({"class": k, "name": CLASSES[k], "acres": round(acres(int(m.sum())), 1),
                         "sustained_low_median": round(float(np.nanmedian(low[m])), 3) if m.any() else np.nan,
                         "vh_low_median_db": round(float(np.nanmedian(vh_low[m])), 1) if m.any() else np.nan})
    table = pd.DataFrame(rows).assign(ndvi_split=round(split, 3), radar_split_db=round(float(radar_split), 1))
    table.attrs["radar_moved_to_crop_acres"] = round(acres(optical_trees - int((cover == 2).sum())), 1)
    table.to_csv(f"{stem}.csv", index=False)
    preview(cover.reshape(shape), low, vh_low, veg, split, Path(f"{stem}.png"), f"aoi{aoi_id}: step 1, since {START}")
    return table


def preview(cover, low, vh_low, veg, split, path: Path, title: str) -> None:
    """Map of the three classes, histogram of the sustained low with the split, and VH low per class."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lut = np.ones((256, 3))
    for k, c in COLOURS.items():
        lut[k] = matplotlib.colors.to_rgb(c)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.2), gridspec_kw={"width_ratios": [1, 0.9, 0.9]})
    axes[0].imshow(lut[cover], interpolation="nearest")
    axes[0].axis("off")
    axes[0].legend(handles=[Patch(color=COLOURS[k], label=CLASSES[k]) for k in COLOURS], loc="lower left",
                   fontsize=8, frameon=True)
    axes[0].set_title("September vegetation, split by its history since 1 May")
    v = low[veg & np.isfinite(low)]
    axes[1].hist(v, bins=120, color="#6d8f4e")
    axes[1].axvline(split, color="#b45f06", lw=2, label=f"split from this AOI: NDVI {split:.2f}")
    axes[1].set_xlabel("lowest NDVI held for two windows since 1 May")
    axes[1].set_ylabel("vegetated pixels")
    axes[1].legend(frameon=False)
    c = cover.ravel()
    for k in (1, 2):
        x = vh_low[(c == k) & np.isfinite(vh_low)]
        axes[2].hist(x, bins=80, alpha=0.6, color=COLOURS[k], label=CLASSES[k])
    axes[2].set_xlabel("second-darkest VH pass since 1 May (dB)")
    axes[2].set_title("radar: a 'never bare' pixel whose VH went dark is crop ground")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def sar_check(aoi_id: int, out_root: str = "processed/_batch/s2_2026") -> pd.DataFrame:
    """How far the radar ALONE reproduces step 1: Otsu's split of the second-darkest VH pass (``vh_low``) among the
    September vegetation, against the optical classes (1 went bare / 2 never bare). Why (user, 30 Sep): can step 1 run on
    radar only? Returns the agreement table (acres) and the split in dB (attribute ``split_db``)."""
    import rasterio

    stem = Path(out_root) / "rice_fresh" / f"aoi{aoi_id}" / f"aoi{aoi_id}_step1"
    with rasterio.open(f"{stem}_cover.tif") as ds:
        cover = ds.read(1).ravel()
    with rasterio.open(f"{stem}_vh_low.tif") as ds:
        vh = ds.read(1).ravel()
    veg = np.isin(cover, (1, 2)) & np.isfinite(vh)
    split = otsu(vh[veg])
    radar = np.where(vh > split, "radar: never dark", "radar: went dark")
    optical = np.where(cover == 2, "optical: never bare", "optical: went bare")
    t = pd.crosstab(optical[veg], radar[veg]).map(lambda v: round(acres(int(v)), 1))
    t.attrs["split_db"] = round(split, 2)
    t.attrs["agreement"] = round(float(((cover[veg] == 2) == (vh[veg] > split)).mean()), 3)
    # second look: the season's swing instead of its darkness (dry bare soil is not as dark as water)
    from . import radar_water as rw

    series = rw.read_series(aoi_id)
    sw = swing(series)
    ok = veg & np.isfinite(sw)
    s_split = otsu(sw[ok])
    radar2 = np.where(sw < s_split, "radar: small swing", "radar: big swing")
    t2 = pd.crosstab(optical[ok], radar2[ok]).map(lambda v: round(acres(int(v)), 1))
    t.attrs["swing"] = t2
    t.attrs["swing_split_db"] = round(s_split, 2)
    t.attrs["swing_agreement"] = round(float(((cover[ok] == 2) == (sw[ok] < s_split)).mean()), 3)
    # third look: the lowest cross-polarisation ratio (bare soil has a low VH - VV even when dry)
    rl = ratio_low(series)
    ok = veg & np.isfinite(rl)
    r_split = otsu(rl[ok])
    radar3 = np.where(rl > r_split, "radar: ratio stayed high", "radar: ratio went low")
    t.attrs["ratio"] = pd.crosstab(optical[ok], radar3[ok]).map(lambda v: round(acres(int(v)), 1))
    t.attrs["ratio_split_db"] = round(r_split, 2)
    t.attrs["ratio_agreement"] = round(float(((cover[ok] == 2) == (rl[ok] > r_split)).mean()), 3)
    t.attrs["ratio_median_by_optical"] = {k: round(float(np.nanmedian(rl[ok & (cover == c)])), 2)
                                          for k, c in (("went bare", 1), ("never bare", 2))}
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--series-root", default="processed/_batch/s2_2026_hyb40m1late")
    p.add_argument("--out-root", default="processed/_batch/s2_2026")
    p.add_argument("--sar-check", action="store_true", help="only compare step 1 with a radar-only split (after a run)")
    args = p.parse_args(argv)
    if args.sar_check:
        t = sar_check(args.aoi, args.out_root)
        print(t.to_string(), f"\nradar split VH {t.attrs['split_db']} dB, agreement {t.attrs['agreement']}\n")
        print(t.attrs["swing"].to_string(),
              f"\nswing split {t.attrs['swing_split_db']} dB, agreement {t.attrs['swing_agreement']}\n")
        print(t.attrs["ratio"].to_string(), f"\nratio split {t.attrs['ratio_split_db']} dB, agreement "
              f"{t.attrs['ratio_agreement']}, medians {t.attrs['ratio_median_by_optical']}")
        return 0
    t = run(args.aoi, args.series_root, out_root=args.out_root)
    print(t.to_string(index=False))
    print(f"radar went dark under 'never bare': {t.attrs['radar_moved_to_crop_acres']} ac moved to crop ground")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
