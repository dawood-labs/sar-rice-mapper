"""Turn the measured cycles of one AOI into a two-class rice map, from Sentinel-2 only.

Why
---
``optical_phenology`` measures a cycle for every pixel, including pixels that carry no crop; it
deliberately forms no verdict. This module forms the verdict, and it is kept separate so the
measurement stays reusable and the decision stays visible in one place.

The rule, and where each part of it comes from
----------------------------------------------
First the AOI's **own season** is measured, then each pixel is judged against it.

*The season*: take every strong, fully observed cycle in the AOI and read the median of their peak
dates. That date is the AOI's canopy peak — 8 August in the first AOI, but somewhere else entirely
in an AOI a few hundred kilometres north. A cycle counts as being in season when its peak falls
within ``tolerance_days`` of it. This is the point a field agronomist makes plainly: within one area
crops do not differ by months. It is also why the *season* is measured rather than written down —
a fixed "peak in July or August" would be right in the delta and wrong in the north, and the 132
AOIs span the country.

*The pixel*: a pixel is **rice** when it has a cycle that satisfies all of:

1. **In season** — its peak is inside the window above. A pixel's *chosen* cycle is the strongest
   one in season, not the strongest overall, because a field can carry a bigger flush of weeds or a
   second crop outside the season and that must not disqualify the rice crop underneath it.
2. **Fully observed** (``complete``) — a cycle running into the start or end of the series has a
   length that is a floor, not a measurement.
3. **A real canopy** — how far the crop climbed, at or above a cut taken from *this AOI's own*
   histogram by Otsu's method. The climb is the larger of two measures: from the cycle's own trough,
   and from the field's pre-season bare level. Two failures forced that: measuring peak-to-shoulder
   made a plainly-rice double-cropped pixel read 0.31 against 0.65 of real growth, because the next
   crop started before this one had come back down; and when the trough itself falls inside a cloudy
   fortnight the interpolated trough is already half way up the rise, which read 0.22 against 0.62.
4. **Long enough to be a crop** — trough to harvest at least ``min_cycle_days``. Nothing that
   greens and is cut inside two months is rice.
5. **Short enough to be one crop** — trough to harvest no longer than ``max_cycle_days``. Beyond
   that the "cycle" is two crops the segmenter failed to separate, or permanent vegetation.
6. **Wet when it started** — ``wet_at_sowing``: at the cycle's trough the NDVI sits *below* the
   field's own pre-season bare level while LSWI moves the *other way*. Drying bare soil takes both
   down together, so it is the opposite directions that mean standing water. This is the paddy test
   that the per-date exports could not answer, because Sentinel-2 never saw the transplanting window
   through the monsoon cloud; the 5-day s2cloudless composites do. Where the baseline stretch falls
   outside the observed period the test cannot be run, and a pixel is **not** failed for that.

Sowing and harvest *dates* are still not conditions in themselves; only the distance from the AOI's
own measured peak is.

The sieve
---------
A rice field is not one pixel. The second output removes any group of pixels smaller than a given
area, replacing it with its largest neighbour, so specks below the minimum mapping unit disappear.
The area is given in acres and converted to a pixel count from the grid's own resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ACRES_PER_M2 = 1.0 / 4046.8564224
#: How far a pixel's peak may sit from the AOI's own median peak and still count as the same season.
#: Generous on purpose: transplanting inside one area spreads over a few weeks and varieties differ,
#: but not by a season. At this value one AOI kept 98% of its strong cycles.
TOLERANCE_DAYS = 45
#: Shortest trough-to-harvest a crop may take. Measured from the cycle's own starting trough rather
#: than from the half-amplitude width, because with light smoothing the *fall* can be cut short by a
#: shallow wiggle: one pixel that is plainly rice measured 30 days of canopy width against 75 days
#: from its trough. The floor is the agronomic one — nothing that grows and is cut inside two months
#: is a rice crop.
MIN_CYCLE_DAYS = 60
#: Longest trough-to-harvest a single crop may take. Beyond it the cycle is two crops merged.
MAX_CYCLE_DAYS = 165


def otsu(values, bins: int = 256) -> float:
    """The cut that best splits a distribution into two groups, chosen by the data.

    Otsu's method maximises the variance *between* the two groups either side of the cut. It needs
    no threshold of its own, which is why it is used here rather than a number picked by eye.
    """
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0 or np.allclose(v, v[0]):
        return float("nan")
    counts, edges = np.histogram(v, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(counts)
    total = weight[-1]
    if total == 0:
        return float("nan")
    w0 = weight[:-1] / total
    w1 = 1.0 - w0
    csum = np.cumsum(counts * centres)
    mean0 = np.divide(csum[:-1], weight[:-1], out=np.zeros(len(counts) - 1), where=weight[:-1] > 0)
    mean1 = np.divide(csum[-1] - csum[:-1], total - weight[:-1],
                      out=np.zeros(len(counts) - 1), where=(total - weight[:-1]) > 0)
    between = w0 * w1 * (mean0 - mean1) ** 2
    ok = np.isfinite(between)
    if not ok.any():
        return float("nan")
    return float(centres[:-1][np.nanargmax(np.where(ok, between, -np.inf))])


def climb(cycles: pd.DataFrame):
    """How far a cycle's canopy rose: the larger of its climb from its own trough and from the
    field's pre-season bare level. See point 3 of the module docstring for why both are needed."""
    return np.fmax(cycles["growth_amplitude"], cycles.get("growth_from_baseline", np.nan))


def season_window(cycles: pd.DataFrame, amplitude_cut: float | None = None,
                  tolerance_days: float = TOLERANCE_DAYS) -> tuple:
    """The AOI's own canopy season, and the cut used to decide which cycles defined it.

    Returns ``(centre, start, end, amplitude_cut)``. The centre is the median peak date of the
    strong, fully observed cycles; everything within ``tolerance_days`` of it is in season.
    """
    strong = cycles.dropna(subset=["peak_date"]).copy()
    strong["growth"] = climb(strong)
    strong = strong.dropna(subset=["growth"])
    cut = otsu(strong["growth"]) if amplitude_cut is None else float(amplitude_cut)
    strong = strong[(strong["growth"] >= cut) & strong["complete"].fillna(False)]
    if strong.empty:
        raise ValueError("no strong, fully observed cycle in this AOI to measure a season from")
    centre = strong["peak_date"].median()
    span = pd.Timedelta(days=tolerance_days)
    return centre, centre - span, centre + span, cut


def pick_season_cycle(cycles: pd.DataFrame, start, end) -> pd.DataFrame:
    """One row per pixel: its strongest cycle whose peak falls inside the season, if it has one."""
    inside = cycles.dropna(subset=["peak_date"]).copy()
    inside["growth"] = climb(inside)
    inside = inside.dropna(subset=["growth"])
    inside = inside[inside["peak_date"].between(start, end)]
    if inside.empty:
        return inside
    chosen = (inside.sort_values("growth", ascending=False)
              .groupby("pid", as_index=False).head(1).sort_values("pid").reset_index(drop=True))
    chosen.attrs.update(cycles.attrs)
    return chosen


def classify(cycles: pd.DataFrame, tolerance_days: float = TOLERANCE_DAYS,
             min_cycle_days: float = MIN_CYCLE_DAYS, max_cycle_days: float = MAX_CYCLE_DAYS,
             amplitude_cut: float | None = None):
    """Rice / not rice per pixel. Returns ``(chosen, rice, used)``; see the module docstring."""
    centre, start, end, cut = season_window(cycles, amplitude_cut, tolerance_days)
    chosen = pick_season_cycle(cycles, start, end)
    complete = chosen["complete"].fillna(False).astype(bool)
    strong = chosen["growth"] >= cut
    long_enough = chosen["start_to_harvest_days"] >= min_cycle_days
    short_enough = chosen["start_to_harvest_days"] <= max_cycle_days
    testable = chosen["prewet_ndvi_drop"].notna() & chosen["prewet_lswi_rise"].notna()
    wet = (~testable) | chosen["wet_at_sowing"].fillna(False).astype(bool)
    rice = complete & strong & long_enough & short_enough & wet
    used = {"season_centre": centre.strftime("%d %b %Y"),
            "season_window": f"{start:%d %b} .. {end:%d %b}",
            "tolerance_days": tolerance_days, "growth_amplitude_cut": round(cut, 4),
            "min_cycle_days": min_cycle_days, "max_cycle_days": max_cycle_days,
            "pixels_with_a_cycle": int(cycles["pid"].nunique()),
            "pixels_with_an_in_season_cycle": int(len(chosen)),
            "failed_complete": int((~complete).sum()),
            "failed_amplitude": int((complete & ~strong).sum()),
            "failed_too_short": int((complete & strong & ~long_enough).sum()),
            "failed_max_cycle": int((complete & strong & long_enough & ~short_enough).sum()),
            "failed_dry_at_sowing": int((complete & strong & long_enough & short_enough & ~wet).sum()),
            "wetness_not_testable": int((~testable).sum())}
    return chosen, rice, used


def acres(n_pixels: int, res_m: float = 10.0) -> float:
    """Pixel count to acres; areas in this project are always acres."""
    return float(n_pixels) * res_m * res_m * ACRES_PER_M2


def min_pixels_for(acres_value: float, res_m: float = 10.0) -> int:
    """Smallest pixel count that still covers ``acres_value`` acres."""
    return int(np.ceil(acres_value / (res_m * res_m * ACRES_PER_M2)))


def write_class_raster(chosen: pd.DataFrame, rice: pd.Series, out_path, shape=None, grid=None,
                       nodata: int = 255):
    """Write the two-class map: 1 rice, 0 not rice, ``nodata`` where no cycle could be measured."""
    import rasterio
    from rasterio.transform import from_origin

    shape = shape or chosen.attrs["shape"]
    grid = grid or chosen.attrs["grid"]
    band = np.full(shape[0] * shape[1], nodata, dtype="uint8")
    band[chosen["pid"].to_numpy()] = rice.to_numpy().astype("uint8")
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint8",
                   crs=grid["crs"], nodata=nodata, compress="deflate", tiled=True,
                   transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(band.reshape(shape), 1)
        ds.update_tags(1, CLASSES="0=not rice, 1=rice, 255=no cycle measured")
    return out_path


def sieve_raster(in_path, out_path, min_acres: float = 0.15, connectivity: int = 8,
                 nodata: int = 255):
    """Copy a class raster with every patch smaller than ``min_acres`` merged into its neighbour."""
    import rasterio
    from rasterio.features import sieve

    with rasterio.open(in_path) as ds:
        band = ds.read(1)
        profile = ds.profile.copy()
        res = abs(ds.transform.a)
    size = min_pixels_for(min_acres, res)
    valid = band != nodata
    cleaned = sieve(np.where(valid, band, 0).astype("uint8"), size=size,
                    mask=valid, connectivity=connectivity)
    out = np.where(valid, cleaned, nodata).astype("uint8")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(out, 1)
        ds.update_tags(1, CLASSES="0=not rice, 1=rice, 255=no cycle measured",
                       SIEVE_MIN_ACRES=str(min_acres), SIEVE_MIN_PIXELS=str(size))
    return out_path, size


def summarise(aoi: str, chosen: pd.DataFrame, rice, used: dict, shape, res_m: float = 10.0) -> dict:
    """One row describing an AOI's map: the areas, and the rule that produced them."""
    total = int(shape[0] * shape[1])
    n_rice = int(np.asarray(rice).sum())
    decided = int(len(chosen))
    return {"aoi": aoi, "total_acres": round(acres(total, res_m), 1),
            "rice_acres": round(acres(n_rice, res_m), 1),
            "not_rice_acres": round(acres(decided - n_rice, res_m), 1),
            "undecided_acres": round(acres(total - decided, res_m), 1),
            "rice_pct_of_decided": round(100 * n_rice / decided, 1) if decided else float("nan"),
            "season_centre": used["season_centre"], "season_window": used["season_window"],
            "growth_cut": used["growth_amplitude_cut"],
            "failed_amplitude": used["failed_amplitude"],
            "failed_too_short": used["failed_too_short"],
            "failed_max_cycle": used["failed_max_cycle"],
            "failed_dry_at_sowing": used["failed_dry_at_sowing"],
            "wetness_not_testable": used["wetness_not_testable"]}


def run_aoi(aoi_id: int, out_root="processed/_batch/optical_v3", sieve_acres=(0.5,),
            folder: str = "s2_windows5d", **cycle_kw) -> dict:
    """Measure one AOI's cycles, classify it and write its maps. Returns the summary row.

    Every AOI is treated on its own terms: its season window, its Otsu cut and its sieve are all
    derived from its own data, so nothing is carried over from whichever AOI happened to be first.
    """
    from . import optical_phenology as op

    cycles = op.aoi_cycles(aoi_id, folder=folder, **cycle_kw)
    chosen, rice, used = classify(cycles)
    shape, grid = cycles.attrs["shape"], cycles.attrs["grid"]
    aoi = cycles.attrs["aoi"]
    folder_out = Path(out_root) / aoi
    base = write_class_raster(chosen, rice, folder_out / f"{aoi}_rice_optical.tif",
                              shape=shape, grid=grid)
    sieved = {}
    for area in sieve_acres:
        tag = f"{area:g}".replace(".", "p")
        path, size = sieve_raster(base, folder_out / f"{aoi}_rice_optical_sieved_{tag}ac.tif", area)
        sieved[area] = (path, size)
    row = summarise(aoi, chosen, rice, used, shape, float(grid["res"]))
    for area, (path, size) in sieved.items():
        import rasterio

        with rasterio.open(path) as ds:
            band = ds.read(1)
        row[f"rice_acres_sieved_{area:g}ac"] = round(acres(int((band == 1).sum()), float(grid["res"])), 1)
        row[f"sieve_{area:g}ac_pixels"] = size
    cycles.to_pickle(folder_out / f"{aoi}_cycles.pkl")
    return row


def run_batch(aoi_ids, out_root="processed/_batch/optical_v3", log=print, **kw) -> pd.DataFrame:
    """Run :func:`run_aoi` over several AOIs, carrying on past one that fails."""
    rows = []
    for aoi_id in aoi_ids:
        try:
            row = run_aoi(aoi_id, out_root, **kw)
            log(f"aoi{aoi_id}: {row['rice_acres']:.0f} of {row['total_acres']:.0f} acres rice "
                f"({row['rice_pct_of_decided']:.0f}% of decided), season {row['season_window']}")
        except Exception as error:  # noqa: BLE001 - one bad AOI must not stop the batch
            log(f"aoi{aoi_id}: FAILED - {type(error).__name__}: {error}")
            row = {"aoi": f"aoi{aoi_id}", "error": f"{type(error).__name__}: {error}"}
        rows.append(row)
    frame = pd.DataFrame(rows)
    Path(out_root).mkdir(parents=True, exist_ok=True)
    frame.to_csv(Path(out_root) / "aoi_rice_acres.csv", index=False)
    return frame
