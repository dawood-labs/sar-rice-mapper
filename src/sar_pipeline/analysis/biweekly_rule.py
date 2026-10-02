"""A colleague's simple radar rule, re-implemented on our stacks so it can be scored the same way.

Why
---
A Google Earth Engine script (September 2026) maps "standing rice" from Sentinel-1 alone with four
thresholds on 15-day median composites of one descending orbit, followed by a morphological
closing and a majority filter. Its maps look right by eye in a rice-dominated delta. Looking right
is not a score, so the rule is rebuilt here, unchanged, on the same GRD passes our own methods use
(this module reads the descending track only, as the script does), and run through the same
reference sets: surveyed plots, optical-derived negatives, and the validated map.

The rule, as written in the script
----------------------------------
For ten half-month periods from 1 May to 30 September: the median of the descending passes in the
period, VH and VV in dB, spatially smoothed (a focal median of 10 m radius; here a 3 x 3 mean in
power, the nearest equivalent on a 10 m grid). Then, per pixel:

* ``flooded``      = minimum VV over the ten periods < -12 dB
* ``vegetated_now``= VH of the second September period (or the first where the second is empty) > -18 dB
* ``seasonal``     = (max VH - min VH over the periods) > 5 dB
* ``not_harvested``= (max VH - VH now) < 3.5 dB
* rice = all four; then a closing (focal max then focal min, 30 m radius) and a majority filter
  (15 m radius).

Where it differs from ours, for the record: there is no requirement that the dark moment comes
*before* the green-up (a field harvested in June and weedy in September passes), no test of how
dark the water was in VH nor of how long it lasted, no check that the ground was bare and smooth
before the water, and one orbit only. -12 dB in VV is a level that dry bare soil, wet fallow and
any rain-soaked field also reach, so the first test is "was ever dark-ish", not "was flooded".
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr
from . import pixel_report as pr
from . import sar_curve
from . import seasonal_stats as ss

SRC = "processed/_batch/s2_2026"
OUT = f"{SRC}/biweekly_rule"
SEASON_KEY = "monsoon2026"
PERIODS = [("05-01", "05-15"), ("05-16", "05-31"), ("06-01", "06-15"), ("06-16", "06-30"), ("07-01", "07-15"),
           ("07-16", "07-31"), ("08-01", "08-15"), ("08-16", "08-31"), ("09-01", "09-15"), ("09-16", "09-30")]
VV_MIN_FLOOD = -12.0
VH_NOW_MIN = -18.0
VH_RANGE_MIN = 5.0
DROP_FROM_PEAK_MAX = 3.5
CLOSING_RADIUS_PX = 3      # 30 m
MAJORITY_RADIUS_PX = 1.5   # 15 m -> the 3 x 3 neighbourhood


def period_composites(aoi_id: int, year: int = 2026, window: int = 3, orbit: str = "DSC") -> dict:
    """``VH`` / ``VV`` (10, rows, cols) half-month medians in dB of the chosen orbit; NaN where the
    period has no pass. Also the pass count per period."""
    loc = pr.locate(aoi_id, 0, season_key=SEASON_KEY)
    tracks = [t["track_id"] for t in loc["cfg"]["s1"]["tracks"] if t["track_id"].endswith(orbit)]
    if not tracks:
        raise ValueError(f"aoi{aoi_id}: no {orbit} track")
    dates, cubes = sar_curve.read_track(loc, tracks[0], window)
    day = dates.to_numpy().astype("datetime64[D]")
    out = {p: np.full((len(PERIODS),) + cubes[p].shape[1:], np.nan, dtype="float32") for p in ("VH", "VV")}
    counts = []
    for k, (a, b) in enumerate(PERIODS):
        sel = (day >= np.datetime64(f"{year}-{a}")) & (day <= np.datetime64(f"{year}-{b}"))
        counts.append(int(sel.sum()))
        if sel.any():
            for p in ("VH", "VV"):
                out[p][k] = np.nanmedian(cubes[p][sel], axis=0)
    out["passes_per_period"] = counts
    out["shape"] = cubes["VH"].shape[1:]
    return out


def _disk(radius: float) -> np.ndarray:
    r = int(np.ceil(radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius * radius


def closing_then_majority(mask: np.ndarray) -> np.ndarray:
    """The script's two kernels: closing with a 30 m disk, then a majority vote in a 15 m disk."""
    from scipy.ndimage import binary_closing, uniform_filter

    closed = binary_closing(mask, structure=_disk(CLOSING_RADIUS_PX))
    k = _disk(MAJORITY_RADIUS_PX).astype(float)
    votes = uniform_filter(closed.astype(float), size=k.shape[0], mode="nearest")
    return votes > 0.5


def biweekly_rule(aoi_id: int, smooth: bool = True) -> dict:
    """The rule on one AOI: ``raw`` (before the kernels), ``rice`` (after), the four tests and the
    summary layers, all on the AOI grid."""
    import warnings

    c = period_composites(aoi_id)
    vh, vv = c["VH"], c["VV"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vh_max, vh_min, vv_min = np.nanmax(vh, axis=0), np.nanmin(vh, axis=0), np.nanmin(vv, axis=0)
        vh_now = np.where(np.isfinite(vh[9]), vh[9], vh[8])
    with np.errstate(invalid="ignore"):
        flooded = vv_min < VV_MIN_FLOOD
        vegetated_now = vh_now > VH_NOW_MIN
        seasonal = (vh_max - vh_min) > VH_RANGE_MIN
        not_harvested = (vh_max - vh_now) < DROP_FROM_PEAK_MAX
    raw = flooded & vegetated_now & seasonal & not_harvested
    rice = closing_then_majority(raw) if smooth else raw
    valid = np.isfinite(vh_max)
    return {"raw": raw & valid, "rice": rice & valid, "flooded": flooded, "vegetated_now": vegetated_now,
            "seasonal": seasonal, "not_harvested": not_harvested, "vh_range": vh_max - vh_min, "vv_min": vv_min,
            "vh_now": vh_now, "valid": valid, "passes_per_period": c["passes_per_period"], "shape": c["shape"]}


def write_map(aoi_id: int, result: dict, out_dir=OUT) -> Path:
    import rasterio

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    src = Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_{SEASON_KEY}_final.tif"
    with rasterio.open(src) as ds:
        profile = ds.profile.copy()
    arr = np.where(result["valid"], result["rice"].astype("uint8"), 255).astype("uint8")
    profile.update(dtype="uint8", nodata=255, count=1, compress="deflate")
    path = Path(out_dir) / f"aoi{aoi_id}_biweekly_rule.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def compare_with_teacher(aoi_id: int, result: dict) -> pd.DataFrame:
    """Acres per (teacher class, rule rice yes/no) inside the AOI."""
    import rasterio

    from . import ndvi_5day as nd

    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_{SEASON_KEY}_final.tif") as ds:
        teacher = ds.read(1).ravel()
    inside = nd.inside_aoi(aoi_id) & result["valid"].ravel()
    t = pd.crosstab(teacher[inside], result["rice"].ravel()[inside]) * mr.acres(1)
    t.columns = [f"rule_rice_{bool(c)}" for c in t.columns]
    return t.round(0)


def score_references(aoi_ids, plots, smooth: bool = True) -> pd.DataFrame:
    """Share of each reference set (plots, negatives) the rule calls rice, per region."""
    from . import validation as va

    rows = []
    for a in aoi_ids:
        res = biweekly_rule(a, smooth)
        refs = va.reference_sets(a, plots)
        hit = res["rice"].ravel()[refs["pixel"].to_numpy()]
        for (s, region), g in refs.assign(hit=hit).groupby(["set", "region"]):
            rows.append({"set": s, "region": region, "pixels": len(g), "hit": int(g["hit"].sum())})
    t = pd.DataFrame(rows).groupby(["set", "region"]).sum().reset_index()
    t["rice_pct"] = np.round(100 * t["hit"] / t["pixels"], 1)
    return t.drop(columns="hit")


def panel(aoi_id: int, sar_only_dir=f"{SRC}/sar_only", out_dir=OUT) -> Path:
    """One picture: the latest clear Sentinel-2 view, the validated map, this rule, and our
    radar-only model side by side over the whole AOI."""
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import ListedColormap

    from ..optical_export import band_index
    from ..review import _stretch, latest_clear
    from . import ndvi_5day as nd

    res = biweekly_rule(aoi_id)
    with rasterio.open(Path(SRC) / f"aoi{aoi_id}" / f"aoi{aoi_id}_{SEASON_KEY}_final.tif") as ds:
        teacher = ds.read(1).astype(float)
    inside = nd.inside_aoi(aoi_id).reshape(teacher.shape)
    path, date, _ = latest_clear(aoi_id)
    with rasterio.open(path) as ds:
        rgb = np.dstack([_stretch(ds.read(band_index(ds, b)).astype("float32")) for b in ("B4", "B3", "B2")])
    colours = {0: "#d9d9d9", 1: "#1f77b4", 2: "#8ad5c8", 3: "#f5a028", 4: "#96643c", 5: "#a0c060", 6: "#e070c0",
               7: "#3c78c8", 8: "#c8aa78", 9: "#009696"}
    cmap = ListedColormap([colours[k] for k in range(len(colours))])
    panels = [("Sentinel-2 " + str(date.date()), None), ("validated map (optical + radar)", np.where(inside, teacher, np.nan)),
              ("15-day radar rule (colleague)", np.where(inside, np.where(res["rice"], 1.0, 0.0), np.nan))]
    so = Path(sar_only_dir) / f"aoi{aoi_id}_sar_only.tif"
    if so.exists():
        with rasterio.open(so) as ds:
            m = ds.read(1).astype(float)
        m[m == 255] = np.nan
        panels.append(("our radar-only model", np.where(inside, m, np.nan)))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 5.5))
    for ax, (title, arr) in zip(axes, panels):
        ax.imshow(rgb)
        if arr is not None:
            ax.imshow(arr, cmap=cmap, vmin=-0.5, vmax=len(colours) - 0.5, alpha=0.75, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"aoi{aoi_id}: blue = rice, pink = young rice, orange = rice-like no water, grey = not rice", fontsize=9)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    f = Path(out_dir) / f"aoi{aoi_id}_panel.png"
    fig.savefig(f, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return f


def main(argv=None) -> int:
    import argparse

    from .compare_runs import PLOT_AOIS, load_plots

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.biweekly_rule", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["map", "score", "panel"])
    p.add_argument("--ids", nargs="*", type=int, default=None)
    p.add_argument("--no-smooth", action="store_true", help="skip the closing and majority kernels")
    args = p.parse_args(argv)
    if args.step == "map":
        for a in args.ids:
            res = biweekly_rule(a, not args.no_smooth)
            path = write_map(a, res)
            print(f"aoi{a}: passes per period {res['passes_per_period']}; rule rice {mr.acres(int(res['rice'].sum())):.0f} ac "
                  f"(before kernels {mr.acres(int(res['raw'].sum())):.0f}) -> {path}")
            print(compare_with_teacher(a, res).to_string())
        return 0
    if args.step == "panel":
        for a in args.ids:
            print(panel(a))
        return 0
    ids = args.ids or list(PLOT_AOIS)
    print(score_references(ids, load_plots(), not args.no_smooth).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
