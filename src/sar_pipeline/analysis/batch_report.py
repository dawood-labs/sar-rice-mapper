"""What the standing-rice map found in each processed AOI, and the evidence behind each class.

Why
---
The acres table says how much of each class the rule found; it does not say *why* a pixel landed
where it did. After the first batch a large share of the rice-like area came out as "standing,
water unconfirmed" (class 3), concentrated in a few AOIs. Following the project's principle that
rice is always transplanted into water, the first suspect is the detection, not the crop. This
module lays out, per AOI and per class, the evidence the rule used:

* **when** - median trough, climb and the end of the bare period before the climb;
* **how well observed** - days to the nearest real Sentinel-2 observation at the trough;
* **radar** - VV/VH dip at the trough, the share of class-3 pixels that missed the 3 dB bar only
  narrowly (2-3 dB) or showed no sign at all (< 1 dB);
* **trough on the edge** - the share of pixels whose trough sits on the first window of the
  110-day search. There the field was already bare when the search began, the argmin may be weeks
  before the real transplanting, and the radar looks for the water on the wrong date;
* **dip at the end of the bare period** - the same radar dip measured at the last bare window
  before the climb. If class-3 pixels turn wet there, the water was present and the rule's date
  was the problem; if they stay dry, the crop is more likely not rice;
* **drop anywhere before the climb** - the season's median backscatter minus the lowest value
  between 10 days before the trough and the climb, best of VV/VH and tracks. Transplanting must
  fall inside that span, so this test does not depend on picking the right date at all.

It also splits the "not rice" class into what it is on the map date or over the season: open water
now, never a canopy (bare, built-up or flooded all season), evergreen (a canopy all season: trees,
orchards), and the rest (a seasonal canopy that failed some condition of the rule).

Nothing here changes the map. A rule change needs ``analysis/validation`` re-run on the field plots.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import monsoon_rule as mr

SERIES_ROOT = "processed/_batch/s2_2026"
OUT = SERIES_ROOT + "/report"
LOW_LEVEL = mr.TROUGH_MAX + 0.05      # "bare" as used by the rule's 40-day criterion
SAMPLE = 20000


def bare_end(ndvi_sub, trough, first, level: float = LOW_LEVEL) -> np.ndarray:
    """Index (into the search windows) of the last bare window between the trough and the climb.

    ``ndvi_sub`` (windows, pixels) is the fitted NDVI over the search windows, ``trough`` the
    trough index and ``first`` the climb index (-1: no climb, then the search runs to the end).
    Falls back to the trough when no later window is bare.
    """
    n = ndvi_sub.shape[0]
    step = np.arange(n)[:, None]
    stop = np.where(first >= 0, first, n)
    mask = (step >= trough[None, :]) & (step < stop[None, :]) & (ndvi_sub <= level)
    last = np.where(mask, step, -1).max(axis=0)
    return np.where(last >= 0, last, trough)


def drop_before_climb(dates, cube, start, stop) -> np.ndarray:
    """Season median minus the minimum inside ``[start, stop]`` per pixel, in dB (NaN: no date inside).

    ``dates`` (n_dates), ``cube`` (n_dates, n_pixels) in dB, ``start``/``stop`` datetime64 per pixel.
    """
    import warnings

    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")[:, None]
    inside = (day >= np.asarray(start)[None, :]) & (day <= np.asarray(stop)[None, :])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nanmedian(cube, axis=0)
        low = np.nanmin(np.where(inside, cube, np.nan), axis=0)
    return level - low


def not_rice_parts(classes, ndvi, lswi, season_mask) -> dict:
    """Acres of class 0 split into ``water_now``, ``never_canopy``, ``evergreen`` and ``other``.

    ``ndvi``/``lswi`` are fitted (windows, pixels); ``season_mask`` selects the season's windows.
    The parts are exclusive and checked in that order: open water on the last window (LSWI above
    NDVI, NDVI below 0.2); season maximum below the canopy floor; season minimum at or above it.
    """
    import warnings

    c0 = np.asarray(classes) == 0
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hi = np.nanmax(ndvi[season_mask], axis=0)
        lo = np.nanmin(ndvi[season_mask], axis=0)
    water = c0 & (lswi[-1] > ndvi[-1]) & (ndvi[-1] < 0.2)
    never = c0 & ~water & (hi < mr.CANOPY_MIN)
    ever = c0 & ~water & ~never & (lo >= mr.CANOPY_MIN)
    other = c0 & ~water & ~never & ~ever
    return {name: round(mr.acres(int(m.sum())), 1)
            for name, m in (("water_now", water), ("never_canopy", never), ("evergreen", ever), ("other", other))}


def aoi_evidence(aoi_id: int, out_dir=OUT, sample: int = SAMPLE, seed: int = 0) -> dict:
    """Summary row for one AOI; also writes a pixel sample of classes 1 and 3 to ``<out_dir>/aoi<N>.parquet``."""
    import rasterio

    from . import ndvi_5day as nd
    from . import radar_water

    d, ev, radar = mr.aoi_events(aoi_id)
    aoi = d["loc"]["aoi"]
    with rasterio.open(Path(SERIES_ROOT) / aoi / f"{aoi}_monsoon2026.tif") as ds:
        classes = ds.read(1).ravel()
    windows = pd.DatetimeIndex(d["windows"])
    earliest = max(pd.Timestamp(mr.SEASON[0]), windows[-1] - pd.Timedelta(days=mr.LOOKBACK_DAYS))
    idx = np.flatnonzero((windows >= earliest) & (windows < mr.SEASON[1]))
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    gaps = d["gapdays5d"].reshape(d["gapdays5d"].shape[0], -1)
    sub = ndvi[idx]
    tdate = ev["trough_date"].to_numpy().astype("datetime64[D]")
    t_i = np.searchsorted(windows[idx].to_numpy().astype("datetime64[D]"), tdate).clip(0, len(idx) - 1)
    cdate = pd.to_datetime(ev["climb_date"]).to_numpy().astype("datetime64[D]")
    c_i = np.where(np.isnat(cdate), -1,
                   np.searchsorted(windows[idx].to_numpy().astype("datetime64[D]"), cdate).clip(0, len(idx) - 1))
    b_i = bare_end(sub, t_i, c_i)
    ev["bare_end_date"] = windows[idx][b_i]
    cols = np.arange(ndvi.shape[1])
    ev["gap_at_trough"] = gaps[idx[t_i], cols]
    ev["trough_on_edge"] = t_i == 0
    ev["class"] = classes
    if radar:
        when = np.where(ev["valid"].to_numpy(), ev["bare_end_date"].to_numpy().astype("datetime64[D]"),
                        np.datetime64("NaT"))
        late = radar_water.pixel_dips(aoi_id, when, d["loc"]["grid"])
        ev["VV_dip_bare_end"] = late["VV_dip"].to_numpy()
        ev["VH_dip_bare_end"] = late["VH_dip"].to_numpy()
        ev["radar_wet_bare_end"] = late["radar_wet"].to_numpy()
        from . import pixel_report as pr
        from . import sar_curve

        loc = pr.locate(aoi_id, 0, season_key="monsoon2026")
        start = tdate - np.timedelta64(10, "D")
        stop = np.where(np.isnat(cdate), windows[-1].to_datetime64().astype("datetime64[D]"), cdate)
        best = np.full(ndvi.shape[1], np.nan)
        for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
            dates, cubes = sar_curve.read_track(loc, track)
            for pol in ("VV", "VH"):
                best = np.fmax(best, drop_before_climb(dates, cubes[pol].reshape(len(dates), -1), start, stop))
        ev["drop_before_climb"] = best
    nd.forget()

    row = {"aoi": aoi, "radar": radar}
    for code, name in mr.CLASSES.items():
        if code != 255:
            row[f"{name.replace(' ', '_')}_acres"] = round(mr.acres(int((classes == code).sum())), 1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    for name, value in not_rice_parts(classes, ndvi, lswi, np.asarray(windows >= mr.SEASON[0])).items():
        row[f"not_rice_{name}_acres"] = value
    for code, tag in ((1, "rice"), (3, "unconf")):
        e = ev[ev["class"] == code]
        row[f"{tag}_n"] = len(e)
        if not len(e):
            continue
        row[f"{tag}_trough"] = e["trough_date"].median().strftime("%d %b")
        row[f"{tag}_bare_end"] = e["bare_end_date"].median().strftime("%d %b")
        row[f"{tag}_climb"] = pd.to_datetime(e["climb_date"]).dropna().median().strftime("%d %b")
        row[f"{tag}_peak"] = round(float(e["peak_after"].median()), 2)
        row[f"{tag}_last"] = round(float(e["last_ndvi"].median()), 2)
        row[f"{tag}_trough_on_edge_pct"] = round(100 * float(e["trough_on_edge"].mean()), 1)
        row[f"{tag}_gap_at_trough_days"] = float(e["gap_at_trough"].median())
        row[f"{tag}_optical_wet_pct"] = round(100 * float(e["wet_at_trough"].mean()), 1)
        if radar:
            best = np.fmax(e["VV_dip"], e["VH_dip"])
            row[f"{tag}_VV_dip"] = round(float(e["VV_dip"].median()), 1)
            row[f"{tag}_VH_dip"] = round(float(e["VH_dip"].median()), 1)
            row[f"{tag}_dip_2to3_pct"] = round(100 * float(((best >= 2) & (best < 3)).mean()), 1)
            row[f"{tag}_dip_below1_pct"] = round(100 * float((best < 1).mean()), 1)
            row[f"{tag}_wet_at_bare_end_pct"] = round(100 * float(e["radar_wet_bare_end"].mean()), 1)
            row[f"{tag}_drop_before_climb"] = round(float(e["drop_before_climb"].median()), 1)
            row[f"{tag}_drop_3db_pct"] = round(100 * float((e["drop_before_climb"] >= 3).mean()), 1)
    keep = ev[np.isin(classes, (1, 3))]
    if len(keep) > sample:
        keep = keep.sample(sample, random_state=seed)
    keep = keep.assign(aoi=aoi)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    keep.drop(columns=[c for c in ("valid",) if c in keep]).to_parquet(Path(out_dir) / f"{aoi}.parquet")
    pd.DataFrame([row]).to_csv(Path(out_dir) / f"{aoi}_summary.csv", index=False)
    return row


def combine(out_dir=OUT, index: pd.DataFrame | None = None, regions: dict | None = None,
            recall_csv="processed/_batch/s2_2026/monsoon2026_rule_plot_recall_final.csv") -> pd.DataFrame:
    """All AOI summary rows, with location, geometry acres, region name and field-plot recall; north first."""
    rows = pd.concat([pd.read_csv(p) for p in sorted(Path(out_dir).glob("aoi*_summary.csv"))], ignore_index=True)
    if index is not None:
        ix = index.assign(aoi="aoi" + index["aoi"].astype(int).astype(str))[["aoi", "lon", "lat", "acres"]]
        rows = ix.merge(rows, on="aoi", how="right")
    if regions:
        rows.insert(1, "region", rows["aoi"].map(regions).fillna("no field plots"))
    if recall_csv and Path(recall_csv).exists():
        rec = pd.read_csv(recall_csv)
        if "rice_pct" in rec:
            rows = rows.merge(rec[["aoi", "rice_pct"]].rename(columns={"rice_pct": "plot_recall_pct"}),
                              on="aoi", how="left")
    rows["confirmed_pct"] = (100 * rows["rice_acres"] / (rows["rice_acres"] + rows["rice_unconfirmed_acres"])
                             ).round(1)
    return rows.sort_values("lat", ascending=False).reset_index(drop=True) if "lat" in rows else rows


def figure(table: pd.DataFrame, pixels: pd.DataFrame, out_path=None):
    """Three panels: AOIs on the map coloured by confirmed share; best radar dip at the trough vs at
    the end of the bare period for class 3; trough date against latitude per class."""
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, SURFACE

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), facecolor=SURFACE)
    for a in axes:
        a.set_facecolor(SURFACE)
        a.grid(True, alpha=.2)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    a1, a2, a3 = axes
    size = 20 + 0.15 * (table["rice_acres"] + table["rice_unconfirmed_acres"])
    sc = a1.scatter(table["lon"], table["lat"], c=table["confirmed_pct"], cmap="RdYlBu", vmin=0, vmax=100,
                    s=size, edgecolor=INK, linewidth=.5)
    for _, r in table.iterrows():
        a1.annotate(r["aoi"].replace("aoi", ""), (r["lon"], r["lat"]), fontsize=7, xytext=(4, 3),
                    textcoords="offset points", color=INK)
    fig.colorbar(sc, ax=a1, fraction=.04, pad=.02).set_label("water confirmed, % of standing rice-like acres")
    a1.set_xlabel("longitude")
    a1.set_ylabel("latitude")
    a1.set_title("where the water was confirmed (size = standing rice-like acres)", loc="left", fontsize=10)

    u = pixels[pixels["class"] == 3]
    bins = np.arange(-4, 12.5, 0.5)
    a2.hist(np.fmax(u["VV_dip"], u["VH_dip"]).clip(-4, 12), bins=bins, alpha=.6, color="#b45f06",
            label="class 3: dip at the trough")
    if "VV_dip_bare_end" in u:
        a2.hist(np.fmax(u["VV_dip_bare_end"], u["VH_dip_bare_end"]).clip(-4, 12), bins=bins, alpha=.5,
                color="#2a78d6", label="class 3: dip at the end of the bare period")
    a2.axvline(3, color=INK, lw=1, ls="--")
    a2.set_xlabel("best radar dip, dB (VV or VH, best track)")
    a2.set_ylabel("pixels (sample)")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("does the water appear when looked for later?", loc="left", fontsize=10)

    for code, color, label in ((1, "#2a78d6", "rice, water confirmed"), (3, "#b45f06", "water unconfirmed")):
        s = table[f"{'rice' if code == 1 else 'unconf'}_trough"].notna()
        doy = pd.to_datetime(table.loc[s, f"{'rice' if code == 1 else 'unconf'}_trough"] + " 2026",
                             format="%d %b %Y").dt.dayofyear
        a3.scatter(doy, table.loc[s, "lat"], color=color, s=40, edgecolor=INK, linewidth=.4, label=label)
    ticks = pd.to_datetime(["2026-06-01", "2026-07-01", "2026-08-01"])
    a3.set_xticks(ticks.dayofyear)
    a3.set_xticklabels([t.strftime("%b") for t in ticks])
    a3.set_ylabel("latitude")
    a3.legend(frameon=False, fontsize=8)
    a3.set_title("median trough date per AOI", loc="left", fontsize=10, color=INK_MUTED)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110, facecolor=SURFACE)
    return fig


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.batch_report", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="+", type=int, required=True)
    args = p.parse_args(argv)
    for aoi_id in args.ids:
        print(pd.Series(aoi_evidence(aoi_id)).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
