"""Why does the radar find no water on pixels whose optical curve looks like rice?

Why
---
After four batches, a large share of the standing rice-like area (class 3 of ``monsoon_rule``) had
no radar water at the trough, and the date-free test in ``batch_report`` found none anywhere
before the climb either. In AOIs that look mostly like rice, that is more likely a flaw in how the
water is measured than a different crop. This module looks at the radar itself instead of a single
summary number:

* :func:`aligned_curves` lines every pixel's VV and VH series up on its own optical trough date and
  returns the per-class median and quartiles for every relative day. If class 3 is rice under
  water for a long time, its curve sits *low* for weeks (the dry reference is already flooded, so
  the dip measured against it is small). If it is a dry-land crop, it never drops.
* the absolute levels matter too: open water under Sentinel-1 is about -18 dB or lower in VV and
  about -24 dB or lower in VH, whatever the field did before.

Nothing here changes the map; a new water test is only adopted after ``analysis/validation`` on the
field plots.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REL_DAYS = np.arange(-90, 121, 6)       # relative-day grid for the aligned curves (half the 12-day repeat)


def sample_pixels(ev: pd.DataFrame, classes, codes=(1, 3), n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Up to ``n`` random pixels of each class code, with their trough date and class."""
    ev = ev.assign(**{"class": np.asarray(classes)})
    parts = []
    for code in codes:
        e = ev[(ev["class"] == code) & ev["valid"]]
        parts.append(e.sample(min(n, len(e)), random_state=seed) if len(e) else e)
    return pd.concat(parts)


def align(dates, cube, pix, trough, rel_days=REL_DAYS, max_gap: int = 7) -> np.ndarray:
    """Per pixel, the value nearest to ``trough + d`` for every ``d`` in ``rel_days`` (NaN if none within ``max_gap``).

    ``dates`` (n_dates), ``cube`` (n_dates, n_pixels) in dB, ``pix`` pixel indices, ``trough``
    datetime64 per selected pixel. Returns (len(pix), len(rel_days)).
    """
    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    t = np.asarray(trough).astype("datetime64[D]")
    want = t[:, None] + rel_days[None, :].astype("timedelta64[D]")            # (p, r)
    diff = np.abs((day[None, None, :] - want[:, :, None]) / np.timedelta64(1, "D"))  # (p, r, d)
    nearest = diff.argmin(axis=2)
    ok = np.take_along_axis(diff, nearest[:, :, None], axis=2)[:, :, 0] <= max_gap
    vals = cube[:, pix].T                                                      # (p, d)
    out = np.take_along_axis(vals, nearest, axis=1)
    return np.where(ok, out, np.nan)


def aligned_curves(aoi_id: int, n: int = 3000, window: int = 5, season_key: str = "monsoon2026",
                   anchor: str = "trough"):
    """Per class and polarisation, quartiles of the trough-aligned radar curves, best track per pixel.

    Returns a long frame: ``class, pol, track, rel_day, q25, median, q75, n`` and the pixel sample.
    Each track is kept separate (their incidence angles differ, so their levels differ).
    ``anchor`` is ``"trough"`` (the rule's date) or ``"climb"`` (the first window with the full NDVI
    rise). The climb is the sharper event: when the trough sits on a long flat low, pixels aligned
    on it are aligned on different stages and their water dips smear out in the median.
    """
    import rasterio

    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import pixel_report as pr
    from . import sar_curve

    d, ev, _ = mr.aoi_events(aoi_id, radar_season=None)
    aoi = d["loc"]["aoi"]
    with rasterio.open(Path("processed/_batch/s2_2026") / aoi / f"{aoi}_monsoon2026.tif") as ds:
        classes = ds.read(1).ravel()
    s = sample_pixels(ev, classes, n=n)
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    rows = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        for pol in ("VV", "VH"):
            a = align(dates, cubes[pol].reshape(len(dates), -1), s.index.to_numpy(),
                      pd.to_datetime(s["trough_date" if anchor == "trough" else "climb_date"]).to_numpy())
            for code in (1, 3):
                m = (s["class"] == code).to_numpy()
                if not m.any():
                    continue
                q = np.nanpercentile(a[m], [25, 50, 75], axis=0)
                cnt = np.isfinite(a[m]).sum(axis=0)
                for j, rd in enumerate(REL_DAYS):
                    rows.append({"aoi": aoi, "anchor": anchor, "class": code, "pol": pol, "track": track, "rel_day": int(rd),
                                 "q25": q[0, j], "median": q[1, j], "q75": q[2, j], "n": int(cnt[j])})
    nd.forget()
    return pd.DataFrame(rows), s


def plot_aligned(curves: pd.DataFrame, out_path=None, title: str = ""):
    """VV and VH (rows) for each track (columns): class 1 vs class 3 median with the quartile band."""
    import matplotlib.pyplot as plt

    from .curves import INK, INK_MUTED, SURFACE

    tracks = sorted(curves["track"].unique())
    fig, axes = plt.subplots(2, len(tracks), figsize=(6.5 * len(tracks), 8), facecolor=SURFACE, squeeze=False)
    for i, pol in enumerate(("VV", "VH")):
        for j, track in enumerate(tracks):
            a = axes[i, j]
            a.set_facecolor(SURFACE)
            a.grid(True, alpha=.2)
            for code, color, label in ((1, "#2a78d6", "rice, water confirmed"), (3, "#b45f06", "water unconfirmed")):
                c = curves[(curves["pol"] == pol) & (curves["track"] == track) & (curves["class"] == code)]
                if c.empty:
                    continue
                a.fill_between(c["rel_day"], c["q25"], c["q75"], color=color, alpha=.15)
                a.plot(c["rel_day"], c["median"], color=color, lw=2, label=label)
            a.axvline(0, color=INK, lw=1, ls="--")
            a.axhline(-18 if pol == "VV" else -24, color=INK_MUTED, lw=.8, ls=":")
            a.set_title(f"{pol}  {track}", loc="left", fontsize=10)
            a.set_xlabel(f"days from the optical {curves['anchor'].iloc[0] if 'anchor' in curves else 'trough'}")
            a.set_ylabel("dB (5x5 mean)")
            if i == 0 and j == 0:
                a.legend(frameon=False, fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=100, facecolor=SURFACE)
    plt.close(fig)
    return fig


def pixel_sheet(aoi_id: int, pid: int, events: pd.DataFrame | None = None, out_dir=None,
                season=("2026-03-15", "2026-09-24")):
    """One pixel, optical and radar on one time axis, plus its clear 5-3-2 chips.

    Top: every raw NDVI observation (filled = Cloud Score+ clear at the pixel, hollow = hazy but
    passed QA60, x = QA60 cloud), the 5-day composite and the fitted curve the rule reads, with the
    rule's trough and climb. Bottom: VV and VH (5x5 mean) of each track. Looking at both together
    answers the questions a summary cannot: did the optical curve put the trough where the field was
    really bare, did the radar see water at some other moment, and is the trough real or haze.
    Returns ``(curve_figure, chips_figure)``.
    """
    import matplotlib.pyplot as plt

    from . import ndvi_5day as nd
    from . import pixel_2026 as p26
    from . import pixel_report as pr
    from . import sar_curve
    from .curves import INK, INK_MUTED, SURFACE

    p = nd.pixel(aoi_id, pid)
    table, chips, clear = p26.chip_stack(aoi_id, pid)
    raw = p["raw"].merge(table[["date", "pixel_clear"]], on="date", how="left")
    ser = p["series"]
    t0, t1 = pd.Timestamp(season[0]), pd.Timestamp(season[1])
    raw, ser = raw[(raw["date"] >= t0) & (raw["date"] < t1)], ser[(ser["window"] >= t0) & (ser["window"] < t1)]
    loc = pr.locate(aoi_id, pid, season_key="monsoon2026")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8.5), facecolor=SURFACE, sharex=True)
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        a.grid(True, alpha=.2)
    cloud = ~raw["qa60_clear"].astype(bool)
    cs_ok = raw["pixel_clear"] >= p26.CLEAR_MIN
    a1.scatter(raw.loc[~cloud & cs_ok, "date"], raw.loc[~cloud & cs_ok, "ndvi"], color="#1f7a3a", s=28,
               label="clear (Cloud Score+)", zorder=3)
    a1.scatter(raw.loc[~cloud & ~cs_ok, "date"], raw.loc[~cloud & ~cs_ok, "ndvi"], facecolor="none",
               edgecolor="#1f7a3a", s=28, label="hazy, passed QA60", zorder=3)
    a1.scatter(raw.loc[cloud, "date"], raw.loc[cloud, "ndvi"], marker="x", color=INK_MUTED, s=18,
               label="QA60 cloud (not used)")
    a1.plot(ser["window"], ser["ndvi_fit"], color="#2a78d6", lw=2, label="fitted NDVI (rule)")
    a1.plot(ser["window"], ser["lswi_fit"], color="#6aa6e8", lw=1, ls="--", label="fitted LSWI")
    title = f"aoi{aoi_id} pid {pid}"
    if events is not None and pid in events.index:
        e = events.loc[pid]
        for key, colour, label in (("trough_date", "#b45f06", "trough"), ("climb_date", "#1f7a3a", "climb"),
                                   ("bare_end_date", "#7a3ab4", "bare end")):
            if key in e and pd.notna(e[key]):
                for a in (a1, a2):
                    a.axvline(pd.Timestamp(e[key]), color=colour, lw=1.2, ls="--")
                a1.text(pd.Timestamp(e[key]), 1.02, label, color=colour, fontsize=8, ha="center")
        title += f" · class {int(e['class'])}" if "class" in e else ""
    a1.set_ylim(-0.3, 1.05)
    a1.set_ylabel("NDVI / LSWI")
    a1.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    a1.set_title(title, loc="left", fontsize=11, color=INK)
    styles = {"VV": "-", "VH": ":"}
    colours = ["#b45f06", "#2a78d6", "#7a3ab4"]
    for k, track in enumerate([t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]):
        dates, cubes = sar_curve.read_track(loc, track)
        for pol in ("VV", "VH"):
            v = cubes[pol].reshape(len(dates), -1)[:, pid]
            a2.plot(dates, v, styles[pol], marker="o", ms=3, color=colours[k % 3], label=f"{pol} {track}")
    a2.set_ylabel("dB (5x5 mean)")
    a2.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    found = pd.DataFrame()
    picked = p26.pixel_clear_chips(table)
    picked = picked[(picked["date"] >= t0) & (picked["date"] < t1)]
    sheet = p26.plot_chips(picked, chips, clear, table["date"].to_numpy(), found) if len(picked) else None
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"aoi{aoi_id}_pid{pid}_curve.png", dpi=90, facecolor=SURFACE)
        if sheet is not None:
            sheet.savefig(out / f"aoi{aoi_id}_pid{pid}_chips.png", dpi=70, facecolor=SURFACE)
    plt.close(fig)
    if sheet is not None:
        plt.close(sheet)
    return fig, sheet


def flood_table(aoi_id: int, frame: pd.DataFrame, pixel_col: str | None = None, anchor_col: str = "climb_date",
                random_anchor: bool = False, seed: int = 0, random_span=("2026-06-20", "2026-09-19")) -> pd.DataFrame:
    """``radar_water.pixel_floods`` for the pixels of ``frame`` (one AOI), joined back to the frame.

    Anchors are the pixels' own optical climb dates, or (``random_anchor``) random dates, which is the
    null: how large a "flood" the search finds on ground that is never flooded (evergreen, bare).
    """
    from . import radar_water as rw

    pix = (frame[pixel_col] if pixel_col else frame.index.to_series()).to_numpy().astype(int)
    if random_anchor:
        rng = np.random.default_rng(seed)
        days = pd.date_range(*random_span, freq="D").to_numpy().astype("datetime64[D]")
        anchor = rng.choice(days, size=len(pix))
    else:
        anchor = pd.to_datetime(frame[anchor_col]).to_numpy().astype("datetime64[D]")
    fl = rw.pixel_floods(aoi_id, anchor, pixels=pix)
    out = frame.reset_index(drop=pixel_col is not None).copy()
    for c in fl.columns:
        out[c] = fl[c].to_numpy()
    out["anchor"] = anchor
    return out


def canopy_before(aoi_id: int, pixels, when, ref=(45, 12)) -> np.ndarray:
    """Highest fitted NDVI in the weeks before ``when`` (the same window the radar drop is measured against).

    A radar drop under a field that carried a canopy just before is a harvest (the canopy's volume
    scattering removed), not water arriving on a bare, puddled field.
    """
    from . import ndvi_5day as nd

    d = nd.load(aoi_id)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)[:, np.asarray(pixels)]
    win = pd.DatetimeIndex(d["windows"]).to_numpy().astype("datetime64[D]")
    w = np.asarray(when).astype("datetime64[D]")
    lag = (w[None, :] - win[:, None]) / np.timedelta64(1, "D")
    with np.errstate(invalid="ignore"):
        inside = (lag >= ref[1]) & (lag <= ref[0])
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(np.where(inside, ndvi, np.nan), axis=0)


def same_class_share(classes_2d, size: int = 5) -> np.ndarray:
    """Per pixel, the share of its ``size`` x ``size`` neighbourhood that carries the same class.

    1.0 means the pixel sits inside a field of its own class; the radar 5x5 mean there reads only
    that field. Low values are edges: roads, bunds, a neighbouring field of another crop.
    """
    from scipy.ndimage import uniform_filter

    c = np.asarray(classes_2d)
    out = np.zeros(c.shape, dtype="float32")
    for code in np.unique(c):
        m = (c == code).astype("float32")
        share = uniform_filter(m, size=size, mode="nearest")
        out = np.where(c == code, share, out)
    return out


def interior_pids(classes_2d, code: int, n: int, size: int = 5, seed: int = 0) -> np.ndarray:
    """Up to ``n`` random pixel ids of class ``code`` whose whole ``size`` x ``size`` neighbourhood is that class."""
    share = same_class_share(classes_2d, size)
    pool = np.flatnonzero((np.asarray(classes_2d) == code).ravel() & (share.ravel() >= 0.999))
    rng = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n, len(pool)), replace=False) if len(pool) else pool


CLEAR_CS = 60          # Cloud Score+ "clear" at the pixel for an observation to count as really seen
SEEN_DAYS = 20         # the trough counts as seen when a clear, low observation lies this close to it


def diagnose(aoi_id: int, n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Per sampled field-interior pixel of classes 1 and 3: why the water was or was not found.

    Columns added to the rule's events: ``flood_*`` (the flood search anchored on the climb, see
    ``radar_water.flood_search``), ``canopy_before`` (NDVI in the weeks before that drop: high means a
    harvest, not water), ``trough_seen`` (a Cloud Score+ clear observation at or below the bare level
    within ``SEEN_DAYS`` of the trough: without one the trough may be haze interpolated across a
    cloudy gap) and ``clear_obs_season`` (clear observations May-September).
    """
    import rasterio

    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import radar_water as rw

    d, ev, radar = mr.aoi_events(aoi_id)
    aoi = d["loc"]["aoi"]
    with rasterio.open(Path("processed/_batch/s2_2026") / aoi / f"{aoi}_monsoon2026.tif") as ds:
        c2 = ds.read(1)
    pids = np.concatenate([interior_pids(c2, code, n, seed=seed) for code in (1, 3)])
    e = ev.loc[pids].copy()
    e["class"] = c2.ravel()[pids]
    anchor = pd.to_datetime(e["climb_date"]).to_numpy().astype("datetime64[D]")
    fl = rw.pixel_floods(aoi_id, anchor, pixels=pids)
    for col in fl.columns:
        e[col] = fl[col].to_numpy()
    e["canopy_before"] = canopy_before(aoi_id, pids, e["flood_date"].to_numpy())
    # was the trough really seen? strict-mask read of the raw dates
    dates, ndvi_raw, _, _, ok_cs, _, _ = nd.read_dates(aoi_id, cs_min=CLEAR_CS)
    raw = ndvi_raw.reshape(len(dates), -1)[:, pids]
    okc = ok_cs.reshape(len(dates), -1)[:, pids]
    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    t = e["trough_date"].to_numpy().astype("datetime64[D]")
    near = np.abs((day[:, None] - t[None, :]) / np.timedelta64(1, "D")) <= SEEN_DAYS
    e["trough_seen"] = (near & okc & (raw <= mr.TROUGH_MAX + 0.05)).any(axis=0)
    season = (day >= np.datetime64("2026-05-01"))[:, None]
    e["clear_obs_season"] = (okc & season).sum(axis=0)
    e["aoi"] = aoi
    nd.forget()
    return e


def diagnosis_summary(e: pd.DataFrame, drop_min: float = 4.0, persist_min: int = 2, canopy_max: float = 0.5) -> pd.DataFrame:
    """Share of each class's interior pixels in each explanation, per AOI."""
    e = e.copy()
    e["old_wet"] = e["radar_wet"].fillna(False) if "radar_wet" in e else False
    e["flood_elsewhere"] = ((e["flood_drop"] >= drop_min) & (e["flood_persist"] >= persist_min)
                            & (e["canopy_before"] <= canopy_max))
    rows = []
    for (aoi, code), g in e.groupby(["aoi", "class"]):
        rows.append({"aoi": aoi, "class": code, "n": len(g),
                     "old_wet_pct": round(100 * g["old_wet"].mean(), 1),
                     "flood_elsewhere_pct": round(100 * g["flood_elsewhere"].mean(), 1),
                     "trough_not_seen_pct": round(100 * (~g["trough_seen"]).mean(), 1),
                     "not_seen_and_no_flood_pct": round(100 * ((~g["trough_seen"]) & ~g["flood_elsewhere"]).mean(), 1),
                     "seen_and_no_flood_pct": round(100 * (g["trough_seen"] & ~g["flood_elsewhere"] & ~g["old_wet"]).mean(), 1),
                     "clear_obs_season_median": float(g["clear_obs_season"].median())})
    return pd.DataFrame(rows)


def group_centers(classes_2d, code: int, n: int, context: int = 9, seed: int = 0) -> np.ndarray:
    """Up to ``n`` pixel ids whose whole ``context`` x ``context`` neighbourhood is class ``code``.

    Such a centre carries a block of pixels from the middle of one field (or pond, or orchard) of
    that class: the unit to inspect. A single pixel, and above all an edge pixel, mixes a field
    with its bunds, roads and neighbours, and was once read as a field while it sat on a fish pond.
    """
    from scipy.ndimage import uniform_filter

    c = np.asarray(classes_2d)
    share = uniform_filter((c == code).astype("float32"), size=context, mode="constant")
    pool = np.flatnonzero((share >= 0.999).ravel())
    rng = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n, len(pool)), replace=False) if len(pool) else pool


def block_pids(center: int, width: int, height: int, block: int = 5) -> np.ndarray:
    """Pixel ids of the ``block`` x ``block`` square centred on ``center`` (clipped to the grid)."""
    r, c = divmod(int(center), int(width))
    h = block // 2
    rows = np.arange(max(0, r - h), min(height, r + h + 1))
    cols = np.arange(max(0, c - h), min(width, c + h + 1))
    return (rows[:, None] * width + cols[None, :]).ravel()


def group_sheet(aoi_id: int, center: int, block: int = 5, out_dir=None, label: str = "",
                season=("2026-03-15", "2026-09-24")):
    """A block of pixels from the middle of a field, optical and radar on one time axis, plus chips.

    Top: per date the median raw NDVI of the block's QA60-clear pixels (filled when most of the
    block is Cloud Score+ clear, hollow when hazy), the block's median fitted NDVI and LSWI, and the
    trough and climb the rule finds on that median curve. Bottom: the block's median VV and VH per
    track. The chips show the whole block in the magenta square.
    """
    import matplotlib.pyplot as plt

    from . import monsoon_rule as mr
    from . import ndvi_5day as nd
    from . import pixel_2026 as p26
    from . import pixel_report as pr
    from . import sar_curve
    from .curves import INK, INK_MUTED, SURFACE

    d = nd.load(aoi_id)
    g = d["loc"]["grid"]
    pids = block_pids(center, int(g["width"]), int(g["height"]), block)
    dates = pd.DatetimeIndex(d["dates"])
    ndvi = d["ndvi"].reshape(len(dates), -1)[:, pids]
    ok = d["ok"].reshape(len(dates), -1)[:, pids]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = np.nanmedian(np.where(ok, ndvi, np.nan), axis=1)
    table, chips, clear = p26.chip_stack(aoi_id, int(center))
    h, b = p26.HALF, block // 2
    cs_share = (clear[:, h - b:h + b + 1, h - b:h + b + 1] >= p26.CLEAR_MIN).mean(axis=(1, 2))
    cs = pd.Series(cs_share, index=pd.DatetimeIndex(table["date"]))
    windows = pd.DatetimeIndex(d["windows"])
    fit = np.nanmedian(d["ndvi5d"].reshape(len(windows), -1)[:, pids], axis=1)
    lswi = np.nanmedian(d["lswi5d"].reshape(len(windows), -1)[:, pids], axis=1)
    ev = mr.pixel_events(fit[:, None], lswi[:, None], windows).iloc[0]
    t0, t1 = pd.Timestamp(season[0]), pd.Timestamp(season[1])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 8.5), facecolor=SURFACE, sharex=True)
    for a in (a1, a2):
        a.set_facecolor(SURFACE)
        a.grid(True, alpha=.2)
    sel = (dates >= t0) & (dates < t1)
    share = cs.reindex(dates).to_numpy()
    good = sel & np.isfinite(raw) & (share >= 0.5)
    hazy = sel & np.isfinite(raw) & ~(share >= 0.5)
    a1.scatter(dates[good], raw[good], color="#1f7a3a", s=28, label="block mostly clear (Cloud Score+)", zorder=3)
    a1.scatter(dates[hazy], raw[hazy], facecolor="none", edgecolor="#1f7a3a", s=28, label="hazy, passed QA60", zorder=3)
    w = (windows >= t0) & (windows < t1)
    a1.plot(windows[w], fit[w], color="#2a78d6", lw=2, label="median fitted NDVI")
    a1.plot(windows[w], lswi[w], color="#6aa6e8", lw=1, ls="--", label="median fitted LSWI")
    for key, colour, name in (("trough_date", "#b45f06", "trough"), ("climb_date", "#1f7a3a", "climb")):
        if pd.notna(ev[key]):
            for a in (a1, a2):
                a.axvline(pd.Timestamp(ev[key]), color=colour, lw=1.2, ls="--")
            a1.text(pd.Timestamp(ev[key]), 1.02, name, color=colour, fontsize=8, ha="center")
    a1.set_ylim(-0.3, 1.05)
    a1.set_ylabel("NDVI / LSWI (block median)")
    a1.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    a1.set_title(f"aoi{aoi_id} block of {len(pids)} px around {center} {label}", loc="left", fontsize=11, color=INK)
    loc = pr.locate(aoi_id, int(center), season_key="monsoon2026")
    colours = ["#b45f06", "#2a78d6", "#7a3ab4"]
    for k, track in enumerate([t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]):
        rd, cubes = sar_curve.read_track(loc, track)
        for pol, style in (("VV", "-"), ("VH", ":")):
            v = np.nanmedian(cubes[pol].reshape(len(rd), -1)[:, pids], axis=1)
            a2.plot(rd, v, style, marker="o", ms=3, color=colours[k % 3], label=f"{pol} {track}")
    a2.set_ylabel("dB (block median of 5x5 means)")
    a2.legend(frameon=False, fontsize=8, ncol=4, loc="lower left")
    fig.tight_layout()
    picked = table[(table["date"] >= t0) & (table["date"] < t1)]
    picked = picked[cs.reindex(picked["date"]).to_numpy() >= 0.5]
    sheet = (p26.plot_chips(picked, chips, clear, table["date"].to_numpy(), pd.DataFrame(), box=block)
             if len(picked) else None)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"aoi{aoi_id}_block{center}_curve.png", dpi=90, facecolor=SURFACE)
        if sheet is not None:
            sheet.savefig(out / f"aoi{aoi_id}_block{center}_chips.png", dpi=70, facecolor=SURFACE)
    plt.close(fig)
    if sheet is not None:
        plt.close(sheet)
    return fig, sheet, ev


LONG_SPAN = (-100, -5)     # flood searched from 100 to 5 days before the optical climb


def evidence_features(aoi_id: int, pixels, trough, climb, season_key: str = "monsoon2026",
                      read_clear: bool = True) -> pd.DataFrame:
    """Everything the group sheets showed matters, per pixel, as numbers.

    * ``flood_drop``/``flood_persist``/``flood_date``: flood search over ``LONG_SPAN`` before the climb;
    * ``flood_vv``/``flood_vh``: the lowest VV and VH (best track = lowest) inside that span, dB;
      flooded paddy goes far below dry soil, a dry-land crop or a tree line does not;
    * ``vh_at_trough``: median VH within 20 days of the optical trough (bare or puddled soil is low,
      a tree line or a settlement stays high);
    * ``canopy_before_trough``: highest fitted NDVI 60 to 20 days before the trough (trees and
      gardens were already green; a field was bare or carried the previous crop);
    * ``canopy_before``: highest fitted NDVI in the weeks before the flood drop (harvest guard);
    * ``trough_seen``: a Cloud Score+ clear low observation within ``SEEN_DAYS`` of the trough.
    """
    import warnings

    from . import ndvi_5day as nd
    from . import pixel_report as pr
    from . import radar_water as rw
    from . import sar_curve

    pix = np.asarray(pixels).astype(int)
    trough = np.asarray(trough).astype("datetime64[D]")
    climb = np.asarray(climb).astype("datetime64[D]")
    out = pd.DataFrame(index=pix)
    fl = rw.pixel_floods(aoi_id, climb, pixels=pix, span=LONG_SPAN)
    for c in fl.columns:
        out[c] = fl[c].to_numpy()
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    vv_min = np.full(len(pix), np.inf)
    vh_min = np.full(len(pix), np.inf)
    vh_trough = np.full(len(pix), -np.inf)
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track)
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        rel = (day[:, None] - climb[None, :]) / np.timedelta64(1, "D")
        near_t = np.abs((day[:, None] - trough[None, :]) / np.timedelta64(1, "D")) <= 20
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            span = (rel >= LONG_SPAN[0]) & (rel <= LONG_SPAN[1])
            vv = cubes["VV"].reshape(len(dates), -1)[:, pix]
            vh = cubes["VH"].reshape(len(dates), -1)[:, pix]
            vv_min = np.fmin(vv_min, np.nanmin(np.where(span, vv, np.nan), axis=0))
            vh_min = np.fmin(vh_min, np.nanmin(np.where(span, vh, np.nan), axis=0))
            vh_trough = np.fmax(vh_trough, np.nanmedian(np.where(near_t, vh, np.nan), axis=0))
    out["flood_vv"] = np.where(np.isfinite(vv_min), vv_min, np.nan)
    out["flood_vh"] = np.where(np.isfinite(vh_min), vh_min, np.nan)
    out["vh_at_trough"] = np.where(np.isfinite(vh_trough), vh_trough, np.nan)
    d = nd.load(aoi_id)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)[:, pix]
    win = pd.DatetimeIndex(d["windows"]).to_numpy().astype("datetime64[D]")
    lag = (trough[None, :] - win[:, None]) / np.timedelta64(1, "D")
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["canopy_before_trough"] = np.nanmax(np.where((lag >= 20) & (lag <= 60), ndvi, np.nan), axis=0)
    out["canopy_before"] = canopy_before(aoi_id, pix, out["flood_date"].to_numpy())
    if read_clear:
        dates, ndvi_raw, _, _, ok_cs, _, _ = nd.read_dates(aoi_id, cs_min=CLEAR_CS)
        raw = ndvi_raw.reshape(len(dates), -1)[:, pix]
        okc = ok_cs.reshape(len(dates), -1)[:, pix]
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        near = np.abs((day[:, None] - trough[None, :]) / np.timedelta64(1, "D")) <= SEEN_DAYS
        from . import monsoon_rule as mr
        out["trough_seen"] = (near & okc & (raw <= mr.TROUGH_MAX + 0.05)).any(axis=0)
    nd.forget()
    return out


def aoi_evidence_sample(aoi_id: int, n_interior: int = 1500, n_edge: int = 800, seed: int = 0,
                        out_dir="processed/_batch/s2_2026/report/evidence") -> pd.DataFrame:
    """Evidence features for a sample of class 1 and class 3 pixels of one AOI, interior and edge.

    Interior = the pixel's whole 5x5 neighbourhood has its class (see :func:`same_class_share`);
    edge = less than half of it does. Written to ``<out_dir>/aoi<N>.parquet``.
    """
    import rasterio

    from . import monsoon_rule as mr

    d, ev, _ = mr.aoi_events(aoi_id, radar_season=None)
    aoi = d["loc"]["aoi"]
    with rasterio.open(Path("processed/_batch/s2_2026") / aoi / f"{aoi}_monsoon2026.tif") as ds:
        c2 = ds.read(1)
    share = same_class_share(c2).ravel()
    flat = c2.ravel()
    rng = np.random.default_rng(seed)
    picks = []
    for code in (1, 3):
        for kind, mask, n in (("interior", share >= 0.999, n_interior), ("edge", share < 0.5, n_edge)):
            pool = np.flatnonzero((flat == code) & mask & ev["valid"].to_numpy() & ev["climb_date"].notna().to_numpy())
            if len(pool):
                sel = rng.choice(pool, size=min(n, len(pool)), replace=False)
                picks.append(pd.DataFrame({"pixel": sel, "class": code, "position": kind}))
    if not picks:
        return pd.DataFrame()
    s = pd.concat(picks, ignore_index=True)
    e = ev.loc[s["pixel"]].reset_index(drop=True)
    s = pd.concat([s, e], axis=1)
    f = evidence_features(aoi_id, s["pixel"].to_numpy(), s["trough_date"].to_numpy(),
                          pd.to_datetime(s["climb_date"]).to_numpy())
    s = s.merge(f.reset_index(names="pixel").drop_duplicates("pixel"), on="pixel", how="left")
    s["aoi"] = aoi
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    s.to_parquet(Path(out_dir) / f"{aoi}.parquet")
    return s


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.water_investigation",
                                description="Evidence features for class 1 / class 3 samples, per AOI.")
    p.add_argument("--ids", nargs="+", type=int, required=True)
    args = p.parse_args(argv)
    for aoi_id in args.ids:
        s = aoi_evidence_sample(aoi_id)
        print(f"aoi{aoi_id}: {len(s)} pixels")
    return 0





def ndvi_at(aoi_id: int, pixels, when) -> np.ndarray:
    """Fitted NDVI of each pixel on its own date ``when`` (nearest 5-day window; NaN for NaT)."""
    from . import ndvi_5day as nd

    d = nd.load(aoi_id)
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)[:, np.asarray(pixels).astype(int)]
    win = pd.DatetimeIndex(d["windows"]).to_numpy().astype("datetime64[D]")
    w = np.asarray(when).astype("datetime64[D]")
    ok = ~np.isnat(w)
    idx = np.clip(np.searchsorted(win, np.where(ok, w, win[0])), 0, len(win) - 1)
    out = ndvi[idx, np.arange(ndvi.shape[1])]
    return np.where(ok, out, np.nan)


def cross_track_support(aoi_id: int, pixels, when, pol, days: int = 14, drop_db: float = 2.0,
                        season_key: str = "monsoon2026") -> np.ndarray:
    """How many acquisitions, over all tracks, within ``days`` of ``when`` sit ``drop_db`` below their own earlier level.

    ``pol`` per pixel ("VV"/"VH", the polarisation of the flood found). Standing water lasts; a
    single speckled or wind-roughened date on one track does not show on the other track two days
    later. Counting over tracks keeps short floods (one pass per track) countable.
    """
    from . import pixel_report as pr
    from . import radar_water as rw
    from . import sar_curve

    pix = np.asarray(pixels).astype(int)
    w = np.asarray(when).astype("datetime64[D]")
    pol = np.asarray(pol)
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    count = np.zeros(len(pix), dtype=int)
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track)
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        near = np.abs((day[:, None] - w[None, :]) / np.timedelta64(1, "D")) <= days
        for p in ("VV", "VH"):
            drops = rw.local_drops(dates, cubes[p].reshape(len(dates), -1)[:, pix])
            with np.errstate(invalid="ignore"):
                hit = (near & (drops >= drop_db)).sum(axis=0)
            count += np.where(pol == p, hit, 0)
    return count


def monsoon_flood_share(aoi_id: int, code: int = 3, n: int = 3000, seed: int = 0,
                        start: str = "2026-05-15", end: str = "2026-09-24") -> dict:
    """For interior pixels of one class: did the radar see standing water at ANY monsoon date?

    The rule looks for the transplanting water before the climb. A field whose water came only
    after the optical climb (a floodplain drowned in August, a late transplanting after an early
    weed flush) or never came at all is left in class 3; this splits the two, without guards on
    the optical dates: a drop of 4 dB below the field's own earlier level, confirmed by a second
    pass on the same track and polarisation.
    """
    import rasterio

    from . import radar_water as rw

    with rasterio.open(Path("processed/_batch/s2_2026") / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
        c2 = ds.read(1)
    pids = interior_pids(c2, code, n, seed=seed)
    if not len(pids):
        return {"aoi": f"aoi{aoi_id}", "n": 0}
    anchor = np.full(len(pids), np.datetime64(end), dtype="datetime64[D]")
    span = (-(np.datetime64(end) - np.datetime64(start)).astype(int), 0)
    fl = rw.pixel_floods(aoi_id, anchor, pixels=pids, span=span)
    wet = (fl["flood_drop"] >= 4) & (fl["flood_persist"] >= 2)
    when = pd.to_datetime(fl.loc[wet, "flood_date"])
    return {"aoi": f"aoi{aoi_id}", "n": int(len(pids)), "any_monsoon_flood_pct": round(100 * float(wet.mean()), 1),
            "flood_month_median": when.median().strftime("%d %b") if len(when) else None}


def block_series(aoi_id: int, center: int, block: int = 5, season=("2026-03-15", "2026-09-24"),
                 tracks_cache: dict | None = None) -> dict:
    """Median series of one field-interior block: raw NDVI (QA60-clear), fitted NDVI, VV/VH per track.

    ``tracks_cache`` (track -> (dates, cubes)) avoids re-reading the AOI's radar stacks per block.
    """
    import warnings

    from . import ndvi_5day as nd
    from . import pixel_report as pr
    from . import sar_curve

    d = nd.load(aoi_id)
    g = d["loc"]["grid"]
    pids = block_pids(center, int(g["width"]), int(g["height"]), block)
    dates = pd.DatetimeIndex(d["dates"])
    ndvi = d["ndvi"].reshape(len(dates), -1)[:, pids]
    ok = d["ok"].reshape(len(dates), -1)[:, pids]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = np.nanmedian(np.where(ok, ndvi, np.nan), axis=1)
    windows = pd.DatetimeIndex(d["windows"])
    fit = np.nanmedian(d["ndvi5d"].reshape(len(windows), -1)[:, pids], axis=1)
    loc = pr.locate(aoi_id, int(center), season_key="monsoon2026")
    radar = {}
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        if tracks_cache is not None and track in tracks_cache:
            rd, cubes = tracks_cache[track]
        else:
            rd, cubes = sar_curve.read_track(loc, track)
            if tracks_cache is not None:
                tracks_cache[track] = (rd, cubes)
        radar[track] = (rd, {p: np.nanmedian(cubes[p].reshape(len(rd), -1)[:, pids], axis=1) for p in ("VV", "VH")})
    t0, t1 = pd.Timestamp(season[0]), pd.Timestamp(season[1])
    return {"dates": dates, "raw": raw, "windows": windows, "fit": fit, "radar": radar, "t0": t0, "t1": t1}


def gallery(aoi_id: int, code: int, n: int = 20, seed: int = 0, out_dir=None, context: int = 9):
    """Small multiples: ``n`` field-interior blocks of one class, NDVI on top, VV/VH below, one panel each.

    Why: one or two pixels can mislead (an edge, a pond rim); twenty blocks from the middles of
    fields show what the class really is, and the class-1 gallery of the same AOI is the reference.
    """
    import matplotlib.pyplot as plt
    import rasterio

    from . import ndvi_5day as nd
    from .curves import INK, INK_MUTED, SURFACE

    with rasterio.open(Path("processed/_batch/s2_2026") / f"aoi{aoi_id}" / f"aoi{aoi_id}_monsoon2026.tif") as ds:
        c = ds.read(1)
    centers = group_centers(c, code, n, context=context, seed=seed)
    if not len(centers):
        centers = group_centers(c, code, n, context=5, seed=seed)
    if not len(centers):
        return None, []
    cols = 5
    rows = int(np.ceil(len(centers) / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(4.2 * cols, 4.4 * rows), facecolor=SURFACE,
                             squeeze=False, gridspec_kw={"height_ratios": [1, 1] * rows})
    colours = ["#b45f06", "#2a78d6", "#7a3ab4"]
    cache: dict = {}
    for i, center in enumerate(centers):
        s = block_series(aoi_id, int(center), tracks_cache=cache)
        a1, a2 = axes[2 * (i // cols), i % cols], axes[2 * (i // cols) + 1, i % cols]
        for a in (a1, a2):
            a.set_facecolor(SURFACE)
            a.grid(True, alpha=.2)
            a.set_xlim(s["t0"], s["t1"])
            a.tick_params(labelsize=6)
        m = (s["dates"] >= s["t0"]) & (s["dates"] < s["t1"])
        a1.scatter(s["dates"][m], s["raw"][m], s=6, color="#1f7a3a")
        w = (s["windows"] >= s["t0"]) & (s["windows"] < s["t1"])
        a1.plot(s["windows"][w], s["fit"][w], color="#2a78d6", lw=1.5)
        a1.set_ylim(-0.3, 1.0)
        a1.set_title(f"{center}", fontsize=7, color=INK_MUTED, loc="left")
        for k, (track, (rd, v)) in enumerate(s["radar"].items()):
            a2.plot(rd, v["VV"], "-", color=colours[k % 3], lw=1)
            a2.plot(rd, v["VH"], ":", color=colours[k % 3], lw=1)
        a2.axhline(-19, color=INK_MUTED, lw=.6, ls="--")
        a2.set_ylim(-28, -2)
        a1.set_xticklabels([])
    for j in range(len(centers), rows * cols):
        axes[2 * (j // cols), j % cols].set_visible(False)
        axes[2 * (j // cols) + 1, j % cols].set_visible(False)
    fig.suptitle(f"aoi{aoi_id} class {code}: {len(centers)} field-interior blocks. Top NDVI (dots clear obs, "
                 f"line fit, -0.3..1). Bottom VV solid / VH dotted per track, dashed -19 dB",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    nd.forget()
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"aoi{aoi_id}_class{code}_gallery.png", dpi=75, facecolor=SURFACE)
    plt.close(fig)
    return fig, list(centers)


def gallery_main(argv=None) -> int:
    """Command line: galleries for the given classes of the given AOIs."""
    import argparse

    p = argparse.ArgumentParser(prog="python -m sar_pipeline.analysis.water_investigation gallery")
    p.add_argument("--ids", nargs="+", type=int, required=True)
    p.add_argument("--codes", nargs="+", type=int, default=[3, 1])
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="processed/_batch/s2_2026/figures/gallery")
    args = p.parse_args(argv)
    for a in args.ids:
        for code in args.codes:
            _, c = gallery(a, code, args.n, seed=args.seed, out_dir=args.out)
            print(f"aoi{a} class {code}: {len(c)} blocks")
    return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "gallery":
        raise SystemExit(gallery_main(sys.argv[2:]))
    raise SystemExit(main())
