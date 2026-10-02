"""Fresh start, step 2: when was each crop field sown? (the last time it was empty before the September crop)

Why
---
User rules (30 Sep 2026): the sowing is the last time the field was empty (bare or under water), which is usually its
lowest point of the season, in May, June or up to mid-July. After mid-July the optical view cannot be trusted on its
own: July and August are the cloudiest months and a cloud or haze makes a green field look empty. So an empty spell
after ``OPTICAL_TRUSTED_UNTIL`` counts only when the radar agrees that the field was empty then; otherwise the last
empty spell before that date is the sowing.

Per crop-ground pixel (class 1 of step 1, ``vegetation_types``):

* **empty**: the fitted NDVI is at the pixel's own low level: its lowest value held for two windows since 1 May
  (``vegetation_types.sustained_low``) plus twice its own normal wobble (``monsoon_rule.optical_noise`` of its clear
  views). Only spells of at least two windows count (one window can be one hazy view).
* **sowing** = the last trough of the smoothed curve since 1 May (``troughs``) that lies at this AOI's bare level
  (``bare_level``): the low point just before the crop standing in September. Earlier versions used an "empty level" (the pixel's season low + wobble): it lagged the
  sowing by ~10 days (pixel 81532) and could not see a second crop's trough above the May low (pixel 59890, 0.33 in
  July); both dropped on the user's instruction (30 Sep).
* **seen or not**: the chosen low counts only when two clear views at bare level lie within ``CLEAR_VIEWS_REACH``
  windows of it; otherwise (a curve filled across a cloudy gap, or one hazy view) the sowing is read from the radar
  (:func:`radar_sowing`: the last low of VH and VV before their rise, median over tracks) and ``_step2_source.tif``
  says so (1 optical, 2 radar). User, 1 Oct: "look at the radar curves for every pixel too".
* **radar agreement** for a spell after mid-July: on some track, the pass nearest that date (within one revisit) is at
  the pixel's own radar low: VH no more than twice its own pass-to-pass wobble above its second-darkest pass since
  1 May, and at least twice that wobble below its second-brightest pass (its green level). A field that was empty or
  flooded then is at its radar low; a field under a canopy is not; a pixel whose radar stays flat all season cannot
  agree (first run on aoi160: flat pixels "agreed" to empty spells in September, when their field was green).

Outputs in ``<out_root>/rice_fresh/aoi<N>/``: ``aoi<N>_step2_sowing.tif`` (day of year, 0 = not crop ground),
``aoi<N>_step2_period.tif`` (1 May, 2 June, 3 July up to mid-July, 4 after mid-July with radar agreement, 5 after
mid-July without it and no earlier empty spell) + ``.qml``, ``aoi<N>_step2.csv`` (acres per period), ``aoi<N>_step2.png``.

Use::

    python -m sar_pipeline.analysis.sowing_fresh --aoi 160 --series-root processed/_batch/s2_2026_hyb40m1late
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from .optical_phenology import acres
from .vegetation_types import START

#: User (30 Sep): the optical view alone is trusted for an empty field up to mid-July; later the radar must agree.
OPTICAL_TRUSTED_UNTIL = "2026-07-15"
#: Windows on each side of a late trough in which two clear views at bare level confirm it without the radar.
CLEAR_VIEWS_REACH = 3
#: Sowing from the radar where the optical did not see the low (user, 1 Oct). OFF: on aoi39, where two clear views
#: did see the sowing, the radar's last low before its rise came 50-85 days later (median; only 22-33 % within
#: 15 days), raw passes or smoothed, VH or VV: the radar's lows in July-August (flooding, rain) are not the optical's
#: bare spell. With it ON late rice rose from 33 to 102 ac. Kept for a better radar start (see ``radar_sowing``).
RADAR_SOWING = False
#: One radar revisit (days): the pass that speaks for a date must lie this close to it.
REVISIT_DAYS = 12
PERIODS = {1: "May", 2: "June", 3: "1-15 July", 4: "after 15 July, confirmed (radar or two clear views)",
           5: "after 15 July, not confirmed (no earlier empty spell)"}
COLOURS = {1: "#1a9850", 2: "#91cf60", 3: "#fee08b", 4: "#fc8d59", 5: "#d73027"}


def last_true(mask, upto=None) -> np.ndarray:
    """Per pixel: index of the last True row of ``mask`` (optionally only rows ``< upto``); -1 if none."""
    m = np.asarray(mask, dtype=bool)
    if upto is not None:
        m = m.copy()
        m[upto:] = False
    n = m.shape[0]
    return np.where(m.any(axis=0), n - 1 - np.argmax(m[::-1], axis=0), -1)


def radar_empty_at(series, day, k: float | None = None, start: str = START) -> np.ndarray:
    """Per pixel: on some track, is the pass nearest ``day`` (within ``REVISIT_DAYS``) at the pixel's own radar low?"""
    from .radar_water import WATER_K, pass_noise

    k = WATER_K if k is None else k
    day = np.asarray(day).astype("datetime64[D]")
    n = len(day)
    cols = np.arange(n)
    agree = np.zeros(n, dtype=bool)
    for dates, flat in series:
        d = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        use = d >= np.datetime64(start)
        d = d[use]
        if len(d) < 3:
            continue
        x = np.asarray(flat["VH"], dtype="float32")[use]
        noise = pass_noise(x)
        low = np.partition(np.where(np.isfinite(x), x, np.inf), 1, axis=0)[1]
        high = -np.partition(np.where(np.isfinite(x), -x, np.inf), 1, axis=0)[1]
        dd = np.where(np.isnat(day), d[0], day)
        j = np.abs(d[:, None] - dd[None, :]).argmin(axis=0)
        near = np.abs(d[j] - dd) <= np.timedelta64(REVISIT_DAYS, "D")
        with np.errstate(invalid="ignore"):
            # at its own low AND clearly below its own green level: a flat radar (dry-sown field, tree) says nothing
            at_low = (x[j, cols] <= low + k * noise) & (x[j, cols] <= high - k * noise)
        agree |= ~np.isnat(day) & near & at_low
    return agree


def radar_sowing(series, start: str = START, k: float | None = None, pols=("VH", "VV")) -> np.ndarray:
    """Per pixel: the sowing read from the radar alone (datetime64[D], NaT where the radar shows no clear start).

    Per track and polarisation: the LAST pass that (a) is lower than every later pass, (b) lies near the pixel's own
    radar low (within ``k`` x its pass-to-pass wobble of its second-darkest pass since ``start``) and (c) is followed by
    a rise to the end of at least ``k`` x that wobble: the start of the crop standing now. The sowing is the median of
    these dates over all tracks and polarisations. Why (user, 1 Oct, aoi39 pixel 12617): no clear optical view from
    19 May to 11 Sep except one hazy view; the filled NDVI put the sowing on 7 Aug, while VV and VH on both tracks were
    lowest around 12-15 July and rose after it."""
    from .radar_water import WATER_K, pass_noise

    k = WATER_K if k is None else k
    found = []
    for dates, flat in series:
        d = pd.DatetimeIndex(dates)
        use = d >= pd.Timestamp(start)
        if use.sum() < 4:
            continue
        dd = d[use].to_numpy().astype("datetime64[D]")
        for pol in pols:
            if pol not in flat:
                continue
            x = np.asarray(flat[pol], dtype="float32")[use]
            with warnings.catch_warnings(), np.errstate(invalid="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                noise = pass_noise(x)
                fin = np.where(np.isfinite(x), x, np.inf)
                low2 = np.partition(fin, 1, axis=0)[1]
                end = np.nanmedian(x[-2:], axis=0)
                later = np.minimum.accumulate(fin[::-1], axis=0)[::-1]       # min of x[i:]
                after = np.vstack([later[1:], np.full((1, x.shape[1]), np.inf)])  # min of x[i+1:]
                ok = np.isfinite(x) & (x < after) & (x <= low2 + k * noise) & (end - x >= k * noise)
            last = last_true(ok)
            found.append(np.where(last >= 0, dd[np.clip(last, 0, None)], np.datetime64("NaT")))
    if not found:
        return np.array([], dtype="datetime64[D]")
    stack = np.stack(found).astype("datetime64[D]")
    days = np.where(np.isnat(stack), np.nan, stack.astype("int64").astype("float64"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _days_to_dates(np.nanmedian(days, axis=0))


def _days_to_dates(days) -> np.ndarray:
    out = np.full(len(days), np.datetime64("NaT"), dtype="datetime64[D]")
    ok = np.isfinite(days)
    out[ok] = np.round(days[ok]).astype("int64").astype("datetime64[D]")
    return out


def bare_level(fit, noise, crop, k: float | None = None) -> np.ndarray:
    """Per pixel: the highest value a sowing trough may have. A field at sowing is bare or under water, so its trough
    lies at this AOI's bare level: the median of the crop-ground pixels' own lows (the lowest value each held for two
    windows, ``vegetation_types.sustained_low``) plus ``k`` x sqrt(their spread^2 + the pixel's own wobble^2); or
    within ``k`` x its own wobble of the pixel's own low. Read from the AOI, not fixed (aoi160: lows around 0.18, the
    user's "a trough is usually around 0.2")."""
    from .radar_water import WATER_K
    from .vegetation_types import sustained_low

    k = WATER_K if k is None else k
    f = np.asarray(fit, dtype="float32")
    noise = np.nan_to_num(np.asarray(noise, dtype="float32"))
    own = sustained_low(f, pd.date_range("2000-01-01", periods=f.shape[0], freq="5D"), start="2000-01-01")
    ref = own[np.asarray(crop, dtype=bool) & np.isfinite(own)]
    if len(ref):
        med = np.median(ref)
        spread = 1.4826 * np.median(np.abs(ref - med))
        aoi = med + k * np.sqrt(spread ** 2 + noise ** 2)
    else:
        aoi = np.full(f.shape[1], -np.inf)
    return np.fmax(aoi, own + k * noise)


def troughs(fit, noise, k: float | None = None, level=None) -> np.ndarray:
    """(windows, pixels) True at the troughs of the smoothed curve: a window lower than the one before (or equal to it,
    so a flat bottom counts at its last window), lower than the one after, followed later by a rise of at least ``k`` x
    the pixel's own normal wobble (``noise``), and, with ``level`` (per pixel, :func:`bare_level`), at or below it: a
    trough of a standing crop is not a sowing (user, 30 Sep: aoi160 pixel 62535, 0.72 -> 0.57 -> 0.86 in July;
    pixel 59890, 0.69 -> 0.33 -> 0.57; both were sown in May, "a trough is usually around 0.2")."""
    from .radar_water import WATER_K

    k = WATER_K if k is None else k
    f = np.asarray(fit, dtype="float32")
    n = f.shape[0]
    with np.errstate(invalid="ignore"):
        down = np.zeros_like(f, dtype=bool)
        down[0] = True
        down[1:] = f[1:] <= f[:-1]
        up = np.zeros_like(f, dtype=bool)
        up[:-1] = f[1:] > f[:-1]
        after = np.full_like(f, -np.inf)
        after[:-1] = np.maximum.accumulate(np.where(np.isfinite(f), f, -np.inf)[::-1], axis=0)[::-1][1:]
        rises = after >= f + k * np.nan_to_num(np.asarray(noise, dtype="float32"))[None, :]
        low = np.ones_like(f, dtype=bool) if level is None else f <= np.asarray(level)[None, :]
    t = down & up & rises & low
    t[n - 1] = False
    return t


def sowing(fit, ndvi_raw, windows, crop, series=None, start: str = START, level=None) -> pd.DataFrame:
    """Per pixel: ``sowing_date``, ``period`` (see ``PERIODS``; 0 = not crop ground) and ``late_spell_date`` (the trough
    after mid-July that was checked with the radar, NaT if none).

    The sowing is the LAST TROUGH of the smoothed curve since ``start`` (:func:`troughs`): the low point just before
    the crop that stands in September, a second crop's trough included (user, 30 Sep, aoi160 pixel 59890: a crop in
    June, a trough at 0.33 around 20 July, the September crop after it). A trough after mid-July counts only when the
    radar agrees (:func:`radar_empty_at`) or two clear views at bare level lie within ``CLEAR_VIEWS_REACH`` windows of
    it; else the last trough up to mid-July. Without any trough (the curve only rises
    from 1 May) the lowest window is taken."""
    from .monsoon_rule import optical_noise

    w = pd.DatetimeIndex(windows)
    use = w >= pd.Timestamp(start)
    # the pixel's normal wobble from ALL its clear views: the few views since May include the crop's own rises and
    # falls and read far too large (pixel 59890: 0.21 instead of 0.04, its July trough was lost)
    noise = optical_noise(np.asarray(ndvi_raw))
    fit = np.asarray(fit, dtype="float32")[use]
    w = w[use]
    crop = np.asarray(crop, dtype=bool)
    level = bare_level(fit, noise, crop) if level is None else np.asarray(level, dtype=float)
    tr = troughs(fit, noise, level=level) & crop[None, :]
    trusted = int(np.searchsorted(w, pd.Timestamp(OPTICAL_TRUSTED_UNTIL), side="right"))
    last = last_true(tr)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        lowest = np.where(np.isfinite(fit).any(axis=0), np.nanargmin(np.where(np.isfinite(fit), fit, np.inf), axis=0), -1)
    last = np.where(crop & (last < 0), lowest, last)
    early = last_true(tr, upto=trusted)
    wd = w.to_numpy().astype("datetime64[D]")
    late = crop & (last >= trusted)
    late_day = np.where(late, wd[np.clip(last, 0, None)], np.datetime64("NaT"))
    agree = radar_empty_at(series, late_day) if series is not None and late.any() else np.zeros(len(crop), bool)
    # ... or two clear views at bare level within three windows (15 days) of the late trough: one view can be haze, two
    # are evidence (user, 1 Oct, aoi160 pixel 68729: 0.21 on 17 Aug, 0.19 on 6 Sep, 0.23 on 16 Sep, trough 1 Sep)
    raw = np.asarray(ndvi_raw, dtype=float)[use]
    near = np.abs(np.arange(len(w))[:, None] - last[None, :]) <= CLEAR_VIEWS_REACH
    with np.errstate(invalid="ignore"):
        seen_bare = (near & (raw <= level[None, :])).sum(axis=0)
    agree = agree | (late & (seen_bare >= 2))
    use_early = late & ~agree & (early >= 0)
    pick = np.where(use_early, early, last)
    has = crop & (pick >= 0)
    day = np.where(has, wd[np.clip(pick, 0, None)], np.datetime64("NaT")).astype("datetime64[D]")
    # the optical low counts only when at least two clear views at bare level saw it; else (a filled curve across a
    # cloudy gap, or one hazy view) the radar's own sowing is used (user, 1 Oct, aoi39 pixel 12617)
    near_pick = np.abs(np.arange(len(w))[:, None] - pick[None, :]) <= CLEAR_VIEWS_REACH
    with np.errstate(invalid="ignore"):
        seen_pick = (near_pick & (raw <= level[None, :])).sum(axis=0)
    rday = radar_sowing(series, start) if (RADAR_SOWING and series is not None) else \
        np.array([], dtype="datetime64[D]")
    if len(rday) != len(crop):
        rday = np.full(len(crop), np.datetime64("NaT"), dtype="datetime64[D]")
    by_radar = has & (seen_pick < 2) & ~np.isnat(rday)
    day = np.where(by_radar, rday, day)
    dt = pd.DatetimeIndex(day)
    month = np.asarray(dt.month)
    after_trusted = np.asarray(dt > pd.Timestamp(OPTICAL_TRUSTED_UNTIL))
    confirmed = np.where(by_radar, True, agree)
    period = np.zeros(len(crop), dtype="uint8")
    period[has & (month == 5)] = 1
    period[has & (month == 6)] = 2
    period[has & ~after_trusted & (month == 7)] = 3
    period[has & after_trusted & confirmed] = 4
    period[has & after_trusted & ~confirmed] = 5
    return pd.DataFrame({"sowing_date": day, "period": period, "late_spell_date": late_day, "radar_agrees": agree,
                         "source": np.where(by_radar, 2, np.where(has, 1, 0)).astype("uint8")})


def run(aoi_id: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late",
        out_root: str = "processed/_batch/s2_2026") -> pd.DataFrame:
    """Step 2 for one AOI (step 1 must exist); returns the acre table."""
    import rasterio

    from . import radar_water as rw

    stem = Path(out_root) / "rice_fresh" / f"aoi{aoi_id}" / f"aoi{aoi_id}_step"
    with rasterio.open(f"{stem}1_cover.tif") as ds:
        cover = ds.read(1)
        profile = ds.profile
    shape = cover.shape
    crop = (cover == 1).ravel()
    d = nd.load(aoi_id, out_root=series_root)
    fit = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    raw = d["ndvi5d_raw"].reshape(d["ndvi5d_raw"].shape[0], -1)
    s = sowing(fit, raw, d["windows"], crop, rw.read_series(aoi_id))
    del fit, raw
    nd.forget()
    doy = np.where(s["period"] > 0, pd.DatetimeIndex(s["sowing_date"]).dayofyear.fillna(0).to_numpy(), 0)
    with rasterio.open(f"{stem}2_sowing.tif", "w", **dict(profile, dtype="uint16", nodata=0, count=1)) as ds:
        ds.write(doy.astype("uint16").reshape(shape)[None])
    with rasterio.open(f"{stem}2_period.tif", "w", **dict(profile, dtype="uint8", nodata=0, count=1)) as ds:
        ds.write(s["period"].to_numpy().reshape(shape)[None])
    with rasterio.open(f"{stem}2_source.tif", "w", **dict(profile, dtype="uint8", nodata=0, count=1)) as ds:
        # 1 = sowing seen by the optical (two clear views), 2 = read from the radar (the optical did not see it)
        ds.write(s["source"].to_numpy().reshape(shape)[None])
    entries = "\n".join(f'        <paletteEntry value="{k}" color="{c}" alpha="255" label="{PERIODS[k]}"/>'
                        for k, c in COLOURS.items())
    Path(f"{stem}2_period.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
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
    p = s["period"].to_numpy()
    table = pd.DataFrame([{"period": k, "name": PERIODS[k], "acres": round(acres(int((p == k).sum())), 1)}
                          for k in PERIODS])
    late = s["late_spell_date"].notna().to_numpy()
    table.attrs["late_spells_acres"] = round(acres(int(late.sum())), 1)
    table.attrs["late_spells_radar_agrees_acres"] = round(acres(int((late & s["radar_agrees"].to_numpy()).sum())), 1)
    table.attrs["sowing_from_radar_acres"] = round(acres(int((s["source"].to_numpy() == 2).sum())), 1)
    table.to_csv(f"{stem}2.csv", index=False)
    preview(p.reshape(shape), s, cover, Path(f"{stem}2.png"), f"aoi{aoi_id}: step 2, sowing (last empty spell since 1 May)")
    return table


def preview(period, s, cover, path: Path, title: str) -> None:
    """Map of the sowing periods and a histogram of the sowing dates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    lut = np.ones((256, 3))
    lut[0] = matplotlib.colors.to_rgb("#d9d9d9")
    for k, c in COLOURS.items():
        lut[k] = matplotlib.colors.to_rgb(c)
    img = lut[period]
    img[cover == 2] = matplotlib.colors.to_rgb("#1b5e20")
    img[cover == 255] = 1.0
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.1]})
    axes[0].imshow(img, interpolation="nearest")
    axes[0].axis("off")
    handles = [Patch(color=COLOURS[k], label=PERIODS[k]) for k in PERIODS]
    handles += [Patch(color="#1b5e20", label="trees / orchards (step 1)"),
                Patch(color="#d9d9d9", label="not vegetation in September")]
    axes[0].legend(handles=handles, loc="lower left", fontsize=8)
    d = pd.DatetimeIndex(s.loc[s["period"] > 0, "sowing_date"])
    axes[1].hist(d, bins=pd.date_range("2026-05-01", "2026-09-30", freq="5D"), color="#4a7d4a")
    axes[1].axvline(pd.Timestamp(OPTICAL_TRUSTED_UNTIL), color="#b45f06", lw=2,
                    label="mid-July: later empty spells need the radar")
    axes[1].set_ylabel("crop pixels")
    axes[1].set_title("Sowing date (last empty spell)")
    axes[1].legend(frameon=False)
    fig.suptitle(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def explain(aoi_id: int, pid: int, series_root: str = "processed/_batch/s2_2026_hyb40m1late") -> pd.DataFrame:
    """One pixel, window by window since 1 May: smoothed and clear NDVI, the troughs of the smoothed curve, and the
    sowing chosen. Why: to answer "why is this pixel's sowing on that date?" from the numbers (user, 30 Sep)."""
    from . import radar_water as rw
    from .monsoon_rule import optical_noise

    d = nd.load(aoi_id, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    fit = d["ndvi5d"].reshape(len(w), -1)[:, [pid]]
    raw = d["ndvi5d_raw"].reshape(len(w), -1)[:, [pid]]
    use = w >= pd.Timestamp(START)
    # the AOI's bare level needs the AOI's crop ground: read it from the step-1 cover of the whole AOI
    import rasterio

    with rasterio.open(Path("processed/_batch/s2_2026/rice_fresh") / f"aoi{aoi_id}" / f"aoi{aoi_id}_step1_cover.tif") as ds:
        crop_all = (ds.read(1) == 1).ravel()
    fit_all = d["ndvi5d"].reshape(len(w), -1)[use]
    noise_all = optical_noise(d["ndvi5d_raw"].reshape(len(w), -1))
    level = bare_level(fit_all, noise_all, crop_all)[pid]
    tr = troughs(fit[use], optical_noise(raw), level=np.array([level]))[:, 0]
    s = sowing(fit, raw, w, np.array([True]), [(dd, {p: v[:, [pid]] for p, v in flat.items()})
                                                for dd, flat in rw.read_series(aoi_id)], level=np.array([level]))
    t = pd.DataFrame({"window": w[use].date, "fit": fit[use, 0].round(2), "clear_view": raw[use, 0].round(2),
                      "trough": tr})
    t.attrs.update(bare_level=round(float(level), 3), sowing=s.loc[0, "sowing_date"], period=int(s.loc[0, "period"]),
                   late_trough=s.loc[0, "late_spell_date"], radar_agrees=bool(s.loc[0, "radar_agrees"]),
                   sowing_from="radar" if int(s.loc[0, "source"]) == 2 else "optical")
    return t


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--series-root", default="processed/_batch/s2_2026_hyb40m1late")
    p.add_argument("--out-root", default="processed/_batch/s2_2026")
    p.add_argument("--explain", type=int, metavar="PIXEL_ID", help="print one pixel's windows and its sowing, then stop")
    args = p.parse_args(argv)
    if args.explain is not None:
        t = explain(args.aoi, args.explain, args.series_root)
        print(t.to_string(index=False))
        print({k: str(v) for k, v in t.attrs.items()})
        return 0
    t = run(args.aoi, args.series_root, args.out_root)
    print(t.to_string(index=False))
    print(f"empty spells after mid-July: {t.attrs['late_spells_acres']} ac, radar agrees on "
          f"{t.attrs['late_spells_radar_agrees_acres']} ac; sowing read from the radar: {t.attrs['sowing_from_radar_acres']} ac")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
