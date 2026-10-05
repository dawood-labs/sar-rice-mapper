"""Per-pixel radar confirmation of the water at the optical transplant date.

Why
---
Rice is transplanted into water, always; what varies is how much and for how long. The optical
water test (LSWI) sees deep, prolonged flooding well and shallow, short flooding badly: on the first
field-plot delivery it found water in 94-100 % of delta plots and in 0-13 % of the capital-region
plots, where the radar found it in 91 %. So water is confirmed here from Sentinel-1, per pixel, the
way ``sar_water_check`` confirmed it per field plot:

* line every pixel's radar series up on **that pixel's** optical trough date;
* **dry level** = median backscatter 40 to 15 days before the trough (after the previous harvest,
  bare and dry, the brightest the field gets);
* **flood level** = the minimum 10 days before to 15 days after the trough;
* **dip** = dry - flood, in dB, per polarisation, the best over the season's tracks.

A dip of ``DIP_MIN_DB`` or more in either polarisation is ``radar_wet``. Where the season run holds
no dates in the dry window (a trough at the very start of the run), the pixel is ``not checkable``
rather than dry: the map must not confuse "no water" with "could not look".

The radar stacks come from the season run of the Sentinel-1 pipeline (stages 1-5), read as 5x5
linear-power means in dB by ``sar_curve.read_track``; the radar grid must be the optical grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pixel_report as pr
from . import sar_curve

DRY_WINDOW = (-40, -15)
FLOOD_WINDOW = (-10, 15)
DIP_MIN_DB = 3.0
#: The flood-window VH the v1 dip must reach (see ``pixel_dips``). -17 and not -18: in the dry-zone
#: plot AOI 10 % of the surveyed rice floods to only -17.5 dB (shallow water on light soils), while
#: the harvest dips this guard is for (a canopy at -13 dB cut to -16) stay above it; the optical
#: trough (bare for 40 days) and the fit floor already rule out the deeper harvest-drop cases.
V1_VH_MAX = -17.0


def dips_for_track(dates, cube, trough, dry=DRY_WINDOW, flood=FLOOD_WINDOW):
    """Dry level, flood level and dip per pixel for one track and one polarisation.

    ``dates`` (n_dates), ``cube`` (n_dates, n_pixels) in dB, ``trough`` (n_pixels) datetime64 with
    NaT where the pixel has no trough. Returns ``(dry_level, flood_level, dip)``; NaN where a window
    holds no date.
    """
    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    t = np.asarray(trough).astype("datetime64[D]")
    gap = (day[:, None] - t[None, :]) / np.timedelta64(1, "D")          # NaT -> nan
    with np.errstate(invalid="ignore"):
        in_dry = (gap >= dry[0]) & (gap <= dry[1])
        in_flood = (gap >= flood[0]) & (gap <= flood[1])
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # all-NaN windows are the "not checkable" case
        dry_level = np.nanmedian(np.where(in_dry, cube, np.nan), axis=0)
        flood_level = np.nanmin(np.where(in_flood, cube, np.nan), axis=0)
    return dry_level, flood_level, dry_level - flood_level


def pixel_dips(aoi_id: int, trough, grid: dict, season_key: str = "monsoon2026", window: int = 5) -> pd.DataFrame:
    """Best dip over tracks per pixel, for VV and VH, plus whether the check was possible at all."""
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    for key in ("crs", "x0", "y0", "width", "height"):
        if loc["grid"][key] != grid[key]:
            raise ValueError(f"radar grid differs from the optical grid in {key}")
    n_pix = int(grid["width"]) * int(grid["height"])
    best = {pol: np.full(n_pix, np.nan) for pol in ("VV", "VH")}
    vh_flood = np.full(n_pix, np.nan)
    checkable = np.zeros(n_pix, dtype=bool)
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        for pol in ("VV", "VH"):
            _, flood_level, dip = dips_for_track(dates, cubes[pol].reshape(len(dates), -1), trough)
            best[pol] = np.fmax(best[pol], dip)
            if pol == "VH":
                vh_flood = np.fmin(vh_flood, flood_level)
            checkable |= np.isfinite(dip)
    out = pd.DataFrame({"VV_dip": best["VV"], "VH_dip": best["VH"], "VH_flood_level": vh_flood,
                        "radar_checkable": checkable})
    with np.errstate(invalid="ignore"):
        # the same VV/VH consistency as water_evidence: the other polarisation must not have risen
        # by more than -OTHER_POL_DROP_MIN over the flood window (unknown is not held against it)
        vv_ok = (out["VV_dip"] >= DIP_MIN_DB) & ~(out["VH_dip"] < OTHER_POL_DROP_MIN)
        vh_ok = (out["VH_dip"] >= DIP_MIN_DB) & ~(out["VV_dip"] < OTHER_POL_DROP_MIN)
        # the dip must end on dark ground (fix plan, issue 9): a 3 dB fall from a bright canopy or
        # garden (-13 -> -16 dB) is a harvest or rain, not transplanting water
        dark = ~(out["VH_flood_level"] > V1_VH_MAX)
    out["radar_wet"] = checkable & (vv_ok | vh_ok) & dark
    return out


def season_run_exists(aoi_id: int, season_key: str = "monsoon2026") -> bool:
    """True when the AOI has a config and a stacked run for the season."""
    try:
        loc = pr.locate(aoi_id, 0, season_key=season_key)
    except Exception:  # noqa: BLE001 - no config for that season is the normal "no" here
        return False
    from pathlib import Path

    return any((Path(loc["run"]) / "stack" / f"track_{t['track_id']}" / "stack_VH.vrt").exists()
               for t in loc["cfg"]["s1"]["tracks"])


FLOOD_SPAN = (-75, 0)       # days relative to the optical climb in which the flooding is searched
REF_WINDOW = (45, 12)       # the "before" level: median of dates 45 to 12 days before each date


def local_drops(dates, cube, ref=REF_WINDOW) -> np.ndarray:
    """For every date, how far the value sits below the pixel's own level of the weeks before, in dB.

    ``drop[i] = median(values dated t_i - ref[0] .. t_i - ref[1]) - value[i]``; NaN where no earlier
    date falls in that window. Why a *local* reference: a flood is a sudden fall from whatever the
    field was just before (dry soil, a wet field, weeds), so it is measured against the preceding
    weeks, not against a fixed window tied to an optical date that may be weeks off.
    """
    import warnings

    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    out = np.full(cube.shape, np.nan, dtype="float32")
    for i in range(len(day)):
        lag = (day[i] - day) / np.timedelta64(1, "D")
        before = (lag >= ref[1]) & (lag <= ref[0])
        if before.any():
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[i] = np.nanmedian(cube[before], axis=0) - cube[i]
    return out


def flood_search(dates, cube, anchor, span=FLOOD_SPAN, ref=REF_WINDOW, persist_db: float = 2.0,
                 persist_days: int = 14):
    """Largest local drop inside ``anchor + span`` per pixel, its date, and a persistence count.

    ``anchor`` is datetime64 per pixel (NaT: skipped). ``persist`` counts the dates within
    ``persist_days`` of the best one that are also ``persist_db`` or more below their own earlier
    level: transplanting water stands for weeks, so a real flood is seen on more than one pass,
    while a single speckled date is not. Returns ``(best_drop, best_date, persist)``.
    """
    import warnings

    drops = local_drops(dates, cube, ref)
    day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
    a = np.asarray(anchor).astype("datetime64[D]")
    rel = (day[:, None] - a[None, :]) / np.timedelta64(1, "D")
    with np.errstate(invalid="ignore"):
        inside = (rel >= span[0]) & (rel <= span[1])
    masked = np.where(inside, drops, np.nan)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        best = np.nanmax(masked, axis=0)
    has = np.isfinite(best)
    arg = np.where(has, np.nanargmax(np.where(np.isfinite(masked), masked, -np.inf), axis=0), 0)
    best_day = np.where(has, day[arg], np.datetime64("NaT"))
    near = np.abs((day[:, None] - best_day[None, :]) / np.timedelta64(1, "D")) <= persist_days
    with np.errstate(invalid="ignore"):
        persist = (near & inside & (drops >= persist_db)).sum(axis=0)
    return best, best_day, np.where(has, persist, 0)


def pixel_floods(aoi_id: int, anchor, pixels=None, season_key: str = "monsoon2026", window: int = 5,
                 span=FLOOD_SPAN) -> pd.DataFrame:
    """:func:`flood_search` over every track and polarisation, best per pixel.

    Returns ``flood_drop`` (dB), ``flood_date``, ``flood_pol``, ``flood_track`` and ``flood_persist``
    (the persistence count of the best track and polarisation). ``pixels`` limits the work to a
    subset (flat indices); ``anchor`` then has one date per selected pixel.
    """
    loc = pr.locate(aoi_id, 0, season_key=season_key)
    n = int(loc["grid"]["width"]) * int(loc["grid"]["height"])
    pix = np.arange(n) if pixels is None else np.asarray(pixels)
    best = np.full(len(pix), -np.inf)
    out = {"flood_date": np.full(len(pix), np.datetime64("NaT"), dtype="datetime64[D]"),
           "flood_pol": np.full(len(pix), "", dtype=object), "flood_track": np.full(len(pix), "", dtype=object),
           "flood_persist": np.zeros(len(pix), dtype=int)}
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        for pol in ("VV", "VH"):
            b, when, persist = flood_search(dates, cubes[pol].reshape(len(dates), -1)[:, pix], anchor, span)
            better = np.isfinite(b) & (b > best)
            best = np.where(better, b, best)
            out["flood_date"] = np.where(better, when, out["flood_date"])
            out["flood_pol"] = np.where(better, pol, out["flood_pol"])
            out["flood_track"] = np.where(better, track, out["flood_track"])
            out["flood_persist"] = np.where(better, persist, out["flood_persist"])
    frame = pd.DataFrame(out, index=pix)
    frame.insert(0, "flood_drop", np.where(np.isfinite(best), best, np.nan))
    return frame


# --- water evidence v2 (docs/11): the flood searched over the whole bare period -------------------
LONG_SPAN = (-100, -5)       # days relative to the optical climb
FLOOD_DROP_MIN = 4.0         # dB below the field's own level of the weeks before
FLOOD_VH_MAX = -19.0         # VH on the flood pass; plots: 90 % at or below -19.4 dB, trees and bare never
FLOOD_NDVI_MAX = 0.5         # no canopy on the field at the flood date (a harvest is also a drop)
SUPPORT_MIN = 2              # passes (any track) within 14 days that also show the drop
VEG_VH_MIN = -15.0           # VH around the trough above this: a canopy or buildings, not a bare field
VEG_FLOOD_VH_MIN = -17.0     # ... and never darker than this in the whole radar season
FLOOD_EARLIEST = "2026-05-15"  # monsoon water only: the plots' floods all came after this (99 %+)
#: Shallow water (fix plan, issue 6): a pass 1 dB short of ``FLOOD_VH_MAX`` still counts when the
#: drop is larger (>= ``SHALLOW_DROP_MIN``) and the flood is seen on more passes (>= ``SHALLOW_SUPPORT_MIN``).
#: Why: in the dry-zone plot AOI 11 % of the surveyed rice sat in class 3 with a 5.5 dB drop to
#: VH -18.1 seen on five passes; dark dry soil never shows such a drop from a brighter level.
SHALLOW_VH_MAX = -18.0
SHALLOW_DROP_MIN = 5.0
SHALLOW_SUPPORT_MIN = 4
#: The radar canopy after a flood must reach this VH: still-dark ground (VH below it) is water or
#: mud, not a crop (a pond's VH went from -32 to -24 dB and looked like a "rise").
RADAR_CANOPY_VH_MIN = -18.0
#: v1 dip only: the other polarisation may not RISE by more than -OTHER_POL_DROP_MIN over the flood
#: window (a drop of -1 dB = a rise of 1 dB). Why (fix plan, issue 15): on 11 Jun 2026 one pass read
#: VH 5-20 dB low on dry fields while VV was at its brightest; water never does that. The v2 flood
#: does NOT use it any more (round 2, aoi110): in a freshly transplanted paddy VV often RISES 2-4 dB
#: (double bounce off the seedlings) while VH falls into the water, so the test refused real floods;
#: v2 is protected instead by its support test (a second pass within 14 days) and by the pass
#: screening of ``final_audit.screen_passes``, which now removes broken passes before any rule runs.
#: ``flood_other_drop`` is still reported.
OTHER_POL_DROP_MIN = -1.0
#: Support for the flood pass (fix plan, stage 2.4, issue 3): a second pass within 14 days that also
#: sits >= 2 dB below its own earlier level, in EITHER polarisation of any track (before: the same
#: polarisation only; a real June flood seen at -28 dB VH on one track failed because the other
#: track's VH drop was 1.6 dB while its VV had dropped 3 dB). A pass that is very dark AND a very
#: large drop (``DEEP_FLOOD_VH_MAX`` / ``DEEP_FLOOD_DROP_MIN``) needs no second pass: nothing but
#: open water reads that dark on a field that was bright weeks before, and broken passes are now
#: screened (issue 15) and must agree between polarisations.
DEEP_FLOOD_VH_MAX = -24.0
DEEP_FLOOD_DROP_MIN = 8.0
#: Clear observations from ``RAW_BEFORE_WINDOWS`` windows before to ``RAW_AFTER_WINDOWS`` after the
#: flood date decide whether a canopy stood on the field then (see ``water_evidence``). A canopy
#: seen in the 10 days BEFORE the drop is the signature of a harvest (the cut is quick) or of trees;
#: 25 days was too long: in a double-crop AOI the summer rice was green until 8 June and the
#: monsoon flood came on 23 June, so 97 % of a paddy block was refused its water.
RAW_BEFORE_WINDOWS = 2
RAW_AFTER_WINDOWS = 0          # a canopy AFTER the drop is the crop growing (one field read 0.56 nine days after its flood)
#: ``bare_near_flood``: a clear observation without canopy (NDVI <= ``BARE_SEEN_NDVI``) from
#: ``BARE_SEEN_BEFORE_WINDOWS`` windows before the flood to ``BARE_SEEN_AFTER_WINDOWS`` after it or
#: to the optical climb, whichever is later. Why: the radar is read over a 5 x 5 box, so a tree line
#: next to flooded paddies also "floods" in the radar; the pixel's own optical history must show
#: ground without a canopy at some point around the flood before the radar alone may call it rice
#: (the radar-trough path). No observation at all in that span counts as unknown, not as a canopy.
#: The level is the rule's canopy level (``monsoon_rule.CANOPY_MIN`` 0.50) less a margin: a double
#: crop's first clear view after its August flood is often 5-6 weeks later, when the seedlings
#: already read 0.40-0.45 (round 2, aoi110); a tree line reads 0.6 and more on every clear date.
BARE_SEEN_NDVI = 0.45
BARE_SEEN_BEFORE_WINDOWS = 18      # 90 days: the last dry-season view of a field can be that old under monsoon cloud
BARE_SEEN_AFTER_WINDOWS = 6        # 30 days, the minimum; the window runs on to the climb
#: ``vh_end``: median VH of the passes in the last ``END_DAYS`` before the series end (any track);
#: ``radar_canopy_rise`` = vh_end - flood_vh. Why: a young crop transplanted in August is often
#: seen clear only through haze by the map date, but the radar sees its canopy: VH climbs from the
#: water (-20 to -25 dB) to -17 / -15 within 3-5 weeks. A rise of 4 dB is well above pass-to-pass
#: noise on a 5 x 5 box (~1 dB).
END_DAYS = 20
#: Own-level canopy measure (fix round 3, stage 2.2; reported only, the rule does not read it yet): on the
#: track where the flood was found, ``vh_end_own`` = median VH of the last ``END_PASSES`` passes up to the
#: series end (counted in passes, not days), ``vh_dry_own`` = the pixel's VH level of the weeks before the
#: flood (flood VH + the VH drop on the flood pass), ``vh_noise_own`` = the pixel's own pass-to-pass scatter
#: on that track (1.4826 x median |second difference| / sqrt 6: a steady rise or fall of a growing crop
#: cancels in the second difference, and the few real jumps are ignored by the median). ``recovery`` = (end - flood) / (dry - flood): 0 still at the water, 1 back at its own dry
#: level. ``rise_z`` = (end - flood) / (noise x sqrt(1/END_PASSES + 1)): the rise in units of its own noise.
END_PASSES = 2
#: Water test v3 (fix round 3; docs/fix_plan_2026_round3.md). ``WATER_K`` is the one unitless factor: how many times
#: its own pass-to-pass noise a pixel must fall to count. ``REF_PASSES``: the "level before" is the median of the
#: preceding passes of the same track (3 passes = one month at the 12-day revisit; a count of orbits, not a dB).
WATER_K = 2.0
REF_PASSES = 3
#: The AOI's water level of a pass is the median of its clearly flooded pixels on that pass; with fewer than this many
#: (a statistical sample size, not a crop threshold) the median of all their passes on the track is used instead.
WATER_PEERS_MIN = 50
#: Round 3 (user 30 Sep, aoi160_003251 / 003253): a single dark pass counts as water when it is the pixel's own darkest
#: pass of the season on that track (within k x its noise) AND the optical shows the field at its own bare low at that
#: time (sowing). Short transplanting water can drain within one revisit. Off by default until tested.
SINGLE_DIP_WATER = False
#: Round 3 (user 30 Sep, aoi160_001045, issue 36): water counts only around the pixel's own sowing, i.e. on passes where
#: its fitted NDVI is below half-way from its own season low to its own season peak (``sowing_seen`` of
#: :func:`flood_v3`). Why: a mid-August radar dip under a 0.84 canopy (VV down on both tracks) was taken as the flood of a
#: field sown in May; water under a standing crop is not transplanting water. Pixel-relative, no NDVI cut-off.
FLOOD_AT_SOWING = True


def own_levels(tracks, track_of, end_day, passes: int = END_PASSES) -> dict:
    """Per pixel, on the track given by ``track_of`` (-1 = none): median VH of the last ``passes`` passes up to
    ``end_day`` (+3 days, as ``vh_end``) and the pixel's own robust pass-to-pass noise on that track.

    ``tracks``: ``[(day, drops, flat), ...]`` as built in :func:`water_evidence`. NaN where there is no track.
    """
    import warnings

    n = len(track_of)
    end = np.full(n, np.nan)
    noise = np.full(n, np.nan)
    for ti, (day, _, flat) in enumerate(tracks):
        use = track_of == ti
        if not use.any():
            continue
        vh = flat["VH"][:, use]
        upto = day <= end_day + np.timedelta64(3, "D")
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            last = np.where(np.isfinite(vh[upto]), vh[upto], np.nan)
            # the last ``passes`` finite values of each pixel
            ranks = np.cumsum(np.isfinite(last[::-1]), axis=0)[::-1]
            end[use] = np.nanmedian(np.where(ranks <= passes, last, np.nan), axis=0)
            # second differences: a steady trend (a growing crop) cancels, only the pass-to-pass scatter is left
            noise[use] = 1.4826 * np.nanmedian(np.abs(np.diff(last, n=2, axis=0)), axis=0) / np.sqrt(6.0)
    return {"end": end, "noise": noise}


def read_series(aoi_id: int, season_key: str = "monsoon2026", window: int = 5, run_id: str | None = None) -> list:
    """Every track of the AOI's radar season as ``[(dates, {"VV": (dates, pixels), "VH": ...}), ...]`` in dB.

    ``run_id``: the radar run to read; None = the run the analysis is pinned to (``config.analysis_run_dir``), so a
    newer run never changes an AOI's results until it is chosen on purpose."""
    loc = pr.locate(aoi_id, 0, season_key=season_key, run_id=run_id)
    series = []
    for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
        dates, cubes = sar_curve.read_track(loc, track, window)
        series.append((dates, {p: cubes[p].reshape(len(dates), -1) for p in ("VV", "VH")}))
    return series


def bare_near(when, climb, win, ndvi_raw) -> np.ndarray:
    """True where a clear view without canopy (NDVI <= ``BARE_SEEN_NDVI``) exists from ``BARE_SEEN_BEFORE_WINDOWS``
    before the flood ``when`` up to the later of ``BARE_SEEN_AFTER_WINDOWS`` after it and the optical climb; True
    also where there is no observation (unknown is not held against a field). Without ``ndvi_raw``: all True."""
    import warnings

    n = len(when)
    if ndvi_raw is None:
        return np.ones(n, dtype=bool)
    when = np.asarray(when).astype("datetime64[D]")
    climb = np.asarray(climb).astype("datetime64[D]")
    has = ~np.isnat(when)
    idx = np.clip(np.searchsorted(win, np.where(has, when, win[0])), 0, len(win) - 1)
    step = np.arange(len(win))[:, None]
    rel = step - idx[None, :]
    # ... up to the optical climb when that is later: between the flood and the climb the field
    # carries no canopy, so any clear view there counts (a double crop's harvest-to-flood gap
    # often falls inside a cloud gap, and its first bare view comes 5-6 weeks after the flood)
    climb_idx = np.clip(np.searchsorted(win, np.where(np.isnat(climb), win[-1], climb)), 0, len(win) - 1)
    upto = np.maximum(idx + BARE_SEEN_AFTER_WINDOWS, climb_idx)
    around = (rel >= -BARE_SEEN_BEFORE_WINDOWS) & (step <= upto[None, :])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seen_min = np.nanmin(np.where(around, ndvi_raw, np.nan), axis=0)
    return ~(seen_min > BARE_SEEN_NDVI)           # unknown (no observation) stays True


def pass_noise(x) -> np.ndarray:
    """Robust pass-to-pass noise per pixel of a (passes, pixels) dB cube: 1.4826 x median |second difference| / sqrt 6.
    A steady rise or fall (a growing crop) cancels in the second difference; the median ignores the few real jumps."""
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return 1.4826 * np.nanmedian(np.abs(np.diff(x, n=2, axis=0)), axis=0) / np.sqrt(6.0)


def _nearest_other(per, ti, key):
    """For every pass of track ``ti``: is ``key`` True on the nearest pass of any other track? (passes, pixels)"""
    a = per[ti]
    other = np.zeros(a[key].shape, dtype=bool)
    for tj, b in enumerate(per):
        if tj != ti:
            j = np.abs(a["day"][:, None] - b["day"][None, :]).argmin(axis=1)
            other |= b[key][j]
    return other


def rise_since(series, since_day, end_day, k_passes: int = END_PASSES, pols=("VV", "VH")) -> np.ndarray:
    """Per pixel: how much the radar rose AFTER ``since_day`` (datetime64 per pixel, NaT = unknown), in own-noise units,
    all tracks together (sum / sqrt(tracks)); on each track and polarisation, the last pass on or before the day
    against the median of the last ``k_passes`` passes up to ``end_day``, the larger of VV and VH.

    Why (user, aoi13_000737 / 000471): a clear view of water from mid-August says nothing about a crop the radar shows
    rising from late August; only a rise after the clear view can overrule it. ``pols=("VH",)`` reads the canopy only
    (volume scattering); VV also follows rain on bare soil."""
    import warnings

    since = np.asarray(since_day).astype("datetime64[D]")
    end = np.datetime64(end_day, "D") + np.timedelta64(3, "D")
    n = len(since)
    z_sum = np.zeros(n)
    count = np.zeros(n, dtype=int)
    cols = np.arange(n)
    for dates, flat in series:
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        use = day <= end
        idx = np.searchsorted(day[use], np.where(np.isnat(since), day[0], since), side="right") - 1
        ok = ~np.isnat(since) & (idx >= 0)
        best = np.full(n, np.nan)
        for pol in pols:
            x = np.asarray(flat[pol], dtype="float32")[use]
            noise = pass_noise(x)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ranks = np.cumsum(np.isfinite(x[::-1]), axis=0)[::-1]
                e = np.nanmedian(np.where(ranks <= k_passes, x, np.nan), axis=0)
                at = x[np.clip(idx, 0, len(x) - 1), cols]
                z = (e - at) / (noise * np.sqrt(1.0 + 1.0 / k_passes))
            best = np.fmax(best, np.where(ok, z, np.nan))
        seen = np.isfinite(best)
        z_sum += np.where(seen, best, 0.0)
        count += seen
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, z_sum / np.sqrt(np.maximum(count, 1)), np.nan)


def brightened_at(series, day, k: float = WATER_K, after_passes: int = 2, until=None) -> np.ndarray:
    """Per pixel: did VH brighten at sowing? For each pass j: the median of passes j .. j+``after_passes``-1 against the
    median of the ``after_passes`` passes before j, in the pixel's own noise units; the largest such rise over the passes
    from the one nearest ``day`` to the one nearest ``until`` (the sowing spell: last bare view to first green view; the
    sowing itself often falls in a cloud gap), all tracks together (sum of the tracks' largest rises / sqrt(tracks)).
    True at ``k`` or more.

    Why (user 30 Sep, aoi160_004992 / 003930): a paddy goes dark under the transplanting water; a crop sown into dry soil
    brightens at once (rain on tilled soil, the young canopy), so a bright start argues against water."""
    import warnings

    day = np.asarray(day).astype("datetime64[D]")
    until = day if until is None else np.asarray(until).astype("datetime64[D]")
    until = np.where(np.isnat(until), day, until)
    n = len(day)
    z_sum = np.zeros(n)
    count = np.zeros(n, dtype=int)
    for dates, flat in series:
        d = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        x = np.asarray(flat["VH"], dtype="float32")
        noise = pass_noise(x)
        dd = np.where(np.isnat(day), d[0], day)
        uu = np.where(np.isnat(until), dd, until)
        j0 = np.abs(d[:, None] - dd[None, :]).argmin(axis=0)
        j1 = np.abs(d[:, None] - uu[None, :]).argmin(axis=0)
        best = np.full(n, -np.inf)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j in range(after_passes, len(d) - after_passes + 1):
                post = np.nanmedian(x[j:j + after_passes], axis=0)
                pre = np.nanmedian(x[j - after_passes:j], axis=0)
                z = (post - pre) / (noise * np.sqrt(2.0 / after_passes))
                inside = (j >= j0) & (j <= np.maximum(j1, j0))
                best = np.where(inside & np.isfinite(z), np.fmax(best, z), best)
        seen = ~np.isnat(day) & np.isfinite(best)
        z_sum += np.where(seen, best, 0.0)
        count += seen
    with np.errstate(invalid="ignore", divide="ignore"):
        zall = np.where(count > 0, z_sum / np.sqrt(np.maximum(count, 1)), np.nan)
    return zall >= k

def optical_water_between_passes(day, water_seen) -> np.ndarray:
    """(passes, pixels): a clear optical view showed open water (NDVI below 0) between this pass and the next pass of
    the same track. ``water_seen`` = (window start dates, bool (windows, pixels)); the interval follows the orbit."""
    w_day, seen = water_seen
    w_day = np.asarray(w_day).astype("datetime64[D]")
    cum = np.vstack([np.zeros((1, seen.shape[1]), dtype="int32"), np.cumsum(seen, axis=0, dtype="int32")])
    lo = np.searchsorted(w_day, day)
    hi = np.searchsorted(w_day, np.r_[day[1:], day[-1] + (day[-1] - day[-2] if len(day) > 1 else np.timedelta64(1, "D"))])
    return (cum[hi] - cum[lo]) > 0


def flood_v3(series, end_day, season_start: str = FLOOD_EARLIEST, k: float = WATER_K,
             ref_passes: int = REF_PASSES, min_peers: int = WATER_PEERS_MIN, water_seen=None,
             bare_seen=None, sowing_seen=None) -> pd.DataFrame:
    """Water at transplanting, judged by each pixel against itself AND against the same AOI's clearly flooded fields.

    Why: the v1/v2 tests used fixed cut-offs (3 / 4 / 5 / 8 dB, VH -17 / -19 / -24, 2 passes in 14 days) and v1 was tied
    to the optical trough, which cloud or haze can put on the wrong date. A pixel's own history alone is not enough
    either (round 3): a summer-rice field holds its own March water in the "dry" season, and some dry soil is nearly as
    dark as water. So two witnesses, both taken from the data, per track and polarisation:

    * the pixel itself: a sudden fall of ``k`` x its own pass-to-pass noise below its ``ref_passes`` preceding passes;
    * darkness: as dark as this AOI's clearly flooded fields on the same pass (their median, within ``k`` x the combined
      scatter of the pixel's own noise and the flooded fields' spread): the AOI's own water level of that date and track. The clear cases are the pixels that fall AND end ``k`` x
      noise below their own pre-monsoon median (at least ``min_peers`` of them, else all their passes on the track).
      "Below its own dry season" never decides alone: a field with a summer crop in the dry season falls below it at
      harvest (bare soil), and a summer-rice field already holds water in it. A clear optical view of open water (NDVI
      below 0) between this pass and the next (``water_seen``) is a third witness: shallow water reads a little brighter
      in the radar than the AOI's deep floods;
    * confirmation: the next pass of the same track is also dark, or the nearest pass of another track is, or a clear
      optical view shows the water (another sensor). One track is enough; a track that shows nothing is unknown, not
      "against".
    * around sowing only (``sowing_seen`` = (window start dates, bool (windows, pixels)), see ``FLOOD_AT_SOWING``): a
      candidate pass must fall in a window where the field was at its sowing (low, not under a canopy). The AOI's water
      level is still taught by all passes.

    The flood is the confirmed candidate followed by the largest rise to the series end (own-noise units), over all
    tracks, VV or VH (the transplanting water comes before the crop; a late flood on a grown crop, or a harvest before
    the water, is followed by less); its date is the first pass of its dark run. Returns per pixel: ``flood_ok``,
    ``flood_date``, ``flood_vh``, ``flood_drop`` (dB below the preceding passes, deciding polarisation), ``flood_pol``,
    ``flood_z`` (fall in own-noise units), ``flood_by_peers`` (True when the pixel was NOT also below its own dry season),
    ``rise_after_flood_z``, ``flood_track``, ``flood_tracks``, ``support`` (1 same track, 2 other track, 3 both),
    ``flood_by_pattern`` (no peer-confirmed flood; a fall below its own dry season held for two passes: water only if the
    crop then rises, see ``monsoon_rule.radar_canopy_own``),
    ``rise_z_all`` (rise after the water summed over every track that saw it / sqrt(tracks)), ``vh_premonsoon`` and
    ``vh_end_track`` (VH levels on the flood track).
    """
    import warnings

    start = np.datetime64(season_start)
    end = np.datetime64(end_day, "D") + np.timedelta64(3, "D")
    per = []
    for dates, flat in series:                      # 1. every pixel against itself
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        pre = day < start
        t = {"day": day, "in_season": ((day >= start) & (day <= end))[:, None]}
        for pol in ("VV", "VH"):
            x = np.asarray(flat[pol], dtype="float32")
            noise = pass_noise(x[day <= end]).astype("float32")
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                p_level = np.nanmedian(x[pre], axis=0) if pre.any() else np.full(x.shape[1], np.nan, "float32")
                n_pre = np.maximum(np.isfinite(x[pre]).sum(axis=0), 1) if pre.any() else np.ones(x.shape[1])
                ref = np.full(x.shape, np.nan, dtype="float32")
                for i in range(1, len(day)):
                    ref[i] = np.nanmedian(x[max(0, i - ref_passes):i], axis=0)
                upto = x[day <= end]
                ranks = np.cumsum(np.isfinite(upto[::-1]), axis=0)[::-1]
                e_level = np.nanmedian(np.where(ranks <= END_PASSES, upto, np.nan), axis=0)
                t[pol] = {"x": x, "noise": noise, "p": p_level, "e": e_level,
                          "fall": ((ref - x) / (noise * np.sqrt(1.0 + 1.0 / ref_passes))).astype("float32"),
                          "own_dark": (p_level - x) / (noise * np.sqrt(1.0 + 1.0 / n_pre)) >= k,
                          "rise": ((e_level - x) / (noise * np.sqrt(1.0 + 1.0 / END_PASSES))).astype("float32"),
                          "drop": ref - x}
        with np.errstate(invalid="ignore"):
            t["dark_own"] = t["VV"]["own_dark"] | t["VH"]["own_dark"]
        per.append(t)
    for ti, t in enumerate(per):                    # 2. the AOI's clear cases give its water level per pass
        # the teaching set must be unmistakable water: both polarisations fall and sit below the dry season (a harvest
        # mostly drops VH only), the next pass of the same track stays dark AND, where the AOI has another track, that
        # track sees it too (one speckled pass or one track's artefact cannot teach the level)
        persist = np.zeros_like(t["dark_own"])
        persist[:-1] = t["dark_own"][1:]
        confirmed = persist & (_nearest_other(per, ti, "dark_own") if len(per) > 1 else True)
        with np.errstate(invalid="ignore"):
            both = ((t["VV"]["fall"] >= k) & t["VV"]["own_dark"] & (t["VH"]["fall"] >= k) & t["VH"]["own_dark"])
        for pol in ("VV", "VH"):
            c = t[pol]
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                clear = t["in_season"] & both & confirmed
                vals = np.where(clear, c["x"], np.nan)
                n_clear = np.isfinite(vals).sum(axis=1)
                level = np.nanmedian(vals, axis=1)
                spread = 1.4826 * np.nanmedian(np.abs(vals - level[:, None]), axis=1)
                if np.isfinite(vals).any():
                    all_level = np.nanmedian(vals)
                    all_spread = 1.4826 * np.nanmedian(np.abs(vals - all_level))
                else:
                    all_level = all_spread = np.nan
                few = n_clear < min_peers
                level = np.where(few, all_level, level)
                spread = np.where(few, all_spread, spread)
                # water is not one value: fields differ in depth and the pixel has its own noise; both widen the band
                band = np.sqrt(c["noise"] ** 2 + spread[:, None] ** 2)
                c["peer_gap"] = ((c["x"] - level[:, None]) / band).astype("float32")     # >0: brighter than the water
                c["peer_level"] = level
                c["peer_spread"] = spread
                c["peer_dark"] = c["peer_gap"] <= k
        # third witness: a clear optical view of open water while the radar is dark (shallow water that the radar
        # alone reads a little brighter than the AOI's deep floods)
        t["optical_water"] = (optical_water_between_passes(t["day"], water_seen) if water_seen is not None
                              else np.zeros(t["VH"]["x"].shape, dtype=bool))
        # single-dip water (SINGLE_DIP_WATER): the pass is the pixel's own darkest of the season on this track and the
        # optical window holding the pass shows the field at its own bare low
        t["single"] = np.zeros(t["VH"]["x"].shape, dtype=bool)
        if SINGLE_DIP_WATER and bare_seen is not None:
            w_day, bare = bare_seen
            idx = np.clip(np.searchsorted(np.asarray(w_day).astype("datetime64[D]"), t["day"], side="right") - 1,
                          0, len(w_day) - 1)
            bare_at = np.asarray(bare)[idx]
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for pol in ("VV", "VH"):
                    c = t[pol]
                    x_season = np.where(t["in_season"], c["x"], np.nan)
                    deepest = (c["x"] - np.nanmin(x_season, axis=0)[None, :]) <= k * c["noise"][None, :]
                    t["single"] |= t["in_season"] & (c["fall"] >= k) & deepest & bare_at
        t["at_sowing"] = np.ones(t["VH"]["x"].shape, dtype=bool)
        if sowing_seen is not None:
            w_day, sow = sowing_seen
            idx = np.clip(np.searchsorted(np.asarray(w_day).astype("datetime64[D]"), t["day"], side="right") - 1,
                          0, len(w_day) - 1)
            t["at_sowing"] = np.asarray(sow, dtype=bool)[idx]
        with np.errstate(invalid="ignore"):
            t["dark"] = t["VV"]["peer_dark"] | t["VH"]["peer_dark"] | t["optical_water"]
    n = per[0]["dark"].shape[1]

    def peer_candidates(t, ti):
        dark = t["dark"]
        persist = np.zeros_like(dark)
        persist[:-1] = dark[1:]
        other = _nearest_other(per, ti, "dark")
        with np.errstate(invalid="ignore"):
            q = {pol: (t[pol]["fall"] >= k) & (t[pol]["peer_dark"] | t["optical_water"]) for pol in ("VV", "VH")}
            # confirmation: the next pass, another track, or the optical view of the water itself (another sensor)
            ok = (t["in_season"] & (q["VV"] | q["VH"]) & (persist | other | t["optical_water"])) | t["single"]
        return q, ok & t["at_sowing"], persist, other, dark

    def pattern_candidates(t, ti):
        # shallow water the AOI test just misses: a sudden fall that stays below the pixel's own dry season for two
        # passes in a row. Only used where no peer-confirmed flood exists, and it counts as water only when the crop
        # rises after it (radar_canopy_own); never alone.
        dark = t["dark_own"]
        persist = np.zeros_like(dark)
        persist[:-1] = dark[1:]
        with np.errstate(invalid="ignore"):
            q = {pol: (t[pol]["fall"] >= k) & t[pol]["own_dark"] for pol in ("VV", "VH")}
            ok = t["in_season"] & (q["VV"] | q["VH"]) & persist & t["at_sowing"]
        return q, ok, persist, np.zeros_like(dark), dark

    def select(candidates):
        cols = np.arange(n)
        best = np.full(n, -np.inf)
        out = {"flood_date": np.full(n, np.datetime64("NaT"), dtype="datetime64[D]"), "flood_vh": np.full(n, np.nan),
               "flood_drop": np.full(n, np.nan), "flood_pol": np.full(n, "", dtype=object), "flood_track": np.full(n, -1),
               "support": np.zeros(n, dtype=int), "flood_z": np.full(n, np.nan), "flood_by_peers": np.zeros(n, dtype=bool),
               "vh_premonsoon": np.full(n, np.nan), "vh_end_track": np.full(n, np.nan), "peer_gap": np.full(n, np.nan),
               "peer_level": np.full(n, np.nan), "peer_spread": np.full(n, np.nan), "pixel_noise": np.full(n, np.nan)}
        tracks_ok = np.zeros(n, dtype=int)
        rise_sum = np.zeros(n)
        for ti, t in enumerate(per):
            q, ok, persist, other, dark = candidates(t, ti)
            run_start = np.zeros(dark.shape, dtype=int)
            for i in range(1, len(t["day"])):
                run_start[i] = np.where(dark[i - 1] & dark[i], run_start[i - 1], i)
            with np.errstate(invalid="ignore"):
                vh_wins = np.where(q["VH"], t["VH"]["rise"], -np.inf) >= np.where(q["VV"], t["VV"]["rise"], -np.inf)
                rise = np.where(vh_wins, t["VH"]["rise"], t["VV"]["rise"])
            seen = ok.any(axis=0)
            tracks_ok += seen
            # the transplanting flood is the confirmed water followed by the largest rise of the crop (own-noise
            # units); a late-season flood on a grown crop, or a harvest before the water, is followed by less.
            # Every track that saw the water adds its own look at the rise (independent looks add up).
            sc = np.where(ok, np.nan_to_num(rise, nan=-1e6), -np.inf)
            rise_sum += np.where(seen, sc.max(axis=0), 0.0)
            arg = sc.argmax(axis=0)
            top = sc[arg, cols]
            better = top > best
            best = np.where(better, top, best)
            pick = lambda a: a[arg, cols]  # noqa: E731
            use_vh = pick(vh_wins)
            out["flood_date"] = np.where(better, t["day"][pick(run_start)], out["flood_date"])
            out["flood_vh"] = np.where(better, pick(t["VH"]["x"]), out["flood_vh"])
            out["flood_drop"] = np.where(better, np.where(use_vh, pick(t["VH"]["drop"]), pick(t["VV"]["drop"])),
                                         out["flood_drop"])
            out["flood_pol"] = np.where(better, np.where(use_vh, "VH", "VV"), out["flood_pol"])
            out["flood_z"] = np.where(better, np.where(use_vh, pick(t["VH"]["fall"]), pick(t["VV"]["fall"])), out["flood_z"])
            own = np.where(use_vh, pick(t["VH"]["own_dark"]), pick(t["VV"]["own_dark"]))
            out["flood_by_peers"] = np.where(better, ~own, out["flood_by_peers"])
            for key, a_vh, a_vv in (("peer_gap", pick(t["VH"]["peer_gap"]), pick(t["VV"]["peer_gap"])),
                                     ("peer_level", t["VH"]["peer_level"][arg], t["VV"]["peer_level"][arg]),
                                     ("peer_spread", t["VH"]["peer_spread"][arg], t["VV"]["peer_spread"][arg]),
                                     ("pixel_noise", t["VH"]["noise"], t["VV"]["noise"])):
                out[key] = np.where(better, np.where(use_vh, a_vh, a_vv), out[key])
            out["flood_track"] = np.where(better, ti, out["flood_track"])
            out["support"] = np.where(better, pick(persist).astype(int) + 2 * pick(other).astype(int), out["support"])
            out["vh_premonsoon"] = np.where(better, t["VH"]["p"], out["vh_premonsoon"])
            out["vh_end_track"] = np.where(better, t["VH"]["e"], out["vh_end_track"])
        return out, best, tracks_ok, rise_sum

    out, best, tracks_ok, rise_sum = select(peer_candidates)
    p_out, p_best, p_tracks, p_rise = select(pattern_candidates)
    use_p = ~np.isfinite(best) & np.isfinite(p_best)
    for key in out:
        out[key] = np.where(use_p, p_out[key], out[key])
    best = np.where(use_p, p_best, best)
    tracks_ok = np.where(use_p, p_tracks, tracks_ok)
    rise_sum = np.where(use_p, p_rise, rise_sum)
    out["flood_by_pattern"] = use_p
    res = pd.DataFrame(out)
    res["flood_ok"] = np.isfinite(best)
    res["rise_after_flood_z"] = np.where(res["flood_ok"], best, np.nan)
    res["flood_tracks"] = tracks_ok
    with np.errstate(invalid="ignore", divide="ignore"):
        res["rise_z_all"] = np.where(tracks_ok > 0, rise_sum / np.sqrt(np.maximum(tracks_ok, 1)), np.nan)
    return res

def water_evidence(aoi_id: int, trough, climb, ndvi, windows, season_key: str = "monsoon2026",
                   window: int = 5, series=None, ndvi_raw=None) -> pd.DataFrame:
    """Per pixel: the flood searched over the whole bare period, and whether the field was ever bare.

    Why (docs/11): the first rule looked for the water only 10 days before to 15 days after the
    optical trough and measured it against a "dry" level 40-15 days before. Where cloud hid the
    field for months, the trough landed at the end of a long flood and the flood itself became the
    "dry" level, so real rice lost its water; where haze faked a trough on trees or houses, nothing
    in the radar said so. This test uses the optical **climb** (a sharper event than the trough) as
    the anchor and lets the radar find its own flood date:

    * ``flood_drop``: largest drop of any pass, over all tracks and both polarisations, below the
      field's own level of the 12-45 days before it, anywhere 100 to 5 days before the climb;
    * ``flood_vh``: VH on the flood pass itself (open water under Sentinel-1 is below -20 dB); the
      flood is only searched from ``FLOOD_EARLIEST`` (monsoon onset), because a dry-season pass on
      a harvested, drying field also drops to about -22 dB;
    * ``ndvi_at_flood``: fitted NDVI on the flood date (a canopy there means a harvest, not water);
    * ``support``: passes (any track) within 14 days of it that also sit 2 dB below their earlier level;
    * ``flood_ok``: all four agree;
    * ``vh_at_trough`` and ``never_bare``: VH stayed high around the trough and its second-darkest
      pass of the whole radar season (dry season included) never went dark (one outlier pass is
      ignored):
      the ground carried a canopy or buildings throughout, so a low optical trough there is haze.

    ``trough``/``climb`` datetime64 per pixel (NaT allowed); ``ndvi`` (windows, pixels) fitted NDVI.
    ``series`` (optional): ``[(dates, {"VV": (dates, n), "VH": (dates, n)}), ...]`` per track, already
    read, e.g. field means (``analysis/field_level``); then the AOI's stacks are not read.
    ``ndvi_raw`` (optional, (windows, pixels), NaN where unobserved): the "no canopy on the flood
    date" test then uses the clear observations from ``RAW_BEFORE_WINDOWS`` windows before to
    ``RAW_AFTER_WINDOWS`` after the flood instead of the fitted value. Why (fix plan, stage 2.3): where cloud hides the weeks around
    transplanting, the fit runs straight across the gap and reads 0.5-0.6 on the flood date, so a
    10 dB two-track flood on a surveyed paddy failed this test; with no observation near the flood
    there is no evidence of a canopy, and the radar decides.
    """
    import warnings

    if series is None:
        series = read_series(aoi_id, season_key, window)
    n = ndvi.shape[1]
    trough = np.asarray(trough).astype("datetime64[D]")
    climb = np.asarray(climb).astype("datetime64[D]")
    # Two selections per pixel, each "the largest drop of any pass, any track, either polarisation":
    # ``ok``: among the passes that pass the whole water test (below); ``any``: among all monsoon
    # passes. The reported flood is the ``ok`` pass where one exists, else the ``any`` pass, so a
    # failed test stays visible in the report. Why not simply the largest drop: on a double-crop
    # plain the season's largest drop is the summer crop's ripening and harvest (a bright canopy
    # falling to -14..-16 dB VH, not water); judging only that pass hid the real transplanting
    # flood of August (-19..-23 dB, a smaller drop from an already cut field) and ~150 ac of
    # monsoon rice per AOI lost their water (fix plan, round 2, aoi110).
    keys = ("best", "when", "pol", "vh", "other", "support", "ndvi", "track")
    def _empty():
        return {"best": np.full(n, -np.inf), "when": np.full(n, np.datetime64("NaT"), dtype="datetime64[D]"),
                "pol": np.full(n, "", dtype=object), "vh": np.full(n, np.nan), "other": np.full(n, np.nan),
                "support": np.zeros(n, dtype=int), "ndvi": np.full(n, np.nan), "track": np.full(n, -1)}
    sel = {"ok": _empty(), "any": _empty()}
    vh_min = np.full(n, np.inf)
    vh_low2 = np.full((2, n), np.inf)      # the two darkest VH passes in the span, over all tracks
    vh_trough = np.full(n, -np.inf)
    earliest = np.datetime64(FLOOD_EARLIEST)
    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    tracks = []
    for dates, flat in series:
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        tracks.append((day, {p: local_drops(dates, flat[p]) for p in ("VV", "VH")}, flat))
    # a pass "supports" a flood when it sits 2 dB below its own earlier level in either polarisation
    hits = [((drops["VV"] >= 2.0) | (drops["VH"] >= 2.0)).astype("float32") for _, drops, _ in tracks]
    cols = np.arange(n)
    for ti, (day, drops, flat) in enumerate(tracks):
        rel = (day[:, None] - climb[None, :]) / np.timedelta64(1, "D")
        with np.errstate(invalid="ignore"):
            span = (rel >= LONG_SPAN[0]) & (rel <= LONG_SPAN[1])
            near_t = np.abs((day[:, None] - trough[None, :]) / np.timedelta64(1, "D")) <= 20
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vh_min = np.fmin(vh_min, np.nanmin(np.where(span, flat["VH"], np.nan), axis=0))
            # over the WHOLE radar season, dry season included: a field that was bare and dry in
            # April (VH -18 or darker) was a field, whatever grew on it later; only ground that never
            # went dark (tree lines, gardens, houses) is "never bare"
            cand = np.where(np.isfinite(flat["VH"]), flat["VH"], np.inf)
            vh_low2 = np.sort(np.concatenate([vh_low2, cand]), axis=0)[:2]
            vh_trough = np.fmax(vh_trough, np.nanmedian(np.where(near_t, flat["VH"], np.nan), axis=0))
            # the flood itself: monsoon passes only. A dry-season pass on a freshly harvested,
            # drying field is also a VH drop to about -22 dB (seen on a field in April).
            monsoon = span & (day >= earliest)[:, None]
            # support of every pass of this track: passes of any track within 14 days that also dropped
            support_p = np.zeros((len(day), n), dtype="float32")
            for (day2, _, _), hit in zip(tracks, hits):
                near = (np.abs((day[:, None] - day2[None, :]) / np.timedelta64(1, "D")) <= 14).astype("float32")
                support_p += near @ hit
            # canopy on each pass date: the clear observations RAW_BEFORE_WINDOWS windows before to
            # RAW_AFTER_WINDOWS after it (see the module notes), or the fitted value without raw data
            idx_p = np.clip(np.searchsorted(win, day), 0, len(win) - 1)
            if ndvi_raw is None:
                canopy_p = ndvi[idx_p]
            else:
                canopy_p = np.stack([np.nanmax(ndvi_raw[max(i - RAW_BEFORE_WINDOWS, 0):i + RAW_AFTER_WINDOWS + 1], axis=0)
                                     for i in idx_p])
            for p, q in (("VV", "VH"), ("VH", "VV")):
                m = np.where(monsoon, drops[p], np.nan)
                vh = flat["VH"]
                deep = (vh <= DEEP_FLOOD_VH_MAX) & (m >= DEEP_FLOOD_DROP_MIN)
                dark = (vh <= FLOOD_VH_MAX) | ((vh <= SHALLOW_VH_MAX) & (m >= SHALLOW_DROP_MIN)
                                               & (support_p >= SHALLOW_SUPPORT_MIN))
                ok_p = ((m >= FLOOD_DROP_MIN) & dark & ~(canopy_p > FLOOD_NDVI_MAX)
                        & ((support_p >= SUPPORT_MIN) | deep))
                for name, mm in (("ok", np.where(ok_p, m, np.nan)), ("any", m)):
                    st = sel[name]
                    b = np.nanmax(mm, axis=0)
                    arg = np.nanargmax(np.where(np.isfinite(mm), mm, -np.inf), axis=0)
                    better = np.isfinite(b) & (b > st["best"])
                    st["best"] = np.where(better, b, st["best"])
                    st["when"] = np.where(better, day[arg], st["when"])
                    st["pol"] = np.where(better, p, st["pol"])
                    st["vh"] = np.where(better, vh[arg, cols], st["vh"])
                    st["other"] = np.where(better, drops[q][arg, cols], st["other"])
                    st["support"] = np.where(better, support_p[arg, cols].astype(int), st["support"])
                    st["ndvi"] = np.where(better, canopy_p[arg, cols], st["ndvi"])
                    st["track"] = np.where(better, ti, st["track"])
    qualified = np.isfinite(sel["ok"]["best"])
    pick = {k: np.where(qualified, sel["ok"][k], sel["any"][k]) for k in keys}
    best, when, pol_of, vh_at_flood, other_drop = pick["best"], pick["when"], pick["pol"], pick["vh"], pick["other"]
    support, ndvi_at = pick["support"], pick["ndvi"]
    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    own = own_levels(tracks, pick["track"], win[-1])
    vh_drop_at_flood = np.where(pol_of == "VH", best, other_drop)
    end_from = win[-1] - np.timedelta64(END_DAYS, "D")
    vh_end_parts = []
    for dates, flat in series:
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        late = (day >= end_from) & (day <= win[-1] + np.timedelta64(3, "D"))
        if late.any():
            vh_end_parts.append(flat["VH"][late])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        vh_end = np.nanmedian(np.concatenate(vh_end_parts), axis=0) if vh_end_parts else np.full(n, np.nan)
    has = ~np.isnat(when)
    idx = np.clip(np.searchsorted(win, np.where(has, when, win[0])), 0, len(win) - 1)
    ndvi_at = np.where(has, ndvi_at, np.nan)
    bare_seen = bare_near(when, climb, win, ndvi_raw)
    out = pd.DataFrame({
        "flood_drop": np.where(np.isfinite(best), best, np.nan),
        "flood_date": when, "flood_pol": pol_of,
        "flood_vh": vh_at_flood,
        "vh_min_span": np.where(np.isfinite(vh_min), vh_min, np.nan),
        "ndvi_at_flood": ndvi_at, "support": support,
        "vh_at_trough": np.where(np.isfinite(vh_trough), vh_trough, np.nan),
        "flood_vh_2nd": np.where(np.isfinite(vh_low2[1]), vh_low2[1], np.nan),
        "flood_other_drop": other_drop,
        "bare_near_flood": bare_seen,
        "vh_end": vh_end,
        "radar_canopy_rise": vh_end - vh_at_flood,
        "vh_dry_own": vh_at_flood + vh_drop_at_flood,
        "vh_end_own": own["end"],
        "vh_noise_own": own["noise"],
    })
    with np.errstate(invalid="ignore", divide="ignore"):
        rise = out["vh_end_own"] - out["flood_vh"]
        out["recovery"] = rise / (out["vh_dry_own"] - out["flood_vh"])
        out["rise_z"] = rise / (out["vh_noise_own"] * np.sqrt(1.0 / END_PASSES + 1.0))
    with np.errstate(invalid="ignore"):
        deep = (out["flood_vh"] <= DEEP_FLOOD_VH_MAX) & (out["flood_drop"] >= DEEP_FLOOD_DROP_MIN)
        no_canopy = ~(out["ndvi_at_flood"] > FLOOD_NDVI_MAX)        # unknown (no observation) passes
        dark_enough = (out["flood_vh"] <= FLOOD_VH_MAX) | ((out["flood_vh"] <= SHALLOW_VH_MAX)
                                                          & (out["flood_drop"] >= SHALLOW_DROP_MIN)
                                                          & (out["support"] >= SHALLOW_SUPPORT_MIN))
        out["flood_ok"] = ((out["flood_drop"] >= FLOOD_DROP_MIN) & dark_enough
                           & no_canopy & ((out["support"] >= SUPPORT_MIN) | deep))
        # the second-darkest pass, not the darkest: one speckled or mis-registered pass must not
        # decide that a village or a tree line was once a bare, wet field
        out["never_bare"] = (out["vh_at_trough"] > VEG_VH_MIN) & (out["flood_vh_2nd"] > VEG_FLOOD_VH_MIN)
    return out
