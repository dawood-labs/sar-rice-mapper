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
#: On the flood pass the other polarisation may not sit more than this above its own earlier level
#: (a drop of -1 dB = a rise of 1 dB). Why (fix plan, issue 15): on 11 Jun 2026 one pass read VH
#: 5-20 dB low on dry fields while VV was at its brightest; water never does that.
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
#: ``BARE_SEEN_BEFORE_WINDOWS`` windows before to ``BARE_SEEN_AFTER_WINDOWS`` after the flood. Why:
#: the radar is read over a 5 x 5 box, so a tree line next to flooded paddies also "floods" in the
#: radar; the pixel's own optical history must show bare ground at some point around the flood
#: before the radar alone may call it rice (the radar-trough path). No observation at all in that
#: span counts as unknown, not as a canopy.
BARE_SEEN_NDVI = 0.40
BARE_SEEN_BEFORE_WINDOWS = 18      # 90 days: the last dry-season view of a field can be that old under monsoon cloud
BARE_SEEN_AFTER_WINDOWS = 6
#: ``vh_end``: median VH of the passes in the last ``END_DAYS`` before the series end (any track);
#: ``radar_canopy_rise`` = vh_end - flood_vh. Why: a young crop transplanted in August is often
#: seen clear only through haze by the map date, but the radar sees its canopy: VH climbs from the
#: water (-20 to -25 dB) to -17 / -15 within 3-5 weeks. A rise of 4 dB is well above pass-to-pass
#: noise on a 5 x 5 box (~1 dB).
END_DAYS = 20


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
        loc = pr.locate(aoi_id, 0, season_key=season_key)
        series = []
        for track in [t["track_id"] for t in loc["cfg"]["s1"]["tracks"]]:
            dates, cubes = sar_curve.read_track(loc, track, window)
            series.append((dates, {p: cubes[p].reshape(len(dates), -1) for p in ("VV", "VH")}))
    n = ndvi.shape[1]
    trough = np.asarray(trough).astype("datetime64[D]")
    climb = np.asarray(climb).astype("datetime64[D]")
    best = np.full(n, -np.inf)
    when = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")
    pol_of = np.full(n, "", dtype=object)
    vh_min = np.full(n, np.inf)
    vh_low2 = np.full((2, n), np.inf)      # the two darkest VH passes in the span, over all tracks
    vh_trough = np.full(n, -np.inf)
    vh_at_flood = np.full(n, np.nan)
    other_drop = np.full(n, np.nan)        # the other polarisation's drop on the flood pass
    earliest = np.datetime64(FLOOD_EARLIEST)
    tracks = []
    for dates, flat in series:
        day = pd.DatetimeIndex(dates).to_numpy().astype("datetime64[D]")
        drops = {p: local_drops(dates, flat[p]) for p in ("VV", "VH")}
        tracks.append((day, drops))
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
            for p, q in (("VV", "VH"), ("VH", "VV")):
                m = np.where(monsoon, drops[p], np.nan)
                b = np.nanmax(m, axis=0)
                arg = np.nanargmax(np.where(np.isfinite(m), m, -np.inf), axis=0)
                better = np.isfinite(b) & (b > best)
                best = np.where(better, b, best)
                when = np.where(better, day[arg], when)
                pol_of = np.where(better, p, pol_of)
                vh_at_flood = np.where(better, flat["VH"][arg, np.arange(n)], vh_at_flood)
                other_drop = np.where(better, drops[q][arg, np.arange(n)], other_drop)
    support = np.zeros(n, dtype=int)
    for day, drops in tracks:
        near = np.abs((day[:, None] - when[None, :]) / np.timedelta64(1, "D")) <= 14
        with np.errstate(invalid="ignore"):
            # a pass counts once, whichever polarisation shows the drop
            hit = (near & ((drops["VV"] >= 2.0) | (drops["VH"] >= 2.0))).sum(axis=0)
        support += hit
    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
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
    bare_seen = np.ones(n, dtype=bool)
    if ndvi_raw is None:
        ndvi_at = np.where(has, ndvi[idx, np.arange(n)], np.nan)
    else:
        step = np.arange(len(win))[:, None]
        rel = step - idx[None, :]
        near = (rel >= -RAW_BEFORE_WINDOWS) & (rel <= RAW_AFTER_WINDOWS)
        around = (rel >= -BARE_SEEN_BEFORE_WINDOWS) & (rel <= BARE_SEEN_AFTER_WINDOWS)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ndvi_at = np.where(has, np.nanmax(np.where(near, ndvi_raw, np.nan), axis=0), np.nan)
            seen_min = np.nanmin(np.where(around, ndvi_raw, np.nan), axis=0)
        bare_seen = ~(seen_min > BARE_SEEN_NDVI)           # unknown (no observation) stays True
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
    })
    with np.errstate(invalid="ignore"):
        # water lowers VV and VH together; a pass where one polarisation falls while the other
        # rises is a broken pass (issue 15), not a flood. Unknown (no reference) is not held against it.
        consistent = ~(out["flood_other_drop"] < OTHER_POL_DROP_MIN)
        deep = (out["flood_vh"] <= DEEP_FLOOD_VH_MAX) & (out["flood_drop"] >= DEEP_FLOOD_DROP_MIN)
        no_canopy = ~(out["ndvi_at_flood"] > FLOOD_NDVI_MAX)        # unknown (no observation) passes
        dark_enough = (out["flood_vh"] <= FLOOD_VH_MAX) | ((out["flood_vh"] <= SHALLOW_VH_MAX)
                                                          & (out["flood_drop"] >= SHALLOW_DROP_MIN)
                                                          & (out["support"] >= SHALLOW_SUPPORT_MIN))
        out["flood_ok"] = ((out["flood_drop"] >= FLOOD_DROP_MIN) & dark_enough
                           & no_canopy & ((out["support"] >= SUPPORT_MIN) | deep)
                           & consistent)
        # the second-darkest pass, not the darkest: one speckled or mis-registered pass must not
        # decide that a village or a tree line was once a bare, wet field
        out["never_bare"] = (out["vh_at_trough"] > VEG_VH_MIN) & (out["flood_vh_2nd"] > VEG_FLOOD_VH_MIN)
    return out
