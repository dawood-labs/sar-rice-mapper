"""The current-season rice rule, per pixel, checked against field plots.

Why this rule and not the earlier one
-------------------------------------
The field survey (plots reported as standing rice in September 2026) showed what the monsoon crop
looks like in the gap-free 5-day series, region by region:

* the field goes to a **trough** — bare or flooded, NDVI 0 to 0.3 — somewhere between late June and
  late August depending on the region (two months apart between the delta and the capital region);
* the canopy then **climbs** by 0.5 to 0.9 NDVI within six to nine weeks;
* at the trough the field is wet. In the delta that is open water (LSWI above NDVI, 79-100 % of
  plots). In the capital region it is not: the field dried out after the summer harvest (LSWI -0.2)
  and was then wetted for transplanting (LSWI up by 0.2-0.3 to about zero) under a shallow, turbid
  layer with dense seedlings, so LSWI never rose above NDVI (0-13 % of plots) — yet the rise from
  the field's own dry level was there in 71-86 % of them. Water is therefore read both ways
  (``wet_open`` or ``wet_relative``), and is **evidence, not a condition**: it is reported per
  pixel, and the rule does not fail a pixel for lacking it.

So the rule is calendar-free within a season window, and needs no AOI-level season: per pixel, the
trough inside the window and the climb after it. Two thresholds, both read off the plots:

* ``trough_max`` = 0.40: the 90th percentile of plot troughs was below 0.2 in most regions and
  0.36-0.41 in the two latest ones;
* ``rise_min`` = 0.30 for **rice**: the plots' rises were 0.5-0.9 wherever the crop was more than
  six weeks old. Where it was younger the rise was 0.15-0.35, so a rise between ``young_min`` = 0.15
  and ``rise_min`` is called **young**: a crop has started, it cannot be confirmed yet.

A third threshold is an absolute floor on the canopy reached, ``canopy_min`` = 0.50 for rice and
``young_canopy_min`` = 0.30 for young. It exists because of a control AOI with no monsoon crop at
all: there the fields went under water (NDVI -0.35) and re-emerged as bare soil (0.2), a "rise" of
0.55 that is not a canopy. NDVI 0.2 is soil; the plots' canopies were 0.65 and above.

On the plots this gives 88-100 % recall where the crop was established and 30-60 % where it was
still young — which is the honest limit of optical data at that date, not a flaw to tune away.

Water is confirmed by radar (``radar_water``), not by the optical LSWI: for every rice pixel the
Sentinel-1 backscatter is lined up on that pixel's own trough and the dip below its dry level is
read. Where the dip is there the pixel is **rice** (1); where the radar could look and saw no dip,
or could not look at all, it is **rice by phenology, water unconfirmed** (3) — kept apart so that
"no water" and "could not check" are never mixed with confirmed rice.

The target is rice **standing on the map date** (decided with the client's manager): the canopy
must be there on the last window (``CANOPY_MIN``) and must not have fallen from its peak by more
than ``STANDING_FALL_MAX`` — a field already cut is not standing rice, however rice-like its
season was. Because a standing crop was transplanted within roughly one rice season of the map date,
the trough is looked for only in the last ``LOOKBACK_DAYS``; this also stops the bare, dry field of
May from being taken for the transplanting of a crop that started in July. Crops still too young
to show a canopy are reported as ``young`` and crops already cut as ``harvested``, as information:
neither is delivered as rice, and the map is re-run with new imagery when the client asks again.

Classes written: 0 not rice, 1 rice standing (water confirmed by radar), 2 young, 3 rice standing
by phenology with water unconfirmed, 4 rice-like but harvested / not standing, 255 no data.
Areas in acres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import ndvi_5day as nd
from .optical_phenology import acres

TROUGH_MAX = 0.40
RISE_MIN = 0.30
YOUNG_MIN = 0.15
CANOPY_MIN = 0.50
YOUNG_CANOPY_MIN = 0.30
#: The season: rice planted in May 2026 or later. The dry-season crop is out of scope.
SEASON = ("2026-05-01", "2026-09-24")
#: A crop standing on the last date was transplanted within about one rice season of it
#: (transplanting to harvest is 100-120 days), so its trough is looked for in this many days
#: before the last window and not earlier.
LOOKBACK_DAYS = 110
#: Where the 110-day window finds no cycle at all (class 0), the window is widened to this so that a
#: crop sown in the first half of May counts (the deliverable says "planted May 2026 or later").
#: Only as a fallback: applied everywhere, the earlier trough shifts the water anchor and ~3 % of
#: the confirmed rice loses its flood (user review of aoi160, 28 Sep 2026; docs/17).
LOOKBACK_FALLBACK_DAYS = 140
#: A standing crop may have lost this much NDVI from its peak (haze, early senescence) and still be
#: standing; a harvested field has lost far more.
STANDING_FALL_MAX = 0.35
#: The field must have been without a canopy (fitted NDVI at or below ``TROUGH_MAX + 0.05``) for at
#: least this many consecutive 5-day windows around its trough — 8 windows, about 40 days. A field
#: really starting a crop has weeks of bare, puddled and seedling-covered ground; a haze dip that
#: the light cloud mask let through on an orchard or a tree line lasts one or two windows. On the
#: validation set this took the evergreen false-positive rate from 7 % to 0 % without touching the
#: recall on the field plots, whose low period is 55 days at the 5th percentile. Counted up to 8
#: windows on each side of the trough.
LOW_WINDOWS_MIN = 8
#: LSWI must rise by this much from the field's own driest level in the 60 days before the trough
#: (and end at or above zero) to count as wetting for transplanting.
WET_RISE_MIN = 0.15
WET_LOOKBACK_DAYS, WET_LOOKAHEAD_DAYS = 60, 25
CLASSES = {0: "not rice", 1: "rice", 2: "young", 3: "rice_unconfirmed", 4: "harvested",
           5: "never_bare", 6: "young_rice", 7: "flooded_not_green", 8: "cut_unconfirmed",
           9: "rice_like_no_water", 255: "no data"}
#: Class 9 "rice-like, no sign of water" is never written by the rule itself: ``analysis/finalize``
#: gives it (switch ``PER_PIXEL_RELABEL``) to the class-3 pixels the generated "class 3 -> rice"
#: relabel would have made rice although their radar brightened right after sowing with no sign of
#: water (``<aoi>_monsoon2026_c3_radar_against.tif``). Why a class of its own: a dry-sown crop or
#: direct-seeded rice looks like rice optically; the client must see these acres apart from both
#: the rice and the plain class 3 (user, 30 Sep). Reported, not delivered.
N_CLASSES = 10
#: Report classes (fix plan, stage 2.5). 7 "flooded, not yet green": a confirmed radar flood within
#: ``FLOODED_RECENT_DAYS`` of the map date and no canopy yet (last fitted NDVI below
#: ``FLOODED_NDVI_MAX``): the next map's rice, hidden in "not rice" before (issue 22). 8 "cut crop,
#: water not confirmed": a rice-like cycle cut before the map date without transplanting water,
#: which the review found to be mostly rain-fed dry-land crops (issue 18), separated from class 4
#: (harvested with water) so the harvested-paddy acres are not inflated.
FLOODED_RECENT_DAYS = 60
FLOODED_NDVI_MAX = 0.20
#: Radar canopy after a confirmed flood (``radar_water.radar_canopy_rise``): at or above
#: ``RADAR_YOUNG_RISE_DB`` the pixel is a young crop even when no clear optical view shows the
#: canopy yet; at or above ``RADAR_VISIBLE_RISE_DB`` the canopy is real enough to deliver as young
#: rice (class 6) without the optical ``YOUNG_VISIBLE`` test. Why: seven reviewed young-rice fields
#: (late-August transplanting) fell to "not rice" once the hazy late observations were removed,
#: while their radar showed the canopy rising 5-8 dB from the water.
RADAR_YOUNG_RISE_DB = 4.0
RADAR_VISIBLE_RISE_DB = 5.0
#: A radar canopy this far above the water (dB) and reaching ``RADAR_MATURE_VH_MIN`` at the season end
#: is a full-grown crop, not a young one: the pixel is class 1 even when no clear optical view showed
#: the canopy (a paddy flooded in late May whose only later clear view was a cloud edge read
#: VH -25 -> -14 by August: "young rice" would misname it).
RADAR_MATURE_RISE_DB = 8.0
RADAR_MATURE_VH_MIN = -15.0
from .radar_water import RADAR_CANOPY_VH_MIN  # noqa: E402  (the radar canopy must reach this VH)
#: A ripening (yellowing) crop is still standing: its NDVI falls from ~0.75 to ~0.45 before the
#: cut. The standing test therefore accepts a last value at or above ``STANDING_MIN`` that has not
#: fallen more than ``STANDING_FALL_MAX`` from the peak; a harvested field reads 0.15-0.30.
STANDING_MIN = 0.40
#: A young crop is delivered as young rice (class 6) when its water is confirmed and its canopy is
#: already visible on the map date: 95 % of the field plots' pixels that sat in "young" were at or
#: above 0.30 (bare soil 0.15-0.25, open water 0-0.1). docs/15.
YOUNG_VISIBLE = 0.30
#: A clear optical view within the last ``CLEAR_RECENT_WINDOWS`` windows (20 days) that reads below
#: ``YOUNG_VISIBLE`` contradicts a canopy seen by the radar alone: the radar canopy (``RADAR_MATURE_*``,
#: ``RADAR_VISIBLE_RISE_DB``) may only stand in for the optical one where the optical end is stale.
#: Why (stage 6): 23 pixels of open water / bare mud (raw NDVI -0.14 to 0.25 on a clear 16 Sep)
#: next to bright bunds were delivered as "rice" by a 12 dB radar rise in a 5 x 5 box.
CLEAR_RECENT_WINDOWS = 4


def last_clear_view(ndvi_raw, windows) -> pd.DataFrame:
    """Per pixel, the last clear observation: ``last_clear_ndvi`` and its ``last_clear_age`` in
    windows before the series end (inf where nothing was ever observed)."""
    raw = np.asarray(ndvi_raw)
    obs = np.isfinite(raw)
    any_obs = obs.any(axis=0)
    last_idx = raw.shape[0] - 1 - np.argmax(obs[::-1], axis=0)
    cols = np.arange(raw.shape[1])
    value = np.where(any_obs, raw[last_idx, cols], np.nan)
    age = np.where(any_obs, raw.shape[0] - 1 - last_idx, np.inf).astype(float)
    return pd.DataFrame({"last_clear_ndvi": value, "last_clear_age": age})
#: Water test versions: "v1" = dip at the optical trough only; "v2" (docs/11) = v1 OR the flood
#: searched over the whole bare period, plus class 5 for "rice-like" curves on ground the radar
#: shows was never bare (trees, houses: the optical trough there is haze).
WATER_DEFAULT = "v2"
#: Radar-defined trough (fix plan, stage 2.3, issues 4, 16, 3/6). Why: where cloud hid the field
#: for the weeks around transplanting, the fitted curve runs straight across the gap and never
#: falls below ``TROUGH_MAX``, so a textbook paddy (deep two-track flood, then canopy) was "not
#: rice"; and a paddy whose fallow after the summer crop was shorter than 40 days failed the
#: ``LOW_WINDOWS_MIN`` test. A confirmed radar flood (``radar_water.water_evidence``: >= 4 dB below
#: the field's own level, VH <= -19 dB, no canopy that day, seen on a second pass) is bare, wet
#: ground by definition, so it can stand in for the optical trough: the climb is then measured
#: from the fitted NDVI on the flood date. Haze dips on trees never come with such a flood, which
#: is why the 40-day bare criterion is not needed on this path.
RADAR_TROUGH_DEFAULT = True


def radar_trough_events(events: pd.DataFrame, ndvi, windows) -> pd.DataFrame:
    """Per pixel, the climb measured from the radar flood: ``peak_after_flood``, ``rise_from_flood``,
    ``standing_after_flood``; NaN / False where there is no confirmed flood (``flood_ok``)."""
    windows = pd.DatetimeIndex(windows)
    win = windows.to_numpy().astype("datetime64[D]")
    n = ndvi.shape[1]
    flood = pd.to_datetime(events["flood_date"]).to_numpy().astype("datetime64[D]")
    ok = events["flood_ok"].to_numpy(dtype=bool) & ~np.isnat(flood)
    idx = np.clip(np.searchsorted(win, np.where(ok, flood, win[0])), 0, len(win) - 1)
    step = np.arange(len(win))[:, None]
    after = np.where(step >= idx[None, :], ndvi, -np.inf)
    with np.errstate(invalid="ignore"):
        peak = np.where(ok, after.max(axis=0), np.nan)
        at = np.where(ok, ndvi[idx, np.arange(n)], np.nan)
        rise = peak - at
        last = ndvi[-1]
        standing = ok & (last >= STANDING_MIN) & (last >= peak - STANDING_FALL_MAX)
    return pd.DataFrame({"ndvi_at_flood_fit": at, "peak_after_flood": peak, "rise_from_flood": rise,
                         "standing_after_flood": standing})


def season_to_series(season, windows) -> tuple[str, str]:
    """The season with its end moved up to the last window of the series when the series runs later.

    Why: ``SEASON[1]`` was written for a series that ends on 24 Sep (last window 21 Sep). The map date is
    always the series' last window, so a series rebuilt with newer images (e.g. to 28 Sep) has windows
    after ``SEASON[1]``; left as it was, the trough search (``windows < season[1]``) would silently skip
    them while the standing test reads them, and a field cut in the new days would have its cut ignored
    by one test and seen by the other. The end only moves later, never earlier, so a series that ends
    before ``SEASON[1]`` (the delivered one) gives exactly the same season as before.
    """
    last = pd.DatetimeIndex(windows)[-1] + pd.Timedelta(days=1)     # up to and including the last window
    end = max(pd.Timestamp(season[1]), last)
    return (season[0], end.strftime("%Y-%m-%d"))


#: Round 3 (user 30 Sep, aoi160_007735): the optical climb counts only when at least two clear views after the trough lie
#: ``WATER_K`` x the pixel's own noise above it, or the radar's VH rose since the trough (``rise_since_trough_z``). Why:
#: one view on the last date (0.22 -> 0.53 in ten days, radar flat) made a bare field "rice"; the fit just reached the
#: 0.50 canopy. One view can be haze, a cloud edge or a neighbour's crop; a second view or the radar confirms it.
#: One clear view is enough when it already reaches this AOI's rice canopy (``at_rice_canopy``, reference = crops with
#: two views): aoi20 plots (cloudy, one clear view of 0.67-0.82 on 6 Sep) were lost without it (99.4 -> 99.0 %).
#: The radar path counts the views after its FLOOD (``views_above_flood``, above the fit on the flood date) or the radar
#: rise after the flood (``rise_z_all``): gating it on the trough turned aoi13 cut fields (trough = the bare spell after
#: the harvest) into rice (cut negatives 38.6 -> 47.2 %).
CLIMB_TWO_VIEWS = True


def views_above_trough(ndvi_raw, windows, trough_date, trough_ndvi, noise) -> np.ndarray:
    """Per pixel: clear views after its trough window that lie ``WATER_K`` x its own optical noise above its trough."""
    from .radar_water import WATER_K

    raw = np.asarray(ndvi_raw)
    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    t = np.asarray(trough_date).astype("datetime64[D]")
    t_idx = np.searchsorted(win, np.where(np.isnat(t), win[-1], t))
    after = np.arange(raw.shape[0])[:, None] > t_idx[None, :]
    level = np.asarray(trough_ndvi, dtype=float) + WATER_K * np.nan_to_num(np.asarray(noise, dtype=float))
    with np.errstate(invalid="ignore"):
        above = after & (raw >= level[None, :])
    return np.where(np.isnat(t), 0, above.sum(axis=0))


def max_view_after(ndvi_raw, windows, day) -> np.ndarray:
    """Per pixel: the highest clear view after the window of ``day`` (NaN where none or ``day`` unknown)."""
    import warnings

    raw = np.asarray(ndvi_raw)
    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    t = np.asarray(day).astype("datetime64[D]")
    t_idx = np.searchsorted(win, np.where(np.isnat(t), win[-1], t))
    after = np.arange(raw.shape[0])[:, None] > t_idx[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        top = np.nanmax(np.where(after, raw, np.nan), axis=0)
    return np.where(np.isnat(t), np.nan, top)


def at_rice_canopy(events: pd.DataFrame, value, reference) -> np.ndarray:
    """``value`` (per pixel) reaches this AOI's rice canopy: not ``WATER_K`` x sqrt(spread^2 + own noise^2) below the
    median peak of the ``reference`` pixels (bool mask; optically confirmed standing crops). False without enough
    reference pixels (``radar_water.WATER_PEERS_MIN``)."""
    from .radar_water import WATER_K, WATER_PEERS_MIN

    peak = np.asarray(events["peak_after"], dtype=float)
    vals = peak[np.asarray(reference, dtype=bool) & np.isfinite(peak)]
    if len(vals) < WATER_PEERS_MIN:
        return np.zeros(len(events), dtype=bool)
    med = np.median(vals)
    spread = 1.4826 * np.median(np.abs(vals - med))
    noise = np.nan_to_num(np.asarray(events["ndvi_noise_own"], dtype=float)) if "ndvi_noise_own" in events else 0.0
    with np.errstate(invalid="ignore"):
        return np.asarray(value, dtype=float) >= med - WATER_K * np.sqrt(spread ** 2 + noise ** 2)


def pixel_events(ndvi, lswi, windows, season=SEASON, lookback_days: int | None = LOOKBACK_DAYS,
                 trough_level=None) -> pd.DataFrame:
    """Per pixel: trough date and value inside the search window, LSWI there, and the climb after it.

    ``ndvi``/``lswi`` are (windows, pixels) fitted series. Vectorised, so an AOI of half a million
    pixels takes seconds. The search window is the season cut to the last ``lookback_days`` before
    the last window (see ``LOOKBACK_DAYS``). ``climb_date`` is the first window after the trough
    where NDVI has risen by ``RISE_MIN`` — the moment the crop became a confirmed canopy.
    ``standing`` says whether that canopy is still there on the last window.

    ``lookback_days=None`` searches the whole season. With ``trough_level`` (per pixel) the trough is the LAST window at or
    below that level that starts a run of at least two such windows (the field's last bare spell before the crop it carries
    now; one window alone is one view, often haze; round 3, user 30 Sep, aoi160_003471); a single window where no run
    exists; the lowest window where the field never reaches the level.
    """
    windows = pd.DatetimeIndex(windows)
    earliest = pd.Timestamp(season[0]) if lookback_days is None else \
        max(pd.Timestamp(season[0]), windows[-1] - pd.Timedelta(days=lookback_days))
    inside = (windows >= earliest) & (windows < season[1])
    idx = np.flatnonzero(inside)
    sub = ndvi[idx]
    n_pix = ndvi.shape[1]
    valid = np.isfinite(sub).all(axis=0)
    trough = np.where(valid, np.nanargmin(np.where(np.isfinite(sub), sub, np.inf), axis=0), 0)
    if trough_level is not None:
        with np.errstate(invalid="ignore"):
            low = sub <= np.asarray(trough_level, dtype=float)[None, :]
        # a low that lasts at least two windows in a row (one window is one view: haze or a cloud edge)
        low2 = np.zeros_like(low)
        low2[:-1] = low[:-1] & low[1:]
        use = np.where(low2.any(axis=0)[None, :], low2, low)
        has_low = use.any(axis=0)
        last_low = len(idx) - 1 - np.argmax(use[::-1], axis=0)
        trough = np.where(valid & has_low, last_low, trough)
    cols = np.arange(n_pix)
    trough_ndvi = sub[trough, cols]
    lswi_sub = lswi[idx]
    lswi_at = lswi_sub[trough, cols]
    # wetting relative to the field's own dry level: driest LSWI in the weeks before the trough
    # against the wettest in the weeks after it (the water is applied at or just after the trough)
    step = np.arange(len(idx))[:, None]
    when = windows[idx].to_numpy()
    day_gap = (when[:, None] - when[trough][None, :]) / np.timedelta64(1, "D")
    before = (day_gap >= -WET_LOOKBACK_DAYS) & (day_gap <= 0)
    after_w = (day_gap >= 0) & (day_gap <= WET_LOOKAHEAD_DAYS)
    lswi_dry = np.where(before, lswi_sub, np.inf).min(axis=0)
    lswi_wet = np.where(after_w, lswi_sub, -np.inf).max(axis=0)
    # highest value after the trough, and the first window where the climb reaches RISE_MIN
    after = np.where(step >= trough[None, :], sub, -np.inf)
    peak_after = after.max(axis=0)
    reached = after >= (trough_ndvi + RISE_MIN)[None, :]
    first = np.where(reached.any(axis=0), reached.argmax(axis=0), -1)
    return pd.DataFrame({
        "valid": valid,
        "trough_date": windows[idx][trough],
        "trough_ndvi": trough_ndvi, "lswi_at_trough": lswi_at,
        "wet_open": lswi_at > trough_ndvi,
        "wet_relative": (lswi_wet - lswi_dry >= WET_RISE_MIN) & (lswi_wet >= 0),
        "wet_at_trough": (lswi_at > trough_ndvi) | ((lswi_wet - lswi_dry >= WET_RISE_MIN) & (lswi_wet >= 0)),
        "rise": peak_after - trough_ndvi, "peak_after": peak_after,
        "climb_date": pd.Series(np.where(first >= 0, windows[idx][np.clip(first, 0, None)], pd.NaT)),
        "last_ndvi": ndvi[-1],
        "standing": (ndvi[-1] >= STANDING_MIN) & (ndvi[-1] >= peak_after - STANDING_FALL_MAX),
        "low_windows": low_run(ndvi, idx[trough], TROUGH_MAX + 0.05),
    })


def low_run(ndvi, trough_idx, level: float, reach: int = 8) -> np.ndarray:
    """Consecutive windows at or below ``level`` around each pixel's trough index, up to ``reach`` per side.

    ``ndvi`` is the full (windows, pixels) series; ``trough_idx`` indexes into it. The trough window
    itself counts as one.
    """
    n_win, n_pix = ndvi.shape
    cols = np.arange(n_pix)
    low = ndvi <= level
    run = np.ones(n_pix, dtype=int)
    for direction in (-1, 1):
        still = np.ones(n_pix, dtype=bool)
        for k in range(1, reach + 1):
            j = trough_idx + direction * k
            inside = (j >= 0) & (j < n_win)
            still &= inside & low[np.clip(j, 0, n_win - 1), cols]
            run += still
    return run


def at_sowing(events: pd.DataFrame) -> np.ndarray:
    """The flood lies at the pixel's own sowing: the fitted NDVI on the flood date is within ``WATER_K`` x the pixel's own
    optical noise of its trough (its own low point), and that low point is real (its highest clear view of the series, an
    earlier crop or the new one, lies k x noise above it).
    Trees and gardens have no such low point. Unknown noise: False."""
    from .radar_water import WATER_K

    with np.errstate(invalid="ignore"):
        at = np.asarray(events["ndvi_at_flood_fit"], dtype=float)
        low = np.asarray(events["trough_ndvi"], dtype=float)
        top = np.asarray(events["ndvi_max_own"], dtype=float) if "ndvi_max_own" in events \
            else np.asarray(events["peak_after"], dtype=float)
        noise = np.asarray(events["ndvi_noise_own"], dtype=float) if "ndvi_noise_own" in events else np.full(len(events), np.nan)
        real_low = (top - low) >= WATER_K * noise        # a flat, always-green line has no low point at all
        return real_low & (at <= low + WATER_K * noise)


def water_sign(events: pd.DataFrame) -> np.ndarray:
    """Any sign of water at sowing (user, 30 Sep): a peer-confirmed flood, or a pattern flood at the pixel's own sowing."""
    ok = np.asarray(events["flood_ok"], dtype=bool).copy()
    if "flood_by_pattern" in events:
        ok &= ~np.asarray(events["flood_by_pattern"], dtype=bool) | at_sowing(events)
    if "open_water_year" in events:
        ok &= ~np.asarray(events["open_water_year"], dtype=bool)
    return ok


def radar_canopy_own(events: pd.DataFrame, grown, standing, wet, contradicted):
    """The crop after the water, read from the radar against the pixel's own levels (water test v3).

    ``canopy_r``: a confirmed flood (with a bare view near it) followed by a rise of at least ``radar_water.WATER_K`` times
    the pixel's own noise, all tracks that saw the water together (``rise_z_all``). ``full``: that rise has come back to the
    pixel's own dry-season level (``vh_premonsoon``), or, for a double crop whose dry-season level was the first crop, to the
    level of this AOI's optically confirmed standing rice on the map date; and no recent clear view says otherwise.
    Pixels that are open water most of the year (``open_water_year``: median clear NDVI below 0) never count as flooded fields.
    Returns ``(canopy_r, full, flood_seen)``; no dB cut-off is involved.
    """
    import warnings

    from .radar_water import WATER_K

    flood_seen = np.asarray(events["flood_ok"], dtype=bool).copy()
    if "bare_near_flood" in events:
        flood_seen &= np.asarray(events["bare_near_flood"], dtype=bool)
    if "open_water_year" in events:        # a pond's water and its ripples are not a crop cycle
        flood_seen &= ~np.asarray(events["open_water_year"], dtype=bool)
    pattern = np.asarray(events["flood_by_pattern"], dtype=bool) if "flood_by_pattern" in events \
        else np.zeros(len(events), dtype=bool)
    flood_seen &= ~pattern | at_sowing(events)       # shallow water counts only at the pixel's own sowing
    if "trough_peer_z" in events:
        # the low point must be as low as the AOI's paddies at sowing (tree lines have a haze dip at 0.56-0.68 instead);
        # unknown (no reference) is not held against a field
        with np.errstate(invalid="ignore"):
            flood_seen &= ~(np.asarray(events["trough_peer_z"], dtype=float) >= WATER_K)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        canopy_r = flood_seen & (np.asarray(events["rise_z_all"], dtype=float) >= WATER_K)
        f = np.asarray(events["flood_vh"], dtype=float)
        e = np.asarray(events["vh_end_track"], dtype=float)
        p = np.asarray(events["vh_premonsoon"], dtype=float)
        rice_opt = grown & standing & wet & np.isfinite(e)
        level = np.nanmedian(e[rice_opt]) if rice_opt.sum() >= 50 else np.nan
        back = np.fmax((e - f) / (p - f), (e - f) / (level - f))
    full = canopy_r & (back >= 1.0) & ~contradicted
    # a pattern flood (shallow water, no peer confirmation) is water only when the crop rises after it
    flood_seen &= ~pattern | canopy_r
    return canopy_r, full, flood_seen


#: Width of one series window in days (``ndvi_5day.STEP_DAYS``); converts ``last_clear_age`` to a date.
STEP_DAYS = 5
#: User decision (30 Sep 2026): "young rice" is only a crop sown AFTER the normal sowing window; a young-rice pixel sown
#: before this date is rice, even when its September canopy still looks young. Sowing date = the radar flood date
#: (transplanting water) where one is confirmed, else the optical trough. A deliverable definition set by the user, to
#: revisit for another season; not a measured cut-off.
YOUNG_SOWN_AFTER = "2026-07-20"


def classify(events: pd.DataFrame, trough_max=TROUGH_MAX, rise_min=RISE_MIN, young_min=YOUNG_MIN,
             canopy_min=CANOPY_MIN, young_canopy_min=YOUNG_CANOPY_MIN, radar_wet=None,
             low_windows_min: int = LOW_WINDOWS_MIN, never_bare=None, radar_trough: bool = False,
             map_date=None):
    """0 not rice, 1 rice standing, 2 young, 3 standing with water unconfirmed, 4 harvested,
    5 rice-like curve on ground that was never bare (radar), 6 young rice (young, water confirmed,
    canopy visible: last NDVI >= ``YOUNG_VISIBLE``), 255 no data.

    ``radar_wet`` (bool per pixel) splits the standing phenology-rice pixels into confirmed (1) and
    unconfirmed (3). Without it every such pixel is unconfirmed. ``never_bare`` (bool per pixel,
    ``radar_water.water_evidence``) moves unconfirmed pixels whose radar shows a canopy or buildings
    all season to class 5. With ``radar_trough`` (see ``RADAR_TROUGH_DEFAULT``) a pixel whose
    optical trough failed (hidden by cloud, or a short fallow) is still rice / young rice when its
    confirmed radar flood is followed by the same climb (``radar_trough_events`` columns). With
    ``map_date`` (the last window) the report classes 7 and 8 are assigned (see ``FLOODED_RECENT_DAYS``).
    """
    out = np.zeros(len(events), dtype="uint8")
    low = events["trough_ndvi"] <= trough_max
    if "low_windows" in events:
        low &= events["low_windows"] >= low_windows_min
    grown = (low & (events["rise"] >= rise_min) & (events["peak_after"] >= canopy_min)).to_numpy()
    standing = events["standing"].to_numpy() if "standing" in events else np.ones(len(events), dtype=bool)
    young = (low & ~grown & (events["rise"] >= young_min) & (events["peak_after"] >= young_canopy_min)).to_numpy()
    wet = np.zeros(len(events), dtype=bool) if radar_wet is None else np.asarray(radar_wet, dtype=bool).copy()
    contradicted = np.zeros(len(events), dtype=bool)
    if "last_clear_ndvi" in events:
        with np.errstate(invalid="ignore"):
            contradicted = ((np.array(events["last_clear_age"], dtype=float) <= CLEAR_RECENT_WINDOWS)
                            & (np.array(events["last_clear_ndvi"], dtype=float) < YOUNG_VISIBLE))
    own = radar_trough and "rise_z_all" in events
    climb_seen = np.ones(len(events), dtype=bool)
    opt_rice = np.zeros(len(events), dtype=bool)
    if own and CLIMB_TWO_VIEWS and "views_above_trough" in events:
        from .radar_water import WATER_K

        with np.errstate(invalid="ignore"):
            climb_seen = (np.asarray(events["views_above_trough"]) >= 2) | \
                (np.asarray(events["rise_since_trough_z"], dtype=float) >= WATER_K)
        # ... or one clear view that already reaches this AOI's rice canopy (cloudy AOIs, aoi20 plots: one 0.7-0.8 view)
        opt_rice = grown & standing & (np.asarray(events["views_above_trough"]) >= 2)
        if "max_view_after_trough" in events:
            climb_seen |= at_rice_canopy(events, events["max_view_after_trough"], opt_rice)
        grown = grown & climb_seen
        young = young & climb_seen
    if own and HARVEST_BACK_TO_BARE and "ndvi_noise_own" in events:
        # plan O5 (round 3, plot 2971): a crop is cut only when its latest value has fallen back to its own bare level
        # (within k x its own optical noise of the trough); a young crop still rising (0.30) or a ripening one (0.45) stands.
        from .radar_water import WATER_K

        with np.errstate(invalid="ignore"):
            noise_o = np.nan_to_num(np.asarray(events["ndvi_noise_own"], dtype=float))
            back_to_bare = events["last_ndvi"].to_numpy() <= events["trough_ndvi"].to_numpy() + WATER_K * noise_o
        standing = ~back_to_bare
    if own:
        canopy_r, full, flood_seen = radar_canopy_own(events, grown, standing, wet, contradicted)
        if "flood_by_pattern" in events:
            # without the crop's rise a pattern flood is not water: it must not turn a pixel into rice by the optical path
            wet = wet & ~(np.asarray(events["flood_by_pattern"], dtype=bool) & ~canopy_r)
    if radar_trough and "rise_from_flood" in events:
        with np.errstate(invalid="ignore"):
            # a confirmed flood is bare, wet ground (NDVI of water is at or below 0.2), so the climb
            # to a canopy is measured by the canopy itself; the fitted value on the flood date can
            # be an interpolation across a cloud gap and is not trusted for the rise
            # an explicit copy: with numpy 2 + pandas 2.2 np.array(series) returns a VIEW, so the
            # in-place & below used to rewrite events["flood_ok"] (issue 35)
            ok = np.asarray(events["flood_ok"], dtype=bool).copy()
            if "bare_near_flood" in events:            # the pixel's own optical history must allow a field
                ok &= np.array(events["bare_near_flood"], dtype=bool)
            if own:                                    # water test v3: the gated flood (ponds, pattern floods)
                ok = flood_seen.copy()
            flood_climb_seen = np.ones(len(events), dtype=bool)
            if own and CLIMB_TWO_VIEWS and "views_above_flood" in events:
                # the radar path's climb is the one after the FLOOD (a cut field's trough is its post-harvest bare spell)
                from .radar_water import WATER_K

                with np.errstate(invalid="ignore"):
                    flood_climb_seen = (np.asarray(events["views_above_flood"]) >= 2) | \
                        (np.asarray(events["rise_z_all"], dtype=float) >= WATER_K)
                if "max_view_after_flood" in events:
                    flood_climb_seen |= at_rice_canopy(events, events["max_view_after_flood"], opt_rice)
            r_grown = ok & (events["peak_after_flood"] >= canopy_min).to_numpy() & flood_climb_seen
            r_young = ~r_grown & ok & (events["peak_after_flood"] >= young_canopy_min).to_numpy() & flood_climb_seen
            if own:                                    # water test v3: the pixel's own levels, no dB cut-offs
                mature = full & ~grown & ~young
                r_grown = r_grown | mature
                r_young = r_young & ~full
                r_young |= ~r_grown & canopy_r
            elif "radar_canopy_rise" in events:          # the canopy seen by the radar alone
                rise_db = np.array(events["radar_canopy_rise"], dtype=float)
                vh_end = np.array(events["vh_end"], dtype=float) if "vh_end" in events else np.full(len(events), np.nan)
                canopy_level = ~(vh_end < RADAR_CANOPY_VH_MIN)
                mature = ok & (rise_db >= RADAR_MATURE_RISE_DB) & (vh_end >= RADAR_MATURE_VH_MIN) & ~contradicted
                r_grown = r_grown | (mature & ~grown & ~young)
                r_young = r_young & ~mature
                r_young |= ~r_grown & ok & (rise_db >= RADAR_YOUNG_RISE_DB) & canopy_level
        r_standing = events["standing_after_flood"].to_numpy(dtype=bool)
        # the radar path only adds: a pixel the optical path already decided keeps that decision
        add = r_grown & ~grown & ~young
        if own:
            r_standing = r_standing | (add & full)
        elif "radar_canopy_rise" in events:
            # a full-grown radar canopy at the season end is standing whatever the stale optical end says
            r_standing = r_standing | (add & (rise_db >= RADAR_MATURE_RISE_DB) & (vh_end >= RADAR_MATURE_VH_MIN))
        standing = np.where(add, r_standing, standing)
        grown = grown | add
        young = young | (r_young & ~grown & ~young)
    rice = grown & standing
    out[rice & wet] = 1
    out[rice & ~wet] = 3
    last = events["last_ndvi"].to_numpy() if "last_ndvi" in events else np.zeros(len(events))
    visible = last >= YOUNG_VISIBLE
    if radar_trough and own:
        visible |= canopy_r & ~contradicted
    elif radar_trough and "radar_canopy_rise" in events:
        with np.errstate(invalid="ignore"):
            at_canopy = np.array(events["vh_end"], dtype=float) >= RADAR_CANOPY_VH_MIN if "vh_end" in events \
                else np.ones(len(events), dtype=bool)
            visible |= ((np.array(events["radar_canopy_rise"], dtype=float) >= RADAR_VISIBLE_RISE_DB) & at_canopy
                        & ~contradicted)
    young_rice = young & wet & visible
    out[grown & ~standing & wet] = 4
    out[grown & ~standing & ~wet] = 8
    out[young] = 2
    out[young_rice] = 6
    if radar_trough and own:
        # flooded, but neither the radar nor the optical shows a crop since: next month's rice (no day limit)
        out[(out == 0) & flood_seen & ~canopy_r] = 7
        # user decision (30 Sep): "young" is not a class. A young crop is young rice unless the optical canopy AND the
        # radar canopy are both very low (within k x the pixel's own noise of its trough / flood level) AND there is no
        # sign of water at sowing; then it is not rice.
        from .radar_water import WATER_K

        with np.errstate(invalid="ignore"):
            noise_o = np.asarray(events["ndvi_noise_own"], dtype=float) if "ndvi_noise_own" in events \
                else np.full(len(events), np.nan)
            optical_low = ~((events["last_ndvi"].to_numpy() - events["trough_ndvi"].to_numpy()) >= WATER_K * noise_o)
            radar_low = ~(np.asarray(events["rise_z_all"], dtype=float) >= WATER_K)
        dead = optical_low & radar_low & ~water_sign(events)
        two = out == 2
        out[two & dead] = 0
        out[two & ~dead] = 6
        # user decision (30 Sep, "b"): water seen on the LATEST clear view, taken after the flood, means no crop is
        # visible yet: flooded, not yet green (7), whatever a slight radar rise says (wind on open water also roughens
        # it), UNLESS the radar rose clearly after that view. NDVI below 0 is open water, a physical boundary.
        if map_date is not None and "last_clear_ndvi" in events and "flood_date" in events:
            with np.errstate(invalid="ignore"):
                age = np.asarray(events["last_clear_age"], dtype=float)
                seen_day = np.datetime64(pd.Timestamp(map_date).date()) - (np.nan_to_num(age, nan=1e6).clip(0, 1e6)
                                                                            * STEP_DAYS).astype("timedelta64[D]")
                flood_day = pd.to_datetime(events["flood_date"]).to_numpy().astype("datetime64[D]")
                last_view = np.asarray(events["last_clear_ndvi"], dtype=float)
                no_crop_yet = last_view < 0
                if LATEST_VIEW_RELATIVE:
                    # user 30 Sep (aoi160_006123, NDVI 0.11): still at the AOI's paddies-at-sowing level = nothing grown
                    no_crop_yet |= ~(level_vs_rice_sowing(events, last_view, with_spread=False) >= WATER_K)
                still_water = no_crop_yet & ~np.isnat(flood_day) & (seen_day >= flood_day) & water_sign(events)
                if "rise_since_last_clear_z" in events:
                    # ... unless the radar rose clearly after that view (aoi13_000737 / 000471: water on 17 Aug, crop from 25 Aug)
                    from .radar_water import WATER_K

                    still_water &= ~(np.asarray(events["rise_since_last_clear_z"], dtype=float) >= WATER_K)
            out[np.isin(out, (1, 6)) & still_water] = 7
        # NOT used (round 3, 30 Sep): "behind the AOI's rice of the same age" (``behind_rice_z``) turned three flooded
        # aoi116 fields into class 7 as the user judged, but dropped the aoi28 plots from 99.3 to 93.8 % and moved aoi63
        # double crops out of rice (their flood date is not their transplanting). Kept as a reported column only.
    elif map_date is not None and "flood_ok" in events and "flood_date" in events:
        with np.errstate(invalid="ignore"):
            since = (np.datetime64(pd.Timestamp(map_date).date()) - pd.to_datetime(events["flood_date"]).to_numpy()
                     .astype("datetime64[D]")) / np.timedelta64(1, "D")
            recent = events["flood_ok"].to_numpy(dtype=bool) & (since >= 0) & (since <= FLOODED_RECENT_DAYS)
        # a young rice seen only by the radar while the optical value is still open water is
        # "flooded, not yet green" (the seedlings are emerging), not a delivered canopy
        out[np.isin(out, (0, 2, 6)) & recent & (last < FLOODED_NDVI_MAX)] = 7
    if radar_trough and own:
        # user decision (30 Sep): young rice only when sown after the normal sowing window, else rice
        flood_day = pd.to_datetime(events["flood_date"]).to_numpy().astype("datetime64[D]") if "flood_date" in events \
            else np.full(len(events), np.datetime64("NaT"), dtype="datetime64[D]")
        trough_day = pd.to_datetime(events["trough_date"]).to_numpy().astype("datetime64[D]")
        sown = np.where(np.asarray(events["flood_ok"], dtype=bool) & ~np.isnat(flood_day), flood_day, trough_day)
        early = ~np.isnat(sown) & (sown < np.datetime64(YOUNG_SOWN_AFTER))
        out[(out == 6) & early] = 1
    if never_bare is not None:
        # trees, gardens and houses: the radar never saw bare ground, so the optical "cycle" is
        # haze. Applied to every rice-like class (fix plan, issues 2 and 5; before only class 3),
        # except where a confirmed flood says the ground was bare and wet after all.
        nb = np.asarray(never_bare, dtype=bool).copy()          # a real copy (issue 35)
        if "flood_ok" in events:
            nb &= ~events["flood_ok"].to_numpy(dtype=bool)
        out[np.isin(out, (1, 2, 3, 4, 6, 8)) & nb] = 5
    out[~events["valid"].to_numpy()] = 255
    return out


#: Season-long trough search for water test v3 (round 3, user 30 Sep). OFF: the AOI rice-at-sowing level was too wide in
#: aoi160 (its rice troughs spread 0.1-0.5), so the "last low moment" moved to July; delivered rice fell 1,374 -> 1,017 ac,
#: plots 99.2 -> 97.6 %, aoi28 evergreen 0 -> 10.5 %. Kept for the next attempt.
RETROUGH_V3 = True
#: Second attempt: the level is the pixel's OWN lowest fitted value of the season plus ``WATER_K`` x its own optical noise.
RETROUGH_MODE = "own"
#: Plan O5 (cut = latest value back at the field's own bare level). OFF (round 3, 30 Sep): real cut fields keep some
#: stubble / weeds above their bare level, so cut negatives became rice (aoi116 0 -> 10.8 %, aoi160 28 -> 37 %) and the
#: plots did not gain. The fixed standing test (last >= 0.40, fall <= 0.35) stays for now.
HARVEST_BACK_TO_BARE = False


def retrough_v3(events: pd.DataFrame, ndvi, lswi, windows, season=SEASON) -> pd.DataFrame:
    """Find the trough again over the whole season, as the field's last bare moment before its current crop.

    Why (user, 30 Sep, aoi160_003471): the 110-day lookback started on 3 Jun, so a field sown in mid-May lost its bare
    May and a single hazy view on 23 Jun (0.41 between 0.70 and 0.76) became its "trough". Now: the trough is the last
    window, from the season start, at or below this AOI's rice at sowing (median of the radar-confirmed rice's troughs
    plus ``WATER_K`` x sqrt(their spread^2 + the pixel's own optical noise^2)); where the field never gets that low, its
    lowest window, as before. The optical columns are replaced; the trough-anchored v1 / v2 columns stay as they were."""
    from .radar_water import WATER_K, WATER_PEERS_MIN

    noise = np.nan_to_num(np.asarray(events["ndvi_noise_own"], dtype=float)) if "ndvi_noise_own" in events else 0.0
    if RETROUGH_MODE == "own":
        # the field's own lowest point of the season, within its own noise: its last visit there is the sowing
        w = pd.DatetimeIndex(windows)
        inside = (w >= pd.Timestamp(season[0])) & (w < pd.Timestamp(season[1]))
        with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            season_min = np.nanmin(np.asarray(ndvi)[inside], axis=0)
        level = season_min + WATER_K * noise
        new = pixel_events(ndvi, lswi, windows, season, lookback_days=None, trough_level=level)
        for c in new.columns:
            events[c] = new[c].to_numpy() if c != "climb_date" else new[c].values
        events["trough_peer_z"] = trough_vs_rice(events)
        return events
    trough = np.asarray(events["trough_ndvi"], dtype=float)
    ref = np.asarray(events["flood_ok"], dtype=bool) & (np.asarray(events["rise_z_all"], dtype=float) >= WATER_K)
    if "flood_by_pattern" in events:
        ref &= ~np.asarray(events["flood_by_pattern"], dtype=bool)
    vals = trough[ref & np.isfinite(trough)]
    if len(vals) < WATER_PEERS_MIN:
        return events
    med = np.median(vals)
    spread = 1.4826 * np.median(np.abs(vals - med))
    level = med + WATER_K * np.sqrt(spread ** 2 + noise ** 2)
    new = pixel_events(ndvi, lswi, windows, season, lookback_days=None, trough_level=level)
    for c in new.columns:
        events[c] = new[c].to_numpy() if c != "climb_date" else new[c].values
    events["trough_peer_z"] = trough_vs_rice(events)
    return events


def apply_water_v3(events: pd.DataFrame, series, climb, windows, ndvi_raw, ndvi_fit=None) -> pd.DataFrame:
    """Replace the flood columns by water test v3 (``radar_water.flood_v3``: each pixel against its own noise and its own
    dry season, any track, no optical anchor) and make it the only water evidence (``radar_wet``; the trough-anchored
    v1 dip is dropped). The v2 columns stay under ``*_v2`` for comparison. The checks that hang on the flood date
    (bare field near the flood, radar canopy rise) are recomputed on the v3 date."""
    from . import radar_water

    win = pd.DatetimeIndex(windows).to_numpy().astype("datetime64[D]")
    water_seen = None
    if ndvi_raw is not None:
        with np.errstate(invalid="ignore"):
            water_seen = (win, np.asarray(ndvi_raw) < 0)       # open water on a clear view (a physical boundary)
    bare_seen = None
    if ndvi_fit is not None and ndvi_raw is not None and radar_water.SINGLE_DIP_WATER:
        # the field at its own bare low: fitted NDVI within k x its own optical noise of its season minimum
        import warnings

        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fit = np.asarray(ndvi_fit)
            low = np.nanmin(fit, axis=0) + radar_water.WATER_K * np.nan_to_num(optical_noise(ndvi_raw))
            bare_seen = (win, fit <= low[None, :])
    # the flood is searched from the season start (deliverable: "planted May 2026 or later"; user 30 Sep), not 15 May
    # the flood search keeps its 15 May start (dry-season drying fields and trees also dip in early May: aoi39, round 3)
    sowing_seen = None
    if ndvi_fit is not None and radar_water.FLOOD_AT_SOWING:
        # the field around its own sowing: fitted NDVI below half-way from its own season low to its own season peak
        # (issue 36: a radar dip under a standing canopy is not transplanting water); unknown (no fit) is not held against it
        import warnings

        fit = np.asarray(ndvi_fit)
        inside = win >= np.datetime64(SEASON[0])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            half = (np.nanmin(fit[inside], axis=0) + np.nanmax(fit[inside], axis=0)) / 2
            sowing_seen = (win, ~(fit >= half[None, :]))
    v3 = radar_water.flood_v3(series, win[-1], season_start=radar_water.FLOOD_EARLIEST, water_seen=water_seen,
                              bare_seen=bare_seen, sowing_seen=sowing_seen)
    if ndvi_raw is not None:
        # the radar's rise after the latest clear view (the user's "b" rule only holds when nothing rose since that view)
        obs = np.isfinite(np.asarray(ndvi_raw))
        last = obs.shape[0] - 1 - np.argmax(obs[::-1], axis=0)
        last_day = np.where(obs.any(axis=0), win[last], np.datetime64("NaT"))
        v3["rise_since_last_clear_z"] = radar_water.rise_since(series, last_day, win[-1])
    for c in ("flood_ok", "flood_date", "flood_vh", "flood_drop", "flood_pol", "support", "bare_near_flood",
              "radar_canopy_rise"):
        if c in events:
            events[f"{c}_v2"] = events[c]
    for c in v3.columns:
        events[c] = v3[c].to_numpy()
    events["bare_near_flood"] = radar_water.bare_near(v3["flood_date"].to_numpy(), climb, win, ndvi_raw)
    events["radar_canopy_rise"] = events["vh_end"] - events["flood_vh"]
    events["radar_wet"] = events["flood_ok"].to_numpy(dtype=bool).copy()
    if ndvi_raw is not None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            # open water most of the year: the median of all clear views is below 0 (water reflects more red than
            # near-infrared: a physical boundary, not a tuned cut-off). Ponds and rivers, not fields.
            events["open_water_year"] = np.nanmedian(ndvi_raw, axis=0) < 0
            # the pixel's own optical noise: robust scatter of second differences of its clear views (gaps skipped)
            events["ndvi_noise_own"] = optical_noise(ndvi_raw)
            events["ndvi_max_own"] = np.nanmax(ndvi_raw, axis=0)      # its highest clear view of the whole series
            events["behind_rice_z"] = behind_rice(events, ndvi_raw, win)
            events["trough_peer_z"] = trough_vs_rice(events)
    return events


#: Rule "b" read relatively (user 30 Sep, issue 37): the latest clear view, taken after the flood, shows no crop when it
#: is not ``WATER_K`` x the pixel's own noise above the TYPICAL low point of this AOI's rice at sowing (median of their
#: troughs; :func:`level_vs_rice_sowing` with ``with_spread=False``), not only when it is open water (NDVI < 0). A field
#: flooded since June whose September view reads 0.11 has nothing growing. With the rice's spread in the band (first
#: try) aoi28 plots 2970 / 2971 (young, 0.33-0.48 on 26 Sep) fell to class 7: plots 99.2 -> 96.3 %.
LATEST_VIEW_RELATIVE = True


def level_vs_rice_sowing(events: pd.DataFrame, values, with_spread: bool = True) -> np.ndarray:
    """How far ``values`` (per pixel) lie ABOVE the low points of this AOI's radar-confirmed rice (their sowing), in units
    of the rice's spread and the pixel's own optical noise; NaN without enough reference pixels. Same reference as
    :func:`trough_vs_rice`. ``with_spread=False``: in units of the pixel's own noise only (above the TYPICAL paddy at
    sowing; rule "b": aoi28 rice troughs spread so widely that a 0.45 young crop sat inside the band)."""
    from .radar_water import WATER_K, WATER_PEERS_MIN

    trough = np.asarray(events["trough_ndvi"], dtype=float)
    ref = np.asarray(events["flood_ok"], dtype=bool) & (np.asarray(events["rise_z_all"], dtype=float) >= WATER_K)
    if "flood_by_pattern" in events:
        ref &= ~np.asarray(events["flood_by_pattern"], dtype=bool)
    vals = trough[ref & np.isfinite(trough)]
    if len(vals) < WATER_PEERS_MIN:
        return np.full(len(events), np.nan)
    med = np.median(vals)
    spread = 1.4826 * np.median(np.abs(vals - med))
    noise = np.asarray(events["ndvi_noise_own"], dtype=float) if "ndvi_noise_own" in events else np.zeros(len(events))
    spread = spread if with_spread else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.asarray(values, dtype=float) - med) / np.sqrt(spread ** 2 + np.nan_to_num(noise) ** 2)


def trough_vs_rice(events: pd.DataFrame) -> np.ndarray:
    """How far the pixel's low point (``trough_ndvi``) lies ABOVE the low points of this AOI's radar-confirmed rice, in
    units of their spread and the pixel's own optical noise.

    Why (round 3): tree lines beside paddies share their radar flood (5 x 5 box) and a haze dip gives them a fitted
    "trough" of 0.56-0.68; a paddy at sowing is bare or under water, as low as the AOI's other paddies. The reference is
    the pixels with a peer-confirmed flood (not a pattern flood) and a radar crop rise, as in :func:`behind_rice`."""
    return level_vs_rice_sowing(events, events["trough_ndvi"])


def behind_rice(events: pd.DataFrame, ndvi_raw, win, sample: int = 200_000, seed: int = 0) -> np.ndarray:
    """How far the pixel's latest clear view lies below what this AOI's radar-confirmed rice showed at the same time
    after its flood, in units of the combined scatter (the rice's spread at that age and the pixel's own noise).

    Why (user verdicts, round 3): fields flooded for two to six months whose latest clear view is still water or bare
    soil (NDVI 0.07-0.29) had the same latest NDVI as young rice seen two to five weeks after its flood. The difference is
    time: rice greens up after the water at a pace the AOI's own rice shows. The reference is built from pixels with a
    peer-confirmed flood (not a pattern flood) and a radar crop rise (``rise_z_all`` >= ``WATER_K``), binned by
    series windows since the flood. NaN where the latest clear view is not after the flood or the age has too few
    reference pixels (``radar_water.WATER_PEERS_MIN``)."""
    import warnings

    from .radar_water import WATER_K, WATER_PEERS_MIN

    raw = np.asarray(ndvi_raw)
    n_win, n = raw.shape
    flood = pd.to_datetime(events["flood_date"]).to_numpy().astype("datetime64[D]")
    has = ~np.isnat(flood)
    f_idx = np.where(has, np.searchsorted(win, np.where(has, flood, win[0])), -1)
    ref = has & np.asarray(events["flood_ok"], dtype=bool) & (np.asarray(events["rise_z_all"], dtype=float) >= WATER_K)
    if "flood_by_pattern" in events:
        ref &= ~np.asarray(events["flood_by_pattern"], dtype=bool)
    ref_pix = np.flatnonzero(ref)
    if len(ref_pix) > sample:
        ref_pix = np.sort(np.random.default_rng(seed).choice(ref_pix, sample, replace=False))
    out = np.full(n, np.nan)
    if len(ref_pix) < WATER_PEERS_MIN:
        return out
    # reference: NDVI of the rice's clear views by age (windows since its flood)
    ages = np.arange(n_win)[:, None] - f_idx[ref_pix][None, :]
    vals = raw[:, ref_pix]
    keep = np.isfinite(vals) & (ages >= 0)
    a, v = ages[keep], vals[keep]
    med = np.full(n_win, np.nan)
    spread = np.full(n_win, np.nan)
    for age in np.unique(a):
        x = v[a == age]
        if len(x) >= WATER_PEERS_MIN:
            med[age] = np.median(x)
            spread[age] = 1.4826 * np.median(np.abs(x - med[age]))
    # the pixel's latest clear view and its age since the flood
    obs = np.isfinite(raw)
    last = n_win - 1 - np.argmax(obs[::-1], axis=0)
    last_val = np.where(obs.any(axis=0), raw[last, np.arange(n)], np.nan)
    age = last - f_idx
    ok = has & (age >= 0) & obs.any(axis=0)
    age_c = np.clip(age, 0, n_win - 1)
    noise = np.asarray(events["ndvi_noise_own"], dtype=float) if "ndvi_noise_own" in events else np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        z = (med[age_c] - last_val) / np.sqrt(spread[age_c] ** 2 + np.nan_to_num(noise) ** 2)
    out[ok] = z[ok]
    return out


def optical_noise(ndvi_raw) -> np.ndarray:
    """Per pixel: 1.4826 x median |second difference| / sqrt 6 of its clear NDVI values in time order (gaps removed).
    A steady green-up cancels in the second difference; the few real jumps are ignored by the median."""
    raw = np.asarray(ndvi_raw, dtype="float64")
    n = raw.shape[1]
    out = np.full(n, np.nan)
    seen = np.isfinite(raw)
    counts = seen.sum(axis=0)
    # pack each pixel's observations to the top, in time order, then take second differences column by column
    order = np.argsort(~seen, axis=0, kind="stable")
    packed = np.take_along_axis(raw, order, axis=0)
    d2 = packed[2:] - 2 * packed[1:-1] + packed[:-2]
    valid = np.arange(d2.shape[0])[:, None] < (counts - 2)[None, :]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = 1.4826 * np.nanmedian(np.where(valid, np.abs(d2), np.nan), axis=0) / np.sqrt(6.0)
    return out


def aoi_events(aoi_id: int, out_root="processed/_batch/s2_2026", season=SEASON,
               radar_season: str | None = "monsoon2026", water: str = WATER_DEFAULT,
               lookback_days: int = LOOKBACK_DAYS):
    """The per-pixel evidence the rule decides on: ``(series, events, radar)``.

    ``events`` holds, per grid pixel, the optical events of :func:`pixel_events` and, when the
    Sentinel-1 season run exists (``radar`` True), the radar dips of ``radar_water.pixel_dips``.
    Kept separate from :func:`run_aoi` so reports can look at the evidence behind a class, not only
    the class.
    """
    from . import radar_water

    d = nd.load(aoi_id, out_root=out_root)
    season = season_to_series(season, d["windows"])     # a longer series moves the season end with it
    ndvi = d["ndvi5d"].reshape(d["ndvi5d"].shape[0], -1)
    lswi = d["lswi5d"].reshape(d["lswi5d"].shape[0], -1)
    events = pixel_events(ndvi, lswi, d["windows"], season, lookback_days=lookback_days)
    radar = bool(radar_season) and radar_water.season_run_exists(aoi_id, radar_season)
    if radar:
        trough = np.where(events["valid"].to_numpy(), events["trough_date"].to_numpy().astype("datetime64[D]"),
                          np.datetime64("NaT"))
        dips = radar_water.pixel_dips(aoi_id, trough, d["loc"]["grid"], radar_season)
        events = pd.concat([events, dips], axis=1)
        if water in ("v2", "v3"):
            climb = pd.to_datetime(events["climb_date"]).to_numpy().astype("datetime64[D]")
            # a young crop has not climbed yet: its flood is searched up to the map date instead
            last = np.datetime64(pd.DatetimeIndex(d["windows"])[-1].date())
            climb = np.where(np.isnat(climb), last, climb)
            climb = np.where(events["valid"].to_numpy(), climb, np.datetime64("NaT"))
            raw = d["ndvi5d_raw"].reshape(d["ndvi5d_raw"].shape[0], -1) if "ndvi5d_raw" in d else None
            series = radar_water.read_series(aoi_id, radar_season)
            v2 = radar_water.water_evidence(aoi_id, trough, climb, ndvi, d["windows"], radar_season, ndvi_raw=raw,
                                            series=series)
            events = pd.concat([events, v2], axis=1)
            events["radar_wet_v1"] = events["radar_wet"]
            events["radar_wet"] = events["radar_wet"] | events["flood_ok"]
            if water == "v3":
                events = apply_water_v3(events, series, climb, d["windows"], raw, ndvi_fit=ndvi)
                if RETROUGH_V3:
                    events = retrough_v3(events, ndvi, lswi, d["windows"], season)
                # radar brightened right after sowing (argues against transplanting water; user 30 Sep, aoi160_004992)
                events["brightened_at_sowing"] = radar_water.brightened_at(
                    series, pd.to_datetime(events["trough_date"]).to_numpy(),
                    until=pd.to_datetime(events["climb_date"]).to_numpy())
                # the climb must be seen twice (two clear views) or by the radar's VH (CLIMB_TWO_VIEWS)
                trough_day = pd.to_datetime(events["trough_date"]).to_numpy()
                events["rise_since_trough_z"] = radar_water.rise_since(
                    series, trough_day, np.datetime64(pd.DatetimeIndex(d["windows"])[-1].date()), pols=("VH",))
                if raw is not None:
                    events["views_above_trough"] = views_above_trough(raw, d["windows"], trough_day,
                                                                      events["trough_ndvi"], events["ndvi_noise_own"])
                    events["max_view_after_trough"] = max_view_after(raw, d["windows"], trough_day)
            del series
            events = pd.concat([events, radar_trough_events(events, ndvi, d["windows"])], axis=1)
            if water == "v3" and raw is not None and "ndvi_noise_own" in events:
                events["views_above_flood"] = views_above_trough(raw, d["windows"], events["flood_date"],
                                                                 events["ndvi_at_flood_fit"], events["ndvi_noise_own"])
                events["max_view_after_flood"] = max_view_after(raw, d["windows"], events["flood_date"])
            if raw is not None:
                events = pd.concat([events, last_clear_view(raw, d["windows"])], axis=1)
    return d, events, radar


def median_day(dates) -> str | None:
    """Median of a date column as "dd Mon", None when it holds no date (an all-empty column can come
    back as float NaN, which has no ``strftime``: issue 34)."""
    d = pd.to_datetime(pd.Series(dates), errors="coerce").dropna()
    return d.median().strftime("%d %b") if len(d) else None


def run_aoi(aoi_id: int, out_root="processed/_batch/s2_2026", season=SEASON, inside_only: bool = True,
            radar_season: str | None = "monsoon2026", water: str = WATER_DEFAULT, suffix: str = "",
            radar_trough: bool = RADAR_TROUGH_DEFAULT, fallback_days: int | None = LOOKBACK_FALLBACK_DAYS) -> dict:
    """Classify one AOI, write ``<aoi>_monsoon2026.tif``, return the acres per class.

    ``radar_season`` names the Sentinel-1 season run used to confirm the water; when that run does
    not exist for the AOI every phenology-rice pixel lands in class 3 and ``radar`` says ``False``.
    """
    import rasterio
    from rasterio.transform import from_origin

    d, events, radar = aoi_events(aoi_id, out_root, season, radar_season, water)
    shape = d["ndvi5d"].shape[1:]
    classes = classify(events, radar_wet=events["radar_wet"] if radar else None,
                       never_bare=events["never_bare"] if radar and "never_bare" in events else None,
                       radar_trough=radar_trough and radar, map_date=pd.DatetimeIndex(d["windows"])[-1])
    if fallback_days and fallback_days > LOOKBACK_DAYS and (classes == 0).any() and not (water == "v3" and RETROUGH_V3):
        # second pass with the wider window, taken only where the first found nothing
        import gc

        none = classes == 0
        del events
        gc.collect()
        _, events, _ = aoi_events(aoi_id, out_root, season, radar_season, water, lookback_days=fallback_days)
        classes2 = classify(events, radar_wet=events["radar_wet"] if radar else None,
                            never_bare=events["never_bare"] if radar and "never_bare" in events else None,
                            radar_trough=radar_trough and radar, map_date=pd.DatetimeIndex(d["windows"])[-1])
        classes = np.where(none, classes2, classes).astype("uint8")
    if inside_only:
        classes[~nd.inside_aoi(aoi_id)] = 255
    grid = d["loc"]["grid"]
    path = Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026{suffix}.tif"
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype="uint8", crs=grid["crs"],
                   nodata=255, compress="deflate", tiled=True,
                   transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]))
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(classes.reshape(shape), 1)
    if "brightened_at_sowing" in events:
        # class-3 pixels whose radar brightened at sowing with no water sign: the per-pixel class-3 relabel skips them
        against = (classes == 3) & np.asarray(events["brightened_at_sowing"], dtype=bool) & ~water_sign(events)
        with rasterio.open(path.with_name(f"{d['loc']['aoi']}_monsoon2026_c3_radar_against.tif"), "w", **profile) as ds:
            ds.write(against.reshape(shape).astype("uint8"), 1)
    counts = {name: int((classes == code).sum()) for code, name in CLASSES.items() if code != 255}
    inside = classes != 255
    wet = events["wet_at_trough"].to_numpy()
    n_rice = counts["rice"] + counts["rice_unconfirmed"]
    return {"aoi": d["loc"]["aoi"], "radar": radar, "water": water if radar else None,
            **{f"{k}_acres": round(acres(v), 1) for k, v in counts.items()},
            "rice_pct_of_decided": round(100 * n_rice / max(inside.sum(), 1), 1),
            "radar_wet_pct_of_rice": round(100 * counts["rice"] / max(n_rice, 1), 1),
            "radar_checkable_pct_of_rice": (round(100 * float(events["radar_checkable"].to_numpy()[
                np.isin(classes, (1, 3))].mean()), 1) if radar and n_rice else None),
            "optical_wet_pct_of_rice": round(100 * float(wet[np.isin(classes, (1, 3))].mean()) if n_rice else 0.0, 1),
            "trough_median": median_day(events.loc[np.isin(classes, (1, 3)), "trough_date"]),
            "climb_median": median_day(events.loc[np.isin(classes, (1, 3)), "climb_date"]),
            "path": str(path)}


def plot_recall(aoi_id: int, plots, out_root="processed/_batch/s2_2026", suffix: str = "") -> dict:
    """Share of field-plot pixels the map calls rice / young / not rice, for one AOI."""
    import rasterio

    from .plot_curves import plot_pixels

    d = nd.load(aoi_id, out_root=out_root)
    with rasterio.open(Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026{suffix}.tif") as ds:
        classes = ds.read(1).ravel()
    pix = np.concatenate(list(plot_pixels(plots, d["loc"]["grid"]).values()))
    c = classes[pix]
    n = max(int((c != 255).sum()), 1)
    return {"aoi": d["loc"]["aoi"], "plot_pixels": int(len(pix)),
            **{f"{name}_pct": round(100 * float((c == code).sum()) / n, 1)
               for code, name in CLASSES.items() if code != 255}}


def plot_recall_by_position(aoi_id: int, plots, out_root="processed/_batch/s2_2026") -> pd.DataFrame:
    """Class shares of field-plot pixels split into plot interior and plot edge, plus per-plot shares.

    Returns a frame with rows ``interior`` and ``edge`` (class percentages) and the attribute
    ``per_plot``: per plot, the share of its pixels called rice, to find whole plots that disagree.
    """
    import rasterio

    from .plot_curves import interior_pixels, plot_pixels

    d = nd.load(aoi_id, out_root=out_root)
    with rasterio.open(Path(out_root) / d["loc"]["aoi"] / f"{d['loc']['aoi']}_monsoon2026.tif") as ds:
        classes = ds.read(1).ravel()
    pixels = plot_pixels(plots, d["loc"]["grid"])
    interior = interior_pixels(pixels, int(d["loc"]["grid"]["width"]))
    rows, per_plot = {}, []
    for name, want in (("interior", True), ("edge", False)):
        pix = np.concatenate([p[interior[k] == want] for k, p in pixels.items()]) if pixels else np.array([], int)
        c = classes[pix.astype(int)] if len(pix) else np.array([], "uint8")
        n = max(int((c != 255).sum()), 1)
        rows[name] = {"pixels": int(len(pix)), **{f"{label}_pct": round(100 * float((c == code).sum()) / n, 1)
                                                 for code, label in CLASSES.items() if code != 255}}
    for k, p in pixels.items():
        c = classes[p]
        per_plot.append({"plot_id": k, "n_pixels": len(p), "rice_pct": round(100 * float(np.isin(c, (1, 3)).mean()), 1),
                         "confirmed_pct": round(100 * float((c == 1).mean()), 1)})
    frame = pd.DataFrame(rows).T
    frame.attrs["per_plot"] = pd.DataFrame(per_plot)
    return frame
