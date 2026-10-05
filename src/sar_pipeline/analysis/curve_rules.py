"""Hybrid rule (NDVI + VH + VV) learned by reading the user's labelled curves (second fresh start, 1 Oct 2026).

Why
---
User (1 Oct): "from these pixels, see whether a hybrid approach can be built that combines NDVI, VH and VV into a
classification formula; direct-seeded rice is a separate class from transplanted rice". While labelling, the user and
Claude read the same few things on every pixel (``curve_labels.describe``): when the field was empty, whether water
stood in it (optical: NDVI near 0 with LSWI above it; radar: VV AND VH down together on one pass), whether the radar
rose after the water (young plants in water raise VV), how fast and how long the crop grew, and what it looks like on
the newest image. This module measures exactly those things per pixel (``features``) so the rule can be written from
them and checked against the labels (``table``).

Use::

    python -m sar_pipeline.analysis.curve_rules table --aoi 160      # the labelled pixels with their measured features
    python -m sar_pipeline.analysis.curve_rules run --aoi 160        # the relative rule on the whole AOI -> map, and
                                                                     # the same map sieved at 0.5 ac (_sieved.tif)
    python -m sar_pipeline.analysis.curve_rules fields --aoi 160     # delineated fields labelled from the sieved map
    python -m sar_pipeline.analysis.curve_rules try-rules --aoi 116  # step 1 on a new AOI: raw map with each finished
                                                                     # AOI's rule set side by side (rule_trials/)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from .curve_labels import YOUNG_MAX_DAYS  # the user's definition of young rice (days on the newest image)

START = "2026-05-01"
#: Crop age counts from the last window the fitted NDVI was within this share of its own rise above its low (aoi160:
#: 0.2). aoi28 (user, 2 Oct, pixel 3345): 0.2 put the start mid-way up the green-up (12 Aug, "45 days") for a crop
#: sown end of July; the end of the empty spell is 0.05.
AGE_LOW_SHARE = 0.2
#: The radar-low path calls a green, still-rising crop young rice without asking its age (aoi160). True: a green crop
#: older than ``YOUNG_MAX_DAYS`` skips that path and is judged as a grown crop (aoi28, pixel 3345: 55 days).
YOUNG_NEEDS_AGE = False
#: Tree / orchard also needs its NDVI low BEFORE this date (dry-season dip); None = not asked (aoi160). aoi28 (user,
#: 2 Oct, pixel 7826): tried (a field whose low came in late June is a crop) and reverted by the user: not used.
TREE_LOW_BEFORE = None
#: When the newest clear view shows water and the radar rose since (VV and VH half their own range, or a VV jump):
#: young rice, whatever the optical curve says (aoi72, user 2 Oct, pixel 42109: water on 13 Sep, no view on 26 Sep,
#: VV -15.7 -> -8.2; the rule had called it "harvested"). Off for aoi160 / aoi28.
YOUNG_AFTER_LAST_WATER = False
#: Young rice also when the radar has not yet reached the top of its own range (VH below ``LOW_NOW``) and shows plants
#: coming up, while the newest clear view is the season's greenest and still rising: the canopy is not closed yet,
#: whatever the day count says (aoi72, user 2 Oct, pixel 19247: water to late July, VV up from early August, NDVI only
#: 0.47 on 13 Sep and still climbing; the fit held flat after that view looked like "15 days at the top" and the age
#: from the first water was 88 days). Off for aoi160 / aoi28.
YOUNG_WHILE_RADAR_LOW = False
#: ...and only while the radar is still climbing on its last passes (VH up ``RISING`` over the last three, or a VV jump
#: on the last pass), not merely risen once since an earlier low: a dense / ripening canopy LOWERS the radar while its
#: NDVI is at the top (aoi39, user 3 Oct, pixel 11226: water to 8 Aug, NDVI 0.88 on 13 Sep, VV -7 -> -13.4 in
#: September: transplanted standing, not young). Off = the aoi72 / aoi116 behaviour (locked).
YOUNG_RADAR_LOW_NEEDS_RISE = False
#: The "grown / ripening canopy" guard (``HELD_DAYS``: a canopy at its top for weeks keeps a low radar off the flooded
#: path) does not apply when the newest clear view shows open water: a ripening canopy never looks like water on the
#: optical (aoi39, user 3 Oct, pixel 21405: no clear view 11 May - 24 Aug, the fit's July "top" was interpolated, under
#: water from mid July to 13 Sep (NDVI -0.85): flooded, not harvested). Off for the locked AOIs.
GROWN_NOT_IF_WATER_VIEW = False
#: Permanent water (fish pond, lake): most clear views of the DRY season (``DRY_SEASON_FROM`` to ``DRY_UNTIL``) show water
#: (NDVI below 0, or LSWI above NDVI) -> flooded / bare. A paddy is dry or bare in March-April; a pond holds water all
#: year (aoi39, user 3 Oct, pixel 30425: teal rectangle with thick bunds in every chip, NDVI -0.5..-1.0 in April, radar
#: dark all year; the rule called it direct seeded). Off for the locked AOIs.
POND_IF_DRY_WATER = False
#: A second crop: a DEEP radar water spell that began after the NDVI curve had already left its top (an earlier crop's
#: peak) is a new, transplanted crop, classed by the radar and its age from that water (young up to ``YOUNG_MAX_DAYS``,
#: else standing transplanted), whatever the NDVI features of the first crop say (aoi39, user 3 Oct, pixel 24716: a
#: crop to NDVI 0.64 in May-June, deep water 15 Jul - end Aug, radar up in Sep; the rule said tree/orchard). "Deep":
#: on one pass VV AND VH both fall at least ``WATER_DEPTH_K`` times their own pass-to-pass wobble below their own
#: dry-season level (``water_depth``), so a tree's small dip does not count (aoi39: trees 90 % below 2.1). Off for the
#: locked AOIs. The crop must also be up now (VH out of the lower part of its range, ``LOW_NOW``): a field still under
#: that water has no crop yet (aoi39 pixel 21405: flooded from mid July, VH at 0.13 of its range -> stays flooded).
SECOND_CROP_BY_RADAR = False
WATER_DEPTH_K = 3.0
#: User rule (3 Oct 2026, aoi39 pixel 21618): "whenever and wherever there is no NDVI, the decision is made by the radar
#: alone". When the radar shows a DEEP water spell (``WATER_DEPTH_K``) and the optical did not see that spell (no clear
#: view showing water during it, and fewer than ``OPTICAL_VIEWS_MIN`` clear views in it), the NDVI tests (tree / orchard,
#: other vegetation, harvested) are not used: radar low now with no plants coming up -> flooded / bare, otherwise
#: transplanted rice, young or standing by its age (a dense canopy lowers the radar too: aoi39 pixel 11226). Off
#: by default (the locked AOIs keep their results); on for aoi39 and, by the user's decision, for every new AOI.
RADAR_DECIDES_WITHOUT_OPTICAL = False
#: No "rice harvested" while the radar is at the top of its own range and still rising (VV and VH both up half their
#: range recently, VH not in its lower part): a cut field's radar does not climb to its season top. Such a crop is
#: standing: transplanted when a water spell or deep water (``WATER_DEPTH_K``) was seen, else direct seeded. Why (aoi39,
#: user 5 Oct, pixels 37571 / 37570 / 39894: one hazy-looking 18 Sep view at NDVI 0.35 after 0.66 on 13 Sep, measured
#: against the April summer crop's peak, read as harvested while VH was at its season top). Off for the locked AOIs.
HARVEST_NOT_IF_RADAR_RISING = False
#: Young rice only when the VH has kept climbing for at least this many days since its last low (the water or the cut);
#: a shorter rise is a field still at water level -> flooded / bare (aoi39, user 5 Oct, pixel 43450: cut, water from
#: mid July, VH up on the last passes only, 12 days: flooded; "young rice needs VH rising for about 40-50 days"; the
#: user relabelled 8391 / 15496 / 25885, 12-24 days, as flooded too). None = not asked (the locked AOIs).
YOUNG_MIN_RISE_DAYS = None
#: Young rice only on a field that held water at some point (radar below its own dry-season level, ``water_depth`` > 0):
#: young rice comes up after the water or the cut; a field that never held water and is green is standing direct-seeded
#: rice (aoi39, user 5 Oct, pixels 29972 / 26437 / 13545: radar never below its dry level, NDVI 0.75-0.92 on 13 Sep, the
#: rule said young). Off for the locked AOIs.
YOUNG_NEEDS_WATER = False
#: Not "flooded / bare" while the newest clear view is the season's greenest and the crop greened over weeks (not
#: faster than ``FAST_RISE_DAYS``): a closing canopy lowers VV and VH (as aoi39 pixel 11226), the radar low is not water.
#: Such a crop is young or standing by its age, transplanted when the radar saw a water spell. Why (aoi63, user 5 Oct,
#: pixel 20447: water late June, NDVI climbing to 0.86 on 28 Sep, VV / VH falling in September -> the rule said
#: flooded; the user: transplanted rice). Off for the locked AOIs.
CANOPY_NOT_FLOODED = False
#: Where the optical never saw the crop's green-up (no clear view between the fitted curve's 10 % and 90 % of its rise:
#: ``rise_unseen``), the radar tells the story and the NDVI only gives its clear views. The fitted curve across such a gap
#: is a smooth line between two views: its rise length, top and peak dates are drawn, not seen, so the NDVI timing tests
#: (other vegetation, grown / past the peak, radar-low-but-green) are skipped:
#: * pond / radar-only / tree come first as usual, but a field that stood in water is not a tree;
#: * newest clear view bare (less than ``HARVEST_LEFT`` of its own amplitude): flooded / bare when it shows water or the
#:   radar is low and not rising, else harvested when the field held water (a paddy that was cut), else flooded / bare;
#: * newest clear view green: a crop now; young or standing by its age (from the water, else from the radar's last low),
#:   transplanted when the radar saw a water spell or deep water.
#: Why (aoi63, user 5 Oct): no clear view from 3 Jul to 26 Sep (85 days); "the NDVI rises over the whole AOI while VV
#: and VH swing a lot: they do not justify each other". The swings are the real season (water, planting, cut); the
#: rise is the fit's line. Off for the locked AOIs.
RADAR_STORY_IN_GAP = False
#: A wet pass with no clear view within one 5-day step counts as "while the crop was small" (water spell): there the
#: fitted NDVI is the line across the cloud gap, not a canopy that was seen. Why (aoi63, user 5 Oct, pixel 31277: summer
#: crop 0.94 in May; July water on both tracks (VH -21.6 / -20.0, VV down 4 dB); the fit's peak is the summer crop, so
#: the July fit 0.60 read as "a grown crop" and no water spell was found -> tree; the user: transplanted rice, sown early
#: June / early July). Off for the locked AOIs.
WATER_SMALL_UNSEEN = False
#: A pond (``POND_IF_DRY_WATER``) only when the newest clear view is NOT green on the pixel's whole-year range: a field
#: flooded in the dry season (summer-rice water) that carries a crop now is a paddy, not a pond. Why (aoi63, user 5 Oct,
#: pixel 41938: water on the early-April views (LSWI above NDVI, VH -24 to -27), a summer crop to 0.89 in May, water
#: again in July, NDVI 0.86 on 28 Sep -> the rule said flooded; the user: transplanted). Off for the locked AOIs.
POND_NEEDS_NO_CROP = False
#: Radar low now (VH in the lower part of its own range), but the newest clear view is the pixel's greenest of the whole
#: year and the optical saw the field empty more than ``YOUNG_MAX_DAYS`` before it: the radar's late low is a dense
#: canopy, not water, and the crop is not young: standing rice, transplanted when the water came early or deep, else
#: direct seeded. Pixels 18853 / 24865 (same day): the late radar low read as a water spell gave "young, 37 days" and
#: then "transplanted"; seen empty 150 days before, no June VV dip -> the user: direct seeded. Why (aoi63, user 5 Oct, pixel 13571: empty to early May, green-up
#: seen from 19 May, VH dip in June, VV / VH falling Aug - Sep to their season low (VV -18.2 on 25 Sep) while NDVI was
#: 0.81 on 28 Sep -> the rule said flooded; the user: transplanted). Off for the locked AOIs.
CANOPY_OVER_RADAR_WATER = False
#: Never "flooded / bare" when the newest clear view is green on the pixel's whole-year range (at least ``HARVEST_LEFT``
#: of it) and does not show water: the optical's last word is a canopy, and it is the only view after the cloud gap.
#: The crop's age is from the last clear view of the empty field (else ``crop_age``): young up to ``YOUNG_MAX_DAYS``,
#: else standing; transplanted when the water came early (spell older than ``YOUNG_MAX_DAYS``) or deep
#: (``WATER_DEPTH_K``), else direct seeded. Applied before the young-rice checks (``YOUNG_NEEDS_WATER``,
#: ``YOUNG_MIN_RISE_DAYS``), which still decide young rice. Why (aoi63, user 5 Oct: 36796, water view 3 Jul + radar dip
#: mid July, NDVI 0.79 on 28 Sep -> flooded, "transplanted, a clear radar dip"; 23348, bare 7 Aug, 0.84 on 28 Sep,
#: radar falling now -> flooded, "not smooth like a flooded field: direct seeded"). Off for the locked AOIs.
GREEN_VIEW_NOT_FLOODED = False
#: Young rice whose field shows a clear radar water dip (a water spell AND VV and VH both deep below their own dry-season
#: level, ``WATER_DEPTH_K``) is called transplanted rice, whatever its age; applied after ``YOUNG_MIN_RISE_DAYS``, so a
#: field whose radar has not yet risen that long stays flooded / bare. Why (aoi13, user 5 Oct: everything harvested
#: until ~18 Jul, water to late August, planted after it, so on 1 Oct the crops are 30-50 days old; "lower the 40 days
#: to 30 for this AOI only, and wherever VV and VH show a clear dip put it in transplanted rice even at 30 days").
#: Off for the locked AOIs.
DIP_MAKES_TRANSPLANTED = False
#: With ``SOWING_FROM_RADAR``: the sowing is never later than the optical saw the crop rising. When a clear view after
#: the field's lowest (empty / water) view already shows the crop above ``AGE_LOW_SHARE`` of its own NDVI amplitude, and
#: that view is earlier than the radar's end of the water spell, the crop was planted at that lowest view; a later radar
#: dip is the canopy or rain, not the transplanting water. Why (aoi13, user 5 Oct, pixel 14994: water view 18 Jul,
#: NDVI 0.33 on 17 Aug, a dip on 29 Aug -> sowing 29 Aug, 33 days, young; the user: sown mid July, transplanted).
#: Off for the locked AOIs.
SOWING_NOT_AFTER_SEEN_CROP = False
#: In the radar story (``RADAR_STORY_IN_GAP``): the newest clear view shows water but VV and VH both rose since
#: (``radar_rose``) -> young rice, then the usual young-rice checks (``YOUNG_MIN_RISE_DAYS``, ``DIP_MAKES_TRANSPLANTED``).
#: Without it, an old water view decided "flooded" whatever the radar did after. Why (aoi13, user 5 Oct, pixel 11791:
#: deep water June - end August, last clear view 19 Aug (water), radar up from 6-10 Sep, VV +12 dB; the user:
#: transplanted end of August). Off for the locked AOIs.
STORY_YOUNG_AFTER_LAST_WATER = False
#: With ``YOUNG_MIN_RISE_DAYS``: a young crop whose radar has risen a shorter time stays YOUNG RICE (not flooded / bare)
#: when the newest clear view shows it green (at least ``HARVEST_LEFT`` of the whole-year range, not water). Why (aoi13,
#: user 5 Oct, pixel 16003: water to end August, radar up from 10 Sep (24 days), 1 Oct NDVI 0.67 -> the rule said
#: flooded; the user: young rice. 9643, whose 1 Oct view is still water, stays flooded). Off for the locked AOIs.
SHORT_RISE_GREEN_YOUNG = False
#: A pixel that is clear on the AOI's LATEST image and still empty there (its NDVI within ``AGE_LOW_SHARE`` of its own
#: whole-year range above its lowest view: the bare / water level) is not rice: flooded / bare, whatever the radar or
#: the fitted curve say. Pixels cloudy on the latest image keep the curve's class. Why (aoi13, user 5 Oct: "after the
#: latest image arrives, every pixel that is still empty is certainly not rice -> flooded or bare; the green ones by
#: their curve; the cloudy ones by the same curve"). Off for the locked AOIs.
NEWEST_EMPTY_NOT_RICE = False
#: UNIVERSAL rules for every new AOI (user, 5 Oct 2026, after aoi13; ``NEW_AOI_SWITCHES``):
#: TONE_AND_CURVE - a class must fit the tone of the pixel's latest clear view (walking back past cloudy dates):
#: ``water`` (the view shows water), ``green`` (at least half of its own whole-year NDVI range), ``empty`` (otherwise).
#: The NEWER news wins: when VV and VH BOTH moved by ``RISE_BOTH`` of their own range on the passes after the view, the
#: radar decides (both fell: the crop was cut -> harvested; both rose: plants came up -> a crop).
#: * green tone: never flooded / bare or harvested, unless both fell after the view (harvested); a rice class by the
#:   curve (young by age, transplanted with a water spell, else direct seeded);
#: * empty tone: never standing or young rice, unless both rose after the view (a new crop: young rice); else
#:   harvested when both fell from their recent top, flooded / bare otherwise;
#: * water tone: standing rice only if both rose after the view (young rice), else flooded / bare.
#: Pixels with no clear view keep the curve's class (radar only where there is no NDVI).
TONE_AND_CURVE = False
#: UNIVERSAL: "rice harvested" only when VV AND VH both fell from their recent top by ``RISE_BOTH`` of their own range
#: (the NDVI agreeing confirms it); one polarisation still up = the crop is standing (user, 5 Oct: "if VV or VH says the
#: crop is still there and VH is not falling, do not put it in harvested").
HARVEST_NEEDS_BOTH_POLS = False
OPTICAL_VIEWS_MIN = 2
DRY_SEASON_FROM = "2026-03-01"
#: Tree / orchard only when the radar saw NO water spell (two wet passes while the crop was small): trees do not stand in
#: water for weeks. Why (aoi116, user 2 Oct, pixel 168977): no clear view from 31 May to 13 Sep, so the NDVI low/peak
#: (0.38 / 0.61) said "never emptied" while both radar tracks showed water mid June - mid July and a crop rising after.
#: User rule: where the optical has no view, the radar decides. Off for aoi160 / aoi28 / aoi72.
TREE_NEEDS_NO_WATER = False
#: Sowing / transplanting date (and the crop age that decides young vs standing) from the RADAR wherever the optical
#: cannot see it (user, 2 Oct, aoi116 pixel 186792: "why not give the radar priority for the sowing date when we know
#: NDVI is not available"): a transplanted crop (radar water spell) was planted at the END of its water spell, the last
#: pass at water level before the radar rises (seedlings in water are invisible to the optical; 186792: water mid June -
#: end of August, radar up on 10-13 Sep: planted ~1 Sep, not 26 Jun when the water came); any other crop at the end of
#: its empty NDVI spell when a clear view lies within one series step of it, else at the last pass with VH at the bottom
#: of its own range before its season top. ``sowing_from`` says which. Off for aoi160 / aoi28 / aoi72 (locked).
SOWING_FROM_RADAR = False
#: How the tracks' "last pass at water level" dates combine into one transplanting date: "median" (the default way every
#: radar feature is combined); "earliest": the first track to show the plants rising (aoi116, user 2 Oct, pixel 116025:
#: the ascending track's VV rose with plants from 8 Aug while the descending track read water again on 17 / 29 Aug; the
#: median, 12 Aug, made a crop at NDVI 0.79 on 13 Sep "31 days old"); "earliest_unless_wet_view": the earliest, unless
#: a clear optical view AFTER it still shows water (LSWI above NDVI): then the water had not ended and the latest track
#: is right (aoi116 pixel 143310: same track split as 116025, but 24 Aug view water and NDVI 0.37 on 13 Sep: young);
#: also the latest when the pixel's newest clear view is greener than fewer than ``BEHIND_SHARE`` of the AOI's clear
#: pixels that day: a crop far behind nearly all fields was planted later (aoi116 pixel 108161: no view from 8 May to
#: 13 Sep, NDVI 0.41 on 13 Sep while 90 % of the AOI's fields were above 0.57; 116025 at 0.79 stays early).
#: User rule: the radar decides where the optical cannot see; where a clear view exists, it settles the tracks' dispute.
WATER_END_TRACKS = "median"
#: See ``WATER_END_TRACKS``: "behind nearly all fields" = below this share (percentile / 100) of the AOI's clear NDVI on
#: the pixel's newest clear date. A date seen clear on less than ``BEHIND_MIN_CLEAR`` of the AOI is not used.
BEHIND_SHARE = 0.10
BEHIND_MIN_CLEAR = 0.25
#: Crop age of a transplanted crop (water spell while small) counts from the water spell's first pass, not from the end
#: of the low NDVI spell (aoi72, user 2 Oct, pixel 27550). Off for aoi160 / aoi28.
AGE_FROM_WATER = False
#: A radar pass is "water" when VV and VH are BOTH this many times the track's own pass-to-pass wobble below the
#: median of the three passes before it (user, 1 Oct: one pass is enough, water can dry within a week; geeli mitti
#: raises both, water lowers both).
WATER_K = 2.0


#: A wet pass: VH and VV both within this many dB of the track's own second-lowest pass since May, and VH below
#: ``WET_VH_MAX`` (a tree's whole season sits near its own low).
WET_NEAR_LOW_DB = 1.5
WET_VH_MAX = -18.0
#: The crop is still small (transplanting time) while the fitted NDVI is below this.
SMALL_CROP_NDVI = 0.35


def _radar_events(dates, vv, vh, k: float = WATER_K):
    """Per pixel (columns of ``vv`` / ``vh``, passes in rows): boolean (passes, pixels) of water passes, and the VV
    rise after each pass (max VV later minus VV on the pass)."""
    from .radar_water import pass_noise

    vv = np.asarray(vv, dtype="float32")
    vh = np.asarray(vh, dtype="float32")
    nv, nh = pass_noise(vv), pass_noise(vh)
    water = np.zeros(vv.shape, dtype=bool)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(1, len(vv)):
            ref_v = np.nanmedian(vv[max(0, i - 3):i], axis=0)
            ref_h = np.nanmedian(vh[max(0, i - 3):i], axis=0)
            water[i] = (vv[i] <= ref_v - k * nv) & (vh[i] <= ref_h - k * nh)
        later_max = np.vstack([np.fmax.accumulate(vv[::-1], axis=0)[::-1][1:], np.full((1, vv.shape[1]), np.nan)])
    return water, later_max - vv


def features(aoi: int, pixels, series_root: str | None = None,
             start: str = START) -> pd.DataFrame:
    """The measured story of each pixel since ``start`` (see the module docstring)."""
    from . import ndvi_5day as nd

    series_root = series_root or nd.analysis_series_root(aoi)   # the AOI's pinned series
    from . import radar_water as rw

    px = np.asarray(pixels, dtype=int)
    d = nd.load(aoi, out_root=series_root)
    dt = pd.DatetimeIndex(d["dates"])
    nv = d["ndvi"].reshape(len(dt), -1)[:, px]
    lv = d["lswi"].reshape(len(dt), -1)[:, px]
    ok = d["ok"].reshape(len(dt), -1)[:, px].astype(bool)
    w = pd.DatetimeIndex(d["windows"])
    fit = d["ndvi5d"].reshape(len(w), -1)[:, px]
    nd.forget()
    newest = dt[ok.any(axis=1)].max()
    rows = []
    series = rw.read_series(aoi)
    radar = []
    for dd, flat in series:
        dd = pd.DatetimeIndex(dd)
        use = dd >= pd.Timestamp(start)
        vv, vh = flat["VV"][use][:, px], flat["VH"][use][:, px]
        water, rise = _radar_events(dd[use], vv, vh)
        radar.append((dd[use], vv, vh, water, rise))
    for j, p in enumerate(px):
        m = ok[:, j] & (dt >= pd.Timestamp(start))
        t, n, l = dt[m], nv[m, j], lv[m, j]
        r = {"pixel": int(p), "clear_views": int(m.sum())}
        if m.sum():
            ipk = int(np.nanargmax(n))
            r.update(low=round(float(np.nanmin(n)), 2), peak=round(float(n[ipk]), 2), peak_date=t[ipk].date(),
                     last=round(float(n[-1]), 2), last_date=t[-1].date(),
                     drop_from_peak=round(float(n[ipk] - n[-1]), 2))
            ow = (n < 0.15) & (l > n)
            r["optical_water"] = ",".join(f"{x:%d%b}" for x in t[ow]) or "-"
            r["water_now_optical"] = bool(ow[-1])
        # timing on the fitted curve: days from the last low before the peak to 0.6 of the way up
        f = fit[:, j]
        fw = w >= pd.Timestamp(start)
        fs, ws = f[fw], w[fw]
        ip = int(np.nanargmax(fs))
        lo = float(np.nanmin(fs[:ip + 1]))
        low_idx = np.flatnonzero(fs[:ip + 1] <= lo + 0.05)
        t0 = int(low_idx[-1]) if len(low_idx) else 0
        half = lo + 0.6 * (fs[ip] - lo)
        reach = np.flatnonzero(fs[t0:ip + 1] >= half)
        r["fit_low_end"] = ws[t0].date()
        r["days_to_60pct"] = int((ws[t0 + reach[0]] - ws[t0]).days) if len(reach) else None
        r["fit_peak_date"] = ws[ip].date()
        # radar: water passes, last water, VV rise after it, VH range
        events, vv_after, vh_rng, end_v, end_h = [], [], [], [], []
        for dd, vv, vh, water, rise in radar:
            events += [x for x, wv in zip(dd, water[:, j]) if wv]
            if water[:, j].any():
                last = np.flatnonzero(water[:, j])[-1]
                vv_after.append(float(rise[last, j]) if np.isfinite(rise[last, j]) else np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vh_rng.append(float(np.nanpercentile(vh[:, j], 90) - np.nanpercentile(vh[:, j], 10)))
                end_v.append(float(np.nanmean(vv[-2:, j])))
                end_h.append(float(np.nanmean(vh[-2:, j])))
        # radar levels per track, then the median over tracks: how low it went (second-lowest pass), how long it stayed
        # there (passes within 1.5 dB of that low), when it last was there, and how far it rose from there to the end
        lows = {"vh": [], "vv": []}
        for dd, vv, vh, water, rise in radar:
            for name, x in (("vh", vh[:, j]), ("vv", vv[:, j])):
                xs = np.where(np.isfinite(x), x, np.inf)
                if np.isfinite(xs).sum() < 3:
                    continue
                low2 = float(np.partition(xs, 1)[1])
                near = np.flatnonzero(x <= low2 + 1.5)
                lows[name].append((low2, len(near), dd[near[-1]], float(np.nanmean(x[-2:])) - low2))
        for name, v in lows.items():
            if v:
                r[f"{name}_low"] = round(float(np.median([a[0] for a in v])), 1)
                r[f"{name}_low_passes"] = int(np.median([a[1] for a in v]))
                r[f"{name}_last_low"] = max(a[2] for a in v).date()
                r[f"{name}_rise_to_end"] = round(float(np.median([a[3] for a in v])), 1)
        # wet passes: on one pass BOTH VH and VV near this track's own season lows (water lowers both, user 1 Oct);
        # the fitted NDVI on that day says whether the crop was still small (water before green-up: transplanted) or
        # already green (water given to a growing direct-seeded crop)
        wet_small, wet_green = [], []
        for dd, vv, vh, water, rise in radar:
            x, y = vh[:, j], vv[:, j]
            if np.isfinite(x).sum() < 3:
                continue
            lh = float(np.partition(np.where(np.isfinite(x), x, np.inf), 1)[1])
            lv_ = float(np.partition(np.where(np.isfinite(y), y, np.inf), 1)[1])
            with np.errstate(invalid="ignore"):
                wet = (x <= lh + WET_NEAR_LOW_DB) & (y <= lv_ + WET_NEAR_LOW_DB) & (x <= WET_VH_MAX)
            for day in dd[wet]:
                ndvi_then = float(np.interp(day.value, ws.asi8 if ws.asi8.max() > 1e17 else ws.as_unit("ns").asi8, fs))
                (wet_small if ndvi_then < SMALL_CROP_NDVI else wet_green).append(day)
        r["wet_small"] = ",".join(f"{x:%d%b}" for x in sorted(wet_small)) or "-"
        r["wet_green"] = ",".join(f"{x:%d%b}" for x in sorted(wet_green)) or "-"
        events = sorted(events)
        r["radar_water"] = ",".join(f"{x:%d%b}" for x in events) or "-"
        r["last_radar_water"] = events[-1].date() if events else None
        r["vv_rise_after_last_water"] = round(float(np.nanmax(vv_after)), 1) if vv_after and np.isfinite(
            np.nanmax(vv_after)) else None
        r["vh_range"] = round(float(np.nanmax(vh_rng)), 1)
        r["vv_end"], r["vh_end"] = round(float(np.nanmean(end_v)), 1), round(float(np.nanmean(end_h)), 1)
        r["newest"] = newest.date()
        rows.append(r)
    return pd.DataFrame(rows)


#: Draft thresholds read off the 32 labelled aoi160 pixels (1 Oct). dB levels are this AOI's; they must be checked on
#: other AOIs (look angles differ) before use. Radar first, NDVI second (user, 1 Oct).
TREE_VH_LOW = -16.5        # a tree's VH never goes this low (second-lowest pass since May)
TREE_NDVI_LOW = 0.35       # ...and its NDVI never falls below this (secondary check)
TREE_VH_RANGE = 6.0        # ...or its VH swings less than this over the season (aoi160 pixel 76937: VH low -17.9)
WET_VH_END = -17.5         # VH of the last two passes below this: water or bare soil now
YOUNG_VV_END = -11.5       # wet now, but VV this high: young plants standing in water
YOUNG_NDVI_LAST = 0.5      # ...or the newest clear view already this green (secondary)
RICE_VH_LOW = -18.5        # a rice field's radar went this low once (water / wet bare soil at sowing)
HARVEST_DROP = 0.35        # NDVI fell this much from its peak by the newest view: harvested


def classify(f: pd.DataFrame) -> pd.Series:
    """Draft hybrid rule (radar first, NDVI second) on ``features`` rows. Steps:
    1 tree/orchard: VH never low AND NDVI never low; 2 wet now (VH low on the last passes): young rice if VV is up
    (plants in water) or the newest view is green, else flooded (newest view wet) / bare (dry); 3 rice if the radar
    went low once (water or wet bare soil at sowing) or, failing that, the crop peaked late (Aug-Sep: a 3-4 month crop)
    -- else other vegetation (peak in June: a two-month crop); 4 rice harvested if NDVI fell ``HARVEST_DROP`` from its
    peak by the newest view."""
    out = []
    for r in f.itertuples(index=False):
        if r.low > TREE_NDVI_LOW and (r.vh_low > TREE_VH_LOW or r.vh_range < TREE_VH_RANGE):
            out.append("tree/orchard")
        elif r.vh_end < WET_VH_END:
            if r.vv_end > YOUNG_VV_END or r.last >= YOUNG_NDVI_LAST:
                out.append("young rice")
            else:
                out.append("flooded" if r.last > 0.15 else "bare")
        else:
            late_peak = pd.Timestamp(r.fit_peak_date).month >= 8
            rice = r.vh_low <= RICE_VH_LOW or late_peak
            if not rice:
                out.append("other vegetation")
            else:
                out.append("rice harvested" if r.drop_from_peak >= HARVEST_DROP else "rice standing")
    return pd.Series(out, index=f.index)


# ------------------------------------------------------------------------------------------- relative / derivative view
#: Radar passes before this date form each track's own dry-season level (April: bare, dry fields; trees as always).
DRY_UNTIL = "2026-05-01"


def relative_features(aoi: int, pixels, series_root: str | None = None,
                      start: str = START) -> pd.DataFrame:
    """Each pixel against ITSELF (user, 1 Oct: "rise above hard-coded numbers: rate of change, acceleration,
    derivatives, relative change"). No dB or NDVI level is compared with a fixed number:

    * radar, per track: every pass minus the track's own dry-season median (April), divided by the track's own
      pass-to-pass wobble (``radar_water.pass_noise``) -> a z-score series; tracks are then combined on a 6-day grid
      (median), so look angles drop out;
    * ``*_end_z``: where the pixel is now (last two grid steps) against its own dry season; ``*_min_z``: how far below
      it ever went (second-lowest); ``*_swing_z``: its season swing (90th - 10th percentile) in wobble units;
    * ``*_slope_end``: the change over the last ~3 weeks in wobble units (first derivative), ``*_accel_end``: the
      change of that slope (second derivative: still speeding up, or levelling off);
    * NDVI on its own scale: ``ndvi_amp`` (peak - low), ``ndvi_left`` = (last - low) / amp (1 = still at the top,
      0 = back to its low: harvested), ``ndvi_rise_days`` = days from 10 % to 90 % of its rise (fast = short crop),
      ``ndvi_slope_end`` = change over the last ~3 weeks / amp;
    * order of events: ``water_before_rise`` = days from the last joint VV + VH low (both below their own dry level
      by 2 wobbles, on the same grid step) to the start of the sustained VH rise (positive: water first, transplanted;
      negative: the crop rose first, water came later)."""
    from . import ndvi_5day as nd

    series_root = series_root or nd.analysis_series_root(aoi)   # the AOI's pinned series
    from . import radar_water as rw
    from .radar_water import pass_noise

    px = np.asarray(pixels, dtype=int)
    grid = pd.date_range(start, periods=40, freq="6D")
    z = {"VH": [], "VV": []}
    for dd, flat in rw.read_series(aoi):
        dd = pd.DatetimeIndex(dd)
        dry = dd < pd.Timestamp(DRY_UNTIL)
        for pol in z:
            x = np.asarray(flat[pol], dtype="float32")[:, px]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                base = np.nanmedian(x[dry], axis=0) if dry.sum() >= 2 else np.nanmedian(x, axis=0)
                zz = (x - base) / np.where(pass_noise(x) > 0, pass_noise(x), np.nan)
            j = np.abs(dd.values[:, None] - grid.values[None, :]).argmin(axis=0)
            near = np.abs(dd.values[j] - grid.values) <= np.timedelta64(6, "D")
            z[pol].append(np.where(near[:, None], zz[j], np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Z = {pol: np.nanmedian(np.stack(v), axis=0) for pol, v in z.items()}       # (grid, pixels)
    keep = ~np.all(np.isnan(Z["VH"]), axis=1)
    grid = grid[keep]
    Z = {k: v[keep] for k, v in Z.items()}
    d = nd.load(aoi, out_root=series_root)
    w = pd.DatetimeIndex(d["windows"])
    fit = d["ndvi5d"].reshape(len(w), -1)[:, px]
    nd.forget()
    fw = (w >= pd.Timestamp(start))
    fit, w = fit[fw], w[fw]
    rows = []
    for j, p in enumerate(px):
        r = {"pixel": int(p)}
        for pol, name in (("VH", "vh"), ("VV", "vv")):
            x = Z[pol][:, j]
            ok = np.isfinite(x)
            xs = x[ok]
            if len(xs) < 6:
                continue
            r[f"{name}_end_z"] = round(float(np.mean(xs[-2:])), 1)
            r[f"{name}_min_z"] = round(float(np.partition(xs, 1)[1]), 1)
            r[f"{name}_swing_z"] = round(float(np.percentile(xs, 90) - np.percentile(xs, 10)), 1)
            r[f"{name}_slope_end"] = round(float(xs[-1] - xs[-4]), 1)
            r[f"{name}_accel_end"] = round(float((xs[-1] - xs[-2]) - (xs[-3] - xs[-4])), 1)
        # order of events: last joint low (VV and VH both 2 wobbles below dry) and the start of the sustained VH rise
        vh, vv = Z["VH"][:, j], Z["VV"][:, j]
        with np.errstate(invalid="ignore"):
            joint = (vh <= -2) & (vv <= -2)
        sm = pd.Series(vh).rolling(3, center=True, min_periods=2).median().to_numpy()
        ip = int(np.nanargmax(sm[len(sm) // 3:])) + len(sm) // 3       # the season's VH top (after early May)
        lo = int(np.nanargmin(sm[:ip + 1])) if ip > 0 else 0          # the lowest point before it
        r["vh_rise_start"] = grid[lo].date()
        r["vh_rise_z"] = round(float(sm[ip] - sm[lo]), 1)
        jl = np.flatnonzero(joint[:ip + 1])
        r["last_joint_low"] = grid[jl[-1]].date() if len(jl) else None
        r["water_before_rise"] = int((grid[lo] - grid[jl[-1]]).days) if len(jl) else None
        # NDVI on its own scale
        f = fit[:, j]
        lo_v, ip_f = float(np.nanmin(f)), int(np.nanargmax(f))
        amp = float(f[ip_f] - np.nanmin(f[:ip_f + 1]))
        base = float(np.nanmin(f[:ip_f + 1]))
        r["ndvi_amp"] = round(amp, 2)
        r["ndvi_left"] = round(float((f[-1] - base) / amp), 2) if amp > 0 else None
        if amp > 0:
            a10 = np.flatnonzero(f[:ip_f + 1] <= base + 0.1 * amp)
            a90 = np.flatnonzero(f[:ip_f + 1] >= base + 0.9 * amp)
            r["ndvi_rise_days"] = int((w[a90[0]] - w[a10[-1]]).days) if len(a10) and len(a90) and a90[0] > a10[-1] \
                else None
            r["ndvi_slope_end"] = round(float((f[-1] - f[-5]) / amp), 2)
            r["ndvi_peak"] = w[ip_f].date()
        rows.append(r)
    return pd.DataFrame(rows)



def water_aware_fit(fit, raw, water, k: float = 2.0, noise=None) -> np.ndarray:
    """(windows, pixels) fitted NDVI that dips to every clear view showing WATER (``water``: LSWI above NDVI on that
    clear view) when the view lies more than ``k`` x the pixel's own wobble below the fit. Why (user, 1 Oct, aoi160
    pixel 69782): the upper-envelope fit treats every low view as haze; it ignored 17 Aug (NDVI 0.06, LSWI 0.36, radar
    under water on both tracks) and drew a standing crop through a flood. Between the clear views before and after
    such a view the curve is drawn straight down to it and back up (never above the original fit)."""
    from .monsoon_rule import optical_noise

    fit = np.array(fit, dtype="float32", copy=True)
    raw = np.asarray(raw, dtype="float32")
    noise = optical_noise(raw) if noise is None else np.asarray(noise)
    with np.errstate(invalid="ignore"):
        forced = np.asarray(water, dtype=bool) & np.isfinite(raw) & (fit - raw > k * np.nan_to_num(noise)[None, :])
    t = np.arange(fit.shape[0])
    for j in np.flatnonzero(forced.any(axis=0)):
        clear = np.flatnonzero(np.isfinite(raw[:, j]))
        for i in np.flatnonzero(forced[:, j]):
            prev = clear[clear < i]
            nxt = clear[clear > i]
            a = int(prev[-1]) if len(prev) else 0
            b = int(nxt[0]) if len(nxt) else fit.shape[0] - 1
            line = np.interp(t[a:b + 1], [a, i, b], [fit[a, j], raw[i, j], fit[b, j]])
            fit[a:b + 1, j] = np.fmin(fit[a:b + 1, j], line)
    return fit


#: A radar pass is "water" when VV and VH are BOTH in the bottom share of the pixel's own season range.
WATER_BOTTOM = 0.15
#: ...entered by a fall: the pass before the bottom spell stands this share of the own range higher in VH AND VV.
WATER_DROP = 0.3
#: None: the fall is measured from the last pass outside the bottom (aoi160 / aoi28). N days: from the highest pass of
#: the last N days (aoi72: a long flood's previous pass is itself water).
WATER_FALL_LOOKBACK_DAYS = None
#: Which polarisations must FALL into the bottom for a wet pass: "both" (aoi160 / aoi28 / aoi72) or "VV": VV must fall,
#: VH only has to be at the bottom (aoi116, user 2 Oct, pixels 217385 / 112734: dry May soil was already dark in VH, so
#: the flooding dropped VV by 7-8 dB but VH by ~1 dB and no water spell was found -> "direct seeded"; the user: these
#: are transplanted; a water dip is a VV dip, as for aoi160 pixel 20951).
WATER_FALL_POLS = "both"
#: Wet passes count from mid-May (April-early May dry bare soil is also at the bottom of the range).
WATER_FROM = "2026-05-15"
#: Two wet passes within this many days (any track) = standing water, not one wet day.
WATER_SPELL_DAYS = 15
#: Wet passes (while the crop is small) needed within ``WATER_SPELL_DAYS`` for a water spell: 2 (aoi160 / aoi28 / aoi72 /
#: aoi116; user 1 Oct: one wet pass is water given later to a direct-seeded crop). 1 for aoi39 (user, 3 Oct, pixel 8384:
#: one pass with VV and VH both at water level on 27 Jul, the passes around it VH low with VV high = seedlings standing
#: in water; transplanted, not direct seeded).
WATER_SPELL_MIN_PASSES = 2
#: "Crop still small": the fitted NDVI below this share of its own rise.
SMALL_SHARE = 0.5


def _nanpct(x, q: float) -> np.ndarray:
    """Per column of ``x`` (rows = time) the ``q``-th percentile of its finite values, linear interpolation as numpy.
    Why (2 Oct): ``np.nanpercentile`` loops over columns in Python; on aoi160 (136k pixels) it took 107 of 112 s."""
    xs = np.sort(np.asarray(x, dtype="float32"), axis=0)          # NaN sort to the end
    n = np.isfinite(xs).sum(axis=0)
    pos = (n - 1) * (q / 100.0)
    lo = np.clip(np.floor(pos).astype(int), 0, None)
    hi = np.clip(lo + 1, None, np.maximum(n - 1, 0))
    cols = np.arange(xs.shape[1])
    f = (pos - lo).astype("float32")
    out = xs[lo, cols] * (1 - f) + xs[hi, cols] * f
    return np.where(n > 0, out, np.nan)


def _last_k(x, k: int) -> np.ndarray:
    """(k, pixels): each column's last ``k`` finite values in time order (NaN-padded at the front)."""
    x = np.asarray(x, dtype="float32")
    fin = np.isfinite(x)
    order = np.argsort(fin, axis=0, kind="stable")               # NaNs first, finite values keep their order
    packed = np.take_along_axis(x, order, axis=0)
    return packed[-k:]


def change_since(pday, pos, since_day) -> np.ndarray:
    """Per pixel, the radar's own-range position on its last two passes minus its position on the last pass at or
    before ``since_day`` (days since 1970; NaN = no view): negative = fell since, positive = rose since. ``pos``:
    (passes, pixels) own-range positions of one track and polarisation, ``pday``: the pass days. NaN passes are skipped
    (the last finite pass up to that day counts)."""
    pos = np.asarray(pos, dtype="float32")
    pday = np.asarray(pday)
    n, m = pos.shape
    fin = np.isfinite(pos)
    idx = np.where(fin, np.arange(n)[:, None], -1)
    last_fin = np.maximum.accumulate(idx, axis=0)                     # last finite pass up to each pass
    since = np.asarray(since_day, dtype=float)
    k = np.searchsorted(pday, np.nan_to_num(since, nan=-1e18), side="right") - 1    # last pass at or before
    ref_i = np.where(k >= 0, last_fin[np.clip(k, 0, None), np.arange(m)], -1)
    ref = np.where(ref_i >= 0, pos[np.clip(ref_i, 0, None), np.arange(m)], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        end = np.nanmean(_last_k(pos, 2), axis=0)
    after = np.isfinite(since) & (ref_i >= 0) & (ref_i < n - 1)        # at least one pass after the view
    return np.where(after, end - ref, np.nan)


def sowing_not_after_seen_crop(sowing, spell_end, ok, v, lo, amp, ilo, tdays):
    """``SOWING_NOT_AFTER_SEEN_CROP``: per pixel, the radar sowing (end of the water spell, days since 1970) moved back to
    the field's lowest clear view (``ilo``) where a later clear view already shows the crop above ``AGE_LOW_SHARE`` of
    its own amplitude before that radar date. Returns (sowing, moved). ``ok`` / ``v``: (dates, pixels) clear flags and
    NDVI; ``lo`` / ``amp``: the pixel's clear-view low and amplitude; ``tdays``: the dates in days since 1970."""
    up = ok & (v >= (lo + AGE_LOW_SHARE * amp)[None, :]) & (tdays[:, None] > tdays[ilo][None, :])
    first_up = np.where(up.any(axis=0), np.min(np.where(up, tdays[:, None], 10 ** 9), axis=0), -1)
    moved = (first_up >= 0) & (sowing >= 0) & (first_up < sowing) & (spell_end >= 0)
    return np.where(moved, tdays[ilo], sowing), moved


def own_range_features(aoi: int, pixels, series_root: str | None = None,
                       start: str = START) -> pd.DataFrame:
    """Every signal on the pixel's OWN season range (0 = its own low since ``start``, 1 = its own high; 10th / 90th
    percentile so one speckle does not set the scale), per radar track on the real passes, then the median over tracks.
    Why (1 Oct): dry bare soil and water are both dark in radar, so "change from the dry season" cannot see water;
    "where is it now within its own range, and which way is it moving" can, without any dB number.

    * ``vh_pos_end`` / ``vv_pos_end``: position of the last two passes in the own range (0 at the low: water / bare
      now; 1 at the top: full canopy now);
    * ``vh_slope_end`` / ``vv_slope_end``: change over the last three passes in own-range units (first derivative);
      ``*_accel_end``: last step minus the step before (second derivative); ``*_step_end``: the last step;
    * ``vh_range_rel``: the own range divided by the own pass-to-pass wobble (a tree hardly swings);
    * ``ndvi_low_peak``: lowest / highest NDVI since ``start`` on clear views (a tree never empties: high ratio);
    * ``ndvi_left``: (last - low) / (peak - low) on clear views; ``ndvi_slope_end``: last clear view minus the one
      before, in own-amplitude units; ``ndvi_rise_days``: days from 10 % to 90 % of the rise on the fitted curve;
      ``days_since_peak``: the series' last window minus the fitted peak; ``days_at_top``: days since the fitted
      curve first reached 90 % of its rise.
    Vectorised over pixels (a whole AOI in one call)."""
    from . import ndvi_5day as nd

    series_root = series_root or nd.analysis_series_root(aoi)   # the AOI's pinned series
    from . import radar_water as rw
    from .radar_water import pass_noise

    px = np.asarray(pixels, dtype=int)
    acc = {}
    joints = []                                   # (dates, (passes, pixels) bool): VV AND VH at their own bottom
    track_pos = []                                # (pass days, VH pos, VV pos) per track: radar change since a view
    depth = []                                    # per track: deepest same-pass VV + VH fall (own wobbles)
    for dd, flat in rw.read_series(aoi):
        dd = pd.DatetimeIndex(dd)
        use = dd >= pd.Timestamp(start)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            pos, cube = [], {}
            for pol in ("VH", "VV"):
                x = np.asarray(flat[pol], dtype="float32")[use][:, px]
                lo_, hi_ = _nanpct(x, 10), _nanpct(x, 90)
                cube[pol] = (x, lo_, hi_)
                pos.append((x - lo_) / np.where(hi_ - lo_ > 0, hi_ - lo_, np.nan))
            bottom = (pos[0] <= WATER_BOTTOM) & (pos[1] <= WATER_BOTTOM)
            # water is a FALL into the bottom: the pass before the bottom spell must stand WATER_DROP higher in both
            # VH and VV. Dry bare soil sits at the bottom from the start (aoi160 pixel 20951, 18 / 21 May: the end of
            # the dry season, not water; user, 2 Oct: "no dip in VV")
            # vectorised: carry the last pass OUTSIDE the bottom forward; a bottom pass is wet when that pass stood
            # WATER_DROP higher in both polarisations (NaN passes are skipped by the forward fill)
            n_p = bottom.shape[0]
            if WATER_FALL_LOOKBACK_DAYS:
                # the fall is measured from the highest pass of the last N days: in a long flood the pass before is
                # itself water (aoi72 pixel 26433: flooded 28 May - late July, no "fall" from the previous pass)
                dday = dd[use].to_numpy().astype("datetime64[D]").astype("int64")
                ref_h = np.full_like(pos[0], np.nan)
                ref_v = np.full_like(pos[1], np.nan)
                for i in range(n_p):
                    w_ = (dday < dday[i]) & (dday >= dday[i] - WATER_FALL_LOOKBACK_DAYS)
                    if w_.any():
                        ref_h[i] = np.nanmax(pos[0][w_], axis=0)
                        ref_v[i] = np.nanmax(pos[1][w_], axis=0)
                wet = bottom & (ref_v - pos[1] >= WATER_DROP) & (
                    (ref_h - pos[0] >= WATER_DROP) | (WATER_FALL_POLS == "VV"))
            else:
                idx = np.where(~bottom & np.isfinite(pos[0]) & np.isfinite(pos[1]), np.arange(n_p)[:, None], -1)
                last_out = np.maximum.accumulate(idx, axis=0)
                cols = np.arange(bottom.shape[1])[None, :]
                ref = np.clip(last_out, 0, None)
                drop_h = pos[0][ref, cols] - pos[0]
                drop_v = pos[1][ref, cols] - pos[1]
                wet = bottom & (last_out >= 0) & (drop_v >= WATER_DROP) & (
                    (drop_h >= WATER_DROP) | (WATER_FALL_POLS == "VV"))
        late = np.asarray(dd[use] >= pd.Timestamp(WATER_FROM))[:, None]
        joints.append((dd[use], wet & late))
        track_pos.append((dd[use].to_numpy().astype("datetime64[D]").astype("int64"), pos[0], pos[1]))
        # radar start of the crop: the last pass with VH at the bottom of its own range before the crop's VH top. The
        # top is searched only AFTER the first pass at water level: a wet-soil spike before the flooding is not the
        # crop (aoi116 pixel 143310, descending track: VH -13.1 on 18 May hid the August water passes)
        with np.errstate(invalid="ignore"):
            ph = pos[0]
            n_ = np.arange(len(ph))[:, None]
            first_bot = np.where((bottom & late).any(axis=0), np.argmax(bottom & late, axis=0), 0)
            itop = np.where(late & np.isfinite(ph) & (n_ >= first_bot[None, :]), ph, -np.inf).argmax(axis=0)
            low_pass = (ph <= WATER_BOTTOM) & late & (np.arange(len(ph))[:, None] < itop[None, :])
        pday = dd[use].to_numpy().astype("datetime64[D]").astype("int64")
        ilow = len(ph) - 1 - np.argmax(low_pass[::-1], axis=0)
        acc.setdefault("_radar_rise_start", []).append(np.where(low_pass.any(axis=0), pday[ilow], np.nan))
        # days the VH has been climbing: from that last low pass to this track's newest pass (YOUNG_MIN_RISE_DAYS)
        acc.setdefault("radar_rise_days", []).append(np.where(low_pass.any(axis=0), pday[-1] - pday[ilow], np.nan))
        # the last pass with VV AND VH both at the bottom of their range before that top: still water level. In a long
        # flood no pass after the first weeks is a FALL (the passes before are water too), so the end of the water is
        # read from the level, not from the wet-pass test (aoi116 pixel 186792: water to 1 Sep, "wet" only to 5 Aug)
        with np.errstate(invalid="ignore"):
            at_bottom = bottom & late & (np.arange(len(ph))[:, None] < itop[None, :])
        ibot = len(ph) - 1 - np.argmax(at_bottom[::-1], axis=0)
        acc.setdefault("_water_level_end", []).append(np.where(at_bottom.any(axis=0), pday[ibot], np.nan))
        # water depth: the deepest same-pass fall of VV and VH below their own dry-season level, in own wobbles
        dry_all = np.asarray(dd < pd.Timestamp(DRY_UNTIL))
        if dry_all.sum() >= 2:
            fall = []
            with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                for pol in ("VV", "VH"):
                    xf = np.asarray(flat[pol], dtype="float32")[:, px]
                    nz = pass_noise(xf)
                    fall.append((np.nanmedian(xf[dry_all], axis=0) - xf) / np.where(nz > 0, nz, np.nan))
                both = np.fmin(fall[0], fall[1])[np.asarray(dd >= pd.Timestamp(WATER_FROM))]
                depth.append(np.nanmax(both, axis=0) if len(both) else np.full(len(px), np.nan))
        for pol in ("VH", "VV"):
            x, lo, hi = cube[pol]
            with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
                warnings.simplefilter("ignore", RuntimeWarning)
                rng = np.where(hi - lo > 0, hi - lo, np.nan)
                v = (_last_k(x, 4) - lo) / rng
                noise = pass_noise(x)
            n = pol.lower()
            acc.setdefault(f"{n}_pos_end", []).append(np.nanmean(v[-2:], axis=0))
            acc.setdefault(f"{n}_slope_end", []).append(v[-1] - v[-3])
            acc.setdefault(f"{n}_accel_end", []).append((v[-1] - v[-2]) - (v[-2] - v[-3]))
            acc.setdefault(f"{n}_step_end", []).append(v[-1] - v[-2])
            # rise since the recent low: the last two passes above the lowest of the last six, in own-range units
            # (plants coming up after a flood, even when the last pass is flat; user, 1 Oct, 69416)
            v6 = (_last_k(x, 6) - lo) / rng
            acc.setdefault(f"{n}_rise_recent", []).append(np.nanmean(v6[-2:], axis=0) - np.nanmin(v6, axis=0))
            # fall from the recent top: the highest of the last six passes above the last two (a cut; HARVEST_NEEDS_BOTH_POLS)
            acc.setdefault(f"{n}_fall_recent", []).append(np.nanmax(v6, axis=0) - np.nanmean(v6[-2:], axis=0))
            acc.setdefault(f"{n}_range_rel", []).append(rng / np.where(noise > 0, noise, np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = {k: np.nanmedian(np.stack(v), axis=0) for k, v in acc.items()}
        out["water_depth"] = np.nanmax(np.stack(depth), axis=0) if depth else np.full(len(px), np.nan)
        track_ends = np.stack(acc["_water_level_end"]) if "_water_level_end" in acc else None
        if WATER_END_TRACKS in ("earliest", "earliest_unless_wet_view") and track_ends is not None:
            out["_water_level_end"] = np.nanmin(track_ends, axis=0)
    d = nd.load(aoi, out_root=series_root)
    dt = pd.DatetimeIndex(d["dates"])
    nv = d["ndvi"].reshape(len(dt), -1)[:, px].astype("float32")
    ok = d["ok"].reshape(len(dt), -1)[:, px].astype(bool) & (dt >= pd.Timestamp(start))[:, None]
    w = pd.DatetimeIndex(d["windows"])
    sel = w >= pd.Timestamp(start)
    fit_all = d["ndvi5d"].reshape(len(w), -1)[:, px].astype("float32")
    raw_all = d["ndvi5d_raw"].reshape(len(w), -1)[:, px].astype("float32")
    # water views per window: a clear date in the window whose LSWI is above its NDVI
    lv = d["lswi"].reshape(len(dt), -1)[:, px].astype("float32")
    wet_date = ok & (lv > nv)
    k_ = np.clip(np.searchsorted(w.values, dt.values, side="right") - 1, 0, len(w) - 1)
    wet_win = np.zeros(fit_all.shape, dtype=bool)
    np.logical_or.at(wet_win, k_, wet_date)
    fit = water_aware_fit(fit_all, raw_all, wet_win)[sel]
    gaps = d["gapdays5d"].reshape(len(w), -1)[:, px][sel]     # days from each window to the nearest clear view
    w = w[sel]
    newest = dt[ok.any(axis=1)].max()
    nd.forget()
    v = np.where(ok, nv, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, pk = np.nanmin(v, axis=0), np.nanmax(v, axis=0)
        amp = pk - lo
        last2 = _last_k(v, 2)
        out["ndvi_low_peak"] = lo / pk
        # when the clear-view low came (days since 1970): a tree / orchard dips in the dry season (April-May), a
        # monsoon crop's low is its sowing in June-August (aoi28 pixel 7826: low 0.39 on 23 Jun, then 0.85 in Sep)
        tdays = dt.to_numpy().astype("datetime64[D]").astype("int64")
        ilo = np.nanargmin(np.where(np.isfinite(v), v, np.inf), axis=0)
        out["ndvi_low_day"] = np.where(np.isfinite(lo), tdays[ilo], np.nan).astype(float)
        # days from that clear-view low to the pixel's newest clear view: how long ago the optical SAW the field empty
        # (CANOPY_OVER_RADAR_WATER; aoi63 pixel 13571: empty in early May, 0.81 on 28 Sep = 147 days)
        inew_ = len(dt) - 1 - np.argmax(ok[::-1], axis=0)
        out["days_since_ndvi_low"] = np.where(np.isfinite(lo) & ok.any(axis=0), tdays[inew_] - tdays[ilo],
                                              np.nan).astype(float)
        # the pixel is clear on the AOI's newest image date (NEWEST_EMPTY_NOT_RICE: only the latest image decides)
        aoi_ok = d["ok"].reshape(len(dt), -1).astype(bool) & (dt >= pd.Timestamp(start))[:, None]
        latest = np.flatnonzero(aoi_ok.any(axis=1)).max() if aoi_ok.any() else -1
        out["seen_on_latest"] = (ok[latest] if latest >= 0 else np.zeros(len(px), dtype=bool)).astype(float)
        # the pixel's newest clear view (walking back past cloudy dates) and how VV / VH moved on the passes after it
        # (TONE_AND_CURVE: the newer news wins; user, 5 Oct)
        view_day = np.where(ok.any(axis=0), tdays[inew_], np.nan).astype(float)
        out["view_day"] = view_day
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for j, name in ((1, "vh"), (2, "vv")):
                ch = [change_since(t[0], t[j], view_day) for t in track_pos]
                out[f"{name}_since_view"] = np.nanmedian(np.stack(ch), axis=0) if ch else np.full(len(px), np.nan)
        out["ndvi_left"] = (last2[-1] - lo) / amp
        # the same on the whole year's clear views (from DRY_SEASON_FROM): "green now" against the pixel's own bare
        # level, not against a summer crop's dip (RADAR_STORY_IN_GAP; aoi63 pixel 31277: summer crop 0.94 in May,
        # 0.73 in June, 0.81 on 28 Sep = "38 % left" since May, 68 % on the year)
        okY = d["ok"].reshape(len(dt), -1)[:, px].astype(bool) & (dt >= pd.Timestamp(DRY_SEASON_FROM))[:, None]
        vY = np.where(okY, nv, np.nan)
        loY, pkY = np.nanmin(vY, axis=0), np.nanmax(vY, axis=0)
        out["ndvi_left_year"] = (last2[-1] - loY) / (pkY - loY)
        # the newest clear view shows water (NDVI below 0, or LSWI above NDVI): the optical's last word is "flooded"
        lv_ = d["lswi"].reshape(len(dt), -1)[:, px].astype("float32")
        lw = _last_k(np.where(ok, lv_, np.nan), 1)[-1]
        out["last_view_water"] = ((last2[-1] < 0) | (lw > last2[-1])).astype(float)
        # share of the dry-season clear views that show water (permanent water / ponds; see POND_IF_DRY_WATER)
        dry_ = ((dt >= pd.Timestamp(DRY_SEASON_FROM)) & (dt < pd.Timestamp(DRY_UNTIL)))[:, None]
        ok_dry = d["ok"].reshape(len(dt), -1)[:, px].astype(bool) & dry_
        wet_dry = ok_dry & ((nv < 0) | (lv_ > nv))
        out["dry_water_share"] = np.where(ok_dry.any(axis=0), wet_dry.sum(axis=0) / np.maximum(ok_dry.sum(axis=0), 1),
                                          np.nan)
        out["ndvi_slope_end"] = (last2[-1] - last2[-2]) / amp
        # green-up seen on clear views: from the last clear view still near the season low (<= 10 % of the rise) to
        # the first clear view at >= 90 %, before the peak view. An UPPER bound of the green-up time: when it is short
        # the crop surely greened fast, whatever the filled curve says between sparse views (user, 1 Oct: 15317 and
        # 16427, the same curve, 35 vs 30 days on the fit)
        tt = dt.to_numpy().astype("datetime64[D]").astype("int64")[:, None]
        ipk = np.nanargmax(np.where(np.isfinite(v), v, -np.inf), axis=0)
        before_pk = np.arange(len(dt))[:, None] <= ipk[None, :]
        lo10 = before_pk & (v <= (lo + 0.1 * amp)[None, :])
        hi90 = before_pk & (v >= (lo + 0.9 * amp)[None, :])
        t_lo = np.where(lo10.any(axis=0), np.max(np.where(lo10, tt, -10 ** 9), axis=0), np.nan)
        t_hi = np.where(hi90.any(axis=0), np.min(np.where(hi90 & (tt >= np.nan_to_num(t_lo, nan=-1e9)[None, :]),
                                                          tt, 10 ** 9), axis=0), np.nan)
        out["ndvi_rise_seen"] = np.where(np.isfinite(t_lo) & np.isfinite(t_hi) & (t_hi < 10 ** 9), t_hi - t_lo,
                                         np.nan)
        fz = np.where(np.isfinite(fit), fit, -np.inf)
        ip = fz.argmax(axis=0)
        t = np.arange(len(w))[:, None]
        before = t <= ip[None, :]
        base = np.nanmin(np.where(before, fit, np.nan), axis=0)
        famp = fit[ip, np.arange(len(px))] - base
        low10 = before & (fit <= (base + 0.1 * famp)[None, :])
        hi90 = before & (fit >= (base + 0.9 * famp)[None, :])
        i10 = np.where(low10.any(axis=0), len(w) - 1 - np.argmax(low10[::-1], axis=0), -1)
        i90 = np.where(hi90.any(axis=0), np.argmax(hi90, axis=0), -1)
        good = (i10 >= 0) & (i90 > i10)
        wd = w.to_numpy().astype("datetime64[D]").astype("int64")
        out["ndvi_rise_days"] = np.where(good, wd[np.clip(i90, 0, None)] - wd[np.clip(i10, 0, None)], np.nan)
        # no clear view INSIDE the fitted green-up (strictly between its 10 % and 90 % windows; the views at the two
        # ends only pin the line): its timing is the fit's line, not seen (RADAR_STORY_IN_GAP; aoi63: no clear view
        # 3 Jul - 26 Sep, the 26 Sep view itself is the 90 % end)
        seen_c = np.vstack([np.zeros((1, len(px)), dtype=int), np.cumsum(np.isfinite(raw_all[sel]), axis=0)])
        cols_ = np.arange(len(px))
        n_seen = seen_c[np.clip(i90, 0, None), cols_] - seen_c[np.clip(i10, 0, None) + 1, cols_]
        out["rise_unseen"] = (good & (n_seen == 0)).astype(float)
        # from the fitted peak to the series' own last window (not the AOI's newest clear date: one pixel seen on
        # 28 Sep made every young crop look "past its peak" on a series ending 26 Sep)
        out["days_since_peak"] = (wd[-1] - wd[ip]).astype(float)
        # days the curve has stayed at its top (first reached 90 % of its rise -> last window): a flat top of a grown
        # crop counts as "past the peak" even when its last window is a hair higher (aoi160 pixel 74380, 0.77-0.79
        # from late July to 26 Sep)
        out["days_at_top"] = np.where(i90 >= 0, wd[-1] - wd[np.clip(i90, 0, None)], np.nan).astype(float)
        # days since the crop was last at its top (>= 90 % of its own rise on the fitted curve). Rice ripens and is cut
        # within weeks of its top; a crop that peaked in June and has only declined since (15689, 15317, 16427,
        # 27089, 50760: 85-110 days) is not the monsoon rice, while 108253 (top on 6 Sep, cut by 26 Sep) is
        top = fit >= (base + 0.9 * famp)[None, :]
        last_top = len(w) - 1 - np.argmax(top[::-1], axis=0)
        out["days_off_top"] = np.where(top.any(axis=0), wd[-1] - wd[last_top], np.nan).astype(float)
        # age of the crop standing now: days since the water-aware fitted curve was last near its own low (within 20 %
        # of its rise) before the peak. The user's "young rice" is rice at most YOUNG_MAX_DAYS old on the newest image
        # (63530: under water to late Aug, 0.62 on 26 Sep, radar already up: ~35 days, young, not standing rice)
        low20 = before & (fit <= (base + AGE_LOW_SHARE * famp)[None, :])
        i20 = np.where(low20.any(axis=0), len(w) - 1 - np.argmax(low20[::-1], axis=0), -1)
        out["crop_age"] = np.where(i20 >= 0, wd[-1] - wd[np.clip(i20, 0, None)], np.nan).astype(float)
        # standing water while the crop was still small (fitted NDVI below half of its own rise), on at least two
        # passes within WATER_SPELL_DAYS (any track): transplanting into a flooded field. One wet pass, or water under
        # a grown crop, is a direct-seeded field that got water later (user, 1 Oct: 78199 vs 82127 / 125633)
        frac = (fit - base[None, :]) / np.where(famp > 0, famp, np.nan)[None, :]
        wdays = w.to_numpy().astype("datetime64[D]").astype("int64")
        all_d, all_m = [], []
        for dd_, jm in joints:
            dday = dd_.to_numpy().astype("datetime64[D]").astype("int64")
            k = np.abs(dday[:, None] - wdays[None, :]).argmin(axis=1)
            small = frac[k] < SMALL_SHARE
            if WATER_SMALL_UNSEEN:
                small = small | (gaps[k] > nd.STEP_DAYS)      # no clear view near the pass: the radar decides
            all_d.append(dday)
            all_m.append(jm & small)
        D = np.concatenate(all_d)
        M = np.concatenate(all_m).astype(int)
        order = np.argsort(D, kind="stable")
        D, M = D[order], M[order]
        cs = np.vstack([np.zeros((1, M.shape[1]), dtype=int), np.cumsum(M, axis=0)])
        hi_i = np.searchsorted(D, D + WATER_SPELL_DAYS, side="right")
        counts = cs[hi_i] - cs[np.arange(len(D))]
        qual = (counts >= WATER_SPELL_MIN_PASSES) & (M > 0)
        out["water_spell"] = qual.any(axis=0).astype(float)
        # the rule's own sowing date (days since 1970): a transplanted crop was planted when the water spell began; any
        # other crop when its empty spell ended (user, 2 Oct, aoi28 pixel 4227: the notebook showed step 2's June
        # trough, the radar puts the flooding at 22-27 July)
        first_wet = np.where(qual.any(axis=0), D[np.argmax(qual, axis=0)], -1)
        low_end = np.where(i20 >= 0, wd[np.clip(i20, 0, None)], -1)
        out["sowing_day"] = np.where(first_wet >= 0, first_wet, low_end).astype(float)
        if AGE_FROM_WATER:
            # a transplanted crop's age counts from its water spell: seedlings in standing water stay hidden from the
            # optical for weeks, so the end of the low NDVI spell comes far too late (aoi72 pixel 27550: water from
            # 12 Jul, NDVI low until 12 Aug, "45 days" -> 76 days)
            out["crop_age"] = np.where(first_wet >= 0, wd[-1] - first_wet, out["crop_age"]).astype(float)
        # where the date came from: 0 NDVI (end of the empty spell), 1 radar (end of the water spell), 2 radar (VH
        # leaves the bottom of its range), 3 radar (start of the water spell, the pre-SOWING_FROM_RADAR date)
        out["sowing_from"] = np.where(first_wet >= 0, 3.0, 0.0)
        rise = out.pop("_radar_rise_start")
        level_end = out.pop("_water_level_end")
        # no clear view in the first half of the crop's life (its start = the water spell, else the radar's last low
        # before its top; its end = the newest clear view): the optical never saw this crop grow, only grown
        # (RADAR_STORY_IN_GAP; aoi63 pixel 31277: the season's NDVI rise was a summer crop seen in April, the monsoon
        # crop grew 3 Jul - 26 Sep without a clear view; the 26 / 28 Sep views only show it grown)
        start_c = np.where(first_wet >= 0, first_wet, np.nan_to_num(rise, nan=-1))
        tday_all = dt.to_numpy().astype("datetime64[D]").astype("int64")[:, None]
        newest_c = np.max(np.where(ok, tday_all, -1), axis=0)
        mid_c = (start_c + newest_c) / 2
        between = ok & (tday_all > start_c[None, :]) & (tday_all <= mid_c[None, :])
        out["crop_unseen"] = ((start_c >= 0) & ~between.any(axis=0)).astype(float)
        # the optical during the water spell (first wet pass -> last pass at water level): how many clear views, and
        # how many of them show water (RADAR_DECIDES_WITHOUT_OPTICAL)
        last_wet_any = np.where((M > 0).any(axis=0), D[len(D) - 1 - np.argmax((M > 0)[::-1], axis=0)], -1)
        w_end = np.where(qual.any(axis=0), np.fmax(np.nan_to_num(level_end, nan=-1), last_wet_any), -1)
        tday = dt.to_numpy().astype("datetime64[D]").astype("int64")[:, None]
        in_spell = ok & (tday >= first_wet[None, :]) & (tday <= w_end[None, :]) & (first_wet >= 0)[None, :]
        out["views_in_water"] = np.where(first_wet >= 0, in_spell.sum(axis=0), np.nan).astype(float)
        out["water_views_in_water"] = np.where(first_wet >= 0, (in_spell & ((nv < 0) | (lv_ > nv))).sum(axis=0),
                                               np.nan).astype(float)
        if SOWING_FROM_RADAR:
            # end of the water spell: the later of the last wet pass and the last pass still at water level (median
            # over tracks) before the radar rises; only for a pixel with a water spell
            last_wet = np.where((M > 0).any(axis=0), D[len(D) - 1 - np.argmax((M > 0)[::-1], axis=0)], -1)
            if WATER_END_TRACKS == "earliest_unless_wet_view" and track_ends is not None:
                # the newest clear view showing water (LSWI above NDVI) since WATER_FROM; later than the earliest
                # track's end -> the water had not ended there: take the latest track's end
                wv = wet_date & (dt >= pd.Timestamp(WATER_FROM))[:, None]
                last_view_wet = np.where(wv.any(axis=0), np.max(np.where(wv, tdays[:, None], -1), axis=0), -1)
                # ...or the newest clear view is behind nearly all of the AOI's fields that day (AOI-relative)
                full_ok = d["ok"].reshape(len(dt), -1).astype(bool) & (dt >= pd.Timestamp(start))[:, None]
                inside = nd.inside_from_loc(d["loc"])
                full_v = d["ndvi"].reshape(len(dt), -1)
                inew = len(dt) - 1 - np.argmax(ok[::-1], axis=0)              # each pixel's newest clear date
                ref = np.full(len(px), np.nan, dtype="float32")
                for j in np.unique(inew[ok.any(axis=0)]):
                    sel_ = full_ok[j] & inside
                    if sel_.sum() >= BEHIND_MIN_CLEAR * inside.sum():
                        ref[inew == j] = np.percentile(full_v[j][sel_], 100 * BEHIND_SHARE)
                with np.errstate(invalid="ignore"):
                    behind = np.where(ok.any(axis=0), nv[inew, np.arange(len(px))], np.nan) < ref
                later = (last_view_wet > np.nan_to_num(level_end, nan=-1)) | behind
                level_end = np.where(later, np.nanmax(track_ends, axis=0), level_end)
            spell_end = np.where(qual.any(axis=0), np.fmax(np.nan_to_num(level_end, nan=-1), last_wet), -1)
            seen = (i20 >= 0) & (gaps[np.clip(i20, 0, None), np.arange(len(px))] <= nd.STEP_DAYS)
            radar_rise = np.isfinite(rise) & ~seen
            sowing = np.where(spell_end >= 0, spell_end, np.where(radar_rise, np.nan_to_num(rise, nan=-1), low_end))
            out["sowing_from"] = np.where(spell_end >= 0, 1.0, np.where(radar_rise, 2.0, 0.0))
            if SOWING_NOT_AFTER_SEEN_CROP:
                # a clear view after the field's lowest view already shows the crop rising (above AGE_LOW_SHARE of its
                # own amplitude) before the radar's water end: the crop was planted by then -> sowing = that lowest view
                # (aoi13 pixel 14994: water view 18 Jul, NDVI 0.33 on 17 Aug, a late-August radar dip made it 29 Aug)
                sowing, early = sowing_not_after_seen_crop(sowing, spell_end, ok, v, lo, amp, ilo, tdays)
                out["sowing_from"] = np.where(early, 4.0, out["sowing_from"])
            out["sowing_day"] = sowing.astype(float)
            out["crop_age"] = np.where(sowing >= 0, wd[-1] - sowing, out["crop_age"]).astype(float)
    out.pop("_radar_rise_start", None)
    out.pop("_water_level_end", None)
    frame = pd.DataFrame({k: np.round(np.asarray(v_, dtype=float), 2) for k, v_ in out.items()})
    frame.insert(0, "pixel", px)
    return frame


#: Own-scale cut points of the relative rule (fractions of the pixel's own range / amplitude, or days), read off the 31
#: labelled aoi160 pixels (1 Oct). No dB and no NDVI level appears.
TREE_LOW_PEAK = 0.45     # NDVI never fell below ~half of its own peak: never emptied (tree / orchard)
LOW_NOW = 0.6            # radar VH now in the lower part of its own season range: water / bare / young plants
RISING = 0.2             # VH climbed this share of its own range over the last three passes: plants coming up
VV_JUMP = 0.3            # ...or VV rose this share of its own range on the last pass (stems standing in water)
RISE_BOTH = 0.5          # ...or VV AND VH both rose half of their own range since their recent low (69416: risen
#                          after the August flood, flat on the last pass; flooded 43298: VH only 0.42)
FAST_RISE_DAYS = 35      # 10 % -> 90 % of the NDVI rise faster than this: a two-month crop, not rice (user)
HARVEST_LEFT = 0.55      # less than this share of its own NDVI amplitude left on the newest view: harvested
RIPEN_DAYS_MAX = 60      # ...and only if it left its top (90 % of its own rise) more than this long ago: rice ripens
#                          and is cut within weeks of its top (108253: top 6 Sep, cut by 26 Sep = rice harvested; the
#                          June-peaking crops 15689 / 15317 / 16427 / 27089 / 50760 left their top 85-110 days ago)
HELD_LEFT = 0.85         # a fast green-up is "other vegetation" only if the crop also ended soon; still holding this
#                          share of its rise on the newest view = a long crop (user, 1 Oct, 48135: greened in 25 days,
#                          stood at 0.72-0.89 for four months: direct-seeded rice)
HELD_DAYS = 15           # a crop that greened over weeks (not faster than FAST_RISE_DAYS) and has stood at its top
#                          for at least three windows is a grown / ripening canopy even when the radar fell to its own
#                          low: dense and ripening rice lower VV and VH (user, 1 Oct, 93578 / 74380 / 110873). A young
#                          crop reaches its top only now; a flood's short green bump rose in under five weeks.


def _crop_by_curve(r) -> str:
    """A crop's class from its curve: young by age (``YOUNG_MAX_DAYS``), transplanted with a water spell."""
    seen = getattr(r, "days_since_ndvi_low", np.nan)
    age = seen if pd.notna(seen) else getattr(r, "crop_age", np.nan)
    if pd.notna(age) and age <= YOUNG_MAX_DAYS:
        return "young rice"
    return "rice standing transplanted" if getattr(r, "water_spell", 0) == 1 else "rice standing direct seeded"


def tone_and_curve(f: pd.DataFrame, out: list) -> list:
    """``TONE_AND_CURVE`` and ``HARVEST_NEEDS_BOTH_POLS`` on the classes ``out`` of the feature rows ``f``."""
    rice = ("rice standing transplanted", "rice standing direct seeded", "young rice")
    res = []
    for c, r in zip(out, f.itertuples(index=False)):
        vh_s, vv_s = getattr(r, "vh_since_view", np.nan), getattr(r, "vv_since_view", np.nan)
        both_fell = pd.notna(vh_s) and pd.notna(vv_s) and vh_s <= -RISE_BOTH and vv_s <= -RISE_BOTH
        both_rose = pd.notna(vh_s) and pd.notna(vv_s) and vh_s >= RISE_BOTH and vv_s >= RISE_BOTH
        cut = getattr(r, "vh_fall_recent", 0) >= RISE_BOTH and getattr(r, "vv_fall_recent", 0) >= RISE_BOTH
        if HARVEST_NEEDS_BOTH_POLS and c == "rice harvested" and not cut and not both_fell:
            # one polarisation still up: the crop stands
            c = "rice standing transplanted" if getattr(r, "water_spell", 0) == 1 else "rice standing direct seeded"
        if TONE_AND_CURVE and pd.notna(getattr(r, "view_day", np.nan)):
            left = getattr(r, "ndvi_left_year", np.nan)
            t = "water" if getattr(r, "last_view_water", 0) == 1 else ("green" if pd.notna(left) and left >= 0.5 else
                                                                         "empty")
            if t == "green":
                if both_fell:
                    c = "rice harvested"
                elif c in ("flooded / bare", "rice harvested"):
                    c = _crop_by_curve(r)
            elif t == "empty" and c in rice:
                c = c if both_rose else ("rice harvested" if cut else "flooded / bare")
            elif t == "water" and c in ("rice standing transplanted", "rice standing direct seeded"):
                c = "young rice" if both_rose else "flooded / bare"
        res.append(c)
    return res


def classify_relative(f: pd.DataFrame) -> pd.Series:
    """Relative rule, radar first (user, 1 Oct). On ``own_range_features`` rows:
    1 tree/orchard: the NDVI never fell below about half of its own peak;
    2 VH now low in its own range (unless the NDVI shows a grown canopy: greened over weeks and at its top for
      ``HELD_DAYS``; ripening rice lowers the radar, aoi160 pixels 93578 / 74380 / 110873): young rice if NDVI peaks on the newest image and still climbs AND VH climbs
      (``RISING``), else flooded / bare;
    3 otherwise: young rice if the crop began to rise within ``YOUNG_MAX_DAYS`` and is still green (``crop_age``);
      else a grown crop: other vegetation if its green-up (10 -> 90 %) took less than ``FAST_RISE_DAYS`` on the
      fitted curve or at most that long between clear views (``ndvi_rise_seen``) AND it did not stay up (less than
      ``HELD_LEFT`` of its rise left on the newest view) AND it left its top more than ``RIPEN_DAYS_MAX`` ago, else
      rice, harvested if less than ``HARVEST_LEFT`` of its own NDVI amplitude is left; standing rice is transplanted
      when water stood in the field while the crop was small (``water_spell``), else direct seeded (user, 1 Oct)."""
    out = []
    for r in f.itertuples(index=False):
        # a real crop that greened over weeks and has stood at its top for weeks: a grown / ripening canopy, whatever
        # the radar does now (a dense or ripening rice canopy lowers VV and VH; user, 1 Oct, 93578 / 74380 / 110873)
        grown = (pd.notna(r.ndvi_rise_days) and r.ndvi_rise_days >= FAST_RISE_DAYS and
                 pd.notna(r.days_at_top) and r.days_at_top >= HELD_DAYS and
                 not (GROWN_NOT_IF_WATER_VIEW and getattr(r, "last_view_water", 0) == 1))
        low_late = (TREE_LOW_BEFORE is not None and pd.notna(getattr(r, "ndvi_low_day", np.nan)) and
                    r.ndvi_low_day >= (pd.Timestamp(TREE_LOW_BEFORE) - pd.Timestamp("1970-01-01")).days)
        radar_rose = (getattr(r, "vh_rise_recent", 0) >= RISE_BOTH and getattr(r, "vv_rise_recent", 0) >= RISE_BOTH) \
            or getattr(r, "vv_step_end", 0) >= VV_JUMP
        radar_water = TREE_NEEDS_NO_WATER and getattr(r, "water_spell", 0) == 1
        second_crop = (SECOND_CROP_BY_RADAR and getattr(r, "water_spell", 0) == 1 and
                       getattr(r, "water_depth", 0) >= WATER_DEPTH_K and pd.notna(getattr(r, "days_off_top", np.nan))
                       and pd.notna(getattr(r, "crop_age", np.nan)) and r.crop_age < r.days_off_top
                       and r.vh_pos_end >= LOW_NOW)
        radar_only = (RADAR_DECIDES_WITHOUT_OPTICAL and getattr(r, "water_spell", 0) == 1 and
                      getattr(r, "water_depth", 0) >= WATER_DEPTH_K and
                      getattr(r, "water_views_in_water", 1) == 0 and getattr(r, "views_in_water", 99) < OPTICAL_VIEWS_MIN)
        green_now = POND_NEEDS_NO_CROP and getattr(r, "ndvi_left_year", np.nan) >= HARVEST_LEFT
        if POND_IF_DRY_WATER and getattr(r, "dry_water_share", 0) >= 0.5 and not green_now:
            # most dry-season clear views show water: a pond / permanent water, not a field
            out.append("flooded / bare")
        elif CANOPY_OVER_RADAR_WATER and r.vh_pos_end < LOW_NOW and getattr(r, "ndvi_left_year", np.nan) >= 1.0 and \
                getattr(r, "days_since_ndvi_low", np.nan) > YOUNG_MAX_DAYS:
            # radar low now, the newest clear view is the year's greenest and the field was seen empty long before: a
            # grown canopy lowers the radar; its age is from the empty field, not from the radar's late "water"
            # (aoi63 pixels 13571: read as deep water -> flooded; 18853: read as a late water spell -> young, 37 days).
            # Transplanted only when the water came while the crop was small (the spell began more than YOUNG_MAX_DAYS
            # ago: 20447, late June) or was deep (WATER_DEPTH_K: 13571); a shallow "spell" found only under the late
            # canopy is the canopy itself -> direct seeded (user, 5 Oct: 18853 / 24865, no June VV dip)
            early = getattr(r, "crop_age", 0) > YOUNG_MAX_DAYS
            deep = getattr(r, "water_depth", 0) >= WATER_DEPTH_K
            out.append("rice standing transplanted" if getattr(r, "water_spell", 0) == 1 and (early or deep) else
                       "rice standing direct seeded")
        elif radar_only:
            # the optical never saw this water spell: the radar alone decides
            # still low: a crop only if VV AND VH both rose half their range (radar_rose); the last passes' small wobble
            # on standing water is not plants (aoi39 pixel 21635: under water to 25 Sep, 13 Sep view NDVI -1.0)
            if r.vh_pos_end < LOW_NOW and not radar_rose:
                out.append("flooded / bare")
            elif getattr(r, "crop_age", 999) <= YOUNG_MAX_DAYS:
                # young or standing by the AOI's own crop age (aoi39: from the start of the water). Counting from the END
                # of the water was tried (pixel 10969) and rejected by the user (3 Oct): it doubled young rice (24 -> 56 ac)
                out.append("young rice")
            else:
                out.append("rice standing transplanted")
        elif RADAR_STORY_IN_GAP and (getattr(r, "rise_unseen", 0) == 1 or getattr(r, "crop_unseen", 0) == 1):
            # the optical never saw the green-up: radar story, NDVI only from its clear views (aoi63); green / bare now
            # on the whole year's range (a summer crop's dip is not the bare level: 31277)
            left = getattr(r, "ndvi_left_year", r.ndvi_left)
            paddy = getattr(r, "water_spell", 0) == 1 or getattr(r, "water_depth", 0) >= WATER_DEPTH_K
            held_water = paddy or getattr(r, "water_depth", 0) > 0
            if r.ndvi_low_peak >= TREE_LOW_PEAK and not low_late and not held_water:
                out.append("tree/orchard")
            elif STORY_YOUNG_AFTER_LAST_WATER and getattr(r, "last_view_water", 0) == 1 and radar_rose:
                # the last clear view was water and VV and VH both rose since: plants in the water the optical has not
                # seen yet (aoi13 pixel 11791: last view 19 Aug water, VV -18.8 -> -6.3 by 22 Sep); the young-rice
                # checks after the loop (radar rise days, a clear dip) then decide
                out.append("young rice")
            elif left < HARVEST_LEFT:
                water_now = getattr(r, "last_view_water", 0) == 1 or (r.vh_pos_end < LOW_NOW and not radar_rose)
                out.append("rice harvested" if held_water and not water_now else "flooded / bare")
            else:
                age = getattr(r, "crop_age", np.nan)
                if getattr(r, "water_spell", 0) != 1 and pd.notna(getattr(r, "radar_rise_days", np.nan)):
                    age = r.radar_rise_days          # no water spell: from the radar's last low before its top
                if pd.notna(age) and age <= YOUNG_MAX_DAYS:
                    out.append("young rice")
                else:
                    out.append("rice standing transplanted" if paddy else "rice standing direct seeded")
        elif second_crop:
            # deep water after the NDVI had left its top: a new transplanted crop; its class comes from its age
            out.append("young rice" if r.crop_age <= YOUNG_MAX_DAYS else "rice standing transplanted")
        elif r.ndvi_low_peak >= TREE_LOW_PEAK and not low_late and not radar_water:
            out.append("tree/orchard")
        elif YOUNG_AFTER_LAST_WATER and getattr(r, "last_view_water", 0) == 1 and radar_rose:
            # the last clear view was water and the radar rose since: plants in the water the optical has not seen
            out.append("young rice")
        elif YOUNG_WHILE_RADAR_LOW and r.vh_pos_end < LOW_NOW and r.ndvi_left >= 1.0 and r.ndvi_slope_end > 0 and (
                r.vh_slope_end >= RISING or getattr(r, "vv_step_end", 0) >= VV_JUMP or
                (radar_rose and not YOUNG_RADAR_LOW_NEEDS_RISE)):
            # the radar is still in the lower part of its range and climbing while the newest clear view is the
            # season's greenest: the canopy is still closing (young), not a grown crop held at its top
            out.append("young rice")
        elif r.vh_pos_end < LOW_NOW and not grown and not (
                YOUNG_NEEDS_AGE and getattr(r, "crop_age", 0) > YOUNG_MAX_DAYS and r.ndvi_left >= HELD_LEFT):
            # plants coming up in the water: VH climbed over the last passes, or VV jumped on the last pass (stems
            # standing in water: double bounce; user, 1 Oct: "neither VV nor VH rose" = flooded, 43298; 69782 VV +0.9)
            plants = (r.vh_slope_end >= RISING or getattr(r, "vv_step_end", 0) >= VV_JUMP or
                      (getattr(r, "vh_rise_recent", 0) >= RISE_BOTH and getattr(r, "vv_rise_recent", 0) >= RISE_BOTH))
            young = r.days_since_peak == 0 and r.ndvi_slope_end > 0 and plants
            canopy = (CANOPY_NOT_FLOODED and r.ndvi_left >= 1.0 and pd.notna(r.ndvi_rise_days) and
                      r.ndvi_rise_days >= FAST_RISE_DAYS)
            if canopy and not young:
                # the greenest view of the season is the newest one: a canopy closing, not water (aoi63 pixel 20447)
                out.append("young rice" if getattr(r, "crop_age", 999) <= YOUNG_MAX_DAYS else
                           "rice standing transplanted" if getattr(r, "water_spell", 0) == 1 else
                           "rice standing direct seeded")
            else:
                out.append("young rice" if young else "flooded / bare")
        elif ((pd.notna(r.ndvi_rise_days) and r.ndvi_rise_days < FAST_RISE_DAYS) or
              (pd.notna(getattr(r, "ndvi_rise_seen", np.nan)) and r.ndvi_rise_seen <= FAST_RISE_DAYS)) and \
                not r.ndvi_left >= HELD_LEFT and not getattr(r, "days_off_top", 999) <= RIPEN_DAYS_MAX:
            out.append("other vegetation")
        elif getattr(r, "crop_age", 999) <= YOUNG_MAX_DAYS and r.ndvi_left >= HELD_LEFT:
            # a crop that began to rise within the last YOUNG_MAX_DAYS and is still green: young rice, even when the
            # radar has already climbed out of the low part of its range (fast-growing transplants, 63530)
            out.append("young rice")
        else:
            rising_top = (HARVEST_NOT_IF_RADAR_RISING and radar_rose and r.vh_pos_end >= LOW_NOW)
            if r.ndvi_left < HARVEST_LEFT and rising_top:
                deep = getattr(r, "water_spell", 0) == 1 or getattr(r, "water_depth", 0) >= WATER_DEPTH_K
                out.append("rice standing transplanted" if deep else "rice standing direct seeded")
            elif r.ndvi_left < HARVEST_LEFT:
                out.append("rice harvested")
            else:
                out.append("rice standing transplanted" if getattr(r, "water_spell", 0) == 1 else
                           "rice standing direct seeded")
    if NEWEST_EMPTY_NOT_RICE and "seen_on_latest" in f:
        # applied before every other late check: the latest image's word on an empty field is final
        left = pd.to_numeric(f.get("ndvi_left_year", pd.Series(np.nan, index=f.index)), errors="coerce").to_numpy()
        seen = pd.to_numeric(f["seen_on_latest"], errors="coerce").to_numpy() == 1
        out = ["flooded / bare" if sn and np.isfinite(lf) and lf <= AGE_LOW_SHARE else c
               for c, lf, sn in zip(out, left, seen)]
    if (TONE_AND_CURVE or HARVEST_NEEDS_BOTH_POLS) and "ndvi_left_year" in f:
        out = tone_and_curve(f, out)
    if GREEN_VIEW_NOT_FLOODED and "ndvi_left_year" in f:
        fixed = []
        for c, r in zip(out, f.itertuples(index=False)):
            if c == "flooded / bare" and getattr(r, "ndvi_left_year", np.nan) >= HARVEST_LEFT and \
                    getattr(r, "last_view_water", 0) != 1:
                seen = getattr(r, "days_since_ndvi_low", np.nan)
                age = seen if pd.notna(seen) else getattr(r, "crop_age", np.nan)
                wet = getattr(r, "water_spell", 0) == 1 and (getattr(r, "crop_age", 0) > YOUNG_MAX_DAYS or
                                                             getattr(r, "water_depth", 0) >= WATER_DEPTH_K)
                c = "young rice" if pd.notna(age) and age <= YOUNG_MAX_DAYS else (
                    "rice standing transplanted" if wet else "rice standing direct seeded")
            fixed.append(c)
        out = fixed
    if YOUNG_NEEDS_WATER and "water_depth" in f:
        depth = pd.to_numeric(f["water_depth"], errors="coerce").to_numpy()
        out = ["rice standing direct seeded" if c == "young rice" and np.isfinite(d) and d <= 0 else c
               for c, d in zip(out, depth)]
    if YOUNG_MIN_RISE_DAYS is not None and "radar_rise_days" in f:
        # a short radar rise: flooded / bare where the field still looks like water (radar low now, or the newest clear
        # view shows water); a green field whose radar never went to water level keeps a crop: standing rice
        # (aoi39 pixel 29260: no radar water dip, NDVI 0.84 on 13 Sep -> transplanted per the user, not flooded)
        fixed = []
        for c, r in zip(out, f.itertuples(index=False)):
            d = pd.to_numeric(getattr(r, "radar_rise_days", np.nan), errors="coerce")
            if c == "young rice" and np.isfinite(d) and d < YOUNG_MIN_RISE_DAYS:
                deep = getattr(r, "water_spell", 0) == 1 or getattr(r, "water_depth", 0) >= WATER_DEPTH_K
                # radar low counts as water only where the field ever went below its own dry-season level (8391: yes,
                # flooded; 29260: never, water_depth -1.1, its radar low now is a dense canopy -> a crop)
                watery = getattr(r, "last_view_water", 0) == 1 or (
                    r.vh_pos_end < LOW_NOW and getattr(r, "water_depth", 0) > 0)
                green_now = (SHORT_RISE_GREEN_YOUNG and getattr(r, "last_view_water", 0) != 1 and
                             getattr(r, "ndvi_left_year", np.nan) >= HARVEST_LEFT)
                if green_now:
                    # the newest clear view shows the young crop green (aoi13 pixel 16003: 1 Oct NDVI 0.67): young
                    # rice, though the radar has risen only a short time; 9643 (1 Oct view water) stays flooded
                    fixed.append("young rice")
                    continue
                c = "flooded / bare" if watery else (
                    "rice standing transplanted" if deep else "rice standing direct seeded")
            fixed.append(c)
        out = fixed
    if DIP_MAKES_TRANSPLANTED and "water_depth" in f:
        depth = pd.to_numeric(f["water_depth"], errors="coerce").to_numpy()
        spell = pd.to_numeric(f.get("water_spell", pd.Series(0, index=f.index)), errors="coerce").to_numpy()
        # only a crop whose radar has risen long enough (YOUNG_MIN_RISE_DAYS); a short rise kept young by
        # SHORT_RISE_GREEN_YOUNG stays young (aoi13 pixel 16003)
        rise = pd.to_numeric(f.get("radar_rise_days", pd.Series(np.nan, index=f.index)), errors="coerce").to_numpy()
        short = np.isfinite(rise) & (rise < YOUNG_MIN_RISE_DAYS) if YOUNG_MIN_RISE_DAYS is not None else \
            np.zeros(len(f), dtype=bool)
        out = ["rice standing transplanted" if c == "young rice" and sp == 1 and np.isfinite(d) and d >= WATER_DEPTH_K
               and not sh else c for c, d, sp, sh in zip(out, depth, spell, short)]
    return pd.Series(out, index=f.index)


# ------------------------------------------------------------------------------------------ locked AOIs and overrides
#: AOIs whose result the user accepted (user, 2 Oct: "aoi160 is done and locked"). Their outputs in
#: ``rice_fresh/aoi<N>/`` are never rewritten (``force=True`` to override on the user's word); the frozen copy, the
#: code and the labels at lock time are in ``rice_fresh/locked/aoi<N>/`` with checksums (MANIFEST.json).
LOCKED_AOIS = {160, 28, 72, 116, 39, 63, 13}
#: Rule changes for ONE AOI (user, 2 Oct: "try the aoi160 rules first; if they do not work, new rules only for that
#: AOI, not 160"): ``{aoi: {"NAME": value, "field_polygons.NAME": value}}``. The module constants are the aoi160 rules;
#: an AOI without an entry runs exactly those.
AOI_OVERRIDES: dict = {
    28: {"AGE_LOW_SHARE": 0.05,         # user, 2 Oct, pixel 3345: age from the end of the empty spell
         "YOUNG_NEEDS_AGE": True,       # ...and young rice only up to 50 days, on every path
         "SIEVE_ACRES": 0.15},          # user, 2 Oct: 0.5 ac removed all of aoi28's young rice
    # aoi72 (user, 2 Oct): start from the aoi28 rules
    72: {"AGE_LOW_SHARE": 0.05, "YOUNG_NEEDS_AGE": True, "SIEVE_ACRES": 0.15,
         "YOUNG_AFTER_LAST_WATER": True,    # pixel 42109: last view water, radar up since -> young rice
         "AGE_FROM_WATER": True,            # pixel 27550: a transplanted crop's age from its water spell
         "WATER_BOTTOM": 0.3,               # pixel 26433: a two-month flood sits at 0.16-0.18 of its own range
         "WATER_FALL_LOOKBACK_DAYS": 45,    # ...and its fall is from the pre-flood high, not the previous pass
         "YOUNG_WHILE_RADAR_LOW": True,     # pixel 19247: radar still low and rising, NDVI still climbing -> young
         "WATER_SPELL_DAYS": 30},           # pixel 42979: water on 18 May (DSC) and 14 Jun (ASC); the DSC mid-June
    #                                         pass has its VH dropped (artefact list), so 15 days found one wet pass only
    # TREE_LOW_BEFORE "2026-06-01" was tried for aoi28 (pixel 7826) and reverted by the user (2 Oct): it moved 11.9 ac
    # of trees into rice
}
#: aoi39 (user, 3 Oct): works on the aoi160 rule set = the module defaults; the aoi72 set was tried and rejected by the
#: user (its map is kept in ``rice_fresh/aoi39/rule_trials/rules_aoi72/``). Its own changes, aoi39 ONLY:
AOI_OVERRIDES[39] = {"YOUNG_WHILE_RADAR_LOW": True,   # pixel 8391: watered late Aug, radar rising, NDVI 0.74 on 13 Sep
                     #                                  and climbing; "15 days past the peak" was the fit held flat
                     "YOUNG_RADAR_LOW_NEEDS_RISE": True,  # pixel 11226: dense canopy (NDVI 0.88) lowering the radar in
                     #                                  September is not young: the radar must still be climbing
                     "AGE_FROM_WATER": True,          # pixel 11226: transplanted, sown 30 Jun = start of its water spell
                     "WATER_BOTTOM": 0.3,             # pixel 8384: VV fell gradually into the water level (-7.8 on
                     "WATER_FALL_LOOKBACK_DAYS": 45,  # 14 Jun, -10.4 on 15 Jul, -10.9 on 27 Jul): fall from the 45-day
                     "WATER_SPELL_MIN_PASSES": 1,     # high, and one wet pass while small (the passes around it: VH
#                                                        low with VV high = seedlings in water) -> transplanted, 61 days
                     "GROWN_NOT_IF_WATER_VIEW": True, # pixel 21405: newest clear view open water -> flooded, not harvested
                     "POND_IF_DRY_WATER": True,       # pixel 30425: water on the March-April views -> fish pond
                     "SECOND_CROP_BY_RADAR": True,    # pixel 24716: deep water after an earlier crop, radar up now ->
#                                                        a new transplanted crop (not tree / other vegetation)
                     "RADAR_DECIDES_WITHOUT_OPTICAL": True,  # pixel 21618: the optical never saw the water -> radar alone
                     "YOUNG_AFTER_LAST_WATER": True,  # pixel 25885: last clear view water (13 Sep), VV and VH up since ->
#                                                        young (the aoi72 rule of 42109; user accepted young 24 -> 56 ac)
                     "HARVEST_NOT_IF_RADAR_RISING": True,  # 37570 / 37571 / 39894 / 39895: radar at its top and rising
                     "YOUNG_MIN_RISE_DAYS": 40,       # 43450 (+ 8391 / 15496 / 25885 relabelled): VH must have risen
#                                                        for 40 days, else flooded / bare
                     "YOUNG_NEEDS_WATER": True,       # 29972 / 26437 / 13545: never held water, green -> direct seeded
                     "SIEVE_ACRES": 0.15}             # as aoi28 / aoi72 / aoi116 (0.5 erases small young-rice patches)
#: Every NEW AOI (after aoi39) starts with RADAR_DECIDES_WITHOUT_OPTICAL on (user, 3 Oct: "from now on, wherever there is
#: no NDVI, the radar alone decides"): put it in that AOI's AOI_OVERRIDES entry together with the chosen rule set.
NEW_AOI_SWITCHES = {"RADAR_DECIDES_WITHOUT_OPTICAL": True,
                    # user, 5 Oct (after aoi13): universal rules for every new AOI - latest clear view's tone + curves,
                    # harvested only when VV and VH agree, no NDVI -> the radar tells the story
                    "TONE_AND_CURVE": True, "HARVEST_NEEDS_BOTH_POLS": True, "RADAR_STORY_IN_GAP": True}
#: aoi116 (user, 2 Oct): starts from the aoi72 rule set, chosen after ``try_rules`` (aoi160 / aoi28 rules: ~3,030 ac
#: direct seeded; aoi72 rules: ~4,770 ac transplanted, which the user found more plausible). A COPY, so a later aoi116
#: change never touches the locked aoi72 entry.
AOI_OVERRIDES[116] = dict(AOI_OVERRIDES[72],
                          TREE_NEEDS_NO_WATER=True,   # pixel 168977: no optical view Jun-Aug, radar water -> not a tree
                          SOWING_FROM_RADAR=True,     # pixel 186792: planted at the end of its water spell (~1 Sep)
                          WATER_END_TRACKS="earliest_unless_wet_view",  # 116025: first track rising (27 Jul);
#                                                       143310: a water view after it -> the later track (29 Aug)
                          WATER_FALL_POLS="VV")       # 217385 / 112734: flooding seen as a VV fall; VH already dark
#: aoi63 (user, 5 Oct): starts from the aoi39 rule set, chosen after ``try_rules`` (aoi39 already carries
#: ``NEW_AOI_SWITCHES`` and the 40-day young-rice rise). A COPY, so a later aoi63 change never touches locked aoi39.
AOI_OVERRIDES[63] = dict(AOI_OVERRIDES[39], RADAR_DECIDES_WITHOUT_OPTICAL=True,   # NEW_AOI_SWITCHES of 5 Oct, frozen
                         HARVEST_NOT_IF_RADAR_RISING=False,  # 12923: two clear views (26 / 28 Sep) at the bare-soil level;
#                                                              the radar rising after the cut is tillage / wet soil
                         CANOPY_NOT_FLOODED=True,            # 20447: NDVI 0.86 and climbing, radar lowered by the canopy
                         RADAR_STORY_IN_GAP=True,            # no clear view 3 Jul - 26 Sep: the radar tells the season
                         WATER_SMALL_UNSEEN=True,            # 31277: July water under an unseen NDVI (summer crop's fit)
                         POND_NEEDS_NO_CROP=True,            # 41938: dry-season water, green crop now -> a paddy
                         CANOPY_OVER_RADAR_WATER=True,       # 13571: radar low under the year's greenest view = canopy
                         GREEN_VIEW_NOT_FLOODED=True)        # 36796 / 23348: green on 26 / 28 Sep -> not flooded
#: aoi13 (user, 5 Oct): harvested everywhere until about 18 Jul, then under water to mid / late August, planted after the
#: water. Starts from the aoi63 set (chosen after ``try_rules``) with the crop's age from the END of its water spell
#: (``SOWING_FROM_RADAR``, as aoi116 186792): from its start the age was ~97 days for crops planted in late August.
#: A COPY, so a later aoi13 change never touches locked aoi63.
AOI_OVERRIDES[13] = dict(AOI_OVERRIDES[63], SOWING_FROM_RADAR=True,
                         YOUNG_MIN_RISE_DAYS=30,             # user: 30 days here (planted late Aug), 40 in aoi39 / aoi63
                         DIP_MAKES_TRANSPLANTED=True,        # user: a clear VV / VH dip -> transplanted, even at 30 days
                         SOWING_NOT_AFTER_SEEN_CROP=True,    # 14994: crop seen on 17 Aug, sown mid July, not 29 Aug
                         STORY_YOUNG_AFTER_LAST_WATER=True,  # 11791: last view water (19 Aug), radar up since -> a crop
                         SHORT_RISE_GREEN_YOUNG=True,        # 16003: radar up 24 days, green on 1 Oct -> young
                         NEWEST_EMPTY_NOT_RICE=True)         # user: empty on the latest image -> not rice


@contextmanager
def rules_for(aoi: int):
    """Applies ``AOI_OVERRIDES[aoi]`` to this module's (and ``field_polygons``') constants for the duration, then
    restores them, so one AOI's special rules never leak into another run."""
    with switches(AOI_OVERRIDES.get(aoi, {}), f"AOI_OVERRIDES[{aoi}]"):
        yield


@contextmanager
def switches(overrides: dict, label: str = "switches"):
    """Applies ``{"NAME": value, "field_polygons.NAME": value}`` for the duration and restores the old values after.
    Shared by ``rules_for`` and ``try_rules`` (which lays ``NEW_AOI_SWITCHES`` over each finished AOI's set)."""
    from . import field_polygons as fl

    saved = []
    try:
        for key, value in overrides.items():
            mod, name = (fl, key.split(".", 1)[1]) if key.startswith("field_polygons.") else (sys.modules[__name__], key)
            if not hasattr(mod, name):
                raise KeyError(f"{label}: unknown rule constant {key}")
            saved.append((mod, name, getattr(mod, name)))
            setattr(mod, name, value)
        yield
    finally:
        for mod, name, value in reversed(saved):
            setattr(mod, name, value)


def lock(aoi: int, sliver_acres: float, fresh: str = "processed/_batch/s2_2026/rice_fresh") -> dict:
    """Freezes an accepted AOI (user, 2 Oct: "aoi160 / aoi28 done, save the outputs"): copies the raw and sieved class
    rasters, the chosen field layer (``sliver_acres``), their styles and the acre table to ``rice_fresh/locked/aoi<N>/``
    together with the rule code and the AOI's labels at that moment, and writes MANIFEST.json (sha256 per file, the
    rules used). Add the AOI to ``LOCKED_AOIS`` afterwards so it is never rewritten."""
    import hashlib
    import json
    import shutil

    from .curve_labels import load

    src = Path(fresh) / f"aoi{aoi}"
    dst = Path(fresh) / "locked" / f"aoi{aoi}"
    (dst / "code").mkdir(parents=True, exist_ok=True)
    tag = f"{int(round(sliver_acres * 100)):03d}"
    names = [f"aoi{aoi}_rel_class.tif", f"aoi{aoi}_rel_class.qml", f"aoi{aoi}_rel_class_sieved.tif",
             f"aoi{aoi}_rel_class_sieved.qml", f"aoi{aoi}_rel_fields_sliver{tag}.gpkg",
             f"aoi{aoi}_rel_fields_sliver{tag}.qml", f"aoi{aoi}_rel_fields_acres.csv", f"aoi{aoi}_rel.csv",
             f"aoi{aoi}_rel_sieved.csv"]
    for n in names:
        if (src / n).exists():
            shutil.copy2(src / n, dst / n)
    here = Path(__file__).parent
    for n in ("curve_rules.py", "field_polygons.py", "curve_labels.py"):
        shutil.copy2(here / n, dst / "code" / n)
    lab = load()
    lab[lab["aoi"] == aoi].to_csv(dst / f"aoi{aoi}_labels_at_lock.csv", index=False)
    from . import ndvi_5day as nd
    from . import pixel_report as pr

    # the inputs the result was made with (user, 2 Oct: newer images come in new folders; a lock must say which it used)
    inputs = {"series_root": nd.analysis_series_root(aoi),
              "radar_run": Path(pr.locate(aoi, 0, season_key="monsoon2026")["run"]).name}
    man = {"aoi": aoi, "locked": str(pd.Timestamp.now().date()), "final": f"aoi{aoi}_rel_fields_sliver{tag}.gpkg",
           "inputs": inputs,
           "sliver_acres": sliver_acres, "rules": "aoi160 rules" + (f" + {AOI_OVERRIDES[aoi]}" if aoi in AOI_OVERRIDES
                                                                    else ""),
           "files": {str(p.relative_to(dst)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(dst.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}}
    (dst / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
    return man


def compare_outputs(aoi: int, new_dir, ref_dir, sliver_acres: float = 0.15) -> dict:
    """Compares one AOI's rule outputs in ``new_dir`` with those in ``ref_dir`` (both folders holding
    ``aoi<N>_rel_*`` files): the raw and sieved class rasters pixel by pixel, the field layer of ``sliver_acres`` by
    polygon count and acres per class. A file missing on either side is reported, not compared.

    Why (2 Oct 2026, handover to a new machine): "the restore is exact" must be shown with numbers, not assumed; a
    missing input (e.g. the radar artefact-pass list) changes results silently."""
    import geopandas as gpd
    import rasterio

    new_dir, ref_dir = Path(new_dir), Path(ref_dir)
    out = {"aoi": aoi}
    for kind, name in (("raw", f"aoi{aoi}_rel_class.tif"), ("sieved", f"aoi{aoi}_rel_class_sieved.tif")):
        a, b = new_dir / name, ref_dir / name
        if not (a.exists() and b.exists()):
            out[kind] = "missing: " + ", ".join(str(p) for p in (a, b) if not p.exists())
            continue
        with rasterio.open(a) as da, rasterio.open(b) as db:
            x, y = da.read(1), db.read(1)
            same_grid = x.shape == y.shape and da.transform == db.transform and da.crs == db.crs
        diff = int((x != y).sum()) if x.shape == y.shape else -1
        out[kind] = {"identical": bool(same_grid and diff == 0), "pixels_differing": diff, "same_grid": same_grid}
    name = f"aoi{aoi}_rel_fields_sliver{int(round(sliver_acres * 100)):03d}.gpkg"
    a, b = new_dir / name, ref_dir / name
    if a.exists() and b.exists():
        fa, fb = gpd.read_file(a), gpd.read_file(b)
        acres_a = fa.groupby("class_name")["acres"].sum().round(2)
        acres_b = fb.groupby("class_name")["acres"].sum().round(2)
        acres_diff = float((acres_a.sub(acres_b, fill_value=0)).abs().max()) if len(fa) or len(fb) else 0.0
        out["fields"] = {"polygons": len(fa), "polygons_ref": len(fb), "max_class_acres_diff": round(acres_diff, 3),
                         "acres": acres_a.to_dict(),
                         "identical": len(fa) == len(fb) and acres_diff == 0.0}
    else:
        out["fields"] = "missing: " + ", ".join(str(p) for p in (a, b) if not p.exists())
    return out


def manifest_mismatches(ref, man: dict) -> list[str]:
    """Files of a locked folder whose sha256 differs from MANIFEST.json (``"<name> (missing)"`` when absent). aoi160 was
    locked by hand before :func:`lock` existed: its manifest names the code files without the ``code/`` folder they
    sit in, so a name not found at the top is looked up in ``code/``."""
    import hashlib

    ref = Path(ref)
    bad = []
    for name, digest in man["files"].items():
        path = ref / name if (ref / name).exists() else ref / "code" / name
        if not path.exists():
            bad.append(f"{name} (missing)")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append(name)
    return bad


def reproduce(aoi: int, scratch, fresh: str = "processed/_batch/s2_2026/rice_fresh",
              series_root: str | None = None) -> dict:
    """Rebuilds a LOCKED AOI from zero in ``scratch`` (raw rule, sieve, fields with its own rules) and compares it with
    the frozen copy in ``rice_fresh/locked/aoi<N>/`` (:func:`compare_outputs`); also re-checks the frozen files'
    sha256 against MANIFEST.json. Only ``scratch`` is written; the real folder and the locked copy are read only.

    Why (2 Oct 2026, handover): a restored project must prove it reproduces the accepted results exactly before work
    continues; the rule reads many inputs (series, radar stacks, artefact passes, delineation) that a copy can miss."""
    import json
    import shutil

    ref = Path(fresh) / "locked" / f"aoi{aoi}"
    scratch = Path(scratch)
    if scratch.resolve() == Path(fresh).resolve() or Path(fresh).resolve() in scratch.resolve().parents:
        raise ValueError("scratch must lie outside the real rice_fresh folder")
    man = json.loads((ref / "MANIFEST.json").read_text())
    bad = manifest_mismatches(ref, man)
    (scratch / f"aoi{aoi}").mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}_step1_cover.tif", scratch / f"aoi{aoi}")
    # force: the guard keys on the AOI number, but nothing outside ``scratch`` is written here
    run(aoi, series_root=series_root, fresh=str(scratch), force=True)
    sieve(aoi, fresh=str(scratch), force=True)
    fields(aoi, fresh=str(scratch), force=True)
    out = compare_outputs(aoi, scratch / f"aoi{aoi}", ref, man["sliver_acres"])
    out["manifest_files_changed"] = bad
    return out


def _guard(aoi: int, force: bool) -> None:
    if aoi in LOCKED_AOIS and not force:
        raise PermissionError(f"aoi{aoi} is locked (accepted by the user); its outputs are not rewritten. "
                              f"Frozen copy: rice_fresh/locked/aoi{aoi}/. Pass force=True / --force only on request.")


#: Map codes of the relative rule.
MAP_CLASSES = {1: ("rice standing direct seeded", "#1a9850"), 7: ("rice standing transplanted", "#00441b"),
               2: ("rice harvested", "#fee08b"), 3: ("young rice", "#9ecae1"),
               4: ("flooded / bare", "#2166ac"), 5: ("other vegetation", "#e3a21a"), 6: ("tree/orchard", "#1b5e20"),
               0: ("no data", "#d9d9d9")}


def run(aoi: int, series_root: str | None = None,
        fresh: str = "processed/_batch/s2_2026/rice_fresh", force: bool = False) -> pd.DataFrame:
    """Locked-AOI guard and per-AOI rules around :func:`_run`."""
    from . import ndvi_5day as nd

    _guard(aoi, force)
    with rules_for(aoi):
        return _run(aoi, series_root or nd.analysis_series_root(aoi), fresh)


def _run(aoi: int, series_root: str, fresh: str) -> pd.DataFrame:
    """The relative rule on every pixel inside the AOI; writes ``aoi<N>_rel_class.tif`` + ``.qml`` and
    ``aoi<N>_rel.csv`` (acres per class) in ``rice_fresh/aoi<N>/``."""
    import rasterio

    from . import ndvi_5day as nd
    from .optical_phenology import acres

    inside = np.flatnonzero(nd.inside_aoi(aoi).ravel())
    nd.forget()
    f = own_range_features(aoi, inside, series_root)
    cls = classify_relative(f)
    code = {v[0]: k for k, v in MAP_CLASSES.items()}
    need = ["ndvi_low_peak", "vh_pos_end", "ndvi_left"]
    vals = np.where(f[need].notna().all(axis=1), cls.map(code).to_numpy(), 0).astype("uint8")
    stem = Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}"
    with rasterio.open(f"{stem}_step1_cover.tif") as ds:
        profile = ds.profile
    out = np.full(profile["height"] * profile["width"], 255, dtype="uint8")
    out[inside] = vals
    with rasterio.open(f"{stem}_rel_class.tif", "w", **dict(profile, dtype="uint8", nodata=255, count=1)) as ds:
        ds.write(out.reshape(profile["height"], profile["width"])[None])
    entries = "\n".join(f'        <paletteEntry value="{k}" color="{c}" alpha="255" label="{k} {n}"/>'
                        for k, (n, c) in MAP_CLASSES.items())
    Path(f"{stem}_rel_class.qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
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
    res = pd.DataFrame([{"class": k, "name": n, "acres": round(acres(int((vals == k).sum())), 1)}
                        for k, (n, _) in MAP_CLASSES.items()])
    res.to_csv(f"{stem}_rel.csv", index=False)
    return res


def nearest_locked(aoi: int, index: pd.DataFrame | None = None, locked=None) -> pd.DataFrame:
    """The locked AOIs ordered by distance (km, between AOI centroids) from ``aoi``. Why (user, 5 Oct 2026): "every new
    AOI starts from the universal rules plus the rules of the locked AOI nearest to it (nearest neighbour)": nearby
    fields share a calendar, water regime and cloud gaps. ``index``: ``prep.batch.aoi_index()`` (read when None)."""
    import math

    if index is None:
        from ..prep.batch import aoi_index
        index = aoi_index()
    locked = sorted(LOCKED_AOIS if locked is None else locked)
    me = index.set_index("aoi").loc[aoi]
    rows = []
    for a in locked:
        o = index.set_index("aoi").loc[a]
        dx = (o.lon - me.lon) * 111.32 * math.cos(math.radians((o.lat + me.lat) / 2))
        dy = (o.lat - me.lat) * 110.57
        rows.append({"locked_aoi": a, "km": round(math.hypot(dx, dy), 1), "acres": o.acres})
    return pd.DataFrame(rows).sort_values("km").reset_index(drop=True)


#: The finished AOIs whose rule sets a new AOI is tried with first (user, 2 Oct, aoi116: "apply the rules of the 3 AOIs
#: we have done, tell me the results of each, then we decide which one to tune"). Each source's ``AOI_OVERRIDES`` entry
#: (none for 160 = the defaults) is applied in turn, with ``NEW_AOI_SWITCHES`` on top (aoi63, 5 Oct: every new AOI
#: carries them, so the trial must too). aoi116 / aoi39 added once locked.
RULE_SOURCES = (160, 28, 72, 116, 39)


def try_rules(aoi: int, sources=RULE_SOURCES, series_root: str | None = None,
              fresh: str = "processed/_batch/s2_2026/rice_fresh", out: str | None = None) -> pd.DataFrame:
    """Step 1 on a new AOI: the raw rule map with each finished AOI's rule set, side by side, so the user can pick the
    set to start tuning from (each with ``NEW_AOI_SWITCHES`` on top). Each set's map goes to ``rice_fresh/aoi<N>/rule_trials/rules_aoi<src>/aoi<N>/``
    (``aoi<N>_rel_class.tif`` + style + acre table); the comparison to ``rule_trials/aoi<N>_rule_trials.csv`` (acres
    per class, one column per rule set). The AOI's own outputs in ``rice_fresh/aoi<N>/`` are not touched, so nothing is
    chosen until the user decides; the AOI's own pinned inputs are used (``ndvi_5day.analysis_series_root`` and the
    radar run pin)."""
    import shutil

    from . import ndvi_5day as nd

    series_root = series_root or nd.analysis_series_root(aoi)
    out = Path(out) if out else Path(fresh) / f"aoi{aoi}" / "rule_trials"
    step1 = Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}_step1_cover.tif"
    cols = []
    for src in sources:
        d = out / f"rules_aoi{src}"
        (d / f"aoi{aoi}").mkdir(parents=True, exist_ok=True)
        shutil.copy2(step1, d / f"aoi{aoi}" / step1.name)          # the rule's output grid / profile
        with rules_for(src), switches(NEW_AOI_SWITCHES, "NEW_AOI_SWITCHES"):
            res = _run(aoi, series_root, str(d))
        cols.append(res.set_index(["class", "name"])["acres"].rename(f"aoi{src} rules"))
    table = pd.concat(cols, axis=1).reset_index()
    table.to_csv(out / f"aoi{aoi}_rule_trials.csv", index=False)
    return table


#: Smallest patch kept on the map (user, 2 Oct: "sieve to remove noise", 0.5 ac = 20 pixels; 0.2 was tried, user chose 0.5);
#: 1 pixel = 0.025 ac.
SIEVE_ACRES = 0.5


def sieve(aoi: int, acres_min: float | None = None, fresh: str = "processed/_batch/s2_2026/rice_fresh",
          force: bool = False) -> pd.DataFrame:
    """Patches smaller than ``acres_min`` of ``aoi<N>_rel_class.tif`` merged into their largest neighbour
    (``rasterio.features.sieve``, 8-connected); outside the AOI stays nodata. Writes ``aoi<N>_rel_class_sieved.tif``
    + ``.qml`` and ``aoi<N>_rel_sieved.csv``; the unsieved map is kept for checking single pixels."""
    import shutil

    import rasterio
    from rasterio.features import sieve as rio_sieve

    _guard(aoi, force)
    with rules_for(aoi):
        acres_min = SIEVE_ACRES if acres_min is None else acres_min

    from .optical_phenology import acres

    stem = Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}"
    with rasterio.open(f"{stem}_rel_class.tif") as ds:
        a = ds.read(1)
        profile = ds.profile
    inside = a != 255
    size = max(int(round(acres_min / 0.025)), 1)
    out = rio_sieve(np.where(inside, a, 0).astype("uint8"), size=size, mask=inside, connectivity=8)
    out = np.where(inside, out, 255).astype("uint8")
    with rasterio.open(f"{stem}_rel_class_sieved.tif", "w", **profile) as ds:
        ds.write(out[None])
    shutil.copyfile(f"{stem}_rel_class.qml", f"{stem}_rel_class_sieved.qml")
    v = out[inside]
    res = pd.DataFrame([{"class": k, "name": n, "acres": round(acres(int((v == k).sum())), 1),
                         "unsieved_acres": round(acres(int((a[inside] == k).sum())), 1)}
                        for k, (n, _) in MAP_CLASSES.items()])
    res.to_csv(f"{stem}_rel_sieved.csv", index=False)
    return res



#: Sliver thresholds compared (user, 2 Oct: "start from 0.05 acres, increments of 0.05 till 0.15").
SLIVER_TRIALS = (0.05, 0.10, 0.15)


def fields_qml(path) -> None:
    """A QGIS style next to a field layer (same name, ``.qml``): polygons outlined and filled by ``class_name`` in the
    map's colours, so the layer opens styled (QGIS loads a same-named .qml automatically)."""
    cats, syms = [], []
    for i, (k, (name, col)) in enumerate((k, v) for k, v in MAP_CLASSES.items() if k):
        h = col.lstrip("#")
        rgb = ",".join(str(int(h[j:j + 2], 16)) for j in (0, 2, 4))
        cats.append(f'<category value="{name}" symbol="{i}" label="{name}" render="true"/>')
        syms.append(f'''<symbol type="fill" name="{i}" alpha="1"><layer class="SimpleFill">
<Option type="Map"><Option type="QString" name="color" value="{rgb},90"/><Option type="QString" name="outline_color"
value="{rgb},255"/><Option type="QString" name="outline_width" value="0.5"/></Option></layer></symbol>''')
    Path(path).with_suffix(".qml").write_text(f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28">
  <renderer-v2 type="categorizedSymbol" attr="class_name">
    <categories>{''.join(cats)}</categories>
    <symbols>{''.join(syms)}</symbols>
  </renderer-v2>
</qgis>
""")


def fields(aoi: int, fresh: str = "processed/_batch/s2_2026/rice_fresh", sieved: bool = True,
           force: bool = False) -> pd.DataFrame:
    """Delineated fields labelled from the relative-rule map (sieved by default), cut where one polygon holds two
    fields of different classes, new polygons where no field exists, and same-label neighbours merged
    (``field_polygons``, ported from the user's sugarcane project). Writes ``aoi<N>_rel_fields_pieces.gpkg`` (every
    piece with its origin and decision, before sliver absorption), ``aoi<N>_rel_fields_sliver<005|010|015>.gpkg`` (the
    final layer per sliver threshold) and ``aoi<N>_rel_fields_acres.csv`` (one row per threshold). Why (user, 2 Oct): the client receives fields, not pixels."""
    import rasterio

    from . import field_polygons as fl
    from .field_rice import RECORD_ONLY_FLAGS, load_fields

    stem = Path(fresh) / f"aoi{aoi}" / f"aoi{aoi}"
    path = f"{stem}_rel_class_sieved.tif" if sieved else f"{stem}_rel_class.tif"
    with rasterio.open(path) as ds:
        classes = ds.read(1)
        transform, crs = ds.transform, ds.crs
    f = load_fields(aoi)
    if "refine_flag" in f:
        f = f[~f["refine_flag"].isin(RECORD_ONLY_FLAGS)]
    names = {k: v[0] for k, v in MAP_CLASSES.items() if k}
    _guard(aoi, force)
    with rules_for(aoi):
        pieces, finals = fl.run(f, classes, transform, crs, names, sliver_acres=SLIVER_TRIALS)
    pieces.to_file(f"{stem}_rel_fields_pieces.gpkg", driver="GPKG")
    rows = []
    for thr, fin in finals.items():
        out_path = f"{stem}_rel_fields_sliver{int(round(thr * 100)):03d}.gpkg"
        fin.to_file(out_path, driver="GPKG")
        fields_qml(out_path)
        rows.append({"sliver_acres": thr, "polygons": len(fin), "pieces_absorbed": int(fin["absorbed"].sum()),
                     "small_alone": int(fin["small_alone"].sum()), "largest_acres": round(float(fin["acres"].max()), 1),
                     "overlap_acres": round(fl.overlap_acres(fin), 4),
                     "tidy_lost_acres": fin.attrs.get("tidy_lost_acres", 0.0),
                     **{n: round(float(fin.loc[fin["class_name"] == n, "acres"].sum()), 1) for n in names.values()}})
    res = pd.DataFrame(rows)
    res.to_csv(f"{stem}_rel_fields_acres.csv", index=False)
    return res


def table(aoi: int) -> pd.DataFrame:
    """The user's labels for one AOI (``curve_labels``) joined with the measured features."""
    from .curve_labels import load

    lab = load()
    lab = lab[lab["aoi"] == aoi]
    f = features(aoi, lab["pixel"].to_numpy())
    out = lab[["pixel", "label", "state", "sowing", "establishment"]].merge(f, on="pixel")
    return out.sort_values(["label", "state", "establishment"]).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("step", choices=["table", "run", "fields", "try-rules", "reproduce"])
    p.add_argument("--aoi", type=int, required=True)
    p.add_argument("--scratch", help="reproduce: a folder OUTSIDE rice_fresh where the locked AOI is rebuilt and compared")
    p.add_argument("--series-root", default=None,
                   help="run: the 5-day Sentinel-2 series the rule reads (default: the AOI's pinned series, "
                        "processed/aoi<N>/monsoon2026/ANALYSIS_SERIES.txt; a newer series has its own folder); the radar "
                        "run is the AOI's pinned one (python -m sar_pipeline --config ... pin-run)")
    p.add_argument("--out", help="also write the table to this CSV")
    p.add_argument("--force", action="store_true", help="rewrite a LOCKED AOI's outputs (only on the user's word)")
    args = p.parse_args(argv)
    if args.step == "run":
        print(run(args.aoi, series_root=args.series_root, force=args.force).to_string(index=False))
        print("sieved:")
        print(sieve(args.aoi, force=args.force).to_string(index=False))
        return 0
    if args.step == "try-rules":
        print(try_rules(args.aoi, series_root=args.series_root).to_string(index=False))
        return 0
    if args.step == "fields":
        print(fields(args.aoi, force=args.force).to_string(index=False))
        return 0
    if args.step == "reproduce":
        # proof that a locked AOI still rebuilds exactly (after a restore, or after new inputs were added beside it)
        if not args.scratch:
            p.error("reproduce needs --scratch")
        import json

        out = reproduce(args.aoi, args.scratch)
        print(json.dumps(out, indent=1, default=str))
        return 0
    t = table(args.aoi)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 50)
    print(t.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(args.out, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
