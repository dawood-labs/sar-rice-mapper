"""Shape of one pixel's year, measured from Sentinel-2 only. No radar, no pass/fail thresholds.

Why
---
The radar draft map was anchored to the wrong months: NDVI showed the canopy peaking in mid-August
and the crop cut by mid-September, while VH did not reach its maximum until November. Before any
radar feature is trusted again, two agronomic statements need an independent optical check:

**H1 — a rice crop is short.** Sowing in May/June, canopy peak around August, harvest from mid
September into October; sowing to harvest is roughly four months, not more.

**H2 — there must be water at sowing.** Paddy is transplanted into a flooded field. A field that was
never wet when its crop started is not paddy.

Why this module contains no thresholds
--------------------------------------
A cut such as "NDVI peak above 0.55" or "NDWI rise above 0.15" is a number chosen by the analyst, not
by the data, and it does not transfer between pixels, AOIs or years: a hazy pixel, a mixed pixel and a
clean pixel have different absolute levels while having the same seasonal *shape*. So every quantity
here is measured **against the pixel's own curve**:

* amplitude is peak minus that pixel's own pre-peak trough;
* the crop's length is the width of its own curve at half of its own amplitude, so it does not depend
  on how green the crop got;
* green-up and senescence are **rates** (NDVI units per day) — a crop changes fast, a tree or a marsh
  does not, whatever their absolute NDVI;
* the NDWI descriptors are kept, but **they do not test for water**. Measured over 904,458 clear
  pixel-dates, NDWI (green, NIR) correlates with NDVI at r = -0.969, and only 0.041 of its 0.149
  spread survives once its NDVI dependence is removed. Both indices are driven by the same NIR band,
  so "NDWI rises at sowing" is very nearly a restatement of "NDVI is low": it confirms bare ground,
  not a flooded field. Separating wet from dry bare soil needs SWIR (B11), where water absorbs
  strongly and dry soil is bright — that is what LSWI = (B8 - B11) / (B8 + B11) uses. The first
  per-date exports carried no B11, so H2 could not be answered from them; the 5-day window
  composites from ``s2_windows`` do carry it, and the wet-at-sowing test in ``optical_rice_map``
  reads LSWI from those;
* **emergence** is found the way the sowing-detection pipeline written for an earlier rice project finds
  it: on the gap-filled, lightly smoothed curve, the crop starts where the rise *accelerates* hardest
  (the maximum of the second derivative), not where NDVI crosses some level. Acceleration is a shape
  property, so it survives a hazy pixel or a mixed pixel that never reaches a "normal" NDVI;
* **timing** is kept as a separate descriptor: how many days the NDWI maximum leads or lags the NDVI
  trough. For a transplanted paddy the two coincide.

Why there are no month windows either
-------------------------------------
The 132 AOIs run from the north of the country to the south, and the same crop is sown weeks apart
between them: agronomically identical, shifted in time. A detector that looks for a peak "between
15 June and 15 November" would therefore work in one region and fail in another, and would quietly
mis-time every AOI in between. So cycles are found **wherever they fall in the series**: the curve
is split at its own local minima, every peak between two minima is one cycle, and *when* that cycle
happened is an output of the measurement, not a condition on it. Comparing sowing dates between
regions then becomes a result the data can give, instead of an assumption built into the code.

Deciding where each descriptor separates rice from everything else is a later, data-driven step
(see ``distributions``), done by looking at the shape of the distributions, not by picking numbers
up front.

Cloud handling: values are interpolated onto a regular grid between clear observations, and a grid
date whose surrounding clear observations are further apart than ``max_gap_days`` is left empty
rather than invented. A pixel with too little clear data is reported as undecidable, never as
non-rice.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import pixel_report as pr

#: How far either side of a cycle's start to look for standing water, in days. This is a length of
#: time, not a calendar date, so it travels between regions: transplanting and the puddling before it
#: occupy two to three weeks wherever the field is.
WATER_WINDOW_DAYS = 20
#: Days from the field's start (transplanting, or direct seeding) to the **green-up onset**, the
#: point where NDVI starts to climb fastest. A 10 m pixel at transplanting is ~90 % water with small
#: seedlings in it, so NDVI stays near zero until the plants have tillered enough to cover the water;
#: the nursery sowing that precedes transplanting happens elsewhere and is invisible here. Measured on
#: 2,950 field plots in September 2026: 15 days in the delta, 20 in the other regions. The trough is
#: the field start whenever it is observed; this lag is only used to date the start from the onset
#: when the trough fell in a cloud gap.
GREENUP_LAG_DAYS = 15
#: Where the field's dry baseline is read: a stretch before the cycle starts, far enough back to be
#: before any puddling. The median over ``[start - BASELINE_FROM, start - BASELINE_TO]`` days.
BASELINE_FROM, BASELINE_TO = 75, 25
#: A local minimum or peak counts when it stands out by this fraction of the pixel's own NDVI range.
#: Relative, so a faint crop and a vigorous one are segmented the same way.
PROMINENCE_FRACTION = 0.3
#: How long the curve must stay down for a dip to count as the end of a crop, in days.
#: A harvested field does not green up again within three weeks, so a dip that recovers faster is
#: residual cloud, not a harvest. Without this one crop was being cut into two cycles by a single
#: contaminated observation, and the larger fragment was then reported as the pixel's main crop.
MIN_LOW_DAYS = 20
#: Cloud Score+ value a pixel-date must reach to be used. Applies only to the per-date exports; the
#: window composites are masked with s2cloudless before compositing and carry no clear score. 0.60
#: let through enough thin haze to put a 0.3-amplitude sawtooth through the whole monsoon; 0.75 keeps
#: a median of 38 clear observations per pixel between April and November, against 47 at 0.60, and 21
#: at 0.85 where the series collapses.
CLEAR_MIN = 75
#: Whittaker smoothing strength. The window composites are already a median of everything clear in
#: their five days, so the series reaching the smoother is far less noisy than single dates were and
#: needs correspondingly less of it: 0.5 bridges a missing window without rounding off the harvest
#: edge that the whole measurement depends on.
LMBD = 0.5
#: Which reader each GCS export folder needs. "windows" files are already on a regular grid;
#: "dates" files are one acquisition each and have to be interpolated onto one first.
SOURCES = {"s2_windows5d": "windows", "s2_reference_swir": "dates", "s2_reference": "dates"}


def read_cube(aoi_id: int, clear_min: float = CLEAR_MIN, cache_root="data/s2_reference",
              folder: str = "s2_reference"):
    """NDVI, NDWI, LSWI and a clear mask for every exported date, shaped (dates, rows, cols).

    Bands are looked up by name, because ``clear`` sits at a different position once SWIR is
    included. LSWI is all-NaN when the files carry no B11.
    """
    import rasterio

    from ..optical_export import band_index

    loc = pr.locate(aoi_id, 0)
    paths = sorted(pr.sync_s2(loc, cache_root, folder).glob("*.tif"))
    dates, ndvi, ndwi, lswi, ok = [], [], [], [], []
    for path in paths:
        dates.append(pd.to_datetime(path.stem.rsplit("_S2_", 1)[1]))
        with rasterio.open(path) as ds:
            read = lambda n: ds.read(band_index(ds, n)).astype("float32")  # noqa: E731
            b3, b4, b8, clear = (read(n) for n in ("B3", "B4", "B8", "clear"))
            b11 = read("B11") if "B11" in ds.descriptions else None
        valid = (b8 + b4 > 0) & (b3 + b8 > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi.append(np.where(valid, (b8 - b4) / (b8 + b4), np.nan))
            ndwi.append(np.where(valid, (b3 - b8) / (b3 + b8), np.nan))
            lswi.append(np.full(valid.shape, np.nan, dtype="float32") if b11 is None
                        else np.where(valid & (b8 + b11 > 0), (b8 - b11) / (b8 + b11), np.nan))
        ok.append(valid & (clear >= clear_min))
    order = np.argsort(dates)
    return (pd.DatetimeIndex(np.array(dates)[order]), np.stack(ndvi)[order], np.stack(ndwi)[order],
            np.stack(lswi)[order], np.stack(ok)[order], loc)



def read_windows(aoi_id: int, cache_root="data/s2_windows5d", folder: str = "s2_windows5d",
                 min_obs: int = 1):
    """NDVI, NDWI, LSWI and a validity mask for every fixed window, shaped (windows, rows, cols).

    Validity comes from the ``n_obs`` band: a window is usable at a pixel only where at least
    ``min_obs`` acquisitions survived the cloud mask. The reflectance bands are unmasked to 0 on
    export, so a zero is not evidence of dark ground — the count is the only thing that says whether
    anything was seen.
    """
    import rasterio

    from ..optical_export import band_index

    loc = pr.locate(aoi_id, 0)
    paths = sorted(pr.sync_s2(loc, cache_root, folder).glob("*.tif"))
    dates, ndvi, ndwi, lswi, ok = [], [], [], [], []
    for path in paths:
        dates.append(pd.to_datetime(path.stem.rsplit("_W", 1)[1].split("_", 1)[1]))
        with rasterio.open(path) as ds:
            read = lambda n: ds.read(band_index(ds, n)).astype("float32")  # noqa: E731
            b3, b4, b8, n_obs = (read(n) for n in ("B3", "B4", "B8", "n_obs"))
            b11 = read("B11") if "B11" in ds.descriptions else None
        seen = n_obs >= min_obs
        with np.errstate(invalid="ignore", divide="ignore"):
            ndvi.append(np.where(seen & (b8 + b4 > 0), (b8 - b4) / (b8 + b4), np.nan))
            ndwi.append(np.where(seen & (b3 + b8 > 0), (b3 - b8) / (b3 + b8), np.nan))
            lswi.append(np.full(seen.shape, np.nan, dtype="float32") if b11 is None
                        else np.where(seen & (b8 + b11 > 0), (b8 - b11) / (b8 + b11), np.nan))
        ok.append(seen)
    order = np.argsort(dates)
    return (pd.DatetimeIndex(np.array(dates)[order]), np.stack(ndvi)[order], np.stack(ndwi)[order],
            np.stack(lswi)[order], np.stack(ok)[order], loc)


def blank_long_gaps(values, days, ok, max_gap_days: float = 35):
    """Empty the smoothed curve across stretches nobody observed.

    The smoother will happily draw a line through a two-month hole. A run of unobserved windows
    longer than ``max_gap_days`` is blanked afterwards, so a break in the curve always means a break
    in the data rather than a confident guess.
    """
    values = np.asarray(values, dtype="float64").copy()
    ok = np.asarray(ok, dtype=bool)
    if ok.all() or not ok.any():
        return values if ok.any() else np.full(len(values), np.nan)
    start = None
    for i, seen in enumerate([*ok, True]):
        if not seen and start is None:
            start = i
        elif seen and start is not None:
            first, last = start, i - 1
            lo = days[first - 1] if first > 0 else days[first]
            hi = days[last + 1] if last + 1 < len(days) else days[last]
            if hi - lo > max_gap_days:
                values[first:last + 1] = np.nan
            start = None
    return values


def regrid(days, values, ok, grid_days, max_gap_days: float = 35):
    """Linear interpolation of one pixel's series onto ``grid_days``; wide gaps stay NaN."""
    t, v = days[ok], values[ok]
    if len(t) < 2:
        return np.full(len(grid_days), np.nan)
    out = np.interp(grid_days, t, v, left=np.nan, right=np.nan)
    after = np.searchsorted(t, grid_days, side="left")
    inside = (after > 0) & (after < len(t))
    lo = np.clip(after - 1, 0, len(t) - 1)
    hi = np.clip(after, 0, len(t) - 1)
    gap = np.where(inside, t[hi] - t[lo], np.inf)
    return np.where(np.isin(grid_days, t) | (gap <= max_gap_days), out, np.nan)



def whittaker(y, lmbd: float = 2.0, d: int = 2, dtd=None, weights=None):
    """Smooth and gap-fill one series in a single step (Whittaker, second-difference penalty).

    Why here, when smoothing was rejected for the radar features: NDVI's gaps are *missing* dates,
    not noise. Fitting through them gives an evenly spaced curve whose first and second derivatives
    mean something, and the derivatives are what locate emergence and harvest. Missing samples get
    zero weight, so they are interpolated by the penalty rather than invented from a neighbour.

    ``lmbd`` is deliberately small (2 by default, on a 5-day grid): enough to bridge a cloud gap,
    not enough to round off a harvest. Adapted from the sowing pipeline used on an earlier rice project.

    ``weights`` (same length as ``y``, 0..1) lets a caller trust some observations less than others;
    missing samples get weight 0 whatever is passed. ``ndvi_5day`` uses it for the upper envelope.
    """
    y = np.asarray(y, dtype="float64")
    mask = np.isfinite(y)
    if not mask.any():
        return y.copy()
    n = len(y)
    if dtd is None:
        D = np.diff(np.eye(n), n=d, axis=0)
        dtd = D.T @ D
    w = mask.astype(float) if weights is None else np.where(mask, np.asarray(weights, float), 0.0)
    rhs = np.where(mask, y, 0.0) * w
    A = np.diag(w) + lmbd * dtd
    try:
        z = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        z = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return np.clip(z, -1.0, 1.0)


def acceleration_onset(series, days, lo_idx, hi_idx):
    """Index between ``lo_idx`` and ``hi_idx`` where the rise accelerates hardest.

    The crop emerges where the curve turns upward most sharply, which is the maximum of the second
    derivative. Unlike "NDVI first exceeds x", this needs no level to be chosen in advance.
    """
    if hi_idx - lo_idx < 3:
        return None
    seg, t = series[lo_idx:hi_idx + 1], days[lo_idx:hi_idx + 1]
    if not np.isfinite(seg).all():
        return None
    accel = np.gradient(np.gradient(seg, t), t)
    return lo_idx + int(np.argmax(accel))


def _extreme(values, largest: bool):
    """``nanmax``/``nanmin`` that returns NaN for an all-NaN slice instead of warning.

    A blanked gap can cover a cycle's whole rise or fall, which leaves no rate to report. That is a
    legitimate missing measurement, not something to warn about on every one of thousands of pixels.
    """
    values = np.asarray(values, dtype="float64")
    if not values.size or not np.isfinite(values).any():
        return np.nan
    return float(np.nanmax(values) if largest else np.nanmin(values))


def _cross(series, lo_idx, hi_idx, level, rising: bool):
    """Index of the first sample in [lo_idx, hi_idx) that has passed ``level``."""
    seg = series[lo_idx:hi_idx]
    hit = np.where(seg >= level if rising else seg <= level)[0]
    return None if not len(hit) else lo_idx + int(hit[0])


def _durable_minima(ndvi, days, peaks, minima, min_low_days: float = MIN_LOW_DAYS):
    """Keep only the dips a crop could actually have ended on.

    A harvest leaves the field low for weeks. A dip that drops and recovers within a few days is a
    contaminated observation, and treating it as a cycle boundary cut single crops in half: one pixel
    whose NDVI ran from 0.06 in May to 0.80 in August was reported as a 40-day cycle starting at the
    end of August, because a 0.24 dip in mid-August cleared the prominence test.

    A dip is kept when the curve stays below the midpoint between the dip and its lower neighbouring
    peak for at least ``min_low_days``.
    """
    keep = []
    for m in minima:
        left, right = peaks[peaks < m], peaks[peaks > m]
        shoulders = [ndvi[left[-1]]] if len(left) else []
        shoulders += [ndvi[right[0]]] if len(right) else []
        if not shoulders:
            keep.append(m)
            continue
        level = ndvi[m] + 0.5 * (min(shoulders) - ndvi[m])
        i = j = int(m)
        while i > 0 and ndvi[i - 1] <= level:
            i -= 1
        while j < len(ndvi) - 1 and ndvi[j + 1] <= level:
            j += 1
        if days[j] - days[i] >= min_low_days:
            keep.append(m)
    return np.array(keep, dtype=int)


def _segment(ndvi, days=None, prominence_fraction: float = PROMINENCE_FRACTION,
             min_low_days: float = MIN_LOW_DAYS):
    """Split one pixel's year into cycles at its own local minima. Returns (peaks, minima) indices.

    Prominence is a fraction of the pixel's own NDVI range, so the segmentation does not depend on
    how green the crop got, and no calendar is involved: a cycle is found wherever it happens. Dips
    that do not last are then discarded by :func:`_durable_minima`.
    """
    finite = np.where(np.isfinite(ndvi), ndvi, np.nan)
    if not np.isfinite(finite).any():
        return np.array([], dtype=int), np.array([], dtype=int)
    span = float(np.nanmax(finite) - np.nanmin(finite))
    filled = np.where(np.isfinite(finite), finite, np.nanmin(finite))
    # A flat pixel (open water, bare rock, a constant fill) is not exactly constant once a smoothing
    # fit has run over it: it carries floating-point noise, and a prominence measured as a fraction
    # of that noise finds "peaks" in it. Anything this flat has no season to measure.
    if not np.isfinite(span) or span <= 1e-6:
        return np.array([], dtype=int), np.array([], dtype=int)
    try:
        from scipy.signal import find_peaks
    except ImportError:  # pragma: no cover - scipy ships with the analysis extra
        return np.array([], dtype=int), np.array([], dtype=int)
    peaks, _ = find_peaks(filled, prominence=span * prominence_fraction)
    minima, _ = find_peaks(-filled, prominence=span * prominence_fraction)
    if days is not None and len(minima):
        minima = _durable_minima(filled, np.asarray(days, dtype="float64"), peaks, minima, min_low_days)
    return peaks, minima


def _flat_shoulder(series, lo_idx, hi_idx, take_last: bool, tolerance: float = 0.02):
    """Where a cycle meets a flat trough that carries no local minimum of its own.

    A fallow field sits at its floor for weeks, so the curve either side of a crop can be flat and
    ``find_peaks`` reports no minimum in it. Falling back to the first or last sample of the whole
    series would tie the cycle's length to when the observations happen to start, which is exactly
    the calendar dependence this module avoids. Instead the shoulder is the sample nearest the peak
    that still sits at the segment's own floor (within ``tolerance`` of its own range), so a crop
    sown six weeks later measures the same length.
    """
    seg = series[lo_idx:hi_idx]
    if not len(seg) or not np.isfinite(seg).any():
        return lo_idx
    floor, ceiling = float(np.nanmin(seg)), float(np.nanmax(seg))
    at_floor = np.where(seg <= floor + tolerance * max(ceiling - floor, 1e-9))[0]
    if not len(at_floor):
        return lo_idx
    return lo_idx + int(at_floor[-1] if take_last else at_floor[0])


def _bounds(ndvi, peak_i, minima):
    """The start and end of the cycle around a peak, and whether both were actually observed.

    ``complete`` is False when a shoulder lands on the first or last sample, which means the trough
    that belongs to this crop happened outside the observed period.
    """
    n = len(ndvi)
    before, after = minima[minima < peak_i], minima[minima > peak_i]
    lo = int(before[-1]) if len(before) else _flat_shoulder(ndvi, 0, peak_i, take_last=True)
    hi = int(after[0]) if len(after) else _flat_shoulder(ndvi, peak_i + 1, n, take_last=False)
    return lo, hi, bool(lo > 0 and hi < n - 1)


def pixel_cycles(grid_dates, ndvi, ndwi, lswi=None, prominence_fraction: float = PROMINENCE_FRACTION,
                 water_window_days: int = WATER_WINDOW_DAYS,
                 min_low_days: float = MIN_LOW_DAYS, observed=None) -> list[dict]:
    """Every growth cycle this pixel went through, described by shape, rate and timing.

    One dict per cycle, ordered in time. No verdict is formed and no date is required to fall in any
    particular month: ``start_date`` and ``harvest_date`` are measurements that a later step can
    compare *between* AOIs to see how the season shifts from north to south.

    ``complete`` is False when the cycle runs into the start or the end of the observed series, so a
    crop that was already growing on the first date is not mistaken for a short one.
    """
    days = (grid_dates - grid_dates[0]).days.to_numpy().astype("float64")
    observed = np.isfinite(ndvi) if observed is None else np.asarray(observed, dtype=bool)
    peaks, minima = _segment(ndvi, days, prominence_fraction, min_low_days)
    slope = np.gradient(ndvi, days)
    margin = None if lswi is None else lswi + 0.05 - ndvi
    out = []
    for k, peak_i in enumerate(peaks):
        lo, hi, complete = _bounds(ndvi, peak_i, minima)
        peak, low, high = float(ndvi[peak_i]), float(ndvi[lo]), float(ndvi[hi])
        cycle = {
            "cycle": k, "complete": complete,
            # start_date is the NDVI trough; transplant_date is set below, from the green-up onset.
            "start_date": grid_dates[lo], "start_ndvi": low,
            "peak_date": grid_dates[peak_i], "peak_ndvi": peak,
            "end_date": grid_dates[hi], "end_ndvi": high,
            # Three amplitudes, because they answer different questions and one of them was wrong.
            # "amplitude" (the shallower shoulder) was chosen so a cycle could not be flattered by a
            # deep trough on one side only -- but on a double-cropped field the next crop starts
            # before this one's NDVI has come back down, so the shallow shoulder is the *next* crop
            # and this crop's amplitude is understated. One pixel that is plainly rice by eye
            # measured 0.31 that way against 0.65 of actual growth. So the canopy test uses
            # "growth_amplitude": how far this crop climbed from its own starting trough, which no
            # later crop can touch.
            "amplitude": peak - max(low, high),
            # The trough itself can land inside a blanked gap, which made growth NaN and silently
            # dropped a real rice pixel from the map; the lowest observed point on the way up is the
            # same quantity and survives that.
            "growth_amplitude": peak - _extreme(ndvi[lo:peak_i + 1], largest=False),
            "fall_amplitude": peak - high,
            "rise_days": float(days[peak_i] - days[lo]),
            "fall_days": float(days[hi] - days[peak_i]),
            "max_rise_per_day": _extreme(slope[lo:peak_i + 1], True) if peak_i > lo else np.nan,
            "max_fall_per_day": _extreme(slope[peak_i:hi + 1], False) if hi > peak_i else np.nan,
        }
        onset = acceleration_onset(ndvi, days, lo, peak_i)
        cycle["greenup_onset"] = None if onset is None else grid_dates[onset]
        # The field start (transplanting / direct seeding) is dated from the green-up onset minus
        # the measured lag. The trough is only a fallback: on a field that sits low and flat for
        # weeks before the crop, the trough lands far too early (one pixel: trough 6 October, onset
        # 10 December), and the water test below would then look two months before the puddling.
        tp_idx = lo if onset is None else onset
        cycle["transplant_from"] = "trough" if onset is None else "greenup"
        cycle["transplant_date"] = grid_dates[tp_idx] - pd.Timedelta(days=GREENUP_LAG_DAYS)
        up = _cross(ndvi, lo, peak_i + 1, low + (peak - low) / 2, True)
        down = _cross(ndvi, peak_i, hi + 1, high + (peak - high) / 2, False)
        cycle["greenup_date"] = None if up is None else grid_dates[up]
        cycle["harvest_date"] = None if down is None else grid_dates[down]
        cycle["fwhm_days"] = np.nan if (up is None or down is None) else float(days[down] - days[up])
        cycle["start_to_harvest_days"] = np.nan if down is None else float(days[down] - days[lo])
        cycle["greenup_to_harvest_days"] = (np.nan if (down is None or onset is None)
                                              else float(days[down] - days[onset]))

        # Water at the start of this cycle, in a window measured in days rather than months.
        near = np.abs(days - days[lo]) <= water_window_days
        for name, series in (("ndwi", ndwi), ("lswi", lswi)):
            if series is None or not np.isfinite(series[near]).any():
                cycle[f"{name}_at_start"] = cycle[f"{name}_at_peak"] = cycle[f"{name}_rise"] = np.nan
                continue
            at_start = float(np.nanmax(np.where(near, series, -np.inf)))
            cycle[f"{name}_at_start"] = at_start
            cycle[f"{name}_at_peak"] = float(series[peak_i])
            cycle[f"{name}_rise"] = at_start - float(series[peak_i])
        if margin is not None and np.isfinite(margin[near]).any():
            cycle["flood_margin"] = float(np.nanmax(np.where(near, margin, -np.inf)))
        else:
            cycle["flood_margin"] = np.nan

        # Water at sowing, read the way an analyst reads it off the chart: before the crop starts the
        # field sits at a steady bare-soil level; when it is puddled the NDVI drops *below* that
        # level while LSWI moves the other way. Drying bare soil moves both down together, so it is
        # the opposite directions, not either curve alone, that mean standing water. Both are
        # measured against this pixel's own pre-season baseline, so no absolute level is assumed.
        # Look for the water around *sowing*, not around the trough: the field is puddled before the
        # seedlings go in, so the wettest moment sits about ten days ahead of the NDVI minimum.
        tp_day = days[tp_idx] - GREENUP_LAG_DAYS
        base = (days >= tp_day - BASELINE_FROM) & (days <= tp_day - BASELINE_TO)
        around = np.abs(days - tp_day) <= water_window_days
        cycle["prewet_ndvi_drop"] = np.nan
        cycle["prewet_lswi_rise"] = np.nan
        cycle["growth_from_baseline"] = np.nan
        if base.any() and around.any() and np.isfinite(ndvi[base]).any():
            baseline_ndvi = float(np.nanmedian(ndvi[base]))
            cycle["prewet_ndvi_drop"] = baseline_ndvi - _extreme(ndvi[around], largest=False)
            # How far the crop climbed above the field's own pre-season bare level. Needed because
            # the trough itself can fall inside a cloudy fortnight: there the interpolated "trough"
            # is already half way up the rise and growth measured from it is far too small.
            cycle["growth_from_baseline"] = peak - baseline_ndvi
            if lswi is not None and np.isfinite(lswi[base]).any() and np.isfinite(lswi[around]).any():
                cycle["prewet_lswi_rise"] = (_extreme(lswi[around], largest=True)
                                             - float(np.nanmedian(lswi[base])))
        cycle["wet_at_transplant"] = bool(cycle["prewet_ndvi_drop"] > 0 and cycle["prewet_lswi_rise"] > 0)

        # How much of this cycle was actually seen. The smoother draws a continuous curve across a
        # cloudy fortnight, which is what lets the cycle be detected at all; these two numbers say
        # how much of it is a fit rather than an observation, so the rule can decide instead of the
        # curve silently disappearing.
        seen = observed[lo:hi + 1]
        cycle["observed_fraction"] = float(seen.mean()) if len(seen) else np.nan
        runs, run, prev = [], 0.0, None
        for i in range(lo, hi + 1):
            if observed[i]:
                runs.append(run)
                run, prev = 0.0, None
            else:
                run += (days[i] - days[prev]) if prev is not None else (
                    days[i] - days[i - 1] if i > 0 else 0.0)
                prev = i
        runs.append(run)
        cycle["longest_gap_days"] = float(max(runs) if runs else 0.0)
        out.append(cycle)
    return out


def aoi_cycles(aoi_id: int, step_days: int = 5, clear_min: float = CLEAR_MIN, max_gap_days: float = 35,
               min_clear: int = 15, start="2025-03-01", end="2026-01-31", lmbd: float = LMBD,
               cache_root: str | None = None, folder: str = "s2_windows5d",
               prominence_fraction: float = PROMINENCE_FRACTION) -> pd.DataFrame:
    """Every cycle of every pixel of an AOI, one row per cycle, from Sentinel-2 only.

    ``folder`` selects the export and, through :data:`SOURCES`, how it is read. Window composites
    are already on a regular grid and go straight to the smoother; per-date exports are interpolated
    onto one first, which is the extra step the window export exists to remove.
    """
    source = SOURCES.get(folder, "dates")
    cache_root = cache_root or f"data/{folder}"
    if source == "windows":
        dates, ndvi, ndwi, lswi, ok, loc = read_windows(aoi_id, cache_root, folder)
    else:
        dates, ndvi, ndwi, lswi, ok, loc = read_cube(aoi_id, clear_min, cache_root, folder)
    keep = (dates >= start) & (dates <= end)
    dates, ndvi, ndwi, lswi, ok = dates[keep], ndvi[keep], ndwi[keep], lswi[keep], ok[keep]
    days = (dates - dates[0]).days.to_numpy().astype("float64")
    if source == "windows":
        grid_dates, grid_days = dates, days
    else:
        grid_dates = pd.date_range(dates[0], dates[-1], freq=f"{step_days}D")
        grid_days = (grid_dates - dates[0]).days.to_numpy().astype("float64")

    D = np.diff(np.eye(len(grid_days)), n=2, axis=0)
    dtd = D.T @ D  # built once; it depends only on the grid length
    h, w = ndvi.shape[1:]
    flat = {k: a.reshape(len(dates), -1) for k, a in
            (("ndvi", ndvi), ("ndwi", ndwi), ("lswi", lswi), ("ok", ok))}
    rows = []
    for p in range(h * w):
        good = flat["ok"][:, p]
        if good.sum() < min_clear:
            rows.append({"pid": p, "n_clear": int(good.sum()), "reason": "too few clear windows"})
            continue
        smooth = {}
        for name in ("ndvi", "ndwi", "lswi"):
            column = flat[name][:, p]
            seen = good & np.isfinite(column)
            if not seen.any():
                smooth[name] = None
                continue
            if source == "windows":
                # No blanking: erasing a long gap also erased the whole cycle around it, and a pixel
                # with one cloudy fortnight is not a pixel without a crop. The fit bridges it and
                # each cycle carries observed_fraction and longest_gap_days so the rule can judge.
                smooth[name] = whittaker(np.where(seen, column, np.nan), lmbd, dtd=dtd)
            else:
                smooth[name] = whittaker(regrid(days, column, good, grid_days, max_gap_days),
                                         lmbd, dtd=dtd)
        cycles = pixel_cycles(grid_dates, smooth["ndvi"], smooth["ndwi"], smooth["lswi"],
                              prominence_fraction,
                              observed=good & np.isfinite(flat["ndvi"][:, p]) if source == "windows"
                              else None)
        if not cycles:
            rows.append({"pid": p, "n_clear": int(good.sum()), "reason": "no cycle found"})
        rows += [{"pid": p, "n_clear": int(good.sum()), "reason": "", **c} for c in cycles]
    frame = pd.DataFrame(rows)
    frame["row"], frame["col"] = frame["pid"] // w, frame["pid"] % w
    frame.attrs.update(shape=(h, w), aoi=loc["aoi"], grid=loc["grid"], source=source, folder=folder,
                       lmbd=lmbd)
    return frame


def main_cycle(cycles: pd.DataFrame) -> pd.DataFrame:
    """One row per pixel: the cycle with the largest amplitude, plus how many cycles that pixel had.

    The largest cycle is the pixel's main crop; the count says whether a second one followed it.
    """
    counts = cycles.dropna(subset=["amplitude"]).groupby("pid").size().rename("n_cycles")
    best = (cycles.dropna(subset=["amplitude"]).sort_values("amplitude", ascending=False)
            .groupby("pid", as_index=False).head(1).sort_values("pid").reset_index(drop=True))
    out = best.merge(counts, on="pid", how="left")
    out.attrs.update(cycles.attrs)
    return out


def distributions(frame: pd.DataFrame, columns=None, bins: int = 20) -> pd.DataFrame:
    """Percentiles of each descriptor, for choosing cuts by looking at the data rather than guessing."""
    columns = columns or [c for c in frame.columns
                          if frame[c].dtype.kind == "f" and c not in ("pid", "row", "col")]
    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    out = frame[columns].describe(percentiles=[x / 100 for x in q]).T
    out["n"] = frame[columns].notna().sum()
    return out


def acres(n_pixels: int, res_m: float = 10.0) -> float:
    """Pixel count to acres (1 ha = 2.47105 acres); areas in this project are always acres."""
    return n_pixels * res_m * res_m / 10_000.0 * 2.47105


def write_map(frame: pd.DataFrame, values, out_path, dtype="float32", nodata=-9999.0):
    """Write any per-pixel column back onto the AOI grid as a GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin

    shape, grid = frame.attrs["shape"], frame.attrs["grid"]
    band = np.full(shape[0] * shape[1], nodata, dtype=dtype)
    v = np.asarray(values, dtype="float64")
    band[frame["pid"].to_numpy()[np.isfinite(v)]] = v[np.isfinite(v)].astype(dtype)
    profile = dict(driver="GTiff", width=shape[1], height=shape[0], count=1, dtype=dtype,
                   crs=grid["crs"], nodata=nodata, compress="deflate", tiled=True,
                   transform=from_origin(grid["x0"], grid["y0"], grid["res"], grid["res"]))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(band.reshape(shape), 1)
    return out_path
