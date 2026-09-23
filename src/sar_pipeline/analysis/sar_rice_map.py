"""A rice map from radar alone, built the way an agronomist described it: two stages, in order.

Why two stages and not one classifier
--------------------------------------
Asked plainly, the question is: *is there a crop here*, and *is it on the monsoon calendar*. Those
are two different physical questions with two different best moments in the year, and answering them
separately keeps both the reasoning and the failure visible:

**Stage 1 — seasonal cropland or not.** Read in the **dry season**, when the monsoon has stopped
making every surface look alike. The direction is the opposite of the obvious one, and getting it
wrong the first time cost a whole run: in this delta the land that carries a crop cycle is
**harvested, bare and smooth** from November to January, so its VH is *low*, while the pixels that
stay high are trees, settlements and permanent rough cover — which is exactly the class with no crop
cycle. So stage 1 keeps the **low** dry-season VH, and what it really separates is *seasonal
cropland* from *permanent cover*, not "vegetation" from "bare". Against the optical canopy mask on
the first AOI that agrees 90.7% with a kappa of 0.73, and it picks 11,173 pixels where the optical
finds 11,169.

**Stage 2 — on the monsoon calendar or not.** Read around the AOI's **own sowing window**. A paddy
field at transplanting is flat, smooth and wet and its VH collapses; vegetation that is not on that
calendar does not. Mid-June VH separated in-season from out-of-season vegetation at 0.90.

Each stage's *window* comes from agronomy and each stage's *threshold* comes from the data, by
Otsu's method on that AOI's own histogram — so nothing is carried between AOIs except the reasoning.

What this does not do
---------------------
The sowing window is still taken from the optical measurement. A radar-only pipeline has to find it
for itself, which is a separate thing to test. And the labels this is scored against are the optical
map, which is a measurement and not ground truth, so every number here is **agreement with that
map**, never accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optical_rice_map import acres, otsu  # noqa: F401  (acres re-exported for callers)

#: Months read for stage 1. After the monsoon, standing vegetation and bare ground stop looking alike.
DRY_MONTHS = (11, 12, 1)
#: Half-width, in days, of the sowing window read for stage 2.
SOWING_HALF_DAYS = 15


def window_level(curves, dates, mask, reducer=np.nanmedian) -> np.ndarray:
    """Summarise a curve over the selected time steps, per pixel."""
    selected = np.asarray(curves)[np.asarray(mask, dtype=bool)]
    if not selected.size:
        raise ValueError("no time step falls inside that window")
    with np.errstate(invalid="ignore"):
        return reducer(selected, axis=0)


def dry_season_level(vh, dates, months=DRY_MONTHS) -> np.ndarray:
    """Stage 1's feature: the dry-season VH level of every pixel."""
    return window_level(vh, dates, pd.DatetimeIndex(dates).month.isin(months))


def sowing_level(vh, dates, centre, half_days: int = SOWING_HALF_DAYS) -> np.ndarray:
    """Stage 2's feature: the VH level around the AOI's own sowing window."""
    dates = pd.DatetimeIndex(dates)
    offset = (dates - pd.Timestamp(centre)).days
    return window_level(vh, dates, np.abs(offset) <= half_days)


def window_mask(dates, centre, lo_days: float, hi_days: float):
    """Steps between ``lo_days`` and ``hi_days`` of a centre date, both inclusive."""
    offset = (pd.DatetimeIndex(dates) - pd.Timestamp(centre)).days
    return (offset >= lo_days) & (offset <= hi_days)


def sowing_min(vh, dates, centre, lo_days: float = -15, hi_days: float = 25) -> np.ndarray:
    """The lowest VH a pixel reaches around sowing.

    A median over a window answers "how bright is this field at that time"; the minimum answers "did
    it ever go flat and wet". Transplanting and puddling last days, not weeks, and the exact day
    moves from field to field, so the level can miss what the minimum catches.
    """
    return window_level(np.asarray(vh), dates, window_mask(dates, centre, lo_days, hi_days),
                        reducer=np.nanmin)


def sowing_drop(vh, dates, centre, lo_days: float = -15, hi_days: float = 25,
                baseline_from: float = 75, baseline_to: float = 25) -> np.ndarray:
    """How far the sowing minimum sits below the field's **own** dry baseline, in dB.

    This is the change form of the same question, and it removes the part of the signal that is just
    what kind of land this is: a permanently smooth field is low all year and scores nothing here.
    """
    vh = np.asarray(vh)
    base = window_level(vh, dates, window_mask(dates, centre, -baseline_from, -baseline_to))
    return base - sowing_min(vh, dates, centre, lo_days, hi_days)


def growth_change(vh, vv, dates, centre, lo_days: float = 20, hi_days: float = 60) -> np.ndarray:
    """Rise of VH - VV between two points after sowing: the canopy filling in, as a change.

    Measured on the first AOI this separated rice from the rest at 0.72 as a change feature, while
    the same quantity as a level barely separated at all.
    """
    ratio = np.asarray(vh) - np.asarray(vv)
    late = window_level(ratio, dates, window_mask(dates, centre, hi_days - 5, hi_days + 5))
    early = window_level(ratio, dates, window_mask(dates, centre, lo_days - 5, lo_days + 5))
    return late - early


def split_high(values, cut: float | None = None, above: bool = True):
    """Threshold a feature at an Otsu cut chosen from its own histogram. Returns ``(mask, cut)``."""
    cut = otsu(values) if cut is None else float(cut)
    finite = np.isfinite(values)
    mask = np.zeros(len(values), dtype=bool)
    mask[finite] = (values[finite] >= cut) if above else (values[finite] <= cut)
    return mask, cut


def classify(vh, dates, sowing_centre, dry_months=DRY_MONTHS, half_days: int = SOWING_HALF_DAYS,
             veg_cut: float | None = None, season_cut: float | None = None,
             cropland_is_low: bool = True, flooded_is_low: bool = True):
    """Radar-only rice mask, in two stages. Returns ``(rice, parts)``.

    Stage 1 keeps the pixels whose dry-season VH is **low** — harvested cropland is bare and smooth
    then, while permanent cover stays rough. Stage 2 keeps, among those, the pixels whose VH around
    sowing is also **low** — a puddled, flat field is a weak scatterer at exactly that moment.

    Both directions are exposed rather than assumed, because the first of them was assumed and was
    backwards. A direction is a physical claim about the landscape and should be checked per region,
    not inherited.
    """
    dry = dry_season_level(vh, dates, dry_months)
    sow = sowing_level(vh, dates, sowing_centre, half_days)
    vegetation, veg_cut = split_high(dry, veg_cut, above=not cropland_is_low)
    flooded, season_cut = split_high(np.where(vegetation, sow, np.nan), season_cut,
                                     above=not flooded_is_low)
    rice = vegetation & flooded
    return rice, {"dry_season_vh": dry, "sowing_vh": sow, "vegetation": vegetation,
                  "veg_cut": veg_cut, "season_cut": season_cut}


def agreement(predicted, reference, valid=None) -> dict:
    """Confusion of a predicted mask against a reference mask, over ``valid`` pixels only."""
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    valid = np.ones(len(predicted), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    p, r = predicted[valid], reference[valid]
    tp, fp = int((p & r).sum()), int((p & ~r).sum())
    fn, tn = int((~p & r).sum()), int((~p & ~r).sum())
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    observed = (tp + tn) / total if total else float("nan")
    expected = (((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / total ** 2) if total else float("nan")
    return {"n": total, "agreement": observed, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else float("nan"),
            "kappa": (observed - expected) / (1 - expected) if expected < 1 else float("nan"),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def split_halves(shape, axis: int = 1):
    """Two spatial halves of a grid, flattened — for fitting a cut on one and testing on the other.

    A random pixel split would put neighbouring pixels on both sides; they are strongly correlated,
    and the score would flatter the method. Splitting the AOI in space does not.
    """
    rows, cols = shape
    index = np.arange(rows * cols)
    coordinate = (index % cols) if axis == 1 else (index // cols)
    first = coordinate < (cols if axis == 1 else rows) / 2
    return first, ~first
