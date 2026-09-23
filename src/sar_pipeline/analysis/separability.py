"""Can the radar tell the two classes apart at all, before any model is fitted.

Why
---
A classifier can be trained on anything and will report a number. That number says how well the
model reproduces its labels, not whether the sensor carries the information — and after the phase
error found in Phase 4d, that distinction is the whole question. So before any model: take the
optical map as the labels, and measure, feature by feature and **date by date**, how far apart the
two classes actually are in the radar data.

The measure is the **AUC** of a single feature: the probability that a randomly chosen rice pixel
has a higher value than a randomly chosen non-rice pixel. It needs no threshold, no training and no
assumption about the distributions; 0.5 means the feature carries nothing, and a value below 0.5 is
just as informative as one above (the classes are separated the other way round), so
``separation = |AUC - 0.5| * 2`` is reported alongside.

Reading the result
------------------
A curve of AUC against date says *when* in the season the radar sees a difference, which is the
thing the calendar-anchored draft model got wrong. If no date and no summary feature gets past about
0.7, the honest conclusion is that this radar data cannot make this map on its own, and that is a
result worth having rather than a model to be tuned.

The labels come from the optical map, which is itself a measurement and not ground truth, so every
number here is agreement with that map — never accuracy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def auc(values, labels) -> float:
    """Probability a positive outranks a negative, from ranks. NaNs are dropped pairwise."""
    values = np.asarray(values, dtype="float64")
    labels = np.asarray(labels, dtype=bool)
    ok = np.isfinite(values)
    values, labels = values[ok], labels[ok]
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype="float64")
    ranks[order] = np.arange(1, len(values) + 1)
    # average the ranks of ties, or a feature with many equal values scores spuriously well
    sorted_values = values[order]
    start = 0
    for i in range(1, len(sorted_values) + 1):
        if i == len(sorted_values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + i + 1) / 2
            start = i
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def separation(value: float) -> float:
    """AUC rescaled to 0 (nothing) .. 1 (perfect), whichever way round the classes sit."""
    return float("nan") if not np.isfinite(value) else abs(value - 0.5) * 2


def per_step(dates, cubes: dict, labels) -> pd.DataFrame:
    """AUC of every feature at every time step. ``cubes`` maps a name to (steps, pixels)."""
    rows = []
    for k, when in enumerate(dates):
        row = {"date": when}
        for name, cube in cubes.items():
            row[name] = auc(cube[k], labels)
        rows.append(row)
    frame = pd.DataFrame(rows)
    for name in cubes:
        frame[f"{name}_sep"] = frame[name].map(separation)
    return frame


def class_curves(dates, cubes: dict, labels, low: float = 25, high: float = 75) -> pd.DataFrame:
    """Median and an inter-quantile band per class per step, for looking at the shapes."""
    labels = np.asarray(labels, dtype=bool)
    rows = []
    for k, when in enumerate(dates):
        row = {"date": when}
        for name, cube in cubes.items():
            for which, mask in (("rice", labels), ("other", ~labels)):
                values = cube[k][mask]
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                row[f"{name}_{which}_med"] = float(np.median(values))
                row[f"{name}_{which}_lo"] = float(np.percentile(values, low))
                row[f"{name}_{which}_hi"] = float(np.percentile(values, high))
        rows.append(row)
    return pd.DataFrame(rows)


def summary_features(cubes: dict) -> dict:
    """Whole-season summaries of each curve: the features a classifier would actually be given."""
    out = {}
    for name, cube in cubes.items():
        with np.errstate(invalid="ignore"):
            out[f"{name}_min"] = np.nanmin(cube, axis=0)
            out[f"{name}_max"] = np.nanmax(cube, axis=0)
            out[f"{name}_mean"] = np.nanmean(cube, axis=0)
            out[f"{name}_range"] = out[f"{name}_max"] - out[f"{name}_min"]
            out[f"{name}_std"] = np.nanstd(cube, axis=0)
    return out


def rank_features(features: dict, labels) -> pd.DataFrame:
    """Every scalar feature ranked by how far apart it puts the two classes."""
    rows = [{"feature": name, "auc": auc(values, labels)} for name, values in features.items()]
    frame = pd.DataFrame(rows)
    frame["separation"] = frame["auc"].map(separation)
    return frame.sort_values("separation", ascending=False).reset_index(drop=True)


def align_cube(cube, dates, event_dates, rel_days, step_days: int = 5):
    """Re-cut each pixel's curve so that column *k* is ``rel_days[k]`` days after **its own** event.

    Calendar alignment is what the draft model got wrong: fields sown five weeks apart have their
    canopy five weeks apart, and averaging them on a calendar smears both. Aligning on each pixel's
    own sowing date is also the only form the operational question takes — in a season still under
    way, "how long since this field was sown" is known and "what month is it" is not useful.

    ``cube`` is (steps, pixels) on a regular ``step_days`` grid starting at ``dates[0]``. Columns
    that would fall outside the observed period come back as NaN.
    """
    cube = np.asarray(cube, dtype="float64")
    n_steps, n_pixels = cube.shape
    start = np.asarray(dates[0], dtype="datetime64[D]")
    event = np.asarray(pd.to_datetime(event_dates).values, dtype="datetime64[D]")
    base = ((event - start).astype("int64") / step_days).round().astype("int64")
    out = np.full((len(rel_days), n_pixels), np.nan)
    columns = np.arange(n_pixels)
    for k, offset in enumerate(rel_days):
        index = base + int(round(offset / step_days))
        inside = (index >= 0) & (index < n_steps) & np.isfinite(base)
        out[k, inside] = cube[index[inside], columns[inside]]
    return out


def earliest_call(steps: pd.DataFrame, feature: str, target: float = 0.7):
    """First relative day at which a feature's separation reaches ``target``, or None."""
    hit = steps[steps[f"{feature}_sep"] >= target]
    return None if hit.empty else hit.iloc[0]
