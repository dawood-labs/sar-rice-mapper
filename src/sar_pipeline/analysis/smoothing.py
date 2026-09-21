"""Temporal smoothing of a per-pixel backscatter series, and the tools to decide whether to use it.

Why this module is cautious about smoothing
-------------------------------------------
Smoothing a SAR time series looks like an obvious win: speckle is noisy, curves look ragged, and a
Whittaker-Henderson smoother produces something far prettier. For **paddy rice it is a trap**, and
the reason is worth stating once, here, so nobody re-discovers it the hard way.

Rice is identified from three statistics of the VH series over a season: its **minimum** (the
flooded field, which reflects the beam away and collapses backscatter), its **maximum** (dense
canopy) and its **variance** (the size of that swing). The Whittaker penalty of order ``d = 2``
penalises the **second difference**, i.e. curvature — and the flooding minimum *is* a curvature
extremum. So the smoother attacks precisely the three features the classification depends on: it
fills in the minimum, shaves the maximum, and shrinks the variance.

That does not make smoothing useless. It makes it useful for different things:

* **extracting a phenology date** (when is the minimum?) — noise can move the argmin by a whole
  time bin, and a smoother stabilises *where* the minimum is even while biasing *how deep* it is;
  published rice calendars fit a smooth function for exactly this purpose;
* **filling gaps** — :func:`whittaker` accepts weights, so an unobserved bin can be given
  ``w = 0`` and reconstructed from its neighbours. That is strictly better than linear
  interpolation and, with a small ``lam``, it barely touches the observed points;
* **plotting** for a human reader.

So: **measure the features on the unsmoothed series, and use the smoothed one only for dates, gap
filling and pictures.** :func:`feature_impact` exists to keep that decision evidence-based on your
own data rather than inherited from this docstring.

Which domain to smooth in
-------------------------
Speckle is *multiplicative*, which is why the rest of the pipeline averages in **linear power** and
converts to dB at the very end (averaging dB biases the mean low). The same applies here: smooth
linear power, then convert. :func:`whittaker` does not know which domain it was handed, so this is
the caller's responsibility.

Unequal spacing
---------------
The classical Whittaker smoother assumes **equally spaced** samples. A Sentinel-1 series is not:
the revisit is 6 or 12 days depending on how many satellites fly the track, and audits routinely
report 18–24 day gaps. Feeding such a series to the equal-spacing version tells the penalty that a
6-day step and a 24-day step are the same thing, which distorts the result exactly where the data
is thinnest. :func:`whittaker` therefore takes an optional ``t`` and builds a spacing-aware second
derivative operator instead.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

#: Var(y[i-1] - 2 y[i] + y[i+1]) = (1 + 4 + 1) sigma^2 for white noise of variance sigma^2.
_SECOND_DIFF_GAIN = 6.0

#: MAD -> sigma for a normal distribution (1 / 0.6744897501960817).
_MAD_TO_SIGMA = 1.4826022185056018


def difference_matrix(n: int, d: int = 2) -> sp.csc_matrix:
    """The ``d``-th order difference operator for ``n`` equally spaced samples, shape ``(n-d, n)``."""
    if n <= d:
        raise ValueError(f"need more than d={d} samples, got {n}")
    D = sp.eye(n, format="csc")
    for _ in range(d):
        D = D[1:] - D[:-1]
    return D.tocsc()


def curvature_matrix(t: np.ndarray) -> sp.csc_matrix:
    """Spacing-aware second-derivative operator for samples at times ``t``, shape ``(n-2, n)``.

    Row ``i`` is the standard three-point second derivative on an uneven grid::

        2 / (h1 + h2) * [ (z[i+1] - z[i]) / h2  -  (z[i] - z[i-1]) / h1 ]

    with ``h1 = t[i] - t[i-1]`` and ``h2 = t[i+1] - t[i]``. On an even grid this reduces to the
    usual ``[1, -2, 1] / h^2``, so the two operators agree where the spacing is regular and differ
    only across gaps — which is the point.
    """
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 3:
        raise ValueError(f"need at least 3 sample times, got shape {t.shape}")
    h = np.diff(t)
    if np.any(h <= 0):
        raise ValueError("sample times must be strictly increasing")

    n = t.size
    h1, h2 = h[:-1], h[1:]
    scale = 2.0 / (h1 + h2)
    left = scale / h1
    right = scale / h2
    rows = np.repeat(np.arange(n - 2), 3)
    cols = np.concatenate([[i, i + 1, i + 2] for i in range(n - 2)])
    vals = np.column_stack([left, -(left + right), right]).ravel()
    return sp.csc_matrix((vals, (rows, cols)), shape=(n - 2, n))


def whittaker(y, lam: float, d: int = 2, w=None, t=None) -> np.ndarray:
    """Whittaker-Henderson smoother.

    Minimises ``sum(w * (y - z)^2) + lam * ||D z||^2`` and returns ``z``.

    Parameters
    ----------
    y : array
        The series to smooth. Smooth **linear power**, not dB (see the module docstring).
    lam : float
        Roughness penalty. Larger is smoother. ``lam`` is not comparable across series of
        different length or spacing, so tune it on your own data; do not carry a value over from
        another project.
    d : int
        Penalty order, used only when ``t`` is None. ``d = 2`` penalises curvature.
    w : array, optional
        Per-sample weights. Use ``w = 0`` for an unobserved sample to have it interpolated, and
        ``w = 1`` for an observation. Defaults to all ones.
    t : array, optional
        Sample times (any consistent unit, e.g. days since the season start). When given, the
        spacing-aware second-derivative operator is used instead of the equally spaced ``d``-th
        difference, which matters on a series with 6-day, 12-day and 24-day steps mixed together.

    Notes
    -----
    ``NaN`` in ``y`` is treated as unobserved: its weight is forced to zero and its value is
    reconstructed, so callers do not have to pre-fill gaps.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    n = y.size

    weights = np.ones(n) if w is None else np.asarray(w, dtype=float).copy()
    if weights.shape != y.shape:
        raise ValueError("w must have the same shape as y")
    missing = ~np.isfinite(y)
    weights[missing] = 0.0
    if not np.any(weights > 0):
        raise ValueError("every sample is unobserved or zero-weighted")
    filled = np.where(missing, 0.0, y)

    D = difference_matrix(n, d) if t is None else curvature_matrix(t)
    A = (sp.diags(weights) + lam * (D.T @ D)).tocsc()
    return spl.spsolve(A, weights * filled)


def noise_sigma(y) -> float:
    """Robustly estimate the white-noise standard deviation of a series from its second differences.

    For a series that is smooth plus white noise, the second difference of the smooth part is small
    while the noise contributes a variance of ``6 * sigma^2``. Using the **median absolute
    deviation** rather than the standard deviation keeps a handful of genuinely sharp transitions —
    a flooding collapse, a harvest drop — from being counted as noise, which a plain
    ``std(diff(y, 2))`` would do.

    Returns ``nan`` when fewer than three finite samples are available.
    """
    y = np.asarray(y, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size < 3:
        return float("nan")
    second = np.diff(finite, n=2)
    mad = np.median(np.abs(second - np.median(second)))
    return float(_MAD_TO_SIGMA * mad / np.sqrt(_SECOND_DIFF_GAIN))


def series_features(y) -> dict:
    """The three statistics paddy-rice detection relies on, plus the peak-to-trough depth."""
    y = np.asarray(y, dtype=float)
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "var": float("nan"), "depth": float("nan"),
                "argmin": -1}
    return {"min": float(finite.min()), "max": float(finite.max()), "var": float(finite.var()),
            "depth": float(finite.max() - finite.min()), "argmin": int(np.nanargmin(y))}


def feature_impact(y, lams, d: int = 2, t=None) -> list[dict]:
    """Measure what each ``lam`` does to the rice features of one series.

    Returns one row per ``lam`` with the smoothed features, the shift in each of them, and the
    fraction of variance and peak-to-trough depth that survives. Use it to decide whether
    smoothing is acceptable **on your own data** before it touches a feature table.
    """
    raw = series_features(y)
    rows = []
    for lam in lams:
        smoothed = series_features(whittaker(y, lam, d=d, t=t))
        rows.append({
            "lam": float(lam),
            "min": smoothed["min"], "min_shift": smoothed["min"] - raw["min"],
            "max": smoothed["max"], "max_shift": smoothed["max"] - raw["max"],
            "var_kept": smoothed["var"] / raw["var"] if raw["var"] else float("nan"),
            "depth_kept": smoothed["depth"] / raw["depth"] if raw["depth"] else float("nan"),
            "argmin_shift": smoothed["argmin"] - raw["argmin"],
        })
    return rows


# --------------------------------------------------------------------------- plotting
def plot_comparison(dates, panels, out_path, lam=None, title=None, subtitle=None):
    """Small multiples of raw vs spatially averaged vs smoothed series, one panel per location.

    ``panels`` is a sequence of ``(label, single_pixel_db, window_mean_db, smoothed_db)``; pass
    ``None`` for a series to leave it out of a panel. All three share the same ``dates``.

    This is a thin convenience wrapper over :func:`sar_pipeline.analysis.curves.plot_panels` that
    fixes the three series names and marks the noisy single-pixel trace as recessive, because this
    exact figure is the one used to decide whether smoothing is worth its cost.
    """
    from . import curves

    names = ("single pixel", "5x5 mean (raw)",
             "5x5 mean + Whittaker" + (f" (lam={lam:.0e})" if lam is not None else ""))
    built = []
    for label, *series in panels:
        entries = [(dates, values, names[i], i == 0)
                   for i, values in enumerate(series) if values is not None]
        built.append((label, entries))
    return curves.plot_panels(built, out_path, title=title, subtitle=subtitle)
