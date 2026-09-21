"""Tests for the unsupervised pattern search. Synthetic and offline."""
from __future__ import annotations

import datetime as dt

import numpy as np

from sar_pipeline.analysis import patterns as pt
from sar_pipeline.analysis import seasonal_stats as ss


def test_half_month_edges_cover_a_year_in_24_bins():
    bins = pt.half_month_edges("2025-05-01", "2026-05-01")
    assert bins.size == 24
    assert len(set(bins)) == 24


def test_half_month_edges_use_the_same_labels_as_the_binning():
    """If these drifted apart, every pixel would silently land in the wrong bin."""
    dates = [dt.date(2025, 5, 3), dt.date(2025, 5, 20), dt.date(2025, 12, 31), dt.date(2026, 1, 2)]
    labels = set(ss.half_month_bins(dates))
    assert labels <= set(pt.half_month_edges("2025-05-01", "2026-05-01"))


def test_fill_gaps_interpolates_inside_and_holds_the_ends():
    out = pt.fill_gaps(np.array([[np.nan, 1.0, np.nan, 3.0, np.nan]]))
    assert np.allclose(out, [[1.0, 1.0, 2.0, 3.0, 3.0]])


def test_cluster_separates_two_shapes_and_numbers_the_larger_first():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 24)
    crop = -18 + 7 * np.sin(t) + rng.normal(0, 0.5, (300, 24))
    flat = -13 + rng.normal(0, 0.5, (100, 24))
    labels, centres = pt.cluster(np.vstack([crop, flat]).astype("float32"), k=2)
    assert (labels[:300] == 0).all() and (labels[300:] == 1).all()
    assert centres.shape == (2, 24)


def test_cluster_table_rows_sum_to_100():
    table = pt.cluster_table(np.array([0, 0, 1, 1, 1]), np.array(["a", "a", "a", "b", "b"]))
    assert np.allclose(table.sum(axis=1), 100.0)
