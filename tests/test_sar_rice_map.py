"""Tests for the two-stage radar rice map. Synthetic arrays only."""
import numpy as np
import pandas as pd
import pytest

from sar_pipeline.analysis import sar_rice_map as srm


def grid(n=60, start="2025-05-01"):
    return pd.date_range(start, periods=n, freq="5D")


def test_dry_season_level_reads_only_the_dry_months():
    dates = grid(60)
    vh = np.zeros((60, 2))
    vh[pd.DatetimeIndex(dates).month.isin(srm.DRY_MONTHS)] = 7.0
    assert np.allclose(srm.dry_season_level(vh, dates), 7.0)


def test_sowing_level_reads_only_around_the_given_centre():
    dates = grid(60)
    vh = np.zeros((60, 2))
    centre = dates[10]
    inside = np.abs((dates - centre).days) <= srm.SOWING_HALF_DAYS
    vh[inside] = -19.0
    assert np.allclose(srm.sowing_level(vh, dates, centre), -19.0)


def test_window_level_refuses_an_empty_window():
    with pytest.raises(ValueError, match="no time step"):
        srm.window_level(np.zeros((5, 2)), grid(5), np.zeros(5, dtype=bool))


def test_split_high_finds_its_own_cut_and_can_go_either_way():
    values = np.concatenate([np.full(200, -19.0), np.full(200, -12.0)])
    high, cut = srm.split_high(values)
    low, _ = srm.split_high(values, above=False)
    assert -19.0 < cut < -12.0
    assert high.sum() == 200 and low.sum() == 200
    assert not (high & low).any()


def test_classify_keeps_only_vegetation_that_is_also_low_at_sowing():
    dates = grid(60)
    centre = dates[6]
    n = 4
    vh = np.zeros((60, n))
    dry = pd.DatetimeIndex(dates).month.isin(srm.DRY_MONTHS)
    sow = np.abs((dates - centre).days) <= srm.SOWING_HALF_DAYS
    # cropland is LOW in the dry season (harvested, smooth); permanent cover stays high.
    # 0: cropland + flooded (rice)  1: cropland, dry at sowing  2/3: permanent cover
    vh[dry] = [-19.0, -19.0, -11.0, -11.0]
    vh[sow] = [-20.0, -13.0, -20.0, -13.0]
    rice, parts = srm.classify(vh, dates, centre)
    assert list(rice) == [True, False, False, False]
    assert list(parts["vegetation"]) == [True, True, False, False]


def test_agreement_counts_a_perfect_prediction():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    out = srm.agreement(truth, truth)
    assert out["agreement"] == 1.0 and out["kappa"] == 1.0 and out["fp"] == 0


def test_agreement_is_not_fooled_by_predicting_everything():
    truth = np.array([True] * 90 + [False] * 10)
    everything = np.ones(100, dtype=bool)
    out = srm.agreement(everything, truth)
    assert out["agreement"] == 0.9
    assert out["kappa"] == 0.0            # no better than guessing the majority
    assert out["recall"] == 1.0 and out["precision"] == 0.9


def test_agreement_respects_the_valid_mask():
    truth = np.array([True, True, False, False])
    predicted = np.array([True, False, False, False])
    out = srm.agreement(predicted, truth, valid=np.array([True, False, True, True]))
    assert out["n"] == 3 and out["agreement"] == 1.0


def test_split_halves_cuts_the_grid_in_space_not_at_random():
    left, right = srm.split_halves((4, 6))
    assert left.sum() == right.sum() == 12
    assert not (left & right).any()
    assert left.reshape(4, 6)[:, :3].all() and not left.reshape(4, 6)[:, 3:].any()


def test_both_directions_are_switchable_because_one_of_them_was_wrong():
    """The dry-season direction is a claim about the landscape, not a law; it must be checkable."""
    dates = grid(60)
    centre = dates[6]
    vh = np.zeros((60, 2))
    dry = pd.DatetimeIndex(dates).month.isin(srm.DRY_MONTHS)
    sow = np.abs((dates - centre).days) <= srm.SOWING_HALF_DAYS
    vh = np.zeros((60, 4))
    vh[dry] = [-19.0, -19.5, -11.0, -11.5]
    vh[sow] = [-20.0, -13.0, -20.0, -13.0]
    _, low = srm.classify(vh, dates, centre, cropland_is_low=True)
    _, high = srm.classify(vh, dates, centre, cropland_is_low=False)
    assert list(low["vegetation"]) == [True, True, False, False]
    assert list(high["vegetation"]) == [False, False, True, True]
