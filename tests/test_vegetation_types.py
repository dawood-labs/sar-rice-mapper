"""Fresh start, step 1: vegetation that went bare since 1 May vs vegetation that never did."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import vegetation_types as vt

WIN = pd.date_range("2026-04-01", "2026-09-26", freq="5D")


def test_sustained_low_ignores_a_single_hazy_window_and_the_time_before_the_start():
    tree = np.full(len(WIN), 0.8)
    tree[20] = 0.3                                     # one hazy view
    crop = np.full(len(WIN), 0.8)
    crop[15:25] = 0.15                                 # bare for ten windows
    early = np.full(len(WIN), 0.8)
    early[:4] = 0.1                                    # bare only in April, before the start
    low = vt.sustained_low(np.stack([tree, crop, early], axis=1), WIN)
    assert low[0] == 0.8 and abs(low[1] - 0.15) < 1e-6 and low[2] == 0.8


def test_second_darkest_needs_two_passes_and_ignores_one_speckle():
    day = pd.date_range("2026-05-05", periods=6, freq="12D")
    vh = np.full((6, 2), -14.0)
    vh[2, 0] = -25.0                                   # one speckled pass
    vh[1:3, 1] = -24.0                                 # a flood of two passes
    assert vt.second_darkest([(day, {"VH": vh})]).tolist() == [-14.0, -24.0]


def test_cover_classes_split_by_this_aois_own_values():
    rng = np.random.default_rng(0)
    low = np.r_[rng.normal(0.15, 0.05, 700), rng.normal(0.7, 0.05, 300), 0.1]
    veg = np.r_[np.ones(1000, bool), False]
    out, split = vt.cover_classes(low, veg, np.ones(1001, bool))
    assert 0.3 < split < 0.55
    assert (out[:700] == 1).mean() > 0.98 and (out[700:1000] == 2).mean() > 0.98 and out[1000] == 0


def test_swing_is_large_for_a_crop_and_small_for_a_tree():
    day = pd.date_range("2026-05-05", periods=12, freq="12D")
    vh = np.full((12, 2), -14.5)
    vh[:, 1] = np.r_[-21, -22, -21, -20, -18, -17, -16, -15, -14.5, -14, -14, -14.2]    # crop
    vh[3, 0] = -22.0                                                                    # one speckled pass on the tree
    sw = vt.swing([(day, {"VH": vh})])
    assert sw[0] < 1.0 and sw[1] > 6.0


def test_ratio_low_is_low_for_bare_soil_and_high_for_a_canopy():
    day = pd.date_range("2026-05-05", periods=8, freq="12D")
    vv = np.full((8, 2), -8.0)
    vh = np.full((8, 2), -14.0)                         # tree: VH - VV = -6 all season
    vh[:3, 1] = -19.0                                   # crop: dry bare soil in May-June, VH - VV = -11
    rl = vt.ratio_low([(day, {"VH": vh, "VV": vv})])
    assert rl[0] == -6.0 and rl[1] == -11.0


def test_water_and_bare_soil_fall_on_the_same_side_of_the_split():
    rng = np.random.default_rng(1)
    low = np.r_[rng.normal(-0.1, 0.03, 500), rng.normal(0.1, 0.03, 500), rng.normal(0.6, 0.05, 150)]
    out, split = vt.cover_classes(low, np.ones(len(low), bool), np.ones(len(low), bool))
    assert 0.25 < split < 0.5
    assert (out[:1000] == 1).mean() > 0.99 and (out[1000:] == 2).mean() > 0.95


def test_the_split_never_falls_below_the_ground_bare_level():
    rng = np.random.default_rng(2)
    low = np.r_[rng.normal(-0.1, 0.03, 500), rng.normal(0.12, 0.03, 500)]       # fields only: water and soil
    out, split = vt.cover_classes(low, np.ones(len(low), bool), np.ones(len(low), bool), floor=0.3)
    assert split == 0.3 and (out == 1).all()


def test_a_never_bare_pixel_whose_radar_went_dark_is_crop_ground():
    rng = np.random.default_rng(3)
    vh_low = np.r_[rng.normal(-21, 0.8, 300), rng.normal(-15, 0.8, 100), -22.0]
    cover = np.r_[np.ones(300), np.full(100, 2), 2].astype("uint8")      # last: optical "tree", radar flooded
    out, split = vt.radar_went_bare(cover, vh_low, np.ones(401, bool))
    assert -20 < split < -16 and out[-1] == 1 and (out[300:400] == 2).mean() > 0.98
