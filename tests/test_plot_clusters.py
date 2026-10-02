"""Grouping the surveyed plots by their NDVI curve."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import plot_clusters as pc


def test_cluster_separates_distinct_curves_and_orders_by_trough():
    rng = np.random.default_rng(0)
    t = np.arange(40)
    early = np.where(t < 10, 0.15, 0.8)                       # trough early
    late = np.where(t < 30, 0.7, 0.15)                        # low only at the end
    x = np.vstack([early + rng.normal(0, 0.02, (30, 40)), late + rng.normal(0, 0.02, (30, 40))])
    x[0, 5] = np.nan                                          # a gap: no group
    labels = pc.cluster(pd.DataFrame(x), k=2)
    assert labels[0] == -1
    assert set(labels[1:30]) == {1} and set(labels[30:]) == {2}


def test_sheet_writes_a_png(tmp_path):
    rng = np.random.default_rng(1)
    cols = [f"2026-05-{d:02d}" for d in range(1, 21)]
    ndvi = pd.DataFrame(rng.uniform(0, 1, (8, 20)), columns=cols)
    vh = pd.DataFrame(rng.uniform(-25, -12, (8, 20)), columns=cols)
    info = pd.DataFrame({"plot_id": range(8), "aoi": "aoi1", "region": ["A"] * 4 + ["B"] * 4, "pixels": 5})
    out = pc.sheet(info, ndvi, vh, np.array([1, 1, 1, 1, 2, 2, 2, 2]), tmp_path / "s.png")
    assert out.exists()


def test_assign_matches_the_nearest_pattern_inside_its_circle():
    rng = np.random.default_rng(0)
    t = np.arange(30)
    rice = np.where(t < 10, 0.15, 0.8)
    trees = np.full(30, 0.65)
    x = np.vstack([rice + rng.normal(0, 0.03, (40, 30)), trees + rng.normal(0, 0.03, (40, 30))])
    labels = np.r_[np.full(40, 1), np.full(40, 2)]
    cents = pc.centroids(x, labels)
    probe = np.vstack([rice, trees, np.full(30, -0.2)])          # rice, tree, open water (matches nothing)
    res, group, _ = pc.assign(probe, cents, {1})
    assert res.tolist() == [1, 2, 3] and group[:2].tolist() == [1, 2]


def test_check_patterns_leaves_the_region_out():
    rng = np.random.default_rng(1)
    t = np.arange(20)
    rice = np.where(t < 8, 0.15, 0.8)
    x = rice + rng.normal(0, 0.02, (60, 20))
    info = pd.DataFrame({"region": ["A"] * 30 + ["B"] * 30})
    out = pc.check_patterns(info, pd.DataFrame(x), np.full(60, 1), pd.DataFrame({"group": [1], "rice": [True]}))
    assert (out["rice (matches a rice pattern)"] > 80).all()


def test_a_shift_lets_the_same_crop_sown_later_match():
    t = np.arange(40)
    pattern = np.where(t < 10, 0.15, 0.8)
    later = np.where(t < 14, 0.15, 0.8)                         # the same crop, four windows later
    cents = {1: (pattern, 0.05)}
    assert pc.assign(later[None, :], cents, {1})[0].tolist() == [3]
    assert pc.assign(later[None, :], cents, {1}, max_shift=4)[0].tolist() == [1]


def test_labelled_groups_are_not_rebuilt(tmp_path, capsys):
    folder = tmp_path / "aoi9"
    folder.mkdir()
    (folder / "aoi9_group_verdicts_k20.csv").write_text("group,rice\n1,True\n")
    out = pc.aoi_groups(9, fresh=str(tmp_path))
    assert out.name == "aoi9_groups_k20.png" and "groups kept" in capsys.readouterr().out
