"""Synthetic checks of the radar-only features and rule (no real data, no location)."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import sar_only as so


def test_fill_gaps_and_resample_hold_the_ends_and_interpolate_between_passes():
    dates = pd.to_datetime(["2026-03-16", "2026-03-28", "2026-04-09", "2026-04-21"])
    v = np.array([[-20.0, np.nan], [np.nan, -12.0], [-16.0, -14.0], [-14.0, np.nan]], dtype="float32")
    f = so.fill_gaps(v)
    assert np.allclose(f[:, 0], [-20, -20, -16, -14]) and np.allclose(f[:, 1], [-12, -12, -14, -14])
    g = so.resample(dates, v, pd.to_datetime(["2026-03-10", "2026-03-22", "2026-04-15", "2026-05-01"]))
    assert np.allclose(g[:, 0], [-20, -20, -15, -14])            # before the first pass: held; midway: the mean


def _features(**over):
    base = {"VH_pre_min": -22.0, "flood_ok": 1.0, "flood_drop": 8.0, "flood_vh": -24.0, "flood_support": 3,
            "canopy_rise": 9.0, "VH_end": -15.0, "VH_end_fall": 0.5, "VV_end_fall": 0.5}
    base.update(over)
    return pd.DataFrame([base])


def test_radar_rule_classes():
    assert so.radar_rule(_features())[0] == 1                                    # flood + canopy + standing
    assert so.radar_rule(_features(VH_pre_min=-12.0))[0] == 0                    # trees / houses before the monsoon
    assert so.radar_rule(_features(flood_ok=0.0))[0] == 0                        # no pass passed the water test
    assert so.radar_rule(_features(flood_support=1))[0] == 0                     # one pass only
    assert so.radar_rule(_features(canopy_rise=4.5, VH_end=-16.0))[0] == 1       # a shallow flood: 4.5 dB up is a canopy
    assert so.radar_rule(_features(canopy_rise=2.5, VH_end=-18.5))[0] == 2       # a young crop, still low
    assert so.radar_rule(_features(canopy_rise=1.0, VH_end=-23.0))[0] == 7       # still under water
    assert so.radar_rule(_features(VH_end_fall=4.0, VV_end_fall=3.5))[0] == 4    # canopy grown, then both fell


def test_pixel_features_on_a_synthetic_series(monkeypatch):
    """A paddy pixel (dry in April, flooded in June, canopy in August) and a tree pixel."""
    dates = pd.date_range("2026-03-16", "2026-09-20", freq="6D")
    n = len(dates)
    vh_paddy = np.full(n, -16.0)
    vh_paddy[(dates >= "2026-06-01") & (dates <= "2026-07-01")] = -24.0
    vh_paddy[dates > "2026-08-01"] = -13.0
    vh_tree = np.full(n, -11.0)
    vh = np.stack([vh_paddy, vh_tree], axis=1).astype("float32")
    vv = vh + 7.0
    from sar_pipeline.analysis import radar_water as rw
    series = {"dates": dates, "shape": (1, 2), "track": np.zeros(n, dtype=int), "VH": vh, "VV": vv,
              "drop_VH": rw.local_drops(dates, vh), "drop_VV": rw.local_drops(dates, vv)}
    monkeypatch.setattr(so, "merged_series", lambda aoi_id, window=5: series)
    f = so.pixel_features(0, inside_only=False)
    F = so.frame(f)
    assert f["X"].shape == (2, len(f["names"])) and f["pixels"].tolist() == [0, 1]
    assert F.loc[0, "flood_vh"] <= -19 and F.loc[0, "flood_drop"] >= 4 and F.loc[0, "canopy_rise"] > 5
    assert F.loc[1, "VH_pre_min"] > so.COVER_VH_MIN
    assert so.radar_rule(F).tolist() == [1, 0]


def test_folds_leave_whole_aois_out():
    table = pd.DataFrame({"aoi": [f"aoi{i}" for i in range(10) for _ in range(3)], "y": 0})
    folds = so.folds_by(table, "aoi", n_folds=5)
    held = [a for _, aois in folds for a in aois]
    assert sorted(held) == sorted(table["aoi"].unique()) and len(folds) == 5
