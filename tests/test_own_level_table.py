"""The AOI rice level and the summary shares of the own-level table (synthetic, no files except tmp)."""
import numpy as np
import pandas as pd

from sar_pipeline.analysis import own_level_table as ot


def test_aoi_rice_level_uses_standing_full_canopy_class1_only():
    n = 200
    ev = pd.DataFrame({"peak_after": np.r_[np.full(100, 0.8), np.full(100, 0.3)], "standing": True,
                       "vh_end_own": np.r_[np.full(100, -14.0), np.full(100, -20.0)]})
    classes = np.r_[np.ones(150, dtype="uint8"), np.full(50, 6, dtype="uint8")]
    assert ot.aoi_rice_level(ev, classes) == -14.0
    assert np.isnan(ot.aoi_rice_level(ev.iloc[:10], classes[:10]))      # too few pixels to set a level


def test_summary_counts_canopy_and_full_by_k(tmp_path):
    t = pd.DataFrame({"set": ["evergreen"] * 2 + ["noted_field"] * 2, "field_id": ["", "", "aoi1_000001", "aoi1_000001"],
                      "flood_ok": [False, True, True, True], "recovery": [np.nan, 0.2, 0.5, 1.1],
                      "recovery_aoi": [np.nan, 0.1, 0.9, 1.2], "rise_z": [np.nan, 1.0, 2.5, 6.0],
                      "radar_canopy_rise": [np.nan, 1.0, 4.0, 8.0], "class": [0, 0, 6, 6]})
    t.to_csv(tmp_path / "aoi1.csv", index=False)
    s = ot.summary(str(tmp_path)).set_index("group")
    assert s.loc["evergreen", "canopy_k2"] == 0.0 and s.loc["aoi1_000001", "canopy_k2"] == 1.0
    assert s.loc["aoi1_000001", "canopy_k3"] == 0.5 and s.loc["aoi1_000001", "full_k2"] == 0.5


def test_combine_tracks_adds_independent_looks():
    pt = pd.DataFrame({"pixel": [1, 1, 1, 2, 2], "rise_z": [2.0, 2.0, 2.0, 2.0, np.nan],
                       "recovery": [0.5, 0.6, 0.7, 0.4, np.nan]})
    c = ot.combine_tracks(pt).set_index("pixel")
    assert np.isclose(c.loc[1, "rise_z_all"], 6 / np.sqrt(3)) and c.loc[1, "tracks"] == 3 and c.loc[1, "tracks_rising"] == 3
    assert np.isclose(c.loc[2, "rise_z_all"], 2.0) and c.loc[2, "tracks"] == 1          # a missing track is left out
    assert np.isclose(c.loc[1, "recovery_all"], 0.6)
