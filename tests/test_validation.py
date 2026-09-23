"""Tests for the validation helpers on synthetic event tables. No files."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sar_pipeline.analysis import validation as va


def _events():
    n = 8
    return pd.DataFrame({
        "set": ["rice_plot_interior"] * 4 + ["evergreen", "water", "bare_or_built", "cut_before_map_date"],
        "region": ["A", "A", "B", "B", "A", "A", "B", "B"],
        "valid": [True] * n,
        "trough_ndvi": [0.1, 0.2, 0.15, 0.3, 0.7, 0.0, 0.1, 0.1],
        "rise": [0.7, 0.6, 0.65, 0.5, 0.1, 0.05, 0.1, 0.6],
        "peak_after": [0.8, 0.8, 0.8, 0.8, 0.8, 0.05, 0.2, 0.7],
        "last_ndvi": [0.8, 0.8, 0.8, 0.8, 0.8, 0.05, 0.2, 0.2],
        "VV_dip": [4.0, 5.0, 2.0, 6.0, 0.5, 0.2, 1.0, 4.0],
        "VH_dip": [5.0, 2.0, 4.0, 1.0, 0.3, 0.1, 0.5, 5.0],
        "VV_dip_min_track": [3.0, 4.0, 1.0, 5.0, 0.2, 0.1, 0.5, 3.0],
        "VH_dip_min_track": [4.0, 1.0, 3.0, 0.5, 0.1, 0.0, 0.2, 4.0],
    })


def test_rule_scores_rice_and_negatives_separately():
    ev = _events()
    called = va.classify_with(ev)
    assert called.tolist() == [True, True, True, True, False, False, False, False]
    table = va.score(ev, called).set_index(["set", "region"])
    assert table.loc[("rice_plot_interior", "A"), "called_pct"] == 100.0
    assert table.loc[("cut_before_map_date", "B"), "called_pct"] == 0.0   # cut crop: not standing


def test_requiring_both_tracks_is_stricter():
    ev = _events()
    assert va.classify_with(ev, both_tracks=True).tolist()[:4] == [True, True, True, True]
    ev.loc[1, "VV_dip_min_track"] = 1.0
    assert va.classify_with(ev, both_tracks=True).tolist()[1] is np.False_ or not va.classify_with(ev, both_tracks=True)[1]


def test_sensitivity_and_leave_one_region_out_run():
    ev = _events()
    s = va.sensitivity(ev, "dip_min", [1.0, 3.0, 10.0])
    assert s["recall_pct"].tolist() == [100.0, 100.0, 0.0]
    lo = va.leave_one_region_out(ev, regions=("A", "B"))
    assert set(lo["held_out"]) == {"A", "B"} and lo["recall_pct"].notna().all()
