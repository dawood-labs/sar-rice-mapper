"""Tests for the per-AOI season screen summary. No Earth Engine calls."""
from __future__ import annotations

import pandas as pd

from sar_pipeline import season_screen as ss


def test_summary_is_in_acres_and_ranked_by_seasonal_area():
    acre = ss.SQM_PER_ACRE
    frame = pd.DataFrame({"aoi": ["a", "b"], "total": [100 * acre, 50 * acre],
                          "seen": [80 * acre, 0.0], "seasonal": [20 * acre, None],
                          "evergreen": [8 * acre, None]})
    out = ss.summarise(frame)
    a = out.set_index("aoi").loc["a"]
    assert out["aoi"].tolist() == ["a", "b"]
    assert (a["total_acres"], a["seen_pct"], a["seasonal_pct_of_seen"], a["evergreen_pct_of_seen"]) == (100.0, 80.0, 25.0, 10.0)
    assert pd.isna(out.set_index("aoi").loc["b", "seasonal_pct_of_seen"])   # never seen: unknown, not 0
