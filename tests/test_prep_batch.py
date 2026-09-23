"""Tests for batch AOI setup: the track-choice rule and config rendering. No network."""
from __future__ import annotations

import pandas as pd
import pytest
import yaml

from sar_pipeline.prep import batch

FLOOD = ("2025-05-01", "2025-08-31")   # sowing runs Jun-Aug


def summary(rows):
    return pd.DataFrame(rows, columns=["track_id", "n_acq_ok_coverage", "max_gap_days",
                                       "mean_aoi_coverage_pct"])


def gaps(rows=()):
    return pd.DataFrame(list(rows), columns=["track_id", "gap_start", "gap_end", "gap_days"])


def test_real_audit_shape_excludes_the_track_with_flooding_window_gaps():
    """The case this rule was written for: three tracks, one blind during May-June."""
    s = summary([("RO143_ASC", 40, 12, 100.0), ("RO033_DSC", 32, 18, 100.0), ("RO106_DSC", 27, 24, 100.0)])
    g = gaps([("RO033_DSC", "2025-07-11", "2025-07-29", 18),
              ("RO106_DSC", "2025-05-01", "2025-05-23", 22),
              ("RO106_DSC", "2025-05-23", "2025-06-16", 24)])
    choice = batch.choose_tracks(s, g, FLOOD)
    assert (choice.primary, choice.secondary) == ("RO143_ASC", "RO033_DSC")
    assert "overlaps the flooding window" in choice.reasons["RO106_DSC"]


def test_an_18_day_gap_in_the_flooding_window_is_tolerated():
    """Shorter than the ~3-week flooded period, so the flood is still observed on either side."""
    s = summary([("RO143_ASC", 40, 12, 100.0), ("RO033_DSC", 32, 18, 100.0)])
    g = gaps([("RO033_DSC", "2025-07-11", "2025-07-29", 18)])
    assert batch.choose_tracks(s, g, FLOOD).secondary == "RO033_DSC"


def test_a_21_day_gap_in_the_flooding_window_excludes():
    s = summary([("A", 40, 12, 100.0), ("B", 32, 21, 100.0)])
    g = gaps([("B", "2025-08-01", "2025-08-22", 21)])
    choice = batch.choose_tracks(s, g, FLOOD)
    assert choice.secondary is None and "excluded" in choice.reasons["B"]


def test_a_long_gap_outside_the_flooding_window_is_allowed():
    s = summary([("A", 40, 12, 100.0), ("B", 30, 30, 100.0)])
    g = gaps([("B", "2025-12-01", "2025-12-31", 30)])
    choice = batch.choose_tracks(s, g, FLOOD)
    assert (choice.primary, choice.secondary) == ("A", "B")


def test_a_gap_of_exactly_the_limit_is_allowed():
    s = summary([("A", 40, 20, 100.0)])
    g = gaps([("A", "2025-06-01", "2025-06-21", 20)])
    assert batch.choose_tracks(s, g, FLOOD).primary == "A"


def test_poor_coverage_and_few_acquisitions_are_excluded():
    s = summary([("A", 40, 12, 100.0), ("B", 38, 12, 60.0), ("C", 10, 12, 100.0)])
    choice = batch.choose_tracks(s, gaps(), FLOOD)
    assert choice.primary == "A" and choice.secondary is None
    assert "coverage" in choice.reasons["B"]
    assert "under 50%" in choice.reasons["C"]


def test_ties_break_on_the_shorter_longest_gap():
    s = summary([("A", 40, 18, 100.0), ("B", 40, 12, 100.0)])
    assert batch.choose_tracks(s, gaps(), FLOOD).primary == "B"


def test_at_most_two_tracks_and_the_third_is_recorded():
    s = summary([("A", 40, 12, 100.0), ("B", 35, 12, 100.0), ("C", 30, 12, 100.0)])
    choice = batch.choose_tracks(s, gaps(), FLOOD)
    assert [t["track_id"] for t in choice.tracks] == ["A", "B"]
    assert "third" in choice.reasons["C"]


def test_no_tracks_at_all_returns_an_empty_choice():
    choice = batch.choose_tracks(summary([]), gaps(), FLOOD)
    assert choice.primary is None and choice.tracks == []


def test_every_considered_track_gets_a_reason():
    s = summary([("A", 40, 12, 100.0), ("B", 5, 12, 100.0)])
    assert set(batch.choose_tracks(s, gaps(), FLOOD).reasons) == {"A", "B"}


TEMPLATE = {
    "aoi": {"key": "demoaoi", "path": "data/aoi/demo_field.gpkg"},
    "season": {"key": "monsoon2025", "timezone": "UTC", "start": "2025-05-01", "end": "2026-02-01"},
    "gcs": {"bucket": "b", "base_folder": "f"},
    "grid": {"crs": "EPSG:32632", "res": 10},
    "s1": {"collection": "COPERNICUS/S1_GRD_FLOAT", "tracks": [{"track_id": "X", "role": "primary"}]},
    "analysis": {"bins": {"kind": "half_month", "start": "2025-05-01"}},
}


def test_render_config_changes_only_the_per_aoi_fields():
    text = batch.render_config(TEMPLATE, 7, "data/aoi/demo_7.gpkg", "year2025", "2025-05-01", "2026-05-01",
                               tracks=[{"track_id": "RO143_ASC", "role": "primary"}])
    cfg = yaml.safe_load(text)
    assert cfg["aoi"] == {"key": "aoi7", "path": "data/aoi/demo_7.gpkg"}
    assert cfg["season"]["key"] == "year2025"
    assert (cfg["season"]["start"], cfg["season"]["end"]) == ("2025-05-01", "2026-05-01")
    assert cfg["analysis"]["bins"]["start"] == "2025-05-01"
    assert cfg["s1"]["tracks"] == [{"track_id": "RO143_ASC", "role": "primary"}]
    assert cfg["grid"] == TEMPLATE["grid"] and cfg["gcs"] == TEMPLATE["gcs"]


def test_render_config_does_not_mutate_the_template():
    batch.render_config(TEMPLATE, 7, "p", "k", "2025-05-01", "2026-05-01")
    assert TEMPLATE["aoi"]["key"] == "demoaoi"
    assert TEMPLATE["s1"]["tracks"] == [{"track_id": "X", "role": "primary"}]


def test_aoi_ids_are_read_from_folder_names(tmp_path):
    for name in ("SET_001", "SET_119", "_batch", "SET_020"):
        (tmp_path / name).mkdir()
    (tmp_path / "SET_999.txt").write_text("not a folder")
    assert batch.aoi_ids_in(tmp_path) == [1, 20, 119]


def test_fallback_secondary_takes_a_track_excluded_only_for_a_flooding_gap():
    s = summary([("RO143_ASC", 40, 12, 100.0), ("RO106_DSC", 27, 24, 100.0)])
    g = gaps([("RO106_DSC", "2025-05-23", "2025-06-16", 24)])
    assert batch.choose_tracks(s, g, FLOOD).secondary is None
    choice = batch.choose_tracks(s, g, FLOOD, fallback_secondary=True)
    assert (choice.primary, choice.secondary) == ("RO143_ASC", "RO106_DSC")
    assert "fallback" in choice.reasons["RO106_DSC"]


def test_fallback_never_promotes_a_gapped_track_to_primary():
    s = summary([("RO106_DSC", 40, 24, 100.0)])
    g = gaps([("RO106_DSC", "2025-05-23", "2025-06-16", 24)])
    choice = batch.choose_tracks(s, g, FLOOD, fallback_secondary=True)
    assert choice.primary is None and choice.secondary is None


def test_fallback_does_not_rescue_a_track_with_poor_coverage():
    s = summary([("A", 40, 12, 100.0), ("B", 30, 24, 50.0)])
    g = gaps([("B", "2025-06-01", "2025-06-25", 24)])
    assert batch.choose_tracks(s, g, FLOOD, fallback_secondary=True).secondary is None


def test_fallback_is_unused_when_an_eligible_secondary_exists():
    s = summary([("A", 40, 12, 100.0), ("B", 30, 12, 100.0), ("C", 35, 24, 100.0)])
    g = gaps([("C", "2025-06-01", "2025-06-25", 24)])
    assert batch.choose_tracks(s, g, FLOOD, fallback_secondary=True).secondary == "B"


def test_pick_spread_takes_the_largest_and_keeps_its_distance():
    import pandas as pd

    index = pd.DataFrame({
        "aoi": [1, 2, 3, 4],
        "lon": [95.0, 95.01, 96.0, 95.5],
        "lat": [17.0, 17.0, 17.0, 18.0],
        "acres": [1000, 900, 500, 100],
    })
    # 2 sits ~1 km from 1, so it is skipped while distant AOIs remain; then it fills the batch.
    assert batch.pick_spread(index, 3, min_km=15) == [1, 3, 4]
    assert batch.pick_spread(index, 4, min_km=15) == [1, 3, 4, 2]
    assert batch.pick_spread(index, 2, min_km=15, exclude=[1]) == [2, 3]
