"""Tests for the Sentinel-1 availability summary. Earth Engine is never contacted."""
from __future__ import annotations

import datetime as dt

from sar_pipeline.prep import s1_availability as av


def epoch_ms(year: int, month: int, day: int) -> int:
    return int(dt.datetime(year, month, day, tzinfo=dt.timezone.utc).timestamp() * 1000)


ROWS = [
    # same track, three distinct dates in May
    (epoch_ms(2026, 5, 1), 143, "ASCENDING", "A"),
    (epoch_ms(2026, 5, 13), 143, "ASCENDING", "A"),
    (epoch_ms(2026, 5, 25), 143, "ASCENDING", "C"),
    # a second track, two dates in May, one of them shared with the first track
    (epoch_ms(2026, 5, 1), 106, "DESCENDING", "A"),
    (epoch_ms(2026, 5, 19), 106, "DESCENDING", "D"),
    # one lonely date in June
    (epoch_ms(2026, 6, 6), 143, "ASCENDING", "A"),
]


def test_build_report_groups_tracks_by_orbit_and_pass():
    report = av.build_report(ROWS, "2026-05-01", "2026-07-01")
    assert report.n_scenes == 6
    assert report.n_tracks == 2
    assert [t.n_scenes for t in report.tracks] == [4, 2]        # most scenes first
    assert report.tracks[0].relative_orbit == 143
    assert report.tracks[0].orbit_pass == "ASCENDING"


def test_same_orbit_in_both_directions_counts_as_two_tracks():
    """Ascending and descending passes see different geometry, so they must not be merged."""
    rows = [(epoch_ms(2026, 5, 1), 143, "ASCENDING", "A"),
            (epoch_ms(2026, 5, 2), 143, "DESCENDING", "A")]
    assert av.build_report(rows, "2026-05-01", "2026-06-01").n_tracks == 2


def test_track_key_matches_the_config_shape():
    track = av.Track(relative_orbit=143, orbit_pass="ASCENDING", n_scenes=1)
    assert track.key == "RO143_ASC"
    assert av.Track(relative_orbit=4, orbit_pass="DESCENDING", n_scenes=1).key == "RO4_DES"


def test_dates_per_month_counts_distinct_dates_not_scenes():
    """Two tracks acquiring on the same day is one date, not two - the time series has one entry."""
    report = av.build_report(ROWS, "2026-05-01", "2026-07-01")
    assert report.dates_per_month == {"2026-05": 4, "2026-06": 1}   # 1, 13, 19, 25 May then 6 June


def test_platforms_are_counted():
    assert av.build_report(ROWS, "2026-05-01", "2026-07-01").platforms == {"A": 4, "C": 1, "D": 1}


def test_months_below_flags_thin_months_in_order():
    report = av.build_report(ROWS, "2026-05-01", "2026-07-01")
    assert report.months_below(4) == ["2026-06"]
    assert report.months_below(2) == ["2026-06"]
    assert report.months_below(1) == []
    assert report.months_below(99) == ["2026-05", "2026-06"]


def test_empty_window_reports_nothing_rather_than_failing():
    """An AOI/window with no scenes is a real answer worth printing, not an error."""
    report = av.build_report([], "2026-05-01", "2026-06-01")
    assert report.n_scenes == 0 and report.n_tracks == 0
    assert report.dates_per_month == {} and report.months_below(1) == []


def test_availability_uses_the_injected_fetcher():
    calls = {}

    def fake_fetch(bounds, start, end, **kwargs):
        calls.update(bounds=bounds, start=start, end=end, kwargs=kwargs)
        return ROWS

    report = av.availability((94.5, 16.0, 96.7, 20.2), "2026-05-01", "2026-07-01",
                             fetch=fake_fetch, collection="COPERNICUS/S1_GRD")
    assert report.n_scenes == 6
    assert calls["bounds"] == (94.5, 16.0, 96.7, 20.2)
    assert calls["kwargs"] == {"collection": "COPERNICUS/S1_GRD"}


def test_summary_mentions_every_track_and_month():
    text = av.build_report(ROWS, "2026-05-01", "2026-07-01").summary()
    assert "RO143_ASC" in text and "RO106_DES" in text
    assert "2026-05" in text and "2026-06" in text
    assert "2026-05-01 .. 2026-07-01" in text
