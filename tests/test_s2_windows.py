"""Tests for the fixed-window Sentinel-2 export. Pure functions only; no Earth Engine, no network."""
import datetime as dt

from sar_pipeline import s2_windows as sw


def test_windows_tile_the_period_without_gaps_or_overlaps():
    w = sw.windows("2025-03-01", "2025-04-01", 5)
    assert w[0] == ("2025-03-01", "2025-03-06")
    assert all(a[1] == b[0] for a, b in zip(w, w[1:]))
    assert w[-1][1] == "2025-04-01"


def test_the_last_window_is_clipped_never_extended():
    w = sw.windows("2025-03-01", "2025-03-13", 5)
    assert w[-1] == ("2025-03-11", "2025-03-13")
    assert dt.date.fromisoformat(w[-1][1]) == dt.date(2025, 3, 13)


def test_windows_are_anchored_to_the_start_so_two_aois_share_a_grid():
    """Every AOI must land on the same calendar windows, or they cannot be compared window by window."""
    a = sw.windows("2025-03-01", "2026-02-01")
    b = sw.windows("2025-03-01", "2025-09-01")
    assert a[:len(b) - 1] == b[:-1]      # all but b's own clipped last window
    assert b[-1][0] == a[len(b) - 1][0]  # which still starts on the shared grid


def test_an_empty_period_yields_no_windows():
    assert sw.windows("2025-03-01", "2025-03-01") == []


def test_object_name_carries_the_aoi_the_step_and_the_window_start():
    name = sw.object_name("base/folder/", "aoiX", "2025-05-16")
    assert name == "base/folder/aoiX/aoiX_W5_2025-05-16"


def test_the_count_band_is_not_one_of_the_reflectance_bands():
    """Validity is read from the count, so it must never collide with a data band."""
    assert sw.COUNT_BAND not in sw.BANDS


def test_the_mask_settings_are_the_ones_the_recipe_expects():
    assert sw.MSK["prob"] == 70 and sw.MSK["drk"] == 0.15
    assert sw.MSK["dist"] == 1 and sw.MSK["buf"] == 50


def test_export_window_retries_before_giving_up(monkeypatch):
    """One rejected submission used to abort a 544-task run and lose two whole AOIs."""
    import ee

    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ee.ee_exception.EEException("a different Operation was already started")

        class Task:
            id = "ok"

            def start(self):
                return None

        return Task()

    monkeypatch.setattr(sw, "window_composite", lambda *a, **k: None)
    monkeypatch.setattr(ee.batch.Export.image, "toCloudStorage", flaky)
    monkeypatch.setattr(ee, "Geometry", lambda g: g)
    monkeypatch.setattr(sw.time, "sleep", lambda s: None)
    task = sw.export_window({}, {"crs": "EPSG:32646", "res": 10, "x0": 0, "y0": 0,
                                 "width": 1, "height": 1}, "b", "p/x", "2025-03-01", "2025-03-06")
    assert task.id == "ok" and calls["n"] == 3


def test_export_window_raises_a_clear_error_after_its_attempts(monkeypatch):
    import ee

    def always_fails(*args, **kwargs):
        raise ee.ee_exception.EEException("nope")

    monkeypatch.setattr(sw, "window_composite", lambda *a, **k: None)
    monkeypatch.setattr(ee.batch.Export.image, "toCloudStorage", always_fails)
    monkeypatch.setattr(ee, "Geometry", lambda g: g)
    monkeypatch.setattr(sw.time, "sleep", lambda s: None)
    import pytest

    with pytest.raises(RuntimeError, match="could not start the export"):
        sw.export_window({}, {"crs": "EPSG:32646", "res": 10, "x0": 0, "y0": 0,
                              "width": 1, "height": 1}, "b", "p/x", "2025-03-01", "2025-03-06",
                         attempts=2)
