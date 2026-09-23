"""Tests for the two-class optical rice map. Synthetic frames only; no rasters, no network."""
import numpy as np
import pandas as pd
import pytest

from sar_pipeline.analysis import optical_rice_map as m


def test_otsu_finds_the_gap_between_two_groups():
    values = np.concatenate([np.full(500, 0.12), np.full(500, 0.62)])
    assert 0.12 < m.otsu(values) < 0.62


def test_otsu_is_not_fooled_by_unequal_group_sizes():
    values = np.concatenate([np.full(950, 0.10), np.full(50, 0.70)])
    assert 0.10 < m.otsu(values) < 0.70


def test_otsu_returns_nan_for_a_constant_distribution():
    assert np.isnan(m.otsu(np.full(100, 0.4)))


def test_min_pixels_covers_the_area_and_never_undershoots():
    assert m.min_pixels_for(0.15, 10.0) == 7          # 0.15 acre = 607 m2 = 6.07 pixels
    assert m.acres(7) >= 0.15 > m.acres(6)
    assert m.min_pixels_for(0.15, 20.0) == 2


def cycles_frame(rows):
    """A cycles frame shaped like the one optical_phenology produces."""
    out = pd.DataFrame(rows)
    out["peak_date"] = pd.to_datetime(out["peak_date"])
    for col, default in (("growth_from_baseline", np.nan), ("complete", True),
                         ("start_to_harvest_days", 110.0), ("prewet_ndvi_drop", 0.1),
                         ("prewet_lswi_rise", 0.05), ("wet_at_sowing", True)):
        out[col] = default if col not in out else out[col].fillna(default)
    out.attrs.update(shape=(1, len(out)), grid={})
    return out


def season(n=60, peak="2025-08-08", growth=0.7, **kw):
    return [dict(pid=i, peak_date=peak, growth_amplitude=growth, **kw) for i in range(n)]


def test_the_season_is_the_median_peak_of_the_strong_cycles():
    frame = cycles_frame(season(50) + season(10, peak="2025-11-20", growth=0.6))
    centre, start, end, cut = m.season_window(frame, tolerance_days=45)
    assert centre == pd.Timestamp("2025-08-08")
    assert (end - start).days == 90


def test_a_cycle_outside_the_season_is_not_rice_however_strong_it_is():
    frame = cycles_frame(season(50) + [dict(pid=99, peak_date="2025-05-25", growth_amplitude=0.9)])
    chosen, rice, used = m.classify(frame, amplitude_cut=0.45)
    assert 99 not in set(chosen["pid"])            # never even chosen
    assert used["pixels_with_an_in_season_cycle"] == 50


def test_the_chosen_cycle_is_the_strongest_one_in_season_not_overall():
    """A bigger flush outside the season must not hide the rice crop underneath it."""
    frame = cycles_frame(season(40) + [
        dict(pid=77, peak_date="2025-08-05", growth_amplitude=0.60),
        dict(pid=77, peak_date="2025-12-01", growth_amplitude=0.95)])
    chosen, rice, _ = m.classify(frame, amplitude_cut=0.45)
    row = chosen[chosen["pid"] == 77].iloc[0]
    assert row["peak_date"] == pd.Timestamp("2025-08-05")


def test_climb_takes_the_larger_of_the_two_measures():
    """A trough inside a cloudy gap understates growth; the pre-season baseline rescues it."""
    frame = cycles_frame([dict(pid=0, peak_date="2025-08-08", growth_amplitude=0.22,
                               growth_from_baseline=0.62)])
    assert float(m.climb(frame).iloc[0]) == 0.62


def test_a_cycle_that_is_too_short_or_too_long_is_not_rice():
    frame = cycles_frame(season(40) + [
        dict(pid=97, peak_date="2025-08-08", growth_amplitude=0.7, start_to_harvest_days=45),
        dict(pid=98, peak_date="2025-08-08", growth_amplitude=0.7, start_to_harvest_days=200)])
    chosen, rice, used = m.classify(frame, amplitude_cut=0.45)
    by_pid = dict(zip(chosen["pid"], rice))
    assert not by_pid[97] and not by_pid[98]
    assert used["failed_too_short"] == 1 and used["failed_max_cycle"] == 1


def test_a_field_that_was_dry_at_sowing_is_not_rice():
    frame = cycles_frame(season(40) + [dict(pid=96, peak_date="2025-08-08", growth_amplitude=0.7,
                                            prewet_lswi_rise=-0.12, wet_at_sowing=False)])
    chosen, rice, used = m.classify(frame, amplitude_cut=0.45)
    assert not dict(zip(chosen["pid"], rice))[96]
    assert used["failed_dry_at_sowing"] == 1


def test_an_untestable_wetness_does_not_fail_a_pixel():
    """The baseline stretch can fall outside the observed period; that is not evidence of dryness."""
    frame = cycles_frame(season(40) + [dict(pid=95, peak_date="2025-08-08", growth_amplitude=0.7,
                                            wet_at_sowing=False)])
    frame.loc[frame["pid"] == 95, ["prewet_ndvi_drop", "prewet_lswi_rise"]] = np.nan
    chosen, rice, used = m.classify(frame, amplitude_cut=0.45)
    assert dict(zip(chosen["pid"], rice))[95]
    assert used["wetness_not_testable"] == 1


def test_sieve_removes_small_patches_and_keeps_nodata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    band = np.zeros((20, 20), dtype="uint8")
    band[2:9, 2:9] = 1          # 49 pixels, well above the minimum
    band[15, 15] = 1            # a single pixel, below it
    band[0, :] = 255            # a nodata row
    path = tmp_path / "cls.tif"
    profile = dict(driver="GTiff", width=20, height=20, count=1, dtype="uint8", nodata=255,
                   crs="EPSG:32646", transform=from_origin(0, 200, 10, 10))
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(band, 1)
    out, size = m.sieve_raster(path, tmp_path / "sieved.tif", min_acres=0.15)
    with rasterio.open(out) as ds:
        cleaned = ds.read(1)
    assert size == 7
    assert cleaned[15, 15] == 0          # the speck is gone
    assert cleaned[5, 5] == 1            # the field survives
    assert (cleaned[0, :] == 255).all()  # nodata untouched


def test_summarise_reports_areas_that_add_up():
    frame = cycles_frame(season(30))
    rice = pd.Series([True] * 20 + [False] * 10)
    used = {"season_centre": "08 Aug 2025", "season_window": "24 Jun .. 22 Sep",
            "growth_amplitude_cut": 0.47, "failed_amplitude": 3, "failed_too_short": 1,
            "failed_max_cycle": 2, "failed_dry_at_sowing": 4, "wetness_not_testable": 5}
    row = m.summarise("aoiX", frame, rice, used, shape=(1, 40))
    assert row["rice_acres"] == round(m.acres(20), 1)
    assert row["undecided_acres"] == round(m.acres(10), 1)     # 40 pixels, 30 decided
    total = row["rice_acres"] + row["not_rice_acres"] + row["undecided_acres"]
    assert abs(total - row["total_acres"]) < 0.2
    assert row["rice_pct_of_decided"] == round(100 * 20 / 30, 1)


def test_run_batch_carries_on_past_a_failing_aoi(tmp_path, monkeypatch):
    calls = []

    def fake(aoi_id, out_root, **kw):
        calls.append(aoi_id)
        if aoi_id == 2:
            raise ValueError("no cycle in this AOI")
        return {"aoi": f"aoi{aoi_id}", "rice_acres": 10.0, "total_acres": 20.0,
                "rice_pct_of_decided": 50.0, "season_window": "x"}

    monkeypatch.setattr(m, "run_aoi", fake)
    frame = m.run_batch([1, 2, 3], out_root=tmp_path, log=lambda *a: None)
    assert calls == [1, 2, 3]
    assert len(frame) == 3
    assert frame.loc[1, "error"].startswith("ValueError")
    assert (tmp_path / "aoi_rice_acres.csv").exists()
