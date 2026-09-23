"""Tests for the Sentinel-2 reference-image export helpers. No Earth Engine calls."""
from __future__ import annotations

from sar_pipeline import optical_export as ox


def test_months_span_a_year_boundary_and_exclude_the_end():
    got = ox.months("2025-11", "2026-03")
    assert got == ["2025-11", "2025-12", "2026-01", "2026-02"]
    assert len(ox.months("2025-01", "2026-05")) == 16


def test_month_bounds_roll_over_december():
    assert ox.month_bounds("2025-12") == ("2025-12-01", "2026-01-01")
    assert ox.month_bounds("2025-02") == ("2025-02-01", "2025-03-01")


def test_export_params_reproduce_the_sar_grid():
    grid = {"crs": "EPSG:32632", "res": 10.0, "x0": 500000.0, "y0": 5500000.0, "width": 743, "height": 698}
    p = ox.grid_export_params(grid)
    assert p["crs"] == "EPSG:32632"
    assert p["crsTransform"] == [10.0, 0, 500000.0, 0, -10.0, 5500000.0]
    assert p["dimensions"] == "743x698"


def test_grid_bounds_are_ordered_lon_lat():
    grid = {"crs": "EPSG:32632", "res": 10.0, "x0": 500000.0, "y0": 5500000.0, "width": 100, "height": 50}
    w, s, e, n = ox.grid_bounds_4326(grid)
    assert w < e and s < n and 8 < w < 10 and 49 < s < 50


def test_object_names_keep_one_folder_per_aoi_and_the_exact_date():
    name = ox.object_name("base/s2_reference/", "aoiX", "2025-10", "2025-10-12")
    assert name == "base/s2_reference/aoiX/aoiX_2025-10_S2_2025-10-12"


def test_qgis_band_mapping_matches_the_file_band_order():
    order = {b: i + 1 for i, b in enumerate(ox.OUTPUT_BANDS)}
    tc = ox.QGIS_COMBINATIONS["true colour (4-3-2)"]
    fc = ox.QGIS_COMBINATIONS["false colour (8-3-2)"]
    re = ox.QGIS_COMBINATIONS["red edge (5-3-2)"]
    assert (tc["red"], tc["green"], tc["blue"]) == (order["B4"], order["B3"], order["B2"])
    assert fc["red"] == order["B8"] and re["red"] == order["B5"]


def test_aoi_geojson_drops_z():
    import geopandas as gpd
    from shapely.geometry import Polygon

    frame = gpd.GeoDataFrame(geometry=[Polygon([(10, 50, 0), (10.1, 50, 0), (10.1, 50.1, 0)])], crs=4326)
    geo = ox.aoi_geojson_2d(frame)
    assert all(len(pt) == 2 for pt in geo["coordinates"][0])


def test_dates_already_on_gcs_are_skipped():
    existing = {"p/aoiX/aoiX_2025-06_S2_2025-06-07.tif"}
    todo = ox.dates_to_export(["2025-06-07", "2025-06-12", "2025-06-12"], lambda d: d[:7],
                              existing, "p", "aoiX")
    assert todo == ["2025-06-12"]


class _FakeDataset:
    """Stand-in for a rasterio dataset: only band descriptions are needed."""

    def __init__(self, descriptions):
        self.descriptions = tuple(descriptions)


def test_band_index_is_looked_up_by_name_not_position():
    without = _FakeDataset(ox.BANDS + ("clear",))
    with_swir = _FakeDataset(ox.BANDS_WITH_SWIR + ("clear",))
    assert ox.band_index(without, "clear") == 6
    assert ox.band_index(with_swir, "clear") == 8
    assert ox.band_index(with_swir, "B8") == ox.band_index(without, "B8")


def test_band_index_rejects_a_missing_band():
    import pytest

    with pytest.raises(KeyError):
        ox.band_index(_FakeDataset(ox.BANDS + ("clear",)), "B11")


def test_swir_band_set_extends_the_default_without_reordering_it():
    assert ox.BANDS_WITH_SWIR[:len(ox.BANDS)] == ox.BANDS
    assert ox.BANDS_WITH_SWIR[len(ox.BANDS):] == ("B11", "B12")


def test_mask_band_set_adds_qa60_and_scl_after_the_swir_set():
    assert ox.BANDS_WITH_MASKS[:len(ox.BANDS_WITH_SWIR)] == ox.BANDS_WITH_SWIR
    assert ox.BANDS_WITH_MASKS[-2:] == ("QA60", "SCL")


def test_qa60_cloud_reads_only_bits_10_and_11():
    import numpy as np

    qa = np.array([0, 1 << 10, 1 << 11, (1 << 10) | (1 << 11), 1 << 9, 1])
    assert ox.qa60_cloud(qa).tolist() == [False, True, True, True, False, False]


def test_start_with_retries_builds_a_fresh_task_each_attempt():
    import ee

    built = []

    class Task:
        def __init__(self, fail):
            self.fail = fail

        def start(self):
            if self.fail:
                raise ee.ee_exception.EEException("request_id collision")

    def make():
        built.append(Task(fail=len(built) < 2))
        return built[-1]

    task = ox.start_with_retries(make, attempts=5, sleep=lambda s: None)
    assert len(built) == 3 and task is built[-1]


def test_start_with_retries_gives_up_after_the_last_attempt():
    import ee
    import pytest

    class Task:
        def start(self):
            raise ee.ee_exception.EEException("down")

    with pytest.raises(RuntimeError, match="after 2 attempts"):
        ox.start_with_retries(Task, attempts=2, sleep=lambda s: None)


def test_submit_each_records_a_failure_and_carries_on():
    def one(date):
        if date == "b":
            raise RuntimeError("could not start")
        return {"date": date}

    rows = ox.submit_each(["a", "b", "c"], one, log=lambda *a: None)
    assert [r.get("date") for r in rows] == ["a", None, "c"]
    assert rows[0]["error"] is None and rows[1]["error"].startswith("RuntimeError")


def test_wait_for_tasks_polls_until_nothing_is_active():
    rounds = iter([{"a": "RUNNING", "b": "READY"}, {"a": "COMPLETED", "b": "RUNNING"},
                   {"a": "COMPLETED", "b": "FAILED"}])
    states = ox.wait_for_tasks(["a", "b", None], status_of=lambda ids: next(rounds),
                               sleep=lambda s: None, log=lambda *a: None)
    assert states == {"a": "COMPLETED", "b": "FAILED"}
