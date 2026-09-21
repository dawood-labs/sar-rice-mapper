"""Tests for the AOI cross-check. No network, no Earth Engine."""
from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from sar_pipeline.prep import aoi_qc

# A projected CRS the test geometries actually fall inside: UTM 32N covers lon 6..12.
# Deliberately unrelated to any real project AOI - this repository is public.
UTM = 32632
LON, LAT = 10.0, 50.0   # neutral point well inside UTM 32N


def square(dx: float, dy: float, side: float = 0.01) -> Polygon:
    """A small lon/lat square offset by ``(dx, dy)`` degrees from the neutral test point."""
    x, y = LON + dx, LAT + dy
    return Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)])


def frame(rows: list[tuple[int, Polygon]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"id": [r[0] for r in rows]},
                            geometry=[r[1] for r in rows], crs="EPSG:4326")


def write_split(tmp_path, rows: list[tuple[int, Polygon]], stem: str = "SET"):
    """Write one-feature-per-file AOIs the way a split/dedup job would."""
    paths = []
    for aoi_id, geom in rows:
        folder = tmp_path / f"{stem}_{aoi_id:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}_{aoi_id:03d}.gpkg"
        frame([(aoi_id, geom)]).to_file(path, driver="GPKG")
        paths.append(path)
    return paths


def test_find_duplicates_counts_distinct_shapes():
    a, b, c = square(0, 0), square(1, 1), square(2, 2)
    report = aoi_qc.find_duplicates(frame([(1, a), (2, a), (3, b), (4, c), (5, c), (6, c), (7, c)]))
    assert report.n_features == 7
    assert report.n_unique == 3
    assert report.n_duplicate_rows == 4
    assert report.group_sizes == {1: 1, 2: 1, 4: 1}
    assert report.is_consistent()


def test_duplicates_ignore_vertex_order():
    """Two rings describing the same square must count as one shape, not two."""
    poly = square(0, 0)
    reversed_ring = Polygon(list(poly.exterior.coords)[::-1])
    assert aoi_qc.find_duplicates(frame([(1, poly), (2, reversed_ring)])).n_unique == 1


def test_crosscheck_accepts_a_faithful_deduplication(tmp_path):
    a, b, c = square(0, 0), square(1, 1), square(2, 2)
    original = frame([(1, a), (2, a), (3, b), (4, c)])       # id 2 is a copy of id 1
    write_split(tmp_path, [(1, a), (3, b), (4, c)])

    report = aoi_qc.crosscheck(original, aoi_qc.read_split_aois(tmp_path.glob("*/*.gpkg")))

    assert report.ok
    assert report.n_split_files == 3
    assert report.missing_from_split == []
    assert report.unexpected_in_split == []
    assert report.dropped_to_kept == {2: 1}
    assert report.orphaned == []


def test_crosscheck_detects_a_lost_polygon(tmp_path):
    """The failure that matters: a distinct shape present in the original and absent from the split."""
    a, b = square(0, 0), square(1, 1)
    original = frame([(1, a), (2, b)])
    write_split(tmp_path, [(1, a)])                          # id 2 silently lost

    report = aoi_qc.crosscheck(original, aoi_qc.read_split_aois(tmp_path.glob("*/*.gpkg")))

    assert not report.ok
    assert len(report.missing_from_split) == 1
    assert report.orphaned == [2]                            # dropped, with no kept twin


def test_crosscheck_detects_a_shape_that_is_not_in_the_original(tmp_path):
    a, stranger = square(0, 0), square(5, 5)
    write_split(tmp_path, [(1, a), (2, stranger)])

    report = aoi_qc.crosscheck(frame([(1, a)]), aoi_qc.read_split_aois(tmp_path.glob("*/*.gpkg")))

    assert not report.ok
    assert len(report.unexpected_in_split) == 1


def test_crosscheck_flags_renumbered_files(tmp_path):
    """A split that renumbers its outputs loses traceability back to the original ids."""
    a = square(0, 0)
    folder = tmp_path / "SET_001"
    folder.mkdir()
    frame([(77, a)]).to_file(folder / "SET_001.gpkg", driver="GPKG")   # filename says 1, field says 77

    report = aoi_qc.crosscheck(frame([(1, a)]), aoi_qc.read_split_aois(tmp_path.glob("*/*.gpkg")))

    assert report.id_mismatches == [(1, 77)]
    assert not report.ok


def test_read_split_aois_refuses_a_multi_feature_file(tmp_path):
    folder = tmp_path / "SET_001"
    folder.mkdir()
    frame([(1, square(0, 0)), (2, square(1, 1))]).to_file(folder / "SET_001.gpkg", driver="GPKG")
    with pytest.raises(ValueError, match="expected exactly 1 feature"):
        aoi_qc.read_split_aois(tmp_path.glob("*/*.gpkg"))


def test_area_table_reports_acres_largest_first():
    small, big = square(0, 0, side=0.001), square(1, 1, side=0.01)
    table = aoi_qc.area_table(frame([(1, small), (2, big)]), UTM)
    assert list(table["id"]) == [2, 1]                       # sorted by area, largest first
    assert table["acres"].iloc[0] > table["acres"].iloc[1]
    # a 0.01 deg square at this latitude is roughly 0.7 x 1.1 km -> a couple of hundred acres
    assert 100 < table["acres"].iloc[0] < 300


def test_area_table_refuses_a_geographic_crs():
    """Areas computed in degrees are meaningless, so they must not be produced at all."""
    with pytest.raises(ValueError, match="geographic"):
        aoi_qc.area_table(frame([(1, square(0, 0))]), 4326)


def test_sqm_to_acres_uses_the_exact_factor():
    assert aoi_qc.sqm_to_acres(aoi_qc.SQM_PER_ACRE) == pytest.approx(1.0)


def test_compare_area_column_identifies_acres_and_hectares():
    geom = square(0, 0, side=0.01)
    acres = aoi_qc.sqm_to_acres(gpd.GeoSeries([geom], crs=4326).to_crs(UTM).area.iloc[0])

    as_acres = gpd.GeoDataFrame({"id": [1], "area": [acres]}, geometry=[geom], crs="EPSG:4326")
    assert aoi_qc.compare_area_column(as_acres, UTM)["likely_unit"] == "acres"

    as_hectares = gpd.GeoDataFrame({"id": [1], "area": [acres / 2.47105]}, geometry=[geom], crs="EPSG:4326")
    assert aoi_qc.compare_area_column(as_hectares, UTM)["likely_unit"] == "hectares"
