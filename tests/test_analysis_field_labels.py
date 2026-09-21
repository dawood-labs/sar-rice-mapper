"""Field labelling: small synthetic geometry and raster checks, no real data."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

from sar_pipeline.analysis import field_labels as fl

CRS = "EPSG:32633"          # any projected CRS in metres; the module never assumes a particular one
NAMES = ["Crop A", "Crop B"]
CODES = [1, 2]


def _frame(geoms, **cols):
    f = gpd.GeoDataFrame(dict(cols), geometry=list(geoms), crs=CRS)
    f["fid"] = np.arange(1, len(f) + 1, dtype="int32")
    return f


def _raster(tmp_path, array, origin=(0.0, 1000.0), res=10.0, nodata=0):
    path = tmp_path / f"class_{array.shape[0]}x{array.shape[1]}_{nodata}.tif"
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype="uint8", crs=CRS, transform=from_origin(origin[0], origin[1], res, res),
                       nodata=nodata) as ds:
        ds.write(array.astype("uint8"), 1)
    return path


# ---------------------------------------------------------------- geometry cleanup

def _staircase(step=0.2, n=200):
    """A 40 x 40 m square whose top edge is a pixel staircase with a vertex every `step` metres."""
    top = []
    for i in range(n):
        x = i * step
        top += [(x, 40.0 + (i % 2) * 0.1), (x + step, 40.0 + (i % 2) * 0.1)]
    return Polygon([(0, 0), (n * step, 0)] + top[::-1])


def test_simplify_outlines_drops_staircase_vertices_without_moving_the_field():
    field = _staircase()
    out, stats = fl.simplify_outlines(gpd.GeoDataFrame(geometry=[field], crs=CRS), 0.5)
    assert stats["vertices_out"] < stats["vertices_in"] / 10
    assert out.area.iloc[0] == pytest.approx(field.area, rel=0.01)
    assert out.geometry.iloc[0].hausdorff_distance(field) <= 0.5


def test_simplify_outlines_zero_tolerance_changes_nothing():
    frame = gpd.GeoDataFrame(geometry=[_staircase()], crs=CRS)
    out, stats = fl.simplify_outlines(frame, 0.0)
    assert out.geometry.iloc[0].equals(frame.geometry.iloc[0])
    assert stats["vertices_out"] == stats["vertices_in"]


def test_explode_fully_keeps_area_hidden_in_a_nested_collection():
    """One explode leaves a MultiPolygon inside a GeometryCollection; a Polygon filter then drops it."""
    nested = GeometryCollection([MultiPolygon([box(0, 0, 10, 10), box(20, 0, 30, 10)])])
    out = fl.explode_fully(gpd.GeoDataFrame({"a": [1]}, geometry=[nested], crs=CRS))
    assert len(out) == 2 and set(out.geom_type) == {"Polygon"}
    assert out.area.sum() == pytest.approx(200.0)


def test_resolve_overlaps_gives_the_ground_to_the_smaller_polygon():
    big, small = box(0, 0, 100, 100), box(50, 50, 80, 80)
    geoms, trimmed = fl.resolve_overlaps(_frame([big, small]))
    assert trimmed == 1
    assert geoms[1].area == pytest.approx(small.area)                     # the finer polygon is untouched
    assert geoms[0].area == pytest.approx(big.area - small.area)          # the coarser one is cut around it
    assert geoms[0].intersection(geoms[1]).area == pytest.approx(0.0)


def test_resolve_overlaps_ignores_neighbours_that_only_share_an_edge():
    """Touching polygons have no ground in common; asking the index for them costs memory for nothing."""
    frame = _frame([box(0, 0, 100, 100), box(100, 0, 200, 100), box(0, 0, 100, 100)])   # neighbour + duplicate
    geoms, trimmed = fl.resolve_overlaps(frame)
    assert trimmed == 1                                                    # only the duplicate pair is cut
    assert geoms[1].area == pytest.approx(10_000.0)                        # the neighbour is untouched
    assert {round(geoms[0].area), round(geoms[2].area)} == {10_000, 0}     # one copy keeps the ground


def test_resolve_overlaps_honours_a_priority_that_is_not_area():
    """`tidy` ranks derived polygons behind traced ones whatever their size."""
    traced, derived = box(0, 0, 100, 100), box(50, 50, 80, 80)
    frame = _frame([traced, derived])
    geoms, _ = fl.resolve_overlaps(frame, priority=[1.0, 2.0])            # traced wins despite being larger
    assert geoms[0].area == pytest.approx(traced.area)
    assert geoms[1].area == pytest.approx(derived.area - 900.0)


def test_despike_removes_a_tail_and_keeps_the_corners():
    body = box(0, 0, 100, 100)
    tail = box(100, 48, 160, 52)                                          # 4 m wide, 60 m long
    frame = gpd.GeoSeries([body.union(tail)], crs=CRS)
    out = fl.despike(frame)
    assert out.iloc[0].area == pytest.approx(body.area, rel=0.02)
    assert out.iloc[0].bounds[2] < 110                                    # the tail is gone
    assert out.iloc[0].contains(Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]))  # a corner survives


def test_despike_drops_a_ribbon_entirely():
    ribbon = gpd.GeoSeries([box(0, 0, 200, 5)], crs=CRS)                   # 5 m wide: no body at all
    assert fl.despike(ribbon).iloc[0] is None


def test_despike_leaves_a_compact_polygon_untouched():
    square = gpd.GeoSeries([box(0, 0, 100, 100)], crs=CRS)
    assert fl.despike(square).iloc[0].equals(square.iloc[0])


def test_field_like_rejects_needles_and_accepts_fields():
    assert fl.field_like(box(0, 0, 100, 100))
    assert not fl.field_like(box(0, 0, 200, 10))                           # narrower than the field width
    assert fl.field_like(box(0, 0, 200, 10), min_width=5.0)                # ... unless a narrower field counts


def test_drop_lines_removes_line_shapes_and_tiny_bodies():
    frame = _frame([box(0, 0, 100, 100),          # a field
                    box(0, 0, 400, 8),            # a line: compactness far below the floor
                    box(0, 0, 10, 10)])           # 100 m2, under the body floor of 0.05 acres
    out, stats = fl.drop_lines(frame)
    assert len(out) == 1 and out.geometry.iloc[0].area == pytest.approx(10_000.0)
    assert stats["dropped_as_lines"] == 1 and stats["dropped_as_too_small"] == 1


def test_repair_reports_acreage_and_removes_overlap():
    fields = gpd.GeoDataFrame({"a": [1, 2]}, geometry=[box(0, 0, 100, 100), box(50, 0, 150, 100)], crs=CRS)
    out, stats = fl.repair(fields, report_coverage=True)
    assert stats["polygons_in"] == 2
    assert fl.repair(fields)[1]["acres_of_ground"] is None          # the dissolve is off by default
    assert stats["acres_of_ground"] == pytest.approx(15_000 / fl.SQM_PER_ACRE, abs=0.05)   # rounded in the summary
    assert out.area.sum() == pytest.approx(15_000.0)                       # overlapping ground counted once
    assert out.geometry.iloc[0].intersection(out.geometry.iloc[1]).area == pytest.approx(0.0)


# ---------------------------------------------------------------- cutting

def test_orientation_follows_the_long_axis():
    assert fl.orientation(box(0, 0, 100, 10)) == pytest.approx(0.0, abs=1e-6)
    assert abs(fl.orientation(box(0, 0, 10, 100))) == pytest.approx(np.pi / 2, abs=1e-6)


def test_best_cut_finds_the_boundary_between_two_classes():
    xs = np.repeat(np.arange(40, dtype=float) * 10 + 5, 4)
    ys = np.tile(np.arange(4, dtype=float) * 10 + 5, 40)
    cls = (xs > 200).astype(int)                                           # clean split at x = 200
    angle, offset, purity = fl.best_cut(xs, ys, cls, 2, (0.0, np.pi / 2))
    assert purity == pytest.approx(1.0)
    assert angle == pytest.approx(0.0) and 195 <= offset <= 210


def test_best_cut_refuses_when_there_is_nothing_to_separate():
    xs = np.arange(40, dtype=float) * 10
    ys = np.zeros(40)
    cls = np.zeros(40, dtype=int)
    _, _, purity = fl.best_cut(xs, ys, cls, 2, (0.0,))
    assert purity == pytest.approx(1.0)                                    # already pure: a cut adds nothing
    assert fl.best_cut(xs[:5], ys[:5], cls[:5], 2, (0.0,))[2] == 0.0       # too few pixels to cut at all


def test_halves_covers_the_polygon_without_overlap():
    geom = box(0, 0, 400, 100)
    left, right = fl.halves(geom, 0.0, 200.0)
    assert left.area + right.area == pytest.approx(geom.area)
    assert left.intersection(right).area == pytest.approx(0.0)


# ---------------------------------------------------------------- raster passes

def test_zonal_counts_and_claimed_mask(tmp_path):
    array = np.ones((10, 10), dtype="uint8")
    array[:, 5:] = 2                                                       # left half class 1, right half class 2
    path = _raster(tmp_path, array)
    frame = _frame([box(0, 900, 50, 1000), box(50, 900, 100, 1000)])       # top-left and top-right quarters
    claimed = np.zeros(array.shape, dtype="uint8")
    z = fl.zonal_counts(frame, path, None, CODES, claimed=claimed)
    assert z["counts"][1].tolist()[:2] == [50, 0]
    assert z["counts"][2].tolist()[:2] == [0, 50]
    assert claimed.sum() == 100 and claimed[:, :].all()


def test_collect_pixels_returns_each_polygons_pixels(tmp_path):
    array = np.ones((10, 10), dtype="uint8")
    array[:, 5:] = 2
    path = _raster(tmp_path, array)
    frame = _frame([box(0, 900, 100, 1000)])
    got = fl.collect_pixels(frame, path, np.array([1]), CODES)
    xs, ys, cls = got[1]
    assert xs.size == 100 and set(np.unique(cls)) == {0, 1}
    assert cls[xs < 50].tolist() == [0] * 50


def _split_raster(tmp_path):
    """40 x 40 pixels of 10 m covering x 0-400, y 0-400: class 1 left of x = 200, class 2 right of it."""
    array = np.ones((40, 40), dtype="uint8")
    array[:, 20:] = 2
    return _raster(tmp_path, array, origin=(0.0, 400.0), res=10.0)


def test_label_polygons_keeps_clean_and_cuts_mixed(tmp_path):
    path = _split_raster(tmp_path)
    clean = box(0, 300, 100, 400)                                          # entirely class 1
    mixed = box(100, 100, 300, 300)                                        # half class 1, half class 2
    frame = _frame([clean, mixed])
    zonal = fl.zonal_counts(frame, path, None, CODES)
    out, tally = fl.label_polygons(frame, zonal, path, NAMES, CODES, target="Crop A")
    assert tally["clean"] == 1 and tally["cut"] == 1
    assert (out.origin == "split").sum() == 2
    split = out[out.origin == "split"]
    assert set(split.majority_class) == {"Crop A", "Crop B"}
    assert (split.majority_share == 1.0).all()                             # each side holds one class only
    assert split.geometry.area.sum() == pytest.approx(mixed.area)          # the cut loses no ground
    assert out[out.origin == "delineation"].decision.iloc[0] == "clean"


def test_label_polygons_cuts_a_mixed_polygon_that_contains_unclassified_pixels(tmp_path):
    """Nodata pixels inside a mixed polygon must not reach the cut search (they carry an extra class index)."""
    array = np.ones((40, 40), dtype="uint8")
    array[:, 20:] = 2
    array[12:16, 12:16] = 0                                                # a nodata patch inside the polygon
    path = _raster(tmp_path, array, origin=(0.0, 400.0), res=10.0)
    frame = _frame([box(100, 100, 300, 300)])
    out, tally = fl.label_polygons(frame, fl.zonal_counts(frame, path, None, CODES), path, NAMES, CODES, "Crop A")
    assert tally["cut"] == 1
    assert set(out[out.origin == "split"].majority_class) == {"Crop A", "Crop B"}


def test_label_polygons_flags_polygons_with_too_few_pixels(tmp_path):
    path = _split_raster(tmp_path)
    frame = _frame([box(0, 380, 20, 400)])                                 # 2 x 2 pixels
    out, tally = fl.label_polygons(frame, fl.zonal_counts(frame, path, None, CODES), path, NAMES, CODES, "Crop A")
    assert tally["too_few_pixels"] == 1
    assert out.decision.iloc[0] == "too few pixels to judge"
    assert out.geometry.iloc[0].equals(frame.geometry.iloc[0])             # the geometry is never changed


def test_label_polygons_keeps_a_polygon_whole_when_a_cut_would_leave_a_sliver(tmp_path):
    path = _split_raster(tmp_path)
    frame = _frame([box(100, 50, 300, 70)])                                # 200 m x 20 m across the boundary
    out, tally = fl.label_polygons(frame, fl.zonal_counts(frame, path, None, CODES), path, NAMES, CODES, "Crop A")
    assert tally["cut"] == 0 and tally["cut_sliver"] == 1 and len(out) == 1
    assert out.decision.iloc[0] == "mixed, the cut would leave a sliver"
    assert out.geometry.iloc[0].equals(frame.geometry.iloc[0])


def test_derived_polygons_keeps_a_field_and_rejects_a_ribbon(tmp_path):
    array = np.zeros((40, 40), dtype="uint8")
    array[5:15, 5:15] = 1                                                  # 100 m x 100 m block: a field
    array[30:31, 0:40] = 1                                                 # 10 m x 400 m ribbon
    path = _raster(tmp_path, array, origin=(0.0, 400.0))
    claimed = np.zeros(array.shape, dtype="uint8")
    out, stats = fl.derived_polygons(claimed, path, NAMES, CODES, CRS, target="Crop A")
    assert stats["blocks"] == 2 and len(out) == 1
    assert out.majority_class.iloc[0] == "Crop A"
    assert out.area.iloc[0] == pytest.approx(10_000.0, rel=0.1)
    assert (out.origin == "derived").all()


def test_derived_polygons_ignores_claimed_ground(tmp_path):
    array = np.zeros((40, 40), dtype="uint8")
    array[5:15, 5:15] = 1
    path = _raster(tmp_path, array, origin=(0.0, 400.0))
    claimed = np.ones(array.shape, dtype="uint8")
    out, stats = fl.derived_polygons(claimed, path, NAMES, CODES, CRS, target="Crop A")
    assert len(out) == 0 and stats["unclaimed_acres"] == 0.0


def test_tidy_leaves_no_overlap_and_prefers_traced_geometry():
    traced = box(0, 0, 100, 100)
    derived = box(50, 50, 150, 150)
    frame = gpd.GeoDataFrame({"origin": ["delineation", "derived"], "majority_class": ["Crop A", "Crop B"]},
                             geometry=[traced, derived], crs=CRS)
    out = fl.tidy(frame, out_crs=CRS)
    assert len(out) == 2
    kept = out.set_index("origin").geometry
    assert kept["delineation"].area == pytest.approx(traced.area, rel=1e-6)
    assert kept["derived"].area == pytest.approx(derived.area - 2500.0, rel=1e-6)
    assert kept["delineation"].intersection(kept["derived"]).area == pytest.approx(0.0, abs=1e-6)
