"""Tests for the field pixel lookup of the crop-stage curves (made-up grid and fields)."""
import geopandas as gpd
from shapely.geometry import box

from sar_pipeline.analysis import crop_stage as cs


def test_field_pixels_groups_pieces_of_one_field(tmp_path):
    gd = {"crs": "EPSG:32646", "x0": 800000.0, "y0": 2000100.0, "res": 10.0, "width": 10, "height": 10}
    f = gpd.GeoDataFrame({"field_id": ["a", "a", "b"]},
                         geometry=[box(800000, 2000080, 800020, 2000100), box(800050, 2000080, 800060, 2000100),
                                   box(800000, 2000000, 800010, 2000010)], crs=32646)
    p = tmp_path / "qc.gpkg"
    f.to_file(p)
    pix, table = cs.field_pixels(p, gd)
    assert len(pix["a"]) == 6 and len(pix["b"]) == 1
    assert table.set_index("field_id").loc["a", "pixels"] == 6


def test_pixel_stage_reads_planting_from_the_dip_and_harvest_from_the_step():
    import numpy as np

    nd_days = np.array([100, 130, 160, 190, 206, 249, 256, 270, 284, 298, 306, 309, 311, 318, 319], dtype=float)
    nd_vals = np.array([.15, .14, .16, .15, .2, .65, .7, .78, .8, .79, .7, .4, .41, .38, .39])
    days = np.arange(105, 250, 12, dtype=float)            # 105 .. 249
    vh = np.array([-21, -21, -20, -19, -17, -14, -15, -18, -20, -20.5, -17, -15, -14.5])
    vv = np.array([-13, -13, -12, -11, -10, -6, -8, -10, -12, -12.5, -9, -8, -7])
    r = cs.pixel_stage(nd_days, nd_vals, [(days, vh, vv)], wet_from=165.0)
    assert r["status"] == "harvested" and (r["harvest_from"], r["harvest_to"]) == (306.0, 309.0)
    assert r["method"].startswith("transplanted") and r["sowing_day"] == 213.0
    standing = cs.pixel_stage(nd_days[:11], nd_vals[:11], [(days, vh, vv)], wet_from=165.0)
    assert standing["status"] == "standing"
