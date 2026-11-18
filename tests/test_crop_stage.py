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
