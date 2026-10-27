"""Tests for the third AOI set's per-polygon files (made-up shapefiles)."""
import geopandas as gpd
from shapely.geometry import box

from sar_pipeline.analysis import aoi_batch3 as ab


def test_prepare_numbers_polygons_in_zip_then_file_order(tmp_path):
    root = tmp_path / "in"
    for z, f, n in (("set_a", "B_AOI", 2), ("set_a", "A_AOI", 1),
                    ("pilot", "Pilot", 1)):
        d = root / z / z
        d.mkdir(parents=True, exist_ok=True)
        gpd.GeoDataFrame({"Company": [f] * n}, geometry=[box(96 + i, 17, 96.01 + i, 17.01) for i in range(n)],
                         crs=4326).to_file(d / f"{f}.shp")
    idx = ab.prepare(root, tmp_path / "out", first=239, zips=("set_a", "pilot"))
    assert list(idx["AOI_NUMBER"]) == [239, 240, 241, 242]
    assert list(idx["source_file"].str.split("/").str[-1]) == ["A_AOI.shp", "B_AOI.shp", "B_AOI.shp", "Pilot.shp"]
    g = gpd.read_file(tmp_path / "out" / "AOI_242" / "AOI_242.gpkg", layer="AOI_242")
    assert g["AOI_NUMBER"].iloc[0] == 242 and g["layer"].iloc[0] == "Pilot"
