import numpy as np
import rasterio
from rasterio.transform import from_origin

from sar_pipeline import delivery


def test_package_aoi_writes_colours_names_and_acres(tmp_path):
    src = tmp_path / "src" / "aoi7"
    src.mkdir(parents=True)
    data = np.array([[1, 1, 3], [5, 0, 255]], dtype="uint8")
    with rasterio.open(src / "aoi7_monsoon2026.tif", "w", driver="GTiff", width=3, height=2, count=1,
                       dtype="uint8", crs="EPSG:32633", transform=from_origin(500000, 4000000, 10, 10),
                       nodata=255) as ds:
        ds.write(data, 1)
    row = delivery.package_aoi("aoi7", tmp_path / "out", "2026-09-19", src_root=tmp_path / "src")
    acre = 100 / 4046.8564224
    assert row["rice, standing, water confirmed (acres)"] == round(2 * acre, 1)
    with rasterio.open(tmp_path / "out" / "aoi7_standing_rice_2026-09-19.tif") as ds:
        assert ds.tags()["class_1"].startswith("rice")
        assert ds.colormap(1)[1][:3] == (0, 140, 60)


def test_class9_has_legend_colour_and_acres(tmp_path):
    """Class 9 (rice-like, no sign of water) is described, coloured and counted, but not delivered as rice."""
    from sar_pipeline import qgis_review
    from sar_pipeline.analysis import monsoon_rule as mr

    assert set(delivery.LEGEND) == set(qgis_review.COLOURS) == set(mr.CLASSES) - {255}
    colours = [v[2] for v in delivery.LEGEND.values()]
    assert len(set(colours)) == len(colours)               # every class its own colour
    assert "reported, not delivered" in delivery.LEGEND[9][1]
    assert 9 in delivery.legend_table()["code"].tolist()
    src = tmp_path / "src" / "aoi7"
    src.mkdir(parents=True)
    with rasterio.open(src / "aoi7_monsoon2026.tif", "w", driver="GTiff", width=3, height=1, count=1,
                       dtype="uint8", crs="EPSG:32633", transform=from_origin(500000, 4000000, 10, 10),
                       nodata=255) as ds:
        ds.write(np.array([[9, 9, 1]], dtype="uint8"), 1)
    row = delivery.package_aoi("aoi7", tmp_path / "out", "2026-09-19", src_root=tmp_path / "src")
    acre = 100 / 4046.8564224
    assert row["rice-like, no sign of water (acres)"] == round(2 * acre, 1)
    with rasterio.open(tmp_path / "out" / "aoi7_standing_rice_2026-09-19.tif") as ds:
        assert ds.colormap(1)[9][:3] == (0, 150, 150)
        assert ds.tags()["class_9"] == "rice-like, no sign of water"


def test_deliver_fields_drops_merged_slivers_and_dropped_monsters(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    from sar_pipeline.delivery import deliver_fields

    g = gpd.GeoDataFrame({"field_id": ["a", "b", "c", "d", "e"], "label": [1, 1, 3, 0, 0],
                          "is_field": [True, False, False, False, False],
                          "refine_flag": ["", "merged", "strip", "monster_dropped", "pond"]},
                         geometry=[box(i * 20, 0, i * 20 + 10, 10) for i in range(5)], crs="EPSG:32630")
    src = tmp_path / "in.gpkg"
    g.to_file(src, layer="fields", driver="GPKG")
    n = deliver_fields(src, tmp_path / "out.gpkg")
    out = gpd.read_file(tmp_path / "out.gpkg")
    assert n == 3 and sorted(out["field_id"]) == ["a", "c", "e"]
