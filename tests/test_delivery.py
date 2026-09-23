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
