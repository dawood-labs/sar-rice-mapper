import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from sar_pipeline import mask_audit as ma


def test_date_table_reports_each_test(tmp_path, monkeypatch):
    names = ["B2", "B4", "B8", "QA60", "SCL", "clear"]
    d = tmp_path / "aoi7"
    d.mkdir()
    for day, b8, qa in (("2026-09-01", 3000, 0), ("2026-09-10", 0, 1024)):
        vals = [500, 500, b8, qa, 6, 80]
        with rasterio.open(d / f"aoi7_2026-09_S2_{day}.tif", "w", driver="GTiff", height=2, width=2, count=6,
                           dtype="uint16", crs="EPSG:32646", transform=from_origin(0, 20, 10, 10)) as ds:
            for i, (n, v) in enumerate(zip(names, vals), 1):
                ds.write(np.full((2, 2), v, "uint16"), i)
                ds.set_band_description(i, n)
    monkeypatch.setattr("sar_pipeline.qgis_review.field_by_id", lambda f: (7, None, np.array([0, 1])))
    t = ma.date_table("aoi7_000001", "2026-08-15", s2_dir=str(tmp_path), map_date="2026-09-05")
    assert t["no_data_b8_or_b4_zero"].tolist() == [0.0, 1.0]
    assert t["qa60_opaque"].tolist() == [0.0, 1.0]
    assert t["kept"].tolist() == [1.0, 0.0]
    assert t["after_map_date"].tolist() == [False, True]


def test_summarise_events_types():
    ev = pd.DataFrame({"trough_ndvi": [0.1, 0.3, 0.2], "standing": [True, True, False],
                       "flood_date": pd.to_datetime(["2026-06-13", "2026-06-13", None])})
    s = ma.summarise_events(ev)
    assert s["trough_ndvi"] == 0.2 and s["standing"] == "0.67 share" and str(s["flood_date"])[:10] == "2026-06-13"


def test_field_ndvi_on_gives_raw_median_per_field(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box

    names = ["B2", "B4", "B8", "QA60", "SCL", "clear"]
    d = tmp_path / "s2" / "aoi7"
    d.mkdir(parents=True)
    b8 = np.array([[3000, 3000, 600, 600]] * 2, dtype="uint16")      # left half canopy, right half bare
    with rasterio.open(d / "aoi7_2026-09_S2_2026-09-26.tif", "w", driver="GTiff", height=2, width=4, count=6,
                       dtype="uint16", crs="EPSG:32646", transform=from_origin(0, 20, 10, 10)) as ds:
        for i, n in enumerate(names, 1):
            ds.write(b8 if n == "B8" else np.full((2, 4), {"B2": 300, "B4": 500, "QA60": 0, "SCL": 4, "clear": 0}[n], "uint16"), i)
            ds.set_band_description(i, n)
    (tmp_path / "fields").mkdir()
    gpd.GeoDataFrame({"field_id": ["aoi7_000001", "aoi7_000002", "aoi7_000003"], "label": [6, 6, 1]},
                     geometry=[box(0, 0, 20, 20), box(20, 0, 40, 20), box(0, 0, 40, 20)], crs="EPSG:32646").to_file(
        tmp_path / "fields" / "aoi7_fields_monsoon2026.gpkg")
    monkeypatch.setattr("sar_pipeline.qgis_review.SRC", str(tmp_path))
    t = ma.field_ndvi_on(7, "2026-09-26", labels=(6,), s2_dir=str(tmp_path / "s2")).set_index("field_id")
    assert list(t.index) == ["aoi7_000001", "aoi7_000002"]                 # only the requested label
    assert t.loc["aoi7_000001", "ndvi_median"] > 0.7 and t.loc["aoi7_000002", "ndvi_median"] < 0.2
    assert (t["pixels"] == 4).all() and (t["cs_median"] == 0).all()        # Cloud Score+ is reported, never used


def test_pixels_of_reads_a_single_pixel_target():
    aoi, pids = ma.pixels_of("aoi160:45142")
    assert aoi == 160 and pids.tolist() == [45142]
