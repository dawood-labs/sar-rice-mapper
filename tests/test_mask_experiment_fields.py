"""field_changes: majority class per field under two rule maps (synthetic rasters, no real data)."""
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from sar_pipeline import qgis_review
from sar_pipeline.analysis import mask_experiment as me


def _write_map(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="uint8",
                       crs="EPSG:32646", transform=from_origin(0, 100, 10, 10)) as ds:
        ds.write(arr, 1)


def test_field_changes_reports_majority_per_mask(tmp_path, monkeypatch):
    base = tmp_path / "s2"
    monkeypatch.setattr(me, "BASE", str(base))
    monkeypatch.setattr(qgis_review, "SRC", str(base))
    ctl = np.zeros((10, 10), dtype="uint8")
    ctl[:, :5] = 2                                   # field A young under the control
    var = ctl.copy()
    var[:, :5] = 6                                   # ... young rice under the variant
    _write_map(base / "aoi1" / "aoi1_monsoon2026.tif", ctl)
    _write_map(tmp_path / "s2_soft" / "aoi1" / "aoi1_monsoon2026.tif", var)
    (base / "fields").mkdir()
    gpd.GeoDataFrame({"field_id": ["aoi1_000001", "aoi1_000002"], "label": [2, 0]},
                     geometry=[box(0, 0, 50, 100), box(50, 0, 100, 100)], crs="EPSG:32646").to_file(
        base / "fields" / "aoi1_fields_monsoon2026.gpkg")
    t = me.field_changes(1, "soft")
    assert t.set_index("field_id")["delivered"].to_dict() == {"aoi1_000001": 2, "aoi1_000002": 0}
    assert t.set_index("field_id")["soft"].to_dict() == {"aoi1_000001": 6, "aoi1_000002": 0}
    assert np.isclose(t["acres"].sum(), 100 * 0.0247105)


def test_compare_delivered_reads_acres_and_noted_labels(tmp_path, monkeypatch):
    base = tmp_path / "s2"
    monkeypatch.setattr(me, "BASE", str(base))
    cols = {f"{c}_acres_fieldmap": 0.0 for c in me.FIELD_CLASSES}
    for folder, rice in ((base / "baseline_v4" / "fields", 10.0), (tmp_path / "s2_sb" / "fields", 12.0)):
        folder.mkdir(parents=True)
        pd.DataFrame([{"aoi": "aoi1", **cols, "rice_acres_fieldmap": rice}]).to_csv(folder / "field_acres_by_class.csv", index=False)
        gpd.GeoDataFrame({"field_id": ["aoi1_000001"], "label": [0 if rice == 10.0 else 6]}, geometry=[box(0, 0, 1, 1)],
                         crs="EPSG:32646").to_file(folder / "aoi1_fields_monsoon2026.gpkg")
    acres, noted = me.compare_delivered([1], "sb", noted=["aoi1_000001"])
    assert acres.loc[0, "delivered"] == "10 -> 12"
    assert noted.to_dict("records") == [{"field_id": "aoi1_000001", "old": 0, "new": 6}]
