"""``qc_sowing.detach`` drops only the sowing columns; the geometries stay byte for byte the same."""
import json

import geopandas as gpd
from shapely.geometry import Polygon

from sar_pipeline.analysis import qc_sowing as qs


def test_detach_drops_sowing_columns_and_keeps_geometry(tmp_path, monkeypatch):
    d = tmp_path / "first_set" / "aoi4"
    d.mkdir(parents=True)
    path = d / "aoi4_fields_qc.gpkg"
    g = gpd.GeoDataFrame({"field_id": ["a", "b"], "area_ac": [1.0, 2.0], "sowing_date": ["2026-07-01", None],
                          "crop_age_15sep": [76, None], "sowing_source": ["x", "y"], "sowing_pixels_pct": [90, 80],
                          "sowing_note": ["", ""]},
                         geometry=[Polygon([(0, 0), (10, 0), (10, 10.123456789), (0, 0)]),
                                   Polygon([(20, 0), (30, 0), (30, 10), (20, 0)])], crs=32646)
    g.to_file(path, driver="GPKG", layer=path.stem)
    path.with_suffix(".json").write_text(json.dumps({"aoi": 4, "sowing": {"fields": 2}}))
    before = gpd.read_file(path).geometry.to_wkb().tolist()
    monkeypatch.setattr(qs, "out_path", lambda aoi: path)
    monkeypatch.setattr(qs, "OUT", str(tmp_path))
    monkeypatch.setattr(qs, "BACKUP", tmp_path / "_backup")
    r = qs.detach(4)
    after = gpd.read_file(path)
    assert r == {"aoi": 4, "columns_dropped": 5, "report_entry_dropped": True}
    assert list(after.columns) == ["field_id", "area_ac", "geometry"]
    assert after.geometry.to_wkb().tolist() == before
    assert "sowing" not in json.loads(path.with_suffix(".json").read_text())
    assert "sowing_date" in gpd.read_file(tmp_path / "_backup" / "first_set" / "aoi4" / path.name).columns
