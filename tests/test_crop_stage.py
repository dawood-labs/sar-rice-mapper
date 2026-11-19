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


def test_deliver_adds_columns_and_keeps_geometry(tmp_path, monkeypatch):
    import json

    import pandas as pd

    qc = tmp_path / "AOI_9.gpkg"
    g = gpd.GeoDataFrame({"field_id": ["x_p000001", None], "acres": [1.0, 2.0]},
                         geometry=[box(0, 0, 10.123456789, 10), box(20, 0, 30, 10)], crs=32646)
    g.to_file(qc, driver="GPKG")
    d = tmp_path / "out"
    d.mkdir()
    (d / "inputs.json").write_text(json.dumps({"qc": str(qc), "run": "r"}))
    pd.DataFrame({"field_id": ["x_p000001", "x_qc001"], "status": ["standing", "harvested"],
                  "sowing_median": ["2026-07-31", "2026-07-16"], "age_median": [107.0, 110.0],
                  "harvest_from": [None, "2026-11-02"], "harvest_to": [None, "2026-11-05"]}).to_csv(d / "field_stage.csv", index=False)
    monkeypatch.setattr(cs, "out_dir", lambda aoi: d)
    out = cs.deliver(9)
    r = gpd.read_file(out)
    assert r.geometry.to_wkb().tolist() == gpd.read_file(qc).geometry.to_wkb().tolist()
    assert list(r["status"]) == ["standing", "harvested"]
    assert r["crop_age"].iloc[0] == 107                     # standing: up to the reference day
    assert r["crop_age"].iloc[1] == 112                     # harvested: 16 July up to its harvest date, 5 November
    assert str(r["harvest_date"].iloc[1])[:10] == "2026-11-05" and pd.isna(r["field_id"].iloc[1])


def test_field_status_ripening_stays_standing_cut_is_harvested():
    import numpy as np

    days = np.arange(0, 200, 5, dtype=float)
    rise = np.clip((days - 40) / 40, 0, 1)
    standing = 0.1 + 0.7 * rise - np.where(days > 150, (days - 150) * 0.001, 0)    # slow ripening fade
    cut = standing.copy()
    cut[days >= 185] = 0.35                                                          # cut on day 185
    assert cs.field_status(days, standing)["status"] == "standing"
    r = cs.field_status(days, cut)
    assert r["status"] == "harvested" and r["harvest_to"] == 185


def test_month_week():
    assert cs.month_week("2026-07-31") == "2026-07-29 to 07-31 (Jul week 5)"
    assert cs.month_week("2026-11-05") == "2026-11-01 to 11-07 (Nov week 1)"
    assert cs.month_week("2026-08-08") == "2026-08-08 to 08-14 (Aug week 2)"
    assert cs.month_week(None) is None
    assert cs.month_week("2026-07-15", 1) == "2026-07-08 to 07-14 (Jul week 2)"     # the day before the pass
    assert cs.month_week("2026-08-08", 1) == "2026-08-01 to 08-07 (Aug week 1)"
    days = ["2026-06-25", "2026-07-03", "2026-07-15", "2026-07-31", "2026-08-08", "2026-09-05", "2026-11-15"]
    labels = [cs.month_week(d) for d in days]
    assert labels == sorted(labels)                     # text order = time order


def test_prepare_fields_cuts_overlap_and_gives_unique_ids(tmp_path):
    src = tmp_path / "f.gpkg"
    g = gpd.GeoDataFrame({"field_id": ["x_p000001", "x_p000001", None], "major_class": "rice", "sub_class": "rice",
                          "origin": ["delineation", "split", None]},
                         geometry=[box(0, 0, 100, 100), box(90, 0, 150, 100), box(300, 0, 400, 100)], crs=32646)
    g.to_file(src, driver="GPKG")
    out, rep = cs.prepare_fields(src)
    f = gpd.read_file(out)
    assert rep["overlapping_pairs"] == 0 and rep["touching_pairs"] == 0
    assert sorted(f["field_id"]) == ["x_p000001", "x_p000001_b", "x_qc001"]
    assert f.loc[f["field_id"] == "x_p000001", "area_ac"].iloc[0] > f.loc[f["field_id"] == "x_p000001_b", "area_ac"].iloc[0]
    assert list(f["fid_no"]) == [1, 2, 3]
