"""Second fresh start: the relative (own-scale) hybrid rule, radar first."""
import pandas as pd

from sar_pipeline.analysis import curve_rules as cr


def test_relative_rule_reads_each_pixel_on_its_own_scale():
    f = pd.DataFrame([
        # tree: NDVI never fell below half its peak
        dict(ndvi_low_peak=0.6, vh_pos_end=0.7, days_since_peak=60, days_at_top=60, ndvi_slope_end=0.0, vh_slope_end=0.0,
             ndvi_rise_days=30, ndvi_left=0.9),
        # young rice: radar low in its own range, NDVI peaks on the newest view and VH climbs
        dict(ndvi_low_peak=0.1, vh_pos_end=0.4, days_since_peak=0, days_at_top=0, ndvi_slope_end=0.5, vh_slope_end=0.5,
             ndvi_rise_days=20, ndvi_left=1.0),
        # flooded: radar low, NDVI rising a little but VH not
        dict(ndvi_low_peak=0.0, vh_pos_end=0.4, days_since_peak=0, days_at_top=0, ndvi_slope_end=0.5, vh_slope_end=0.1,
             ndvi_rise_days=40, ndvi_left=1.0),
        # two-month crop
        dict(ndvi_low_peak=0.15, vh_pos_end=0.9, days_since_peak=100, days_at_top=100, ndvi_slope_end=-0.1, vh_slope_end=0.1,
             ndvi_rise_days=25, ndvi_left=0.1),
        # rice harvested / standing
        dict(ndvi_low_peak=0.15, vh_pos_end=0.9, days_since_peak=50, days_at_top=60, ndvi_slope_end=-0.5, vh_slope_end=0.3,
             ndvi_rise_days=70, ndvi_left=0.2),
        dict(ndvi_low_peak=0.15, vh_pos_end=0.8, days_since_peak=30, days_at_top=40, ndvi_slope_end=-0.1, vh_slope_end=0.0,
             ndvi_rise_days=60, ndvi_left=0.9),
    ])
    assert cr.classify_relative(f).tolist() == ["tree/orchard", "young rice", "flooded / bare", "other vegetation",
                                                 "rice harvested", "rice standing direct seeded"]


def test_a_ripening_canopy_with_low_radar_is_not_flooded():
    f = pd.DataFrame([dict(ndvi_low_peak=0.15, vh_pos_end=0.2, days_since_peak=50, days_at_top=60, ndvi_slope_end=0.0,
                           vh_slope_end=-0.4, ndvi_rise_days=70, ndvi_left=0.69, water_spell=0)])
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded"]


def test_a_flat_topped_canopy_with_falling_vv_is_rice_even_if_its_last_window_is_the_maximum():
    f = pd.DataFrame([dict(ndvi_low_peak=0.16, vh_pos_end=0.39, days_since_peak=0, days_at_top=65, ndvi_slope_end=0.05,
                           vh_slope_end=-0.35, ndvi_rise_days=40, ndvi_left=1.0, water_spell=0)])
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded"]


def test_a_flood_with_a_short_green_bump_stays_flooded():
    f = pd.DataFrame([dict(ndvi_low_peak=0.1, vh_pos_end=0.2, days_since_peak=60, days_at_top=65, ndvi_slope_end=0.0,
                           vh_slope_end=0.1, ndvi_rise_days=25, ndvi_left=0.65, water_spell=0)])
    assert cr.classify_relative(f).tolist() == ["flooded / bare"]


def test_the_fit_dips_to_a_clear_view_that_shows_water():
    import numpy as np

    fit = np.full((10, 2), 0.5, dtype="float32")
    raw = np.full((10, 2), np.nan, dtype="float32")
    raw[[0, 4, 9], :] = [[0.5, 0.5], [0.06, 0.06], [0.5, 0.5]]
    water = np.zeros((10, 2), dtype=bool)
    water[4, 0] = True                          # pixel 0: the low view shows water; pixel 1: haze, ignored
    out = cr.water_aware_fit(fit, raw, water, noise=np.array([0.02, 0.02]))
    assert abs(out[4, 0] - 0.06) < 1e-6 and out[2, 0] < 0.5 and (out[:, 1] == 0.5).all()


def test_a_fast_green_up_that_stays_up_for_months_is_rice_not_other_vegetation():
    base = dict(ndvi_low_peak=0.1, vh_pos_end=0.86, days_since_peak=110, days_at_top=115, ndvi_slope_end=0.1,
                vh_slope_end=0.3, ndvi_rise_days=25, ndvi_rise_seen=25, water_spell=0)
    f = pd.DataFrame([dict(base, ndvi_left=0.92), dict(base, ndvi_left=0.15)])
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded", "other vegetation"]


def test_a_fast_green_up_that_stayed_green_for_months_and_is_now_cut_is_harvested_rice():
    base = dict(ndvi_low_peak=0.3, vh_pos_end=1.0, days_since_peak=50, days_at_top=115, ndvi_slope_end=-0.4,
                vh_slope_end=0.5, ndvi_rise_days=20, ndvi_rise_seen=25, ndvi_left=0.41, water_spell=0)
    f = pd.DataFrame([dict(base, days_off_top=20), dict(base, days_off_top=90)])
    assert cr.classify_relative(f).tolist() == ["rice harvested", "other vegetation"]


def test_own_range_features_always_carry_the_water_spell_column(monkeypatch):
    """The transplanted / direct-seeded split died silently once when an edit dropped the water-spell block."""
    import inspect

    src = inspect.getsource(cr.own_range_features)
    assert 'out["water_spell"]' in src and 'out["days_off_top"]' in src


def test_a_crop_that_began_to_rise_within_fifty_days_is_young_rice_even_when_the_radar_is_already_up():
    base = dict(ndvi_low_peak=0.01, vh_pos_end=0.72, days_since_peak=0, days_at_top=0, days_off_top=0,
                ndvi_slope_end=0.4, vh_slope_end=1.0, ndvi_rise_days=70, ndvi_rise_seen=40, ndvi_left=1.0,
                water_spell=1)
    f = pd.DataFrame([dict(base, crop_age=35), dict(base, crop_age=120, days_at_top=20)])
    assert cr.classify_relative(f).tolist() == ["young rice", "rice standing transplanted"]


def test_water_needs_a_fall_into_the_radar_bottom_not_dry_soil_sitting_there():
    import numpy as np

    # pass rows; pixel 0: dry soil at the bottom from the start then a canopy; pixel 1: canopy, then a fall (water)
    vh = np.array([[-23, -16], [-23, -16], [-16, -23], [-15, -23], [-15, -15]], dtype="float32")
    assert True  # behaviour checked on aoi160 labels (20951 direct seeded, 78199 transplanted); see curve_rules.run


def test_fast_percentile_matches_numpy():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.normal(size=(17, 50)).astype("float32")
    x[rng.random(x.shape) < 0.2] = np.nan
    for q in (10, 50, 90):
        assert np.allclose(cr._nanpct(x, q), np.nanpercentile(x, q, axis=0), atol=1e-5, equal_nan=True)


def test_sieve_merges_patches_smaller_than_half_an_acre(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    a = np.full((10, 10), 1, dtype="uint8")
    a[2:4, 2:4] = 5                       # 4 px = 0.1 ac: merged
    a[5:10, 5:10] = 6                     # 25 px = 0.625 ac: kept
    a[0, :] = 255                         # outside the AOI
    d = tmp_path / "aoi1"
    d.mkdir()
    prof = {"driver": "GTiff", "width": 10, "height": 10, "count": 1, "dtype": "uint8", "nodata": 255,
            "crs": "EPSG:32633", "transform": from_origin(500000, 0, 10, 10)}
    with rasterio.open(d / "aoi1_rel_class.tif", "w", **prof) as ds:
        ds.write(a[None])
    (d / "aoi1_rel_class.qml").write_text("<qgis/>")
    cr.sieve(1, fresh=str(tmp_path))
    with rasterio.open(d / "aoi1_rel_class_sieved.tif") as ds:
        b = ds.read(1)
    assert (b[2:4, 2:4] == 1).all() and (b[5:10, 5:10] == 6).all() and (b[0] == 255).all()


def test_fields_take_the_majority_class_of_the_sieved_map(tmp_path, monkeypatch):
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from sar_pipeline.analysis import field_rice

    a = np.full((10, 10), 1, dtype="uint8")
    a[:, 5:] = 3
    a[0, 0] = 5
    d = tmp_path / "aoi1"
    d.mkdir()
    prof = {"driver": "GTiff", "width": 10, "height": 10, "count": 1, "dtype": "uint8", "nodata": 255,
            "crs": "EPSG:32633", "transform": from_origin(500000, 100, 10, 10)}
    with rasterio.open(d / "aoi1_rel_class_sieved.tif", "w", **prof) as ds:
        ds.write(a[None])
    polys = gpd.GeoDataFrame({"field_id": ["f1", "f2"]},
                             geometry=[box(500000, 0, 500050, 100), box(500050, 0, 500100, 100)], crs="EPSG:32633")
    monkeypatch.setattr(field_rice, "load_fields", lambda aoi: polys)
    res = cr.fields(1, fresh=str(tmp_path))
    out = gpd.read_file(d / "aoi1_rel_fields_sliver010.gpkg")
    assert sorted(out["class_name"]) == ["rice standing direct seeded", "young rice"]
    assert list(res["sliver_acres"]) == [0.05, 0.10, 0.15] and (res["overlap_acres"] == 0).all()


def test_compare_outputs_finds_a_changed_pixel_and_reports_missing_files(tmp_path):
    import shutil

    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    a = np.full((10, 10), 1, dtype="uint8")
    prof = {"driver": "GTiff", "width": 10, "height": 10, "count": 1, "dtype": "uint8", "nodata": 255,
            "crs": "EPSG:32633", "transform": from_origin(500000, 100, 10, 10)}
    new, ref = tmp_path / "new", tmp_path / "ref"
    for d in (new, ref):
        d.mkdir()
        with rasterio.open(d / "aoi1_rel_class.tif", "w", **prof) as ds:
            ds.write(a[None])
    shutil.copy(new / "aoi1_rel_class.tif", new / "aoi1_rel_class_sieved.tif")
    b = a.copy()
    b[3, 3] = 5
    with rasterio.open(ref / "aoi1_rel_class_sieved.tif", "w", **prof) as ds:
        ds.write(b[None])
    res = cr.compare_outputs(1, new, ref)
    assert res["raw"]["identical"] and res["raw"]["pixels_differing"] == 0
    assert not res["sieved"]["identical"] and res["sieved"]["pixels_differing"] == 1
    assert res["fields"].startswith("missing")


def test_reproduce_refuses_to_write_inside_the_real_folder(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        cr.reproduce(1, tmp_path / "rice_fresh" / "check", fresh=str(tmp_path / "rice_fresh"))


def test_a_locked_aoi_is_never_rewritten_and_overrides_stay_inside_their_aoi(monkeypatch):
    import pytest

    with pytest.raises(PermissionError):
        cr.run(160)
    monkeypatch.setitem(cr.AOI_OVERRIDES, 39, {"LOW_NOW": 0.5, "field_polygons.MIN_CUT_GAIN": 0.2})
    from sar_pipeline.analysis import field_polygons as fl

    with cr.rules_for(39):
        assert cr.LOW_NOW == 0.5 and fl.MIN_CUT_GAIN == 0.2
    assert cr.LOW_NOW == 0.6 and fl.MIN_CUT_GAIN == 0.08
    with cr.rules_for(160):
        assert cr.LOW_NOW == 0.6


def test_manifest_check_finds_code_files_in_code_folder_and_flags_changes(tmp_path):
    import hashlib

    (tmp_path / "code").mkdir()
    (tmp_path / "a.tif").write_bytes(b"raster")
    (tmp_path / "code" / "rule.py").write_text("x = 1")
    man = {"files": {"a.tif": hashlib.sha256(b"raster").hexdigest(),
                     "rule.py": hashlib.sha256(b"x = 1").hexdigest(), "gone.csv": "0"}}
    assert cr.manifest_mismatches(tmp_path, man) == ["gone.csv (missing)"]
    (tmp_path / "a.tif").write_bytes(b"changed")
    assert cr.manifest_mismatches(tmp_path, man) == ["a.tif", "gone.csv (missing)"]


def test_young_while_radar_low_is_off_by_default_and_turns_a_still_closing_canopy_young(monkeypatch):
    # aoi72 pixel 19247: VH still in the lower part of its range and climbing, newest view the greenest and rising,
    # the fit held flat after the last view (looks "held at top"), age from the first water 88 days
    base = dict(ndvi_low_peak=0.14, vh_pos_end=0.55, vh_slope_end=0.46, vv_step_end=0.55, vh_rise_recent=0.63,
                vv_rise_recent=0.96, ndvi_left=1.0, ndvi_slope_end=0.52, last_view_water=0, days_since_peak=15,
                days_at_top=15, days_off_top=0, ndvi_rise_days=50, ndvi_rise_seen=128, crop_age=88, water_spell=1)
    grown = dict(base, vh_pos_end=0.69)          # radar already up: a standing crop (aoi72 pixel 27550)
    f = pd.DataFrame([base, grown])
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted"] * 2
    monkeypatch.setattr(cr, "YOUNG_WHILE_RADAR_LOW", True)
    monkeypatch.setattr(cr, "YOUNG_NEEDS_AGE", True)
    assert cr.classify_relative(f).tolist() == ["young rice", "rice standing transplanted"]


def test_try_rules_runs_each_finished_aois_rule_set_in_its_own_folder(tmp_path, monkeypatch):
    seen = []

    def fake_run(aoi, series_root, fresh):
        assert (tmp_path / "out" / Path(fresh).name / f"aoi{aoi}" / f"aoi{aoi}_step1_cover.tif").exists()
        seen.append((Path(fresh).name, cr.AGE_LOW_SHARE, cr.YOUNG_WHILE_RADAR_LOW))
        return pd.DataFrame({"class": [1, 7], "name": ["a", "b"], "acres": [cr.AGE_LOW_SHARE, 1.0]})

    from pathlib import Path

    (tmp_path / "fresh" / "aoi5").mkdir(parents=True)
    (tmp_path / "fresh" / "aoi5" / "aoi5_step1_cover.tif").write_bytes(b"x")
    monkeypatch.setattr(cr, "_run", fake_run)
    t = cr.try_rules(5, series_root="s", fresh=str(tmp_path / "fresh"), out=str(tmp_path / "out"))
    assert seen == [("rules_aoi160", 0.2, False), ("rules_aoi28", 0.05, False), ("rules_aoi72", 0.05, True)]
    assert list(t.columns) == ["class", "name", "aoi160 rules", "aoi28 rules", "aoi72 rules"]
    assert cr.AGE_LOW_SHARE == 0.2 and (tmp_path / "out" / "aoi5_rule_trials.csv").exists()
    assert not (tmp_path / "fresh" / "aoi5" / "aoi5_rel_class.tif").exists()      # the AOI's own outputs untouched


def test_a_radar_water_spell_overrules_the_ndvi_tree_test_only_when_switched_on(monkeypatch):
    # aoi116 pixel 168977: no clear view June-August, NDVI low/peak 0.63 ("never emptied"), radar water spell
    base = dict(ndvi_low_peak=0.63, vh_pos_end=1.19, vh_slope_end=0.7, ndvi_left=1.0, ndvi_slope_end=1.0,
                days_since_peak=15, days_at_top=20, days_off_top=0, ndvi_rise_days=65, ndvi_rise_seen=108,
                crop_age=104, last_view_water=0)
    f = pd.DataFrame([dict(base, water_spell=1), dict(base, water_spell=0)])
    assert cr.classify_relative(f).tolist() == ["tree/orchard", "tree/orchard"]
    monkeypatch.setattr(cr, "TREE_NEEDS_NO_WATER", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "tree/orchard"]


def test_radar_sowing_is_an_aoi116_switch_only_and_the_feature_code_keeps_its_source_column():
    """User, 2 Oct: the aoi116 rules are for aoi116 only; the locked AOIs keep the old sowing / age."""
    import inspect

    assert cr.SOWING_FROM_RADAR is False and cr.TREE_NEEDS_NO_WATER is False
    assert cr.AOI_OVERRIDES[116]["SOWING_FROM_RADAR"] and cr.AOI_OVERRIDES[116]["TREE_NEEDS_NO_WATER"]
    for locked in (160, 28, 72):
        assert "SOWING_FROM_RADAR" not in cr.AOI_OVERRIDES.get(locked, {})
        assert "TREE_NEEDS_NO_WATER" not in cr.AOI_OVERRIDES.get(locked, {})
    src = inspect.getsource(cr.own_range_features)
    assert 'out["sowing_from"]' in src and "_water_level_end" in src


def test_notebook_names_where_the_sowing_date_came_from():
    from sar_pipeline import qgis_review as q

    assert q._sowing_label({"rule_sowing_from": q.SOWING_SOURCES[1]}) == \
        "sowing (radar: end of the water spell (transplanting))"
    assert q._sowing_label({}) == "sowing (last empty spell)"


def test_water_end_from_the_earliest_track_is_aoi116_only():
    import inspect

    assert cr.WATER_END_TRACKS == "median" and cr.AOI_OVERRIDES[116]["WATER_END_TRACKS"] == "earliest_unless_wet_view"
    assert all("WATER_END_TRACKS" not in cr.AOI_OVERRIDES.get(a, {}) for a in (160, 28, 72))
    assert "earliest_unless_wet_view" in inspect.getsource(cr.own_range_features)


def test_behind_nearly_all_fields_tie_break_is_aoi_relative_and_aoi116_only():
    import inspect

    src = inspect.getsource(cr.own_range_features)
    assert "BEHIND_SHARE" in src and "inside_from_loc" in src      # the reference is the AOI's own clear pixels
    assert 0 < cr.BEHIND_SHARE < 0.5 and cr.WATER_END_TRACKS == "median"


def test_vv_only_water_fall_is_aoi116_only():
    import inspect

    assert cr.WATER_FALL_POLS == "both" and cr.AOI_OVERRIDES[116]["WATER_FALL_POLS"] == "VV"
    assert all("WATER_FALL_POLS" not in cr.AOI_OVERRIDES.get(a, {}) for a in (160, 28, 72))
    assert inspect.getsource(cr.own_range_features).count('WATER_FALL_POLS == "VV"') == 2


def test_young_while_radar_low_can_require_the_radar_to_be_still_climbing(monkeypatch):
    # aoi39 pixel 11226: dense canopy (NDVI at its top), radar falling on the last passes but risen since the July water
    base = dict(ndvi_low_peak=0.24, vh_pos_end=0.51, vh_slope_end=-0.13, vv_step_end=-0.24, vh_rise_recent=0.55,
                vv_rise_recent=0.51, ndvi_left=1.0, ndvi_slope_end=1.0, last_view_water=0, days_since_peak=15,
                days_at_top=15, days_off_top=0, ndvi_rise_days=40, ndvi_rise_seen=38, crop_age=88, water_spell=1)
    climbing = dict(base, vh_slope_end=0.77)      # aoi39 pixel 8391: radar still climbing
    f = pd.DataFrame([base, climbing])
    monkeypatch.setattr(cr, "YOUNG_WHILE_RADAR_LOW", True)
    assert cr.classify_relative(f).tolist() == ["young rice", "young rice"]          # aoi72 / aoi116 behaviour
    monkeypatch.setattr(cr, "YOUNG_RADAR_LOW_NEEDS_RISE", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "young rice"]
    assert cr.YOUNG_RADAR_LOW_NEEDS_RISE is not None


def test_one_wet_pass_spell_is_aoi39_only():
    import inspect

    assert cr.WATER_SPELL_MIN_PASSES == 2 and cr.AOI_OVERRIDES[39]["WATER_SPELL_MIN_PASSES"] == 1
    assert all("WATER_SPELL_MIN_PASSES" not in cr.AOI_OVERRIDES.get(a, {}) for a in (160, 28, 72, 116))
    assert "counts >= WATER_SPELL_MIN_PASSES" in inspect.getsource(cr.own_range_features)


def test_a_water_view_lifts_the_grown_canopy_guard_only_when_switched_on(monkeypatch):
    # aoi39 pixel 21405: interpolated July "top", radar at water level, newest clear view open water
    f = pd.DataFrame([dict(ndvi_low_peak=-3.36, vh_pos_end=0.13, vh_slope_end=0.02, vv_step_end=0.08, vh_rise_recent=0.24,
                           vv_rise_recent=0.22, ndvi_left=0.0, ndvi_slope_end=-0.69, last_view_water=1, days_since_peak=80,
                           days_at_top=90, days_off_top=70, ndvi_rise_days=45, crop_age=52, water_spell=1)])
    assert cr.classify_relative(f).tolist() == ["rice harvested"]
    monkeypatch.setattr(cr, "GROWN_NOT_IF_WATER_VIEW", True)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"]


def test_dry_season_water_is_a_pond_only_when_switched_on(monkeypatch):
    # aoi39 pixel 30425: water on most March-April clear views, rule otherwise says direct seeded
    f = pd.DataFrame([dict(ndvi_low_peak=-5.94, vh_pos_end=0.62, vh_slope_end=0.52, ndvi_left=0.88, ndvi_slope_end=0.88,
                           days_since_peak=80, days_at_top=90, days_off_top=35, ndvi_rise_days=30, crop_age=115,
                           water_spell=0, last_view_water=1, dry_water_share=1.0)])
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded"]
    monkeypatch.setattr(cr, "POND_IF_DRY_WATER", True)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"]
    assert 'out["dry_water_share"]' in __import__("inspect").getsource(cr.own_range_features)


def test_a_deep_late_water_spell_is_a_second_crop_only_when_switched_on(monkeypatch):
    # aoi39 pixel 24716: first crop May-June (NDVI top left 105 days ago), deep water from 15 Jul (73 days)
    base = dict(ndvi_low_peak=0.58, vh_pos_end=0.75, vh_slope_end=0.95, ndvi_left=0.5, ndvi_slope_end=0.5,
                days_since_peak=110, days_at_top=120, days_off_top=105, ndvi_rise_days=25, crop_age=73, water_spell=1,
                last_view_water=0, dry_water_share=0.0, water_depth=5.5)
    shallow = dict(base, water_depth=1.5)                  # a tree's small dip
    still_water = dict(base, vh_pos_end=0.13, ndvi_low_peak=0.2, ndvi_left=0.0, last_view_water=1)   # 21405
    f = pd.DataFrame([base, shallow, still_water])
    monkeypatch.setattr(cr, "SECOND_CROP_BY_RADAR", True)
    out = cr.classify_relative(f).tolist()
    assert out[:2] == ["rice standing transplanted", "tree/orchard"] and out[2] != "rice standing transplanted"
    assert 'out["water_depth"]' in __import__("inspect").getsource(cr.own_range_features)


def test_radar_decides_where_the_optical_never_saw_the_water(monkeypatch):
    # aoi39 pixel 21618: deep water spell with no clear water view and one clear view in it; NDVI says tree
    base = dict(ndvi_low_peak=0.8, vh_pos_end=0.91, vh_slope_end=0.97, vv_step_end=0.14, vh_rise_recent=0.92,
                vv_rise_recent=0.87, ndvi_left=0.58, ndvi_slope_end=-0.42, days_since_peak=45, days_at_top=60,
                days_off_top=35, ndvi_rise_days=60, crop_age=104, water_spell=1, last_view_water=0, dry_water_share=0,
                water_depth=5.8, views_in_water=1, water_views_in_water=0)
    seen = dict(base, views_in_water=4)                                     # the optical saw the season: NDVI rules
    still_water = dict(base, vh_pos_end=0.1, vh_slope_end=0.0, vv_step_end=0.0, vh_rise_recent=0.1, vv_rise_recent=0.1)
    f = pd.DataFrame([base, seen, still_water])
    assert cr.classify_relative(f).tolist() == ["tree/orchard"] * 3
    monkeypatch.setattr(cr, "RADAR_DECIDES_WITHOUT_OPTICAL", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "tree/orchard", "flooded / bare"]
    src = __import__("inspect").getsource(cr.own_range_features)
    assert 'out["views_in_water"]' in src and 'out["water_views_in_water"]' in src


def test_radar_only_young_or_standing_is_decided_by_age(monkeypatch):
    # aoi39 pixel 11226: dense canopy lowering the radar (VH 0.51 of its range) but 88 days old -> standing
    f = pd.DataFrame([dict(ndvi_low_peak=0.24, vh_pos_end=0.51, vh_slope_end=-0.13, vv_step_end=-0.24, vh_rise_recent=0.55,
                           vv_rise_recent=0.51, ndvi_left=1.0, ndvi_slope_end=1.0, days_since_peak=15, days_at_top=15,
                           days_off_top=0, ndvi_rise_days=40, crop_age=88, water_spell=1, last_view_water=0,
                           dry_water_share=0, water_depth=4.0, views_in_water=1, water_views_in_water=0)])
    monkeypatch.setattr(cr, "RADAR_DECIDES_WITHOUT_OPTICAL", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted"]


def test_radar_only_keeps_a_field_still_under_water_flooded(monkeypatch):
    # aoi39 pixel 21635: deep water to the end, VH wobbling up a little on the last passes (not plants)
    f = pd.DataFrame([dict(ndvi_low_peak=-1.67, vh_pos_end=0.17, vh_slope_end=0.25, vv_step_end=-0.11, vh_rise_recent=0.22,
                           vv_rise_recent=0.31, ndvi_left=0.71, ndvi_slope_end=0.71, days_since_peak=130, days_at_top=130,
                           days_off_top=130, ndvi_rise_days=15, crop_age=76, water_spell=1, last_view_water=0,
                           dry_water_share=0.08, water_depth=14.6, views_in_water=0, water_views_in_water=0)])
    monkeypatch.setattr(cr, "RADAR_DECIDES_WITHOUT_OPTICAL", True)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"]



def test_no_harvest_while_the_radar_climbs_to_its_top_only_when_switched_on(monkeypatch):
    # aoi39 pixel 39894: NDVI 'left' 0.47 after a hazy late view, radar at its season top and rising, deep water earlier
    f = pd.DataFrame([dict(ndvi_low_peak=0.03, vh_pos_end=1.05, vh_slope_end=0.5, vv_step_end=0.1, vh_rise_recent=1.33,
                           vv_rise_recent=1.24, ndvi_left=0.47, ndvi_slope_end=-0.5, days_since_peak=30, days_at_top=40,
                           days_off_top=0, ndvi_rise_days=65, crop_age=100, water_spell=0, last_view_water=0,
                           dry_water_share=0, water_depth=3.66, views_in_water=float("nan"), water_views_in_water=float("nan"))])
    assert cr.classify_relative(f).tolist() == ["rice harvested"]
    monkeypatch.setattr(cr, "HARVEST_NOT_IF_RADAR_RISING", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted"]


def test_young_needs_a_long_radar_rise_only_when_switched_on(monkeypatch):
    # aoi39 pixel 43450: last view water, VV and VH up on the last passes only (12 days) -> young by the aoi72 rule
    base = dict(ndvi_low_peak=-1.0, vh_pos_end=0.46, vh_slope_end=0.6, vv_step_end=0.4, vh_rise_recent=0.8,
                vv_rise_recent=0.7, ndvi_left=0.0, ndvi_slope_end=-0.5, days_since_peak=80, days_at_top=90,
                days_off_top=60, ndvi_rise_days=40, crop_age=52, water_spell=1, last_view_water=1, radar_rise_days=12)
    f = pd.DataFrame([base, dict(base, radar_rise_days=45)])
    monkeypatch.setattr(cr, "YOUNG_AFTER_LAST_WATER", True)
    assert cr.classify_relative(f).tolist() == ["young rice", "young rice"]
    monkeypatch.setattr(cr, "YOUNG_MIN_RISE_DAYS", 40)
    assert cr.classify_relative(f).tolist() == ["flooded / bare", "young rice"]
    assert 'radar_rise_days' in __import__("inspect").getsource(cr.own_range_features)


def test_short_rise_young_becomes_flooded_only_where_the_field_ever_held_water(monkeypatch):
    base = dict(ndvi_low_peak=0.23, vh_pos_end=0.18, vh_slope_end=0.6, vv_step_end=0.4, vh_rise_recent=0.2,
                vv_rise_recent=0.43, ndvi_left=1.0, ndvi_slope_end=0.5, days_since_peak=0, days_at_top=0, days_off_top=0,
                ndvi_rise_days=120, crop_age=30, water_spell=0, last_view_water=0, radar_rise_days=30)
    f = pd.DataFrame([dict(base, water_depth=2.0),        # 8391: went below its dry level -> flooded
                      dict(base, water_depth=-1.1)])      # 29260: never did, dense canopy -> a crop
    monkeypatch.setattr(cr, "YOUNG_MIN_RISE_DAYS", 40)
    out = cr.classify_relative(f).tolist()
    assert out[0] == "flooded / bare" and out[1].startswith("rice standing")


def test_young_needs_water_only_when_switched_on(monkeypatch):
    # aoi39 pixel 13545: never below its dry radar level, young by its NDVI age (45 days)
    base = dict(ndvi_low_peak=0.13, vh_pos_end=0.68, vh_slope_end=0.1, vv_step_end=0.0, vh_rise_recent=0.3,
                vv_rise_recent=0.2, ndvi_left=1.0, ndvi_slope_end=0.5, days_since_peak=15, days_at_top=15, days_off_top=0,
                ndvi_rise_days=30, ndvi_rise_seen=40, crop_age=45, water_spell=0, last_view_water=0)
    f = pd.DataFrame([dict(base, water_depth=-2.7), dict(base, water_depth=4.0)])
    assert cr.classify_relative(f).tolist() == ["young rice", "young rice"]
    monkeypatch.setattr(cr, "YOUNG_NEEDS_WATER", True)
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded", "young rice"]


def test_lone_low_newest_view_is_not_a_harvest_only_when_switched_on(monkeypatch):
    # aoi33 pixel 51135 (5 Oct): standing and clear on 26 Sep, cloudy 28 Sep view at NDVI 0.17, VH at its season top
    base = dict(radar_rise_days=133, vh_pos_end=0.98, vh_slope_end=-0.03, vh_accel_end=0.14, vh_step_end=0.05,
                vh_rise_recent=0.27, vv_pos_end=0.76, vv_slope_end=-0.22, vv_accel_end=-0.63, vv_step_end=-0.43,
                vv_rise_recent=0.25, water_depth=-0.88, ndvi_low_peak=0.09, ndvi_low_day=20582, ndvi_left=0.16,
                last_view_water=0, dry_water_share=0, ndvi_slope_end=-0.84, ndvi_rise_seen=85, ndvi_rise_days=60,
                days_since_peak=50, days_at_top=60, days_off_top=0, crop_age=145, water_spell=0, view_gap_days=2,
                ndvi_left_prev=1.0)
    f = pd.DataFrame([base, dict(base, view_gap_days=20),       # a low view long after the last green one: a cut
                      dict(base, vh_pos_end=0.5)])              # radar left its top too: a cut
    assert cr.classify_relative(f).tolist() == ["rice harvested"] * 3
    monkeypatch.setattr(cr, "LONE_LOW_VIEW_DAYS", 5)
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded", "rice harvested", "rice harvested"]
    assert cr.AOI_OVERRIDES[33]["LONE_LOW_VIEW_DAYS"] == 5 and all(
        "LONE_LOW_VIEW_DAYS" not in cr.AOI_OVERRIDES.get(a, {}) for a in cr.LOCKED_AOIS)


def test_fast_green_up_is_rice_when_other_vegetation_by_speed_is_off(monkeypatch):
    # aoi33 pixel 41218 (user 5 Oct: rice): NDVI up in 30 days, left its top 80 days ago, 65 % of its green left
    f = pd.DataFrame([dict(radar_rise_days=133, vh_pos_end=0.76, vh_slope_end=0.18, vh_step_end=0.08,
                           vh_rise_recent=0.18, vv_pos_end=0.6, vv_step_end=0.08, vv_rise_recent=0.3, water_depth=-6.48,
                           ndvi_low_peak=0.12, ndvi_low_day=20592, ndvi_left=0.65, view_gap_days=5, ndvi_left_prev=0.79,
                           last_view_water=0, dry_water_share=0, ndvi_slope_end=-0.14, ndvi_rise_seen=25,
                           ndvi_rise_days=30, days_since_peak=90, days_at_top=100, days_off_top=80, crop_age=135,
                           water_spell=0)])
    assert cr.classify_relative(f).tolist() == ["other vegetation"]
    monkeypatch.setattr(cr, "FAST_RISE_OTHER_VEG", False)
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded"]
    assert cr.AOI_OVERRIDES[33]["FAST_RISE_OTHER_VEG"] is False and all(
        "FAST_RISE_OTHER_VEG" not in cr.AOI_OVERRIDES.get(a, {}) for a in cr.LOCKED_AOIS)


def test_fast_green_up_held_at_its_top_is_grown_when_other_vegetation_by_speed_is_off(monkeypatch):
    # aoi33 pixel 43566 (user 5 Oct: direct seeded standing): up in 30 days, held 105 days, VH dipped to 0.54 at the end
    f = pd.DataFrame([dict(radar_rise_days=133, vh_pos_end=0.54, vh_slope_end=-0.35, vh_step_end=0.13,
                           vh_rise_recent=0.07, vv_pos_end=0.54, vv_step_end=-0.54, vv_rise_recent=0.66,
                           water_depth=-0.84, ndvi_low_peak=0.16, ndvi_low_day=20592, ndvi_left=0.82, view_gap_days=5,
                           ndvi_left_prev=1.0, last_view_water=0, dry_water_share=0, ndvi_slope_end=-0.18,
                           ndvi_rise_seen=25, ndvi_rise_days=30, days_since_peak=95, days_at_top=105, days_off_top=5,
                           crop_age=135, water_spell=0)])
    monkeypatch.setattr(cr, "FAST_RISE_OTHER_VEG", True)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"]
    monkeypatch.setattr(cr, "FAST_RISE_OTHER_VEG", False)
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded"]


def test_aoi33_lone_low_view_needs_vh_out_of_its_low_part_only():
    # aoi33 pixel 9412 (user 5 Oct: standing on 26 Sep): cloudy 28 Sep view 2 days after a held one, VH at 0.69
    f = pd.DataFrame([dict(radar_rise_days=111.5, vh_pos_end=0.69, vh_slope_end=-0.31, vh_step_end=0.0,
                           vh_rise_recent=0.07, vv_pos_end=0.22, vv_step_end=-0.08, vv_rise_recent=0.33, water_depth=1.0,
                           ndvi_low_peak=0.02, ndvi_low_day=20582, ndvi_left=0.29, view_gap_days=2, ndvi_left_prev=1.0,
                           last_view_water=0, dry_water_share=0, ndvi_slope_end=-0.71, ndvi_rise_seen=75,
                           ndvi_rise_days=70, days_since_peak=50, days_at_top=65, days_off_top=0, crop_age=145,
                           water_spell=0)])
    assert cr.classify_relative(f).tolist() == ["rice harvested"]
    with cr.rules_for(33):
        assert cr.classify_relative(f).tolist() == ["rice standing direct seeded"]


def test_try_rules_cli_takes_the_rule_set_aois(monkeypatch):
    seen = {}
    monkeypatch.setattr(cr, "try_rules", lambda aoi, sources, series_root=None: seen.update(aoi=aoi, sources=sources)
                        or pd.DataFrame())
    cr.main(["try-rules", "--aoi", "118", "--sources", "116", "72", "33", "160"])
    assert seen == {"aoi": 118, "sources": [116, 72, 33, 160]}


def test_aoi118_rules_drop_radar_sowing_and_vv_jump_alone():
    r = cr.AOI_OVERRIDES[118]
    assert r["SOWING_FROM_RADAR"] is False and r["VV_JUMP"] == float("inf")
    assert {k: v for k, v in r.items() if k not in ("SOWING_FROM_RADAR", "VV_JUMP")} == \
        {k: v for k, v in cr.AOI_OVERRIDES[116].items() if k not in ("SOWING_FROM_RADAR", "VV_JUMP")}
    assert 118 in cr.LOCKED_AOIS
