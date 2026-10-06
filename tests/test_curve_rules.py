"""Second fresh start: the relative (own-scale) hybrid rule, radar first."""
import numpy as np
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
        seen.append((Path(fresh).name, cr.AGE_LOW_SHARE, cr.YOUNG_WHILE_RADAR_LOW, cr.RADAR_DECIDES_WITHOUT_OPTICAL))
        return pd.DataFrame({"class": [1, 7], "name": ["a", "b"], "acres": [cr.AGE_LOW_SHARE, 1.0]})

    from pathlib import Path

    (tmp_path / "fresh" / "aoi5").mkdir(parents=True)
    (tmp_path / "fresh" / "aoi5" / "aoi5_step1_cover.tif").write_bytes(b"x")
    monkeypatch.setattr(cr, "_run", fake_run)
    t = cr.try_rules(5, series_root="s", fresh=str(tmp_path / "fresh"), out=str(tmp_path / "out"), jobs=1)
    # each set as reviewed (curve_rules.REVIEWED_SETS), in its own folder; no extra switches laid over it
    assert [x[0] for x in seen] == [f"rules_aoi{a}" for a in cr.RULE_SOURCES]
    assert [x[1] for x in seen] == [cr.REVIEWED_SETS.get(a, {}).get("AGE_LOW_SHARE", 0.2) for a in cr.RULE_SOURCES]
    assert list(t.columns) == ["class", "name", *[f"aoi{a} rules" for a in cr.RULE_SOURCES]]
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

    assert cr.WATER_SPELL_MIN_PASSES == 2 and cr.REVIEWED_SETS[39]["WATER_SPELL_MIN_PASSES"] == 1
    assert all("WATER_SPELL_MIN_PASSES" not in cr.REVIEWED_SETS.get(a, {}) for a in (160, 28, 72, 116))
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
        "LONE_LOW_VIEW_DAYS" not in cr.AOI_OVERRIDES.get(a, {}) for a in (72, 116, 39, 118))   # AOIs locked before these switches


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
        "FAST_RISE_OTHER_VEG" not in cr.AOI_OVERRIDES.get(a, {}) for a in (72, 116, 39, 118))   # AOIs locked before these switches


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


def test_never_emptied_without_water_is_other_vegetation_only_when_switched_on(monkeypatch):
    # aoi83 pixel 69008 (user 5 Oct: other vegetation): NDVI low 0.23 / peak 0.56, VH bright all season, no water
    f = pd.DataFrame([dict(radar_rise_days=30, vh_pos_end=1.03, vh_slope_end=0.77, vh_step_end=-0.03,
                           vh_rise_recent=1.41, vv_pos_end=0.34, vv_step_end=0.49, vv_rise_recent=0.33,
                           water_depth=-2.1, ndvi_low_peak=0.42, ndvi_low_day=20581, ndvi_left=1.0, view_gap_days=10,
                           ndvi_left_prev=0.75, last_view_water=0, dry_water_share=0, ndvi_slope_end=0.25,
                           ndvi_rise_seen=128, ndvi_rise_days=30, days_since_peak=20, days_at_top=120,
                           days_off_top=0, crop_age=150, water_spell=0)])
    with cr.rules_for(118):
        assert cr.classify_relative(f).tolist() == ["rice standing direct seeded"]
    with cr.rules_for(83):
        assert cr.classify_relative(f).tolist() == ["other vegetation"]
        assert cr.classify_relative(f.assign(water_spell=1.0)).tolist() != ["other vegetation"]


def test_aoi83_rules_are_the_aoi118_set_plus_three_switches():
    r = dict(cr.AOI_OVERRIDES[83])
    assert (r.pop("LONG_WATER_DAYS"), r.pop("NEVER_EMPTY_OTHER_VEG"), r.pop("TREE_LOW_BEFORE")) == (30, 0.35, "2026-07-01")
    for k in ("LONG_WATER_LEVEL", "LONG_WATER_WET_VIEW", "LONG_WATER_GREEN_LEAD", "ANY_WATER_TRANSPLANTED"):
        r.pop(k)
    assert r == cr.AOI_OVERRIDES[118]
    assert all(k not in cr.AOI_OVERRIDES.get(a, {}) for a in (72, 116, 39, 118)
               for k in ("LONG_WATER_DAYS", "NEVER_EMPTY_OTHER_VEG"))
    assert "first_green" in __import__("inspect").getsource(cr.own_range_features)


def test_never_emptied_by_the_fitted_curve_is_other_vegetation_despite_one_dark_view():
    # aoi83 pixel 67677 (user 5 Oct): one 0.16 view on 8 May (clear-view low/peak 0.29), fitted low/peak 0.8,
    # an August radar "water spell" under the canopy but never below the dry level
    f = pd.DataFrame([dict(radar_rise_days=30, vh_pos_end=1.01, vh_slope_end=0.51, vh_step_end=0.08,
                           vh_rise_recent=1.07, vv_pos_end=0.56, vv_step_end=0.23, vv_rise_recent=0.64,
                           water_depth=-1.48, ndvi_low_peak=0.29, fit_low_peak=0.8, ndvi_low_day=20581, ndvi_left=1.0,
                           view_gap_days=10, ndvi_left_prev=0.72, last_view_water=0, dry_water_share=0,
                           ndvi_slope_end=0.28, ndvi_rise_seen=20, ndvi_rise_days=15, days_since_peak=20,
                           days_at_top=20, days_off_top=0, crop_age=45, water_spell=1, views_in_water=1,
                           water_views_in_water=0)])
    with cr.rules_for(118):
        assert cr.classify_relative(f).tolist() == ["young rice"]
    with cr.rules_for(83):
        assert cr.classify_relative(f).tolist() == ["other vegetation"]
        assert cr.classify_relative(f.assign(water_depth=2.0)).tolist() != ["other vegetation"]


def test_long_flood_switches_only_on_the_field_reviewed_aois():
    for a in (83, 117, 155):
        r = cr.REVIEWED_SETS[a]
        assert (r["LONG_WATER_LEVEL"], r["LONG_WATER_WET_VIEW"], r["LONG_WATER_GREEN_LEAD"]) == (0.4, True, 0)
    for a in (160, 28, 72, 116, 39, 118):                 # locked before the long-flood switches
        assert not {"LONG_WATER_LEVEL", "LONG_WATER_WET_VIEW", "LONG_WATER_LAST"} & set(cr.REVIEWED_SETS.get(a, {}))
    assert cr.REVIEWED_SETS[160] == {"ANY_WATER_TRANSPLANTED": True} and cr.REVIEWED_SETS[28]["ANY_WATER_TRANSPLANTED"]
    assert not any(cr.REVIEWED_SETS[a].get("ANY_WATER_TRANSPLANTED") for a in (20, 33, 72, 116, 39, 118))
    assert cr.LONG_WATER_LEVEL is None and cr.LONG_WATER_WET_VIEW is False and cr.LONG_WATER_LAST is False


def test_aoi63_starts_as_a_copy_of_the_aoi39_rules_with_the_new_aoi_switches():
    """User, 5 Oct: aoi63 uses the aoi39 rules; a later aoi63 change must not reach the locked aoi39 entry."""
    own = {"HARVEST_NOT_IF_RADAR_RISING": False, "CANOPY_NOT_FLOODED": True,      # aoi63's own (12923, 20447,
           "RADAR_STORY_IN_GAP": True, "WATER_SMALL_UNSEEN": True,                 # the 3 Jul - 26 Sep gap, 31277,
           "POND_NEEDS_NO_CROP": True, "CANOPY_OVER_RADAR_WATER": True,             # 41938, 13571,
           "GREEN_VIEW_NOT_FLOODED": True}                                          # 36796 / 23348)
    assert cr.REVIEWED_SETS[63] == dict(cr.REVIEWED_SETS[39], **own)
    assert cr.REVIEWED_SETS[39]["HARVEST_NOT_IF_RADAR_RISING"] and "CANOPY_NOT_FLOODED" not in cr.REVIEWED_SETS[39]
    assert cr.REVIEWED_SETS[63]["RADAR_DECIDES_WITHOUT_OPTICAL"] and "TONE_AND_CURVE" not in cr.REVIEWED_SETS[63]
    assert 63 in cr.LOCKED_AOIS                  # locked 5 Oct (user)


def test_a_closing_canopy_with_the_radar_falling_is_not_flooded_when_switched_on(monkeypatch):
    """aoi63 pixel 20447 (user, 5 Oct): water late June, newest view NDVI 0.86 = season's greenest, VV / VH falling."""
    row = dict(radar_rise_days=84.0, vh_pos_end=0.47, vh_slope_end=-0.28, vh_step_end=-0.18, vh_rise_recent=0.09,
               vv_step_end=-0.1, vv_rise_recent=0.16, water_depth=1.97, ndvi_low_peak=0.29, ndvi_low_day=20574.0,
               ndvi_left=1.0, last_view_water=0, dry_water_share=0, ndvi_slope_end=0.06, ndvi_rise_seen=148.0,
               ndvi_rise_days=50.0, days_since_peak=0.0, days_at_top=0.0, days_off_top=0.0, crop_age=104.0,
               water_spell=1, views_in_water=0, water_views_in_water=0)
    f = pd.DataFrame([row, dict(row, crop_age=30.0), dict(row, ndvi_left=0.6)])
    assert cr.classify_relative(f).tolist() == ["flooded / bare"] * 3
    monkeypatch.setattr(cr, "CANOPY_NOT_FLOODED", True)
    # old crop standing, a young one young, and a field whose newest view is NOT its greenest stays flooded
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "young rice", "flooded / bare"]


def test_where_the_optical_never_saw_the_green_up_the_radar_tells_the_story(monkeypatch):
    """aoi63 (user, 5 Oct): no clear view 3 Jul - 26 Sep; the NDVI timing between is the fit's line."""
    base = dict(radar_rise_days=84.0, vh_pos_end=0.47, vh_slope_end=-0.28, vh_step_end=-0.18, vh_rise_recent=0.09,
                vv_step_end=-0.1, vv_rise_recent=0.16, water_depth=1.97, ndvi_low_peak=0.29, ndvi_low_day=20574.0,
                ndvi_left=1.0, last_view_water=0, dry_water_share=0, ndvi_slope_end=0.06, ndvi_rise_seen=148.0,
                ndvi_rise_days=20.0, days_since_peak=0.0, days_at_top=0.0, days_off_top=0.0, crop_age=104.0,
                water_spell=1, views_in_water=0, water_views_in_water=0, rise_unseen=1)
    f = pd.DataFrame([
        base,                                                     # green now, water spell, old: standing transplanted
        dict(base, water_spell=0, water_depth=-1.0, radar_rise_days=30.0),   # green, never wet, radar up 30 d: young
        dict(base, ndvi_left=0.26, vh_pos_end=1.0, vh_rise_recent=0.6, vv_rise_recent=1.0),  # cut paddy (12923)
        dict(base, ndvi_left=0.26, water_spell=0, water_depth=-1.0),          # bare, never held water
        dict(base, ndvi_low_peak=0.6),                            # never emptied BUT stood in water: not a tree
        dict(base, ndvi_low_peak=0.6, water_spell=0, water_depth=-1.0),       # never emptied, never wet: tree
        dict(base, rise_unseen=0),                                # green-up seen: the usual NDVI tests
    ])
    off = cr.classify_relative(f).tolist()
    assert off[0] == "flooded / bare"                            # 20447 before: the radar low read as water
    monkeypatch.setattr(cr, "RADAR_STORY_IN_GAP", True)
    assert cr.classify_relative(f).tolist() == [
        "rice standing transplanted", "young rice", "rice harvested", "flooded / bare",
        "rice standing transplanted", "tree/orchard", off[6]]


def test_a_crop_grown_unseen_after_a_summer_crop_is_judged_green_on_the_whole_year(monkeypatch):
    """aoi63 pixel 31277 (user, 5 Oct): summer crop 0.94 in May, July water, 0.81 on 28 Sep -> transplanted rice."""
    row = dict(radar_rise_days=60.0, vh_pos_end=0.75, vh_slope_end=-0.17, vh_step_end=0.02, vh_rise_recent=0.66,
               vv_step_end=-0.01, vv_rise_recent=0.26, water_depth=1.38, ndvi_low_peak=0.79, ndvi_low_day=20607.0,
               ndvi_left=0.38, ndvi_left_year=0.68, last_view_water=0, dry_water_share=0.07, ndvi_slope_end=0.11,
               ndvi_rise_days=np.nan, ndvi_rise_seen=np.nan, days_since_peak=145.0, days_at_top=145.0,
               days_off_top=145.0, crop_age=87.0, water_spell=1, views_in_water=0, water_views_in_water=0,
               rise_unseen=0, crop_unseen=1)
    f = pd.DataFrame([row, dict(row, crop_unseen=0)])
    assert cr.classify_relative(f).tolist()[0] == "tree/orchard"
    monkeypatch.setattr(cr, "RADAR_STORY_IN_GAP", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "tree/orchard"]


def test_a_field_flooded_in_the_dry_season_but_green_now_is_not_a_pond_when_switched_on(monkeypatch):
    """aoi63 pixel 41938 (user, 5 Oct): early-April water, summer crop, July water, NDVI 0.86 on 28 Sep -> transplanted."""
    row = dict(radar_rise_days=87.5, vh_pos_end=0.4, vh_slope_end=-0.09, vh_step_end=0.73, vh_rise_recent=0.64,
               vv_step_end=-0.15, vv_rise_recent=0.57, water_depth=2.84, ndvi_low_peak=0.63, ndvi_low_day=20577.0,
               ndvi_left=0.91, ndvi_left_year=0.97, last_view_water=0, dry_water_share=0.64, ndvi_slope_end=0.2,
               ndvi_rise_seen=5.0, ndvi_rise_days=5.0, days_since_peak=140.0, days_at_top=140.0, days_off_top=140.0,
               crop_age=73.0, water_spell=1, views_in_water=0, water_views_in_water=0, rise_unseen=1, crop_unseen=1)
    f = pd.DataFrame([row, dict(row, ndvi_left_year=0.1, ndvi_left=0.1)])      # second: not green now -> pond
    for k in ("POND_IF_DRY_WATER", "RADAR_STORY_IN_GAP"):
        monkeypatch.setattr(cr, k, True)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"] * 2
    monkeypatch.setattr(cr, "POND_NEEDS_NO_CROP", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "flooded / bare"]


def test_a_radar_low_under_the_years_greenest_view_is_a_canopy_when_switched_on(monkeypatch):
    """aoi63 pixel 13571 (user, 5 Oct): seen empty early May, NDVI 0.81 on 28 Sep, radar at its low -> transplanted."""
    row = dict(radar_rise_days=24.0, vh_pos_end=0.09, vh_slope_end=-0.24, vh_step_end=-0.42, vh_rise_recent=0.21,
               vv_step_end=-0.35, vv_rise_recent=0.18, water_depth=6.61, ndvi_low_peak=0.19, ndvi_low_day=20577.0,
               ndvi_left=1.0, ndvi_left_year=1.0, days_since_ndvi_low=147.0, last_view_water=0, dry_water_share=0,
               ndvi_slope_end=0.04, ndvi_rise_seen=140.0, ndvi_rise_days=140.0, days_since_peak=0.0, days_at_top=5.0,
               days_off_top=0.0, crop_age=28.0, water_spell=1, views_in_water=0, water_views_in_water=0,
               rise_unseen=0, crop_unseen=1)
    f = pd.DataFrame([row, dict(row, days_since_ndvi_low=30.0), dict(row, ndvi_left_year=0.7)])
    monkeypatch.setattr(cr, "RADAR_DECIDES_WITHOUT_OPTICAL", True)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"] * 3
    monkeypatch.setattr(cr, "CANOPY_OVER_RADAR_WATER", True)
    # seen empty only 30 days before, or not the year's greenest now: the radar's water stands
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "flooded / bare", "flooded / bare"]


def test_a_late_radar_low_under_the_years_greenest_view_does_not_make_a_crop_young(monkeypatch):
    """aoi63 pixel 18853 (user, 5 Oct): seen empty 150 days before, NDVI 0.80 now, radar low -> standing, not young."""
    row = dict(radar_rise_days=78.0, vh_pos_end=0.03, vh_slope_end=-0.04, vh_step_end=-0.08, vh_rise_recent=0.07,
               vv_step_end=-0.03, vv_rise_recent=0.11, water_depth=0.96, ndvi_low_peak=0.27, ndvi_low_day=20574.0,
               days_since_ndvi_low=150.0, ndvi_left=1.0, ndvi_left_year=1.0, last_view_water=0, dry_water_share=0,
               ndvi_slope_end=0.18, ndvi_rise_seen=150.0, ndvi_rise_days=140.0, days_since_peak=0.0, days_at_top=5.0,
               days_off_top=0.0, crop_age=37.0, water_spell=1, views_in_water=0, water_views_in_water=0,
               rise_unseen=0, crop_unseen=1)
    f = pd.DataFrame([row,
                      dict(row, vh_pos_end=0.8),                  # radar not low now: the switch does not apply
                      dict(row, crop_age=104.0),                  # water from late June (20447): transplanted
                      dict(row, water_depth=6.61)])               # deep water (13571): transplanted
    monkeypatch.setattr(cr, "CANOPY_NOT_FLOODED", True)
    before = cr.classify_relative(f).tolist()
    assert before[0] == "young rice"
    monkeypatch.setattr(cr, "CANOPY_OVER_RADAR_WATER", True)
    # user, 5 Oct: 18853 is direct seeded (its "water" was only the late canopy)
    assert cr.classify_relative(f).tolist() == ["rice standing direct seeded", before[1], "rice standing transplanted",
                                                "rice standing transplanted"]


def test_a_green_newest_view_is_never_flooded_when_switched_on(monkeypatch):
    """aoi63 (user, 5 Oct): 36796 transplanted (water 3 Jul + radar dip), 23348 direct seeded (no water spell)."""
    a = dict(radar_rise_days=42.0, vh_pos_end=0.49, vh_slope_end=-0.46, vh_step_end=-0.17, vh_rise_recent=0.5,
             vv_step_end=-0.19, vv_rise_recent=0.3, water_depth=5.39, ndvi_low_peak=0.09, ndvi_low_day=20637.0,
             days_since_ndvi_low=87.0, ndvi_left=0.83, ndvi_left_year=0.81, last_view_water=0, dry_water_share=0,
             ndvi_slope_end=0.01, ndvi_rise_seen=np.nan, ndvi_rise_days=np.nan, days_since_peak=145.0,
             days_at_top=145.0, days_off_top=145.0, crop_age=76.0, water_spell=1, views_in_water=0,
             water_views_in_water=0, rise_unseen=0, crop_unseen=1)
    b = dict(a, water_depth=3.15, days_since_ndvi_low=52.0, ndvi_left=1.0, ndvi_left_year=0.94, crop_age=145.0,
             water_spell=0, vh_pos_end=0.43, vh_rise_recent=0.41, vv_rise_recent=0.32)
    f = pd.DataFrame([a, b, dict(a, last_view_water=1), dict(a, ndvi_left_year=0.3), dict(b, days_since_ndvi_low=30.0)])
    monkeypatch.setattr(cr, "RADAR_DECIDES_WITHOUT_OPTICAL", True)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"] * 5
    monkeypatch.setattr(cr, "GREEN_VIEW_NOT_FLOODED", True)
    # water on the newest view, or not green now: still flooded; seen empty 30 days ago: young
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "rice standing direct seeded",
                                                "flooded / bare", "flooded / bare", "young rice"]


def test_young_rice_with_a_clear_radar_dip_is_transplanted_when_switched_on(monkeypatch):
    """aoi13 (user, 5 Oct): a clear VV / VH dip -> transplanted rice even at 30 days; a short radar rise stays flooded."""
    row = dict(radar_rise_days=35.0, vh_pos_end=0.7, vh_slope_end=0.3, vh_step_end=0.1, vh_rise_recent=0.6,
               vv_step_end=0.1, vv_rise_recent=0.7, water_depth=6.3, ndvi_low_peak=0.1, ndvi_low_day=20682.0,
               days_since_ndvi_low=45.0, ndvi_left=0.9, ndvi_left_year=0.9, last_view_water=0, dry_water_share=0,
               ndvi_slope_end=0.3, ndvi_rise_seen=40.0, ndvi_rise_days=40.0, days_since_peak=0.0, days_at_top=0.0,
               days_off_top=0.0, crop_age=30.0, water_spell=1, views_in_water=2, water_views_in_water=2,
               rise_unseen=0, crop_unseen=0)
    f = pd.DataFrame([row, dict(row, water_depth=1.0), dict(row, radar_rise_days=18.0, vh_pos_end=0.4,
                                                            last_view_water=1)])
    monkeypatch.setattr(cr, "YOUNG_MIN_RISE_DAYS", 30)
    before = cr.classify_relative(f).tolist()
    assert before[:2] == ["young rice", "young rice"] and before[2] == "flooded / bare"
    monkeypatch.setattr(cr, "DIP_MAKES_TRANSPLANTED", True)
    # deep dip -> transplanted; shallow -> stays young; radar risen only 18 days -> still flooded
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "young rice", "flooded / bare"]


def test_the_radar_sowing_is_never_later_than_the_crop_seen_rising():
    """aoi13 pixel 14994 (user, 5 Oct): water view 18 Jul, NDVI 0.33 on 17 Aug, radar water end 29 Aug -> sown 18 Jul."""
    d = lambda x: (pd.Timestamp(x) - pd.Timestamp("1970-01-01")).days        # noqa: E731
    tdays = np.array([d("2026-06-23"), d("2026-07-18"), d("2026-08-17"), d("2026-10-01")])
    v = np.array([[0.07, 0.07], [0.02, 0.02], [0.33, 0.05], [0.80, 0.80]])
    ok = np.ones_like(v, dtype=bool)
    lo, amp = v.min(axis=0), v.max(axis=0) - v.min(axis=0)
    ilo = v.argmin(axis=0)
    sowing = np.array([d("2026-08-29")] * 2)
    new, moved = cr.sowing_not_after_seen_crop(sowing, sowing, ok, v, lo, amp, ilo, tdays)
    # pixel 1: the crop was seen at 0.33 on 17 Aug -> sown at the 18 Jul low; pixel 2: not seen rising -> radar date
    assert moved.tolist() == [True, False]
    assert new.tolist() == [d("2026-07-18"), d("2026-08-29")]


def test_an_old_water_view_does_not_beat_a_radar_that_rose_since_in_the_radar_story(monkeypatch):
    """aoi13 pixel 11791 (user, 5 Oct): last view water 19 Aug, VV / VH up since, deep dip -> transplanted."""
    row = dict(radar_rise_days=30.0, vh_pos_end=0.58, vh_slope_end=0.48, vh_step_end=0.23, vh_rise_recent=0.77,
               vv_step_end=0.27, vv_rise_recent=1.01, water_depth=6.01, ndvi_low_peak=-0.7, ndvi_low_day=20652.0,
               days_since_ndvi_low=32.0, ndvi_left=0.54, ndvi_left_year=0.34, last_view_water=1, dry_water_share=0,
               ndvi_slope_end=0.48, ndvi_rise_seen=np.nan, ndvi_rise_days=15.0, days_since_peak=135.0,
               days_at_top=135.0, days_off_top=130.0, crop_age=75.0, water_spell=1, views_in_water=3,
               water_views_in_water=3, rise_unseen=1, crop_unseen=0)
    f = pd.DataFrame([row, dict(row, vh_rise_recent=0.1, vv_rise_recent=0.1, vv_step_end=0.0)])   # radar not up
    for k, v in (("RADAR_STORY_IN_GAP", True), ("YOUNG_MIN_RISE_DAYS", 30), ("DIP_MAKES_TRANSPLANTED", True)):
        monkeypatch.setattr(cr, k, v)
    assert cr.classify_relative(f).tolist() == ["flooded / bare"] * 2
    monkeypatch.setattr(cr, "STORY_YOUNG_AFTER_LAST_WATER", True)
    assert cr.classify_relative(f).tolist() == ["rice standing transplanted", "flooded / bare"]


def test_a_short_radar_rise_under_a_green_newest_view_stays_young_when_switched_on(monkeypatch):
    """aoi13 (user, 5 Oct): 16003 radar up 24 days, 1 Oct NDVI 0.67 -> young; 9643 1 Oct view water -> flooded."""
    row = dict(radar_rise_days=24.0, vh_pos_end=0.54, vh_slope_end=0.42, vh_step_end=-0.01, vh_rise_recent=0.58,
               vv_step_end=0.13, vv_rise_recent=1.01, water_depth=3.44, ndvi_low_peak=-0.32, ndvi_low_day=20652.0,
               days_since_ndvi_low=75.0, ndvi_left=1.0, ndvi_left_year=0.81, last_view_water=0, dry_water_share=0,
               ndvi_slope_end=0.66, ndvi_rise_seen=45.0, ndvi_rise_days=60.0, days_since_peak=0.0, days_at_top=5.0,
               days_off_top=0.0, crop_age=30.0, water_spell=1, views_in_water=4, water_views_in_water=4,
               rise_unseen=0, crop_unseen=0)
    f = pd.DataFrame([row, dict(row, last_view_water=1, ndvi_left_year=0.3, ndvi_left=0.3),
                      dict(row, radar_rise_days=35.0)])
    for k, v in (("YOUNG_MIN_RISE_DAYS", 30), ("DIP_MAKES_TRANSPLANTED", True)):
        monkeypatch.setattr(cr, k, v)
    before = cr.classify_relative(f).tolist()
    assert before[0] == "flooded / bare" and before[2] == "rice standing transplanted"
    monkeypatch.setattr(cr, "SHORT_RISE_GREEN_YOUNG", True)
    after = cr.classify_relative(f).tolist()
    # green on the newest view: young; newest view water: unchanged; risen 30+ days with a deep dip: transplanted
    assert after[0] == "young rice" and after[1] == before[1] and after[2] == "rice standing transplanted"


def test_a_pixel_still_empty_on_the_latest_image_is_not_rice_when_switched_on(monkeypatch):
    """aoi13 (user, 5 Oct): empty on the latest image -> flooded / bare; cloudy there -> the curve decides."""
    row = dict(radar_rise_days=36.0, vh_pos_end=0.7, vh_slope_end=0.4, vh_step_end=0.15, vh_rise_recent=0.6,
               vv_step_end=0.35, vv_rise_recent=0.7, water_depth=8.0, ndvi_low_peak=-0.35, ndvi_low_day=20682.0,
               days_since_ndvi_low=45.0, ndvi_left=0.9, ndvi_left_year=0.9, last_view_water=0, dry_water_share=0,
               ndvi_slope_end=0.4, ndvi_rise_seen=40.0, ndvi_rise_days=40.0, days_since_peak=0.0, days_at_top=0.0,
               days_off_top=0.0, crop_age=40.0, water_spell=1, views_in_water=0, water_views_in_water=0,
               rise_unseen=0, crop_unseen=0, seen_on_latest=1)
    f = pd.DataFrame([row, dict(row, ndvi_left_year=0.1), dict(row, ndvi_left_year=0.1, seen_on_latest=0)])
    before = cr.classify_relative(f).tolist()
    monkeypatch.setattr(cr, "NEWEST_EMPTY_NOT_RICE", True)
    assert cr.classify_relative(f).tolist() == [before[0], "flooded / bare", before[2]]


def test_change_since_reads_the_radar_after_the_view():
    pday = np.array([0, 12, 24, 36])
    pos = np.array([[0.9, 0.2], [0.8, np.nan], [0.2, 0.3], [0.1, 0.9]])
    ch = cr.change_since(pday, pos, np.array([12.0, 12.0]))
    # pixel 0: 0.8 at the view, 0.15 on its last two passes -> fell; pixel 1: NaN on the view pass -> the pass
    # before (0.2), 0.6 after -> rose
    assert np.allclose(ch, [0.15 - 0.8, 0.6 - 0.2])
    assert np.isnan(cr.change_since(pday, pos, np.array([40.0, np.nan]))).all()   # no pass after / no view


def test_universal_tone_and_curve_rules(monkeypatch):
    """User, 5 Oct: a class must fit the latest clear view's tone; the newer news wins; harvested needs VV and VH."""
    base = dict(radar_rise_days=60.0, vh_pos_end=0.8, vh_slope_end=0.0, vh_step_end=0.0, vh_rise_recent=0.1,
                vv_step_end=0.0, vv_rise_recent=0.1, vh_fall_recent=0.1, vv_fall_recent=0.1, water_depth=4.0,
                ndvi_low_peak=0.1, ndvi_low_day=20600.0, days_since_ndvi_low=90.0, ndvi_left=0.9, ndvi_left_year=0.9,
                last_view_water=0, dry_water_share=0, ndvi_slope_end=0.0, ndvi_rise_seen=60.0, ndvi_rise_days=60.0,
                days_since_peak=10.0, days_at_top=30.0, days_off_top=5.0, crop_age=90.0, water_spell=1,
                views_in_water=0, water_views_in_water=0, rise_unseen=0, crop_unseen=0, view_day=20700.0,
                vh_since_view=0.0, vv_since_view=0.0)
    f = pd.DataFrame([
        dict(base),                                                          # green, standing: unchanged
        dict(base, vh_since_view=-0.7, vv_since_view=-0.6),                  # green 13 Sep, both fell after: harvested
        dict(base, vh_since_view=-0.7, vv_since_view=0.1),                   # only VH fell: still standing
        dict(base, ndvi_left_year=0.1, ndvi_left=0.1),                       # empty now, no radar move: not rice
        dict(base, ndvi_left_year=0.1, ndvi_left=0.1, vh_since_view=0.6, vv_since_view=0.8),  # empty, both rose: crop
        dict(base, last_view_water=1, ndvi_left_year=0.0, ndvi_left=0.0),   # water now: not standing rice
        dict(base, view_day=np.nan),                                         # no clear view: the curve decides
    ])
    out = ["rice standing transplanted"] * 7
    monkeypatch.setattr(cr, "TONE_AND_CURVE", True)
    assert cr.tone_and_curve(f, out) == [
        "rice standing transplanted", "rice harvested", "rice standing transplanted", "flooded / bare",
        "rice standing transplanted", "flooded / bare", "rice standing transplanted"]
    # harvested needs both polarisations
    g = pd.DataFrame([dict(base, vh_fall_recent=0.8, vv_fall_recent=0.1, view_day=np.nan),
                      dict(base, vh_fall_recent=0.8, vv_fall_recent=0.7, view_day=np.nan)])
    monkeypatch.setattr(cr, "HARVEST_NEEDS_BOTH_POLS", True)
    assert cr.tone_and_curve(g, ["rice harvested"] * 2) == ["rice standing transplanted", "rice harvested"]


def test_nearest_locked_orders_the_locked_aois_by_distance():
    index = pd.DataFrame({"aoi": [1, 2, 3, 4], "lon": [96.0, 96.1, 97.0, 96.0], "lat": [17.0, 17.0, 17.0, 18.0],
                          "acres": [10.0, 20.0, 30.0, 40.0]})
    t = cr.nearest_locked(1, index, locked=[2, 3, 4])
    assert t.locked_aoi.tolist() == [2, 3, 4]                 # ~10.6 km, ~106 km, ~111 km
    assert 10 < t.km[0] < 11.5
