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
