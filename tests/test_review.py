import numpy as np
import pandas as pd

from sar_pipeline import review as rv


def _ev(**cols):
    base = dict(field_id=["f"], area_acres=[1.0], label=[1], late_clear_ndvi_max=[np.nan], last_ndvi=[np.nan],
                vh_drop_end=[3.0], flood_passes=[1], dark_monsoon_passes=[3], vh_median=[-20.0],
                vh_second_darkest=[-25.0], dry_season_ndvi=[0.1], best_monsoon_drop=[8.0],
                field_rule_label=[1], days_since_clear=[5])
    base.update({k: [v] for k, v in cols.items()})
    return pd.DataFrame(base)


def test_clean_rice_is_not_suspect():
    assert rv.suspects(_ev()).empty


def test_harvested_with_standing_canopy_is_flagged():
    s = rv.suspects(_ev(label=4, late_clear_ndvi_max=0.55))
    assert "harvested_but_standing" in s["category"].tolist()


def test_not_rice_with_flood_and_canopy_is_flagged():
    s = rv.suspects(_ev(label=0, late_clear_ndvi_max=0.8))
    assert "water_and_canopy_not_rice" in s["category"].tolist()


def test_village_in_young_is_flagged():
    s = rv.suspects(_ev(label=2, vh_median=-13.0, vh_second_darkest=-16.0, flood_passes=0,
                        dark_monsoon_passes=0, best_monsoon_drop=1.0))
    assert "builtup_or_trees_in_crop_class" in s["category"].tolist()


def test_rice_judged_on_an_old_observation_is_flagged():
    s = rv.suspects(_ev(days_since_clear=45))
    assert s["category"].tolist() == ["standing_extrapolated"]


def test_stretch_ignores_zeros_and_scales_to_unit():
    a = np.array([[0.0, 100.0], [200.0, 300.0]])
    out = rv._stretch(a, 0, 100)
    assert out.min() == 0.0 and out.max() == 1.0 and out[0, 0] == 0.0


def test_spread_sample_covers_the_corners():
    import geopandas as gpd
    from shapely.geometry import box

    from sar_pipeline import qgis_review as qr

    boxes, ids = [], []
    for j in range(6):
        for i in range(6):
            boxes.append(box(i * 100, j * 100, i * 100 + 40, j * 100 + 40))
            ids.append(f"a_{j}{i}")
    g = gpd.GeoDataFrame({"field_id": ids, "label": 1, "area_acres": 1.0, "is_field": True}, geometry=boxes, crs="EPSG:32646")
    pick = qr.spread_sample(g, [1], per_label=10)
    assert len(pick) == len(set(pick)) == 10
    assert {"a_00", "a_05", "a_50", "a_55"} <= set(pick) or len({p[2] for p in pick}) >= 3
    xs = g.set_index("field_id").loc[pick].geometry.centroid.x
    assert xs.min() < 100 and xs.max() > 400
    assert qr.spread_sample(g, [1], per_label=10) == pick


def test_fresh_at_reads_the_fresh_start_steps_for_the_pixels(tmp_path):
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.transform import from_origin

    from sar_pipeline import qgis_review as q

    folder = tmp_path / "aoi7"
    folder.mkdir()
    prof = dict(driver="GTiff", width=3, height=1, count=1, crs="EPSG:32633", transform=from_origin(500000, 5000000, 10, 10))
    for name, data, dtype in (("step1_cover", [1, 1, 2], "uint8"), ("step2_period", [1, 2, 0], "uint8"),
                              ("step2_sowing", [134, 160, 0], "uint16")):
        with rasterio.open(folder / f"aoi7_{name}.tif", "w", dtype=dtype, **prof) as ds:
            ds.write(np.array([data], dtype=dtype)[None])
    out = q.fresh_at(7, [0, 1, 2], fresh_root=str(tmp_path))
    assert out["sowing_date"] == pd.Timestamp("2026-05-14") + pd.Timedelta(days=13)      # median of day 134 and 160
    assert "May" in out["step2_sowing_period"] and len(out["step1_cover"]) == 2
    assert q.fresh_at(8, [0], fresh_root=str(tmp_path)) == {}


def test_annotate_rule_writes_the_live_class_and_the_features_on_the_curve():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sar_pipeline import qgis_review as q

    fig, _ = plt.subplots(2, 1)
    q.annotate_rule(fig, {"rule_class": "young rice", "rule_shares": {"young rice": "100 %"},
                          "rule_features": {"vh_pos_end": 0.3, "water_spell": 1.0},
                          "your_labels": {1: "young rice / standing"}})
    assert "rule now: young rice" in fig._suptitle.get_text() and "your label" in fig._suptitle.get_text()
    assert any("water while crop small" in t.get_text() for t in fig.axes[0].texts)
    plt.close(fig)
