import pandas as pd

from sar_pipeline.analysis import mask_experiment as me


def test_variants_are_distinct_folders_and_summary_reads():
    roots = {me.root(v) for v in me.VARIANTS}
    assert len(roots) == len(me.VARIANTS) and all(r != me.BASE for r in roots)
    assert me.VARIANTS["qa60"]["cs_min"] is None and me.VARIANTS["hyb60"]["keep_dark"]
    refs = pd.DataFrame([
        {"variant": "qa60", "aoi": "aoi1", "set": "rice_plot_interior", "region": "R", "pixels": 100, "delivered_pct": 90.0},
        {"variant": "qa60", "aoi": "aoi2", "set": "rice_plot_interior", "region": "R", "pixels": 300, "delivered_pct": 98.0},
        {"variant": "hyb60", "aoi": "aoi1", "set": "rice_plot_interior", "region": "R", "pixels": 100, "delivered_pct": 96.0},
        {"variant": "hyb60", "aoi": "aoi2", "set": "rice_plot_interior", "region": "R", "pixels": 300, "delivered_pct": 98.0}])
    text = me.summarise(refs, pd.DataFrame())
    assert "96.0" in text and "97.5" in text      # pixel-weighted: (100*96 + 300*98) / 400 = 97.5


def test_into_standard_redirects_the_series_root(monkeypatch):
    monkeypatch.setattr(me, "_INTO_STANDARD", True)
    assert me.root("hyb60") == me.BASE
    monkeypatch.setattr(me, "_INTO_STANDARD", False)
    assert me.root("hyb60") == f"{me.BASE}_hyb60"


def test_late_variant_builds_with_its_own_end_and_the_same_mask(monkeypatch):
    from sar_pipeline.analysis import ndvi_5day as nd

    seen = {}
    monkeypatch.setattr(nd, "build", lambda aoi_id, **kw: seen.update(kw) or {"aoi": f"aoi{aoi_id}"})
    monkeypatch.setattr(nd, "forget", lambda: None)
    monkeypatch.setattr(me, "root", lambda v: "/nonexistent/sandbox")
    me.build_one((1, "hyb40m1late"))
    assert seen["end"] == me.SERIES_END["hyb40m1late"] and seen["cs_min"] == me.VARIANTS["hyb40m1"]["cs_min"]
    assert me.VARIANTS["hyb40m1late"] == me.VARIANTS["hyb40m1"]
    seen.clear()
    me.build_one((1, "hyb40m1"))
    assert "end" not in seen                       # the other variants keep build's default end


def test_compare_against_another_sandbox(tmp_path, monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box

    cols = {f"{c}_acres_fieldmap": 0.0 for c in me.FIELD_CLASSES}
    for name, rice, label in (("old", 10.0, 1), ("new", 7.0, 4)):
        d = tmp_path / name / "fields"
        d.mkdir(parents=True)
        pd.DataFrame([{"aoi": "aoi1", **cols, "rice_acres_fieldmap": rice}]).to_csv(d / "field_acres_by_class.csv", index=False)
        gpd.GeoDataFrame({"field_id": ["aoi1_000001"], "label": [label]}, geometry=[box(500000, 5000000, 500010, 5000010)],
                         crs="EPSG:32633").to_file(d / "aoi1_fields_monsoon2026.gpkg")
    monkeypatch.setattr(me, "root", lambda v: str(tmp_path / v))
    acres, fields = me.compare_delivered([1], "new", noted=["aoi1_000001"], against="old")
    assert acres.loc[0, "delivered"] == "10 -> 7"
    assert fields.to_dict("records") == [{"field_id": "aoi1_000001", "old": 1, "new": 4}]
