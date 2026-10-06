import json

from sar_pipeline.analysis import rice_map_delivery as d


def test_rice_classes_are_standing_and_young_only():
    from sar_pipeline.analysis import curve_rules as cr
    assert sorted(cr.MAP_CLASSES[k][0] for k in d.RICE_CODES) == [
        "rice standing direct seeded", "rice standing transplanted", "young rice"]


def test_status_is_written_atomically_and_read_back(tmp_path):
    d._save_status(7, {"fields": {"done": True}}, root=str(tmp_path))
    assert d.load_status(7, root=str(tmp_path)) == {"fields": {"done": True}}
    assert not list((tmp_path / d.DELIVERY / "aoi7").glob("*.tmp"))


def test_qml_styles_by_major_class(tmp_path):
    d.fields_qml(tmp_path / "x.gpkg")
    q = (tmp_path / "x.qml").read_text()
    assert 'attr="major_class"' in q and 'value="rice"' in q and 'value="non-rice"' in q


def test_s2_name():
    assert d.s2_name(5, "2026-10-01") == "aoi5_2026-10_S2_2026-10-01.tif"
    json.dumps(d.STEPS)
