import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from sar_pipeline.analysis import qc_compare as q
from sar_pipeline.analysis import rice_map_delivery as rd


def _delivery(tmp_path, aoi=9991):
    d = tmp_path / rd.DELIVERY / f"aoi{aoi}"
    d.mkdir(parents=True)
    f = gpd.GeoDataFrame({"field_id": ["a", "b", "c"], "major_class": ["rice", "rice", "non-rice"],
                          "sub_class": ["young rice", "rice standing transplanted", "tree/orchard"],
                          "acres": [0.5, 0.5, 0.5]}, geometry=[box(0, 0, 45, 45), box(50, 0, 95, 45), box(100, 0, 145, 45)])
    f.to_file(d / f"aoi{aoi}_fields.gpkg", driver="GPKG")
    (d / "MANIFEST.json").write_text(json.dumps({"files": {}}))
    return d


def _fake(monkeypatch, calls):
    def tyf(aoi, days=None, root=None, max_dry_share=None, name=None, **kw):
        calls.append(name)
        t = pd.DataFrame({"sub_class": ["young rice", "rice standing transplanted"], "acres": [0.5, 0.5],
                          "too_young": [True, False]}, index=pd.Index(["a", "b"], name="field_id"))
        t.attrs["s2_date"] = "2026-09-13"
        return t
    monkeypatch.setattr(q, "too_young_fields", tyf)


def test_dry_run_counts_and_writes_nothing(tmp_path, monkeypatch):
    d = _delivery(tmp_path)
    _fake(monkeypatch, [])
    before = (d / "aoi9991_fields.gpkg").read_bytes()
    r = rd.step_too_young(9991, dry_run=True, root=str(tmp_path))
    assert r["too_young_fields"] == 1 and r["too_young_acres"] == 0.5 and r["rice_fields"] == 2
    assert (d / "aoi9991_fields.gpkg").read_bytes() == before
    assert not (d / "aoi9991_fields_too_young.gpkg").exists()


def test_step_writes_a_new_file_and_never_touches_the_delivered_one(tmp_path, monkeypatch):
    d = _delivery(tmp_path)
    calls = []
    _fake(monkeypatch, calls)
    before, man_before = (d / "aoi9991_fields.gpkg").read_bytes(), (d / "MANIFEST.json").read_bytes()
    rd.step_too_young(9991, root=str(tmp_path))
    first = (d / "aoi9991_fields_too_young.gpkg").read_bytes()
    rd.step_too_young(9991, root=str(tmp_path))                 # running again keeps the same bytes
    assert (d / "aoi9991_fields_too_young.gpkg").read_bytes() == first
    assert (d / "aoi9991_fields.gpkg").read_bytes() == before     # delivered layer unchanged (user, 7 Oct)
    assert (d / "MANIFEST.json").read_bytes() == man_before
    f = gpd.read_file(d / "aoi9991_fields_too_young.gpkg").set_index("field_id")
    assert f.loc["a", "major_class"] == "non-rice" and f.loc["a", "sub_class"] == rd.TOO_YOUNG_CLASS
    assert f.loc["b", "major_class"] == "rice" and len(f) == 3
    assert calls == ["aoi9991_fields.gpkg"]                          # the second run keeps the written file
    j = json.loads((d / "aoi9991_too_young.json").read_text())
    assert j["too_young_fields"] == 1 and j["file"] == "aoi9991_fields_too_young.gpkg"


def test_too_young_runs_between_package_and_upload():
    assert rd.STEPS.index("package") < rd.STEPS.index("too_young") < rd.STEPS.index("upload")


def test_trees_cut_runs_after_too_young_and_before_upload():
    assert rd.STEPS.index("too_young") < rd.STEPS.index("trees_cut") < rd.STEPS.index("upload")


def test_a_failing_trees_cut_is_recorded_and_does_not_stop_the_delivery(tmp_path, monkeypatch):
    from sar_pipeline.analysis import basemap_trees as bt

    d = _delivery(tmp_path)
    monkeypatch.setattr(rd, "OUT_ROOT", str(tmp_path))
    monkeypatch.setattr(bt, "cut_aoi", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("credentials expired")))
    r = rd.step_trees_cut(9991, root=str(tmp_path))
    assert "credentials expired" in r["error"] and not (d / "aoi9991_fields_trees_cut.gpkg").exists()


def test_second_set_aois_get_their_client_names_in_the_bucket():
    assert rd.bucket_name(19) == "aoi19" and rd.bucket_name(19, "aoi19_fields.gpkg") == "aoi19_fields.gpkg"
    assert rd.bucket_name(1133) == "b2_aoi133"
    assert rd.bucket_name(1133, "aoi1133_fields_trees_cut.gpkg") == "b2_aoi133_fields_trees_cut.gpkg"
    assert rd.bucket_name(1133, "MANIFEST.json") == "MANIFEST.json"


def test_client_file_renames_a_second_set_aoi_inside_its_files(tmp_path):
    """b2 delivery copies carry the client's name inside; the first set's files are uploaded as they are."""
    import json

    import geopandas as gpd
    from shapely.geometry import box

    from sar_pipeline.analysis import rice_map_delivery as rd

    g = tmp_path / "aoi1135_fields.gpkg"
    gpd.GeoDataFrame({"field_id": ["aoi1135_p000000", "aoi11350_x"]}, geometry=[box(0, 0, 1, 1)] * 2,
                     crs=32646).to_file(g, driver="GPKG")
    j = tmp_path / "aoi1135_too_young.json"
    j.write_text(json.dumps({"aoi": 1135, "file": "aoi1135_fields.gpkg", "sha256": "old"}))
    cg = rd.client_file(1135, g)
    assert cg.name == "b2_aoi135_fields.gpkg"
    assert gpd.read_file(cg)["field_id"].tolist() == ["b2_aoi135_p000000", "aoi11350_x"]
    cj = json.loads(rd.client_file(1135, j).read_text())
    assert cj["aoi"] == "b2_aoi135" and cj["file"] == "b2_aoi135_fields.gpkg" and cj["sha256"] == rd._sha(cg)
    first = tmp_path / "aoi19_fields.gpkg"
    assert rd.client_file(19, first) == first
    m = cg.stat().st_mtime_ns
    assert rd.client_file(1135, g).stat().st_mtime_ns == m          # unchanged source: copy kept, md5 stable
