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
