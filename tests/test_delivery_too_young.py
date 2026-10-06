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
    assert not (d / "aoi9991_fields_before_too_young.gpkg").exists()


def test_step_relabels_keeps_the_before_copy_and_repeats_the_same(tmp_path, monkeypatch):
    d = _delivery(tmp_path)
    calls = []
    _fake(monkeypatch, calls)
    rd.step_too_young(9991, root=str(tmp_path))
    rd.step_too_young(9991, root=str(tmp_path))                 # a second run reads the kept copy again
    f = gpd.read_file(d / "aoi9991_fields.gpkg").set_index("field_id")
    assert f.loc["a", "major_class"] == "non-rice" and f.loc["a", "sub_class"] == rd.TOO_YOUNG_CLASS
    assert f.loc["b", "major_class"] == "rice" and len(f) == 3                     # polygons kept, others untouched
    b = gpd.read_file(d / "aoi9991_fields_before_too_young.gpkg").set_index("field_id")
    assert b.loc["a", "major_class"] == "rice"                                      # the copy is the packaged layer
    assert calls == ["aoi9991_fields.gpkg", "aoi9991_fields_before_too_young.gpkg"]
    man = json.loads((d / "MANIFEST.json").read_text())
    assert man["too_young"]["too_young_fields"] == 1 and "aoi9991_fields_before_too_young.gpkg" in man["files"]


def test_too_young_runs_between_package_and_upload():
    assert rd.STEPS.index("package") < rd.STEPS.index("too_young") < rd.STEPS.index("upload")
