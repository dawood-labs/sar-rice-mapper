import pandas as pd

from sar_pipeline.analysis import rule_records as rr


def test_record_keeps_history_and_rewrites_the_switch_header(tmp_path):
    rr.record(160, "chosen", "set A", reason="best", when="t1", root=tmp_path)
    rr.record(160, "changed", "set B", scores=pd.DataFrame({"set": ["A"], "right %": [54.0]}), when="t2",
              root=tmp_path)
    t = (tmp_path / "aoi160.md").read_text()
    assert t.count("## Switches now") == 1 and t.index("t1 - chosen") < t.index("t2 - changed")
    assert "| set | right % |" in t and "Why: best" in t


def test_record_never_writes_a_storage_path(tmp_path):
    rr.record(5, "delivered", "gs://some-bucket/some-project/rice_map_x/aoi5/ (3 dates)", when="t", root=tmp_path)
    t = (tmp_path / "aoi5.md").read_text()
    assert "gs://" not in t and "some-bucket" not in t and "rice_map_x/aoi5/" in t
