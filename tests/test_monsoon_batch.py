import pandas as pd

from sar_pipeline import monsoon_batch as mb


def test_fits_respects_the_limit_but_never_blocks_on_an_empty_queue():
    assert mb.fits(2000, 400, 2500)
    assert not mb.fits(2200, 400, 2500)
    assert mb.fits(0, 4000, 2500)


def test_wait_for_headroom_polls_until_the_job_fits():
    seen = iter([2400, 2300, 1000])
    naps = []
    active = mb.wait_for_headroom(300, 2500, count=lambda: next(seen), sleep=naps.append,
                                  poll_seconds=5, log=lambda *_: None)
    assert active == 1000 and naps == [5, 5]


def test_merge_rows_replaces_the_same_aoi_and_keeps_others(tmp_path):
    path = tmp_path / "acres.csv"
    pd.DataFrame({"aoi": ["aoi1", "aoi2"], "rice_acres": [1.0, 2.0]}).to_csv(path, index=False)
    mb.merge_rows(pd.DataFrame({"aoi": ["aoi2", "aoi3"], "rice_acres": [5.0, 6.0]}), path)
    out = pd.read_csv(path).set_index("aoi")["rice_acres"].to_dict()
    assert out == {"aoi1": 1.0, "aoi2": 5.0, "aoi3": 6.0}


def test_wait_for_s2_returns_the_failed_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "OUT_ROOT", str(tmp_path))
    pd.DataFrame({"date": ["2026-07-01", "2026-07-06"], "task_id": ["a", "b"]}).to_csv(
        tmp_path / "aoi7_export_manifest.csv", index=False)
    states = {"a": "COMPLETED", "b": "FAILED"}
    failed = mb.wait_for_s2(7, log=lambda *_: None, status_of=lambda ids: {i: states[i] for i in ids},
                            sleep=lambda *_: None)
    assert failed == ["2026-07-06"]
    assert mb.wait_for_s2(8, log=lambda *_: None) == []


def test_rule_skips_an_aoi_without_an_export(tmp_path, monkeypatch):
    monkeypatch.setattr(mb, "OUT_ROOT", str(tmp_path))
    table = mb.rule([9], log=lambda *_: None)
    assert "no Sentinel-2 export manifest" in table.loc[0, "error"]
