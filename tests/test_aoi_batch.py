import json

from sar_pipeline.analysis import aoi_batch as b


def test_season_end_is_moved_only_in_the_season_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    p = tmp_path / "config" / "aoi7_monsoon2026.yaml"
    p.write_text("season: {key: monsoon2026, timezone: X, start: '2026-03-15', end: '2026-09-24'}\n"
                 "other: {end: '2020-01-01'}\n")
    b._set_season_end(7)
    t = p.read_text()
    assert f"end: '{b.SEASON_END}'}}" in t.splitlines()[0] and "2020-01-01" in t


def test_status_marks_are_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "STATE", tmp_path)
    assert not b._done(5, "imagery")
    b._mark(5, "imagery", {"run": "r1"})
    assert b._done(5, "imagery") and json.loads((tmp_path / "aoi5" / "status.json").read_text())["imagery"]["result"]


def test_candidate_sets_exist_as_reviewed_sets():
    from sar_pipeline.analysis import curve_rules as cr
    assert all(s == 160 or s in cr.REVIEWED_SETS or s in cr.NAMED_SETS for s in b.CANDIDATE_SETS)


def test_flow_with_nothing_to_do_returns_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "STATE", tmp_path)
    for a in (1, 2):
        b._mark(a, "imagery", {"run": "r"})
        b._mark(a, "sheets", {"fields": 0})
    b.flow([1, 2], tmp_path / "logs", sleep=lambda s: (_ for _ in ()).throw(AssertionError("must not wait")))


def test_resample_never_removes_an_earlier_rounds_lock(tmp_path, monkeypatch):
    import pandas as pd
    from sar_pipeline.analysis import field_review as fr
    monkeypatch.setattr(b, "STATE", tmp_path / "state")
    monkeypatch.setattr(b, "FRESH", str(tmp_path / "fresh"))
    monkeypatch.setattr(fr, "review_dir", lambda a, fresh=None: tmp_path / f"r{a}")
    monkeypatch.setattr(b, "target_sample", lambda a: 200)
    for a, finished in ((1, True), (2, False)):
        (tmp_path / f"r{a}").mkdir()
        pd.DataFrame({"field_id": ["x"] * 5}).to_csv(tmp_path / f"r{a}" / "fields.csv", index=False)
        b._mark(a, "sheets", {"fields": 5})
        if finished:
            b._mark(a, "finish", {})
        (tmp_path / "fresh" / "locked" / f"aoi{a}").mkdir(parents=True)
    assert b.resample([1, 2]) == [1, 2]
    assert not (tmp_path / "fresh" / "locked" / "aoi1").exists()       # made by this runner: removed
    assert (tmp_path / "fresh" / "locked" / "aoi2").exists()           # an earlier round's lock: kept
    assert not b._done(1, "sheets") and not b._done(2, "sheets")


def test_universal_set_is_a_named_candidate():
    from sar_pipeline.analysis import curve_rules as cr
    assert "universal" in b.CANDIDATE_SETS and 63 in b.CANDIDATE_SETS
    assert cr.NAMED_SETS["universal"] == cr.NEW_AOI_SWITCHES and cr.NAMED_SETS["universal"]["TONE_AND_CURVE"]
    b._fill_universal()
    assert b.CANDIDATE_SWITCHES["universal"] == cr.NEW_AOI_SWITCHES


def test_stage_summary_counts_each_aoi_once_from_its_files(tmp_path, monkeypatch):
    from sar_pipeline.analysis import field_review as fr

    monkeypatch.setattr(b, "STATE", tmp_path / "state")
    monkeypatch.setattr(fr, "review_dir", lambda a, fresh=None: tmp_path / f"r{a}")
    b._mark(1, "imagery", {})
    b._mark(2, "imagery", {})
    b._mark(2, "sheets", {"fields": 2})
    b._mark(3, "sheets", {"fields": 2})
    (tmp_path / "r3" / "verdicts").mkdir(parents=True)
    (tmp_path / "r3" / "verdicts" / "a.json").write_text("{}")
    b._mark(4, "sheets", {"fields": 1})
    (tmp_path / "r4" / "verdicts").mkdir(parents=True)
    (tmp_path / "r4" / "verdicts" / "a.json").write_text("{}")
    b._mark(5, "finish", {})
    s = b.stage_summary([1, 2, 3, 4, 5, 6, 7], first_round=[7]).set_index("stage")["aois"]
    assert s.sum() == 7
    assert [s[b.STAGES[i]] for i in range(7)] == [1, 1, 1, 1, 1, 1, 1]
