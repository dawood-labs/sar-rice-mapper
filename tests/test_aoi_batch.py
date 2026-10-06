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
    assert all(s == 160 or s in cr.REVIEWED_SETS for s in b.CANDIDATE_SETS)


def test_flow_with_nothing_to_do_returns_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "STATE", tmp_path)
    for a in (1, 2):
        b._mark(a, "imagery", {"run": "r"})
        b._mark(a, "sheets", {"fields": 0})
    b.flow([1, 2], tmp_path / "logs", sleep=lambda s: (_ for _ in ()).throw(AssertionError("must not wait")))
