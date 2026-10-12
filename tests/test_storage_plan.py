"""storage_plan: reads are traced, VRT sources followed, paths collapsed to patterns."""
from pathlib import Path

from sar_pipeline import storage_plan as sp


def test_trace_reads_sees_python_open_and_vrt_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "src.tif").write_bytes(b"0")
    (tmp_path / "s.vrt").write_text('<VRTDataset><SourceFilename relativeToVRT="1">src.tif</SourceFilename></VRTDataset>')
    got = sp.trace_reads(lambda: [Path("a.txt").read_text(), Path("s.vrt").read_text()])
    assert {"a.txt", "s.vrt"} <= got
    assert "src.tif" in sp.with_vrt_sources(got)


def test_patterns_collapse_ids_and_dates():
    p = sp.patterns(["processed/aoi12/runs/v001_20261006/VH_20260316.vrt",
                     "processed/aoi7/runs/v001_20261006/VH_20260322.vrt"])
    assert p == {"processed/aoi<N>/runs/v001_<date>/VH_<date>.vrt": 2}
