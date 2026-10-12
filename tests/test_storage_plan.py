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


def test_backup_list_keeps_what_inspection_reads(tmp_path):
    sr = tmp_path / sp.SR

    def put(rel, data="x"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(data)

    put(f"{sp.SR}/_batch/{sp.KEEP_SERIES}/aoi1/a.tif")                 # pinned series: kept
    put(f"{sp.SR}/_batch/s2_2026_hyb50/aoi1/a.tif")                     # other series: moved
    put(f"{sp.SR}/_batch/s2_2026/rice_fresh/aoi1/field_review/sheets/f.png")   # moved
    put(f"{sp.SR}/_batch/s2_2026/rice_fresh/aoi1/field_review/verdicts/f.json")  # kept
    put(f"{sp.SR}/_batch/s2_2026/report/final_audit/bad_passes_all.csv")  # kept
    put(f"{sp.SR}/_batch/s2_2026/report/other.csv")                      # moved
    put(f"{sp.SR}/_batch/s2_2026/aoi1/aoi1_monsoon2026_final.tif")       # kept
    put(f"{sp.SR}/_batch/s2_2026/aoi1/aoi1_monsoon2026_x.tif")           # moved
    put(f"{sp.SR}/aoi1/year2025/grid/grid_def.json")                     # kept
    put(f"{sp.SR}/aoi1/year2025/runs/v1/a.tif")                          # moved
    put(f"{sp.SR}/aoi1/monsoon2026/ANALYSIS_RUN.txt", "v002\n")
    put(f"{sp.SR}/aoi1/monsoon2026/LATEST.txt", "v002\n")
    put(f"{sp.SR}/aoi1/monsoon2026/runs/v002/a.tif")                     # pinned run: kept
    put(f"{sp.SR}/aoi1/monsoon2026/runs/v001/a.tif")                     # old run: moved
    got = {r for r, _ in sp.backup_list(tmp_path)}
    assert got == {f"{sp.SR}/_batch/s2_2026_hyb50/aoi1/a.tif",
                   f"{sp.SR}/_batch/s2_2026/rice_fresh/aoi1/field_review/sheets/f.png",
                   f"{sp.SR}/_batch/s2_2026/report/other.csv", f"{sp.SR}/_batch/s2_2026/aoi1/aoi1_monsoon2026_x.tif",
                   f"{sp.SR}/aoi1/year2025/runs/v1/a.tif", f"{sp.SR}/aoi1/monsoon2026/runs/v001/a.tif"}
