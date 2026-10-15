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


def test_round2_maps_s2_exports_to_their_bucket_copy():
    assert sp.gcs_name("sar-rice-mapper/data/s2_dates_masks/aoi1/aoi1_2026-10_S2_2026-10-06.tif", "B", "B/X") == \
        "B/s2_dates_masks/aoi1/aoi1_2026-10_S2_2026-10-06.tif"
    assert sp.gcs_name("data/delineation/a.gpkg", "B", "B/X") == "B/X/data/delineation/a.gpkg"


def test_backup_list_all_takes_data_not_code(tmp_path):
    for rel in ("sar-rice-mapper/processed/a.tif", "data/b.gpkg", "sar-rice-mapper/src/c.py",
                "sar-rice-mapper/.venv/d.py", "outputs/e.png"):
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    assert [r for r, _ in sp.backup_list_all(tmp_path)] == ["data/b.gpkg", "outputs/e.png",
                                                            "sar-rice-mapper/processed/a.tif"]


def test_aoi_prefixes_do_not_catch_other_aois():
    pre = tuple(sp.aoi_prefixes(1))
    assert "sar-rice-mapper/processed/aoi1/monsoon2026/LATEST.txt".startswith(pre)
    assert not "sar-rice-mapper/processed/aoi11/monsoon2026/LATEST.txt".startswith(pre)
    assert not "sar-rice-mapper/processed/_batch/s2_2026/baseline_v3/aoi11_monsoon2026_final.tif".startswith(pre)


def test_internal_field_id_maps_the_client_name():
    from sar_pipeline.qgis_review import internal_field_id

    assert internal_field_id("b2_aoi198_p000277") == "aoi1198_p000277"
    assert internal_field_id("aoi66_p000013") == "aoi66_p000013"


def test_pixel_at_round_trips_with_locate(monkeypatch):
    """pixel_at is the inverse of pixel_report.locate's lon / lat for a pixel centre."""
    from sar_pipeline import qgis_review as q
    from sar_pipeline import storage_plan
    from sar_pipeline.analysis import pixel_report as pr

    grid = {"crs": "EPSG:32646", "x0": 800000.0, "y0": 2000000.0, "res": 10.0, "width": 50, "height": 40}
    monkeypatch.setattr(storage_plan, "ensure_aoi", lambda *a, **k: {})
    monkeypatch.setattr(pr, "locate", lambda *a, **k: {"grid": grid})
    import pyproj

    pid = 7 * 50 + 13
    x, y = grid["x0"] + 13.5 * 10, grid["y0"] - 7.5 * 10
    lon, lat = pyproj.Transformer.from_crs("EPSG:32646", "EPSG:4326", always_xy=True).transform(x, y)
    assert q.pixel_at(1, lon, lat) == pid
