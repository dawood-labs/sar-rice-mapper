"""Bringing in newer imagery without changing accepted results: the radar-run pin, the frozen raw dates of a
5-day series, and the read-only "is there anything new?" helpers. No network."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from sar_pipeline import cli, config, newest_imagery as ni
from sar_pipeline.analysis import mask_experiment as me
from sar_pipeline.analysis import ndvi_5day as nd
from sar_pipeline.errors import PipelineError


# ------------------------------------------------------------------ radar run pin
def test_analysis_reads_the_pinned_run_after_a_newer_run_is_created(make_project):
    cfg = make_project()
    first = config.new_run(cfg).name
    assert config.analysis_run_dir(cfg).name == first          # nothing pinned: the newest run, as before
    config.pin_analysis_run(cfg)
    assert config.pinned_run(cfg) == first
    second = config.new_run(cfg).name
    assert second != first
    assert config.run_dir(cfg).name == second                  # the pipeline stages still see the new run
    assert config.analysis_run_dir(cfg).name == first          # the analysis does not move silently
    assert config.analysis_run_dir(cfg, second).name == second  # ... unless asked for the new run


def test_switching_a_pin_needs_replace(make_project):
    cfg = make_project()
    first = config.new_run(cfg).name
    config.pin_analysis_run(cfg, first)
    second = config.new_run(cfg).name
    with pytest.raises(PipelineError):
        config.pin_analysis_run(cfg, second)
    assert config.pin_analysis_run(cfg, second, replace=True) == second
    assert config.analysis_run_dir(cfg).name == second
    assert config.pin_analysis_run(cfg, second) == second        # pinning the same run again is fine


def test_pin_refuses_a_run_that_does_not_exist(make_project):
    cfg = make_project()
    config.new_run(cfg)
    with pytest.raises(FileNotFoundError):
        config.pin_analysis_run(cfg, "v099_20990101")


def test_cli_pin_run(make_project, capsys):
    cfg = make_project({"s1": {"tracks": [{"track_id": "RO001_ASC", "role": "primary"}]}})
    first = config.new_run(cfg).name
    assert cli.main(["--config", cfg["_config_path"], "pin-run"]) == 0
    assert first in capsys.readouterr().out
    second = config.new_run(cfg).name
    assert cli.main(["--config", cfg["_config_path"], "pin-run", "--run", second]) == 1     # refused without --replace
    assert cli.main(["--config", cfg["_config_path"], "pin-run", "--run", second, "--replace"]) == 0
    assert config.pinned_run(cfg) == second


# ------------------------------------------------------------------ frozen raw dates of a 5-day series
def _files(tmp_path, dates):
    paths = []
    for d in dates:
        p = tmp_path / f"aoi1_{d[:7]}_S2_{d}.tif"
        p.write_bytes(b"")
        paths.append(p)
    return paths


def test_select_dates_reads_exactly_the_recorded_dates(tmp_path):
    paths = _files(tmp_path, ["2026-09-26", "2026-09-28", "2026-10-01"])
    picked = nd.select_dates(paths, ["2026-09-26", "2026-09-28"])
    assert [nd.file_date(p) for p in picked] == ["2026-09-26", "2026-09-28"]   # the newer date stays out


def test_select_dates_fails_when_a_recorded_date_is_gone(tmp_path):
    paths = _files(tmp_path, ["2026-09-26"])
    with pytest.raises(FileNotFoundError, match="2026-09-28"):
        nd.select_dates(paths, ["2026-09-26", "2026-09-28"])


def test_series_dates_record_is_written_once(tmp_path):
    assert nd.series_dates(tmp_path, "aoi1") is None                 # an old series without a record
    nd.write_series_dates(tmp_path, "aoi1", ["2026-09-28", "2026-09-26", "2026-09-26"])
    assert nd.series_dates(tmp_path, "aoi1") == ["2026-09-26", "2026-09-28"]
    nd.write_series_dates(tmp_path, "aoi1", ["2026-09-26", "2026-09-28"])    # same list: fine
    with pytest.raises(FileExistsError):
        nd.write_series_dates(tmp_path, "aoi1", ["2026-09-26", "2026-09-28", "2026-10-01"])


def test_new_series_variant_has_the_old_mask_and_its_own_folder():
    assert me.VARIANTS["hyb40m1late_20261001"] == me.VARIANTS["hyb40m1late"]
    assert me.root("hyb40m1late_20261001") != me.root("hyb40m1late")
    assert me.SERIES_END["hyb40m1late_20261001"] > me.SERIES_END["hyb40m1late"]


# ------------------------------------------------------------------ what is new
def test_passes_after_keeps_only_newer_passes_of_configured_tracks():
    slices = pd.DataFrame({
        "track_id": ["RO001_ASC", "RO001_ASC", "RO001_ASC", "RO002_DSC", "RO009_ASC"],
        "datetime_utc": ["2026-09-13T11:45:00Z", "2026-09-25T11:45:00Z", "2026-09-25T11:45:20Z",
                         "2026-09-22T23:30:00Z", "2026-09-30T11:00:00Z"],
        "platform": ["A", "D", "D", "C", "A"],
    })
    after = {"RO001_ASC": pd.Timestamp("2026-09-13"), "RO002_DSC": pd.Timestamp("2026-09-22")}
    out = ni.passes_after(slices, after)
    assert out.to_dict("records") == [{"track_id": "RO001_ASC", "date_utc": pd.Timestamp("2026-09-25"),
                                       "platforms": "D", "n_slices": 2}]
    assert ni.passes_after(slices.iloc[0:0], after).empty


def test_run_last_dates_reads_each_track_stack(tmp_path):
    for track, dates in (("RO001_ASC", ["2026-09-01", "2026-09-13"]), ("RO002_DSC", ["2026-09-22"])):
        d = tmp_path / "stack" / f"track_{track}"
        d.mkdir(parents=True)
        pd.DataFrame({"date_utc": dates}).to_csv(d / "dates.csv", index=False)
    assert ni.run_last_dates(tmp_path) == {"RO001_ASC": pd.Timestamp("2026-09-13"),
                                           "RO002_DSC": pd.Timestamp("2026-09-22")}


def test_diff_stats_counts_only_pixels_valid_in_both():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.5, 3.0, np.nan])
    s = ni.diff_stats(a, b, tol=0.1)
    assert s["pixels"] == 2 and s["pct_over_tol"] == 50.0 and s["median_abs"] == 0.25 and s["median_diff"] == 0.25


def test_compare_series_window_by_window(tmp_path):
    def write(root, windows, values, name="ndvi5d"):
        p = root / "aoi1" / f"aoi1_{name}.tif"
        p.parent.mkdir(parents=True, exist_ok=True)
        cube = np.asarray(values, dtype="float32").reshape(len(windows), 1, 2)
        with rasterio.open(p, "w", driver="GTiff", width=2, height=1, count=len(windows), dtype="float32",
                           nodata=-9999.0, crs="EPSG:32631", transform=from_origin(500000, 10, 10, 10)) as ds:
            ds.write(cube)
            ds.descriptions = tuple(windows)
    new_windows = ["2026-09-21", "2026-09-26", "2026-10-01"]
    write(tmp_path / "old", ["2026-09-21", "2026-09-26"], [0.5, 0.6, 0.7, 0.8])
    write(tmp_path / "old", ["2026-09-21", "2026-09-26"], [0, 0, 5, 15], name="gapdays5d")
    write(tmp_path / "new", new_windows, [0.5, 0.6, 0.7, 0.9, 0.9, 0.9])
    write(tmp_path / "new", new_windows, [0.5, 0.6, 0.7, 0.9, 0.9, -9999.0], name="ndvi5d_raw")
    t = ni.compare_series(1, tmp_path / "old", tmp_path / "new")
    assert t["in"].tolist() == ["both", "both", "new"]
    assert t.loc[0, "pct_over_tol"] == 0.0 and t.loc[1, "pct_over_tol"] == 50.0
    assert t.loc[2, "observed_pct"] == 50.0
    assert t.loc[1, "old_gap_days_median"] == 10.0 and t.loc[1, "median_diff"] == 0.05                       # one of two pixels seen on the new date


def test_check_report_is_json_serialisable():
    s2 = pd.DataFrame({"date": ["2026-10-01"], "aoi_clear_pct": [82.0]})
    s2["mostly_clear"] = s2["aoi_clear_pct"] >= ni.MOSTLY_CLEAR_PCT
    assert json.loads(s2.to_json(orient="records"))[0]["mostly_clear"] is True


# ------------------------------------------------------------------ series pin (which 5-day series an AOI reads)
def test_series_pin_chooses_the_series_per_aoi_and_switching_needs_replace(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "repo_root", lambda: tmp_path)
    for root in ("s_old", "s_new"):
        (tmp_path / root / "aoi7").mkdir(parents=True)
        (tmp_path / root / "aoi7" / "aoi7_ndvi5d.tif").write_bytes(b"")
    assert nd.analysis_series_root(7) == nd.ANALYSIS_SERIES_DEFAULT          # no pin: the accepted series
    nd.pin_analysis_series(7, "s_old")
    assert nd.analysis_series_root(7) == "s_old" and nd.analysis_series_root(8) == nd.ANALYSIS_SERIES_DEFAULT
    with pytest.raises(FileExistsError):
        nd.pin_analysis_series(7, "s_new")
    nd.pin_analysis_series(7, "s_new", replace=True)
    assert nd.analysis_series_root(7) == "s_new"
    with pytest.raises(FileNotFoundError):
        nd.pin_analysis_series(8, "s_new")                                  # no aoi8 series there


def test_notebook_reads_the_pinned_series_unless_one_is_forced(monkeypatch):
    from sar_pipeline import qgis_review as q

    monkeypatch.setattr(nd, "analysis_series_root", lambda aoi: f"pinned_for_{aoi}")
    assert q.series_root(13) == "pinned_for_13"
    monkeypatch.setattr(q, "SERIES_ROOT", "forced")
    assert q.series_root(13) == "forced"


# ------------------------------------------------------------------ artefact list: only new passes are added
def test_new_passes_are_appended_and_old_rows_never_change(tmp_path, monkeypatch):
    from sar_pipeline.analysis import final_audit as fa

    old = pd.DataFrame([dict(aoi="aoi7", track="T1", pol=pol, date="2026-09-13", stable_pixels=500,
                             stable_jump_db=0.2, bad=False, reason="") for pol in ("VV", "VH")])
    old.to_csv(tmp_path / "bad_passes_all.csv", index=False)

    def judged(aoi_id, run_id=None):            # the newer run: the old pass reads a bit differently, plus a new pass
        return pd.DataFrame([dict(aoi=f"aoi{aoi_id}", track="T1", pol=pol, date=d, stable_pixels=500,
                                  stable_jump_db=j, bad=False)
                             for d, j in (("2026-09-13", -3.5), ("2026-09-25", 0.1)) for pol in ("VV", "VH")])

    monkeypatch.setattr(fa, "bad_passes", judged)
    new = fa.add_new_passes([7], "v002", audit_dir=tmp_path)
    allr = pd.read_csv(tmp_path / "bad_passes_all.csv")
    assert list(new["date"].unique()) == ["2026-09-25"] and not new["bad"].any()
    assert len(allr) == 4 and not allr[allr["date"] == "2026-09-13"]["bad"].any()   # old verdict kept
    assert (tmp_path / "bad_passes_all_prev.csv").exists()
