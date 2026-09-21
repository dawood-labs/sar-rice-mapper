"""CLI tests: argument parsing, --yes gating and dispatch. No network.

Pipeline modules are replaced with fakes in sys.modules; the CLI imports them lazily via
importlib, so it picks up the fakes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from sar_pipeline import cli, config
from sar_pipeline.errors import AuditRequired, DecisionRequired, PipelineError


class Recorder:
    def __init__(self):
        self.calls = []

    def fn(self, name, result=None, exc=None):
        def _f(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            if exc is not None:
                raise exc
            return result(*args, **kwargs) if callable(result) else result

        return _f

    def names(self):
        return [c[0] for c in self.calls]

    def kwargs(self, name):
        return [c[2] for c in self.calls if c[0] == name]


def _stacks_result(cfg, run_path, track_ids, force=False):
    return {t: f"stack/track_{t}" for t in track_ids}


def _manifest(states, chunks=None):
    return pd.DataFrame({"task_key": [f"t{i}" for i in range(len(states))],
                         "track_id": ["RO123_ASC"] * len(states), "state": states,
                         "chunk_name": chunks or ["chunk_r00c00"] * len(states)})


@pytest.fixture
def project(make_project):
    cfg = make_project({"s1": {"tracks": [{"track_id": "RO123_ASC", "role": "primary"}]}})
    config.new_run(cfg)
    return cfg


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()
    monkeypatch.setitem(sys.modules, "sar_pipeline.auth", SimpleNamespace(init_ee=r.fn("init_ee")))
    return r


def edit_user_config(cfg, section, key, value):
    """Edit the user's YAML (not the frozen run copy) after the run was created."""
    p = Path(cfg["_config_path"])
    d = yaml.safe_load(p.read_text())
    d.setdefault(section, {})[key] = value
    p.write_text(yaml.safe_dump(d, sort_keys=False))


def run(cfg, *argv):
    return cli.main(["--config", cfg["_config_path"], *argv])


# ---------------------------------------------------------------- parsing
def test_config_is_required():
    with pytest.raises(SystemExit) as e:
        cli.main(["grid"])
    assert e.value.code == 2


def test_export_needs_pilot_or_all(project):
    with pytest.raises(SystemExit) as e:
        run(project, "export")
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        run(project, "export", "--pilot", "chunk_r00c00", "--all")


@pytest.mark.parametrize("extra", [[], ["--lon", "1.0"], ["--pid", "5", "--lon", "1", "--lat", "2"]])
def test_pixel_requires_pid_xor_lonlat(project, extra):
    with pytest.raises(SystemExit) as e:
        run(project, "pixel", "--track", "RO123_ASC", *extra)
    assert e.value.code == 2


def test_parser_lists_all_commands():
    help_text = cli.build_parser().format_help()
    for cmd in ["grid", "audit", "new-run", "export", "monitor", "download", "stack", "pixel", "resources"]:
        assert cmd in help_text


# ---------------------------------------------------------------- export gating
@pytest.fixture
def fake_export(monkeypatch, rec):
    mod = SimpleNamespace(
        plan_exports=rec.fn("plan_exports", _manifest(["PLANNED", "PLANNED", "NOT_COVERED"],
                                                       ["chunk_r00c00", "chunk_r00c01", "chunk_r00c00"])),
        confirm_exports=rec.fn("confirm_exports", _manifest(["PENDING"])),
        retry_failed=rec.fn("retry_failed", _manifest(["PENDING"])),
        submit_pending=rec.fn("submit_pending", _manifest(["SUBMITTED", "SUBMITTED"])),
        monitor=rec.fn("monitor", _manifest(["VERIFIED", "FAILED"])),
        write_failed_report=rec.fn("write_failed_report", "failed_chunks.csv"),
    )
    monkeypatch.setitem(sys.modules, "sar_pipeline.export", mod)
    monkeypatch.setitem(sys.modules, "sar_pipeline.manifest",
                        SimpleNamespace(read_manifest=rec.fn("read_manifest", _manifest(["COMPLETED", "FAILED", "PLANNED"]))))
    return mod


def test_export_without_yes_plans_but_does_not_confirm_or_submit(project, rec, fake_export, capsys):
    assert run(project, "export", "--all") == cli.EXIT_CONFIRM
    assert "plan_exports" in rec.names()
    assert "confirm_exports" not in rec.names() and "submit_pending" not in rec.names()
    out = capsys.readouterr().out
    assert "--yes" in out and "To export    : 2 tasks" in out and "Not covered  : 1" in out


def test_export_with_yes_confirms_scope_then_submits(project, rec, fake_export, capsys):
    assert run(project, "export", "--pilot", "chunk_r00c00", "--yes") == cli.EXIT_OK
    assert rec.kwargs("plan_exports")[0]["chunk_names"] == ["chunk_r00c00"]
    assert rec.kwargs("confirm_exports")[0]["chunk_names"] == ["chunk_r00c00"]
    assert rec.kwargs("submit_pending")[0]["confirmed"] is True
    assert rec.kwargs("plan_exports")[0]["force"] is False
    assert rec.names().index("confirm_exports") < rec.names().index("submit_pending")
    assert "To export    : 1 tasks" in capsys.readouterr().out


def test_export_force_rebuilds_plan_only(project, rec, fake_export):
    assert run(project, "export", "--all", "--force") == cli.EXIT_CONFIRM
    assert rec.kwargs("plan_exports")[0]["force"] is True
    assert "submit_pending" not in rec.names() and "confirm_exports" not in rec.names()


def test_export_all_passes_no_chunk_filter(project, rec, fake_export):
    run(project, "export", "--all", "--yes")
    assert rec.kwargs("plan_exports")[0]["chunk_names"] is None
    assert rec.kwargs("confirm_exports")[0]["chunk_names"] is None


def test_export_retry_failed_gating(project, rec, fake_export, capsys):
    assert run(project, "export", "--retry-failed") == cli.EXIT_CONFIRM
    assert "retry_failed" not in rec.names() and "submit_pending" not in rec.names()
    assert "FAILED rows  : 1" in capsys.readouterr().out
    assert run(project, "export", "--retry-failed", "--yes") == cli.EXIT_OK
    assert rec.names().index("retry_failed") < rec.names().index("submit_pending")
    assert "plan_exports" not in rec.names()
    assert rec.kwargs("retry_failed")[0]["cfg"]["_config_path"].endswith("run_config.yaml")   # validated run


def _foreign_monitor_lock(cfg):
    import json

    run_path = config.run_dir(cfg)
    (run_path / ".monitor.lock").write_text(json.dumps({"pid": 1, "space": "another-machine", "token": "t"}))
    return run_path


@pytest.mark.parametrize("scope", [["--all"], ["--pilot", "chunk_r00c00"], ["--retry-failed"]])
def test_export_yes_refused_while_a_monitor_holds_the_run(project, rec, fake_export, capsys, scope):
    _foreign_monitor_lock(project)
    assert run(project, "export", *scope, "--yes") == cli.EXIT_ERROR
    assert not {"confirm_exports", "retry_failed", "submit_pending"} & set(rec.names())   # nothing changed
    assert "monitor" in capsys.readouterr().err


def test_export_scope_flags_are_exclusive_and_required(project):
    with pytest.raises(SystemExit) as e:
        run(project, "export", "--all", "--retry-failed")
    assert e.value.code == 2


def test_monitor_gating(project, rec, fake_export, capsys):
    assert run(project, "monitor") == cli.EXIT_CONFIRM
    assert "monitor" not in rec.names() and "init_ee" not in rec.names()
    out = capsys.readouterr().out
    assert "Planned  : 1 rows NOT confirmed" in out and "Pending  : 0" in out
    assert run(project, "monitor", "--yes") == cli.EXIT_OK
    assert rec.kwargs("monitor")[0]["confirmed"] is True
    assert "write_failed_report" in rec.names()
    assert "confirm_exports" not in rec.names()          # monitor never confirms PLANNED rows


# ---------------------------------------------------------------- download gating
@pytest.fixture
def fake_download(monkeypatch, rec):
    est = {"n_files": 3, "bytes": 3 * 1024**2, "disk_free_bytes": 10 * 1024**3, "fits": True}
    mod = SimpleNamespace(
        estimate_download=rec.fn("estimate_download", est),
        download_completed=rec.fn("download_completed", _manifest(["VERIFIED"])),
    )
    monkeypatch.setitem(sys.modules, "sar_pipeline.download", mod)
    return mod


def test_download_gating(project, rec, fake_download, capsys):
    assert run(project, "download") == cli.EXIT_CONFIRM
    assert "download_completed" not in rec.names()
    assert "Files      : 3" in capsys.readouterr().out
    assert run(project, "download", "--yes") == cli.EXIT_OK
    assert rec.kwargs("download_completed")[0] == dict(confirmed=True, force=False, retry_failed=False, deep=False)
    assert run(project, "download", "--yes", "--force", "--retry-failed", "--deep") == cli.EXIT_OK
    assert rec.kwargs("download_completed")[1] == dict(confirmed=True, force=True, retry_failed=True, deep=True)


def test_run_id_is_forwarded(project, rec, fake_download):
    run_id = config.list_runs(project)[-1]
    assert run(project, "download", "--run", run_id) == cli.EXIT_CONFIRM
    _, args, _ = [c for c in rec.calls if c[0] == "estimate_download"][0]
    assert args[1].name == run_id


def test_unknown_run_is_error(project, rec, fake_download, capsys):
    assert run(project, "download", "--run", "v999_20000101") == cli.EXIT_ERROR
    assert "ERROR" in capsys.readouterr().err


# ---------------------------------------------------------------- other dispatch
def test_grid_builds_grid_and_index(project, monkeypatch, rec):
    gd = {"width": 10, "height": 8, "res": 10, "crs": "EPSG:6933", "chunks": [{}], "chunk_px": 256, "pid_dtype": "int32"}
    monkeypatch.setitem(sys.modules, "sar_pipeline.grid", SimpleNamespace(build_grid=rec.fn("build_grid", gd)))
    monkeypatch.setitem(sys.modules, "sar_pipeline.index",
                        SimpleNamespace(build_pixel_index=rec.fn("build_pixel_index", "pixel_index.tif")))
    assert run(project, "grid") == cli.EXIT_OK
    assert rec.names() == ["build_grid", "build_pixel_index"]


def test_audit_inits_ee_then_runs(project, monkeypatch, rec):
    monkeypatch.setitem(sys.modules, "sar_pipeline.audit", SimpleNamespace(run_audit=rec.fn("run_audit", "audit/x")))
    assert run(project, "audit") == cli.EXIT_OK
    assert rec.names() == ["init_ee", "run_audit"]
    assert rec.kwargs("run_audit")[0] == {"force": False}


def test_audit_resume_older_folder_and_force(project, monkeypatch, rec):
    monkeypatch.setitem(sys.modules, "sar_pipeline.audit", SimpleNamespace(run_audit=rec.fn("run_audit", "audit/x")))
    assert run(project, "audit", "--audit-date", "20260915", "--force") == cli.EXIT_OK
    assert rec.kwargs("run_audit")[0] == {"force": True, "audit_date": "20260915"}


def test_stack_uses_config_tracks(project, monkeypatch, rec):
    monkeypatch.setitem(sys.modules, "sar_pipeline.stack", SimpleNamespace(build_stacks=rec.fn("build_stacks", _stacks_result)))
    assert run(project, "stack") == cli.EXIT_OK
    assert [c[1][2] for c in rec.calls] == [["RO123_ASC"]]
    assert rec.kwargs("build_stacks")[0]["force"] is False
    assert run(project, "stack", "--force") == cli.EXIT_OK
    assert rec.kwargs("build_stacks")[1]["force"] is True


def test_grid_has_no_force_flag(project):
    with pytest.raises(SystemExit):
        run(project, "grid", "--force")


def test_stack_without_tracks_is_audit_required(make_project, monkeypatch, rec, capsys):
    cfg = make_project()
    config.new_run(cfg)
    monkeypatch.setitem(sys.modules, "sar_pipeline.stack", SimpleNamespace(build_stacks=rec.fn("build_stacks")))
    assert run(cfg, "stack") == cli.EXIT_ERROR
    assert "AuditRequired" in capsys.readouterr().err


def test_pipeline_errors_exit_1(project, monkeypatch, rec, capsys):
    monkeypatch.setitem(sys.modules, "sar_pipeline.stack",
                        SimpleNamespace(build_stacks=rec.fn("build_stacks", exc=PipelineError("boom"))))
    assert run(project, "stack", "--track", "RO123_ASC") == cli.EXIT_ERROR
    assert "boom" in capsys.readouterr().err


def test_pixel_lonlat_dispatch(project, monkeypatch, rec, capsys):
    df = pd.DataFrame({"date_utc": ["2026-07-05"], "VV_db": [-10.0], "VH_db": [-16.0]})
    monkeypatch.setitem(sys.modules, "sar_pipeline.pixel_query", SimpleNamespace(
        pixel_timeseries=rec.fn("pixel_timeseries", df),
        pixel_timeseries_lonlat=rec.fn("pixel_timeseries_lonlat", df),
    ))
    assert run(project, "pixel", "--track", "RO123_ASC", "--lon", "1.5", "--lat", "2.5") == cli.EXIT_OK
    assert rec.names() == ["pixel_timeseries_lonlat"]
    assert rec.calls[0][1][3:] == (1.5, 2.5)
    assert "-16.0" in capsys.readouterr().out


def test_pixel_pid_dispatch(project, monkeypatch, rec):
    df = pd.DataFrame({"date_utc": ["2026-07-05"]})
    monkeypatch.setitem(sys.modules, "sar_pipeline.pixel_query", SimpleNamespace(
        pixel_timeseries=rec.fn("pixel_timeseries", df),
        pixel_timeseries_lonlat=rec.fn("pixel_timeseries_lonlat", df),
    ))
    assert run(project, "pixel", "--track", "RO123_ASC", "--pid", "42") == cli.EXIT_OK
    assert rec.calls[0][1][3] == 42


def test_resources_command_prints_machine(project, capsys):
    assert run(project, "resources") == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "usable CPUs" in out and "Download threads" in out and "CPU workers" in out


def test_new_run_creates_folder(project, capsys):
    before = config.list_runs(project)
    assert run(project, "new-run") == cli.EXIT_OK
    assert len(config.list_runs(project)) == len(before) + 1


# ---------------------------------------------------------------- frozen run config
def test_new_run_requires_tracks(make_project, capsys):
    cfg = make_project()
    assert run(cfg, "new-run") == cli.EXIT_ERROR
    assert "AuditRequired" in capsys.readouterr().err
    assert config.list_runs(cfg) == []


def _fake_stack(monkeypatch, rec):
    monkeypatch.setitem(sys.modules, "sar_pipeline.stack", SimpleNamespace(build_stacks=rec.fn("build_stacks", _stacks_result)))


def test_run_scoped_commands_use_frozen_config_and_warn(project, monkeypatch, rec, capsys):
    _fake_stack(monkeypatch, rec)
    frozen_dtype = project["export"]["dtype"]
    edit_user_config(project, "export", "dtype", "int16")
    edit_user_config(project, "ard", "speckle_kernel", 9)

    assert run(project, "stack") == cli.EXIT_OK
    used_cfg = rec.calls[0][1][0]
    assert used_cfg["_config_path"].endswith("run_config.yaml")
    assert used_cfg["export"]["dtype"] == frozen_dtype
    err = capsys.readouterr().err
    assert "WARNING" in err and "export.dtype" in err and "ard.speckle_kernel" in err and "new run" in err


def test_no_warning_when_config_unchanged(project, monkeypatch, rec, capsys):
    _fake_stack(monkeypatch, rec)
    assert run(project, "stack") == cli.EXIT_OK
    assert "WARNING" not in capsys.readouterr().err


def test_acknowledged_issues_come_from_user_config(project, monkeypatch, rec, capsys):
    _fake_stack(monkeypatch, rec)
    issue = "LOW_VALID:RO123_ASC:20260623"
    edit_user_config(project, "qa", "acknowledged_issues", [issue])

    assert run(project, "stack") == cli.EXIT_OK
    assert rec.calls[0][1][0]["qa"]["acknowledged_issues"] == [issue]
    assert "WARNING" not in capsys.readouterr().err


def test_tracks_changed_after_run_are_not_applied(project, monkeypatch, rec, capsys):
    _fake_stack(monkeypatch, rec)
    edit_user_config(project, "s1", "tracks", [{"track_id": "RO045_DSC", "role": "primary"}])
    assert run(project, "stack") == cli.EXIT_OK
    assert [c[1][2] for c in rec.calls] == [["RO123_ASC"]]
    assert "s1.tracks" in capsys.readouterr().err


def test_pixel_and_export_receive_run_config(project, monkeypatch, rec, fake_export):
    df = pd.DataFrame({"date_utc": ["2026-07-05"]})
    monkeypatch.setitem(sys.modules, "sar_pipeline.pixel_query", SimpleNamespace(
        pixel_timeseries=rec.fn("pixel_timeseries", df),
        pixel_timeseries_lonlat=rec.fn("pixel_timeseries_lonlat", df),
    ))
    run(project, "pixel", "--track", "RO123_ASC", "--pid", "1")
    run(project, "export", "--all")
    for name in ("pixel_timeseries", "plan_exports"):
        used = [c for c in rec.calls if c[0] == name][0][1][0]
        assert used["_config_path"].endswith("run_config.yaml"), name


def test_config_differences_helper():
    a = {"ard": {"speckle_kernel": 5}, "qa": {"acknowledged_issues": ["x"], "min_valid_pct": 80}, "rain": {"level": "aoi"}}
    b = {"ard": {"speckle_kernel": 7}, "qa": {"acknowledged_issues": [], "min_valid_pct": 80}, "rain": {"level": "chunk"}}
    # audit/rain differences are warned too (the frozen values are used); acknowledged_issues is excluded
    assert cli.config_differences(a, b) == ["ard.speckle_kernel", "rain.level"]


def test_ee_project_stays_the_runs_project_with_warning(project, monkeypatch, rec, capsys):
    """The Earth Engine project belongs to the run (its tasks and queue live there)."""
    _fake_stack(monkeypatch, rec)
    edit_user_config(project, "auth", "project", "other-project")
    edit_user_config(project, "auth", "key_file", "secrets/laptop_key.json")
    assert run(project, "stack") == cli.EXIT_OK
    used_cfg = rec.calls[0][1][0]
    assert used_cfg["auth"]["project"] == project["auth"]["project"]
    assert used_cfg["auth"]["key_file"] == "secrets/laptop_key.json"
    err = capsys.readouterr().err
    assert "WARNING" in err and "other-project" in err and "run's project is used" in err


def test_machine_sections_come_from_user_config_without_warning(project, monkeypatch, rec, capsys):
    """resources and auth.key_file describe the machine running the command, not the run."""
    _fake_stack(monkeypatch, rec)
    edit_user_config(project, "resources", "cpu_workers", 64)
    edit_user_config(project, "auth", "key_file", "secrets/other_machine_key.json")

    assert run(project, "stack") == cli.EXIT_OK
    used_cfg = rec.calls[0][1][0]
    assert used_cfg["_config_path"].endswith("run_config.yaml")
    assert used_cfg["resources"]["cpu_workers"] == 64
    assert used_cfg["auth"]["key_file"] == "secrets/other_machine_key.json"
    assert "WARNING" not in capsys.readouterr().err


def test_stack_builds_all_tracks_in_one_call_and_reports_decision(make_project, monkeypatch, rec, capsys):
    cfg = make_project({"s1": {"tracks": [{"track_id": "RO123_ASC", "role": "primary"},
                                          {"track_id": "RO045_DSC", "role": "secondary"}]}})
    config.new_run(cfg)
    monkeypatch.setitem(sys.modules, "sar_pipeline.stack", SimpleNamespace(
        build_stacks=rec.fn("build_stacks", exc=DecisionRequired("3 unacknowledged QA issue(s) for track(s) ['RO045_DSC', 'RO123_ASC']"))))
    assert run(cfg, "stack") == cli.EXIT_ERROR
    assert rec.names() == ["build_stacks"]
    assert rec.calls[0][1][2] == ["RO123_ASC", "RO045_DSC"]
    err = capsys.readouterr().err
    assert "RO045_DSC" in err and "RO123_ASC" in err


def test_stack_prints_every_track(make_project, monkeypatch, rec, capsys):
    cfg = make_project({"s1": {"tracks": [{"track_id": "RO123_ASC", "role": "primary"},
                                          {"track_id": "RO045_DSC", "role": "secondary"}]}})
    config.new_run(cfg)
    monkeypatch.setitem(sys.modules, "sar_pipeline.stack", SimpleNamespace(build_stacks=rec.fn("build_stacks", _stacks_result)))
    assert run(cfg, "stack") == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "Stack for RO123_ASC" in out and "Stack for RO045_DSC" in out


def test_audit_and_rain_differences_are_warned(project, monkeypatch, rec, capsys):
    _fake_stack(monkeypatch, rec)
    edit_user_config(project, "audit", "max_gap_days", 30)
    edit_user_config(project, "rain", "level", "chunk")
    assert run(project, "stack") == cli.EXIT_OK
    err = capsys.readouterr().err
    assert "audit.max_gap_days" in err and "rain.level" in err
    assert rec.calls[0][1][0]["audit"]["max_gap_days"] != 30          # the frozen value is used
