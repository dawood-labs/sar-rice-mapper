"""Config loading and the processed/ folder conventions.

Folder layout (details: docs/05_outputs_and_folders.md):
processed/<aoi.key>/<season.key>/{grid, audit/<YYYYMMDD>, runs/v<NNN>_<YYYYMMDD>, LATEST.txt}

All paths in a config are relative to the project root. The project root is
- `project_root` from the config (relative to the config file's folder), if given, else
- the parent of the `config/` folder that contains the config file.
A config anywhere else without `project_root` is rejected, so paths never resolve silently
against the wrong folder.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .errors import PipelineError

log = logging.getLogger(__name__)


def repo_root(start: str | Path | None = None) -> Path:
    """The repository root, found by walking up to the directory holding ``pyproject.toml``.

    Why this exists: paths such as ``config/`` and ``data/`` are written relative to the repository,
    and a notebook runs with its own folder as the working directory. Library code should not depend
    on where it was called from, so anything resolving those paths goes through here instead of
    trusting ``Path.cwd()``. Falls back to the current directory when no marker is found.
    """
    here = Path(start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def load_config(config_path: str | Path, project_root: str | Path | None = None) -> dict:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Private keys (leading underscore) are added at load time and never written back to YAML.
    cfg["_config_path"] = str(config_path)
    if project_root is not None:
        cfg["_root"] = str(Path(project_root).resolve())
    elif cfg.get("project_root"):
        cfg["_root"] = str((config_path.parent / cfg["project_root"]).resolve())
    elif config_path.parent.name == "config":
        cfg["_root"] = str(config_path.parents[1])
    else:
        raise PipelineError(
            f"{config_path} is not inside a 'config/' folder and has no 'project_root' key, so relative "
            "paths cannot be resolved safely. Move it to <project>/config/ or add project_root."
        )
    return cfg


def root(cfg: dict) -> Path:
    return Path(cfg["_root"])


def season_dir(cfg: dict) -> Path:
    return root(cfg) / "processed" / cfg["aoi"]["key"] / cfg["season"]["key"]


def grid_dir(cfg: dict) -> Path:
    return season_dir(cfg) / "grid"


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def audit_dir(cfg: dict, audit_date: str | None = None) -> Path:
    """Audit folder; named by UTC date so machines in different time zones agree."""
    return season_dir(cfg) / "audit" / (audit_date or utc_today())


def latest_audit_dir(cfg: dict) -> Path:
    dirs = sorted(p for p in (season_dir(cfg) / "audit").glob("[0-9]" * 8) if p.is_dir())
    if not dirs:
        raise FileNotFoundError("No audit found - run the audit stage first.")
    return dirs[-1]


def aoi_path(cfg: dict) -> Path:
    return root(cfg) / cfg["aoi"]["path"]


def gcs_prefix(cfg: dict, *parts: str) -> str:
    """Object prefix (without bucket) under <base_folder>/<aoi>/<season>/<parts...>."""
    segs = [cfg["gcs"]["base_folder"], cfg["aoi"]["key"], cfg["season"]["key"], *parts]
    return "/".join(str(s).strip("/") for s in segs if s)


# ---------------------------------------------------------------- runs (versioning)
_RUN_RE = re.compile(r"^v(\d{3})_(\d{8})$")


def list_runs(cfg: dict) -> list[str]:
    """Complete runs (folder name v<NNN>_<date> containing run_config.yaml), oldest first."""
    runs_dir = season_dir(cfg) / "runs"
    if not runs_dir.exists():
        return []
    return sorted(p.name for p in runs_dir.iterdir()
                  if p.is_dir() and _RUN_RE.match(p.name) and (p / "run_config.yaml").exists())


def new_run(cfg: dict) -> Path:
    """Create a new immutable run folder v<NNN>_<UTC date> with a frozen copy of the config.

    The folder is prepared under a temporary name and renamed at the end, so a crash never leaves
    a half-initialised run. `run_uid.txt` gets a random suffix: folder names are only unique on one
    machine, and the uid keeps GCS prefixes and Earth Engine task descriptions globally unique.
    """
    runs = list_runs(cfg)
    runs_dir = season_dir(cfg) / "runs"
    existing = [p.name for p in runs_dir.iterdir()] if runs_dir.exists() else []
    numbers = [int(m.group(1)) for n in existing if (m := _RUN_RE.match(n))]
    nxt = max(numbers) + 1 if numbers else 1
    run_id = f"v{nxt:03d}_{utc_today()}"
    final = runs_dir / run_id
    tmp = runs_dir / f".{run_id}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)  # leftover from a crashed new_run; it never became a run
    tmp.mkdir(parents=True)
    shutil.copy(cfg["_config_path"], tmp / "run_config.yaml")
    (tmp / "run_uid.txt").write_text(f"{run_id}_{secrets.token_hex(3)}\n")
    os.replace(tmp, final)

    latest = season_dir(cfg) / "LATEST.txt"
    latest_tmp = latest.with_suffix(".txt.tmp")
    latest_tmp.write_text(run_id + "\n")
    os.replace(latest_tmp, latest)
    log.info("created run %s (previous runs: %s)", run_id, ", ".join(runs) or "none")
    return final


def run_dir(cfg: dict, run_id: str | None = None) -> Path:
    """Folder of `run_id`, or of the newest complete run when run_id is None.

    LATEST.txt is a convenience pointer for humans. If a crash happened between creating a run and
    updating LATEST.txt, the newest complete run still wins and a warning names the mismatch.
    """
    if run_id is None:
        runs = list_runs(cfg)
        if not runs:
            raise FileNotFoundError("No run yet - create one with the new-run stage first.")
        run_id = runs[-1]
        latest = season_dir(cfg) / "LATEST.txt"
        pointer = latest.read_text().strip() if latest.exists() else ""
        if pointer != run_id:
            log.warning("LATEST.txt points to %r but the newest complete run is %r; using %r", pointer, run_id, run_id)
    d = season_dir(cfg) / "runs" / run_id
    if not (d / "run_config.yaml").exists():
        raise FileNotFoundError(f"Run folder missing or incomplete: {d}")
    return d


def load_run_config(run_path: Path) -> dict:
    """The run's frozen config, with paths resolved against the original project root."""
    run_path = Path(run_path).resolve()
    # processed/<aoi>/<season>/runs/<run> -> project root
    return load_config(run_path / "run_config.yaml", project_root=run_path.parents[4])
