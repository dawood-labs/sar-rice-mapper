"""Public-repository hygiene: nothing private may end up in files git would track.

The private values are read at test time from the local (gitignored) config files, the service
account key and `secrets/private_terms.txt` (one forbidden word per line). This test file itself
therefore contains no private information. On a fresh clone without those files the private-value
checks are skipped; the structural checks (notebook outputs, .gitignore) always run.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".ipynb", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".json", ".csv", ""}
REQUIRED_IGNORES = ["secrets/", "data/", "processed/", "reference/", "config/*.yaml", "!config/*.example.yaml",
                    "*.log", "*.png", "*.gpkg", "*.shp"]


def _candidate_files() -> list[Path]:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [REPO / line for line in out.splitlines() if line.strip()]


def _private_values() -> list[str]:
    values: set[str] = set()
    for cfg_path in (REPO / "config").glob("*.yaml"):
        if cfg_path.name.endswith(".example.yaml"):
            continue
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
        gcs, auth, aoi = cfg.get("gcs") or {}, cfg.get("auth") or {}, cfg.get("aoi") or {}
        values.update(str(v) for v in (gcs.get("bucket"), gcs.get("base_folder"), auth.get("project")) if v)
        for p in (auth.get("key_file"), aoi.get("path")):
            if p:
                values.add(Path(str(p)).stem)
        key = REPO / str(auth.get("key_file", ""))
        if auth.get("key_file") and key.is_file():
            data = json.loads(key.read_text())
            values.update(str(data[k]) for k in ("client_email", "private_key_id", "client_id") if data.get(k))
    terms = REPO / "secrets" / "private_terms.txt"
    if terms.exists():
        values.update(t.strip() for t in terms.read_text().splitlines() if t.strip())
    return sorted(v for v in values if len(v) >= 4)


def test_gitignore_protects_private_paths():
    text = (REPO / ".gitignore").read_text()
    missing = [p for p in REQUIRED_IGNORES if p not in text.splitlines()]
    assert not missing, f".gitignore lacks {missing}"


def test_no_private_values_in_tracked_files():
    values = _private_values()
    if not values:
        pytest.skip("no local private config/terms to check against")
    hits = []
    for f in _candidate_files():
        if f.suffix.lower() not in TEXT_SUFFIXES or not f.is_file() or f.stat().st_size > 2_000_000:
            continue
        text = f.read_text(errors="ignore").lower()
        for v in values:
            if v.lower() in text:
                hits.append(f"{f.relative_to(REPO)}: contains a private value ({len(v)} chars, starts with {v[:2]!r})")
    assert not hits, "\n".join(hits)


def test_no_secret_files_tracked():
    bad = [f for f in _candidate_files() if f.suffix.lower() in {".pem", ".key"}
           or f.parts[len(REPO.parts)] in {"secrets", "data", "processed", "reference"}]
    assert not bad, bad


def test_notebooks_have_no_outputs():
    for nb in (REPO / "notebooks").glob("*.ipynb"):
        data = json.loads(nb.read_text())
        for cell in data.get("cells", []):
            if cell.get("cell_type") == "code":
                assert not cell.get("outputs"), f"{nb.name} has outputs (clear them before committing)"
                assert cell.get("execution_count") is None, f"{nb.name} has execution counts"
